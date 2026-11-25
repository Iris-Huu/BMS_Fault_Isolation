"""Reproduce the Week 3 quantitative analysis and draft paper figures."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.distinguishability import (colored_covariance, profile, short_boundary,
                                    whiten, window_signatures)
from src.io import load_panasonic_data
from src.model import simulate
from src.preparation import file_digest

DURATIONS = (10, 30, 60, 300, 600, 1800, 3600)
STATES = ['soc', 'v1_v', 'v2_v', 'temperature_c']
INPUTS = ['time_s', 'current_discharge_positive_a', 'ambient_input_c']
COVARIANCES = ['cycle3_colored', 'total_white', 'sensor_white']
AMBIENT = .35
ALPHA = 1 / 3600
POWER = .90
THRESHOLD = (norm.isf(ALPHA) + norm.ppf(POWER)) ** 2 / 2


def read_records(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rest_segments(t, current):
    rest = np.abs(current) <= .05
    splits = np.r_[0, np.flatnonzero((rest[:-1] != rest[1:]) | (np.diff(t) > 2.)) + 1, len(t)]
    return [(a, b) for a, b in zip(splits[:-1], splits[1:]) if rest[a] and b - a > 1]


def select_windows(t, segments, reference_soc, soc_grid, duration, limit):
    starts = []
    for a, b in segments:
        last = t[b - 1] - duration
        if last >= t[a]:
            candidates = np.unique(np.searchsorted(t, t[a] + np.arange(int(last - t[a]) + 1)))
            ends = np.searchsorted(t, t[candidates] + duration)
            available = ends < b
            candidates, ends = candidates[available], ends[available]
            valid = reference_soc[candidates] <= soc_grid[-1] + 1e-8
            valid &= reference_soc[candidates] >= soc_grid[0]
            valid &= reference_soc[ends] >= soc_grid[0]
            valid &= reference_soc[ends] <= soc_grid[-1] + 1e-8
            starts.extend(candidates[valid])
    starts = np.unique(starts).astype(int)
    if len(starts) > limit:
        chosen = starts[np.linspace(0, len(starts) - 1, limit, dtype=int)]
    else:
        chosen = starts
    return chosen, len(starts)


def fixed_bias_band(bias, short, ambient, limit, amplitude):
    """Resistance interval inside 10–1000 ohm where the fixed pair is below threshold."""
    zero = np.zeros_like(bias)

    def margin(g):
        return profile(g * short - amplitude * bias, zero, ambient, limit)[0] - THRESHOLD

    lo, hi = .001, .1
    minimum = minimize_scalar(margin, bounds=(lo, hi), method='bounded',
                              options={'xatol': 1e-10})
    center = min((lo, minimum.x, hi), key=margin)
    if margin(center) >= 0:
        return np.nan, np.nan
    lower_g = lo if margin(lo) <= 0 else brentq(margin, lo, center)
    upper_g = hi if margin(hi) <= 0 else brentq(margin, center, hi)
    return 1 / upper_g, 1 / lower_g


def evaluate(signatures, factor, channels, covariance, base):
    s = whiten(signatures, factor, channels, covariance != 'cycle3_colored')
    bias, short, ambient = s.T
    record = dict(base, covariance=covariance, channels='V+T' if len(channels) == 2 else 'V')
    for ambient_case, limit in [('known', 0.), ('profiled', 2 * AMBIENT)]:
        resistance, status = short_boundary(bias, short, ambient, limit, THRESHOLD)
        result = dict(record, ambient_case=ambient_case, boundary_ohm=resistance,
                      boundary_status=status)
        for r in [10, 30, 100, 300, 1000]:
            d, rival, offset = profile(short / r, bias, ambient, limit)
            result[f'D_R{r}'] = d
            result[f'closest_bias_mA_R{r}'] = 1000 * rival
            result[f'ambient_difference_C_R{r}'] = offset
        for ma in [-10, -25, -50, -100, 10, 25, 50, 100]:
            magnitude = ma / 1000
            d, g, _ = profile(magnitude * bias, short, ambient, limit, rival_nonnegative=True)
            result[f'D_bias_{ma}_mA'] = d
            result[f'closest_conductance_S_bias_{ma}_mA'] = g
            if ma < 0:
                low, high = fixed_bias_band(bias, short, ambient, limit, magnitude)
                result[f'fixed_bias_{ma}_ambiguous_from_ohm'] = low
                result[f'fixed_bias_{ma}_ambiguous_to_ohm'] = high
        yield result


def sample_outputs(time, output, target):
    """Evaluate predicted means on the covariance grid, within the native window."""
    flat = output.reshape(len(time), -1)
    interpolated = np.column_stack([np.interp(target, time, column) for column in flat.T])
    return interpolated.reshape(len(target), *output.shape[1:])


def nonlinear_check(t, i, amb, p, ocv, initial, target, sig, nominal, base):
    rows = []
    for fault, units in [('fisc', [1/10, 1/100, 1/300, 1/1000]), ('fi', [-.05])]:
        for u in units:
            sim = simulate(t, i, amb, p, ocv, soc0=initial[0], polarization0=initial[1:3],
                           temperature0=initial[3], faults={fault: u})
            actual = sample_outputs(t, sim[['voltage_v', 'temperature_c']].to_numpy(), target) - nominal
            linear = sig[:, :, int(fault == 'fisc')] * u
            row = dict(base, fault=fault, magnitude=u)
            for index, channel in enumerate(['V', 'T']):
                error = np.linalg.norm(actual[:, index] - linear[:, index])
                row[f'relative_l2_{channel}'] = error / max(np.linalg.norm(actual[:, index]), 1e-12)
                row[f'rmse_{channel}'] = error / np.sqrt(len(target))
            for name in (['V', 'T'], ['V']):
                columns = [0, 1] if len(name) == 2 else [0]
                factor = noise_factor(base['temperature_c'], base['window_s'],
                                      base['observation_stride_s'], 'cycle3_colored', tuple(columns))
                # Compare both signatures in the same projected noise metric.
                packed = np.stack([sig[:, :, 0], actual, sig[:, :, 2]], axis=-1)
                a, b, c = whiten(packed, factor, columns, False).T
                packed[:, :, 1] = linear
                _, linear_w, _ = whiten(packed, factor, columns, False).T
                if fault == 'fisc':
                    row['nonlinear_D_' + '+'.join(name)] = profile(b, a, c, 2 * AMBIENT)[0]
                    row['linear_D_' + '+'.join(name)] = profile(linear_w, a, c, 2 * AMBIENT)[0]
            rows.append(row)
    return rows


@lru_cache(maxsize=16)
def noise_factor(temperature, duration, stride, covariance, channels):
    folder = ROOT / f'results/week3/preparation/noise/{temperature}C'
    if covariance == 'cycle3_colored':
        with np.load(folder / 'cycle3_colored_autocovariance.npz') as source:
            matrix = colored_covariance(source['gamma'], duration // stride, stride, [0, 1])
        if channels == (0,):
            matrix = matrix[::2, ::2]
    else:
        filename = ('sensor_surrogate_covariance.csv' if covariance == 'sensor_white'
                    else 'sensor_and_model_covariance.csv')
        raw = pd.read_csv(folder / filename, index_col=0).to_numpy()
        matrix = raw[np.ix_(np.array(channels) + 1, np.array(channels) + 1)]
    return cholesky(matrix, lower=True, check_finite=True)


def datasets():
    prep = ROOT / 'results/week3/preparation'
    segments = pd.read_csv(prep / 'operating_segments.csv')
    for record in read_records(prep / 'input_interface.json'):
        cycle = record['cycle']
        if cycle not in ['HPPC', 'US06', 'UDDS']:
            continue
        desired = 'zero_terminal_current' if cycle == 'HPPC' else 'driving'
        picked = segments[(segments.temperature_c == record['temperature_c']) &
                          (segments.cycle == cycle) & (segments.kind == desired)]
        path = ROOT / record['input_file']
        if not path.is_file() or file_digest(path) != record['input_sha256']:
            # Raw data can regenerate portable input when Windows-generated CSV hashes differ.
            raw = ROOT / record['source']
            if file_digest(raw) != record['source_sha256']:
                raise ValueError(f'Raw source hash mismatch: {raw}')
            frame = load_panasonic_data(raw, ambient_setpoint_c=record['temperature_c'])
            frame = standardize(frame)
            if len(frame) != record['samples']:
                raise ValueError(f'Prepared row indices no longer match {raw}')
            starts = frame.time_s.to_numpy()[picked.start_row]
            ends = frame.time_s.to_numpy()[picked.stop_row_exclusive - 1]
            if not (np.allclose(starts, picked.start_s, atol=1e-5, rtol=0)
                    and np.allclose(ends, picked.end_s, atol=1e-5, rtol=0)):
                raise ValueError(f'Prepared segment times no longer match {raw}')
        else:
            frame = pd.read_csv(path)
        yield record, frame, list(zip(picked.start_row, picked.stop_row_exclusive))
    path = next((ROOT / 'data/raw/Panasonic_18650PF/10degC Trise with pause').glob('*.mat'))
    frame = standardize(load_panasonic_data(path))
    record = dict(temperature_c=10, cycle='Trise_pause', ambient_source='measured',
                  source=path.relative_to(ROOT).as_posix(), source_sha256=file_digest(path),
                  model_parameters='results/week3/preparation/model_v1/10C.json')
    t, current, _ = frame[INPUTS].to_numpy().T
    yield record, frame, [(a, b) for a, b in rest_segments(t, current) if t[b-1]-t[a] >= 600]


def standardize(frame):
    return frame.rename(columns={'Time': 'time_s', 'Current_discharge_positive': INPUTS[1],
                                'Chamber_Temp_degC': INPUTS[2],
                                'Battery_Temp_degC': 'measured_temperature_c'})


def run(output, limit):
    output.mkdir(parents=True, exist_ok=True)
    rows, validation, support, sources = [], [], [], []
    for record, frame, segments in datasets():
        temperature, cycle = record['temperature_c'], record['cycle']
        print(f'{temperature} C {cycle}', flush=True)
        sources.append(record)
        model = read_records(ROOT / record['model_parameters'])
        p = dict(model['parameters'], exact_parameter_lookup=True)
        ocv = pd.read_csv(ROOT / model['base_ocv_file'])
        if cycle == 'Trise_pause':
            p['effective_ambient_offset_c'] = 0.
        t, current, ambient = frame[INPUTS].to_numpy().T
        measured_t = frame.measured_temperature_c.to_numpy()
        reference = simulate(t, current, ambient, p, ocv, temperature0=measured_t[0])
        reference = reference[STATES].to_numpy()
        for duration in DURATIONS:
            chosen, count = select_windows(t, segments, reference[:, 0], ocv.soc.to_numpy(),
                                           duration, limit)
            support.append(dict(temperature_c=temperature, cycle=cycle, window_s=duration,
                                eligible_start_count=count, evaluated_start_count=len(chosen)))
            for j, start in enumerate(chosen):
                stride = 5 if duration > 600 else 1
                target = t[start] + np.arange(stride, duration + 1, stride)
                stop = np.searchsorted(t, target[-1]) + 1
                native = slice(start, stop)
                initial = reference[start].copy()
                initial[3] = measured_t[start]
                sig, healthy = window_signatures(t[native], current[native], ambient[native],
                                                 p, ocv, initial)
                sig = sample_outputs(t[native], sig, target)
                healthy = sample_outputs(t[native], healthy, target)
                base = dict(temperature_c=temperature, cycle=cycle, window_s=duration,
                            onset_s=t[start], reference_soc=initial[0], initial_temperature_c=initial[3],
                            min_ambient_c=ambient[native].min(), max_ambient_c=ambient[native].max(),
                            ambient_source=record['ambient_source'], observation_stride_s=stride)
                for covariance in COVARIANCES:
                    for channels in [(0,), (0, 1)]:
                        factor = noise_factor(temperature, duration, stride, covariance, channels)
                        rows.extend(evaluate(sig, factor, list(channels), covariance, base))
                if j == len(chosen) // 2:
                    validation.extend(nonlinear_check(t[native], current[native], ambient[native], p,
                                                     ocv, initial, target, sig, healthy, base))
    data = pd.DataFrame(rows)
    data.to_csv(output / 'window_distinguishability.csv', index=False)
    pd.DataFrame(validation).to_csv(output / 'nonlinear_validation.csv', index=False)
    pd.DataFrame(support).to_csv(output / 'window_support.csv', index=False)
    metadata = dict(alpha_per_comparison=ALPHA, target_power=POWER, threshold_KL=THRESHOLD,
                    ambient_per_hypothesis_C=AMBIENT, max_windows_per_stratum=limit,
                    model='model-v1 with continuous lookup of the same parameter maps',
                    current_noise='conditional on the observed current input',
                    input_sources=sources)
    (output / 'analysis_settings.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    make_figures(data, output)


def make_figures(data, output):
    primary = data[(data.covariance == 'cycle3_colored') & (data.ambient_case == 'profiled')]
    colors = {25: '#a9412a', 10: '#156675', 0: '#626676'}
    cycles = ['HPPC', 'US06', 'UDDS', 'Trise_pause']
    labels = ['HPPC rest', 'US06 active', 'UDDS active', 'Trise long rest']
    fig3, axes3 = plt.subplots(2, 4, figsize=(12, 5.8), sharex='col', sharey=True,
                              layout='constrained')
    fig4, axes4 = plt.subplots(2, 2, figsize=(9, 6), sharey=True, layout='constrained')
    summary = []
    for col, (cycle, label) in enumerate(zip(cycles, labels)):
        ax4 = axes4.flat[col]
        for channel_index, channel in enumerate(['V', 'V+T']):
            ax3 = axes3[channel_index, col]
            ax3.set_title(f'{label} · {channel}')
            band = primary[(primary.cycle == cycle) & (primary.channels == channel)]
            for temp, group in band.groupby('temperature_c'):
                med = group.groupby('window_s')[['D_R100', 'D_R300']].median()
                for resistance, linestyle in [(100, '-'), (300, ':')]:
                    ax3.plot(med.index, med[f'D_R{resistance}'].clip(lower=1e-9),
                             linestyle, marker='o', markersize=3, color=colors[temp])
                if channel_index == 1:
                    for duration, g in group.groupby('window_s'):
                        values = g.boundary_ohm.fillna(10.)
                        mid = float(values.median())
                        status = 'below_10' if mid == 10 and (g.boundary_status == 'below_10').any() else (
                            'at_least_1000' if mid == 1000 else 'within_grid')
                        summary.append(dict(cycle=cycle, temperature_c=temp, window_s=duration,
                                            n=len(g), plotted_boundary_ohm=mid, status=status,
                                            count_below_10=sum(g.boundary_status == 'below_10'),
                                            count_at_least_1000=sum(g.boundary_status == 'at_least_1000')))
            ax3.axhline(THRESHOLD, color='.45', linestyle='--', linewidth=.8)
            ax3.set(xscale='log', yscale='log')
            ax3.grid(alpha=.18, which='both')
        for temp in colors:
            points = pd.DataFrame([x for x in summary if x['cycle'] == cycle and x['temperature_c'] == temp])
            if points.empty:
                continue
            ax4.plot(points.window_s, points.plotted_boundary_ohm, color=colors[temp], alpha=.7)
            for status, marker in [('within_grid', 'o'), ('below_10', 'v'), ('at_least_1000', '^')]:
                p = points[points.status == status]
                ax4.scatter(p.window_s, p.plotted_boundary_ohm, marker=marker, color=colors[temp], s=21)
        ax4.set(title=label, xscale='log', yscale='log', ylim=(8, 1250))
        ax4.grid(alpha=.2, which='both')
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, label=f'{t} °C' + (' (supp.)' if t == 0 else ''))
               for t, c in colors.items()]
    axes3[0, 0].set_ylabel('Profiled KL distance')
    axes3[1, 0].set_ylabel('Profiled KL distance')
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    for col, cycle in enumerate(cycles):
        durations = sorted(primary.loc[primary.cycle == cycle, 'window_s'].unique())
        for ax in (axes3[1, col], axes4.flat[col]):
            ax.xaxis.set_major_locator(FixedLocator(durations))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
            ax.xaxis.set_minor_formatter(NullFormatter())
    axes3[0, 0].yaxis.set_major_locator(FixedLocator([1e-9, 1e-7, 1e-5, 1e-3, .1, 10]))
    axes3[0, 0].yaxis.set_major_formatter(FuncFormatter(
        lambda value, _: r'$\leq 10^{-9}$' if value == 1e-9 else rf'$10^{{{np.log10(value):.0f}}}$'))
    fig3.supxlabel('Observation window (s)')
    fig3.legend(handles=handles + [Line2D([0], [0], color='.4', label='100 Ω'),
                Line2D([0], [0], color='.4', ls=':', label='300 Ω'),
                Line2D([0], [0], color='.4', ls='--', label='Design target')],
                loc='outside upper center', ncols=6, fontsize=8)
    fig4.supxlabel('Observation window (s)')
    fig4.supylabel('Median resistance boundary (Ω)')
    fig4.legend(handles=handles + [Line2D([0], [0], color='.4', marker='v', ls='none', label='<10 Ω'),
                                   Line2D([0], [0], color='.4', marker='^', ls='none', label='≥1000 Ω')],
                loc='outside upper center', ncols=5, fontsize=8)
    for fig, name in [(fig3, 'figure3_distinguishability'), (fig4, 'figure4_resistance_boundary')]:
        fig.savefig(output / (name + '.png'), dpi=230)
        fig.savefig(output / (name + '.pdf'))
        plt.close(fig)
    pd.DataFrame(summary).to_csv(output / 'figure4_boundary_summary.csv', index=False)
    compare = primary.pivot(index=['temperature_c', 'cycle', 'window_s', 'onset_s'],
                            columns='channels', values=['D_R100', 'D_R300'])
    compare.columns = ['_'.join(pair) for pair in compare.columns]
    for r in [100, 300]:
        compare[f'gain_R{r}'] = compare[f'D_R{r}_V+T'] - compare[f'D_R{r}_V']
        compare[f'crosses_design_target_R{r}'] = ((compare[f'D_R{r}_V+T'] >= THRESHOLD) &
                                                    (compare[f'D_R{r}_V'] < THRESHOLD)).astype(int)
    compare.reset_index().groupby(['temperature_c', 'cycle', 'window_s']).median().to_csv(
        output / 'thermal_contribution_summary.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/week3/quantitative')
    parser.add_argument('--windows-per-stratum', type=int, default=12)
    args = parser.parse_args()
    if args.windows_per_stratum < 1:
        parser.error('windows-per-stratum must be positive')
    run(args.output_dir, args.windows_per_stratum)
