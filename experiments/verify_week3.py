"""Complete Week 3 checks and sensitivity calculations from frozen inputs."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_week3 import (AMBIENT, DURATIONS, INPUTS, ROOT, STATES, THRESHOLD,
                                  evaluate, noise_factor, read_records, sample_outputs,
                                  select_windows)
from src.distinguishability import profile, short_boundary, whiten, window_signatures
from src.model import simulate
from tests.check_week3_distinguishability import check

OUT = ROOT / 'results/week3/quantitative'


def mixed_driving():
    rows, coverage = [], []
    inputs = read_records(ROOT / 'results/week3/preparation/input_interface.json')
    for record in inputs:
        if record['cycle'] not in ['US06', 'UDDS']:
            continue
        temperature, cycle = record['temperature_c'], record['cycle']
        print(f'Whole driving record {temperature} C {cycle}', flush=True)
        frame = pd.read_csv(ROOT / record['input_file'])
        time, current, ambient = frame[INPUTS].to_numpy().T
        model = read_records(ROOT / record['model_parameters'])
        parameters = dict(model['parameters'], exact_parameter_lookup=True)
        ocv = pd.read_csv(ROOT / model['base_ocv_file'])
        reference = simulate(time, current, ambient, parameters, ocv,
                             temperature0=frame.measured_temperature_c.iloc[0])[STATES].to_numpy()
        active = np.flatnonzero(np.abs(current) > .05)
        splits = np.r_[active[0], np.flatnonzero(np.diff(time) > 2.) + 1, active[-1] + 1]
        splits = np.unique(splits[(splits >= active[0]) & (splits <= active[-1] + 1)])
        segments = list(zip(splits[:-1], splits[1:]))
        for duration in DURATIONS:
            chosen, available = select_windows(time, segments, reference[:, 0], ocv.soc.values,
                                               duration, 3)
            coverage.append(dict(temperature_c=temperature, cycle=cycle, window_s=duration,
                                 eligible_start_count=available, evaluated_start_count=len(chosen)))
            for start in chosen:
                stride = 5 if duration > 600 else 1
                target = time[start] + np.arange(stride, duration + 1, stride)
                stop = np.searchsorted(time, target[-1]) + 1
                sl = slice(start, stop)
                initial = reference[start].copy()
                initial[3] = frame.measured_temperature_c.iloc[start]
                signatures, _ = window_signatures(time[sl], current[sl], ambient[sl], parameters, ocv, initial)
                signatures = sample_outputs(time[sl], signatures, target)
                base = dict(temperature_c=temperature, cycle=cycle, window_s=duration,
                            onset_s=time[start], observation_stride_s=stride,
                            window_definition='whole driving record including traffic rests')
                for channels in [(0,), (0, 1)]:
                    factor = noise_factor(temperature, duration, stride, 'cycle3_colored', channels)
                    rows.extend(evaluate(signatures, factor, list(channels), 'cycle3_colored', base))
    pd.DataFrame(rows).to_csv(OUT / 'whole_driving_distinguishability.csv', index=False)
    pd.DataFrame(coverage).to_csv(OUT / 'whole_driving_window_support.csv', index=False)


def thermal_sensitivity():
    rows = []
    data = pd.read_csv(OUT / 'window_distinguishability.csv')
    selected = data.query("covariance=='cycle3_colored' and ambient_case=='profiled' and channels=='V+T'")
    for (temperature, cycle), group in selected.groupby(['temperature_c', 'cycle']):
        if cycle not in ['HPPC', 'Trise_pause']:
            continue
        longest = group[group.window_s == group.window_s.max()].sort_values('onset_s')
        row = longest.iloc[len(longest) // 2]
        print(f'Thermal sensitivity {temperature} C {cycle}', flush=True)
        parameters = read_records(ROOT / f'results/week3/preparation/model_v1/{temperature}C.json')['parameters']
        time = np.arange(int(row.window_s) + 1, dtype=float)
        # A separate constant-OCV equilibrium benchmark tests the thermal scale
        # without conflating it with SOC lookup, polarization or chamber drift.
        voltage = 3.7
        table = pd.DataFrame({'soc': [0., 1.], 'ocv_v': [voltage, voltage]})
        base = {key: parameters[key] for key in ['Q_Ah', 'R0_ohm', 'R1_ohm', 'C1_F',
                                                'R2_ohm', 'C2_F', 'Cth_J_K', 'hA_W_K']}
        stride = int(row.observation_stride_s)
        factor = noise_factor(temperature, int(row.window_s), stride, 'cycle3_colored', (0, 1))
        for capacity in [base['Cth_J_K'], 40., 45., 50.]:
            p = dict(base, Cth_J_K=capacity, hA_W_K=base['hA_W_K'] * capacity / base['Cth_J_K'])
            sig, _ = window_signatures(time, time * 0., time * 0. + temperature, p, table,
                                       [.5, 0., 0., float(temperature)])
            bias, short, chamber = whiten(sig[stride::stride], factor, [0, 1], False).T
            resistance, status = short_boundary(bias, short, chamber, 2 * AMBIENT, THRESHOLD)
            rows.append(dict(temperature_c=temperature, source_window=cycle, window_s=int(row.window_s),
                             Cth_J_K=capacity, hA_W_K=p['hA_W_K'], boundary_ohm=resistance,
                             boundary_status=status,
                             D_R100=profile(short / 100, bias, chamber, 2 * AMBIENT)[0],
                             scenario='3.7 V equilibrium thermal-scale benchmark'))
    pd.DataFrame(rows).to_csv(OUT / 'thermal_scale_sensitivity.csv', index=False)


def plot_supplement():
    import matplotlib.pyplot as plt
    data = pd.read_csv(OUT / 'whole_driving_distinguishability.csv')
    selected = data[data.ambient_case == 'profiled']
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout='constrained')
    colors = {25: '#a9412a', 10: '#156675', 0: '#626676'}
    for column, cycle in enumerate(['US06', 'UDDS']):
        for (temperature, channels), group in selected[selected.cycle == cycle].groupby(['temperature_c', 'channels']):
            median = group.groupby('window_s')[['D_R100', 'boundary_ohm']].median()
            style = '-' if channels == 'V+T' else ':'
            axes[0, column].plot(median.index, median.D_R100.clip(lower=1e-9), style,
                                  marker='o', markersize=3, color=colors[temperature],
                                  label=f'{temperature} C {channels}')
            if channels == 'V+T':
                axes[1, column].plot(median.index, median.boundary_ohm, marker='o',
                                      color=colors[temperature], markersize=3)
        axes[0, column].axhline(THRESHOLD, color='.4', ls='--', lw=.8)
        axes[0, column].set(title=f'{cycle} with traffic rests', xscale='log', yscale='log')
        axes[1, column].set(xscale='log', xlabel='Observation window (s)', ylabel='Resistance boundary (ohm)')
        axes[0, column].set_ylabel('KL distance at 100 ohm')
        for ax in axes[:, column]:
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7, ncols=2)
    for suffix in ['pdf', 'png']:
        fig.savefig(OUT / ('whole_driving_comparison.' + suffix), dpi=200)
    plt.close(fig)


def validate():
    synthetic = check()
    supplementary = pd.read_csv(OUT / 'whole_driving_distinguishability.csv')
    coverage = pd.read_csv(OUT / 'whole_driving_window_support.csv')
    assert len(supplementary) == coverage.evaluated_start_count.sum() * 4
    assert (supplementary.filter(regex=r'^D_').to_numpy() >= 0.).all()
    assert (np.diff(supplementary[['D_R1000', 'D_R300', 'D_R100', 'D_R30', 'D_R10']].to_numpy(), axis=1) >= -1e-9).all()
    data = pd.read_csv(OUT / 'window_distinguishability.csv')
    support = pd.read_csv(OUT / 'window_support.csv')
    assert len(data) == support.evaluated_start_count.sum() * 12
    assert np.isfinite(data.filter(regex=r'^D_').to_numpy()).all()
    assert (data.filter(regex=r'^D_').to_numpy() >= 0.).all()
    assert not data.duplicated(['temperature_c', 'cycle', 'window_s', 'onset_s', 'channels',
                                'covariance', 'ambient_case']).any()
    grouped = data.pivot(index=['temperature_c', 'cycle', 'window_s', 'onset_s',
                                'covariance', 'ambient_case'], columns='channels', values='D_R100')
    assert (grouped['V+T'] >= grouped['V'] - 1e-9).all()
    distances = data[['D_R1000', 'D_R300', 'D_R100', 'D_R30', 'D_R10']].to_numpy()
    assert (np.diff(distances, axis=1) >= -1e-9).all()
    result = dict(synthetic_check=synthetic, zero_fault_and_monotonicity='passed',
                  main_rows=len(data), evaluated_windows=int(support.evaluated_start_count.sum()),
                  nonnegative_finite_distances='passed', temperature_information_monotonicity='passed',
                  duplicate_window_check='passed', whole_driving_windows=int(
                      pd.read_csv(OUT / 'whole_driving_window_support.csv').evaluated_start_count.sum()),
                  thermal_scale_scenarios=len(pd.read_csv(OUT / 'thermal_scale_sensitivity.csv')),
                  interpretation='Week 3 prediction and figure drafts complete')
    (OUT / 'verification_status.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    check()
    mixed_driving()
    thermal_sensitivity()
    plot_supplement()
    validate()
    print('Week 3 verification passed', flush=True)
