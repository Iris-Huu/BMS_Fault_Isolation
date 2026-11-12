"""Reproduce the first Week 2 calibration step; no raw data are modified.

Example: python experiments/calibrate_ocv_25c.py --project-root PATH
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_dir or root / "results" / "week2" / "ocv_25c"
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))
    from src.io import load_panasonic_data
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import ocv as cal
    candidates = list((root / "data" / "raw" / "Panasonic_18650PF").rglob("*C20_25dC.mat"))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one C20_25dC file; found {len(candidates)}")
    source = candidates[0]
    df = load_panasonic_data(source)
    phases = cal.summarize_phases(df)
    phases.to_csv(output / "phase_summary.csv", index=False)
    eligible = phases[(phases.state == 1) &
                      (phases.Chamber_Temp_degC_min >= 24.5) &
                      (phases.Chamber_Temp_degC_max <= 25.5)]
    if len(eligible) != 1:
        raise ValueError("Expected one 25C discharge; inspect phase_summary.csv")
    phase = eligible.iloc[0]
    part, capacity = cal.discharge_soc(df.iloc[int(phase.start_row):int(phase.stop_row)])
    part.to_csv(output / "discharge_soc.csv", index=False)
    x, y = part.soc_relative.to_numpy(), part.Voltage.to_numpy()
    # Interleaved holdout checks interpolation only, not independent OCV accuracy.
    holdout = np.arange(len(part)) % 5 == 2
    holdout[[0, -1]] = False
    trial = cal.fit_ocv_proxy(x[~holdout], y[~holdout])
    holdout_metrics = cal.error_metrics(y[holdout], trial(x[holdout]))
    fitted = cal.fit_ocv_proxy(x, y)
    grid = np.linspace(0, 1, 1001)
    curve = fitted(grid)
    pd.DataFrame(dict(soc_relative=grid, voltage_proxy_v=curve)).to_csv(
        output / "ocv_25C_preliminary.csv", index=False)
    pd.DataFrame(dict(soc_relative=x, measured_voltage_v=y,
                      fitted_proxy_v=fitted(x), residual_mv=1000*(fitted(x)-y),
                      interpolation_holdout=holdout)).to_csv(output / "fit_residuals.csv", index=False)
    # Charging uses the SAME Ah coordinate anchored at discharge end, not a new 0..1 scale.
    charges = phases[(phases.state == -1) & (phases.start_s > phase.end_s)]
    charge_range = None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(x, y, ".", ms=2, alpha=.5, label="C/20 discharge measurements")
    axes[0].plot(grid, curve, lw=1.5, label="Discharge OCV proxy")
    if len(charges):
        ch = charges.iloc[0]
        charge = df.iloc[int(ch.start_row):int(ch.stop_row)].copy()
        charge["soc_relative"] = (charge.Ah - float(phase.ah_end)) / capacity
        charge.to_csv(output / "charge_soc_same_reference.csv", index=False)
        charge_range = [float(charge.soc_relative.min()), float(charge.soc_relative.max())]
        axes[0].plot(charge.soc_relative, charge.Voltage, lw=.8,
                     alpha=.7, label="Charge (same Ah reference; not fitted)")
    axes[0].set(xlabel="Relative SOC", ylabel="Terminal voltage (V)",
                title="25 C C/20: preliminary discharge OCV proxy")
    axes[0].legend(fontsize=8)
    axes[1].plot(x, 1000*(fitted(x)-y), lw=.8)
    axes[1].axhline(0, color="black", lw=.5)
    axes[1].set(xlabel="Relative SOC", ylabel="Proxy minus measurement (mV)",
                title="Curve approximation error (not model validation)")
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "ocv_25C_preliminary.png", dpi=180)
    plt.close(fig)
    metrics = dict(source=str(source.relative_to(root)),
                   method="SOC-bin medians + PCHIP; discharge terminal-voltage proxy",
                   phase=int(phase.phase), capacity_ah=capacity,
                   integrated_discharge_ah=float(phase.integrated_discharge_ah),
                   charge_soc_range_same_reference=charge_range,
                   full_curve_approximation=cal.error_metrics(y, fitted(x)),
                   interleaved_interpolation_holdout=holdout_metrics,
                   negative_grid_increments=int(np.sum(np.diff(curve) < 0)),
                   limitations=["Terminal voltage under load is not equilibrium OCV.",
                       "SOC endpoints are defined by this discharge window.",
                       "Holdout checks interpolation, not accuracy on a new experiment.",
                       "May C/20 and March HPPC/US06 dates differ; transfer is unvalidated.",
                       "Only 25C C/20 located; no independent 10C or 0C OCV calibration.",
                       "No extrapolation beyond relative SOC [0,1]."])
    (output / "calibration_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("Output:", output)


if __name__ == "__main__":
    main()
