"""Week 2: auditable C/20 phase extraction and a preliminary OCV proxy.

Current is positive during discharge. Ah follows the original tester convention.
This module does not claim that loaded terminal voltage is equilibrium OCV.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def summarize_phases(df, threshold_a=0.01):
    """Split consecutive charge/rest/discharge states; retain source row bounds."""
    if not np.isfinite(threshold_a) or threshold_a <= 0:
        raise ValueError("threshold_a must be positive")
    current = df["Current_discharge_positive"].to_numpy()
    state = np.where(current > threshold_a, 1,
                     np.where(current < -threshold_a, -1, 0))
    bounds = np.r_[0, np.flatnonzero(np.diff(state)) + 1, len(df)]
    rows = []
    for phase, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
        part = df.iloc[start:stop]
        row = dict(phase=phase, state=int(state[start]), start_row=int(start),
                   stop_row=int(stop), count=len(part),
                   start_s=float(part.Time.iloc[0]), end_s=float(part.Time.iloc[-1]),
                   mean_current_a=float(part.Current_discharge_positive.mean()),
                   ah_start=float(part.Ah.iloc[0]), ah_end=float(part.Ah.iloc[-1]),
                   integrated_discharge_ah=float(np.trapezoid(
                       part.Current_discharge_positive, part.Time) / 3600))
        for col in ["Voltage", "Battery_Temp_degC", "Chamber_Temp_degC"]:
            row[col + "_min"] = float(part[col].min())
            row[col + "_max"] = float(part[col].max())
        rows.append(row)
    return pd.DataFrame(rows)


def discharge_soc(part):
    """Map ONE full discharge segment to relative SOC 1..0 using its own Ah span.

The capacity is segment-specific, not a capacity for every test date.
"""
    result = part.copy()
    ah = result.Ah.to_numpy()
    capacity = float(ah[0] - ah[-1])
    if capacity <= 0 or np.any(np.diff(ah) > 1e-5):
        raise ValueError("Expected decreasing Ah in a single discharge segment")
    soc = 1 - (ah[0] - ah) / capacity
    if np.any(np.diff(soc) >= 0):
        raise ValueError("SOC must strictly decrease in the selected segment")
    result["soc_relative"] = soc
    return result, capacity


def fit_ocv_proxy(soc, voltage, knots=51):
    """PCHIP through SOC-bin medians. Endpoints use the observed endpoint values.

No extrapolation and no forced monotonicity; reversals remain diagnosable.
"""
    soc, voltage = np.asarray(soc), np.asarray(voltage)
    if (len(soc) < 2 * knots or len(soc) != len(voltage)
            or not np.all(np.isfinite(soc)) or not np.all(np.isfinite(voltage))):
        raise ValueError("Insufficient or invalid curve samples")
    order = np.argsort(soc)
    x, y = soc[order], voltage[order]
    if np.any(np.diff(x) <= 0):
        raise ValueError("SOC coordinates must be unique")
    # More resolution near empty/full SOC, where voltage curvature is strongest.
    fractions = (1 - np.cos(np.linspace(0, np.pi, knots - 1))) / 2
    edges = x[0] + (x[-1] - x[0]) * fractions
    xs, ys = [float(x[0])], [float(y[0])]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x > lo) & (x < hi)
        if mask.any():
            xs.append(float(np.median(x[mask])))
            ys.append(float(np.median(y[mask])))
    xs.append(float(x[-1]))
    ys.append(float(y[-1]))
    return PchipInterpolator(xs, ys, extrapolate=False)


def error_metrics(measured, predicted):
    residual = np.asarray(predicted) - np.asarray(measured)
    if not np.all(np.isfinite(residual)):
        raise ValueError("Prediction outside fitted SOC domain")
    return dict(rmse_mv=float(1000 * np.sqrt(np.mean(residual ** 2))),
                mae_mv=float(1000 * np.mean(np.abs(residual))),
                max_abs_mv=float(1000 * np.max(np.abs(residual))))
