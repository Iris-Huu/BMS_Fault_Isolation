"""Audit 25 C HPPC pulses and explore 2RC relaxation fits.

All parameter estimates are provisional, not validated model parameters.
Never integrate across omitted discharge intervals in the HPPC file.
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
from scipy.optimize import least_squares


def fit_relaxation(t, v, amplitude_a, pulse_s, last_loaded_v):
    """V=Vinf-Afast exp(-t/taufast)-Aslow exp(-t/tauslow).

Time zero is the last loaded sample; switching instant is uncertain by the
off-edge sampling interval. R=A/[I*(1-exp(-pulse_s/tau))] assumes zero initial
polarization and a constant-current pulse. Results require joint validation.
"""
    # Uniform in log time to reduce bias from mixed 0.1s/1s logging rates.
    grid = np.geomspace(t[0], t[-1], 200)
    indices = np.unique(np.clip(np.searchsorted(t, grid), 0, len(t)-1))
    x, y = t[indices], v[indices]
    def predict(q, time):
        return q[0]-q[1]*np.exp(-time/np.exp(q[3]))-q[2]*np.exp(-time/np.exp(q[4]))
    bounds = ([2, 0, 0, np.log(.05), np.log(.05)],
              [4.5, 1, 1, np.log(5000), np.log(5000)])
    fits = []
    for fast, slow in [(1, 30), (5, 150), (20, 600)]:
        guess = [v[-1], max((v[-1]-v[0])*.5, .001),
                 max((v[-1]-v[0])*.5, .001), np.log(fast), np.log(slow)]
        fit = least_squares(lambda q: predict(q, x)-y, guess, bounds=bounds,
                            max_nfev=500)
        fits.append(fit)
    best = min(fits, key=lambda result: np.sum(result.fun**2))
    q = best.x
    order = np.argsort(np.exp(q[3:]))
    tau = np.exp(q[3:])[order]
    amps = q[1:3][order]
    resistance = amps / (amplitude_a * -np.expm1(-pulse_s/tau))
    capacitance = tau / resistance
    r0 = (q[0]-amps.sum()-last_loaded_v)/amplitude_a
    early_dt = float(np.median(np.diff(t[t <= 5]))) if np.sum(t <= 5)>2 else float(t[0])
    flags = []
    if tau[0] < 5*max(early_dt, float(t[0])):
        flags.append("fast_time_constant_poorly_resolved")
    if tau[1] > t[-1]/3:
        flags.append("slow_relaxation_window_short")
    if tau[1]/tau[0] < 3:
        flags.append("time_constants_not_well_separated")
    if min(amps) < .001:
        flags.append("polarization_amplitude_below_1mV")
    if r0 <= 0:
        flags.append("nonpositive_R0_extrapolation")
    if not best.success:
        flags.append("optimizer_not_converged")
    residual = predict(q,t)-v
    row = dict(R0_relaxation_candidate_ohm=float(r0), R1_candidate_ohm=resistance[0],
               C1_candidate_f=capacitance[0], R2_candidate_ohm=resistance[1],
               C2_candidate_f=capacitance[1], tau_fast_s=tau[0], tau_slow_s=tau[1],
               voltage_asymptote_v=q[0], relaxation_rmse_mv=float(1000*np.sqrt(np.mean(residual**2))),
               jacobian_condition=float(np.linalg.cond(best.jac)),
               numerical_flags=";".join(flags),
               status="exploratory_requires_joint_pulse_and_drive_validation")
    return row, predict(q,t)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_dir or root/"archive"/"results"/"hppc_relaxation_25c"
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0,str(root))
    from src.io import load_panasonic_data
    from src.ocv import summarize_phases
    raw = root/"data"/"raw"/"Panasonic_18650PF"
    files = list(raw.rglob("*25degC_5Pulse_HPPC*.mat"))
    if len(files) != 1:
        raise ValueError("Expected one 25C HPPC source")
    source = files[0]
    df = load_panasonic_data(source)
    companion = load_panasonic_data(source.parent/"03-11-17_10.10 3390_dis5_10p.mat")
    capacity_source = next(raw.rglob("03-09-17_17.59*Dis1C_1.mat"))
    cap_df = load_panasonic_data(capacity_source)
    cap_phases = summarize_phases(cap_df)
    cap_phase = cap_phases[cap_phases.state==1].iloc[0]
    capacity = float(cap_phase.ah_start-cap_phase.ah_end)
    t, i, v, ah = [df[c].to_numpy() for c in ["Time","Current_discharge_positive","Voltage","Ah"]]
    on = i > .2
    bounds = np.r_[0,np.flatnonzero(np.diff(on))+1,len(df)]
    pulses = [(a,z) for a,z in zip(bounds[:-1],bounds[1:]) if on[a]]
    inventory, fits, curves = [], [], []
    for k,(a,z) in enumerate(pulses,1):
        next_start = pulses[k][0] if k<len(pulses) else len(df)
        end = next_start
        if z<end:
            # A zero-current row after a gap or an Ah jump is NOT a contiguous rest.
            cuts = np.flatnonzero((np.diff(t[z:end])>2) | (np.abs(np.diff(ah[z:end]))>1e-4))
            if len(cuts):
                end = z+int(cuts[0])+1
        mean_i=float(i[a:z].mean())
        duration=float(t[z-1]-t[a])
        off_gap=float(t[z]-t[z-1]) if z<len(df) else np.nan
        rest_s=float(t[end-1]-t[z]) if end>z else 0.
        flags=[]
        if duration<9:
            flags.append("cutoff_truncated_pulse")
        if off_gap>.2:
            flags.append("off_edge_coarse_sampling")
        if rest_s<300:
            flags.append("short_contiguous_rest")
        start_i=i[a]-i[a-1] if a else np.nan
        row=dict(pulse=k,start_row=a,stop_row=z,rest_stop_row=end,start_s=t[a],
                 duration_observed_s=duration,current_mean_a=mean_i,
                 ah_start=ah[a],ah_end=ah[z-1],
                 soc_start_reference=1+(ah[a]-ah[0])/capacity,
                 soc_end_reference=1+(ah[z-1]-ah[0])/capacity,
                 R_on_apparent_ohm=(v[a-1]-v[a])/start_i if a else np.nan,
                 R_off_apparent_ohm=(v[z]-v[z-1])/(i[z-1]-i[z]) if z<len(df) else np.nan,
                 off_edge_gap_s=off_gap,contiguous_rest_s=rest_s,
                 rest_samples=end-z,flags=";".join(flags))
        inventory.append(row)
        if duration>=9 and end-z>=50 and rest_s>=300 and off_gap<=.2:
            rt=t[z:end]-t[z-1]
            fit,pred=fit_relaxation(rt,v[z:end],mean_i,duration,v[z-1])
            fit.update(pulse=k,soc_reference=row["soc_start_reference"],current_a=mean_i)
            fits.append(fit)
            if len(curves)<3 and mean_i<6:
                curves.append((k,rt,v[z:end],pred,fit))
    table=pd.DataFrame(inventory)
    table.to_csv(output/"pulse_inventory.csv",index=False)
    fit_table=pd.DataFrame(fits)
    fit_table.to_csv(output/"relaxation_2rc_candidates.csv",index=False)
    # Align companion by cumulative Ah context only; its Time is file-local.
    from src.ocv import summarize_phases
    comp_phases=summarize_phases(companion,threshold_a=.2)
    comp_phases.to_csv(output/"companion_phase_summary.csv",index=False)
    gaps=np.flatnonzero((np.diff(t)>2)|(np.abs(np.diff(ah))>.003))
    pd.DataFrame(dict(row_before=gaps,time_before_s=t[gaps],time_after_s=t[gaps+1],
                      dt_s=np.diff(t)[gaps],ah_change=np.diff(ah)[gaps])).to_csv(output/"time_and_ah_gaps.csv",index=False)
    summary=dict(source=str(source.relative_to(root)),capacity_source=str(capacity_source.relative_to(root)),
                 capacity_reference_ah=capacity,pulse_count=len(table),
                 truncated_pulses=table.loc[table.duration_observed_s<9,"pulse"].tolist(),
                 current_levels_a=sorted(set(np.round(table.current_mean_a,1))),
                 relaxation_fits_attempted=len(fit_table),
                 numerically_flagged_fits=int((fit_table.numerical_flags!="").sum()),
                 relaxation_rmse_median_mv=float(fit_table.relaxation_rmse_mv.median()),
                 limitations=["SOC assumes full charge at HPPC start and uses a nearby 1C capacity, not equilibrium capacity.",
                    "Apparent edge resistance contains sample timing and fast polarization effects; not pure R0.",
                    "Relaxation fits assume negligible initial polarization and locally constant OCV after each pulse.",
                    "Small voltage error does not prove unique 2RC parameters.",
                    "No fit is accepted for model use until joint pulse and independent drive validation.",
                    "Companion contains additional sequences and Ah resets; do not concatenate by Time."])
    (output/"hppc_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    fig,ax=plt.subplots(3,1,figsize=(12,8),sharex=True)
    # NaN breaks prevent connecting unobserved portions in the figure.
    for axis,values,label in zip(ax,[i,v,1+(ah-ah[0])/capacity],["Discharge current (A)","Voltage (V)","Reference SOC"]):
        xx,yy=t.copy(),values.copy()
        yy[np.r_[False,np.diff(t)>2]]=np.nan
        axis.plot(xx/3600,yy,lw=.65)
        axis.set_ylabel(label)
        axis.grid(alpha=.25)
    ax[0].set_title("25 C HPPC: 67 pulses; gaps are not interpolated")
    ax[-1].set_xlabel("HPPC file time (h)")
    fig.tight_layout()
    fig.savefig(output/"hppc_pulse_overview.png",dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,6.5))
    for col,(k,rt,obs,pred,fit) in enumerate(curves):
        axes[0,col].plot(rt,obs,".",ms=1.5,label="Measured")
        axes[0,col].plot(rt,pred,lw=1,label="2-exponential fit")
        axes[0,col].set(title=f"Pulse {k}: exploratory fit",ylabel="Voltage (V)")
        axes[0,col].legend(fontsize=8)
        axes[1,col].plot(rt,1000*(pred-obs),lw=.7)
        axes[1,col].set(xlabel="Time after last loaded sample (s)",ylabel="Fit minus measurement (mV)")
        for axis in axes[:,col]:
            axis.set_xscale("log")
            axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output/"relaxation_fit_examples.png",dpi=160)
    plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
