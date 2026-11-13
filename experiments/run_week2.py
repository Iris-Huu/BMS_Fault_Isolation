"""Reproduce Week 2 numerical baseline, validation and evidence of unmet gates.

Uses only HPPC + Cycle1/2 for calibration; US06/UDDS are held-out validation.
No network access, raw-data writes, or repository commits are performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from src.io import load_panasonic_data
from src.model import simulate
from src.calibration import pulse_windows,rested_ocv,fit_joint_pulse,covariance_from_rests,fit_thermal_integrals


def dump(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")


def metrics(predicted,measured):
    r=np.asarray(predicted)-np.asarray(measured)
    return dict(rmse=float(np.sqrt(np.mean(r*r))),mae=float(np.mean(np.abs(r))),
                bias=float(np.mean(r)),max_abs=float(np.max(np.abs(r))))


def run(root,out):
    raw=root/"data"/"raw"/"Panasonic_18650PF"
    allfiles=list(raw.rglob("*.mat"))
    out.mkdir(parents=True,exist_ok=True)
    summary=out/"summary"
    parameters=out/"parameters"
    summary.mkdir(exist_ok=True);parameters.mkdir(exist_ok=True)
    manifest=[]; parameter_rows=[]; errors=[]; all_fit_rows=[]; issues=[]
    def find(temp,kind):
        candidates=[p for p in allfiles if f" {temp}degC_" in p.name and kind in p.name and "trise" not in str(p).lower()]
        if len(candidates)!=1: raise ValueError(f"Ambiguous {temp} {kind}: {candidates}")
        return candidates[0]
    def load(path,temp,role):
        frame=load_panasonic_data(path,ambient_setpoint_c=float(temp))
        report=frame.attrs["quality_report"]
        manifest.append(dict(file=str(path.relative_to(root)),role=role,temperature_c=temp,
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             rows=len(frame),ambient_source=report["ambient_source"],
                             ambient_imputed_rows=report["ambient_imputed_rows"],
                             original_invalid_values=report["original_invalid_values"],
                             max_sampling_interval_s=report["max_sampling_interval_s"],
                             intervals_over_15s=int(np.sum(np.diff(frame.Time)>15)),
                             removed_duplicate_records=report["duplicate_time_records_handled"]))
        return frame
    for temp in [25,10,0]:
        folder=out/"calibration"/f"{temp}C"
        noise_folder=out/"noise"/f"{temp}C"
        validation_folder=out/"validation"/f"{temp}C"
        for directory in [folder,noise_folder,validation_folder]: directory.mkdir(parents=True,exist_ok=True)
        print(f"{temp}C: HPPC joint calibration",flush=True)
        hppc=load(find(temp,"HPPC"),temp,"electrical_calibration_and_noise")
        windows=pulse_windows(hppc)
        ocv,ocv_raw=rested_ocv(hppc,windows)
        ocv.to_csv(parameters/f"ocv_{temp}C.csv",index=False)
        ocv_raw.to_csv(folder/"ocv_source_points.csv",index=False)
        fits=[]; examples=[]
        for w in windows:
            if not (9<=w["duration_s"]<=11 and w["current_a"]<=3.5 and w["rest_s"]>=600): continue
            row,(idx,pred)=fit_joint_pulse(hppc,w)
            row["temperature_c"]=temp
            fits.append(row)
            if len(examples)<3: examples.append((w["pulse"],hppc.Time.iloc[idx].to_numpy()-hppc.Time.iloc[w["start"]],hppc.Voltage.iloc[idx].to_numpy(),pred))
        fit_table=pd.DataFrame(fits)
        if fit_table.empty: raise ValueError(f"No fit windows at {temp}C")
        fit_table.to_csv(folder/"joint_pulse_fits.csv",index=False)
        all_fit_rows.extend(fits)
        selected=fit_table[(fit_table["flags"]=="")&(fit_table.rmse_mv<10)&fit_table.soc_reference.between(.2,.9)]
        selection="unflagged_mid_soc_pulses"
        if len(selected)<3:
            selected=fit_table[(fit_table.rmse_mv<10)&fit_table.soc_reference.between(.2,.9)]
            selection="fallback_includes_flagged_fits"
            issues.append(f"{temp}C: insufficient unflagged fits; aggregate includes flagged candidates")
        if len(selected)<1: raise ValueError("No usable mid-SOC fit")
        selected.to_csv(folder/"aggregate_source_pulses.csv",index=False)
        p={"Q_Ah":2.9,"Cth_J_K":50.,"hA_W_K":.05}
        for k in ["R0_ohm","R1_ohm","R2_ohm"]: p[k]=float(selected[k].median())
        p["C1_F"]=float(selected.tau1_s.median()/p["R1_ohm"])
        p["C2_F"]=float(selected.tau2_s.median()/p["R2_ohm"])
        print(f"{temp}C: {len(fits)} fits, {len(selected)} aggregate sources; thermal calibration",flush=True)
        records=[]; calibration_frames=[]
        for cycle in [1,2]:
            frame=load(find(temp,f"Cycle_{cycle}_Pan"),temp,"thermal_calibration")
            # Nominal SOC starts at 1 after the protocol's full recharge.
            soc=1+(frame.Ah.to_numpy()-frame.Ah.iloc[0])/p["Q_Ah"]
            equilibrium=np.interp(soc,ocv.soc,ocv.ocv_v)
            heat=frame.Current_discharge_positive.to_numpy()*(equilibrium-frame.Voltage.to_numpy())
            records.append((frame.Time.to_numpy(),frame.Battery_Temp_degC.to_numpy(),frame.Chamber_Temp_degC.to_numpy(),heat))
            calibration_frames.append(frame)
        thermal=fit_thermal_integrals(records)
        p.update({k:thermal[k] for k in ["Cth_J_K","hA_W_K"]})
        sigma,effective,noise,residual=covariance_from_rests(hppc,windows)
        labels=noise["channels"]
        pd.DataFrame(sigma,index=labels,columns=labels).to_csv(noise_folder/"sigma_raw_detrended.csv")
        pd.DataFrame(effective,index=labels,columns=labels).to_csv(noise_folder/"sigma_quantization_sensitivity.csv")
        dump(noise_folder/"noise_assumptions.json",noise)
        # Identify thermal calibration residual bias without using held-out drives.
        thermal_cal_errors=[]; voltage_cal_errors=[]
        calibration_metrics=[]
        for cycle,frame in enumerate(calibration_frames,1):
            sim=simulate(frame.Time,frame.Current_discharge_positive,frame.Chamber_Temp_degC,p,ocv,
                         temperature0=float(frame.Battery_Temp_degC.iloc[0]))
            rt=sim.temperature_c.to_numpy()-frame.Battery_Temp_degC.to_numpy()
            rv=sim.voltage_v.to_numpy()-frame.Voltage.to_numpy()
            thermal_cal_errors.append(rt);voltage_cal_errors.append(rv)
            calibration_metrics.append(dict(cycle=cycle,voltage=metrics(sim.voltage_v,frame.Voltage),
                                            temperature=metrics(sim.temperature_c,frame.Battery_Temp_degC)))
        # Model-discrepancy sensitivity covariance comes from CALIBRATION only.
        discrepancy=effective.copy()
        discrepancy[1,1]+=float(np.mean(np.concatenate(voltage_cal_errors)**2))
        discrepancy[2,2]+=float(np.mean(np.concatenate(thermal_cal_errors)**2))
        pd.DataFrame(discrepancy,index=labels,columns=labels).to_csv(noise_folder/"sigma_model_error_sensitivity.csv")
        info=dict(temperature_c=temp,parameters=p,capacity_basis="2.9Ah nominal/protocol reference; not independently identified at each date",
                  ocv_method="monotone rested-HPPC voltage proxy; not independent C/20 calibration",
                  supported_soc=[float(ocv.soc.min()),float(ocv.soc.max())],
                  parameter_aggregation=selection,joint_fit_count=len(fits),aggregate_count=len(selected),
                  thermal=thermal,thermal_heat_basis="measured calibration current times (rested-voltage proxy minus measured voltage)",
                  calibration_metrics=calibration_metrics,
                  status="provisional_baseline_not_acceptance_certified",
                  initial_conditions="SOC=1 and polarization=0 after recharge; measured first cell temperature",
                  fault_interface="fi[A], fv[V], fT[C], fisc[S], fQ[fraction], fR[fraction], onset_s[s]",
                  missing_requirements=["independent date-specific capacity identification","independent 10C/0C C20 curves" if temp!=25 else "C20/rested-OCV/date reconciliation",
                                        "validated colored/quantized measurement likelihood","model-error acceptance review"])
        dump(parameters/f"{temp}C.json",info)
        parameter_rows.append(dict(temperature_c=temp,**p,aggregate_count=len(selected),
                                   joint_fit_rmse_median_mv=float(fit_table.rmse_mv.median()),
                                   thermal_integral_rmse_c=thermal["integral_rmse_c"],status=info["status"]))
        fig,axes=plt.subplots(1,3,figsize=(12,3.7))
        for ax,(number,t,v,pred) in zip(axes,examples):
            ax.plot(t,v,".",ms=2,label="Measured")
            ax.plot(t,pred,lw=.9,label="Joint fit")
            ax.set(xlabel="Time since first loaded sample (s)",ylabel="Voltage (V)",title=f"{temp} C pulse {number}")
            ax.set_xscale("symlog",linthresh=10)
            ax.grid(alpha=.25)
        axes[0].legend(fontsize=8)
        fig.tight_layout();fig.savefig(folder/"joint_pulse_examples.png",dpi=160);plt.close(fig)
        print(f"{temp}C: held-out US06/UDDS validation",flush=True)
        for cycle in ["US06","UDDS"]:
            frame=load(find(temp,f"{cycle}_Pan"),temp,"held_out_validation")
            sim=simulate(frame.Time,frame.Current_discharge_positive,frame.Chamber_Temp_degC,p,ocv,
                         temperature0=float(frame.Battery_Temp_degC.iloc[0]))
            ev=metrics(sim.voltage_v,frame.Voltage);et=metrics(sim.temperature_c,frame.Battery_Temp_degC)
            rt=sim.temperature_c.to_numpy()-frame.Battery_Temp_degC.to_numpy()
            rv=sim.voltage_v.to_numpy()-frame.Voltage.to_numpy()
            corr=float(np.corrcoef(sim.soc,rt)[0,1]) if np.std(rt)>1e-12 else 0.
            errors.append(dict(temperature_c=temp,cycle=cycle,n=len(frame),duration_s=float(frame.Time.iloc[-1]-frame.Time.iloc[0]),
                               voltage_rmse_mv=1000*ev["rmse"],voltage_mae_mv=1000*ev["mae"],voltage_bias_mv=1000*ev["bias"],voltage_max_abs_mv=1000*ev["max_abs"],
                               temperature_rmse_c=et["rmse"],temperature_mae_c=et["mae"],temperature_bias_c=et["bias"],temperature_max_abs_c=et["max_abs"],
                               temperature_error_soc_correlation=corr,
                               outside_ocv_support_fraction=float(sim.ocv_outside_supported_soc.mean()),
                               ambient_source=frame.attrs["quality_report"]["ambient_source"],
                               status="held_out_open_loop_baseline_not_acceptance_certified"))
            # Save 1-second trajectories for B; simulation itself used all original rows.
            keep=np.r_[True,np.floor(frame.Time.to_numpy()[1:])!=np.floor(frame.Time.to_numpy()[:-1])]
            trajectory=sim.loc[keep].copy().reset_index(drop=True)
            for name,col in [("measured_voltage_v","Voltage"),("measured_current_a","Current_discharge_positive"),
                             ("measured_temperature_c","Battery_Temp_degC"),("ambient_c","Chamber_Temp_degC")]:
                trajectory[name]=frame.loc[keep,col].to_numpy()
            trajectory.to_csv(validation_folder/f"trajectory_{cycle.lower()}_1s.csv",index=False)
            # These files are sampled for exchange; users must not confuse them with
            # averaged/resampled inputs for rerunning the full-resolution simulation.
            t=frame.Time.to_numpy()/3600
            stride=max(1,len(t)//7000)
            fig,ax=plt.subplots(2,2,figsize=(12,7),sharex=True)
            ax[0,0].plot(t[::stride],frame.Voltage.iloc[::stride],lw=.6,label="Measured")
            ax[0,0].plot(t[::stride],sim.voltage_v.iloc[::stride],lw=.7,label="Predicted")
            ax[1,0].plot(t[::stride],rv[::stride]*1000,lw=.6)
            ax[0,1].plot(t[::stride],frame.Battery_Temp_degC.iloc[::stride],lw=.6,label="Measured")
            ax[0,1].plot(t[::stride],sim.temperature_c.iloc[::stride],lw=.8,label="Predicted")
            ax[1,1].plot(t[::stride],rt[::stride],lw=.6)
            ax[0,0].set_ylabel("Voltage (V)");ax[0,1].set_ylabel("Temperature (C)")
            ax[1,0].set_ylabel("Voltage error (mV)");ax[1,1].set_ylabel("Temperature error (C)")
            for aa in ax.flat: aa.grid(alpha=.25)
            for aa in ax[0]: aa.legend(fontsize=8)
            for aa in ax[1]: aa.set_xlabel("Time (h)");aa.axhline(0,color="black",lw=.5)
            fig.suptitle(f"{temp} C {cycle}: held-out open-loop baseline")
            fig.tight_layout();fig.savefig(validation_folder/f"validation_{cycle.lower()}.png",dpi=160);plt.close(fig)
        if thermal["at_bound"]: issues.append(f"{temp}C thermal integral estimate hits bound")
        if noise["current_noise_unidentified"]: issues.append(f"{temp}C resting-current noise not identifiable from logged zeros")
    pd.DataFrame(parameter_rows).to_csv(summary/"parameter_table.csv",index=False)
    pd.DataFrame(errors).to_csv(summary/"model_error_table.csv",index=False)
    pd.DataFrame(all_fit_rows).to_csv(out/"calibration"/"joint_pulse_fits_all.csv",index=False)
    dump(summary/"data_manifest.json",manifest)
    dump(summary/"run_status.json",dict(status="baseline_implemented_acceptance_not_complete",issues=issues,
        global_gates=["independent low-temperature C20 unavailable; rested-HPPC voltage remains a labeled substitute",
                      "capacity/date reconciliation","systematic model-error review","quantized/colored noise likelihood"],
        calibration_validation_overlap=bool(set(m["sha256"] for m in manifest if "calibration" in m["role"])&set(m["sha256"] for m in manifest if m["role"]=="held_out_validation"))))
    print(pd.DataFrame(parameter_rows).to_string(index=False),flush=True)
    print(pd.DataFrame(errors).to_string(index=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args()
    target=args.output_dir or args.project_root/"results"/"week2"
    run(args.project_root.resolve(),target)
    from experiments.comparisons.compare_parameter_maps import run as compare_maps
    compare_maps(args.project_root.resolve(),target)
    from experiments.comparisons.compare_entropy_heat import run as compare_entropy
    compare_entropy(args.project_root.resolve(),target)
    from tools.audit_dataset import audit as audit_dataset
    audit_dataset(args.project_root.resolve(),target/"audit")
    from experiments.refine_week2 import run as refine
    refine(args.project_root.resolve(),target)
    from tools.close_week2_checks import run as close_checks
    close_checks(args.project_root.resolve(),target)
    from tools.build_week2_report import run as summarize
    summarize(target)
