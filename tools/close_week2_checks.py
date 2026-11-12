"""Data lineage, SOC-coordinate equivalence, OCV evidence, noise and one-shot checks."""
import argparse
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from src.io import load_panasonic_data
from src.model import simulate
from src.calibration import pulse_windows
from src.noise import fit_empirical_rest_noise


def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")


def error_row(frame,sim):
    voltage=sim.voltage_v.to_numpy()-frame.Voltage.to_numpy()
    thermal=sim.temperature_c.to_numpy()-frame.Battery_Temp_degC.to_numpy()
    return dict(voltage_rmse_mv=float(np.sqrt(np.mean(voltage**2))*1000),
                voltage_bias_mv=float(voltage.mean()*1000),temperature_rmse_c=float(np.sqrt(np.mean(thermal**2))),
                temperature_bias_c=float(thermal.mean()))


def run(root,out,temperatures=(25,10,0)):
    source_results=out if (out/"parameters"/"25C.json").exists() else root/"results"/"week2"
    refined=out/"comparisons"/"calibration_refinement"
    audit=out/"audit";audit.mkdir(parents=True,exist_ok=True)
    summary=out/"summary";summary.mkdir(exist_ok=True)
    raw=root/"data"/"raw"/"Panasonic_18650PF"
    inventory=pd.read_csv(audit/"dataset_inventory.csv")
    original_manifest=json.loads((source_results/"summary"/"data_manifest.json").read_text(encoding="utf-8"))
    lineage=inventory[inventory.file.isin([m["file"] for m in original_manifest])].copy()
    lineage["epoch"]=np.where(lineage.actual_start_date<"2017-05-01","March_April",
                              np.where(lineage.actual_start_date<"2017-07-01","May_June","July_later"))
    lineage.to_csv(audit/"model_data_lineage.csv",index=False)
    noise_rows=[];ocv_rows=[];confirmations=[];coverage_rows=[]
    if set(temperatures)!={0,10,25}:
        noise_rows=pd.read_csv(summary/"noise_holdout_coverage.csv").query("temperature_c not in @temperatures").to_dict("records")
        ocv_rows=pd.read_csv(audit/"ocv_evidence.csv").query("temperature_c not in @temperatures").to_dict("records")
        confirmations=pd.read_csv(refined/"confirmation_error_table.csv").query("temperature_c not in @temperatures").to_dict("records")
        coverage_rows=pd.read_csv(refined/"uncertainty_coverage.csv").query("temperature_c not in @temperatures").to_dict("records")
    for temp in temperatures:
        print(f"{temp}C: OCV cross-validation, empirical-noise holdout, independent Cycle3/4 checks",flush=True)
        hp=list(raw.rglob(f"* {temp}degC_5*HPPC*.mat"))
        hppc=load_panasonic_data(hp[0],ambient_setpoint_c=float(temp))
        windows=pulse_windows(hppc)
        model,samples,boundaries=fit_empirical_rest_noise(hppc,windows)
        noise_dir=out/"noise"/f"{temp}C";noise_dir.mkdir(parents=True,exist_ok=True)
        dump(noise_dir/"empirical_colored_noise.json",model)
        baseline_assumptions=noise_dir/"noise_assumptions.json"
        if baseline_assumptions.exists():
            old=json.loads(baseline_assumptions.read_text(encoding="utf-8"))
            old["baseline_surrogate_only"]=True
            old["primary_empirical_noise_model"]="empirical_colored_noise.json"
            dump(baseline_assumptions,old)
        np.savez_compressed(noise_dir/"empirical_rest_blocks.npz",samples=samples,boundaries=boundaries)
        for evaluation in model["point_and_block_coverage"]:
            noise_rows.append(dict(temperature_c=temp,**evaluation))
        # Leave complete pulse groups out, not adjacent nearly-identical points.
        group=0;mapping={};last_current=0.
        for w in windows:
            if w["current_a"]<2 and last_current>2:group+=1
            mapping[w["pulse"]]=group;last_current=w["current_a"]
        points=pd.read_csv(source_results/"calibration"/f"{temp}C"/"ocv_source_points.csv")
        points["group"]=points.pulse.map(mapping)
        cv=[]
        for g in sorted(points.group.unique()):
            train=points[points.group!=g].groupby("soc",as_index=False).ocv_v.median().sort_values("soc")
            test=points[points.group==g]
            values=IsotonicRegression(increasing=True).fit_transform(train.soc,train.ocv_v)
            covered=test.soc.between(train.soc.min(),train.soc.max())
            for record in test[covered].itertuples():
                estimate=float(np.interp(record.soc,train.soc,values))
                cv.append(dict(group=int(g),soc=float(record.soc),error_mv=1000*(estimate-record.ocv_v)))
        cv=pd.DataFrame(cv)
        folder=refined/f"{temp}C"
        cv.to_csv(folder/"ocv_group_crossvalidation.csv",index=False)
        ocv_rows.append(dict(temperature_c=temp,independent_c20_available=temp==25,
                            held_out_groups=int(cv.group.nunique()),covered_points=len(cv),
                            group_cv_rmse_mv=float(np.sqrt(np.mean(cv.error_mv**2))),
                            group_cv_max_abs_mv=float(abs(cv.error_mv).max()),
                            interpretation="HPPC rested-voltage interpolation across held-out SOC groups, not independent equilibrium OCV accuracy"))
        info=json.loads((folder/"parameters.json").read_text(encoding="utf-8"));p=info["parameters"]
        ocv=pd.read_csv(source_results/"parameters"/f"ocv_{temp}C.csv")
        # OCVC query table uses the SAME withdrawn-charge coordinate as dynamics.
        grid=np.linspace(float(ocv.soc.min()),1,1001)
        curve=np.interp(grid,ocv.soc,ocv.ocv_v)+np.interp(grid,p["resistance_soc_model"]["soc_knots"],p["resistance_soc_model"]["ocv_correction_v"])
        pd.DataFrame(dict(soc_reference=grid,withdrawn_charge_ah=2.9*(1-grid),effective_ocv_v=curve)).to_csv(folder/"effective_ocv_charge_coordinate.csv",index=False)
        def find(kind):
            choices=[q for q in raw.rglob(f"* {temp}degC_{kind}_Pan18650PF.mat") if "trise" not in str(q).lower()]
            if len(choices)!=1:raise ValueError(choices)
            return choices[0]
        # Cycle3 calibrates prediction-error bands only, not electrical/thermal parameters.
        calibration=load_panasonic_data(find("Cycle_3"),ambient_setpoint_c=float(temp))
        calibration_sim=simulate(calibration.Time,calibration.Current_discharge_positive,calibration.Chamber_Temp_degC,p,ocv,
                                 temperature0=float(calibration.Battery_Temp_degC.iloc[0]))
        cal_v=calibration_sim.voltage_v.to_numpy()-calibration.Voltage.to_numpy()
        cal_t=calibration_sim.temperature_c.to_numpy()-calibration.Battery_Temp_degC.to_numpy()
        cal_bins=np.clip((calibration_sim.soc.to_numpy()*5).astype(int),0,4)
        widths=[]
        for bin_index in range(5):
            mask=cal_bins==bin_index
            if mask.sum()<100:mask=np.ones(len(calibration),dtype=bool)
            widths.append([float(np.quantile(abs(cal_v[mask]),.95)),float(np.quantile(abs(cal_t[mask]),.95))])
        dump(folder/"prediction_error_bands.json",dict(source="Cycle3 only",soc_bins=[0,.2,.4,.6,.8,1.],
             pointwise_absolute_error_95_quantiles=widths,units=["V","C"],
             limitations="Empirical bands under changing operating conditions; serial dependence prevents an iid coverage guarantee."))
        for kind in ["Cycle_4","US06","UDDS"]:
            frame=load_panasonic_data(find(kind),ambient_setpoint_c=float(temp))
            sim=simulate(frame.Time,frame.Current_discharge_positive,frame.Chamber_Temp_degC,p,ocv,
                         temperature0=float(frame.Battery_Temp_degC.iloc[0]))
            if kind=="Cycle_4":confirmations.append(dict(temperature_c=temp,cycle=kind,**error_row(frame,sim)))
            bins=np.clip((sim.soc.to_numpy()*5).astype(int),0,4)
            band=np.asarray(widths)[bins]
            vv=sim.voltage_v.to_numpy();tt=sim.temperature_c.to_numpy()
            lower_v=vv-band[:,0];upper_v=vv+band[:,0]
            lower_t=tt-band[:,1];upper_t=tt+band[:,1]
            # A +/-0.5C effective-ambient profile is sensitivity, NOT a confidence bound.
            for variant in info["thermal_calibration"]["sensitivity_profiles"]:
                if abs(variant["effective_ambient_offset_c"]-p["effective_ambient_offset_c"])<1e-10:continue
                altered=dict(p,**variant)
                sensitivity=simulate(frame.Time,frame.Current_discharge_positive,frame.Chamber_Temp_degC,altered,ocv,
                                     temperature0=float(frame.Battery_Temp_degC.iloc[0]))
                lower_v=np.minimum(lower_v,sensitivity.voltage_v.to_numpy()-band[:,0])
                upper_v=np.maximum(upper_v,sensitivity.voltage_v.to_numpy()+band[:,0])
                lower_t=np.minimum(lower_t,sensitivity.temperature_c.to_numpy()-band[:,1])
                upper_t=np.maximum(upper_t,sensitivity.temperature_c.to_numpy()+band[:,1])
            measured_v=frame.Voltage.to_numpy();measured_t=frame.Battery_Temp_degC.to_numpy()
            coverage_rows.append(dict(temperature_c=temp,cycle=kind,band_source="Cycle3 95% residual quantiles + calibrated ambient-offset profiles",
                                      voltage_coverage=float(np.mean((measured_v>=lower_v)&(measured_v<=upper_v))),
                                      temperature_coverage=float(np.mean((measured_t>=lower_t)&(measured_t<=upper_t))),
                                      median_voltage_halfwidth_mv=float(np.median((upper_v-lower_v)/2)*1000),
                                      median_temperature_halfwidth_c=float(np.median((upper_t-lower_t)/2))))
            if kind=="US06":
                t=frame.Time.to_numpy()/3600;stride=max(1,len(t)//4000)
                fig,axes=plt.subplots(1,2,figsize=(12,4))
                for ax,actual,pred,lo,hi,label in zip(axes,[measured_v,measured_t],[vv,tt],
                        [lower_v,lower_t],[upper_v,upper_t],["Voltage (V)","Temperature (C)"]):
                    ax.fill_between(t[::stride],lo[::stride],hi[::stride],alpha=.25,label="Empirical + sensitivity envelope")
                    ax.plot(t[::stride],actual[::stride],lw=.6,label="Measured")
                    ax.plot(t[::stride],pred[::stride],lw=.7,label="Predicted")
                    ax.set(xlabel="Time (h)",ylabel=label);ax.grid(alpha=.25)
                axes[0].legend(fontsize=7)
                fig.tight_layout();fig.savefig(folder/"us06_uncertainty_envelope.png",dpi=160);plt.close(fig)
    pd.DataFrame(ocv_rows).to_csv(audit/"ocv_evidence.csv",index=False)
    pd.DataFrame(noise_rows).to_csv(summary/"noise_holdout_coverage.csv",index=False)
    pd.DataFrame(confirmations).to_csv(refined/"confirmation_error_table.csv",index=False)
    pd.DataFrame(coverage_rows).to_csv(refined/"uncertainty_coverage.csv",index=False)
    dump(audit/"soc_coordinate_policy.json",dict(reference_capacity_ah=2.9,
         equations=["q_removed(t) = integral Icell dt / 3600","soc_reference = 1-q_removed/2.9",
                    "change Qref to Qnew => soc_new=1-(Qref/Qnew)*(1-soc_reference); transform every OCV/parameter SOC knot identically"],
         conclusion="Healthy predictions depend on withdrawn charge and curve coordinates, not the arbitrary reference-capacity label, when transformed consistently.",
         physical_capacity_status="Incomplete low-temperature discharges cannot identify date-specific equilibrium capacity; usable 1C capacities are separate audited observations.",
         counterfactual_capacity_fault="A true capacity-change fault changes charge dynamics relative to fixed calibrated curves; it is not a coordinate relabeling."))
    print(pd.DataFrame(ocv_rows).to_string(index=False),flush=True)
    print(pd.DataFrame(confirmations).to_string(index=False),flush=True)
    print(pd.DataFrame(coverage_rows).to_string(index=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--project-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--temperatures",type=int,nargs="+",choices=[0,10,25],default=[25,10,0])
    args=parser.parse_args();run(args.project_root,args.output_dir,args.temperatures)
