"""Close Week 2 audit, calibration and uncertainty checks with existing data.

Fits Cycle1/2 only. US06/UDDS remain development validation, never fit targets.
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
from src.calibration import (prepare_dynamic_calibration,fit_dynamic_electrical,calibrated_heat,
                             fit_thermal_offset,pulse_windows)


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")


def run(root,out,temperatures=(25,10,0)):
    raw=root/"data"/"raw"/"Panasonic_18650PF"
    source_results=out if (out/"parameters"/"25C.json").exists() else root/"results"/"week2"
    destination=out/"comparisons"/"calibration_refinement"
    destination.mkdir(parents=True,exist_ok=True)
    params=[json.loads((source_results/"parameters"/f"{t}C.json").read_text(encoding="utf-8"))["parameters"] for t in [0,10,25]]
    reference_temperatures=np.array([0.,10.,25.])
    activations=[]
    for key in ["R0_ohm","R1_ohm","R2_ohm"]:
        coefficient=np.polyfit(1/(reference_temperatures+273.15),np.log([p[key] for p in params]),1)[0]
        activations.append(float(np.clip(coefficient,0,6000)))
    errors=[];calibration_errors=[];provenance=[]
    if set(temperatures)!={0,10,25}:
        errors=pd.read_csv(destination/"model_error_table.csv").query("temperature_c not in @temperatures").to_dict("records")
        calibration_errors=pd.read_csv(destination/"calibration_error_table.csv").query("temperature_c not in @temperatures").to_dict("records")
        provenance=[m for m in json.loads((destination/"source_manifest.json").read_text(encoding="utf-8"))
                    if not any(f" {temp}degC_" in m["file"] for temp in temperatures)]
    def source(temp,kind):
        choices=list(raw.rglob(f"* {temp}degC_{kind}*.mat"))
        choices=[p for p in choices if "trise" not in str(p).lower()]
        if len(choices)!=1:raise ValueError(choices)
        return choices[0]
    def load(path,temp,role):
        frame=load_panasonic_data(path,ambient_setpoint_c=float(temp))
        provenance.append(dict(file=str(path.relative_to(root)),role=role,
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               ambient_source=frame.attrs["quality_report"]["ambient_source"]))
        return frame
    for temp in temperatures:
        print(f"{temp}C: fit calibration-cycle electrical dynamics and readout response",flush=True)
        folder=destination/f"{temp}C";folder.mkdir(exist_ok=True)
        original=json.loads((source_results/"parameters"/f"{temp}C.json").read_text(encoding="utf-8"))
        p=original["parameters"].copy()
        ocv=pd.read_csv(source_results/"parameters"/f"ocv_{temp}C.csv")
        fits=pd.read_csv(source_results/"calibration"/f"{temp}C"/"joint_pulse_fits.csv").fillna({"flags":""})
        good=fits[(fits["flags"]=="")&(fits.rmse_mv<10)]
        groups=good.assign(soc_bin=(good.soc_reference/.05).round()*.05).groupby("soc_bin").median(numeric_only=True)
        prior=np.concatenate([np.interp(np.linspace(0,1,6),groups.index,groups[key]) for key in ["R0_ohm","R1_ohm","R2_ohm"]])
        prior=np.clip(prior,.0001,.3)
        tau1=p["R1_ohm"]*p["C1_F"];tau2=p["R2_ohm"]*p["C2_F"]
        frames=[load(source(temp,f"Cycle_{cycle}_Pan18650PF"),temp,f"calibration_cycle_{cycle}") for cycle in [1,2]]
        excluded=[]
        for k,frame in enumerate(frames):
            within=1+(frame.Ah-frame.Ah.iloc[0])/2.9>=float(ocv.soc.min())+.005
            excluded.append(int((~within).sum()))
            if not within.all():
                # Crop the unsupported tail; a clipped OCV curve must not be
                # "explained" by inventing huge low-SOC resistances.
                first=int(np.flatnonzero(~within.to_numpy())[0])
                frames[k]=frame.iloc[:first].copy()
        records=[prepare_dynamic_calibration(frame,ocv,tau1,tau2,temp,activations) for frame in frames]
        model,blend,scores=fit_dynamic_electrical(records,prior,temp,activations,tau1,tau2)
        p["resistance_soc_model"]=model;p["voltage_readout_blend"]=blend
        pd.DataFrame(scores).to_csv(folder/"readout_response_selection.csv",index=False)
        # Offset prior from low-current, long HPPC rest tails, not validation temperature.
        hp=[q for q in raw.rglob(f"* {temp}degC_5*HPPC*.mat")]
        if len(hp)!=1:raise ValueError(hp)
        hppc=load(hp[0],temp,"hppc_noise_and_thermal_offset_prior")
        windows=pulse_windows(hppc)
        offsets=[]
        for w in windows:
            if w["current_a"]>3.5 or w["rest_s"]<600:continue
            tail=hppc.iloc[w["stop"]:w["end"]]
            tail=tail[tail.Time>=tail.Time.iloc[-1]-120]
            offsets.append(float(np.median(tail.Battery_Temp_degC-tail.Chamber_Temp_degC)))
        offset_prior=float(np.median(offsets))
        heats=[calibrated_heat(record,model) for record in records]
        thermal=fit_thermal_offset(list(zip(records,heats)),offset_prior)
        p.update({k:thermal[k] for k in ["Cth_J_K","hA_W_K","effective_ambient_offset_c"]})
        p.pop("entropy_k0_v_k",None);p.pop("entropy_k1_v_k",None)
        info=dict(parameters=p,temperature_c=temp,thermal_calibration=thermal,
                  capacity_coordinate="reference=2.9Ah; OCV is a function of withdrawn Ah, physical capacity is not inferred",
                  current_voltage_response="causal one-sample physical-voltage blend fitted using Cycle1, selected using Cycle2",
                  observation_parameter_is_not_a_proven_hardware_delay=True,
                  source_ocv=f"parameters/ocv_{temp}C.csv",status="development_validation_with_explicit_uncertainty")
        info["calibration_rows_excluded_below_ocv_support"]=excluded
        info["calibration_ocv_soc_lower_limit"]=float(ocv.soc.min())+.005
        dump(folder/"parameters.json",info)
        for cycle,frame in enumerate(frames,1):
            sim=simulate(frame.Time,frame.Current_discharge_positive,frame.Chamber_Temp_degC,p,ocv,
                         temperature0=float(frame.Battery_Temp_degC.iloc[0]))
            ev=sim.voltage_v.to_numpy()-frame.Voltage.to_numpy()
            et=sim.temperature_c.to_numpy()-frame.Battery_Temp_degC.to_numpy()
            calibration_errors.append(dict(temperature_c=temp,cycle=cycle,voltage_rmse_mv=float(np.sqrt(np.mean(ev**2))*1000),
                                           temperature_rmse_c=float(np.sqrt(np.mean(et**2)))))
        print(f"{temp}C: readout blend={blend}, Cth={p['Cth_J_K']:.2f}, hA={p['hA_W_K']:.3f}, offset={p['effective_ambient_offset_c']:.3f}; validate",flush=True)
        for cycle in ["US06","UDDS"]:
            frame=load(source(temp,f"{cycle}_Pan18650PF"),temp,"development_validation")
            sim=simulate(frame.Time,frame.Current_discharge_positive,frame.Chamber_Temp_degC,p,ocv,
                         temperature0=float(frame.Battery_Temp_degC.iloc[0]))
            ev=sim.voltage_v.to_numpy()-frame.Voltage.to_numpy()
            et=sim.temperature_c.to_numpy()-frame.Battery_Temp_degC.to_numpy()
            errors.append(dict(temperature_c=temp,cycle=cycle,n=len(frame),
                               voltage_rmse_mv=float(np.sqrt(np.mean(ev**2))*1000),voltage_bias_mv=float(ev.mean()*1000),
                               voltage_max_abs_mv=float(abs(ev).max()*1000),temperature_rmse_c=float(np.sqrt(np.mean(et**2))),
                               temperature_bias_c=float(et.mean()),temperature_max_abs_c=float(abs(et).max()),
                               temperature_error_soc_correlation=float(np.corrcoef(sim.soc,et)[0,1]),
                               ambient_source=frame.attrs["quality_report"]["ambient_source"]))
            keep=np.r_[True,np.floor(frame.Time.to_numpy()[1:])!=np.floor(frame.Time.to_numpy()[:-1])]
            trajectory=sim.loc[keep].copy().reset_index(drop=True)
            for key,col in [("measured_voltage_v","Voltage"),("measured_temperature_c","Battery_Temp_degC"),
                            ("measured_current_a","Current_discharge_positive"),("ambient_c","Chamber_Temp_degC")]:
                trajectory[key]=frame.loc[keep,col].to_numpy()
            trajectory.to_csv(folder/f"trajectory_{cycle.lower()}_1s.csv",index=False)
            t=frame.Time.to_numpy()/3600;stride=max(1,len(frame)//7000)
            fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
            for col,(observed,predicted,error,label) in enumerate([
                (frame.Voltage,sim.voltage_v,ev*1000,"Voltage (V)"),
                (frame.Battery_Temp_degC,sim.temperature_c,et,"Temperature (C)")]):
                axes[0,col].plot(t[::stride],observed.iloc[::stride],lw=.6,label="Measured")
                axes[0,col].plot(t[::stride],predicted.iloc[::stride],lw=.8,label="Predicted")
                axes[0,col].set_ylabel(label);axes[0,col].legend(fontsize=8)
                axes[1,col].plot(t[::stride],error[::stride],lw=.6)
                axes[1,col].set(xlabel="Time (h)",ylabel="Error (mV)" if col==0 else "Error (C)")
                axes[1,col].axhline(0,color="black",lw=.5)
            for axis in axes.flat:axis.grid(alpha=.25)
            fig.suptitle(f"{temp} C {cycle}: calibration-only refinement")
            fig.tight_layout();fig.savefig(folder/f"validation_{cycle.lower()}.png",dpi=160);plt.close(fig)
    pd.DataFrame(errors).to_csv(destination/"model_error_table.csv",index=False)
    pd.DataFrame(calibration_errors).to_csv(destination/"calibration_error_table.csv",index=False)
    dump(destination/"source_manifest.json",provenance)
    print(pd.DataFrame(errors).to_string(index=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--project-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--temperatures",type=int,nargs="+",choices=[0,10,25],default=[25,10,0])
    args=parser.parse_args();run(args.project_root,args.output_dir or args.project_root/"results"/"week2",args.temperatures)
