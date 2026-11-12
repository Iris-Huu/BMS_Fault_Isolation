"""Compare fixed-temperature constants with HPPC-only SOC/temperature maps.

No parameters or thresholds are fitted to US06/UDDS. These datasets have
already been inspected during development, so they are development validation,
not an untouched blind test set.
"""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.io import load_panasonic_data
from src.model import simulate


def run(root,baseline):
    maps=[]
    raw=root/"data"/"raw"/"Panasonic_18650PF"
    for temp in [0,10,25]:
        table=pd.read_csv(baseline/"calibration"/f"{temp}C"/"joint_pulse_fits.csv").fillna({"flags":""})
        source=next(raw.rglob(f"* {temp}degC_5*HPPC*.mat"))
        hppc=load_panasonic_data(source,ambient_setpoint_c=float(temp))
        from src.calibration import pulse_windows
        windows={w["pulse"]:w for w in pulse_windows(hppc)}
        selected=table[(table["flags"]=="")&(table.rmse_mv<10)].copy()
        if len(selected)<3: raise ValueError("Insufficient well-behaved fits for parameter map")
        selected["soc_bin"]=(selected.soc_reference/.05).round()*.05
        keys=["R0_ohm","R1_ohm","R2_ohm","tau1_s","tau2_s"]
        grouped=selected.groupby("soc_bin",as_index=False)[keys].median().sort_values("soc_bin")
        temperatures=[float(hppc.Battery_Temp_degC.iloc[windows[int(k)]["start"]]) for k in selected.pulse]
        curve=dict(cell_temperature_c=float(np.median(temperatures)),ambient_temperature_c=temp,
                   soc=grouped.soc_bin.tolist(),source_pulses=selected.pulse.tolist(),
                   **{k:grouped[k].tolist() for k in keys})
        maps.append(curve)
    (baseline/"parameters"/"electrical_parameter_maps.json").write_text(json.dumps(maps,indent=2),encoding="utf-8")
    errors=[]
    destination=baseline/"comparisons"/"soc_temperature_maps"
    destination.mkdir(parents=True,exist_ok=True)
    for temp in [25,10,0]:
        folder=destination/f"{temp}C";folder.mkdir(exist_ok=True)
        info=json.loads((baseline/"parameters"/f"{temp}C.json").read_text(encoding="utf-8"))
        p=dict(info["parameters"],electrical_map=maps)
        info["parameters"]=p
        info["status"]="development_validation_baseline_not_acceptance_certified"
        info["electrical_map_notes"]="Interpolates HPPC-only fits over SOC and cell temperature; clamps outside support. Lookup resolution 0.001 SOC / 0.1 C. No voltage or temperature feedback after initial condition."
        (folder/"parameters.json").write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding="utf-8")
        ocv=pd.read_csv(baseline/"parameters"/f"ocv_{temp}C.csv")
        for cycle in ["US06","UDDS"]:
            print(f"SOC/T map: {temp}C {cycle}",flush=True)
            source=next(raw.rglob(f"* {temp}degC_{cycle}_Pan18650PF.mat"))
            frame=load_panasonic_data(source,ambient_setpoint_c=float(temp))
            sim=simulate(frame.Time,frame.Current_discharge_positive,frame.Chamber_Temp_degC,p,ocv,
                         temperature0=float(frame.Battery_Temp_degC.iloc[0]))
            rv=sim.voltage_v.to_numpy()-frame.Voltage.to_numpy()
            rt=sim.temperature_c.to_numpy()-frame.Battery_Temp_degC.to_numpy()
            delta=np.abs(np.diff(frame.Current_discharge_positive,prepend=frame.Current_discharge_positive.iloc[0]))
            # Supplemental diagnostic only; all-sample errors remain primary.
            stable=(delta<=2)&np.r_[True,delta[:-1]<=2]
            covered=(sim.temperature_c>=maps[0]["cell_temperature_c"])&(sim.temperature_c<=maps[-1]["cell_temperature_c"])
            error=dict(temperature_c=temp,cycle=cycle,n=len(frame),voltage_rmse_mv=float(np.sqrt(np.mean(rv**2))*1000),
                       voltage_mae_mv=float(np.mean(abs(rv))*1000),voltage_bias_mv=float(rv.mean()*1000),
                       voltage_max_abs_mv=float(np.max(abs(rv))*1000),
                       temperature_rmse_c=float(np.sqrt(np.mean(rt**2))),temperature_bias_c=float(rt.mean()),
                       temperature_max_abs_c=float(np.max(abs(rt))),
                       temperature_error_soc_correlation=float(np.corrcoef(sim.soc,rt)[0,1]),
                       supplemental_stable_voltage_rmse_mv=float(np.sqrt(np.mean(rv[stable]**2))*1000),
                       supplemental_stable_fraction=float(np.mean(stable)),
                       temperature_map_outside_support_fraction=float(1-np.mean(covered)),
                       outside_ocv_support_fraction=float(sim.ocv_outside_supported_soc.mean()),
                       ambient_source=frame.attrs["quality_report"]["ambient_source"])
            errors.append(error)
            worst=np.argsort(np.abs(rv))[-20:][::-1]
            context=frame.iloc[worst][["Time","Voltage","Current_discharge_positive","Ah","Battery_Temp_degC"]].copy()
            context["predicted_voltage_v"]=sim.voltage_v.iloc[worst].to_numpy()
            context["voltage_error_mv"]=rv[worst]*1000
            context["current_change_a"]=delta[worst]
            context.to_csv(folder/f"worst_voltage_errors_{cycle.lower()}.csv",index=False)
            keep=np.r_[True,np.floor(frame.Time.to_numpy()[1:])!=np.floor(frame.Time.to_numpy()[:-1])]
            trajectory=sim.loc[keep].copy().reset_index(drop=True)
            for name,col in [("measured_voltage_v","Voltage"),("measured_current_a","Current_discharge_positive"),
                             ("measured_temperature_c","Battery_Temp_degC"),("ambient_c","Chamber_Temp_degC")]:
                trajectory[name]=frame.loc[keep,col].to_numpy()
            trajectory.to_csv(folder/f"trajectory_{cycle.lower()}_1s.csv",index=False)
            t=frame.Time.to_numpy()/3600;stride=max(1,len(t)//7000)
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
            fig.suptitle(f"{temp} C {cycle}: HPPC SOC/temperature-map baseline")
            fig.tight_layout();fig.savefig(folder/f"validation_{cycle.lower()}.png",dpi=160);plt.close(fig)
    table=pd.DataFrame(errors)
    table.to_csv(destination/"model_error_table.csv",index=False)
    print(table.to_string(index=False),flush=True)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--project-root",type=Path,required=True)
    parser.add_argument("--baseline-dir",type=Path,required=True)
    args=parser.parse_args();run(args.project_root,args.baseline_dir)
