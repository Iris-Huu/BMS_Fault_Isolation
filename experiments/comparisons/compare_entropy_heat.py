"""Test a calibration-only effective entropy-heat correction; retain baseline."""
import argparse
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
from src.calibration import fit_thermal_entropy


def run(root,baseline):
    raw=root/"data"/"raw"/"Panasonic_18650PF"
    destination=baseline/"comparisons"/"entropy_sensitivity";destination.mkdir(parents=True,exist_ok=True)
    rows=[];thermal_rows=[]
    fig,axes=plt.subplots(3,2,figsize=(12,10))
    for row,temp in enumerate([25,10,0]):
        folder=destination/f"{temp}C";folder.mkdir(exist_ok=True)
        info=json.loads((baseline/"parameters"/f"{temp}C.json").read_text(encoding="utf-8"))
        p=info["parameters"].copy()
        ocv=pd.read_csv(baseline/"parameters"/f"ocv_{temp}C.csv")
        records=[]
        for cycle in [1,2]:
            source=next(raw.rglob(f"* {temp}degC_Cycle_{cycle}_Pan18650PF.mat"))
            frame=load_panasonic_data(source,ambient_setpoint_c=float(temp))
            soc=1+(frame.Ah.to_numpy()-frame.Ah.iloc[0])/p["Q_Ah"]
            current=frame.Current_discharge_positive.to_numpy()
            heat=current*(np.interp(soc,ocv.soc,ocv.ocv_v)-frame.Voltage.to_numpy())
            records.append((frame.Time.to_numpy(),frame.Battery_Temp_degC.to_numpy(),frame.Chamber_Temp_degC.to_numpy(),heat,soc,current))
        thermal=fit_thermal_entropy(records)
        p.update({k:thermal[k] for k in ["Cth_J_K","hA_W_K","entropy_k0_v_k","entropy_k1_v_k"]})
        thermal_rows.append(dict(temperature_c=temp,**thermal))
        info["parameters"]=p
        info["entropy_thermal_fit"]=thermal
        info["status"]="effective_entropy_sensitivity_not_thermodynamically_validated"
        (folder/"parameters.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
        for col,cycle in enumerate(["US06","UDDS"]):
            print(f"Entropy sensitivity: {temp}C {cycle}",flush=True)
            source=next(raw.rglob(f"* {temp}degC_{cycle}_Pan18650PF.mat"))
            frame=load_panasonic_data(source,ambient_setpoint_c=float(temp))
            sim=simulate(frame.Time,frame.Current_discharge_positive,frame.Chamber_Temp_degC,p,ocv,
                         temperature0=float(frame.Battery_Temp_degC.iloc[0]))
            e=sim.temperature_c.to_numpy()-frame.Battery_Temp_degC.to_numpy()
            v=sim.voltage_v.to_numpy()-frame.Voltage.to_numpy()
            rows.append(dict(temperature_c=temp,cycle=cycle,voltage_rmse_mv=float(np.sqrt(np.mean(v*v))*1000),
                             temperature_rmse_c=float(np.sqrt(np.mean(e*e))),temperature_bias_c=float(e.mean()),
                             temperature_max_abs_c=float(abs(e).max()),
                             temperature_error_soc_correlation=float(np.corrcoef(sim.soc,e)[0,1])))
            t=frame.Time.to_numpy()/3600;stride=max(1,len(t)//5000)
            ax=axes[row,col]
            ax.plot(t[::stride],frame.Battery_Temp_degC.iloc[::stride],lw=.65,label="Measured")
            ax.plot(t[::stride],sim.temperature_c.iloc[::stride],lw=.8,label="Effective entropy model")
            ax.set(title=f"{temp} C {cycle}",xlabel="Time (h)",ylabel="Temperature (C)")
            ax.grid(alpha=.25)
            if row==0:ax.legend(fontsize=8)
    pd.DataFrame(rows).to_csv(destination/"model_error_table.csv",index=False)
    pd.DataFrame(thermal_rows).to_csv(destination/"thermal_parameter_table.csv",index=False)
    fig.tight_layout();fig.savefig(destination/"temperature_validation_all.png",dpi=160);plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--project-root",type=Path,required=True)
    parser.add_argument("--baseline-dir",type=Path,required=True)
    args=parser.parse_args();run(args.project_root,args.baseline_dir)
