"""Audit actual timestamps, capacity references and current integration conventions."""
import argparse
import hashlib
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat


def audit(root,output):
    raw=root/"data"/"raw"/"Panasonic_18650PF"
    output.mkdir(parents=True,exist_ok=True)
    inventory=[];capacities=[];timing=[]
    for path in sorted(raw.rglob("*.mat")):
        m=loadmat(path,simplify_cells=True).get("meas")
        if not isinstance(m,dict):continue
        t=np.asarray(m["Time"]).reshape(-1)
        i=-np.asarray(m["Current"]).reshape(-1)
        v=np.asarray(m["Voltage"]).reshape(-1)
        ah=np.asarray(m["Ah"]).reshape(-1)
        stamps=np.asarray(m.get("TimeStamp",[])).reshape(-1)
        actual_start=str(stamps[0]) if len(stamps) else ""
        actual_end=str(stamps[-1]) if len(stamps) else ""
        try:
            date=pd.to_datetime(actual_start,format="%m/%d/%Y %I:%M:%S %p").strftime("%Y-%m-%d")
        except (ValueError,TypeError):date="unknown"
        match=re.match(r"(\d\d)-(\d\d)-(\d\d)_",path.name)
        filename_date=f"20{match[3]}-{match[1]}-{match[2]}" if match else "unknown"
        ambient=np.asarray(m.get("Chamber_Temp_degC",np.full(len(t),np.nan)))
        temp=np.asarray(m["Battery_Temp_degC"])
        inventory.append(dict(file=str(path.relative_to(root)),actual_start=actual_start,actual_end=actual_end,
                              actual_start_date=date,filename_date=filename_date,date_mismatch=date!=filename_date,
                              samples=len(t),duration_s=float(t[-1]-t[0]),
                              ambient_missing_fraction=float(np.mean(~np.isfinite(ambient))),
                              cell_temperature_min_c=float(np.nanmin(temp)),cell_temperature_max_c=float(np.nanmax(temp)),
                              ah_start=float(ah[0]),ah_end=float(ah[-1]),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        on=i>.05
        regular_dt=np.diff(t);regular_dt=regular_dt[regular_dt>0]
        max_gap=max(15.,3*float(np.median(regular_dt))) if len(regular_dt) else 15.
        b=np.r_[0,np.flatnonzero((np.diff(on)!=0)|(np.diff(t)>max_gap)|(np.diff(ah)>.01))+1,len(t)]
        for a,z in zip(b[:-1],b[1:]):
            if z-a<20 or not on[a] or t[z-1]-t[a]<600:continue
            current=i[a:z]
            if np.std(current)>.05*max(float(np.mean(current)),.1):continue
            if v[a]<3.8 or v[z-1]>2.6:continue
            capacities.append(dict(file=str(path.relative_to(root)),date=date,
                                   current_a=float(np.mean(current)),start_voltage_v=float(v[a]),end_voltage_v=float(v[z-1]),
                                   capacity_ah=float(ah[a]-ah[z-1]),cell_temperature_start_c=float(temp[a]),
                                   cell_temperature_end_c=float(temp[z-1]),
                                   method="observed constant-current usable discharge span; not equilibrium capacity"))
        if any(f" {temp}degC_" in path.name for temp in [25,10,0]) and any(s in path.name for s in ["HPPC","Cycle_1_Pan","Cycle_2_Pan","US06_Pan","UDDS_Pan"]):
            keep=np.r_[True,np.diff(t)>0]
            tt,ii,aa=t[keep],i[keep],ah[keep]
            dt=np.diff(tt); valid=(dt>0)&(dt<2)&(np.diff(aa)<=.001)
            reference=-np.diff(aa)*3600
            for scheme,estimate in [("left",ii[:-1]*dt),("right",ii[1:]*dt),("trapezoid",.5*(ii[:-1]+ii[1:])*dt)]:
                error=(estimate-reference)[valid]
                timing.append(dict(file=str(path.relative_to(root)),scheme=scheme,valid_intervals=int(valid.sum()),
                                   charge_increment_rmse_as=float(np.sqrt(np.mean(error**2))),
                                   total_charge_difference_ah=float(np.sum(error)/3600)))
    pd.DataFrame(inventory).to_csv(output/"dataset_inventory.csv",index=False)
    pd.DataFrame(capacities).to_csv(output/"capacity_references.csv",index=False)
    pd.DataFrame(timing).to_csv(output/"current_integration_audit.csv",index=False)
    print("Audited MAT files:",len(inventory),"; date mismatches:",sum(x["date_mismatch"] for x in inventory))
    print(pd.DataFrame(capacities).to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--project-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args();audit(args.project_root,args.output_dir)
