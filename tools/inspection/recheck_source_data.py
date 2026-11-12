"""Recheck raw schemas and slow-rate segments without the project loader."""
import argparse
import json
from pathlib import Path
import re
from zipfile import ZipFile
import numpy as np
import pandas as pd
from scipy.io import loadmat


def run(root,out):
    raw=root/"data"/"raw";out.mkdir(parents=True,exist_ok=True)
    rows=[];segments=[];schemas={};archives=[]
    for path in sorted(raw.rglob("*.mat")):
        cell="Panasonic_18650PF" if "Panasonic_18650PF" in path.parts else "LG_18650HG2"
        record=loadmat(path,simplify_cells=True)
        top=[k for k in record if not k.startswith("__")]
        m=record.get("meas")
        if not isinstance(m,dict):
            rows.append(dict(cell=cell,file=str(path.relative_to(root)),status="no_meas_structure",top_variables=";".join(top)))
            continue
        fields=sorted(m)
        schema_key=cell+":"+";".join(fields)
        schemas.setdefault(schema_key,dict(cell=cell,fields=fields,top_variables=top,files=0))["files"]+=1
        t=np.asarray(m["Time"],dtype=float).reshape(-1)
        i=np.asarray(m["Current"],dtype=float).reshape(-1)
        v=np.asarray(m["Voltage"],dtype=float).reshape(-1)
        ah=np.asarray(m["Ah"],dtype=float).reshape(-1)
        temp=np.asarray(m["Battery_Temp_degC"],dtype=float).reshape(-1)
        ambient=m.get("Chamber_Temp_degC")
        if ambient is None:
            ambient_count=0;ambient_status="column_absent";ambient_array=np.full(len(t),np.nan)
        else:
            ambient_array=np.asarray(ambient,dtype=float).reshape(-1)
            ambient_count=int(np.isfinite(ambient_array).sum())
            ambient_status="all_NaN_or_Inf" if ambient_count==0 else "all_finite" if ambient_count==len(ambient_array) else "partly_missing"
        group=next((part for part in path.parts if re.fullmatch(r"(?:n|-)?\d+degC(?: Trise(?: with pause)?)?",part)),"unknown")
        aliases=[k for k in fields if re.search(r"temp|chamb|amb",k,re.I)]
        rows.append(dict(cell=cell,file=str(path.relative_to(root)),status="raw_meas_checked",group=group,
                         samples=len(t),ambient_column=ambient_status,ambient_finite_values=ambient_count,
                         temperature_related_fields=";".join(aliases),top_variables=";".join(top),
                         name_mentions_c20=bool(re.search(r"c[_ /-]?20|ocv",path.name,re.I))))
        if len(t)<3:continue
        slow=(np.abs(i)>=.05)&(np.abs(i)<=.3)
        positive_dt=np.diff(t);positive_dt=positive_dt[positive_dt>0]
        typical=float(np.median(positive_dt)) if len(positive_dt) else 1.
        breaks=(np.diff(slow)!=0)|(np.diff(t)>max(15,3*typical))|(np.diff(t)<0)
        bounds=np.r_[0,np.flatnonzero(breaks)+1,len(t)]
        for a,z in zip(bounds[:-1],bounds[1:]):
            if not slow[a] or z-a<20 or t[z-1]-t[a]<3600:continue
            current=i[a:z]
            if np.std(current)>.05*abs(np.mean(current)):continue
            voltage_change=float(v[z-1]-v[a])
            if abs(voltage_change)<.5:continue
            measured_ambient=ambient_array[a:z];measured_ambient=measured_ambient[np.isfinite(measured_ambient)]
            segments.append(dict(cell=cell,file=str(path.relative_to(root)),group=group,
                                  kind="slow_discharge" if voltage_change<0 else "slow_charge",
                                  first_row=a,last_row=z-1,duration_h=float((t[z-1]-t[a])/3600),
                                  current_mean_raw_a=float(np.mean(current)),voltage_start_v=float(v[a]),voltage_end_v=float(v[z-1]),
                                  absolute_ah_change=float(abs(ah[z-1]-ah[a])),
                                  cell_temperature_min_c=float(np.min(temp[a:z])),cell_temperature_max_c=float(np.max(temp[a:z])),
                                  cell_temperature_median_c=float(np.median(temp[a:z])),
                                  ambient_finite_values=len(measured_ambient),
                                  ambient_median_c=float(np.median(measured_ambient)) if len(measured_ambient) else None))
    for path in raw.rglob("*.zip"):
        with ZipFile(path) as archive:
            names=archive.namelist()
            archives.append(dict(file=str(path.relative_to(root)),members=len(names),
                                 matching_members=[name for name in names if re.search(r"c20|ocv|readme|chamb",name,re.I)],
                                 examples=names[:15]))
    pd.DataFrame(rows).to_csv(out/"source_data_recheck.csv",index=False)
    pd.DataFrame(segments).to_csv(out/"slow_rate_segments_recheck.csv",index=False)
    (out/"raw_schema_and_archives_recheck.json").write_text(json.dumps(dict(schemas=list(schemas.values()),archives=archives),indent=2,ensure_ascii=False),encoding="utf-8")
    print(pd.DataFrame(rows).groupby(["cell","status"]).size().to_string())
    print(pd.DataFrame(segments)[["cell","group","kind","duration_h","current_mean_raw_a","cell_temperature_median_c","file"]].to_string(index=False))
    print(json.dumps(archives,indent=2,ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--project-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args();run(args.project_root,args.output_dir)
