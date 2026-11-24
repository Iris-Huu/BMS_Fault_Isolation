"""Prepare fixed-model evidence and interfaces; do not recalibrate parameters."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.io import load_panasonic_data
from src.model import simulate
from src.preparation import (file_digest,rest_tail_metrics,hppc_rising_edges,lagged_autocovariance,
                             residual_noise_models,effective_sample_counts)


def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def fingerprint(path):return file_digest(path)


def metrics(frame, voltage, temperature):
    v=voltage-frame.Voltage.to_numpy();t=temperature-frame.Battery_Temp_degC.to_numpy()
    return dict(voltage_rmse_mv=float(np.sqrt(np.mean(v*v))*1000),voltage_bias_mv=float(v.mean()*1000),
                temperature_rmse_c=float(np.sqrt(np.mean(t*t))),temperature_bias_c=float(t.mean()))


def run(root,out):
    source=root/'results/week2'; raw=root/'data/raw/Panasonic_18650PF'
    local=root/'data/processed/week3_preparation';local.mkdir(parents=True,exist_ok=True)
    out.mkdir(parents=True,exist_ok=True)
    inventory=pd.read_csv(source/'audit/dataset_inventory.csv')
    inventory['end_clock']=pd.to_datetime(inventory.actual_end,format='%m/%d/%Y %I:%M:%S %p',errors='coerce')
    provenance={};ambient_rows=[];edge_rows=[];validation=[];thermal=[];noise_stats=[];ess=[]
    edge_figure,edge_axes=plt.subplots(1,3,figsize=(13,3.5))
    temperature_figure,temperature_axes=plt.subplots(3,2,figsize=(12,9))
    def record(path):provenance[path.relative_to(root).as_posix()]=fingerprint(path)
    def find(temp,kind):
        choices=[p for p in raw.rglob(f'* {temp}degC_{kind}_Pan18650PF.mat') if 'trise' not in str(p).lower()]
        if len(choices)!=1:raise ValueError(choices)
        return choices[0]
    for row_index,temp in enumerate([25,10,0]):
        print(f'{temp}C: audit HPPC and preceding rests',flush=True)
        hp=next(raw.rglob(f'* {temp}degC_5*HPPC*.mat'));record(hp)
        frame=load_panasonic_data(hp,ambient_setpoint_c=float(temp))
        edges=hppc_rising_edges(frame);edges.insert(0,'temperature_c',temp);edge_rows.extend(edges.to_dict('records'))
        for k in edges.row.iloc[:8]:
            sl=frame.iloc[int(k)-2:int(k)+4];x=sl.Time-sl.Time.iloc[2]
            drop=(sl.Voltage.iloc[1]-sl.Voltage)/(sl.Voltage.iloc[1]-sl.Voltage.iloc[3])
            edge_axes[row_index].plot(x,drop,'o-',lw=.7,alpha=.65)
        edge_axes[row_index].axvline(0,color='k',ls=':');edge_axes[row_index].set(title=f'{temp} C: pulse onset',xlabel='Time relative to current jump (s)',ylabel='Voltage drop / next-sample drop');edge_axes[row_index].grid(alpha=.2)
        param_path=source/f'comparisons/calibration_refinement/{temp}C/parameters.json';record(param_path)
        info=json.loads(param_path.read_text(encoding='utf-8'));p=deepcopy(info['parameters']);old_blend=p.get('voltage_readout_blend',0.)
        p['voltage_readout_blend']=0.
        frozen=dict(model_id='model-v1',parameters=p,temperature_c=temp,primary_result=temp in [25,10],
                    base_ocv_file=f'results/week2/parameters/ocv_{temp}C.csv',
                    forward_model='src/model.py::simulate',input_current_convention='discharge_positive_A',
                    source_parameters=param_path.relative_to(root).as_posix(),source_sha256=fingerprint(param_path),
                    change='remove unsupported readout blend; physical electrical and thermal parameters unchanged; no refit',
                    previous_readout_blend=old_blend,
                    ambient_nuisance=dict(parameter='delta_ambient_c',sensitivity_halfwidth_c=.35,
                                          interpretation='sensitivity range from feedback, not a confidence interval'))
        dump(out/f'model_v1/{temp}C.json',frozen)
        ocv_path=source/f'parameters/ocv_{temp}C.csv';record(ocv_path);ocv=pd.read_csv(ocv_path)
        rest_candidates=inventory[inventory.file.str.contains('Charges and Pauses',regex=False)&inventory.file.str.contains('\\'+f'{temp}degC'+'\\',regex=False)]
        for kind in ['Cycle_1','Cycle_2','Cycle_3','Cycle_4','US06','UDDS']:
            target=find(temp,kind);record(target);row=inventory[inventory.file==str(target.relative_to(root))].iloc[0]
            start=pd.to_datetime(row.actual_start,format='%m/%d/%Y %I:%M:%S %p')
            candidates=rest_candidates[(rest_candidates.end_clock<start)&(rest_candidates.end_clock>=start-pd.Timedelta(hours=12))].sort_values('end_clock',ascending=False)
            best=None
            for item in candidates.itertuples():
                path=root/item.file
                m=loadmat(path,simplify_cells=True)['meas']
                item_result=rest_tail_metrics(m,start)
                if item_result is not None:
                    best=dict(item_result,rest_source=item.file);record(path);break
            result=dict(temperature_c=temp,cycle=kind,target_source=target.relative_to(root).as_posix(),target_start=str(start),
                        **(best if best else dict(qualified=False,flags='no_terminal_rest_in_previous_12h',rest_source='')))
            ambient_rows.append(result)
        for kind in ['Cycle_3','US06','UDDS']:
            print(f'{temp}C: frozen prediction {kind}',flush=True)
            path=find(temp,kind);record(path);frame=load_panasonic_data(path,ambient_setpoint_c=float(temp))
            sim=simulate(frame.Time,frame.Current_discharge_positive,frame.Chamber_Temp_degC,p,ocv,
                         temperature0=float(frame.Battery_Temp_degC.iloc[0]))
            physical=sim.voltage_v.to_numpy();temperature=sim.temperature_c.to_numpy()
            former=(1-old_blend)*physical+old_blend*np.r_[physical[0],physical[:-1]]
            validation.append(dict(temperature_c=temp,cycle=kind,variant='accepted_week2',**metrics(frame,former,temperature)))
            validation.append(dict(temperature_c=temp,cycle=kind,variant='model_v1_no_delay',**metrics(frame,physical,temperature)))
            if kind=='Cycle_3':
                residual=np.c_[physical-frame.Voltage.to_numpy(),temperature-frame.Battery_Temp_degC.to_numpy()]
                trajectory=pd.DataFrame(dict(time_s=frame.Time,current_discharge_positive_a=frame.Current_discharge_positive,
                           measured_voltage_v=frame.Voltage,measured_temperature_c=frame.Battery_Temp_degC,
                           ambient_input_c=frame.Chamber_Temp_degC,reference_soc=sim.soc,
                           voltage_error_v=residual[:,0],temperature_error_c=residual[:,1]))
                file=local/f'{temp}C_cycle3_residuals.csv';trajectory.to_csv(file,index=False)
                # Original samples selected at about 1 s; do not interpolate across gaps.
                t=frame.Time.to_numpy();keep=np.r_[True,np.floor(t[1:])!=np.floor(t[:-1])]
                selected=residual[keep];selected_time=t[keep]
                cuts=np.r_[0,np.flatnonzero((np.diff(selected_time)>2)|(np.diff(selected_time)<=0))+1,len(selected_time)]
                gamma=lagged_autocovariance(selected,cuts,3600)
                sensor=pd.read_csv(source/f'noise/{temp}C/sigma_raw_detrended.csv',index_col=0).to_numpy()
                noise_path=source/f'noise/{temp}C/empirical_colored_noise.json';record(noise_path)
                proxy=json.loads(noise_path.read_text())['current_plateau_fluctuation_std_proxy_a'];sensor[0,0]=proxy**2
                model_cov,total_vt=residual_noise_models(gamma[0],sensor[1:,1:])
                total=sensor.copy();total[1:,1:]=total_vt
                gamma_total=gamma.copy();gamma_total[0]=total_vt
                folder=out/f'noise/{temp}C';folder.mkdir(parents=True,exist_ok=True)
                pd.DataFrame(sensor,index=['current_A','voltage_V','temperature_C'],columns=['current_A','voltage_V','temperature_C']).to_csv(folder/'sensor_surrogate_covariance.csv')
                pd.DataFrame(total,index=['current_A','voltage_V','temperature_C'],columns=['current_A','voltage_V','temperature_C']).to_csv(folder/'sensor_and_model_covariance.csv')
                np.savez_compressed(folder/'cycle3_colored_autocovariance.npz',gamma=gamma_total,lag_s=np.arange(3601))
                e=effective_sample_counts(gamma_total,[10,30,60,300,600,1800,3600]);e.insert(0,'temperature_c',temp);ess.extend(e.to_dict('records'))
                info=dict(channel_order=['current_A','voltage_V','temperature_C'],
                          colored_channel_order=['voltage_V','temperature_C'],
                          current_temporal_model='independent 1-second surrogate; current sensor not separately identified',
                          native_covariance_cadence_s=1.,
                          selected_sample_interval_quantiles_s=np.quantile(np.diff(selected_time),[.05,.5,.95]).tolist(),
                          original_samples=len(frame),selected_samples=len(selected),segments=cuts.tolist(),
                          bias_voltage_v=float(residual[:,0].mean()),bias_temperature_c=float(residual[:,1].mean()),
                          residual_file=file.relative_to(root).as_posix(),residual_sha256=fingerprint(file),
                          source_file=path.relative_to(root).as_posix(),source_sha256=fingerprint(path),
                          frozen_parameter_sha256=fingerprint(out/f'model_v1/{temp}C.json'),
                          lag_orientation='Gamma[k,a,b]=E[e_a(t)*e_b(t+k)]; negative lags use transpose',
                          covariance_construction='PSD positive part of Cycle3 residual covariance minus sensor covariance; avoids counting sensor variance twice',
                          model_discrepancy_covariance=model_cov.tolist(),
                          limitations=['Current variance is controller-ripple plus sensor proxy, not independently identified sensor noise.',
                                       'Global residual mean is retained separately as systematic bias, not whitened away as iid noise.',
                                       'Residual covariance is a stationary Cycle3 surrogate; transfer to rest and other cycles is unproven.',
                                       'Samples are selected without interpolation at about 1 s; sub-second covariance is not observed.',
                                       'Ambient perturbation must be treated as a nuisance separately; bias is not a measured ambient offset.'])
                dump(folder/'noise_interface.json',info);noise_stats.append(dict(temperature_c=temp,**metrics(frame,physical,temperature)))
            else:
                col=0 if kind=='US06' else 1;ax=temperature_axes[row_index,col];stride=max(1,len(frame)//4000)
                ax.plot(frame.Time.iloc[::stride]/3600,frame.Battery_Temp_degC.iloc[::stride],label='Measured',lw=.7)
                ax.plot(frame.Time.iloc[::stride]/3600,temperature[::stride],label='Frozen model',lw=.8)
                rest=next(a for a in ambient_rows if a['temperature_c']==temp and a['cycle']==kind)
                if rest.get('qualified'):
                    q=dict(p,effective_ambient_offset_c=0.)
                    alternative=simulate(frame.Time,frame.Current_discharge_positive,np.full(len(frame),rest['tail_median_temperature_c']),q,ocv,
                                         temperature0=float(frame.Battery_Temp_degC.iloc[0]))
                    validation.append(dict(temperature_c=temp,cycle=kind,variant='pre_rest_ambient_sensitivity',**metrics(frame,alternative.voltage_v.to_numpy(),alternative.temperature_c.to_numpy())))
                    ax.plot(frame.Time.iloc[::stride]/3600,alternative.temperature_c.iloc[::stride],label='Pre-rest proxy, no extra offset',lw=.8)
                ax.set(title=f'{temp} C {kind}',xlabel='Time (h)',ylabel='Cell temperature (C)');ax.grid(alpha=.2)
        c=p['Cth_J_K'];h=p['hA_W_K']
        for target_c in [40.,45.,50.]:
            scale=target_c/c
            thermal.append(dict(temperature_c=temp,fitted_Cth_J_K=c,fitted_hA_W_K=h,
                         prior_Cth_J_K=target_c,common_heat_Cth_hA_scale=scale,
                         scaled_hA_W_K=h*scale,thermal_time_constant_s=c/h,
                         excess_heat_capacity_J_K=c-target_c,
                         equivalent_short_resistance_at_3p7V_0p35C_ohm=3.7**2/(h*.35),
                         equivalent_extra_mass_at_assumed_900J_kgK_g=(c-target_c)/900*1000))
    pd.DataFrame(ambient_rows).to_csv(out/'pretest_ambient_estimates.csv',index=False)
    pd.DataFrame(edge_rows).to_csv(out/'hppc_rising_edges.csv',index=False)
    edges=pd.DataFrame(edge_rows);summary=[]
    for temp,g in edges.groupby('temperature_c'):
        summary.append(dict(temperature_c=int(temp),edges=len(g),median_immediate_fraction=float(g.immediate_fraction.median()),
                            delayed_fraction=float(g.voltage_not_yet_jumped.mean()),decision='blend=0; universal one-sample voltage delay not supported'))
    dump(out/'hppc_timing_summary.json',summary)
    pd.DataFrame(validation).to_csv(out/'fixed_model_validation.csv',index=False)
    pd.DataFrame(thermal).to_csv(out/'thermal_scale_audit.csv',index=False)
    pd.DataFrame(ess).to_csv(out/'cycle3_effective_samples.csv',index=False)
    dump(out/'thermal_scale_assumptions.json',dict(mass_g_reference=[47.,48.],mass_is_not_measured_in_this_experiment=True,
        manufacturer_sources=['https://api.pim.na.industrial.panasonic.com/file_stream/main/fileversion/3447',
                              'https://mediac.industry.panasonic.eu/p/2020-09/NCR18650BD-Improved_Lithium-ion_Flyer_2020-09-07_0.pdf'],
        heat_capacity_prior_J_K=[40.,50.],prior_source='senior feedback; not independently measured on this test fixture',
        specific_heat_900_J_kgK='illustrative equivalent-mass conversion only',
        identity='P -> a P, Cth -> a Cth, hA -> a hA leaves the healthy temperature trajectory invariant; ISC heat must not be rescaled as a fitted healthy discrepancy',
        conclusion='Healthy trajectories alone cannot distinguish attached heat capacity from multiplicative healthy heat-model error. Independent calorimetry/fixture data are missing.'))
    dump(out/'source_manifest.json',provenance)
    edge_figure.tight_layout();edge_figure.savefig(out/'hppc_onset_comparison.png',dpi=150);plt.close(edge_figure)
    temperature_axes[0,0].legend(fontsize=7);temperature_figure.tight_layout();temperature_figure.savefig(out/'ambient_sensitivity_comparison.png',dpi=150);plt.close(temperature_figure)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print(pd.DataFrame(validation).to_string(index=False),flush=True)


def prepare_inputs(root, out):
    """Export native inputs and explicit rest/driving windows without model fitting."""
    raw=root/'data/raw/Panasonic_18650PF';local=root/'data/processed/week3_preparation'
    local.mkdir(parents=True,exist_ok=True)
    records=[];segments=[];leading=[]
    for temp in [25,10,0]:
        for kind in ['HPPC','Cycle_3','US06','UDDS']:
            path=(next(raw.rglob(f'* {temp}degC_5*HPPC*.mat')) if kind=='HPPC' else
                  next(p for p in raw.rglob(f'* {temp}degC_{kind}_Pan18650PF.mat') if 'trise' not in str(p).lower()))
            frame=load_panasonic_data(path,ambient_setpoint_c=float(temp))
            t=frame.Time.to_numpy();i=frame.Current_discharge_positive.to_numpy()
            table=pd.DataFrame(dict(time_s=t,current_discharge_positive_a=i,
                  measured_voltage_v=frame.Voltage,measured_temperature_c=frame.Battery_Temp_degC,
                  ambient_input_c=frame.Chamber_Temp_degC))
            export=local/f'{temp}C_{kind.lower()}_inputs.csv';table.to_csv(export,index=False)
            records.append(dict(temperature_c=temp,cycle=kind,source=path.relative_to(root).as_posix(),
                source_sha256=fingerprint(path),input_file=export.relative_to(root).as_posix(),
                input_sha256=fingerprint(export),samples=len(frame),
                model_parameters=(out/f'model_v1/{temp}C.json').relative_to(root).as_posix(),
                noise_interface=(out/f'noise/{temp}C/noise_interface.json').relative_to(root).as_posix(),
                noise_role='Cycle3 supplies discrepancy; US06/UDDS are evaluation inputs',
                ambient_source=frame.attrs['quality_report']['ambient_source']))
            rest=np.abs(i)<=.05
            split=np.r_[0,np.flatnonzero((rest[1:]!=rest[:-1])|(np.diff(t)>2))+1,len(t)]
            for a,b in zip(split[:-1],split[1:]):
                if b-a<2 or t[b-1]-t[a]<10:continue
                segments.append(dict(temperature_c=temp,cycle=kind,input_file=export.relative_to(root).as_posix(),
                    start_row=int(a),stop_row_exclusive=int(b),start_s=float(t[a]),end_s=float(t[b-1]),
                    duration_s=float(t[b-1]-t[a]),kind='zero_terminal_current' if rest[a] else 'driving',
                    interpretation='zero current does not imply zero polarization or thermal equilibrium' if rest[a] else 'active current'))
            # Some drive files contain their own initial rest when no separate Pause file exists.
            # Such an estimate becomes available only after the initial rest; no backward use.
            m=loadmat(path,simplify_cells=True)['meas'];ri=np.ravel(m['Current']);active=np.flatnonzero(abs(ri)>.05)
            k=int(active[0]) if len(active) else 0
            item=dict(temperature_c=temp,cycle=kind,source=path.relative_to(root).as_posix(),qualified=False,
                      flags='no_long_leading_rest',application='post-rest windows only; not independent test-wide ambient measurement')
            if k>1:
                stamp=np.ravel(m['TimeStamp'])
                target=pd.to_datetime(stamp[k],format='%m/%d/%Y %I:%M:%S %p')
                before={name:np.ravel(m[name])[:k] for name in ['TimeStamp','Current','Battery_Temp_degC']}
                estimate=rest_tail_metrics(before,target)
                if estimate:
                    item.update(estimate)
                    item['application']='post-rest windows only; do not use subsequent test temperatures'
                    item['available_from_time_s']=float(np.ravel(m['Time'])[k])
            leading.append(item)
    segment_table=pd.DataFrame(segments)
    segment_table.to_csv(out/'operating_segments.csv',index=False)
    availability=[]
    for (temp,cycle,kind),group in segment_table.groupby(['temperature_c','cycle','kind']):
        for window in [10,30,60,300,600,1800,3600]:
            support=group[group.duration_s>=window]
            availability.append(dict(temperature_c=int(temp),cycle=cycle,kind=kind,window_s=window,
                supporting_segments=len(support),max_segment_s=float(group.duration_s.max()),
                overlapping_1s_step_windows=int(((support.duration_s-window)//1+1).sum())))
    pd.DataFrame(availability).to_csv(out/'window_availability.csv',index=False)
    pd.DataFrame(leading).to_csv(out/'leading_rest_estimates.csv',index=False)
    dump(out/'input_interface.json',records)
    dump(out/'ambient_nuisance_definition.json',dict(
        name='delta_ambient_c',units='C',nominal_halfwidth_c=.35,
        halfwidth_source='feedback sensitivity scenario, not a measured confidence interval',
        action='add one unknown constant to ambient input across a window, before the frozen effective offset',
        requirements=['Keep this nuisance in competing-hypothesis profiling or sensitivity analysis.',
                      'Do not equate it to independent per-sample temperature noise.',
                      'Use a pre-test proxy only if its quality flags pass.',
                      'Leading-rest estimates become available only after that rest and cannot initialize earlier predictions.']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();root=args.project_root.resolve();out=args.output_dir or root/'results/week3/preparation'
    run(root,out)
    prepare_inputs(root,out)
