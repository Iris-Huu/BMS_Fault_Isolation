"""Physical and data-integrity checks for the Week 2 implementation."""
import argparse
import hashlib
import json
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.model import simulate
from src.io import load_panasonic_data
from src.calibration import fit_joint_pulse,pulse_windows,fit_thermal_integrals
from src.noise import quantized_normal_log_probability,separable_window_covariance,sample_empirical_blocks


def check(root,results=None):
    p=dict(Q_Ah=2.9,R0_ohm=.025,R1_ohm=.01,C1_F=200.,R2_ohm=.02,C2_F=1500.,Cth_J_K=50.,hA_W_K=.05)
    ocv=pd.DataFrame(dict(soc=[0.,1.],ocv_v=[3.8,3.8]))
    t=np.arange(0,100.01,.1)
    constant=np.ones(len(t))
    model=simulate(t,constant,np.full(len(t),25.),p,ocv)
    assert np.isclose(model.soc.iloc[-1],1-100/(3600*2.9),atol=1e-11)
    assert np.isclose(model.v1_v.iloc[-1],.01*(1-np.exp(-100/2)),atol=1e-10)
    assert np.isclose(model.v2_v.iloc[-1],.02*(1-np.exp(-100/30)),atol=1e-10)
    healthy=simulate(t,np.zeros(len(t)),np.full(len(t),25.),p,ocv)
    assert np.allclose(healthy.voltage_v,3.8) and np.allclose(healthy.temperature_c,25.)
    shorted=simulate(t,np.zeros(len(t)),np.full(len(t),25.),p,ocv,faults=dict(fisc=.01))
    assert shorted.soc.iloc[-1]<healthy.soc.iloc[-1]
    assert shorted.temperature_c.iloc[-1]>healthy.temperature_c.iloc[-1]
    assert np.allclose(shorted.leakage_current_a,.01*shorted.voltage_v)
    assert np.allclose(shorted.cell_current_a,shorted.leakage_current_a)
    assert np.allclose(shorted.voltage_v,3.8-.025*shorted.cell_current_a-shorted.v1_v-shorted.v2_v)
    # Known synthetic pulse includes baseline charge depletion; test actual joint fitter.
    time=np.unique(np.r_[np.arange(0,30,.1),np.arange(30,1250,1.)])
    current=np.where((time>=10)&(time<20),2.9,0.)
    soc=1-np.r_[0,np.cumsum(current[:-1]*np.diff(time))]/(3600*2.9)
    dynamic_ocv=pd.DataFrame(dict(soc=[0.,1.],ocv_v=[3.3,4.1]))
    truth=simulate(time,current,np.full(len(time),25.),p,dynamic_ocv)
    frame=pd.DataFrame(dict(Time=time,Voltage=truth.voltage_v,Ah=(soc-1)*2.9,
                            Current_discharge_positive=current,Battery_Temp_degC=25.,Chamber_Temp_degC=25.))
    estimate,_=fit_joint_pulse(frame,pulse_windows(frame)[0])
    for key in ["R0_ohm","R1_ohm","R2_ohm","C1_F","C2_F"]:
        # Midpoint switching approximation introduces small differences from ZOH truth.
        assert np.isclose(estimate[key],p[key],rtol=.15),(key,estimate[key],p[key])
    # Integral thermal regression must recover two independently excited coefficients.
    heat=2.+np.sin(t/10)
    ambient=np.full(len(t),25.)
    temperature=np.empty(len(t));temperature[0]=25.
    for n in range(len(t)-1):
        a=np.exp(-.05*.1/50)
        temperature[n+1]=25+heat[n]/.05+(temperature[n]-25-heat[n]/.05)*a
    long_t=np.arange(0,1800.1,.1)
    long_heat=1.5+np.sin(long_t/150)
    long_temperature=np.empty(len(long_t));long_temperature[0]=25.
    for n in range(len(long_t)-1):
        a=np.exp(-.05*.1/50)
        long_temperature[n+1]=25+long_heat[n]/.05+(long_temperature[n]-25-long_heat[n]/.05)*a
    thermal=fit_thermal_integrals([(long_t,long_temperature,np.full(len(long_t),25.),long_heat)])
    assert np.isclose(thermal["Cth_J_K"],50.,rtol=.02)
    assert np.isclose(thermal["hA_W_K"],.05,rtol=.02)
    low=next((root/"data"/"raw"/"Panasonic_18650PF").rglob("* 10degC_5*HPPC*.mat"))
    try:
        load_panasonic_data(low)
    except ValueError:
        pass
    else:
        raise AssertionError("Strict ambient-missing check must remain default")
    frame=load_panasonic_data(low,ambient_setpoint_c=10.)
    assert frame.attrs["quality_report"]["ambient_imputed_rows"]>0
    assert np.all(frame.Chamber_Temp_degC==10.)
    assert np.all(np.diff(frame.Time)>0)
    if results is not None:
        manifest=json.loads((results/"summary"/"data_manifest.json").read_text(encoding="utf-8"))
        training={m["sha256"] for m in manifest if "calibration" in m["role"]}
        validation={m["sha256"] for m in manifest if m["role"]=="held_out_validation"}
        assert training.isdisjoint(validation)
        assert len(validation)==6
        for source in manifest:
            assert hashlib.sha256((root/source["file"]).read_bytes()).hexdigest()==source["sha256"],source["file"]
        for temp in [25,10,0]:
            for name in ["sigma_raw_detrended.csv","sigma_quantization_sensitivity.csv","sigma_model_error_sensitivity.csv"]:
                covariance=pd.read_csv(results/"noise"/f"{temp}C"/name,index_col=0).to_numpy()
                assert np.allclose(covariance,covariance.T)
                assert np.linalg.eigvalsh(covariance).min()>=-1e-12
        errors=pd.read_csv(results/"summary"/"model_error_table.csv")
        assert set(zip(errors.temperature_c,errors.cycle))=={(temp,cycle) for temp in [25,10,0] for cycle in ["US06","UDDS"]}
        assert np.all(np.isfinite(errors.select_dtypes(include="number")))
        latest_path=results/"comparisons"/"calibration_refinement"/"model_error_table.csv"
        if latest_path.exists():
            latest=pd.read_csv(latest_path)
            compared=errors.merge(latest,on=["temperature_c","cycle"],suffixes=("_initial","_new"))
            assert len(compared)==6
            assert np.all(compared.voltage_rmse_mv_new<compared.voltage_rmse_mv_initial)
            assert np.all(compared.temperature_rmse_c_new<compared.temperature_rmse_c_initial)
            for temp in [25,10,0]:
                parameter=json.loads((latest_path.parent/f"{temp}C"/"parameters.json").read_text(encoding="utf-8"))["parameters"]
                for key in ["R0_ohm","R1_ohm","R2_ohm","C1_F","C2_F","Cth_J_K","hA_W_K"]:
                    assert np.isfinite(parameter[key]) and parameter[key]>0
                for branch in ["R0_ohm","R1_ohm","R2_ohm"]:
                    assert np.all(np.array(parameter["resistance_soc_model"][branch])>0)
                empirical=np.load(results/"noise"/f"{temp}C"/"empirical_rest_blocks.npz")
                assert empirical["boundaries"][0]==0
                assert empirical["boundaries"][-1]==len(empirical["samples"])
                assert np.all(np.diff(empirical["boundaries"])>0)
            print("PASS: six refined development cases improve both RMSEs; positive parameters; empirical noise block integrity.")
        print("PASS: raw source hashes unchanged, calibration/validation source separation, six finite validation rows, covariance PSD.")
    print("PASS: charge balance, exact RC response, healthy equilibrium, short-circuit algebra/heat,")
    print("joint 2RC parameter recovery, independent thermal coefficient recovery,")
    print("strict missing-ambient default, explicit fallback provenance, increasing time.")


def check_refinement():
    from copy import deepcopy
    p=dict(Q_Ah=2.9,R0_ohm=.025,R1_ohm=.01,C1_F=200.,R2_ohm=.02,C2_F=1500.,Cth_J_K=50.,hA_W_K=.05)
    t=np.arange(0,200.1,.5);i=1+np.sin(t/7);ambient=np.full(len(t),25.)
    ocv=pd.DataFrame(dict(soc=[0.,.5,1.],ocv_v=[3.3,3.7,4.1]))
    physical=simulate(t,i,ambient,p,ocv)
    delayed=simulate(t,i,ambient,dict(p,voltage_readout_blend=1.),ocv)
    assert np.allclose(delayed.voltage_v.iloc[1:],physical.voltage_v.iloc[:-1])
    assert np.allclose(delayed.heat_w,physical.heat_w)
    assert np.allclose(delayed.temperature_c,physical.temperature_c)
    model=dict(reference_temperature_c=25.,soc_knots=[0.,.5,1.],
               R0_ohm=[.04,.025,.03],R1_ohm=[.02,.01,.015],R2_ohm=[.04,.02,.03],
               ocv_correction_v=[0.,.001,0.],R0_activation_k=2000.,R1_activation_k=1800.,R2_activation_k=2200.,
               tau1_s=2.,tau2_s=30.)
    original=dict(p,resistance_soc_model=model,exact_parameter_lookup=True)
    renamed=deepcopy(original);renamed["Q_Ah"]=3.2
    transformed_ocv=ocv.copy();transformed_ocv["soc"]=1-(2.9/3.2)*(1-ocv.soc)
    renamed["resistance_soc_model"]["soc_knots"]=(1-(2.9/3.2)*(1-np.array(model["soc_knots"]))).tolist()
    a=simulate(t,i,ambient,original,ocv);b=simulate(t,i,ambient,renamed,transformed_ocv)
    assert np.allclose(a.voltage_v,b.voltage_v,atol=1e-10)
    assert np.allclose(a.temperature_c,b.temperature_c,atol=1e-10)
    # A true capacity fault keeps the calibrated coordinate curve fixed and must
    # change voltage evolution, unlike a reference-coordinate relabeling.
    reduced=simulate(t,i,ambient,original,ocv,faults=dict(fQ=.2))
    assert not np.allclose(a.voltage_v,reduced.voltage_v,atol=1e-4)
    bins=np.arange(-100,101)*.2
    pmf=np.exp(quantized_normal_log_probability(bins,0.,1.,.2))
    assert np.isclose(pmf.sum(),1.,atol=1e-12)
    assert np.all(np.isfinite(quantized_normal_log_probability(np.array([-30.,30.]),0.,1.,.2)))
    covariance=separable_window_covariance(np.diag([1e-6,1e-8,.01]),20,.1,1.,.7)
    assert covariance.shape==(60,60) and np.linalg.eigvalsh(covariance).min()>0
    samples=np.arange(90.).reshape(30,3);bounds=np.array([0,10,30])
    block=sample_empirical_blocks(samples,bounds,8,np.random.default_rng(12))
    assert np.all(np.diff(block,axis=0)==3)
    print("PASS: readout response preserves physical heat, charge-coordinate invariance, true capacity fault,")
    print("normalized quantized likelihood, PSD colored covariance, contiguous empirical-noise blocks.")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--project-root",type=Path,required=True)
    parser.add_argument("--results-dir",type=Path)
    args=parser.parse_args()
    check(args.project_root,args.results_dir)
    check_refinement()
