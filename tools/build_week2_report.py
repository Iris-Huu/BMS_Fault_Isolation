"""Build the Chinese results report from saved numerical outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd




def run(out):
    summary=out/"summary"
    summary.mkdir(parents=True,exist_ok=True)
    params=pd.read_csv(summary/"parameter_table.csv")
    errors=pd.read_csv(summary/"model_error_table.csv")
    mapped=pd.read_csv(out/"comparisons"/"soc_temperature_maps"/"model_error_table.csv")
    entropy=pd.read_csv(out/"comparisons"/"entropy_sensitivity"/"model_error_table.csv")
    noise_rows=[];robustness=[]
    for temp in [25,10,0]:
        folder=out/"noise"/f"{temp}C"
        calibration=out/"calibration"/f"{temp}C"
        info=json.loads((folder/"noise_assumptions.json").read_text(encoding="utf-8"))
        sigma=pd.read_csv(folder/"sigma_raw_detrended.csv",index_col=0).to_numpy()
        noise_rows.append(dict(温度=temp,静置窗口=info["rest_windows"],
                               电压残差标准差_mV=1000*np.sqrt(sigma[1,1]),
                               温度残差标准差_C=np.sqrt(sigma[2,2]),
                               最小观测温度间隔_C=info["temperature_min_observed_spacing_c"],
                               温度较大跳变尺度_C=info["temperature_large_jump_scale_c"],
                               恒流段电流波动代理_mA=1000*info["active_plateau_current_std_proxy_a"]))
        fits=pd.read_csv(calibration/"joint_pulse_fits.csv").fillna({"flags":""})
        selected=pd.read_csv(calibration/"aggregate_source_pulses.csv")
        for key in ["R0_ohm","R1_ohm","C1_F","R2_ohm","C2_F","tau1_s","tau2_s"]:
            robustness.append(dict(temperature_c=temp,parameter=key,source_count=len(selected),
                                   median=float(selected[key].median()),q25=float(selected[key].quantile(.25)),
                                   q75=float(selected[key].quantile(.75)),
                                   all_fit_count=len(fits),flagged_fit_count=int((fits["flags"]!="").sum())))
    pd.DataFrame(robustness).to_csv(summary/"parameter_spread.csv",index=False)
    pd.DataFrame(noise_rows).to_csv(summary/"noise_summary.csv",index=False)
    comparison=errors[["temperature_c","cycle","voltage_rmse_mv","temperature_rmse_c"]].copy()
    comparison=comparison.rename(columns={"voltage_rmse_mv":"固定参数_电压RMSE_mV","temperature_rmse_c":"固定参数_温度RMSE_C"})
    comparison["SOC温度映射_电压RMSE_mV"]=mapped.voltage_rmse_mv.to_numpy()
    comparison["SOC温度映射_温度RMSE_C"]=mapped.temperature_rmse_c.to_numpy()
    comparison["熵热敏感性_温度RMSE_C"]=entropy.temperature_rmse_c.to_numpy()
    comparison.to_csv(summary/"model_comparison.csv",index=False)
    if (out/"comparisons"/"calibration_refinement"/"uncertainty_coverage.csv").exists():
        build_refinement_report(out)


def build_refinement_report(out,baseline=None):
    baseline=baseline or out
    summary=out/"summary";summary.mkdir(parents=True,exist_ok=True)
    refined=out/"comparisons"/"calibration_refinement"
    initial=pd.read_csv(baseline/"summary"/"model_error_table.csv")
    latest=pd.read_csv(refined/"model_error_table.csv")
    comparison=initial[["temperature_c","cycle","voltage_rmse_mv","temperature_rmse_c"]].rename(
        columns={"voltage_rmse_mv":"initial_voltage_rmse_mv","temperature_rmse_c":"initial_temperature_rmse_c"})
    comparison=comparison.merge(latest[["temperature_c","cycle","voltage_rmse_mv","temperature_rmse_c"]],on=["temperature_c","cycle"])
    comparison.to_csv(summary/"refinement_comparison.csv",index=False)
    ocv=pd.read_csv(out/"audit"/"ocv_evidence.csv")
    independent=pd.read_csv(refined/"confirmation_error_table.csv")
    coverage=pd.read_csv(refined/"uncertainty_coverage.csv")
    noise=pd.read_csv(summary/"noise_holdout_coverage.csv")
    status=dict(source_data_recheck="audit/source_data_recheck.csv",computational_work="completed_for_available_data",final_scientific_acceptance="conditional_not_unqualified",
                closed_items=["198-file timestamp and capacity-reference audit","charge-coordinate consistency with numerical invariance test",
                              "held-out HPPC group interpolation check","calibration-only current/voltage response fit",
                              "positive SOC/temperature resistance calibration","forward thermal fit and ambient-offset sensitivity",
                              "empirical block-noise model, quantized likelihood and colored covariance implementation",
                              "six development validations plus three additional fixed-parameter Cycle4 evaluations"],
                irreducible_data_limits=["independent Panasonic 10C/0C C20 not located locally; LG counterparts exist",
                                        "missing low-temperature chamber measurements cannot be reconstructed independently",
                                        "physical equilibrium capacity and current-sensor-only noise not separately identifiable"],
                observed_validation_failures=["nominal 95% prediction envelopes do not achieve 95% voltage coverage in every operating condition",
                                              "noise window exchangeability and sub-native-cadence correlation are not proven"],
                usage="model and algorithms may be studied conditionally; do not label weak-fault boundaries as unconditional experimentally validated limits")
    (summary/"week2_completion_status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding="utf-8")
    selection=dict(preferred_numerical_model="calibration_refinement",status="conditional_development_model",
                   parameter_files={f"{t}C":f"comparisons/calibration_refinement/{t}C/parameters.json" for t in [25,10,0]},
                   base_ocv_files={f"{t}C":f"parameters/ocv_{t}C.csv" for t in [25,10,0]},
                   noise_files={f"{t}C":f"noise/{t}C/empirical_colored_noise.json" for t in [25,10,0]},
                   comparison_file="summary/refinement_comparison.csv",limitations_file="summary/week2_completion_status.json")
    (summary/"model_selection.json").write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding="utf-8")
    reference_rows=[]
    for temp in [25,10,0]:
        parameter_path=refined/f"{temp}C"/"parameters.json"
        info=json.loads(parameter_path.read_text(encoding="utf-8"));p=info["parameters"];curve=p["resistance_soc_model"]
        # Scalar fields are a documented fallback at SOC=.5 and group reference
        # temperature. The full simulator still uses the complete dynamic map.
        for branch in ["R0","R1","R2"]:
            p[branch+"_ohm"]=float(np.interp(.5,curve["soc_knots"],curve[branch+"_ohm"]))
        p["C1_F"]=curve["tau1_s"]/p["R1_ohm"];p["C2_F"]=curve["tau2_s"]/p["R2_ohm"]
        info["scalar_fallback_basis"]=dict(soc_reference=.5,cell_temperature_c=temp,
                                           note="Use resistance_soc_model for full simulation; scalar values alone are an approximation")
        parameter_path.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
        for soc in [.2,.5,.8]:
            r0,r1,r2=[float(np.interp(soc,curve["soc_knots"],curve[k+"_ohm"])) for k in ["R0","R1","R2"]]
            reference_rows.append(dict(temperature_c=temp,soc_reference=soc,withdrawn_charge_ah=2.9*(1-soc),
                                       R0_ohm=r0,R1_ohm=r1,C1_F=curve["tau1_s"]/r1,R2_ohm=r2,C2_F=curve["tau2_s"]/r2,
                                       Cth_J_K=p["Cth_J_K"],hA_W_K=p["hA_W_K"],
                                       effective_ambient_offset_c=p["effective_ambient_offset_c"],
                                       voltage_readout_blend=p["voltage_readout_blend"]))
    pd.DataFrame(reference_rows).to_csv(summary/"refined_parameter_table.csv",index=False)
    # (summary/"第二周校核与改进说明.md").write_text(text,encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--results-dir",type=Path,required=True)
    run(parser.parse_args().results_dir)
