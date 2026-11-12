"""Build the Chinese results report from saved numerical outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def markdown_table(frame):
    def fmt(x):
        if isinstance(x,(float,np.floating)): return f"{x:.4g}"
        return str(x).replace("|","/")
    rows=["| "+" | ".join(map(str,frame.columns))+" |",
          "| "+" | ".join(["---"]*len(frame.columns))+" |"]
    rows.extend("| "+" | ".join(fmt(x) for x in row)+" |" for row in frame.itertuples(index=False,name=None))
    return "\n".join(rows)


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
    parameter_display=params[["temperature_c","R0_ohm","R1_ohm","C1_F","R2_ohm","C2_F","Cth_J_K","hA_W_K"]]
    text="""# 第二周电池模型标定与验证说明

本次已完成三个温度的数值模型基线、HPPC 联合辨识、热参数拟合、噪声检查，以及六组驾驶工况验证。代码和结果可以复现。第二周的科学验收尚未全部通过：Panasonic 低温独立 C/20 数据缺失，日期与容量未独立对齐，温度模型仍有系统误差，噪声也不能直接视为独立高斯白噪声。参数文件均标为 provisional，不能把这些结果表述成已经验证的可隔离边界。

## 1 研究内容与结构结论

本研究包括数据处理、电热模型建立、参数标定、结构分析与定量验证。结构结论为：无热通道时电流偏置与内短路不可隔离；加入热模型和温度测量后可以隔离。本轮已完成数值基线实现、参数标定计算和工况验证，结构分析与定量诊断的完成状态需依据相应代码和结果单独核实。

## 2 数据和标定验证的划分

电学参数只用各温度的 HPPC；噪声用 HPPC 内连续、充分静置后的原始采样段。热参数用各温度的 Cycle 1 与 Cycle 2。US06 和 UDDS 不参与参数数值拟合，但已经在开发过程中用于检查模型结构，因此称为开发验证集，不是未接触过的盲测集。data_manifest.json 记录了文件、角色、SHA256、重复行和温箱温度补缺情况。三个温度各使用 5 份文件，共 15 份。

原始 10°C、0°C 的 HPPC、Cycle 1/2、US06、UDDS 温箱温度列存在整列缺失。只有显式传入 ambient_setpoint_c 才会采用标称设定温度补缺；默认仍拒绝无效数据。报告保留原始缺失数量，原始 MAT 不改动。Trise 禁止此回退。低温热参数与误差均以设定环境温度为条件。

不均匀采样按实际时间间隔处理。连续仿真在相邻输入之间采用零阶保持；对长时间无记录区间，这是一项条件假设，不等于知道缺失区间的真实输入。清单记录最大间隔和大于 15 秒的间隔数，热参数积分拟合不使用含此类大间隔的窗口。误差指标按原始有效采样点等权计算，不是按持续时间加权。

## 3 SOC 和 OCV

先前的 25°C C/20 初版采用该放电段自身的 2.99491 Ah，把本段起止归一化为 SOC 1 和 0。这里的跨温度 HPPC 基线采用数据集 2.9 Ah 的标称容量和协议坐标：SOC = 1 + (Ah - Ah_initial)/2.9，假设每次工况开始前已充满。两者不是同一条经过独立验证的 SOC 标尺，不能直接混合使用。邻近 1C 放电约 2.798 Ah 是倍率相关的可用容量，也不能自动当作平衡容量。

数据说明只明确 C/20 在前序做一次，低温重复 HPPC、EIS 和驾驶工况。目前只找到一份 25°C C/20；无法声称已经完成 10°C、0°C 独立 C/20 拟合。为让模型能运行，另保存了各温度的 HPPC 静置电压近似曲线 parameters/ocv_25C.csv、ocv_10C.csv、ocv_0C.csv，使用静置前后电压、充分静置尾段以及单调回归。它是有来源的替代基线，仍可能含残余极化、温度变化和老化影响，不是完成了原要求的三温度 C/20 标定。

## 4 电学参数

HPPC 里的缺失放电不能按零电流积分。每个脉冲后只保留到下一处时间缺口或 Ah 跳变之前的恢复段。标定使用未提前截止的约 10 秒低倍率脉冲和至少 600 秒连续恢复段，联合拟合加载与恢复，使用 Ah 相关的局部电压基线。开关时刻取采样中点，剔除切换附近 0.25 秒，限制电阻、电容为正，并区分两条时间尺度。采用多个初值，记录贴边参数、基线偏移和误差，不把优化器返回数值等同于物理可辨识。

固定参数基线采用中间 SOC 的未标记拟合中位数。tau = R*C；先汇总 R 与 tau，再计算 C。parameter_spread.csv 保留各 SOC/倍率参数的四分位范围，这些变化不等同于统计置信区间。

"""+markdown_table(parameter_display)+"""

Q_Ah 均为标称参考值 2.9，尚未独立辨识；上表均为候选基线参数，不是已验收参数。参数表里的温度是实验分组，不能理解为电芯在整个实验中始终处于该温度。

## 5 热模型

单纯静置、忽略产热时，dT/dt = -(hA/Cth)*(T-Tamb)，主要确定二者比值。这里联合使用有产热的驾驶段和静置段，通过 60 秒窗口的积分能量平衡分别估计 1/Cth 和 hA/Cth，避免直接对台阶状温度求导。标定产热使用 I*(OCV_proxy - measured_voltage)，因此热参数也受 OCV 近似和电压测量误差影响。

验证时不再使用实测电压来计算产热，而由预测电压计算；实测温度只用于初值，之后没有输出反馈。根据残差的 SOC 相关性，额外测试了有效熵热项 I*(T+273.15)*(k0+k1*(SOC-0.5))。系数只用 Cycle1/2 拟合，不能当作独立测量的 dOCV/dT；该修正没有改善验证误差，因此保留为敏感性实验，不作为推荐模型。

## 6 独立工况运行结果与模型比较

主误差表包含全部原始有效采样点，单位分别为 mV 和 °C。没有为了改善数字而删除电流切换或截止电压点。SOC/温度映射版本只用 HPPC 的未标记参数插值；温度只由模型自身预测。超出参数映射支撑区间会采用端点值，并报告比例。特别是 25°C US06 中电芯升温超出 HPPC 标定温度范围，不能把映射当作已验证的温度规律。

"""+markdown_table(comparison)+"""

映射版本改善部分工况，但不是全面改善，因此同时保留固定基线与映射模型，不按每个工况挑选最优版本拼表。温度误差仍有方向性偏差；三温度模型都不能直接用于宣称弱内短路的精确定量边界。模型误差的量级明显大于小故障预期信号时，必须先改进模型或把不确定性传播到结论中。

一些最大电压误差恰好出现在电流突变处。局部原始记录显示电流已经改变，而电压仍接近前一拍，提示测量通道同步或记录时序需要进一步核实；这只是数据证据支持的假设，尚未证明固定延迟。没有人为平移通道。映射版本另报告“排除当前和前一拍大于 2A 跳变”的辅助误差，但全采样误差仍是主结果，且最差 20 个点完整保存。

## 7 噪声与温度量化

"""+markdown_table(pd.DataFrame(noise_rows))+"""

最小观测温度间隔和较大跳变尺度不是同一回事。约 0.011°C 的细间隔与约 0.2°C 的台阶都存在；不能只凭最小相邻值差判定传感器分辨率，更不能把分辨率当作准确度。

每个温度提供三套矩阵，通道顺序固定为 [电流 A, 电压 V, 温度 °C]：

- sigma_raw_detrended.csv：连续静置窗口分别去趋势后的样本协方差。静置电流常为精确零，不能据此识别真实电流传感器噪声。
- sigma_quantization_sensitivity.csv：在上述矩阵上加入恒流脉冲内部电流波动代理，以及温度较大跳变尺度平方/12 的敏感性项和极小数值正则。电流波动包含控制器纹波，温度项也不是已证明的均匀量化模型；它们是显式假设，不是纯测量噪声的精确估计。
- sigma_model_error_sensitivity.csv：再加入 Cycle1/2 标定残差的均方量级，用来考察模型误差的影响；不能当成残差已独立、零均值的证明。

noise_assumptions.json 记录样本量、窗口数和滞后一拍相关性。第三周不能未经检查就把这些 3×3 矩阵复制成整个窗口的独立高斯协方差。电流作为模型输入，其噪声还要传播到输出残差；定量分析需要明确这一噪声传播关系。

## 8 数值模型与数据接口

src/model.py 的 simulate(time, current_measured, ambient, parameters, ocv_table, soc0, temperature0, polarization0, faults) 返回 SOC、两条极化电压、电芯温度、端电压、电芯电流、漏电流和产热。电流以放电为正。数值模型保留 i_s=fisc*v、Icell=i+i_s，内短路耗电并通过 i_s*v 额外产热，热方程使用 Icell。内短路代数关系已解析求解，不需要把故障重复写进符号模型。

故障字段为 fi[A]、fv[V]、fT[°C]、fisc[S]、fQ[比例]、fR[比例] 和 onset_s[s]。fisc=1/Risc。数值模型提供灵敏度分析与故障注入原型接口；结构分析、式(7)和诊断器需分别实现与验证。

parameters/25C.json、10C.json、0C.json 记录固定基线参数、单位含义、方法、初始条件和限制。validation/ 各温度目录内的 trajectory_*_1s.csv 是从全分辨率模拟中按秒抽样的交换文件，不是均值重采样输入，也不应把稀疏输出重新当作原始电流运行模型。正式验证使用原始有效采样间隔，OCV 支撑范围外采用端点值并报告比例。方案对比的独立参数保留在 comparisons/ 对应方案内，不与固定基线混用。

## 9 检查与复现

在项目根目录执行：

```powershell
.\\.venv\\Scripts\\python.exe -B tests\\verify_week2.py --project-root .
.\\.venv\\Scripts\\python.exe -B experiments\\run_week2.py --project-root .
```

物理检查覆盖电荷守恒、恒流 RC 解析响应、健康静置平衡、内短路耗电与产热、漏电代数一致性、已知 2RC 参数恢复、热参数独立恢复、缺失温箱温度的默认拒绝和显式补缺。重跑会更新 results/week2 下的生成结果，不修改原始数据。

## 10 验收状态和后续工作

已完成可复现的计算流程、三个温度的候选参数表、六组误差表、噪声敏感性矩阵、模型接口和图表。尚未满足的要求包括：三温度独立 C/20；按日期独立确认容量与 SOC 标尺；解释和降低系统性电压/温度误差；确认测量时序；验证量化、相关噪声的统计模型。应把这些作为明确的研究限制和补充数据需求，不把文件齐全当成验收通过。

本接口可用于算法单元核对、模拟算例和不确定性敏感性分析，但目前不宜把计算出的弱内短路边界写成已经有实测证据支撑的最终结论。结构分析结论不因这些数值误差而自动改变。
"""
    if (out/"comparisons"/"calibration_refinement"/"model_error_table.csv").exists():
        text=text.replace("# 第二周电池模型标定与验证说明",
                          "# 第二周固定参数基线说明\n\n此页保留固定参数基线及早期比较。最新校核与改进结果见 [第二周校核与改进说明](第二周校核与改进说明.md)。",1)
    (summary/"第二周工作说明.md").write_text(text,encoding="utf-8")
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
    status=dict(computational_work="completed_for_available_data",final_scientific_acceptance="conditional_not_unqualified",
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
    text="""# 第二周校核与改进说明

本轮已完成数据日期和容量证据核对、SOC 坐标统一、测量响应建模、电热参数改进、噪声模型重构及额外工况检查。三个温度的六组开发验证误差均低于固定参数基线。现有数据不能恢复Panasonic 低温独立 C/20、缺失温箱实测温度或纯电流传感器噪声，不能将这些不可辨识量写成已完成的独立实测标定。预测区间也尚未在所有工况达到名义覆盖率，因此最终科学验收仍有条件限制。

## 1 日期与容量证据

已逐一检查全部 198 个 MAT 文件，15 个文件名日期与内部实际起始日期不一致。模型数据来源以 TimeStamp 核实；例如 10°C US06 文件名为 2017-03-27，实际起始为 2017-04-06。audit/dataset_inventory.csv 记录实际起止时间、缺失、哈希和日期冲突；model_data_lineage.csv 记录模型所用数据的实际阶段。

独立恒流放电记录显示：三月 1C 可用放电量约 2.798、2.752 Ah，七月约 2.434、2.354 Ah；五月 C/20 段为 2.99491 Ah。这些值既有测试阶段差异，也有倍率和截止电压差异，不能互相替代为同一个平衡容量。未对中间日期虚构插值容量。

## 2 电量与 SOC 坐标

模型统一使用累计释放电量 q = integral(Icell dt)/3600，并以 s = 1-q/2.9 表示参考 SOC。OCV 和参数查询点沿同一电量坐标定义。2.9 Ah 是坐标参考，不是每个日期独立测出的真实平衡容量。

参考容量改成 Qnew 时，须同时把所有查询点改为 s_new = 1-(2.9/Qnew)*(1-s)。这样正常模型预测不变；数值测试已验证。真正容量衰减保持已标定曲线不变、改变电荷动态，不能与坐标改名混同。由此解决了此前不同归一化曲线直接混用的问题，同时保留物理容量不可独立辨识的事实。

## 3 OCV 替代方法与适用范围

在当前下载的 Panasonic 原始数据中未找到 10°C、0°C 的独立 C/20；LG 数据中存在对应文件，但属于不同电芯。各温度采用 HPPC 充分静置电压代理，并按整组脉冲进行留组检验，避免用相邻近重复点检验自身。结果只反映静置电压曲线的跨 SOC 插值误差，不等同于独立平衡 OCV 准确度。

"""+markdown_table(ocv[["temperature_c","held_out_groups","covered_points","group_cv_rmse_mv","group_cv_max_abs_mv"]])+"""

改进模型另保存 effective_ocv_charge_coordinate.csv，同时列出参考 SOC、释放电量和有效电压。小幅 OCV 修正只用标定工况拟合。0°C 标定工况尾段超出 HPPC 电压曲线支撑域，已明确截去该未支撑部分，而非用巨大电阻吸收曲线端点钳制造成的误差。域外低 SOC 不作准确性承诺。

## 4 电学与测量响应

电阻随 SOC 插值，并采用由三温度 HPPC 得到的温度变化斜率；电阻保持为正。RC 时间常数仍由 HPPC 提供。使用 Cycle1 拟合、Cycle2 选择因果电压读出混合系数，之后再用两组标定工况确定系数。没有平移验证数据，也没有把读出系数宣称为已证明的硬件延迟。

读出模型只改变预测的测量电压，物理产热仍用瞬时端电压，避免把记录滞后当成电芯额外发热。该分离已通过数值检查。SOC、温度参数查询的缓存精度为 0.001 SOC、0.1°C，可在检查中启用精确查询。

## 5 热模型与环境不确定性

使用标定工况的前向温度轨迹辨识热容、换热系数和健康基线的有效环境偏移。静置段提供偏移先验。该偏移综合了温箱设定误差、测温偏移及模型不足，不是独立测出的环境温度或温度传感器偏置。内短路的额外产热保持 i_s*v，物理温度不依赖电压读出混合。

对有效偏移上下各 0.5°C，重新在标定数据上拟合热参数，形成显式敏感性情形。它们不是置信区间，也不能说明真实环境必然位于其中。低温温箱原始测量仍缺失，采用设定值的条件已保留。

## 6 开发验证结果

下表电压单位为 mV，温度单位为 °C，均按全部原始有效采样点计算，没有剔除突变或截止点。

"""+markdown_table(comparison)+"""

US06/UDDS 已用于开发诊断，因此不能称作未接触过的盲测数据。Cycle3 只用于建立误差区间，未用于拟合电学和热学参数；参数固定后另在 Cycle4 做点预测检查：

"""+markdown_table(independent)+"""

## 7 噪声与区间检验

已保存原始连续静置残差块及边界，提供按块抽样接口，保留量化和时间依赖；另提供带相关性的高斯比较协方差与量化正态区间概率计算接口。后两者是显式比较模型，不是已证明的仪器噪声定律。静置尾段的原生采样约为 1 秒，向 0.1 秒窗口延伸相关性属于假设。

恒流段电流波动仅作控制器纹波与测量波动的混合代理，不能独立辨识纯传感器噪声。噪声窗口采用约四分之三训练、四分之一留出；所有覆盖结果保存在 noise_holdout_coverage.csv。不同温度的覆盖不完全一致，不将高斯白噪声假设视为已通过。

预测区间用 Cycle3 的分 SOC 残差 95% 分位数，加上标定环境偏移敏感性情形；下表是实际覆盖率与区间宽度，不是宣称 95% 保证：

"""+markdown_table(coverage[["temperature_c","cycle","voltage_coverage","temperature_coverage","median_voltage_halfwidth_mv","median_temperature_halfwidth_c"]])+"""

部分工况电压覆盖明显低于 95%，说明跨工况模型误差不是一个固定协方差就能充分描述。这个失败结果完整保留，未用验证残差重新调整区间。温度区间覆盖较高，但宽度限制了小产热的分辨能力。不能把平均误差下降等同于弱内短路定量边界已获验证。

## 8 文件与复现

- audit/：全部源数据审计、容量记录、时间与电量积分检查、OCV 检验、坐标约定。
- comparisons/calibration_refinement/：改进模型的参数、读出系数选择、有效 OCV、预测轨迹、图表、额外检查和区间覆盖。
- noise/各温度/empirical_colored_noise.json 与 empirical_rest_blocks.npz：经验噪声模型和原始连续残差块。
- summary/refinement_comparison.csv：新旧模型比较；week2_completion_status.json：完成项、不可辨识量和未通过的统计要求。

原始数据与固定参数基线均保留。完整复现仍使用 experiments/run_week2.py；tests/verify_week2.py 增加了参考坐标不变性、读出响应不改变产热、量化概率归一化、相关协方差正定和连续噪声块检查。

## 9 当前结论

可由现有数据执行的计算、校核、改进和验证已落实；模型误差显著下降。Panasonic 低温独立 C/20、真实温箱温度与纯传感器噪声缺乏独立观测，不能通过继续拟合恢复。模型与噪声接口可用于条件性算法研究；只有在明确这些条件、传播模型不确定性且验证区间覆盖后，才适合形成最终可区分边界。
"""
    (summary/"第二周校核与改进说明.md").write_text(text,encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--results-dir",type=Path,required=True)
    run(parser.parse_args().results_dir)
