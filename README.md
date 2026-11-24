# BMS 电流偏置与内短路故障可隔离性研究

本项目使用单体锂离子电池的电压、电流和温度数据，研究电流传感器偏置与内短路的检测及区分。模型由二阶 RC 等效电路和集中参数热模型组成。

**第二周已按反馈通过验收。当前工作是冻结模型、准备第三周的故障敏感性与定量可区分度计算。** 主结果采用25°C和10°C，0°C保留作补充。

## 1. 项目目录

| 目录或文件 | 用途 |
| --- | --- |
| `src/` | 数据读取、参数标定、数值电热模型、符号结构模型及噪声方法 |
| `experiments/` | 标定、验证、比较和数据准备入口 |
| `tests/` | 数据读取、物理关系和数值一致性检查 |
| `tools/` | 数据来源审计及汇总工具 |
| `results/week1/` | US06原始测量图与`structural/`无热/含热结构分析结果 |
| `results/week2/` | 第二周基线、改进参数与原有验证证据，保留原样 |
| `results/week3/preparation/` | 时序与环境核查、冻结参数、Cycle3噪声接口及复核结果 |
| `data/raw/` | 原始实验数据，Git忽略 |
| `data/processed/` | 本地输入与逐行残差，Git忽略 |
| `archive/` | 历史探索和本地快照 |
| `requirements.txt` | UTF-8依赖清单，含`faultdiagnosistoolbox==0.12.5`和scikit-learn |

根目录README是唯一正式说明。CSV、JSON及图片保留具体数值、来源和研究证据。

## 2. 环境与数据

现有环境为Python 3.12。在项目根目录用PowerShell执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

主数据为[Panasonic 18650PF，版本1](https://data.mendeley.com/datasets/wykht8y7tg/1)，标称2.9 Ah；另保留[LG 18650HG2，版本2](https://data.mendeley.com/datasets/b5mj79w5w9/2)，用于独立数据检查。

数据分别放在`data/raw/Panasonic_18650PF/`、`data/raw/LG_18650HG2/`，保留原文件名和嵌套结构。

按官方目录核查反馈，Panasonic只有25°C独立C/20。三温度数值模型统一使用HPPC静置电压代理；独立25°C C/20结果另外保存。LG的低温C/20属于另一种电芯。

计算电流使用`Current_discharge_positive = -Current`，放电为正；时间单位为秒。所选低温文件的温箱温度缺测时，采用有来源记录的设定值。Trise实验要求实测温箱温度。

## 3. 第一周与第二周成果

第一周完成MAT读取、字段检查、重复时间处理、电流方向统一、分段重采样和25°C US06可视化。

第二周完成：

- 独立25°C C/20电压曲线：放电量2.99491 Ah，全曲线RMSE 0.331927 mV、同实验插值留出RMSE 0.313534 mV。
- 三温度HPPC加载与恢复联合辨识，以及静置电压代理。
- 热容Cth和换热系数hA标定。
- 电流、电压、温度统计及经验噪声模型。
- US06、UDDS开发验证和Cycle4额外检查。

OCV代理、参考SOC与温度分组均有明确来源。2.9 Ah用于电量坐标，不作为逐日期测出的真实容量。

电气关系为`V = OCV(SOC) - R0*Icell - v1 - v2`，每个极化支路时间常数为`R*C`。热关系为`Cth*dT/dt = Pheat - hA*(T-Tamb)`。内短路使用`i_s = f_isc*V`、`Icell = i+i_s`，并保留`i_s*V`附加产热。

## 4. 两条建模路线与冻结模型

| 路线 | 入口 | 参数与主要输出 |
| --- | --- | --- |
| 固定参数基线 | `experiments/run_week2.py::run` | `results/week2/parameters/`、`validation/`、`summary/model_error_table.csv` |
| 动态标定改进 | `experiments/refine_week2.py` | `results/week2/comparisons/calibration_refinement/` |
| 第三周冻结版本model-v1 | `experiments/prepare_week3.py`准备并验证 | `results/week3/preparation/model_v1/{25C,10C,0C}.json` |

改进模型增加SOC/温度相关电阻、OCV修正和有效环境偏移，仍由`src/model.py::simulate()`计算。完整JSON中的映射必须与共享的`results/week2/parameters/ocv_*C.csv`一起使用。

model-v1保留第二周的物理电气、热参数，仅移除没有普遍一拍滞后证据的0.75电压读出混合。原第二周参数和误差表保持原样。当前及未来标定中，电压读出统一使用完整电压观测算子，默认混合系数为0。

冻结后不增加模型结构；环境偏移和热尺度通过显式干扰或敏感性情形研究。

## 5. 反馈核查结果

### 5.1 HPPC采样时序

25°C、10°C、0°C分别检查67、59、54个上升沿。电流跳变同一拍的电压下降量，相对下一拍下降量的中位比例为0.773、0.772、0.727；没有边沿满足预设的“同拍响应小于下一拍25%”判据。

这不支持普遍一拍电压滞后，故model-v1的`voltage_readout_blend=0`。数据仍可能有亚采样时序差异；本轮不再用额外系数拟合。

证据：`hppc_rising_edges.csv`、`hppc_timing_summary.json`和`hppc_onset_comparison.png`。

### 5.2 逐工况环境估计

使用内部时间戳关联试验前的静置，取最后600秒电芯温度中位数。检查静置时长、尾段漂移、样本数和距试验开始的时间，拒绝过时或未稳定的候选。

18个Cycle1至4、US06、UDDS案例中，7个通过当前质量筛选。完整记录在`pretest_ambient_estimates.csv`。

25°C US06和UDDS的前置静置温度都为25.6307°C。以该值替代环境并去掉原有效偏移时，UDDS温度RMSE由0.394降至0.157°C，但US06由0.367升至0.748°C。因此，试验前静置温度作为可检验的环境代理保留，未统一替换冻结模型的参考环境输入。

环境干扰定义见`ambient_nuisance_definition.json`。±0.35°C是敏感性幅度，不是置信区间。后续窗口分析应将环境偏移作为共同干扰量处理，不能当每拍独立温度白噪声。

`leading_rest_estimates.csv`另列数据文件内部的前置静置候选；只有静置结束后才能使用，不能倒用于更早的预测。

### 5.3 热容与产热尺度

当前Cth为65.625、57.908、58.253 J/K。反馈提供的40–50 J/K仅作为物理量级先验。

健康热方程中同时将产热、Cth、hA乘同一系数，温度轨迹保持不变；因此仅凭轨迹无法区分夹具附加热容和整体产热估计偏大。当前没有独立夹具质量或热量计数据，本轮不强行修改Cth、hA。

厂家规格给出裸电芯最大质量47 g，另一规格列含外套最大48 g；这些不是本实验实测质量。参考来源和尺度情形保存在`thermal_scale_assumptions.json`、`thermal_scale_audit.csv`。

以3.7 V、环境扰动0.35°C作稳态量级计算，现有hA对应约341、297、307 Ω的短路产热。这只是查错参照，实际窗口的可区分边界还要计算。

## 6. model-v1固定参数验证

下表使用原有环境输入与固定有效偏移，未重新拟合参数。误差方向为预测减实测。

| 温度 | 工况 | 电压RMSE/mV | 电压Bias/mV | 温度RMSE/°C | 温度Bias/°C |
| --- | --- | ---: | ---: | ---: | ---: |
| 25°C | US06 | 23.571 | +8.522 | 0.367 | +0.266 |
| 25°C | UDDS | 9.628 | -3.705 | 0.394 | -0.382 |
| 10°C | US06 | 28.422 | +8.019 | 0.328 | +0.002 |
| 10°C | UDDS | 13.626 | -4.131 | 0.229 | -0.103 |
| 0°C | US06 | 37.675 | +5.020 | 0.497 | +0.033 |
| 0°C | UDDS | 16.402 | +9.089 | 0.340 | -0.314 |

移除0.75后，25°C US06电压RMSE由17.685升至23.571 mV，10°C US06由21.025升至28.422 mV。完整新旧对比保留在`fixed_model_validation.csv`，没有按验证工况择优切换模型。

25°C、10°C用于主结果，0°C保留为补充，并保留低温RC贴边问题的记录。

## 7. 两套噪声与数据接口

`results/week3/preparation/noise/T/`中：

- `sensor_surrogate_covariance.csv`：HPPC静置电压、温度协方差及电流平台波动代理；电流项仍混有控制器纹波。
- `sensor_and_model_covariance.csv`：加入Cycle3模型误差尺度的总协方差。Cycle3残差本身已含测量噪声，通过协方差差值的半正定正部分构造附加模型项，避免简单重复相加。
- `cycle3_colored_autocovariance.npz`：约1秒采样下电压、温度的滞后自协方差，保留跨通道和时间相关性。
- `noise_interface.json`：单位、来源、采样间隔、均值偏差、通道顺序及模型哈希。

三通道协方差顺序为`[I,V,T]`；有色自协方差数组的通道为`[V,T]`，负滞后使用转置。电流暂用独立采样代理，纯传感器电流噪声尚未独立测出。

Cycle3均值偏差单独记录，不能通过当成零均值白噪声而消失。经验统计跨工况迁移仍需验证。`cycle3_effective_samples.csv`给出10秒至1小时窗口的有效样本数，不能按窗口采样点数直接累积独立信息。

原生输入和逐行残差保存在`data/processed/week3_preparation/`。公开接口见：

- `input_interface.json`：输入、模型、噪声路径及哈希；
- `operating_segments.csv`：静置/工况段的行号边界；
- `window_availability.csv`：真实连续段能支持哪些窗口长度；
- `freeze_manifest.json`：冻结文件和哈希；
- `preparation_status.json`：已完成项目和待补依据。

静置零端口电流不表示零极化或已经热平衡。相邻重叠窗口也不是独立重复实验。

## 8. 运行命令

以下命令均在项目根目录执行。重现原标定会写结果；建议写入独立本地目录。

```powershell
# 第一周测量图
.\.venv\Scripts\python.exe -B experiments\plot_us06.py

# 结构分析：不标定电池参数
.\.venv\Scripts\python.exe -B experiments\run_structural.py

# 第二周独立25°C C/20
.\.venv\Scripts\python.exe -B experiments\calibrate_ocv_25c.py --project-root . --output-dir data\processed\week2_reproduction\ocv_25c

# 第二周完整流程：基础、比较、改进和验证
.\.venv\Scripts\python.exe -B experiments\run_week2.py --project-root . --output-dir data\processed\week2_reproduction

# 第三周准备：固定物理参数，只做核查、预测和接口导出
.\.venv\Scripts\python.exe -B experiments\prepare_week3.py --project-root .

# 数值与数据检查；含合成参数恢复检查
.\.venv\Scripts\python.exe -B tests\verify_week2.py --project-root . --results-dir results\week2
.\.venv\Scripts\python.exe -B tests\check_week3_preparation.py
.\.venv\Scripts\python.exe -B tests\check_structural.py
```

仅运行基础模型：

```powershell
.\.venv\Scripts\python.exe -B -c "from pathlib import Path; from experiments.run_week2 import run; root=Path.cwd(); run(root,root/'data'/'processed'/'baseline_only')"
```

基于上述基础结果另行标定改进模型：

```powershell
.\.venv\Scripts\python.exe -B experiments\refine_week2.py --project-root . --output-dir data\processed\baseline_only
```

历史第二周结果对应当时0.75读出设置。新标定默认零混合，不承诺重现历史数字；后续主计算读取model-v1冻结参数，不运行重新标定来追逐验证误差。

## 9. 结构可隔离性分析

结构分支的符号模型与分析已整合：`src/structural_model.py`建立方程，`src/structural.py`执行分析，`experiments/run_structural.py`为独立入口。`src/model.py`继续用于数值仿真，model-v1冻结文件不变。

结果保存到`results/week1/structural/`，包含检测性、积分/混合因果矩阵、DM分解、MSO及详细方程、低指标MSO、测试选择和故障签名矩阵。`figure2_structural_isolability.png`为两模型矩阵草图，`run_summary.json`记录来源和方法。

| 参考模型 | 方程/未知量 | MSO | 低指标MSO | 保留测试 | 结构结论 |
| --- | --- | ---: | ---: | ---: | --- |
| 无热通道 | 11/10 | 1 | 1 | 1 | 电流偏置与内短路不能隔离 |
| 含热平衡及温度测量 | 14/12 | 6 | 6 | 6 | 六类参考故障两两可隔离 |

可隔离性矩阵中1表示不能隔离、0表示能隔离，对角线为1；FSM中的1则表示对故障敏感。两者含义不同。所有矩阵按model.f原顺序保存，不随图形重排。

工具箱的最小隔离测试选择对无热模型返回空集，因为没有可隔离故障对。现保留单个MSO以同时满足检测要求。原非热DM错误记录作为`legacy_DM_plot_error.txt`保留；新图直接使用同一工具箱的DM分区和整数索引，修复空方块索引问题。

按记录的方程集，MSO1有12条方程、11个未知量，MSO2有13条方程、12个未知量，冗余度均为1。`MSO_details.json`给出逐项方程、未知量及签名，便于复核。

MSO1通过电压、电量和极化动态约束电芯电流，再检查热平衡与温度测量的一致性，对内短路敏感而不包含电流偏置。MSO2去掉漏电本构方程，以反推电芯电流与测得端口电流之差作为候选漏电流，并检查其产热与温度的一致性，从而约束电流偏置。

结构分析采用常参数符号参考模型；model-v1的SOC/温度参数映射、环境干扰及模型误差下的定量分辨能力仍由后续灵敏度和D_ij计算检验。本次没有将未提供的“12条修订清单”或文献[2]阅读标记为完成。

## 10. 下一阶段

基于冻结模型核对式（7）及D_ij，再分别比较静置/工况、有/无温度测量、10秒至1小时窗口。主结论采用包含Cycle3模型误差的有色统计，并显式处理环境温度干扰。图3、图4和诊断器属于下一步计算，不以本轮准备图替代。
