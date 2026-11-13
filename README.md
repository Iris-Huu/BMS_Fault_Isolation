# BMS 电流传感器偏置与内短路故障可隔离性研究

本项目研究：能否利用单体锂离子电池的电压、电流和温度测量，区分电流传感器偏置与内短路故障。

项目使用二阶 RC 等效电路和集中参数热模型，开展实验数据检查、参数辨识、模型验证及后续故障可隔离性研究。

**当前进度：已建立第二周基础模型与改进模型，完成三个温度下的参数计算和开发验证；最终科学验收仍有条件限制。**

当前结果可以支持后续模型敏感性和诊断算法原型研究，尚不能直接作为已获实验验证的微小故障检测或隔离边界。

## 1. 研究目标

本项目主要回答以下问题：

1. 没有热通道和加入热通道时，两类故障能否在模型结构上区分？
2. 在真实驾驶工况、测量噪声和模型误差下，两类故障的响应差异有多大？
3. 温度测量、工况激励和参数不确定性怎样影响可区分能力？

当前工作树主要包含数据处理、数值建模、标定和验证代码。结构可隔离性分析与第三周定量可区分度程序，需要依据对应代码和结果另行确认，不能仅凭数值模型中的故障接口视为已经完成。

## 2. 项目目录

```text
BMS_Fault_Isolation/
├── README.md                 项目正式说明
├── requirements.txt          Python依赖版本
├── .gitignore                数据、环境和临时文件排除规则
├── src/                      可复用的数据、模型和辨识方法
├── experiments/              实验入口与模型比较
│   └── comparisons/          参数映射和熵热比较
├── tests/                    数据读取与数值实现检查
├── tools/                    数据审计、结果汇总和文件探查
│   └── inspection/           原始文件结构与缺失数据复查
├── data/
│   ├── raw/                  原始实验数据，不提交Git
│   └── processed/            本地处理数据及复现输出，不提交Git
├── results/
│   ├── week1/figures/        第一周US06可视化
│   └── week2/
│       ├── audit/            数据来源、日期、容量坐标和OCV证据
│       ├── calibration/      HPPC逐脉冲辨识及筛选记录
│       ├── parameters/       基础模型参数和共享OCV代理表
│       ├── noise/            协方差、经验噪声及统计假设
│       ├── validation/       基础模型预测图和轨迹
│       ├── ocv_25c/          独立25°C C/20电压代理拟合
│       ├── comparisons/     改进模型及比较实验
│       └── summary/          参数、误差、来源和状态汇总
└── archive/
    ├── experiments/          早期探索脚本
    ├── results/              对应探索结果
    └── backups/              尚未确认可完全替代的本地快照
```

原始数据、处理数据、环境、缓存、凭据及历史快照不进入Git。部分逐行轨迹和连续残差块也仅保存在本地；新克隆的仓库不包含运行全部数据检查所需的所有文件。

## 3. Python环境

现有项目使用 Python 3.12。主要计算依赖为：

- NumPy、Pandas：数组和数据表；
- SciPy：MAT读取、积分、插值和参数优化；
- scikit-learn：单调回归；
- Matplotlib：绘图。

在项目根目录使用 PowerShell：

```text
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

后续命令直接调用虚拟环境中的Python，不要求先激活环境。

`requirements.txt` 记录现有环境的固定版本。安装成功与否需要在目标环境检查，不能把版本清单等同于已经验证的跨平台兼容性保证。

## 4. 实验数据及来源

### 4.1 Panasonic 18650PF

主模型使用 Panasonic 18650PF 数据，标称容量为2.9 Ah。数据由 Phillip Kollmeyer 提供，实验在 University of Wisconsin–Madison 开展。

官方来源：

[Panasonic 18650PF Li-ion Battery Data，版本1](https://data.mendeley.com/datasets/wykht8y7tg/1)\
DOI：`10.17632/wykht8y7tg.1`

数据包含HPPC脉冲、驾驶工况、阻抗及相关实验。当前主要建模温度为25°C、10°C和0°C。

### 4.2 LG 18650HG2

LG数据属于另一种电芯，标称容量为3 Ah，实验在 McMaster University 开展。

官方来源：

[LG 18650HG2 Li-ion Battery Data，版本2](https://data.mendeley.com/datasets/b5mj79w5w9/2)\
DOI：`10.17632/b5mj79w5w9.2`

当前项目已对LG原始文件结构和低倍率数据进行复查，但没有把LG参数作为Panasonic电芯参数使用，也没有完成本工程中的LG独立电热模型验证。

### 4.3 本地存放方式

```text
data/raw/
├── Panasonic_18650PF/
│   ├── Panasonic 18650PF Data/25degC/
│   ├── 10degC/
│   ├── 0degC/
│   └── 其他原始实验目录
└── LG_18650HG2/
    ├── LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/
    └── LG_HG2_Prepared_Dataset_McMasterUniversity_Jan_2020/
```

保留原始文件名、嵌套目录和数据集附带说明。脚本根据这些信息寻找实验文件。

### 4.4 已确认的数据限制

- 当前本地Panasonic数据中找到25°C C/20实验，未找到10°C、0°C独立C/20。该结论限定于本地复查范围，不代表所有公开版本都不存在。
- LG有相应低温C/20数据，但不能直接替代Panasonic OCV。
- 所选Panasonic低温HPPC、Cycle1/2、US06、UDDS文件的温箱温度缺测，模型采用显式记录的设定值回退。
- Panasonic其他0°C实验及Trise数据存在有效温箱记录，不能把缺失情况推广到整个数据集。
- LG原始文件没有`Chamber_Temp_degC`字段，电芯温度不能直接代替环境温度。

复查证据保存在：

- `results/week2/audit/source_data_recheck.csv`
- `results/week2/audit/slow_rate_segments_recheck.csv`
- `results/week2/audit/raw_schema_and_archives_recheck.json`

数据使用与论文写作应引用原始数据来源。

## 5. 已完成的工作

### 第一周：数据准备

- 下载并检查实验数据；
- 建立Panasonic MAT读取方法；
- 检查无效数值、重复记录和时间顺序；
- 保留原始电流，统一计算时放电为正；
- 建立分段重采样与质量报告；
- 生成25°C US06电压、电流、电芯温度和温箱温度图。

### 第二周：模型标定与开发验证

- 拟合独立25°C C/20放电电压代理；
- 建立三温度HPPC静置电压代理；
- 联合拟合HPPC加载与恢复段的二阶RC参数；
- 辨识基础和改进热模型参数；
- 估计静置噪声，记录量化、电流代理和时间相关假设；
- 完成US06、UDDS开发验证；
- 比较固定参数、HPPC映射、有效熵热和动态标定改进方案；
- 补充Cycle4固定参数验证及经验误差包络覆盖率检查；
- 审计源文件日期、容量参考及SOC坐标。

这些计算成果不等同于全部参数均已获得独立物理验证。

## 6. 模型与参数辨识方法

### 6.1 数据约定

- 时间单位：秒；
- 电压：V；
- 电流：A，放电为正；
- 容量：Ah；
- 电阻：Ω；
- 电容：F；
- 热容：J/K；
- 换热系数：W/K。

原始`Current`列保留，计算使用：

```text
Current_discharge_positive = -Current
```

时间必须严格递增。缺失环境温度默认报错，只有显式指定温箱设定值时才允许补充并记录来源。Trise实验禁止使用这一恒温回退。

### 6.2 OCV–SOC

OCV表示开路电压，SOC表示荷电状态。

独立25°C C/20实验使用该放电段自身的Ah跨度定义相对SOC，再通过SOC分箱中位数和PCHIP插值建立电压曲线。

已有结果：

- 放电段容量：2.99491 Ah；
- 全曲线RMSE：0.331927 mV；
- 同实验内交错留出插值RMSE：0.313534 mV。

这是低倍率带载电压代理。上述误差不等于独立平衡OCV的测量准确度。

三温度数值模型使用另一套基础曲线：

```text
results/week2/parameters/ocv_25C.csv
results/week2/parameters/ocv_10C.csv
results/week2/parameters/ocv_0C.csv
```

这些曲线来自HPPC脉冲前及充分恢复后的静置电压。改进模型在其上增加有界OCV修正。

两套曲线的SOC参考不同，不能未经坐标转换直接混用。

### 6.3 二阶RC电路

模型包含瞬时内阻和两个极化支路：

$$
V=OCV(SOC)-R_0I_{\mathrm{cell}}-V_1-V_2
$$

$$
\dot V_j=-\frac{V_j}{R_jC_j}+\frac{I_{\mathrm{cell}}}{C_j}
$$

其中，$R_0$ 描述瞬时压降，两组RC支路描述不同速度的电压恢复过程，时间常数为 $\tau_j=R_jC_j$。

HPPC辨识联合使用约10秒低倍率加载和至少600秒连续恢复，并记录误差、边界和时间尺度警告。

25°C、10°C、0°C分别得到28、25、23组联合拟合。0°C其中12组参数接近边界，不能仅凭参数为正和拟合误差小判断其唯一可辨识。

### 6.4 热模型

$$
C_{\mathrm{th}}\dot T=P_{\mathrm{heat}}-hA(T-T_{\mathrm{amb}})
$$

- `C_th`：等效热容；
- `hA`：等效换热能力；
- `P_heat`：模型估计的产热。

纯零产热静置降温主要确定`hA/C_th`。现有代码使用带产热的Cycle1/2数据，在产热和环境假设下分别估计两个参数。

基础模型采用积分热平衡；改进模型拟合正向温度轨迹，并增加有效环境偏移。该偏移混合了环境、测温和模型误差，不是独立辨识的传感器偏置。

内短路保留漏电流：

$$
i_s=f_{\mathrm{isc}}V,\qquad I_{\mathrm{cell}}=i+i_s
$$

并包含额外短路产热`i_s V`。

### 6.5 噪声估计

通道顺序固定为：

```text
[current_A, voltage_V, temperature_C]
```

静置尾段去趋势后的实测残差标准差为：

| 温度   | 电压／mV  | 温度／°C   |
| ---- | ------ | ------- |
| 25°C | 0.1328 | 0.05523 |
| 10°C | 0.1134 | 0.06514 |
| 0°C  | 0.1352 | 0.05566 |

静置电流记录为零，无法独立辨识真实电流传感器噪声。脉冲电流波动代理包含控制器纹波。

温度跳变、量化项、时间相关高斯近似及数值正则均需按其假设解释。模型预测误差不能全部归为传感器噪声。

`noise/*/sigma_model_error_sensitivity.csv`来自基础模型残差，不能直接作为改进模型的匹配残差协方差。

## 7. 基础模型与改进模型

| 项目   | 基础模型          | 当前改进模型                                |
| ---- | ------------- | ------------------------------------- |
| 电气参数 | 各温度固定RC参数     | SOC相关电阻及温度修正                          |
| 时间常数 | HPPC汇总值       | 保留HPPC提供的时间常数                         |
| OCV  | HPPC静置代理      | 共享代理加拟合修正                             |
| 动态标定 | 主要使用HPPC      | HPPC先验与Cycle1/2                       |
| 电压读出 | 当前物理电压        | 前后采样物理电压混合                            |
| 热参数  | 积分热平衡         | 正向轨迹拟合及有效环境偏移                         |
| 参数目录 | `parameters/` | `comparisons/calibration_refinement/` |
| 验证目录 | `validation/` | `comparisons/calibration_refinement/` |

两条路线共用`src/model.py::simulate()`。改进模型必须读取完整JSON及基础OCV，不能只取参考参数表中的五个RC数值代替全部映射。

当前三温度的`voltage_readout_blend`均为0.75。在仿真中：

[\
V\_{\mathrm{out},k}=0.25V_k+0.75V\_{k-1}\
]

它是经验测量响应机制，没有独立证明为真实硬件延迟。

当前标定设计混合电流，正向仿真混合完整物理电压，两者一般不严格等价。后续灵敏度研究前需要统一或论证这一近似。

另外两项比较实验保留：

- `soc_temperature_maps`：HPPC参数映射，部分工况改善、部分变差；
- `entropy_sensitivity`：有效熵热探索，没有改善现有六组温度RMSE，未被当前推荐模型采用。

模型选择入口：

`results/week2/summary/model_selection.json`

## 8. 运行与复现

以下命令均在项目根目录执行。

### 8.1 检查数据读取

```text
.\.venv\Scripts\python.exe -B tests\check_us06_io.py
.\.venv\Scripts\python.exe -B tests\check_c20_io.py
.\.venv\Scripts\python.exe -B tests\check_hppc_io.py
```

这些脚本检查读取和数据约定，不代表参数辨识或预测精度已通过科学验收。

### 8.2 第一周可视化

```text
.\.venv\Scripts\python.exe -B experiments\plot_us06.py
```

输出到`results/week1/figures/`，重复运行会更新对应图片。

### 8.3 完整第二周复现

建议使用被Git忽略的独立输出目录，保留已有正式结果：

```text
# 独立25°C C/20实验
.\.venv\Scripts\python.exe -B experiments\calibrate_ocv_25c.py --project-root . --output-dir data\processed\week2_reproduction\ocv_25c

# 三温度基础标定、比较实验、动态改进、补充验证与汇总
.\.venv\Scripts\python.exe -B experiments\run_week2.py --project-root . --output-dir data\processed\week2_reproduction
```

完整流程依次运行：

```text
基础模型
→ HPPC参数映射比较
→ 有效熵热比较
→ 日期与容量审计
→ 动态标定改进
→ OCV、经验噪声及额外验证
→ 结果汇总
```

C/20独立拟合不包含在总入口中，因此单独执行。

若要复查Panasonic与LG全部原始字段和低倍率片段，另执行：

```text
.\.venv\Scripts\python.exe -B tools\inspection\recheck_source_data.py --project-root . --output-dir data\processed\week2_reproduction\audit
```

该复查脚本不会由总入口自动调用。

不指定输出目录时，总入口默认写入`results/week2/`，会更新已有生成结果。

### 8.4 分别运行基础与改进路线

仅运行基础模型，使用当前已有的Python函数入口：

```text
.\.venv\Scripts\python.exe -B -c "from pathlib import Path; from experiments.run_week2 import run; root=Path.cwd(); run(root, root/'data'/'processed'/'week2_baseline')"
```

随后基于该目录中的基础参数运行改进模型：

```text
.\.venv\Scripts\python.exe -B experiments\refine_week2.py --project-root . --output-dir data\processed\week2_baseline
```

上述分步方式只执行相应标定与US06/UDDS预测，不自动补齐全部审计、经验误差包络和汇总。完整复现使用上一节总入口。

### 8.5 实现与结果检查

完整复现后：

```text
.\.venv\Scripts\python.exe -B tests\verify_week2.py --project-root . --results-dir data\processed\week2_reproduction
```

检查内容包括电荷守恒、RC响应、故障关系、合成参数恢复、源文件哈希、协方差和部分结果一致性。

该脚本包含合成数据拟合；通过不代表实测参数已证明唯一。

检查既有正式结果时，将`--results-dir`改为`results\week2`。完整检查还需要本地原始数据和生成的经验残差块；新克隆的仓库本身不包含全部数据。

## 9. 第二周验证结果

误差方向为：

```text
预测值 − 实测值
```

RMSE表示均方根误差，Bias表示平均偏差。下表取自现有基础／改进比较文件，单位分别为mV和°C。

| 温度   | 工况   | 基础电压RMSE | 改进电压RMSE | 基础温度RMSE | 改进温度RMSE |
| ---- | ---- | -------- | -------- | -------- | -------- |
| 25°C | US06 | 30.549   | 17.685   | 1.003    | 0.367    |
| 25°C | UDDS | 16.237   | 8.702    | 0.533    | 0.394    |
| 10°C | US06 | 51.205   | 21.025   | 2.158    | 0.328    |
| 10°C | UDDS | 29.933   | 12.967   | 0.612    | 0.229    |
| 0°C  | US06 | 85.434   | 34.827   | 2.017    | 0.497    |
| 0°C  | UDDS | 38.400   | 15.788   | 0.632    | 0.340    |

来源：`results/week2/summary/refinement_comparison.csv`。

改进模型在六组开发验证中均降低了两种RMSE，但仍存在：

- 电压瞬态最大误差；
- 温度方向性偏差；
- 工况和参数支持范围限制；
- 概率误差包络覆盖不足。

25°C、0°C UDDS的改进温度Bias分别约为−0.382°C、−0.314°C。

额外Cycle4验证的电压RMSE为39.278、15.529、20.594 mV，对应温度RMSE为0.262、0.233、0.242°C。

### 数据划分

- HPPC：初始RC、静置电压代理及噪声；
- Cycle1/2：动态电气和热标定；
- Cycle3：经验预测误差带；
- US06/UDDS：开发验证；
- Cycle4：额外固定参数验证。

US06/UDDS未用于参数数值拟合，但在开发过程中已被反复查看，不属于完全未接触的盲测。

验证在初始化后不使用实测电压、实测电芯温度反馈修正状态。误差按原始有效采样点等权统计，不是持续时间加权。

## 10. 结果导航与适用限制

### 主要结果

| 内容       | 路径                                                                              |
| -------- | ------------------------------------------------------------------------------- |
| 基础参数     | `results/week2/parameters/{25C,10C,0C}.json`                                    |
| 共享OCV代理  | `results/week2/parameters/ocv_*C.csv`                                           |
| 基础误差     | `results/week2/summary/model_error_table.csv`                                   |
| 当前完整改进参数 | `results/week2/comparisons/calibration_refinement/{25C,10C,0C}/parameters.json` |
| 改进误差及图   | `results/week2/comparisons/calibration_refinement/`                             |
| 改进参考参数表  | `results/week2/summary/refined_parameter_table.csv`                             |
| 新旧比较     | `results/week2/summary/refinement_comparison.csv`                               |
| 噪声与假设    | `results/week2/noise/`                                                          |
| 来源与缺失审计  | `results/week2/audit/`                                                          |
| 当前完成状态   | `results/week2/summary/week2_completion_status.json`                            |

不同目录下的同名参数或误差表属于不同模型，不可混用。

### 主要限制

1. Panasonic低温独立C/20尚未在本地找到，三温度OCV以代理方法运行。
2. 2.9 Ah是参考电量坐标，不是逐日期独立辨识的真实平衡容量。
3. 低温环境温度采用设定值，热参数受产热和环境假设影响。
4. 纯静置降温不能分别确定Cth、hA；现有估计依赖带产热的工况数据。
5. 参数贴边和先验约束表明物理可辨识性仍需检查。
6. 经验读出机制未获独立硬件验证，标定与仿真形式尚需统一。
7. 电流传感器噪声不能与控制器纹波独立分离。
8. 约1秒静置噪声相关性向更高采样率外推属于假设。
9. Cycle3误差分位数与环境敏感性组合不是具有保证的统计置信区间。
10. 25°C、10°C、0°C US06电压包络实际覆盖率约为73.23%、77.92%、86.52%，低于名义95%。

提高健康模型预测精度，不等于已经证明故障隔离性能改善。

## 11. 后续研究

下一阶段计划：

1. 固定数值模型、故障变量、测量方程和噪声口径；
2. 统一读出混合的标定与正向仿真形式；
3. 检查参数查询取整对小扰动导数的影响；
4. 核对无热模型和含热模型的结构可隔离性及对应代码；
5. 沿实际工况轨迹计算故障灵敏度；
6. 研究噪声、量化、模型偏差和参数不确定性下的定量可区分度；
7. 使用独立工况及可信故障情景评价误报、检测与隔离性能。

历史探索脚本和失败结果保留在`archive/`，用于追溯，不用于替代当前推荐参数。

项目只维护根目录这一份正式README。CSV、JSON及图片继续作为机器可读结果和研究证据保留。
