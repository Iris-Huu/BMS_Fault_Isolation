
import sys
from pathlib import Path
import numpy as np

# 定位项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 让 Python 能找到 src 文件夹
sys.path.insert(0, str(PROJECT_ROOT))

from src.io import (
    load_panasonic_data,
    print_quality_report,
    resample_data,
    print_resampling_report,
)

# Panasonic 25°C US06 数据路径
file_path = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Panasonic_18650PF"
    / "Panasonic 18650PF Data"
    / "25degC"
    / "Drive cycles"
    / "03-20-17_01.43 25degC_US06_Pan18650PF.mat"
)

# 1. 读取并清洗原始数据
df = load_panasonic_data(
    file_path,
    gap_threshold=0.15,
)

print("\n========== 第一步：数据读取与清洗 ==========")
print_quality_report(df)

# 2. 检查电流符号
print("\n========== 第二步：电流符号检查 ==========")

print(
    df[
        ["Current", "Current_discharge_positive"]
    ].head(10)
)

print("\n原始电流最小值：", df["Current"].min())
print("原始电流最大值：", df["Current"].max())

print(
    "放电为正电流最小值：",
    df["Current_discharge_positive"].min()
)

print(
    "放电为正电流最大值：",
    df["Current_discharge_positive"].max()
)

# 3. 检查时间是否严格递增
time = df["Time"].to_numpy()

print("\n========== 第三步：时间检查 ==========")
print("时间是否严格递增：", (time[1:] > time[:-1]).all())
print("处理后数据行数：", len(df))

# 4. 执行重采样
resampled_df = resample_data(
    df,
    target_interval=0.1,
    gap_threshold=0.15,
)

print("\n========== 第四步：重采样检查 ==========")
print_resampling_report(resampled_df)

print("\n重采样后前五行：")
print(resampled_df.head())

print("\n重采样后时间是否严格递增：",
      (resampled_df["Time"].diff().dropna() > 0).all())


# 5. 统计重采样后的实际采样间隔
print("\n========== 第五步：采样间隔分布 ==========")

dt = np.diff(resampled_df["Time"].to_numpy())

print("最小采样间隔：", dt.min())
print("最大采样间隔：", dt.max())
print("平均采样间隔：", dt.mean())

print("小于 0.1 s 的间隔数量：", np.sum(dt < 0.1 - 1e-8))
print("约等于 0.1 s 的间隔数量：", np.sum(np.isclose(dt, 0.1, atol=1e-8)))
print("大于 0.15 s 的间隔数量：", np.sum(dt > 0.15))

print("\n前 20 个采样间隔：")
print(dt[:20])

# 6. 计算电池温度数据的实际分辨率
print("\n========== 第六步：温度分辨率检查 ==========")

# 使用清洗后的原始数据，而不是重采样数据
temp_values = df["Battery_Temp_degC"].dropna().to_numpy()

# 提取所有不同的温度值，并从小到大排序
unique_temps = np.sort(np.unique(temp_values))

# 计算相邻不同温度值之间的差值
temp_steps = np.diff(unique_temps)

# 排除数值计算产生的极小误差
temp_steps = temp_steps[temp_steps > 1e-8]

print("不同温度值的数量：", len(unique_temps))
print("温度最小值：", unique_temps.min(), "°C")
print("温度最大值：", unique_temps.max(), "°C")

print("最小温度变化步长：", temp_steps.min(), "°C")
print("最大温度变化步长：", temp_steps.max(), "°C")
print("平均温度变化步长：", temp_steps.mean(), "°C")

# 统计最常见的温度变化步长
steps, counts = np.unique(
    np.round(temp_steps, 4),
    return_counts=True
)

order = np.argsort(counts)[::-1]

print("\n最常见的温度变化步长：")
for i in order[:10]:
    print(f"{steps[i]:.4f} °C：出现 {counts[i]} 次")
    
print("\n========== 测试结束 ==========")
