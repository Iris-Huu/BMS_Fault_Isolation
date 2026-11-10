
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.io import (
    load_panasonic_data,
    print_quality_report,
)

# C/20 OCV 文件路径
file_path = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Panasonic_18650PF"
    / "Panasonic 18650PF Data"
    / "25degC"
    / "C20 OCV and 1C discharge tests_start_of_tests"
    / "05-08-17_13.26 C20 OCV Test_C20_25dC.mat"
)

print("========== C/20 OCV 数据读取测试 ==========")

# 1. 使用统一接口读取数据
df = load_panasonic_data(
    file_path,
    gap_threshold=0.15,
)

# 2. 打印数据质量报告
print_quality_report(df)

# 3. 检查时间范围
time = df["Time"].to_numpy()
dt = np.diff(time)

print("\n========== 时间信息 ==========")

print("起始时间:", time[0], "s")
print("结束时间:", time[-1], "s")
print("总记录数:", len(df))
print("平均采样间隔:", np.mean(dt), "s")
print("最小采样间隔:", np.min(dt), "s")
print("最大采样间隔:", np.max(dt), "s")
print("时间严格递增:", np.all(dt > 0))

# 4. 检查电流
print("\n========== 电流信息 ==========")

print("原始电流最小值:", df["Current"].min())
print("原始电流最大值:", df["Current"].max())

print(
    "统一电流最小值:",
    df["Current_discharge_positive"].min()
)

print(
    "统一电流最大值:",
    df["Current_discharge_positive"].max()
)

print(
    "电流为零的记录数:",
    int((df["Current"] == 0).sum())
)

print(
    "非零电流的记录数:",
    int((df["Current"] != 0).sum())
)

# 5. 检查电压和温度范围
print("\n========== 电压与温度信息 ==========")

for col in [
    "Voltage",
    "Battery_Temp_degC",
    "Chamber_Temp_degC",
    "Ah",
]:

    print(
        f"{col}:",
        df[col].min(),
        "至",
        df[col].max()
    )

# 6. 查看部分数据
print("\n========== 数据前五行 ==========")

print(
    df[
        [
            "Time",
            "Voltage",
            "Current",
            "Current_discharge_positive",
            "Ah",
            "Battery_Temp_degC",
            "Chamber_Temp_degC",
        ]
    ].head()
)

print("\nC/20 OCV 数据读取测试结束。")
