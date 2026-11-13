
from pathlib import Path
import sys
import numpy as np

# 获取项目根目录，并加入 Python 模块搜索路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io import load_panasonic_data


DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "Panasonic_18650PF"
    / "Panasonic 18650PF Data"
    / "25degC"
    / "C20 OCV and 1C discharge tests_start_of_tests"
    / "05-08-17_13.26 C20 OCV Test_C20_25dC.mat"
)


df = load_panasonic_data(DATA_PATH)

time_h = df["Time"].to_numpy() / 3600
current = df["Current_discharge_positive"].to_numpy()

# 将电流分为放电、充电和静置三种状态
threshold = 0.01

state = np.zeros(len(current), dtype=int)
state[current > threshold] = 1
state[current < -threshold] = -1

state_names = {
    1: "Discharge",
    0: "Rest",
    -1: "Charge",
}

# 找到状态发生变化的位置
change_indices = np.where(state[1:] != state[:-1])[0] + 1

boundaries = np.r_[0, change_indices, len(df)]

print("\nC/20 test phase summary")
print("=" * 65)

for i in range(len(boundaries) - 1):
    start = boundaries[i]
    end = boundaries[i + 1]

    segment = df.iloc[start:end]

    print(
        f"{i + 1}. "
        f"{state_names[state[start]]:10s} | "
        f"{time_h[start]:8.3f} h - "
        f"{time_h[end - 1]:8.3f} h | "
        f"Current = {segment['Current_discharge_positive'].mean():.5f} A | "
        f"Voltage = {segment['Voltage'].min():.3f} - "
        f"{segment['Voltage'].max():.3f} V"
    )

print("\nTemperature range")
print("=" * 65)
print(
    "Battery temperature:",
    df["Battery_Temp_degC"].min(),
    "~",
    df["Battery_Temp_degC"].max(),
    "°C",
)
print(
    "Chamber temperature:",
    df["Chamber_Temp_degC"].min(),
    "~",
    df["Chamber_Temp_degC"].max(),
    "°C",
)
