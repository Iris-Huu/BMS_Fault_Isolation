
from pathlib import Path
import sys
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io import load_panasonic_data

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

df = load_panasonic_data(file_path)

# 时间转换为小时，便于观察长时间测试过程
time_h = df["Time"] / 3600

fig, axes = plt.subplots(
    4, 1,
    figsize=(14, 12),
    sharex=True
)

# 电压
axes[0].plot(time_h, df["Voltage"])
axes[0].set_ylabel("Voltage (V)")
axes[0].set_title("Panasonic 25°C C/20 OCV Test")

# 电流：使用统一的放电为正约定
axes[1].plot(
    time_h,
    df["Current_discharge_positive"]
)
axes[1].set_ylabel("Current (A)")

# 电池温度
axes[2].plot(
    time_h,
    df["Battery_Temp_degC"]
)
axes[2].set_ylabel("Battery Temp (°C)")

# 环境温度
axes[3].plot(
    time_h,
    df["Chamber_Temp_degC"]
)
axes[3].set_ylabel("Chamber Temp (°C)")
axes[3].set_xlabel("Time (h)")

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()

output_dir = PROJECT_ROOT / "archive" / "results" / "c20_exploration"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "Panasonic_C20_overview.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("图片已保存至:", output_path)
