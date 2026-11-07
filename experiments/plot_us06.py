
from pathlib import Path
import matplotlib.pyplot as plt

from src.io import load_panasonic_data


# 1. 项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

figure_dir = PROJECT_ROOT / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)


# 2. 读取数据
df = load_panasonic_data(file_path)

time = df["Time"]


# 3. 电压曲线
plt.figure(figsize=(12, 5))

plt.plot(time, df["Voltage"], linewidth=0.8)

plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("Panasonic 18650PF - US06 Voltage at 25°C")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    figure_dir / "us06_voltage.png",
    dpi=300
)

plt.close()


# 4. 电流曲线
plt.figure(figsize=(12, 5))

plt.plot(time, df["Current"], linewidth=0.8)

plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.title("Panasonic 18650PF - US06 Current at 25°C")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    figure_dir / "us06_current.png",
    dpi=300
)

plt.close()


# 5. 温度曲线
plt.figure(figsize=(12, 5))

plt.plot(
    time,
    df["Battery_Temp_degC"],
    label="Battery Temperature",
    linewidth=1
)

plt.plot(
    time,
    df["Chamber_Temp_degC"],
    label="Chamber Temperature",
    linewidth=1
)

plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.title("Panasonic 18650PF - US06 Temperature at 25°C")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    figure_dir / "us06_temperature.png",
    dpi=300
)

plt.close()


print("三张实验曲线已经生成！")
print("保存位置:", figure_dir)
