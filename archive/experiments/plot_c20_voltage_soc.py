
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.io import load_panasonic_data


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Panasonic_18650PF"
    / "Panasonic 18650PF Data"
    / "25degC"
    / "C20 OCV and 1C discharge tests_start_of_tests"
    / "05-08-17_13.26 C20 OCV Test_C20_25dC.mat"
)

OUTPUT_DIR = PROJECT_ROOT / "archive" / "results" / "c20_exploration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = load_panasonic_data(DATA_PATH)

time_h = df["Time"].to_numpy() / 3600
voltage = df["Voltage"].to_numpy()
current = df["Current_discharge_positive"].to_numpy()
ah = df["Ah"].to_numpy()

# 先观察 Ah 的实际变化，不自行假设其符号定义
print("Ah information")
print("Minimum:", ah.min(), "Ah")
print("Maximum:", ah.max(), "Ah")
print("Initial:", ah[0], "Ah")
print("Final:", ah[-1], "Ah")

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

axes[0].plot(time_h, voltage)
axes[0].set_ylabel("Voltage (V)")
axes[0].set_title("Panasonic C/20 Voltage and Ah Analysis")
axes[0].grid(True, alpha=0.3)

axes[1].plot(time_h, current)
axes[1].set_ylabel("Current (A)")
axes[1].grid(True, alpha=0.3)

axes[2].plot(time_h, ah)
axes[2].set_ylabel("Ah (Ah)")
axes[2].set_xlabel("Time (h)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()

output_path = OUTPUT_DIR / "Panasonic_C20_voltage_ah.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nFigure saved to:", output_path)
