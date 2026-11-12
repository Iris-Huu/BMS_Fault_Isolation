
from pathlib import Path
import sys

import matplotlib.pyplot as plt


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ensure Python can find the src package
sys.path.insert(0, str(PROJECT_ROOT))

from src.io import load_panasonic_data


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Panasonic_18650PF"
    / "Panasonic 18650PF Data"
    / "25degC"
    / "Drive cycles"
    / "03-20-17_01.43 25degC_US06_Pan18650PF.mat"
)

FIGURE_DIR = PROJECT_ROOT / "results" / "week1" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Load data
# ============================================================

df = load_panasonic_data(DATA_PATH)

time_s = df["Time"]

voltage = df["Voltage"]
current = df["Current_discharge_positive"]
battery_temp = df["Battery_Temp_degC"]
chamber_temp = df["Chamber_Temp_degC"]


# ============================================================
# 3. Individual figures
# ============================================================

# Voltage
plt.figure(figsize=(12, 4))

plt.plot(time_s, voltage, linewidth=0.8)

plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("Panasonic 18650PF - US06 Battery Voltage (25°C)")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "us06_voltage.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# Current
plt.figure(figsize=(12, 4))

plt.plot(time_s, current, linewidth=0.8)

plt.xlabel("Time (s)")
plt.ylabel("Current (A)")
plt.title("Panasonic 18650PF - US06 Battery Current (25°C)")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "us06_current.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# Battery and chamber temperature
plt.figure(figsize=(12, 4))

plt.plot(
    time_s,
    battery_temp,
    label="Battery Temperature",
    linewidth=1
)

plt.plot(
    time_s,
    chamber_temp,
    label="Chamber Temperature",
    linewidth=1
)

plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.title("Panasonic 18650PF - US06 Temperature (25°C)")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "us06_temperature.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 4. Composite figure for project submission
# ============================================================

fig, axes = plt.subplots(
    4,
    1,
    figsize=(12, 11),
    sharex=True
)

# Battery voltage
axes[0].plot(time_s, voltage, linewidth=0.8)
axes[0].set_ylabel("Voltage (V)")
axes[0].set_title("Battery Voltage")
axes[0].grid(True, alpha=0.3)

# Battery current
axes[1].plot(time_s, current, linewidth=0.8)
axes[1].set_ylabel("Current (A)")
axes[1].set_title("Battery Current (Discharge Positive)")
axes[1].grid(True, alpha=0.3)

# Battery temperature
axes[2].plot(time_s, battery_temp, linewidth=1)
axes[2].set_ylabel("Temperature (°C)")
axes[2].set_title("Battery Surface Temperature")
axes[2].grid(True, alpha=0.3)

# Chamber temperature
axes[3].plot(time_s, chamber_temp, linewidth=1)
axes[3].set_ylabel("Temperature (°C)")
axes[3].set_title("Chamber Temperature")
axes[3].set_xlabel("Time (s)")
axes[3].grid(True, alpha=0.3)

fig.suptitle(
    "Panasonic 18650PF - 25°C US06 Test",
    fontsize=14
)

fig.tight_layout(rect=[0, 0, 1, 0.97])

composite_path = FIGURE_DIR / "Panasonic_25degC_US06.png"

fig.savefig(
    composite_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 5. Output information
# ============================================================

print("US06 figures generated successfully!")
print("Figure directory:", FIGURE_DIR)
print("Composite figure:", composite_path)
print("Number of data points:", len(df))
print("Time range:", time_s.min(), "-", time_s.max(), "s")
print("Current convention: discharge positive")
