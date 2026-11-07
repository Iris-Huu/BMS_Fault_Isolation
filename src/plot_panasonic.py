
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


# 1. Locate the project and the Panasonic dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "panasonic_18650pf"

target_name = "03-20-17_01.43 25degC_US06_Pan18650PF.mat"
files = list(DATA_DIR.rglob(target_name))

if not files:
    raise FileNotFoundError(
        f"Cannot find {target_name} under {DATA_DIR}"
    )

# 2. Load the MATLAB data
data = loadmat(files[0])
m = data["meas"][0, 0]

time = np.asarray(m["Time"]).ravel()
voltage = np.asarray(m["Voltage"]).ravel()
current = np.asarray(m["Current"]).ravel()
temperature = np.asarray(m["Battery_Temp_degC"]).ravel()

# 3. Check the data
print("Dataset:", files[0].name)
print("Number of samples:", len(time))
print(f"Duration: {time.min():.2f} to {time.max():.2f} s")
print(f"Voltage range: {voltage.min():.3f} to {voltage.max():.3f} V")
print(f"Current range: {current.min():.3f} to {current.max():.3f} A")
print(
    f"Battery temperature range: "
    f"{temperature.min():.3f} to {temperature.max():.3f} °C"
)

# 4. Plot voltage, current and temperature
signals = [
    (voltage, "Voltage (V)", "Voltage"),
    (current, "Current (A)", "Current"),
    (temperature, "Battery Temperature (°C)", "Temperature"),
]

for values, ylabel, name in signals:
    plt.figure(figsize=(10, 4))
    plt.plot(time, values)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(f"Panasonic 18650PF - US06 at 25 degC: {name}")
    plt.grid(True)
    plt.tight_layout()

    output_path = PROJECT_DIR / "figures" / f"panasonic_us06_{name.lower()}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print("Saved figure:", output_path)

plt.show()
