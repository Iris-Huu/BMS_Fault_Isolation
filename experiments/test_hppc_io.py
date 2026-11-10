
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.io import (
    load_panasonic_data,
    print_quality_report,
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Panasonic_18650PF"
    / "Panasonic 18650PF Data"
    / "25degC"
    / "5 pulse disch"
)

MAT_FILES = [
    "03-11-17_08.47 25degC_5Pulse_HPPC_Pan18650PF.mat",
    "03-11-17_10.10 3390_dis5_10p.mat",
]

for filename in MAT_FILES:

    print("\n\n" + "=" * 65)
    print("正在测试:", filename)
    print("=" * 65)

    file_path = DATA_DIR / filename

    df = load_panasonic_data(
        file_path,
        gap_threshold=0.15,
    )

    print_quality_report(df)

    print("\n电流范围检查：")

    print(
        "原始 Current:",
        df["Current"].min(),
        "至",
        df["Current"].max(),
    )

    print(
        "放电为正 Current:",
        df["Current_discharge_positive"].min(),
        "至",
        df["Current_discharge_positive"].max(),
    )

    print("\n时间范围：")

    print("起始时间:", df["Time"].iloc[0], "s")
    print("结束时间:", df["Time"].iloc[-1], "s")

    print(
        "时间严格递增:",
        np.all(np.diff(df["Time"]) > 0),
    )

    print("最终数据行数:", len(df))

print("\n两份 HPPC 文件读取测试结束。")
