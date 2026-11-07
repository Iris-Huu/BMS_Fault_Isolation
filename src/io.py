
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


# ==================================================
# 1. 定义统一的数据字段
# ==================================================

REQUIRED_COLUMNS = [
    "Time",
    "Voltage",
    "Current",
    "Ah",
    "Battery_Temp_degC",
    "Chamber_Temp_degC",
]


# ==================================================
# 2. 读取 MATLAB 文件并提取 meas 数据
# ==================================================

def load_panasonic_data(file_path, drop_exact_duplicates=True):
    """
    读取 Panasonic 18650PF MATLAB 数据。

    Parameters
    ----------
    file_path : str or Path
        MAT 文件路径。

    drop_exact_duplicates : bool
        是否删除完全重复的数据行，默认保留第一条。

    Returns
    -------
    pandas.DataFrame
        包含六个标准字段的数据表。
    """

    # ---------- 读取文件 ----------

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到 MAT 文件: {file_path}"
        )

    raw_data = loadmat(
        file_path,
        simplify_cells=True
    )

    # ---------- 检查 meas 结构 ----------

    if "meas" not in raw_data:
        raise KeyError(
            "MAT 文件中不存在 meas 变量"
        )

    meas = raw_data["meas"]

    if not isinstance(meas, dict):
        raise TypeError(
            "meas 不是预期的 Python 字典结构"
        )

    # ---------- 检查必要字段 ----------

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in meas
    ]

    if missing_columns:
        raise KeyError(
            f"缺少必要字段: {missing_columns}"
        )

    # ---------- 提取并统一数据类型 ----------

    arrays = {}

    for col in REQUIRED_COLUMNS:

        values = np.asarray(
            meas[col],
            dtype=np.float64
        ).reshape(-1)

        arrays[col] = values

    # ---------- 检查数据长度 ----------

    lengths = {
        col: len(values)
        for col, values in arrays.items()
    }

    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"各字段数据长度不一致: {lengths}"
        )

    # ---------- 建立 DataFrame ----------

    df = pd.DataFrame(arrays)

    original_rows = len(df)

    # ---------- 检查无效数值 ----------

    invalid_counts = {}

    for col in REQUIRED_COLUMNS:

        invalid_counts[col] = int(
            (~np.isfinite(df[col])).sum()
        )

    if any(invalid_counts.values()):

        raise ValueError(
            f"数据存在 NaN 或 Inf: {invalid_counts}"
        )

    # ---------- 检查时间顺序 ----------

    time = df["Time"].to_numpy()

    dt = np.diff(time)

    if np.any(dt < 0):

        raise ValueError(
            "Time 存在倒退，请检查原始数据"
        )

    # ---------- 删除完全重复的数据行 ----------

    removed_duplicates = 0

    if drop_exact_duplicates:

        before = len(df)

        df = df.drop_duplicates(
            subset=REQUIRED_COLUMNS,
            keep="first"
        )

        removed_duplicates = before - len(df)

    df = df.reset_index(drop=True)

    # ---------- 检查采样间隔 ----------

    time = df["Time"].to_numpy()

    dt = np.diff(time)

    zero_intervals = int(np.sum(dt == 0))

    if len(dt) > 0:

        min_dt = float(np.min(dt))
        max_dt = float(np.max(dt))
        mean_dt = float(np.mean(dt))

        large_gaps = int(np.sum(dt > 0.15))

    else:

        min_dt = max_dt = mean_dt = np.nan
        large_gaps = 0

    # ---------- 保存数据质量报告 ----------

    quality_report = {

        "file": str(file_path),

        "original_rows": original_rows,

        "final_rows": len(df),

        "removed_duplicates": removed_duplicates,

        "invalid_values": invalid_counts,

        "duplicate_time_points": zero_intervals,

        "min_sampling_interval_s": min_dt,

        "max_sampling_interval_s": max_dt,

        "mean_sampling_interval_s": mean_dt,

        "large_gaps_over_0.15s": large_gaps,

    }

    df.attrs["quality_report"] = quality_report

    return df


# ==================================================
# 3. 打印数据质量报告
# ==================================================

def print_quality_report(df):

    report = df.attrs.get("quality_report")

    if report is None:

        print("没有找到数据质量报告")

        return

    print("\n========== DATA QUALITY REPORT ==========")

    for key, value in report.items():

        print(f"{key}: {value}")

    print("==========================================")
