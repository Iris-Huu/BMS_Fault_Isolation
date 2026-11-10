
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

# 项目模型约定：放电为正，充电为负
CURRENT_SIGN_CONVENTION = "discharge_positive"

# 新增的统一电流字段
STANDARD_CURRENT_COLUMN = "Current_discharge_positive"


# ==================================================
# 2. 读取 Panasonic MAT 文件
# ==================================================

def load_panasonic_data(
    file_path,
    drop_exact_duplicates=True,
    drop_duplicate_times=True,
    gap_threshold=0.15,
):
    """
    读取 Panasonic 18650PF MATLAB 数据，并进行基础清洗。

    Parameters
    ----------
    file_path : str or Path
        MAT 文件路径。

    drop_exact_duplicates : bool
        是否删除完全重复的数据行。

    drop_duplicate_times : bool
        是否删除重复时间点，保留第一条记录。

    gap_threshold : float
        判断较大采样间隔的阈值，单位为秒。

    Returns
    -------
    pandas.DataFrame
        包含原始字段和统一电流字段的数据表。
    """

    # ---------- 检查输入参数 ----------

    if not np.isfinite(gap_threshold) or gap_threshold <= 0:
        raise ValueError("gap_threshold 必须是有限的正数")

    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"找不到 MAT 文件: {file_path}"
        )

    # ---------- 读取 MAT 文件 ----------

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

    # ---------- 提取数据并统一数据类型 ----------

    arrays = {}

    for col in REQUIRED_COLUMNS:

        try:
            values = np.asarray(
                meas[col],
                dtype=np.float64
            ).reshape(-1)

        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"字段 {col} 无法转换为数值类型"
            ) from exc

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

    if df.empty:
        raise ValueError("MAT 文件中没有有效的数据记录")

    # ---------- 检查无效数值 ----------

    invalid_counts = {}

    for col in REQUIRED_COLUMNS:

        invalid_counts[col] = int(
            (~np.isfinite(df[col].to_numpy())).sum()
        )

    if any(invalid_counts.values()):

        raise ValueError(
            f"数据存在 NaN 或 Inf: {invalid_counts}"
        )

    # ---------- 检查时间是否倒退 ----------

    original_time = df["Time"].to_numpy()

    original_dt = np.diff(original_time)

    time_reversals = int(
        np.sum(original_dt < 0)
    )

    if time_reversals > 0:
        raise ValueError(
            f"Time 存在 {time_reversals} 处倒退，请检查原始数据"
        )

    # ---------- 统计重复时间点 ----------

    # duplicated() 统计的是重复记录中多出来的行数，
    # 不包括每组重复时间点中的第一条记录。
    duplicate_time_records_detected = int(
        df["Time"].duplicated().sum()
    )

    # ---------- 删除完全重复的数据行 ----------

    removed_exact_duplicates = 0

    if drop_exact_duplicates:

        before = len(df)

        df = df.drop_duplicates(
            subset=REQUIRED_COLUMNS,
            keep="first"
        )

        removed_exact_duplicates = before - len(df)

    # ---------- 删除同一时间点的其他记录 ----------

    removed_duplicate_times = 0

    if drop_duplicate_times:

        before = len(df)

        df = df.drop_duplicates(
            subset=["Time"],
            keep="first"
        )

        removed_duplicate_times = before - len(df)

    elif df["Time"].duplicated().any():

        raise ValueError(
            "数据仍存在重复时间点，请启用 drop_duplicate_times"
        )

    # ---------- 重置索引 ----------

    df = df.reset_index(drop=True)

    # ---------- 检查处理后的时间顺序 ----------

    time = df["Time"].to_numpy()

    dt = np.diff(time)

    if np.any(dt <= 0):
        raise ValueError(
            "处理后 Time 仍未严格递增"
        )

    # ---------- 新增放电为正的电流字段 ----------

    # 保留原始 Current，不覆盖原始数据。
    # 原始数据：放电为负，充电为正。
    # 统一数据：放电为正，充电为负。
    df[STANDARD_CURRENT_COLUMN] = -df["Current"]

    # ---------- 检查采样间隔 ----------

    if len(dt) > 0:

        min_dt = float(np.min(dt))
        max_dt = float(np.max(dt))
        mean_dt = float(np.mean(dt))

        large_gaps = int(
            np.sum(dt > gap_threshold)
        )

    else:

        min_dt = np.nan
        max_dt = np.nan
        mean_dt = np.nan
        large_gaps = 0

    # ---------- 统计重复记录处理情况 ----------

    duplicate_time_records_handled = (
        removed_exact_duplicates
        + removed_duplicate_times
    )

    duplicate_time_records_remaining = int(
        df["Time"].duplicated().sum()
    )

    # ---------- 保存数据质量报告 ----------

    quality_report = {

        "file": str(file_path),

        "original_rows": original_rows,

        "final_rows": len(df),

        "removed_exact_duplicates": removed_exact_duplicates,

        "duplicate_time_records_detected":
            duplicate_time_records_detected,

        "removed_duplicate_times": removed_duplicate_times,

        "duplicate_time_records_handled":
            duplicate_time_records_handled,

        "duplicate_time_records_remaining":
            duplicate_time_records_remaining,

        "time_reversals_detected": time_reversals,

        "time_strictly_increasing": bool(
            np.all(np.diff(df["Time"].to_numpy()) > 0)
        ),

        "invalid_values": invalid_counts,

        "min_sampling_interval_s": min_dt,

        "max_sampling_interval_s": max_dt,

        "mean_sampling_interval_s": mean_dt,

        "gap_threshold_s": gap_threshold,

        "large_gaps": large_gaps,

        "current_sign_convention": CURRENT_SIGN_CONVENTION,

        "standard_current_column": STANDARD_CURRENT_COLUMN,

    }

    df.attrs["quality_report"] = quality_report

    return df


# ==================================================
# 3. 分段重采样函数
# ==================================================

def resample_data(
    df,
    target_interval=0.1,
    gap_threshold=0.15,
):
    """
    对数据进行分段线性重采样。

    大于 gap_threshold 的时间间隔视为时间缺口，
    不在缺口两侧进行插值。

    每一段独立重采样，并保留该段的原始起点和终点。

    Parameters
    ----------
    df : pandas.DataFrame
        已完成重复时间点处理的数据。

    target_interval : float
        目标采样间隔，单位为秒。

    gap_threshold : float
        时间缺口判断阈值，单位为秒。

    Returns
    -------
    pandas.DataFrame
        重采样后的数据。
    """

    # ---------- 检查输入参数 ----------

    if not np.isfinite(target_interval) or target_interval <= 0:
        raise ValueError(
            "target_interval 必须是有限的正数"
        )

    if not np.isfinite(gap_threshold) or gap_threshold <= 0:
        raise ValueError(
            "gap_threshold 必须是有限的正数"
        )

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df 必须是 pandas DataFrame")

    if "Time" not in df.columns:
        raise KeyError("数据中缺少 Time 字段")

    if df.empty:
        raise ValueError("不能对空数据进行重采样")

    # ---------- 检查时间数据 ----------

    result = df.copy().reset_index(drop=True)

    time = result["Time"].to_numpy(dtype=float)

    if not np.all(np.isfinite(time)):
        raise ValueError("Time 字段存在 NaN 或 Inf")

    dt = np.diff(time)

    if np.any(dt <= 0):
        raise ValueError(
            "重采样前 Time 必须严格递增"
        )

    # ---------- 检查需要插值的数值字段 ----------

    value_columns = [
        col for col in result.columns
        if col != "Time"
        and pd.api.types.is_numeric_dtype(result[col])
    ]

    for col in value_columns:

        values = result[col].to_numpy(dtype=float)

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"字段 {col} 存在 NaN 或 Inf，不能进行插值"
            )

    # ---------- 找到较大的时间缺口 ----------

    gap_positions = np.where(
        dt > gap_threshold
    )[0]

    # ---------- 根据时间缺口划分数据段 ----------

    segment_starts = [0] + [
        int(pos + 1) for pos in gap_positions
    ]

    segment_ends = [
        int(pos + 1) for pos in gap_positions
    ] + [len(result)]

    resampled_segments = []

    # ---------- 对每一段独立重采样 ----------

    for start, end in zip(segment_starts, segment_ends):

        segment = result.iloc[start:end].copy()

        segment_time = segment["Time"].to_numpy(dtype=float)

        # 只有一个数据点时，不进行插值
        if len(segment) == 1:

            resampled_segments.append(segment)

            continue

        # 生成目标时间序列
        new_time = np.arange(
            segment_time[0],
            segment_time[-1],
            target_interval
        )

        # 保留原始段末端
        if (
            len(new_time) == 0
            or not np.isclose(
                new_time[-1],
                segment_time[-1],
                rtol=1e-9,
                atol=1e-9
            )
        ):

            new_time = np.append(
                new_time,
                segment_time[-1]
            )

        # ---------- 对各数值字段进行线性插值 ----------

        new_segment = {
            "Time": new_time
        }

        for col in value_columns:

            values = segment[col].to_numpy(dtype=float)

            new_segment[col] = np.interp(
                new_time,
                segment_time,
                values
            )

        resampled_segments.append(
            pd.DataFrame(new_segment)
        )

    # ---------- 合并各数据段 ----------

    resampled_df = pd.concat(
        resampled_segments,
        ignore_index=True
    )

    # ---------- 检查重采样结果 ----------

    resampled_time = resampled_df["Time"].to_numpy(dtype=float)

    resampled_dt = np.diff(resampled_time)

    if np.any(resampled_dt <= 0):
        raise ValueError(
            "重采样后 Time 未能保持严格递增"
        )

    # ---------- 保留原始质量报告 ----------

    resampled_df.attrs.update(df.attrs)

    # ---------- 生成重采样报告 ----------

    if len(resampled_dt) > 0:

        resampled_min_dt = float(np.min(resampled_dt))

        resampled_max_dt = float(np.max(resampled_dt))

        resampled_mean_dt = float(np.mean(resampled_dt))

    else:

        resampled_min_dt = np.nan

        resampled_max_dt = np.nan

        resampled_mean_dt = np.nan

    resampled_df.attrs["resampling_report"] = {

        "target_interval_s": target_interval,

        "gap_threshold_s": gap_threshold,

        "original_rows": len(df),

        "resampled_rows": len(resampled_df),

        "number_of_gaps": len(gap_positions),

        "number_of_segments": len(segment_starts),

        "gaps_not_interpolated": True,

        "time_strictly_increasing": bool(
            np.all(resampled_dt > 0)
        ),

        "min_sampling_interval_s": resampled_min_dt,

        "max_sampling_interval_s": resampled_max_dt,

        "mean_sampling_interval_s": resampled_mean_dt,

    }

    return resampled_df


# ==================================================
# 4. 打印数据质量报告
# ==================================================

def print_quality_report(df):
    """
    打印数据读取和清洗阶段的数据质量报告。
    """

    report = df.attrs.get("quality_report")

    if report is None:

        print("没有找到数据质量报告")

        return

    print("\n========== DATA QUALITY REPORT ==========")

    for key, value in report.items():

        print(f"{key}: {value}")

    print("==========================================")


# ==================================================
# 5. 打印重采样报告
# ==================================================

def print_resampling_report(df):
    """
    打印重采样阶段的数据质量报告。
    """

    report = df.attrs.get("resampling_report")

    if report is None:

        print("没有找到重采样报告")

        return

    print("\n========== RESAMPLING REPORT ==========")

    for key, value in report.items():

        print(f"{key}: {value}")

    print("=======================================")
