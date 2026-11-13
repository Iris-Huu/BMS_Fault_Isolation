from pathlib import Path
from scipy.io import loadmat
import numpy as np


# =====================================
# 1. 获取项目根目录
# =====================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =====================================
# 2. 设置 MAT 文件路径
# =====================================

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


# =====================================
# 3. 检查文件是否存在
# =====================================

print("检查文件路径...")

if not file_path.exists():
    raise FileNotFoundError(
        f"找不到文件: {file_path}"
    )

print("文件存在！")
print("文件路径:", file_path)


# =====================================
# 4. 读取 MATLAB 文件
# =====================================

print("\n正在读取 MAT 文件...")

data = loadmat(
    file_path,
    simplify_cells=True
)

print("读取成功！")


# =====================================
# 5. 查看内部变量名称
# =====================================

print("\nMAT 文件包含的变量：")

for key in data.keys():

    if key.startswith("__"):
        continue

    print("-", key)


# =====================================
# 6. 递归查看数据结构
# =====================================

def show_structure(name, value, level=0):

    indent = "    " * level

    print(f"\n{indent}变量名称: {name}")
    print(f"{indent}数据类型: {type(value)}")

    if isinstance(value, dict):

        print(f"{indent}包含的子变量:")

        for key, item in value.items():

            show_structure(
                key,
                item,
                level + 1
            )

    elif isinstance(value, np.ndarray):

        print(f"{indent}数据形状: {value.shape}")
        print(f"{indent}数据类型: {value.dtype}")

        if value.size <= 10:

            print(f"{indent}数据内容: {value}")

    else:

        print(f"{indent}数据内容: {repr(value)[:200]}")


print("\n========== 详细数据结构 ==========")

for key, value in data.items():

    if key.startswith("__"):
        continue

    show_structure(key, value)





# =====================================
# 7. 检查六个目标字段的实际数据
# =====================================

meas = data["meas"]

fields = [
    "Time",
    "Voltage",
    "Current",
    "Ah",
    "Battery_Temp_degC",
    "Chamber_Temp_degC"
]

print("\n========== 六个字段的数据检查 ==========")

for field in fields:

    values = np.asarray(meas[field]).reshape(-1)

    print(f"\n字段名称: {field}")

    print("数据类型:", values.dtype)

    print("数据数量:", len(values))

    print("前五个数据:", values[:5])

    print("后五个数据:", values[-5:])

    # 数值统计
    numeric_values = values.astype(float)

    finite_values = numeric_values[
        np.isfinite(numeric_values)
    ]

    print("缺失/无效数值数量:",
          len(values) - len(finite_values))

    if len(finite_values) > 0:

        print("最小值:", np.min(finite_values))

        print("最大值:", np.max(finite_values))

        print("平均值:", np.mean(finite_values))


# =====================================
# 8. 检查时间采样
# =====================================

time = np.asarray(meas["Time"], dtype=float).reshape(-1)

dt = np.diff(time)

print("\n========== 时间采样检查 ==========")

print("起始时间:", time[0])

print("结束时间:", time[-1])

print("总时间:", time[-1] - time[0])

print("最小采样间隔:", np.min(dt))

print("最大采样间隔:", np.max(dt))

print("平均采样间隔:", np.mean(dt))

print("重复时间点数量:", np.sum(dt == 0))

print("时间倒退数量:", np.sum(dt < 0))


# =====================================
# 9. 检查 TimeStamp
# =====================================

timestamp = np.asarray(meas["TimeStamp"]).reshape(-1)

print("\n========== TimeStamp 检查 ==========")

print("前五个时间戳:", timestamp[:5])

print("数据类型:", timestamp.dtype)

print("\n========== 检查完成 ==========")


# =====================================
# 10. 定位重复时间点和异常采样间隔
# =====================================

import pandas as pd

# 获取六个目标字段
meas = data["meas"]

df = pd.DataFrame({
    field: np.asarray(meas[field]).reshape(-1)
    for field in fields
})

time = df["Time"].to_numpy()

dt = np.diff(time)


# ---------- 检查重复时间点 ----------

print("\n========== 重复时间点详细检查 ==========")

duplicate_indices = np.where(dt == 0)[0]

for idx in duplicate_indices:

    print(f"\n重复时间位置: {idx}")

    print("前一条记录：")
    print(df.iloc[idx].to_string())

    print("\n后一条记录：")
    print(df.iloc[idx + 1].to_string())


# ---------- 检查最大采样间隔 ----------

print("\n========== 最大采样间隔详细检查 ==========")

max_idx = np.argmax(dt)

print("最大间隔位置:", max_idx)

print("间隔前的时间:", time[max_idx])

print("间隔后的时间:", time[max_idx + 1])

print("最大采样间隔:", dt[max_idx])

print("\n异常间隔附近的实验数据：")

start = max(0, max_idx - 3)

end = min(len(df), max_idx + 5)

print(
    df.iloc[start:end].to_string(index=True)
)


# ---------- 检查较大的采样间隔 ----------

print("\n========== 较大采样间隔统计 ==========")

print("间隔 > 0.15 s 的数量:", np.sum(dt > 0.15))

print("间隔 > 0.5 s 的数量:", np.sum(dt > 0.5))

print("间隔 > 1.0 s 的数量:", np.sum(dt > 1.0))


# 检查重复时间点的全部原始字段

print("\n========== 全部原始字段重复检查 ==========")

all_fields = [
    key for key in meas.keys()
]

for idx in duplicate_indices:

    for field in all_fields:

        values = np.asarray(meas[field]).reshape(-1)

        same = values[idx] == values[idx + 1]

        print(
            f"{field}: "
            f"{'相同' if same else '不同'}"
        )

# =====================================
# 11. 检查全部较大采样间隔
# =====================================

print("\n========== 全部较大采样间隔 ==========")

large_gap_indices = np.where(dt > 0.15)[0]

for idx in large_gap_indices:

    print("\n--------------------------------")

    print("位置:", idx)

    print("开始时间:", time[idx])

    print("结束时间:", time[idx + 1])

    print("采样间隔:", dt[idx])

    print("间隔前电压:", df["Voltage"].iloc[idx])

    print("间隔后电压:", df["Voltage"].iloc[idx + 1])

    print("间隔前电流:", df["Current"].iloc[idx])

    print("间隔后电流:", df["Current"].iloc[idx + 1])

    print("间隔前电池温度:",
          df["Battery_Temp_degC"].iloc[idx])

    print("间隔后电池温度:",
          df["Battery_Temp_degC"].iloc[idx + 1])