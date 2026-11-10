
from pathlib import Path

import numpy as np
from scipy.io import loadmat


# ==================================================
# 1. 定位项目和 MAT 文件
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Panasonic_18650PF"
    / "Panasonic 18650PF Data"
    / "25degC"
    / "C20 OCV and 1C discharge tests_start_of_tests"
)

MAT_FILES = [
    "05-08-17_13.26 C20 OCV Test_C20_25dC.mat",
]



# ==================================================
# 2. 检查单个 MAT 文件
# ==================================================

def inspect_mat_file(file_path):

    print("\n" + "=" * 65)
    print("文件名称:", file_path.name)
    print("=" * 65)

    if not file_path.is_file():
        print("错误：文件不存在！")
        return

    # 读取 MAT 文件
    mat_data = loadmat(
        file_path,
        simplify_cells=True
    )

    # ---------- 检查顶层变量 ----------

    root_keys = [
        key for key in mat_data.keys()
        if not key.startswith("__")
    ]

    print("\n【一、MAT 文件顶层变量】")

    for key in root_keys:

        value = mat_data[key]

        print(
            f"{key}: "
            f"type={type(value).__name__}"
        )

    # ---------- 检查 meas ----------

    if "meas" not in mat_data:

        print("\n没有找到 meas 变量。")
        print("需要进一步检查顶层变量结构。")
        return

    meas = mat_data["meas"]

    print("\n【二、meas 结构】")
    print("meas 类型:", type(meas).__name__)

    if isinstance(meas, dict):

        print("\n【三、meas 内部字段】")

        for key, value in meas.items():

            print(f"\n字段名称: {key}")
            print("数据类型:", type(value).__name__)

            try:

                array = np.asarray(value)

                print("数据形状:", array.shape)
                print("数据长度:", array.size)

                if array.size > 0 and array.dtype.kind in "biufc":

                    print(
                        "前五个数据:",
                        array.reshape(-1)[:5]
                    )

            except Exception as exc:

                print("无法转换为数组:", exc)

    else:

        print(
            "\nmeas 不是字典结构，"
            "需要进一步分析其内部结构。"
        )


# ==================================================
# 3. 依次检查两份文件
# ==================================================

if __name__ == "__main__":

    for filename in MAT_FILES:

        file_path = DATA_DIR / filename

        inspect_mat_file(file_path)

    print("\n所有文件检查结束。")
