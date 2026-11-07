
from pathlib import Path

from src.io import (
    load_panasonic_data,
    print_quality_report
)


# =====================================
# 1. 定义项目根目录
# =====================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =====================================
# 2. 指定 Panasonic US06 数据
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
# 3. 调用正式数据接口
# =====================================

df = load_panasonic_data(file_path)


# =====================================
# 4. 打印数据质量报告
# =====================================

print_quality_report(df)


# =====================================
# 5. 检查 DataFrame
# =====================================

print("\n========== DATAFRAME ==========")

print(df.head())

print("\n数据维度:", df.shape)

print("\n字段名称:")

print(df.columns.tolist())


# =====================================
# 6. 验证数据是否符合要求
# =====================================

assert len(df) == 48060

assert df.shape[1] == 6

assert df["Time"].is_monotonic_increasing

assert df.isna().sum().sum() == 0

assert df.duplicated().sum() == 0

print("\n所有测试通过！")
