
from pathlib import Path
from scipy.io import whosmat

# 获取项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# LG 18650HG2 25°C US06 数据文件
file_path = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "LG_18650HG2"
    / "LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020"
    / "25degC"
    / "10-29-18_10.08 551_US06_25degC_LGHG2.mat"
)

print("=" * 60)
print("LG 18650HG2 - US06 MAT 文件结构检查")
print("=" * 60)

# 先检查文件是否存在
print(f"项目根目录：{PROJECT_ROOT}")
print(f"目标文件：{file_path}")

if not file_path.exists():
    print("\n错误：找不到文件！")
    print("请检查文件夹名称和文件名是否一致。")
else:
    print("\n文件存在，开始读取！\n")

    variables = whosmat(str(file_path))

    for name, shape, dtype in variables:
        print(f"变量名称：{name}")
        print(f"数据维度：{shape}")
        print(f"数据类型：{dtype}")
        print("-" * 40)
