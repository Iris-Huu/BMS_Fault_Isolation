import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
import faultdiagnosistoolbox as fdt

#create symbolic model
def build_battery_model(with_thermal=True):
#定义模型变量
    x_names = [
        "I_cell", "i", "i_s", "v", "SOC",
        "dSOC", "v1", "dv1", "v2", "dv2"
    ]
    if with_thermal:
        x_names.extend(["T", "dT"])

#定义故障变量
    f_names = [
        "f_i", "f_v", "f_isc", "f_Q", "f_R"
    ]
    if with_thermal:
        f_names.append("f_T")

#定义已知变量
    z_names = [
        "y_i", "y_v"
    ]
    if with_thermal:
        z_names.extend(["T_amb", "y_T"])

#定义模型参数
    parameter_names = [
        "R0", "R1", "C1",
        "R2", "C2", "Q"
    ]

    if with_thermal:
        parameter_names.extend(["Cth", "hA"])

#创建数学符号
    x_symbols = sym.symbols(" ".join(x_names))
    f_symbols = sym.symbols(" ".join(f_names))
    z_symbols = sym.symbols(" ".join(z_names))
    parameter_symbols = sym.symbols(" ".join(parameter_names))

# 建立名称与数学符号之间的对应关系
    x = dict(zip(x_names, x_symbols))
    f = dict(zip(f_names, f_symbols))
    z = dict(zip(z_names, z_symbols))
    p = dict(zip(parameter_names, parameter_symbols))

#定义OCV函数
    OCV = sym.Function("OCV")

#建立方程集合
    rels = []
    rels.append(x["I_cell"] - x["i"] - x["i_s"])
    rels.append(x["i_s"] - f["f_isc"] * x["v"])
    rels.append(x["dSOC"] +x["I_cell"] / (3600 * p["Q"] * (1 - f["f_Q"])))
    rels.append(x["dv1"] + x["v1"] / (p["R1"] * p["C1"]) - x["I_cell"] / p["C1"])
    rels.append( x["dv2"] + x["v2"] / (p["R2"] * p["C2"]) - x["I_cell"] / p["C2"])
    rels.append(x["v"] - OCV(x["SOC"]) + p["R0"] * (1 + f["f_R"]) * x["I_cell"] + x["v1"] + x["v2"])
    rels.append(z["y_i"] - x["i"] - f["f_i"])
    rels.append(z["y_v"] - x["v"] - f["f_v"])

#添加微分约束
    rels.extend([
        fdt.DiffConstraint("dSOC", "SOC"),
        fdt.DiffConstraint("dv1", "v1"),
        fdt.DiffConstraint("dv2", "v2"),
    ])

#热方程
    if with_thermal:
        rels.append(
            p["Cth"] * x["dT"]
            - x["I_cell"] * (OCV(x["SOC"]) - x["v"])
            - x["i_s"] * x["v"]
            + p["hA"] * (x["T"] - z["T_amb"])
        )
# 温度传感器方程
        rels.append(
            z["y_T"] - x["T"] - f["f_T"]
        )

# 温度微分约束
        rels.append(
            fdt.DiffConstraint("dT", "T")
        )

#from model to Fault Diagnosis Toolbox

    model_def = {
        "type": "Symbolic",
        "x": x_names,
        "f": f_names,
        "z": z_names,
        "parameters": parameter_names,
        "rels": rels,
    }

    model = fdt.DiagnosisModel(model_def) #交给诊断，创建一个真正的诊断模型对象

#检查模型
    model.Lint()

    return model


#if __name__ == "__main__":
    model = build_battery_model(with_thermal=True)

    #print("\n模型创建成功！")

if __name__ == "__main__":

    print("\n========== FULL THERMAL MODEL ==========")
    model_thermal = build_battery_model(with_thermal=True)

    print("\n========== NON-THERMAL MODEL ==========")
    model_no_thermal = build_battery_model(with_thermal=False)

    print("\n两个模型创建完成！")





