import sympy as sym
import faultdiagnosistoolbox as fdt


def build_battery_model(with_thermal=True):

    # Unknown variables
    x_names = [
        "I_cell", "i", "i_s", "v", "SOC",
        "dSOC", "v1", "dv1", "v2", "dv2"
    ]

    if with_thermal:
        x_names.extend(["T", "dT"])

    # Fault variables
    f_names = [
        "f_i", "f_v", "f_isc", "f_Q", "f_R"
    ]

    if with_thermal:
        f_names.append("f_T")

    # Known variables
    z_names = ["y_i", "y_v"]

    if with_thermal:
        z_names.extend(["T_amb", "y_T"])

    # Parameters
    parameter_names = [
        "R0", "R1", "C1",
        "R2", "C2", "Q"
    ]

    if with_thermal:
        parameter_names.extend(["Cth", "hA"])

    # Create SymPy symbols
    x_symbols = sym.symbols(" ".join(x_names))
    f_symbols = sym.symbols(" ".join(f_names))
    z_symbols = sym.symbols(" ".join(z_names))
    p_symbols = sym.symbols(" ".join(parameter_names))

    x = dict(zip(x_names, x_symbols))
    f = dict(zip(f_names, f_symbols))
    z = dict(zip(z_names, z_symbols))
    p = dict(zip(parameter_names, p_symbols))

    # OCV function
    OCV = sym.Function("OCV")

    # Equations
    rels = []

    # (1) Current balance
    rels.append(
        x["I_cell"] - x["i"] - x["i_s"]
    )

    # (2) Internal short circuit current
    rels.append(
        x["i_s"] - f["f_isc"] * x["v"]
    )

    # (3) SOC dynamics
    rels.append(
        x["dSOC"]
        + x["I_cell"] /
        (3600 * p["Q"] * (1 - f["f_Q"]))
    )

    # (4) First RC branch
    rels.append(
        x["dv1"]
        + x["v1"] / (p["R1"] * p["C1"])
        - x["I_cell"] / p["C1"]
    )

    # (5) Second RC branch
    rels.append(
        x["dv2"]
        + x["v2"] / (p["R2"] * p["C2"])
        - x["I_cell"] / p["C2"]
    )

    # (6) Terminal voltage
    rels.append(
        x["v"]
        - OCV(x["SOC"])
        + p["R0"] * (1 + f["f_R"]) * x["I_cell"]
        + x["v1"]
        + x["v2"]
    )

    # Current sensor
    rels.append(
        z["y_i"] - x["i"] - f["f_i"]
    )

    # Voltage sensor
    rels.append(
        z["y_v"] - x["v"] - f["f_v"]
    )

    # Differential constraints
    rels.extend([
        fdt.DiffConstraint("dSOC", "SOC"),
        fdt.DiffConstraint("dv1", "v1"),
        fdt.DiffConstraint("dv2", "v2"),
    ])

    # Thermal model
    if with_thermal:

        # Single-node thermal balance
        rels.append(
            p["Cth"] * x["dT"]
            - x["I_cell"] * (OCV(x["SOC"]) - x["v"])
            - x["i_s"] * x["v"]
            + p["hA"] * (x["T"] - z["T_amb"])
        )

        # Temperature sensor
        rels.append(
            z["y_T"] - x["T"] - f["f_T"]
        )

        # Temperature differential constraint
        rels.append(
            fdt.DiffConstraint("dT", "T")
        )

    # ==========================================
    # CHECK: Equations containing internal short current i_s
    # ==========================================

    print("\n========== EQUATIONS CONTAINING i_s ==========")

    for k, rel in enumerate(rels):
        if isinstance(rel, sym.Expr) and x["i_s"] in rel.free_symbols:
            print(f"Equation {k}: {rel}")

    print("==============================================\n")

    print("\n========== FAULT OCCURRENCE CHECK ==========")

    for k, rel in enumerate(rels):
        if isinstance(rel, sym.Expr):
            fault_list = [
                name for name, symbol in f.items()
                if symbol in rel.free_symbols
            ]

            if fault_list:
                print(f"Equation {k}: {rel}")
                print(f"    Faults: {fault_list}")

    print("============================================\n")

    # Model definition
    model_def = {
        "type": "Symbolic",
        "x": x_names,
        "f": f_names,
        "z": z_names,
        "parameters": parameter_names,
        "rels": rels,
    }

    model_name = (
        "Battery model with thermal dynamics"
        if with_thermal
        else "Battery model without thermal dynamics"
    )

    model = fdt.DiagnosisModel(
        model_def,
        name=model_name
    )

    return model