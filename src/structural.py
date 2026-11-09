
import csv

import numpy as np
import matplotlib.pyplot as plt

def save_matrix(matrix, row_names, col_names, filepath):

    matrix = np.asarray(matrix)

    with open(filepath, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([""] + list(col_names))

        for name, row in zip(row_names, matrix):
            writer.writerow([name] + list(row))


# ============================================================
# 4. Print isolability matrix
# ============================================================

def print_isolability(matrix, fault_names, title):

    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)

    print("Rows and columns follow model fault order.")

    print(" " * 12 + " ".join(
        f"{name:>9}" for name in fault_names
    ))

    for name, row in zip(fault_names, matrix):

        print(
            f"{name:>10}  "
            + " ".join(f"{int(value):>9}" for value in row)
        )


# ============================================================
# 5. Plot and save isolability analysis
# ============================================================

def save_isolability_plot(model, causality, fault_names,
                          title, filepath):

    fig, ax = plt.subplots(figsize=(8, 7))

    matrix = model.IsolabilityAnalysis(
        ax=ax,
        causality=causality
    )

    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print_isolability(
        matrix,
        fault_names,
        title
    )

    return np.asarray(matrix)


# ============================================================
# 6. Structural analysis for one model
# ============================================================

def run_structural_analysis(model, model_label, output_dir):

    print("\n")
    print("#" * 75)
    print(f"STRUCTURAL ANALYSIS: {model_label}")
    print("#" * 75)

    output_dir.mkdir(parents=True, exist_ok=True)

    fault_names = list(model.f)

    # --------------------------------------------------------
    # 6.1 Model validation
    # --------------------------------------------------------

    print("\n[1] MODEL INFORMATION AND VALIDATION")

    print("Unknown variables:", model.x)
    print("Fault variables:", model.f)
    print("Known variables:", model.z)

    model.Lint()

    # --------------------------------------------------------
    # 6.2 Detectability analysis
    # --------------------------------------------------------

    print("\n[2] DETECTABILITY ANALYSIS")

    detectable, not_detectable = model.DetectabilityAnalysis(
        causality="mixed"
    )

    print("Detectable faults:")
    print(detectable)

    print("Non-detectable faults:")
    print(not_detectable)

    with open(
        output_dir / "detectability.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write("Detectable faults:\n")
        file.write(str(detectable))

        file.write("\n\nNon-detectable faults:\n")
        file.write(str(not_detectable))

    # --------------------------------------------------------
    # 6.3 Integral causality isolability
    # --------------------------------------------------------

    print("\n[3] INTEGRAL CAUSALITY ISOLABILITY")

    integral_matrix = save_isolability_plot(
        model,
        causality="int",
        fault_names=fault_names,
        title=f"{model_label}: Integral Causality",
        filepath=output_dir / "isolability_integral.png"
    )

    save_matrix(
        integral_matrix,
        fault_names,
        fault_names,
        output_dir / "isolability_integral.csv"
    )

    # --------------------------------------------------------
    # 6.4 Mixed causality isolability
    # --------------------------------------------------------

    print("\n[4] MIXED CAUSALITY ISOLABILITY")

    mixed_matrix = save_isolability_plot(
        model,
        causality="mixed",
        fault_names=fault_names,
        title=f"{model_label}: Mixed Causality",
        filepath=output_dir / "isolability_mixed.png"
    )

    save_matrix(
        mixed_matrix,
        fault_names,
        fault_names,
        output_dir / "isolability_mixed.csv"
    )

    # --------------------------------------------------------
    # 6.5 Dulmage-Mendelsohn decomposition
    # --------------------------------------------------------

    print("\n[5] DULMAGE-MENDELSOHN DECOMPOSITION")

    fig, ax = plt.subplots(figsize=(12, 9))

    # ---------- DM decomposition plot ----------
    print("\n正在生成 DM 分解图...")

    try:
        fig, ax = plt.subplots(figsize=(12, 8))

        model.PlotDM(
            ax=ax,
            fault=True,
            eqclass=False,
            verbose=True
        )

        fig.tight_layout()
        fig.savefig(
            output_dir / "DM_decomposition.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(fig)

        print("DM 分解图保存成功！")

    except Exception as e:
        plt.close("all")

        print("\nDM 分解图生成失败，但继续执行其他分析。")
        print("错误信息：", repr(e))

        with open(output_dir / "DM_plot_error.txt", "w") as file:
            file.write(repr(e))

    ax.set_title(f"{model_label}: DM Decomposition")

    fig.tight_layout()

    fig.savefig(
        output_dir / "DM_decomposition.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("DM decomposition saved.")

    # --------------------------------------------------------
    # 6.6 MSO enumeration
    # --------------------------------------------------------

    print("\n[6] MSO ANALYSIS")

    msos = model.MSO()

    print("Total number of MSO sets:", len(msos))

    with open(
        output_dir / "MSO_list.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(f"Total MSO sets: {len(msos)}\n\n")

        for index, mso in enumerate(msos):

            file.write(
                f"MSO {index + 1}: {np.asarray(mso).tolist()}\n"
            )

    for index, mso in enumerate(msos):

        print(f"MSO {index + 1}: {np.asarray(mso).tolist()}")

        # Print detailed equations for each MSO
        print("\n========== MSO EQUATION DETAILS ==========")

        for index, mso in enumerate(msos):

            print(f"\nMSO {index + 1}:")

            for eq_index in mso:
                print(
                    f"  Equation {int(eq_index)}: "
                    f"{model.syme[int(eq_index)]}"
                )

        print("==========================================")


    # --------------------------------------------------------
    # 6.7 Low-index MSO analysis
    # --------------------------------------------------------

    print("\n[7] LOW-INDEX MSO ANALYSIS")

    low_index_msos = []

    for mso in msos:

        if model.IsLowIndex(mso):
            low_index_msos.append(mso)

    print("Total MSO sets:", len(msos))

    print("Low-index MSO sets:", len(low_index_msos))

    print("High-index MSO sets:", len(msos) - len(low_index_msos))

    with open(
        output_dir / "low_index_MSO.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"Total MSO sets: {len(msos)}\n"
        )

        file.write(
            f"Low-index MSO sets: {len(low_index_msos)}\n\n"
        )

        for index, mso in enumerate(low_index_msos):

            file.write(
                f"Low-index MSO {index + 1}: "
                f"{np.asarray(mso).tolist()}\n"
            )

    # --------------------------------------------------------
    # 6.8 Test selection
    # --------------------------------------------------------

    print("\n[8] TEST SELECTION")

    if len(low_index_msos) > 0:

        selected_indices = model.TestSelection(
            low_index_msos
        )

        selected_msos = [
            low_index_msos[int(index)]
            for index in selected_indices
        ]

        print("Selected MSO indices:")
        print(selected_indices)

        print("Number of selected tests:", len(selected_msos))

        with open(
            output_dir / "selected_tests.txt",
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                f"Number of selected tests: {len(selected_msos)}\n\n"
            )

            for index, mso in enumerate(selected_msos):

                file.write(
                    f"Test {index + 1}: "
                    f"{np.asarray(mso).tolist()}\n"
                )

        # ----------------------------------------------------
        # 6.9 Fault Signature Matrix
        # ----------------------------------------------------

        print("\n[9] FAULT SIGNATURE MATRIX (FSM)")

        fsm = model.FSM(selected_msos)

        print("FSM shape:", fsm.shape)

        print("\nFSM:")
        print(fsm)

        np.savetxt(
            output_dir / "FSM.csv",
            fsm,
            delimiter=",",
            fmt="%d"
        )

        # Save selected test isolability
        selected_isolability = model.IsolabilityAnalysisFSM(
            fsm
        )

        print("\nIsolability from selected tests:")
        print(selected_isolability)

        save_matrix(
            selected_isolability,
            fault_names,
            fault_names,
            output_dir / "selected_test_isolability.csv"
        )

        # Save FSM with fault labels
        with open(
            output_dir / "FSM_labeled.csv",
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                ["Test"] + fault_names
            )

            for index, row in enumerate(fsm):

                writer.writerow(
                    [f"Test_{index + 1}"] + list(row)
                )

    else:

        print("No low-index MSOs available. Test selection skipped.")

    # --------------------------------------------------------
    # 6.10 Summary
    # --------------------------------------------------------

    print("\n[10] SUMMARY")

    print("Model:", model_label)
    print("Number of faults:", len(fault_names))
    print("Number of MSOs:", len(msos))
    print("Number of low-index MSOs:", len(low_index_msos))

    print("\nResults saved in:")
    print(output_dir.resolve())

    return {
        "model": model_label,
        "fault_names": fault_names,
        "detectable": detectable,
        "not_detectable": not_detectable,
        "integral_matrix": integral_matrix,
        "mixed_matrix": mixed_matrix,
        "msos": msos,
        "low_index_msos": low_index_msos,
    }