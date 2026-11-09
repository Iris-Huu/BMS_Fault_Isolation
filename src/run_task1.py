
"""
Task 1: Structural analysis of battery fault isolability

Includes:
1. Thermal and non-thermal symbolic battery models
2. Model validation
3. Detectability analysis
4. Isolability analysis: integral and mixed causality
5. Dulmage-Mendelsohn decomposition
6. MSO enumeration
7. Low-index MSO analysis
8. Test selection
9. Fault Signature Matrix (FSM)

Run:
    python run_task1.py
"""

from pathlib import Path
import csv
import traceback

import numpy as np
import matplotlib.pyplot as plt

from src.model import build_battery_model
from src.structural import run_structural_analysis


# ============================================================
# 1. Project paths
# ============================================================

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "task1"

RESULTS.mkdir(parents=True, exist_ok=True)

def main():

    print("\n")
    print("=" * 75)
    print("BATTERY BMS FAULT ISOLABILITY - TASK 1")
    print("=" * 75)

    all_results = {}

    # --------------------------------------------------------
    # 7.1 Full thermal model
    # --------------------------------------------------------

    print("\nBUILDING FULL THERMAL MODEL...")

    thermal_model = build_battery_model(
        with_thermal=True
    )

    try:

        all_results["thermal"] = run_structural_analysis(
            thermal_model,
            "thermal_model",
            RESULTS / "thermal_model"
        )

    except Exception as error:

        print("\nERROR IN THERMAL MODEL ANALYSIS:")
        print(error)
        traceback.print_exc()

    # --------------------------------------------------------
    # 7.2 Non-thermal model
    # --------------------------------------------------------

    print("\nBUILDING NON-THERMAL MODEL...")

    nonthermal_model = build_battery_model(
        with_thermal=False
    )

    try:

        all_results["nonthermal"] = run_structural_analysis(
            nonthermal_model,
            "nonthermal_model",
            RESULTS / "nonthermal_model"
        )

    except Exception as error:

        print("\nERROR IN NON-THERMAL MODEL ANALYSIS:")
        print(error)
        traceback.print_exc()

    # --------------------------------------------------------
    # 7.3 Final report
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("TASK 1 FINISHED")
    print("=" * 75)

    print("Successfully completed models:")

    for name in all_results:
        print("-", name)

    print("\nAll results directory:")
    print(RESULTS.resolve())


if __name__ == "__main__":
    main()