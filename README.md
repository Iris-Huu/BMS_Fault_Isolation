# BMS Fault Isolation

This project studies how to distinguish **current sensor bias** from an **internal short circuit** in a lithium-ion battery using voltage, current, and temperature measurements.

The main dataset is Panasonic 18650PF. The battery model combines a second-order RC circuit and a thermal model.

## Current progress

Week 2 calibration and validation results are available for **25°C, 10°C, and 0°C**:

- OCV–SOC voltage curves and RC circuit parameters.
- Thermal parameters and measurement-noise estimates.
- Voltage and temperature predictions for US06 and UDDS driving cycles.

The model is ready for further study, with remaining limitations listed below.

## Project folders

| Folder | Contents |
| --- | --- |
| `src/` | Reusable code for data loading, battery simulation, and calibration |
| `experiments/` | Scripts that run calibration, validation, and plotting |
| `tests/` | Checks for data loading and model calculations |
| `tools/` | Data inspection and report tools |
| `results/` | Parameter tables, error tables, and figures |
| `archive/` | Earlier exploration scripts |
| `data/` | Local datasets, excluded from Git |

## Run the project

Use PowerShell in the project root. These commands use Python 3.12.

**1. Install the environment**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**2. Prepare the data**

Place the original Panasonic dataset in `data/raw/Panasonic_18650PF/`. Keep its original filenames and folder structure. Experimental data is not included in this repository.

**3. Run calibration and validation**

```powershell
# Fit the separate 25°C C/20 voltage curve
.\.venv\Scripts\python.exe -B experiments\calibrate_ocv_25c.py --project-root .

# Run the three-temperature calibration and validation workflow
.\.venv\Scripts\python.exe -B experiments\run_week2.py --project-root .
```

Generated files are saved in `results/week2/`. Running these commands again updates existing results.

**4. Check the results**

```powershell
.\.venv\Scripts\python.exe -B tests\verify_week2.py --project-root . --results-dir results\week2
```

This check needs the local raw data and generated noise files. Run the workflow first after setting up a fresh copy of the repository.

## Where to find the results

| What to read | File or folder |
| --- | --- |
| Current model parameters | [Refined parameter table](results/week2/summary/refined_parameter_table.csv) |
| US06 and UDDS prediction errors | [Model error table](results/week2/comparisons/calibration_refinement/model_error_table.csv) |
| Current model files and validation figures | [Refined model results](results/week2/comparisons/calibration_refinement/) |
| Current model selection | [Model selection](results/week2/summary/model_selection.json) |
| Detailed methods and limitations, in Chinese | [Week 2 research notes](results/week2/summary/第二周校核与改进说明.md) |

The current model uses the full parameter files in `comparisons/calibration_refinement/`. Files directly under `parameters/` and `summary/model_error_table.csv` describe the earlier baseline.

## Remaining limitations

- Independent Panasonic C/20 data was not found locally for 10°C and 0°C. Their voltage curves use rested HPPC measurements as an approximation of OCV.
- Missing low-temperature chamber measurements are replaced by explicitly recorded temperature setpoints.
- Some circuit parameters remain uncertain. Thermal parameters also depend on the assumed heat and ambient-temperature models.
- Temperature prediction still has systematic errors in some cycles. Noise estimates include assumptions, especially for current measurements.

These results support the next stage of research, but do not yet establish fully validated fault-isolation limits. The recorded US06 and UDDS results are development validation, rather than untouched final tests.

## Data conventions

Discharge current is positive in calculations. The original current column is preserved, and the model uses `Current_discharge_positive`. Time is measured in seconds.

Raw data, processed measurement records, virtual environments, caches, credentials, and local backups are excluded from Git.
