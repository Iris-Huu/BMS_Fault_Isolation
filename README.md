# BMS Fault Isolation: Current Sensor Bias and Internal Short Circuit

This project studies whether current sensor bias and an internal short circuit can be distinguished in a lithium-ion cell using electrical and thermal measurements. It combines a second-order RC equivalent circuit, a lumped thermal model, parameter calibration, measurement-noise analysis, and operating-cycle validation.

## Current research status

**Week 2 computational results are available; scientific acceptance remains conditional.** The current numerical model can support algorithm development and sensitivity studies. Its outputs do not yet establish unconditional experimentally validated weak-fault detection or isolation limits.

The main limitations are:

- Independent Panasonic 10 and 0 degC C/20 measurements were not located in the local dataset. Rested HPPC voltage is used as an explicitly labeled OCV proxy.
- Chamber temperature is missing in the selected low-temperature calibration and validation files. An explicit chamber-setpoint fallback is recorded in the data provenance.
- Some HPPC parameters approach optimization bounds, especially at 0 degC. Positive parameters and low fitting error do not establish unique identifiability.
- Thermal coefficients depend on the assumed heat model and an effective ambient-temperature offset. Their statistical confidence intervals have not been established.
- Temperature residuals retain systematic bias in some operating conditions, and nominal prediction-envelope coverage is not achieved in every case.
- Current-sensor-only noise cannot be separated from controller ripple using the available logged measurements.

## Start with these files

| Purpose | File |
| --- | --- |
| Identify the selected numerical model and its inputs | [Model selection](results/week2/summary/model_selection.json) |
| Read current conditional completion status | [Completion status](results/week2/summary/week2_completion_status.json) |
| Inspect selected model parameters at representative SOC values | [Refined parameter table](results/week2/summary/refined_parameter_table.csv) |
| Inspect current US06 and UDDS prediction errors | [Refined model error table](results/week2/comparisons/calibration_refinement/model_error_table.csv) |
| Compare the initial and refined models | [Refinement comparison](results/week2/summary/refinement_comparison.csv) |
| Inspect additional fixed-parameter Cycle 4 evaluations | [Confirmation error table](results/week2/comparisons/calibration_refinement/confirmation_error_table.csv) |
| Review OCV evidence and its interpretation | [OCV evidence](results/week2/audit/ocv_evidence.csv) |
| Review empirical noise holdout coverage | [Noise coverage](results/week2/summary/noise_holdout_coverage.csv) |
| Review raw source provenance | [Source manifest](results/week2/summary/data_manifest.json) |

Detailed existing research notes are also available in Chinese: [baseline methods and results](results/week2/summary/第二周工作说明.md), [refinement and limitations](results/week2/summary/第二周校核与改进说明.md), and [raw-data availability recheck](results/week2/summary/原始数据缺失复查.md).

**Version distinction:** `summary/parameter_table.csv` and `summary/model_error_table.csv` describe the initial fixed-parameter baseline. The selected refined model is under `comparisons/calibration_refinement/`. Do not combine baseline parameters with refined-model error claims.

## Repository layout

```text
BMS_Fault_Isolation/
|-- src/                         Reusable data, model, and calibration modules
|-- experiments/                 Calibration and plotting entry points
|   `-- comparisons/             Alternative model comparison scripts
|-- tests/                       Data-reading and numerical correctness checks
|-- tools/                       Data audits, report generation, and inspection
|   `-- inspection/              Exploratory MAT-file and source-data inspection
|-- results/
|   |-- week1/figures/           Initial US06 figures
|   `-- week2/
|       |-- audit/               Source dates, capacity coordinates, and OCV evidence
|       |-- calibration/         Pulse fits, quality flags, and aggregation sources
|       |-- parameters/          Initial baseline parameters and HPPC OCV tables
|       |-- noise/               Covariances, empirical summaries, and assumptions
|       |-- validation/          Initial baseline prediction figures
|       |-- comparisons/         Alternative and refined model results
|       |-- ocv_25c/             Independent 25 degC C/20 proxy fit
|       `-- summary/             Navigation, error tables, provenance, and limitations
|-- archive/                     Historical exploration scripts and local archives
|-- data/raw/                    Local experimental files; excluded from Git
|-- data/processed/              Local processed data; excluded from Git
|-- requirements.txt            Pinned Python dependencies
`-- README.md
```

The repository root README contains directory guidance. A separate `docs/` directory is not used.

### Core modules

| Module | Responsibility |
| --- | --- |
| `src/io.py` | MAT-file loading, data-quality checks, discharge-positive current, duplicate-time handling, explicit ambient fallback, and segmented resampling |
| `src/ocv.py` | Charge/discharge segmentation, relative SOC, monotone OCV-proxy fitting, and interpolation metrics |
| `src/calibration.py` | Joint pulse/rest 2RC fitting, dynamic parameter refinement, and thermal coefficient fitting |
| `src/model.py` | Electrical/thermal simulation, fault inputs, and the selected voltage observation model |
| `src/noise.py` | Rest-window statistics, quantization assumptions, colored covariance, and empirical residual-block sampling |

## Environment setup

The commands below use PowerShell from the repository root. The existing local environment uses Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Activation is optional because the examples invoke the environment's Python directly. The virtual environment is local and must not be committed.

## Local experimental data

The main calibration dataset is Panasonic 18650PF. LG 18650HG2 is a separate cell dataset and must not be silently substituted for missing Panasonic measurements.

Place the original dataset under:

```text
data/raw/
|-- Panasonic_18650PF/           Preserve the supplied nested dataset structure
`-- LG_18650HG2/                 Optional separate-cell inspection data
```

The Panasonic files used by the current workflow include HPPC pulse tests, their companion `dis5_10p` files for data inspection, Cycle 1 through Cycle 4, US06, UDDS, and the located 25 degC C/20 test. The recorded manifests contain relative source paths, hashes, and provenance. Preserve filenames and nesting so the scripts can locate the expected files.

Raw files and row-level processed measurements are not distributed in Git. Parameters, aggregate fitting diagnostics, provenance, error tables, and figures are retained as research evidence. Source filenames and hashes in the manifests are metadata rather than the experimental files themselves.

## Reproduce the workflow

### 1. Check source-file loading

```powershell
.\.venv\Scripts\python.exe -B tests\check_us06_io.py
.\.venv\Scripts\python.exe -B tests\check_c20_io.py
.\.venv\Scripts\python.exe -B tests\check_hppc_io.py
```

These scripts inspect current sign, increasing time, duplicate handling, fields, and sampling. Successful loading does not establish model accuracy or parameter identifiability.

### 2. Generate the initial US06 figures

```powershell
.\.venv\Scripts\python.exe -B experiments\plot_us06.py
```

Output: `results/week1/figures/`.

### 3. Fit the independent 25 degC C/20 voltage proxy

```powershell
.\.venv\Scripts\python.exe -B experiments\calibrate_ocv_25c.py --project-root .
```

Output: `results/week2/ocv_25c/`. This fit uses a discharge terminal-voltage proxy and does not replace the HPPC-based OCV tables used by the current three-temperature simulation.

### 4. Run the complete Week 2 pipeline

```powershell
.\.venv\Scripts\python.exe -B experiments\run_week2.py --project-root .
```

This entry point generates the three-temperature baseline, model comparisons, source-date and capacity audits, refined electrical and thermal parameters, empirical noise summaries, Cycle 3 prediction bands, Cycle 4 evaluations, and result reports. The independent C/20 fit above is a separate entry point.

The complete workflow writes generated outputs under `results/week2/` and can overwrite existing outputs there. It does not modify raw measurements. To keep a separate generated result set, use:

```powershell
.\.venv\Scripts\python.exe -B experiments\run_week2.py --project-root . --output-dir data\processed\week2_reproduction
```

This alternative location is excluded from Git. Reproducing the complete workflow requires the expected local raw datasets.

### 5. Verify model implementation and existing results

```powershell
.\.venv\Scripts\python.exe -B tests\verify_week2.py --project-root . --results-dir results\week2
```

The verifier checks charge balance, exact RC response, healthy equilibrium, short-circuit algebra and heating, synthetic parameter recovery, ambient fallback provenance, source hashes, calibration/validation file separation, covariance properties, and selected output consistency.

Synthetic recovery checks are implementation tests, not proofs of identifiability for the experimental data. Complete result verification also requires locally generated `empirical_rest_blocks.npz` files. These residual blocks are excluded from Git; generate them with the complete pipeline after providing the raw data. A fresh clone alone cannot perform every data-dependent check.

## Model and data conventions

- Preserve the original `Current` column. Computation uses `Current_discharge_positive`, with discharge positive and charge negative.
- Time is in seconds and must increase strictly after duplicate handling. Large sampling gaps are recorded; segmented resampling does not interpolate across detected gaps.
- The loader rejects missing required measurements by default. A chamber-setpoint fallback must be requested explicitly and recorded. The Trise workflow requires measured chamber temperature rather than a constant-temperature fallback.
- The reference capacity is 2.9 Ah for the charge/SOC coordinate. It is not an independently identified equilibrium capacity for every experiment date. Changing the reference requires a consistent change of OCV and parameter lookup coordinates.
- The second-order circuit contains `R0`, two polarization branches `R1/C1` and `R2/C2`, and a lumped thermal state with `Cth` and `hA`.
- Internal leakage is retained explicitly: `I_cell = i + i_s`, with `i_s = f_isc * v` and additional short-circuit heat `i_s * v`. Heat uses the actual cell current.
- Fault inputs include `fi` [A], `fv` [V], `fT` [degC], `fisc` [S], `fQ` [fraction], `fR` [fraction], and `onset_s` [s].
- Covariance channel order is `[current_A, voltage_V, temperature_C]`. Effective quantization and model-discrepancy covariances are sensitivity assumptions, not independently validated white Gaussian sensor noise.

### Selected refined model

Load the complete parameter JSON for the selected temperature from `results/week2/comparisons/calibration_refinement/{25C,10C,0C}/parameters.json`, together with its referenced OCV table. The model includes SOC/temperature-dependent resistances, OCV corrections, fixed branch time constants, an effective ambient offset, and a fitted voltage readout blend.

The scalar RC values in the refined summary table represent a reference SOC and temperature. They are not a replacement for the full parameter maps. The voltage readout blend is an empirical observation model, not a proven hardware delay; it must be included when constructing sensitivities for subsequent diagnosis work. For derivatives involving parameter maps, review the simulator's lookup rounding and `exact_parameter_lookup` option.

## Recorded validation results

The selected refined model's saved US06/UDDS results are:

| Temperature | Cycle | Voltage RMSE (mV) | Temperature RMSE (degC) |
| --- | --- | ---: | ---: |
| 25 degC | US06 | 17.685 | 0.367 |
| 25 degC | UDDS | 8.702 | 0.394 |
| 10 degC | US06 | 21.025 | 0.328 |
| 10 degC | UDDS | 12.967 | 0.229 |
| 0 degC | US06 | 34.827 | 0.497 |
| 0 degC | UDDS | 15.788 | 0.340 |

These values summarize existing outputs rather than a guarantee for future runs or other cells. See the linked refined error table for bias and maximum errors. In particular, the 25 and 0 degC UDDS temperature mean errors are approximately -0.382 and -0.314 degC.

### Calibration and evaluation separation

- HPPC provides initial electrical fits, rested-voltage proxies, and noise windows.
- Cycle 1 and Cycle 2 provide dynamic electrical/thermal calibration; Cycle 2 also participates in selecting the voltage readout response.
- Cycle 3 provides prediction-error bands and is not an independent validation of those bands.
- US06 and UDDS use files distinct from calibration sources. They have been inspected during model development and should be described as development validation rather than untouched blind tests.
- Cycle 4 provides additional fixed-parameter evaluation.

File hashes, internal timestamps, and source roles are recorded in the manifests and audit tables. Filenames can disagree with internal timestamps; use the audit rather than filenames alone.

## Interpretation of uncertainty

The 25 degC C/20 proxy has a saved full-curve RMSE of about 0.332 mV and same-experiment interpolation holdout RMSE of about 0.314 mV. Neither metric establishes independent equilibrium OCV accuracy.

Pure zero-heat cooling identifies `hA/Cth`, not both coefficients separately. The implemented thermal calibration uses excited operating data to estimate both coefficients, conditional on the heat and ambient models. Saved ambient-offset sensitivity profiles are scenario analyses, not statistical confidence intervals.

Resting current is logged as zero in the selected noise windows. Its zero empirical variance is not evidence of a noiseless sensor. Active-current fluctuations include controller ripple. Observed large temperature jumps are used for quantization sensitivity; they are not a proven ADC quantization interval.

`noise/*/sigma_model_error_sensitivity.csv` belongs to the initial baseline calibration. Do not silently use it as the refined model's current residual covariance. Likewise, nominal prediction-band coverage should be checked against saved coverage tables rather than assumed.

## Historical file locations

The initial `experiments/test_*` data checks now live in `tests/check_*`. MAT inspection scripts moved from `experiments/` to `tools/inspection/`; the old `inspect_hppc_mat.py` was renamed `inspect_c20_mat.py` because it inspects C/20 data. The four original US06 figures moved from `results/figures/` to `results/week1/figures/`.

Historical exploration scripts are kept under `archive/experiments/`. Their local result folders and backup snapshots are excluded from the Week 2 delivery. They are not inputs to the selected numerical model.

## Git delivery policy

Commit reusable code, runnable entry points, checks, parameter and error tables, provenance metadata, diagnostic figures, and explicit research limitations. Exclude raw datasets, row-level processed measurements, empirical residual blocks, virtual environments, caches, credentials, local backups, and superseded exploration outputs.

Check `.gitignore` and the actual staged diff before committing. The intended Week 2 commit message is:

```text
Complete week 2 calibration and conditional validation
```
