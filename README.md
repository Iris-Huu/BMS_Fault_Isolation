# BMS Fault Detectability and Isolability

## 1. Project Overview

This project focuses on fault detectability and isolability in Battery Management Systems (BMS).

The research aims to investigate how different battery faults can be detected and distinguished through system modeling, analysis, and experimental data.

## 2. Project Structure

```text
BMS_Fault_Isolation/
├── data/
│   ├── raw/                  Original experimental data (not tracked by Git)
│   └── processed/            Processed data (not tracked by Git)
├── src/                      Reusable source code
│   └── io.py                 Data loading and preprocessing interface
├── experiments/              Experimental and analysis scripts
│   └── plot_us06.py          Panasonic US06 plotting script
├── results/
│   └── figures/              Generated figures
├── requirements.txt          Python dependencies
├── .gitignore
└── README.md
```

## 3. Data

The project primarily uses Panasonic 18650PF battery experimental data.

LG 18650HG2 battery data is also collected for cross-cell validation.

Raw and processed datasets are excluded from this Git repository.

## 4. Development Environment

The project uses Python and the following main packages:

- NumPy
- Pandas
- SciPy
- Matplotlib
- SymPy
- Scikit-learn

Exact package versions are recorded in `requirements.txt`.

## 5. Current Progress

- Implemented a data loading interface for Panasonic battery measurements.
- Added data quality checks, duplicate-record handling, and current sign convention conversion.
- Implemented segmented resampling for time-series data.
- Completed initial data quality checks for the 25°C US06 drive-cycle dataset.
- Generated voltage, current, and temperature plots for the Panasonic US06 dataset.

## 6. Notes

This repository is under active development.