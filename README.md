# Health Indifference Analysis

This repository hosts the analysis script used in the Health Indifference (HI) study of the JASTIS dataset. `health-indifference-analysis.py` provides a pipeline that executes data preprocessing, causal inference, and figure generation.

## Key Features
- Data integrity checks and memory optimization
- Calculation of the Health Indifference score, including Cronbach's alpha
- Variable creation according to the study protocol
- Missing data imputation via multiple imputation by chained equations (MICE)
- Statistical tests using t-test, Brunner–Munzel test, and chi-square test to generate Tables 1–3
- Causal inference integrating CORL, DirectLiNGAM, GOLEM, and DAGMA (optional 5-fold cross-validation)
- Automatic saving of results and figures locally and, when available, to Google Drive

## Dependencies
The script automatically installs required packages at runtime. Key dependencies include:

- numpy, pandas, matplotlib, seaborn
- scikit-learn, statsmodels, scipy, tqdm, openpyxl, missingno, joblib, networkx, torch
- gcastle, dagma

## Data Preparation
Prepare a longitudinal CSV file containing variables such as `UserID`, `Age_21`, `HealthOrientationScore_22`, etc. The file path is specified when running the script.

## Usage
```bash
python health-indifference-analysis.py --hi_var_type continuous --use_cv
```
- `--hi_var_type`: `continuous` (HI score) or `binary` (High_HI)
- `--use_cv`: Enable 5-fold cross-validation for causal inference

Given the heavy computational load, running on a GPU environment is recommended.

## Outputs
The script generates the following tables and figures:
- Table 1: Participant characteristics by Health Indifference
- Table 2: Determinants of Health Indifference (mutually adjusted model)
- Table 3: Outcomes associated with Health Indifference
- Causal inference figures, participant flowcharts, and forest plots

Generated files are saved locally and, in Google Colab environments, automatically copied to Google Drive.

## License
This project is licensed under the MIT License.
