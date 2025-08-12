**JASTIS Health Indifference Analysis**

This repository contains the analysis script for investigating health indifference and its association with health outcomes using the JASTIS (Japan Society and New Integrated Survey) longitudinal dataset. The script implements a comprehensive epidemiological analysis pipeline with causal discovery methods.

**Key Features**

* **Data Processing**: Automated loading of 3-year panel data (2021-2023) with encoding detection and column name flexibility
* **Health Indifference Score**: Calculation of 13-item HI score with reverse coding, z-standardization, and quartile categorization
* **IPCW Weighting**: Inverse Probability of Censoring Weighting with stabilization to handle attrition bias
* **Multiple Imputation**: MICE (Multiple Imputation by Chained Equations) for missing covariate data with Rubin's rules for pooling
* **Primary Analysis**: Modified Poisson regression with robust standard errors for risk ratios and standardized risk differences (aRD)
* **Causal Discovery**: Integration of DirectLiNGAM, GOLEM (with EV→NV two-stage execution), CORL (with DAG masks), and DAGMA algorithms
* **Sensitivity Analyses**: Quartile comparisons, interaction tests, and stratified analyses
* **Visualization**: Forest plots, spline curves, and temporal causal networks with CUD-compliant color schemes

**Dependencies**

Core packages (automatically checked at runtime):
* `numpy`, `pandas`, `matplotlib`, `scipy`, `statsmodels`
* `scikit-learn`, `patsy`, `networkx`
* `torch` (optional, for neural causal methods)
* `castle` (optional, for gCastle algorithms)
* `dagma` (optional, for DAGMA algorithm)

**Data Structure**

Place the following CSV files in your folder:
* `2021_row.csv`: Baseline covariates
* `2022_row.csv`: Health Indifference exposure
* `2023_row.csv`: Health outcomes

Update the `DRIVE_FOLDER` path in the script to point to your location.

**Usage**

```python
# Run in Google Colab or local environment
python jastis_hi_analysis.py
```

The script automatically:
1. Loads and merges 3-year panel data
2. Calculates IPCW weights
3. Performs multiple imputation (m=5-10)
4. Runs primary and sensitivity analyses
5. Generates all figures and tables

**Hyperparameter Configuration**

The script includes optimized hyperparameters for causal discovery algorithms:
* **DirectLiNGAM**: threshold=0.03, with temporal and immutable variable constraints
* **GOLEM**: λ₁=2e-3, λ₂=5.0, 100k iterations, two-stage EV→NV execution
* **CORL**: BIC scoring, transformer encoder, 1000 iterations with DAG mask constraints
* **DAGMA**: warm-up 50k iterations, max 80k iterations, threshold=0.03

**Outputs**

**Main Manuscript Files:**
* `Table1_characteristics.csv`: Baseline characteristics by HI quartile
* `Figure2_forest_main.png`: Forest plot of adjusted risk ratios
* `Figure3_causal_network_disease.png`: Causal network for disease outcomes
* `Figure4_spline.png`: Spline curve for hospitalization risk

**Supplementary Materials:**
* `SupplementaryTable1_main_results.csv`: Complete results with rRR and aRD
* `SupplementaryTable2_quartile_analysis.csv`: Quartile-based sensitivity analysis
* `SupplementaryTable3_interaction.csv`: Effect modification tests
* `SupplementaryTable5_absolute_risk.csv`: Absolute risks by quartile
* `SupplementaryTable6_causal_edges.csv`: Causal edges from 5-fold CV
* `SupplementaryFigure1_causal_network_symptoms.png`: Causal network for symptom outcomes

**Statistical Methods**

* **Primary Analysis**: Modified Poisson regression with log link and robust standard errors
* **Weighting**: Stabilized IPCW with 1st/99th percentile trimming
* **Multiple Comparisons**: FDR correction (Benjamini-Hochberg) and Bonferroni adjustment
* **Causal Discovery**: 5-fold cross-validation with edge retention threshold ≥3 folds
* **Risk Differences**: G-computation with delta method for variance estimation

**Performance Considerations**

* GPU recommended for GOLEM and CORL algorithms

**License**

This project is licensed under the MIT License.

**Citation**

If you use this code, please cite the associated publication.
