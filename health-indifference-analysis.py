#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
JASTIS Health Indifference Analysis - Final Revised Version (Hyperparameter Optimized)
- Data source: /content/2021_row.csv, /content/2022_row.csv, /content/2023_row.csv
- Main analysis: modified Poisson (log link, robust SE) + stabilized IPCW
- Causal discovery: gCastle/DAGMA compatible (if available) + prior knowledge constraints
- Improvements: GOLEM EV→NV two-stage execution, CORL dag_mask added
"""

# ===== Hyperparameter Settings (Optimized Version) =====
CAUSAL_PARAMS = {
    "DirectLiNGAM": { #Execution instant
        "thresh": 0.03,            # threshold. 0.01: Good balance between detection power and false positives
        "measure": "pwling",
        "prior_knowledge": None    # Set dynamically at runtime
    },
    
    "GOLEM": {                     #Execution time 10-20min/5Fold approximately
        "lambda_1": 2e-3,          # Recommended range for NV
        "lambda_2": 5.0,           # Recommended value
        "learning_rate": 1e-3,     # Recommended value
        "num_iter": 100000,        # Recommended value (realistic with A100) #100000
        "checkpoint_iter": 5000,   # Intermediate logging
        "graph_thres": 0.001,        # threshold. Default value (suppress unnecessary edges)
        "equal_variances": True    # True if prioritizing convergence with EV
    },
    
    "GOLEM_CONSTRAINTS": {
        "use_temporal": True,      # Use temporal constraints
        "use_immutable": True,     # Use immutable variable constraints
        "immutable_vars": ["male", "age"],
        "init_scale": 0.05,        # Suppress initial random fluctuation
        "use_two_stage": True      # Enable EV→NV two-stage execution
    },
    
    "CORL": { #Execution time/fold approximately
        "batch_size": 32,              # Must be at least n_nodes
        "input_dim": 64,              # Feature dimension
        "embed_dim": 128,              # Expand embedding dimension
        "encoder_name": "transformer",
        "encoder_heads": 4,            # Increase number of heads
        "encoder_blocks": 4,           # Increase number of blocks
        "encoder_dropout_rate": 0.1,   # Adjust dropout rate
        "decoder_name": "lstm",
        "reward_mode": "episodic",
        "reward_score_type": "BIC",    # Changed to BIC (improved stability)
        "reward_regression_type": "LR",
        "lambda_iter_num": 500,        # Score update interval
        "actor_lr": 1e-5,              # Lower learning rate
        "critic_lr": 1e-4,             # Lower learning rate
        "iteration": 1000,              # Significantly increased #1000
        "use_float64": True,           # Must be True to avoid double/float mismatch error
        "use_dag_mask": True,          # Use dag_mask
        "edge_threshold": 1e-4         # Edge extraction threshold (added)              
    },
    
    "DAGMA": {                         # 20min/fold approximately
        "dims": [None, 10, 1],         # Input dimension set at runtime
        "bias": True,
        "lambda1": 2e-2,               # Paper/API default
        "lambda2": 5e-3,               # Paper/API default
        "lr": 2e-5,                    # Paper/API default
        "w_threshold": 0.03,            # threshold. Auto-prune small edges
        "T": 4,
        "mu_init": 0.1,
        "mu_factor": 0.1,
        "s": 1.0,
        "warm_iter": 50000,            # 5e4: Paper default #50000
        "max_iter": 80000,             # 8e4: Paper default #80000
        "checkpoint": 1000
    },
    
    "CV": {
        "n_folds": 5,
        "random_state": 2025,
        "fold_threshold": 3            # Ensure recall
    }
}

# ===== Standard/External Libraries =====
import os
import sys
import json
import math
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold

from patsy import dmatrix

import networkx as nx

# torch (if available)
try:
    import torch
    TORCH_AVAILABLE = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except:
    TORCH_AVAILABLE = False
    device = None

# gCastle (if available)
try:
    import castle
    from castle.algorithms import DirectLiNGAM, GOLEM, CORL
    GCASTLE_AVAILABLE = True
except:
    GCASTLE_AVAILABLE = False

# DAGMA (if available)
try:
    import dagma
    from dagma.nonlinear import DagmaMLP, DagmaNonlinear
    DAGMA_AVAILABLE = True
except:
    DAGMA_AVAILABLE = False

# ===== Logger =====
import logging
logger = logging.getLogger("jastis")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

# ===== Paths =====
# Specify folder 
DRIVE_FOLDER = "" 

# Data file paths
DATA_2021 = f"{DRIVE_FOLDER}/2021_row.csv"
DATA_2022 = f"{DRIVE_FOLDER}/2022_row.csv"
DATA_2023 = f"{DRIVE_FOLDER}/2023_row.csv"

# Output destination also set to same folder
OUT_DIR = DRIVE_FOLDER

# Create folder if it doesn't exist
import os
os.makedirs(OUT_DIR, exist_ok=True)

# ====== 1) Data Loading ======
def read_csv_known_enc(path: str) -> Tuple[pd.DataFrame, str]:
    for enc in ["utf-8", "utf-8-sig", "cp932", "shift_jis"]:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            logger.info(f"Loaded {os.path.basename(path)} with encoding={enc}, shape={df.shape}")
            return df, enc
        except Exception:
            continue
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded {os.path.basename(path)} with default encoding, shape={df.shape}")
    return df, "default"

# ====== 2) Column Name Search ======
def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None

# ====== 3) 2021 Covariates ======
def build_covariates_2021(df21: pd.DataFrame) -> pd.DataFrame:
    d = df21.copy()
    out = pd.DataFrame({"USER_ID": d["USER_ID"]})

    sex_col = find_column(d, ["SEX", "Sex", "sex"])
    if sex_col:
        out["male"] = pd.to_numeric(d[sex_col], errors="coerce").apply(
            lambda x: 1 if x == 1 else (0 if x == 2 else np.nan)
        )
    else:
        out["male"] = np.nan

    age_col = find_column(d, ["AGE", "Age", "age"])
    if age_col:
        out["age"] = pd.to_numeric(d[age_col], errors="coerce")
    else:
        out["age"] = np.nan

    edu_col = find_column(d, ["Q12", "q12"])
    if edu_col:
        out["low_edu"] = pd.to_numeric(d[edu_col], errors="coerce").apply(
            lambda x: 1 if x in [1,2] else (0 if pd.notna(x) else np.nan)
        )
    else:
        out["low_edu"] = np.nan

    inc_col = find_column(d, ["Q10", "q10"])
    if inc_col:
        out["low_income"] = pd.to_numeric(d[inc_col], errors="coerce").apply(
            lambda x: 1 if x in [1,2,3,4] else (0 if pd.notna(x) else np.nan)
        )
    else:
        out["low_income"] = np.nan

    hh_col = find_column(d, ["Q7.1", "Q7_1", "Q7"])
    if hh_col:
        out["living_alone"] = pd.to_numeric(d[hh_col], errors="coerce").apply(
            lambda x: 1 if x == 1 else (0 if pd.notna(x) else np.nan)
        )
    else:
        out["living_alone"] = np.nan

    soc_col = find_column(d, ["Q17.7", "Q17_7", "Q17"])
    if soc_col:
        out["lack_social_support"] = pd.to_numeric(d[soc_col], errors="coerce").apply(
            lambda x: 1 if x >= 4 else (0 if pd.notna(x) else np.nan)
        )
    else:
        out["lack_social_support"] = np.nan

    smk_col = find_column(d, ["Q9", "q9"])
    if smk_col:
        out["current_smoking"] = pd.to_numeric(d[smk_col], errors="coerce").apply(
            lambda x: 1 if x in [1,2] else (0 if pd.notna(x) else np.nan)
        )
    else:
        out["current_smoking"] = np.nan

    drf_col = find_column(d, ["Q41", "q41"])
    dra_col = find_column(d, ["Q42", "q42"])
    if drf_col and dra_col:
        freq = pd.to_numeric(d[drf_col], errors="coerce")
        amount = pd.to_numeric(d[dra_col], errors="coerce")
        out["heavy_drinking"] = ((freq >= 5) & (amount >= 3)).astype(float)
        out.loc[freq.isna() | amount.isna(), "heavy_drinking"] = np.nan
    else:
        out["heavy_drinking"] = np.nan

    wlk_col = find_column(d, ["Q25.4", "Q25_4"])
    vig_col = find_column(d, ["Q25.7", "Q25_7"])
    if wlk_col and vig_col:
        walk = pd.to_numeric(d[wlk_col], errors="coerce")
        vigorous = pd.to_numeric(d[vig_col], errors="coerce")
        out["low_physical_activity"] = ((walk <= 2) & (vigorous == 1)).astype(float)
        out.loc[walk.isna() | vigorous.isna(), "low_physical_activity"] = np.nan
    else:
        out["low_physical_activity"] = np.nan

    hck_col = find_column(d, ["Q34.3", "Q34_3"])
    if hck_col:
        out["no_health_checkup"] = pd.to_numeric(d[hck_col], errors="coerce").apply(
            lambda x: 1 if x == 2 else (0 if x == 1 else np.nan)
        )
    else:
        out["no_health_checkup"] = np.nan

    srh_col = find_column(d, ["Q78", "q78"])
    if srh_col:
        out["poor_self_rated_health"] = pd.to_numeric(d[srh_col], errors="coerce").apply(
            lambda x: 1 if x in [4,5] else (0 if x in [1,2,3] else np.nan)
        )
    else:
        out["poor_self_rated_health"] = np.nan

    ht_col = find_column(d, ["Q83.1", "Q83_1"])
    wt_col = find_column(d, ["Q83.2", "Q83_2"])
    if ht_col and wt_col:
        h = pd.to_numeric(d[ht_col], errors="coerce") / 100.0
        w = pd.to_numeric(d[wt_col], errors="coerce")
        out["bmi"] = w / (h ** 2)
        out["bmi"] = out["bmi"].clip(10, 60)
    else:
        out["bmi"] = np.nan

    return out

# ====== 4) 2022 HI (13 items) ======
def score_hi_2022(df22: pd.DataFrame) -> pd.DataFrame:
    d = df22.copy()
    out = pd.DataFrame({"USER_ID": d["USER_ID"]})

    items = []
    for i in range(1, 14):
        col = find_column(d, [f"Q32.{i}", f"Q32_{i}"])
        if col:
            items.append(col)

    if len(items) < 13:
        logger.warning(f"Only {len(items)} HI items found (expected 13)")

    for col in items:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    for i in range(9, 13):
        col = find_column(d, [f"Q32.{i}", f"Q32_{i}"])
        if col:
            d[col] = 5 - d[col]

    if items:
        out["HI_sum_22"] = d[items].sum(axis=1, min_count=10)
        mean = out["HI_sum_22"].mean()
        sd = out["HI_sum_22"].std(ddof=1)
        median = out["HI_sum_22"].median()
        logger.info(f"HI raw mean={mean:.2f}, sd={sd:.2f}, median={median:.0f}")

        out["HI_z_22"] = (out["HI_sum_22"] - mean) / sd
        out["HI_quartile_22"] = pd.qcut(out["HI_sum_22"], q=4, labels=[1,2,3,4], duplicates='drop')
        out["High_HI_22"] = (out["HI_sum_22"] >= median).astype(float)
    else:
        out["HI_sum_22"] = np.nan
        out["HI_z_22"] = np.nan
        out["HI_quartile_22"] = np.nan
        out["High_HI_22"] = np.nan

    return out

# ====== 5) 2023 Outcomes ======
def build_outcomes_2023(df23: pd.DataFrame) -> pd.DataFrame:
    d = df23.copy()
    out = pd.DataFrame({"USER_ID": d["USER_ID"]})

    # Hospitalization
    hosp_col = find_column(d, ["Q91", "q91"])
    if hosp_col:
        out["HospitalizationPastYear_23"] = pd.to_numeric(d[hosp_col], errors="coerce").apply(
            lambda x: 1 if x == 1 else (0 if pd.notna(x) else np.nan)
        )
    else:
        out["HospitalizationPastYear_23"] = np.nan

    # COVID infection (Q23.*-1 with 1-3=infected, any positive = 1)
    INFECTION_TRIGGER_COLS = [
        "Q23.1-1","Q23.2-1","Q23.3-1","Q23.4-1","Q23.5-1",
        "Q23.6-1","Q23.7-1","Q23.8-1","Q23.9-1","Q23.10-1"
    ]
    trigger_cols = [c for c in INFECTION_TRIGGER_COLS if c in d.columns]
    if len(trigger_cols) == 0:
        out["COVID19_Infection_23"] = np.nan
        logger.error("[COVID] Q23.*-1 not found.")
    else:
        vals = d[trigger_cols].apply(pd.to_numeric, errors="coerce")
        infected = vals.isin([1,2,3]).any(axis=1)
        out["COVID19_Infection_23"] = infected.astype(float)
        n1 = int((out["COVID19_Infection_23"] == 1).sum())
        n0 = int((out["COVID19_Infection_23"] == 0).sum())
        n_na = int(out["COVID19_Infection_23"].isna().sum())
        all_missing_rows = int(vals.notna().any(axis=1).eq(False).sum())
        logger.info(f"[COVID] triggers={len(trigger_cols)}/10, pos={n1}, neg={n0}, NaN={n_na}, all-missing(-1)={all_missing_rows}")

    # COVID vaccine
    covid_vac_col = find_column(d, ["Q20", "q20"])
    if covid_vac_col:
        val = pd.to_numeric(d[covid_vac_col], errors="coerce")
        out["COVID19_Vaccination_23"] = val.apply(
            lambda x: 1 if x in [1,2,3,4,5] else (0 if x in [6,7,8,9] else np.nan)
        )
    else:
        out["COVID19_Vaccination_23"] = np.nan

    # Influenza vaccine
    flu_col = find_column(d, ["Q37.10", "Q37_10"])
    if flu_col:
        out["InfluenzaVaccination_23"] = pd.to_numeric(d[flu_col], errors="coerce").apply(
            lambda x: 1 if x == 1 else (0 if x == 2 else np.nan)
        )
    else:
        out["InfluenzaVaccination_23"] = np.nan

    # Chronic diseases
    CHRONIC = {
        "Hypertension_23": ["Q89.1", "Q89_1"],
        "Diabetes_23": ["Q89.2", "Q89_2"],
        "Dyslipidemia_23": ["Q89.3", "Q89_3"],
        "Asthma_23": ["Q89.5", "Q89_5"],
        "Periodontitis_23": ["Q89.8", "Q89_8"],
        "DentalCaries_23": ["Q89.9", "Q89_9"],
        "AnginaMI_23": ["Q89.10", "Q89_10"],
        "Stroke_23": ["Q89.11", "Q89_11"],
        "COPD_23": ["Q89.12", "Q89_12"],
        "CKD_23": ["Q89.13", "Q89_13"],
        "ChronicLiverDisease_23": ["Q89.14", "Q89_14"],
        "Cancer_23": ["Q89.16", "Q89_16"],
        "ChronicPain_23": ["Q89.17", "Q89_17"],
        "Depression_23": ["Q89.18", "Q89_18"],
    }
    for name, candidates in CHRONIC.items():
        col = find_column(d, candidates)
        if col:
            out[name] = pd.to_numeric(d[col], errors="coerce").apply(
                lambda x: 1 if x in [3,4,5] else (0 if x in [1,2] else np.nan)
            )
        else:
            out[name] = np.nan

    # Symptoms
    SYMPTOM = {
        "GIDiscomfort_23": ["Q26.1", "Q26_1"],
        "BackPain_23": ["Q26.2", "Q26_2"],
        "JointPain_23": ["Q26.3", "Q26_3"],
        "Headache_23": ["Q26.4", "Q26_4"],
        "ChestPain_23": ["Q26.5", "Q26_5"],
        "Dyspnea_23": ["Q26.6", "Q26_6"],
        "Dizziness_23": ["Q26.7", "Q26_7"],
        "SleepDisturbance_23": ["Q26.9", "Q26_9"],
        "MemoryDisorder_23": ["Q26.13", "Q26_13"],
        "ConcentrationDecline_23": ["Q26.14", "Q26_14"],
        "ReducedLibido_23": ["Q26.17", "Q26_17"],
        "Fatigue_23": ["Q26.18", "Q26_18"],
        "Cough_23": ["Q26.19", "Q26_19"],
        "Fever_23": ["Q26.20", "Q26_20"],
    }
    for name, candidates in SYMPTOM.items():
        col = find_column(d, candidates)
        if col:
            val = pd.to_numeric(d[col], errors="coerce")
            out[name] = (val >= 3).astype(float)
            out.loc[val.isna(), name] = np.nan
        else:
            out[name] = np.nan

    return out

# ====== 6) Panel Merge ======
def merge_panel(cov21: pd.DataFrame, hi22: pd.DataFrame, y23: pd.DataFrame) -> pd.DataFrame:
    a = pd.merge(cov21, hi22, on="USER_ID", how="inner")
    panel = pd.merge(a, y23, on="USER_ID", how="inner")
    logger.info(f"Merged panel size: {panel.shape[0]:,}")
    return panel

# ====== 7) IPCW ======
def compute_ipcw_weights(df21_all: pd.DataFrame, panel_ids: pd.Series) -> pd.DataFrame:
    d = df21_all.copy()
    d["complete3"] = d["USER_ID"].isin(set(panel_ids)).astype(int)

    covars = []
    for col_candidates in [
        ["SEX"], ["AGE"], ["Q12"], ["Q10"], ["Q7.1", "Q7_1"],
        ["Q17.7", "Q17_7"], ["Q9"], ["Q41"], ["Q42"],
        ["Q83.1", "Q83_1"], ["Q83.2", "Q83_2"],
        ["Q25.4", "Q25_4"], ["Q25.7", "Q25_7"],
        ["Q34.3", "Q34_3"], ["Q78"]
    ]:
        col = find_column(d, col_candidates)
        if col:
            covars.append(col)

    if not covars:
        logger.warning("No covariates found for IPCW")
        return pd.DataFrame({"USER_ID": d["USER_ID"], "ipcw_weight": 1.0, "complete3": d["complete3"]})

    X = d[covars].copy()
    for col in covars:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median())

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    y = d["complete3"].values
    lr = LogisticRegression(max_iter=1000)
    lr.fit(Xs, y)
    p_hat = lr.predict_proba(Xs)[:,1]
    p_hat = np.clip(p_hat, 1e-6, 1-1e-6)

    p_bar = y.mean()
    sw = np.where(y==1, p_bar / p_hat, (1-p_bar) / (1-p_hat))
    w = pd.DataFrame({"USER_ID": d["USER_ID"], "ipcw_raw": sw, "complete3": y})

    ql, qh = np.nanpercentile(w.loc[w["complete3"]==1,"ipcw_raw"], [1, 99])
    w["ipcw_weight"] = w["ipcw_raw"].clip(lower=ql, upper=qh)

    logger.info(f"IPCW weights trimmed to [{ql:.3f}, {qh:.3f}]")
    return w[["USER_ID","ipcw_weight","complete3"]]

# ====== 8) Multiple Imputation ======
COVARS = ["male","age","low_edu","low_income","living_alone","lack_social_support",
          "current_smoking","heavy_drinking","low_physical_activity","no_health_checkup",
          "poor_self_rated_health","bmi"]

def multiple_impute_covars(panel: pd.DataFrame, m: int = 5, random_state: int = 123) -> List[pd.DataFrame]:
    imputed_list = []
    X = panel[COVARS].copy()
    imp = IterativeImputer(random_state=random_state, max_iter=25, sample_posterior=True)
    for k in range(m):
        X_imp = pd.DataFrame(imp.fit_transform(X), columns=COVARS, index=panel.index)
        d = panel.copy()
        d[COVARS] = X_imp[COVARS]
        d["mi_id"] = k+1
        imputed_list.append(d)
    return imputed_list

# ====== 9) Modified Poisson ======
def fit_modified_poisson(df: pd.DataFrame, outcome: str, exposure: str, weights: str, adjust: List[str]):
    rhs = " + ".join([exposure] + adjust)
    formula = f"{outcome} ~ {rhs}"
    model = smf.glm(formula=formula, data=df, family=sm.families.Poisson(),
                    freq_weights=None, var_weights=df[weights])
    res = model.fit(cov_type="HC0")
    b = res.params[exposure]
    se = res.bse[exposure]
    rr = math.exp(b)
    lcl = math.exp(b - 1.96*se)
    ucl = math.exp(b + 1.96*se)
    p = res.pvalues[exposure]
    return rr, lcl, ucl, p

# ====== 10) Rubin's Rules ======
def pool_mi_results(mi_dfs: List[pd.DataFrame], outcome: str, exposure: str, weights: str, adjust: List[str]):
    betas, ses2 = [], []
    for d in mi_dfs:
        rhs = " + ".join([exposure] + adjust)
        formula = f"{outcome} ~ {rhs}"
        model = smf.glm(formula=formula, data=d, family=sm.families.Poisson(), var_weights=d[weights])
        res = model.fit(cov_type="HC0")
        b = res.params[exposure]
        se = res.bse[exposure]
        betas.append(b)
        ses2.append(se**2)

    m = len(betas)
    b_bar = np.mean(betas)
    W = np.mean(ses2)
    B = np.var(betas, ddof=1) if m > 1 else 0.0
    T = W + (1 + 1/m) * B
    se_bar = math.sqrt(T)
    rr = math.exp(b_bar)
    lcl = math.exp(b_bar - 1.96*se_bar)
    ucl = math.exp(b_bar + 1.96*se_bar)

    z = b_bar / se_bar if se_bar > 0 else np.inf
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return rr, lcl, ucl, p, {"beta_bar": b_bar, "se_bar": se_bar, "W": W, "B": B, "T": T}

# ====== 10b) Standardized Risk Difference (aRD) + 95%CI with MI pooling [ARD_STD] ======
def standardized_riskdiff_mi(mi_list: List[pd.DataFrame],
                             outcome: str, weights: str,
                             exposure: str, adjust: List[str],
                             delta: float = 1.0) -> Tuple[float, float, float]:
    """
    Using g-computation from Poisson(log) model,
    calculate adjusted risk difference (in % points) and 95%CI
    when HI_z_22 is increased by +delta, pooled using Rubin's rules.

    Returns: (aRD_pctpt, LCL_pctpt, UCL_pctpt)
    """
    rd_list = []
    var_list = []

    for d in mi_list:
        need = [outcome, weights, exposure] + adjust
        df = d.dropna(subset=need).copy()
        if df.empty:
            continue

        # Fit
        rhs = " + ".join([exposure] + adjust)
        formula = f"{outcome} ~ {rhs}"
        mdl = smf.glm(formula=formula, data=df, family=sm.families.Poisson(),
                      var_weights=df[weights])
        res = mdl.fit(cov_type="HC0")

        # Coefficients and covariance
        params = res.params.copy()
        V = res.cov_params().values
        param_names = list(res.params.index)

        # Intercept name
        if "Intercept" in params.index:
            intercept_name = "Intercept"
        elif "const" in params.index:
            intercept_name = "const"
        else:
            # Just in case (shouldn't happen)
            intercept_name = param_names[0]

        # Linear predictor η0 = b0 + bx*x + Σ bk*zk
        b0 = params[intercept_name]
        bx = params[exposure]
        x = df[exposure].values
        Z = df[adjust].values if len(adjust) > 0 else np.zeros((len(df), 0))

        # Adjustment coefficient vector (order follows adjust)
        bZ = np.array([params[a] for a in adjust]) if len(adjust) > 0 else np.array([])

        eta0 = b0 + bx * x + (Z @ bZ if Z.size > 0 else 0.0)
        mu0 = np.exp(eta0)
        mu1 = np.exp(eta0 + bx * delta)  # Only change x -> x+delta

        # Clip predictions to [0,1] (safety for Poisson properties)
        mu0 = np.clip(mu0, 0, 1)
        mu1 = np.clip(mu1, 0, 1)

        w = df[weights].values
        sumw = np.sum(w)

        # RD (probability difference)
        rd = float(np.sum(w * (mu1 - mu0)) / sumw)  # Proportion
        # Gradient g = (1/sumw) * Σ w*(μ1 X1 - μ0 X0)
        # X is [1, x, Z...], but X1's 2nd column is (x+delta)
        diff = (mu1 - mu0)

        g_map = {}
        g_map[intercept_name] = float(np.sum(w * diff) / sumw)  # Intercept

        # exposure
        g_map[exposure] = float(np.sum(w * (mu1 * (x + delta) - mu0 * x)) / sumw)

        # adjust
        for j, a in enumerate(adjust):
            z = Z[:, j] if Z.size > 0 else 0.0
            g_map[a] = float(np.sum(w * diff * z) / sumw)

        # Arrange g in parameter order
        g = np.array([g_map[name] if name in g_map else 0.0 for name in param_names])

        # Variance (delta method, using robust V)
        var_rd = float(g.T @ V @ g)

        rd_list.append(rd)
        var_list.append(var_rd)

    if not rd_list:
        return np.nan, np.nan, np.nan

    # Pool with Rubin's rules (rd is proportion, var is its variance)
    m = len(rd_list)
    rd_bar = float(np.mean(rd_list))
    W = float(np.mean(var_list))
    B = float(np.var(rd_list, ddof=1)) if m > 1 else 0.0
    T = W + (1 + 1/m) * B
    se_bar = math.sqrt(max(T, 0.0))

    # Convert to percentage points
    rd_pct = rd_bar * 100.0
    lcl_pct = (rd_bar - 1.96 * se_bar) * 100.0
    ucl_pct = (rd_bar + 1.96 * se_bar) * 100.0

    return rd_pct, lcl_pct, ucl_pct

# ====== 11) Outcome List ======
def outcome_list_all() -> List[str]:
    base = [
        "HospitalizationPastYear_23",
        "COVID19_Vaccination_23","InfluenzaVaccination_23","COVID19_Infection_23",
    ]
    chronic = ["Hypertension_23","Diabetes_23","Dyslipidemia_23","Asthma_23",
               "Periodontitis_23","DentalCaries_23","AnginaMI_23","Stroke_23",
               "COPD_23","CKD_23","ChronicLiverDisease_23","Cancer_23",
               "ChronicPain_23","Depression_23"]
    symptom = ["GIDiscomfort_23","BackPain_23","JointPain_23","Headache_23",
               "ChestPain_23","Dyspnea_23","Dizziness_23","SleepDisturbance_23",
               "MemoryDisorder_23","ConcentrationDecline_23","ReducedLibido_23",
               "Fatigue_23","Cough_23","Fever_23"]
    return base + chronic + symptom

# ====== 12) Causal Discovery (5-fold CV) (Improved Version) ======
class CausalDiscoveryCV:
    def __init__(self, hi_var_type="continuous"):
        self.hi_var_type = hi_var_type
        self.n_folds = CAUSAL_PARAMS["CV"]["n_folds"]
        self.random_state = CAUSAL_PARAMS["CV"]["random_state"]
        self.fold_threshold = CAUSAL_PARAMS["CV"]["fold_threshold"]
        self.device = device

    def _prepare(self, df: pd.DataFrame, subset="all") -> Tuple[np.ndarray, List[str]]:
        # Baseline variables (2021)
        det21 = ["male","low_edu","lack_social_support","current_smoking","no_health_checkup",
                 "low_income","living_alone","age","poor_self_rated_health",
                 "heavy_drinking","low_physical_activity","bmi"]
        
        # Exposure variable (2022)
        hi22 = ["HI_sum_22"] if self.hi_var_type=="continuous" else ["High_HI_22"]
        
        # Outcome variables (2023)
        # Disease/hospitalization/vaccine related (for Figure3)
        disease_outcomes = [
            "HospitalizationPastYear_23",
            "COVID19_Infection_23",
            "COVID19_Vaccination_23",
            "InfluenzaVaccination_23",
            "Hypertension_23",
            "Diabetes_23",
            "Dyslipidemia_23",
            "Asthma_23",
            "Periodontitis_23",
            "DentalCaries_23",
            "AnginaMI_23",
            "Stroke_23",
            "COPD_23",
            "CKD_23",
            "ChronicLiverDisease_23",
            "Cancer_23",
            "ChronicPain_23",
            "Depression_23"
        ]
        
        # Symptom related (for Supplementary Figure2)
        symptom_outcomes = [
            "GIDiscomfort_23",
            "BackPain_23",
            "JointPain_23",
            "Headache_23",
            "ChestPain_23",
            "Dyspnea_23",
            "Dizziness_23",
            "SleepDisturbance_23",
            "MemoryDisorder_23",
            "ConcentrationDecline_23",
            "ReducedLibido_23",
            "Fatigue_23",
            "Cough_23",
            "Fever_23"
        ]

        # Select variables according to subset
        if subset == "disease":
            use = det21 + hi22 + disease_outcomes
        elif subset == "symptoms":
            use = det21 + hi22 + symptom_outcomes
        else:
            # Default (only representative variables)
            use = det21 + hi22 + disease_outcomes[:4] + symptom_outcomes[:4]

        # Use only variables that exist in dataframe
        use = [c for c in use if c in df.columns]
        
        # Remove missing values
        m = df[use].dropna()
        if m.empty:
            logger.warning(f"[{subset}] No complete cases after removing NAs")
            return None, []
        
        logger.info(f"[{subset}] Using {len(use)} variables, {len(m)} complete cases")

        # Set data type
        dtype = np.float64 if CAUSAL_PARAMS["CORL"]["use_float64"] else np.float32
        X = m.copy().astype(dtype)
        
        # Standardize continuous variables
        cont = [c for c in use if not set(pd.unique(m[c])).issubset({0,1})]
        if cont:
            sc = RobustScaler()
            X[cont] = sc.fit_transform(m[cont].values.astype(dtype)).astype(dtype)
            logger.info(f"[{subset}] Standardized {len(cont)} continuous variables")

        return X.values.astype(dtype), use

    def _create_prior_knowledge(self, var_names: List[str], strictness: str = "moderate") -> np.ndarray:
        """
        Create prior knowledge matrix for DirectLiNGAM (stepwise constraints)
        prior_knowledge[i,j] = 1 means "prohibit j→i"
        
        strictness:
        - "minimal": Temporal constraints only
        - "moderate": Temporal constraints + most important immutable variables only
        - "strict": Temporal constraints + all immutable variables
        """
        n_vars = len(var_names)
        prior_knowledge = np.zeros((n_vars, n_vars))
        
        # Classify variables by year
        vars_2021 = []
        vars_2022 = []
        vars_2023 = []
        
        for i, var in enumerate(var_names):
            if var in ["HI_sum_22", "High_HI_22"]:
                vars_2022.append(i)
            elif "_23" in var:
                vars_2023.append(i)
            else:
                vars_2021.append(i)
        
        # Temporal constraints: prohibit future→past (applied at all levels)
        # Prohibit causality from 2023 to 2021/2022
        for i in vars_2021 + vars_2022:
            for j in vars_2023:
                prior_knowledge[i, j] = 1
        
        # Prohibit causality from 2022 to 2021
        for i in vars_2021:
            for j in vars_2022:
                prior_knowledge[i, j] = 1
        
        if strictness in ["moderate", "strict"]:
            # Prohibit causality to most critical immutable variables (male, age)
            critical_immutable = ["male", "age"]
            for target_var in critical_immutable:
                if target_var in var_names:
                    target_idx = var_names.index(target_var)
                    # Prohibit causality from other variables in same year to immutable variables
                    for source_idx in vars_2021:
                        if source_idx != target_idx:
                            prior_knowledge[target_idx, source_idx] = 1
        
        if strictness == "strict":
            # Also prohibit causality to low_edu
            if "low_edu" in var_names:
                edu_idx = var_names.index("low_edu")
                for source_idx in vars_2021:
                    if source_idx != edu_idx:
                        prior_knowledge[edu_idx, source_idx] = 1
        
        # Check if constraints are not too strong
        # Verify each variable can be an effect of at least one variable
        can_be_effect = np.sum(prior_knowledge == 0, axis=0) - 1  # Exclude self
        cant_be_effect = np.sum(can_be_effect == -1)
        
        logger.info(f"Prior knowledge ({strictness}): {n_vars}x{n_vars}, "
                   f"constraints: {int(np.sum(prior_knowledge))}, "
                   f"variables that can't be effects: {cant_be_effect}")
        
        return prior_knowledge

    def _create_dag_mask_for_corl(self, var_names: List[str]) -> np.ndarray:
        """
        Create dag_mask for CORL
        dag_mask[i,j] = 1 means "allow j→i", 0 means "prohibit j→i"
        (opposite meaning from prior_knowledge)
        """
        n_vars = len(var_names)
        # Explicitly specify data type (float32 or float64)
        dtype = np.float64 if CAUSAL_PARAMS["CORL"]["use_float64"] else np.float32
        dag_mask = np.ones((n_vars, n_vars), dtype=dtype)
        
        # Classify variables by year
        vars_2021 = []
        vars_2022 = []
        vars_2023 = []
        
        for i, var in enumerate(var_names):
            if var in ["HI_sum_22", "High_HI_22"]:
                vars_2022.append(i)
            elif "_23" in var:
                vars_2023.append(i)
            else:
                vars_2021.append(i)
        
        # Temporal constraints: prohibit future→past
        for i in vars_2021 + vars_2022:
            for j in vars_2023:
                dag_mask[i, j] = 0
        
        for i in vars_2021:
            for j in vars_2022:
                dag_mask[i, j] = 0
        
        # Prohibit causality to immutable variables
        immutable_vars = CAUSAL_PARAMS["GOLEM_CONSTRAINTS"]["immutable_vars"]
        for target_var in immutable_vars:
            if target_var in var_names:
                target_idx = var_names.index(target_var)
                for source_idx in range(n_vars):
                    if source_idx != target_idx:
                        dag_mask[target_idx, source_idx] = 0
        
        # Diagonal elements are 0 (prohibit self-loops)
        np.fill_diagonal(dag_mask, 0)
        
        logger.info(f"CORL dag_mask: {n_vars}x{n_vars}, allowed edges: {int(np.sum(dag_mask))}, dtype: {dag_mask.dtype}")
        
        return dag_mask

    def _create_B_init_for_golem(self, var_names: List[str], ev_result: np.ndarray = None) -> np.ndarray:
        """
        Create initial value matrix for GOLEM (reflecting constraints)
        B_init[i,j] is initial value of coefficient j→i
        ev_result: EV stage result (if provided, use as initial value)
        """
        n_vars = len(var_names)
        
        if ev_result is not None:
            # Use EV result as initial value - ensure float32
            B_init = ev_result.copy().astype(np.float32)
        else:
            # Random initialization
            init_scale = CAUSAL_PARAMS.get("GOLEM_CONSTRAINTS", {}).get("init_scale", 0.05)
            B_init = np.random.randn(n_vars, n_vars).astype(np.float32) * init_scale
        
        # Get constraint usage settings
        use_temporal = CAUSAL_PARAMS.get("GOLEM_CONSTRAINTS", {}).get("use_temporal", True)
        use_immutable = CAUSAL_PARAMS.get("GOLEM_CONSTRAINTS", {}).get("use_immutable", True)
        immutable_vars = CAUSAL_PARAMS.get("GOLEM_CONSTRAINTS", {}).get("immutable_vars", ["male", "age"])
        
        # Classify variables by year
        vars_2021 = []
        vars_2022 = []
        vars_2023 = []
        
        for i, var in enumerate(var_names):
            if var in ["HI_sum_22", "High_HI_22"]:
                vars_2022.append(i)
            elif "_23" in var:
                vars_2023.append(i)
            else:
                vars_2021.append(i)
        
        # Count non-zero elements before applying constraints
        non_zero_before = np.sum(np.abs(B_init) > 1e-10)
        
        if use_temporal:
            for i in vars_2021 + vars_2022:
                for j in vars_2023:
                    B_init[i, j] = 0.0  # Use 0.0 instead of 0            
                # Set 2022→2021 to 0
                for i in vars_2021:
                    for j in vars_2022:
                        B_init[i, j] = 0.0
        
        if use_immutable:
            # Set causality to immutable variables to 0
            for target_var in immutable_vars:
                if target_var in var_names:
                    target_idx = var_names.index(target_var)
                    for source_idx in range(n_vars):
                        if source_idx != target_idx:
                            B_init[target_idx, source_idx] = 0.0
        
        # Count non-zero elements after applying constraints
        non_zero_after = np.sum(np.abs(B_init) > 1e-10)
        
        logger.info(f"GOLEM B_init: {n_vars}x{n_vars} matrix, "
                  f"non-zero elements: {non_zero_before} → {non_zero_after} "
                  f"(removed {non_zero_before - non_zero_after} constraints)")
        
        return B_init

    def _cv_edges(self, edges_by_fold: List[Dict[str,float]]) -> Dict[str, Dict]:
        counter, coefmap = {}, {}
        for ed in edges_by_fold:
            for k,v in ed.items():
                counter[k] = counter.get(k,0)+1
                coefmap.setdefault(k, []).append(v)
        sig = {k: {"folds":c, "coef": float(np.mean(coefmap[k]))}
               for k,c in counter.items() if c >= self.fold_threshold}
        return sig

    def _filter_temporal_and_immutable(self, A: np.ndarray, names: List[str]) -> np.ndarray:
        """
        Apply temporal constraints and immutable variable constraints post-hoc
        """
        out = A.copy().astype(np.float32)
        
        # Apply temporal constraints
        idx21 = [i for i,n in enumerate(names) if not "_23" in n and n not in ["HI_sum_22","High_HI_22"]]
        idx22 = [i for i,n in enumerate(names) if n in ["HI_sum_22","High_HI_22"]]
        idx23 = [i for i,n in enumerate(names) if "_23" in n]
        
        # Remove 2023→2021/2022
        for i in idx21+idx22:
            for j in idx23:
                out[i,j] = 0.0
        
        # Remove 2022→2021
        for i in idx21:
            for j in idx22:
                out[i,j] = 0.0
        
        # Forcibly remove causality to immutable variables (Age, Male, low_edu) 
        # Add more constraints here if needed
        immutable_vars = ["age", "male","low_edu"]
        for var in immutable_vars:
            if var in names:
                var_idx = names.index(var)
                # Set causality from all variables to immutable variable to 0
                for source_idx in range(len(names)):
                    if source_idx != var_idx:
                        out[var_idx, source_idx] = 0.0
        
        # Set diagonal elements to 0 (prohibit self-loops)
        np.fill_diagonal(out, 0.0)
        
        # Log output
        removed_edges = np.sum((A != 0) & (out == 0))
        if removed_edges > 0:
            logger.debug(f"Post-processing removed {int(removed_edges)} edges (temporal + immutable constraints)")
        
        return out

    def _run_directlingam(self, X, names) -> Dict[str,float]:
        if not GCASTLE_AVAILABLE:
            return {}
        
        # Try from strictest constraints (strict→moderate→minimal)
        for strictness in ["strict", "moderate", "minimal"]:
            try:
                # Create prior knowledge matrix
                prior_knowledge = self._create_prior_knowledge(names, strictness=strictness)
                
                params = CAUSAL_PARAMS["DirectLiNGAM"].copy()
                params["prior_knowledge"] = prior_knowledge
                
                mdl = DirectLiNGAM(**params)
                mdl.learn(X)
                
                # Apply temporal and immutable variable constraints post-hoc
                A = self._filter_temporal_and_immutable(mdl.causal_matrix, names)
                
                edges={}
                for i in range(len(names)):
                    for j in range(len(names)):
                        if i!=j and abs(A[i,j])>1e-3:
                            edges[f"{names[j]}→{names[i]}"]=float(A[i,j])
                
                logger.info(f"DirectLiNGAM succeeded with {strictness} constraints, found {len(edges)} edges")
                return edges
                
            except ValueError as ve:
                if "argmax of an empty sequence" in str(ve):
                    logger.warning(f"DirectLiNGAM failed with {strictness} constraints: {ve}")
                    continue
                else:
                    raise ve
            except Exception as e:
                logger.warning(f"DirectLiNGAM failed with {strictness} constraints: {e}")
                continue
        
        # If failed at all constraint levels, run without constraints
        try:
            logger.info("DirectLiNGAM: trying without constraints")
            params = CAUSAL_PARAMS["DirectLiNGAM"].copy()
            params["prior_knowledge"] = None
            
            mdl = DirectLiNGAM(**params)
            mdl.learn(X)
            
            # Apply temporal and immutable variable constraints post-hoc
            A = self._filter_temporal_and_immutable(mdl.causal_matrix, names)
            
            edges={}
            for i in range(len(names)):
                for j in range(len(names)):
                    if i!=j and abs(A[i,j])>1e-3:
                        edges[f"{names[j]}→{names[i]}"]=float(A[i,j])
            
            logger.info(f"DirectLiNGAM succeeded without constraints, found {len(edges)} edges")
            return edges
            
        except Exception as e:
            logger.warning(f"DirectLiNGAM failed completely: {e}")
            return {}

    def _run_golem(self, X, names) -> Dict[str,float]:
        """Improved version: Support EV→NV two-stage execution + immutable variable constraints"""
        if not GCASTLE_AVAILABLE or not TORCH_AVAILABLE:
            return {}
        
        try:
            X = X.astype(np.float32)
            use_two_stage = CAUSAL_PARAMS.get("GOLEM_CONSTRAINTS", {}).get("use_two_stage", False)
            
            if use_two_stage:
                # Stage 1: Run with EV (equal_variances=True)
                logger.info("GOLEM Stage 1: Running with equal_variances=True")
                params_ev = CAUSAL_PARAMS["GOLEM"].copy()
                params_ev["device_type"] = "gpu" if torch.cuda.is_available() else "cpu"
                params_ev["equal_variances"] = True
                params_ev["lambda_1"] = 2e-2
                params_ev["num_iter"] = min(params_ev["num_iter"], 50000)
                
                B_init_ev = self._create_B_init_for_golem(names)
                params_ev["B_init"] = B_init_ev
                
                golem_ev = GOLEM(**params_ev)
                golem_ev.learn(X)
                
                # Apply constraints to EV result
                ev_result = self._filter_temporal_and_immutable(golem_ev.causal_matrix, names)
                
                # Stage 2: Run NV (equal_variances=False) with EV result as initial value
                logger.info("GOLEM Stage 2: Running with equal_variances=False using EV result as init")
                params_nv = CAUSAL_PARAMS["GOLEM"].copy()
                params_nv["device_type"] = "gpu" if torch.cuda.is_available() else "cpu"
                params_nv["equal_variances"] = False
                params_nv["B_init"] = self._create_B_init_for_golem(names, ev_result=ev_result)
                
                golem_nv = GOLEM(**params_nv)
                golem_nv.learn(X)
                
                # Apply constraints to final result
                A = self._filter_temporal_and_immutable(golem_nv.causal_matrix, names)
                
            else:
                # Single-stage execution (conventional method)
                B_init = self._create_B_init_for_golem(names)
                params = CAUSAL_PARAMS["GOLEM"].copy()
                params["device_type"] = "gpu" if torch.cuda.is_available() else "cpu"
                params["B_init"] = B_init
                
                golem = GOLEM(**params)
                golem.learn(X)
                
                # Apply constraints
                A = self._filter_temporal_and_immutable(golem.causal_matrix, names)
            
            # Extract edges
            edges = {}
            graph_thres = CAUSAL_PARAMS["GOLEM"].get("graph_thres", 0.3)
            for i in range(len(names)):
                for j in range(len(names)):
                    if i != j and abs(A[i,j]) > graph_thres:
                        edges[f"{names[j]}→{names[i]}"] = float(A[i,j])
            
            logger.info(f"GOLEM found {len(edges)} edges {'(two-stage)' if use_two_stage else ''}")
            return edges
            
        except Exception as e:
            logger.warning(f"GOLEM failed: {e}")
            return {}

    def _run_corl(self, X, names) -> Dict[str,float]:
        """Improved version: Use dag_mask + immutable variable constraints"""
        if not GCASTLE_AVAILABLE or not TORCH_AVAILABLE:
            return {}
        
        try:
            dtype = np.float64 if CAUSAL_PARAMS["CORL"]["use_float64"] else np.float32
            X = np.ascontiguousarray(X, dtype=dtype)
            X = X + np.random.randn(*X.shape).astype(dtype) * 1e-5
            var = np.var(X, axis=0)
            valid_cols = var > 1e-10
            if not np.all(valid_cols):
                logger.warning(f"CORL: Removing {np.sum(~valid_cols)} low-variance columns")
                X = X[:, valid_cols]
                names = [n for i, n in enumerate(names) if valid_cols[i]]
                if X.shape[1] < 3:
                    logger.warning("CORL: Too few variables after filtering")
                    return {}
            
            params = CAUSAL_PARAMS["CORL"].copy()
            params.pop("use_float64", None)
            params.pop("use_dag_mask", None)
            params.pop("edge_threshold", None)
            params["device_type"] = "gpu" if torch.cuda.is_available() else "cpu"
            
            n_samples, n_features = X.shape
            if n_samples < params["batch_size"] * 2:
                params["batch_size"] = max(8, n_samples // 4)
            
            # Create and use dag_mask
            use_dag_mask = CAUSAL_PARAMS["CORL"].get("use_dag_mask", False)
            if use_dag_mask:
                dag_mask = self._create_dag_mask_for_corl(names)
                dag_mask = dag_mask.astype(dtype)
                corl = CORL(**params)
                corl.learn(X, dag_mask=dag_mask)
                logger.info("CORL: Using dag_mask for constraints")
            else:
                corl = CORL(**params)
                corl.learn(X)
            
            # Apply constraints
            A = self._filter_temporal_and_immutable(corl.causal_matrix, names)
            
            # Edge extraction
            edge_threshold = CAUSAL_PARAMS["CORL"].get("edge_threshold", 1e-3)
            edges = {}
            for i in range(len(names)):
                for j in range(len(names)):
                    if i != j and abs(A[i,j]) > edge_threshold:
                        edges[f"{names[j]}→{names[i]}"] = float(A[i,j])
            
            logger.info(f"CORL found {len(edges)} edges {'(with dag_mask)' if use_dag_mask else ''} (threshold={edge_threshold})")
            return edges
            
        except Exception as e:
            logger.warning(f"CORL failed: {e}")
            return {}

    def _run_dagma(self, X, names) -> Dict[str,float]:
        """DAGMA execution + immutable variable constraints"""
        if not DAGMA_AVAILABLE or not TORCH_AVAILABLE:
            return {}
        try:
            n = X.shape[1]
            params = CAUSAL_PARAMS["DAGMA"].copy()
            params["dims"][0] = n
            eq_model = DagmaMLP(dims=params["dims"], bias=params["bias"])
            mdl = DagmaNonlinear(model=eq_model)
            W = mdl.fit(X.astype(np.float64),
                      lambda1=params["lambda1"], lambda2=params["lambda2"],
                      lr=params["lr"], w_threshold=params["w_threshold"],
                      T=params["T"], mu_init=params["mu_init"],
                      mu_factor=params["mu_factor"], s=params["s"],
                      warm_iter=params["warm_iter"], max_iter=params["max_iter"],
                      checkpoint=params["checkpoint"])
            
            # Apply constraints
            A = self._filter_temporal_and_immutable(W.T, names)
            
            # Apply w_threshold
            w_threshold = params.get("w_threshold", 0.3)
            edges = {}
            for i in range(len(names)):
                for j in range(len(names)):
                    if i != j and abs(A[i,j]) > w_threshold:
                        edges[f"{names[j]}→{names[i]}"] = float(A[i,j])
            
            logger.info(f"DAGMA found {len(edges)} edges (threshold={w_threshold})")
            return edges
        except Exception as e:
            logger.warning(f"DAGMA failed: {e}")
            return {}

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        
        # Save variable list used in each subset (added)
        subset_variables = {}
        
        for subset in ["disease","symptoms"]:
            X, names = self._prepare(df, subset=subset)
            if X is None:
                logger.info(f"[{subset}] no data; skip")
                continue
            
            # Record variables used in this subset (added)
            subset_variables[subset] = names
            
            logger.info(f"[{subset}] Variables: {len(names)}, Samples: {X.shape[0]}")
            logger.info(f"[{subset}] Creating prior knowledge constraints...")
            
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            folds = list(kf.split(X))
            methods = [("DirectLiNGAM", self._run_directlingam),
                      ("GOLEM", self._run_golem),
                      ("CORL", self._run_corl),
                      ("DAGMA", self._run_dagma)]
            for mname, runner in methods:
                edges_by_fold=[]
                for fi,(tr,_te) in enumerate(folds,1):
                    ed = runner(X[tr], names)
                    edges_by_fold.append(ed)
                    if ed:
                        logger.info(f"[{mname}][{subset}] fold {fi}/{self.n_folds}: {len(ed)} edges")
                sig = self._cv_edges(edges_by_fold)
                for k,v in sig.items():
                    results.append({"Path":k,"Method":mname,"Subset":subset,
                                  "FoldCount":v["folds"],"Avg_Coefficient":v["coef"]})
        
        if not results:
            return pd.DataFrame()
        
        dfres = pd.DataFrame(results)
        
        # Add variable list for each subset (as metadata)
        dfres.attrs['subset_variables'] = subset_variables
        
        return dfres

# ====== 13) Absolute Risk (Quartiles) ======
def absolute_risk_by_quartile(df: pd.DataFrame, outcome: str, weight_col: str,
                              qcol: str="HI_quartile_22") -> pd.DataFrame:
    g = df.dropna(subset=[outcome, qcol, weight_col]).copy()
    g[qcol] = g[qcol].astype(int)
    rows = []
    for q in [1,2,3,4]:
        sub = g[g[qcol]==q]
        w = sub[weight_col].values
        y = sub[outcome].values
        n = sub.shape[0]
        risk = np.average(y, weights=w) if n > 0 else np.nan
        rows.append({"quartile": q, "n": n, "risk_pct": risk*100.0,
                    "events_unweighted": int(np.nansum(y)),
                    "sum_weights": float(np.sum(w))})
    return pd.DataFrame(rows)

# ====== 14) Forest Plot ======
def plot_forest(results_df: pd.DataFrame, out_path: str):
    df = results_df.copy()
    df = df.sort_values(by="RR", ascending=True)

    ylabels = df["Outcome"].tolist()
    rr = df["RR"].values
    lcl = df["LCL"].values
    ucl = df["UCL"].values
    sig = df["FDR_q"] < 0.05

    label_map = {
        "HospitalizationPastYear_23": "Hospitalization",
        "COVID19_Vaccination_23": "COVID-19 Vaccination",
        "InfluenzaVaccination_23": "Influenza Vaccination",
        "COVID19_Infection_23": "COVID-19 Infection",
        "Hypertension_23": "Hypertension",
        "Diabetes_23": "Diabetes",
        "Dyslipidemia_23": "Dyslipidemia",
        "Asthma_23": "Asthma",
        "Periodontitis_23": "Periodontitis",
        "DentalCaries_23": "Dental Caries",
        "AnginaMI_23": "Angina/MI",
        "Stroke_23": "Stroke",
        "COPD_23": "COPD",
        "CKD_23": "CKD",
        "ChronicLiverDisease_23": "Chronic Liver Disease",
        "Cancer_23": "Cancer",
        "ChronicPain_23": "Chronic Pain",
        "Depression_23": "Depression",
        "GIDiscomfort_23": "GI Discomfort",
        "BackPain_23": "Back Pain",
        "JointPain_23": "Joint Pain",
        "Headache_23": "Headache",
        "ChestPain_23": "Chest Pain",
        "Dyspnea_23": "Dyspnea",
        "Dizziness_23": "Dizziness",
        "SleepDisturbance_23": "Sleep Disturbance",
        "MemoryDisorder_23": "Memory Disorder",
        "ConcentrationDecline_23": "Concentration Decline",
        "ReducedLibido_23": "Reduced Libido",
        "Fatigue_23": "Fatigue",
        "Cough_23": "Cough",
        "Fever_23": "Fever"
    }
    ylabels = [label_map.get(label, label) for label in ylabels]
    y = np.arange(len(df))

    plt.figure(figsize=(12, max(8, 0.35*len(df))))
    # CUD color scheme: significant is blue, non-significant is gray
    plt.hlines(y, lcl, ucl, colors=np.where(sig, "#0072B2", "#999999"), linewidth=2)
    colors = np.where(sig, "#0072B2", "#999999")
    plt.scatter(rr, y, s=60, c=colors, zorder=5)

    plt.axvline(1.0, linestyle="--", color="#D55E00", alpha=0.5, linewidth=1)
    plt.yticks(y, ylabels)
    plt.xscale("log")

    min_val = min(lcl.min(), 0.7)
    max_val = max(ucl.max(), 1.5)
    major_ticks = []
    tick_val = 0.7
    while tick_val <= max_val:
        major_ticks.append(tick_val)
        tick_val += 0.1
    ax = plt.gca()
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([f"{x:.1f}" for x in major_ticks])
    ax.grid(True, axis='x', which='major', alpha=0.3, linestyle='-', linewidth=0.5)

    plt.xlabel("Adjusted Risk Ratio (95% CI)", fontsize=12)
    plt.title("Health Indifference and Health Outcomes\n(per 1 SD increase in HI score)", fontsize=14)

    for i, (r, l, u) in enumerate(zip(rr, lcl, ucl)):
        text = f"{r:.2f} ({l:.2f}-{u:.2f})"
        if sig[i]:
            ax.text(max_val*1.02, i, text, va='center', fontweight='bold', fontsize=9)
        else:
            ax.text(max_val*1.02, i, text, va='center', fontsize=9, color='#666666')

    ax.set_xlim(min_val, max_val)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved forest plot to {out_path}")

# ====== 15) Spline (Revised Version) ======
def plot_spline_hospitalization(panel: pd.DataFrame, wcol: str, out_path: str,
                                df_spline: int = 3, x_min: int = 13, x_max: int = 52, x_ref: int = 29):
    dat = panel.dropna(subset=["HospitalizationPastYear_23","HI_sum_22", wcol] + COVARS).copy()
    if dat.shape[0] < 100:
        logger.warning("Not enough observations for spline analysis; skipped")
        return
    try:
        dat["HI_sum_22_clip"] = dat["HI_sum_22"].clip(lower=x_min, upper=x_max)
        bs_terms = dmatrix(f"bs(HI_sum_22_clip, df={df_spline}, lower_bound={x_min}, upper_bound={x_max}, include_intercept=False)",
                           {"HI_sum_22_clip": dat["HI_sum_22_clip"]}, return_type='dataframe')
        X_df = pd.concat([bs_terms.reset_index(drop=True),
                         dat[COVARS].reset_index(drop=True)], axis=1)
        X_with_const = sm.add_constant(X_df, has_constant='skip')
        y = dat["HospitalizationPastYear_23"].values
        w = dat[wcol].values
        model = sm.GLM(y, X_with_const, family=sm.families.Poisson(), var_weights=w)
        res = model.fit(cov_type="HC0")

        ref_df = pd.DataFrame({"HI_sum_22_clip": [x_ref]})
        Xref_bs = dmatrix(f"bs(HI_sum_22_clip, df={df_spline}, lower_bound={x_min}, upper_bound={x_max}, include_intercept=False)",
                          ref_df, return_type="dataframe")
        cov_means = dat[COVARS].mean().to_frame().T.reset_index(drop=True)
        Xref_df = pd.concat([Xref_bs.reset_index(drop=True), cov_means], axis=1)
        Xref = sm.add_constant(Xref_df, has_constant='skip').values

        xs = np.arange(x_min, x_max+1)
        Xs_list = []
        for xv in xs:
            xi_bs = dmatrix(f"bs(HI_sum_22_clip, df={df_spline}, lower_bound={x_min}, upper_bound={x_max}, include_intercept=False)",
                            {"HI_sum_22_clip": [xv]}, return_type="dataframe")
            Xi_df = pd.concat([xi_bs.reset_index(drop=True), cov_means], axis=1)
            Xi = sm.add_constant(Xi_df, has_constant='skip').values
            Xs_list.append(Xi[0,:])
        Xs = np.vstack(Xs_list)

        b = res.params.values
        V = res.cov_params().values
        diff = Xs - Xref
        logRR = diff @ b
        se = np.sqrt(np.einsum("ij,jk,ik->i", diff, V, diff))
        RR = np.exp(logRR)
        LCL = np.exp(logRR - 1.96*se)
        UCL = np.exp(logRR + 1.96*se)

        plt.figure(figsize=(6,4))
        plt.plot(xs, RR, label="aRR (vs median HI=29)", linewidth=2, color="#0072B2")
        plt.fill_between(xs, LCL, UCL, alpha=0.2, color="#0072B2")
        plt.axhline(1.0, linestyle="--", alpha=0.3, linewidth=1, label="aRR = 1.0", color="#D55E00")
        plt.axvline(x_ref, linestyle="--", alpha=0.3, label="Reference (Median HI = 29)", color="#009E73")
        plt.xlabel("Health Indifference score (13–52)")
        plt.ylabel("Adjusted Risk Ratio (aRR)")
        plt.title("Spline: Hospitalization vs Health Indifference")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Saved spline plot to {out_path}")

    except Exception as e:
        logger.warning(f"Spline analysis failed: {e}")
        try:
            quartiles = pd.qcut(dat["HI_sum_22"], q=4, labels=[1,2,3,4], duplicates='drop')
            q_means, q_risks = [], []
            for q in [1,2,3,4]:
                sub = dat[quartiles==q]
                if len(sub) > 0:
                    q_means.append(sub["HI_sum_22"].mean())
                    q_risks.append(sub["HospitalizationPastYear_23"].mean())
            if q_means:
                plt.figure(figsize=(6,4))
                plt.plot(q_means, q_risks, 'o-', markersize=8, linewidth=2, color="#0072B2")
                plt.axhline(np.mean(q_risks), linestyle="--", alpha=0.5, label="Mean risk", color="#D55E00")
                plt.xlabel("Health Indifference score (mean by quartile)")
                plt.ylabel("Hospitalization rate (%)")
                plt.title("Hospitalization vs Health Indifference (simplified)")
                plt.legend()
                plt.grid(True, alpha=0.2)
                plt.tight_layout()
                plt.savefig(out_path, dpi=300)
                plt.close()
                logger.info(f"Saved simplified plot to {out_path}")
        except Exception as e2:
            logger.warning(f"Even simplified plot failed: {e2}")

# ====== 16) Network Visualization ======
def plot_causal_network(causal_df: pd.DataFrame, out_path: str, subset_name: str = None, all_variables: List[str] = None):
    """
    Visualize causal network (concise version: only 3 temporal groups)
    """
    # Continue processing even with empty DataFrame if all_variables is provided
    if causal_df.empty and not all_variables:
        logger.warning("No causal edges to plot and no variables provided")
        return
    
    # Filter by subset
    if subset_name and not causal_df.empty:
        causal_df = causal_df[causal_df["Subset"] == subset_name].copy()
    
    # Initialize graph
    G = nx.DiGraph()
    
    # Add all variables as nodes
    if all_variables:
        for var in all_variables:
            G.add_node(var)
        logger.info(f"Added {len(all_variables)} nodes to graph for subset {subset_name}")
    
    # Add edges
    edge_methods = {}
    if not causal_df.empty:
        for _, row in causal_df.iterrows():
            path = row["Path"]
            method = row["Method"]
            if "→" in path:
                src, tgt = path.split("→")
                src, tgt = src.strip(), tgt.strip()
                
                if src not in G.nodes():
                    G.add_node(src)
                if tgt not in G.nodes():
                    G.add_node(tgt)
                
                edge_key = (src, tgt)
                edge_methods.setdefault(edge_key, []).append(method)
        
        for edge_key, methods in edge_methods.items():
            G.add_edge(edge_key[0], edge_key[1],
                       methods=methods, n_methods=len(set(methods)))
    
    if G.number_of_nodes() == 0:
        logger.warning("No nodes in causal graph")
        return
    
    logger.info(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Identify isolated nodes
    isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0]
    connected_nodes = [node for node in G.nodes() if G.degree(node) > 0]
    
    # Variable name mapping (concise version)
    name_map = {
        # 2021 Baseline - Demographics
        "male": "Male",
        "age": "Age",
        "low_edu": "Low education",
        "low_income": "Low income",
        "living_alone": "Live alone",
        # 2021 Baseline - Social/Behavioral
        "lack_social_support": "No social support",
        "current_smoking": "Smoking",
        "heavy_drinking": "Heavy Drink",
        "low_physical_activity": "Low physical activity",
        "no_health_checkup": "No health checkup",
        # 2021 Baseline - Health Status
        "poor_self_rated_health": "Poor self rated health",
        "bmi": "BMI",
        # 2022 Exposure
        "HI_sum_22": "Health indifference",
        "High_HI_22": "High health indifference",
        # 2023 Disease Outcomes  
        "HospitalizationPastYear_23": "Hospitalization",
        "COVID19_Infection_23": "COVID Infection",
        "COVID19_Vaccination_23": "COVID Vaccination",
        "InfluenzaVaccination_23": "Flu Vaccination",
        "Hypertension_23": "Hypertension",
        "Diabetes_23": "Diabetes mellitus",
        "Dyslipidemia_23": "Dyslipidemia",
        "Asthma_23": "Asthma",
        "Periodontitis_23": "Periodontitis",
        "DentalCaries_23": "Dental caries",
        "AnginaMI_23": "Angina/Myocardial infarction",
        "Stroke_23": "Stroke",
        "COPD_23": "COPD",
        "CKD_23": "Chronic kidney disease",
        "ChronicLiverDisease_23": "Liver Disease",
        "Cancer_23": "Cancer",
        "ChronicPain_23": "Chronic pain",
        "Depression_23": "Depression",
        # 2023 Symptom Outcomes
        "GIDiscomfort_23": "GI Discomfort",
        "BackPain_23": "Back Pain",
        "JointPain_23": "Joint Pain",
        "Headache_23": "Headache",
        "ChestPain_23": "Chest Pain",
        "Dyspnea_23": "Dyspnea",
        "Dizziness_23": "Dizziness",
        "SleepDisturbance_23": "Sleep disturbance",
        "MemoryDisorder_23": "Memory disorder",
        "ConcentrationDecline_23": "Concentration decline",
        "ReducedLibido_23": "Low Libido",
        "Fatigue_23": "Fatigue",
        "Cough_23": "Cough",
        "Fever_23": "Fever"
    }

    # Classify nodes by year (simple version)
    nodes_2021 = []
    nodes_2022 = []
    nodes_2023 = []
    
    for node in G.nodes():
        if "_23" in node:
            nodes_2023.append(node)
        elif "HI_" in node or "High_HI" in node:
            nodes_2022.append(node)
        else:
            nodes_2021.append(node)

    # Layout calculation (simple version)
    pos = {}
    spacing = 1.5  # Node spacing
    
    # 2021 variable placement
    for i, node in enumerate(sorted(nodes_2021)):  # Sort alphabetically
        pos[node] = (0, i * spacing)
    
    # 2022 (HI) placement (centered)
    y_center = (len(nodes_2021) * spacing) / 2 if nodes_2021 else 0
    for i, node in enumerate(sorted(nodes_2022)):
        pos[node] = (5, y_center + i * spacing)
    
    # 2023 variable placement
    for i, node in enumerate(sorted(nodes_2023)):
        pos[node] = (10, i * spacing)

    # Dynamically adjust figure size
    n_max = max(len(nodes_2021), len(nodes_2022), len(nodes_2023))
    fig_height = max(12, n_max * 0.8)
    fig_width = 16
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.axes([0.05, 0.05, 0.75, 0.9])

    # Node color settings (by year only)
    node_colors = []
    for node in G.nodes():
        if node in nodes_2021:
            node_colors.append("#E8F4FD")  # Light blue
        elif node in nodes_2022:
            node_colors.append("#FFF4E6")  # Orange
        else:  # 2023
            node_colors.append("#E8F6F3")  # Mint green

    # Edge color map
    method_colors = {
        4: "#000000", 3: "#D55E00", 2: "#E69F00",
        "DirectLiNGAM": "#0072B2", "GOLEM": "#009E73",
        "DAGMA": "#CC79A7", "CORL": "#56B4E9"
    }

    # Draw edges
    for u, v in G.edges():
        data = G[u][v]
        n_methods = data['n_methods']
        methods = data['methods']
        is_hi_related = ("HI_" in u or "High_HI" in u) or ("HI_" in v or "High_HI" in v)

        if n_methods >= 2:
            edge_color = method_colors.get(min(n_methods, 4), "#666666")
        else:
            edge_color = method_colors.get(methods[0] if methods else "unknown", "#666666")

        linestyle = '-' if is_hi_related else '--'
        linewidth = 2.5 if is_hi_related else 1.0
        alpha = 0.7 if is_hi_related else 0.5

        nx.draw_networkx_edges(G, pos, [(u, v)],
                               edge_color=edge_color, style=linestyle,
                               width=linewidth, alpha=alpha,
                               arrows=True, arrowsize=12,
                               connectionstyle="arc3,rad=0.1",
                               min_target_margin=15, ax=ax)

    # Set node size
    node_size = 1000 if n_max > 20 else 1200
    
    # Draw nodes
    # Connected nodes
    if connected_nodes:
        connected_colors = [node_colors[list(G.nodes()).index(n)] for n in connected_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=connected_nodes,
                               node_color=connected_colors,
                               node_size=node_size, 
                               edgecolors='#333333', 
                               linewidths=1.2, ax=ax)
    
    # Isolated nodes
    if isolated_nodes:
        isolated_colors = [node_colors[list(G.nodes()).index(n)] for n in isolated_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=isolated_nodes,
                               node_color=isolated_colors,
                               node_size=node_size, 
                               edgecolors='#999999', 
                               linewidths=0.8, 
                               alpha=0.4, ax=ax)

    # Draw labels
    labels = {node: name_map.get(node, node.replace("_23", "").replace("_22", "")) 
              for node in G.nodes()}
    
    font_size = 7 if n_max > 20 else 8
    
    # Connected node labels
    if connected_nodes:
        connected_labels = {node: labels[node] for node in connected_nodes}
        nx.draw_networkx_labels(G, pos, connected_labels, 
                                font_size=font_size, 
                                font_weight='bold', ax=ax)
    
    # Isolated node labels
    if isolated_nodes:
        for node in isolated_nodes:
            x, y = pos[node]
            ax.text(x, y, labels[node], 
                   fontsize=font_size-1, 
                   ha='center', va='center',
                   alpha=0.5, fontweight='normal')

    # Time series labels
    ax.text(0, -2, "2021 Baseline", fontsize=12, fontweight='bold', ha='center')
    ax.text(5, -2, "2022 Exposure", fontsize=12, fontweight='bold', ha='center', color='#D55E00')
    ax.text(10, -2, "2023 Outcomes", fontsize=12, fontweight='bold', ha='center')

    # Vertical lines (time series dividers)
    ax.axvline(x=2.5, color='#999999', linestyle='--', alpha=0.3)
    ax.axvline(x=7.5, color='#999999', linestyle='--', alpha=0.3)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color=method_colors[4], lw=2.5, label='4 methods'),
        plt.Line2D([0], [0], color=method_colors[3], lw=2.5, label='3 methods'),
        plt.Line2D([0], [0], color=method_colors[2], lw=2.5, label='2 methods'),
        plt.Line2D([0], [0], color=method_colors["DirectLiNGAM"], lw=2.5, label='DirectLiNGAM'),
        plt.Line2D([0], [0], color=method_colors["GOLEM"], lw=2.5, label='GOLEM'),
        plt.Line2D([0], [0], color=method_colors["DAGMA"], lw=2.5, label='DAGMA'),
        plt.Line2D([0], [0], color=method_colors["CORL"], lw=2.5, label='CORL'),
        plt.Line2D([0], [0], color='#333333', lw=2.5, linestyle='-', label='HI-related'),
        plt.Line2D([0], [0], color='#333333', lw=1, linestyle='--', label='Non-HI'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#999999', 
                   markersize=8, alpha=0.4, label='Isolated')
    ]
    
    ax.legend(handles=legend_elements, loc='center left', 
              bbox_to_anchor=(1.02, 0.5), fontsize=9,
              frameon=True, fancybox=True, shadow=True,
              title="Edge Types", title_fontsize=10)

    # Title
    if subset_name == "disease":
        title = "Figure 3. Causal Network: Disease and Vaccination Outcomes"
    elif subset_name == "symptoms":
        title = "Supplementary Figure 2. Causal Network: Symptom Outcomes"
    else:
        title = "Causal Network: Temporal Structure"
    
    subtitle = f"({G.number_of_nodes()} nodes: {len(connected_nodes)} connected, {len(isolated_nodes)} isolated | {G.number_of_edges()} edges)"
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.text(5, max([p[1] for p in pos.values()]) + 2 if pos else 0, subtitle, 
            fontsize=10, ha='center', style='italic')
    
    ax.axis('off')
    ax.set_xlim(-2, 12)
    if pos:
        y_min = min([p[1] for p in pos.values()]) - 3
        y_max = max([p[1] for p in pos.values()]) + 3
    else:
        y_min, y_max = -3, 3
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved causal network ({subset_name}) to {out_path}")

# ====== 17) Table 1 (Complete version with all variables) ======
def summarize_table1(panel: pd.DataFrame, wcol: str) -> pd.DataFrame:
    """
    Create Table 1 with All participants, Low HI, and High HI groups
    Maintains compatibility with original code structure
    """
    tbl = []
    
    def add_row(variable, all_value, low_value, high_value, smd, pvalue):
        """Add a row to the table with all groups"""
        tbl.append({
            "Variable": variable,
            "All participants": all_value,
            "Low HI": low_value,
            "High HI": high_value,
            "SMD": smd,
            "p-value": pvalue
        })
    
    # Create HI_category (split by median)
    if "High_HI_22" not in panel.columns:
        # Split by median of HI_sum_22
        if "HI_sum_22" in panel.columns:
            median_hi = panel["HI_sum_22"].median()
            panel["HI_category"] = pd.cut(panel["HI_sum_22"], 
                                         bins=[-np.inf, median_hi, np.inf],
                                         labels=["Low HI", "High HI"])
            logger.info(f"Created HI_category using median HI score: {median_hi:.1f}")
        else:
            logger.error("HI_sum_22 not found, cannot create HI_category")
            return pd.DataFrame(tbl)
    else:
        # Create from High_HI_22
        panel["HI_category"] = panel["High_HI_22"].apply(
            lambda x: "High HI" if x == 1 else "Low HI"
        )
    
    # Split into groups
    d = panel.copy()
    d_all = d.dropna(subset=[wcol]).copy()
    d_low = d[d["HI_category"] == "Low HI"].dropna(subset=[wcol]).copy()
    d_high = d[d["HI_category"] == "High HI"].dropna(subset=[wcol]).copy()
    
    # Sample sizes
    n_all = len(d_all)
    n_low = len(d_low)
    n_high = len(d_high)
    
    # Add header row with sample sizes
    add_row(
        "Characteristic",
        f"(n={n_all:,})",
        f"(n={n_low:,})",
        f"(n={n_high:,})",
        "",
        ""
    )
    
    # Define all variables with their labels and types
    variables = [
        # 2022 Health Indifference
        ("HI_sum_22", "HI score (2022), mean±SD", "continuous"),
        
        # 2021 Baseline Demographics
        ("age", "Age (2021), years, mean±SD", "continuous"),
        ("male", "Male sex, n (%)", "binary"),
        ("low_edu", "Low education (<HS), n (%)", "binary"),
        ("low_income", "Low household income, n (%)", "binary"),
        ("living_alone", "Living alone, n (%)", "binary"),
        ("lack_social_support", "Lack of social support, n (%)", "binary"),
        
        # 2021 Health Behaviors
        ("current_smoking", "Current smoking, n (%)", "binary"),
        ("heavy_drinking", "Heavy drinking, n (%)", "binary"),
        ("low_physical_activity", "Low physical activity, n (%)", "binary"),
        ("no_health_checkup", "No health checkup, n (%)", "binary"),
        
        # 2021 Health Status
        ("poor_self_rated_health", "Poor self-rated health, n (%)", "binary"),
        ("bmi", "BMI (2021), kg/m², mean±SD", "continuous"),
        
        # 2023 Hospitalization and Vaccination
        ("HospitalizationPastYear_23", "Hospitalization past year (2023), n (%)", "binary"),
        ("COVID19_Infection_23", "COVID-19 infection (2023), n (%)", "binary"),
        ("COVID19_Vaccination_23", "COVID-19 vaccination (2023), n (%)", "binary"),
        ("InfluenzaVaccination_23", "Influenza vaccination (2023), n (%)", "binary"),
        
        # 2023 Chronic Diseases
        ("Hypertension_23", "Hypertension (2023), n (%)", "binary"),
        ("Diabetes_23", "Diabetes mellitus (2023), n (%)", "binary"),
        ("Dyslipidemia_23", "Dyslipidemia (2023), n (%)", "binary"),
        ("Asthma_23", "Asthma (2023), n (%)", "binary"),
        ("Periodontitis_23", "Periodontitis (2023), n (%)", "binary"),
        ("DentalCaries_23", "Dental caries (2023), n (%)", "binary"),
        ("AnginaMI_23", "Angina/MI (2023), n (%)", "binary"),
        ("Stroke_23", "Stroke (2023), n (%)", "binary"),
        ("COPD_23", "COPD (2023), n (%)", "binary"),
        ("CKD_23", "Chronic kidney disease (2023), n (%)", "binary"),
        ("ChronicLiverDisease_23", "Chronic liver disease (2023), n (%)", "binary"),
        ("Cancer_23", "Cancer (2023), n (%)", "binary"),
        ("ChronicPain_23", "Chronic pain (2023), n (%)", "binary"),
        ("Depression_23", "Depression (2023), n (%)", "binary"),
        
        # 2023 Symptoms
        ("GIDiscomfort_23", "GI discomfort (2023), n (%)", "binary"),
        ("BackPain_23", "Back pain (2023), n (%)", "binary"),
        ("JointPain_23", "Joint pain (2023), n (%)", "binary"),
        ("Headache_23", "Headache (2023), n (%)", "binary"),
        ("ChestPain_23", "Chest pain (2023), n (%)", "binary"),
        ("Dyspnea_23", "Dyspnea (2023), n (%)", "binary"),
        ("Dizziness_23", "Dizziness (2023), n (%)", "binary"),
        ("SleepDisturbance_23", "Sleep disturbance (2023), n (%)", "binary"),
        ("MemoryDisorder_23", "Memory disorder (2023), n (%)", "binary"),
        ("ConcentrationDecline_23", "Concentration decline (2023), n (%)", "binary"),
        ("ReducedLibido_23", "Reduced libido (2023), n (%)", "binary"),
        ("Fatigue_23", "Fatigue (2023), n (%)", "binary"),
        ("Cough_23", "Cough (2023), n (%)", "binary"),
        ("Fever_23", "Fever (2023), n (%)", "binary"),
    ]
    
    # Process each variable
    for var, label, var_type in variables:
        if var not in panel.columns:
            # Variable not found, add NA row
            add_row(label, "NA", "NA", "NA", "NA", "NA")
            continue
        
        if var_type == "continuous":
            # Process continuous variables
            
            # All participants
            sub_all = d_all.dropna(subset=[var])
            if len(sub_all) > 0:
                mu_all = np.average(sub_all[var], weights=sub_all[wcol])
                var_all = np.average((sub_all[var] - mu_all)**2, weights=sub_all[wcol])
                sd_all = np.sqrt(var_all)
                all_value = f"{mu_all:.1f} ± {sd_all:.1f}"
            else:
                all_value = "NA"
            
            # Low HI
            sub_low = d_low.dropna(subset=[var])
            if len(sub_low) > 0:
                mu_low = np.average(sub_low[var], weights=sub_low[wcol])
                var_low = np.average((sub_low[var] - mu_low)**2, weights=sub_low[wcol])
                sd_low = np.sqrt(var_low)
                low_value = f"{mu_low:.1f} ± {sd_low:.1f}"
            else:
                low_value = "NA"
            
            # High HI
            sub_high = d_high.dropna(subset=[var])
            if len(sub_high) > 0:
                mu_high = np.average(sub_high[var], weights=sub_high[wcol])
                var_high = np.average((sub_high[var] - mu_high)**2, weights=sub_high[wcol])
                sd_high = np.sqrt(var_high)
                high_value = f"{mu_high:.1f} ± {sd_high:.1f}"
            else:
                high_value = "NA"
            
            # Calculate p-value (t-test)
            if len(sub_low) > 0 and len(sub_high) > 0:
                from scipy import stats
                _, p_value = stats.ttest_ind(sub_low[var], sub_high[var])
            else:
                p_value = np.nan
            
            # Calculate SMD
            if len(sub_low) > 0 and len(sub_high) > 0:
                pooled_sd = np.sqrt(((len(sub_low)-1)*var_low + (len(sub_high)-1)*var_high) / 
                                   (len(sub_low) + len(sub_high) - 2))
                if pooled_sd > 0:
                    smd = abs(mu_high - mu_low) / pooled_sd
                else:
                    smd = 0
            else:
                smd = np.nan
            
        else:  # binary
            # Process binary variables
            
            # All participants
            sub_all = d_all.dropna(subset=[var])
            if len(sub_all) > 0:
                p_all = np.average(sub_all[var], weights=sub_all[wcol])
                n_all_events = int(sub_all[var].sum())
                all_value = f"{n_all_events:,} ({p_all*100:.1f})"
            else:
                all_value = "NA"
            
            # Low HI
            sub_low = d_low.dropna(subset=[var])
            if len(sub_low) > 0:
                p_low = np.average(sub_low[var], weights=sub_low[wcol])
                n_low_events = int(sub_low[var].sum())
                low_value = f"{n_low_events:,} ({p_low*100:.1f})"
            else:
                low_value = "NA"
                p_low = None
            
            # High HI
            sub_high = d_high.dropna(subset=[var])
            if len(sub_high) > 0:
                p_high = np.average(sub_high[var], weights=sub_high[wcol])
                n_high_events = int(sub_high[var].sum())
                high_value = f"{n_high_events:,} ({p_high*100:.1f})"
            else:
                high_value = "NA"
                p_high = None
            
            # Calculate p-value (chi-square or Fisher's exact test)
            if len(sub_low) > 0 and len(sub_high) > 0:
                n_low_yes = int(sub_low[var].sum())
                n_low_no = len(sub_low) - n_low_yes
                n_high_yes = int(sub_high[var].sum())
                n_high_no = len(sub_high) - n_high_yes
                
                contingency = np.array([[n_low_yes, n_low_no],
                                       [n_high_yes, n_high_no]])
                
                # Use Fisher's exact test for small samples
                if np.min(contingency) < 5:
                    from scipy.stats import fisher_exact
                    _, p_value = fisher_exact(contingency)
                else:
                    from scipy.stats import chi2_contingency
                    _, p_value, _, _ = chi2_contingency(contingency)
            else:
                p_value = np.nan
            
            # Calculate SMD for binary variables
            if p_low is not None and p_high is not None:
                pooled_p = (p_low + p_high) / 2
                if pooled_p > 0 and pooled_p < 1:
                    smd = abs(p_high - p_low) / np.sqrt(pooled_p * (1 - pooled_p))
                else:
                    smd = 0
            else:
                smd = np.nan
        
        # Format p-value
        if np.isnan(p_value):
            p_str = "NA"
        elif p_value < 0.001:
            p_str = "<0.001"
        else:
            p_str = f"{p_value:.3f}"
        
        # Format SMD
        if np.isnan(smd):
            smd_str = "NA"
        else:
            smd_str = f"{smd:.3f}"
        
        # Add row to table
        add_row(label, all_value, low_value, high_value, smd_str, p_str)
    
    # Log summary statistics
    logger.info(f"Table 1 created with {len(tbl)-1} variables (excluding header)")
    
    return pd.DataFrame(tbl)

# ====== 18) Sensitivity Analysis (Quartiles) ======
def sensitivity_quartile_analysis(mi_list: List[pd.DataFrame], outcome: str, weights: str, adjust: List[str]):
    results = []
    for q in [2, 3, 4]:
        betas, ses2 = [], []
        for df in mi_list:
            df[f"Q{q}_vs_Q1"] = (df["HI_quartile_22"] == q).astype(float)
            rhs = " + ".join([f"Q{q}_vs_Q1"] + adjust)
            formula = f"{outcome} ~ {rhs}"
            try:
                model = smf.glm(formula=formula, data=df, family=sm.families.Poisson(),
                                var_weights=df[weights])
                res = model.fit(cov_type="HC0")
                b = res.params[f"Q{q}_vs_Q1"]
                se = res.bse[f"Q{q}_vs_Q1"]
                betas.append(b)
                ses2.append(se**2)
            except:
                continue
        if betas:
            m = len(betas)
            b_bar = np.mean(betas)
            W = np.mean(ses2)
            B = np.var(betas, ddof=1) if m > 1 else 0.0
            T = W + (1 + 1/m) * B
            se_bar = math.sqrt(T)
            rr = math.exp(b_bar)
            lcl = math.exp(b_bar - 1.96*se_bar)
            ucl = math.exp(b_bar + 1.96*se_bar)
            z = b_bar / se_bar if se_bar > 0 else np.inf
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            results.append({
                "Comparison": f"Q{q} vs Q1",
                "RR": rr,
                "LCL": lcl,
                "UCL": ucl,
                "p": p
            })
    return pd.DataFrame(results)

# ====== 19) Sensitivity Analysis (Interaction) ======
def sensitivity_interaction_analysis(mi_list: List[pd.DataFrame], outcome: str, weights: str,
                                    exposure: str, modifier: str, adjust: List[str]):
    betas_int, ses2_int = [], []
    for df in mi_list:
        df[f"{exposure}_X_{modifier}"] = df[exposure] * df[modifier]
        rhs = " + ".join([exposure, modifier, f"{exposure}_X_{modifier}"] +
                         [a for a in adjust if a != modifier])
        formula = f"{outcome} ~ {rhs}"
        try:
            model = smf.glm(formula=formula, data=df, family=sm.families.Poisson(),
                            var_weights=df[weights])
            res = model.fit(cov_type="HC0")
            b = res.params[f"{exposure}_X_{modifier}"]
            se = res.bse[f"{exposure}_X_{modifier}"]
            betas_int.append(b)
            ses2_int.append(se**2)
        except:
            continue
    if betas_int:
        m = len(betas_int)
        b_bar = np.mean(betas_int)
        W = np.mean(ses2_int)
        B = np.var(betas_int, ddof=1) if m > 1 else 0.0
        T = W + (1 + 1/m) * B
        se_bar = math.sqrt(T)
        z = b_bar / se_bar if se_bar > 0 else np.inf
        p_interaction = 2 * (1 - stats.norm.cdf(abs(z)))
        return {
            "modifier": modifier,
            "beta_interaction": b_bar,
            "se_interaction": se_bar,
            "p_interaction": p_interaction
        }
    return None

# ====== 20) Sensitivity Analysis (Stratified) ======
def sensitivity_stratified_analysis(mi_list: List[pd.DataFrame], outcome: str, weights: str,
                                   exposure: str, stratifier: str, strata: List, adjust: List[str]):
    results = []
    for stratum_value, stratum_label in strata:
        mi_stratum = []
        for df in mi_list:
            if stratifier == "age_group":
                df["age_group"] = pd.cut(df["age"], bins=[0, 65, 100], labels=[0, 1])
                stratum_df = df[df["age_group"] == stratum_value].copy()
            else:
                stratum_df = df[df[stratifier] == stratum_value].copy()
            if len(stratum_df) > 100:
                mi_stratum.append(stratum_df)
        if mi_stratum:
            try:
                rr, lcl, ucl, p, info = pool_mi_results(mi_stratum, outcome=outcome,
                                                       exposure=exposure,
                                                       weights=weights,
                                                       adjust=[a for a in adjust if a != stratifier])
                abs_risk = 0
                n_events = 0
                n_total = 0
                for df in mi_stratum:
                    d = df.dropna(subset=[outcome, weights])
                    abs_risk += np.average(d[outcome], weights=d[weights])
                    n_events += d[outcome].sum()
                    n_total += len(d)
                abs_risk = abs_risk / len(mi_stratum) * 100
                results.append({
                    "Stratum": stratum_label,
                    "N": n_total,
                    "Events": int(n_events / len(mi_stratum)),
                    "AbsoluteRisk": f"{abs_risk:.1f}%",
                    "RR": rr,
                    "LCL": lcl,
                    "UCL": ucl,
                    "p": p
                })
            except:
                continue
    return pd.DataFrame(results)

# ====== 21) Stratified Forest Plot ======
def plot_stratified_forest(stratified_results: Dict[str, pd.DataFrame], out_path: str):
    all_data = []
    for category, df in stratified_results.items():
        if not df.empty:
            for _, row in df.iterrows():
                all_data.append({
                    "Category": category,
                    "Stratum": row["Stratum"],
                    "N": row["N"],
                    "Events": row["Events"],
                    "RR": row["RR"],
                    "LCL": row["LCL"],
                    "UCL": row["UCL"]
                })
    if not all_data:
        logger.warning("No stratified data to plot")
        return

    df_plot = pd.DataFrame(all_data)
    y_pos = list(range(len(df_plot)))
    y_labels = []
    # CUD color scheme
    colors = {"Age": "#0072B2", "Sex": "#E69F00", "SES": "#009E73"}

    plt.figure(figsize=(12, max(6, 0.5*len(df_plot))))
    for i, row in df_plot.iterrows():
        label = f"{row['Category']}: {row['Stratum']} (n={row['N']:,}, events={row['Events']})"
        y_labels.append(label)
        color = colors.get(row['Category'], "#999999")
        plt.hlines(y_pos[i], row['LCL'], row['UCL'], colors=color, linewidth=2)
        plt.scatter(row['RR'], y_pos[i], s=60, c=color, zorder=5)

    plt.axvline(1.0, linestyle="--", color="#D55E00", alpha=0.5, linewidth=1)
    plt.yticks(y_pos, y_labels)
    plt.xscale("log")

    min_val = max(df_plot['LCL'].min(), 0.7)
    max_val = min(df_plot['UCL'].max(), 1.5)

    major_ticks = []
    tick_val = 0.7
    while tick_val <= max_val:
        major_ticks.append(tick_val)
        tick_val += 0.1
    ax = plt.gca()
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([f"{x:.1f}" for x in major_ticks])
    ax.grid(True, axis='x', which='major', alpha=0.3, linestyle='-', linewidth=0.5)

    plt.xlabel("Adjusted Risk Ratio (95% CI)", fontsize=12)
    plt.title("Supplementary Figure: Stratified Analysis by Subgroups", fontsize=14)

    for i, row in df_plot.iterrows():
        text = f"{row['RR']:.2f} ({row['LCL']:.2f}-{row['UCL']:.2f})"
        ax.text(max_val*1.02, y_pos[i], text, va='center', fontsize=9)

    ax.set_xlim(min_val, max_val)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved stratified forest plot to {out_path}")

# ====== 22) Main Execution ======
def main():
    logger.info("=== 1/9 Loading data ===")
    df21, enc21 = read_csv_known_enc(DATA_2021)
    df22, enc22 = read_csv_known_enc(DATA_2022)
    df23, enc23 = read_csv_known_enc(DATA_2023)
    for name, df in [("2021", df21), ("2022", df22), ("2023", df23)]:
        if "USER_ID" not in df.columns:
            raise KeyError(f"[{name}] USER_ID column not found")

    logger.info("=== 2/9 Scoring HI ===")
    cov21 = build_covariates_2021(df21)
    hi22 = score_hi_2022(df22)

    logger.info("=== 3/9 Building outcomes ===")
    y23 = build_outcomes_2023(df23)

    logger.info("=== 4/9 Merging panel & IPCW ===")
    panel = merge_panel(cov21, hi22, y23)

    ipcw = compute_ipcw_weights(df21, panel["USER_ID"])
    panel = panel.merge(ipcw[["USER_ID","ipcw_weight","complete3"]], on="USER_ID", how="left")
    panel = panel[panel["complete3"]==1].copy()
    panel.drop(columns=["complete3"], inplace=True)

    logger.info(f"Panel size after IPCW: {len(panel):,}")

    key_outcomes = ["COVID19_Infection_23", "HospitalizationPastYear_23", "COVID19_Vaccination_23"]
    for outcome in key_outcomes:
        if outcome in panel.columns:
            d_complete = panel.dropna(subset=[outcome, "HI_z_22", "ipcw_weight"] + COVARS)
            n_complete = len(d_complete)
            n_events = int(d_complete[outcome].sum())
            event_rate = n_events / n_complete * 100 if n_complete > 0 else 0
            logger.info(f"{outcome}: N_complete={n_complete:,}, Events={n_events:,}, Rate={event_rate:.1f}%")
            vc = d_complete[outcome].value_counts()
            logger.info(f"  Value distribution: {vc.to_dict()}")

    panel.to_csv(os.path.join(OUT_DIR, "panel_ready.csv"), index=False)

    logger.info("=== 5/9 Multiple imputation (covariates) ===")
    mi_list = multiple_impute_covars(panel, m=5, random_state=2025)

    logger.info("=== 6/9 PRIMARY ANALYSIS: Continuous HI (1 SD increase) ===")
    outcomes = outcome_list_all()
    rows = []
    for out in outcomes:
        if out not in panel.columns:
            continue
        try:
            # RR (Rubin)
            rr, lcl, ucl, p, info = pool_mi_results(mi_list, outcome=out, exposure="HI_z_22",
                                                    weights="ipcw_weight", adjust=COVARS)

            # aRD (standardized, Rubin)
            ard, ard_lcl, ard_ucl = standardized_riskdiff_mi(
                mi_list, outcome=out, weights="ipcw_weight",
                exposure="HI_z_22", adjust=COVARS, delta=1.0
            )

            # Descriptive marginal proportion (%)
            d = panel.dropna(subset=[out, "ipcw_weight"])
            abs_risk = np.average(d[out], weights=d["ipcw_weight"]) * 100
            n_events = int(d[out].sum())
            n_total = len(d)

            rows.append({
                "Outcome": out,
                "N_total": n_total,
                "N_events": n_events,
                "RR": rr, "LCL": lcl, "UCL": ucl, "p": p,
                "beta_bar": info["beta_bar"], "se_bar": info["se_bar"],
                # aRD (% points)
                "aRD": ard, "aRD_LCL": ard_lcl, "aRD_UCL": ard_ucl,
                "AbsoluteRisk_pct": abs_risk
            })
        except Exception as e:
            logger.warning(f"Failed to analyze {out}: {e}")

    res = pd.DataFrame(rows)
    if res.shape[0] == 0:
        logger.warning("No outcomes were analyzed. Check data columns.")
        res.to_csv(os.path.join(OUT_DIR, "SupplementaryTable1_main_results.csv"), index=False)
        return

    # Multiple comparison (FDR)
    rej, qvals, _, _ = multipletests(res["p"].values, alpha=0.05, method="fdr_bh")
    res["FDR_reject"] = rej
    res["FDR_q"] = qvals
    alpha_bonf = 0.05 / res.shape[0]
    res["Bonf_threshold"] = alpha_bonf
    res["Bonf_reject"] = res["p"] < alpha_bonf

    # Create figure with conventional column names
    plot_forest(res, os.path.join(OUT_DIR, "Figure2_forest_main.png"))

    # Format main table, column order, column names
    # Convert CI to string
    res["RR_CI_str"] = res.apply(lambda r: f"{r['LCL']:.2f}-{r['UCL']:.2f}", axis=1)
    res["aRD_CI_str"] = res.apply(lambda r: f"{r['aRD_LCL']:.2f}-{r['aRD_UCL']:.2f}", axis=1)

    # Change to rRR column name and reorder (CSV will have duplicate column name '95%CI(LCL-UCL)')
    out_cols = [
        "Outcome","N_total","N_events",
        "RR","RR_CI_str",
        "aRD","aRD_CI_str",
        "p","beta_bar","se_bar","FDR_reject","FDR_q","Bonf_threshold","Bonf_reject",
        "AbsoluteRisk_pct"
    ]
    res_out = res[out_cols].copy()
    res_out.rename(columns={"RR":"rRR"}, inplace=True)

    # Header names (set duplicate column names as requested)
    res_out.columns = [
        "Outcome","N_total","N_events",
        "rRR","95%CI(LCL-UCL)",
        "aRD","95%CI(LCL-UCL)",
        "p","beta_bar","se_bar","FDR_reject","FDR_q","Bonf_threshold","Bonf_reject",
        "AbsoluteRisk_pct"
    ]
    # Save (main table)
    res_out.to_csv(os.path.join(OUT_DIR, "SupplementaryTable1_main_results.csv"), index=False)

    logger.info("=== 7/9 SENSITIVITY ANALYSES ===")
    key_outcomes = ["HospitalizationPastYear_23", "COVID19_Vaccination_23",
                    "Depression_23", "Hypertension_23"]
    all_quartile_results = []
    for out in key_outcomes:
        if out in panel.columns:
            q_res = sensitivity_quartile_analysis(mi_list, out, "ipcw_weight", COVARS)
            q_res["Outcome"] = out
            all_quartile_results.append(q_res)
    if all_quartile_results:
        quartile_df = pd.concat(all_quartile_results, ignore_index=True)
        quartile_df.to_csv(os.path.join(OUT_DIR, "SupplementaryTable2_quartile_analysis.csv"), index=False)

    interactions = []
    for out in ["HospitalizationPastYear_23"]:
        if out not in panel.columns:
            continue
        int_age = sensitivity_interaction_analysis(mi_list, out, "ipcw_weight", "HI_z_22", "age", COVARS)
        if int_age:
            int_age["Outcome"] = out
            int_age["Modifier"] = "Age"
            interactions.append(int_age)
        int_sex = sensitivity_interaction_analysis(mi_list, out, "ipcw_weight", "HI_z_22", "male", COVARS)
        if int_sex:
            int_sex["Outcome"] = out
            int_sex["Modifier"] = "Sex"
            interactions.append(int_sex)
        int_ses = sensitivity_interaction_analysis(mi_list, out, "ipcw_weight", "HI_z_22", "low_income", COVARS)
        if int_ses:
            int_ses["Outcome"] = out
            int_ses["Modifier"] = "Low Income"
            interactions.append(int_ses)
    if interactions:
        int_df = pd.DataFrame(interactions)
        int_df.to_csv(os.path.join(OUT_DIR, "SupplementaryTable3_interaction.csv"), index=False)

    logger.info("=== 8/9 Table 1 and Additional Analyses ===")
    t1 = summarize_table1(panel, "ipcw_weight")
    t1.to_csv(os.path.join(OUT_DIR, "Table1_characteristics.csv"), index=False)

    abs_results = []
    for out in ["HospitalizationPastYear_23", "COVID19_Vaccination_23"]:
        if out in panel.columns:
            abs_df = absolute_risk_by_quartile(panel, out, "ipcw_weight")
            abs_df["Outcome"] = out
            abs_results.append(abs_df)
    if abs_results:
        pd.concat(abs_results, ignore_index=True).to_csv(
            os.path.join(OUT_DIR, "SupplementaryTable5_absolute_risk.csv"), index=False)

    plot_spline_hospitalization(panel, "ipcw_weight",
                               os.path.join(OUT_DIR, "Figure4_spline.png"),
                               df_spline=3, x_min=13, x_max=52, x_ref=29)

    if GCASTLE_AVAILABLE or DAGMA_AVAILABLE:
        cd = CausalDiscoveryCV(hi_var_type="continuous")
        causal_df = cd.run(panel)
        
        # Get subset variable list
        subset_variables = getattr(causal_df, 'attrs', {}).get('subset_variables', {})
        
        if not causal_df.empty:
            causal_df.to_csv(os.path.join(OUT_DIR, "SupplementaryTable6_causal_edges.csv"), index=False)
        
        # Check number of edges for each subset
        disease_edges = causal_df[causal_df["Subset"] == "disease"] if not causal_df.empty else pd.DataFrame()
        symptom_edges = causal_df[causal_df["Subset"] == "symptoms"] if not causal_df.empty else pd.DataFrame()
        logger.info(f"Disease subset: {len(disease_edges)} edges found")
        logger.info(f"Symptoms subset: {len(symptom_edges)} edges found")
        
        # Disease+Vaccination DAG (Figure 3)
        plot_causal_network(causal_df, 
                          os.path.join(OUT_DIR, "Figure3_causal_network_disease.png"),
                          subset_name="disease",
                          all_variables=subset_variables.get("disease", []))
        
        # Symptoms DAG (Supplementary Figure 2)
        plot_causal_network(causal_df,
                          os.path.join(OUT_DIR, "SupplementaryFigure1_causal_network_symptoms.png"),
                          subset_name="symptoms",
                          all_variables=subset_variables.get("symptoms", []))
    else:
        logger.warning("Causal discovery libraries not available")

    logger.info("=" * 60)
    logger.info("Analysis completed with optimized hyperparameters. Files for JAMA Network Open:")
    logger.info("Main Manuscript:")
    logger.info("  - Table1_characteristics.csv")
    logger.info("  - Figure2_forest_main.png")
    logger.info("  - Figure3_causal_network_disease.png (Disease & Vaccination outcomes)")
    logger.info("  - Figure4_spline.png")
    logger.info("Supplementary Materials:")
    logger.info("  - SupplementaryTable1_main_results.csv  [rRR & aRD integrated]")
    logger.info("  - SupplementaryTable2_quartile_analysis.csv")
    logger.info("  - SupplementaryTable3_interaction.csv")
    logger.info("  - SupplementaryTable4_stratified.csv")
    logger.info("  - SupplementaryTable5_absolute_risk.csv")
    logger.info("  - SupplementaryTable6_causal_edges.csv")
    logger.info("  - SupplementaryFigure2_causal_network_symptoms.png (Symptom outcomes)")
    logger.info("=" * 60)

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            main()
        except Exception as e:
            logger.error(f"Fatal error in main execution: {e}")
            import traceback
            traceback.print_exc()
