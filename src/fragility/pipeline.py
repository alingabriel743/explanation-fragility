"""Data preparation, recording-error scenarios, models and SHAP explanations."""
import json
import os
import time
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data import load

warnings.filterwarnings("ignore")

ROOT = os.environ.get("FRAGILITY_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS = os.path.join(ROOT, "results")
TABLES = os.path.join(ROOT, "tables")
FIGURES = os.path.join(ROOT, "figures")

DATASETS = ["telco", "german_credit", "hr_attrition"]
DATASET_ID = {name: i for i, name in enumerate(DATASETS)}
TREE_MODELS = ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]
ALL_MODELS = TREE_MODELS + ["Logistic Regression"]
STRATEGIES = ["none", "smote", "smotenc", "class_weight", "undersample"]
NOISE_TYPES = ["gauss_mult", "gauss_sd", "laplace_mult", "mixed", "relational"]


def feature_types(X):
    """Classify every raw field as categorical, binary, ordinal, integer or continuous."""
    types = {}
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            types[c] = "categorical"
            continue
        v = X[c].dropna()
        is_int = np.allclose(v % 1, 0)
        n = v.nunique()
        if n <= 2:
            types[c] = "binary"
        elif is_int and n <= 10:
            types[c] = "ordinal"
        else:
            types[c] = "integer" if is_int else "continuous"
    return types


def make_preprocessor(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
    ])


def prepare(name, seed, strategy):
    """Split, preprocess and (optionally) resample one dataset for one seed."""
    X, y, display = load(name)
    ftypes = feature_types(X)
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    pre = make_preprocessor(num_cols, cat_cols)
    Xt_tr = pre.fit_transform(X_tr)
    Xt_te = pre.transform(X_te)

    # index of the raw field behind every transformed column, to sum one-hot attributions
    raw_names = list(X.columns)
    groups = [raw_names.index(c) for c in num_cols]
    for c, cats in zip(cat_cols, pre.named_transformers_["cat"].named_steps["oh"].categories_):
        groups += [raw_names.index(c)] * len(cats)

    Xt_fit, y_fit = Xt_tr, y_tr.values
    if strategy == "smote":
        Xt_fit, y_fit = SMOTE(random_state=seed).fit_resample(Xt_tr, y_tr.values)
    elif strategy == "undersample":
        Xt_fit, y_fit = RandomUnderSampler(random_state=seed).fit_resample(Xt_tr, y_tr.values)
    elif strategy == "smotenc":
        # resample the raw records so that discrete fields keep admissible values
        Xr = X_tr.copy()
        for c in num_cols:
            Xr[c] = Xr[c].fillna(X_tr[c].median())
        discrete = [i for i, c in enumerate(raw_names) if ftypes[c] in ("categorical", "binary", "ordinal")]
        Xr, y_fit = SMOTENC(categorical_features=discrete, random_state=seed).fit_resample(Xr, y_tr.values)
        for c in raw_names:
            if ftypes[c] == "integer":
                Xr[c] = np.round(Xr[c].astype(float))
        Xt_fit = pre.transform(Xr)

    background = Xt_tr[np.random.default_rng(seed).choice(len(Xt_tr), size=min(50, len(Xt_tr)), replace=False)]
    return SimpleNamespace(
        name=name, display=display, seed=seed, strategy=strategy,
        X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te, Xt_tr=Xt_tr, Xt_te=Xt_te, Xt_fit=Xt_fit, y_fit=y_fit,
        pre=pre, groups=np.array(groups), raw_names=raw_names, ftypes=ftypes,
        num_cols=num_cols, cat_cols=cat_cols, background=background,
        train_sd={c: float(X_tr[c].std()) for c in num_cols},
        train_min={c: float(X_tr[c].min()) for c in num_cols},
        train_max={c: float(X_tr[c].max()) for c in num_cols},
        cat_freq={c: X_tr[c].value_counts(normalize=True) for c in cat_cols},
    )


def perturb(P, x, K, eps, noise, rng):
    """K re-measurements of the raw record x, in original units.

    gauss_mult    Gaussian error with SD eps*|x_j| (eps if x_j = 0)
    gauss_sd      Gaussian error with SD eps*SD_j
    laplace_mult  Laplace error with the variance of gauss_mult
    mixed         gauss_mult plus mis-recording of every discrete field with probability eps
    relational    gauss_mult, then deterministic relationships between fields are restored
    Integer fields are rounded and non-negative fields truncated at zero.
    """
    nb = pd.DataFrame([x.values] * K, columns=x.index)
    for c in P.raw_names:
        kind, v = P.ftypes[c], x[c]
        if kind in ("continuous", "integer"):
            if pd.isna(v):
                continue
            if noise == "gauss_sd":
                z = rng.normal(0.0, eps * P.train_sd[c], K)
            else:
                sd = eps * abs(v) if v != 0 else eps
                z = rng.laplace(0.0, sd / np.sqrt(2.0), K) if noise == "laplace_mult" else rng.normal(0.0, sd, K)
            new = float(v) + z
            if kind == "integer":
                new = np.round(new)
            if P.train_min[c] >= 0:
                new = np.maximum(new, 0.0)
            nb[c] = new
        elif noise == "mixed":
            flip = rng.random(K) < eps
            if not flip.any():
                continue
            col = nb[c].to_numpy(dtype=object).copy()
            if kind == "binary":
                lo, hi = P.train_min[c], P.train_max[c]
                col[flip] = hi if v == lo else lo
            elif kind == "ordinal":
                lo, hi = P.train_min[c], P.train_max[c]
                cand = float(v) + rng.choice([-1, 1], size=K)
                cand = np.where(cand < lo, float(v) + 1, cand)
                cand = np.where(cand > hi, float(v) - 1, cand)
                col[flip] = cand[flip]
            else:
                freq = P.cat_freq[c].drop(labels=[v], errors="ignore")
                if len(freq):
                    col[flip] = rng.choice(freq.index.values, size=int(flip.sum()), p=freq.values / freq.values.sum())
            nb[c] = col
    for c in P.num_cols:
        nb[c] = pd.to_numeric(nb[c])
    if noise == "relational":
        nb = enforce_relations(P.name, x, nb)
    return nb


def enforce_relations(name, x, nb):
    """Telco: total charges move with tenure x monthly charges.
    HR: years in role, since promotion and with manager <= years at company <= total working years <= age - 18."""
    nb = nb.copy()
    if name == "telco":
        base = float(x["tenure"]) * float(x["MonthlyCharges"])
        if base > 0 and not pd.isna(x["TotalCharges"]):
            ratio = nb["tenure"] * nb["MonthlyCharges"] / base
            nb["TotalCharges"] = np.where(nb["tenure"] > 0, float(x["TotalCharges"]) * ratio, np.nan)
    elif name == "hr_attrition":
        nb["TotalWorkingYears"] = np.minimum(nb["TotalWorkingYears"], np.maximum(nb["Age"] - 18, 0))
        nb["YearsAtCompany"] = np.minimum(nb["YearsAtCompany"], nb["TotalWorkingYears"])
        for c in ("YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"):
            nb[c] = np.minimum(nb[c], nb["YearsAtCompany"])
    return nb


DEFAULT_PARAMS = {
    "Random Forest": dict(n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=2),
    "XGBoost": dict(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8),
    "LightGBM": dict(n_estimators=200, max_depth=-1, num_leaves=31, learning_rate=0.1, subsample=0.8,
                     colsample_bytree=0.8),
    "CatBoost": dict(iterations=500, depth=6, learning_rate=0.05),
    "Logistic Regression": dict(C=1.0),
}


def make_model(model, params, seed, class_weight=False, y=None):
    p = dict(params)
    if model == "Random Forest":
        if class_weight:
            p["class_weight"] = "balanced"
        return RandomForestClassifier(random_state=seed, n_jobs=1, **p)
    if model == "XGBoost":
        if class_weight:
            p["scale_pos_weight"] = float((y == 0).sum() / max((y == 1).sum(), 1))
        return XGBClassifier(random_state=seed, n_jobs=1, eval_metric="logloss", verbosity=0, **p)
    if model == "LightGBM":
        if class_weight:
            p["class_weight"] = "balanced"
        return LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1, **p)
    if model == "CatBoost":
        if class_weight:
            p["auto_class_weights"] = "Balanced"
        return CatBoostClassifier(random_seed=seed, thread_count=1, verbose=0, allow_writing_files=False, **p)
    if model == "Logistic Regression":
        if class_weight:
            p["class_weight"] = "balanced"
        return LogisticRegression(max_iter=5000, random_state=seed, **p)
    raise ValueError(model)


def tuned_params(name, seed, model):
    with open(os.path.join(RESULTS, "tuned_params.json")) as f:
        return json.load(f)[name][str(seed)][model]


def fit_model(P, model, variant="default"):
    params = tuned_params(P.name, P.seed, model) if variant == "tuned" else DEFAULT_PARAMS[model]
    m = make_model(model, params, P.seed, class_weight=P.strategy == "class_weight", y=P.y_fit)
    t0 = time.time()
    m.fit(P.Xt_fit, P.y_fit)
    fit_time = time.time() - t0
    prob = m.predict_proba(P.Xt_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    perf = dict(accuracy=accuracy_score(P.y_te, pred), f1=f1_score(P.y_te, pred),
                balanced_accuracy=balanced_accuracy_score(P.y_te, pred), roc_auc=roc_auc_score(P.y_te, prob),
                pr_auc=average_precision_score(P.y_te, prob), fit_time=fit_time)
    return m, perf


def group_attr(sv, groups, M):
    """Sum attributions of the transformed columns of every raw field."""
    out = np.zeros((sv.shape[0], M))
    np.add.at(out.T, groups, sv.T)
    return out


class ShapExplainer:
    """Interventional TreeSHAP of the predicted probability for tree models and exact linear
    SHAP (log-odds) for logistic regression, grouped to the raw fields."""

    def __init__(self, model, model_name, P):
        self.P = P
        self.linear = model_name == "Logistic Regression"
        if self.linear:
            self.exp = shap.LinearExplainer(model, P.background)
        else:
            self.exp = shap.TreeExplainer(model, data=P.background, feature_perturbation="interventional",
                                          model_output="probability")

    def __call__(self, Xt):
        sv = self.exp.shap_values(Xt) if self.linear else self.exp.shap_values(Xt, check_additivity=False)
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        return group_attr(sv, self.P.groups, len(self.P.raw_names))
