"""Why SMOTE appears to destabilise explanations under naive perturbations.

2 x 2 x 2 factorial on HR Attrition and Telco, 5 seeds, N=100, K=20:
    explainer  path-dependent TreeSHAP of the raw output vs interventional TreeSHAP of the probability
    features   one-hot columns vs attributions grouped to raw fields
    noise      naive (fractional noise on every numeric field) vs type-respecting
crossed with no treatment, SMOTE and SMOTE-NC.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed

from fragility.measures import cies
from fragility.pipeline import DATASET_ID, RESULTS, TABLES, fit_model, group_attr, perturb, prepare


def naive_noise(x, num_cols, k, eps):
    """Noise on every numeric field, without rounding or truncation."""
    rows = []
    for _ in range(k):
        z = x.copy()
        for c in num_cols:
            z[c] = x[c] + np.random.normal(0, abs(x[c] * eps) if x[c] != 0 else eps)
        rows.append(z)
    return pd.DataFrame(rows)


def as_matrix(sv):
    """SHAP values of the positive class as an (instances x columns) array."""
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv.reshape(1, -1) if sv.ndim == 1 else sv


def run(ds, model, strategy, seed):
    P = prepare(ds, seed, strategy)
    m, _ = fit_model(P, model)
    explainers = {"path_dependent": shap.TreeExplainer(m),
                  "interventional": shap.TreeExplainer(m, data=P.background, feature_perturbation="interventional",
                                                model_output="probability")}
    inst = P.X_te.sample(n=100, random_state=seed)
    rows = []
    for noise in ["naive", "typed"]:
        blocks = []
        for pos, (_, x) in enumerate(inst.iterrows()):
            if noise == "naive":
                np.random.seed([seed, pos])
                nb = naive_noise(x, P.num_cols, 20, 0.03)
            else:
                nb = perturb(P, x, 20, 0.03, "gauss_mult", np.random.default_rng([seed, DATASET_ID[ds], pos, 0, 30]))
            blocks.append(pd.concat([pd.DataFrame([x.values], columns=x.index), nb], ignore_index=True))
        raw = pd.concat(blocks, ignore_index=True)
        for c in P.num_cols:
            raw[c] = pd.to_numeric(raw[c])
        Xt = P.pre.transform(raw)
        for name, E in explainers.items():
            sv = as_matrix(E.shap_values(Xt) if name == "path_dependent" else E.shap_values(Xt, check_additivity=False))
            for features in ["dummies", "grouped"]:
                A = sv if features == "dummies" else group_attr(sv, P.groups, len(P.raw_names))
                scores = [cies(A[i * 21], A[i * 21 + 1:(i + 1) * 21]) for i in range(len(inst))]
                rows.append(dict(dataset=ds, model=model, strategy=strategy, seed=seed, noise=noise,
                                 explainer=name, features=features, cies=np.nanmean(scores)))
    return rows


if __name__ == "__main__":
    jobs = itertools.product(["hr_attrition", "telco"], ["LightGBM", "XGBoost", "Random Forest"],
                             ["none", "smote", "smotenc"], range(5))
    res = Parallel(n_jobs=int(sys.argv[1]) if len(sys.argv) > 1 else -1)(delayed(run)(*j) for j in jobs)
    df = pd.DataFrame([r for rr in res for r in rr])
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "smote_factorial.csv"), index=False)
    base = df[df.strategy == "none"].drop(columns="strategy").rename(columns={"cies": "cies_none"})
    j = df[df.strategy != "none"].merge(base, on=["dataset", "model", "seed", "noise", "explainer", "features"])
    j["d"] = j.cies - j.cies_none
    os.makedirs(TABLES, exist_ok=True)
    j.groupby(["dataset", "model", "strategy", "noise", "explainer", "features"]).d.agg(["mean", "std"]).to_csv(
        os.path.join(TABLES, "T_smote_mechanism.csv"))
