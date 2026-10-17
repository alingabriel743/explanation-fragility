"""CIES of the boosted models explained on the probability and on the log-odds scale
(same instances, neighbours and background; default models, no treatment, 5 seeds)."""
import itertools
import os

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from scipy.stats import spearmanr

from fragility.measures import cies
from fragility.pipeline import DATASET_ID, DATASETS, RESULTS, TABLES, fit_model, group_attr, perturb, prepare

K = 20


def run(ds, seed, model):
    P = prepare(ds, seed, "none")
    m, _ = fit_model(P, model)
    inst = P.X_te.sample(n=100, random_state=seed)
    blocks = []
    for pos, (_, x) in enumerate(inst.iterrows()):
        nb = perturb(P, x, K, 0.03, "gauss_mult", np.random.default_rng([seed, DATASET_ID[ds], pos, 0, 30]))
        blocks.append(pd.concat([pd.DataFrame([x.values], columns=x.index), nb], ignore_index=True))
    raw = pd.concat(blocks, ignore_index=True)
    for c in P.num_cols:
        raw[c] = pd.to_numeric(raw[c])
    Xt = P.pre.transform(raw)
    rows = [dict(dataset=ds, seed=seed, model=model, pos=i) for i in range(len(inst))]
    for scale, output in (("probability", "probability"), ("log-odds", "raw")):
        E = shap.TreeExplainer(m, data=P.background, feature_perturbation="interventional", model_output=output)
        sv = np.asarray(E.shap_values(Xt, check_additivity=False))
        phi = group_attr(sv[:, :, 1] if sv.ndim == 3 else sv, P.groups, len(P.raw_names))
        for i, r in enumerate(rows):
            r[f"cies_{scale}"] = cies(phi[i * (K + 1)], phi[i * (K + 1) + 1:(i + 1) * (K + 1)])
    return pd.DataFrame(rows)


if __name__ == "__main__":
    jobs = itertools.product(DATASETS, range(5), ["XGBoost", "LightGBM", "CatBoost"])
    df = pd.concat(Parallel(n_jobs=-1)(delayed(run)(*j) for j in jobs), ignore_index=True)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "scale_check.csv"), index=False)
    pair = ["cies_probability", "cies_log-odds"]
    inst_rho = df.groupby(["dataset", "seed", "model"]).apply(lambda g: spearmanr(g[pair[0]], g[pair[1]])[0])
    blocks = df.groupby(["dataset", "seed", "model"])[pair].mean().reset_index()
    order_rho = blocks.groupby(["dataset", "seed"]).apply(lambda g: spearmanr(g[pair[0]], g[pair[1]])[0])
    os.makedirs(TABLES, exist_ok=True)
    pd.DataFrame([dict(instance_rho_mean=inst_rho.mean(), instance_rho_min=inst_rho.min(),
                       model_order_identical_share=(order_rho == 1).mean(),
                       mean_abs_diff=(df[pair[0]] - df[pair[1]]).abs().mean())]).to_csv(
        os.path.join(TABLES, "T_scale_check.csv"), index=False)
    df.groupby(["dataset", "model"])[pair].mean().to_csv(os.path.join(TABLES, "T_scale_check_means.csv"))
