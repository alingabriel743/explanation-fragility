"""LIME explanations: own sampling randomness versus input perturbation.

For every instance x (seed 0, no treatment, default models):
    L0  x explained R times with different LIME seeds, no input noise
    L1  K neighbours explained with the seed used for x
    L2  K neighbours explained with a different seed each
at 500 and 5,000 LIME samples, plus the VSI/CSI indices of Visani et al. (2022).
"""
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lime.lime_tabular import LimeTabularExplainer
from lime_stability.stability import LimeTabularExplainerOvr
from sklearn.utils import check_random_state

from fragility.measures import stability_metrics
from fragility.pipeline import ALL_MODELS, DATASET_ID, DATASETS, NOISE_TYPES, RESULTS, fit_model, group_attr, perturb, prepare

N_INST, K, R = 30, 10, 10
EPS, NOISE = 0.03, "gauss_mult"


def lime_vector(explainer, row, predict, n_cols, seed, n_samples, P):
    explainer.random_state = check_random_state(seed)
    e = explainer.explain_instance(row, predict, num_features=n_cols, num_samples=n_samples, labels=(1,))
    v = np.zeros(n_cols)
    for j, w in e.local_exp[1]:
        v[j] = w
    return group_attr(v[None, :], P.groups, len(P.raw_names))[0]


def run(ds, model, n_inst=N_INST):
    P = prepare(ds, 0, "none")
    m, _ = fit_model(P, model)
    n_cols = P.Xt_tr.shape[1]
    kw = dict(mode="classification", categorical_features=list(range(len(P.num_cols), n_cols)),
              discretize_continuous=True, random_state=0)
    explainer, ovr = LimeTabularExplainer(P.Xt_tr, **kw), LimeTabularExplainerOvr(P.Xt_tr, **kw)
    predict = m.predict_proba
    rows = []
    for pos, (idx, x) in enumerate(P.X_te.sample(n=n_inst, random_state=0).iterrows()):
        rng = np.random.default_rng([0, DATASET_ID[ds], pos, NOISE_TYPES.index(NOISE), 30])
        xt = P.pre.transform(pd.DataFrame([x]))[0]
        nbt = P.pre.transform(perturb(P, x, K, EPS, NOISE, rng))
        px, pnb = predict(xt[None, :])[0, 1], predict(nbt)[:, 1]
        seed = 1000 + pos
        for ns in (500, 5000):
            phx = lime_vector(explainer, xt, predict, n_cols, seed, ns, P)
            repeat = np.array([lime_vector(explainer, xt, predict, n_cols, seed + 1 + r, ns, P) for r in range(R)])
            fixed = np.array([lime_vector(explainer, nbt[k], predict, n_cols, seed, ns, P) for k in range(K)])
            varied = np.array([lime_vector(explainer, nbt[k], predict, n_cols, seed + 100 + k, ns, P) for k in range(K)])
            xr = np.repeat(xt[None, :], R, 0)
            for cond, ph, X2, p2 in [("L0 repeat (no noise)", repeat, xr + 1e-9, np.repeat(px, R)),
                                     ("L1 noise, fixed seed", fixed, nbt, pnb),
                                     ("L2 noise, varied seed", varied, nbt, pnb)]:
                met = stability_metrics(phx, ph, xt, X2, px, p2)
                rows.append(dict(dataset=P.display, model=model, instance=int(idx), num_samples=ns, condition=cond,
                                 **{k: met[k] for k in ("cies", "cies_uniform", "topk3", "rbo", "wkendall", "avg_sens")}))
        ovr.random_state = check_random_state(2000 + pos)
        csi, vsi = ovr.check_stability(xt, predict, labels=(1,), num_features=5, num_samples=5000, n_calls=10)
        rows.append(dict(dataset=P.display, model=model, instance=int(idx), num_samples=5000,
                         condition="Visani indices", csi=csi, vsi=vsi))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else -1
    res = Parallel(n_jobs=n_jobs)(delayed(run)(d, m) for d in DATASETS for m in ALL_MODELS)
    os.makedirs(RESULTS, exist_ok=True)
    pd.concat(res, ignore_index=True).to_csv(os.path.join(RESULTS, "lime.csv"), index=False)
