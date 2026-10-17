"""Explains N=100 test instances per configuration under recording-error scenarios.

For every instance, K=20 diagnostic neighbours give the stability measures and the
direct counts, and K'=30 independent neighbours give the decision consequences.
The first 20 independent neighbours also give the test-retest values.

Grids
    main        3 datasets x 5 treatments x 5 models x {default, tuned}, seeds 0-4
    fresh       as main, seeds 5-9 (evaluation data of the monitoring task)
    mixed       default models, mixed noise, seeds 0-4
    mixedfresh  as mixed, seeds 5-9
    relational  Telco and HR Attrition, no treatment and SMOTE, relational noise, seeds 0-4
    noise       four noise models x four noise levels, seeds 0-1, no independent neighbours

    python experiments/run_grid.py main [n_jobs]
"""
import glob
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from fragility.measures import cies, consequence_bounds, decision_consequences, stability_metrics, w_harmonic
from fragility.pipeline import (ALL_MODELS, DATASET_ID, DATASETS, NOISE_TYPES, RESULTS, STRATEGIES, ShapExplainer,
                      fit_model, perturb, prepare)

N_INST, K_DIAG, K_FRESH = 100, 20, 30
NOISE_CODE = {n: i for i, n in enumerate(NOISE_TYPES)}


def grid_tasks(grid):
    seeds = range(5, 10) if grid in ("fresh", "mixedfresh") else range(5)
    if grid in ("main", "fresh"):
        it = itertools.product(DATASETS, seeds, STRATEGIES, ALL_MODELS, ["default", "tuned"], ["gauss_mult"], [0.03])
    elif grid in ("mixed", "mixedfresh"):
        it = itertools.product(DATASETS, seeds, STRATEGIES, ALL_MODELS, ["default"], ["mixed"], [0.03])
    elif grid == "relational":
        it = itertools.product(["telco", "hr_attrition"], seeds, ["none", "smote"], ALL_MODELS, ["default"],
                               ["relational"], [0.03])
    elif grid == "noise":
        it = itertools.product(DATASETS, range(2), ["none", "smote"], ALL_MODELS, ["default"],
                               NOISE_TYPES[:4], [0.01, 0.03, 0.05, 0.10])
    else:
        raise ValueError(grid)
    keys = ["dataset", "seed", "strategy", "model", "variant", "noise", "eps"]
    tasks = [dict(zip(keys, v)) for v in it]
    # slowest jobs first
    return sorted(tasks, key=lambda t: (t["model"] != "Random Forest", t["dataset"] != "telco"))


def counts(phi_x, phi_nb, p_x, p_nb, prefix):
    c = decision_consequences(phi_x, phi_nb, p_x, p_nb)
    return {prefix + k[5:]: v for k, v in c.items()}


def run_task(t, out_dir, with_fresh=True):
    path = os.path.join(out_dir, "__".join(str(v).replace(" ", "") for v in t.values()) + ".csv")
    if os.path.exists(path):
        return path
    ds, seed, noise, eps = t["dataset"], t["seed"], t["noise"], t["eps"]
    P = prepare(ds, seed, t["strategy"])
    model, perf = fit_model(P, t["model"], t["variant"])
    explain = ShapExplainer(model, t["model"], P)

    # the same instances and the same noise draws for every model and treatment of a seed
    inst = P.X_te.sample(n=min(N_INST, len(P.X_te)), random_state=seed)
    k_tot = K_DIAG + (K_FRESH if with_fresh else 0)
    blocks = []
    for pos, (_, x) in enumerate(inst.iterrows()):
        rng = np.random.default_rng([seed, DATASET_ID[ds], pos, NOISE_CODE[noise], int(round(eps * 1000))])
        blocks.append(pd.concat([pd.DataFrame([x.values], columns=x.index), perturb(P, x, k_tot, eps, noise, rng)],
                                ignore_index=True))
    raw = pd.concat(blocks, ignore_index=True)
    for c in P.num_cols:
        raw[c] = pd.to_numeric(raw[c])
    Xt = P.pre.transform(raw)

    t0 = time.time()
    phi = explain(Xt)
    t_explain = time.time() - t0
    p = model.predict_proba(Xt)[:, 1]
    base = float(np.mean(model.predict_proba(P.background)[:, 1]))

    rows, step = [], 1 + k_tot
    for pos, idx in enumerate(inst.index):
        s = pos * step
        diag = slice(s + 1, s + 1 + K_DIAG)
        r = dict(t, instance=int(idx), pos=pos, y_true=int(P.y_te.loc[idx]), p_x=p[s], p_minus_base=p[s] - base,
                 phi_l1=float(np.abs(phi[s]).sum()), phi_wmag=float(np.sum(w_harmonic(phi[s]) * np.abs(phi[s]))))
        r.update(stability_metrics(phi[s], phi[diag], Xt[s], Xt[diag], p[s], p[diag]))
        r.update(consequence_bounds(phi[s], r["cies"]))
        r.update(counts(phi[s], phi[diag], p[s], p[diag], "count_"))
        for k in (5, 10):
            first_k = slice(s + 1, s + 1 + k)
            r[f"cies_d{k}"] = cies(phi[s], phi[first_k])
            r.update(counts(phi[s], phi[first_k], p[s], p[first_k], f"cnt_d{k}_"))
        if with_fresh:
            fresh = slice(s + 1 + K_DIAG, s + step)
            retest = slice(s + 1 + K_DIAG, s + 1 + 2 * K_DIAG)
            r.update(decision_consequences(phi[s], phi[fresh], p[s], p[fresh]))
            rt = stability_metrics(phi[s], phi[retest], Xt[s], Xt[retest], p[s], p[retest])
            rt.update(counts(phi[s], phi[retest], p[s], p[retest], "count_"))
            r.update({"retest_" + k: v for k, v in rt.items()})
            for k in (5, 10, 20, 30):
                r[f"cies_K{k}"] = cies(phi[s], phi[s + 1 + K_DIAG:s + 1 + K_DIAG + k])
        rows.append(r)
    df = pd.DataFrame(rows)
    for k, v in perf.items():
        df["perf_" + k] = v
    df["runtime_per_instance"] = t_explain / len(Xt) * (1 + K_DIAG)   # one CIES evaluation
    df.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return path


if __name__ == "__main__":
    grid = sys.argv[1]
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    out_dir = os.path.join(RESULTS, grid)
    os.makedirs(out_dir, exist_ok=True)
    Parallel(n_jobs=n_jobs, batch_size=1)(delayed(run_task)(t, out_dir, grid != "noise") for t in grid_tasks(grid))
    files = sorted(glob.glob(os.path.join(out_dir, "*.csv")))
    pd.concat([pd.read_csv(f) for f in files], ignore_index=True).to_csv(os.path.join(RESULTS, f"{grid}.csv"),
                                                                         index=False)
