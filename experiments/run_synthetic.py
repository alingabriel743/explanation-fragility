"""Benchmark with known attributions and planted instabilities.

Model: x ~ N(0, I_20), logit f(x) = sum_j beta_j g_j(x_j) with decreasing beta_j. Smooth features
have g(x) = x, four tipping-point features have g(x) = tanh(20 x). With independent features
the interventional SHAP values of the logit are phi_j(x) = beta_j g_j(x_j). Re-measurements add
N(0, 0.05^2) to every feature.

Instance types (250 each): A stable, B a low-importance tipping feature near its cut-off,
C a high-importance tipping feature near its cut-off, D two leading features tied within 1%.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from fragility.measures import (DECISIONS, METRIC_DIRECTION, METRIC_LABEL, cies, consequence_bounds, decision_consequences,
                      stability_metrics)
from fragility.pipeline import RESULTS

M = 20
BETA = np.array([3.0, 2.6, 2.2, 1.9, 1.6, 1.4, 1.2, 1.0, 0.85, 0.7, 0.6, 0.5, 0.4, 0.33, 0.27, 0.22, 0.18, 0.15,
                 0.12, 0.1])
TIP_HIGH, TIP_LOW = [0, 2], [15, 17]
TIP = TIP_HIGH + TIP_LOW
SIGMA = 0.05
K, K_FRESH, N_PER = 20, 200, 250
OUT = os.path.join(RESULTS, "synthetic")


def phi_exact(X):
    G = X.copy()
    G[..., TIP] = np.tanh(20.0 * X[..., TIP])
    return BETA * G


def logit(X):
    return phi_exact(X).sum(-1)


def prob(X):
    return 1 / (1 + np.exp(-logit(X)))


def sample_instance(kind, rng):
    x = rng.normal(0, 1, M)
    x[TIP] = rng.uniform(0.3, 2.0, len(TIP)) * rng.choice([-1, 1], len(TIP))
    near = lambda: rng.uniform(0.04, 0.15) * rng.choice([-1, 1])   # 0.8 to 3 noise SDs from the cut-off
    if kind == "B":
        x[rng.choice(TIP_LOW)] = near()
    elif kind == "C":
        x[rng.choice(TIP_HIGH)] = near()
    elif kind == "D":
        x[1] = np.sign(rng.normal()) * BETA[0] / BETA[1] * (1 + rng.uniform(-0.01, 0.01))
        x[3] = np.clip(x[3], -1.0, 1.0)
    return x


def faithfulness(phi, x, rng, n_subsets=50, size=5):
    """Faithfulness correlation (Bhatt et al., 2020) with the population mean 0 as baseline."""
    a, b = [], []
    for _ in range(n_subsets):
        s = rng.choice(M, size, replace=False)
        z = x.copy()
        z[s] = 0.0
        a.append(phi[s].sum())
        b.append(logit(x) - logit(z))
    return spearmanr(a, b)[0]


def benchmark():
    rng = np.random.default_rng(2026)
    rows = []
    for kind in "ABCD":
        for i in range(N_PER):
            x = sample_instance(kind, rng)
            nb = x + rng.normal(0, SIGMA, (K, M))
            fresh = x + rng.normal(0, SIGMA, (K_FRESH, M))
            phx, phn, phf = phi_exact(x), phi_exact(nb), phi_exact(fresh)
            r = dict(kind=kind, i=i)
            r.update(stability_metrics(phx, phn, x, nb, prob(x), prob(nb)))
            r.update(decision_consequences(phx, phf, prob(x), prob(fresh)))
            r.update(consequence_bounds(phx, r["cies"]))
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "instances.csv"), index=False)

    res = []
    for m in [m for m in METRIC_DIRECTION if m in df]:
        inst = -METRIC_DIRECTION[m] * df[m]

        def auc(pos, neg):
            sel = df.kind.isin(pos + neg)
            return roc_auc_score(df.kind[sel].isin(pos), inst[sel])

        r = dict(metric=m, label=METRIC_LABEL[m], auc_C_vs_A=auc(["C"], ["A"]), auc_B_vs_A=auc(["B"], ["A"]),
                 auc_D_vs_A=auc(["D"], ["A"]), auc_C_vs_ABD=auc(["C"], ["A", "B", "D"]))
        r.update({"rho_cons_" + c: spearmanr(inst, df["cons_" + c])[0] for c in DECISIONS})
        res.append(r)
    pd.DataFrame(res).to_csv(os.path.join(OUT, "discrimination.csv"), index=False)

    bounds = [dict(consequence=c, coverage=float((df[f"cons_{c}"] <= df[f"B_{c}"] + 1e-12).mean()),
                   informative=float((df[f"B_{c}"] < 1).mean()), rho_bound=spearmanr(df[f"B_{c}"], df[f"cons_{c}"])[0],
                   rho_cies=spearmanr(-df["cies"], df[f"cons_{c}"])[0]) for c in DECISIONS]
    pd.DataFrame(bounds).to_csv(os.path.join(OUT, "bounds.csv"), index=False)

    # stable but unfaithful explainers
    glob_importance = np.abs(phi_exact(rng.normal(0, 1, (5000, M)))).mean(0)
    noise_rng = np.random.default_rng(7)

    def scaffold(X):        # explains a surrogate that hides the tipping features
        Z = X.copy()
        Z[..., TIP] = 0.0
        return BETA * Z

    explainers = {"Exact SHAP (faithful)": phi_exact,
                  "Global importance (constant)": lambda X: np.broadcast_to(glob_importance, X.shape).copy(),
                  "Scaffolded surrogate (hides cut-offs)": scaffold,
                  "Noisy unbiased estimator": lambda X: phi_exact(X) + noise_rng.normal(0, 0.25, X.shape)}
    sub = df.sample(400, random_state=0)
    demo = []
    for name, fn in explainers.items():
        cs, fa, agree = [], [], []
        r = np.random.default_rng(11)
        for kind in sub.kind:
            x = sample_instance(kind, r)
            nb = x + r.normal(0, SIGMA, (K, M))
            phx, phn = fn(x[None, :])[0], fn(nb)
            cs.append(cies(phx, phn))
            fa.append(faithfulness(phx, x, r))
            agree.append(spearmanr(np.abs(phx), np.abs(phi_exact(x)))[0])
        demo.append(dict(explainer=name, CIES=np.nanmean(cs), faithfulness=np.nanmean(fa),
                         rank_agreement_with_truth=np.nanmean(agree)))
    pd.DataFrame(demo).to_csv(os.path.join(OUT, "stable_but_unfaithful.csv"), index=False)


PLANT_RANKS = [1, 2, 3, 5, 8, 12, 16, 19]
PLANT_METRICS = ["cies", "cies_unbounded", "cies_uniform", "avg_sens", "lip_mean", "ris", "topk3", "spearman",
                 "wkendall", "rbo", "cies_sym", "sri", "cies_star", "cies_v4", "cies_sym_b", "cies_star_b"]


def plant(phx, phn, kind, r, rng, delta=0.3, q=0.3):
    """Plant the same instability at the feature of rank r in a share q of the neighbours."""
    order = np.argsort(-np.abs(phx), kind="stable")
    j = order[r - 1]
    phn = phn.copy()
    hit = rng.random(len(phn)) < q
    if kind == "jitter":
        phn[:, j] += delta * rng.normal(0, 1, len(phn))
    elif kind == "signflip":
        phn[hit, j] = -phn[hit, j]
    elif kind == "swap":
        j2 = order[r]
        a, b = phn[hit, j].copy(), phn[hit, j2].copy()
        phn[hit, j], phn[hit, j2] = np.sign(a) * np.abs(b), np.sign(b) * np.abs(a)
    return phn


def controlled(n=300, seed=5):
    """Standardised response of every measure to an identical planted instability at different ranks."""
    rng = np.random.default_rng(seed)
    cache, base = [], []
    for x in rng.normal(0, 1, (n, M)):
        nb = x + rng.normal(0, SIGMA, (K, M))
        cache.append((x, nb, BETA * x, BETA * nb))
        base.append(stability_metrics(BETA * x, BETA * nb, x, nb, prob(x), prob(nb)))
    base = pd.DataFrame(base)
    rows = []
    for kind in ["jitter", "signflip", "swap"]:
        for r in PLANT_RANKS:
            v = pd.DataFrame([stability_metrics(phx, plant(phx, phn, kind, r, rng), x, nb, prob(x), prob(nb))
                              for x, nb, phx, phn in cache])
            for m in PLANT_METRICS:
                d = -METRIC_DIRECTION[m] * (v[m] - base[m])
                rows.append(dict(kind=kind, rank=r, metric=m, effect=d.mean() / (base[m].std() + 1e-12),
                                 detect_rate=float((d > 1e-9).mean())))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "controlled_planting.csv"), index=False)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    benchmark()
    controlled()
