"""Pre-registered monitoring task (preregistration/PREREGISTRATION_A3_monitoring.md).

Within each configuration an analyst reviews a fixed share of the explained instances. Rankings
are computed from the diagnostic neighbours, the target is the rate of explanation changes on the
independent neighbours. Development data: main (seeds 0-4); evaluation data: fresh (seeds 5-9).
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from analyze import CFG, TAB, load
from fragility.measures import DECISIONS
from fragility.stats import interval, strat_boot

K_DIAG = 20
BUDGETS = (0.05, 0.10, 0.20)
TARGETS = {"weighted": None, "sign3": "sign3", "reasons4": "reasons4"}
RANKERS = ["R1_cies", "R2_rbo", "R3_uniform", "R4_count", "R5_count_sign3", "R6_count_tiebreak", "R7_cies_rbo"]
HYPOTHESES = {"H1 count - CIES": ("R4_count", "R1_cies"), "H2 tiebreak - count": ("R6_count_tiebreak", "R4_count"),
              "H3 CIES - RBO": ("R1_cies", "R2_rbo"), "count - RBO": ("R4_count", "R2_rbo")}


def events(g, c):
    """Number of diagnostic neighbours in which consequence c changed (integers, so that equal counts tie exactly)."""
    return np.rint(g["count_" + c].astype(float) * K_DIAG).astype(int)


def scores(g, target):
    inst = 1 - g["cies"]
    count = sum(events(g, c) for c in DECISIONS) if target == "weighted" else events(g, target)
    return {"R1_cies": inst, "R2_rbo": 1 - g["rbo"], "R3_uniform": 1 - g["cies_uniform"], "R4_count": count,
            "R5_count_sign3": events(g, "sign3"), "R6_count_tiebreak": count + inst / 2,
            "R7_cies_rbo": (inst.rank() + (1 - g["rbo"]).rank()) / 2}


def capture(score, y, budget):
    """Share of the target held by the top `budget` share of instances; ties at the cut-off are allocated
    proportionally (the expectation under random tie-breaking)."""
    total = y.sum()
    if total <= 0:
        return np.nan
    slots = budget * len(y)
    levels = pd.DataFrame({"s": score, "y": y}).groupby("s")["y"].agg(["sum", "count"]).iloc[::-1]
    got = used = 0.0
    for level_sum, n in zip(levels["sum"], levels["count"]):
        if used + n <= slots + 1e-9:
            got, used = got + level_sum, used + n
        else:
            got += level_sum * (slots - used) / n
            break
    return got / total


def per_config(df, grid):
    rows = []
    for key, g in df.groupby(CFG):
        for name, c in TARGETS.items():
            y = sum(g["cons_" + d] for d in DECISIONS) / len(DECISIONS) if c is None else g["cons_" + c]
            r = dict(zip(CFG, key), grid=grid, target=name, total=float(y.sum()), prevalence=float((y > 0).mean()))
            positive = (y > 0).astype(int)
            for rk, s in scores(g, name).items():
                for b in BUDGETS:
                    r[f"{rk}|cap{int(b * 100)}"] = capture(s.values, y.values, b)
                r[f"{rk}|ap"] = average_precision_score(positive, s) if 0 < positive.sum() < len(positive) else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def summarise(P):
    caps, diffs = [], []
    for (grid, target), q in P.groupby(["grid", "target"]):
        ev = q[q.total > 0]
        for metric in [f"cap{int(b * 100)}" for b in BUDGETS] + ["ap"]:
            cols = [f"{rk}|{metric}" for rk in RANKERS]
            point, draws, _ = strat_boot(ev, cols)
            lo, hi = interval(draws)
            for j, rk in enumerate(RANKERS):
                caps.append(dict(grid=grid, target=target, metric=metric, ranker=rk, value=point[j], lo=lo[j], hi=hi[j],
                                 n_configs=int(ev[cols[j]].notna().sum()), n_excluded=int((q.total <= 0).sum())))
            f = ev[["dataset", "seed"]].copy()
            for label, (a, b) in HYPOTHESES.items():
                f[label] = ev[f"{a}|{metric}"] - ev[f"{b}|{metric}"]
            point, draws, _ = strat_boot(f, list(HYPOTHESES))
            lo, hi = interval(draws)
            for j, label in enumerate(HYPOTHESES):
                diffs.append(dict(grid=grid, target=target, metric=metric, comparison=label, diff=point[j],
                                  lo=lo[j], hi=hi[j]))
    pd.DataFrame(caps).to_csv(os.path.join(TAB, "T_monitoring_capture.csv"), index=False)
    pd.DataFrame(diffs).to_csv(os.path.join(TAB, "T_monitoring_diffs.csv"), index=False)


if __name__ == "__main__":
    os.makedirs(TAB, exist_ok=True)
    summarise(pd.concat([per_config(load(g), g) for g in ["main", "fresh", "mixedfresh"]], ignore_index=True))
