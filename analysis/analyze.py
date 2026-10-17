"""Tables (tables/*.csv) and figures (figures/*.png) of the paper, from the per-instance results."""
import json
import os
from collections import Counter

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr
from scipy.stats import t as tdist

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from fragility.measures import CONSEQUENCES, DECISIONS, METRIC_DIRECTION, METRIC_LABEL  # noqa: E402
from fragility.pipeline import ALL_MODELS, DATASETS, DEFAULT_PARAMS, FIGURES, RESULTS, TABLES  # noqa: E402
from fragility.stats import boot_ci, interval, strat_boot, t_halfwidth  # noqa: E402

TAB, FIG = TABLES, FIGURES
CFG = ["dataset", "seed", "strategy", "model", "variant", "noise", "eps"]
DEC_CONS = [f"cons_{c}" for c in DECISIONS]

MEASURES = ["cies", "cies_unbounded", "cies_uniform", "cies_nbw", "avg_sens", "max_sens", "lip_max", "lip_mean",
            "ris", "ros", "topk3", "topk5", "spearman", "kendall", "wkendall", "rbo", "pred_stab",
            "cies_sym", "sri", "cies_star", "cies_v4", "cies_sym_b", "cies_star_b", "cies_rbo"]
COUNTS = ["count_sign3", "count_matched"]

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MODEL_COLOR = {"Random Forest": "#2a78d6", "XGBoost": "#eb6834", "LightGBM": "#1baf7a",
               "CatBoost": "#eda100", "Logistic Regression": "#e87ba4"}
MODEL_MARK = {"Random Forest": "o", "XGBoost": "s", "LightGBM": "^", "CatBoost": "D", "Logistic Regression": "v"}
MODEL_SHORT = {"Random Forest": "RF", "XGBoost": "XGB", "LightGBM": "LGBM", "CatBoost": "CB", "Logistic Regression": "LR"}
STRAT_LABEL = {"none": "None (raw)", "smote": "SMOTE", "smotenc": "SMOTE-NC",
               "class_weight": "Class weights", "undersample": "Undersampling"}
CONS_LABEL = {"cons_top1": "Top-1 feature changes", "cons_top3set": "Top-3 set changes",
              "cons_sign3": "Top-3 direction reverses", "cons_reasons4": "Adverse-action reasons change",
              "cons_action": "Primary risk driver changes", "cons_predflip": "Predicted class flips"}
DS_LABEL = {"telco": "Telco Customer Churn", "german_credit": "German Credit Risk", "hr_attrition": "IBM HR Attrition"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.titleweight": "bold", "axes.titlesize": 9.5, "axes.axisbelow": True, "axes.titlecolor": INK,
    "figure.dpi": 150, "savefig.dpi": 300, "lines.linewidth": 2.0, "lines.markersize": 5,
})


def load(grid):
    return pd.read_csv(os.path.join(RESULTS, f"{grid}.csv"))


def save_table(df, name, index=False):
    df.to_csv(os.path.join(TAB, name), index=index)


def save_fig(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def instability(df, m):
    return -METRIC_DIRECTION[m] * df[m]


# ------------------------------------------------------------------ models (RQ1, RQ3)
def model_summary(df, tag):
    seed_means = df.groupby(["dataset", "variant", "strategy", "model", "seed"]).agg(
        cies=("cies", "mean"), f1=("perf_f1", "first"), roc_auc=("perf_roc_auc", "first"),
        pr_auc=("perf_pr_auc", "first")).reset_index()
    keys = ["dataset", "variant", "strategy", "model"]
    out = seed_means.groupby(keys).agg(n_seeds=("seed", "nunique"), cies_mean=("cies", "mean"),
                                       cies_sd_seeds=("cies", "std"), f1=("f1", "mean"),
                                       roc_auc=("roc_auc", "mean"), pr_auc=("pr_auc", "mean")).reset_index()
    p05 = df.groupby(keys)["cies"].quantile(0.05).rename("cies_p05").reset_index()
    out = out.merge(p05, on=keys)
    save_table(out, f"T_summary_{tag}.csv")

    ranks, fried, agree = [], [], []
    for var, sm in seed_means.groupby("variant"):
        piv = sm.pivot_table(index=["dataset", "strategy", "seed"], columns="model", values="cies").dropna()
        trees = [c for c in piv.columns if c != "Logistic Regression"]
        best = piv.idxmax(axis=1)
        rk = piv[trees].rank(axis=1, ascending=False)
        for mdl in piv.columns:
            ranks.append(dict(variant=var, model=mdl, share_best_all_models=(best == mdl).mean(),
                              mean_rank_trees=rk[mdl].mean() if mdl in trees else np.nan, n_blocks=len(piv)))
        st, p = friedmanchisquare(*[piv[c] for c in trees])
        fried.append(dict(variant=var, friedman_chi2_trees=st, p=p, n_blocks=len(piv)))
    if seed_means.variant.nunique() == 2:
        tree_means = seed_means[seed_means.model != "Logistic Regression"]
        mm = tree_means.groupby(["dataset", "strategy", "variant", "model"]).cies.mean().unstack("variant")
        for (d, s), g in mm.groupby(level=[0, 1]):
            agree.append(dict(dataset=d, strategy=s, spearman_default_vs_tuned=spearmanr(g["default"], g["tuned"])[0]))
    save_table(pd.DataFrame(ranks), f"T_model_ranks_{tag}.csv")
    save_table(pd.DataFrame(fried), f"T_friedman_{tag}.csv")
    save_table(pd.DataFrame(agree), f"T_rank_agreement_default_tuned_{tag}.csv")
    return out


def fig_models(out, tag):
    for var in out.variant.unique():
        o = out[out.variant == var]
        strategies = [s for s in STRAT_LABEL if s in o.strategy.unique()]
        x = np.arange(len(strategies))
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
        for ax, ds in zip(axes, DATASETS):
            od = o[o.dataset == ds]
            for i, mdl in enumerate(ALL_MODELS):
                om = od[od.model == mdl].set_index("strategy").reindex(strategies)
                err = tdist.ppf(0.975, np.maximum(om.n_seeds - 1, 1)) * om.cies_sd_seeds / np.sqrt(om.n_seeds)
                ax.errorbar(x + (i - 2) * 0.12, om.cies_mean, yerr=[err], fmt=MODEL_MARK[mdl], ms=5,
                            color=MODEL_COLOR[mdl], elinewidth=1.2, capsize=0, mec="white", mew=0.6, label=mdl)
            ax.set_title(DS_LABEL[ds])
            ax.set_xticks(x)
            ax.set_xticklabels([STRAT_LABEL[s] for s in strategies], rotation=30, ha="right")
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel("Mean CIES (95% CI over 5 seeds)")
        axes[-1].legend(loc="lower left", bbox_to_anchor=(1.0, 0.0), fontsize=8)
        save_fig(fig, f"F_models_strategies_{tag}_{var}.png")


def fig_tradeoff(out):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, ds in zip(axes, DATASETS):
        g = out[out.dataset == ds]
        for mdl in ALL_MODELS:
            h = g[g.model == mdl]
            dft = h[h.variant == "default"].set_index("strategy")
            tun = h[h.variant == "tuned"].set_index("strategy")
            for s in dft.index.intersection(tun.index):
                ax.plot([dft.roc_auc[s], tun.roc_auc[s]], [dft.cies_mean[s], tun.cies_mean[s]],
                        color=MODEL_COLOR[mdl], lw=0.8, alpha=0.45)
            ax.scatter(dft.roc_auc, dft.cies_mean, marker=MODEL_MARK[mdl], s=26, facecolors="white",
                       edgecolors=MODEL_COLOR[mdl], linewidths=1.2)
            ax.scatter(tun.roc_auc, tun.cies_mean, marker=MODEL_MARK[mdl], s=30, color=MODEL_COLOR[mdl],
                       edgecolors="white", linewidths=0.6, label=mdl)
        ax.set_title(DS_LABEL[ds])
        ax.set_xlabel("ROC-AUC (mean over 5 seeds)")
    axes[0].set_ylabel("Mean CIES")
    handles = [Line2D([], [], marker=MODEL_MARK[m], color=MODEL_COLOR[m], ls="", label=m) for m in ALL_MODELS]
    handles += [Line2D([], [], marker="o", mfc="white", mec=INK2, ls="", label="Default hyper-parameters"),
                Line2D([], [], marker="o", color=INK2, ls="", label="Tuned hyper-parameters")]
    axes[-1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    save_fig(fig, "F_tradeoff.png")


def hyperparameter_table():
    params = json.load(open(os.path.join(RESULTS, "tuned_params.json")))
    auc = json.load(open(os.path.join(RESULTS, "tuned_cv_auc.json")))
    rows = []
    for ds in params:
        seeds = [s for s in sorted(params[ds]) if int(s) < 5]
        for mdl in params[ds]["0"]:
            mode, n = Counter(json.dumps(params[ds][s][mdl], sort_keys=True) for s in seeds).most_common(1)[0]
            aucs = [auc[ds][s][mdl] for s in seeds]
            rows.append(dict(dataset=ds, model=mdl, default=str(DEFAULT_PARAMS[mdl]), tuned=str(json.loads(mode)),
                             splits_with_this_config=n, cv_auc_min=min(aucs), cv_auc_max=max(aucs)))
    save_table(pd.DataFrame(rows), "T_hyperparameters.csv")


# ------------------------------------------------------------------ imbalance (RQ2)
def imbalance_effects(df, tag):
    keys = ["dataset", "variant", "model", "seed", "pos"]
    base = df[df.strategy == "none"].set_index(keys)
    rows = []
    for s in [s for s in STRAT_LABEL if s != "none"]:
        j = df[df.strategy == s].set_index(keys).join(base, rsuffix="_base", how="inner").reset_index()
        j["d_cies"] = j.cies - j.cies_base
        per_seed = j.groupby(["dataset", "variant", "model", "seed"]).agg(
            d_cies=("d_cies", "mean"), f1=("perf_f1", "first"), f1_base=("perf_f1_base", "first"),
            pr=("perf_pr_auc", "first"), pr_base=("perf_pr_auc_base", "first")).reset_index()
        for (d, v, mdl), g in per_seed.groupby(["dataset", "variant", "model"]):
            rows.append(dict(dataset=d, variant=v, model=mdl, strategy=s, d_cies=g.d_cies.mean(),
                             d_cies_ci=t_halfwidth(g.d_cies), d_f1=(g.f1 - g.f1_base).mean(),
                             d_pr_auc=(g.pr - g.pr_base).mean()))
    out = pd.DataFrame(rows)
    save_table(out, f"T_imbalance_effects_{tag}.csv")

    o = out[out.variant == "default"]
    strategies = [s for s in STRAT_LABEL if s != "none"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        od = o[o.dataset == ds]
        for i, mdl in enumerate(ALL_MODELS):
            om = od[od.model == mdl].set_index("strategy").reindex(strategies)
            ax.errorbar(np.arange(len(strategies)) + (i - 2) * 0.12, om.d_cies, yerr=om.d_cies_ci,
                        fmt=MODEL_MARK[mdl], color=MODEL_COLOR[mdl], ms=5, elinewidth=1.2, mec="white", mew=0.6,
                        label=mdl)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_title(DS_LABEL[ds])
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels([STRAT_LABEL[s] for s in strategies], rotation=30, ha="right")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Change in CIES vs. no treatment\n(paired, 95% CI over seeds)")
    axes[-1].legend(loc="lower left", bbox_to_anchor=(1.0, 0.0), fontsize=8)
    save_fig(fig, f"F_imbalance_effects_{tag}.png")


def imbalance_intervals(df, tag):
    """Paired change in mean CIES and in the per-configuration 5th percentile, pooled and per subgroup."""
    keys = ["dataset", "variant", "model", "seed", "pos"]
    base = df[df.strategy == "none"].set_index(keys)["cies"]
    p5 = df.groupby(["dataset", "variant", "model", "seed", "strategy"]).cies.quantile(0.05)
    rows = []
    for s in ["smote", "smotenc", "class_weight", "undersample"]:
        d = (df[df.strategy == s].set_index(keys)["cies"] - base).dropna().rename("d").reset_index()
        q = (p5.xs(s, level="strategy") - p5.xs("none", level="strategy")).rename("d5").reset_index()
        groups = [("all", "all", d, q)] + [(v, "all", d[d.variant == v], q[q.variant == v]) for v in ["default", "tuned"]]
        groups += [(v, m, d[(d.variant == v) & (d.model == m)], q[(q.variant == v) & (q.model == m)])
                   for v in ["default", "tuned"] for m in ALL_MODELS]
        for var, mdl, dd, qq in groups:
            m, lo, hi, n = boot_ci(dd, "d")
            m5, lo5, hi5, _ = boot_ci(qq, "d5")
            rows.append(dict(strategy=s, variant=var, model=mdl, mean_change=m, ci_lo=lo, ci_hi=hi,
                             p05_change=m5, p05_lo=lo5, p05_hi=hi5, n_clusters=n))
    save_table(pd.DataFrame(rows), f"T_imbalance_cis_{tag}.csv")


# ------------------------------------------------------------------ predictive validity (RQ4)
def per_config_validity(df):
    """Spearman correlation between instability on the diagnostic neighbours and every consequence
    on the independent neighbours, per configuration. Constant predictors or outcomes are skipped."""
    rows = []
    for key, g in df.groupby(CFG):
        r = dict(zip(CFG, key))
        g = g.assign(cies_rbo=-(instability(g, "cies").rank() + instability(g, "rbo").rank()) / 2,
                     retest_cies_rbo=-((-g["retest_cies"]).rank() + (-g["retest_rbo"]).rank()) / 2)
        for m in MEASURES + COUNTS:
            if (m if m != "count_matched" else "count_sign3") not in g:
                continue
            for c in CONSEQUENCES:
                x = g["count_" + c[5:]] if m == "count_matched" else instability(g, m)
                if x.nunique() > 1 and g[c].nunique() > 1:
                    r[f"{m}|{c}"] = spearmanr(x, g[c])[0]
            if "retest_" + m in g and g["retest_" + m].nunique() > 1:
                r[f"{m}|retest"] = spearmanr(g[m], g["retest_" + m])[0]
        rows.append(r)
    return pd.DataFrame(rows)


def validity(df, tag):
    per = per_config_validity(df)
    save_table(per, f"T_validity_per_config_{tag}.csv")
    rows = []
    for m in MEASURES + COUNTS:
        r = dict(metric=m, label=METRIC_LABEL.get(m, m))
        for c in CONSEQUENCES + ["retest"]:
            if f"{m}|{c}" in per:
                r[c] = per[f"{m}|{c}"].mean()
                r["n_" + c] = int(per[f"{m}|{c}"].notna().sum())
        r["mean_decision"] = np.nanmean([r.get(c, np.nan) for c in DEC_CONS])
        rows.append(r)
    summary = pd.DataFrame(rows)
    save_table(summary, f"T_validity_summary_{tag}.csv")
    return per, summary


def fig_validity(summary, tag):
    cols = DEC_CONS + ["cons_predflip", "retest"]
    order = ["cies", "cies_unbounded", "cies_rbo", "cies_uniform", "cies_nbw", "avg_sens", "max_sens", "lip_mean",
             "lip_max", "ris", "ros", "topk3", "topk5", "spearman", "kendall", "wkendall", "rbo", "pred_stab"] + COUNTS
    s = summary.set_index("metric")[cols].reindex(order)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    im = ax.imshow(s.values, cmap=LinearSegmentedColormap.from_list("div", ["#d03b3b", "#f0efec", "#2a78d6"]),
                   vmin=-0.8, vmax=0.8, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([CONS_LABEL.get(c, "Test-retest reliability") for c in cols], rotation=35, ha="right")
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels([METRIC_LABEL[m] for m in s.index])
    ax.grid(False)
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            v = s.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=INK if abs(v) < 0.5 else "white")
    ax.axvline(len(cols) - 1.5, color="white", lw=3)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Mean Spearman correlation (instability vs. outcome)", color=INK2)
    cb.outline.set_visible(False)
    save_fig(fig, f"F_validity_heatmap_{tag}.png")


def _paired(per, other):
    """CIES minus `other` per configuration and consequence, where both are defined."""
    out = per[["dataset", "seed"]].copy()
    for c in DEC_CONS:
        a, b = f"cies|{c}", f"{other}|{c}"
        out[c] = per[a] - per[b] if a in per and b in per else np.nan
    return out


def paired_differences(per, tag, others=("cies_uniform", "avg_sens", "lip_mean", "ris", "ros", "topk3", "wkendall",
                                         "rbo", "pred_stab", "count_sign3", "count_matched")):
    """Per consequence: mean difference over the configurations where both are defined. Summary: unweighted
    mean of the five consequence-specific differences, as in the validity table."""
    rows = []
    for o in others:
        f = _paired(per, o)
        point, draws, n = strat_boot(f, DEC_CONS)
        lo, hi = interval(draws)
        for j, c in enumerate(DEC_CONS):
            dd = f[c].dropna()
            rows.append(dict(other=o, outcome=c, diff=point[j], ci_lo=lo[j], ci_hi=hi[j], n_clusters=n,
                             n_configs=len(dd), share_cies_better=float((dd > 0).mean()) if len(dd) else np.nan))
        md = np.nanmean(draws, axis=1)
        rows.append(dict(other=o, outcome="mean_decision", diff=float(np.nanmean(point)), ci_lo=np.percentile(md, 2.5),
                         ci_hi=np.percentile(md, 97.5), n_clusters=n,
                         n_configs=int(f[DEC_CONS].notna().any(axis=1).sum()), share_cies_better=np.nan))
    save_table(pd.DataFrame(rows), f"T_paired_cis_{tag}.csv")


def paired_by_dataset(per, tag, others=("cies_uniform", "rbo", "count_sign3", "count_matched", "topk3")):
    rows = []
    j = DEC_CONS.index("cons_sign3")
    for o in others:
        f = _paired(per, o)
        subsets = [(d, f[f.dataset == d]) for d in DATASETS] + [("without " + d, f[f.dataset != d]) for d in DATASETS]
        for label, sub in subsets:
            point, draws, _ = strat_boot(sub, DEC_CONS)
            md = np.nanmean(draws, axis=1)
            rows.append(dict(other=o, subset=label, mean_decision=float(np.nanmean(point)),
                             md_lo=np.percentile(md, 2.5), md_hi=np.percentile(md, 97.5), sign3=point[j],
                             sign3_lo=np.nanpercentile(draws[:, j], 2.5), sign3_hi=np.nanpercentile(draws[:, j], 97.5),
                             n_sign3=int(sub["cons_sign3"].notna().sum())))
    save_table(pd.DataFrame(rows), f"T_paired_by_dataset_{tag}.csv")


def counts_vs_cies(df, tag):
    """CIES against the direct count of the same consequence on the first K diagnostic neighbours."""
    def rho(x, y):
        return x.corr(y, method="spearman") if x.nunique() > 1 and y.nunique() > 1 else np.nan

    cols = {5: ("cies_d5", "cnt_d5_"), 10: ("cies_d10", "cnt_d10_"), 20: ("cies", "count_")}
    rows = []
    for key, g in df.groupby(CFG):
        r = dict(zip(CFG, key))
        for K, (cc, cnt) in cols.items():
            for c in DECISIONS:
                y, n = g["cons_" + c], g[cnt + c]
                r[f"cies|{K}|{c}"] = rho(-g[cc], y)
                r[f"count|{K}|{c}"] = rho(n, y)
                zero = n == 0
                r[f"zero|{K}|{c}"] = rho(-g[cc][zero], y[zero]) if zero.sum() >= 10 else np.nan
                r[f"zero_share|{K}|{c}"] = zero.mean()
        rows.append(r)
    P = pd.DataFrame(rows)
    out, zeros = [], []
    for K in cols:
        f = P[["dataset", "seed"]].copy()
        for c in DECISIONS:
            both = P[f"cies|{K}|{c}"].notna() & P[f"count|{K}|{c}"].notna()
            f[c] = (P[f"cies|{K}|{c}"] - P[f"count|{K}|{c}"]).where(both)
            f["a_" + c] = P[f"cies|{K}|{c}"].where(both)
            f["b_" + c] = P[f"count|{K}|{c}"].where(both)
        point, draws, _ = strat_boot(f, DECISIONS)
        pa = strat_boot(f, ["a_" + c for c in DECISIONS])[0]
        pb = strat_boot(f, ["b_" + c for c in DECISIONS])[0]
        lo, hi = interval(draws)
        for j, c in enumerate(DECISIONS):
            out.append(dict(K=K, outcome=c, rho_cies=pa[j], rho_count=pb[j], diff=point[j], lo=lo[j], hi=hi[j],
                            n_configs=int(f[c].notna().sum())))
        md = np.nanmean(draws, axis=1)
        out.append(dict(K=K, outcome="mean", rho_cies=np.nanmean(pa), rho_count=np.nanmean(pb),
                        diff=np.nanmean(point), lo=np.percentile(md, 2.5), hi=np.percentile(md, 97.5),
                        n_configs=int(f[DECISIONS].notna().any(axis=1).sum())))
        for c in DECISIONS:
            p, dr, _ = strat_boot(P, [f"zero|{K}|{c}"])
            zeros.append(dict(K=K, outcome=c, share_zero_count=P[f"zero_share|{K}|{c}"].mean(),
                              rho_cies_within_zero=p[0], lo=np.nanpercentile(dr[:, 0], 2.5),
                              hi=np.nanpercentile(dr[:, 0], 97.5), n_configs=int(P[f"zero|{K}|{c}"].notna().sum())))
    save_table(pd.DataFrame(out), f"T_counts_vs_cies_{tag}.csv")
    save_table(pd.DataFrame(zeros), f"T_counts_zero_{tag}.csv")


# ------------------------------------------------------------------ calibration and secondary analyses
BINS = [0, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 1.0001]


def calibration(df, tag):
    d = df[df.variant == "default"].copy()
    d["bin"] = pd.cut(d.cies, BINS, right=False)
    cal = d.groupby(["dataset", "bin"], observed=True).agg(n=("cies", "size"),
                                                          **{c: (c, "mean") for c in CONSEQUENCES}).reset_index()
    save_table(cal, f"T_calibration_bins_{tag}.csv")
    rows = []
    for ds, g in d.groupby("dataset"):
        for tau in [0.7, 0.8, 0.9, 0.95]:
            flag = g.cies < tau
            r = dict(dataset=ds, tau=tau, share_flagged=flag.mean(),
                     cons_reasons4_flagged=g.loc[flag, "cons_reasons4"].mean(),
                     cons_reasons4_not=g.loc[~flag, "cons_reasons4"].mean())
            gaps = []
            for s in g.seed.unique():
                h = g[g.seed != s]
                f = h.cies < tau
                if f.any() and (~f).any():
                    gaps.append(h.loc[f, "cons_reasons4"].mean() - h.loc[~f, "cons_reasons4"].mean())
            r["reasons4_gap_loso_min"] = np.min(gaps) if gaps else np.nan
            r["reasons4_gap_loso_max"] = np.max(gaps) if gaps else np.nan
            rows.append(r)
    save_table(pd.DataFrame(rows), f"T_thresholds_{tag}.csv")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    styles = list(zip(["cons_reasons4", "cons_top3set", "cons_action", "cons_sign3"],
                      ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], ["o", "s", "^", "D"]))
    for ax, ds in zip(axes, DATASETS):
        c = cal[(cal.dataset == ds) & (cal.n >= 30)]
        mids = [(b.left + min(b.right, 1.0)) / 2 for b in c.bin]
        for cons, color, mark in styles:
            ax.plot(mids, c[cons], marker=mark, color=color, label=CONS_LABEL[cons], ms=5, mec="white", mew=0.6)
        ax.set_title(DS_LABEL[ds])
        ax.set_xlabel("CIES (binned)")
        ax.set_xlim(0.55, 1.0)
    axes[0].set_ylabel("Probability under an independent\nre-measurement of the instance")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    save_fig(fig, f"F_calibration_{tag}.png")


def secondary(df, tag):
    d = df[df.variant == "default"]
    trees = d[d.model != "Logistic Regression"]
    cols = ["cies", "cies_uniform", "avg_sens", "lip_mean", "lip_max", "topk3", "rbo", "wkendall", "pred_stab"]
    save_table(trees.groupby("model")[cols].mean(), f"T_comparator_means_{tag}.csv", index=True)

    wcols = ["cies", "cies_exp", "cies_log", "cies_top5", "cies_uniform", "cies_nbw"]
    blocks = d.groupby(["dataset", "strategy", "seed", "model"])[wcols].mean()
    agree = [dict(scheme=w, rho=spearmanr(g["cies"], g[w])[0])
             for _, g in blocks.groupby(level=[0, 1, 2]) for w in wcols[1:]]
    save_table(pd.DataFrame(agree).groupby("scheme").rho.agg(["mean", "min"]), f"T_weighting_rank_agreement_{tag}.csv",
               index=True)

    conv = {f"K{k}": (d[f"cies_K{k}"] - d["cies_K30"]).abs().mean() for k in (5, 10, 20)}
    conv["retest_spearman_K20"] = d.groupby(CFG).apply(lambda g: spearmanr(g.cies, g.retest_cies)[0]).mean()
    save_table(pd.DataFrame([conv]), f"T_K_convergence_{tag}.csv")
    save_table(d.groupby(["dataset", "model"]).runtime_per_instance.mean().unstack(), f"T_runtime_{tag}.csv", index=True)
    smooth = d.groupby(CFG).apply(lambda g: spearmanr(g.cies, g.pred_stab)[0]).rename("rho").reset_index()
    save_table(smooth.groupby("model").rho.agg(["mean", "std"]), f"T_smoothness_{tag}.csv", index=True)


def tree_ranks(df):
    tr = df[df.model != "Logistic Regression"]
    m = tr.groupby(["variant", "dataset", "strategy", "seed", "model"]).cies.mean().reset_index()
    m["rank"] = m.groupby(["variant", "dataset", "strategy", "seed"]).cies.rank(ascending=False)
    save_table(m.groupby(["variant", "model"])["rank"].mean().unstack(0), "T_model_ranks_trees_main.csv", index=True)


# ------------------------------------------------------------------ noise models (RQ5)
NOISE_LABEL = {"gauss_mult": "Gaussian, proportional", "gauss_sd": "Gaussian, SD-scaled",
               "laplace_mult": "Laplace, proportional", "mixed": "Mixed (numeric + mis-recorded categories)"}


def noise_sensitivity(df):
    m = df.groupby(["dataset", "strategy", "noise", "eps", "model", "seed"]).cies.mean().reset_index()
    tab = m.groupby(["dataset", "strategy", "noise", "eps", "model"]).cies.mean().unstack("model")
    save_table(tab, "T_noise_sensitivity.csv", index=True)
    ref = tab.xs(("gauss_mult", 0.03), level=("noise", "eps"))
    agree = [dict(dataset=ds, strategy=st, noise=nz, eps=e,
                  spearman_vs_reference=spearmanr(row.values, ref.loc[(ds, st)].values)[0])
             for (ds, st, nz, e), row in tab.iterrows()]
    save_table(pd.DataFrame(agree), "T_noise_rank_agreement.csv")

    d = m[m.strategy == "none"].groupby(["dataset", "noise", "eps", "model"]).cies.mean().reset_index()
    fig, axes = plt.subplots(3, 4, figsize=(11, 7.2), sharex=True, sharey="row")
    for i, ds in enumerate(DATASETS):
        for j, nz in enumerate(NOISE_LABEL):
            ax = axes[i, j]
            for mdl in ALL_MODELS:
                g = d[(d.dataset == ds) & (d.noise == nz) & (d.model == mdl)]
                ax.plot(g.eps, g.cies, marker=MODEL_MARK[mdl], color=MODEL_COLOR[mdl], label=mdl, ms=4.5,
                        mec="white", mew=0.5)
            if i == 0:
                ax.set_title(NOISE_LABEL[nz], fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"{DS_LABEL[ds]}\nmean CIES")
            if i == 2:
                ax.set_xlabel("Noise level ε")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    save_fig(fig, "F_noise_sensitivity.png")


# ------------------------------------------------------------------ synthetic benchmark and LIME
SYN_METRICS = ["cies", "cies_uniform", "avg_sens", "lip_mean", "topk3", "spearman", "wkendall", "rbo"]
SYN_COLOR = dict(zip(SYN_METRICS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]))
SYN_MARK = dict(zip(SYN_METRICS, ["o", "s", "^", "D", "v", "P", "X", "*"]))


def synthetic_figures():
    cp = pd.read_csv(os.path.join(RESULTS, "synthetic", "controlled_planting.csv"))
    titles = {"jitter": "(a) Same-size magnitude jitter", "signflip": "(b) Direction reversal, |φ| unchanged",
              "swap": "(c) Order swap with the next-ranked feature"}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    for ax, kind in zip(axes, titles):
        for m in SYN_METRICS:
            g = cp[(cp.kind == kind) & (cp.metric == m)].sort_values("rank")
            top = g.effect.abs().max()
            y = g.effect / top if top >= 0.1 else g.effect * 0   # responses below 0.1 SD drawn at zero
            ax.plot(g["rank"], y, marker=SYN_MARK[m], color=SYN_COLOR[m], label=METRIC_LABEL[m], ms=5,
                    mec="white", mew=0.5, lw=1.6)
        ax.set_title(titles[kind])
        ax.set_xlabel("Rank of the feature carrying the planted instability")
        ax.set_xticks([1, 2, 3, 5, 8, 12, 16, 19])
    axes[0].set_ylabel("Response (share of the measure's\nmaximum response)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7.5)
    save_fig(fig, "F_synthetic_controlled.png")

    dis = pd.read_csv(os.path.join(RESULTS, "synthetic", "discrimination.csv")).set_index("metric").reindex(SYN_METRICS)
    panels = [("auc_C_vs_A", "Consequential tipping\n(top feature) vs. stable"),
              ("auc_B_vs_A", "Inconsequential tipping\n(bottom feature) vs. stable"),
              ("auc_D_vs_A", "Near-tie at the top\nvs. stable")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0), sharey=True)
    for ax, (c, title) in zip(axes, panels):
        y = np.arange(len(dis))
        ax.barh(y, dis[c] - 0.5, left=0.5, color=[SYN_COLOR[m] for m in dis.index], height=0.6)
        for yi, v in zip(y, dis[c]):
            ax.text(max(v, 0.5) + 0.01, yi, f"{v:.2f}", va="center", fontsize=7.5, color=INK2)
        ax.axvline(0.5, color=AXIS, lw=1)
        ax.set_xlim(0.35, 1.08)
        ax.set_title(title)
        ax.set_xlabel("AUC")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(dis)))
    axes[0].set_yticklabels([METRIC_LABEL[m] for m in dis.index])
    axes[0].invert_yaxis()
    save_fig(fig, "F_synthetic_discrimination.png")


def lime_figures():
    d = pd.read_csv(os.path.join(RESULTS, "lime.csv"))
    t = d[d.condition != "Visani indices"].groupby(["dataset", "model", "num_samples", "condition"])[
        ["cies", "rbo", "topk3"]].mean().reset_index()
    save_table(t, "T_lime_conditions.csv")
    save_table(d[d.condition == "Visani indices"].groupby(["dataset", "model"])[["vsi", "csi"]].mean(),
               "T_lime_visani.csv", index=True)
    conditions = {"L0 repeat (no noise)": "#898781", "L1 noise, fixed seed": "#2a78d6",
                  "L2 noise, varied seed": "#eb6834"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    x = np.arange(len(ALL_MODELS))
    for ax, ns in zip(axes, [500, 5000]):
        g = t[t.num_samples == ns]
        for i, (c, color) in enumerate(conditions.items()):
            v = g[g.condition == c].groupby("model").cies.mean().reindex(ALL_MODELS)
            ax.bar(x + (i - 1) * 0.26, v.values, width=0.24, color=color, label=c)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_SHORT[m] for m in ALL_MODELS])
        ax.set_title(f"LIME, {ns} samples per explanation")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Mean CIES (3 datasets)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    save_fig(fig, "F_lime_conditions.png")


# ------------------------------------------------------------------ pre-registered analyses
def registered_validity(d, measures):
    """Registered rule: mean over the five consequences of the configuration-averaged correlation."""
    rows = []
    for _, g in d.groupby(CFG):
        r = {}
        for m in measures:
            x = instability(g, m)
            for c in DECISIONS:
                r[f"{m}|{c}"] = x.corr(g[f"cons_{c}"], method="spearman") if g[f"cons_{c}"].nunique() > 1 else np.nan
            vals = [r[f"{m}|{c}"] for c in DECISIONS]
            r[m] = np.nanmean(vals) if any(pd.notna(v) for v in vals) else np.nan
            r[m + "|rt"] = g[m].corr(g["retest_" + m], method="spearman")
        rows.append(r)
    R = pd.DataFrame(rows)
    return {m: (np.mean([R[f"{m}|{c}"].mean() for c in DECISIONS]), R[m + "|rt"].mean()) for m in measures}, R


def preregistered_variants(main, mixedfresh):
    measures = ["cies_unbounded", "cies_sym", "sri", "cies_star", "cies_v4", "cies", "cies_sym_b", "cies_star_b", "rbo"]
    names = {"cies_unbounded": "V0 unbounded CIES (ratio of averaged distance)", "cies_sym": "V1 symmetric weights",
             "sri": "V2 signed-rank index", "cies_star": "V3 average of V1 and V2", "cies_v4": "V4 average of V0 and V2",
             "cies": "V0b bounded CIES (adopted)", "cies_sym_b": "V1b bounded symmetric weights",
             "cies_star_b": "V3b average of V1b and V2", "rbo": "RBO (reference)"}
    dev, _ = registered_validity(main[main.seed.isin([0, 1, 2])], measures)
    con, R = registered_validity(main[main.seed.isin([3, 4])], measures)
    fresh, _ = registered_validity(mixedfresh, measures)
    syn = pd.read_csv(os.path.join(RESULTS, "synthetic", "discrimination.csv")).set_index("metric")
    rows = [dict(Measure=names[m], dev=dev[m][0], confirm=con[m][0],
                 confirm_better_than_V0=(R[m] > R["cies_unbounded"]).mean() if m != "cies_unbounded" else np.nan,
                 fresh_mixed=fresh[m][0], fresh_mixed_retest=fresh[m][1],
                 syn_C=syn.loc[m, "auc_C_vs_A"], syn_B=syn.loc[m, "auc_B_vs_A"]) for m in measures]
    save_table(pd.DataFrame(rows), "T_prereg_variants.csv")


def consequence_bound_tables(main, mixed, mixedfresh, K=20, delta=0.05):
    rows = []
    for label, d in {"Main grid, seeds 3-4": main[main.seed.isin([3, 4])], "Mixed noise": mixed,
                     "Mixed noise, seeds 5-9": mixedfresh}.items():
        for c in DECISIONS:
            B, obs = d[f"B_{c}"], d[f"cons_{c}"]
            rho_b = [g[f"B_{c}"].corr(g[f"cons_{c}"], method="spearman") for _, g in d.groupby(CFG)
                     if g[f"cons_{c}"].nunique() > 1 and g[f"B_{c}"].nunique() > 1]
            rho_c = [(-g["cies"]).corr(g[f"cons_{c}"], method="spearman") for _, g in d.groupby(CFG)
                     if g[f"cons_{c}"].nunique() > 1]
            rows.append(dict(data=label, consequence=c, coverage=(obs <= B + 1e-12).mean(), informative=(B < 1).mean(),
                             rho_bound=np.nanmean(rho_b), rho_cies=np.nanmean(rho_c)))
    save_table(pd.DataFrame(rows), "T_bounds_A2.csv")

    allowance = np.sqrt(np.log(1 / delta) / (2 * K))
    rows = []
    for c in DECISIONS:
        mu, B, obs = main[f"mu_{c}"], main[f"B_{c}"], main[f"cons_{c}"]
        Bh = np.where(mu <= 0, 1.0, np.minimum(1.0, (1 - main.cies + allowance) / np.minimum(1.0, mu)))
        rows.append(dict(consequence=c, zero_margin=(mu <= 0).mean(), coverage_plugin=(obs <= B + 1e-12).mean(),
                         informative_plugin=(B < 1).mean(), coverage_hoeffding=(obs <= Bh + 1e-12).mean(),
                         informative_hoeffding=(Bh < 1).mean(), median_margin=np.median(np.minimum(1.0, mu))))
    save_table(pd.DataFrame(rows), "T_bounds_finite_sample.csv")


if __name__ == "__main__":
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    main = load("main")
    summary = model_summary(main, "main")
    fig_models(summary, "main")
    fig_tradeoff(summary)
    tree_ranks(main)
    hyperparameter_table()
    imbalance_effects(main, "main")
    imbalance_intervals(main, "main")
    per, s = validity(main, "main")
    fig_validity(s, "main")
    paired_differences(per, "main")
    paired_by_dataset(per, "main")
    counts_vs_cies(main, "main")
    calibration(main, "main")
    secondary(main, "main")

    mixed = load("mixed")
    model_summary(mixed, "mixed")
    per, s = validity(mixed, "mixed")
    fig_validity(s, "mixed")
    paired_differences(per, "mixed")
    paired_by_dataset(per, "mixed")
    counts_vs_cies(mixed, "mixed")

    fresh = load("fresh")
    per, _ = validity(fresh, "fresh")
    paired_differences(per, "fresh")

    matched = main[(main.variant == "default") & main.dataset.isin(["telco", "hr_attrition"])
                   & main.strategy.isin(["none", "smote"])]
    validity(matched, "rel_gauss_mult")
    validity(load("relational"), "rel_relational")

    noise_sensitivity(load("noise"))
    synthetic_figures()
    lime_figures()
    mixedfresh = load("mixedfresh")
    preregistered_variants(main, mixedfresh)
    consequence_bound_tables(main, mixed, mixedfresh)
