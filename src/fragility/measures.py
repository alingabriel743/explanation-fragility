"""CIES, the comparator stability measures, decision consequences and the consequence bound."""
import numpy as np
from scipy.stats import kendalltau, spearmanr, weightedtau

N_REASONS = 4
EPS_MIN = 1e-4      # denominator floor of RIS/ROS (Agarwal et al., 2022)
EPS_DIV = 1e-6      # guard for element-wise relative changes of zero entries


# ---------------------------------------------------------------- weights
# Ties in |phi| are broken in favour of the lower feature index, both for the rank weights and
# for the top-k sets of the decision artefacts (np.argsort with a stable sort).

def ranks_desc(a):
    """Rank 1 = largest |a|."""
    return ranks_rows(np.atleast_2d(a))[0]


def ranks_rows(A):
    order = np.argsort(-np.abs(A), axis=1, kind="stable")
    r = np.empty_like(order)
    np.put_along_axis(r, order, np.arange(1, A.shape[1] + 1)[None, :].repeat(len(A), 0), axis=1)
    return r


def harmonic_by_rank(A):
    """Harmonic rank weights h(r) = (1/r) / H_M, row-wise for 2-D input."""
    A = np.atleast_2d(A)
    return (1.0 / ranks_rows(A)) / np.sum(1.0 / np.arange(1, A.shape[1] + 1))


def normalise(w):
    return w / w.sum()


def w_harmonic(phi):
    return normalise(1.0 / ranks_desc(phi))


WEIGHTS = {
    "exp": lambda phi: normalise(np.exp(-0.5 * ranks_desc(phi))),
    "log": lambda phi: normalise(1.0 / np.log2(ranks_desc(phi) + 1)),
    "top5": lambda phi: normalise((ranks_desc(phi) <= 5).astype(float)),
}


# ---------------------------------------------------------------- CIES
def weighted_cies(phi_x, phi_nb, w):
    """Mean over neighbours of max(0, 1 - weighted distance / weighted magnitude)."""
    n = np.sum(w * np.abs(phi_x))
    if n < 1e-12:
        return np.nan
    d = np.sum(w * np.abs(phi_nb - phi_x), axis=1)
    return float(np.mean(np.maximum(0.0, 1.0 - d / n)))


def cies(phi_x, phi_nb):
    """CIES with harmonic rank weights, equation (5)."""
    return weighted_cies(phi_x, phi_nb, harmonic_by_rank(phi_x)[0])


def ratio_score(phi_x, phi_nb, w):
    """Unbounded form max(0, 1 - mean distance / magnitude), the initial form of CIES (V0)."""
    mag = np.sum(w * np.abs(phi_x))
    if mag < 1e-12:
        return np.nan
    return max(0.0, 1.0 - np.mean(np.sum(w * np.abs(phi_nb - phi_x), axis=1)) / mag)


def uniform_index(phi_x, phi_nb):
    """CIES with equal weights and the same bounded aggregation."""
    mag = np.mean(np.abs(phi_x))
    if mag <= 1e-12:
        return np.nan
    d = np.mean(np.abs(phi_nb - phi_x), axis=1)
    return float(np.mean(np.maximum(0.0, 1.0 - d / mag)))


# pre-registered variants (preregistration/PREREGISTRATION_cies_variants.md)
def _symmetric_weights(phi_x, phi_nb):
    w = np.maximum(harmonic_by_rank(phi_x)[0][None, :], harmonic_by_rank(phi_nb))
    return w / w.sum(axis=1, keepdims=True)


def cies_symmetric(phi_x, phi_nb):
    """V1: a feature counts if it leads in either explanation."""
    w = _symmetric_weights(phi_x, phi_nb)
    num = np.sum(w * np.abs(phi_nb - phi_x), axis=1).mean()
    den = np.sum(w * (np.abs(phi_x)[None, :] + np.abs(phi_nb)) / 2, axis=1).mean()
    return np.nan if den < 1e-12 else max(0.0, 1.0 - num / den)


def cies_symmetric_bounded(phi_x, phi_nb):
    w = _symmetric_weights(phi_x, phi_nb)
    d = np.sum(w * np.abs(phi_nb - phi_x), axis=1)
    n = np.sum(w * (np.abs(phi_x)[None, :] + np.abs(phi_nb)) / 2, axis=1)
    ok = n > 1e-12
    return float(np.mean(np.maximum(0.0, 1.0 - d[ok] / n[ok]))) if ok.any() else np.nan


def signed_rank_index(phi_x, phi_nb):
    """V2: compares the signed harmonic rank weights of the two explanations."""
    sx = np.sign(phi_x) * harmonic_by_rank(phi_x)[0]
    sn = np.sign(phi_nb) * harmonic_by_rank(phi_nb)
    return float(1.0 - np.mean(np.sum(np.abs(sn - sx), axis=1)) / 2.0)


# ---------------------------------------------------------------- comparators
def rbo_ext(r1, r2, p=0.9):
    """Extrapolated rank-biased overlap (Webber et al., 2010) of two full rankings."""
    k = len(r1)
    s1, s2 = set(), set()
    overlap, acc = 0, 0.0
    for d in range(1, k + 1):
        a, b = r1[d - 1], r2[d - 1]
        overlap += 1 if a == b else (a in s2) + (b in s1)
        s1.add(a)
        s2.add(b)
        acc += overlap / d * p ** d
    return overlap / k * p ** k + (1 - p) / p * acc


def topk_set(phi, k):
    return set(np.argsort(-np.abs(phi), kind="stable")[:k])


def _relative(a, b):
    den = np.where(np.abs(a) > EPS_DIV, a, np.sign(a) * EPS_DIV + (a == 0) * EPS_DIV)
    return (a - b) / den


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def relative_stability(phi_x, phi_nb, xt, xt_nb, p_x, p_nb):
    """RIS and ROS (Agarwal et al., 2022, eqs. 2 and 5) over neighbours that keep the predicted class."""
    keep = (p_nb >= 0.5) == (p_x >= 0.5)
    if not keep.any():
        return np.nan, np.nan, 0.0
    rphi = np.linalg.norm(_relative(phi_x[None, :], phi_nb[keep]), axis=1)
    rx = np.linalg.norm(_relative(xt[None, :], xt_nb[keep]), axis=1)
    ro = np.abs(_logit(p_nb[keep]) - _logit(p_x))
    return (float(np.max(rphi / np.maximum(rx, EPS_MIN))), float(np.max(rphi / np.maximum(ro, EPS_MIN))),
            float(keep.mean()))


def stability_metrics(phi_x, phi_nb, xt, xt_nb, p_x, p_nb):
    """All stability measures of one instance on one set of neighbours."""
    out = {"cies": cies(phi_x, phi_nb),
           "cies_unbounded": ratio_score(phi_x, phi_nb, w_harmonic(phi_x)),
           "cies_uniform": uniform_index(phi_x, phi_nb)}
    # alternative weights, with the same bounded aggregation as CIES
    mean_abs = np.mean(np.abs(np.vstack([phi_x[None, :], phi_nb])), axis=0)
    out["cies_nbw"] = weighted_cies(phi_x, phi_nb, w_harmonic(mean_abs))
    for name, wf in WEIGHTS.items():
        out[f"cies_{name}"] = weighted_cies(phi_x, phi_nb, wf(phi_x))

    dphi = np.linalg.norm(phi_nb - phi_x, axis=1)
    dx = np.linalg.norm(xt_nb - xt, axis=1)
    ok = dx > 1e-10
    out["avg_sens"], out["max_sens"] = float(np.mean(dphi)), float(np.max(dphi))
    out["lip_max"] = float(np.max(dphi[ok] / dx[ok])) if ok.any() else 0.0
    out["lip_mean"] = float(np.mean(dphi[ok] / dx[ok])) if ok.any() else 0.0
    out["ris"], out["ros"], out["share_same_class"] = relative_stability(phi_x, phi_nb, xt, xt_nb, p_x, p_nb)

    ax = np.abs(phi_x)
    order_x = np.argsort(-ax, kind="stable")
    t3, t5 = topk_set(phi_x, 3), topk_set(phi_x, 5)
    k3, k5, sp, kd, wk, rb = [], [], [], [], [], []
    for ph in phi_nb:
        an = np.abs(ph)
        k3.append(len(t3 & topk_set(ph, 3)) / 3)
        k5.append(len(t5 & topk_set(ph, 5)) / 5)
        sp.append(spearmanr(ax, an)[0])
        kd.append(kendalltau(ax, an)[0])
        wk.append(weightedtau(ax, an)[0])
        rb.append(rbo_ext(order_x, np.argsort(-an, kind="stable")))
    out.update(topk3=np.mean(k3), topk5=np.mean(k5), spearman=np.nanmean(sp), kendall=np.nanmean(kd),
               wkendall=np.nanmean(wk), rbo=np.mean(rb), pred_stab=float(1.0 - np.mean(np.abs(p_nb - p_x))))

    out["cies_sym"] = cies_symmetric(phi_x, phi_nb)
    out["sri"] = signed_rank_index(phi_x, phi_nb)
    out["cies_star"] = (out["cies_sym"] + out["sri"]) / 2
    out["cies_v4"] = (out["cies_unbounded"] + out["sri"]) / 2
    out["cies_sym_b"] = cies_symmetric_bounded(phi_x, phi_nb)
    out["cies_star_b"] = (out["cies_sym_b"] + out["sri"]) / 2
    return out


# +1: higher value = more stable, -1: higher value = less stable
METRIC_DIRECTION = {
    "cies": +1, "cies_unbounded": +1, "cies_uniform": +1, "cies_nbw": +1,
    "cies_exp": +1, "cies_log": +1, "cies_top5": +1,
    "avg_sens": -1, "max_sens": -1, "lip_max": -1, "lip_mean": -1, "ris": -1, "ros": -1,
    "topk3": +1, "topk5": +1, "spearman": +1, "kendall": +1, "wkendall": +1, "rbo": +1,
    "pred_stab": +1, "cies_rbo": +1,
    "cies_sym": +1, "sri": +1, "cies_star": +1, "cies_v4": +1, "cies_sym_b": +1, "cies_star_b": +1,
    "count_sign3": -1, "count_matched": -1,
}

METRIC_LABEL = {
    "cies": "CIES", "cies_unbounded": "CIES, unbounded form", "cies_uniform": "Uniform-weight index",
    "cies_nbw": "CIES (neighbourhood weights)", "cies_exp": "Exponential weights", "cies_log": "Log (nDCG) weights",
    "cies_top5": "Top-5 binary weights",
    "avg_sens": "Avg-sensitivity (Yeh 2019)", "max_sens": "Max-sensitivity (Yeh 2019)",
    "lip_max": "Local Lipschitz, max (Alvarez-Melis 2018)", "lip_mean": "Local Lipschitz, mean",
    "ris": "RIS (Agarwal 2022)", "ros": "ROS (Agarwal 2022)",
    "topk3": "Top-3 intersection (Ghorbani 2019)", "topk5": "Top-5 intersection",
    "spearman": "Spearman rank corr.", "kendall": "Kendall tau", "wkendall": "Weighted Kendall tau (Vigna 2015)",
    "rbo": "RBO, p=0.9 (Webber 2010)", "pred_stab": "Prediction stability", "cies_rbo": "CIES + RBO (paired diagnostic)",
    "cies_sym": "Symmetric-weight CIES (V1)", "sri": "Signed-rank index (V2)", "cies_star": "Average of V1 and V2 (V3)",
    "cies_v4": "V4 (CIES+V2)/2", "cies_sym_b": "V1b symmetric, bounded", "cies_star_b": "V3b (V1b+V2)/2",
    "count_sign3": "Direct count of direction reversals", "count_matched": "Direct count of the same change",
}


# ---------------------------------------------------------------- decision consequences
DECISIONS = ["top1", "top3set", "sign3", "reasons4", "action"]
CONSEQUENCES = [f"cons_{c}" for c in DECISIONS] + ["cons_predflip"]


def _reasons(phi):
    pos = np.where(phi > 0)[0]
    return set(pos[np.argsort(-phi[pos], kind="stable")][:N_REASONS])


def decision_consequences(phi_x, phi_nb, p_x, p_nb):
    """Share of neighbours in which each decision artefact changes:
    top1 (leading feature), top3set (set of the three leading features), sign3 (direction of one of
    them), reasons4 (four adverse-action reasons), action (strongest risk driver), predflip (class)."""
    t1, t3, r4, act = int(np.argmax(np.abs(phi_x))), topk_set(phi_x, 3), _reasons(phi_x), int(np.argmax(phi_x))
    top3 = np.array(sorted(t3))
    c = {k: [] for k in DECISIONS + ["predflip"]}
    for ph, pn in zip(phi_nb, p_nb):
        c["top1"].append(int(np.argmax(np.abs(ph))) != t1)
        c["top3set"].append(topk_set(ph, 3) != t3)
        c["sign3"].append(bool(np.any(np.sign(ph[top3]) * np.sign(phi_x[top3]) < 0)))
        c["reasons4"].append(_reasons(ph) != r4)
        c["action"].append(int(np.argmax(ph)) != act)
        c["predflip"].append((pn >= 0.5) != (p_x >= 0.5))
    return {f"cons_{k}": float(np.mean(v)) for k, v in c.items()}


# ---------------------------------------------------------------- consequence bound (Proposition 3)
def margins(phi):
    """Lower bounds on D_R for any neighbour whose artefact changes, and ||phi||_w."""
    a = np.abs(phi)
    w = harmonic_by_rank(phi)[0]
    order = np.argsort(-a, kind="stable")
    out = {}
    for name, k in (("top1", 1), ("top3set", 3)):
        others = order[k:]
        out[name] = float(np.min(w[others] * (a[order[k - 1]] - a[others]))) if len(others) else np.inf
    out["sign3"] = float(np.min(w[order[:3]] * a[order[:3]]))
    act = int(np.argmax(phi))
    rest = np.array([l for l in range(len(phi)) if l != act])
    out["action"] = float(np.min(np.minimum(w[act], w[rest]) * (phi[act] - phi[rest])))
    pos = np.where(phi > 0)[0]
    R = pos[np.argsort(-phi[pos], kind="stable")][:N_REASONS]
    notR = np.array([l for l in range(len(phi)) if l not in set(R)])
    cands = []
    if len(R) and len(notR):
        cands.append(min(min(w[j], w[l]) * (phi[j] - phi[l]) for j in R for l in notR))
    if len(R):
        cands.append(float(np.min(w[R] * phi[R])))
    if len(R) < N_REASONS and len(notR):
        cands.append(float(np.min(w[notR] * np.abs(phi[notR]))))
    out["reasons4"] = float(min(cands)) if cands else 0.0
    return out, float(np.sum(w * a))


def consequence_bounds(phi_x, cies_value):
    """Relative margins mu_A and the plug-in bound B_A = min(1, (1 - CIES) / min(1, mu_A))."""
    m, norm = margins(phi_x)
    out = {}
    for name, mv in m.items():
        mu = mv / norm if norm > 1e-12 else 0.0
        out[f"mu_{name}"] = mu
        out[f"B_{name}"] = 1.0 if (mu <= 0 or np.isnan(cies_value)) else float(min(1.0, (1.0 - cies_value) / min(1.0, mu)))
    return out
