"""Interval estimates that respect the dependence between configurations of the same split."""
import numpy as np
from scipy.stats import t as tdist


def strat_boot(frame, cols, n_boot=10000, seed=0):
    """Pooled means sum/count of `cols` over the rows where each is defined, with bootstrap draws
    that resample seeds within each dataset (rows of a split stay together, datasets keep their weight)."""
    cols = list(cols)
    g = frame.groupby(["dataset", "seed"])
    S = g[cols].sum(min_count=1).fillna(0.0).values
    N = g[cols].count().values.astype(float)
    ds = g.size().index.get_level_values(0).values
    rng = np.random.default_rng(seed)
    W = np.zeros((n_boot, len(ds)))
    for d in np.unique(ds):
        idx = np.where(ds == d)[0]
        W[:, idx] = rng.multinomial(len(idx), np.ones(len(idx)) / len(idx), size=n_boot)
    with np.errstate(invalid="ignore", divide="ignore"):
        return S.sum(0) / N.sum(0), (W @ S) / (W @ N), len(ds)


def boot_ci(frame, col, **kw):
    point, draws, n = strat_boot(frame, [col], **kw)
    return float(point[0]), float(np.nanpercentile(draws[:, 0], 2.5)), float(np.nanpercentile(draws[:, 0], 97.5)), n


def interval(draws):
    return np.nanpercentile(draws, 2.5, axis=0), np.nanpercentile(draws, 97.5, axis=0)


def t_halfwidth(x):
    """Half-width of a t-based 95% interval of the mean."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return tdist.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
