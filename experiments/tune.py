"""Hyper-parameter tuning within the training part of every split.

Randomized search (30 candidates, 6 for logistic regression), three-fold stratified
cross-validation on ROC-AUC, preprocessing refitted inside every fold. The selected
configuration is shared by all imbalance treatments of the split.

    python experiments/tune.py 0 10        # seeds 0-9
"""
import json
import os
import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from fragility.pipeline import ALL_MODELS, DATASETS, RESULTS, make_model, make_preprocessor, prepare

SPACES = {
    "Random Forest": dict(n_estimators=[100, 200, 500], max_depth=[None, 4, 6, 10, 20],
                          min_samples_leaf=[1, 2, 5, 10, 20], max_features=["sqrt", 0.3, 0.5]),
    "XGBoost": dict(n_estimators=[100, 200, 400, 800], max_depth=[2, 3, 4, 6, 8],
                    learning_rate=[0.01, 0.03, 0.1, 0.3], subsample=[0.7, 0.85, 1.0],
                    colsample_bytree=[0.6, 0.8, 1.0], min_child_weight=[1, 5, 10], reg_lambda=[0.1, 1.0, 10.0]),
    "LightGBM": dict(n_estimators=[100, 200, 400, 800], num_leaves=[7, 15, 31, 63],
                     learning_rate=[0.01, 0.03, 0.1], min_child_samples=[5, 20, 50],
                     subsample=[0.7, 0.85, 1.0], subsample_freq=[1],
                     colsample_bytree=[0.6, 0.8, 1.0], reg_lambda=[0.0, 1.0, 10.0]),
    "CatBoost": dict(iterations=[200, 500, 1000], depth=[4, 6, 8],
                     learning_rate=[0.01, 0.03, 0.1], l2_leaf_reg=[1, 3, 10]),
    "Logistic Regression": dict(C=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]),
}


def tune(ds, seed, model):
    P = prepare(ds, seed=seed, strategy="none")
    pipe = Pipeline([("pre", make_preprocessor(P.num_cols, P.cat_cols)), ("m", make_model(model, {}, seed=seed))])
    search = RandomizedSearchCV(pipe, {f"m__{k}": v for k, v in SPACES[model].items()},
                                n_iter=6 if model == "Logistic Regression" else 30, scoring="roc_auc",
                                cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                                random_state=seed, n_jobs=1, refit=False)
    search.fit(P.X_tr, P.y_tr.values)
    best = {k[3:]: (v.item() if isinstance(v, np.generic) else v) for k, v in search.best_params_.items()}
    return ds, seed, model, best, search.best_score_


if __name__ == "__main__":
    first, last = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 10)
    jobs = [(d, s, m) for d in DATASETS for s in range(first, last) for m in ALL_MODELS]
    res = Parallel(n_jobs=-1)(delayed(tune)(*j) for j in jobs)
    os.makedirs(RESULTS, exist_ok=True)
    paths = [os.path.join(RESULTS, f) for f in ("tuned_params.json", "tuned_cv_auc.json")]
    params, auc = [json.load(open(p)) if os.path.exists(p) else {} for p in paths]
    for ds, seed, model, best, score in res:
        params.setdefault(ds, {}).setdefault(str(seed), {})[model] = best
        auc.setdefault(ds, {}).setdefault(str(seed), {})[model] = score
    for p, obj in zip(paths, (params, auc)):
        with open(p, "w") as f:
            json.dump(obj, f, indent=2)
