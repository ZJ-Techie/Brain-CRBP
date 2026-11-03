"""Evaluation protocol: splitting, marker selection and downstream metrics.

The paper reports two protocols, both implemented here verbatim:

``holdout`` (SNP-VBM, SNP-FreeSurfer)
    Stratified subject-level 80/20 split per seed.  Upstream projections, the
    Top-k marker list, and every downstream model are fitted on the training
    participants only; the held-out 20% is scored once.

``oof`` (SNP-Protein)
    The upstream projection is fitted once on the complete-case cohort, then
    the downstream classifier is scored with stratified 10-fold out-of-fold
    predictions and the cognitive regressors are fitted on all available rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


def subject_split(bl_dx, test_size: float = 0.2, seed: int = 20260907):
    """Stratified subject-level holdout; returns ``(train_idx, test_idx)``."""
    labels = np.asarray(bl_dx, dtype=float).reshape(-1)
    if not np.isfinite(labels).all():
        raise ValueError("stratification labels must be finite")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(seed))
    train, test = next(splitter.split(np.zeros((len(labels), 1)), labels))
    if set(train).intersection(test):
        raise ValueError("train and test subjects overlap")
    return np.sort(train), np.sort(test)


def top_features(weights, names, k: int = 10) -> list[str]:
    """Top-``k`` features by descending absolute weight with stable ties."""
    values = np.asarray(weights, dtype=float).reshape(-1)
    names = list(names)
    if values.size != len(names):
        raise ValueError("weights and feature names must be aligned")
    order = np.argsort(-np.abs(values), kind="stable")[: int(k)]
    return [names[index] for index in order]


def marker_arrays(cohort, snp_names, img_names_per_visit) -> tuple[np.ndarray, np.ndarray]:
    """Slice the selected markers out of the cohort: ``(N,k)`` and ``(N,T,k)``."""
    snp_index = {name: index for index, name in enumerate(cohort.snp_names)}
    img_index = {name: index for index, name in enumerate(cohort.img_names)}
    snp = cohort.snp[:, [snp_index[name] for name in snp_names]]
    img = np.stack([
        cohort.img[:, visit, [img_index[name] for name in names]]
        for visit, names in enumerate(img_names_per_visit)
    ], axis=1)
    return snp, img


def joint_features(snp: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Concatenate ``[SNP_k, img_t1_k, ..., img_tT_k]`` as the SVM/SVR input."""
    if snp.ndim != 2 or img.ndim != 3 or snp.shape[0] != img.shape[0]:
        raise ValueError("expected snp (N,k) and img (N,T,k)")
    return np.column_stack([snp] + [img[:, visit, :] for visit in range(img.shape[1])])


def ccc(actual, predicted) -> float:
    """Lin's concordance correlation coefficient."""
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    covariance = np.mean((actual - actual.mean()) * (predicted - predicted.mean()))
    denominator = actual.var() + predicted.var() + (actual.mean() - predicted.mean()) ** 2
    if denominator == 0:
        return 1.0 if np.array_equal(actual, predicted) else 0.0
    return float(2 * covariance / denominator)


_SVC_GRID = tuple((c, gamma) for c in (0.1, 1.0, 10.0, 100.0) for gamma in ("scale", 0.01, 0.1))
_SVR_GRID = tuple(
    (c, gamma, epsilon)
    for c in (0.1, 1.0, 10.0, 100.0)
    for gamma in ("scale", 0.01, 0.1)
    for epsilon in (0.05, 0.1, 0.2)
)


def _classifier(seed: int, params=None):
    params = params or {"kernel": "rbf", "C": 1.0, "gamma": "scale"}
    return make_pipeline(
        StandardScaler(),
        SVC(class_weight="balanced", random_state=int(seed), **params),
    )


def _regressor(params=None):
    params = params or {"kernel": "rbf", "C": 1.0, "gamma": "scale", "epsilon": 0.1}
    return make_pipeline(StandardScaler(), SVR(**params))


def tune_classifier(features, labels, seed: int, folds: int = 5):
    """Inner-CV selection of ``(C, gamma)`` on the training partition only."""
    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    best = None
    for c, gamma in _SVC_GRID:
        f1_values, accuracy_values = [], []
        for train_rows, test_rows in splitter.split(features, labels):
            model = _classifier(seed, {"kernel": "rbf", "C": c, "gamma": gamma})
            model.fit(features[train_rows], labels[train_rows])
            predicted = model.predict(features[test_rows])
            f1_values.append(f1_score(labels[test_rows], predicted, labels=[1, 3],
                                      average="macro", zero_division=0))
            accuracy_values.append(accuracy_score(labels[test_rows], predicted))
        key = (float(np.mean(f1_values)), float(np.mean(accuracy_values)))
        if best is None or key > best[0]:
            best = (key, {"kernel": "rbf", "C": c, "gamma": gamma})
    return best[1]


def tune_regressor(features, target, seed: int, folds: int = 5):
    """Inner-CV selection of ``(C, gamma, epsilon)`` by RMSE, then CCC."""
    splitter = KFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    best = None
    for c, gamma, epsilon in _SVR_GRID:
        rmse_values, ccc_values = [], []
        for train_rows, test_rows in splitter.split(features):
            model = _regressor({"kernel": "rbf", "C": c, "gamma": gamma, "epsilon": epsilon})
            model.fit(features[train_rows], target[train_rows])
            predicted = model.predict(features[test_rows])
            rmse_values.append(float(np.sqrt(mean_squared_error(target[test_rows], predicted))))
            ccc_values.append(ccc(target[test_rows], predicted))
        key = (-float(np.mean(rmse_values)), float(np.mean(ccc_values)))
        if best is None or key > best[0]:
            best = (key, {"kernel": "rbf", "C": c, "gamma": gamma, "epsilon": epsilon})
    return best[1]


def evaluate_holdout(features, test_indices, bl_dx, adas, mmse, seed: int, downstream=None):
    """Fit SVM/SVR on the training rows and score the held-out partition.

    ``downstream`` may carry the frozen hyper-parameters selected by the
    original runs (``classifier`` / ``regression``); otherwise the defaults
    ``C=1, gamma=scale, epsilon=0.1`` are used, or ``tune=True`` re-selects
    them with inner cross-validation on the training partition only.
    """
    downstream = downstream or {}
    tuning = bool(downstream.get("tune", False))
    folds = int(downstream.get("inner_folds", 5))
    fixed_classifier = downstream.get("classifier")
    fixed_regression = downstream.get("regression") or {}
    train_mask = np.ones(len(bl_dx), dtype=bool)
    train_mask[test_indices] = False
    test_mask = ~train_mask
    labels = np.asarray(bl_dx, dtype=float)
    keep_train = train_mask & np.isin(labels, [1, 3])
    keep_test = test_mask & np.isin(labels, [1, 3])

    params = fixed_classifier
    if params is None and tuning:
        params = tune_classifier(features[keep_train], labels[keep_train].astype(int), seed, folds)
    model = _classifier(seed, params)
    model.fit(features[keep_train], labels[keep_train].astype(int))
    predicted = model.predict(features[keep_test])
    rows = [{
        "task": "DX13_classification",
        "visit": "BL",
        "n_train": int(keep_train.sum()),
        "n_test": int(keep_test.sum()),
        "accuracy": float(accuracy_score(labels[keep_test].astype(int), predicted)),
        "f1": float(f1_score(labels[keep_test].astype(int), predicted, labels=[1, 3],
                             average="macro", zero_division=0)),
        "ccc": np.nan,
        "rmse": np.nan,
    }]

    for modality, scores in (("ADAS", adas), ("MMSE", mmse)):
        per_visit = []
        for visit in range(4):
            values = np.asarray(scores, dtype=float)[:, visit]
            fit_rows = train_mask & np.isfinite(values)
            score_rows = test_mask & np.isfinite(values)
            params = (fixed_regression.get(modality) or {}).get(("BL", "M6", "M12", "M24")[visit])
            if params is None and tuning:
                params = tune_regressor(features[fit_rows], values[fit_rows], seed + visit, folds)
            regressor = _regressor(params)
            regressor.fit(features[fit_rows], values[fit_rows])
            predicted_score = regressor.predict(features[score_rows])
            actual = values[score_rows]
            per_visit.append({
                "task": f"{modality}_regression",
                "visit": ("BL", "M6", "M12", "M24")[visit],
                "n_train": int(fit_rows.sum()),
                "n_test": int(score_rows.sum()),
                "accuracy": np.nan,
                "f1": np.nan,
                "ccc": ccc(actual, predicted_score),
                "rmse": float(np.sqrt(mean_squared_error(actual, predicted_score))),
            })
        rows.extend(per_visit)
        rows.append({
            "task": f"{modality}_regression",
            "visit": "mean_BL_M6_M12_M24",
            "n_train": int(np.mean([row["n_train"] for row in per_visit])),
            "n_test": int(np.mean([row["n_test"] for row in per_visit])),
            "accuracy": np.nan,
            "f1": np.nan,
            "ccc": float(np.mean([row["ccc"] for row in per_visit])),
            "rmse": float(np.mean([row["rmse"] for row in per_visit])),
        })
    return rows


def evaluate_oof(features, bl_dx, adas, mmse, seed: int, folds: int = 10):
    """Stratified out-of-fold classification plus in-sample cognitive SVR."""
    labels = np.asarray(bl_dx, dtype=float)
    keep = np.isin(labels, [1, 3])
    actual = labels[keep].astype(int)
    values = np.asarray(features, dtype=float)[keep]

    predicted = np.empty(len(actual), dtype=int)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
    for fit_rows, score_rows in splitter.split(values, actual):
        model = _classifier(seed)
        model.fit(values[fit_rows], actual[fit_rows])
        predicted[score_rows] = model.predict(values[score_rows])

    rows = [{
        "task": "DX13_classification",
        "visit": "BL",
        "n_train": int(len(actual)),
        "n_test": int(len(actual)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "f1": float(f1_score(actual, predicted, labels=[1, 3], average="macro", zero_division=0)),
        "ccc": np.nan,
        "rmse": np.nan,
    }]

    for modality, scores in (("ADAS", adas), ("MMSE", mmse)):
        per_visit = []
        for visit in range(4):
            target = np.asarray(scores, dtype=float)[:, visit]
            mask = np.isfinite(target)
            regressor = _regressor()
            regressor.fit(np.asarray(features, dtype=float)[mask], target[mask])
            predicted_score = regressor.predict(np.asarray(features, dtype=float)[mask])
            per_visit.append({
                "task": f"{modality}_regression",
                "visit": ("BL", "M6", "M12", "M24")[visit],
                "n_train": int(mask.sum()),
                "n_test": int(mask.sum()),
                "accuracy": np.nan,
                "f1": np.nan,
                "ccc": ccc(target[mask], predicted_score),
                "rmse": float(np.sqrt(mean_squared_error(target[mask], predicted_score))),
            })
        rows.extend(per_visit)
        rows.append({
            "task": f"{modality}_regression",
            "visit": "mean_BL_M6_M12_M24",
            "n_train": int(np.mean([row["n_train"] for row in per_visit])),
            "n_test": int(np.mean([row["n_test"] for row in per_visit])),
            "accuracy": np.nan,
            "f1": np.nan,
            "ccc": float(np.mean([row["ccc"] for row in per_visit])),
            "rmse": float(np.mean([row["rmse"] for row in per_visit])),
        })
    return rows


def summarize(rows_by_seed: list[dict]) -> pd.DataFrame:
    """Mean and sample standard deviation (ddof=1) across seeds."""
    frame = pd.DataFrame([row for rows in rows_by_seed for row in rows])
    grouped = frame.groupby(["task", "visit"], as_index=False).agg(
        **{
            "accuracy_mean": ("accuracy", "mean"),
            "accuracy_std": ("accuracy", lambda x: x.std(ddof=1)),
            "f1_mean": ("f1", "mean"),
            "f1_std": ("f1", lambda x: x.std(ddof=1)),
            "ccc_mean": ("ccc", "mean"),
            "ccc_std": ("ccc", lambda x: x.std(ddof=1)),
            "rmse_mean": ("rmse", "mean"),
            "rmse_std": ("rmse", lambda x: x.std(ddof=1)),
            "n_seeds": ("accuracy", "size"),
        }
    )
    return grouped
