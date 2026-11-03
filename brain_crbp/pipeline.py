"""End-to-end orchestration: train, export markers, evaluate, summarise."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import protocol
from .data import load_imaging_cohort, load_prior, load_protein_cohort, standardize_visits
from .train import load_checkpoint, save_checkpoint, train_model

DEFAULT_SEEDS = (20260907, 20260908, 20260909)


def load_config(path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("name", "modality", "protocol", "training"):
        if key not in config:
            raise ValueError(f"config {path} is missing {key!r}")
    config.setdefault("seeds", list(DEFAULT_SEEDS))
    config.setdefault("top_k", 10)
    config.setdefault("test_size", 0.2)
    config.setdefault("upstream", "train_split")
    config.setdefault("standardize", {"snp": False, "img": False, "scope": "train"})
    config.setdefault("association", False)
    return config


def build_cohort(root, config: dict):
    if config["modality"] == "protein":
        return load_protein_cohort(root)
    return load_imaging_cohort(root, config["modality"])


def _standardizer(cohort, config, fit_indices):
    """Return a callable that maps the cohort arrays into model space."""
    standardize = config.get("standardize") or {}
    do_snp = bool(standardize.get("snp", False))
    do_img = bool(standardize.get("img", False))
    if not (do_snp or do_img):
        return None, {"snp": False, "img": False}

    fit = np.arange(len(cohort)) if standardize.get("scope", "train") == "all" else np.asarray(fit_indices, dtype=int)
    stats = {"snp": do_snp, "img": do_img}

    if do_snp:
        flat = cohort.snp[fit]
        stats["snp_mean"] = flat.mean(axis=0)
        stats["snp_scale"] = np.where(flat.std(axis=0) > 1e-8, flat.std(axis=0), 1.0)
    if do_img:
        flat = cohort.img[fit].transpose(0, 2, 1).reshape(-1, cohort.img.shape[2])
        stats["img_mean"] = flat.mean(axis=0)
        stats["img_scale"] = np.where(flat.std(axis=0) > 1e-8, flat.std(axis=0), 1.0)
    return stats, {"snp": do_snp, "img": do_img}


def _apply_standardizer(cohort, stats):
    if not stats:
        return cohort
    cohort.snp = (cohort.snp - stats["snp_mean"]) / stats["snp_scale"] if stats.get("snp") else cohort.snp
    if stats.get("img"):
        cohort.img = (cohort.img - stats["img_mean"][None, None, :]) / stats["img_scale"][None, None, :]
    return cohort


def train_seed(root, config: dict, seed: int, output_dir, device: str = "cpu", verbose: bool = True):
    """Fit one seed and write ``seed_<seed>.pt``; returns the checkpoint payload."""
    cohort = build_cohort(root, config)
    log = print if verbose else (lambda message: None)

    if config["protocol"] == "holdout":
        train_indices, _ = protocol.subject_split(cohort.bl_dx, config["test_size"], seed)
    else:
        train_indices = np.arange(len(cohort))

    stats, _ = _standardizer(cohort, config, train_indices)
    cohort = _apply_standardizer(cohort, stats)

    priors = config.get("priors") or {}
    snp_prior = load_prior(root / priors["snp"], priors["snp_column"], cohort.snp_names) if priors.get("snp") else None
    img_prior = load_prior(root / priors["img"], priors["img_column"], cohort.img_names) if priors.get("img") else None

    log(f"[{config['name']}] seed {seed}: training on {len(train_indices)} participants")
    model, history = train_model(
        cohort, train_indices, config["training"], seed, device=device,
        snp_prior=snp_prior, img_prior=img_prior, log=log,
    )

    snp_markers = protocol.top_features(
        model.gene_weight.detach().cpu().numpy().reshape(-1), cohort.snp_names, config["top_k"])
    img_markers = [
        protocol.top_features(model.image_weights[visit].detach().cpu().numpy().reshape(-1),
                              cohort.img_names, config["top_k"])
        for visit in range(cohort.n_visits)
    ]
    config_with_stats = dict(config)
    config_with_stats["preprocess"] = {
        key: (np.asarray(value).tolist() if isinstance(value, np.ndarray) else value)
        for key, value in (stats or {}).items()
    }
    path = save_checkpoint(
        Path(output_dir) / f"seed_{seed}.pt", model, cohort, config_with_stats, seed,
        snp_markers, img_markers, history,
    )
    log(f"[{config['name']}] seed {seed}: checkpoint -> {path}")
    return load_checkpoint(path)


def evaluate_checkpoint(root, config: dict, payload: dict, device: str = "cpu"):
    """Score one checkpoint with the protocol recorded in the config."""
    cohort = build_cohort(root, config)
    stats = (payload.get("config") or {}).get("preprocess") or {}
    if stats:
        stats = {key: (np.asarray(value) if isinstance(value, list) else value) for key, value in stats.items()}
        cohort = _apply_standardizer(cohort, stats)

    snp, img = protocol.marker_arrays(cohort, payload["snp_markers"], payload["img_names"])
    features = protocol.joint_features(snp, img)
    seed = int(payload["seed"])

    if config["protocol"] == "holdout":
        _, test_indices = protocol.subject_split(cohort.bl_dx, config["test_size"], seed)
        downstream = dict(config.get("downstream") or {})
        if downstream.get("use_frozen", True):
            downstream.update(payload.get("downstream_params") or {})
        rows = protocol.evaluate_holdout(features, test_indices, cohort.bl_dx, cohort.adas, cohort.mmse,
                                         seed, downstream=downstream)
        if config.get("association"):
            rows = rows + _association_rows(cohort, payload, test_indices)
    else:
        rows = protocol.evaluate_oof(features, cohort.bl_dx, cohort.adas, cohort.mmse, seed,
                                     folds=int(config.get("folds", 10)))
    tasks = config.get("tasks")
    if tasks:
        def keep(row):
            if row["task"] == "association":
                return "association" in tasks
            if row["task"].endswith("classification"):
                return "classification" in tasks
            return "regression" in tasks

        rows = [row for row in rows if keep(row)]
    return rows


def _normalize(values):
    centered = np.asarray(values, dtype=float) - np.asarray(values, dtype=float).mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    norms = np.where(norms > 0, norms, 1.0)
    return centered / norms


def _association_rows(cohort, payload, test_indices):
    """Held-out genotype-phenotype projection CC/RMSE on the selected markers."""
    train_mask = np.ones(len(cohort), dtype=bool)
    train_mask[test_indices] = False
    weights = payload["state_dict"]
    u = weights["gene_weight"].numpy().reshape(-1)
    v = weights["image_weights"].numpy()[:, :, 0]
    snp_index = {name: i for i, name in enumerate(cohort.snp_names)}
    img_index = {name: i for i, name in enumerate(cohort.img_names)}

    snp_cols = [snp_index[name] for name in payload["snp_markers"]]
    rows = []
    for visit in range(cohort.n_visits):
        cols = [img_index[name] for name in payload["img_names"][visit]]
        u_sub = u[snp_cols]
        v_sub = v[visit, cols]
        train_gene = _normalize(cohort.snp[train_mask][:, snp_cols]) @ u_sub
        test_gene = _normalize(cohort.snp[~train_mask][:, snp_cols]) @ u_sub
        train_img = _normalize(cohort.img[train_mask][:, visit, :][:, cols]) @ v_sub
        test_img = _normalize(cohort.img[~train_mask][:, visit, :][:, cols]) @ v_sub
        # The released runs calibrate the genetic projection scale on the
        # training partition before reporting RMSE; the correlation is
        # invariant to that rescaling.
        denominator = float(train_gene @ train_gene)
        if np.isfinite(denominator) and denominator > 1e-12:
            scale = float((train_gene @ train_img) / denominator)
            if np.isfinite(scale):
                train_gene = scale * train_gene
                test_gene = scale * test_gene
        cc = float(np.corrcoef(test_gene, test_img)[0, 1]) if np.std(test_gene) and np.std(test_img) else 0.0
        rows.append({
            "task": "association",
            "visit": f"t{visit + 1}",
            "n_train": int(train_mask.sum()),
            "n_test": int((~train_mask).sum()),
            "accuracy": np.nan,
            "f1": np.nan,
            "ccc": abs(cc),
            "rmse": float(np.sqrt(np.mean((test_gene - test_img) ** 2))),
        })
    rows.append({
        "task": "association",
        "visit": "mean",
        "n_train": int(train_mask.sum()),
        "n_test": int((~train_mask).sum()),
        "accuracy": np.nan,
        "f1": np.nan,
        "ccc": float(np.mean([row["ccc"] for row in rows])),
        "rmse": float(np.mean([row["rmse"] for row in rows])),
    })
    return rows


def run(root, config_path, output_dir, seeds=None, mode="train", device: str = "cpu", verbose=True):
    """Train (optional) and evaluate every seed; write tidy result tables."""
    root = Path(root)
    config_path = Path(config_path)
    config = load_config(config_path if config_path.is_absolute() else root / config_path)
    seeds = [int(seed) for seed in (seeds or config["seeds"])]
    output = Path(output_dir)
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)

    metrics_by_seed = []
    for seed in seeds:
        checkpoint = output / "checkpoints" / f"seed_{seed}.pt"
        if mode == "train":
            payload = train_seed(root, config, seed, output / "checkpoints", device, verbose)
        else:
            payload = load_checkpoint(checkpoint, device)
        rows = evaluate_checkpoint(root, config, payload, device)
        for row in rows:
            metrics_by_seed.append({"seed": seed, **row})
        if verbose:
            summary = {row["task"] + "/" + row["visit"]: row for row in rows}
            for key in ("DX13_classification/BL",):
                if key in summary:
                    print(f"[{config['name']}] seed {seed}: accuracy={summary[key]['accuracy']:.4f} "
                          f"f1={summary[key]['f1']:.4f}")

    frame = pd.DataFrame(metrics_by_seed)
    frame.to_csv(output / "metrics_by_seed.csv", index=False)
    summary = protocol.summarize([
        [row for row in metrics_by_seed if row["seed"] == seed] for seed in seeds
    ])
    summary.to_csv(output / "metrics_mean_std.csv", index=False)
    (output / "protocol.json").write_text(json.dumps({
        "config": config,
        "seeds": seeds,
        "mode": mode,
        "subjects": len(build_cohort(root, config)),
    }, indent=2), encoding="utf-8")
    return frame, summary
