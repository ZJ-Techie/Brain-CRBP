"""Upstream training loop (full-batch Adam over the exact paper objective)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from .model import BrainCRBP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(cohort, train_indices, config: dict, seed: int, device: str = "cpu",
                snp_prior=None, img_prior=None, log=None):
    """Fit one :class:`BrainCRBP` on ``train_indices`` and return it."""
    set_seed(int(seed))
    batch = cohort.batch(train_indices, device=device)
    model = BrainCRBP(
        n_snp=batch["snp"].shape[1],
        n_img=batch["img_visits"].shape[2],
        n_visits=cohort.n_visits,
        contrast_dim=int(config["contrast_dim"]),
        temperature=float(config["temperature"]),
        snp_penalty=snp_prior,
        img_penalty=img_prior,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    history = []
    for epoch in range(int(config["epochs"])):
        model.train()
        optimizer.zero_grad()
        out = model(
            batch,
            lambda_cca=float(config["lambda_cca"]),
            lambda_con=float(config["lambda_con"]),
            lambda_snp=float(config["lambda_snp"]),
            lambda_img=float(config["lambda_img"]),
            lambda_reg=float(config["lambda_reg"]),
        )
        if not bool(torch.isfinite(out["loss_total"])):
            raise FloatingPointError(f"non-finite loss at epoch {epoch}")
        out["loss_total"].backward()
        optimizer.step()
        if log is not None and (epoch % 50 == 0 or epoch == int(config["epochs"]) - 1):
            log(f"  epoch {epoch:4d}  total={float(out['loss_total'].detach()):.6f}  "
                f"cca={float(out['loss_cca'].detach()):.4f}  con={float(out['loss_con'].detach()):.4f}  "
                f"reg={float(out['loss_reg'].detach()):.4f}")
        history.append({
            "epoch": epoch,
            "loss_total": float(out["loss_total"].detach()),
            "loss_cca": float(out["loss_cca"].detach()),
            "loss_con": float(out["loss_con"].detach()),
            "loss_snp_l1": float(out["loss_snp_l1"].detach()),
            "loss_img_l1": float(out["loss_img_l1"].detach()),
            "loss_reg": float(out["loss_reg"].detach()),
        })
    model.eval()
    return model, history


def save_checkpoint(path, model, cohort, config, seed: int, snp_names, img_names, history):
    """Persist weights + the marker lists needed for evaluation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "seed": int(seed),
        "modality": cohort.modality,
        "n_visits": cohort.n_visits,
        "snp_markers": list(snp_names),
        "img_names": [list(names) for names in img_names],
        "config": config,
        "history": history,
    }
    torch.save(payload, path)
    path.with_suffix(".json").write_text(json.dumps({
        "modality": cohort.modality,
        "seed": int(seed),
        "n_visits": cohort.n_visits,
        "snp_markers": list(snp_names),
        "img_markers": [list(names) for names in img_names],
        "config": config,
        "final_loss": history[-1] if history else None,
    }, indent=2), encoding="utf-8")
    return path


def load_checkpoint(path, device: str = "cpu") -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    if "state_dict" not in payload:
        raise ValueError(f"{path} is not a Brain-CRBP checkpoint")
    return payload


def model_from_checkpoint(payload, device: str = "cpu") -> BrainCRBP:
    state = payload["state_dict"]
    model = BrainCRBP(
        n_snp=int(state["gene_weight"].shape[0]),
        n_img=int(state["image_weights"].shape[1]),
        n_visits=int(state["image_weights"].shape[0]),
        contrast_dim=int(state["image_con_head.0.weight"].shape[0]),
        temperature=float(payload["config"]["temperature"]),
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)
