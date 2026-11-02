"""Generic array inputs and optional prior loaders. No cohort data are bundled."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass
class Cohort:
    """User-supplied arrays, with participants on the first axis.

    snp: (N, P), img: (N, T, Q), dx: (N, T).
    The caller supplies aligned, preprocessed arrays and feature identifiers.
    """

    ids: list[str]
    snp: np.ndarray
    img: np.ndarray
    dx: np.ndarray
    snp_names: list[str]
    img_names: list[str]
    n_visits: int
    modality: str = "phenotype"

    def __len__(self) -> int:
        return len(self.ids)

    def batch(self, indices=None, device="cpu") -> dict:
        snp = self.snp if indices is None else self.snp[indices]
        img = self.img if indices is None else self.img[indices]
        dx = self.dx if indices is None else self.dx[indices]
        from .model import build_pair_masks

        dx_tensor = torch.as_tensor(np.ascontiguousarray(dx), dtype=torch.long, device=device)
        masks, _ = build_pair_masks(dx_tensor)
        return {
            "snp": torch.as_tensor(np.ascontiguousarray(snp), dtype=torch.float32, device=device),
            "img_visits": torch.as_tensor(np.ascontiguousarray(img), dtype=torch.float32, device=device),
            "dx_visits": dx_tensor,
            "pair_masks": masks.to(device),
        }


def load_prior(path, column: str, feature_names) -> torch.Tensor:
    """Read fixed positive penalty weights by exact feature name from a CSV."""
    frame = pd.read_csv(path)
    if column not in frame or "wj" not in frame:
        raise ValueError(f"{path} must contain {column} and wj columns")
    weights = dict(zip(frame[column].astype(str).str.strip(), frame["wj"].astype(float)))
    missing = [name for name in feature_names if name not in weights]
    if missing:
        raise ValueError(f"{path} is missing {len(missing)} features, first: {missing[:3]}")
    values = np.asarray([weights[name] for name in feature_names], dtype=np.float32)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("prior weights must be finite and positive")
    return torch.from_numpy(values).reshape(-1, 1)


def load_gwas_beta(path, column: str, beta_column: str, feature_names):
    """Align signed effects to SNP names, excluding uncovered features.

    Effects must already refer to the allele encoded in each SNP identifier.
    Allele harmonization and sign flipping are the caller's responsibility.
    """
    frame = pd.read_csv(path, dtype={column: "string"})
    if column not in frame or beta_column not in frame:
        raise ValueError(f"{path} must contain {column} and {beta_column} columns")
    ids = frame[column].str.strip()
    if ids.isna().any() or ids.eq("").any():
        raise ValueError("GWAS SNP identifiers must not be empty")
    if ids.duplicated().any():
        raise ValueError("duplicate SNP identifiers in GWAS beta file")
    effects = pd.to_numeric(frame[beta_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(effects).all():
        raise ValueError("GWAS beta coefficients must be numeric and finite")
    lookup = dict(zip(ids.tolist(), effects))
    names = list(feature_names)
    covered = np.asarray([name in lookup for name in names], dtype=bool)
    if not covered.any():
        raise ValueError("no GWAS SNP identifiers match the cohort; check SNP IDs and effect alleles")
    beta = np.asarray([lookup.get(name, 0.) for name in names], dtype=np.float32)
    return torch.from_numpy(beta).reshape(-1, 1), torch.from_numpy(covered).reshape(-1, 1)
