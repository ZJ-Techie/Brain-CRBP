"""Cohort loading and leakage-safe preprocessing.

Two data sources are used by the paper:

* ``SNP-VBM`` / ``SNP-FreeSurfer``: 409 ADNI participants, 10,000 SNPs and 90
  regional imaging traits measured at four visits (BL, M6, M12, M24).
* ``SNP-Protein``: 273 participants with complete plasma protein observations
  at baseline and month 12 (146 paired markers).

Clinical targets are ADAS and MMSE at BL/M6/M12/M24.  ``BL_DX`` (1 = healthy
control, 2 = MCI, 3 = AD) supervises the downstream SVM; the joint regression
term inside the model uses the last observed diagnosis code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

VISITS = ("t1", "t2", "t3", "t4")
SCORE_VISITS = ("BL", "M6", "M12", "M24")
SCORE_SHEETS = {
    "ADAS": ("ADAS_BL", "ADAS_M6", "ADAS_M12", "ADAS_M24"),
    "MMSE": ("MMSE_SC", "MMSE_M6", "MMSE_M12", "MMSE_M24"),
}


def _as_names(values) -> list[str]:
    names = []
    for value in np.asarray(values, dtype=object).reshape(-1):
        item = np.asarray(value).reshape(-1)
        text = str(item[0] if item.size else "").strip()
        if not text:
            raise ValueError("empty identifier in MAT file")
        names.append(text)
    return names


@dataclass
class Cohort:
    """Aligned arrays for one experiment (subjects on the first axis)."""

    ids: list[str]
    snp: np.ndarray
    img: np.ndarray
    dx: np.ndarray
    snp_names: list[str]
    img_names: list[str]
    adas: np.ndarray
    mmse: np.ndarray
    bl_dx: np.ndarray
    n_visits: int
    modality: str

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


def load_basic_info(path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="Sheet1")
    if "SubjID" not in frame or "BL_DX" not in frame:
        raise ValueError(f"{path} must contain SubjID and BL_DX columns")
    frame = frame.copy()
    frame["SubjID"] = frame["SubjID"].astype(str).str.strip()
    frame["BL_DX"] = pd.to_numeric(frame["BL_DX"], errors="coerce")
    if frame["SubjID"].duplicated().any():
        raise ValueError("duplicate SubjID values in the basic-info table")
    return frame


def load_scores(path, modality: str) -> np.ndarray:
    """Return ``[n_rows, 4]`` ADAS/MMSE values ordered BL, M6, M12, M24."""
    sheets = SCORE_SHEETS[modality.upper()]
    frames = [pd.read_excel(path, sheet_name=sheet) for sheet in sheets]
    n_rows = max(len(frame) for frame in frames)
    out = np.full((n_rows, 4), np.nan, dtype=float)
    for column, frame in enumerate(frames):
        if frame.shape[1] != 1:
            raise ValueError(f"{path} sheet {sheets[column]} must hold exactly one column")
        values = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
        out[: len(values), column] = values
    return out


def load_panel(path) -> dict:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    panel = {
        "ids": _as_names(mat["sbj_id"]),
        "snp": np.asarray(mat["snp_data2"], dtype=float),
        "vbm": np.asarray(mat["Imagingvbm_T"], dtype=float),
        "fs": np.asarray(mat["Imagingfs_T"], dtype=float),
        "snp_names": _as_names(mat["SNP_name_chr19_snp_data2"]),
        "vbm_names": _as_names(mat["VBM_ImagingID"]),
        "fs_names": _as_names(mat["FS_ImagingID"]),
        "dx": np.column_stack(
            [np.asarray(v, dtype=float).reshape(-1) for v in np.asarray(mat["DX_person"], dtype=object).reshape(-1)]
        ),
    }
    if panel["snp"].ndim != 2 or len(panel["ids"]) != panel["snp"].shape[0]:
        raise ValueError("invalid SNP matrix in panel.mat")
    if panel["vbm"].shape[:2] != (len(panel["ids"]), len(panel["vbm_names"])):
        raise ValueError("invalid VBM tensor in panel.mat")
    return panel


def load_imaging_cohort(root, modality: str = "vbm") -> Cohort:
    """Load SNP + VBM/FreeSurfer features aligned to the clinical tables."""
    root = Path(root)
    panel = load_panel(root / "data" / "panel.mat")
    basic = load_basic_info(root / "data" / "longitudinal_basic_infos.xlsx")
    lookup = {subject_id: index for index, subject_id in enumerate(panel["ids"])}
    matched = basic["SubjID"].isin(lookup).to_numpy()
    if int(matched.sum()) != len(panel["ids"]):
        raise ValueError("the imaging panel is not fully covered by the basic-info table")

    basic_matched = basic.loc[matched].reset_index(drop=True)
    order = np.asarray([lookup[subject_id] for subject_id in basic_matched["SubjID"]], dtype=int)
    modality = modality.lower()
    if modality not in {"vbm", "fs"}:
        raise ValueError("modality must be vbm or fs")
    img = panel[modality][order].transpose(0, 2, 1)  # (N, visit, feature)
    snp = panel["snp"][order]
    dx = panel["dx"][order]

    adas = load_scores(root / "data" / "longitudinal_cognitive_scores_ADAS.xlsx", "ADAS")[matched]
    mmse = load_scores(root / "data" / "longitudinal_cognitive_scores_MMSE.xlsx", "MMSE")[matched]
    bl_dx = pd.to_numeric(basic_matched["BL_DX"], errors="coerce").to_numpy(dtype=float)

    if not np.isfinite(snp).all() or not np.isfinite(img).all():
        raise ValueError("imaging cohort contains non-finite feature values")
    if not np.isfinite(dx).all():
        raise ValueError("DX_person must be complete for every visit")
    return Cohort(
        ids=basic_matched["SubjID"].tolist(),
        snp=snp,
        img=img,
        dx=dx.astype(int),
        snp_names=panel["snp_names"],
        img_names=panel["vbm_names"] if modality == "vbm" else panel["fs_names"],
        adas=adas,
        mmse=mmse,
        bl_dx=bl_dx,
        n_visits=4,
        modality=modality,
    )


def _paired_protein_columns(names: list[str]) -> tuple[list[str], list[int], list[int]]:
    metadata = {"RBM_ID", "Sample_Received_Date"}
    baseline, followup = {}, {}
    for index, name in enumerate(names):
        if name.startswith("BL_") and name[3:] not in metadata:
            baseline[name[3:]] = index
        elif name.startswith("M12_"):
            raw = name[4:]
            if raw.endswith("_2"):
                base = raw[:-2]
                if base in metadata or base in followup:
                    continue
            else:
                base = raw
            if base not in metadata:
                followup[base] = index
    paired = sorted(set(baseline).intersection(followup))
    if not paired:
        raise ValueError("no paired BL_/M12_ protein features found")
    return paired, [baseline[name] for name in paired], [followup[name] for name in paired]


def load_protein_cohort(root) -> Cohort:
    """Complete-case SNP-protein cohort (273 participants, two visits)."""
    root = Path(root)
    panel = load_panel(root / "data" / "panel.mat")
    mat = loadmat(root / "data" / "protein" / "fs_vbm_proteomics_integrated.mat",
                  squeeze_me=True, struct_as_record=False)
    raw = np.asarray(mat["Protein_raw_data"], dtype=float)
    missing = np.asarray(mat["Protein_raw_missing_mask"], dtype=bool)
    all_names = _as_names(mat["Protein_feature_names"])
    protein_ids = _as_names(mat["sbj_id"])
    names, bl_idx, m12_idx = _paired_protein_columns(all_names)

    lookup = {subject_id: index for index, subject_id in enumerate(protein_ids)}
    if len(lookup) != len(panel["ids"]) or set(lookup) != set(panel["ids"]):
        raise ValueError("protein cohort must match the SNP panel subject IDs")
    rows = np.asarray([lookup[subject_id] for subject_id in panel["ids"]], dtype=int)
    protein = raw[rows]
    mask = missing[rows]

    protein_dx_all = np.column_stack([
        np.asarray(v, dtype=float).reshape(-1)
        for v in np.asarray(mat["DX_person"], dtype=object).reshape(-1)
    ])
    dx = panel["dx"][:, (0, 2)]
    if not np.array_equal(dx, protein_dx_all[rows][:, (0, 2)]):
        raise ValueError("protein and panel DX_person labels disagree")

    selected = np.asarray([*bl_idx, *m12_idx], dtype=int)
    keep = (~mask[:, selected].any(axis=1)) & np.isfinite(protein[:, selected]).all(axis=1)
    protein = protein[keep][:, selected]
    dx = dx[keep]
    panel_indices = np.flatnonzero(keep)

    basic = load_basic_info(root / "data" / "longitudinal_basic_infos.xlsx")
    basic_lookup = {subject_id: index for index, subject_id in enumerate(basic["SubjID"])}
    ids = [panel["ids"][index] for index in panel_indices]
    basic_rows = np.asarray([basic_lookup[subject_id] for subject_id in ids], dtype=int)
    matched = np.zeros(len(basic), dtype=bool)
    matched[basic_rows] = True
    adas = load_scores(root / "data" / "longitudinal_cognitive_scores_ADAS.xlsx", "ADAS")[matched]
    mmse = load_scores(root / "data" / "longitudinal_cognitive_scores_MMSE.xlsx", "MMSE")[matched]
    bl_dx = pd.to_numeric(basic.iloc[basic_rows]["BL_DX"], errors="coerce").to_numpy(dtype=float)

    visits = np.stack([protein[:, : len(names)], protein[:, len(names):]], axis=1)
    return Cohort(
        ids=ids,
        snp=panel["snp"][panel_indices],
        img=visits,
        dx=dx.astype(int),
        snp_names=panel["snp_names"],
        img_names=names,
        adas=adas,
        mmse=mmse,
        bl_dx=bl_dx,
        n_visits=2,
        modality="protein",
    )


def standardize_visits(img: np.ndarray, train_indices) -> np.ndarray:
    """Train-only z-scoring of every imaging feature across all visits.

    The mean and scale are estimated on the training participants only, so the
    held-out partition never influences preprocessing.
    """
    img = np.asarray(img, dtype=float)
    train = np.asarray(train_indices, dtype=int)
    flat = img[train].transpose(0, 2, 1).reshape(-1, img.shape[2])
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (img - mean[None, None, :]) / scale[None, None, :]


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
