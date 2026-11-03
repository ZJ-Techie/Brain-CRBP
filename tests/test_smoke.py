"""Self-contained checks that need no ADNI data (synthetic cohort)."""

from __future__ import annotations

import numpy as np
import torch

from brain_crbp.data import Cohort
from brain_crbp.model import BrainCRBP, build_pair_masks, directional_orders
from brain_crbp.protocol import ccc, joint_features, top_features, tune_classifier
from brain_crbp.train import model_from_checkpoint, train_model


def _synthetic_cohort(n=60, n_snp=200, n_img=24, n_visits=4, seed=0):
    rng = np.random.default_rng(seed)
    snp = rng.integers(0, 3, size=(n, n_snp)).astype(float)
    img = rng.normal(size=(n, n_visits, n_img))
    dx = rng.integers(1, 4, size=(n, n_visits))
    adas = rng.normal(size=(n, 4))
    mmse = rng.normal(size=(n, 4))
    bl_dx = dx[:, 0].astype(float)
    return Cohort(
        ids=[f"S{i:03d}" for i in range(n)],
        snp=snp, img=img, dx=dx,
        snp_names=[f"snp_{i}" for i in range(n_snp)],
        img_names=[f"roi_{i}" for i in range(n_img)],
        adas=adas, mmse=mmse, bl_dx=bl_dx,
        n_visits=n_visits, modality="vbm",
    )


def test_directional_orders_are_cyclic():
    assert directional_orders(4) == ((1, 2, 3), (2, 3, 0), (3, 0, 1), (0, 1, 2))
    assert directional_orders(2) == ((1,), (0,))


def test_pair_masks_count_changed_subjects():
    dx = torch.tensor([[1, 3], [2, 2], [3, 1]])
    masks, counts = build_pair_masks(dx)
    assert masks.shape == (2, 3, 3)
    assert counts.tolist() == [[2], [2]]
    # subject 1 never changes diagnosis, so it is never a positive anchor
    assert masks[:, 1, :].sum() == 0


def test_top_features_is_stable_for_ties():
    names = ["a", "b", "c"]
    assert top_features([1.0, -1.0, 0.5], names, 2) == ["a", "b"]


def test_ccc_bounds():
    x = np.linspace(0, 1, 20)
    assert abs(ccc(x, x) - 1.0) < 1e-12
    assert ccc(x, -x) < 0


def test_model_trains_and_reduces_loss():
    cohort = _synthetic_cohort()
    config = {
        "epochs": 20, "lr": 0.01, "contrast_dim": 8, "temperature": 0.07,
        "lambda_cca": 20.0, "lambda_con": 0.009, "lambda_snp": 0.03,
        "lambda_img": 0.02, "lambda_reg": 1e-6,
    }
    indices = np.arange(40)
    model, history = train_model(cohort, indices, config, seed=1)
    assert history[0]["loss_total"] > history[-1]["loss_total"]
    assert model.gene_weight.shape == (200, 1)
    assert model.image_weights.shape == (4, 24, 1)


def test_checkpoint_round_trip(tmp_path):
    cohort = _synthetic_cohort()
    config = {
        "epochs": 3, "lr": 0.01, "contrast_dim": 8, "temperature": 0.07,
        "lambda_cca": 1.0, "lambda_con": 0.0, "lambda_snp": 0.0,
        "lambda_img": 0.0, "lambda_reg": 0.0,
    }
    model, _ = train_model(cohort, np.arange(40), config, seed=2)
    payload = {"state_dict": model.state_dict(), "config": config}
    restored = model_from_checkpoint(payload)
    batch = cohort.batch(np.arange(20))
    with torch.no_grad():
        first = model(batch, **{"lambda_cca": 1.0})["score"]
        second = restored(batch, **{"lambda_cca": 1.0})["score"]
    assert torch.allclose(first, second)


def test_joint_features_layout():
    snp = np.arange(6).reshape(3, 2)
    img = np.arange(12).reshape(3, 2, 2)
    joined = joint_features(snp, img)
    assert joined.shape == (3, 6)
    assert np.array_equal(joined[:, :2], snp)
    assert np.array_equal(joined[:, 2:4], img[:, 0])


def test_tuning_selects_from_grid():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 5))
    labels = np.repeat([1, 3], 20)
    features[labels == 1, 0] -= 3.0
    params = tune_classifier(features, labels, seed=0, folds=4)
    assert set(params) == {"kernel", "C", "gamma"}
