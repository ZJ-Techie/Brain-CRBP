"""Software checks use only arrays generated here, never research data."""

import numpy as np
import pytest
import torch


def _cohort():
    from brain_crbp.data import Cohort

    rng = np.random.default_rng(4)
    n, p, q, t = 24, 12, 6, 2
    return Cohort(
        ids=[f"synthetic_{i}" for i in range(n)],
        snp=rng.integers(0, 3, size=(n, p)).astype(float),
        img=rng.normal(size=(n, t, q)),
        dx=rng.integers(1, 4, size=(n, t)),
        snp_names=[f"variant_{j}_A" for j in range(p)],
        img_names=[f"phenotype_{j}" for j in range(q)],
        n_visits=t,
        modality="synthetic",
    )


def _config(epochs=10):
    return {
        "epochs": epochs, "lr": .01, "contrast_dim": 8, "temperature": .07,
        "lambda_cca": 20., "lambda_con": .009, "lambda_snp": .03,
        "lambda_img": .02, "lambda_reg": 1e-12,
        "lambda_gwas": .5, "gwas_scale": 2.,
    }


def test_public_api_imports_without_downstream_modules():
    import brain_crbp

    assert callable(brain_crbp.train_model)
    assert callable(brain_crbp.top_features)
    assert callable(brain_crbp.load_gwas_beta)


def test_array_cohort_produces_model_inputs():
    cohort = _cohort()
    batch = cohort.batch(np.arange(8))
    assert batch["snp"].shape == (8, 12)
    assert batch["img_visits"].shape == (8, 2, 6)
    assert batch["pair_masks"].shape == (2, 8, 8)


def test_core_training_runs_without_gwas_or_clinical_score_tables():
    from brain_crbp.train import train_model

    cohort = _cohort()
    model, history = train_model(cohort, np.arange(len(cohort)), _config(), seed=5)
    assert model.gwas_beta is None
    assert all(row["loss_gwas"] == 0 for row in history)
    assert all(np.isfinite(row["loss_total"]) for row in history)
    assert history[-1]["loss_total"] < history[0]["loss_total"]


def test_signed_gwas_anchor_matches_hand_calculated_gradient():
    from brain_crbp.model import BrainCRBP

    cohort = _cohort()
    beta = torch.zeros(12)
    beta[:2] = torch.tensor([.5, -.5])
    mask = torch.zeros(12, dtype=torch.bool)
    mask[:2] = True
    model = BrainCRBP(12, 6, n_visits=2, gwas_beta=beta, gwas_mask=mask)
    with torch.no_grad():
        model.gene_weight.fill_(100.)
        model.gene_weight[:2] = torch.tensor([[1.], [-2.]])
    result = model(cohort.batch(), lambda_cca=0, lambda_gwas=4, gwas_scale=2)
    assert result["loss_total"].item() == pytest.approx(2.)
    result["loss_total"].backward()
    expected = torch.zeros(12, 1)
    expected[1] = -4.
    torch.testing.assert_close(model.gene_weight.grad, expected)


def test_beta_loader_aligns_only_supplied_effects(tmp_path):
    from brain_crbp.data import load_gwas_beta

    path = tmp_path / "external_beta.csv"
    path.write_text("snp_id,beta\nvariant_B_T,-0.2\nvariant_A_C,0.4\n", encoding="utf-8")
    beta, mask = load_gwas_beta(path, "snp_id", "beta", ["variant_A_C", "variant_B_T", "variant_C_G"])
    torch.testing.assert_close(beta, torch.tensor([[.4], [-.2], [0.]]))
    assert mask.tolist() == [[True], [True], [False]]


def test_beta_loader_rejects_duplicate_effects(tmp_path):
    from brain_crbp.data import load_gwas_beta

    path = tmp_path / "external_beta.csv"
    path.write_text("snp_id,beta\nvariant_A_C,0.1\nvariant_A_C,0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_gwas_beta(path, "snp_id", "beta", ["variant_A_C"])


@pytest.mark.parametrize("with_gwas", [False, True])
def test_checkpoint_roundtrip_preserves_core_model(tmp_path, with_gwas):
    from brain_crbp.features import top_features
    from brain_crbp.train import load_checkpoint, model_from_checkpoint, save_checkpoint, train_model

    cohort = _cohort()
    beta = torch.linspace(-.2, .2, 12) if with_gwas else None
    config = _config(epochs=3)
    model, history = train_model(cohort, np.arange(len(cohort)), config, 5, gwas_beta=beta)
    snp_names = top_features(model.gene_weight.detach().numpy(), cohort.snp_names, 3)
    img_names = [top_features(model.image_weights[t].detach().numpy(), cohort.img_names, 3)
                 for t in range(cohort.n_visits)]
    path = save_checkpoint(tmp_path / "model.pt", model, cohort, config, 5,
                           snp_names, img_names, history)
    restored = model_from_checkpoint(load_checkpoint(path))
    batch = cohort.batch()
    before = model(batch, lambda_gwas=.5, gwas_scale=2)
    after = restored(batch, lambda_gwas=.5, gwas_scale=2)
    torch.testing.assert_close(before["score"], after["score"])
    torch.testing.assert_close(before["loss_total"], after["loss_total"])


def test_feature_selection_keeps_stable_absolute_weight_order():
    from brain_crbp.features import top_features

    assert top_features([1., -1., .5], ["a", "b", "c"], 2) == ["a", "b"]
