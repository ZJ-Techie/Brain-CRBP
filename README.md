# Brain-CRBP

Reference implementation of **Brain-CRBP: Multimodal Causal Reasoning and
LLM-Informed Biological Priors for Imaging Genetics**.

The model learns a sparse genotype projection `u` and one sparse
genotype–phenotype projection `v_t` per visit, and trains them with four
terms: a CCA association residual, a within-participant contrastive
regulariser, prior-weighted L1 penalties, and a joint clinical regression.
Selected biomarkers are then scored with an SVM/SVR downstream model.

This repository is the code of the paper. The processed cohorts, the frozen
LLM/GWAS priors and the 12 released checkpoints (three seeds × four settings)
ship in the accompanying archive, not in git (ADNI data cannot be
redistributed through GitHub).

---

## 1. Install

```bash
python -m pip install -r requirements.txt
```

Tested with Python 3.11, `torch 2.11` (CPU is enough), `scikit-learn 1.5`.

## 2. Data layout (archive)

```
data/
  panel.mat                                   # 409 x 10,000 SNPs, 90 x 4 VBM and FreeSurfer traits
  longitudinal_basic_infos.xlsx               # SubjID, BL_DX (1 HC / 2 MCI / 3 AD)
  longitudinal_cognitive_scores_ADAS.xlsx     # BL / M6 / M12 / M24
  longitudinal_cognitive_scores_MMSE.xlsx     # BL / M6 / M12 / M24
  protein/fs_vbm_proteomics_integrated.mat    # raw plasma protein, 273 complete cases x 146 x 2 visits
  priors/snp_weights_eta1.csv                 # frozen LLM-RAG penalties  w_j = I_j^(-eta)
  priors/vbm_weights_eta1.csv
  priors/fs_weights_eta1.csv
  priors/protein_weights_eta1.csv
weights/
  snp_vbm/seed_*.pt          # clinical SNP-VBM model (lambda_con = 0.009)
  snp_vbm_assoc/seed_*.pt    # association SNP-VBM model (lambda_con = 0.003, lambda_reg = 0)
  snp_fs/seed_*.pt
  snp_protein/seed_*.pt
results/reference/*.csv      # reference metric tables reproduced by the command below
```

Each `seed_*.pt` stores the weights, the selected Top-10 marker lists, the
train-only preprocessing statistics and the frozen downstream
hyper-parameters, so evaluation needs nothing but the archive.

## 3. Quick start

Evaluate the released checkpoints (no training, ~1 min on CPU):

```bash
python -m brain_crbp.cli --root . evaluate \
    --config configs/snp_vbm.json --weights weights/snp_vbm --out runs/snp_vbm

python -m brain_crbp.cli --root . check \
    --expected results/reference/snp_vbm.csv --produced runs/snp_vbm/metrics_mean_std.csv
```

Retrain one setting from scratch and evaluate it:

```bash
python -m brain_crbp.cli --root . train \
    --config configs/snp_vbm.json --out runs/snp_vbm_retrain --seeds 20260907 20260908 20260909
```

`configs/` holds the four frozen configurations. All commands take
`--seeds`; the reported runs use `20260907 20260908 20260909`.

## 4. Reproduced results (three seeds, mean ± sample SD)

| Setting | Accuracy | Macro-F1 | ADAS CCC | MMSE CCC | Association CC |
|---|---|---|---|---|---|
| SNP–VBM (clinical) | 0.859 ± 0.013 | 0.854 ± 0.011 | 0.544 ± 0.081 | 0.460 ± 0.054 | 0.206 ± 0.044 |
| SNP–VBM (association) | – | – | – | – | 0.230 ± 0.017 |
| SNP–FreeSurfer | 0.919 ± 0.013 | 0.913 ± 0.016 | 0.703 ± 0.029 | 0.639 ± 0.045 | 0.323 ± 0.110 |
| SNP–Protein | 0.852 ± 0.085 | 0.849 ± 0.084 | – | – | – |

`results/reference/` contains the full per-task tables (`metrics_mean_std.csv`
format), including the per-visit CCC/RMSE. Loading the released checkpoints
reproduces these numbers to the fourth decimal.

Retraining on CPU reproduces the method but not bit-identical weights: the
original runs were executed on GPU, and floating-point reduction order differs.
Expect small deviations (typically one held-out subject).

## 5. Method

For participant `i`, visit `t`, genotype `x_i ∈ R^p`, phenotype `z_i^(t) ∈ R^q`:

```
s_i  = x_i^T u + sum_t z_i^(t) T v_t                                  (score, Eq. 12)
g    = X u,      h^(t) = Z^(t) v^(t)                                  (Eq. 8)

L =  lambda_cca * sum_t ||N(g) - N(h^(t))||^2 / n                      (CCA, Eq. 9-10)
   + lambda_con * sum_t L_rep,t                                       (contrast, Eq. 1-2)
   + lambda_snp * sum_j w_j^SNP |u_j|                                 (Eq. 3-4)
   + lambda_img * sum_t sum_j w_j^IMG |v_tj|                          (Eq. 5)
   + lambda_reg * ||d_i^(T) - s_i||_2^2                               (Eq. 13)
```

`L_rep,t` is the CDReg-style `-SupCon`: the anchor is the participant's visit
`t`, the candidate pool is every participant at every *other* visit, and the
positive set is the same participant at a visit with a **different** diagnosis.
Minimising it with `lambda_con > 0` pushes those pairs apart, i.e. it removes
the individual-specific component that both visits share. All candidate
participants stay in the denominator, so stable participants are still
negatives.

Code map:

| Paper | Code |
|---|---|
| Eq. 1–2, directional masks | `brain_crbp/model.py: directional_orders`, `build_pair_masks`, `BrainCRBP.contrastive` |
| Eq. 8–10, CCA residual | `BrainCRBP.cca_residual` |
| Eq. 3–7, prior-weighted L1 | `BrainCRBP.forward` (`loss_snp_l1`, `loss_img_l1`) |
| Eq. 11 global objective | `BrainCRBP.forward` |
| Eq. 12–14, clinical head | `score`, `loss_reg` |
| Top-k biomarker export | `brain_crbp/protocol.py: top_features` |
| Downstream SVM/SVR, CCC | `brain_crbp/protocol.py: evaluate_holdout` |
| Training loop | `brain_crbp/train.py: train_model` |

### Priors

`w_j = I_j^(-eta)` with `eta = 1`. `I_j` is the LLM importance score produced by
RABPS: the feature annotation and the RAG-retrieved evidence are passed to the
LLM, which returns a per-feature relevance score for AD. The scores are frozen
before training and only weight the sparsity penalty; they are never derived
from fitted coefficients. `data/priors/*.csv` store the resulting `w_j`.

## 6. Protocol

* **SNP–VBM / SNP–FreeSurfer** — subject-level stratified 80/20 split per seed
  (`StratifiedShuffleSplit`, `test_size=0.2`, `random_state=seed`). Upstream
  training, Top-10 selection, preprocessing statistics and the downstream
  SVM/SVR are fitted on the training 80% only; the held-out 20% is scored once.
* **SNP–Protein** — same 80/20 split over the 273 complete-case participants.
  Only diagnosis (DX = 1 vs 3) is reported for this cohort.
* **Association metrics** — the genotype and phenotype projections of the
  Top-10 markers are normalised independently on train/test, the genetic
  scale is calibrated on the training partition, and CC/RMSE are reported on
  the held-out partition, averaged over visits.
* **Downstream models** — `StandardScaler` + RBF SVM (`class_weight="balanced"`)
  for diagnosis, RBF SVR for ADAS/MMSE. The hyper-parameters used by the
  released checkpoints are stored in the checkpoint
  (`downstream_params`); setting `"use_frozen": false` and `"tune": true` in a
  config re-selects them with inner cross-validation on the training partition
  only (`4 x 3` SVM grid, `4 x 3 x 3` SVR grid).

Metrics: accuracy, macro-F1 with labels `[1, 3]`, Lin's CCC and RMSE.

## 7. Repository layout

```
brain_crbp/
  model.py      # BrainCRBP, pair masks, losses
  data.py       # cohort loaders, priors, train-only standardisation
  protocol.py   # splits, Top-k markers, downstream SVM/SVR, metrics
  train.py      # training loop, checkpoint I/O
  pipeline.py   # end-to-end orchestration
  cli.py        # train / evaluate / check
configs/        # frozen configurations for the four settings
tests/          # synthetic smoke tests (no ADNI data required)
scripts/        # reproduce.py helper
results/        # reference metric tables
```

## 8. Tests

```bash
python -m pytest tests -q
```

The suite builds a synthetic cohort, so it runs without the archive.

## 9. Notes and limitations

* The released checkpoints reproduce the paper's tables; retraining depends on
  hardware reduction order (see §4).
* The protein cohort supports diagnosis only in the paper; its cognitive
  scores are not reported and are disabled in `configs/snp_protein.json`.
* ADNI data use is governed by the ADNI Data Use Agreement. `data/` is
  therefore distributed with the archive, not via this repository.

## Citation

```bibtex
@article{zhang2026braincrbp,
  title   = {Brain-CRBP: Multimodal Causal Reasoning and LLM-Informed Biological Priors for Imaging Genetics},
  author  = {Zhang, Jin and Li, Xingle and Wang, Ying and others},
  journal = {Under review},
  year    = {2026}
}
```
