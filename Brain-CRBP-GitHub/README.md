# Brain-CRBP

Core implementation for sparse genotype-phenotype learning with biological
priors and optional external GWAS direction anchoring.

This public release contains the model, core training loop, feature ranking
and generic array interfaces. Research datasets, cohort-specific preprocessing,
pretrained research checkpoints, experiment results, and downstream SVM/SVR
evaluation are excluded. The runnable example generates artificial arrays
in memory; its output is a software demonstration, not a reproduction of
the paper's experiments.

## Install

Run the following commands from the repository root:

```bash
python -m pip install -r requirements.txt
```

Use Python 3.10 or later. No dataset download or external service is needed
for the synthetic demonstration.

## Quick start

Train on artificial inputs without GWAS effects:

```bash
python -m brain_crbp.demo --epochs 20
```

Exercise the optional anchor using artificial signed beta coefficients:

```bash
python -m brain_crbp.demo --epochs 20 --with-gwas
```

Both commands keep arrays in memory and report training losses. They do not
run classification/regression validation or create data files. To save a
checkpoint from the synthetic demonstration, add `--out outputs/demo`.

## Core model

The model learns a shared genetic projection `u` and a phenotype projection
`v_t` for each visit:

```text
g = X u
h_t = Z_t v_t
s = g + sum_t h_t

L = lambda_cca * L_cca
  + lambda_con * L_con
  + lambda_snp * sum_j w_gene_j |u_j|
  + lambda_img * sum_t sum_j w_pheno_j |v_tj|
  + lambda_reg * sum_i (dx_i,last - s_i)^2
  + lambda_gwas / 2 * sum_j mask_j (u_j - gwas_scale * beta_j)^2
```

- The association loss aligns centered, L2-normalized genetic and phenotype
  projections.
- The contrastive term uses within-participant visits with different
  diagnosis codes, with other visits forming the candidate pool.
- Fixed positive penalty weights can encode feature-level biological priors.
  Higher importance can be represented by a smaller L1 penalty weight.
- The existing prediction term is squared error against the last-visit
  diagnosis code. Its definition and coefficient are retained in this release.
- The entire GWAS term is omitted when `gwas_beta=None`. With beta input,
  `lambda_gwas` corresponds to the anchoring strength and `gwas_scale` to its
  rescaling factor. Only covered SNPs participate in the anchor.

Default training parameters are in [`configs/model.json`](configs/model.json).
The demo overrides only the requested number of epochs.

## Supply your own arrays

The caller prepares and aligns the arrays before training:

| Array | Shape | Meaning |
|---|---|---|
| `snp` | `(N, P)` | Genotype features |
| `img` | `(N, T, Q)` | Longitudinal phenotype features |
| `dx` | `(N, T)` | Diagnosis codes at each visit |

Construct a `brain_crbp.Cohort` with these arrays, matching subject and feature
identifiers, and the number of visits. Then call:

```python
import json
import numpy as np
from brain_crbp import train_model

with open("configs/model.json", encoding="utf-8") as handle:
    config = json.load(handle)

# cohort is a Cohort constructed from your own authorized, aligned arrays.
train_indices = np.asarray(your_training_indices, dtype=int)
model, history = train_model(cohort, train_indices, config, seed=7)
```

Only the indexed participants are used by the training loop. The release
does not perform cohort-specific preprocessing, train/test splitting, or
downstream evaluation. Feature identifiers can be ranked using
`brain_crbp.top_features` and the learned projection weights.

## Optional biological priors and external GWAS effects

`load_prior(path, id_column, feature_names)` reads fixed positive L1 penalty
weights from a user-supplied CSV with an identifier column and a `wj` column.
Pass the returned tensors as `snp_prior` and/or `img_prior` to `train_model`.
Without these inputs, uniform L1 penalty weights are used. Generation of
LLM/RAG importance scores is outside this core release.

External effects can be loaded with:

```python
from brain_crbp import load_gwas_beta, train_model

beta, mask = load_gwas_beta(
    "path/to/your_external_beta.csv", "snp_id", "beta", cohort.snp_names
)
model, history = train_model(
    cohort, train_indices, config, seed=7, gwas_beta=beta, gwas_mask=mask
)
```

CSV rows are aligned by exact SNP identifier, not file order. Signed effects
must refer to the same coded allele as each genotype feature. No allele
harmonization or automatic sign flipping is performed. Unmatched SNPs are
excluded from the whole anchor, including its L2 part; a supplied beta of
zero is a valid effect. Duplicate identifiers, non-finite effects and files
with no matching SNPs are rejected.

Leaving out `gwas_beta` skips the anchor entirely. Setting `lambda_gwas=0`
also disables its contribution. New checkpoints store the aligned beta
coefficients and mask for restoration without the source CSV.

## Repository layout

```text
brain_crbp/
  model.py       sparse projections, contrastive loss and optional GWAS anchor
  data.py        generic array container and optional CSV prior loaders
  train.py       training and checkpoint I/O
  features.py    feature ranking from projection weights
  demo.py        artificial in-memory training example
configs/
  model.json     model/training parameters only
tests/
  test_core.py   synthetic software tests
```

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

All tests construct artificial inputs. Research data and pretrained weights
are not needed.

## Data and license

Research datasets are not distributed with this repository. Users supply
their own authorized data. Ignore rules exclude data files, model weights,
experiment outputs and local archives from routine Git staging.

Code is distributed under the [MIT license](LICENSE).
