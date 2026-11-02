"""Exercise core training using entirely artificial, in-memory arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import Cohort
from .features import top_features
from .train import save_checkpoint, train_model


def synthetic_cohort(seed: int = 7) -> Cohort:
    rng = np.random.default_rng(seed)
    n, p, q, visits = 40, 128, 24, 4
    return Cohort(
        ids=[f"synthetic_{i}" for i in range(n)],
        snp=rng.integers(0, 3, size=(n, p)).astype(float),
        img=rng.normal(size=(n, visits, q)),
        dx=rng.integers(1, 4, size=(n, visits)),
        snp_names=[f"variant_{j}_A" for j in range(p)],
        img_names=[f"phenotype_{j}" for j in range(q)],
        n_visits=visits,
        modality="synthetic",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "model.json"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--with-gwas", action="store_true", help="use artificial signed effects for this demonstration")
    parser.add_argument("--out", type=Path, help="optionally save a synthetic-demo checkpoint")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config["epochs"] = args.epochs
    cohort = synthetic_cohort(args.seed)
    beta = None
    if args.with_gwas:
        rng = np.random.default_rng(args.seed + 1)
        beta = torch.from_numpy(rng.normal(0, .1, size=cohort.snp.shape[1]).astype(np.float32))
    print("Synthetic arrays only; no research data or downstream evaluation.")
    print(f"SNP={cohort.snp.shape}, phenotype={cohort.img.shape}, GWAS enabled={beta is not None}")
    model, history = train_model(
        cohort, np.arange(len(cohort)), config, args.seed, device=args.device,
        gwas_beta=beta, log=print,
    )
    snp_names = top_features(model.gene_weight.detach().cpu().numpy(), cohort.snp_names)
    img_names = [
        top_features(model.image_weights[t].detach().cpu().numpy(), cohort.img_names)
        for t in range(cohort.n_visits)
    ]
    print(f"Training completed: {len(history)} epochs")
    print(f"loss_total: {history[0]['loss_total']:.6f} -> {history[-1]['loss_total']:.6f}")
    print(f"Selected synthetic SNP identifiers: {snp_names}")
    if args.out is not None:
        path = save_checkpoint(args.out / "synthetic_model.pt", model, cohort, config,
                               args.seed, snp_names, img_names, history)
        print(f"Synthetic-demo checkpoint: {path}")


if __name__ == "__main__":
    main()
