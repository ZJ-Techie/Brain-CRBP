"""Command line entry point: ``python -m brain_crbp.cli <command>``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import pipeline


def _root(args):
    return Path(args.root).resolve()


def cmd_train(args):
    frame, summary = pipeline.run(
        _root(args), args.config, args.out, seeds=args.seeds,
        mode="train", device=args.device, verbose=not args.quiet,
    )
    print(summary.to_string(index=False))
    return frame


def cmd_evaluate(args):
    root = _root(args)
    config = pipeline.load_config(root / args.config if not Path(args.config).is_absolute() else args.config)
    seeds = args.seeds or config["seeds"]
    weights = Path(args.weights)
    rows = []
    for seed in seeds:
        checkpoint = weights / f"seed_{seed}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing checkpoint {checkpoint}")
        payload = pipeline.load_checkpoint(checkpoint, args.device)
        rows.extend({"seed": int(seed), **row} for row in pipeline.evaluate_checkpoint(root, config, payload, args.device))
        print(f"[{config['name']}] seed {seed}: evaluated")
    frame = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "metrics_by_seed.csv", index=False)
    summary = pipeline.protocol.summarize([[row for row in rows if row["seed"] == seed] for seed in seeds])
    summary.to_csv(out / "metrics_mean_std.csv", index=False)
    print(summary.to_string(index=False))
    return frame


def cmd_check(args):
    """Compare a produced summary against the reference numbers in ``results/``."""
    expected = pd.read_csv(Path(args.expected))
    produced = pd.read_csv(Path(args.produced))
    merged = expected.merge(produced, on=["task", "visit"], suffixes=("_ref", "_run"))
    metrics = ["accuracy_mean", "f1_mean", "ccc_mean", "rmse_mean"]
    for metric in metrics:
        merged[metric + "_delta"] = merged[metric + "_run"].fillna(0.0) - merged[metric + "_ref"].fillna(0.0)
    columns = ["task", "visit"] + [column for metric in metrics
                                   for column in (metric + "_ref", metric + "_run", metric + "_delta")]
    print(merged[columns].to_string(index=False))
    deltas = merged[[metric + "_delta" for metric in metrics]].abs()
    failed = merged[(deltas > args.tol).any(axis=1)]
    if len(failed):
        raise SystemExit(f"verification failed for {len(failed)} rows (tol={args.tol})")
    print(f"OK: {len(merged)} rows within {args.tol}")


def build_parser():
    parser = argparse.ArgumentParser("Brain-CRBP")
    parser.add_argument("--root", default=".", help="repository root holding data/ and configs/")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train and evaluate the configured seeds")
    train.add_argument("--config", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--seeds", nargs="*", type=int)
    train.add_argument("--device", default="cpu")
    train.add_argument("--quiet", action="store_true")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate", help="evaluate released checkpoints")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--weights", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--seeds", nargs="*", type=int)
    evaluate.add_argument("--device", default="cpu")
    evaluate.set_defaults(func=cmd_evaluate)

    check = sub.add_parser("check", help="verify a summary against reference numbers")
    check.add_argument("--expected", required=True)
    check.add_argument("--produced", required=True)
    check.add_argument("--tol", type=float, default=0.002)
    check.set_defaults(func=cmd_check)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
