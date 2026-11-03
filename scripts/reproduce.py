"""Reproduce every released setting: evaluate the frozen checkpoints."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SETTINGS = ("snp_vbm", "snp_vbm_assoc", "snp_fs", "snp_protein")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="runs")
    parser.add_argument("--settings", nargs="*", default=list(SETTINGS))
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    for name in args.settings:
        out = Path(args.out) / name
        command = [sys.executable, "-X", "utf8", "-m", "brain_crbp.cli", "--root", str(root),
                   "evaluate", "--config", f"configs/{name}.json",
                   "--weights", f"weights/{name}", "--out", str(out)]
        print(f"== {name} ==")
        subprocess.run(command, check=True, cwd=root)
        subprocess.run([sys.executable, "-X", "utf8", "-m", "brain_crbp.cli", "--root", str(root),
                        "check", "--expected", f"results/reference/{name}.csv",
                        "--produced", str(out / "metrics_mean_std.csv")], check=True, cwd=root)


if __name__ == "__main__":
    main()
