"""Feature ranking from learned projection weights."""

import numpy as np


def top_features(weights, names, k: int = 10) -> list[str]:
    """Top-k features by descending absolute weight, preserving ties."""
    values = np.asarray(weights, dtype=float).reshape(-1)
    names = list(names)
    if values.size != len(names):
        raise ValueError("weights and feature names must be aligned")
    order = np.argsort(-np.abs(values), kind="stable")[: int(k)]
    return [names[index] for index in order]
