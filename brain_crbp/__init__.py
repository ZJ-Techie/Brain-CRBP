"""Brain-CRBP: causal-reasoning and biological-prior guided imaging genetics."""

from .model import BrainCRBP, build_pair_masks, directional_orders
from .protocol import ccc, joint_features, top_features
from .train import load_checkpoint, train_model

__all__ = [
    "BrainCRBP",
    "build_pair_masks",
    "directional_orders",
    "ccc",
    "joint_features",
    "top_features",
    "train_model",
    "load_checkpoint",
]
__version__ = "1.0.0"
