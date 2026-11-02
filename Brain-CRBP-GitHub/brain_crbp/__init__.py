"""Core Brain-CRBP model and training interfaces."""

# Load NumPy before Torch for Windows environments with MKL/OpenMP libraries.
import numpy as _np

from .model import BrainCRBP, build_pair_masks, directional_orders
from .data import Cohort, load_gwas_beta, load_prior
from .features import top_features
from .train import load_checkpoint, model_from_checkpoint, save_checkpoint, train_model

__all__ = [
    "BrainCRBP", "Cohort", "build_pair_masks", "directional_orders",
    "load_prior", "load_gwas_beta", "top_features", "train_model",
    "save_checkpoint", "load_checkpoint", "model_from_checkpoint",
]

__version__ = "1.0.0"
