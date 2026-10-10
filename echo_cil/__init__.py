"""ECHO-CIL: physics-aware class-incremental learning for CWRU."""

from .model import ECHOModel
from .data import EchoCILDataset

__all__ = ["ECHOModel", "EchoCILDataset"]
