"""XQP — Cross-layer Query-aware Predictor.

Public surface:
- features.extract(...)
- predictor.ClosedFormXQP / TinyMLPXQP
- policy.XQPPolicy
- trace.TraceCollector
"""

from .features import extract_features, FEATURE_NAMES, FEATURE_DIM
from .predictor import ClosedFormXQP, PairwiseXQP, TinyMLPXQP
from .policy import XQPPolicy

__all__ = [
    "extract_features",
    "FEATURE_NAMES",
    "FEATURE_DIM",
    "ClosedFormXQP",
    "PairwiseXQP",
    "TinyMLPXQP",
    "XQPPolicy",
]
__version__ = "0.1.0"
