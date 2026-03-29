"""ML strategy utilities: feature engineering and labeling."""

from strategies.ml.labeler import TripleBarrierLabeler
from strategies.ml.mss_entry_strategy import MSSEntryStrategy

try:  # pragma: no cover - dependency optional during test collection
    from strategies.ml.feature_engineer import FeatureEngineer
except ImportError:  # pragma: no cover - pandas-ta may be unavailable
    FeatureEngineer = None  # type: ignore[assignment]

__all__ = ["FeatureEngineer", "TripleBarrierLabeler", "MSSEntryStrategy"]
