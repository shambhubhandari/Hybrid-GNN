"""Post-DFT proxy ablation: drop the leaky columns and rescore.

The gap between the full-feature and dropped-feature runs is the honest measure
of what the framework adds over feeding it DFT outputs.
"""
from typing import List

import pandas as pd

from src.logging_util import get_logger

logger = get_logger(__name__)

# Post-DFT proxies: direct copies of DFT/MP output labels, unknowable before
# running the calculation the model claims to replace. predicted_band_gap is
# deliberately excluded — it is a GNN's own prediction (a legitimate feature)
# kept in the honest run; its leakage is a separate, embedding-axis mechanism.
POST_DFT_COLUMNS = [
    "is_metal",
    "is_gap_direct",
    "is_stable",
    "is_magnetic",
    "num_magnetic_sites",
]


def drop_leaky_features(X: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """Drop the post-DFT proxy columns (or a custom leaky-column set)."""
    columns = columns or POST_DFT_COLUMNS
    present = [c for c in columns if c in X.columns]
    if not present:
        logger.warning("No leaky columns found in X; nothing dropped.")
    else:
        logger.info("Dropping leaky feature(s): %s", ", ".join(present))
    return X.drop(columns=present, errors="ignore")
