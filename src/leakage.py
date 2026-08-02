# =============================================================================
# leakage.py — the ablation the paper does not run. Copied verbatim from
# src/leakage/leakage.py (the __main__ CLI is omitted; the notebook calls the
# functions directly).
#
# Retrain the tabular stage with the post-DFT proxy columns dropped and report
# the metrics beside the full-feature run. The gap between them is the honest
# measure of what the framework contributes over feeding it DFT outputs.
# =============================================================================
from typing import Dict, List

import pandas as pd

from src.logging_util import get_logger

logger = get_logger(__name__)

LEAKY_COLUMNS = ["predicted_band_gap"]

# Post-DFT proxies: direct copies of DFT/MP output labels, so they cannot be
# known before running the very calculation the ML model claims to replace.
# Feeding these to predict band_gap is target-proxy leakage (Comment paper F1).
# NOTE: predicted_band_gap is deliberately NOT here — it is a GNN's own band-gap
# prediction (a legitimate model output, computable without DFT), so it is kept
# as a feature in the honest run alongside the embeddings. Its leakiness is a
# *different* mechanism (pretrained/in-sample embedding contamination), tracked
# separately below and on the embedding axis, not the boolean-proxy axis.
POST_DFT_COLUMNS = [
    "is_metal",
    "is_gap_direct",
    "is_stable",
    "is_magnetic",
    "num_magnetic_sites",
]

# The GNN's own band-gap prediction. Kept in the honest run (legitimate feature),
# but if you want the full-leak floor it is dropped via ALL_LEAK_COLUMNS.
PRETRAINED_LEAK_COLUMNS = ["predicted_band_gap"]

# Full leak set (5 proxies + the GNN prediction) — only for the "drop everything"
# comparison; the canonical honest run drops POST_DFT_COLUMNS alone.
ALL_LEAK_COLUMNS = POST_DFT_COLUMNS + PRETRAINED_LEAK_COLUMNS


def drop_leaky_features(X: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """Remove the post-DFT proxy features (or a custom leaky-column set)."""
    columns = columns or LEAKY_COLUMNS
    present = [c for c in columns if c in X.columns]
    if not present:
        logger.warning("No leaky columns found in X; nothing dropped.")
    else:
        logger.info("Dropping leaky feature(s): %s", ", ".join(present))
    return X.drop(columns=present, errors="ignore")


def ablation_report(with_feature: Dict[str, float], without_feature: Dict[str, float]) -> Dict[str, float]:
    """Side-by-side metrics plus the delta attributable to the leaky features."""
    delta = {f"delta_{k}": with_feature[k] - without_feature[k] for k in ("r2", "mae", "mse")}
    report = {"with_predicted_band_gap": with_feature,
              "without_predicted_band_gap": without_feature, **delta}
    logger.info("Ablation: R2 %.4f -> %.4f (delta %+.4f)",
                with_feature["r2"], without_feature["r2"], delta["delta_r2"])
    return report
