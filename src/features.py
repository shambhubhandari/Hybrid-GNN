# Copied from src/models/common.py — resolve_feature_columns. The feature matrix
# every entry point (training, ablation, SHAP) must agree on, so leaky and honest
# runs score on the identical column set.
from src import config
from src.logging_util import get_logger

logger = get_logger(__name__)


def resolve_feature_columns(df, target: str) -> list:
    """Select exactly the model feature columns declared in config.FEATURES.

    Expands pca_* to the GNN PCA columns and silently skips any declared column
    that is absent from a given arm (tabular has neither pca_* nor
    predicted_band_gap). Mirrors the resolver the paper's pipeline uses.
    """
    feat_cfg = config.models_cfg.get("features", {})
    feature_cols = []
    for group, cols in feat_cfg.items():
        if group == "gnn":
            for c in cols:
                if c == "pca_*":
                    feature_cols.extend([col for col in df.columns if col.startswith("pca_")])
                elif c in df.columns:
                    feature_cols.append(c)
        else:
            for c in cols:
                if c in df.columns:
                    feature_cols.append(c)
                else:
                    logger.warning("Feature column '%s' not in dataframe — skipping.", c)

    if not feature_cols:
        feature_cols = [c for c in df.columns if c != target and c not in config.ID_COLS]
    return feature_cols
