"""src — the minimal, self-contained slice of the paper's pipeline needed
to reproduce the leakage ablation. Every module here is copied verbatim from the
full audit codebase (only the import lines and a local config are adapted), so a
reviewer sees the *exact* functions used, not a re-implementation.

Provenance (full repo path -> here):
  src/leakage/leakage.py       -> src/leakage.py
  src/leakage/protocol.py      -> src/protocol.py
  src/models/common.py         -> src/features.py   (resolve_feature_columns)
  src/models/gbdt.py           -> src/gbdt.py        (train_xgboost)
  src/models/wrappers.py       -> src/wrappers.py
  src/evaluation/metrics.py    -> src/metrics.py
  src/utils/logging.py         -> src/logging_util.py
  src/config.py (values only)  -> src/config.py
"""
__all__ = ["config", "leakage", "protocol", "features", "gbdt", "wrappers", "metrics"]
