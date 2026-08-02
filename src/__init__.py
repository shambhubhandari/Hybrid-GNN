"""Minimal, self-contained slice of the paper's pipeline used by the audit.

The notebook imports the exact functions the paper's stages use — the 70/15/15
split, the feature resolver, XGBoost training, the leaky-column dropper, and
metrics — so a reviewer sees the real code, not a re-implementation.
"""
__all__ = ["config", "leakage", "protocol", "features", "gbdt", "wrappers", "metrics"]
