# =============================================================================
# config.py — the *values* the ablation needs, lifted verbatim from the full
# repo's src/config.py + configs/models.yaml. Kept as a plain module so the
# audit codebase has no YAML / path machinery to carry.
# =============================================================================

TARGET = "band_gap"
RANDOM_STATE = 42
ID_COLS = ["material_id", "formula_pretty"]

# The exact XGBoost hyper-parameters the paper's pipeline uses (Optuna-tuned).
XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "learning_rate": 0.03865291106813676,
    "max_depth": 9,
    "min_child_weight": 11.689593895871,
    "subsample": 0.8809231752905414,
    "colsample_bytree": 0.864323016559627,
    "colsample_bylevel": 0.9986567609362819,
    "gamma": 0.0015113421306209107,
    "reg_alpha": 24.332514099721607,
    "reg_lambda": 3.062237425668688,
    "tree_method": "hist",
    "eval_metric": "rmse",
    "num_boost_round": 1336,
    "early_stopping_rounds": 100,
}

# The model feature groups (configs/models.yaml -> features). resolve_feature_columns
# selects exactly these from the feature table; pca_* expands to the GNN PCA columns.
FEATURES = {
    "physical": [
        "MagpieData maximum Electronegativity", "MagpieData mean CovalentRadius",
        "MagpieData mean NUnfilled", "MagpieData mean NsValence", "nelements",
        "volume", "density", "density_atomic", "is_stable", "is_gap_direct",
        "is_metal", "is_magnetic", "num_magnetic_sites", "num_unique_magnetic_sites",
        "is_transition_metal",
    ],
    "target_encoded": ["spacegroup_symbol", "crystal_system", "ordering"],
    "derived": ["soc_avg"],
    "gnn": ["predicted_band_gap", "pca_*"],
}

# Mirror the interface resolve_feature_columns / train_xgboost expect.
models_cfg = {"models": {"xgboost": XGBOOST_PARAMS}, "features": FEATURES}
