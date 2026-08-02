==================================================
Audit: Comment on Aydin et al.
==================================================

.. image:: https://img.shields.io/badge/Python-3.11-blue.svg
   :target: https://www.python.org/
   :alt: Python 3.11

.. image:: https://img.shields.io/badge/Notebook-Jupyter-orange.svg
   :target: https://jupyter.org/
   :alt: Jupyter

.. image:: https://img.shields.io/badge/XGBoost-2.1.4-brightgreen.svg
   :target: https://xgboost.readthedocs.io/
   :alt: XGBoost 2.1.4

.. image:: https://img.shields.io/badge/scikit--learn-1.3.2-F7931E.svg
   :target: https://scikit-learn.org/
   :alt: scikit-learn 1.3.2

.. image:: https://img.shields.io/badge/Data-10.5281%2Fzenodo.18481208-blue.svg
   :target: https://doi.org/10.5281/zenodo.18481208
   :alt: Dataset DOI

.. image:: https://img.shields.io/badge/Reproducible-Run%20All-success.svg
   :alt: Reproducible: Run All

**Reproducible evidence for the Comment on** *"Hybrid Graph–Machine Learning
Framework for Accurate and Interpretable Band Gap Prediction"*
(`J. Chem. Inf. Model. 2026, 66, 3787
<https://pubs.acs.org/jcisd8/article/66/7/3787/5138424/Hybrid-Graph-Machine-Learning-Framework-for>`_),
computed on the authors' published dataset
(`Zenodo 10.5281/zenodo.18481208 <https://doi.org/10.5281/zenodo.18481208>`_,
132,364 entries).

Quick start
===========

.. code-block:: bash

   python -m venv .venv && source .venv/bin/activate      # Python 3.11
   pip install -r requirements.txt
   jupyter notebook notebook/audit_leakage.ipynb          # Run All

Runs in a few minutes on a laptop CPU.

Layout
======

::

    audit/
    ├── notebook/
    │   └── audit_leakage.ipynb     # The audit: live checks + leakage decomposition
    ├── src/                        # Pipeline functions copied verbatim from the full repository
    │   ├── protocol.py             # 70/15/15 stratified split
    │   ├── features.py             # Feature resolver
    │   ├── gbdt.py                 # train_xgboost
    │   ├── leakage.py              # Leaky-column dropper
    │   ├── metrics.py              # R², MAE, MSE
    │   ├── config.py               # Optuna-tuned XGBoost parameters
    │   └── wrappers.py             # Model wrappers
    ├── data/                       # Feature tables for all four arms (~150 MB)
    │   ├── features_tabular.parquet
    │   ├── features_cgcnn.parquet
    │   ├── features_megnet.parquet
    │   ├── features_schnet.parquet
    │   ├── composition.parquet     # Composition table
    │   ├── eval_material_ids.parquet
    │   └── megnet_2018_train_ids.parquet
    ├── outputs/                    # Cached stacking-run JSON (decomposition renders
    │   └── stacking_*.json         #   without retraining)
    ├── figures/                    # Figures written by the notebook
    │   ├── shap_attribution_all_arms.png
    │   └── shap_is_metal.png
    └── requirements.txt            # Pinned dependencies

Software environment
====================

Every number in the Comment is produced by the stack below. Versions are pinned
in ``requirements.txt`` for bit-for-bit agreement with the manuscript.

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Package
     - Version
     - Role in the audit
   * - ``numpy``
     - 1.23.5
     - Numerics underlying every step
   * - ``pandas``
     - 1.5.3
     - Feature tables, identifier joins, overlap checks
   * - ``pyarrow``
     - 17.0.0
     - Parquet I/O for ``data/``
   * - ``scikit-learn``
     - 1.3.2
     - Stratified split, RidgeCV meta-learner, R²/MAE/MSE
   * - ``xgboost``
     - 2.1.4
     - Base regressor for the ablation; exact TreeSHAP via ``pred_contribs`` (§4.4)
   * - ``matplotlib``
     - 3.7.5
     - Figures written to ``figures/``
   * - ``jupyter``
     - 1.0.0
     - Runs ``notebook/audit_leakage.ipynb``

What is precomputed, and why
----------------------------

The audit has two tiers. Every **live check** — the leakage evidence in §1–§3,
the ablation in §4.2, and the TreeSHAP attribution in §4.4 — runs here on CPU in
minutes using a single XGBoost regressor, which establishes the *direction* of
each effect. The **headline figures** come from a six-model RidgeCV stacking
ensemble that needs GPU embedding extraction and hours of base-model fitting on
an HPC cluster; those results are bundled as small JSON files so §4.1 and §4.3
render instantly. Nothing in ``outputs/`` is required for the live ablation.

Consequently the graph networks and the non-XGBoost base learners are **not
retrained here and are not dependencies of this repository.** They enter as
fixed inputs:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Input
     - Provenance
   * - ``features_{cgcnn,schnet}.parquet``
     - Embeddings regenerated **held-out** under ``--protocol-split``, with the
       test materials withheld — the leakage-free counterfactual
   * - ``features_megnet.parquet``
     - Pretrained ``Bandgap_MP_2018`` inference; **cannot** be de-contaminated,
       so MEGNet is excluded from the honest floor and stands as an upper bound
   * - ``megnet_2018_train_ids.parquet``
     - MEGNet's 2018.6.1 training identifiers, intersected with the evaluation
       set for the 30.4% contamination figure (§1.2)
   * - ``stacking_{arm}.json``
     - As-published RidgeCV ensemble R²/MAE/MSE — the leaky headline
   * - ``stacking_{arm}_honest.json``
     - Same ensemble with the five post-DFT proxies dropped

Each GNN block contributes ``predicted_band_gap`` plus ``pca_1``–``pca_32``, the
PCA-reduced penultimate-layer embedding. The 19 tabular features are common to
all four arms: four Magpie descriptors from matminer's ``ElementProperty``, ten
structural and Materials Project fields, and the five post-DFT proxies.

Stacking base learners
----------------------

The ensemble searches every subset of six base models and keeps the best-scoring
one — ``xgboost``, ``lightgbm``, ``catboost``, ``random_forest``, ``mlp``, and
``ftt`` (FT-Transformer). Recovering the winning subset from each arm's
``best_combo`` returns **five or six** members in every case, never the three the
paper reports (§3.2). Only ``xgboost`` is installed here; the other five are
represented solely by their cached ensemble output.

Contents
========

**§1 Data leakage.**
Four mechanisms, each traced to the authors' code with a live check where
possible: post-DFT tabular proxies (``is_metal`` coincides with
``band_gap = 0``; metals are 53.2% of the dataset), pretrained MEGNet
contamination (30.4% identifier overlap with MEGNet's 2018.6.1 set), SchNet
trained without a holdout, and CGCNN's uncoordinated split.

**§2 Feature misattribution.**
``is_metal``, ``is_gap_direct``, and ``is_stable`` are Materials Project labels
rather than the Magpie descriptors the paper describes.

**§3 Methodological discrepancies.**
A single static split rather than 10-fold cross-validation; an optimal ensemble
of five or six models rather than three; ``num_boost_round = 1336`` above the
stated Optuna bound; a single DFTB spin–orbit table rather than a hybrid scheme.

**§4 Ablation and decomposition.**
Faithful reproduction of the authors' reported metrics (MEGNet 0.921, CGCNN
0.909, SchNet 0.871), recovered to within 0.012 R²; the single-variable
leakage-removal ablation; per-arm leakage-free results measured against a
tabular baseline; and TreeSHAP attribution.

Method
======

The ablation is single-variable. Both configurations share the same split,
feature resolver, hyperparameters, and code, and only the dropped columns
differ.

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Step
     - Function in ``src/``
   * - 70/15/15 stratified split (seed 42)
     - ``protocol.stratified_holdout``
   * - Resolve the model feature set
     - ``features.resolve_feature_columns``
   * - Train XGBoost with the paper's hyperparameters
     - ``gbdt.train_xgboost``, ``config.XGBOOST_PARAMS``
   * - Drop the five post-DFT proxy columns
     - ``leakage.drop_leaky_features``, ``leakage.POST_DFT_COLUMNS``
   * - Score R², MAE, and MSE
     - ``metrics.regression_metrics``

The live cells fit a single XGBoost regressor to establish the direction of each
effect. Figures compared against the authors' reported values come from the
six-model RidgeCV stacking ensemble, run once in the full audit repository and
cached under ``outputs/``.
