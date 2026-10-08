"""XGBoost early-warning classifier described in the project proposal.

Each tree is fit to the residual error of the ensemble so far. The objective
is logistic loss with L1/L2 penalties on the leaf weights. The summed tree
outputs pass through a sigmoid, which is the risk probability.
"""

from __future__ import annotations

import xgboost as xgb


def build_classifier(scale_pos_weight: float, n_estimators: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=0.1,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )
