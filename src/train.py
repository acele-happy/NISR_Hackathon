"""Fit the early-warning model for a placeholder number of boosting rounds.

The default is one round. That is the formative pipeline check, not a finished score.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from src.data import FEATURE_COLUMNS, PROCESSED_PATH, load_frame, save_frame
from src.model import build_classifier

ROOT = Path(__file__).resolve().parents[1]
CATEGORICAL = ["sex", "province", "district", "urban_rural", "consumption_quintile"]


def _design_matrix(train: pd.DataFrame, val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = pd.get_dummies(train, columns=CATEGORICAL, dummy_na=True)
    val_x = pd.get_dummies(val, columns=CATEGORICAL, dummy_na=True)
    val_x = val_x.reindex(columns=train_x.columns, fill_value=0)
    return train_x.astype(float), val_x.astype(float)


def train(n_estimators: int) -> Path:
    frame = load_frame()
    save_frame(frame)
    risk_rate = float(frame["at_risk"].mean())
    print(f"eligible people aged 12-17: {len(frame)}")
    print(f"share flagged at risk: {risk_rate:.3f}")

    train_df, val_df = train_test_split(
        frame,
        test_size=0.2,
        random_state=42,
        stratify=frame["at_risk"],
    )
    y_train = train_df["at_risk"].to_numpy()
    y_val = val_df["at_risk"].to_numpy()
    x_train, x_val = _design_matrix(train_df[FEATURE_COLUMNS], val_df[FEATURE_COLUMNS])

    positives = max(int((y_train == 1).sum()), 1)
    negatives = max(int((y_train == 0).sum()), 1)
    model = build_classifier(negatives / positives, n_estimators)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        verbose=False,
    )

    proba = model.predict_proba(x_val)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_val, proba)),
        "pr_auc": float(average_precision_score(y_val, proba)),
        "f1": float(f1_score(y_val, pred)),
        "n_estimators": n_estimators,
        "validation_people": int(len(y_val)),
    }
    print(
        f"placeholder metrics  ROC-AUC={metrics['roc_auc']:.3f}  "
        f"PR-AUC={metrics['pr_auc']:.3f}  F1={metrics['f1']:.3f}"
    )
    print("One boosting round only shows that the pipeline runs.")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = ROOT / "outputs" / "runs" / f"xgb-{n_estimators}round-{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    history = model.evals_result()
    for step, value in enumerate(history["validation_0"]["logloss"]):
        writer.add_scalar("loss/train", value, step)
    for step, value in enumerate(history["validation_1"]["logloss"]):
        writer.add_scalar("loss/val", value, step)
    writer.add_scalar("auc/val", metrics["roc_auc"], n_estimators)
    writer.add_scalar("pr_auc/val", metrics["pr_auc"], n_estimators)
    writer.add_text(
        "hparams",
        "\n".join(
            [
                f"n_estimators: {n_estimators}",
                "max_depth: 3",
                "learning_rate: 0.1",
                "objective: binary:logistic",
                "reg_alpha: 0.1",
                "reg_lambda: 1.0",
                f"scale_pos_weight: {negatives / positives:.4f}",
            ]
        ),
    )
    writer.close()

    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "placeholder_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    scored = val_df.loc[:, ["hhid", "pid", "at_risk"]].copy()
    scored["risk_score"] = proba
    scored.to_csv(output_dir / "placeholder_scores.csv", index=False)
    model.save_model(output_dir / "placeholder_model.json")

    print(f"feature table: {PROCESSED_PATH}")
    print(f"scores:        {output_dir / 'placeholder_scores.csv'}")
    print(f"tensorboard:   {log_dir}")
    return log_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of boosting rounds. 1 is the formative placeholder.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1.")
    try:
        train(args.epochs)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
