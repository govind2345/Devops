# src/train.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.pipeline import Pipeline

from src.config_loader import load_config
from src.data_pipeline import (
    DataConfig,
    load_data,
    build_preprocessor,
    split_features_labels,
)

from src.logger import get_logger
logger = get_logger()


def main(config_path: str = "config.yaml") -> None:
    logger.info("Training started")

    cfg = load_config(config_path)
    logger.info(f"Config loaded from {config_path}")

    train_path = cfg["paths"]["train_data"]
    test_path = cfg["paths"].get("test_data", None)
    model_dir = Path(cfg["paths"]["model_dir"])
    metrics_dir = Path(cfg["paths"]["metrics_dir"])

    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = DataConfig(
        id_cols=cfg["columns"]["id"],
        numeric_cols=cfg["columns"]["numeric"],
        label_col=cfg["columns"].get("label", None),
    )

    logger.info(f"Loading training data from {train_path}")
    train_df = load_data(train_path)
    X_train, y_train = split_features_labels(train_df, data_cfg)

    if y_train is not None:
        normal_mask = y_train == 0
        logger.info(f"Using {normal_mask.sum()} normal samples out of {len(y_train)}")
        X_train = X_train[normal_mask]
    else:
        logger.info("No labels found — training on all samples")

    preprocessor = build_preprocessor(data_cfg)

    contamination = cfg["training"]["contamination"]
    random_state = cfg["training"]["random_state"]
    n_estimators = cfg["training"]["n_estimators"]

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    logger.info("Fitting model...")
    clf.fit(X_train)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Versioning folder
    version_dir = model_dir / "versions"
    version_dir.mkdir(parents=True, exist_ok=True)

    model_path = version_dir / f"model_v{timestamp}.joblib"
    latest_path = model_dir / "model_latest.joblib"

    joblib.dump(clf, model_path)
    joblib.dump(clf, latest_path)

    logger.info(f"Saved versioned model: {model_path}")
    logger.info(f"Updated latest model: {latest_path}")

    metrics = {
        "timestamp_utc": timestamp,
        "training_samples": int(len(X_train)),
        "model_path": str(model_path),
        "contamination": float(contamination),
    }

    # Evaluation (optional)
    if test_path is not None and Path(test_path).exists():
        logger.info(f"Running evaluation on test data: {test_path}")
        test_df = load_data(test_path)
        X_test, y_test = split_features_labels(test_df, data_cfg)

        if y_test is not None:
            X_test_trans = clf.named_steps["preprocess"].transform(X_test)
            y_raw = clf.named_steps["model"].predict(X_test_trans)
            y_scores = clf.named_steps["model"].decision_function(X_test_trans)

            y_pred = np.where(y_raw == -1, 1, 0)

            auc = roc_auc_score(y_test, -y_scores)
            metrics["roc_auc"] = float(auc)
            metrics["classification_report"] = classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            )

            logger.info(f"ROC-AUC: {auc:.4f}")
        else:
            logger.info("Test set has no labels — skipping evaluation")
    else:
        logger.info("No test data available — skipping evaluation")

    metrics_path = metrics_dir / f"training_metrics_{timestamp}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Training metrics saved to {metrics_path}")
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
