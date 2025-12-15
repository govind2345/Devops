from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config_loader import load_config
from src.data_pipeline import DataConfig, split_features_labels, load_data


def run_inference(
    data_path: str,
    model_path: str,
    output_path: str,
    config_path: str = "config.yaml",
) -> None:
    cfg = load_config(config_path)
    data_cfg = DataConfig(
        id_cols=cfg["columns"]["id"],
        numeric_cols=cfg["columns"]["numeric"],
        label_col=cfg["columns"].get("label", None),
    )

    print(f"[INFER] Loading model from {model_path}")
    clf = joblib.load(model_path)

    print(f"[INFER] Loading data from {data_path}")
    df = load_data(data_path)
    X, y = split_features_labels(df, data_cfg)

    print("[INFER] Running anomaly detection...")
    y_pred_raw = clf.predict(X)
    y_scores = clf.decision_function(X)

    df_out = df.copy()
    df_out["anomaly_score"] = -y_scores
    df_out["is_phishing_pred"] = np.where(y_pred_raw == -1, 1, 0)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"[INFER] Saved predictions to {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phishing Detection on Network Data")
    parser.add_argument("--data", required=True, help="Path to input CSV file")
    parser.add_argument("--model", default="models/model_latest.joblib", help="Model path")
    parser.add_argument("--output", default="metrics/predictions.csv", help="Output file")
    args = parser.parse_args()

    run_inference(args.data, args.model, args.output)
