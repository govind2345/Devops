import json
from pathlib import Path
import pandas as pd
from src.config_loader import load_config


def compute_stats(scores):
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
    }


def monitor_predictions(predictions_path: str, config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    pred_path = Path(predictions_path)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path.resolve()}")

    df = pd.read_csv(pred_path)

    if "anomaly_score" not in df.columns:
        raise ValueError("Prediction file missing 'anomaly_score' column")

    current_stats = compute_stats(df["anomaly_score"])

    # Baseline file path
    baseline_path = metrics_dir / "training_stats.json"

    # If baseline doesn't exist → Save baseline
    if not baseline_path.exists():
        with open(baseline_path, "w") as f:
            json.dump(current_stats, f, indent=2)
        
        print("[MONITOR] Baseline created. Run monitoring again to compare drift.")
        return

    # Load baseline for comparison
    with open(baseline_path, "r") as f:
        baseline_stats = json.load(f)

    # Compute drift
    drift_threshold = cfg["monitoring"]["drift_alert_threshold"]
    mean_shift = abs(current_stats["mean"] - baseline_stats["mean"])
    relative_shift = mean_shift / (abs(baseline_stats["mean"]) + 1e-6)

    alert = relative_shift > drift_threshold

    result = {
        "baseline_stats": baseline_stats,
        "current_stats": current_stats,
        "relative_shift": relative_shift,
        "threshold": drift_threshold,
        "alert": alert
    }

    # Save monitoring results
    monitor_path = metrics_dir / "monitoring_report.json"
    with open(monitor_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[MONITOR] Monitoring report saved: {monitor_path}")
    if alert:
        print("🚨 ALERT: Possible data drift detected!")
    else:
        print("✅ No significant drift detected.")


if __name__ == "__main__":
    monitor_predictions("metrics/predictions.csv")
