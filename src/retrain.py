# src/retrain.py

import json
from pathlib import Path

from src.train import main as train_main
from src.logger import get_logger

logger = get_logger()

def check_for_retrain(report_path="metrics/monitoring_report.json"):
    report_file = Path(report_path)

    if not report_file.exists():
        print("Monitoring report not found. Run monitoring first.")
        logger.info("Retrain skipped — no monitoring report found.")
        return

    with open(report_file, "r") as f:
        report = json.load(f)

    alert = report.get("alert", False)

    if alert:
        print("⚠ Drift detected → Retraining model...")
        logger.warning("Drift detected — retraining started.")
        train_main()
        print("Retraining completed.")
        logger.info("Retraining completed.")
    else:
        print("No drift detected → Retraining not required.")
        logger.info("Retrain skipped — no drift.")

if __name__ == "__main__":
    check_for_retrain()
