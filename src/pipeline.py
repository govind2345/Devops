# src/pipeline.py

from src.train import main as train_main
from src.predict import run_inference
from src.monitoring import monitor_predictions
from src.retrain import check_for_retrain
from src.logger import get_logger

logger = get_logger()

def main():
    logger.info("=== MLOps Pipeline Started ===")
    print("🔵 Training model...")
    train_main()

    print("🟢 Running inference...")
    run_inference(
        data_path="data/network_test.csv",
        model_path="models/model_latest.joblib",
        output_path="metrics/predictions.csv"
    )

    print("🟡 Running monitoring...")
    monitor_predictions("metrics/predictions.csv")

    print("🟠 Checking for drift + retraining if needed...")
    check_for_retrain()

    logger.info("=== MLOps Pipeline Completed ===")
    print("✅ Pipeline completed successfully.")


if __name__ == "__main__":
    main()
