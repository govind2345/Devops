import logging
from pathlib import Path

def get_logger(name="mlops"):
    Path("metrics").mkdir(parents=True, exist_ok=True)
    log_file = Path("metrics") / "mlops.log"

    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    return logging.getLogger(name)
