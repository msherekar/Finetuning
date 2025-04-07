import mlflow
import logging
from pathlib import Path
import os


# === Directory Setup === #
ROOT_DIR = Path(__file__).parent.parent.resolve()
LOGS_DIR = ROOT_DIR / "logs"
STORAGE_DIR = ROOT_DIR / "finetune_results"

# Ensure directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# === MLflow Configuration === #
MODEL_REGISTRY = STORAGE_DIR / "mlflow"
MLFLOW_TRACKING_URI = f"file://{MODEL_REGISTRY}"
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Optional: Enable experiment name here if needed
DEFAULT_EXPERIMENT_NAME = "Protein_Finetune_Experiments"
mlflow.set_experiment(DEFAULT_EXPERIMENT_NAME)

# === Logger Configuration === #
logger = logging.getLogger("mlflow")
logger.setLevel(logging.DEBUG)

# Optional: Add console handler if script is run directly
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
