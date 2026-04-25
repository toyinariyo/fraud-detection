"""Project paths for data and model artifacts."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DB_PATH = DATA_DIR / "fraud_detection.db"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.pkl"
