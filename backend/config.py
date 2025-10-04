# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Paramètres existants...
APP_NAME = os.getenv("APP_NAME", "Exoplanet Classification API")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
WORKERS = int(os.getenv("WORKERS", 1))

# Nouveaux paramètres pour les modèles d'exoplanètes
KEPLER_MODEL_PATH = BASE_DIR / os.getenv("KEPLER_MODEL_PATH", "models/kepler/model.joblib")
KEPLER_IMPUTER_PATH = BASE_DIR / os.getenv("KEPLER_IMPUTER_PATH", "models/kepler/imputer.joblib")
KEPLER_SCALER_PATH = BASE_DIR / os.getenv("KEPLER_SCALER_PATH", "models/kepler/scaler.joblib")

KEPLER2_MODEL_PATH = BASE_DIR / os.getenv("KEPLER2_MODEL_PATH", "models/kepler2/model.joblib")
KEPLER2_IMPUTER_PATH = BASE_DIR / os.getenv("KEPLER2_IMPUTER_PATH", "models/kepler2/imputer.joblib")
KEPLER2_SCALER_PATH = BASE_DIR / os.getenv("KEPLER2_SCALER_PATH", "models/kepler2/scaler.joblib")

TESS_MODEL_PATH = BASE_DIR / os.getenv("TESS_MODEL_PATH", "models/tess/model.joblib")
TESS_IMPUTER_PATH = BASE_DIR / os.getenv("TESS_IMPUTER_PATH", "models/tess/imputer.joblib")
TESS_SCALER_PATH = BASE_DIR / os.getenv("TESS_SCALER_PATH", "models/tess/scaler.joblib")