import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Application settings
APP_NAME = os.getenv("APP_NAME", "Text Classification API")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
WORKERS = int(os.getenv("WORKERS", 1))

# Model settings
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", 50))
KEPLER_MODEL_PATH = BASE_DIR / os.getenv("KEPLER_MODEL_PATH", "models/kepler/model.h5")
KEPLER_IMPUTER_PATH = BASE_DIR / os.getenv("KEPLER_IMPUTER_PATH", "models/kepler/imputer.joblib")
KEPLER_SCALER_PATH = BASE_DIR / os.getenv("KEPLER_SCALER_PATH", "models/kepler/scaler.joblib")

KEPLER2_MODEL_PATH = BASE_DIR / os.getenv("KEPLER2_MODEL_PATH", "models/kepler2/model.h5")
KEPLER2_IMPUTER_PATH = BASE_DIR / os.getenv("KEPLER2_IMPUTER_PATH", "models/kepler2/imputer.joblib")
KEPLER2_SCALER_PATH = BASE_DIR / os.getenv("KEPLER2_SCALER_PATH", "models/kepler2/scaler.joblib")

TESS_MODEL_PATH = BASE_DIR / os.getenv("TESS_MODEL_PATH", "models/tess/model.h5")
TESS_IMPUTER_PATH = BASE_DIR / os.getenv("TESS_IMPUTER_PATH", "models/tess/imputer.joblib")
TESS_SCALER_PATH = BASE_DIR / os.getenv("TESS_SCALER_PATH", "models/tess/scaler.joblib")
