from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import logging
import joblib

from config import (
    KEPLER_MODEL_PATH,
    KEPLER_IMPUTER_PATH,
    KEPLER_SCALER_PATH,
    KEPLER2_MODEL_PATH,
    KEPLER2_IMPUTER_PATH,
    KEPLER2_SCALER_PATH,
    TESS_MODEL_PATH,
    TESS_IMPUTER_PATH,
    TESS_SCALER_PATH,
)

logger = logging.getLogger(__name__)

def load_model_and_tokenizer():
    """Load the trained model and tokenizer from disk."""
    try:
        logger.info(f"Loading model from: {KEPLER_MODEL_PATH}")
        model = load_model(KEPLER_MODEL_PATH)

        logger.info(f"Loading tokenizer from: {KEPLER_TOKENIZER_PATH}")
        with open(KEPLER_TOKENIZER_PATH, "r", encoding="utf-8") as f:
            tokenizer = tokenizer_from_json(f.read())

        logger.info("Model and tokenizer loaded successfully")
        return {

        }

    except FileNotFoundError as e:
        logger.error(f"Model or tokenizer file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}")
        raise
