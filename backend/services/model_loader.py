# services/model_loader.py
import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

try:
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
except ImportError:
    # Valeurs par défaut si config n'est pas disponible
    BASE_DIR = Path(__file__).resolve().parent.parent
    KEPLER_MODEL_PATH = BASE_DIR / "models/kepler/model.joblib"
    KEPLER_IMPUTER_PATH = BASE_DIR / "models/kepler/imputer.joblib"
    KEPLER_SCALER_PATH = BASE_DIR / "models/kepler/scaler.joblib"
    KEPLER2_MODEL_PATH = BASE_DIR / "models/kepler2/model.joblib"
    KEPLER2_IMPUTER_PATH = BASE_DIR / "models/kepler2/imputer.joblib"
    KEPLER2_SCALER_PATH = BASE_DIR / "models/kepler2/scaler.joblib"
    TESS_MODEL_PATH = BASE_DIR / "models/tess/model.joblib"
    TESS_IMPUTER_PATH = BASE_DIR / "models/tess/imputer.joblib"
    TESS_SCALER_PATH = BASE_DIR / "models/tess/scaler.joblib"

logger = logging.getLogger(__name__)


def load_model_and_tokenizer():
    """
    Charge tous les modèles d'exoplanètes avec leurs imputers et scalers.
    Retourne un dictionnaire structuré avec tous les composants.
    """
    try:
        models_dict = {}
        
        # Charger Kepler
        logger.info(f"Loading Kepler model from: {KEPLER_MODEL_PATH}")
        models_dict['kepler'] = {
            'model': joblib.load(KEPLER_MODEL_PATH),
            'imputer': joblib.load(KEPLER_IMPUTER_PATH),
            'scaler': joblib.load(KEPLER_SCALER_PATH)
        }
        logger.info("Kepler model loaded successfully")
        
        # Charger Kepler2 (K2)
        logger.info(f"Loading Kepler2 model from: {KEPLER2_MODEL_PATH}")
        models_dict['k2'] = {
            'model': joblib.load(KEPLER2_MODEL_PATH),
            'imputer': joblib.load(KEPLER2_IMPUTER_PATH),
            'scaler': joblib.load(KEPLER2_SCALER_PATH)
        }
        logger.info("Kepler2 model loaded successfully")
        
        # Charger TESS (TOI)
        logger.info(f"Loading TESS model from: {TESS_MODEL_PATH}")
        models_dict['toi'] = {
            'model': joblib.load(TESS_MODEL_PATH),
            'imputer': joblib.load(TESS_IMPUTER_PATH),
            'scaler': joblib.load(TESS_SCALER_PATH)
        }
        logger.info("TESS model loaded successfully")
        
        logger.info("All models, imputers, and scalers loaded successfully")
        
        # Pour compatibilité avec l'architecture existante, on retourne model et tokenizer
        # mais ici "model" est notre dict et "tokenizer" est None
        return models_dict, None
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


def pretraitement(data: pd.DataFrame, destination: str) -> pd.DataFrame:
    """
    Prétraite les données selon la destination.
    
    Args:
        data: DataFrame avec les données brutes
        destination: 'kepler', 'toi', ou 'k2'
    
    Returns:
        DataFrame prétraité (AVEC la colonne label si elle existe)
    """
    # Copier pour éviter les modifications inattendues
    data = data.copy()
    
    # Supprimer les colonnes entièrement vides
    missing_cols = data.columns[data.isna().all()]
    data = data.drop(columns=missing_cols)
    
    if destination == "kepler":
        # Colonnes à supprimer pour Kepler
        cols_to_drop = ['kepler_name', 'koi_teq_err1', 'koi_teq_err2', 
                        'kepid', 'kepoi_name', 'koi_pdisposition', 'koi_score']
        # Ne supprimer que les colonnes qui existent
        cols_to_drop = [col for col in cols_to_drop if col in data.columns]
        data = data.drop(columns=cols_to_drop)
        
        # 🔹 CORRECTION : Remplir et encoder koi_tce_delivname sans inplace
        if 'koi_tce_delivname' in data.columns:
            mode_value = data["koi_tce_delivname"].mode()
            if len(mode_value) > 0:
                data["koi_tce_delivname"] = data["koi_tce_delivname"].fillna(mode_value[0])
            data = pd.get_dummies(data, columns=["koi_tce_delivname"])
        
        # 🔹 CORRECTION : Si koi_disposition existe et qu'on n'a pas encore de label, créer label
        if 'koi_disposition' in data.columns and 'label' not in data.columns:
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            data['label'] = encoder.fit_transform(data['koi_disposition'])
        
        # Supprimer koi_disposition (mais GARDER label)
        if 'koi_disposition' in data.columns:
            data = data.drop(columns=['koi_disposition'])
            
    elif destination == "toi":
        # 🔹 Si tfopwg_disp existe et qu'on n'a pas encore de label, créer label
        if 'tfopwg_disp' in data.columns and 'label' not in data.columns:
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            data['label'] = encoder.fit_transform(data['tfopwg_disp'])
        
        # Supprimer tfopwg_disp (mais GARDER label)
        if 'tfopwg_disp' in data.columns:
            data = data.drop(columns=['tfopwg_disp'])
        
        # Garder uniquement les colonnes numériques (y compris label si présente)
        data = data.select_dtypes(include=['number'])
        
    elif destination == "k2":
        # 🔹 Si disposition existe et qu'on n'a pas encore de label, créer label
        if 'disposition' in data.columns and 'label' not in data.columns:
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            data['label'] = encoder.fit_transform(data['disposition'])
        
        # Supprimer disposition (mais GARDER label)
        if 'disposition' in data.columns:
            data = data.drop(columns=['disposition'])
        
        # Garder uniquement les colonnes numériques (y compris label si présente)
        data = data.select_dtypes(include=['number'])
        
    else:
        raise ValueError(f"Unknown destination: '{destination}'. Must be 'kepler', 'toi', or 'k2'")
    
    return data
