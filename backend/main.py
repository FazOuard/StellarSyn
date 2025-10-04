# main.py
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging
import time
import traceback
import pandas as pd
import numpy as np
import io
import joblib
from typing import List, Optional, Dict, Any
from services.model_loader import pretraitement

from config import APP_NAME, APP_VERSION, DEBUG, LOG_LEVEL

# ------------------ Logging ------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------ FastAPI app ------------------
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="API for exoplanet candidate classification using pre-trained .joblib models",
    debug=DEBUG
)

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Global models ------------------
models = {}

# ------------------ Pydantic Models ------------------
class PredictionRequest(BaseModel):
    destination: str = Field(..., description="Destination: 'kepler', 'k2', or 'toi'")
    features: List[float] = Field(..., description="List of feature values")

class BatchPredictionRequest(BaseModel):
    destination: str
    data: List[List[float]]
    true_labels: Optional[List[int]] = None

class PredictionResponse(BaseModel):
    destination: str
    prediction: int
    probability: Optional[float] = None

class BatchPredictionItem(BaseModel):
    prediction: int
    probability: Optional[float]

class MetricsResponse(BaseModel):
    accuracy: float
    confusion_matrix: List[List[int]]
    f1_score: float
    classification_report: Dict[str, Any]

class BatchPredictionResponse(BaseModel):
    destination: str
    predictions: List[BatchPredictionItem]
    total_samples: int
    metrics: Optional[MetricsResponse] = None

class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    timestamp: str
    version: str

class ModelInfoResponse(BaseModel):
    destination: str
    model_type: str
    is_loaded: bool
    additional_info: Optional[Dict[str, Any]] = None

# ------------------ Helper Functions ------------------
def detect_destination(df: pd.DataFrame) -> str:
    """
    Détecte automatiquement la destination en fonction des colonnes du DataFrame.
    """
    columns = set(df.columns)
    
    kepler_indicators = {'koi_disposition', 'kepoi_name', 'koi_score', 'koi_tce_delivname', 'koi_depth'}
    k2_indicators = {'disposition', 'epic_candname', 'k2_campaign_str', 'epic_id'}
    toi_indicators = {'tfopwg_disp', 'toi', 'tid', 'toipfx', 'st_tmag'}
    
    kepler_match = len(columns.intersection(kepler_indicators))
    k2_match = len(columns.intersection(k2_indicators))
    toi_match = len(columns.intersection(toi_indicators))
    
    logger.info(f"🔍 Detection scores - Kepler: {kepler_match}, K2: {k2_match}, TOI: {toi_match}")
    logger.info(f"📋 Sample columns: {list(columns)[:5]}")
    
    if kepler_match >= k2_match and kepler_match >= toi_match and kepler_match > 0:
        return 'kepler'
    elif toi_match > k2_match and toi_match > 0:
        return 'toi'
    elif k2_match > 0:
        return 'k2'
    else:
        # Fallback basé sur les noms de colonnes courantes
        if any('koi_' in col.lower() for col in columns):
            return 'kepler'
        elif any('toi' in col.lower() or 'tid' in col.lower() for col in columns):
            return 'toi'
        elif any('epic' in col.lower() for col in columns):
            return 'k2'
        else:
            logger.warning("⚠️ Could not detect destination reliably, defaulting to kepler")
            return 'kepler'

# ------------------ Load models on startup ------------------
@app.on_event("startup")
async def startup_event():
    global models
    try:
        logger.info("🔄 Loading trained models from 'models/' directory...")

        models = {
            "kepler": {
                "model": joblib.load("models/kepler/model.joblib"),
                "imputer": joblib.load("models/kepler/imputer.joblib"),
                "scaler": joblib.load("models/kepler/scaler.joblib")
            },
            "k2": {
                "model": joblib.load("models/k2/model.joblib"),
                "imputer": joblib.load("models/k2/imputer.joblib"),
                "scaler": joblib.load("models/k2/scaler.joblib")
            },
            "toi": {
                "model": joblib.load("models/toi/model.joblib"),
                "imputer": joblib.load("models/toi/imputer.joblib"),
                "scaler": joblib.load("models/toi/scaler.joblib")
            }
        }

        logger.info("✅ All models loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# ------------------ Middleware ------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} ({duration:.3f}s)")
    return response

# ------------------ Exception Handler ------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc) if DEBUG else "Internal server error"}
    )

# ------------------ Routes ------------------
@app.get("/")
async def root():
    return {
        "message": f"Welcome to {APP_NAME}",
        "version": APP_VERSION,
        "models": list(models.keys()) if models else [],
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    models_status = {dst: dst in models for dst in ['kepler', 'k2', 'toi']} if models else {}
    return HealthResponse(
        status="ok" if models else "error",
        models_loaded=models_status,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        version=APP_VERSION
    )

@app.get("/model-info/{destination}", response_model=ModelInfoResponse)
async def model_info(destination: str):
    if destination not in models:
        raise HTTPException(status_code=404, detail=f"Model '{destination}' not found")
    model = models[destination]['model']
    info = {"model_type": type(model).__name__}
    if hasattr(model, "n_estimators"): info["n_estimators"] = model.n_estimators
    if hasattr(model, "n_features_in_"): info["n_features"] = model.n_features_in_
    return ModelInfoResponse(
        destination=destination,
        model_type=type(model).__name__,
        is_loaded=True,
        additional_info=info
    )

# ------------------ Prediction ------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest):
    destination = request.destination.lower()
    if destination not in models:
        raise HTTPException(status_code=400, detail=f"Invalid destination: {destination}")
    try:
        bundle = models[destination]
        X = np.array(request.features).reshape(1, -1)
        X = bundle["imputer"].transform(X)
        X = bundle["scaler"].transform(X)
        y_pred = bundle["model"].predict(X)
        if hasattr(bundle["model"], "predict_proba"):
            prob = bundle["model"].predict_proba(X).max()
        else:
            prob = None
        return PredictionResponse(destination=destination, prediction=int(y_pred[0]), probability=prob)
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ------------------ Batch Prediction ------------------
@app.post("/predict-file", response_model=BatchPredictionResponse)
async def predict_from_file(
    file: UploadFile = File(...), 
    has_labels: bool = False,
    destination: Optional[str] = None
):
    try:
        # Lire le CSV avec détection automatique du délimiteur
        contents = await file.read()
        content_str = contents.decode('utf-8')
        
        # Détecter le délimiteur (virgule ou tabulation)
        first_line = content_str.split('\n')[0]
        delimiter = '\t' if '\t' in first_line else ','
        
        delimiter_name = 'TAB' if delimiter == '\t' else 'COMMA'
        logger.info(f"📄 Detected delimiter: {delimiter_name}")
        
        df = pd.read_csv(io.StringIO(content_str), delimiter=delimiter)
        logger.info(f"📊 Loaded file with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"📋 Columns: {list(df.columns)[:10]}...")  # Log first 10 columns

        # Détection automatique si destination non fournie
        if destination is None:
            destination = detect_destination(df)
            logger.info(f"🎯 Auto-detected destination: {destination.upper()}")
        else:
            destination = destination.lower()
            logger.info(f"📌 Using provided destination: {destination.upper()}")
        
        if destination not in models:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid or unsupported destination: {destination}"
            )

        # Sauvegarder les vrais labels si présents
        true_labels = None
        if has_labels:
            label_columns = ['label', 'koi_disposition', 'tfopwg_disp', 'disposition']
            label_col = next((col for col in label_columns if col in df.columns), None)
            if label_col:
                from sklearn.preprocessing import LabelEncoder
                true_labels = LabelEncoder().fit_transform(df[label_col])
                logger.info(f"✅ Found and extracted {len(true_labels)} labels from column '{label_col}'")

        # Prétraitement
        df_processed = pretraitement(df, destination)
        logger.info(f"📋 After preprocessing, columns: {list(df_processed.columns)}")

        features_df = df_processed.copy()
        
        # Récupérer le bundle du modèle
        bundle = models[destination]

        # Imputation avec toutes les colonnes (y compris label si présente)
        logger.info(f"🔧 Imputing {len(features_df)} samples...")
        X_imp = bundle["imputer"].transform(features_df)
        
        X_imp_df = pd.DataFrame(X_imp, columns=features_df.columns)
        
        # Retirer label avant scaling/prédiction
        if 'label' in X_imp_df.columns:
            if true_labels is None and has_labels:
                true_labels = X_imp_df['label'].values
            X_imp_df = X_imp_df.drop(columns=['label'])
        
        logger.info(f"🔧 Scaling {len(X_imp_df)} samples...")
        X_scaled = bundle["scaler"].transform(X_imp_df)

        # Prédiction
        logger.info(f"🚀 Predicting {len(X_scaled)} samples...")
        y_pred = bundle["model"].predict(X_scaled)

        # Probabilités si disponibles
        probs = None
        if hasattr(bundle["model"], "predict_proba"):
            probs_array = bundle["model"].predict_proba(X_scaled)
            # Prendre la probabilité maximale pour chaque prédiction
            probs = probs_array.max(axis=1)

        # Préparer les résultats
        predictions = [
            BatchPredictionItem(
                prediction=int(y_pred[i]), 
                probability=float(probs[i]) if probs is not None else None
            )
            for i in range(len(y_pred))
        ]

        # Calculer les métriques si on a les labels
        metrics = None
        if true_labels is not None:
            from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
            
            accuracy = accuracy_score(true_labels, y_pred)
            conf_matrix = confusion_matrix(true_labels, y_pred)
            f1 = f1_score(true_labels, y_pred, average='weighted')
            class_report = classification_report(true_labels, y_pred, output_dict=True)
            
            metrics = MetricsResponse(
                accuracy=float(accuracy),
                confusion_matrix=conf_matrix.tolist(),
                f1_score=float(f1),
                classification_report=class_report
            )
            
            logger.info(f"📊 Metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        return BatchPredictionResponse(
            destination=destination,
            predictions=predictions,
            total_samples=len(y_pred),
            metrics=metrics
        )

    except Exception as e:
        logger.error(f"❌ Batch prediction failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")