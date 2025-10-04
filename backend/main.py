# main.py
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import shap
import logging
import time
import traceback
import pandas as pd
import seaborn as sns
import numpy as np
import io
from fastapi.responses import Response
import joblib
from typing import List, Optional, Dict, Any
from services.model_loader import pretraitement
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, f1_score
from sklearn.preprocessing import label_binarize

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
                "scaler": joblib.load("models/kepler/scaler.joblib"),
                "X_test_sc": joblib.load("models/kepler/X_test_sc.joblib"),
                "y_test": joblib.load("models/kepler/y_test.joblib")
            },
            "k2": {
                "model": joblib.load("models/k2/model.joblib"),
                "imputer": joblib.load("models/k2/imputer.joblib"),
                "scaler": joblib.load("models/k2/scaler.joblib"),
                "X_test_sc": joblib.load("models/k2/X_test_sc.joblib"),
                "y_test": joblib.load("models/k2/y_test.joblib")
            },
            "toi": {
                "model": joblib.load("models/toi/model.joblib"),
                "imputer": joblib.load("models/toi/imputer.joblib"),
                "scaler": joblib.load("models/toi/scaler.joblib"),
                "X_test_sc": joblib.load("models/toi/X_test_sc.joblib"),
                "y_test": joblib.load("models/toi/y_test.joblib")
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
    

# ------------------ ROC Curve Endpoint ------------------


@app.get("/roc-curve/{destination}")
async def roc_curve_endpoint(destination: str):
    """
    Génère et retourne la courbe ROC pour le modèle choisi (binaire ou multiclasses).
    """
    
    if destination not in models:
        raise HTTPException(status_code=400, detail=f"Destination '{destination}' non trouvée")

    if destination=="kepler":
        class_names = {
            0: "False Positive",
            1: "Confirmed Planet",
            2: "Candidate"
        }
    elif destination=="k2":
        class_names = {
            0: "Confirmed",
            1: "Candidate",
            2: "False Positive",
            3: "Refuted"
        }
    elif destination=="toi":
        class_names = {
            0: "False Positive (FP)",
            1: "Planetary Candidate (PC)",
            2: "Known Planet (KP)",
            3: "Ambiguous (APC)",
            4: "False Alarm (FA)",
            5: "Confirmed Planet (CP)"
        }
    bundle = models[destination]
    model = bundle.get("model")
    X_test = bundle.get("X_test_sc")
    y_test = bundle.get("y_test")
    sns.set_style("darkgrid", {
        'axes.facecolor': '#ECE0E000',         # fond transparent/dégradé
        'figure.facecolor': '#ECE0E000',
        'axes.edgecolor': 'black',             # couleur bordure axes
        'xtick.color': 'white',                # couleur ticks axe x
        'ytick.color': 'white',                # couleur ticks axe y
        'axes.labelcolor': 'white',            # couleur labels axes
        'grid.color': 'gray'                   # couleur grille
    })
    if X_test is None or y_test is None:
        logger.warning(f"Aucune donnée test pour {destination}, ROC non disponible.")
        return {"message": "ROC non disponible pour ce modèle (X_test/y_test manquant)"}

    try:
        y_score = model.predict_proba(X_test)

        # Cas binaire
        if y_score.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(7, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            #plt.title(f'Courbe ROC - {destination}')
            plt.legend(loc="lower right")

        # Cas multiclasses
        else:
            classes = np.unique(y_test)
            y_test_bin = label_binarize(y_test, classes=classes)
            n_classes = y_test_bin.shape[1]

            colors = ['darkorange', 'green', 'red', 'blue', 'purple', 'brown']
            plt.figure(figsize=(8, 6))

            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                label = f"{class_names[classes[i]]} (AUC = {roc_auc:.2f})"
                plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=label)


            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            #plt.title(f'Courbes ROC Multiclasses - {destination}')
            plt.legend(loc="lower right")

        # Sauvegarde et envo
        roc_path = f"roc_{destination}.png"
        plt.savefig(roc_path)
        plt.close()

        with open(roc_path, "rb") as f:
            image_bytes = f.read()
        return Response(content=image_bytes, media_type="image/png", headers={"Access-Control-Allow-Origin": "*"})

    except Exception as e:
        logger.error(f"Erreur génération ROC ({destination}): {e}")
        raise HTTPException(status_code=500, detail=str(e))
    


@app.get("/pr-curve/{destination}")
async def pr_curve_endpoint(destination: str):
    """
    Génère et retourne la courbe Precision-Recall pour le modèle choisi.
    """
    if destination not in models:
        raise HTTPException(status_code=400, detail=f"Destination '{destination}' non trouvée")

    # Définition des noms de classes
    if destination == "kepler":
        class_names = {0: "False Positive", 1: "Confirmed Planet", 2: "Candidate"}
    elif destination == "k2":
        class_names = {0: "Confirmed", 1: "Candidate", 2: "False Positive", 3: "Refuted"}
    elif destination == "toi":
        class_names = {0: "False Positive (FP)", 1: "Planetary Candidate (PC)", 2: "Known Planet (KP)",
                       3: "Ambiguous (APC)", 4: "False Alarm (FA)", 5: "Confirmed Planet (CP)"}

    bundle = models[destination]
    model = bundle.get("model")
    X_test = bundle.get("X_test_sc")
    y_test = bundle.get("y_test")

    sns.set_style("darkgrid", {
        'axes.facecolor': '#ECE0E000',
        'figure.facecolor': '#ECE0E000',
        'axes.edgecolor': 'black',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'axes.labelcolor': 'white',
        'grid.color': 'gray'
    })

    if X_test is None or y_test is None:
        logger.warning(f"Aucune donnée test pour {destination}, PR non disponible.")
        return {"message": "PR non disponible pour ce modèle (X_test/y_test manquant)"}

    try:
        y_score = model.predict_proba(X_test)

        plt.figure(figsize=(8, 6))
        # Cas binaire
        if y_score.shape[1] == 2:
            precision, recall, _ = precision_recall_curve(y_test, y_score[:, 1])
            ap = average_precision_score(y_test, y_score[:, 1])
            plt.plot(recall, precision, color='darkorange', lw=2, label=f'Precision-Recall (AP = {ap:.2f})')
        
        # Cas multiclasses
        else:
            classes = np.unique(y_test)
            y_test_bin = label_binarize(y_test, classes=classes)
            n_classes = y_test_bin.shape[1]
            colors = ['darkorange', 'green', 'red', 'blue', 'purple', 'brown']

            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
                ap = average_precision_score(y_test_bin[:, i], y_score[:, i])
                plt.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                         label=f"{class_names[classes[i]]} (AP = {ap:.2f})")

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="lower left")

        # Sauvegarde et retour
        pr_path = f"pr_{destination}.png"
        plt.savefig(pr_path)
        plt.close()

        with open(pr_path, "rb") as f:
            image_bytes = f.read()
        return Response(content=image_bytes, media_type="image/png", headers={"Access-Control-Allow-Origin": "*"})

    except Exception as e:
        logger.error(f"Erreur génération PR ({destination}): {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/f1-curve/{destination}")
async def f1_curve_endpoint(destination: str):
    if destination not in models:
        raise HTTPException(status_code=400, detail=f"Destination '{destination}' non trouvée")

    bundle = models[destination]
    model = bundle.get("model")
    X_test = bundle.get("X_test_sc")
    y_test = bundle.get("y_test")

    if X_test is None or y_test is None or X_test.shape[0] == 0:
        raise HTTPException(status_code=400, detail="F1 curve non disponible (X_test/y_test manquant ou vide)")

    try:
        # score
        y_score = model.predict_proba(X_test)
        n_classes = y_score.shape[1]

        thresholds = np.linspace(0, 1, 100)
        f1_scores = []

        if n_classes == 2:
            y_score = y_score[:, 1]
            for t in thresholds:
                y_pred = (y_score >= t).astype(int)
                f1 = f1_score(y_test, y_pred, average='binary')
                f1_scores.append(f1)
        else:  # multi-class
            for t in thresholds:
                y_pred = np.argmax(y_score, axis=1)  # ou ajuster selon le seuil si besoin
                f1 = f1_score(y_test, y_pred, average='macro')
                f1_scores.append(f1)

        # création image en mémoire
        plt.figure(figsize=(7, 6))
        plt.plot(thresholds, f1_scores, color='purple', lw=2)
        plt.xlabel("Threshold")
        plt.ylabel("F1 Score")
        #plt.title(f"F1 Score vs Threshold - {destination}")
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        return Response(content=buf.getvalue(), media_type="image/png", headers={"Access-Control-Allow-Origin": "*"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/confidence-distribution/{destination}")
async def confidence_distribution(destination: str):
    """
    Génère la distribution des probabilités (confiance) pour chaque classe du modèle.
    """

    if destination not in models:
        raise HTTPException(status_code=400, detail=f"Destination '{destination}' non trouvée")
    
    if destination == "kepler":
        class_names = {0: "False Positive", 1: "Confirmed Planet", 2: "Candidate"}
    elif destination == "k2":
        class_names = {0: "Confirmed", 1: "Candidate", 2: "False Positive", 3: "Refuted"}
    elif destination == "toi":
        class_names = {0: "False Positive (FP)", 1: "Planetary Candidate (PC)", 2: "Known Planet (KP)",
                       3: "Ambiguous (APC)", 4: "False Alarm (FA)", 5: "Confirmed Planet (CP)"}

    bundle = models[destination]
    model = bundle.get("model")
    X_test = bundle.get("X_test_sc")
    y_test = bundle.get("y_test")
    sns.set_style("darkgrid", {
        'axes.facecolor': '#ECE0E000',
        'figure.facecolor': '#ECE0E000',
        'axes.edgecolor': 'black',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'axes.labelcolor': 'white',
        'grid.color': 'gray'
    })
    if X_test is None or y_test is None:
        raise HTTPException(status_code=400, detail="Données de test manquantes")
    
    try:
        # Probabilités prédites pour chaque classe
        y_probs = model.predict_proba(X_test)  # shape (n_samples, n_classes)
        n_classes = y_probs.shape[1]
        
        plt.figure(figsize=(8,6))
        for class_idx in range(n_classes):
            label=class_names.get(class_idx, f"Class {class_idx}")
            plt.hist(y_probs[:, class_idx], bins=20, alpha=0.5, label=label)
        
        plt.xlabel("Probability")
        plt.ylabel("Number of samples")
        #plt.title(f"Model Confidence Distribution - {destination}")
        plt.legend()
        plt.grid(True)
        
        # Sauvegarde en mémoire et envoi
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png",
                        headers={"Access-Control-Allow-Origin": "*"})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@app.get("/shap-plot/{destination}")
async def shap_plot(destination: str):
    """
    Génère et retourne la SHAP summary plot pour le modèle choisi.
    """
    if destination not in models:
        raise HTTPException(status_code=400, detail=f"Destination '{destination}' non trouvée")

    bundle = models[destination]
    model = bundle.get("model")
    X_test = bundle.get("X_test_sc")
    y_test = bundle.get("y_test")

    if X_test is None or y_test is None:
        return {"message": "SHAP plot non disponible pour ce modèle (X_test/y_test manquant)"}

    try:
        # 1️⃣ Créer un mini-modèle final pour expliquer le final_estimator
        base_preds = np.column_stack([
            est.predict_proba(X_test)[:, 1] for name, est in model.named_estimators_.items()
        ])
        final_model = model.final_estimator_

        # 2️⃣ Explainer pour final estimator
        explainer = shap.Explainer(final_model, base_preds)
        shap_values = explainer(base_preds)

        # 3️⃣ Plot SHAP summary
        plt.figure(figsize=(8,6))
        shap.summary_plot(shap_values, base_preds, show=False)
        plt.title(f"SHAP Summary Plot - {destination}", fontsize=14)

        # 4️⃣ Sauvegarde image temporaire
        shap_path = f"shap_{destination}.png"
        plt.savefig(shap_path, bbox_inches='tight')
        plt.close()

        # 5️⃣ Retourne l'image
        with open(shap_path, "rb") as f:
            image_bytes = f.read()
        return Response(content=image_bytes, media_type="image/png", headers={"Access-Control-Allow-Origin": "*"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))