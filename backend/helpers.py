# helpers.py
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from services.model_loader import pretraitement

logger = logging.getLogger(__name__)


def predict(
    data: pd.DataFrame,
    destination: str,
    models: dict
) -> Tuple[int, float]:
    """
    Effectue une prédiction pour un seul échantillon.
    
    Args:
        data: DataFrame avec une seule ligne de données
        destination: 'kepler', 'k2', ou 'toi'
        models: Dictionnaire contenant tous les modèles et leurs composants
    
    Returns:
        Tuple (prediction, probability)
        - prediction: classe prédite (0 ou 1)
        - probability: probabilité de la classe positive
    """
    try:
        # Valider que le modèle existe
        if destination not in models:
            raise ValueError(f"Model for destination '{destination}' not found")
        
        model_components = models[destination]
        model = model_components['model']
        imputer = model_components['imputer']
        scaler = model_components['scaler']
        
        # 1. Prétraitement
        logger.info(f"Preprocessing data for {destination}")
        processed_data = pretraitement(data.copy(), destination)
        
        # 2. Imputation
        logger.info(f"Imputing missing values for {destination}")
        columns = processed_data.columns
        imputed_data = imputer.transform(processed_data)
        processed_data = pd.DataFrame(imputed_data, columns=columns)
        
        # 3. Scaling
        logger.info(f"Scaling features for {destination}")
        scaled_data = scaler.transform(processed_data)
        
        # 4. Prédiction
        logger.info(f"Making prediction with {destination} model")
        prediction = model.predict(scaled_data)[0]
        
        # 5. Obtenir la probabilité si disponible
        probability = None
        try:
            probabilities = model.predict_proba(scaled_data)
            # Prendre la probabilité de la classe positive (classe 1)
            probability = float(probabilities[0, 1])
        except AttributeError:
            logger.warning(f"Model {destination} does not support predict_proba")
            # Si pas de probabilité disponible, utiliser la prédiction comme confiance
            probability = float(prediction)
        
        return int(prediction), probability
        
    except Exception as e:
        logger.error(f"Error in predict: {str(e)}")
        raise


def predict_batch(
    data: pd.DataFrame,
    destination: str,
    models: dict,
    true_labels: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Effectue des prédictions par batch et calcule les métriques si les vraies étiquettes sont fournies.
    
    Args:
        data: DataFrame avec plusieurs lignes de données
        destination: 'kepler', 'k2', ou 'toi'
        models: Dictionnaire contenant tous les modèles et leurs composants
        true_labels: Étiquettes réelles (optionnel) pour calculer les métriques
    
    Returns:
        Dictionnaire contenant:
        - predictions: array des prédictions
        - probabilities: array des probabilités
        - metrics (si true_labels fourni):
            - accuracy: score de précision
            - confusion_matrix: matrice de confusion
            - f1_score: F1 score
            - classification_report: rapport détaillé
    """
    try:
        # Valider que le modèle existe
        if destination not in models:
            raise ValueError(f"Model for destination '{destination}' not found")
        
        model_components = models[destination]
        model = model_components['model']
        imputer = model_components['imputer']
        scaler = model_components['scaler']
        
        # 1. Prétraitement
        logger.info(f"Preprocessing {len(data)} samples for {destination}")
        processed_data = pretraitement(data.copy(), destination)
        
        # 2. Imputation
        logger.info(f"Imputing missing values for {destination}")
        columns = processed_data.columns
        imputed_data = imputer.transform(processed_data)
        processed_data = pd.DataFrame(imputed_data, columns=columns)
        
        # 3. Scaling
        logger.info(f"Scaling features for {destination}")
        scaled_data = scaler.transform(processed_data)
        
        # 4. Prédictions
        logger.info(f"Making batch predictions with {destination} model")
        predictions = model.predict(scaled_data)
        
        # 5. Obtenir les probabilités si disponible
        probabilities = None
        try:
            proba = model.predict_proba(scaled_data)
            # Prendre la probabilité de la classe positive (classe 1)
            probabilities = proba[:, 1]
        except AttributeError:
            logger.warning(f"Model {destination} does not support predict_proba")
        
        # Préparer le résultat
        result = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'total_samples': len(predictions)
        }
        
        # 6. Calculer les métriques si les vraies étiquettes sont fournies
        if true_labels is not None:
            logger.info("Calculating evaluation metrics")
            
            # Accuracy
            accuracy = accuracy_score(true_labels, predictions)
            
            # Confusion Matrix
            conf_matrix = confusion_matrix(true_labels, predictions)
            
            # F1 Score
            f1 = f1_score(true_labels, predictions, average='weighted')
            
            # Classification Report
            class_report = classification_report(
                true_labels, 
                predictions,
                output_dict=True
            )
            
            result['metrics'] = {
                'accuracy': float(accuracy),
                'confusion_matrix': conf_matrix.tolist(),
                'f1_score': float(f1),
                'classification_report': class_report
            }
            
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"F1 Score: {f1:.4f}")
            logger.info(f"Confusion Matrix:\n{conf_matrix}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in predict_batch: {str(e)}")
        raise