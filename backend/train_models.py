import requests
import pandas as pd
from typing import Dict, List

API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Vérifie si l'API est en ligne"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        result = response.json()
        print("🟢 API Status:", result['status'])
        print("📦 Modèles chargés:")
        for model, loaded in result['models_loaded'].items():
            status = "✅" if loaded else "❌"
            print(f"   {status} {model}")
        return True
    except Exception as e:
        print(f"🔴 API non disponible: {e}")
        return False

def predict_single(destination: str, data: Dict):
    """Prédiction pour un seul échantillon"""
    print(f"\n{'='*60}")
    print(f"🔮 Prédiction unique - {destination.upper()}")
    print(f"{'='*60}")

    response = requests.post(
        f"{API_BASE_URL}/predict",
        json={
            "destination": destination,
            "data": data
        }
    )

    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction']}")
        print(f"Probability: {result.get('probability', 'N/A')}")
    else:
        print(f"Erreur {response.status_code}: {response.text}")

def predict_batch(destination: str, batch_data: List[Dict], true_labels: List[int] = None):
    """Prédiction par lot"""
    print(f"\n{'='*60}")
    print(f"🔮 Prédiction batch - {destination.upper()}")
    print(f"{'='*60}")

    payload = {
        "destination": destination,
        "data": batch_data
    }
    if true_labels is not None:
        payload["true_labels"] = true_labels

    response = requests.post(f"{API_BASE_URL}/predict-batch", json=payload)

    if response.status_code == 200:
        result = response.json()
        print(f"Total samples: {result['total_samples']}")
        for i, pred in enumerate(result['predictions']):
            print(f"Sample {i}: Prediction={pred['prediction']}, Probability={pred.get('probability', 'N/A')}")
        if 'metrics' in result and result['metrics'] is not None:
            print("Metrics:")
            print(f" - Accuracy: {result['metrics']['accuracy']}")
            print(f" - F1 Score: {result['metrics']['f1_score']}")
    else:
        print(f"Erreur {response.status_code}: {response.text}")

if __name__ == "__main__":
    # Exemple d'utilisation
    if check_api_health():
        # Exemple prédiction unique
        sample_data = {
            "feature1": 0.5,
            "feature2": 1.2,
            "feature3": 3.4,
            # Ajoutez toutes les colonnes nécessaires
        }
        predict_single("kepler", sample_data)

        # Exemple prédiction batch
        batch_samples = [
            {"feature1": 0.5, "feature2": 1.2, "feature3": 3.4},
            {"feature1": 0.6, "feature2": 1.1, "feature3": 3.5}
        ]
        true_labels_batch = [1, 0]  # Optionnel, pour évaluation
        predict_batch("kepler", batch_samples, true_labels_batch)

