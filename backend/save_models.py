"""
Script de sauvegarde rapide si tu as déjà entraîné tes modèles dans un notebook
Utilise ce script après avoir entraîné chaque modèle
"""

import joblib
from pathlib import Path


def save_kepler_model(stacking_clf, imputer, scaler):
    """Sauvegarde le modèle Kepler"""
    Path("models/kepler").mkdir(parents=True, exist_ok=True)
    joblib.dump(stacking_clf, "models/kepler/model.joblib")
    joblib.dump(imputer, "models/kepler/imputer.joblib")
    joblib.dump(scaler, "models/kepler/scaler.joblib")
    print("✅ Modèle Kepler sauvegardé dans models/kepler/")


def save_k2_model(stacking_clf, imputer, scaler):
    """Sauvegarde le modèle K2"""
    Path("models/kepler2").mkdir(parents=True, exist_ok=True)
    joblib.dump(stacking_clf, "models/kepler2/model.joblib")
    joblib.dump(imputer, "models/kepler2/imputer.joblib")
    joblib.dump(scaler, "models/kepler2/scaler.joblib")
    print("✅ Modèle K2 sauvegardé dans models/kepler2/")


def save_toi_model(stacking_clf, imputer, scaler):
    """Sauvegarde le modèle TOI (TESS)"""
    Path("models/tess").mkdir(parents=True, exist_ok=True)
    joblib.dump(stacking_clf, "models/tess/model.joblib")
    joblib.dump(imputer, "models/tess/imputer.joblib")
    joblib.dump(scaler, "models/tess/scaler.joblib")
    print("✅ Modèle TOI sauvegardé dans models/tess/")


# ============================================================================
# UTILISATION DANS TON CODE D'ENTRAÎNEMENT
# ============================================================================

"""
Après avoir entraîné ton modèle Kepler, ajoute ces lignes :

# Import
from save_models import save_kepler_model

# ... ton code d'entraînement Kepler ...
# Après cette ligne: stacking_clf.fit(X_train_sc, y_train)

# Sauvegarder
save_kepler_model(stacking_clf, imputer, scaler)


Pour K2:
from save_models import save_k2_model
# ... entraînement K2 ...
save_k2_model(stacking_clf, imputer, scaler)


Pour TOI:
from save_models import save_toi_model
# ... entraînement TOI ...
save_toi_model(stacking_clf, imputer, scaler)
"""


# ============================================================================
# OU SI TU AS DÉJÀ LES VARIABLES EN MÉMOIRE (dans un notebook par exemple)
# ============================================================================

if __name__ == "__main__":
    print("Ce script doit être importé et les fonctions appelées après l'entraînement")
    print("\nExemple d'utilisation:")
    print("=" * 60)
    print("""
# Dans ton notebook ou script d'entraînement:

# 1. Importer
from save_models import save_kepler_model, save_k2_model, save_toi_model

# 2. Après avoir entraîné Kepler
data_kepler = pretraitement('kepler_data.tsv', 'kepler')
X_train_sc, X_test_sc, y_train, y_test, imputer, scaler = imputationsmote(data_kepler)
stacking_clf = train_stacking_model(X_train_sc, y_train)

# Sauvegarder immédiatement
save_kepler_model(stacking_clf, imputer, scaler)

# 3. Répéter pour K2
data_k2 = pretraitement('k2_data.tsv', 'k2')
X_train_sc, X_test_sc, y_train, y_test, imputer, scaler = imputationsmote(data_k2)
stacking_clf = train_stacking_model(X_train_sc, y_train)
save_k2_model(stacking_clf, imputer, scaler)

# 4. Répéter pour TOI
data_toi = pretraitement('toi_data.tsv', 'toi')
X_train_sc, X_test_sc, y_train, y_test, imputer, scaler = imputationsmote(data_toi)
stacking_clf = train_stacking_model(X_train_sc, y_train)
save_toi_model(stacking_clf, imputer, scaler)
    """)
    print("=" * 60)