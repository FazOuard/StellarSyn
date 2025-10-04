import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE


# -------------------- Prétraitement --------------------
def pretraitement(file, destination):
    data = pd.read_csv(file, delimiter=',', engine='python')
    missing_cols = data.columns[data.isna().all()]
    data = data.drop(columns=missing_cols)
    
    if destination == "kepler":
        data = data.drop(columns=['kepler_name','koi_teq_err1','koi_teq_err2','kepid','kepoi_name','koi_pdisposition','koi_score'], errors='ignore')
        data["koi_tce_delivname"].fillna(data["koi_tce_delivname"].mode()[0], inplace=True)
        data = pd.get_dummies(data, columns=["koi_tce_delivname"])
        encoder = LabelEncoder()
        data['label'] = encoder.fit_transform(data['koi_disposition'])
        data = data.drop(columns=['koi_disposition'], errors='ignore')

    elif destination == "toi":
        encoder = LabelEncoder()
        data['label'] = encoder.fit_transform(data['tfopwg_disp'])
        data = data.drop(columns=['tfopwg_disp'], errors='ignore')
        data = data.select_dtypes(include=['number'])

    elif destination == "k2":
        encoder = LabelEncoder()
        data['label'] = encoder.fit_transform(data['disposition'])
        data = data.select_dtypes(include=['number'])

    else:
        raise ValueError(f"Unknown destination: {destination}")
    
    return data


# -------------------- Imputation, SMOTE, Scaling --------------------
def imputation_smote_scaling(data, destination):
    # Imputation
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(data)
    data = pd.DataFrame(imputed_data, columns=data.columns)

    # Sauvegarde de l’imputer
    Path(f"models/{destination}").mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, f"models/{destination}/imputer.joblib")

    # SMOTE
    smote = SMOTE()
    X, y = smote.fit_resample(data.drop('label', axis=1), data['label'])
    y = y.astype('int')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Sauvegarde du scaler
    joblib.dump(scaler, f"models/{destination}/scaler.joblib")

    return X_train_sc, X_test_sc, y_train, y_test


# -------------------- Entraînement du modèle --------------------
def train_and_save_model(file, destination):
    print(f"\n🚀 Entraînement du modèle pour {destination.upper()}")

    data = pretraitement(file, destination)
    X_train_sc, X_test_sc, y_train, y_test = imputation_smote_scaling(data, destination)

    # Définir les modèles de base et le méta-modèle
    base_learners = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
        ('ada', AdaBoostClassifier(n_estimators=50, random_state=42))
    ]
    meta_learner = LogisticRegression(max_iter=1000)
    stacking_clf = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5, n_jobs=-1)

    # Entraînement
    stacking_clf.fit(X_train_sc, y_train)

    # Évaluation
    y_pred = stacking_clf.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy ({destination}): {acc:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Sauvegarde du modèle
    joblib.dump(stacking_clf, f"models/{destination}/model.joblib")
    print(f"💾 Modèle sauvegardé dans models/{destination}/model.joblib")


# -------------------- MAIN --------------------
if __name__ == "__main__":
    # ⚠️ Remplace ces chemins par tes vrais fichiers de données
    train_and_save_model("data/KEPLER.csv", "kepler")
    train_and_save_model("data/K2.csv", "k2")
    train_and_save_model("data/TESS.csv", "toi")
