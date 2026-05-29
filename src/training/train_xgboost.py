# src/training/train_xgboost.py

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# Aligner le path pour importer les configurations
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    TECHNICAL_FEATURES,
    TARGET_COLUMN
)

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def train_xgboost_model(n_trees=40, use_early_stopping=False):
    """
    Entraîne le modèle XGBoost sur les features du CAC40.
    """
    input_path = Path(DATA_PROCESSED_DIR) / "cac40_technical_features.parquet"
    model_output_dir = Path(MODELS_DIR) / "saved_models"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        logging.error(f"Fichier de features introuvable : {input_path}")
        raise FileNotFoundError(f"Lancez d'abord technical_indicators.py et macro_data.py")
        
    logging.info("Chargement du dataset enrichi (Technique + Macro)...")
    df = pd.read_parquet(input_path)
    
    # Élimination du benchmark global
    df = df[df["ticker"] != "^FCHI"].copy()
    
    # Filtrer les lignes récentes
    df_ml = df.dropna(subset=[TARGET_COLUMN]).copy()
    df_ml = df_ml.sort_values("date").reset_index(drop=True)
    
    # Split temporel strict (2015-2024 vs 2025-2026)
    split_date = "2025-01-01"
    train_mask = df_ml["date"] < split_date
    test_mask = df_ml["date"] >= split_date
    
    X_train = df_ml.loc[train_mask, TECHNICAL_FEATURES]
    y_train = df_ml.loc[train_mask, TARGET_COLUMN].astype(int)
    
    X_test = df_ml.loc[test_mask, TECHNICAL_FEATURES]
    y_test = df_ml.loc[test_mask, TARGET_COLUMN].astype(int)
    
    logging.info(f"Train set : {X_train.shape[0]} lignes | Test set : {X_test.shape[0]} lignes")
    
    # =========================================================
    # HYPERPARAMÈTRES RÉAJUSTÉS POUR CORRIGER LE RECALL
    # =========================================================
    # Calcul automatique du poids des classes pour corriger le biais (Recall 0)
    ratio_pos_neg = (len(y_train) - y_train.sum()) / y_train.sum()
    
    robust_params = {
        "n_estimators": n_trees,           
        "max_depth": 6,               # Augmenté pour capturer les interactions macro
        "learning_rate": 0.05,        
        "subsample": 0.8,             
        "colsample_bytree": 0.8,      # Augmenté pour donner plus de chance aux colonnes macro
        "scale_pos_weight": ratio_pos_neg, # Ajustement dynamique pour équilibrer la précision/rappel
        "eval_metric": "auc",         
        "random_state": 42,
        "n_jobs": -1
    }
    
    if use_early_stopping:
        robust_params["early_stopping_rounds"] = 40
    
    # 1. Initialisation de l'XGBClassifier
    logging.info("Initialisation de l'XGBClassifier (Configuration optimisée)...")
    model = xgb.XGBClassifier(**robust_params)

    # 2. Entraînement du modèle
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=10
    )

    # 3. Récupération de la meilleure itération si early stopping actif
    if use_early_stopping and hasattr(model, "best_iteration"):
        actual_trees = model.best_iteration
        tree_log = f"stoppé à l'arbre optimal : {actual_trees}"
    else:
        actual_trees = n_trees
        tree_log = f"entraîné complètement sur {n_trees} arbres"
        
    logging.info(f"✔ Modèle {tree_log}.")

    # =========================================================
    # ÉVALUATION
    # =========================================================
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)
    
    print("\n" + "="*50)
    print(f"📊 RÉSULTATS DU MODÈLE XGBOOST OPTIMISÉ")
    print("="*50)
    print(f"Arbres effectifs : {actual_trees}")
    print(f"Accuracy Score   : {accuracy:.4f}")
    print(f"ROC AUC Score    : {auc_score:.4f}")
    print("-"*50)
    print("Rapport de Classification complet :")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("="*50 + "\n")
    
    logging.info("Importance des Features :")
    importance = pd.Series(model.feature_importances_, index=TECHNICAL_FEATURES).sort_values(ascending=False)
    print(importance)
    
    # Sauvegarde du modèle
    model_path = model_output_dir / "xgboost_cac40.json"
    model.save_model(str(model_path))
    logging.info(f"✔ Modèle sauvegardé avec succès : {model_path}")
    
    return model

if __name__ == "__main__":
    # Force 100 arbres, sans early stopping pour laisser le modèle converger
    train_xgboost_model(n_trees=100, use_early_stopping=False)