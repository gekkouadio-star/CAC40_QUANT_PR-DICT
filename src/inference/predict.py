# src/inference/predict.py

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import xgboost as xgb

# Alignement des chemins vers la racine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    TECHNICAL_FEATURES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_latest_predictions():
    features_path = Path(DATA_PROCESSED_DIR) / "cac40_technical_features.parquet"
    model_path = Path(MODELS_DIR) / "saved_models" / "xgboost_cac40.json"
    
    if not features_path.exists() or not model_path.exists():
        logging.error("Fichiers requis introuvables. Lancez d'abord le pipeline.")
        return None

    # 1. Charger les données
    df = pd.read_parquet(features_path)
    
    # Exclure le benchmark global pour l'analyse par actif
    df_stocks = df[df["ticker"] != "^FCHI"].copy()
    
    # Trouver la date la plus récente disponible dans le dataset
    latest_date = df_stocks["date"].max()
    logging.info(f"Extraction des données pour la dernière séance disponible : {latest_date}")
    
    # Isoler les lignes de cette dernière journée
    df_latest = df_stocks[df_stocks["date"] == latest_date].copy()
    
    if df_latest.empty:
        logging.error("Aucune donnée trouvée pour la date la plus récente.")
        return None
        
    # 2. Charger le modèle optimal
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    
    # 3. Prédire les probabilités de hausse
    logging.info("Calcul des scores de probabilité IA...")
    df_latest["proba_hausse"] = model.predict_proba(df_latest[TECHNICAL_FEATURES])[:, 1]
    
    # Tri des actions par probabilité décroissante
    df_predictions = df_latest[["date", "ticker", "proba_hausse"]].sort_values(
        by="proba_hausse", ascending=False
    ).reset_index(drop=True)
    
    # Sauvegarde des prédictions du jour pour le dashboard
    output_path = Path(DATA_PROCESSED_DIR) / "latest_predictions.csv"
    df_predictions.to_csv(output_path, index=False)
    logging.info(f"🎯 Top 5 Recommandations sauvegardées dans : {output_path}")
    
    return df_predictions

if __name__ == "__main__":
    preds = generate_latest_predictions()
    if preds is not None:
        print("\n" + "="*50)
        print(f"🎯 TOP 5 DES ACTIONS À ACHETER POUR LA PROCHAINE SÉANCE")
        print("="*50)
        print(preds.head(5).to_string(index=False))
        print("="*50 + "\n")