# src/evaluation/backtest.py

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb

# Alignement des chemins
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    TECHNICAL_FEATURES,
    TARGET_COLUMN
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_backtest():
    features_path = Path(DATA_PROCESSED_DIR) / "cac40_technical_features.parquet"
    model_path = Path(MODELS_DIR) / "saved_models" / "xgboost_cac40.json"
    
    if not features_path.exists() or not model_path.exists():
        logging.error("Modèle ou features introuvables. Vérifiez vos fichiers.")
        return

    logging.info("Chargement des données et du modèle pour le Backtest...")
    df = pd.read_parquet(features_path)
    
    # Isoler le jeu de test (données hors-échantillon 2025-2026)
    df_test = df[(df["date"] >= "2025-01-01") & (df["ticker"] != "^FCHI")].copy()
    df_test = df_test.sort_values(["date", "ticker"]).reset_index(drop=True)
    
    # Recharger le modèle XGBoost
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    
    # Générer les probabilités de hausse
    logging.info("Générations des prédictions quantitatives...")
    df_test["proba_up"] = model.predict_proba(df_test[TECHNICAL_FEATURES])[:, 1]
    
    # =========================================================
    # SIMULATION DE LA STRATÉGIE (Top 5 Long)
    # =========================================================
    logging.info("Simulation de la stratégie Long-Only (Top 5 actions)...")
    
    # Calcul du rendement réel à 1 jour pour chaque ligne
    # Note : return_1d actuel est une feature passée. On calcule le rendement FUTUR (le gain du jour suivant)
    df_test["next_day_return"] = df_test.groupby("ticker")["price"].pct_change(1).shift(-1)
    df_test = df_test.dropna(subset=["next_day_return"])
    
    strategy_returns = []
    market_returns = []
    dates = []
    
    # Boucle jour par jour
    for date, group in df_test.groupby("date"):
        if len(group) < 10: # Ignorer les jours avec trop peu d'actifs valides
            continue
            
        # 1. Performance du marché de référence ce jour-là (Moyenne équipondérée du CAC40)
        mean_market_return = group["next_day_return"].mean()
        
        # 2. Performance de notre stratégie : Sélection des 5 meilleures probabilités
        top_5 = group.nlargest(5, "proba_up")
        mean_strat_return = top_5["next_day_return"].mean()
        
        dates.append(date)
        market_returns.append(mean_market_return)
        strategy_returns.append(mean_strat_return)
        
    # Création du DataFrame de performance
    df_perf = pd.DataFrame({
        "date": dates,
        "market_return": market_returns,
        "strategy_return": strategy_returns
    })
    
    # Calcul des rendements cumulés
    df_perf["cum_market"] = (1 + df_perf["market_return"]).cumprod() - 1
    df_perf["cum_strategy"] = (1 + df_perf["strategy_return"]).cumprod() - 1
    
    # Métriques de performance
    sharpe_strat = (df_perf["strategy_return"].mean() / df_perf["strategy_return"].std()) * np.sqrt(252) if df_perf["strategy_return"].std() != 0 else 0
    sharpe_market = (df_perf["market_return"].mean() / df_perf["market_return"].std()) * np.sqrt(252) if df_perf["market_return"].std() != 0 else 0
    
    print("\n" + "="*50)
    print("🏁 RÉSULTATS DU BACKTEST (PÉRIODE HORIZON 2025 - 2026)")
    print("="*50)
    print(f"Rendement Cumulé Marché   : {df_perf['cum_market'].iloc[-1]*100:.2f} %")
    print(f"Rendement Cumulé Stratégie : {df_perf['cum_strategy'].iloc[-1]*100:.2f} %")
    print("-"*50)
    print(f"Ratio de Sharpe Marché    : {sharpe_market:.2f}")
    print(f"Ratio de Sharpe Stratégie : {sharpe_strat:.2f}")
    print("="*50 + "\n")
    
    # Sauvegarde des résultats
    output_perf_path = Path(DATA_PROCESSED_DIR) / "backtest_performance.csv"
    df_perf.to_csv(output_perf_path, index=False)
    logging.info(f"Résultats sauvegardés dans : {output_perf_path}")

if __name__ == "__main__":
    run_backtest()