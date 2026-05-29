# src/features/technical_indicators.py

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import ta

from src.utils.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    HORIZON_PREDICTION,
    TARGET_COLUMN
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def compute_technical_indicators():
    input_path = Path(DATA_RAW_DIR) / "cac40_raw_prices.parquet"
    output_path = Path(DATA_PROCESSED_DIR) / "cac40_technical_features.parquet"
    macro_path = Path(DATA_PROCESSED_DIR) / "macro_data.parquet"
    print(f"DEBUG: Cherche le fichier macro ici : {macro_path.absolute()}") # Ajoute cette ligne
    
    if not input_path.exists():
        raise FileNotFoundError(f"Le fichier brut n'existe pas : {input_path}")
        
    logging.info("Lecture des données brutes Parquet...")
    df = pd.read_parquet(input_path)
    
    # 1. Fusion des données macro (Correction du KeyError)
    if macro_path.exists():
        df_macro = pd.read_parquet(macro_path)
        df['date'] = pd.to_datetime(df['date'])
        df_macro['date'] = pd.to_datetime(df_macro['date'])
        df = df.merge(df_macro, on="date", how="left")
        # Remplissage des valeurs manquantes macro pour éviter les NaN
        df = df.ffill().bfill()
        
        # AJOUT DU RÉGIME DE VOLATILITÉ (Feature Engineering pour aider XGBoost)
        # 1 si le VIX est supérieur à sa moyenne mobile 20 jours
        df["vix_regime"] = (df["vix_close"] > df["vix_close"].rolling(20).mean()).astype(int)
        
        logging.info("Fusion des données macro réussie et calcul du régime de volatilité.")
    else:
        logging.warning("Fichier macro introuvable, attention aux colonnes manquantes !")
    
    if "ticker" not in df.columns:
        df = df.rename(columns={"index": "ticker"}) if "index" in df.columns else df.reset_index()

    # =========================================================
    # EXTRACTION ET PRÉ-CALCUL DU BENCHMARK (CAC 40 global)
    # =========================================================
    logging.info("Calcul des rendements futurs du benchmark (^FCHI) pour la target...")
    df_bench_raw = df[df["ticker"] == "^FCHI"].sort_values("date").copy()
    
    df_bench_raw["mkt_forward_return"] = df_bench_raw["price"].pct_change(periods=-HORIZON_PREDICTION).mul(-1)
    mkt_return_map = dict(zip(df_bench_raw["date"], df_bench_raw["mkt_forward_return"]))

    processed_shares = []
    logging.info("Début du calcul des indicateurs techniques par action...")
    
    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("date").copy()
        
        # Indicateurs techniques
        group["return_1d"] = group["price"].pct_change(1)
        group["return_5d"] = group["price"].pct_change(5)
        
        ma_5_raw = group["price"].rolling(window=5).mean()
        ma_20_raw = group["price"].rolling(window=20).mean()
        ma_50_raw = group["price"].rolling(window=50).mean()
        
        group["ma_5"] = (group["price"] - ma_5_raw) / ma_5_raw
        group["ma_20"] = (group["price"] - ma_20_raw) / ma_20_raw
        group["ma_50"] = (group["price"] - ma_50_raw) / ma_50_raw
        
        log_returns = np.log(group["price"] / group["price"].shift(1))
        group["volatility_20"] = log_returns.rolling(window=20).std()
        
        group["rsi"] = ta.momentum.rsi(close=group["price"], window=14)
        macd_obj = ta.trend.MACD(close=group["price"], window_slow=26, window_fast=12, window_sign=9)
        group["macd"] = macd_obj.macd_diff()
        group["volume_change"] = group["volume"].pct_change(1)
        
        # Target hebdomadaire
        group["forward_return"] = group["price"].pct_change(periods=-HORIZON_PREDICTION).mul(-1)
        group["mkt_forward_return"] = group["date"].map(mkt_return_map)
        group[TARGET_COLUMN] = (group["forward_return"] > group["mkt_forward_return"]).astype(int)
        
        processed_shares.append(group)
        
    df_features = pd.concat(processed_shares, ignore_index=True)
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    
    # =========================================================
    # STANDARDISATION CROSS-SECTIONNELLE
    # =========================================================
    logging.info("Application de la standardisation cross-sectionnelle par date...")
    
    df_stocks = df_features[df_features["ticker"] != "^FCHI"].copy()
    df_bench = df_features[df_features["ticker"] == "^FCHI"].copy()
    
    # On inclut ici les colonnes macro dans la liste de celles qui doivent être traitées
    features_to_scale = ["return_1d", "return_5d", "ma_5", "ma_20", "ma_50", 
                         "volatility_20", "rsi", "macd", "volume_change",
                         "vix_close", "vix_return_1d", "sp500_return_1d", "sp500_return_5d"]
    
    # Filtrer uniquement les colonnes présentes pour éviter erreur si une manque
    features_to_scale = [f for f in features_to_scale if f in df_stocks.columns]
    
    grouped = df_stocks.groupby("date")[features_to_scale]
    means = grouped.transform("mean")
    stds = grouped.transform("std")
    
    df_stocks[features_to_scale] = (df_stocks[features_to_scale] - means) / stds.replace(0, 1)
    
    df_final = pd.concat([df_stocks, df_bench], ignore_index=True)
    df_final = df_final.dropna(subset=[TARGET_COLUMN]).sort_values(by=["date", "ticker"]).reset_index(drop=True)
    
    df_final.to_parquet(output_path, index=False)
    logging.info(f"Calcul et normalisation terminés. Shape finale : {df_final.shape}")
    return df_final

if __name__ == "__main__":
    compute_technical_indicators()