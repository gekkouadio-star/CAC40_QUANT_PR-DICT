# src/data/macro_data.py

import logging
from pathlib import Path
import pandas as pd
import yfinance as yf

from src.utils.config import (
    START_DATE,
    END_DATE,
    DATA_PROCESSED_DIR
)

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def download_and_merge_macro_data():
    # Définition du chemin de sortie macro
    output_macro_path = Path(DATA_PROCESSED_DIR) / "macro_data.parquet"
    
    logging.info("Téléchargement des données macroéconomiques et indices globaux (VIX, S&P500)...")
    
    macro_tickers = {
        "^VIX": "vix_close",       # Indice de la peur
        "^GSPC": "sp500_close"     # Tendance du marché US
    }
    
    try:
        macro_raw = yf.download(
            tickers=list(macro_tickers.keys()),
            start=START_DATE,
            end=END_DATE,
            group_by="ticker",
            auto_adjust=False,
            progress=False
        )
    except Exception as e:
        logging.error(f"Erreur lors du téléchargement des données macro : {e}")
        raise

    macro_frames = []
    
    for ticker, col_name in macro_tickers.items():
        if ticker in macro_raw.columns.levels[0]:
            df_ticker = macro_raw[ticker][["Close"]].copy()
            df_ticker = df_ticker.reset_index()
            df_ticker.columns = ["date", col_name]
            df_ticker["date"] = pd.to_datetime(df_ticker["date"])
            
            # Calcul des features macro stationnaires
            if ticker == "^VIX":
                df_ticker["vix_return_1d"] = df_ticker["vix_close"].pct_change(1)
                
                # AJOUT : Calcul du Z-Score du VIX (1 mois) pour normaliser la volatilité
                # Cela permet au modèle de détecter si le VIX est anormalement élevé
                rolling_mean = df_ticker["vix_close"].rolling(20).mean()
                rolling_std = df_ticker["vix_close"].rolling(20).std()
                df_ticker["vix_zscore"] = (df_ticker["vix_close"] - rolling_mean) / rolling_std
                
            elif ticker == "^GSPC":
                df_ticker["sp500_return_1d"] = df_ticker["sp500_close"].pct_change(1)
                df_ticker["sp500_return_5d"] = df_ticker["sp500_close"].pct_change(5)
                df_ticker = df_ticker.drop(columns=["sp500_close"])
                
            macro_frames.append(df_ticker)

    # Fusion des données macro entre elles
    df_macro = macro_frames[0]
    for df_m in macro_frames[1:]:
        df_macro = pd.merge(df_macro, df_m, on="date", how="outer")
        
    # Tri et traitement des valeurs manquantes
    df_macro = df_macro.sort_values("date").ffill().bfill()
    
    # SAUVEGARDE DU FICHIER MACRO INDÉPENDANT
    df_macro.to_parquet(output_macro_path, index=False)
    
    logging.info(f"Données macro sauvegardées avec succès dans : {output_macro_path}")
    return df_macro

if __name__ == "__main__":
    download_and_merge_macro_data()