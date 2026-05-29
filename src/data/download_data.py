# src/data/download_data.py

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.config import (
    CAC40_TICKERS,
    BENCHMARK_TICKER,
    START_DATE,
    END_DATE,
    DATA_RAW_DIR
)

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================================================
# DOWNLOAD FUNCTION
# =========================================================

def download_cac40_data():

    logging.info(
        f"Début téléchargement : {START_DATE} → {END_DATE}"
    )

    all_tickers = CAC40_TICKERS + [BENCHMARK_TICKER]

    try:

        raw_data = yf.download(
            tickers=all_tickers,
            start=START_DATE,
            end=END_DATE,
            group_by="ticker",
            auto_adjust=False,
            progress=True
        )

    except Exception as e:
        logging.error(f"Erreur téléchargement : {e}")
        raise

    if raw_data.empty:
        raise ValueError("Aucune donnée téléchargée.")

    processed_frames = []

    for ticker in all_tickers:

        try:

            if ticker not in raw_data.columns.levels[0]:
                logging.warning(f"{ticker} absent")
                continue

            ticker_df = raw_data[ticker].copy()

            ticker_df = ticker_df.reset_index()

            ticker_df["ticker"] = ticker

            # Utilisation du prix ajusté
            if "Adj Close" in ticker_df.columns:
                ticker_df["price"] = ticker_df["Adj Close"]

            processed_frames.append(ticker_df)

            logging.info(f"{ticker} téléchargé")

        except Exception as e:
            logging.warning(f"Erreur ticker {ticker} : {e}")

    if not processed_frames:
        raise ValueError("Aucune donnée exploitable.")

    df_final = pd.concat(
        processed_frames,
        ignore_index=True
    )

    # =====================================================
    # CLEANING
    # =====================================================

    df_final.columns = [
        col.lower().replace(" ", "_")
        for col in df_final.columns
    ]

    # Tri de sécurité avant imputation
    df_final = df_final.sort_values(by=["ticker", "date"]).reset_index(drop=True)

    # Imputation propre sans perdre le nom de la colonne 'ticker'
    df_final = (
        df_final
        .groupby("ticker", group_keys=False)
        .apply(lambda x: x.ffill().bfill(), include_groups=True)
        .reset_index(drop=True)
    )
    
    # Sécurité absolue : on s'assure que les colonnes indispensables sont bien là
    if 'ticker' not in df_final.columns and 'index' in df_final.columns:
        # Si pandas a renommé le ticker en 'index' à cause du multi-index résiduel
        df_final = df_final.rename(columns={'index': 'ticker'})

    # =====================================================
    # SAVE
    # =====================================================

    output_path = (
        Path(DATA_RAW_DIR)
        / "cac40_raw_prices.parquet"
    )

    df_final.to_parquet(
        output_path,
        index=False
    )

    logging.info(
        f"Données sauvegardées : {output_path}"
    )

    logging.info(
        f"Shape finale : {df_final.shape}"
    )

    return df_final


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    df = download_cac40_data()

    print(df.head())