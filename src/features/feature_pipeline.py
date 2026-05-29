# src/features/feature_pipeline.py

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD

from src.utils.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    HORIZON_PREDICTION
)

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================================================
# LOAD DATA
# =========================================================

INPUT_FILE = Path(DATA_RAW_DIR) / "cac40_raw_prices.parquet"

OUTPUT_FILE = (
    Path(DATA_PROCESSED_DIR)
    / "cac40_features.parquet"
)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:

    logging.info("Début feature engineering...")

    # -----------------------------------------------------
    # TRI
    # -----------------------------------------------------

    df = df.sort_values(
        by=["ticker", "date"]
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # RETURNS
    # -----------------------------------------------------

    logging.info("Calcul des rendements...")

    df["return_1d"] = (
        df.groupby("ticker")["price"]
        .pct_change(1)
    )

    df["return_5d"] = (
        df.groupby("ticker")["price"]
        .pct_change(5)
    )

    df["return_20d"] = (
        df.groupby("ticker")["price"]
        .pct_change(20)
    )

    # -----------------------------------------------------
    # MOVING AVERAGES
    # -----------------------------------------------------

    logging.info("Calcul des moyennes mobiles...")

    df["ma_5"] = (
        df.groupby("ticker")["price"]
        .transform(lambda x: x.rolling(5).mean())
    )

    df["ma_20"] = (
        df.groupby("ticker")["price"]
        .transform(lambda x: x.rolling(20).mean())
    )

    df["ma_50"] = (
        df.groupby("ticker")["price"]
        .transform(lambda x: x.rolling(50).mean())
    )

    # -----------------------------------------------------
    # VOLATILITY
    # -----------------------------------------------------

    logging.info("Calcul de la volatilité...")

    df["volatility_20"] = (
        df.groupby("ticker")["return_1d"]
        .transform(lambda x: x.rolling(20).std())
    )

    # -----------------------------------------------------
    # VOLUME FEATURES
    # -----------------------------------------------------

    logging.info("Calcul des indicateurs volume...")

    df["volume_change"] = (
        df.groupby("ticker")["volume"]
        .pct_change()
    )

    df["volume_ma_20"] = (
        df.groupby("ticker")["volume"]
        .transform(lambda x: x.rolling(20).mean())
    )

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    logging.info("Calcul du RSI...")

    df["rsi"] = (
        df.groupby("ticker")["price"]
        .transform(
            lambda x: RSIIndicator(
                close=x,
                window=14
            ).rsi()
        )
    )

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    logging.info("Calcul du MACD...")

    def compute_macd(series):

        macd = MACD(
            close=series,
            window_slow=26,
            window_fast=12,
            window_sign=9
        )

        return macd.macd()

    df["macd"] = (
        df.groupby("ticker")["price"]
        .transform(compute_macd)
    )

    # -----------------------------------------------------
    # Z-SCORE CROSS-SECTIONNEL
    # -----------------------------------------------------

    logging.info("Calcul des Z-Scores cross-sectionnels...")

    features_to_standardize = [
        "return_1d",
        "return_5d",
        "return_20d",
        "volatility_20",
        "rsi",
        "macd",
        "volume_change"
    ]

    for feature in features_to_standardize:

        zscore_name = f"{feature}_zscore"

        df[zscore_name] = (
            df.groupby("date")[feature]
            .transform(
                lambda x: (
                    (x - x.mean()) / x.std()
                )
            )
        )

    # -----------------------------------------------------
    # FUTURE RETURNS
    # -----------------------------------------------------

    logging.info("Calcul des rendements futurs...")

    df["future_return"] = (
        df.groupby("ticker")["price"]
        .shift(-HORIZON_PREDICTION)
        / df["price"]
    ) - 1

    # -----------------------------------------------------
    # TARGET
    # -----------------------------------------------------

    logging.info("Création de la target...")

    df["target"] = np.where(
        df["future_return"] > 0,
        1,
        0
    )

    # -----------------------------------------------------
    # CLEANING FINAL
    # -----------------------------------------------------

    logging.info("Nettoyage final...")

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna()

    logging.info(
        f"Dataset final : {df.shape}"
    )

    return df

# =========================================================
# MAIN PIPELINE
# =========================================================

def run_feature_pipeline():

    logging.info("Chargement des données brutes...")

    df = pd.read_parquet(INPUT_FILE)

    logging.info(
        f"Données chargées : {df.shape}"
    )

    df_features = create_features(df)

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    logging.info("Sauvegarde des features...")

    df_features.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    logging.info(
        f"Features sauvegardées : {OUTPUT_FILE}"
    )

    return df_features

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    df = run_feature_pipeline()

    print(df.head())