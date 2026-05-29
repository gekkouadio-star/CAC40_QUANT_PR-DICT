import logging
from pathlib import Path
import joblib

import pandas as pd

from src.utils.config import (
    DATA_PROCESSED_DIR,
    MODELS_DIR
)

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================================================
# PATHS
# =========================================================

FEATURES_FILE = (
    Path(DATA_PROCESSED_DIR)
    / "cac40_features.parquet"
)

MODEL_FILE = (
    Path(MODELS_DIR)
    / "xgboost"
    / "xgb_model.pkl"
)

OUTPUT_FILE = (
    Path(DATA_PROCESSED_DIR)
    / "predictions.parquet"
)

# =========================================================
# FEATURES
# =========================================================

FEATURE_COLUMNS = [

    "return_1d_zscore",
    "return_5d_zscore",
    "return_20d_zscore",

    "volatility_20_zscore",

    "rsi_zscore",
    "macd_zscore",

    "volume_change_zscore"
]

# =========================================================
# LOAD DATA
# =========================================================

logging.info("Chargement des données features...")

df = pd.read_parquet(FEATURES_FILE)

logging.info(f"Dataset : {df.shape}")

# =========================================================
# LOAD MODEL
# =========================================================

logging.info("Chargement modèle IA...")

model = joblib.load(MODEL_FILE)

# =========================================================
# PREDICTIONS
# =========================================================

logging.info("Calcul des probabilités...")

df["probability_up"] = model.predict_proba(
    df[FEATURE_COLUMNS]
)[:, 1]

# =========================================================
# DAILY RANKING
# =========================================================

logging.info("Création du ranking journalier...")

df["rank"] = (
    df.groupby("date")["probability_up"]
    .rank(
        ascending=False,
        method="first"
    )
)

# =========================================================
# KEEP ONLY USEFUL COLUMNS
# =========================================================

predictions = df[
    [
        "date",
        "ticker",
        "probability_up",
        "rank"
    ]
].copy()

# =========================================================
# SAVE
# =========================================================

predictions.to_parquet(
    OUTPUT_FILE,
    index=False
)

logging.info(
    f"Prédictions sauvegardées : {OUTPUT_FILE}"
)

# =========================================================
# DISPLAY TOP 5
# =========================================================

latest_date = predictions["date"].max()

top5 = (
    predictions[
        predictions["date"] == latest_date
    ]
    .sort_values(
        by="probability_up",
        ascending=False
    )
    .head(5)
)

print("\n===== TOP 5 IA =====\n")

print(top5)