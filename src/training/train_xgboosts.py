# src/training/train_xgboost.py

import logging
from pathlib import Path
import joblib

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)

from xgboost import XGBClassifier

from src.utils.config import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    TARGET_COLUMN,
    XGBOOST_PARAMS
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

INPUT_FILE = (
    Path(DATA_PROCESSED_DIR)
    / "cac40_features.parquet"
)

MODEL_OUTPUT = (
    Path(MODELS_DIR)
    / "xgboost"
    / "xgb_model.pkl"
)

# =========================================================
# LOAD DATA
# =========================================================

logging.info("Chargement des features...")

df = pd.read_parquet(INPUT_FILE)

logging.info(f"Dataset chargé : {df.shape}")

# =========================================================
# FEATURES
# =========================================================

FEATURES = [

    "return_1d_zscore",
    "return_5d_zscore",
    "return_20d_zscore",

    "volatility_20_zscore",

    "rsi_zscore",
    "macd_zscore",

    "volume_change_zscore"

]

X = df[FEATURES]

y = df[TARGET_COLUMN]

# =========================================================
# TEMPORAL SPLIT
# =========================================================

logging.info("Split temporel...")

split_date = "2023-01-01"

train_df = df[df["date"] < split_date]
test_df = df[df["date"] >= split_date]

X_train = train_df[FEATURES]
y_train = train_df[TARGET_COLUMN]

X_test = test_df[FEATURES]
y_test = test_df[TARGET_COLUMN]

logging.info(
    f"Train shape : {X_train.shape}"
)

logging.info(
    f"Test shape : {X_test.shape}"
)

# =========================================================
# MODEL
# =========================================================

logging.info("Entraînement XGBoost...")

model = XGBClassifier(
    **XGBOOST_PARAMS,
    objective="binary:logistic",
    eval_metric="auc"
)

model.fit(X_train, y_train)

# =========================================================
# PREDICTIONS
# =========================================================

logging.info("Prédictions...")

y_pred = model.predict(X_test)

y_proba = model.predict_proba(X_test)[:, 1]

# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

auc = roc_auc_score(y_test, y_proba)

logging.info(f"Accuracy : {accuracy:.4f}")

logging.info(f"AUC : {auc:.4f}")

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\nFeature Importance\n")

print(importance_df)

# =========================================================
# SAVE MODEL
# =========================================================

logging.info("Sauvegarde du modèle...")

MODEL_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_OUTPUT
)

logging.info(
    f"Modèle sauvegardé : {MODEL_OUTPUT}"
)