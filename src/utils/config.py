# src/utils/config.py

from pathlib import Path
from datetime import datetime

# =========================================================
# CHEMINS PROJET
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"

DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_EXTERNAL_DIR = DATA_DIR / "external"

MODELS_DIR = BASE_DIR / "models"

# Création automatique des dossiers
for directory in [
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_EXTERNAL_DIR,
    MODELS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# =========================================================
# PARAMÈTRES TEMPORELS
# =========================================================

START_DATE = "2015-01-01"

# Date dynamique automatique
END_DATE = datetime.today().strftime("%Y-%m-%d")

# Horizon de prédiction :
# 5 jours ouvrés ≈ 1 semaine de trading
HORIZON_PREDICTION = 5

# =========================================================
# TICKERS CAC40
# =========================================================

CAC40_TICKERS = [
    "AC.PA",    # Accor
    "AI.PA",    # Air Liquide
    "AIR.PA",   # Airbus
    "ALO.PA",   # Alstom
    "MT.PA",    # ArcelorMittal
    "CS.PA",    # AXA
    "BNP.PA",   # BNP Paribas
    "EN.PA",    # Bouygues
    "CAP.PA",   # Capgemini
    "CA.PA",    # Carrefour
    "ACA.PA",   # Crédit Agricole
    "BN.PA",    # Danone
    "DSY.PA",   # Dassault Systèmes
    "EDEN.PA",  # Edenred
    "ENGI.PA",  # Engie
    "EL.PA",    # EssilorLuxottica
    "ERF.PA",   # Eurofins Scientific
    "RMS.PA",   # Hermès
    "KER.PA",   # Kering
    "OR.PA",    # L'Oréal
    "LR.PA",    # Legrand
    "MC.PA",    # LVMH
    "ML.PA",    # Michelin
    "ORA.PA",   # Orange
    "RI.PA",    # Pernod Ricard
    "PUB.PA",   # Publicis
    "RNO.PA",   # Renault
    "SAF.PA",   # Safran
    "SGO.PA",   # Saint-Gobain
    "SAN.PA",   # Sanofi
    "SU.PA",    # Schneider Electric
    "GLE.PA",   # Société Générale
    "STMPA.PA", # STMicroelectronics
    "TEP.PA",   # Teleperformance
    "HO.PA",    # Thales
    "TTE.PA",   # TotalEnergies
    "VIE.PA",   # Veolia
    "DG.PA",    # VINCI
    "VIV.PA",   # Vivendi
    "WLN.PA"    # Worldline
]

# =========================================================
# BENCHMARKS
# =========================================================

BENCHMARK_TICKER = "^FCHI"

# =========================================================
# FEATURES TECHNIQUES
# =========================================================

TECHNICAL_FEATURES = [
    "return_1d",
    "return_5d",
    "ma_5",
    "ma_20",
    "ma_50",
    "volatility_20",
    "rsi",
    "macd",
    "volume_change",
    # Nouvelles variables de contexte macro :
    "vix_close",
    "vix_return_1d",
    "sp500_return_1d",
    "sp500_return_5d"
]

# =========================================================
# PARAMÈTRES ML
# =========================================================

TEST_SIZE = 0.2

RANDOM_STATE = 42

TARGET_COLUMN = "target"

# =========================================================
# MODÈLE XGBOOST
# =========================================================

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE
}