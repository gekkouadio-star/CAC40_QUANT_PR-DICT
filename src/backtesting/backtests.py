import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.config import (
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
# PATHS
# =========================================================

PREDICTIONS_FILE = (
    Path(DATA_PROCESSED_DIR)
    / "predictions.parquet"
)

FEATURES_FILE = (
    Path(DATA_PROCESSED_DIR)
    / "cac40_features.parquet"
)

# =========================================================
# LOAD DATA
# =========================================================

logging.info("Chargement des prédictions...")

predictions = pd.read_parquet(PREDICTIONS_FILE)

logging.info("Chargement dataset features...")

features = pd.read_parquet(FEATURES_FILE)

# =========================================================
# KEEP TOP 5
# =========================================================

portfolio = predictions[
    predictions["rank"] <= 5
].copy()

# =========================================================
# MERGE FUTURE RETURNS
# =========================================================

portfolio = portfolio.merge(

    features[
        [
            "date",
            "ticker",
            "future_return"
        ]
    ],

    on=["date", "ticker"],
    how="left"
)

# =========================================================
# TRANSACTION COSTS
# =========================================================

transaction_cost = 0.001

portfolio["net_return"] = (
    portfolio["future_return"]
    - transaction_cost
)

# =========================================================
# DAILY PORTFOLIO RETURNS
# =========================================================

daily_returns = (
    portfolio.groupby("date")["net_return"]
    .mean()
)

daily_returns = daily_returns.sort_index()

# =========================================================
# CUMULATIVE RETURNS
# =========================================================

cumulative_returns = (
    (1 + daily_returns)
    .cumprod()
)

# =========================================================
# CAGR
# =========================================================

n_years = len(daily_returns) / 252

cagr = (
    cumulative_returns.iloc[-1]
    ** (1 / n_years)
) - 1

# =========================================================
# VOLATILITY
# =========================================================

annual_volatility = (
    daily_returns.std()
    * np.sqrt(252)
)

# =========================================================
# SHARPE RATIO
# =========================================================

sharpe_ratio = (
    daily_returns.mean()
    / daily_returns.std()
) * np.sqrt(252)

# =========================================================
# MAX DRAWDOWN
# =========================================================

rolling_max = cumulative_returns.cummax()

drawdown = (
    cumulative_returns
    / rolling_max
) - 1

max_drawdown = drawdown.min()

# =========================================================
# RESULTS
# =========================================================

print("\n===== BACKTEST RESULTS =====\n")

print(f"CAGR : {cagr*100:.2f}%")

print(
    f"Annual Volatility : "
    f"{annual_volatility*100:.2f}%"
)

print(f"Sharpe Ratio : {sharpe_ratio:.2f}")

print(
    f"Max Drawdown : "
    f"{max_drawdown*100:.2f}%"
)

# =========================================================
# PLOT
# =========================================================

plt.figure(figsize=(14, 7))

plt.plot(
    cumulative_returns,
    label="AI Portfolio"
)

plt.title(
    "AI Portfolio Performance"
)

plt.xlabel("Date")

plt.ylabel("Cumulative Return")

plt.grid()

plt.legend()

plt.show()