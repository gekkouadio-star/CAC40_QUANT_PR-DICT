# =========================================================
# IMPORTS
# =========================================================
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from src.utils.config import DATA_PROCESSED_DIR

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Dashboard de Prédictions CAC40",
    page_icon="📈",
    layout="wide"
)

# =========================================================
# CSS GLOBAL
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.metric-card {
    background-color: #161B22;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #30363D;
}

.title-box {
    border: 2px solid #00FFCC;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    background-color: #161B22;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🌍 MARCHÉ EN DIRECT")

    try:

        cac40 = yf.Ticker("^FCHI")

        hist = cac40.history(period="1d")

        last_price = hist["Close"].iloc[-1]

        st.metric(
            "CAC40 LIVE",
            f"{last_price:,.2f} PTS"
        )

    except:

        st.warning(
            "Flux live indisponible."
        )

    st.markdown("---")

    st.header("Paramètres")

# =========================================================
# TITLE
# =========================================================

st.markdown("""
<div class="title-box">

<h1 style="color:white;">
DASHBOARD DE PRÉDICTIONS CAC40
</h1>

<h4 style="color:#00FFCC;">
Platforme de Prédictions — CAC40
</h4>

</div>
""", unsafe_allow_html=True)

st.markdown("")

# =========================================================
# LOAD DATA
# =========================================================

predictions_file = (
    Path(DATA_PROCESSED_DIR)
    / "predictions.parquet"
)

features_file = (
    Path(DATA_PROCESSED_DIR)
    / "cac40_features.parquet"
)

backtest_file = (
    Path(DATA_PROCESSED_DIR)
    / "backtest_performance.csv"
)

# =========================================================
# CHECK FILES
# =========================================================

required_files = [
    predictions_file,
    features_file,
    backtest_file
]

for file in required_files:

    if not file.exists():

        st.error(f"Fichier manquant : {file}")

        st.stop()

# =========================================================
# READ FILES
# =========================================================

predictions = pd.read_parquet(
    predictions_file
)

features = pd.read_parquet(
    features_file
)

backtest_df = pd.read_csv(
    backtest_file
)

backtest_df["date"] = pd.to_datetime(
    backtest_df["date"]
)

# =========================================================
# DATE FILTER
# =========================================================

selected_date = st.sidebar.selectbox(

    "Date d'analyse",

    sorted(
        predictions["date"].unique()
    )[::-1]
)

# =========================================================
# DAILY PREDICTIONS
# =========================================================

daily_predictions = (

    predictions[
        predictions["date"]
        == selected_date
    ]

    .sort_values(
        by="probability_up",
        ascending=False
    )
)

# =========================================================
# KPIs
# =========================================================

strategy_return = (
    backtest_df["cum_strategy"].iloc[-1]
) * 100

market_return = (
    backtest_df["cum_market"].iloc[-1]
) * 100

volatility = (
    backtest_df["strategy_return"].std()
    * np.sqrt(252)
) * 100

sharpe = (

    backtest_df["strategy_return"].mean()

    /

    backtest_df["strategy_return"].std()

) * np.sqrt(252)

cum_prices = (
    1 + backtest_df["strategy_return"]
).cumprod()

max_drawdown = (

    (
        cum_prices
        - cum_prices.cummax()
    )

    /

    cum_prices.cummax()

).min() * 100

# =========================================================
# KPI DISPLAY
# =========================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Performance IA",
    f"{strategy_return:.2f} %",
    f"{strategy_return - market_return:.2f}% vs CAC40"
)

col2.metric(
    "Sharpe Ratio",
    f"{sharpe:.2f}"
)

col3.metric(
    "Volatilité",
    f"{volatility:.2f}%"
)

col4.metric(
    "Max Drawdown",
    f"{max_drawdown:.2f}%"
)

# =========================================================
# PERFORMANCE CHART
# =========================================================

st.markdown("---")

st.subheader(
    "Performance du Portefeuille vs CAC40"
)

fig_perf = go.Figure()

fig_perf.add_trace(

    go.Scatter(

        x=backtest_df["date"],

        y=backtest_df["cum_strategy"] * 100,

        mode="lines",

        name="AI Portfolio",

        line=dict(
            color="#00FFCC",
            width=3
        )
    )
)

fig_perf.add_trace(

    go.Scatter(

        x=backtest_df["date"],

        y=backtest_df["cum_market"] * 100,

        mode="lines",

        name="CAC40",

        line=dict(
            color="#FF4B4B",
            dash="dash"
        )
    )
)

fig_perf.update_layout(

    template="plotly_dark",

    hovermode="x unified",

    height=550,

    xaxis_title="Date",

    yaxis_title="Performance (%)"
)

st.plotly_chart(
    fig_perf,
    use_container_width=True
)

# =========================================================
# TOP 5 IA
# =========================================================

st.markdown("---")

st.subheader(
    "Top 5 — Prochaine Séance"
)

top5 = daily_predictions.head(5).copy()

top5["probability_up"] = (
    top5["probability_up"]
    * 100
).round(2)

top5 = top5.rename(columns={

    "ticker": "Ticker",

    "probability_up": "Probabilité Hausse (%)",

    "rank": "Classement"
})

st.dataframe(

    top5[
        [
            "Ticker",
            "Probabilité Hausse (%)",
            "Classement"
        ]
    ],

    use_container_width=True,
    hide_index=True
)

# =========================================================
# BAR CHART TOP 10
# =========================================================

st.subheader(
    "Ranking — Top 10"
)

fig_bar = px.bar(

    daily_predictions.head(10),

    x="ticker",

    y="probability_up",

    color="probability_up",

    title="Probabilité de Hausse"
)

fig_bar.update_layout(
    template="plotly_dark"
)

st.plotly_chart(
    fig_bar,
    use_container_width=True
)

# =========================================================
# TECHNICAL DETAILS
# =========================================================

st.markdown("---")

with st.expander(
    "Architecture IA & Méthodologie Quant"
):

    st.markdown("""

### Machine Learning
- XGBoost Classifier
- Optimisation sur métrique AUC

### Feature Engineering
- RSI
- MACD
- Momentum
- Volatilité
- Volume anomalies
- Z-score cross-sectionnel

### Allocation
- Top 5 actions
- Portefeuille équipondéré
- Rebalancing quotidien

### Gestion du Risque
- Benchmark CAC40
- Volatilité annualisée
- Sharpe Ratio
- Maximum Drawdown

""")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Dashboard de Prédictions CAC40 — Quantitative Research Platform"
)