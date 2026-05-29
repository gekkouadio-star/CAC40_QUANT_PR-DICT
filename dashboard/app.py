# dashboard/app.py

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

# Alignement du chemin vers la racine du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import DATA_PROCESSED_DIR

# Configuration de la page Streamlit
st.set_page_config(
    page_title="DASHBOARD - CAC40 QUANT PRÉDICT",
    page_icon="📈",
    layout="wide"
)

# =========================================================
# SIDEBAR : MARCHÉ EN TEMPS RÉEL
# =========================================================
with st.sidebar:
    st.header("🌍 MARCHÉ EN DIRECT")
    try:
        # Récupération du cours actuel du CAC 40
        cac40_ticker = yf.Ticker("^FCHI")
        hist = cac40_ticker.history(period="1d")
        last_price = hist["Close"].iloc[-1]
        st.metric("CAC 40 ACTUEL", f"{last_price:,.2f} PTS")
    except Exception as e:
        st.warning("FLUX LIVE INDISPONIBLE.")

# TITRE PRINCIPAL ENCADRÉ (MÉTHODE CSS ROBUSTE)
st.markdown("""
    <style>
    .title-box {
        border: 3px solid #00FFCC;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        background-color: #0E1117;
    }
    </style>
    <div class="title-box">
        <h1 style="color: #FFFFFF; margin: 0;">DASHBOARD - CAC40 QUANT PRÉDICT</h1>
    </div>
    <br>
""", unsafe_allow_html=True)

st.subheader("ANALYSE DE LA STRATÉGIE QUANTITATIVE PILOTÉE PAR IA (XGBOOST)")

# Chargement des données de performance du backtest
csv_path = Path(DATA_PROCESSED_DIR) / "backtest_performance.csv"

if not csv_path.exists():
    st.error(f"FICHIER DE PERFORMANCE INTROUVABLE : {csv_path}. VEUILLEZ LANCER LE SCRIPT DE BACKTEST EN PREMIER.")
else:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # =========================================================
    # CALCUL DES MÉTRIQUES POUR AFFICHAGE KPI
    # =========================================================
    # Rendements finaux
    strat_final_return = df["cum_strategy"].iloc[-1] * 100
    market_final_return = df["cum_market"].iloc[-1] * 100
    
    # Volatilités annualisées
    vol_strat = df["strategy_return"].std() * np.sqrt(252) * 100
    vol_market = df["market_return"].std() * np.sqrt(252) * 100
    
    # Ratios de Sharpe
    sharpe_strat = (df["strategy_return"].mean() / df["strategy_return"].std()) * np.sqrt(252) if df["strategy_return"].std() != 0 else 0
    sharpe_market = (df["market_return"].mean() / df["market_return"].std()) * np.sqrt(252) if df["market_return"].std() != 0 else 0
    
    # Max Drawdowns
    cum_strat_prices = (1 + df["strategy_return"]).cumprod()
    max_drawdown_strat = ((cum_strat_prices - cum_strat_prices.cummax()) / cum_strat_prices.cummax()).min() * 100
    
    cum_market_prices = (1 + df["market_return"]).cumprod()
    max_drawdown_mkt = ((cum_market_prices - cum_market_prices.cummax()) / cum_market_prices.cummax()).min() * 100

    # =========================================================
    # AFFICHAGE DES CLASSIQUES METRICS CARDS
    # =========================================================
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="RENDEMENT STRATÉGIE (VS CAC40)", 
            value=f"{strat_final_return:.2f} %", 
            delta=f"+{(strat_final_return - market_final_return):.2f} % VS MARCHÉ"
        )
    with col2:
        st.metric(
            label="RATIO DE SHARPE", 
            value=f"{sharpe_strat:.2f}", 
            delta=f"+{(sharpe_strat - sharpe_market):.2f} VS INDEX",
            delta_color="normal"
        )
    with col3:
        st.metric(
            label="⚡ VOLATILITÉ ANNUALISÉE", 
            value=f"{vol_strat:.2f} %", 
            delta=f"MARCHÉ: {vol_market:.2f} %",
            delta_color="inverse"
        )
    with col4:
        st.metric(
            label="MAXIMUM DRAWDOWN", 
            value=f"{max_drawdown_strat:.2f} %", 
            delta=f"MARCHÉ: {max_drawdown_mkt:.2f} %",
            delta_color="inverse"
        )

    st.markdown("---")

    # =========================================================
    # GRAPHIQUE INTERACTIF PLOTLY
    # =========================================================
    st.subheader("ÉVOLUTION DE L'ÉQUITÉ CUMULÉE (PÉRIODE DE TEST OUT-OF-SAMPLE)")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df["date"], 
        y=df["cum_strategy"] * 100,
        mode='lines',
        name='STRATÉGIE IA (TOP 5 SÉLECTION LONG)',
        line=dict(color='#00FFCC', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 204, 0.05)'
    ))
    
    fig.add_trace(go.Scatter(
        x=df["date"], 
        y=df["cum_market"] * 100,
        mode='lines',
        name='CAC40 MOYEN (ÉQUIPONDÉRÉ)',
        line=dict(color='#FF4B4B', width=1.5, dash='dash')
    ))
    
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="DATE",
        yaxis_title="RENDEMENT CUMULÉ (%)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=20, r=20, t=20, b=20),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # =========================================================
    # ENCART DE PRÉDICTION EN DIRECT (INFÉRENCE LIVE)
    # =========================================================
    st.markdown("---")
    st.subheader("SIGNAUX D'ALLOCATION D'ACTIFS POUR LA PROCHAINE SÉANCE")
    
    pred_path = Path(DATA_PROCESSED_DIR) / "latest_predictions.csv"
    
    if not pred_path.exists():
        st.info("AUCUN SIGNAL LIVE GÉNÉRÉ. LANCEZ `PYTHON -M SRC.INFERENCE.PREDICT` POUR CALCULER LES SIGNAUX DU JOUR.")
    else:
        df_preds = pd.read_csv(pred_path)
        last_date_str = pd.to_datetime(df_preds["date"].iloc[0]).strftime('%d/%m/%Y')
        
        st.markdown(f"CES PRÉDICTIONS ONT ÉTÉ CALCULÉES SUR LA BASE DES COURS DE CLÔTURE DU **{last_date_str}**.")
        
        top_5_display = df_preds.head(5).copy()
        top_5_display["proba_hausse"] = (top_5_display["proba_hausse"] * 100).round(2).astype(str) + " %"
        top_5_display.columns = ["DATE DE CALCUL", "CODE TICKER (ACTION)", "PROBABILITÉ DE SURPERFORMANCE"]
        
        st.dataframe(
            top_5_display,
            use_container_width=True,
            hide_index=True
        )

    # =========================================================
    # SECTION EXPLICATION TECHNIQUE
    # =========================================================
    st.markdown("---")
    with st.expander("VOIR LA MÉTHODOLOGIE DE L'ARCHITECTURE IA & RISQUE"):
        st.markdown("""
        **SPÉCIFICATIONS DU FRAMEWORK QUANTITATIF :**
        * **MODÈLE PRÉDICTIF :** `XGBOOST CLASSIFIER` OPTIMISÉ SUR LA MÉTRIQUE AUC.
        * **STATIONNARISATION :** STANDARDISATION PAR Z-SCORE CROSS-SECTIONNELLE PAR DATE.
        * **GESTION DU RISQUE :** ALLOCATION JOURNALIÈRE ÉQUIPONDÉRÉE SUR LE TOP 5 DES ACTIFS.
        """)