# Dashboard — CAC40 Quant Prédict

## Description

**Dashboard — CAC40 Quant Prédict** est une plateforme d’analyse quantitative dédiée aux actions du CAC 40.

Le projet combine :

* Data Engineering
* Machine Learning
* Backtesting
* Visualisation interactive
* Analyse quantitative multi-actifs

afin de générer des signaux de trading basés sur une stratégie de sélection **Long Only** des actifs présentant les meilleures probabilités de surperformance à horizon court terme.

L’objectif est d’optimiser l’allocation d’actifs en s’appuyant sur des indicateurs techniques stationnarisés (*Z-Score cross-sectionnel*) afin d’identifier les opportunités de hausse relative sur le marché français.

---

# Fonctionnalités principales

* Téléchargement automatique des données du CAC40
* Pipeline de feature engineering
* Modèle de prédiction XGBoost
* Génération de signaux de trading
* Backtesting de stratégie
* Dashboard interactif Streamlit
* Analyse des performances du portefeuille
* Visualisation des prédictions et métriques financières

---

# Stack Technique

| Domaine             | Technologies             |
| ------------------- | ------------------------ |
| Langage             | Python 3.x               |
| Machine Learning    | XGBoost                  |
| Dashboard           | Streamlit                |
| Visualisation       | Plotly                   |
| Data Processing     | Pandas / NumPy           |
| Données financières | Yahoo Finance (yfinance) |
| Backtesting         | VectorBT                 |
| API                 | FastAPI                  |
| Base de données     | PostgreSQL               |
| Versioning          | Git / GitHub             |

---

# Structure du projet

```plaintext
AlphaIntelligence/
│
├── dashboard/              # Application Streamlit
│   └── app.py
│
├── data/
│   ├── raw/                # Données brutes téléchargées
│   ├── processed/          # Données feature engineering
│   └── external/
│
├── models/                 # Modèles entraînés
│   ├── xgboost/
│   └── saved_models/
│
├── src/
│   ├── data/               # Pipeline ingestion
│   ├── features/           # Feature engineering
│   ├── training/           # Entraînement ML
│   ├── inference/          # Génération prédictions
│   ├── backtesting/        # Backtests stratégie
│   └── utils/
│
├── notebooks/              # Recherche & expérimentation
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# Méthodologie IA

## Modélisation

Le moteur prédictif repose sur un classifieur **XGBoost** entraîné sur des données historiques multi-actifs du CAC40.

Le modèle cherche à prédire la probabilité de hausse d’une action à horizon de 5 jours ouvrés.

---

## Variables utilisées

Les features incluent notamment :

* Rendements historiques
* Moyennes mobiles
* Volatilité
* Momentum
* RSI
* MACD
* Volume relatif
* Performance relative au CAC40

---

## Standardisation

Les variables sont normalisées via un :

* **Z-score cross-sectionnel par date**

afin de réduire le bruit temporel et rendre comparables les actifs entre eux.

---

## Gestion du risque

La stratégie applique une logique :

* **Long Only**
* équipondérée
* sur le Top 5 des actifs présentant les meilleures probabilités de hausse.

---

# Pipeline Quantitatif

```plaintext
Téléchargement données
        ↓
Feature Engineering
        ↓
Machine Learning
        ↓
Prédictions
        ↓
Backtesting
        ↓
Dashboard interactif
```

---

# Objectifs du projet

* Construire une plateforme de recherche quantitative
* Développer une IA de prédiction boursière
* Implémenter un pipeline ML financier complet
* Optimiser les signaux de trading
* Évaluer les performances via backtesting

---

# Évolutions futures

* Intégration de données macroéconomiques
* Analyse de sentiment NLP (FinBERT)
* LSTM / Transformers temporels
* Optimisation portefeuille Markowitz
* Déploiement cloud
* API temps réel
* MLOps & monitoring

---

# Avertissement

Ce projet est développé à des fins :

* pédagogiques,
* expérimentales,
* de recherche quantitative.

Les prédictions financières comportent un risque important et ne constituent pas des conseils d’investissement.