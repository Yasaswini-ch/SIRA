# 📦 SIRA | Smart Inventory Restock Advisor
> **AI-Powered Demand Forecasting & Automated Replenishment Engine**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1E293B?style=for-the-badge&logo=xgboost&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

---

### 🚀 Overview
**SIRA** is an end-to-end intelligent inventory management system built for the **AIMERS Hackathon 2026 (Problem Statement 5)**. It leverages machine learning to predict product demand, calculates optimal restock quantities using Economic Order Quantity (EOQ) principles, and provides real-time visual alerts to prevent stockouts.

### 💡 Problem Statement
In retail, stock management is often reactive. Businesses either suffer from **frozen capital** (overstocking) or **lost revenue** (stockouts). SIRA solves this by transforming raw sales data into actionable logistics directives.

---

### 🛠️ Tech Stack
*   **Core Logic:** Python 3.11, Pandas, NumPy
*   **ML Engine:** Scikit-Learn (Linear Regression, Random Forest), XGBoost
*   **Backend:** Flask (REST API)
*   **Frontend:** HTML5, CSS3 (Glassmorphism), Vanilla JavaScript
*   **Environment:** Project IDX (Dockerized DevNix)

---

### 🏃 Quick Start (3 Steps)
1.  **Clone & Setup**: 
    ```bash
    git clone https://github.com/Yasaswini-ch/SIRA.git && cd inventory-restock-advisor
    pip install -r requirements.txt
    ```
2.  **Generate Model**: 
    ```bash
    python train.py
    ```
3.  **Launch App**: 
    ```bash
    python app.py
    ```
    *Access via `http://localhost:5000`*

---

### 📂 Project Structure
```text
├── data/               # Raw datasets (Train.csv, Test.csv)
├── models/             # Serialized .pkl artifacts (Model, Scaler, Encoders)
├── static/
│   ├── css/            # Premium dark-mode styling
│   └── charts/         # Dynamically generated ML visualizations
├── templates/          # Responsive HTML5 dashboards
├── app.py              # Main Flask REST API server
├── train.py            # ML Model training & comparison pipeline
├── preprocess.py       # EDA & Feature Engineering logic
├── predict.py          # Batch and Single-item inference engine
└── utils.py            # EOQ Restock Logic & Charting Engine
```

---

### 📊 Model Performance
*Evaluated on the BigMart Sales Dataset using 5-Fold Cross Validation.*

| Model | CV RMSE (Mean) | Train R² Score | Generalization |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | **1132.07** | 0.5635 | **Winner (Best Generalizer)** |
| Random Forest | 1137.08 | 0.9378 | Overfitting detected |
| XGBoost Regressor | 1144.24 | 0.8374 | Balanced |

---

### 🎯 Core Features
1.  **Real-Time Predictor**: Single-item sales forecasting with intelligent fallback for missing data.
2.  **Batch Analysis**: Drag-and-drop CSV uploader that generates bulk restock reports instantly.
3.  **Smart Decision Engine**: Automated reorder point calculation based on lead-time and safety stock.

---

### 🖼️ Screenshots
*(Include images of your Landing Page, Dashboard, and Heatmaps here)*

---

### 👥 Team Section
*   **Yasaswini and Team** - ML Architecture & Backend
*   **College**: GVPCEW (Gayatri Vidya Parishad College of Engineering for Women)

---
*Built for the AIMERS ML 2026 Hackathon competition.*
