# 📦 SIRA | Smart Inventory Restock Advisor
> **AI-Powered Demand Forecasting & Automated Replenishment Engine**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-F97316?style=for-the-badge&logo=xgboost&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chart.js&logoColor=white)

> **AIMERS Club ML Hackathon 2026 · Problem Statement 5**
> Team **Mission:Imputable** — Yasaswini Chebolu, Amrutha Dadi, Kavitha Manikyamba · 3 CSM-1, GVPCEW

---

## 🚀 Overview

**SIRA** is a full-stack, multi-page intelligent inventory management system. It uses machine learning to predict product demand, calculates optimal restock quantities using Economic Order Quantity (EOQ) principles, and delivers real-time AI-driven recommendations through a professional 6-page web dashboard.

The system transforms raw BigMart sales data into **actionable logistics directives** — telling store managers exactly what to order, how much, and why.

---

## 💡 Problem Statement

In retail, stock management is often reactive. Businesses either suffer from **frozen capital** (overstocking perishables) or **lost revenue** (stockouts of fast-moving items). SIRA solves this by:

- Forecasting monthly demand per SKU using XGBoost
- Calculating safety stock, reorder points, and suggested order quantities
- Flagging critical inventory items before stockouts occur
- Generating plain-English AI recommendations and professional PDF reports

---

## 🛠️ Tech Stack

| Layer | Technologies |
|:---|:---|
| **ML Engine** | XGBoost (Primary), Random Forest (Secondary), Scikit-Learn, 5-Fold CV |
| **Data** | Pandas, NumPy, BigMart Sales Dataset (Kaggle) |
| **Backend** | Flask (REST API), 8 API Endpoints, ReportLab (PDF), Joblib |
| **Frontend** | HTML5, CSS3 (Glassmorphism), Vanilla JavaScript, Chart.js |
| **Model Persistence** | Joblib (.pkl) — Model, Scaler, Label Encoders, OHE |

---

## 📊 Model Performance
*Evaluated on the BigMart Sales Dataset using 5-Fold Cross Validation.*

| Model | CV RMSE | R² Score | Verdict |
|:---|:---|:---|:---|
| Linear Regression | ~1180 | 0.41 | Baseline |
| Random Forest | ~1050 | 0.54 | Good — retained as fallback |
| **XGBoost ✓** | **~1022** | **0.57** | **🏆 Primary Model** |

> XGBoost outperformed baseline Linear Regression by ~13% on RMSE with 500 estimators and 5-fold cross-validation.

---

## 🗂️ Application Pages (6 Standalone Routes)

| Route | Page | Purpose |
|:---|:---|:---|
| `/` | **Landing Page** | Hero intro, model stats, CTA buttons |
| `/dashboard` | **Overview** | Inventory health, alert distribution, loss simulator |
| `/predictor` | **AI Predictor Hub** | On-Demand Simulation Hub with 3 AI features |
| `/simulator` | **Scenario Simulator** | What-If supply chain scenario builder |
| `/upload` | **Batch Upload** | CSV drag-and-drop bulk forecasting |
| `/analytics` | **Market Intelligence** | AI Executive Summary, radar charts, loss reports |

---

## ✨ Key Features

### 🧠 AI Predictor Hub (`/predictor`)
- **On-Demand Simulation Hub** — configure 5 inputs (Category, MRP, Outlet Type, Location Tier, Stock Level) and run a live ML prediction
- **AI Smart Advice Engine** — auto-generates plain-English restock recommendation based on alert level and demand delta
- **Predictive Seasonality Chart** — 12-month demand sparkline with Indian retail seasons (Diwali +40%, Holi +15%, New Year +20%) auto-detected
- **Scenario Comparison Delta** — remembers previous run and shows exact % demand shift with trend arrow

### 📈 Market Intelligence Analytics (`/analytics`)
- **AI Executive Summary** — dynamically generated business insight paragraph from live model weights
- **Location Intelligence Radar** — demand distribution across outlet types and tiers
- **Feature Importance Chart** — XGBoost feature weights visualized
- **Financial Loss Simulation** — stacked bar chart of overstock vs stockout losses by category
- **One-click PDF Report** — professional ReportLab-generated report ready for stakeholders

### 🎛️ Scenario Simulator (`/simulator`)
- Sliders for demand multiplier, supplier lead time, and safety buffer factor
- Saved Scenarios panel — store and compare multiple configurations
- Live Delta indicators vs base scenario

### 📦 Batch Upload (`/upload`)
- Drag-and-drop CSV processing
- Bulk restock report generation with alert classification
- Download official PDF report

---

## 📂 Project Structure

```text
SIRA/
├── templates/
│   ├── index.html          # Landing page
│   ├── dashboard.html      # Overview & loss simulator
│   ├── predictor.html      # AI Predictor Hub (On-Demand Simulation Hub)
│   ├── simulator.html      # What-If scenario builder
│   ├── upload.html         # CSV batch upload
│   └── analytics.html      # Market Intelligence page
├── models/
│   ├── demand_model.pkl    # Trained XGBoost model
│   ├── scaler.pkl          # StandardScaler
│   ├── encoders.pkl        # Label + OHE encoders
│   └── model_metadata.json # Training metadata
├── static/
│   └── charts/             # Dynamically generated chart assets
├── app.py                  # Flask server — all routes & API endpoints
├── train.py                # ML training & cross-validation pipeline
├── preprocess.py           # EDA & feature engineering
├── predict.py              # Single & batch inference engine + trend detection
├── utils.py                # EOQ restock logic, alert engine, PDF generator
└── requirements.txt
```

---

## 🏃 Quick Start (3 Steps)

**1. Clone & Install**
```bash
git clone https://github.com/Yasaswini-ch/SIRA.git
cd SIRA
pip install -r requirements.txt
```

**2. Train the Model**
```bash
python train.py
```

**3. Launch the App**
```bash
python app.py
```
Open **http://localhost:5000** in your browser.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/` | Landing page |
| `GET` | `/dashboard` | Main dashboard |
| `GET` | `/predictor` | AI Predictor Hub |
| `GET` | `/simulator` | Scenario simulator |
| `GET` | `/upload` | Batch upload page |
| `GET` | `/analytics` | Market intelligence |
| `POST` | `/predict` | Single-item ML prediction + trend data |
| `POST` | `/upload-csv` | Batch CSV inference |
| `GET` | `/api/dashboard-stats` | Feature importance, category demand, alert distribution |
| `GET` | `/api/loss-report` | Financial loss simulation data |
| `GET` | `/download-report` | PDF report download |

---

## 👥 Team

| Name | Role |
|:---|:---|
| **Yasaswini Chebolu** | Team Lead — ML Architecture & Backend |
| **Amrutha Dadi** | Frontend & Visualization |
| **Kavitha Manikyamba** | Data Processing & Model Evaluation |

**Department:** 3 CSM-1 · **College:** Gayatri Vidya Parishad College of Engineering for Women (GVPCEW)

---

*Built for the AIMERS ML Hackathon 2026 · Problem Statement 5 — Smart Inventory Restock Advisor*
