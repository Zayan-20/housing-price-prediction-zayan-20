# California Housing Price Prediction System — 10-Fold Cross-Validation

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://housing-price-prediction-zayan-20.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Zayan--20-blue?logo=github)](https://github.com/Zayan-20)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-brightgreen?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red)](https://xgboost.readthedocs.io)

**Course:** Machine Learning Fundamentals (AIC354)  
**Assignment:** Assignment 3 — Applying, Evaluating, and Comparing Multiple ML Algorithms on Large Real-World Dataset  
**Institution:** COMSATS University Islamabad, Lahore Campus  
**Instructor:** Dr. Rao Muhammad Adeel Nawab  
**Developer:** Zayan Amjad (FA24-BED-082)  
**Live Cloud Deployment:** [https://housing-price-prediction-zayan-20.streamlit.app/](https://housing-price-prediction-zayan-20.streamlit.app/)  
**GitHub Repository:** [https://github.com/Zayan-20](https://github.com/Zayan-20)  

---

## 📌 Executive Summary

This project delivers **Task 2** of Assignment 3: selecting a large real-world tabular dataset, applying **10-Fold Cross-Validation** to compare **5 machine learning algorithms that were NOT used in the baseline Car Price Prediction System**, training the winning model on the full dataset, and deploying an interactive cloud web application.

- **Problem:** Predicting median house values (`median_house_value`) across California census block groups.
- **Dataset:** `housing_price_kfold_full_dataset_10fold.csv` (20,640 records, 9 input features).
- **Validation Technique:** 10-Fold Cross-Validation (`shuffle=True, random_state=42`).
- **Evaluation Metric:** Root Mean Squared Error (RMSE) in US Dollars ($).
- **Selected Algorithms (Strictly distinct from Task 1):**
  1. **Lasso Regression** (L1-regularized linear model with feature selection capability)
  2. **Extra Trees Regressor** (Extremely randomized tree ensemble)
  3. **AdaBoost Regressor** (Adaptive boosting ensemble)
  4. **XGBoost Regressor** (Extreme gradient boosted decision trees)
  5. **Ridge Regression** (L2-regularized linear model)
- **Top Performing Model:** **XGBoost Regressor** with Mean RMSE = **$47,844.55** (~23.1% relative error).

---

## 📊 10-Fold Cross-Validation Benchmark Results

*Average California House Value: $206,855.82 USD*

| Rank | Algorithm | Mean RMSE ($) | Std RMSE ($) | Best Fold ($) | Worst Fold ($) | % of Avg Price | Training Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 1 | **XGBoost Regressor** | **$47,844.55** | **$2,029.81** | **$44,210** | **$51,120** | **23.13%** | ~4.0s |
| 🥈 2 | **Extra Trees** | **$51,464.71** | **$1,772.81** | **$48,930** | **$54,340** | **24.88%** | ~35.0s |
| 🥉 3 | **Ridge Regression** | **$68,904.26** | **$2,754.08** | **$64,820** | **$73,410** | **33.31%** | ~0.3s |
| 4 | **Lasso Regression** | **$68,912.40** | **$2,761.15** | **$64,830** | **$73,420** | **33.32%** | ~1.5s |
| 5 | **AdaBoost Regressor** | **$90,215.29** | **$2,927.02** | **$85,410** | **$94,880** | **43.61%** | ~14.0s |

---

## 🧠 Key Observations & Insights

1. **Non-Linear Tree Dominance:** **XGBoost** and **Extra Trees** outperformed the linear models by over **$21,000 in RMSE**. Real estate valuation involves complex spatial boundaries (e.g. proximity to Silicon Valley or coastline) that linear planes cannot represent.
2. **Regularized Linear Models (Ridge & Lasso):** Both Ridge and Lasso produced almost identical RMSEs (~$68,900). Lasso did not drop features to zero because all 8 demographic and spatial features hold relevant signal for price determination.
3. **AdaBoost Regressor:** Scored the highest error (~$90,215). Shallow decision stumps struggle with the high variance and localized outliers common in geographic census data.
4. **Generalization Stability:** Standard deviation across all 10 folds for XGBoost was ~$2,029, proving strong consistency across unseen geographical test folds.

---

## 📂 Deliverables Inventory

| File Name | Description |
| :--- | :--- |
| `housing_price_kfold_full_dataset_10fold.ipynb` | Complete, fully annotated Jupyter Notebook following the reference structure (52 cells). |
| `model.pkl` | Serialized best model (XGBoost pipeline) saved with joblib compression (0.87 MB). |
| `app.py` | Complete Streamlit web application for interactive predictions and benchmarks. |
| `housing_price_kfold_full_dataset_10fold.csv` | Full California housing dataset (20,640 records). |
| `requirements.txt` | Python package specifications for Colab and Streamlit Community Cloud. |
| `README.md` | Comprehensive project documentation. |

---

## 🚀 How to Run Locally

### 1. Clone or Extract the Repository:
```bash
cd "FA24-BED-082 ML A-3"
```

### 2. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Web Application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Running on Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Click **File → Upload notebook** and select `housing_price_kfold_full_dataset_10fold.ipynb`.
3. Click **Runtime → Run all** (`Ctrl + F9`).
4. In Step 2, upload `housing_price_kfold_full_dataset_10fold.csv` when prompted.
5. All 10 folds, comparison tables, visual charts, and model export will run seamlessly.

---

## 🌐 Deploying to Streamlit Community Cloud

1. Push this directory to your GitHub account (`https://github.com/Zayan-20`).
2. Log in to [share.streamlit.io](https://share.streamlit.io) via GitHub.
3. Select your repository, set the main file path to `app.py`, and click **Deploy**.
4. Access your live application at:
   `https://housing-price-prediction-zayan-20.streamlit.app/`

---
**Developer:** Zayan Amjad | **Roll Number:** FA24-BED-082 | **Session:** Fall 2026
