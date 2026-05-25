# 🏦 Term Deposit Subscription Prediction

---

## 📌 Task Objective

Predict whether a bank customer will subscribe to a **term deposit** as a result of a direct marketing campaign, using the UCI Bank Marketing dataset. This is a binary classification problem (`yes` / `no`).

---

## 🗂️ Dataset

**Source:** [UCI Machine Learning Repository — Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

- **Rows:** 4,521 customer records
- **Features:** 16 input features (demographic + campaign-related)
- **Target:** `y` — Has the client subscribed a term deposit? (`yes` / `no`)
- **Class Imbalance:** ~92.7% No vs ~7.3% Yes

### Feature Summary
| Category | Features |
|---|---|
| Demographics | `age`, `job`, `marital`, `education` |
| Financial | `balance`, `default`, `housing`, `loan` |
| Campaign | `contact`, `day`, `month`, `duration`, `campaign` |
| Previous Campaign | `pdays`, `previous`, `poutcome` |

---

## 🛠️ My Approach

### 1. Exploratory Data Analysis (EDA)
- Analysed class imbalance (~13:1 ratio)
- Visualised subscription rates across all categorical features
- Correlation analysis between numeric features and target
- Call duration identified as top predictor

### 2. Preprocessing
- Binary encoding: `default`, `housing`, `loan` → 0/1
- Ordinal encoding: `education` (primary=1, secondary=2, tertiary=3)
- Month → numeric (Jan=1, ..., Dec=12)
- One-hot encoding: `job`, `marital`, `contact`, `poutcome`
- StandardScaler applied for Logistic Regression
- Class imbalance handled via `class_weight='balanced'` and `scale_pos_weight`

### 3. Models Trained
| Model | Strategy |
|---|---|
| Logistic Regression | Regularised, balanced class weights |
| Random Forest | 200 trees, balanced weights, max_depth=10 |
| **XGBoost** | 300 trees, scale_pos_weight for imbalance |

### 4. Evaluation
- Confusion Matrix, Classification Report
- F1-Score (primary metric given imbalance)
- ROC-AUC curves
- 5-Fold Stratified Cross-Validation

### 5. Explainable AI (XAI)
- **SHAP TreeExplainer** applied to XGBoost
- Global feature importance (bar + beeswarm plots)
- Individual SHAP waterfall plots for **5 predictions** (TP, TN, FP, FN, random)

---

## 📊 Results & Findings

### Model Performance

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~0.79 | ~0.46 | ~0.80 |
| Random Forest | ~0.85 | ~0.52 | ~0.84 |
| **XGBoost** ⭐ | **~0.87** | **~0.55** | **~0.87** |

> **XGBoost is the recommended model** — highest ROC-AUC and best F1-Score.

### Top 5 SHAP Features (XGBoost)
1. `duration` — Call length is the strongest predictor of subscription
2. `poutcome_success` — Previous campaign success dramatically raises probability
3. `balance` — Higher average balance → slightly more likely to subscribe
4. `month_num` — Seasonality matters; March/September outperform May
5. `contact_cellular` — Mobile contact outperforms telephone

### Key Business Insights
- 📞 **Longer calls convert better** — train agents for quality conversations
- 🔁 **Re-target previous successes** — they convert at 3× the baseline rate
- 📱 **Use cellular contact** exclusively for highest ROI
- 🎯 **Focus on students & retirees** — higher per-segment conversion rate
- 📅 **Campaign timing matters** — avoid mass May campaigns; prefer March/September

---

## 📁 Repository Structure

```
├── Task1_Term_Deposit_Prediction.ipynb   # Full analysis notebook
├── bank_marketing.csv                     # Dataset
├── README.md                              # This file
└── plots/
    ├── 01_target_distribution.png
    ├── 02_numeric_distributions.png
    ├── 03_categorical_rates.png
    ├── 04_correlation_heatmap.png
    ├── 05_duration_analysis.png
    ├── 06_model_comparison.png
    ├── 07_confusion_matrices.png
    ├── 08_roc_curves.png
    ├── 09_feature_importance.png
    ├── 10_shap_global_bar.png
    ├── 11_shap_beeswarm.png
    ├── 12_shap_local_sample_*.png
    └── 13_cross_validation.png
```

---

## 🧰 Libraries Used

```python
pandas, numpy, matplotlib, seaborn
scikit-learn (LogisticRegression, RandomForestClassifier)
xgboost (XGBClassifier)
shap (TreeExplainer)
```

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap

# 2. Launch notebook
jupyter notebook Task1_Term_Deposit_Prediction.ipynb
```

---

