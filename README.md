# Sales Prediction using Python 📈

> **End-to-End Machine Learning Project & Interactive Marketing ROI Dashboard**

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

## 📌 Project Overview
Predict unit sales based on advertising expenditures across **TV**, **Radio**, and **Newspaper** channels, target audience segments, and primary delivery platforms. This repository includes complete data cleaning, exploratory data analysis (EDA), multi-model benchmarking, marginal ROI calculation, and a budget optimizer powered by SciPy.

---

## 🚀 Key Features

- **Data Cleaning & Engineering:** Standard scaling for numerical spend and one-hot encoding for categorical variables (`Target_Segment`, `Platform`).
- **Multi-Model Regression Suite:** Evaluates Linear Regression, Ridge, Lasso, Decision Trees, Random Forests, and Gradient Boosting.
- **Top Model Accuracy:** Linear Regression achieved **$R^2 = 94.93\%$** with **$\text{MAE} = 0.880\text{k units}$**.
- **Channel ROI Analysis:** Radio yields highest return (+$214.18 per $1k spend) vs TV (+$46.37) and Newspaper (+$6.32).
- **Budget Optimizer:** Constrained SciPy solver to maximize top-line sales given a budget limit.
- **Interactive Web App Dashboard:** Real-time budget sliders, live sales predictor, and Chart.js analytics at `http://localhost:8000`.

---

## 🛠️ Repository Structure

```text
.
├── Advertising.csv              # Dataset (200 records)
├── sales_prediction.py          # Complete ML & Optimization pipeline
├── build_notebook.py            # Script to generate Jupyter Notebook
├── sales_prediction.ipynb       # Executable Jupyter Notebook
├── server.py                    # REST API backend server
├── index.html                   # Dashboard frontend HTML
├── styles.css                   # Glassmorphism dark mode CSS
├── app.js                       # Frontend interaction & Chart.js logic
├── models/                      # Saved trained models & metrics JSON
└── plots/                       # Generated EDA & performance charts
```

---

## ⚙️ How to Run Locally

### 1. Install Dependencies
Ensure Python 3.8+ is installed. Install required packages:
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn joblib
```

### 2. Run the Machine Learning Pipeline
```bash
python sales_prediction.py
```

### 3. Launch the Interactive Dashboard
```bash
python server.py
```
Open your browser and visit: **`http://localhost:8000`**

---

## 📊 Model Benchmark Results

| Model Name | Test $R^2$ Score | Adj $R^2$ | 5-Fold CV $R^2$ | MAE ($1000s) | RMSE ($1000s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | **0.9493** | **0.9418** | **0.9425** | **0.880** | **1.138** |
| **Ridge Regression** | 0.9484 | 0.9408 | 0.9415 | 0.888 | 1.148 |
| **Lasso Regression** | 0.9252 | 0.9141 | 0.9243 | 1.139 | 1.382 |
| **Gradient Boosting** | 0.8959 | 0.8806 | 0.8683 | 1.290 | 1.630 |
| **Random Forest** | 0.8812 | 0.8638 | 0.8650 | 1.457 | 1.741 |
| **Decision Tree** | 0.7174 | 0.6759 | 0.7823 | 2.042 | 2.686 |

---

## 📄 License
This project is open source under the MIT License.
