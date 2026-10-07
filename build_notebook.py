import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Sales Prediction using Python\n",
            "### Data Science End-to-End Project: Machine Learning Modeling, EDA & ROI Optimization\n",
            "\n",
            "**Author:** Data Science Team  \n",
            "**Dataset:** `Advertising.csv`  \n",
            "**Objective:** Predict sales based on advertising expenditures (TV, Radio, Newspaper), target customer segment, and dominant marketing platform, and optimize future marketing budget allocation.\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "--- \n",
            "## 1. Environment & Library Setup"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import json\n",
            "import joblib\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "from sklearn.model_selection import train_test_split, cross_val_score\n",
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
            "from sklearn.compose import ColumnTransformer\n",
            "from sklearn.pipeline import Pipeline\n",
            "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
            "from sklearn.linear_model import LinearRegression, Ridge, Lasso\n",
            "from sklearn.tree import DecisionTreeRegressor\n",
            "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n",
            "from scipy.optimize import minimize\n",
            "\n",
            "# Formatting & Aesthetics\n",
            "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "--- \n",
            "## 2. Data Preparation & Exploration (EDA)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load Dataset\n",
            "df = pd.read_csv('Advertising.csv')\n",
            "print(f\"Dataset Shape: {df.shape[0]} Rows, {df.shape[1]} Columns\")\n",
            "df.head()\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Summary Statistics & Data Hygiene Check\n",
            "print('=== Missing Values & Data Types ===')\n",
            "print(df.isnull().sum())\n",
            "\n",
            "print('=== Descriptive Statistics ===')\n",
            "df.describe().T\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Exploratory Data Plots\n",
            "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
            "sns.regplot(x='TV', y='Sales', data=df, ax=axes[0], color='#1f77b4')\n",
            "axes[0].set_title('TV Spend vs Sales')\n",
            "\n",
            "sns.regplot(x='Radio', y='Sales', data=df, ax=axes[1], color='#ff7f0e')\n",
            "axes[1].set_title('Radio Spend vs Sales')\n",
            "\n",
            "sns.regplot(x='Newspaper', y='Sales', data=df, ax=axes[2], color='#2ca02c')\n",
            "axes[2].set_title('Newspaper Spend vs Sales')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "--- \n",
            "## 3. Data Transformation & Preprocessing Pipeline"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Separate Features & Target Variable\n",
            "numeric_features = ['TV', 'Radio', 'Newspaper']\n",
            "categorical_features = ['Target_Segment', 'Platform']\n",
            "\n",
            "X = df[numeric_features + categorical_features]\n",
            "y = df['Sales']\n",
            "\n",
            "# Construct Transformer Pipeline\n",
            "preprocessor = ColumnTransformer(\n",
            "    transformers=[\n",
            "        ('num', StandardScaler(), numeric_features),\n",
            "        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)\n",
            "    ]\n",
            ")\n",
            "\n",
            "# 80/20 Train-Test Split\n",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
            "print(f\"Train observations: {X_train.shape[0]}, Test observations: {X_test.shape[0]}\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "--- \n",
            "## 4. Machine Learning Model Training & Evaluation"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Model Evaluation Suite\n",
            "models = {\n",
            "    'Linear Regression': LinearRegression(),\n",
            "    'Ridge Regression': Ridge(alpha=1.0),\n",
            "    'Lasso Regression': Lasso(alpha=0.1),\n",
            "    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),\n",
            "    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),\n",
            "    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)\n",
            "}\n",
            "\n",
            "results = []\n",
            "for name, model in models.items():\n",
            "    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])\n",
            "    pipe.fit(X_train, y_train)\n",
            "    y_pred = pipe.predict(X_test)\n",
            "    \n",
            "    cv_r2 = np.mean(cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2'))\n",
            "    r2 = r2_score(y_test, y_pred)\n",
            "    mae = mean_absolute_error(y_test, y_pred)\n",
            "    rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n",
            "    \n",
            "    results.append({\n",
            "        'Model': name,\n",
            "        'R2 Score': r2,\n",
            "        '5-Fold CV R2': cv_r2,\n",
            "        'MAE': mae,\n",
            "        'RMSE': rmse\n",
            "    })\n",
            "\n",
            "results_df = pd.DataFrame(results).sort_values(by='R2 Score', ascending=False)\n",
            "results_df\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "--- \n",
            "## 5. Channel Impact & Feature Importance Analysis"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Unscaled Marginal Coef Analysis\n",
            "lr = LinearRegression()\n",
            "lr.fit(X_train[['TV', 'Radio', 'Newspaper']], y_train)\n",
            "\n",
            "print('=== Marginal Sales Revenue per $1,000 Spend ===')\n",
            "print(f\"TV Spend:        +${lr.coef_[0]*1000:.2f} per $1k\")\n",
            "print(f\"Radio Spend:     +${lr.coef_[1]*1000:.2f} per $1k\")\n",
            "print(f\"Newspaper Spend: +${lr.coef_[2]*1000:.2f} per $1k\")\n",
            "print(f\"Baseline Sales:  ${lr.intercept_:.2f}k\")\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "--- \n",
            "## 6. Constrained Budget Optimization & Business Actionable Insights"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Scipy Constrained Budget Allocator\n",
            "best_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])\n",
            "best_pipe.fit(X_train, y_train)\n",
            "\n",
            "def optimize_budget(budget=200.0):\n",
            "    def loss(weights):\n",
            "        df_in = pd.DataFrame([{\n",
            "            'TV': weights[0],\n",
            "            'Radio': weights[1],\n",
            "            'Newspaper': weights[2],\n",
            "            'Target_Segment': 'Mass Market',\n",
            "            'Platform': 'Omnichannel Digital'\n",
            "        }])\n",
            "        return -best_pipe.predict(df_in)[0]\n",
            "    \n",
            "    res = minimize(loss, [budget/3]*3, bounds=[(0, budget)]*3, constraints={'type': 'eq', 'fun': lambda w: budget - sum(w)})\n",
            "    return res.x, -res.fun\n",
            "\n",
            "allocation, max_sales = optimize_budget(200.0)\n",
            "print('Optimal Allocation for $200k Total Spend:')\n",
            "print(f\"TV: ${allocation[0]:.2f}k | Radio: ${allocation[1]:.2f}k | Newspaper: ${allocation[2]:.2f}k\")\n",
            "print(f\"Max Predicted Sales: ${max_sales:.2f}k units\")\n"
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('sales_prediction.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("Generated sales_prediction.ipynb successfully.")
