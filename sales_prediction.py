import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.optimize import minimize

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

def run_sales_prediction_pipeline():
    print("=" * 70)
    print("      SALES PREDICTION MACHINE LEARNING PIPELINE & ANALYSIS")
    print("=" * 70)

    # ---------------------------------------------------------
    # STEP 1: DATA LOADING & EXPLORATION
    # ---------------------------------------------------------
    df = pd.read_csv('Advertising.csv')
    print("\n[Step 1] Dataset Loaded Successfully.")
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nFirst 5 Rows:")
    print(df.head())
    
    print("\nData Types & Missing Values:")
    info_df = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Unique Values': df.nunique()
    })
    print(info_df)

    print("\nSummary Statistics (Numerical):")
    print(df.describe().T)

    # ---------------------------------------------------------
    # STEP 2: EXPLORATORY DATA ANALYSIS & VISUALIZATION
    # ---------------------------------------------------------
    print("\n[Step 2] Generating EDA Visualizations...")

    # Correlation Matrix for Numeric Features
    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='Blues', fmt='.3f', linewidths=0.5)
    plt.title('Correlation Matrix of Numeric Features & Sales', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png', dpi=300)
    plt.close()

    # Scatter Plots: Spend vs Sales
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    channels = ['TV', 'Radio', 'Newspaper']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, ch in enumerate(channels):
        sns.regplot(x=ch, y='Sales', data=df, ax=axes[i], color=colors[i],
                    scatter_kws={'alpha': 0.6}, line_kws={'color': 'darkred', 'linewidth': 2})
        axes[i].set_title(f'{ch} Spend vs Sales', fontsize=12)
        axes[i].set_xlabel(f'{ch} Spend ($1000s)')
        axes[i].set_ylabel('Sales (1000s of units)')
    plt.tight_layout()
    plt.savefig('plots/spend_vs_sales.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # STEP 3: DATA PREPARATION & FEATURE ENGINEERING
    # ---------------------------------------------------------
    print("\n[Step 3] Data Cleaning & Preprocessing Pipeline...")

    # Check for duplicates & handle missing values
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Removed {duplicates} duplicate rows.")

    # Define X and y
    numeric_features = ['TV', 'Radio', 'Newspaper']
    categorical_features = ['Target_Segment', 'Platform']
    
    X = df[numeric_features + categorical_features]
    y = df['Sales']

    # Create Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )

    # Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train Set Size: {X_train.shape[0]} | Test Set Size: {X_test.shape[0]}")

    # ---------------------------------------------------------
    # STEP 4: MODELING & EVALUATION
    # ---------------------------------------------------------
    print("\n[Step 4] Training & Evaluating Machine Learning Models...")

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    }

    results = []

    best_model_name = None
    best_r2 = -float('inf')
    best_pipeline = None

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        # Train model
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)

        # Cross-validation R2 score
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        cv_r2_mean = np.mean(cv_scores)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        n = len(y_test)
        p = X_train.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        results.append({
            'Model': name,
            'R2 Score': r2,
            'Adj R2 Score': adj_r2,
            'CV R2 (5-fold)': cv_r2_mean,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        })

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline

    results_df = pd.DataFrame(results).sort_values(by='R2 Score', ascending=False).reset_index(drop=True)
    print("\nModel Comparison Table:")
    print(results_df.to_string(index=False))

    # Save Best Model
    joblib.dump(best_pipeline, 'models/best_sales_model.pkl')
    print(f"\nBest Model selected: {best_model_name} (R2 = {best_r2:.4f}) saved to 'models/best_sales_model.pkl'.")

    # Plot Actual vs Predicted for Best Model
    y_best_pred = best_pipeline.predict(X_test)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_best_pred, alpha=0.7, color='#1f77b4', edgecolors='k')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Ideal Fit Line')
    plt.xlabel('Actual Sales ($1000s)')
    plt.ylabel('Predicted Sales ($1000s)')
    plt.title(f'Actual vs Predicted Sales ({best_model_name})', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/actual_vs_predicted.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # STEP 5: IMPACT ANALYSIS & FEATURE IMPORTANCE
    # ---------------------------------------------------------
    print("\n[Step 5] Analyzing Channel Impact & Feature Importance...")

    # Extract feature names after preprocessing
    ohe_cols = list(best_pipeline.named_steps['preprocessor']
                    .named_transformers_['cat']
                    .get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + ohe_cols

    regressor = best_pipeline.named_steps['regressor']
    
    if hasattr(regressor, 'feature_importances_'):
        importances = regressor.feature_importances_
        imp_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
        imp_df = imp_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        print("\nFeature Importances (Tree-based model):")
        print(imp_df.to_string(index=False))

        plt.figure(figsize=(10, 5))
        sns.barplot(x='Importance', y='Feature', data=imp_df, palette='Blues_r')
        plt.title(f'Feature Importance Breakdown ({best_model_name})', fontsize=14)
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png', dpi=300)
        plt.close()

    # Also fit Linear Regression specifically to extract interpretable coefficients / ROI per channel
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_model = lr_pipeline.named_steps['regressor']
    scaler = lr_pipeline.named_steps['preprocessor'].named_transformers_['num']
    
    # Unscaled coefficients for TV, Radio, Newspaper
    lr_unscaled = LinearRegression()
    lr_unscaled.fit(X_train[['TV', 'Radio', 'Newspaper']], y_train)
    
    print("\nChannel Sensitivity / Marginal Sales Impact per $1,000 Spend (Linear Model):")
    print(f" - TV Ad Spend:        +${lr_unscaled.coef_[0]*1000:.2f} sales revenue per $1,000 spent")
    print(f" - Radio Ad Spend:     +${lr_unscaled.coef_[1]*1000:.2f} sales revenue per $1,000 spent")
    print(f" - Newspaper Ad Spend: +${lr_unscaled.coef_[2]*1000:.2f} sales revenue per $1,000 spent")
    print(f" - Base Sales (No Spend): {lr_unscaled.intercept_:.2f} ($1000s)")

    # ---------------------------------------------------------
    # STEP 6: BUSINESS MARKETING OPTIMIZER & ROI SIMULATION
    # ---------------------------------------------------------
    print("\n[Step 6] Running Marketing Budget Optimizer...")

    def optimize_budget(total_budget=200.0, default_seg='Mass Market', default_plat='Omnichannel Digital'):
        # Objective: Maximize predicted sales -> Minimize -sales
        def objective(weights):
            tv_val, radio_val, news_val = weights
            input_df = pd.DataFrame([{
                'TV': tv_val,
                'Radio': radio_val,
                'Newspaper': news_val,
                'Target_Segment': default_seg,
                'Platform': default_plat
            }])
            pred = best_pipeline.predict(input_df)[0]
            return -pred

        # Constraints & Bounds
        constraints = ({'type': 'eq', 'fun': lambda w: total_budget - np.sum(w)})
        bounds = [(0, total_budget) for _ in range(3)]
        initial_guess = [total_budget / 3.0] * 3

        opt_res = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        opt_tv, opt_radio, opt_news = opt_res.x
        max_predicted_sales = -opt_res.fun

        # Equal allocation baseline
        baseline_df = pd.DataFrame([{
            'TV': total_budget / 3.0,
            'Radio': total_budget / 3.0,
            'Newspaper': total_budget / 3.0,
            'Target_Segment': default_seg,
            'Platform': default_plat
        }])
        baseline_sales = best_pipeline.predict(baseline_df)[0]

        uplift = ((max_predicted_sales - baseline_sales) / baseline_sales) * 100

        print(f"\nOptimal Budget Allocation for ${total_budget:,.2f}k Total Spend:")
        print(f" - TV Spend:        ${opt_tv:.2f}k ({opt_tv/total_budget*100:.1f}%)")
        print(f" - Radio Spend:     ${opt_radio:.2f}k ({opt_radio/total_budget*100:.1f}%)")
        print(f" - Newspaper Spend: ${opt_news:.2f}k ({opt_news/total_budget*100:.1f}%)")
        print(f" - Optimized Forecasted Sales: ${max_predicted_sales:.2f}k units")
        print(f" - Baseline (Equal Split) Sales: ${baseline_sales:.2f}k units")
        print(f" - Revenue Uplift from Optimization: +{uplift:.2f}%")

        return {
            'budget': total_budget,
            'opt_tv': round(opt_tv, 2),
            'opt_radio': round(opt_radio, 2),
            'opt_news': round(opt_news, 2),
            'max_sales': round(max_predicted_sales, 2),
            'baseline_sales': round(baseline_sales, 2),
            'uplift_pct': round(uplift, 2)
        }

    opt_summary = optimize_budget(total_budget=200.0)

    # Save summary metrics to JSON
    summary_data = {
        'best_model': best_model_name,
        'results': results_df.to_dict(orient='records'),
        'optimization': opt_summary,
        'marginal_impact': {
            'TV_coef': round(lr_unscaled.coef_[0], 4),
            'Radio_coef': round(lr_unscaled.coef_[1], 4),
            'Newspaper_coef': round(lr_unscaled.coef_[2], 4),
            'Intercept': round(lr_unscaled.intercept_, 4)
        }
    }

    with open('models/pipeline_results.json', 'w') as f:
        json.dump(summary_data, f, indent=4)

    print("\n" + "=" * 70)
    print("      PIPELINE EXECUTION COMPLETE & ALL ARTIFACTS GENERATED")
    print("=" * 70)

if __name__ == '__main__':
    run_sales_prediction_pipeline()
