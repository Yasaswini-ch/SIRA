import os
import joblib
import pandas as pd
import numpy as np
import datetime
import json
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_CHARTS = True
except ImportError:
    HAS_CHARTS = False

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, KFold

from preprocess import perform_eda, preprocess

def train_and_evaluate(models_dir='models', charts_dir='static/charts'):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)
    
    print("=" * 70)
    print("1. LOADING AND PREPROCESSING DATA")
    print("=" * 70)
    
    train_raw, test_raw = perform_eda()
    if train_raw is None:
        print("Data could not be loaded. Aborting training.")
        return
        
    prep_result = preprocess(train_raw, test_raw, models_dir=models_dir)
    
    X_train = prep_result['X_train']
    y_train = prep_result['y_train']
    feature_names = prep_result['feature_names']
    
    print("\n" + "=" * 70)
    print("2. INITIALIZING MODELS")
    print("=" * 70)
    
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=50,
        random_state=42, 
        n_jobs=-1
    )
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": rf
    }
    
    if HAS_XGB:
        xgb = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        models["XGBoost Regressor"] = xgb
        
        # Create Ensemble: Voting Regressor
        models["Ensemble (RF + XGB)"] = VotingRegressor(
            estimators=[('rf', rf), ('xgb', xgb)],
            n_jobs=-1
        )
    
    results = []
    
    print("Performing 5-fold Cross Validation (RMSE)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"--> Training {name}...")
        
        # Cross Validation for RMSE out-of-fold generalization
        cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
        rmse_mean = -cv_scores.mean()
        rmse_std = cv_scores.std()
        
        # Fit on whole train set for full train performance metrics
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        
        # Compute Training Metrics
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        mae = mean_absolute_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)
        mape = mean_absolute_percentage_error(y_train, y_pred)
        
        results.append({
            'Model': name,
            'CV RMSE Mean': rmse_mean,
            'CV RMSE Std': rmse_std,
            'Train RMSE': rmse,
            'Train MAE': mae,
            'Train R2': r2,
            'Train MAPE': mape
        })
        
    results_df = pd.DataFrame(results).round(4)
    
    print("\n" + "=" * 70)
    print("3. MODEL COMPARISON RESULTS")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    # Determine the winner based on CV generalization performance
    best_idx = results_df['CV RMSE Mean'].idxmin()
    best_model_name = results_df.loc[best_idx, 'Model']
    best_model_instance = models[best_model_name]
    
    print("\n" + "=" * 70)
    print(f"WINNING MODEL: {best_model_name}")
    print("=" * 70)
    print(f"Reason: '{best_model_name}' achieved the lowest Cross-Validated Root Mean Squared Error "
          f"({results_df.loc[best_idx, 'CV RMSE Mean']} ± {results_df.loc[best_idx, 'CV RMSE Std']}). "
          f"This indicates it generalizes best to unseen data and avoids major overfitting compared to other models.")
          
    # Save the best model locally
    best_model_path = os.path.join(models_dir, 'demand_model.pkl')
    joblib.dump(best_model_instance, best_model_path)
    print(f"\nBest model effectively serialized and saved to: {best_model_path}")
    
    # Export Metadata for App Landing Page
    metadata = {
        "model_name": best_model_name,
        "cv_rmse": results_df.loc[best_idx, 'CV RMSE Mean'],
        "train_r2": results_df.loc[best_idx, 'Train R2'],
        "last_trained": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "top_features": []
    }
    
    # Generate and save Feature Importances Plot
    print("\nGenerating feature importances visualization...")
    importances = None
    if hasattr(best_model_instance, 'feature_importances_'):
        importances = best_model_instance.feature_importances_
    elif hasattr(best_model_instance, 'coef_'):
        importances = np.abs(best_model_instance.coef_)
    elif isinstance(best_model_instance, VotingRegressor):
        # Average importances from base estimators
        base_importances = []
        for name, est in best_model_instance.estimators_:
            if hasattr(est, 'feature_importances_'):
                base_importances.append(est.feature_importances_)
            elif hasattr(est, 'coef_'):
                base_importances.append(np.abs(est.coef_))
        
        if base_importances:
            importances = np.mean(base_importances, axis=0)
        
    if importances is not None:
        feat_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Save top 5 features to metadata
        metadata["top_features"] = feat_imp_df.head(5).to_dict(orient='records')
        
        # Focusing on Top 20 for readability in plot
        plot_df = feat_imp_df.head(20)
        
        if HAS_CHARTS:
            sns.set_style("whitegrid")
            plt.figure(figsize=(11, 8))
            sns.barplot(data=plot_df, x='Importance', y='Feature', hue='Feature', legend=False, palette='Oranges_r')
            plt.title(f'Top 20 Critical Demand Drivers ({best_model_name})', fontsize=16)
            plt.xlabel('Relative Importance (Absolute Coefficient / IG)')
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, 'feature_importance.png'), dpi=150)
            plt.close()
            print("Feature importance chart rendered and saved to static/charts/feature_importance.png")
        else:
            print("Feature importance chart skipped due to missing optional plotting libraries.")
    
    # Save metadata JSON
    with open(os.path.join(models_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Model metadata exported to: {os.path.join(models_dir, 'model_metadata.json')}")

if __name__ == "__main__":
    train_and_evaluate()
