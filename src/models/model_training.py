import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def train_model_random_forest_regressor(x, y, n_splits=5):
    # Enhanced RandomForest parameters
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [4, 6, 8, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False],
        'max_samples': [0.7, 0.8, 0.9],
        'criterion': ['squared_error', 'absolute_error', 'poisson'],
        'min_impurity_decrease': [0.0, 0.1, 0.2]
    }

    # Initialize base model with better defaults
    base_model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        warm_start=True,
        oob_score=True
    )
    
    # Initialize TimeSeriesSplit with more sophisticated settings
    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        gap=1,
        test_size=6  # 6 months test size
    )
    
    # Initialize enhanced GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring={
            'rmse': 'neg_root_mean_squared_error',
            'mae': 'neg_mean_absolute_error',
            'r2': 'r2',
            'mape': 'neg_mean_absolute_percentage_error'
        },
        refit='rmse',
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    
    # Fit GridSearchCV
    grid_search.fit(x, y)
    
    print("\n🎯 Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  • {param}: {value}")
    
    print("\n📊 Cross-validation Scores:")
    print(f"  • Best RMSE: {abs(grid_search.best_score_):.2f}")
    
    # Model configuration details
    print("\n🌳 Random Forest Model Configuration:")
    print("  • Number of features:", x.shape[1])
    print("  • Number of samples:", x.shape[0])
    print("  • Cross-validation folds:", n_splits)
    print("  • Test size:", "6 months")
    print("  • Gap size:", "1 month")
    
    # Add OOB score if available
    if grid_search.best_estimator_.oob_score_:
        print(f"📊 Out-of-Bag Score: {grid_search.best_estimator_.oob_score_:.3f}")
    
    # Get feature importance
    importances = grid_search.best_estimator_.feature_importances_
    std = np.std([tree.feature_importances_ for tree in 
                  grid_search.best_estimator_.estimators_], axis=0)
    
    feature_importance = pd.DataFrame({
        'feature': x.columns,
        'importance': importances,
        'std': std
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 Top 10 Most Important Features:")
    for _, row in feature_importance.head(10).iterrows():
        print(f"  • {row['feature']}: {row['importance']:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance.head(10)['feature'], feature_importance.head(10)['importance'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 10 Feature Importance')
    plt.tight_layout()
    plt.savefig('reports/feature_importance.png')
    plt.close()
    
    # Cross-validation with multiple metrics
    cv_scores = cross_validate(
        grid_search.best_estimator_,
        x, y,
        cv=tscv,
        scoring=['neg_root_mean_squared_error', 'r2', 'neg_mean_absolute_error'],
        return_train_score=True
    )
    
    print("\n📈 Model Performance Metrics:")
    print(f"  • Test RMSE: {abs(cv_scores['test_neg_root_mean_squared_error'].mean()):.2f} (±{cv_scores['test_neg_root_mean_squared_error'].std() * 2:.2f})")
    print(f"  • Test MAE: {abs(cv_scores['test_neg_mean_absolute_error'].mean()):.2f} (±{cv_scores['test_neg_mean_absolute_error'].std() * 2:.2f})")
    print(f"  • Test R²: {cv_scores['test_r2'].mean():.3f} (±{cv_scores['test_r2'].std() * 2:.3f})")
    
    return grid_search.best_estimator_

