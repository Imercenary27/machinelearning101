"""
MODEL COMPARISON: Linear vs Ridge vs Lasso
==========================================
This script compares all three regression techniques on the same dataset
to help you understand their differences and choose the right model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load data
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split data (same random_state as individual files)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train all three models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (L2)': Ridge(alpha=1.0),
    'Lasso (L1)': Lasso(alpha=0.01, max_iter=10000)
}

results = []

print("=" * 80)
print("MODEL COMPARISON: Linear vs Ridge vs Lasso")
print("=" * 80)

for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Evaluate
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Count non-zero coefficients
    n_features = np.sum(model.coef_ != 0) if hasattr(model, 'coef_') else len(X.columns)
    
    results.append({
        'Model': name,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Test RMSE': test_rmse,
        'Test MAE': test_mae,
        'Features Used': n_features,
        'Overfitting': train_r2 - test_r2
    })
    
    print(f"\n{name}:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Features Used: {n_features}/{len(X.columns)}")
    print(f"  Overfitting Gap: {train_r2 - test_r2:.4f}")

# Create comparison dataframe
df_results = pd.DataFrame(results)

print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(df_results.to_string(index=False))

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model Comparison: Linear vs Ridge vs Lasso', fontsize=16, fontweight='bold')

# Plot 1: R² Comparison
metrics = ['Train R²', 'Test R²']
x = np.arange(len(df_results))
width = 0.35

axes[0].bar(x - width/2, df_results['Train R²'], width, label='Train R²', alpha=0.8)
axes[0].bar(x + width/2, df_results['Test R²'], width, label='Test R²', alpha=0.8)
axes[0].set_ylabel('R² Score')
axes[0].set_title('R² Score Comparison')
axes[0].set_xticks(x)
axes[0].set_xticklabels(df_results['Model'])
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: RMSE Comparison
axes[1].bar(df_results['Model'], df_results['Test RMSE'], color=['blue', 'green', 'orange'])
axes[1].set_ylabel('RMSE')
axes[1].set_title('Test RMSE Comparison (Lower is Better)')
axes[1].grid(True, alpha=0.3)

# Plot 3: Feature Usage
axes[2].bar(df_results['Model'], df_results['Features Used'], color=['blue', 'green', 'orange'])
axes[2].set_ylabel('Number of Features')
axes[2].set_title('Features Used by Each Model')
axes[2].axhline(y=len(X.columns), color='r', linestyle='--', label='Total Features')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')

print("\n✓ Comparison visualization saved as 'model_comparison.png'")
print("=" * 80)
