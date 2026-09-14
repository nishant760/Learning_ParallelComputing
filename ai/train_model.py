# train_model.py
# ------------------------------------------------------------------
# Project: AI-Assisted Parallel Numerical Computation & Scheduling
# Description: Trains a Random Forest Regressor to predict OpenMP speedup
#              from empirical benchmark parameters (N, threads, schedule, chunk).
#              Evaluates performance on unseen workload sizes (e.g. N = 75M).
# ------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_and_evaluate():
    csv_path = os.path.join("data", "performance.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please run benchmark first.")

    df = pd.read_csv(csv_path)
    print("Dataset Loaded Successfully.")
    print(f"Total Rows: {len(df)}")
    print(df.head())

    # Separate unseen workload (N = 75,000,000) for zero-shot testing
    unseen_workload = 75000000
    
    # If 75M is in dataset, split training and unseen testing set
    if unseen_workload in df['N'].values:
        train_df = df[df['N'] != unseen_workload].copy()
        test_df = df[df['N'] == unseen_workload].copy()
        print(f"\nWorkload Split: Training on sizes {sorted(train_df['N'].unique())}")
        print(f"Testing on unseen workload size N = {unseen_workload}")
    else:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        print("\nStandard train/test split applied.")

    feature_cols = ['N', 'threads', 'schedule', 'chunk']
    target_col = 'speedup'

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(categories=[['static', 'dynamic', 'guided']], handle_unknown='ignore'), ['schedule']),
            ('num', 'passthrough', ['N', 'threads', 'chunk'])
        ]
    )

    # Random Forest Regressor Pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42))
    ])

    model.fit(X_train, y_train)

    # Evaluation on unseen workload
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print("\n=========================================================================")
    print(" AI MODEL TRAINING & UNSEEN WORKLOAD EVALUATION RESULTS")
    print("=========================================================================")
    print(f" Unseen Workload Size (N) : {unseen_workload:,}")
    print(f" Test Samples Count        : {len(test_df)}")
    print(f" Mean Absolute Error (MAE) : {mae:.4f}")
    print(f" Root Mean Sq Error (RMSE) : {rmse:.4f}")
    print(f" Mean Abs Pct Error (MAPE) : {mape:.2f}%")
    print(f" R² Determination Score   : {r2:.4f}")
    print("=========================================================================")

    # Print sample comparison for unseen workload
    sample_res = test_df.copy()
    sample_res['predicted_speedup'] = y_pred
    sample_res['abs_error'] = np.abs(sample_res['speedup'] - sample_res['predicted_speedup'])
    print("\nSample Predictions on Unseen Workload (N = 75M):")
    print(sample_res[['threads', 'schedule', 'chunk', 'speedup', 'predicted_speedup', 'abs_error']].head(10).to_string(index=False))

    # Save model pipeline
    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", "speedup_model.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel pipeline saved successfully to: {model_path}\n")

if __name__ == '__main__':
    train_and_evaluate()
