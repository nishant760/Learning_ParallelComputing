# train_model.py
# ------------------------------------------------------------------
# Project: AI-Assisted OpenMP Scheduling for Parallel Numerical Computation
# Description: Trains a Random Forest Regressor on empirical OpenMP benchmark dataset.
#              Evaluates generalization via GroupKFold (grouped by workload size N)
#              and a dedicated holdout evaluation on unseen N = 75M workload.
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
from sklearn.model_selection import GroupKFold

def train_and_evaluate():
    csv_path = os.path.join("data", "performance.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Run 'make run-benchmark' first.")

    df = pd.read_csv(csv_path)
    print("Dataset Loaded Successfully.")
    print(f"Total Benchmark Rows: {len(df)}")
    print("Columns:", list(df.columns))

    feature_cols = ['N', 'threads', 'schedule', 'chunk']
    target_col = 'speedup'

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(categories=[['static', 'dynamic', 'guided']], handle_unknown='ignore'), ['schedule']),
            ('num', 'passthrough', ['N', 'threads', 'chunk'])
        ]
    )

    def create_model():
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42))
        ])

    # 1. GroupKFold Workload-Grouped Cross-Validation (Phase 5)
    print("\n-------------------------------------------------------------------------")
    print(" 1. WORKLOAD-GROUPED CROSS-VALIDATION (GroupKFold by N)")
    print("-------------------------------------------------------------------------")
    gkf = GroupKFold(n_splits=min(5, len(df['N'].unique())))
    
    cv_r2, cv_mae, cv_rmse, cv_mape = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df[target_col], groups=df['N'])):
        train_fold = df.iloc[train_idx]
        val_fold = df.iloc[val_idx]
        held_out_workloads = val_fold['N'].unique()

        model_cv = create_model()
        model_cv.fit(train_fold[feature_cols], train_fold[target_col])
        y_val_pred = model_cv.predict(val_fold[feature_cols])
        y_val_true = val_fold[target_col]

        r2 = r2_score(y_val_true, y_val_pred)
        mae = mean_absolute_error(y_val_true, y_val_pred)
        rmse = np.sqrt(mean_squared_error(y_val_true, y_val_pred))
        mape = np.mean(np.abs((y_val_true - y_val_pred) / y_val_true)) * 100

        cv_r2.append(r2)
        cv_mae.append(mae)
        cv_rmse.append(rmse)
        cv_mape.append(mape)

        print(f" Fold {fold + 1} | Held-out N = {held_out_workloads} | R²: {r2:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}%")

    print("\n Workload GroupKFold Average Metrics:")
    print(f"   Mean R² Score          : {np.mean(cv_r2):.4f}")
    print(f"   Mean Absolute Error (MAE): {np.mean(cv_mae):.4f}")
    print(f"   Root Mean Sq Error (RMSE): {np.mean(cv_rmse):.4f}")
    print(f"   Mean Abs Pct Error (MAPE): {np.mean(cv_mape):.2f}%")

    # 2. Dedicated Unseen Workload Holdout Test (N = 75,000,000)
    unseen_workload = 75000000
    print("\n-------------------------------------------------------------------------")
    print(f" 2. DEDICATED UNSEEN WORKLOAD HOLDOUT EVALUATION (N = {unseen_workload:,})")
    print("-------------------------------------------------------------------------")

    if unseen_workload in df['N'].values:
        train_df = df[df['N'] != unseen_workload].copy()
        test_df = df[df['N'] == unseen_workload].copy()
    else:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    model_holdout = create_model()
    model_holdout.fit(train_df[feature_cols], train_df[target_col])

    y_test_pred = model_holdout.predict(test_df[feature_cols])
    y_test_true = test_df[target_col]

    holdout_r2 = r2_score(y_test_true, y_test_pred)
    holdout_mae = mean_absolute_error(y_test_true, y_test_pred)
    holdout_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    holdout_mape = np.mean(np.abs((y_test_true - y_test_pred) / y_test_true)) * 100

    print(f" Unseen Workload N         : {unseen_workload:,}")
    print(f" Test Sample Size          : {len(test_df)} rows")
    print(f" R² Determination Score    : {holdout_r2:.4f}")
    print(f" Mean Absolute Error (MAE)  : {holdout_mae:.4f}")
    print(f" Root Mean Sq Error (RMSE)  : {holdout_rmse:.4f}")
    print(f" Mean Abs Pct Error (MAPE)  : {holdout_mape:.2f}%")

    # 3. Train Final Saved Model on Full Dataset
    final_model = create_model()
    final_model.fit(df[feature_cols], df[target_col])

    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", "speedup_model.pkl")
    joblib.dump(final_model, model_path)
    print("\n=========================================================================")
    print(f" Final Random Forest model trained and saved to: {model_path}")
    print("=========================================================================\n")

if __name__ == '__main__':
    train_and_evaluate()
