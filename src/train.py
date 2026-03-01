"""
train.py
--------
EduAnalytics training pipeline.

Trains three models on the student performance dataset:
  1. Linear Regression
  2. Random Forest Regressor
  3. Neural Network (scikit-learn MLPRegressor)

Evaluates each on RMSE and R², prints a comparison table,
selects the best model by RMSE, and saves it to models/.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from preprocess import (
    load_dataset,
    fit_label_encoders,
    prepare_features,
    get_target,
    save_preprocessing_objects,
    MODELS_DIR,
    TARGET_COLUMN,
)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ─── Paths ────────────────────────────────────────────────────────────────────

DATA_PATH       = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "student-mat.csv")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
META_PATH       = os.path.join(MODELS_DIR, "best_model_meta.pkl")


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"Model": name, "RMSE": round(rmse, 4), "R²": round(r2, 4)}


# ─── Main Training Pipeline ───────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  EduAnalytics — Student Performance Training Pipeline")
    print("=" * 60)

    # 1. Load data
    df = load_dataset(DATA_PATH)

    # 2. Fit preprocessing objects
    encoders = fit_label_encoders(df)
    X, scaler, feature_columns = prepare_features(
        df, encoders, fit_scaler=True
    )
    y = get_target(df).values

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n[train] Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

    results = []
    trained_models = {}

    # ── Model 1: Linear Regression ───────────────────────────────────────────
    print("\n[train] Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_result = evaluate("Linear Regression", y_test, lr_pred)
    results.append(lr_result)
    trained_models["Linear Regression"] = ("sklearn", lr)
    print(f"  RMSE: {lr_result['RMSE']}  |  R²: {lr_result['R²']}")

    # ── Model 2: Random Forest ────────────────────────────────────────────────
    print("\n[train] Training Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_result = evaluate("Random Forest", y_test, rf_pred)
    results.append(rf_result)
    trained_models["Random Forest"] = ("sklearn", rf)
    print(f"  RMSE: {rf_result['RMSE']}  |  R²: {rf_result['R²']}")

    # ── Model 3: Neural Network (MLP) ────────────────────────────────────────
    print("\n[train] Training Neural Network (MLPRegressor)...")
    nn = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=42,
        verbose=False,
    )
    nn.fit(X_train.values, y_train)
    nn_pred = nn.predict(X_test.values)
    nn_result = evaluate("Neural Network", y_test, nn_pred)
    results.append(nn_result)
    trained_models["Neural Network"] = ("sklearn", nn)
    print(f"  RMSE: {nn_result['RMSE']}  |  R²: {nn_result['R²']}")

    # 4. Comparison table
    print("\n" + "=" * 60)
    print("  Model Comparison")
    print("=" * 60)
    results_df = pd.DataFrame(results).sort_values("RMSE")
    print(results_df.to_string(index=False))
    print("=" * 60)

    # 5. Select best model by lowest RMSE
    best_row   = results_df.iloc[0]
    best_name  = best_row["Model"]
    best_rmse  = best_row["RMSE"]
    best_r2    = best_row["R²"]
    model_type, best_model = trained_models[best_name]

    print(f"\n[train] ✅ Best model: {best_name}  (RMSE={best_rmse}, R²={best_r2})")

    # 6. Save preprocessing objects
    save_preprocessing_objects(encoders, scaler, feature_columns)

    # 7. Save best model
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump({"type": "sklearn", "model": best_model}, BEST_MODEL_PATH)

    # Save metadata for the API / predict script
    joblib.dump({"name": best_name, "rmse": best_rmse, "r2": best_r2}, META_PATH)
    print(f"[train] Saved best model meta → {META_PATH}")
    print(f"[train] Saved best model      → {BEST_MODEL_PATH}")
    print("\n[train] 🎉 Training complete!")


if __name__ == "__main__":
    train()
