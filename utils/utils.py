import numpy as np
import pandas as pd
import pickle
import mlflow

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    r2_score, explained_variance_score,
    mean_absolute_percentage_error, make_scorer
)
from minio import Minio
from pathlib import Path

# ------------------------
# -- Feature Importance --
# ------------------------

def show_linear_model_feature_importance(model, feature_names, top_n=None):
    """    Displays the feature importance of a regression model."""

    coefs = model.coef_
    importance = pd.Series(coefs, index=feature_names)
    importance_abs = importance.abs()
    importance_abs_normalized = (importance_abs / importance_abs.max()).sort_values(ascending=False)

    if top_n:
        importance_abs_normalized = importance_abs_normalized.head(top_n)

    print("Feature Importance (Absolute Values):")
    for feature, value in importance_abs_normalized.items():
        print(f"{feature}: {value:.4f}")

def show_tree_model_feature_importance(model, feature_names):
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance_abs = importance.abs()
    importance_abs_normalized = (importance_abs / importance_abs.max()).sort_values(ascending=False)

    print("Feature Importance (Absolute Values):")
    for feature, value in importance_abs_normalized.items():
        print(f"{feature}: {value:.4f}")

# ------------------------
# ------- METRICS --------
# ------------------------
def get_regression_scorers():
    return {
        'mae': make_scorer(mean_absolute_error, greater_is_better=False),
        'mse': make_scorer(mean_squared_error, greater_is_better=False),
        'rmse': make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)), greater_is_better=False),
        'mape': make_scorer(mean_absolute_percentage_error, greater_is_better=False),
        'r2': make_scorer(r2_score),
        'evs': make_scorer(explained_variance_score)
    }

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2_score(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'evs': explained_variance_score(y_true, y_pred)
    }

def extract_cv_metrics(results, best_idx):
    return {
        'mae_train': -results['mean_test_mae'][best_idx],
        'mse_train': -results['mean_test_mse'][best_idx],
        'rmse_train': -results['mean_test_rmse'][best_idx],
        'mape_train': -results['mean_test_mape'][best_idx],
        'evs_train': results['mean_test_evs'][best_idx]
    }

# ------------------------
# -------- Model ---------
# ------------------------
def save_model(model, model_name):
    if mlflow.active_run() is None:
        with mlflow.start_run():
            mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=model_name)
    else:
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=model_name)

    print(f"Model {model_name} saved and registered in MLflow.")