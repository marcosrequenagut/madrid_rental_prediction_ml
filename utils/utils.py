import numpy as np

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    r2_score, explained_variance_score,
    mean_absolute_percentage_error, make_scorer
)

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