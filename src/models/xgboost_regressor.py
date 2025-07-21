import mlflow
import mlflow.sklearn
import time
import numpy as np

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from utils.utils import get_regression_scorers, extract_cv_metrics, calculate_metrics, \
    show_linear_model_feature_importance, show_tree_model_feature_importance, save_model


def train_and_log_xgboost_regressor(X_train, X_test, y_train, y_test):
    # Initial configuration
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Set the name of the experiment
    mlflow.set_experiment("TFM_column_group2")

    model_name = "xgboost_regressor_80pct"

    # Name of the model which will be saved
    run_name = f"{model_name}_{int(time.time())}"

    # Define custom scorers
    scoring = get_regression_scorers()

    with mlflow.start_run(run_name=run_name):
        # Define the grid of hyperparameters
        grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
        }

        '''grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 2]
}'''

        # Define the model
        xgb = XGBRegressor(objective='reg:squarederror', n_jobs=-1)

        # Cross-validation
        xgb_cv = GridSearchCV(estimator=xgb, param_grid=grid,
                              scoring=scoring,
                              refit='r2',
                              cv=3, n_jobs=-1,
                              return_train_score=True)

        # Train
        xgb_cv.fit(X_train, y_train)

        # Important features
        feature_names = X_train.columns
        show_tree_model_feature_importance(xgb_cv.best_estimator_, feature_names)

        # Prediction and metrics on training
        r2_train = xgb_cv.best_score_
        best_idx = xgb_cv.best_index_
        results = xgb_cv.cv_results_

        metrics_train = extract_cv_metrics(results, best_idx)

        # Predictions and Metrics on Test with the best model
        y_pred = xgb_cv.predict(X_test)
        metrics_test = calculate_metrics(y_test, y_pred)

        # Best hyperparams of the model
        best_params = xgb_cv.best_params_

        # Log with MlFlow
        mlflow.log_param("best_hyperparameters", best_params)
        for metric, value in metrics_test.items():
            mlflow.log_metric(f"{metric}_test", value)
        for metric, value in metrics_train.items():
            mlflow.log_metric(metric, value)
        mlflow.log_metric("r2_train", r2_train)

        # Save the model
        save_model(xgb_cv.best_estimator_, "xgboost_regressor")

        print("Show the r^2 for XGBoost Regressor:")
        print(f"R2 on test: {metrics_test['r2']:.2f}")
        print(f"R2 on training: {r2_train:.2f}")
        print(f"Best Params: {best_params}")

        return xgb_cv.best_estimator_
