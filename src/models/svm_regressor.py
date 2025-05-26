import mlflow
import mlflow.sklearn
import time
import numpy as np

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from utils.utils import get_regression_scorers, extract_cv_metrics, calculate_metrics


def train_and_log_svm_regressor(X_train, X_test, y_train, y_test):
    # Initial configuration
    model_name = "svm_regressor"

    # Name of the model which will be saved
    run_name = f"{model_name}_{int(time.time())}"

    # Define custom scorers
    scoring = get_regression_scorers()

    with mlflow.start_run(run_name=run_name):
        # Define the grid of hyperparameters
        grid = {
            'kernel': ['rbf', 'linear'],
            'C': [10, 100],
            'epsilon': [0.2, 0.3],
            'gamma': ['scale', 'auto'],
            'degree': [2, 3]
        }

        # Define the model
        svr = SVR()

        # Cross-validation
        svr_cv = GridSearchCV(estimator=svr, param_grid=grid,
                              scoring=scoring,
                              refit='r2',
                              cv=5, n_jobs=-1,
                              return_train_score=True)

        # Train
        svr_cv.fit(X_train, y_train)

        # Prediction and metrics on training
        r2_train = svr_cv.best_score_
        best_idx = svr_cv.best_index_
        results = svr_cv.cv_results_

        metrics_train = extract_cv_metrics(results, best_idx)

        # Predictions and Metrics on Test with the best model
        y_pred = svr_cv.predict(X_test)
        metrics_test = calculate_metrics(y_test, y_pred)

        # Best hyperparams of the model
        best_params = svr_cv.best_params_

        # Log with MlFlow
        mlflow.log_param("best_hyperparameters", best_params)
        for metric, value in metrics_test.items():
            mlflow.log_metric(f"{metric}_test", value)
        for metric, value in metrics_train.items():
            mlflow.log_metric(metric, value)

        # Save the model
        mlflow.sklearn.log_model(svr_cv.best_estimator_, "random_forest_regressor")

        print("Show the r^2 for Super Vector Machine Regressor:")
        print(f"R2 on test: {metrics_test['r2']:.2f}")
        print(f"R2 on training: {r2_train:.2f}")
        print(f"Best Params: {best_params}")

        return svr_cv.best_estimator_
