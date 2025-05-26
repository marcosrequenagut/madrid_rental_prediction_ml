import mlflow
import mlflow.sklearn
import time
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from utils.utils import get_regression_scorers, extract_cv_metrics, calculate_metrics


def train_and_log_linear_regressor(X_train, X_test, y_train, y_test):
    # Initial configuration
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # O usa una ruta local
    model_name = "linear_regressor"

    run_name = f"{model_name}_{int(time.time())}"

    # Define custom scorers
    scoring = get_regression_scorers()

    # Save the results of the model
    with mlflow.start_run(run_name=run_name):
        # Define the Grid
        grid = {
            'fit_intercept': [True, False]
        }

        # Define the model
        lr = LinearRegression()

        # Define the cross-validation
        lr_cv = GridSearchCV(estimator=lr, param_grid=grid,
                             scoring=scoring,
                             refit='r2',
                             cv=5, n_jobs=-1,
                             return_train_score=True)

        # Train the model
        lr_cv.fit(X_train, y_train)

        # Prediction and metrics on training
        r2_train = lr_cv.best_score_
        best_idx = lr_cv.best_index_
        results = lr_cv.cv_results_

        metrics_train = extract_cv_metrics(results, best_idx)

        # Predictions
        y_pred = lr_cv.predict(X_test)
        metrics_test = calculate_metrics(y_test, y_pred)

        # Best hyperparams of the model
        best_params = lr_cv.best_params_

        # Log with MlFlow
        mlflow.log_param("best_hyperparameters", best_params)
        for metric, value in metrics_test.items():
            mlflow.log_metric(f"{metric}_test", value)
        for metric, value in metrics_train.items():
            mlflow.log_metric(metric, value)

        # Save the model
        mlflow.sklearn.log_model(lr_cv.best_estimator_, "random_forest_regressor")

        print("Show the r^2 for KNN Regressor:")
        print(f"R2 on test: {metrics_test['r2']:.2f}")
        print(f"R2 on training: {r2_train:.2f}")
        print(f"Best Params: {best_params}")

        return lr_cv.best_estimator_