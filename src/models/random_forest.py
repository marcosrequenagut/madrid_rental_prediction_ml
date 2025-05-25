import mlflow
import mlflow.sklearn
import time
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, explained_variance_score, make_scorer, mean_absolute_percentage_error
from utils.utils import setup_mlflow, calculate_metrics, extract_cv_metrics

setup_mlflow()

def train_and_log_random_forest_regressor(X_train, X_test, y_train, y_test):
    # Initial configuration
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # O usa una ruta local
    model_name = "random_forest_regressor"

    run_name = f"{model_name}_{int(time.time())}"

    # Define custom scorers
    scoring = {
        'mae': make_scorer(mean_absolute_error, greater_is_better=False),
        'mse': make_scorer(mean_squared_error, greater_is_better=False),
        'rmse': make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)), greater_is_better=False),
        'mape': make_scorer(mean_absolute_percentage_error, greater_is_better=False),
        'r2': make_scorer(r2_score),
        'evs': make_scorer(explained_variance_score)
    }

    # Save the results of the model
    with mlflow.start_run(run_name=run_name):
        # Define the grid
        grid = {
            'n_estimators': [100, 150],
            'max_depth': [20, 30],
            'min_samples_split': [3, 10]
        }

        ''''min_samples_split': [3, 10],
            'min_samples_leaf': [2,4,6],
            'boostrap': [True, False],'''

        # Chose the model
        rf = RandomForestRegressor(n_jobs=-1)

        # Define the cross-validation
        rf_cv = GridSearchCV(estimator=rf, param_grid=grid,
                             scoring=scoring,
                             refit='r2',
                             cv = 5, n_jobs=-1,
                             return_train_score=True)

        # Train the model
        rf_cv.fit(X_train, y_train)

        # Prediction and metrics on training
        r2_train = rf_cv.best_score_
        best_idx = rf_cv.best_index_
        results = rf_cv.cv_results_

        metrics_train = extract_cv_metrics(results, best_idx)

        # Predictions and Metrics on Test
        y_pred = rf_cv.predict(X_test)
        metrics_test = calculate_metrics(y_test, y_pred)

        # Best hyperparams of the model
        best_params = rf_cv.best_params_

        # Log with MlFlow
        mlflow.log_param("best_hyperparameters", best_params)
        for metric, value in metrics_test.items():
            mlflow.log_metric(f"{metric}_test", value)
        for metric, value in metrics_train.items():
            mlflow.log_metric(f"{metric}_train", value)


        # Save the model
        mlflow.sklearn.log_model(rf_cv.best_estimator_, "random_forest_regressor")

        print("Show the r^2 for Random Forest Regressor:")
        print(f"R2 on test: {metrics_test['r2']:.2f}")
        print(f"R2 on training: {r2_train:.2f}")
        print(f"Best Params: {best_params}")

        return rf_cv.best_estimator_
