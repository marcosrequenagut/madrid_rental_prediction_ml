import mlflow
import mlflow.sklearn
import time
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def train_and_log_random_forest_regressor(X_train, X_test, y_train, y_test):
    # Initial configuration
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # O usa una ruta local
    model_name = "random_forest_regressor"

    run_name = f"{model_name}_{int(time.time())}"

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
                             scoring='neg_mean_absolute_error',
                             cv = 5, n_jobs=-1)

        # Train the model
        rf_cv.fit(X_train, y_train)

        # Prediction and metrics on training
        y_train_pred = rf_cv.predict(X_train)
        r2_train = r2_score(y_train, y_train_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)

        # Predictions and Metrics on Test
        y_pred = rf_cv.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        best_params = rf_cv.best_params_

        # Log with MlFlow
        mlflow.log_param("best_hyperparameters", best_params)
        mlflow.log_metric("mae_test", mae)
        mlflow.log_metric("r2_test", r2)
        mlflow.log_metric("mse_test", mse)
        mlflow.log_metric("rmse_test", rmse)
        mlflow.log_metric("mae_train", mae_train)
        mlflow.log_metric("r2_train", r2_train)

        # Save the model
        mlflow.sklearn.log_model(rf_cv.best_estimator_, "modelo_linear_regressor")

        print(f"Linear Regressor MAE: {mae:.2f}")
        print(f"KNN MSE: {mse:.2f}")
        print(f"KNN RMSE: {rmse:.2f}")
        print(f"Linear Regressor R2: {r2:.2f}")
        print(f"Linear Regressor Best Params: {best_params}")

        return rf_cv.best_estimator_
