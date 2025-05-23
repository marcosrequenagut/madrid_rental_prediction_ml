import mlflow
import mlflow.sklearn
import time
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from utils.utils import setup_mlflow

setup_mlflow()
print("Tracking URI:", mlflow.get_tracking_uri())

def train_and_log_linear_regressor(X_train, X_test, y_train, y_test):
    #Configuración inicial
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # O usa una ruta local
    model_name = "linear_regressor"

    run_name = f"{model_name}_{int(time.time())}"

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
                             scoring='neg_mean_absolute_error',
                             cv=5, n_jobs=-1)

        # Train the model
        lr_cv.fit(X_train, y_train)

        # Predictions
        y_pred = lr_cv.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        best_params = lr_cv.best_params_

        # Log with MLflow
        mlflow.log_param("best_hyperparameters", best_params)
        mlflow.log_metric("mae_test", mae)
        mlflow.log_metric("r2_test", r2)
        mlflow.log_metric("mse_test", mse)
        mlflow.log_metric("rmse_test", rmse)

        # Save the model
        mlflow.sklearn.log_model(lr_cv.best_estimator_, "modelo_linear_regressor")

        print(f"Linear Regressor MAE: {mae:.2f}")
        print(f"KNN MSE: {mse:.2f}")
        print(f"KNN RMSE: {rmse:.2f}")
        print(f"Linear Regressor R2: {r2:.2f}")
        print(f"Linear Regressor Best Params: {best_params}")

        return lr_cv.best_estimator_