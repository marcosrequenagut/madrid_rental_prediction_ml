import mlflow
import mlflow.sklearn
import time
import numpy as np

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def train_and_log_xgboost_regressor(X_train, X_test, y_train, y_test):
    # Initial configuration
    model_name = "xgboost_regressor"

    # Name of the model which will be saved
    run_name = f"{model_name}_{int(time.time())}"

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
                              scoring='neg_mean_absolute_error',
                              cv=3, n_jobs=-1)

        # Train
        xgb_cv.fit(X_train, y_train)

        # Prediction and metrics on training
        y_train_pred = xgb_cv.predict(X_train)
        r2_train = r2_score(y_train, y_train_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)

        # Predictions and Metrics on test
        y_pred = xgb_cv.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        best_params = xgb_cv.best_params_

        # Log with MlFlow
        mlflow.log_param("best_hyperparameters", best_params)
        mlflow.log_metric("mae_test", mae)
        mlflow.log_metric("r2_test", r2)
        mlflow.log_metric("mse_test", mse)
        mlflow.log_metric("rmse_test", rmse)
        mlflow.log_metric("mae_train", mae_train)
        mlflow.log_metric("r2_train", r2_train)

        # Save the model
        mlflow.sklearn.log_model(xgb_cv.best_estimator_, "model_xgboost_regressor")

        print(f"XGBoost Regressor MAE: {mae:.2f}")
        print(f"KNN MSE: {mse:.2f}")
        print(f"KNN RMSE: {rmse:.2f}")
        print(f"XGBoost Regressor R2: {r2:.2f}")
        print(f"XGBoost Regressor Best Params: {best_params}")

        return xgb_cv.best_estimator_