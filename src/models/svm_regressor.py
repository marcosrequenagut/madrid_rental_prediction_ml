import mlflow
import mlflow.sklearn
import time

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

def train_and_log_svm_regressor(X_train, X_test, y_train, y_test):
    # Initial configuration
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model_name = "svm_regressor"

    # Name of the model which will be saved
    run_name = f"{model_name}_{int(time.time())}"

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
                              scoring='neg_mean_absolute_error',
                              cv=5, n_jobs=-1)

        # Train
        svr_cv.fit(X_train, y_train)

        # Prediction and metrics on training
        y_train_pred = svr_cv.predict(X_train)
        r2_train = r2_score(y_train, y_train_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)

        # Predictions and Metrics on test
        y_pred = svr_cv.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        best_params = svr_cv.best_params_

        # Log with MLFlow
        mlflow.log_param("best_hyperparameters", best_params)
        mlflow.log_metric("mae_test", mae)
        mlflow.log_metric("r2_test", r2)
        mlflow.log_metric("mae_train", mae_train)
        mlflow.log_metric("r2_train", r2_train)

        # Save the model
        mlflow.sklearn.log_model(svr_cv.best_estimator_, "model_svm_regressor")

        print(f"SVM Regressor MAE: {mae:.2f}")
        print(f"SVM Regressor R2: {r2:.2f}")
        print(f"SVM Regressor Best Params: {best_params}")

        return svr_cv.best_estimator_
