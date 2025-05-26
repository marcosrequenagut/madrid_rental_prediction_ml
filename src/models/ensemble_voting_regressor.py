import mlflow
import mlflow.sklearn
import time

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor

from utils.utils import get_regression_scorers, calculate_metrics


def train_ensemble_model(X_train, X_test, y_train, y_test):
    # Initial configuration
    model_name = "voting_regressor"

    run_name = f"{model_name}_{int(time.time())}"

    with mlflow.start_run(run_name=run_name):
        # Best hyperparameters found in previous analysis
        best_params_xgb = {
            'colsample_bytree': 1.0,
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 200,
            'subsample': 0.8}

        best_params_rf = {
            'max_depth': 30,
            'min_samples_split': 3,
            'n_estimators': 150}

        best_params_knn = {
            'metric': 'manhattan',
            'n_neighbors': 7,
            'weights': 'distance'}

        # Crear instancias de los modelos
        xgb = XGBRegressor(objective='reg:squarederror', n_jobs=-1, **best_params_xgb)
        rf = RandomForestRegressor(n_jobs=-1, **best_params_rf)
        knn = KNeighborsRegressor(n_jobs=-1,**best_params_knn)

        # Crear el VotingRegressor
        ensemble = VotingRegressor(estimators=[
            ('xgb', xgb),
            ('rf', rf),
            ('svr', knn)
        ])

        # Train the model
        ensemble.fit(X_train, y_train)

        # Predictions on test and training
        y_train_pred = ensemble.predict(X_train)
        y_test_pred = ensemble.predict(X_test)

        # Metrics
        metrics_train = calculate_metrics(y_train, y_train_pred)
        metrics_test = calculate_metrics(y_test, y_test_pred)

        # Log in MLflow
        for metric, value in metrics_train.items():
            mlflow.log_metric(f"{metric}_train", value)
        for metric, value in metrics_test.items():
            mlflow.log_metric(f"{metric}_test", value)

        mlflow.sklearn.log_model(ensemble, "ensemble_model")

        # Show some key metrics
        print(f"R2 test: {metrics_test['r2']:.2f}")
        print(f"MAE test: {metrics_test['mae']:.2f}")
        print(f"RMSE test: {metrics_test['rmse']:.2f}")

        return ensemble
