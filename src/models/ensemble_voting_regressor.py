import mlflow
import mlflow.sklearn
import time

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor


def train_ensemble_model(X_train, X_test, y_train, y_test):
    # Configuración inicial
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
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

        # Train the ensemble
        ensemble.fit(X_train, y_train)

        # Predictions
        y_train_pred = ensemble.predict(X_train)
        y_test_pred = ensemble.predict(X_test)

        # Metrics
        r2_train = r2_score(y_train, y_train_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)

        # Log en MLflow
        mlflow.log_metric("r2_train", r2_train)
        mlflow.log_metric("mae_train", mae_train)
        mlflow.log_metric("r2_test", r2_test)
        mlflow.log_metric("mae_test", mae_test)

        mlflow.sklearn.log_model(ensemble, "ensemble_model")

        print(f"Voting Regressor - MAE Test: {mae_test:.2f}, R2 Test: {r2_test:.2f}")

        return ensemble
