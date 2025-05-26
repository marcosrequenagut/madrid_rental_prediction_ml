import mlflow
import mlflow.sklearn
import time

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from utils.utils import get_regression_scorers, extract_cv_metrics, calculate_metrics


def train_and_log_knn(X_train, X_test, y_train, y_test):
    # Initial configuration
    model_name = "knn_regressor"

    run_name = f"{model_name}_{int(time.time())}"

    # Define custom scorers
    scoring = get_regression_scorers()

    # Save the results of the model
    with mlflow.start_run(run_name=run_name):
        # Define the Grid
        grid = {
            'n_neighbors': [3, 5, 7, 10],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan'],
        }

        '''grid = {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2],  # solo se usa si metric='minkowski'
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [20, 30, 40]
    }'''

        # Define the model
        knn = KNeighborsRegressor()

        # Grid Search with CV
        knn_cv = GridSearchCV(estimator=knn,
                              param_grid=grid,
                              scoring=scoring,
                              refit='r2',
                              cv=5,n_jobs=-1,
                              return_train_score=True)

        # Take an example of input for the log
        input_example = X_test.iloc[0].values.reshape(1, -1)

        # Train
        knn_cv.fit(X_train, y_train)

        # Prediction and metrics on training
        r2_train = knn_cv.best_score_
        best_idx = knn_cv.best_index_
        results = knn_cv.cv_results_

        metrics_train = extract_cv_metrics(results, best_idx)

        # Predictions
        y_pred = knn_cv.predict(X_test)
        metrics_test = calculate_metrics(y_test, y_pred)

        # Best hyperparams of the model
        best_params = knn_cv.best_params_

        # Log with MlFlow
        mlflow.log_param("best_hyperparameters", best_params)
        for metric, value in metrics_test.items():
            mlflow.log_metric(f"{metric}_test", value)
        for metric, value in metrics_train.items():
            mlflow.log_metric(metric, value)

        # Save the model
        mlflow.sklearn.log_model(knn_cv.best_estimator_, "random_forest_regressor")

        print("Show the r^2 for KNN Regressor:")
        print(f"R2 on test: {metrics_test['r2']:.2f}")
        print(f"R2 on training: {r2_train:.2f}")
        print(f"Best Params: {best_params}")

        return knn_cv.best_estimator_