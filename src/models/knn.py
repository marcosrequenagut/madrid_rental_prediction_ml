import mlflow
import mlflow.sklearn

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score

def train_and_log_knn(X_train, X_test, y_train, y_test):
    # Define the Grid
    grid = {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    # Define the model
    knn = KNeighborsRegressor()

    # Grid Search with CV
    knn_cv = GridSearchCV(estimator=knn, param_grid=grid,
                          scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)

    # Take an example of input for the log
    input_example = X_test.iloc[0].values.reshape(1, -1)

    # Save the results of the model
    with mlflow.start_run(run_name="KNN_Model"):
        # Train
        knn_cv.fit(X_train, y_train)

        # Predictions
        y_pred = knn_cv.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        best_params = knn_cv.best_params_

        # Log with MLflow
        mlflow.log_param("best_hyperparameters", best_params)
        mlflow.log_metric("mae_test", mae)
        mlflow.log_metric("r2_test", r2)

        # Save the model
        mlflow.sklearn.log_model(knn_cv.best_estimator_, "modelo_knn", input_example=input_example)

        print(f"KNN MAE: {mae:.2f}")
        print(f"KNN R2: {r2:.2f}")
        print(f"KNN Best Params: {best_params}")

        return knn_cv.best_estimator_