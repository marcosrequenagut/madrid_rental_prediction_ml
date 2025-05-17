from src.data_preprocessing import load_and_preproces_data
from src.models.ensemble_voting_regressor import train_ensemble_model
from src.models.knn import train_and_log_knn
from src.models.linear_regressor import train_and_log_linear_regressor
from src.models.random_forest import train_and_log_random_forest_regressor
from src.models.svm_regressor import train_and_log_svm_regressor
from src.models.xgboost_regressor import train_and_log_xgboost_regressor

# Path where data is stored
URL = r'C:\Users\34651\Desktop\MASTER\TFM\Data\EDA_Madrid_SCALED.csv'

# Obtain the train and test datasets
X_train, X_test, y_train, y_test = load_and_preproces_data(URL)

# Train and predict with KNN
train_and_log_knn(X_train, X_test, y_train, y_test)

# Train and predict with Linear Regressor
#train_and_log_linear_regressor(X_train, X_test, y_train, y_test)

# Train and predict with Random Forest
#train_and_log_random_forest_regressor(X_train, X_test, y_train, y_test)

# Train and Predict with XGBoost Regressor
#train_and_log_xgboost_regressor(X_train, X_test, y_train, y_test)

# Train and Predict with SVM Regressor
#train_and_log_svm_regressor(X_train, X_test, y_train, y_test)

# Train and Predict with Ensembles
#train_ensemble_model(X_train, X_test, y_train, y_test)

