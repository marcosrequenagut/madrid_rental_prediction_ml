import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score


# Load the data
df = pd.read_csv(r'C:\Users\34651\Desktop\MASTER\TFM\Data\EDA_Madrid_SCALED.csv')

# Chose only the most important features, they other ones will be removed
df_new = df[['PRICE', 'UNITPRICE', 'CONSTRUCTEDAREA',
            'HASTERRACE', 'HASLIFT', 'ISPARKINGSPACEINCLUDEDINPRICE',
            'ROOMNUMBER', 'BATHNUMBER',
            'HASSWIMMINGPOOL', 'HASGARDEN', 'ISINTOPFLOOR',
            'DISTANCE_TO_CITY_CENTER', 'DISTANCE_TO_METRO', 'DISTANCE_TO_CASTELLANA',
            'CADMAXBUILDINGFLOOR', 'FLOORCLEAN', 'PERIOD_201803',
            'PERIOD_201806', 'PERIOD_201809', 'PERIOD_201812']]

# Drop target features
X = df_new.drop(['PRICE', 'UNITPRICE'], axis = 1)

# Asing target feature
y = df_new['PRICE']

# Split classes into rares and frequent ones
frequent_classes = y.value_counts()[y.value_counts() > 1].index
rare_classes = y.value_counts()[y.value_counts() == 1].index

# Stratify only frequent classes
stratified_sample = y.loc[y.isin(frequent_classes)]

# Divide the data into train and test each array
# Subset X and y for frequent classes
X_stratified = X.loc[y.isin(frequent_classes)]
y_stratified = y.loc[y.isin(frequent_classes)]
X_train_st, X_test_st, y_train_st, y_test_st = train_test_split(X_stratified, y_stratified, test_size = 0.8, stratify = y_stratified, random_state = 42)

# Subset X and y for rare classes
X_rare = X.loc[y.isin(rare_classes)]
y_rare = y.loc[y.isin(rare_classes)]
X_train_rare, X_test_rare, y_train_rare, y_test_rare = train_test_split(X_rare, y_rare, test_size = 0.8, random_state = 42)

# Concatenate the rare and frequent data
X_train = pd.concat([X_train_st, X_train_rare])
X_test = pd.concat([X_test_st, X_test_rare])
y_train = pd.concat([y_train_st, y_train_rare])
y_test = pd.concat([y_test_st, y_test_rare])

# Verifica que las columnas estén alineadas correctamente
print("X_train columns:", X_train.columns.tolist())
print("X_test columns:", X_test.columns.tolist())

# Verifica que las columnas de X_train y X_test sean exactamente iguales
if list(X_train.columns) == list(X_test.columns):
        print("Las columnas están alineadas correctamente.")
else:
        print("Las columnas NO están alineadas correctamente.")

# Define the Grid
grid = {
    'n_neighbors': [3, 5, 7, 10],  # Number of neighbors
    'weights': ['uniform', 'distance'],  # Weight function used in prediction
    'metric': ['euclidean', 'manhattan']  # Distance metric
}

# Chose se model
KNNR = KNeighborsRegressor()

# Define the cross validation
KNNR_CV = GridSearchCV(estimator = KNNR, param_grid = grid, scoring = 'neg_mean_absolute_error', cv = 5, n_jobs = -1)

# Seleccionamos un ejemplo de entrada (por ejemplo, la primera fila del conjunto de test)
input_example = X_test.iloc[0].values.reshape(1, -1)

print("¿Columnas en el mismo orden?", list(X_train.columns) == list(X_test.columns))
print("¿Nombres iguales?", set(X_train.columns) == set(X_test.columns))

with mlflow.start_run():
    # Train the model
    KNNR_CV.fit(X_train, y_train)

    # Metrics and params
    X_test = X_test[X_train.columns]

    # Verifica que las columnas estén alineadas correctamente
    print("X_train columns:", X_train.columns.tolist())
    print("X_test columns:", X_test.columns.tolist())

    # Verifica que las columnas de X_train y X_test sean exactamente iguales
    if list(X_train.columns) == list(X_test.columns):
        print("Las columnas están alineadas correctamente.")
    else:
        print("Las columnas NO están alineadas correctamente.")

    # Prediction
    y_pred = KNNR_CV.predict(X_test)

    # Metrics and params
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    best_params = KNNR_CV.best_params_

    # Save hyperparams and mae
    mlflow.log_param("best hyperparameters", best_params)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Save the model
    # Guardar el modelo con ejemplo de entrada
    mlflow.sklearn.log_model(KNNR_CV.best_estimator_, "modelo_knn", input_example=input_example)