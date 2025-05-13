import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.knn import train_and_log_knn
from src.models.linear_regressor import train_and_log_linear_regressor
from src.models.random_forest import train_and_log_random_forest_regressor

# Cargar datos
df = pd.read_csv(r'C:\Users\34651\Desktop\MASTER\TFM\Data\EDA_Madrid_SCALED.csv')

# Selección de variables
df_new = df[['PRICE', 'UNITPRICE', 'CONSTRUCTEDAREA',
             'HASTERRACE', 'HASLIFT', 'ISPARKINGSPACEINCLUDEDINPRICE',
             'ROOMNUMBER', 'BATHNUMBER', 'HASSWIMMINGPOOL', 'HASGARDEN',
             'ISINTOPFLOOR', 'DISTANCE_TO_CITY_CENTER', 'DISTANCE_TO_METRO',
             'DISTANCE_TO_CASTELLANA', 'CADMAXBUILDINGFLOOR', 'FLOORCLEAN',
             'PERIOD_201803', 'PERIOD_201806', 'PERIOD_201809', 'PERIOD_201812']]

X = df_new.drop(['PRICE', 'UNITPRICE'], axis=1)
y = df_new['PRICE']

# Separación entre clases frecuentes y raras
frequent_classes = y.value_counts()[y.value_counts() > 1].index
rare_classes = y.value_counts()[y.value_counts() == 1].index

X_stratified = X.loc[y.isin(frequent_classes)]
y_stratified = y.loc[y.isin(frequent_classes)]
X_train_st, X_test_st, y_train_st, y_test_st = train_test_split(
    X_stratified, y_stratified, test_size=0.8, stratify=y_stratified, random_state=42)

X_rare = X.loc[y.isin(rare_classes)]
y_rare = y.loc[y.isin(rare_classes)]
X_train_rare, X_test_rare, y_train_rare, y_test_rare = train_test_split(
    X_rare, y_rare, test_size=0.8, random_state=42)

# Juntar clases raras y frecuentes
X_train = pd.concat([X_train_st, X_train_rare])
X_test = pd.concat([X_test_st, X_test_rare])
y_train = pd.concat([y_train_st, y_train_rare])
y_test = pd.concat([y_test_st, y_test_rare])

# Train and predict with KNN
#train_and_log_knn(X_train, X_test, y_train, y_test)

# Train and predict with Linear Regressor
#train_and_log_linear_regressor(X_train, X_test, y_train, y_test)

# Train and predict with Random Forest
train_and_log_random_forest_regressor(X_train, X_test, y_train, y_test)