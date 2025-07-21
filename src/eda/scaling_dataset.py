import pandas as pd
import joblib
import mlflow

from sklearn.preprocessing import StandardScaler

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set the name of the experiment
mlflow.set_experiment("TFM")

df = pd.read_csv(r'C:\Users\34651\Desktop\MASTER\TFM\Data\EDA_MADRID.csv')

df1 = df[['PRICE', 'UNITPRICE', 'CONSTRUCTEDAREA',
                 'HASTERRACE', 'ISPARKINGSPACEINCLUDEDINPRICE',
                 'ROOMNUMBER', 'BATHNUMBER', 'HASSWIMMINGPOOL',
                 'ISINTOPFLOOR', 'DISTANCE_TO_CITY_CENTER', 'DISTANCE_TO_METRO',
                 'DISTANCE_TO_CASTELLANA', 'CADMAXBUILDINGFLOOR', 'FLOORCLEAN']]

# Drop the 'GEOMETRY' column as it is not necessary because we have the 'LATITUDE' and 'LONGITUDE' column.
df2 = df1.drop(['PRICE', 'UNITPRICE'], axis = 1)


# Scale the dataset using StandardScaler
scaler = StandardScaler()

continuous_features = ['CONSTRUCTEDAREA', 'CADMAXBUILDINGFLOOR',
       'DISTANCE_TO_CITY_CENTER', 'DISTANCE_TO_METRO', 'DISTANCE_TO_CASTELLANA', 'ROOMNUMBER',
       'BATHNUMBER', 'FLOORCLEAN']

df2[continuous_features] = scaler.fit_transform(df2[continuous_features])

# Save the scaler locally
joblib.dump(scaler, "scaler.pkl")

# Log with MLflow
mlflow.set_experiment("TFM")

# Save the scaler in MLFlow to use it in the predict endpoint of the API
with mlflow.start_run(run_name="StandardScaler"):
    mlflow.log_artifact("scaler.pkl", artifact_path="scaler")

print("Dataset scaled!")
