import mlflow.sklearn
import os
import joblib
import pandas as pd
import logging
import json
import unidecode

from fastapi import APIRouter, HTTPException
from .PropertyFeatures import PropertyFeatures
from unidecode import unidecode

print("loaded predict.py")

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set the MLflow tracking URI
mlflow_run_id = "http://127.0.0.1:5000"

# Set the scaler run id
scaler_run_id = "3235ac7dbbd24123a5954bc24351ea03"

# URI where the MLFlow server is running
mlflow.set_tracking_uri(mlflow_run_id)

# Load the model from the Model Registry (last version of the model registered), if we want to use Random Forest
#model = mlflow.sklearn.load_model("models:/random_forest_regressor/latest")

# Load the model from the Model Registry (last version of the model registered), if we want to use Ensemble Voting
model = mlflow.sklearn.load_model("models:/voting_regressor/latest")

# Directory of predict.py
current_dir = os.path.dirname(__file__)

# Going up 3 levels to reach the root of the project
project_root = os.path.abspath(os.path.join(current_dir, "../.."))

# Import the scale object (scale.pkl) from the correct path
scaler_path = os.path.join(project_root, "src", "eda", "scaler.pkl")

# Load the scaler object
try:
    # The scaler.pkl is in the directory "scaler" of the MLFlow experiment
    logged_run_uri = f"runs:/{scaler_run_id}/scaler/scaler.pkl"
    scaler = joblib.load(mlflow.artifacts.download_artifacts(logged_run_uri))
except Exception as e:
    print("Error loading scaler:", e)

# Register the router, it is used to define the endpoints of the API
router = APIRouter()

# Define the columns for the district features
DISTRICT_COLUMNS = [
    'DISTRICTS_ARGANZUELA', 'DISTRICTS_BARAJAS', 'DISTRICTS_CARABANCHEL', 'DISTRICTS_CENTRO',
    'DISTRICTS_CHAMARTIN', 'DISTRICTS_CHAMBERI', 'DISTRICTS_CIUDAD LINEAL',
    'DISTRICTS_FUENCARRAL-EL PARDO', 'DISTRICTS_HORTALEZA', 'DISTRICTS_LATINA',
    'DISTRICTS_MONCLOA-ARAVACA', 'DISTRICTS_MORATALAZ', 'DISTRICTS_PUENTE DE VALLECAS',
    'DISTRICTS_RETIRO', 'DISTRICTS_SALAMANCA', 'DISTRICTS_SAN BLAS-CANILLEJAS',
    'DISTRICTS_TETUAN', 'DISTRICTS_USERA', 'DISTRICTS_VICALVARO',
    'DISTRICTS_VILLA DE VALLECAS', 'DISTRICTS_VILLAVERDE'
]

# Define the columns for the location features
LOCATION_COLUMNS = [
    'LOCATIONNAME_0', 'LOCATIONNAME_1', 'LOCATIONNAME_2',
    'LOCATIONNAME_3', 'LOCATIONNAME_4', 'LOCATIONNAME_5',
    'LOCATIONNAME_6', 'LOCATIONNAME_7', 'LOCATIONNAME_8', 'LOCATIONNAME_9'
]

# Read the json where location name is map to its group
json_path = r'C:\Users\34651\Desktop\MASTER\TFM\madrid_rental_prediction_ml\data\new_data\locationnameGroup.json'
with open(json_path, 'r', encoding='utf-8') as file:
    location_name_map = json.load(file)

print("0: Loaded the model and the scaler")

# Define the prediction endpoint
@router.post("", summary="Predict the price of a property")
def predict(features: PropertyFeatures):
    """
    Predict the price of a property based on its features.

    - Accepts a POST request with structured property features.
    - Encodes categorical variables for district and location using one-hot encoding.
    - Normalizes continuous features using a pre-trained scaler (`scaler.pkl`).
    - Loads a Voting Regressor model from the MLflow Model Registry.
    - Performs inference using the processed input data.
    - Returns the predicted property price.

    :param features: a `PropertyFeatures` object containing the input data for prediction.

    :returns: JSON with the predicted label (price in euros).

    :raises 400 if the district or location name is not valid.
    :raises 500 if any error occurs during data preparation, transformation, or prediction.
    """
    print("1: Entering into the endpoint predict")

    district_name = 'DISTRICTS_'+unidecode(features.district).upper()
    location_name = unidecode(features.location).upper()

    try:
        print("2: Preparing the data")
        data_dict = {
            'CONSTRUCTEDAREA': features.constructed_area,
            'HASTERRACE': features.has_terrace,
            'ISPARKINGSPACEINCLUDEDINPRICE': features.is_parkingspace_included,
            'ROOMNUMBER': features.number_of_rooms,
            'BATHNUMBER': features.number_of_bathrooms,
            'HASSWIMMINGPOOL': features.has_swimming_pool,
            'ISINTOPFLOOR': features.is_top_floor,
            'DISTANCE_TO_CITY_CENTER': features.distance_to_city_center,
            'DISTANCE_TO_METRO': features.distance_to_city_metro,
            'DISTANCE_TO_CASTELLANA': features.distance_to_city_castellana,
            'CADMAXBUILDINGFLOOR': features.constructed_year,
            'FLOORCLEAN': features.floorclean
        }

        group_name = None
        # Find which group the location belongs to
        for key, value in location_name_map.items():
            if location_name == key:
                group_name = 'LOCATIONNAME_' + str(value)
                break

        if group_name is None:
            raise HTTPException(status_code=400, detail=f"Invalid location name: {location_name}")

        # Add the value of each location column to the data_dict
        for location in LOCATION_COLUMNS:
            if group_name == location:
                data_dict[location] = 1
            else:
                data_dict[location] = 0

        # Add the value of each district column to the data_dict
        for district in DISTRICT_COLUMNS:
            if district_name == district:
                data_dict[district] = 1
            else:
                data_dict[district] = 0

        if district_name not in DISTRICT_COLUMNS:
            raise HTTPException(status_code=400, detail=f"Invalid district name: {district_name}")

        print(f"Data dictionary prepared: {data_dict}")
        df = pd.DataFrame([data_dict])
        print("3: Data transformed into a DataFrame")

        continuous_features = ['CONSTRUCTEDAREA', 'CADMAXBUILDINGFLOOR',
                               'DISTANCE_TO_CITY_CENTER', 'DISTANCE_TO_METRO', 'DISTANCE_TO_CASTELLANA', 'ROOMNUMBER',
                               'BATHNUMBER', 'FLOORCLEAN']
        df[continuous_features] = scaler.transform(df[continuous_features])
        print("4: Data scaled")

        prediction = model.predict(df)
        print("5: Prediction finished")

        return {"Predicted label": prediction[0]}

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
