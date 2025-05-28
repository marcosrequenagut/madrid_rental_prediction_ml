import numpy as np
import mlflow.sklearn
import os
import joblib
import pandas as pd
import logging

from fastapi import APIRouter, HTTPException
from .PropertyFeatures import PropertyFeatures

print("loaded predict.py")

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# URI where the MLFlow server is running
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load the model from the Model Registry (last version of the model registred)
model = mlflow.sklearn.load_model("models:/random_forest_regressor/latest")

# Directory of predict.py
current_dir = os.path.dirname(__file__)

# Going up 3 levels to reach the root of the project
project_root = os.path.abspath(os.path.join(current_dir, "../.."))

# Import the scale object (scale.pkl) from the correct path
scaler_path = os.path.join(project_root, "src", "eda", "scaler.pkl")

# Load the scaler object
try:
    scaler = joblib.load(scaler_path)
except Exception as e:
    print("Error loading scaler:", e)

# Register the router, it is used to define the endpoints of the API
router = APIRouter()

# Define the prediction endpoint
@router.post("", summary="Predict the price of a property")
def predict(features: PropertyFeatures):
    print("⏩ Paso 1: Entrando al endpoint predict")

    try:
        print("⏩ Paso 2: Preparando datos")
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
        df = pd.DataFrame([data_dict])
        print("⏩ Paso 3: Datos convertidos a DataFrame")

        continuous_features = ['CONSTRUCTEDAREA', 'CADMAXBUILDINGFLOOR',
                               'DISTANCE_TO_CITY_CENTER', 'DISTANCE_TO_METRO', 'DISTANCE_TO_CASTELLANA', 'ROOMNUMBER',
                               'BATHNUMBER', 'FLOORCLEAN']
        df[continuous_features] = scaler.transform(df[continuous_features])
        print("⏩ Paso 4: Datos escalados")

        prediction = model.predict(df)
        print("⏩ Paso 5: Predicción realizada")

        return {"Predicted label": prediction[0]}

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
