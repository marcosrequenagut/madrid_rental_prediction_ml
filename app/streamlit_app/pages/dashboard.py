import streamlit as st
import requests

# URL of the api
api_url = "http://127.0.0.1:8001"

st.title("Find out how much your property is worth!")

st.write("Introduce the required features:")

# Inputs
number_of_bathrooms = st.number_input("Number of bathrooms", min_value = 1, max_value = 8, value = 2)
number_of_rooms = st.number_input("Number of rooms", min_value = 1, max_value = 20, value = 3)
constructed_area = st.number_input("Constructed area (m^2)", min_value = 20, max_value = 985, value = 100)
distance_to_city_center = st.number_input("Distance to city center (km)", min_value=0.0, max_value=50.0, value=5.0)
distance_to_city_metro = st.number_input("Distance to metro (km)", min_value=0.0, max_value=50.0, value=5.0)
distance_to_city_castellana = st.number_input("Distance to castellana (km)", min_value=0.0, max_value=50.0, value=5.0)
constructed_year = st.number_input("Constructed year", min_value=1700, max_value=2025, value=2010)
floor = st.number_input("Floor", min_value=0, max_value=40, value=4)
has_terrace = st.radio("Has terrace?", ("Yes", "No"))
is_parkingspace_included = st.radio("Is parking space included in price?", ("Yes", "No"))
has_swimming_pool = st.radio("Has swimming pool?", ("Yes", "No"))
is_top_floor = st.radio("Is it on the top floor?", ("Yes", "No"))

# Using the API to get the prediction
endpoint_predict = "/api/v1/predict"
api_url_endpoint_predict = api_url + endpoint_predict

# Button to make the prediction
if st.button("Predict"):
    # Prepare the input data
    input_data = {
        "constructed_area": constructed_area,
        "has_terrace": 1 if has_terrace == "Yes" else 0,
        "is_parkingspace_included": 1 if is_parkingspace_included == "Yes" else 0,
        "number_of_rooms": number_of_rooms,
        "number_of_bathrooms": number_of_bathrooms,
        "has_swimming_pool": 1 if has_swimming_pool == "Yes" else 0,
        "is_top_floor": 1 if is_top_floor == "Yes" else 0,
        "distance_to_city_center": distance_to_city_center,
        "distance_to_city_metro": distance_to_city_metro,
        "distance_to_city_castellana": distance_to_city_castellana,
        "constructed_year": constructed_year,
        "floorclean": floor,
    }

    try:
        # Make the API prediction request
        response = requests.post(api_url_endpoint_predict, json=input_data)

        # Check if the request was successful
        if response.status_code == 200:
            prediction = response.json().get("Predicted label")
            st.success(f"The predicted price of the property is: {prediction:.2f} €")
        else:
            st.error("Error in the prediction request")
            st.error(f"Error: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred when connecting with the API: {e}")
