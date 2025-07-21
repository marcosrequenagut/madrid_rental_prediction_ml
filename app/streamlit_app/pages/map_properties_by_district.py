import streamlit as st
import requests
import os
import json
import streamlit.components.v1 as components

# URL of the API
api_url = "http://127.0.0.1:8001"

st.title("🗺️ Visualization of the properties in Madrid by district")

# Use the API to get the map
endpoint_map = "/api/v1/map_properties_by_district"
api_url_endpoint_map = api_url + endpoint_map

base_dir = os.path.dirname(__file__)
json_path = os.path.abspath(os.path.join(base_dir, '..', '..', '..', 'data', 'new_data', 'districts_and_neighborhood.json'))

with open(json_path, 'r', encoding='utf-8') as f:
    districts_neighborhoods_json = json.load(f)

# Ask the user to select the district to filter
districts = list(districts_neighborhoods_json.keys())
selected_district = st.selectbox("Select a district:", options=districts)

# Ask the user to select some features to filter
# Rooms
filter_rooms = st.checkbox("Filter by number of rooms")
if filter_rooms:
    room_number = st.number_input("Number of rooms", min_value=1, max_value=20, value=3)
else:
    room_number = None

# Bathrooms
filter_bathrooms = st.checkbox("Filter by number of bathrooms")
if filter_bathrooms:
    bathroom_number = st.number_input("Number of bathrooms", min_value=1, max_value=8, value=3)
else:
    bathroom_number = None

# Price
filter_price = st.checkbox("Filter by property price")
if filter_price:
    # Price range, with min and max that you can adjust according to your dataset
    price_range = st.slider(
        "Total price range(€)",
        min_value=30000,
        max_value=3000000,
        value=(100000, 500000),  # Initial value: a default range
        step=1000
    )
else:
    price_range = None

# Constructed area
constructed_area = st.checkbox("Filter by constructed area")
if constructed_area:
    # Constructed area range, with min and max that can be adjusted according to the dataset
    constructed_area_range = st.slider(
        "Constructed area range(€)",
        min_value=20,
        max_value=2000,
        value=(80, 120),  # Initial value: a default range
        step=5
    )
else:
    constructed_area_range = None


params = {"district": selected_district}

if room_number is not None:
    params["room_number"] = room_number
if bathroom_number is not None:
    params["bathroom_number"] = bathroom_number
if price_range is not None:
    params["price_min"] = price_range[0]
    params["price_max"] = price_range[1]
if constructed_area_range is not None:
    params["constructed_area_min"] = constructed_area_range[0]
    params["constructed_area_max"] = constructed_area_range[1]


# Build the URL only with valid params
map_url = requests.Request("GET", api_url_endpoint_map, params=params).prepare().url
try:
    response = requests.get(map_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Display the map
        map = response.text
        components.html(map, height=800)
    else:
        st.error("Error in the map request")
        st.error(f"Error: {response.status_code} - {response.text}")

except requests.exceptions.RequestException as e:
    st.error(f"An error occurred when connecting with the API: {e}")

