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

districts = list(districts_neighborhoods_json.keys())
selected_district = st.selectbox("Select a district:", options=districts)

# Construct the URL for the map request
map_url = f"{api_url_endpoint_map}?district={selected_district}"


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

