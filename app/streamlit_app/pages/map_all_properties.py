import streamlit as st
import streamlit.components.v1 as components
import requests

# URL of the API
api_url = "http://127.0.0.1:8001"

st.title("📊 Visualization of the properties in Madrid")

# Use the API to get the map
endpoint_properties = "/api/v1/map_all_properties"
api_url_endpoint_properties = api_url + endpoint_properties

try:
    # Make the API request to get the map of Madrid
    response = requests.get(api_url_endpoint_properties)

    # Check if the request was successful
    if response.status_code == 200:
        # ✅ Display the interactive folium map as HTML
        components.html(response.content.decode("utf-8"), height=600, scrolling=True)
    else:
        st.error("Error in the properties request")
        st.error(f"Error: {response.status_code} - {response.text}")

except requests.exceptions.RequestException as e:
    st.error(f"An error occurred when connecting with the API: {e}")
