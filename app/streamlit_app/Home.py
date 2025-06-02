import streamlit as st
import requests

# URL of the api
api_url = "http://127.0.0.1:8001"

st.title("🏠 Madrid Housing Price Prediction")

st.markdown("""
Welcome to the real estate prediction for the cuty of **Madrid**.
Here you can:

- 💶 **Estimate the price of a property** based on its characteristics (area, bathrooms, location, etc.).
- 🗺️ **Explore interactive maps** with prices by neighborhoods and districts.
- 📊 **Visualize market trends** in the city.

Use the sidebar menu to get started 👉""")

