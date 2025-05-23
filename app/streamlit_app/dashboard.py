import streamlit as st

URL = " http://192.168.18.107:8501"

st.title("Find out how much your property is worth!")

st.write("Introduce the required features:")

# Inputs
number_of_bathrooms = st.number_input("Number of bathrooms", min_value = 1, max_value = 8, value = 2)
number_of_rooms = st.number_input("Number of rooms", min_value = 1, max_value = 20, value = 3)
constructed_area = st.number_input("Constructed area (m^2)", min_value = 65, max_value = 985, value = 100)

# Yes/No CheckBox for HASTERRACE feature
has_terrace = st.radio("Has terrace?", ("Yes", "No"))

