import streamlit as st

def menu():
    # Determine if a user is logged in or not, then show the correct
    # navigation menu
    st.sidebar.page_link("pages/dashboard.py", label="Property Price Prediction")
    st.sidebar.page_link("pages/map_all_properties.py", label="Madrid Property Price Heatmap")
    st.sidebar.page_link("pages/map_properties_by_district.py", label="EExplore Similar Properties & Compare Market Prices")
