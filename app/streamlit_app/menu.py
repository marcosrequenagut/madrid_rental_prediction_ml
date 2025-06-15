import streamlit as st

def menu():
    # Determine if a user is logged in or not, then show the correct
    # navigation menu
    st.sidebar.page_link("pages/dashboard.py", label="Estimate My Rent")
    st.sidebar.page_link("pages/map_all_properties.py", label="Explore Rent Prices")
    st.sidebar.page_link("pages/map_properties_by_district.py", label="Explore Similar Properties")
    st.sidebar.page_link("pages/Maps.py", label="Map of Madrid")
