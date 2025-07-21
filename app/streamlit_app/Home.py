import streamlit as st
from menu import menu

# API endpoint
api_url = "http://127.0.0.1:8001"

# Set wide layout for the app
st.set_page_config(page_title="Madrid Housing Price App", layout="wide")

# Title and introduction
st.title("🏡 Madrid Housing Price Prediction")

st.markdown("""
Welcome to the interactive application for real estate price analysis and prediction in the city of **Madrid**.

This tool enables you to **explore, compare, and estimate housing prices** using real data extracted from the Idealista property portal.

---

### 🔍 What can you do with this app?

#### 💶 Estimate the price of a property
Enter the main characteristics of a property (square meters, number of rooms, district, etc.) and receive a market value estimation powered by a trained machine learning model.

#### 🗺️ Explore properties on the map
Access an **interactive map of Madrid** displaying thousands of real property listings. Use filters such as district, number of rooms or bathrooms, square meters, price, etc, to narrow down the properties shown. You can compare your estimation against similar listings in the selected district.  
**Note:** Filtering by district is mandatory, while the remaining filters are optional.

#### 📈 Analyze market trends
Discover pricing patterns across neighborhoods, monitor the evolution of price per square meter, and identify emerging or overvalued areas. You’ll find a **global interactive heatmap** highlighting the most and least expensive zones in Madrid. Draw your own insights based on data-driven evidence!.

---

💡 Use the **sidebar menu** to navigate between sections and start your customized analysis of the Madrid housing market.
""")

# Render the navigation menu
menu()
