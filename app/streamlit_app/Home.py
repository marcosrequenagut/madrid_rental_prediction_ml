import streamlit as st
from menu import menu

# URL of the API
api_url = "http://127.0.0.1:8001"

# Set wide layout
st.set_page_config(page_title="Madrid Housing Price App", layout="wide")

# Title and intro
st.title("🏡 Predicción de Precios de Viviendas en Madrid")

st.markdown("""
Bienvenido a la aplicación de análisis y predicción del mercado inmobiliario en la ciudad de **Madrid**.

Esta herramienta te permite **explorar, comparar y estimar precios de viviendas** a partir de datos reales extraídos del portal Idealista.

---

### 🔍 ¿Qué puedes hacer aquí?

#### 💶 Estimar el precio de una vivienda
Introduce las características principales (metros cuadrados, número de habitaciones, barrio, etc.) y obtén una estimación precisa del valor de mercado gracias a un modelo avanzado de machine learning.

#### 🗺️ Explorar propiedades en el mapa
Visualiza un **mapa interactivo** de Madrid con miles de viviendas reales disponibles, filtrables por distrito. Compara tu predicción con propiedades similares en su zona.

#### 📈 Analizar tendencias del mercado
Observa patrones de precios por barrio, evolución del valor por metro cuadrado y detecta zonas emergentes o sobrevaloradas.

---

💡 Utiliza el **menú lateral** para navegar entre las secciones disponibles y comenzar tu análisis personalizado del mercado inmobiliario madrileño.
""")

# Render the dynamic menu
menu()
