# Predicting Rental Prices in Madrid

This is a personal project. The goal is to predict rental prices for properties in Madrid using Machine Learning techniques and present the results interactively through a dashboard.

---

##  Objectives

- Perform exploratory data analysis (EDA)
- Clean and preprocess real estate data
- Train and compare various prediction models (e.g., KNN, Random Forest)
- Evaluate model performance using metrics like RMSE and R²
- Create an interactive dashboard for visualization

---

## Technologies Used

- **Programming language:** Python
- **Data analysis:** Pandas, NumPy
- **Visualization:** Plotly, Seaborn, Matplotlib
- **Geographic visualization:** Folium
- **Machine Learning:** scikit-learn
- **Dashboard:** Streamlit, FastAPI, Pydantic
- **Development tools:** Jupyter Notebook, VSCode, PyCharm

---

##  Installation

Clone the repository and install dependencies in a virtual environment:

```bash
git clone https://github.com/marcosrequenagut/madrid_rental_prediction_ml
````

Virtual environment setup:

Create virtual environment
```bash
python -m venv .venv
```

Activate the virtual environment and install the required packages:
```bash
source .venv/Scripts/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Run Jupyter Notebooks
To explore the data and preprocessing steps:
jupyter notebook

## Viewing Saved Models in MLflow
To view the models saved during training, run the following command in the root of the project:

```bash
mlflow ui
```
This will start the MLflow UI, where you can track the models and their performance metrics interactively and download the model used.

## Running the Application manually
To run the app, execute the following commands from the root of the project in different bash terminals. Make sure that MLflow is also running.
Run the FastAPI server with hot reload to access the API documentation and support the Streamlit dashboard.

```bash
python -m uvicorn app.fastapi_api.app:app --reload --port 8001
```

If you prefer to manually restart the server after code changes (e.g., for production-like debugging), omit the --reload flag:

```bash
python -m uvicorn app.fastapi_api.app:app --port 8001
```

Launch the Streamlit dashboard for interactive property price prediction.
```bash
streamlit run app/streamlit_app/Home.py
```


## Data Sources
https://github.com/paezha/idealista18

## Running the Application with Docker
To run the application using Docker, execute the following command to build and launch the containers required for the different services: **MLflow**, **FastAPI**, and **Streamlit**.


```bash
docker-compose -f docker-compose-full.yml up --build
```

After running this command, you will be able to access three different interfaces:
- **MLFlow UI**:  http://localhost:5000. This interface allows you to view all user-generated experiments, as well as the trained models and the scaler used during preprocessing.
- **FastAPI**:  http://localhost:8001/docs. This provides interactive API documentation where you can test all available endpoints
- **Streamlit**: http://localhost:8501. This is an interactive dashboard where you can explore results, make predictions, and use various application features.

## Author
Juan Marcos Requena Gutiérrez
