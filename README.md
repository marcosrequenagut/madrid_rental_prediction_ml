# Predicting Rental Prices in Madrid

This is a personal project. The goal is to predict rental prices for properties in Madrid using Machine Learning techniques and present the results interactively through a dashboard.

---

##  Objectives

- Perform exploratory data analysis (EDA)
- Clean and preprocess real estate data
- Train and compare various prediction models (e.g., KNN, Random Forest)
- Evaluate model performance using test data
- Create an interactive dashboard for visualization

---

## Technologies Used

- **Programming language:** Python
- **Data analysis:** Pandas, NumPy
- **Visualization:** Plotly, Seaborn, Matplotlib
- **Geographic visualization:** Folium
- **Machine Learning:** scikit-learn, TensorFlow
- **Dashboard:** Streamlit (or Dash)
- **Development tools:** Jupyter Notebook, VSCode, PyCharm

---

##  Installation

Clone the repository and install dependencies in a virtual environment:

```bash
git clone https://github.com/your-username/tfm-madrid-rental-prediction.git
cd tfm-madrid-rental-prediction
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
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

## Running the Application
To run the app, execute the following commands from the root of the project in different bash terminals:
```bash
streamlit run app/streamlit_app/dashboard.py
```
Launch the Streamlit dashboard for interactive property price prediction.

```bash
python -m uvicorn app.fastapi_api.app:app --reload --port 8001
```
Run the FastAPI server with hot reload to access the API documentation and support the Streamlit dashboard.

## Data Sources
https://github.com/paezha/idealista18

## Author
Juan Marcos Requena Gutiérrez
