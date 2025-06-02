from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.routes import predict, maps

print("loaded predict.py")
app = FastAPI(
    title = "Predict Rental Prices in Madrid",
    description = "A RESTful API for predicting property prices in Madrid based on various features.",
    version="0.1.0",
)

@app.get(
    "/",
    response_class=HTMLResponse,
    summary="Beautiful front page for the data service",
    description="This is the front page of the data service. It provides a welcome message and a link to the API documentation.",
    tags=["front-page"],
)


async def root_page():
    return """<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Madrid Rental Price Prediction API</title>
    </head>
    <body>
        <h1>Welcome to the Madrid Rental Price Prediction API</h1>
        <p>This service provides rental price predictions for properties in Madrid based on their features.</p>
        <p>Check out the API documentation <a href="/docs">here</a> to learn how to use the available endpoints.</p>
        <p>Visit our project repository on GitHub <a href="https://github.com/marcosrequenagut/madrid_rental_prediction_ml">here</a>.</p>
    </body>
</html>"""

app.include_router(
    predict.router,
    prefix="/api/v1/predict",
)

app.include_router(
    maps.router,
    prefix="/api/v1/maps",
)