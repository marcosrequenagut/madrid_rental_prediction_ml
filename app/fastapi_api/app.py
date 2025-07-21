import os
import logging

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from app.routes import predict, map_all_properties, map_properties_by_district


app = FastAPI(
    title = "Predict Rental Prices in Madrid",
    description = "A RESTful API for predicting property prices in Madrid based on various features.",
    version="0.1.0",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
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
    <title>Madrid Rental Price Prediction</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f9f9f9;
            margin: 0;
            padding: 0;
            color: #333;
        }

        header {
            background-color: #004aad;
            color: white;
            padding: 30px 20px;
            text-align: center;
        }

        header h1 {
            margin: 0;
            font-size: 2.5em;
        }

        main {
            padding: 40px 20px;
            max-width: 900px;
            margin: 0 auto;
        }

        section {
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }

        section h2 {
            color: #004aad;
            margin-top: 0;
        }

        a {
            color: #004aad;
            text-decoration: none;
        }

        a:hover {
            text-decoration: underline;
        }

        footer {
            text-align: center;
            padding: 20px;
            background-color: #eee;
            font-size: 0.9em;
        }
    </style>
</head>
<body>

    <header>
        <h1>Madrid Rental Price Prediction API</h1>
        <p>Discover the estimated rental value of your property in Madrid</p>
    </header>

    <main>

        <section>
            <h2>What can you do with this app?</h2>
            <p>This platform lets you estimate your property's rental value in Madrid, explore prices in different districts, and visualize data on interactive maps.</p>
        </section>

        <section>
            <h2>1. Property Valuation</h2>
            <p>Enter your home's features (number of rooms, area, neighborhood, etc.) and get an accurate rental price estimate.</p>
        </section>

        <section>
            <h2>2. General Property Explorer</h2>
            <p>View all appraised properties. Discover average prices by neighborhood and compare different areas across Madrid.</p>
        </section>

        <section>
            <h2>3. District-Based Analysis</h2>
            <p>Filter by district to focus on properties that interest you. Each one shows key attributes used in the valuation along with their values.</p>
        </section>

        <section>
            <h2>4. Interactive Madrid Map</h2>
            <p>Explore a detailed map of Madrid to better understand its district layout and geographical context.</p>
        </section>

        <section>
            <h2>Documentation & Source Code</h2>
            <p>Check out the <a href="/docs">API documentation</a> to learn how to use the available endpoints.</p>
            <p>The project is available on <a href="https://github.com/marcosrequenagut/madrid_rental_prediction_ml" target="_blank">GitHub</a>.</p>
        </section>

    </main>

    <footer>
        &copy; 2025 Madrid Rental Prediction | Developed by marcosrequenagut
    </footer>

</body>
</html>"""

@app.get("/map", response_class=HTMLResponse, summary="Interactive map with property points")
async def serve_map():
    file_path = os.path.join("app", "static", "map.html")
    return FileResponse(file_path)

app.include_router(
    predict.router,
    prefix="/api/v1/predict",
)


app.include_router(
    map_all_properties.router,
    prefix="/api/v1/map_all_properties",
)

app.include_router(
    map_properties_by_district.router,
    prefix="/api/v1/map_properties_by_district",
)

