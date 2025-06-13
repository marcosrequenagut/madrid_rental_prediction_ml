import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import io
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from shapely import wkt

from app.streamlit_app.pages.map_properties_by_district import districts

router = APIRouter()

@router.get("", response_class=Response, summary="Combined map: districts and properties with prices")
def show_combined_map():
    """
    Returns a combined map of Madrid's districts and real estate properties.

    - Loads a GeoJSON file containing the geometries of Madrid districts.
    - Loads a CSV file with real estate properties and their prices.

    :returns: the map as a PNG image in the HTTP response.

    Raises:
        - 404 if data files are not found.
        - 500 if any other error occurs during processing.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Load the map of districts
        data_path_geojson = os.path.join(script_dir, '..', '..', 'data/new_data', 'madrid-districts.geojson.txt')
        gdf = gpd.read_file(data_path_geojson)
        df_districts = pd.DataFrame(gdf)
        gdf_districts = gpd.GeoDataFrame(df_districts, geometry='geometry', crs='EPSG:4326')

        # Load the map of properties
        properties_path = os.path.join(script_dir, '..', '..', 'data/new_data', 'EDA_MADRID_SCALED_Geometry_Column.csv')
        df_properties = pd.read_csv(properties_path, encoding='utf-8')
        df_properties['GEOMETRY'] = df_properties['GEOMETRY'].apply(wkt.loads)
        gdf_properties = gpd.GeoDataFrame(df_properties[['PRICE', 'GEOMETRY']], geometry='GEOMETRY', crs='EPSG:4326')

        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the districts
        gdf_districts.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.9)

        # Plot properties with its price
        gdf_properties.plot(
            column='PRICE',
            ax=ax,
            cmap='plasma',
            legend=True,
            markersize=10,
            alpha=0.5,
            vmin=100_000,
            vmax=500_000
        )

        # Remove the axis
        ax.set_axis_off()

        # Save a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        return Response(content=buf.read(), media_type="image/png")

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
