import folium
import pandas as pd
import os
import geopandas as gpd

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from shapely import wkt
from typing import Optional
from folium import IFrame



# Define the APIRouter instance
router = APIRouter()

@router.get("", response_class=Response)
def map(
        district: Optional[str] = Query(None, description="District name to filter by"),
        room_number: Optional[int] = Query(None, description="Number of rooms to filter by"),
        bathroom_number: Optional[int] = Query(None, description="Number of bathrooms to filter by"),
        price_min: Optional[int] = Query(None, description="Minimum price to filter by"),
        price_max: Optional[int] = Query(None, description="Maximum price to filter by"),
        constructed_area_min: Optional[int] = Query(None, description="Minimum constructed area to filter by"),
        constructed_area_max: Optional[int] = Query(None, description="Maximum constructed area to filter by"),
       ):
    """
    Generate an interactive HTML map of Madrid showing real estate properties filtered by district
    and optional property attributes.

    ## Features
    - Loads geographic district boundaries from a CSV (with WKT geometries).
    - Loads property listings from a separate CSV, also with geometries.
    - Filters the properties by:
        - District name (required)
        - Number of rooms (exact match)
        - Number of bathrooms (exact match)
        - Price range (min and/or max)
        - Constructed area range (min and/or max)
    - Performs a **spatial join** to match properties inside the selected district.
    - Uses **Folium** to render an interactive web map with:
        - District polygons
        - Red circle markers for each property
        - Interactive popups showing property details

    ## Parameters
    :param district: (Optional[str]) Name of the district to filter by. Must match district names in the source CSV.
    :param room_number: (Optional[int]) Exact number of rooms to match.
    :param bathroom_number: (Optional[int]) Exact number of bathrooms to match.
    :param price_min: (Optional[int]) Minimum property price in euros.
    :param price_max: (Optional[int]) Maximum property price in euros.
    :param constructed_area_min: (Optional[int]) Minimum constructed area in square meters.
    :param constructed_area_max: (Optional[int]) Maximum constructed area in square meters.

    ## Returns
    - HTML response rendering the Folium map with the filtered properties.

    ## Raises
    - HTTPException 404: If the district is not found.
    - HTTPException 500: For any unexpected errors during processing.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Load districts
        districts_path = os.path.join(script_dir, '..', '..', 'data/old_data', 'Madrid_Districts_Polygons.csv')
        df_districts = pd.read_csv(districts_path, encoding='utf-8')
        df_districts['geometry'] = df_districts['geometry'].apply(wkt.loads)
        gdf_districts = gpd.GeoDataFrame(df_districts, geometry='geometry', crs='EPSG:4326')

        # Load properties
        properties_path = os.path.join(script_dir, '..', '..', 'data/new_data', 'EDA_MADRID_NOT_SCALED_DISTRICTS.csv')
        df_properties = pd.read_csv(properties_path, encoding='utf-8')
        df_properties['GEOMETRY'] = df_properties['GEOMETRY'].apply(wkt.loads)

        # Ensure that the columns have the correct data types
        df_properties['PRICE'] = pd.to_numeric(df_properties['PRICE'], errors='coerce')
        df_properties['ROOMNUMBER'] = pd.to_numeric(df_properties['ROOMNUMBER'], errors='coerce')
        df_properties['BATHNUMBER'] = pd.to_numeric(df_properties['BATHNUMBER'], errors='coerce')

        gdf_properties = gpd.GeoDataFrame(
            df_properties[[
                'PRICE', 'UNITPRICE', 'ROOMNUMBER',
                'BATHNUMBER','CADCONSTRUCTIONYEAR',
                'GEOMETRY', 'CONSTRUCTEDAREA'
            ]], geometry='GEOMETRY', crs='EPSG:4326')

        # Create a Folium map centered on Madrid
        m = folium.Map(location=[40.4168, -3.7038], zoom_start=12)

        gdf_filtered = gdf_properties.copy()

        # Filter properties based on query parameters
        if room_number is not None:
            gdf_filtered = gdf_filtered[gdf_filtered['ROOMNUMBER'] == room_number]

        if bathroom_number is not None:
            gdf_filtered = gdf_filtered[gdf_filtered['BATHNUMBER'] == bathroom_number]

        if price_min is not None:
            gdf_filtered = gdf_filtered[gdf_filtered['PRICE'] >= price_min]

        if price_max is not None:
            gdf_filtered = gdf_filtered[gdf_filtered['PRICE'] <= price_max]

        if constructed_area_min is not None:
            gdf_filtered = gdf_filtered[gdf_filtered['CONSTRUCTEDAREA'] >= constructed_area_min]

        if constructed_area_max is not None:
            gdf_filtered = gdf_filtered[gdf_filtered['CONSTRUCTEDAREA'] <= constructed_area_max]

        # Filter districts by name:
        selected_district = gdf_districts[gdf_districts['DISTRICTS'] == district]

        if selected_district.empty:
            raise HTTPException(status_code=404, detail="District not found")

        # Make the spatial intersection
        gdf_filtered = gpd.sjoin(gdf_filtered, selected_district, predicate="within")

        # Add districts to the map
        folium.GeoJson(
            selected_district,
            name='Districts',
            # Define the general style of the map
            style_function=lambda feature: {
                'fillColor': 'lightblue',
                'color': 'black',
                'weight': 2,
                'fillOpacity': 0.2,
            },
            # Define the style when hovering over a district with the mouse
            highlight_function=lambda feature: {
                'fillColor': 'lightblue',
                'color': 'blue',
                'weight': 2,
                'fillOpacity': 0.6,
            },
            # Add a tooltip to show the district name when the mouse hovers over it
            tooltip=folium.GeoJsonTooltip(fields=['DISTRICTS'], aliases=['District'])
        ).add_to(m)

        # Create a popup
        for _, row in gdf_filtered.iterrows():
            lat, lon = row['GEOMETRY'].x, row['GEOMETRY'].y
            popup_html = f"""
                <div style="font-size: 13px; padding: 5px;">
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr><td><b>Price:</b></td><td>{row['PRICE']:.0f} €</td></tr>
                        <tr><td><b>Price/m²:</b></td><td>{row['UNITPRICE']:.0f} €/m²</td></tr>
                        <tr><td><b>Constructed Area:</b></td><td>{row['CONSTRUCTEDAREA']} m²</td></tr>
                        <tr><td><b>Rooms:</b></td><td>{row['ROOMNUMBER']}</td></tr>
                        <tr><td><b>Bathrooms:</b></td><td>{row['BATHNUMBER']}</td></tr>
                        <tr><td><b>Construction Year:</b></td><td>{row['CADCONSTRUCTIONYEAR']}</td></tr>
                    </table>
                </div>
            """

            iframe = IFrame(popup_html, width=250, height=150)
            popup = folium.Popup(iframe, max_width=300)

            # Add properties to the map
            folium.CircleMarker(
                location=[lon, lat],
                radius=5,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.6,
                popup=popup,
            ).add_to(m)


        # Add layer control
        folium.LayerControl().add_to(m)
        return Response(content=m._repr_html_(), media_type="text/html")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
