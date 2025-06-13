import folium
import pandas as pd
import os
import geopandas as gpd

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from shapely import wkt
from typing import Optional
from folium import IFrame

router = APIRouter()

@router.get("", response_class=Response)
def map(district: Optional[str] = Query(None, description="District name to filter by")):
    """
    Generate an interactive HTML map of Madrid displaying real estate properties filtered by a specific district.

    - Loads a GeoJSON file with the geometry of Madrid's districts.
    - Loads a CSV file containing property data with geographic information.
    - Filters the districts based on the query parameter `district`.
    - Performs a spatial join to select only the properties located within the selected district.
    - Uses Folium to generate an interactive map.
    - Each property is represented as a red circle marker.
    - Clicking a marker opens a popup with detailed information.


    :param district: Optional; the name of the district to filter by
    :type district: str
    :return: HTML content representing the interactive map with properties in the selected district
    :rtype: fastapi.responses.Response
    :raises HTTPException 404: If the district is not found
    :raises HTTPException 500: If an unexpected error occurs
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
        gdf_properties = gpd.GeoDataFrame(
            df_properties[[
                'PRICE', 'UNITPRICE', 'ROOMNUMBER',
                'BATHNUMBER','CADCONSTRUCTIONYEAR',
                'GEOMETRY', 'CONSTRUCTEDAREA'
            ]], geometry='GEOMETRY', crs='EPSG:4326')

        m = folium.Map(location=[40.4168, -3.7038], zoom_start=12)

        # Filter districts by name:
        selected_district = gdf_districts[gdf_districts['DISTRICTS'] == district]

        if selected_district.empty:
            raise HTTPException(status_code=404, detail="District not found")

        # Make the spatial intersection
        gdf_filtered = gpd.sjoin(gdf_properties, selected_district, predicate="within")

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
