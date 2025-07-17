import pandas as pd
import geopandas as gpd
import folium
import os
import branca.colormap as cm

from folium.plugins import HeatMap, MarkerCluster
from shapely import wkt
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

router = APIRouter()

@router.get("", response_class=HTMLResponse, summary="Interactive Price Map of Madrid")
def show_interactive_map():
    """
    Returns an interactive map of Madrid with:
    - District boundaries
    - Price-colored markers
    - Fixed color legend

    :return: HTML with the interactive map
    :raises HTTPException 404: If files are missing
    :raises HTTPException 500: If a processing error occurs
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Load district boundaries
        districts_path = os.path.join(script_dir, '..', '..', 'data/new_data', 'madrid-districts.geojson.txt')
        gdf_districts = gpd.read_file(districts_path)

        # Drop unnecessary columns that could cause JSON issues
        gdf_districts = gdf_districts.drop(columns=['created_at', 'updated_at'], errors='ignore')

        # Load property data
        properties_path = os.path.join(script_dir, '..', '..', 'data/new_data', 'EDA_MADRID_SCALED_Geometry_Column.csv')
        df_properties = pd.read_csv(properties_path, encoding='utf-8')
        df_properties['GEOMETRY'] = df_properties['GEOMETRY'].apply(wkt.loads)

        # Keep only necessary columns
        df_clean = df_properties[['PRICE', 'GEOMETRY']].copy()

        # Rename geometry column to 'geometry' as expected by GeoDataFrame
        df_subset = df_clean.rename(columns={'GEOMETRY': 'geometry'})

        gdf_properties = gpd.GeoDataFrame(df_subset, geometry='geometry', crs='EPSG:4326')

        # Create base map centered in Madrid
        m = folium.Map(location=[40.4168, -3.7038], zoom_start=12, tiles='CartoDB positron')

        # Add districts with name tooltip and hover highlight
        folium.GeoJson(
            gdf_districts,
            name="Districts",
            style_function=lambda x: {
                "fillColor": "gray",
                "color": "black",
                "weight": 1,
                "fillOpacity": 0.1
            },
            highlight_function=lambda x: {
                "color": "black",
                "weight": 3,
                "fillOpacity": 0.2
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["name"],
                aliases=["District:"],
                sticky=True,
                opacity=0.8,
                direction='top'
            )
        ).add_to(m)

        # Define color scale by price (from blue to red)
        min_price = gdf_properties['PRICE'].min()
        max_price = 1000000  # Optional cap to limit color scaling

        colormap = cm.LinearColormap(
            colors=['blue', 'lightblue', 'yellow', 'orange', 'red'],
            vmin=min_price,
            vmax=max_price,
            caption='Property Prices (€)'
        )

        # Add each property as a CircleMarker colored by price
        for _, row in gdf_properties.iterrows():
            price = row.PRICE
            location = [row.geometry.y, row.geometry.x]
            folium.CircleMarker(
                location=location,
                radius=1,
                color=colormap(price),
                fill=True,
                fill_color=colormap(price),
                fill_opacity=0.05,
                popup=f"Price: €{int(price):,}".replace(",", ".")
            ).add_to(m)

        # Add fixed color legend
        colormap.add_to(m)

        # Return embedded HTML
        return HTMLResponse(content=m.get_root().render())

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
