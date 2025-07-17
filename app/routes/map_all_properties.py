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

@router.get("", response_class=HTMLResponse, summary="Mapa interactivo de precios en Madrid")
def show_interactive_map():
    """
    Devuelve un mapa interactivo de Madrid con:
    - Distritos delimitados
    - Mapa de calor (HeatMap) según precios
    - Marcadores agrupados con precios

    :return: HTML con el mapa interactivo
    :raises HTTPException 404: Si faltan archivos
    :raises HTTPException 500: Si ocurre un error de procesamiento
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Cargar distritos
        districts_path = os.path.join(script_dir, '..', '..', 'data/new_data', 'madrid-districts.geojson.txt')
        gdf_districts = gpd.read_file(districts_path)

        # Elimina columnas que pueden causar problemas con JSON
        gdf_districts = gdf_districts.drop(columns=['created_at', 'updated_at'], errors='ignore')

        # Cargar propiedades
        properties_path = os.path.join(script_dir, '..', '..', 'data/new_data', 'EDA_MADRID_SCALED_Geometry_Column.csv')
        df_properties = pd.read_csv(properties_path, encoding='utf-8')
        df_properties['GEOMETRY'] = df_properties['GEOMETRY'].apply(wkt.loads)

        # Limpiamos el dataframe, quedándonos solo las columnas necesarias
        df_clean = df_properties[['PRICE', 'GEOMETRY']].copy()

        # Renombra la geometría a 'geometry', que es el nombre por defecto que espera GeoDataFrame
        df_subset = df_clean.rename(columns={'GEOMETRY': 'geometry'})

        gdf_properties = gpd.GeoDataFrame(df_subset, geometry='geometry', crs='EPSG:4326')

        # Crear mapa base centrado en Madrid
        m = folium.Map(location=[40.4168, -3.7038], zoom_start=12, tiles='CartoDB positron')

        # Añadir distritos con nombre y resaltado al pasar el ratón
        folium.GeoJson(
            gdf_districts,
            name="Distritos",
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
                aliases=["Distrito:"],
                sticky=True,
                opacity=0.8,
                direction='top'
            )
        ).add_to(m)

        # Escalar colores por precio (de azul a rojo)
        min_price = gdf_properties['PRICE'].min()
        #max_price = gdf_properties['PRICE'].max()
        max_price = 1000000

        colormap = cm.LinearColormap(
            colors=['blue', 'lightblue', 'yellow', 'orange', 'red'],
            vmin=min_price,
            vmax=max_price,
            caption='Precio de las propiedades (€)'
        )

        # Añadir cada propiedad como CircleMarker coloreado por precio
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
                popup=f"Precio: €{int(price):,}".replace(",", ".")
            ).add_to(m)

        # Añadir leyenda fija
        colormap.add_to(m)

        # Añadir control de capas
        #folium.LayerControl().add_to(m)

        # Devolver HTML embebido
        return HTMLResponse(content=m.get_root().render())

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocurrió un error: {str(e)}")
