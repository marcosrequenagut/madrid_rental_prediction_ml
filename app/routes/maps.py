import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import io
import os

from fastapi.responses import Response
from fastapi import APIRouter, HTTPException
from shapely import wkt

router = APIRouter()


@router.get("", response_class=Response, summary="Show the districts of Madrid on a map")
def show_map():
    """
    Show the districts of Madrid on a map.
    """
    try:
        # Path of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Path of the CSV file
        data_path = os.path.join(script_dir, '..', '..', 'data', 'Madrid_Districts_Polygons.csv')

        # Read the CSV file
        df = pd.read_csv(data_path, encoding='utf-8')

        df['geometry2'] = df['geometry'].apply(wkt.loads)

        map_df = gpd.GeoDataFrame(df, geometry = 'geometry2', crs='EPSG:4326')


        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the districts
        map_df.plot(column='DISTRICTS',
                    ax=ax,
                    cmap='tab20',
                    legend=True,
                    edgecolor='black'
                    )

        # Adjust the position of the legend
        leg = ax.get_legend()
        if leg:
            leg.set_bbox_to_anchor((1.05, 1))

        # Remove the axes
        ax.set_axis_off()

        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format = 'png', bbox_inches = 'tight', pad_inches = 0)
        plt.close(fig)
        buf.seek(0)

        return Response(content=buf.read(), media_type = "image/png")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
