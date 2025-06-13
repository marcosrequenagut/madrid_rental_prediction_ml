import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import io
import os
import matplotlib.patches as mpatches
import matplotlib.cm as cm

from fastapi.responses import Response
from fastapi import APIRouter, HTTPException


router = APIRouter()


@router.get("", response_class=Response, summary="Show the districts of Madrid on a map")
def show_map():
    """
    Generate a static map showing the districts of Madrid with labels and a custom legend.

    - Loads a GeoJSON file containing the geometry and metadata of Madrid's districts.
    - Transforms the data into a GeoDataFrame with district names and IDs.
    - Plots the districts using a color map (`tab20b`), differentiating them by `DistrictID`.
    - Adds the district ID as a text label at each district's centroid.
    - Creates a custom legend showing the district ID and its corresponding name.
    - Adjusts the layout so the legend doesn't overlap with the map.
    - Returns the final plot as a PNG image.

    :returns:
        PNG image in the HTTP response displaying the map of Madrid's districts with labeled IDs and a legend.

    :raises 404 if the GeoJSON file is not found.
    :raises 500 if any other error occurs during the processing or plotting.
"""
    try:
        # Path of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Path for district geojson
        data_path_geojson = os.path.join(script_dir, '..', '..', 'data/new_data', 'madrid-districts.geojson.txt')

        gdf = gpd.read_file(data_path_geojson)

        df = pd.DataFrame({
            'DISTRICTS': gdf['name'],
            'DistrictID': gdf['cartodb_id'],
            'GEOMETRY': gdf['geometry'],
        })

        # Map DistrictID (number) to district name
        map_district_number = dict(zip(df['DistrictID'], df['DISTRICTS']))

        map_df = gpd.GeoDataFrame(df, geometry='GEOMETRY', crs='EPSG:4326')

        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 10))

        #ax.legend(df['DistrictID'])

        # Plot the districts
        map_df.plot(column='DistrictID',
                    ax=ax,
                    cmap='tab20b',
                    legend=False,
                    edgecolor=(0, 0, 0, 0.3), # Black lines with 30% opacity
                    linewidth=1
                    )

        # Add the DistricID number into each district on the map
        for _, row in map_df.iterrows():
            centroid = row['GEOMETRY'].centroid
            ax.text(centroid.x, centroid.y, str(row['DistrictID']),
                    ha='center', va='center', fontsize=8, fontweight='bold', color='black')

        # Create a custom legend with DistrictID and district name
        unique_ids = sorted(map_df['DistrictID'].unique())
        cmap = plt.get_cmap('tab20b', len(unique_ids))

        # Add the names on the legend
        patches = []
        for i, district_id in enumerate(unique_ids):
            color = cmap(i)
            label = f"{district_id}: {map_district_number[district_id]}"
            patches.append(mpatches.Patch(color=color, label=label))

        # Adjust the position of the legend
        ax.legend(handles=patches, title="Districts", bbox_to_anchor=(1.05, 1), loc='upper left')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])  # move the axis to the left


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
