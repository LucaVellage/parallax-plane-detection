"""
Script enables users to draw a polygon on an interactive map to determine 
area of interest (AOI) which will be passed to the detection pipeline
"""

import ee
import geemap 


def GEE_map():
    """
    Loads lightweight Google Earth map with Roadmap basemap.
    Basemap can be changed if e.g. "SATELLITE" is preferred.
    """
    m = geemap.Map(
        center=(51.0, 10.0),
        zoom=4,
        basemap='ROADMAP'
    )

    #reatining only basemap layer
    m.layers = m.layers[:2]

    #drawing tool
    m.add_drawn_features = True
    draw_control = m.draw_control
    draw_control.polyline = {}
    draw_control.polygon = {}
    draw_control.circlemarker = {}
    draw_control.marker = {}
    #keeping only rectangle shape
    draw_control.rectangle = {'shapeOptions': {'color': '#e63946'}}

    return m


def get_aoi(m):
    """
    Function extracts geocoordinates from drawn polygon
    """
    if not m.user_roi:
        print('No AOI selected. Please draw a rectangle on the map.')
        return None
    aoi = m.user_roi
    return aoi
    

def get_bbox(aoi):
    bounds = aoi.getInfo()['coordinates'][0]
    lons = [c[0] for c in bounds]
    lats = [c[1] for c in bounds]

    bbox = {
        "aoi_lon_min": min(lons),
        "aoi_lon_max": max(lons),
        "aoi_lat_min": min(lats),
        "aoi_lat_max": max(lats),
    }

    return bbox

def aoi_summary(aoi):
    bbox = get_bbox(aoi)
    print(f'AOI CRS  : {aoi.projection().getInfo()["crs"]}')
    print(f'West     : {bbox["aoi_lon_min"]:.4f}')
    print(f'East     : {bbox["aoi_lon_max"]:.4f}')
    print(f'South    : {bbox["aoi_lat_min"]:.4f}')
    print(f'North    : {bbox["aoi_lat_max"]:.4f}')






