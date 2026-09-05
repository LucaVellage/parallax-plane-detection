"""
Script exports matched geodataframe as GeoJsON
"""

import geopandas as gpd
from pathlib import Path
import datetime
from datetime import timedelta
import ee
import pandas as pd

from pipeline.db import repository
from pipeline.settings import get_settings

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='pipeline.s5_evaluation.export')


#---helpers---
def _prepare_gdf(gdf, image_id):
    """
    Prepares a scene's evaluation points for GeoJSON export/QGIS annotation. 
    Note: `id` (the evaluation_point primary key) is renamed to eval_point_id and kept in the export.
    """
    gdf = gdf.copy()

    if 'image_id' not in gdf.columns:
        gdf['image_id'] = image_id

    gdf = gdf.rename(columns={'id': 'eval_point_id'})

    gdf['cloud_pct'] = gdf['cloud_pct'].fillna(_load_cloud_pct(image_id))

    if 'outside_bounds' not in gdf.columns:
        gdf['outside_bounds'] = False
    else:
        gdf['outside_bounds'] = gdf['outside_bounds'].fillna(False)

    for col in ['match_distance_m', 'lat', 'lon',
                'baroaltitude', 'geoaltitude', 'speed_kmh',
                'angle_deg', 'residual_m']:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors='coerce').round(2)

    col_order = [
        'eval_point_id', 'image_id', 'source',
        'inspected', 'is_visible', 'is_flying', 'outside_bounds', 'near_bounds', 'seamline_duplicate', 'notes',
        'in_pipeline', 'in_adsb',
        'matched', 'match_distance_m',
        'matched_icao24', 'matched_callsign',
        'lat', 'lon',
        'icao24', 'callsign',
        'baroaltitude', 'geoaltitude',
        'speed_kmh', 'heading',
        'angle_deg', 'residual_m', 'candidate_id',
        'geometry', 'cloud_pct',
    ]
    col_order = [c for c in col_order if c in gdf.columns]
    extras    = [c for c in gdf.columns if c not in col_order]
    gdf       = gdf[col_order + extras]

    return gdf


def _export_single(gdf, image_id, output_dir):
    """
    Exports a single GeoDataFrame to GeoJSON
    """
    out_path = Path(output_dir) / f"{image_id}_eval.geojson"
    gdf.to_file(out_path, driver='GeoJSON')

    n_pipeline = len(gdf[gdf['source'] == 'pipeline'])
    n_adsb     = len(gdf[gdf['source'] == 'adsb'])
    n_matched  = len(gdf[gdf['matched'] == True])

    return out_path


def _load_cloud_pct(image_id):
    """
    Reads cloud cover percentage for a given image_id from the
    tile_scan_params cache table. Returns None if not found.
    """
    date, tile = image_id.split('_', 1)
    date_dashed = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    cached = repository.get_tile_scan_params(tile.lstrip('T'), date_dashed)
    return cached.get('cloud_pct') if cached else None


#---main---
def run_export(run_id, results, output_dir=None):
    """
    Exports each scene's matched evaluation points to GeoJSON for QGIS
    annotation. 
    
    results: dict mapping image_id -> GeoDataFrame (from evaluate.run_evaluation()).
    """
    output_dir = output_dir if output_dir is not None else str(get_settings().run_subdir(run_id, "step5_annotations"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    out_paths = []
    for image_id, gdf in results.items():
        gdf_clean = _prepare_gdf(gdf, image_id)
        out_path  = _export_single(gdf_clean, image_id, output_dir)
        out_paths.append(out_path)

    print(f"\nExported {len(out_paths)} GeoJSON files to {output_dir}")
    return out_paths


#export S-2 tile from GEE
def export_rgb_tile_to_drive(tile, date):
    """
    Exports true colour of a tile to Google Drive.
    - Function can be used to download image tiles for visual inspections. 
    - Step not integrated in end-to-end pipeline.
    - Downloads in GEE to connected Google drive from where tiles need to be downloaded manually
    """
    date_next = (datetime.datetime.strptime(date, '%Y-%m-%d') 
                 + timedelta(days=1)).strftime('%Y-%m-%d')
    
    image = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
        .filter(ee.Filter.eq('MGRS_TILE', tile))
        .filterDate(date, date_next)
        .first())
    
    rgb      = image.select(['B4', 'B3', 'B2'])
    filename = f"{date.replace('-','')}_{tile}_rgb"
    aoi      = image.geometry()
    native_crs = image.select('B3').projection().getInfo()['crs']

    task = ee.batch.Export.image.toDrive(
        image          = rgb,
        description    = filename,
        folder         = 's2_rgb',
        fileNamePrefix = filename,
        scale          = 10,
        region         = aoi,
        crs            = native_crs,
        maxPixels      = 1e13,
        fileFormat     = 'GeoTIFF'
    )
    task.start()
    print(f"Task ID:  {task.id}")
    print(f"Status:   {task.status()['state']}")
    print(f"Filename: {filename}.tif")
    print(f"Check progress at code.earthengine.google.com/tasks")



