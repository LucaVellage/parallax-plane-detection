"""
Script implements candidate screening in GEE. 
Handles export limits (i.e. large date range) + large areas (e.g. if aoi covers too large area for export)
"""

import ee 
import geemap
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import pyproj
import tifffile
import pyproj
import datetime
from datetime import timedelta
import rasterio
from rasterio.merge import merge

from pipeline.db import repository
from pipeline.settings import get_settings
from pipeline.config import REFL_THRESHOLD

from pipeline.utils.io import extract_geotransform, transform_coords, epsg_from_tile, pixel_to_utm

from pipeline.s1_gee_candidates.aoi_selection import get_bbox


def param_summary(aoi, start_date, end_date):
    """
    Prints summary of selected parameters for user to confirm before starting pipeline.
    """
    bounds = aoi.bounds().getInfo()['coordinates'][0]
    lons = [c[0] for c in bounds]
    lats = [c[1] for c in bounds]

    print('=' * 40)
    print('Pipeline Configuration')
    print('=' * 40)
    #print(f'AOI: {aoi_name}')
    print(f'    West:   {min(lons):.4f}')
    print(f'    East:   {max(lons):.4f}')
    print(f'    South:  {min(lats):.4f}')
    print(f'    North:  {max(lats):.4f}')
    print(f'Date range')
    print(f'    Start:  {start_date}')
    print(f'    End:    {end_date}')


#---helpers---
def _load_s2_collection(aoi, start_date, end_date, tile_name=None):
    """
    Function loads s2 image collection from Google Earth Engine for selected 
    area of interes and date range 

    - Args: AOI, start date, end date, tile (optional)
    - Returns: Sentinel-2 image collection 
    """
    s2 = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date)

    if tile_name is not None:
        s2 = s2.filter(ee.Filter.eq('MGRS_TILE', tile_name))

    return s2

def _get_metadata(image):
    """
    Extracts date and tile ID from individual S2 images
    """
    image_id = image.id().getInfo()
    date     = image_id[16:24]
    tile     = image_id[-6:]

    return date, tile


def _generate_filepath(date: str, tile: str, mask_dir: str):
    """
    Function defines filepath under which binary image masks are stored
    """
    return f'{mask_dir}/{date}_{tile}_candidates.tif'


def _find_candidates(image):
    """
    Function divides reflectance values by 10000 and applies global reflectance threshold 
    to identify reflectance anomalies in band 3 pixels.
    - Args: S2 images (tiles)
    - Returns: binary mask (0=no candidate, 1=candidate)
    """

    #Dividing reflectance value by 10000
    b2 = image.select('B2').divide(10000)  #blue
    b3 = image.select('B3').divide(10000)  #green
    b4 = image.select('B4').divide(10000)  #red

    #Criterion: rho_B3 > rho_B2 + 0.05 AND rho_B3 > rho_B4 + 0.05
    candidates = b3.gt(b2.add(REFL_THRESHOLD)) \
                   .And(b3.gt(b4.add(REFL_THRESHOLD)))
    
    return candidates.rename('candidate')


def _count_candidates(candidates, clip_geom, image):
    """
    Counts number of candidate pixels per mask
    """
    native_crs = image.select('B3').projection().getInfo()['crs']
    result = candidates.toUint8().reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=clip_geom,
        scale=10,
        crs=native_crs,
        maxPixels=1e9
    ).getInfo()
    
    count = result.get('candidate')
    return int(count) if count is not None else 0


def _get_unique_tiles(s2) -> list[str]:
    """
    Returns sorted list of unique mgrs tile IDs in the collection
    """
    tiles = s2.aggregate_array('MGRS_TILE').getInfo()
    return sorted(set(tiles))


def _split_geometry(geom, n=2):
    """
    This function handles the download of large image tiles.
    Splits a GEE geometry into an n×n grid of sub-geometries.
    """
    coords = geom.bounds().coordinates().get(0).getInfo()
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)

    lon_step = (lon_max - lon_min) / n
    lat_step = (lat_max - lat_min) / n

    cells = []
    for row in range(n):
        for col in range(n):
            west  = lon_min + col * lon_step
            east  = lon_min + (col + 1) * lon_step
            south = lat_min + row * lat_step
            north = lat_min + (row + 1) * lat_step
            cells.append(ee.Geometry.Rectangle([west, south, east, north]))
    return cells


def _export_mask_direct(candidates, filepath, clip_geom, native_crs, _max_splits=8, _depth=0):
    """
    Function attempts download for full image tile and splits tile into quadrants
    if 50MB download limit is exceeded.

    Reuses filepath if it already exists on disk (also applies per quadrant)
    """
    indent = '  ' * _depth

    if Path(filepath).exists():
        print(f'{indent}already exists, reusing: {Path(filepath).name}')
        return True

    geemap.ee_export_image(
        candidates,
        filename=filepath,
        scale=10,
        region=clip_geom,
        crs=native_crs,
    )
    if Path(filepath).exists():
        return True

    print(f'{indent} Too large at depth {_depth}, splitting into 2×2 quadrants...')
    
    #maximum splits
    if _max_splits == 0:
        print(f'{indent}')
        return False
    quadrants  = _split_geometry(clip_geom)
    quad_paths = []

    for q_idx, quad_geom in enumerate(quadrants):
        sub_geom  = clip_geom.intersection(quad_geom, 1)

        # Skipping empty quadrants, e.g. at tile edges
        sub_area = sub_geom.area(1).getInfo()
        if sub_area < 1e4:
            print(f'{indent}  Q{q_idx}: negligible area ({sub_area:.0f} m²), skipping')
            continue

        quad_path = filepath.replace('.tif', f'_d{_depth}_q{q_idx}.tif')
        print(f'{indent}  Q{q_idx}: area {sub_area/1e6:.1f} km², attempting download: {Path(quad_path).name}')

        ok = _export_mask_direct(
            candidates,
            quad_path,
            sub_geom,
            native_crs,
            _max_splits=_max_splits - 1,
            _depth=_depth + 1,
        )

        if ok:
            print(f'{indent}  Q{q_idx}: success')
            quad_paths.append(quad_path)
        else:
            print(f'{indent}  Q{q_idx}: failed, no file produced')

    if not quad_paths:
        print(f'{indent}: All quadrants failed at depth {_depth}')
        return False

    print(f'{indent}: Merging {len(quad_paths)} quadrants at depth {_depth}...')
    _merge_quadrants(quad_paths, filepath)
    for p in quad_paths:
        Path(p).unlink(missing_ok=True)

    if Path(filepath).exists():
        print(f'{indent}: Merged successfully to {Path(filepath).name}')
        return True
    else:
        print(f'{indent}: Merge produced no file')
        return False
  

def _merge_quadrants(quad_paths, out_path):
    """
    Function merges quadrants into one GeoTIFF using rasterio
    """
    
    datasets = [rasterio.open(p) for p in quad_paths]
    mosaic, transform = merge(datasets)
    profile = datasets[0].profile.copy()
    profile.update(width=mosaic.shape[2], height=mosaic.shape[1], transform=transform)

    for ds in datasets:
        ds.close()

    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(mosaic)


#---main---
def run_candidate_screening(
    run_id: int,
    aoi,
    start_date: str,
    end_date: str,
    tile_name: str = None,
    mode: str = 'baseline',
    mask_dir: str = None,
):
    """
    Function runs candidate screening for all selected S2 image tiles

    Functionality:
    - Loads all S2 images for the selected AOI and date range.
    - Runs find_candidates() on each image and exports as binary mask to GeoTIFF.
    - Records one `mask` row per successfully exported GeoTIFF, scoped by run_id.

    Args:
        run_id
        aoi
        start_date
        end_date
        mask_dir: path to save GeoTIFF masks. Defaults to a directory
            scoped to this run_id (Settings.run_subdir)

    mode: 'baseline': fixed reflectance threshold currently implemented as default
    """
    mask_dir = mask_dir if mask_dir is not None else str(get_settings().run_subdir(run_id, "step1_masks"))
    os.makedirs(mask_dir, exist_ok=True)

    # Clears any masks this run_id has already registered 
    # (i.e. existing masks for this run_id on disk will be overwritten)
    n_cleared = repository.clear_masks(run_id)
    if n_cleared:
        print(f"Cleared {n_cleared} existing mask row(s) for run_id={run_id}")

    #Accessing s2 images
    bbox = get_bbox(aoi)
    s2 = _load_s2_collection(aoi, start_date, end_date, tile_name=tile_name)
    s2_list   = s2.toList(s2.size())
    n_images  = s2.size().getInfo()
    total_candidates = 0
    skipped = []

    print('=' * 40)
    print('Candidate Screening Selection:')
    print('=' * 40)
    print(f'AOI Bounding Box: lon {bbox["aoi_lon_min"]:.4f}→{bbox["aoi_lon_max"]:.4f}  '
          f'lat {bbox["aoi_lat_min"]:.4f}→{bbox["aoi_lat_max"]:.4f}')
    print(f'Date range: {start_date} - {end_date}')
    print(f'S2 images:  {n_images}')
    unique_tiles = _get_unique_tiles(s2)
    print(f'Unique tiles: {len(unique_tiles)} — {unique_tiles}')

    #iterating over images for candidate screening
    for i in range(n_images):

        image    = ee.Image(s2_list.get(i))
        date, tile = _get_metadata(image)
        filepath = _generate_filepath(date, tile, mask_dir)

        image_id = image.id().getInfo()
        print(f'  full id: {image_id}')

        #for sequential download
        # Clip to the intersection of this tile's geometry with the AOI
        native_crs = image.select('B3').projection().getInfo()['crs']
        tile_geom  = image.geometry()
        clip_geom  = tile_geom.intersection(aoi, 1)

        # Skip negligible overlaps
        overlap_area = clip_geom.area(1).getInfo()
        if overlap_area < 1e6:
            print(f'[{i+1}/{n_images}] {date} {tile}: negligible overlap ({overlap_area/1e6:.2f} km²), skipping')
            continue    

        candidates = _find_candidates(image).clip(clip_geom)

        try:
            count = _count_candidates(candidates, clip_geom, image)
        except Exception as e:
            print(f'[{i+1}/{n_images}] {date} {tile}: count failed: {e}')
            count = -1

        total_candidates += max(count, 0)
        
        try:
            ok = _export_mask_direct(candidates, filepath, clip_geom, native_crs)
            if ok:
                print(f'[{i+1}/{n_images}] {date} {tile}: candidates: {count} ✓')
                # extracting integer from native crs
                epsg = int(native_crs.split(':')[-1]) if ':' in native_crs else None
                repository.insert_mask(
                    run_id,
                    date=date,
                    tile=tile,
                    image_id=f"{date}_{tile}",
                    file_path=filepath,
                    epsg=epsg,
                    candidate_pixel_count=count if count >= 0 else None,
                )
            else:
                print(f'[{i+1}/{n_images}] {date} {tile}: download produced no file')
                skipped.append((date, tile, 'no file after export'))
        except Exception as e:
            print(f'[{i+1}/{n_images}] {date} {tile}: export failed: {e}')
            skipped.append((date, tile, str(e)))


    print(f'Step 1: Candidate screening completed.')
    print('=' * 40)
    print(f'Step 1: Candidate screening + binary mask download completed')
    print(f'Total candidates found: {total_candidates}')
    print(f'Masks saved to: {mask_dir}')
    if skipped:
        print(f'Skipped ({len(skipped)}):')
        for date, tile, err in skipped:
            print(f'  {date} {tile}: {err}')



#---inspection---
def inspect_candidate_mask(run_id, date, tile, mask_dir=None):
    """
    Visual inspection of a binary mask.
    Counts pixels + candidates in array of exported tile.
    """
    mask_dir = mask_dir if mask_dir is not None else str(get_settings().run_subdir(run_id, "step1_masks"))

    #loading GeoTIFF as array
    mask_file = f'{mask_dir}/{date}_{tile}_candidates.tif'
    arr = np.array(Image.open(mask_file))

    print('=' * 40)
    print('Candidate Screening Inspection')
    print('=' * 40)
    print(f"File:               {date}_{tile}_candidates.tif")
    print(f'Shape:              {arr.shape}')
    print(f'Unique values:      {np.unique(arr)}')
    print(f'Candidate pixels:   {int(arr.sum())}')
    print(f'Total pixels:       {arr.size}')
    print(f'Candidate fraction: {arr.sum()/arr.size*100:.4f}%')

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(arr, cmap="Greys", interpolation="nearest", alpha=0.3)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    #candidate pixels as white crosses, otherwise not visible due to small size
    rows, cols = np.where(arr == 1)
    ax.scatter(cols, rows, 
               c='darkred', 
               s=50, marker='x', 
               label=f"Magnified candidates (n={len(rows)})"
               )

    ax.set_title(f"Step 1 candidates: {date}_{tile}")
    ax.set_xlabel("col (pixels)")
    ax.set_ylabel("row (pixels)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()



def inspect_mask_export(run_id: int, date: str, tile: str, aoi, mask_dir=None) -> None:
    """
    Inspects downloaded GeoTIFF to verify candidate counts in export match those in GEE prior to export.
    Comparing AOI and exported raster boundary after reprojection to UTM in local CRS.
    Safeguard integrated to validate correct reprojection.
    """
    mask_dir = mask_dir if mask_dir is not None else str(get_settings().run_subdir(run_id, "step1_masks"))
    mask_path = Path(mask_dir) / f"{date}_{tile}_candidates.tif"


    arr = np.array(Image.open(mask_path)).astype(np.uint8)

    with tifffile.TiffFile(str(mask_path)) as tif:
        tags = {tag.code: tag.value for tag in tif.pages[0].tags.values()}

    #extracting geotransform from exported Geotiff tile
    geo = extract_geotransform(mask_path)
    x_min        = geo["x_min"]
    y_max        = geo["y_max"]
    pixel_width  = geo["pixel_width"]
    pixel_height = geo["pixel_height"]
    
    #reading EPSG from exported tile
    file_epsg = epsg_from_tile(mask_path)

    #file extent in UTM 
    h, w  = arr.shape
    x_max = x_min + w * pixel_width
    y_min = y_max + h * pixel_height

    #transform AOI rectangle corners from EPSG:4326 to UTM CRS
    coords = aoi.getInfo()["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]

    xs_utm, ys_utm = transform_coords(lons, lats, 4326, file_epsg)

    aoi_x_min, aoi_x_max = min(xs_utm), max(xs_utm)
    aoi_y_min, aoi_y_max = min(ys_utm), max(ys_utm)


    #checking coordinates against AOI bounding box
    rows, cols = np.where(arr == 1)


    cand_xs, cand_ys = pixel_to_utm(cols, rows, geo)
    outside_aoi = (
        (cand_xs < aoi_x_min) | (cand_xs > aoi_x_max) |
        (cand_ys < aoi_y_min) | (cand_ys > aoi_y_max)
    )

    #summary
    print("=" * 40)
    print("Mask Export Inspection")
    print("=" * 40)
    print(f"File             : {mask_path.name}")
    print(f"CRS              : EPSG:{file_epsg}")
    print(f"Shape            : {arr.shape}")
    print(f"Unique values    : {np.unique(arr)}")
    print(f"Candidate pixels : {int(arr.sum())}")
    print(f"Total pixels     : {arr.size}")
    print(f"Candidate frac.  : {arr.sum() / arr.size * 100:.4f}%")

    print(f"\nExported raster boundary (EPSG:{file_epsg}, metres):")
    print(f"  x: {x_min:.2f} → {x_max:.2f}")
    print(f"  y: {y_min:.2f} → {y_max:.2f}")

    print(f"\nAOI boundary reprojected to EPSG:{file_epsg} (metres):")
    print(f"  x: {aoi_x_min:.2f} → {aoi_x_max:.2f}")
    print(f"  y: {aoi_y_min:.2f} → {aoi_y_max:.2f}")

    print(f"\nDiff (raster boundary vs AOI boundary)")
    print(f"  x: {abs(x_min - aoi_x_min):.2f} m")
    print(f"  y: {abs(y_min - aoi_y_min):.2f} m")
    print("*Diff of <10m expected due to grid snapping")

    print(f"\nCandidates inside  AOI : {(~outside_aoi).sum()}")
    print(f"Candidates outside AOI : {outside_aoi.sum()}")
    print("=" * 50)
    

