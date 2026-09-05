"""
Helper functions for loading files and geotransformations
"""
from pathlib import Path
import tifffile
import pyproj
from functools import lru_cache
import numpy as np
import pandas as pd
import math

from pipeline.config import METRES_PER_PIXEL
from pipeline.db import repository

#Loading files
def list_files(dir: str, pattern: str) -> list[Path]:
    """
    Returns files in the specified directory,
    sorted by name
    """
    files = sorted(Path(dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No  files found in {dir}")
    return files
 
 
def parse_filename(path: str| Path) -> dict[str, str]:
    """
    Parses date and tile from a Step 1 filename stem
    - Assumes the convention: YYYYMMDD_TILE_<suffix>
    - Splits on underscore and takes the first two parts.

    Examples:
    - parse_filename("20230601_T32TNT_candidates.tif")
    → {"date": "20230601", "tile": "T32TNT"}

    - parse_filename("20230601_T32TNT_centroids.csv")
    → {"date": "20230601", "tile": "T32TNT"}
    """
    parts = Path(path).stem.split("_")
    return {
        "date": parts[0],
        "tile": parts[1],
    }

def load_chip(chip_path: str | Path) -> np.ndarray:
    """
    Reads 3-band GeoTIFF chip (B2, B3, B4)
    - Returns float32 array (3, H, W).
    - Assumes following storage order in GeoTIFF: (H, W, 3)
    - Band order: B2=0, B3=1, B4=2.
    """
    img = tifffile.imread(str(chip_path))
    if img.ndim == 3 and img.shape[2] == 3:
        img = np.moveaxis(img, -1, 0)

    else:
        raise ValueError("Invalid chip shape. Chip shape assumed: (H, W, 3)")

    chip = img.astype(np.float32)
    return chip


def get_bands(chip):
    """
    Extracts individual bands from chip array
    """
    B2, B3, B4 = chip[0], chip[1], chip[2]
    return B2, B3, B4


def load_chip_by_idx(run_id: int, idx: int) -> tuple[np.ndarray, Path]:
    """
    Loads a chip by its candidate.id (idx), looking up the file path
    chip_download.run_chip_download() recorded in the `chip` table
    """
    chips_df = repository.get_chips(run_id)
    match = chips_df[chips_df["candidate_id"] == idx]
    if match.empty:
        raise FileNotFoundError(f"No chip found for candidate_id={idx} in run_id={run_id}")

    row = match.iloc[0]
    if not row["file_path"]:
        raise FileNotFoundError(
            f"Candidate {idx} has no chip file recorded (status={row['status']!r})"
        )

    chip_path = Path(row["file_path"])
    chip = load_chip(chip_path)
    return chip, chip_path


def _load_conf_row(run_id: int, idx: int) -> dict | None:
    """
    Returns the confirmed_detection row for one candidate
    """
    df = repository.get_confirmed_detections(run_id)
    rows = df[df["candidate_id"] == idx]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _load_col_rows(run_id: int, idx: int) -> pd.DataFrame:
    """
    Returns all collinearity_triplet rows for one candidate
    """
    df = repository.get_collinearity_triplets(run_id)
    return df[df["candidate_id"] == idx].copy()

#Metadata extraction
def extract_geotransform(mask_file: str) -> dict:
    """
    Extracts affine geotransform from GeoTIFF exported by GEE in native UTM CRS.
    Reads ModelPixelScaleTag (33550) + ModelTiepointTag (33922).
    Returns dict holding x_min, y_max in metres (easting/northing), pixel size in metres.
    """
    with tifffile.TiffFile(str(mask_file)) as tif:
        tags = {tag.code: tag.value for tag in tif.pages[0].tags.values()}

    if 34264 in tags:
        t = tags[34264]
        return {
            "x_min":        t[3],
            "y_max":        t[7],
            "pixel_width":  t[0],
            "pixel_height": t[5],
        }
    elif 33550 in tags and 33922 in tags:
        scale    = tags[33550]
        tiepoint = tags[33922]
        return {
            "x_min":        tiepoint[3],
            "y_max":        tiepoint[4],
            "pixel_width":  scale[0],
            "pixel_height": -scale[1],
        }
    else:
        raise ValueError(
            f"No geotransform tags found in {mask_file}. "
            "Expected tag 34264, or tags 33550 and 33922."
        )
    

# coordinate conversion functions
def epsg_from_tile(mask_file: str) -> int:
    """
    Extracts EPSG code from GeoTIFF
    """
    with tifffile.TiffFile(str(mask_file)) as tif:
        tags = {tag.code: tag.value for tag in tif.pages[0].tags.values()}

    if 34735 not in tags:
        raise ValueError(f"No GeoKeyDirectoryTag (34735) found in {mask_file}.")

    geo_keys = tags[34735]
    n_keys   = geo_keys[3]
    for i in range(n_keys):
        base = 4 + i * 4
        if geo_keys[base] == 3072 and geo_keys[base + 3] != 32767:
            return geo_keys[base + 3]

    raise ValueError(f"No ProjectedCSTypeGeoKey (3072) found in {mask_file}.")
    
@lru_cache(maxsize=16)
def get_transformer(epsg_source: int, epsg_target: int) -> pyproj.Transformer:
    """
    Function for transforming coordinates between different CRS systems
    """
    crs_transformer = pyproj.Transformer.from_crs(
        f'EPSG:{epsg_source}', f'EPSG:{epsg_target}', always_xy=True
    )
    return crs_transformer


def transform_coords(x: float, y: float, epsg_source: int, epsg_target: int) -> tuple[float, float]:
    """
    Transforms a coordinate pair between two EPSG coordinate systems 
    """
    return get_transformer(epsg_source, epsg_target).transform(x, y)


def pixel_to_utm(col: float, row: float, geotransform: dict) -> tuple[float, float]:
    """ 
    Function converts centroid pixel location (center of any pixel) from 
    raster coordinates to UTM easting/northing in metres.

    Functionality:
    - Uses the affine geotransform extracted in extract_geotransform()
    - Returns (easting, northing), not lon/lat!
    - to get lon/lat, pass output from pixel_to_utm() to transform_coords()
    """
    easting = geotransform['x_min'] + (col + 0.5) * geotransform['pixel_width']
    northing = geotransform['y_max'] + (row + 0.5) * geotransform['pixel_height']
    return easting, northing


def _pixel_dist_m(col_a, row_a, col_b, row_b) -> float:
    return math.sqrt(((col_a - col_b) * METRES_PER_PIXEL) ** 2 +
                     ((row_a - row_b) * METRES_PER_PIXEL) ** 2)


def get_mask_bbox_wgs84(mask_path):
    """
    Derives AOI bounding box in EPSG:4326 from a GeoTIFF mask file.
    Works globally because reads the native UTM CRS from the GeoTIFF tags and 
    reprojects corners to WGS84 using pyproj.

    Args:
        mask_path : str or Path (path to GeoTIFF mask file)
    Returns:
        dict with keys: west, east, south, north (in decimal degrees)
    """
    import numpy as np
    from PIL import Image

    mask_path = Path(mask_path)

    # Read geotransform and native CRS from GeoTIFF tags
    geo  = extract_geotransform(mask_path)
    epsg = epsg_from_tile(mask_path)

    # Get raster dimensions
    arr  = np.array(Image.open(mask_path))
    h, w = arr.shape

    # Compute all four corners in native UTM metres
    x_min = geo['x_min']
    y_max = geo['y_max']
    x_max = x_min + w * geo['pixel_width']
    y_min = y_max + h * geo['pixel_height']  # pixel_height is negative

    # Transform all four corners from native UTM to WGS84
    # Using all four corners handles non-rectangular projections
    # correctly at high latitudes
    xs_utm = [x_min, x_max, x_min, x_max]
    ys_utm = [y_min, y_min, y_max, y_max]

    lons, lats = transform_coords(xs_utm, ys_utm, epsg, 4326)

    return {
        'west'  : min(lons),
        'east'  : max(lons),
        'south' : min(lats),
        'north' : max(lats),
    }




