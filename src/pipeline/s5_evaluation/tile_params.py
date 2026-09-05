"""
Fetches Sentinel-2 tile acquisition parameters from GEE and caches 
them locally so each tile/date combination is only queried once.
"""

import datetime
from datetime import timezone, timedelta
from pathlib import Path
import ee

from pipeline.db import repository
from pipeline.utils import io
from pipeline.config import (
    ADSB_BUFFER_S,
    TILE_HEIGHT,
    SCAN_DURATION_S,
    MASK_DIR
)


def _fetch_from_gee(tile, date, mask_dir=None):
    """
    Fetches Sentinel-2 acquisition parameters from GEE.
    Returns dict with all parameters 
    """
    mask_dir = mask_dir if mask_dir is not None else MASK_DIR
    date_next = (
        datetime.datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)
    ).strftime("%Y-%m-%d")

    image = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filter(ee.Filter.eq("MGRS_TILE", tile))
        .filterDate(date, date_next)
        .first()
    )

    if image.getInfo() is None:
        raise ValueError(f"No S2 image found for tile {tile} on {date}")

    props = image.getInfo()["properties"]

    # Acquisition times
    t_ms = props["system:time_start"]
    t_top = datetime.datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc)
    t_bottom = t_top + timedelta(seconds=SCAN_DURATION_S)

    # ADS-B query window
    t_query_start = t_top    - timedelta(seconds=ADSB_BUFFER_S)
    t_query_end = t_bottom + timedelta(seconds=ADSB_BUFFER_S)

    # Bounding box
    # handles both image ID s starting with or without T
    mask_candidates = [
        Path(mask_dir) / f"{date.replace('-', '')}_T{tile}_candidates.tif",
        Path(mask_dir) / f"{date.replace('-', '')}_{tile}_candidates.tif",
    ]
    mask_path = next((p for p in mask_candidates if p.exists()), None)

    if mask_path is not None:
        print(f"  AOI bbox from mask : {mask_path.name}")
        bbox = io.get_mask_bbox_wgs84(mask_path)
    else:
        print(f"WARNING: mask not found for {tile} {date} "
              f"falling back to full tile bbox from GEE")
        gee_bbox = image.geometry().bounds().getInfo()['coordinates'][0]
        lons     = [c[0] for c in gee_bbox]
        lats     = [c[1] for c in gee_bbox]
        bbox     = {
            'west'  : min(lons),
            'east'  : max(lons),
            'south' : min(lats),
            'north' : max(lats),
        }

    return {
        "tile"          : tile,
        "date"          : date,
        "t_top"         : t_top,
        "t_bottom"      : t_bottom,
        "t_query_start" : t_query_start,
        "t_query_end"   : t_query_end,
        "west"          : bbox['west'],
        "east"          : bbox['east'],
        "south"         : bbox['south'],
        "north"         : bbox['north'],
        "cloud_pct"     : props.get("CLOUDY_PIXEL_PERCENTAGE"),
    }


#---main---
def get_cached_tile_params(tile, date, mask_dir=None):
    """
    Gets Sentinel-2 acquisition params for a tile/date from the
    `tile_scan_params` cache table if available, otherwise fetches from
    GEE and caches. Not run-scoped!

    Args:
        tile, date ('YYYY-MM-DD')
        mask_dir: only used on a cache miss, to derive the AOI bbox from
            an actual mask file
    Returns:
        dict with param keys
    """
    # Normalise tile: strip leading T if present
    tile     = tile.lstrip("T")
    image_id = f"{date.replace('-', '')}_{tile}"

    cached = repository.get_tile_scan_params(tile, date)
    if cached is not None:
        print(f"tile_params cache hit:  {image_id}")
        cached["image_id"] = image_id
        return cached

    print(f"Fetching from GEE:      {image_id}")
    raw = _fetch_from_gee(tile, date, mask_dir=mask_dir)
    repository.insert_tile_scan_params(raw)
    print(f"Cached to database")

    raw["image_id"] = image_id
    return raw
