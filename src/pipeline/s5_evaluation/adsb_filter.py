"""
Filters returned ADS-B records to get unique record at satellite overpass time
"""

import pandas as pd
import numpy as np
from pathlib import Path

from pipeline.config import (
    ADSB_MIN_ALTITUDE_M,
    MIN_VELO_MS,
    TILE_HEIGHT, 
    SCAN_DURATION_S
)

#---helpers---
def _apply_filters(df, params):
    """
    Applies filter to state vectors
    """
    n0 = len(df)

    df = df.copy()
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)

    # Temporal filter
    df = df[
        (df['datetime'] >= params['t_query_start']) &
        (df['datetime'] <= params['t_query_end'])
    ]

    # Spatial filter: clips to AOI if smaller than full tile
    df = df[
        (df['lon'] >= params['west'])  &
        (df['lon'] <= params['east'])  &
        (df['lat'] >= params['south']) &
        (df['lat'] <= params['north'])
    ]

    # Altitude filter to remove e.g. taxiing aircraft
    df = df[
        (df['onground'] == False) &
        (df['baroaltitude'].notna()) &
        (df['baroaltitude'] >= ADSB_MIN_ALTITUDE_M) &
        (df['velocity'].notna()) &
        (df['velocity'] >= MIN_VELO_MS)
    ]

    # Drop NA
    df = df.dropna(subset=['lat', 'lon'])
    df = df.reset_index(drop=True)

    return df

def _select_best_ping(group, params):
    """
    Selects pings closest to row correctedt acquisition time based on location
    """
    group     = group.sort_values('datetime').copy()
    lat_range = params['north'] - params['south']

    #find ping at tile center
    t_centre  = params['t_top'] + pd.Timedelta(seconds=SCAN_DURATION_S / 2)
    group['dt'] = (group['datetime'] - t_centre).abs()
    best      = group.loc[group['dt'].idxmin()]
    lat_0     = best['lat']

    # estimates row from lat
    row_est   = int(((params['north'] - lat_0) / lat_range) * TILE_HEIGHT)
    row_est   = max(0, min(TILE_HEIGHT - 1, row_est))

    # computes acquistion time for that row
    t_correct = params['t_top'] + pd.Timedelta(
        seconds=(row_est / TILE_HEIGHT) * SCAN_DURATION_S
    )

    # selects ping closest to row corrected time
    group['dt_corrected'] = (group['datetime'] - t_correct).abs()
    best_corrected        = group.loc[group['dt_corrected'].idxmin()]

    return {
        **best_corrected.to_dict(),
        'row_estimate' : row_est,
        't_correct'    : t_correct.isoformat(),
        'speed_kmh'    : best_corrected['velocity'] * 3.6,
    }

def _get_unique_aircraft(df, params):
    """
    returns unique aircraft per scene
    """
    records = [
        _select_best_ping(group, params)
        for _, group in df.groupby('icao24')
    ]
    unique = pd.DataFrame(records).reset_index(drop=True)
    print(f"  Unique aircraft        : {len(unique)}")

    return unique


#---main---
def run_adsb_filter(df_raw, params):
    
    df_filtered = _apply_filters(df_raw, params)
    unique      = _get_unique_aircraft(df_filtered, params)

    return unique