"""
Orchestration script for Evaluation
"""

import pandas as pd

from pipeline.db import repository
from pipeline.settings import get_settings
from pipeline.s5_evaluation.tile_params  import get_cached_tile_params
from pipeline.s5_evaluation.adsb_query   import run_adsb_query
from pipeline.s5_evaluation.adsb_filter  import run_adsb_filter
from pipeline.s5_evaluation.adsb_match   import run_adsb_match

#---helpers---
def _load_detections(run_id):
    """
    Loads a run's confirmed detections joined with their candidate's tile/date/lon/lat
    """
    df = repository.get_confirmed_detections_with_candidates(run_id)
    df_confirmed = df[df['confirmed'] == True].copy()

    print(f"Total candidates  : {len(df):,}")
    print(f"Confirmed         : {len(df_confirmed):,}")
    print(f"Unique tile/dates : "
          f"{df_confirmed[['tile', 'date']].drop_duplicates().shape[0]}")

    return df_confirmed


def _get_unique_pairs(df):
    """
    Returns unique (tile, date) pairs
    """
    return (
        df[['tile', 'date']]
        .drop_duplicates()
        .values.tolist()
    )


def _run_single(run_id, tile, date, df_all_detections, mask_dir=None):
    """
    Runs the full evaluation pipeline for a single tile/date combination.
    Returns a GeoDataFrame with matched results, or None if there are no
    confirmed detections for this scene.
    """
    # filters detections to this scene directly by tile/date 
    df_detections = df_all_detections[
        (df_all_detections['tile'] == tile) & (df_all_detections['date'] == date)
    ].copy()

    if df_detections.empty:
        print(f"No confirmed detections for {date}_{tile}: skipping")
        return None

    image_id = f"{date}_{tile}"

    # get_cached_tile_params() strips 'T' from tile itself and expects a 'YYYY-MM-DD' date 
    date_dashed = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    params = get_cached_tile_params(tile, date_dashed, mask_dir=mask_dir)

    # runs cross matching
    df_adsb_raw      = run_adsb_query(params)
    df_adsb_filtered = run_adsb_filter(df_adsb_raw, params)
    gdf_matched = run_adsb_match(run_id, image_id, df_detections, df_adsb_filtered, params)

    return gdf_matched


#---main---
def run_evaluation(run_id, mask_dir=None):
    """
    Runs full step 5 evaluation for a run, inserting `evaluation_point`
    rows and returning a dict mapping image_id -> GeoDataFrame of
    matched results for a given scene.
    """
    print(f"\n{'='*40}")
    print(f"Step 5: ADS-B Evaluation")
    print(f"{'='*40}\n")

    mask_dir = (
        mask_dir if mask_dir is not None
        else str(get_settings().run_subdir(run_id, "step1_masks"))
    )

    n_cleared = repository.clear_evaluation_points(run_id)
    if n_cleared:
        print(f"Cleared {n_cleared} existing evaluation_point row(s) for run_id={run_id}")

    # loads detections
    df_confirmed = _load_detections(run_id)
    pairs = _get_unique_pairs(df_confirmed)

    print(f"\nProcessing {len(pairs)} tile/date combinations...\n")

    # processes each tile + date
    results = {}
    for tile, date in pairs:
        print(f"\n{'─'*40}")
        print(f"Processing {tile} - {date}")
        print(f"{'─'*40}")

        gdf = _run_single(run_id, tile, date, df_confirmed, mask_dir=mask_dir)
        if gdf is not None:
            image_id          = f"{date}_{tile}"
            results[image_id] = gdf

    # summary
    print(f"\n{'─'*40}")
    print(f"Evaluation complete: {len(results)} scenes processed")
    print(f"{'─'*40}\n")

    total_det     = sum(
        len(gdf[gdf['source'] == 'pipeline']) for gdf in results.values()
    )
    total_matched = sum(
        len(gdf[(gdf['source'] == 'pipeline') & (gdf['matched'])])
        for gdf in results.values()
    )
    total_adsb    = sum(
        len(gdf[gdf['source'] == 'adsb']) for gdf in results.values()
    )

    print(f"Total detections       : {total_det}")
    print(f"Matched to ADS-B       : {total_matched} / {total_det}")
    print(f"Total ADS-B records    : {total_adsb}")

    return results