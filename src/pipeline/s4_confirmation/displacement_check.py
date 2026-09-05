"""
Confirmation Step 3: Displacement ratio

Logic: Displacement between triplet points is known due to fixed time offsets in band recordings.

Functionality:
- Computes euclidian distance between all collinear triplet pixel points per chip.
- Computes displacement residual and records triplet combination with lowest residual as aircraft parallax detection.
"""


import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pipeline.db import repository
from pipeline.config import (DISPLACEMENT_RATIO,
                             DISPLACEMENT_OFFSET,
                             DISPLACEMENT_TOLERANCE,
                             METRES_PER_PIXEL,
                             T_BAND2_4)


# Fallback result when a candidate has no colinear triplets at all 
REJECTED = {
    'confirmed':   False,
    'D_Band2_3_m': float('nan'),
    'D_Band3_4_m': float('nan'),
    'residual_m':  float('nan'),
    'angle_deg':   float('nan'),
    'speed_kmh':   float('nan'),
    'c_b2_col':    float('nan'), 'c_b2_row': float('nan'),
    'c_b3_col':    float('nan'), 'c_b3_row': float('nan'),
    'c_b4_col':    float('nan'), 'c_b4_row': float('nan'),
}

#---helpers---
def _pixel_dist(col_a, row_a, col_b, row_b):
    """
    Euclidian distance in metres between pixel points
    """
    dx = (col_a - col_b) * METRES_PER_PIXEL
    dy = (row_a - row_b) * METRES_PER_PIXEL
    return math.sqrt(dx * dx + dy * dy)


def check_displacement(triplets):
    best = None
    best_residual = float('inf')
    best_overall = None

    for triplet in triplets:
        D_23     = _pixel_dist(triplet['c_b2_col'], triplet['c_b2_row'],
                                     triplet['c_b3_col'], triplet['c_b3_row'])
        D_34     = _pixel_dist(triplet['c_b3_col'], triplet['c_b3_row'],
                                     triplet['c_b4_col'], triplet['c_b4_row'])
        D_24 = _pixel_dist(triplet['c_b2_col'], triplet['c_b2_row'],
                           triplet['c_b4_col'], triplet['c_b4_row'])
        residual = abs(DISPLACEMENT_RATIO * D_23 + DISPLACEMENT_OFFSET - D_34)

        #tracking all best residuals even if rejected
        if residual < best_residual:
            best_residual = residual
            best_overall  = {
                'D_Band2_3_m': D_23,
                'D_Band3_4_m': D_34,
                'residual_m':  residual,
                'angle_deg':   triplet['angle_deg'],
                'speed_kmh':   (D_24 / T_BAND2_4) * 3.6,
                'c_b2_col':    triplet['c_b2_col'], 'c_b2_row': triplet['c_b2_row'],
                'c_b3_col':    triplet['c_b3_col'], 'c_b3_row': triplet['c_b3_row'],
                'c_b4_col':    triplet['c_b4_col'], 'c_b4_row': triplet['c_b4_row'],
            }

        #tracking best passing combination
        if residual <= DISPLACEMENT_TOLERANCE and residual < float('inf'):
            if best is None or residual < best['residual_m']:
                best = {'confirmed': True, **best_overall}

    if best is not None:
        return best
    
    if best_overall is not None:
        return {'confirmed': False, **best_overall}

    return dict(REJECTED)


#---main---
def run_displacement_check(run_id: int, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    For each candidate, tests every colinear B2-B3-B4 triplet for proportional inter-band displacement 
    and inserts the best result into the `confirmed_detection` table

    This step produces the final confirmed detections.
    """
    df = df if df is not None else repository.get_collinearity_triplets(run_id)

    n_chips_total         = df['candidate_id'].nunique()
    n_chips_with_colinear = df[df['colinear_found'] == True]['candidate_id'].nunique()

    rows = []
    for candidate_id, chip_df in df.groupby('candidate_id'):
        # only passing collinear triplets are considered
        colinear = chip_df[chip_df['colinear_found'] == True].to_dict('records')
        result = check_displacement(colinear)
        rows.append({'candidate_id': candidate_id, **result})

    result_df = repository.insert_confirmed_detections(run_id, rows)
    n_confirmed_chips = int(result_df['confirmed'].sum())

    #summary
    print(f"\n{'=' * 40}")
    print(f"Displacement Check Summary")
    print(f"{'=' * 40}")
    print(f"Downloaded chips            : {n_chips_total}")
    print(f"Chips with collinear combos : {n_chips_with_colinear}")
    print(f"→ Confirmed                 : {n_confirmed_chips}")
    print(f"→ Rejected by displacement  : {n_chips_with_colinear - n_confirmed_chips}")
    print(f"Confirmed / downloaded chips: {100 * n_confirmed_chips / n_chips_total:.1f}%")

    # #tile lookup for summary print
    candidates_df = repository.get_candidates(run_id)
    tile_by_candidate = dict(zip(candidates_df['id'], candidates_df['tile']))
    result_df = result_df.assign(tile=result_df['candidate_id'].map(tile_by_candidate))

    print(f"\nPer-tile breakdown:")
    tile_summary = (
        result_df.groupby('tile')
        .agg(
            candidates=('confirmed', 'count'),
            confirmed =('confirmed', 'sum'),
        )
        .assign(rate_pct=lambda x: (
            x['confirmed'] / x['candidates'] * 100
        ).round(1))
    )
    print(tile_summary.to_string())

    confirmed_df = result_df[result_df['confirmed'] == True]
    if not confirmed_df.empty:
        speeds = confirmed_df['speed_kmh'].dropna()
        print(f"\nSpeed (confirmed aircraft):")
        print(f"  mean   : {speeds.mean():.0f} km/h")
        print(f"  median : {speeds.median():.0f} km/h")
        print(f"  min    : {speeds.min():.0f} km/h")
        print(f"  max    : {speeds.max():.0f} km/h")

    return result_df


