"""
Confirmation Step 2: Collinearity 

Logic: RGB triplets are nearly linearly distributed 

Functionality:
- Computes angle at vertex for each possible triplet per chip
- Only triples >= COLINEAR_ANGLE_MIN pass collinearity confirmation
"""

import math
import numpy as np
import pandas as pd

from pipeline.db import repository
from pipeline.config import COLINEAR_ANGLE_MIN

#---helpers---
def _angle_at_vertex(a, v, b):
    """
    Angle at vertex V in triangle A-V-B

    Uses dot product: cos(θ) = (VA · VB) / (|VA| |VB|)
    Returns value in [0, 180].
    Returns 0.0 if either vector has zero length (coincident points).
    """

    #vectors relative to vertex V
    va = np.array([a[0] - v[0], a[1] - v[1]], dtype=float)
    vb = np.array([b[0] - v[0], b[1] - v[1]], dtype=float)

    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    cos_theta = np.clip(np.dot(va, vb) / (norm_a * norm_b), -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


#single chip entry point
def compute_triplet_angles(b2_candidates, b3, b4_candidates):
    """
    Computes angles of all possible triplets
    """
    all_triplets = []
    for b2 in b2_candidates:
        for b4 in b4_candidates:
            angle = _angle_at_vertex(b2, b3, b4)
            all_triplets.append({
                'c_b2_col':  b2[0], 'c_b2_row':  b2[1], 'c_b2_area': b2[2],
                'c_b3_col':  b3[0], 'c_b3_row':  b3[1], 'c_b3_area': b3[2],
                'c_b4_col':  b4[0], 'c_b4_row':  b4[1], 'c_b4_area': b4[2],
                'angle_deg': angle,
                'colinear_found': angle >= COLINEAR_ANGLE_MIN,
            })
    return all_triplets

#---main---
def run_collinearity_check(run_id: int, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Tests every B2xB4 centroid-triplet combination per candidate for collinearity with its B3 centroid,
    and inserts the results into the `collinearity_triplet` table.
    """
    df = df if df is not None else repository.get_reflectance_centroids(run_id)

    rows = []
    for candidate_id, chip_df in df.groupby('candidate_id'):
        b2_list = list(chip_df[chip_df['band'] == 'B2'][['col', 'row', 'area']].itertuples(index=False, name=None))
        b3_list = list(chip_df[chip_df['band'] == 'B3'][['col', 'row', 'area']].itertuples(index=False, name=None))
        b4_list = list(chip_df[chip_df['band'] == 'B4'][['col', 'row', 'area']].itertuples(index=False, name=None))

        # band missing
        if not b3_list or not b2_list or not b4_list:
            rows.append({
                'candidate_id': candidate_id,
                'colinear_found': False,
                'angle_deg':  float('nan'),
                'c_b2_col':   float('nan'), 'c_b2_row': float('nan'),
                'c_b2_area':  float('nan'),
                'c_b3_col':   float('nan'), 'c_b3_row': float('nan'),
                'c_b3_area':  float('nan'),
                'c_b4_col':   float('nan'), 'c_b4_row': float('nan'),
                'c_b4_area':  float('nan'),
            })
            continue

        b3 = b3_list[0]

        triplet_angles = compute_triplet_angles(b2_list, b3, b4_list)

        for triplet_angle in triplet_angles:
            rows.append({'candidate_id': candidate_id, **triplet_angle})

    result_df = repository.insert_collinearity_triplets(run_id, rows)

    n_chips    = df['candidate_id'].nunique()
    n_colinear = result_df[result_df['colinear_found'] == True]['candidate_id'].nunique()
    n_combos_passing = result_df['colinear_found'].sum()
    n_combos_notpassing = (result_df['colinear_found'] == False).sum()
    n_combos_total = result_df['angle_deg'].notna().sum()

    print(f"\n{'=' * 40}")
    print(f"Collinearity Check Summary")
    print(f"{'=' * 40}")
    print(f"Total chips          : {n_chips}")
    print(f"With colinear combos : {n_colinear}")
    print(f"No colinear combos   : {n_chips - n_colinear}")
    print(f"Total combinations   : {n_combos_total}")
    print(f"Passing combos       : {n_combos_passing}")
    print(f"Non-passing combos   : {n_combos_notpassing}")

    # candidate_id lookup for summary print
    candidates_df = repository.get_candidates(run_id)
    tile_by_candidate = dict(zip(candidates_df['id'], candidates_df['tile']))
    result_df = result_df.assign(tile=result_df['candidate_id'].map(tile_by_candidate))

    print(f"\nPer-tile breakdown:")
    tile_summary = (result_df.groupby('tile').agg(chips=('candidate_id', 'nunique'), colinear=('colinear_found', 'sum')))

    print(tile_summary.to_string())

    return result_df
