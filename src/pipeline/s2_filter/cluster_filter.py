"""
Script removes cluster anomalies

Case: 
For example reflectance anomalies or sun-glint over water, i.e. cases in which centroids are grouped very close to each other.

Main functionality:
- Haversine matrix computes pairwise circle distances in metres between points.
- Threshold: Candidate removed if >= 3 centroids within buffer (r = 1500m) of a singular centroid. 
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from pipeline.db import repository
from pipeline.settings import get_settings
from pipeline.utils.io import (
    epsg_from_tile,
    transform_coords
)

from pipeline.config import (
    CLUSTER_BUFFER_M,
    CLUSTER_MAX
)

#---helpers---
def _haversine_matrix(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    #Earth radius
    R = 6_371_000.0 

    #converting degrees to radians
    #necessary for trigonometric fct
    lats = np.radians(lats)
    lons = np.radians(lons)

    #computing pairwise angle differences
    #produces difference matrix 
    dlat = lats[:, None] - lats[None, :]
    dlon = lons[:, None] - lons[None, :]

    #haversine formula applied element-wise across /
    #difference matrices
    #values (0-1) are stored in matrix a
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lats[:, None]) * np.cos(lats[None, :])
         * np.sin(dlon / 2) ** 2)
    
    #converting to metres
    return 2 * R * np.arcsin(np.sqrt(a))


def _exclude_clusters(candidate_centroids: list[dict]) -> list[dict]:
    """
    Function removes candidate centrois that sit at dense clusters.

    Counts per candidate how many other candidates fall in cluster of one centroid.
    If count >= CLUSTER_MAX, then candidate removed.

    Current setting:
    neighbours (excl self) < 3 --> kept
    neighbours (excl self) >= 3 --> candidate and all those neighbours removed
    """

    if len(candidate_centroids) <= CLUSTER_MAX:
        return candidate_centroids
    
    lons = np.array([c["lon"] for c in candidate_centroids])
    lats = np.array([c["lat"] for c in candidate_centroids])

    #includes distance to itself at diagnoal
    dist_matrix = _haversine_matrix(lons, lats)
    np.fill_diagonal(dist_matrix, np.inf)  # exclude self
    
    #produces boolean matrix (True: point within 1500m)
    #result: Sum of TRUE values per row (i.e. per candidate)
    neighbour_count = (dist_matrix <= CLUSTER_BUFFER_M).sum(axis=1)
    in_dense_buffer = neighbour_count >= CLUSTER_MAX 
    exclude = np.zeros(len(candidate_centroids), dtype=bool)
    for i, is_dense in enumerate(in_dense_buffer):
        if is_dense:
            # exclude i and all its neighbours
            exclude[i] = True
            exclude[dist_matrix[i] <= CLUSTER_BUFFER_M] = True

    return [c for c, ex in zip(candidate_centroids, exclude) if not ex]


#---main---
def run_cluster_exclusion(run_id: int, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Marks dense-cluster candidates as cluster_excluded in the `candidate` 
    table and returns the surviving DataFrame
    """
    df = df if df is not None else repository.get_candidates(run_id)
    print(f"Total candidates before cluster filter: {len(df)}")

    survivors = []
    excluded_ids: list[int] = []
    for image_id, group in df.groupby("image_id"):
        candidates = group.to_dict(orient="records")
        filtered   = _exclude_clusters(candidates)

        # inspection output per tile
        lons = np.array([c["lon"] for c in candidates])
        lats = np.array([c["lat"] for c in candidates])

        dist_matrix = _haversine_matrix(lons, lats)
        dist_inspect = dist_matrix.copy()
        np.fill_diagonal(dist_inspect, np.inf)
        min_distances = dist_inspect.min(axis=1)

        print(f"\n{image_id}:")
        print(f"  candidates before : {len(candidates)}")
        print(f"  candidates after  : {len(filtered)}")
        print(f"  excluded          : {len(candidates) - len(filtered)}")
        print(f"  min distance      : {min_distances.min():.1f} m")
        print(f"  max distance      : {min_distances.max():.1f} m")

        print(f"{image_id}: {len(candidates)} → {len(filtered)}")

        survivor_ids = {c["id"] for c in filtered}
        excluded_ids.extend(c["id"] for c in candidates if c["id"] not in survivor_ids)
        survivors.extend(filtered)

    repository.set_candidate_flags(excluded_ids, cluster_excluded=True)

    df_filtered = pd.DataFrame(survivors)
    print(f"\nTotal candidates after cluster filter: {len(df_filtered)}")
    print(f"Marked {len(excluded_ids)} candidate(s) cluster_excluded in the database")

    return df_filtered
    
#---inspection---
def cluster_filtered_plot(run_id, date, tile, mask_dir=None):
    """
    Plots candidates pre- and post-filter application with visible buffer circles
    """
    image_id = f"{date}_{tile}"
    scene = repository.get_candidates(run_id)
    scene_before = scene[scene["image_id"] == image_id]
    removed      = scene_before[scene_before["cluster_excluded"] == True]
    scene_after  = scene_before[scene_before["cluster_excluded"] == False]

    print('=' * 40)
    print('Cluster Filter Inspection')
    print('=' * 40)
    print(f"Before:     {len(scene_before)} centroids")
    print(f"Survivors:  {len(scene_after)} centroids")
    print(f"Removed:    {len(removed)} centroids")

    mask_dir = mask_dir if mask_dir is not None else str(get_settings().run_subdir(run_id, "step1_masks"))
    mask_path = Path(mask_dir) / f"{date}_{tile}_candidates.tif"
    epsg       = epsg_from_tile(mask_path)
    lat_centre = scene_before["lat"].mean()
    lon_centre = scene_before["lon"].mean()
    e1, n1     = transform_coords(lon_centre,       lat_centre, 4326, epsg)
    e2, n2     = transform_coords(lon_centre + 1.0, lat_centre, 4326, epsg)
    m_per_deg  = np.sqrt((e2 - e1)**2 + (n2 - n1)**2)
    deg_radius = CLUSTER_BUFFER_M / m_per_deg

    #plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Cluster exclusion: buffer: {CLUSTER_BUFFER_M}m, threshold: {CLUSTER_MAX} neighbours",
                 fontsize=12)


    # left: all candidates with removed ones highlighted
    axes[0].scatter(scene_before["lon"], scene_before["lat"],
                    c="steelblue", s=30, zorder=3, label=f"surviving ({len(scene_after)})")
    axes[0].scatter(removed["lon"], removed["lat"],
                    c="red", s=60, marker="x", zorder=4, linewidths=1.5,
                    label=f"excluded (cluster) ({len(removed)})")

    #buffer circles around removed candidates
    for _, row in removed.iterrows():
        #approx 1500m in degrees at this latitude
        axes[0].add_patch(plt.Circle(
            (row["lon"], row["lat"]), deg_radius,
            color="red", fill=False, linewidth=0.8, alpha=0.4
        ))
    axes[0].set_title(f"{date}_{tile}: Before cluster filter")

    #right: survivors only
    axes[1].scatter(scene_after["lon"], scene_after["lat"],
                    c="steelblue", s=30, zorder=3, label=f"surviving ({len(scene_after)})")

    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()
