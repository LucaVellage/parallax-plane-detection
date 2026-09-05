"""
Confirmation Step 1: Inter-band reflectance anomaly segmentation

For each band, segments inter-band reflectance anomalies with contrast method: 
- computes difference images between B3 and the other two bands
- Applies focal mean + 2xSTD threshold to each difference image
- intersects the two binary masks --> this is the anomaly mask for that band 
- Runs CCL to extract centroids 
"""

import numpy as np
import cv2
from scipy.ndimage import uniform_filter
from pathlib import Path
import pandas as pd
import tifffile

from pipeline.db import repository
from pipeline.config import (
    FOCAL_RADIUS_PX,
    FOCAL_STD_MULT,
    CHIP_CENTRE_PX,
    MAX_CAND_BAND,
)

#---helpers---
def _focal_stats(image:np.ndarray, radius: int):
    """"
    Function computes local means + STD 
    Uses square uniform filter of size (2*radius + 1)
    """
    #21x21 square kernel
    kernel = 2 * radius + 1
    local_mean = uniform_filter(image, size=kernel, mode='reflect')
    local_sq_mean = uniform_filter(image ** 2, size=kernel, mode='reflect')
    variance = np.maximum(local_sq_mean - local_mean ** 2, 0.0)
    local_std = np.sqrt(variance)
    return local_mean, local_std


def _threshold_anomaly(diff: np.ndarray):
    """"
    Computes difference image using the threshold: local_mean + 2 * local_std
    Returns binary mask (1 = anomaly, 0 = background).
    """
    mean, std = _focal_stats(diff, FOCAL_RADIUS_PX)
    diff_img = (diff > mean + FOCAL_STD_MULT * std).astype(np.uint8)
    return diff_img


def _segment_band_anomaly(band_a: np.ndarray, band_b: np.ndarray, band_c: np.ndarray):
    """
    Computes anomaly mask for each band using the other two 
    mask = threshold(band_a - band_b) AND threshold(band_a - band_c)
    """
    diff_img1 = _threshold_anomaly(band_a - band_b)
    diff_img2 = _threshold_anomaly(band_a - band_c)
    diff_imgs = cv2.bitwise_and(diff_img1, diff_img2)
    return diff_imgs


def _extract_centroids(diff_imgs: np.ndarray):
    """
    Runs 4-connected connected component labelling 
    (same as in initial transformation step)
    Goal: Dilate and label pixel blobs + extract centroids 
    """ 
    n_labels, _, stats, centroids_xy = cv2.connectedComponentsWithStats(
        diff_imgs, connectivity=4, ltype=cv2.CV_32S
    )
    result = []
    for label in range(1, n_labels):   
        area = int(stats[label, cv2.CC_STAT_AREA])
        #x=col, y=row
        cx, cy = centroids_xy[label]   
        result.append((float(cx), float(cy), area))
    return result


def _n_closest_to_centre(centroids: list[tuple[float, float, int]],
                         n: int,
                         cx: float = CHIP_CENTRE_PX,
                         cy: float = CHIP_CENTRE_PX) -> list[tuple[float, float, int]]:
    """
    Returns n closest centroid pairs. 

    Note: 
    Returning anomalies closest to centroid is the current solution to deal with 
    noisy difference images, can be replaced with different method.
    """
    sorted_cents = sorted(centroids,
                          key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)
    return sorted_cents[:n]


def run_segmentation(chip: np.ndarray) -> tuple[
    list[tuple[float, float, int]],   # C_B2: top-N closest to centre
    tuple[float, float, int] | None,  # C_B3: single closest to centre
    list[tuple[float, float, int]],   # C_B4: top-N closest to centre
    dict[str, np.ndarray],            # anomaly masks keyed 'B2', 'B3', 'B4'
]:
    """
    Segments inter-band reflectance anomalies in 3-band chip.
    Only keeps N nearest centroids from other bands to B3 centroids.
    Returns 3 sets of centroids: 
    C_B2: top-N band 2 centroids closets to centre
    C_B3: Known band 3 centroid 
    C_B4: top_n band 4 centroids closest to centre
    Segment inter-band reflectance anomalies in a 3-band chip.

    Entry point for single chip
    """
    B2, B3, B4 = chip[0], chip[1], chip[2]

    #anomaly masks
    mask_B2 = _segment_band_anomaly(B2, B3, B4)
    mask_B3 = _segment_band_anomaly(B3, B2, B4)
    mask_B4 = _segment_band_anomaly(B4, B2, B3)

    masks = {'B2': mask_B2, 'B3': mask_B3, 'B4': mask_B4}

    #centroids
    all_B2 = _extract_centroids(mask_B2)
    all_B3 = _extract_centroids(mask_B3)
    all_B4 = _extract_centroids(mask_B4)

    #If multiple B3 anomalies detected --> select closest to chip centre 
    if len(all_B3) == 0:
        C_B3 = None
    else:
        C_B3 = min(
            all_B3,
            key=lambda c: (c[0] - CHIP_CENTRE_PX) ** 2
                        + (c[1] - CHIP_CENTRE_PX) ** 2,
        )

    # B2, B4: top-N closest to chip centre
    C_B2 = _n_closest_to_centre(all_B2, n=MAX_CAND_BAND)
    C_B4 = _n_closest_to_centre(all_B4, n=MAX_CAND_BAND)

    return C_B2, C_B3, C_B4, masks


#---main---
def run_reflectance_check(run_id: int, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Batch entry point to run reflectance anomaly confirmation for all chips.
    Inserts the resulting per-band centroids into the `reflectance_centroid` table.
    """
    df = df if df is not None else repository.get_candidates(run_id, survivors_only=True)
    chips_df = repository.get_chips(run_id)
    chip_path_by_candidate = dict(zip(chips_df["candidate_id"], chips_df["file_path"]))

    all_rows = []
    n_missing_chip = 0
    n_b3_found = 0
    missing_b3 = []
    print("Running Reflectance Anomaly Check...")

    for _, cand in df.iterrows():
        candidate_id = int(cand["id"])
        chip_path = chip_path_by_candidate.get(candidate_id)

        if not chip_path or not Path(chip_path).exists():
            print(f"Chip not found for candidate {candidate_id:04d}")
            n_missing_chip += 1
            continue

        raw  = tifffile.imread(str(chip_path))
        chip = np.moveaxis(raw, -1, 0).astype(np.float32)

        C_B2, C_B3, C_B4, _ = run_segmentation(chip)

        for col, row, area in C_B2:
            all_rows.append({"candidate_id": candidate_id, "band": "B2",
                            "col": col, "row": row, "area": area})

        #only recorded if B3 anomaly found
        if C_B3 is not None:
            all_rows.append({"candidate_id": candidate_id, "band": "B3",
                            "col": C_B3[0], "row": C_B3[1], "area": C_B3[2]})
            n_b3_found += 1
        else:
            missing_b3.append((candidate_id, cand["lon"], cand["lat"]))

        for col, row, area in C_B4:
            all_rows.append({"candidate_id": candidate_id, "band": "B4",
                            "col": col, "row": row, "area": area})

    result_df = repository.insert_reflectance_centroids(run_id, all_rows)

    #summary
    n_chips_total = len(df)
    n_chips_ok = n_chips_total - n_missing_chip
    n_b3_missing = n_chips_ok - n_b3_found

    print(f"\n{'=' * 40}")
    print(f"Reflectance Segmentation Summary")
    print(f"{'=' * 40}")
    print(f"Total candidates     : {n_chips_total}")
    print(f"Chips found on disk  : {n_chips_ok}")
    print(f"Chips missing        : {n_missing_chip}")
    print(f"B3 anomaly found     : {n_b3_found}")
    print(f"B3 anomaly missing   : {n_b3_missing}  ← chip drops out of pipeline")
    print(f"Total centroids      : {len(result_df)}")
    print(f"  B2 centroids       : {(result_df['band'] == 'B2').sum()}")
    print(f"  B3 centroids       : {(result_df['band'] == 'B3').sum()}")
    print(f"  B4 centroids       : {(result_df['band'] == 'B4').sum()}")

    #inspecting missing B3 anomaly chips
    if missing_b3:
        print(f"\nChips with no B3 anomaly found:")
        for candidate_id, lon, lat in missing_b3:
            print(f"  candidate_id={candidate_id:04d}  lon={lon:.4f}  lat={lat:.4f}")

    return result_df
        



