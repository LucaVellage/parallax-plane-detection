"""
Script removes vehicles on roads

Logic: 
- Detects candidates which are located linearly which are likely representing vehicles on roads.
- These candidates lie at a position where over time, multiple candidates accumulated in a straight line. 
- These are likely cars, driving on a straight road.

Functionality: 
- Computes frequency images per tile, i.e. binary mask which hold all candidate positions over time per tile.
- Defines linear kernel of shape [51, 51]
- Defines rotation matrix. Kernel is supposed to rotate over frequency image in 10° steps by 
multiplying kernel x rotation matrix.
- By passing the rotating kernel in sliding-window movement over each pixel in 18 different angles, 
pixels receive road values at each angle. 
- A max_response array holds the maximum road value at each pixel.
- candidate centroids which overlap with pixels that have a road value > 5 are excluded. 
"""
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

from pipeline.db import repository
from pipeline.settings import get_settings
from pipeline.utils.io import (
    list_files,
)

from pipeline.s2_filter.transform import (
    _read_mask
)

from pipeline.config import (
    ROAD_KERNEL_HALF,
    ROAD_ANGLE_STEP,
    ROAD_FREQUENCY_THRESHOLD,
)


#---helpers---
def _get_mask_shape(tile: str, mask_dir: str) -> tuple[int, int]:
    """
    Function finds all mask files of a given tile, reads first to extract dimensions.
    BUilds on the known property that all mask files of the same tile have same dimensions.
    Returns height and width per tile.
    """
    mask_files = list_files(mask_dir, pattern="*_candidates.tif")
    tile_masks = [f for f in mask_files if tile in Path(f).name]
    if not tile_masks:
        raise FileNotFoundError(f"No masks found for tile {tile} in {mask_dir}")
    mask = _read_mask(tile_masks[0])

    #(height, width)
    return mask.shape 


def _build_frequency_image(df_tile: pd.DataFrame, mask_shape: tuple[int, int]) -> np.ndarray:
    """
    Function builds 2-D grid with the same size as mask of a given tile
    """
    #creates empty grid first
    freq = np.zeros(mask_shape, dtype=np.uint16)

    #gets row and col values from the tile dfs
    rows = df_tile["row"].astype(int).values
    cols = df_tile["col"].astype(int).values

    #if a value lies outside of grid, gets clipped to nearest edge 
    rows = np.clip(rows, 0, mask_shape[0] - 1)
    cols = np.clip(cols, 0, mask_shape[1] - 1)

    #adds 1 for each centroid into grid. if centroid in same pixel, increments to 2, 3, etc
    np.add.at(freq, (rows, cols), 1)

    #outputs a sparse array 
    return freq


def _build_linear_kernel(half: int = ROAD_KERNEL_HALF, width = 2) -> np.ndarray:
    """
    Builds a horizontal linear kernel of shape (2*half+1, 2*half+1) as numpy array.
    With the default width=2, the centre row and the row directly above it
    are set to ones; all other rows are zero.
    """
    size = 2 * half + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    #kernel[half, :] = 1.0
    centre = half
    for i in range(width):
        offset = i // 2 if i % 2 == 0 else -(i // 2 + 1)
        kernel[centre + offset, :] = 1.0
    return kernel
   


def _rotate_kernel(kernel: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Function rotates kernel 

    Properties:
        - Rotates kernel by angle_deg and corrects decimal pixel values to binary
        - cv2.getRotationMatrix2D rotates counter-clockwise
        - Border values (i.e. outside of binary mask) are filled with 0 
        - if kernel lies between pixel grid, pixel values are blended across 4 nearest neighbours 
        (instead of directly snapping to nearest neighbour)
        - Decimals are corrected by snapping to 1 if k > 0.1 and 0 if k < 0.1
    """
    h, w = kernel.shape

    #identifying center of kernel
    center = (w // 2, h // 2)

    #Building rotation matrix
    #scale=1 keeps kernel at constant size
    M = cv2.getRotationMatrix2D(center, angle_deg, scale=1.0)

    #applies affine transformation to the binary mask (i.e. the image)
    rotated = cv2.warpAffine(kernel, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    #fixes decimals which arise from interpolation
    rotated_binary = (rotated > 0.1).astype(np.float32)

    return rotated_binary


def _build_max_response(freq_image: np.ndarray) -> np.ndarray:
    """
    Convolves frequency image with 18 rotated linear kernels, return pixel-wise max.
    by comparing response road value at each angle with maximum road value (per pixel).
    Stores maximum value in array.
    """
    kernel_base = _build_linear_kernel()
    
    #moves in 10 degree steps
    angles = range(0, 180, ROAD_ANGLE_STEP)  
    freq_f32 = freq_image.astype(np.float32)

    max_response = np.zeros_like(freq_f32)
    for angle in angles:
        k = _rotate_kernel(kernel_base, angle)

        #no padding at borders 
        response = cv2.filter2D(freq_f32, ddepth=-1, kernel=k,
                                borderType=cv2.BORDER_CONSTANT)
        np.maximum(max_response, response, out=max_response)
    return max_response 


def _apply_road_filter(df_tile: pd.DataFrame, max_response: np.ndarray) -> pd.DataFrame:
    """
    Function looks up maximum response (i.e. road value) at every candidate centroid location
    and flags road hits if road value > 5. 
    """
    #Extracts candidate centroid coordinates
    #safeguards against candidate that are outside of maximum_response raster boundary
    rows = df_tile["row"].astype(int).values.clip(0, max_response.shape[0] - 1)
    cols = df_tile["col"].astype(int).values.clip(0, max_response.shape[1] - 1)

    df_tile = df_tile.copy()
    #for each centroid, reads maximum road value and filters those larger than 5
    df_tile["road_score"] = max_response[rows, cols]
    df_tile["road_exclude"] = df_tile["road_score"] > ROAD_FREQUENCY_THRESHOLD
    return df_tile

def _save_frequency_image(freq_image: np.ndarray, tile: str, road_freq_dir: str) -> None:
    """
    Saves the frequency image as a .npy file for now
    """
    Path(road_freq_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(road_freq_dir) / f"{tile}_frequency.npy"
    np.save(out_path, freq_image)
    print(f"  → frequency image saved: {out_path}")


#---main---
def run_road_exclusion(
    run_id: int,
    df: pd.DataFrame | None = None,
    mask_dir: str | None = None,
    road_freq_dir: str | None = None,
) -> pd.DataFrame:
    """
    Flags road/vehicle-pattern candidates as road_excluded in `candidate` table
    """
    df = df if df is not None else repository.get_candidates(run_id, survivors_only=True)
    mask_dir = (
        mask_dir if mask_dir is not None
        else str(get_settings().run_subdir(run_id, "step1_masks"))
    )
    road_freq_dir = (
        road_freq_dir if road_freq_dir is not None
        else str(get_settings().run_subdir(run_id, "step2_road_frequency"))
    )

    updates: list[dict] = []
    survivors = []

    for tile, df_tile in df.groupby("tile"):
        print(f"Processing tile {tile}: {len(df_tile)} candidates across "
              f"{df_tile['image_id'].nunique()} dates")

        mask_shape = _get_mask_shape(tile, mask_dir)
        freq_image = _build_frequency_image(df_tile, mask_shape)
        max_response = _build_max_response(freq_image)

        _save_frequency_image(freq_image, tile, road_freq_dir)

        df_tile = _apply_road_filter(df_tile, max_response)

        updates.extend(
            {
                "id": int(row["id"]),
                "road_score": float(row["road_score"]),
                "road_excluded": bool(row["road_exclude"]),
            }
            for _, row in df_tile.iterrows()
        )
        survivors.append(df_tile[~df_tile["road_exclude"]])

        n_total = len(df_tile)
        n_excluded = int(df_tile["road_exclude"].sum())
        n_kept = n_total - n_excluded
        print(f"  → excluded {n_excluded} / {n_total}, kept {n_kept}")

    repository.bulk_update_candidates(updates)
    print(f"\nUpdated road_score/road_excluded for {len(updates)} candidate(s) in the database")

    if not survivors:
        return df.iloc[0:0]
    df_filtered = pd.concat(survivors, ignore_index=True)
    return df_filtered.drop(columns=["road_score", "road_exclude"], errors="ignore")


#---inspection---
def kernel_viz(a_1: int, a_2: int, a_3: int, a_4: int, a_5: int):

    fig, axes = plt.subplots(1, 6, figsize=(12, 4))

    #building kernel 
    kernel_base = _build_linear_kernel()
    axes[0].imshow(kernel_base, cmap="gray")
    axes[0].set_title("0° (base)")
    axes[1].imshow(_rotate_kernel(kernel_base, a_1), cmap="gray")
    axes[1].set_title("10°")
    axes[2].imshow(_rotate_kernel(kernel_base, a_2), cmap="gray")
    axes[2].set_title("20°")
    axes[3].imshow(_rotate_kernel(kernel_base, a_3), cmap="gray")
    axes[3].set_title("30°")
    axes[4].imshow(_rotate_kernel(kernel_base, a_4), cmap="gray")
    axes[4].set_title("90°")
    axes[5].imshow(_rotate_kernel(kernel_base, a_5), cmap="gray")
    axes[5].set_title("160°")
    plt.tight_layout()
    plt.show()


def freq_image_plot(run_id, tile, highlight_date, road_freq_dir=None):
    """
    Plots frequency image per tile
    """
    road_freq_dir = (
        road_freq_dir if road_freq_dir is not None
        else str(get_settings().run_subdir(run_id, "step2_road_frequency"))
    )
    freq_path = Path(road_freq_dir) / f"{tile}_frequency.npy"
    freq_img = np.load(freq_path)

    all_candidates = repository.get_candidates(run_id)
    df_tile = all_candidates[
        (all_candidates["tile"] == tile) & (all_candidates["cluster_excluded"] == False)
    ].copy()
    df_tile["date"] = df_tile["date"].astype(str)
    n_dates = df_tile["date"].nunique()

    # frequency image check
    freq_rows,  freq_cols  = np.where(freq_img > 0)
    cand_rows = df_tile["row"].astype(int).values
    cand_cols = df_tile["col"].astype(int).values

    freq_positions = set(zip(freq_rows.tolist(), freq_cols.tolist()))
    cand_positions = set(zip(cand_rows.tolist(), cand_cols.tolist()))

    print("=" * 40)
    print("Frequency Image Inspection")
    print("=" * 40)
    print(f"Frequency image            : {freq_path.name}")
    print(f"Shape                      : {freq_img.shape}")
    print(f"Dates Captured             : {n_dates}")
    print(f"Non-zero pixels            : {len(freq_positions)}")
    print(f"Unique centroid positions  : {len(cand_positions)}")
    print(f"In freq not in centroids   : {len(freq_positions - cand_positions)}")
    print(f"In centroids not in freq   : {len(cand_positions - freq_positions)}")
    print("=" * 40)
 
    #plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, linewidth=0.4, alpha=0.5)

    ax.scatter(df_tile['col'], df_tile['row'],
               s=20, c= "steelblue", label=f'all dates ({len(df_tile)})')
    
    if highlight_date:
        df_hi = df_tile[df_tile['date'] == highlight_date]
        print(f"Highlight date {highlight_date}: {len(df_hi)} candidates")
        ax.scatter(df_hi['col'], df_hi['row'],
                   s=20, c='#ff5b00', label=f'{highlight_date} ({len(df_hi)})')
    
    ax.invert_yaxis()
    fig.suptitle(f"{tile}: Frequency image with candidates", fontsize=14)
    ax.set_title(f"Candidates of Selected Date Highlighted | Frequency Image on {freq_img.shape[0]}×{freq_img.shape[1]} Pixel Grid",
             fontsize=10)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def road_filtered_plot(run_id, tile: str) -> None:
    """
    Split plot showing candidate distribution pre- and post-road-filter
    """
    all_candidates = repository.get_candidates(run_id)
    tile_candidates = all_candidates[all_candidates["tile"] == tile]
    pre     = tile_candidates[tile_candidates["cluster_excluded"] == False]
    removed = pre[pre["road_excluded"] == True]
    post    = pre[pre["road_excluded"] == False]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Road exclusion: frequency threshold >{ROAD_FREQUENCY_THRESHOLD}, "
                 f"kernel half-size {ROAD_KERNEL_HALF}px", fontsize=12)
    
    #summary
    print('=' * 40)
    print('Road Filter Inspection')
    print('=' * 40)
    print(f"Before:     {len(pre)} centroids")
    print(f"Survivors:  {len(post)} centroids")
    print(f"Removed:    {len(removed)} centroids")

    # left 
    axes[0].scatter(pre["lon"], pre["lat"],
                    s=20, c="steelblue", label=f"surviving ({len(pre)})", zorder=2)
    axes[0].scatter(removed["lon"], removed["lat"],
                    s=40, c="red", marker="x", label=f"excluded (road) ({len(removed)})", zorder=3)
    axes[0].set_title(f"{tile}: Before road filter")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].legend()

    # right
    axes[1].scatter(post["lon"], post["lat"],
                    s=20, c="steelblue", label=f"surviving ({len(post)})", zorder=2)
    axes[1].set_title(f"{tile}: After road filter")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


