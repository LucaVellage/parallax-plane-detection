"""
File downloads image chips of filtered candidates 
"""

import ee
import requests
import numpy as np
import pandas as pd
import tifffile
import hashlib
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from pipeline.db import repository
from pipeline.settings import get_settings

from pipeline.config import (
    CHIP_SIZE_PX,
    CHIP_RADIUS_M,
    BANDS,
    CHIP_CACHE_DIR,
    CHIP_CACHE_TOL_PX,
)

#---helpers---
def _load_scene_image(date, tile):
    """
    Loads the Sentinel-2 image matching date and tile.
    Returns ee.Image with bands B2, B3, B4.
    Loads full images from GEE server and extracts image chip per candidate centroid, 
    instead of downloading each image chip individually.
    """
    dt    = datetime.strptime(str(date), "%Y%m%d")
    start = dt.strftime("%Y-%m-%d")
    end   = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
    
    col = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
            .filter(ee.Filter.date(start, end)) \
            .filter(ee.Filter.stringContains("system:index", tile)) \
            .select(BANDS)
    
    count = col.size().getInfo()
    print(f"Images found for {date} {tile}: {count}")
    return col.first()


def _lazy_scene_image(date, tile):
    """
    Checks chip cache first before triggering chip download via GEE call.
    """
    lock  = threading.Lock()
    cache = {}

    def get():
        with lock:
            if "image" not in cache:
                cache["image"] = _load_scene_image(date, tile)
            return cache["image"]

    return get


def _cache_dir():
    path = Path(CHIP_CACHE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _symlink(target, link_path):
    """
    Points link_path at target (in chip cache on disk)
    """
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(target)


def _download_chip(get_image, lon, lat, image_id, candidate_id, chips_dir, tile, date, col, row):
    """
    Downloads a 301×301px chip centred on (lon, lat) from the scene image.
    Saves as GeoTIFF to chips_dir, named using the candidate's stable database id (candidate_id)

    - Checks cache before GEE call 
    - Returns path and source (cache/download)
    """
    out_path = Path(chips_dir) / f"{image_id}_{candidate_id:04d}_chip.tif"
    #only downloads chips that are not downloaded
    if out_path.exists():
        return out_path, "existing"

    cached = repository.find_cached_chip(tile, date, col, row, CHIP_CACHE_TOL_PX)
    if cached is not None:
        cached_path = Path(cached["file_path"])
        if cached_path.exists() and _sha256(cached_path) == cached["checksum"]:
            _symlink(cached_path, out_path)
            return out_path, "cache"
        print(f"chip cache entry for {image_id} col={col} row={row} is stale/corrupted "
              f"(missing file or checksum mismatch): falling back to live download")

    image  = get_image()
    point  = ee.Geometry.Point([lon, lat])
    region = point.buffer(CHIP_RADIUS_M).bounds()
    native_crs = image.select('B2').projection().getInfo()['crs']

    url = image.getDownloadURL({
        "bands":  BANDS,
        "region": region,
        #"scale":  SCALE,
        "format": "GEO_TIFF",
        "dimensions": f"{CHIP_SIZE_PX}x{CHIP_SIZE_PX}",
        "crs":        native_crs
    })

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    cache_path = _cache_dir() / f"{image_id}_{col:.2f}_{row:.2f}_chip.tif"
    cache_path.write_bytes(response.content)
    repository.insert_chip_cache_entry(
        tile, date, image_id, col, row, str(cache_path), _sha256(cache_path),
    )

    _symlink(cache_path, out_path)
    return out_path, "download"


def _download_scene_chips(get_image, df_scene, chips_dir, max_workers=4):
    """
    Downloads all chips for one scene in parallel.
    - Returns a list of (candidate_id, path, source) tuples
    - source is "existing" / "cache" / "download" per _download_chip(), or "failed".
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _download_chip,
                get_image,
                row["lon"], row["lat"],
                row["image_id"],
                int(row["id"]),
                chips_dir,
                row["tile"], row["date"],
                row["col"], row["row"],
            ): int(row["id"])
            for _, row in df_scene.iterrows()
        }
        for future in as_completed(futures):
            candidate_id = futures[future]
            try:
                path, source = future.result()
                results.append((candidate_id, path, source))
            except Exception as e:
                print(f"failed to download candidate {candidate_id:04d}: {e}")
                results.append((candidate_id, None, "failed"))
    return results


#---main---
def run_chip_download(
    run_id: int,
    df: pd.DataFrame | None = None,
    chips_dir: str | None = None,
) -> pd.DataFrame:
    """
    Downloads a GeoTIFF chip per surviving Step 2 candidate and records one `chip` row each in `chip`table.
    """
    df = df if df is not None else repository.get_candidates(run_id, survivors_only=True)
    chips_dir = chips_dir if chips_dir is not None else str(get_settings().run_subdir(run_id, "step3_chips"))
    Path(chips_dir).mkdir(parents=True, exist_ok=True)

    print(f"Surviving candidates: {len(df)}")
    summary = []

    for image_id, df_scene in df.groupby("image_id"):
        date = str(df_scene["date"].iloc[0])
        tile = str(df_scene["tile"].iloc[0])
        print(f"\n{image_id}: {len(df_scene)} candidates")

        get_image = _lazy_scene_image(date, tile)
        results   = _download_scene_chips(get_image, df_scene, chips_dir)

        for candidate_id, path, source in results:
            repository.insert_chip(
                candidate_id, run_id,
                file_path=str(path) if path else None,
                status="ok" if path else "failed",
            )

        n_downloaded = sum(1 for _, p, s in results if s == "download")
        n_cached     = sum(1 for _, p, s in results if s in ("cache", "existing"))
        n_fail       = sum(1 for _, p, s in results if s == "failed")
        print(f"  → downloaded {n_downloaded}, from cache {n_cached}, failed {n_fail}")

        summary.append({"image_id": image_id, "total": len(df_scene),
                        "downloaded": n_downloaded, "cached": n_cached, "failed": n_fail})

    print("\n── chip download summary ──")
    print(f"{'scene':<25} {'total':>6} {'downloaded':>10} {'cached':>8} {'failed':>8}")
    print("─" * 60)
    for row in summary:
        print(f"{row['image_id']:<25} {row['total']:>6} {row['downloaded']:>10} "
              f"{row['cached']:>8} {row['failed']:>8}")
    total      = sum(r["total"]      for r in summary)
    downloaded = sum(r["downloaded"] for r in summary)
    cached     = sum(r["cached"]     for r in summary)
    failed     = sum(r["failed"]     for r in summary)
    print("─" * 60)
    print(f"{'all scenes':<25} {total:>6} {downloaded:>10} {cached:>8} {failed:>8}")


#---inspection--- 

def _normalize_chip(path, percentile=2):
    """
    Helper function for inspection to normalize chip colors:
    - Displays a chip as an RGB image (B4=red, B3=green, B2=blue)
    - Stretches contrast using percentile clipping
    """
    arr = tifffile.imread(path) 
    if arr.shape[2] == 3:
        arr = arr.transpose(2, 0, 1)
    rgb = np.stack([arr[2], arr[1], arr[0]], axis=-1).astype(float)
    for c in range(3):
        lo, hi       = np.percentile(rgb[:, :, c], [percentile, 100 - percentile])
        rgb[:, :, c] = np.clip((rgb[:, :, c] - lo) / (hi - lo + 1e-6), 0, 1)
    return rgb


def plot_chip(run_id, date, tile, candidate_idx, percentile=2):
    """
    Displays single chip
    """
    chips_df = repository.get_chips(run_id)
    match = chips_df[chips_df["candidate_id"] == candidate_idx]
    if match.empty or not match.iloc[0]["file_path"]:
        raise FileNotFoundError(
            f"No chip found for candidate_id={candidate_idx} in run_id={run_id}"
        )
    path = Path(match.iloc[0]["file_path"])

    rgb = _normalize_chip(str(path), percentile)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.set_title(path.stem)
    ax.axis("off")
    plt.tight_layout()
    plt.show()



def plot_chips(run_id, date: str, tile: str, max_chips: int = 24, percentile: int = 2) -> None:
    """
    Displays a grid of chips for a given date and tile
    """
    image_id = f"{date}_{tile}"
    candidates = repository.get_candidates(run_id)
    scene_ids  = candidates[candidates["image_id"] == image_id]["id"]

    chips_df = repository.get_chips(run_id)
    scene_chips = chips_df[
        chips_df["candidate_id"].isin(scene_ids) & chips_df["file_path"].notna()
    ]
    paths = [Path(p) for p in scene_chips["file_path"].tolist()][:max_chips]

    if not paths:
        raise FileNotFoundError(f"No chips found for {image_id} in run_id={run_id}")

    n_cols = 4
    n_rows = int(np.ceil(len(paths) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes      = np.array(axes).flatten()

    for ax, path in zip(axes, paths):
        rgb = _normalize_chip(str(path), percentile)
        ax.imshow(rgb)
        ax.set_title(Path(path).stem, fontsize=8)
        ax.axis("off")

    for ax in axes[len(paths):]:
        ax.axis("off")

    fig.suptitle(f"{date}_{tile}: {len(paths)} chips", fontsize=12)
    plt.tight_layout()
    plt.show()