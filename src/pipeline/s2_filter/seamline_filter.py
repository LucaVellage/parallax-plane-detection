"""
Script filters false positives which result from cloud movement between seam-lines

Functionality: 
- Uses MSAC to fit lines through all possible pairs of candidates in given scene
- Counts inliers per line and retains line with lowest MSAC cost that has a minimum 
of required inliers defined by SEAM_MIN_INLIERS.
- Retains only lines at 68°-82° to est-west axis 
- Repeated 7 times since a S2-tile can contain up to 7 seamlines
"""
import numpy as np
import pandas as pd
from shapely.geometry import Point
from pathlib import Path
import matplotlib.pyplot as plt

from pipeline.db import repository
from pipeline.utils.io import (
    transform_coords,
    epsg_from_tile
)

from pipeline.config import (
    SEAM_MAX_DIST_M,
    SEAM_MIN_INLIERS,
    SEAM_ITERATIONS,
    SEAM_DIR_MIN,
    SEAM_DIR_MAX,
)

#---helpers---
def _convert_to_utm(df, epsg):
    """
    Function converts lon lat coordinates to UTM to measure seam-line buffer in m
    """
    eastings, northings = transform_coords(df["lon"].values, df["lat"].values, 4326, epsg)
    return np.column_stack([eastings, northings])


def _fit_line_normal_form(p1, p2):
    """
    Fits a line through two points in normal form:
        x·cos(θ) + y·sin(θ) = ρ    (ρ ≥ 0)
 
    Returns (None, None) for degenerate input (duplicate points)
    """
    #computes direction vector
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    #calculates hypotenuse + filter duplicates
    length = np.hypot(dx, dy)
    if length < 1e-6:
        return None, None
    
    #computes unit normal vector, i.e. perpendicular vector
    nx = -dy / length
    ny =  dx / length

    #rho = perpendicular distance from origin to line
    #projects p1 onto normal direction 
    rho = p1[0] * nx + p1[1] * ny
    if rho < 0:
        nx, ny, rho = -nx, -ny, -rho
    #converts normal vector backto angle 
    theta = np.arctan2(ny, nx)
 
    #converts normal vector backto angle 
    return theta, rho


def _point_line_distance(
    pts: np.ndarray,
    theta: float,
    rho: float,
) -> np.ndarray:
    """
    Measures how far a given point is from line (i.e. violates line equation)
    i.e. measures points perpendicular distance from line.
    x·cos(θ) + y·sin(θ) = ρ.
    """
    #dot product of each point with unit normal (i.e. how for along normal dir each point is placed)
    projection = pts[:, 0] * np.cos(theta) + pts[:, 1] * np.sin(theta) - rho
    #unsigned perpendicular distance of point to line
    distance = np.abs(projection)
    return distance


def _line_angle_deg(theta_normal):
    """
    Function computes angle between line and east-west axis.
    Needed for angle check: seam-line must lie between 68°-82°.
    """
    line_deg = np.degrees(theta_normal + np.pi / 2) % 180
    return line_deg


def _msac_cost(distance):
    """
    Defines MSAC cost function which penalises points that are further away from line
    """
    threshold_sq = SEAM_MAX_DIST_M ** 2
    return np.sum(np.minimum(distance ** 2, threshold_sq))


def _msac_best_line(pts):
    """
    Function fits a line through every possible pair of points in a given scene. 
    Counts inliers per scene and retains line with lowest MSAC cost 
    """
    n = len(pts)
    best_theta   = None
    best_rho     = None
    best_inliers = np.array([], dtype=int)
    best_cost    = np.inf

    #runs for every unique pair
    for i in range(n - 1):
        for j in range(i + 1, n):

            #fits line for this pair of points
            theta, rho = _fit_line_normal_form(pts[i], pts[j])
            if theta is None:
                continue

            #distance of all remaining points to fitted line
            dists    = _point_line_distance(pts, theta, rho)
            #detects inliers within buffer 
            inliers  = np.where(dists <= SEAM_MAX_DIST_M)[0]
            #computes MSAC cost for this line
            cost     = _msac_cost(dists)

            #updates metrics and stores line with lowest cost 
            if cost < best_cost:
                best_cost    = cost
                best_theta   = theta
                best_rho     = rho
                best_inliers = inliers

    #if no line with min required number of inliers detected 
    if best_theta is None or len(best_inliers) < SEAM_MIN_INLIERS:
        return None

    return best_theta, best_rho, best_inliers


def _run_scene(df, epsg):
    """
    Function iteratues maximum 7 times over same scene, since one 
    tile can contain up to 7 seam-lines
    """
    df = df.copy()
    df["seam_exclude"] = False

    if len(df) < SEAM_MIN_INLIERS:
        return df

    pts = _convert_to_utm(df, epsg)
    active_mask = np.ones(len(df), dtype=bool)

    for iteration in range(SEAM_ITERATIONS):
        active_idx = np.where(active_mask)[0]
        if len(active_idx) < SEAM_MIN_INLIERS:
            break

        pts_active = pts[active_idx]
        result = _msac_best_line(pts_active)

        if result is None:
            print(f"iteration {iteration+1}: no line found, stopping")
            break

        theta, rho, inlier_local_idx = result
        line_angle = _line_angle_deg(theta)

        print(f"iteration {iteration+1}: best line {line_angle:.1f}°  inliers={len(inlier_local_idx)}  seam={SEAM_DIR_MIN <= line_angle <= SEAM_DIR_MAX}")

        if not (SEAM_DIR_MIN <= line_angle <= SEAM_DIR_MAX):
            print(f"not a seam-line direction, stopping")
            break

        inlier_global_idx = active_idx[inlier_local_idx]
        df.iloc[inlier_global_idx, df.columns.get_loc("seam_exclude")] = True
        active_mask[inlier_global_idx] = False

    return df


#---main---
def run_seamline_exclusion(run_id: int, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Flags satellite seam-line artefact candidates as seam_excluded in the `candidate` table
    """
    df = df if df is not None else repository.get_candidates(run_id, survivors_only=True)

    summary = []
    survivors = []
    excluded_ids: list[int] = []

    for image_id, df_scene in df.groupby("image_id"):
        epsg = int(df_scene["epsg"].iloc[0])
        print(f"\n{image_id}: {len(df_scene)} candidates")
        df_scene = _run_scene(df_scene, epsg)

        if len(df_scene) < SEAM_MIN_INLIERS:
            print(f"Scene has too few candidates to detect seam-line: Minimum {SEAM_MIN_INLIERS} candidates required.")
            excluded_ids.extend(df_scene["id"].tolist())
            continue

        n_total = len(df_scene)
        n_excluded = int(df_scene["seam_exclude"].sum())
        n_kept = n_total - n_excluded
        print(f"{n_excluded} candidates removed")
        print(f"{len(df_scene)} candidates → {n_kept} candidates")

        excluded_ids.extend(df_scene.loc[df_scene["seam_exclude"], "id"].tolist())
        survivors.append(df_scene[~df_scene["seam_exclude"]].drop(columns=["seam_exclude"]))

        summary.append({"image_id": image_id, "total": n_total,
                        "excluded": n_excluded, "kept": n_kept})

    repository.set_candidate_flags(excluded_ids, seam_excluded=True)

    print('=' * 40)
    print("Seam-line Exclusion Summary")
    print('=' * 40)
    print(f"{'scene':<25} {'total':>6} {'excluded':>9} {'kept':>6}")
    print("─" * 50)
    for row in summary:
        print(f"{row['image_id']:<25} {row['total']:>6} {row['excluded']:>9} {row['kept']:>6}")
    total = sum(r["total"]    for r in summary)
    kept  = sum(r["kept"]     for r in summary)
    excl  = sum(r["excluded"] for r in summary)
    print("─" * 50)
    print(f"{'all scenes':<25} {total:>6} {excl:>9} {kept:>6}")
    print(f"Marked {len(excluded_ids)} candidate(s) seam_excluded in the database")

    if not survivors:
        return df.iloc[0:0]
    return pd.concat(survivors, ignore_index=True)


#---inspection---
def plot_seamline_exclusion(run_id, date, tile) -> None:
    """
    Split plot showing distribution of candidates pre- and post-seamline-filter
    """
    image_id = f"{date}_{tile}"
    all_candidates = repository.get_candidates(run_id)
    df_seam = all_candidates[
        (all_candidates["image_id"] == image_id)
        & (all_candidates["cluster_excluded"] == False)
        & (all_candidates["road_excluded"] == False)
    ]

    epsg      = int(df_seam["epsg"].iloc[0])
    excluded  = df_seam[df_seam["seam_excluded"] == True]
    surviving = df_seam[df_seam["seam_excluded"] == False]

    #refitting seam line through excluded points
    seam_line = None
    if len(excluded) >= SEAM_MIN_INLIERS:
        #reprojecting to utm to fit line
        utm_pts = _convert_to_utm(excluded, epsg)

        #centering points for plot
        centre      = utm_pts.mean(axis=0)
        utm_centred = utm_pts - centre
        result      = _msac_best_line(utm_centred)

        if result is not None:
            theta, rho, _ = result
            line_angle    = _line_angle_deg(theta)
            print(f"Plot line angle: {line_angle:.1f}°")

            # reconstruct line points in original UTM space
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            t_vals       = np.linspace(-60000, 60000, 500)
            line_e       = centre[0] + rho * cos_t - t_vals * sin_t
            line_n       = centre[1] + rho * sin_t + t_vals * cos_t
            line_lon, line_lat = transform_coords(line_e, line_n, epsg, 4326)

            # clip to excluded points extent + margin
            margin  = 0.02
            mask = (
                (line_lon >= excluded["lon"].min() - margin) &
                (line_lon <= excluded["lon"].max() + margin) &
                (line_lat >= excluded["lat"].min() - margin) &
                (line_lat <= excluded["lat"].max() + margin)
            )
            if mask.sum() > 1:
                seam_line = (line_lon[mask], line_lat[mask], theta)

    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"Seam-line exclusion: direction {SEAM_DIR_MIN}–{SEAM_DIR_MAX}°,  "
        f"buffer {SEAM_MAX_DIST_M}m,  min inliers {SEAM_MIN_INLIERS}",
        fontsize=12
    )

    #left
    axes[0].scatter(surviving["lon"], surviving["lat"],
                    s=20, c="steelblue", zorder=2, label=f"surviving ({len(surviving)})")
    axes[0].scatter(excluded["lon"], excluded["lat"],
                    s=40, c="red", marker="x", zorder=3,
                    label=f"excluded (seam) ({len(excluded)})")

    if seam_line is not None:
        line_lon, line_lat, theta = seam_line

        #seam line
        axes[0].plot(line_lon, line_lat,
                     c="red", linewidth=1.5, linestyle="--",
                     zorder=4, label="seam line")

 
        lat_centre    = excluded["lat"].mean()
        buf_lat       = SEAM_MAX_DIST_M / 111320
        buf_lon       = SEAM_MAX_DIST_M / (111320 * np.cos(np.radians(lat_centre)))
        
    axes[0].set_title(f"{image_id}: Before seam-line filter")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].legend(loc="upper right")

    #right
    axes[1].scatter(surviving["lon"], surviving["lat"],
                    s=20, c="steelblue", zorder=2, label=f"surviving ({len(surviving)})")
    axes[1].set_title(f"{image_id}: After seam-line filter")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()

