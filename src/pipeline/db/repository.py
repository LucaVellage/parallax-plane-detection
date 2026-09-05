"""
Repository layer connecting stage modules to database

- Centralizes DB read/write logic
- Defines PostGIS geometry encoding/decoding + date-format conversion

See db/models.py for full table overview and schema design
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import geopandas as gpd
import pandas as pd
from geoalchemy2.shape import from_shape
from shapely.geometry import Point, box
from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from pipeline.db.base import get_engine, get_session
from pipeline.db.models import (
    AdsbQueryCache,
    Candidate,
    Chip,
    ChipCache,
    CollinearityTriplet,
    ConfirmedDetection,
    EvaluationPoint,
    Mask,
    MetricsByBackground,
    MetricsByScene,
    PipelineRun,
    ReflectanceCentroid,
    TileScanParams,
)

#---helpers---
def _nan_to_none(value):
    """
    Converts any "missing" sentinels (e.g. NaN, NA) to "None"
    """
    if value is None:
        return None
    if pd.isna(value):
        return None
    return value


def _nan_to_int(value):
    """
    Converts value to int if not None
    """
    value = _nan_to_none(value)
    return int(value) if value is not None else None


def _parse_yyyymmdd(date_str: str) -> dt.date:
    """
    Converts the pipeline's 'YYYYMMDD' filename date convention to a
    real date for storing in Postgres DATE column
    """
    return dt.datetime.strptime(str(date_str), "%Y%m%d").date()

#---run-scoped pipeline steps---
def create_pipeline_run(
    *,
    start_date: str,
    end_date: str,
    aoi_name: str | None = None,
    tile_name: str | None = None,
    notes: str | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
    lat_min: float | None = None,
    lat_max: float | None = None,
    config_snapshot: dict | None = None,
) -> int:
    """
    Creates a `pipeline_run` row and returns id

    Functionality: 
    - start_date/end_date are accepted in 'YYYY-MM-DD' form
    - lon_min/lon_max/lat_min/lat_max (all four requried or or none) 
    populate both the bbox float columns and aoi_geom 
    """
    aoi_geom = None
    if None not in (lon_min, lon_max, lat_min, lat_max):
        aoi_geom = from_shape(box(lon_min, lat_min, lon_max, lat_max), srid=4326)

    with get_session() as session:
        run = PipelineRun(
            aoi_name=aoi_name,
            aoi_geom=aoi_geom,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            start_date=dt.date.fromisoformat(start_date),
            end_date=dt.date.fromisoformat(end_date),
            tile_name=tile_name,
            notes=notes,
            status="running",
            config_snapshot=config_snapshot,
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        return run.id


def update_run_status(run_id: int, status: str) -> None:
    """
    Updates a pipeline_run's status ('running' / 'completed' / 'failed')
    """
    with get_session() as session:
        session.execute(
            update(PipelineRun).where(PipelineRun.id == run_id).values(status=status)
        )
        session.commit()


def get_pipeline_run(run_id: int) -> dict | None:
    """
    Returns a single pipeline_run row as a dict or None
    """
    stmt = select(PipelineRun).where(PipelineRun.id == run_id)
    with get_session() as session:
        row = session.execute(stmt).scalar_one_or_none()
        if row is None:
            return None
        return {
            "id": row.id, "aoi_name": row.aoi_name,
            "lon_min": row.lon_min, "lon_max": row.lon_max,
            "lat_min": row.lat_min, "lat_max": row.lat_max,
            "start_date": row.start_date, "end_date": row.end_date,
            "tile_name": row.tile_name, "status": row.status,
            "config_snapshot": row.config_snapshot, "notes": row.notes,
            "created_at": row.created_at, "updated_at": row.updated_at,
        }


def list_pipeline_runs() -> pd.DataFrame:
    """
    Returns all pipeline_run rows
    """
    stmt = select(
        PipelineRun.id, PipelineRun.aoi_name, PipelineRun.tile_name,
        PipelineRun.start_date, PipelineRun.end_date, PipelineRun.status,
        PipelineRun.notes, PipelineRun.created_at,
    ).order_by(PipelineRun.created_at.desc())
    return pd.read_sql(stmt, get_engine())


def clear_masks(run_id: int) -> int:
    """
    Deletes this run's existing `mask` rows, 
    
    Returns: 
    - Number of deleted mask rows 
    
    Note: 
    - Called once at the start of candidate_screening.run_candidate_screening()
    - This way old masks are cleared and replaced when re-running Step 1 
    for an existing run_id 
    """
    with get_session() as session:
        result = session.execute(delete(Mask).where(Mask.run_id == run_id))
        session.commit()
        return result.rowcount


def insert_mask(
    run_id: int,
    *,
    date: str,
    tile: str,
    image_id: str,
    file_path: str,
    epsg: int | None = None,
    width: int | None = None,
    height: int | None = None,
    candidate_pixel_count: int | None = None,
) -> int:
    """
    Inserts one `mask` row per exported Step 1 GeoTIFF 
    
    Returns: 
        - Mask_id 

    Note:
        - GeoTIFF itself lives on disk
        - This step enables querying masks and links them to candidate_ids
    """
    with get_session() as session:
        mask = Mask(
            run_id=run_id,
            date=_parse_yyyymmdd(date),
            tile=tile,
            image_id=image_id,
            file_path=str(file_path),
            epsg=int(epsg) if epsg is not None else None,
            width=int(width) if width is not None else None,
            height=int(height) if height is not None else None,
            candidate_pixel_count=(
                int(candidate_pixel_count) if candidate_pixel_count is not None else None
            ),
        )
        session.add(mask)
        session.commit()
        session.refresh(mask)
        return mask.id


def get_masks(run_id: int) -> pd.DataFrame:
    """
    Returns a run's `mask` rows: 
    - id, date, tile, image_id, file_path, epsg, width, height, candidate_pixel_count
    - Used by transform.run_transformation to look up which mask row a candidate 
    centroid was extracted from
    """
    stmt = select(
        Mask.id, Mask.date, Mask.tile, Mask.image_id, Mask.file_path,
        Mask.epsg, Mask.width, Mask.height, Mask.candidate_pixel_count,
    ).where(Mask.run_id == run_id)

    df = pd.read_sql(stmt, get_engine())
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    return df


def insert_candidates(run_id: int, records: list[dict]) -> pd.DataFrame:
    """
    Bulk-inserts one `candidate` row per record. 

    Functionality: 
    - Each dict in `records` must have: image_id, date, tile, col, row, lon, lat, epsg
    - mask_id optional (i.e. if mask not registered in `maks`table, e.g. when 
    working with pre-downloaded masks)
    """
    if records:
        with get_session() as session:
            candidates = [
                Candidate(
                    run_id=run_id,
                    mask_id=int(r["mask_id"]) if r.get("mask_id") is not None else None,
                    image_id=r["image_id"],
                    date=_parse_yyyymmdd(r["date"]),
                    tile=r["tile"],
                    col=float(r["col"]),
                    row=float(r["row"]),
                    geom=from_shape(Point(float(r["lon"]), float(r["lat"])), srid=4326),
                    epsg=int(r["epsg"]),
                )
                for r in records
            ]
            session.add_all(candidates)
            session.commit()

    return get_candidates(run_id)


def get_candidates(run_id: int, *, survivors_only: bool = False) -> pd.DataFrame:
    """
    Returns candidate rows for a run as a df.
    survivors_only=True shows only candidates that passed all filters in step 2
    """
    stmt = select(
        Candidate.id,
        Candidate.run_id,
        Candidate.mask_id,
        Candidate.image_id,
        Candidate.date,
        Candidate.tile,
        Candidate.col,
        Candidate.row,
        func.ST_X(Candidate.geom).label("lon"),
        func.ST_Y(Candidate.geom).label("lat"),
        Candidate.epsg,
        Candidate.cluster_excluded,
        Candidate.road_excluded,
        Candidate.road_score,
        Candidate.seam_excluded,
    ).where(Candidate.run_id == run_id)

    if survivors_only:
        stmt = stmt.where(
            Candidate.cluster_excluded.is_(False),
            Candidate.road_excluded.is_(False),
            Candidate.seam_excluded.is_(False),
        )

    df = pd.read_sql(stmt, get_engine())
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    return df


def set_candidate_flags(ids: list[int], **flags) -> None:
    """
    Bulk-sets the same flag value across many candidates, e.g.
    `set_candidate_flags(excluded_ids, cluster_excluded=True)`.
    """
    if not ids:
        return
    ids = [int(i) for i in ids]
    with get_session() as session:
        session.execute(update(Candidate).where(Candidate.id.in_(ids)).values(**flags))
        session.commit()


def bulk_update_candidates(records: list[dict]) -> None:
    """
    Updates many candidates with per-row-different values e.g. road_filter's per-candidate road_score
    """
    if not records:
        return
    with get_session() as session:
        session.bulk_update_mappings(Candidate, records)
        session.commit()


def insert_chip(candidate_id: int, run_id: int, *, file_path: str | None, status: str) -> None:
    """
    Records the download outcome for one candidate's chip
    """
    now = dt.datetime.now(dt.timezone.utc)
    stmt = pg_insert(Chip).values(
        candidate_id=int(candidate_id),
        run_id=run_id,
        file_path=str(file_path) if file_path else None,
        status=status,
        downloaded_at=now if status == "ok" else None,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[Chip.candidate_id],
        set_={
            "file_path": stmt.excluded.file_path,
            "status": stmt.excluded.status,
            "downloaded_at": stmt.excluded.downloaded_at,
        },
    )
    with get_session() as session:
        session.execute(stmt)
        session.commit()


def get_chips(run_id: int) -> pd.DataFrame:
    """
    Returns a run's `chip` rows
    """
    stmt = select(
        Chip.id, Chip.candidate_id, Chip.file_path, Chip.status, Chip.downloaded_at,
    ).where(Chip.run_id == run_id)
    return pd.read_sql(stmt, get_engine())


#---Chip cache---
def _find_closest_chip(rows: list[dict], col: float, row: float, tol_px: float) -> dict | None:
    """
    Helper fct to find a cached chip for a given (tile, date, col, row)
    """
    if not rows:
        return None
    best = min(rows, key=lambda r: (r["col"] - col) ** 2 + (r["row"] - row) ** 2)
    dist_px = ((best["col"] - col) ** 2 + (best["row"] - row) ** 2) ** 0.5
    return best if dist_px <= tol_px else None


def find_cached_chip(tile: str, date: str, col: float, row: float, tol_px: float) -> dict | None:
    """
    Returns the closest cached chip pointer for a given chip. 
    Verifies that file still exists and that checksum matches.
    """
    stmt = select(
        ChipCache.col, ChipCache.row, ChipCache.file_path, ChipCache.checksum,
    ).where(
        ChipCache.tile == tile,
        ChipCache.date == _parse_yyyymmdd(date),
    )
    with get_session() as session:
        rows = [dict(r._mapping) for r in session.execute(stmt)]
    return _find_closest_chip(rows, float(col), float(row), tol_px)


def insert_chip_cache_entry(
    tile: str, date: str, image_id: str, col: float, row: float,
    file_path: str, checksum: str,
) -> None:
    """
    Records a downloaded chip's cache pointer for future runs + upserts on file_path
    """
    stmt = pg_insert(ChipCache).values(
        tile=tile, date=_parse_yyyymmdd(date), image_id=image_id,
        col=float(col), row=float(row),
        file_path=str(file_path), checksum=checksum,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[ChipCache.file_path],
        set_={
            "tile": stmt.excluded.tile,
            "date": stmt.excluded.date,
            "image_id": stmt.excluded.image_id,
            "col": stmt.excluded.col,
            "row": stmt.excluded.row,
            "checksum": stmt.excluded.checksum,
            "cached_at": func.now(),
        },
    )
    with get_session() as session:
        session.execute(stmt)
        session.commit()


def insert_reflectance_centroids(run_id: int, records: list[dict]) -> pd.DataFrame:
    """
    Bulk-inserts one `reflectance_centroid` row per detected per-band
    anomaly centroid
    
    Note:
    - Clears a run's existing reflectance_centroid rows first
    - So re-running step 4a for a given run_id replaces its results
    """
    with get_session() as session:
        session.execute(delete(ReflectanceCentroid).where(ReflectanceCentroid.run_id == run_id))
        if records:
            rows = [
                ReflectanceCentroid(
                    run_id=run_id,
                    candidate_id=int(r["candidate_id"]),
                    band=r["band"],
                    col=float(r["col"]),
                    row=float(r["row"]),
                    area=int(r["area"]),
                )
                for r in records
            ]
            session.add_all(rows)
        session.commit()
    return get_reflectance_centroids(run_id)


def get_reflectance_centroids(run_id: int) -> pd.DataFrame:
    """
    Returns a run's `reflectance_centroid` rows
    """
    stmt = select(
        ReflectanceCentroid.id,
        ReflectanceCentroid.candidate_id,
        ReflectanceCentroid.band,
        ReflectanceCentroid.col,
        ReflectanceCentroid.row,
        ReflectanceCentroid.area,
    ).where(ReflectanceCentroid.run_id == run_id)
    return pd.read_sql(stmt, get_engine())


def insert_collinearity_triplets(run_id: int, records: list[dict]) -> pd.DataFrame:
    """
    Bulk-inserts one `collinearity_triplet` row per tested B2-B3-B4 combination 

    Note: 
    - Clears a run's existing collinearity_triplet rows first
    - So re-running step 4b for a given run_id replaces its results
    """
    with get_session() as session:
        session.execute(delete(CollinearityTriplet).where(CollinearityTriplet.run_id == run_id))
        if records:
            rows = [
                CollinearityTriplet(
                    run_id=run_id,
                    candidate_id=int(r["candidate_id"]),
                    c_b2_col=_nan_to_none(r.get("c_b2_col")),
                    c_b2_row=_nan_to_none(r.get("c_b2_row")),
                    c_b2_area=_nan_to_none(r.get("c_b2_area")),
                    c_b3_col=_nan_to_none(r.get("c_b3_col")),
                    c_b3_row=_nan_to_none(r.get("c_b3_row")),
                    c_b3_area=_nan_to_none(r.get("c_b3_area")),
                    c_b4_col=_nan_to_none(r.get("c_b4_col")),
                    c_b4_row=_nan_to_none(r.get("c_b4_row")),
                    c_b4_area=_nan_to_none(r.get("c_b4_area")),
                    angle_deg=_nan_to_none(r.get("angle_deg")),
                    colinear_found=bool(r.get("colinear_found", False)),
                )
                for r in records
            ]
            session.add_all(rows)
        session.commit()
    return get_collinearity_triplets(run_id)


def get_collinearity_triplets(run_id: int) -> pd.DataFrame:
    """
    Returns a run's `collinearity_triplet` rows
    """
    stmt = select(
        CollinearityTriplet.id,
        CollinearityTriplet.candidate_id,
        CollinearityTriplet.c_b2_col,
        CollinearityTriplet.c_b2_row,
        CollinearityTriplet.c_b2_area,
        CollinearityTriplet.c_b3_col,
        CollinearityTriplet.c_b3_row,
        CollinearityTriplet.c_b3_area,
        CollinearityTriplet.c_b4_col,
        CollinearityTriplet.c_b4_row,
        CollinearityTriplet.c_b4_area,
        CollinearityTriplet.angle_deg,
        CollinearityTriplet.colinear_found,
    ).where(CollinearityTriplet.run_id == run_id)
    return pd.read_sql(stmt, get_engine())


def insert_confirmed_detections(run_id: int, records: list[dict]) -> pd.DataFrame:
    """
    Bulk-inserts one `confirmed_detection` row per candidate (i.e. pipeline's final 
    confirmed detection result)
    
    Note:
    - Clears a run's existing confirmed_detection rows first
    - So re-running step 4c for a given run_id replaces its results 
    """
    with get_session() as session:
        session.execute(delete(ConfirmedDetection).where(ConfirmedDetection.run_id == run_id))
        if records:
            rows = [
                ConfirmedDetection(
                    run_id=run_id,
                    candidate_id=int(r["candidate_id"]),
                    confirmed=bool(r.get("confirmed", False)),
                    d_band2_3_m=_nan_to_none(r.get("D_Band2_3_m")),
                    d_band3_4_m=_nan_to_none(r.get("D_Band3_4_m")),
                    residual_m=_nan_to_none(r.get("residual_m")),
                    angle_deg=_nan_to_none(r.get("angle_deg")),
                    speed_kmh=_nan_to_none(r.get("speed_kmh")),
                    c_b2_col=_nan_to_none(r.get("c_b2_col")),
                    c_b2_row=_nan_to_none(r.get("c_b2_row")),
                    c_b3_col=_nan_to_none(r.get("c_b3_col")),
                    c_b3_row=_nan_to_none(r.get("c_b3_row")),
                    c_b4_col=_nan_to_none(r.get("c_b4_col")),
                    c_b4_row=_nan_to_none(r.get("c_b4_row")),
                )
                for r in records
            ]
            session.add_all(rows)
        session.commit()
    return get_confirmed_detections(run_id)


def get_confirmed_detections(run_id: int) -> pd.DataFrame:
    """
    Returns this run's `confirmed_detection` rows
    """
    stmt = select(
        ConfirmedDetection.id,
        ConfirmedDetection.candidate_id,
        ConfirmedDetection.confirmed,
        ConfirmedDetection.d_band2_3_m,
        ConfirmedDetection.d_band3_4_m,
        ConfirmedDetection.residual_m,
        ConfirmedDetection.angle_deg,
        ConfirmedDetection.speed_kmh,
        ConfirmedDetection.c_b2_col,
        ConfirmedDetection.c_b2_row,
        ConfirmedDetection.c_b3_col,
        ConfirmedDetection.c_b3_row,
        ConfirmedDetection.c_b4_col,
        ConfirmedDetection.c_b4_row,
    ).where(ConfirmedDetection.run_id == run_id)
    return pd.read_sql(stmt, get_engine())


def get_confirmed_detections_with_candidates(run_id: int) -> pd.DataFrame:
    """
    Returns this run's `confirmed_detection` rows joined with their candidate's 
    image_id/tile/date/lon/lat
    """
    stmt = (
        select(
            ConfirmedDetection.candidate_id,
            ConfirmedDetection.confirmed,
            ConfirmedDetection.speed_kmh,
            ConfirmedDetection.angle_deg,
            ConfirmedDetection.residual_m,
            Candidate.image_id,
            Candidate.tile,
            Candidate.date,
            func.ST_X(Candidate.geom).label("lon"),
            func.ST_Y(Candidate.geom).label("lat"),
        )
        .join(Candidate, Candidate.id == ConfirmedDetection.candidate_id)
        .where(ConfirmedDetection.run_id == run_id)
    )
    df = pd.read_sql(stmt, get_engine())
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    return df



#---global caches---
def get_tile_scan_params(tile: str, date: str) -> dict | None:
    """
    Returns cached Sentinel-2 acquisition parameters for a (tile, date) pair
    """
    stmt = select(TileScanParams).where(
        TileScanParams.tile == tile,
        TileScanParams.date == dt.date.fromisoformat(date),
    )
    with get_session() as session:
        row = session.execute(stmt).scalar_one_or_none()
        if row is None:
            return None
        return {
            "tile": row.tile,
            "date": row.date.strftime("%Y-%m-%d"),
            "t_top": row.t_top,
            "t_bottom": row.t_bottom,
            "t_query_start": row.t_query_start,
            "t_query_end": row.t_query_end,
            "west": row.west,
            "east": row.east,
            "south": row.south,
            "north": row.north,
            "cloud_pct": row.cloud_pct,
        }


def insert_tile_scan_params(params: dict) -> None:
    """
    Caches Sentinel-2 acquisition parameters for a (tile, date) pair
    """
    with get_session() as session:
        row = TileScanParams(
            tile=params["tile"],
            date=dt.date.fromisoformat(params["date"]),
            t_top=params["t_top"],
            t_bottom=params["t_bottom"],
            t_query_start=params["t_query_start"],
            t_query_end=params["t_query_end"],
            west=float(params["west"]),
            east=float(params["east"]),
            south=float(params["south"]),
            north=float(params["north"]),
            cloud_pct=_nan_to_none(params.get("cloud_pct")),
        )
        session.add(row)
        session.commit()


def get_adsb_cache_entry(tile: str, date: str) -> dict | None:
    """
    Returns the cached ADS-B query pointer for a (tile, date) pair.
    Cached file stays on disk
    """
    stmt = select(AdsbQueryCache).where(
        AdsbQueryCache.tile == tile,
        AdsbQueryCache.date == dt.date.fromisoformat(date),
    )
    with get_session() as session:
        row = session.execute(stmt).scalar_one_or_none()
        if row is None:
            return None
        return {"image_id": row.image_id, "file_path": row.file_path, "row_count": row.row_count}


def insert_adsb_cache_entry(tile: str, date: str, image_id: str, file_path: str, row_count: int) -> None:
    """
    Records where an ADS-B Parquet cache file lives
    """
    stmt = pg_insert(AdsbQueryCache).values(
        tile=tile,
        date=dt.date.fromisoformat(date),
        image_id=image_id,
        file_path=str(file_path),
        row_count=int(row_count),
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[AdsbQueryCache.tile, AdsbQueryCache.date],
        set_={
            "image_id": stmt.excluded.image_id,
            "file_path": stmt.excluded.file_path,
            "row_count": stmt.excluded.row_count,
        },
    )
    with get_session() as session:
        session.execute(stmt)
        session.commit()


#---Evaluation---
def clear_evaluation_points(run_id: int) -> int:
    """
    Deletes this run's existing `evaluation_point` rows
    
    Returns: 
    - number of evaluation points removed

    Note: 
    - Called once at the start of evaluate.run_evaluation() before ads-b cross-referencing
    - Re-running step 5 for an existing run_id replaces results
    """
    with get_session() as session:
        result = session.execute(delete(EvaluationPoint).where(EvaluationPoint.run_id == run_id))
        session.commit()
        return result.rowcount


def insert_evaluation_points(run_id: int, records: list[dict]) -> None:
    """
    Bulk-inserts one `evaluation_point` row per pipeline detection or ADS-B record for a scene
    """
    if not records:
        return
    with get_session() as session:
        rows = [
            EvaluationPoint(
                run_id=run_id,
                image_id=r.get("image_id"),
                source=r["source"],
                candidate_id=int(r["candidate_id"]) if r.get("candidate_id") is not None else None,
                geom=from_shape(Point(float(r["lon"]), float(r["lat"])), srid=4326),
                in_pipeline=bool(r.get("in_pipeline", False)),
                in_adsb=bool(r.get("in_adsb", False)),
                matched=bool(r.get("matched", False)),
                match_distance_m=_nan_to_none(r.get("match_distance_m")),
                matched_icao24=r.get("matched_icao24"),
                matched_callsign=r.get("matched_callsign"),
                icao24=r.get("icao24"),
                callsign=r.get("callsign"),
                baroaltitude=_nan_to_none(r.get("baroaltitude")),
                geoaltitude=_nan_to_none(r.get("geoaltitude")),
                heading=_nan_to_none(r.get("heading")),
                speed_kmh=_nan_to_none(r.get("speed_kmh")),
                angle_deg=_nan_to_none(r.get("angle_deg")),
                residual_m=_nan_to_none(r.get("residual_m")),
                outside_bounds=bool(r.get("outside_bounds", False)),
                cloud_pct=_nan_to_none(r.get("cloud_pct")),
            )
            for r in records
        ]
        session.add_all(rows)
        session.commit()


_EVAL_POINT_COLUMNS = (
    EvaluationPoint.id,
    EvaluationPoint.image_id,
    EvaluationPoint.source,
    EvaluationPoint.candidate_id,
    EvaluationPoint.in_pipeline,
    EvaluationPoint.in_adsb,
    EvaluationPoint.matched,
    EvaluationPoint.match_distance_m,
    EvaluationPoint.matched_icao24,
    EvaluationPoint.matched_callsign,
    EvaluationPoint.icao24,
    EvaluationPoint.callsign,
    EvaluationPoint.baroaltitude,
    EvaluationPoint.geoaltitude,
    EvaluationPoint.heading,
    EvaluationPoint.speed_kmh,
    EvaluationPoint.angle_deg,
    EvaluationPoint.residual_m,
    EvaluationPoint.outside_bounds,
    EvaluationPoint.inspected,
    EvaluationPoint.is_visible,
    EvaluationPoint.is_flying,
    EvaluationPoint.seamline_duplicate,
    EvaluationPoint.near_bounds,
    EvaluationPoint.background_type,
    EvaluationPoint.notes,
    EvaluationPoint.note_code,
    EvaluationPoint.cloud_pct,
)


def get_evaluation_points(run_id: int, image_id: str | None = None) -> gpd.GeoDataFrame:
    """
    Returns this run's `evaluation_point` rows as a GeoDataFrame
    """
    stmt = select(
        *_EVAL_POINT_COLUMNS,
        func.ST_X(EvaluationPoint.geom).label("lon"),
        func.ST_Y(EvaluationPoint.geom).label("lat"),
    ).where(EvaluationPoint.run_id == run_id)
    if image_id is not None:
        stmt = stmt.where(EvaluationPoint.image_id == image_id)

    df = pd.read_sql(stmt, get_engine())
    geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])] if not df.empty else []
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


def update_evaluation_point_annotations(records: list[dict]) -> None:
    """
    Updates annotation fields (from the QGIS ground-truth annotations) on
    existing `evaluation_point` rows. Each dict must have an id (the
    evaluation_point's own primary key), since rows are updated based on it.
    """
    if not records:
        return
    now = dt.datetime.now(dt.timezone.utc)
    boolean_fields = ("is_visible", "is_flying", "seamline_duplicate", "near_bounds")
    for r in records:
        r.setdefault("annotated_at", now)
        for field in boolean_fields:
            if field in r:
                r[field] = _nan_to_none(r[field])
        if "background_type" in r:
            r["background_type"] = _nan_to_int(r["background_type"])
        if "id" in r:
            r["id"] = int(r["id"])
    with get_session() as session:
        session.bulk_update_mappings(EvaluationPoint, records)
        session.commit()


def insert_manual_evaluation_points(run_id: int, records: list[dict]) -> None:
    """
    Inserts new `evaluation_point` rows for points added manually during 
    manual annotation (e.g. in QGIS)
    """
    if not records:
        return
    now = dt.datetime.now(dt.timezone.utc)
    with get_session() as session:
        rows = [
            EvaluationPoint(
                run_id=run_id,
                image_id=r.get("image_id"),
                source="manual",
                geom=from_shape(Point(float(r["lon"]), float(r["lat"])), srid=4326),
                in_pipeline=bool(r.get("in_pipeline", False)),
                in_adsb=bool(r.get("in_adsb", False)),
                inspected=bool(r.get("inspected", True)),
                is_visible=_nan_to_none(r.get("is_visible")),
                is_flying=_nan_to_none(r.get("is_flying")),
                seamline_duplicate=_nan_to_none(r.get("seamline_duplicate")),
                near_bounds=_nan_to_none(r.get("near_bounds")),
                background_type=_nan_to_int(r.get("background_type")),
                notes=r.get("notes"),
                note_code=r.get("note_code"),
                cloud_pct=_nan_to_none(r.get("cloud_pct")),
                annotated_at=now,
            )
            for r in records
        ]
        session.add_all(rows)
        session.commit()


#---evaluation metrics---
def insert_metrics_by_background(run_id: int, df: pd.DataFrame) -> None:
    """
    Bulk-inserts the by-background metrics table
    """
    if df.empty:
        return
    with get_session() as session:
        rows = [
            MetricsByBackground(
                run_id=run_id,
                background_type=int(r["background_type"]),
                background_label=r["background_label"],
                tp=int(r["TP"]), fp=int(r["FP"]), fn=int(r["FN"]), tp1=int(r["TP1"]),
                n_adsb_total=int(r["n_adsb_total"]),
                n_adsb_visible=int(r["n_adsb_visible"]),
                precision=_nan_to_none(r.get("precision")),
                recall=_nan_to_none(r.get("recall")),
                f1=_nan_to_none(r.get("f1")),
                fpr=_nan_to_none(r.get("fpr")),
                fnr=_nan_to_none(r.get("fnr")),
                detectable_fraction=_nan_to_none(r.get("detectable_fraction")),
                tp1_pct=_nan_to_none(r.get("tp1_pct")),
            )
            for r in df.to_dict("records")
        ]
        session.add_all(rows)
        session.commit()


def get_metrics_by_background(run_id: int) -> pd.DataFrame:
    """
    Returns this run's `metrics_by_background` rows
    """
    stmt = select(
        MetricsByBackground.background_type, MetricsByBackground.background_label,
        MetricsByBackground.tp, MetricsByBackground.fp,
        MetricsByBackground.fn, MetricsByBackground.tp1,
        MetricsByBackground.n_adsb_total, MetricsByBackground.n_adsb_visible,
        MetricsByBackground.precision, MetricsByBackground.recall, MetricsByBackground.f1,
        MetricsByBackground.fpr, MetricsByBackground.fnr,
        MetricsByBackground.detectable_fraction, MetricsByBackground.tp1_pct,
    ).where(MetricsByBackground.run_id == run_id)
    return pd.read_sql(stmt, get_engine())


def insert_metrics_by_scene(run_id: int, df: pd.DataFrame) -> None:
    """
    Bulk-inserts the per-scene metrics table
    """
    if df.empty:
        return
    with get_session() as session:
        rows = [
            MetricsByScene(
                run_id=run_id,
                image_id=r["image_id"],
                tile=r.get("tile"),
                date=dt.datetime.strptime(r["date"], "%Y%m%d").date() if r.get("date") else None,
                cloud_pct=_nan_to_none(r.get("cloud_pct")),
                modal_background=_nan_to_int(r.get("modal_background")),
                tp=int(r["TP"]), fp=int(r["FP"]), fn=int(r["FN"]), tp1=int(r["TP1"]),
                n_adsb_total=int(r["n_adsb_total"]),
                n_adsb_visible=int(r["n_adsb_visible"]),
                precision=_nan_to_none(r.get("precision")),
                recall=_nan_to_none(r.get("recall")),
                f1=_nan_to_none(r.get("f1")),
                fpr=_nan_to_none(r.get("fpr")),
                fnr=_nan_to_none(r.get("fnr")),
                detectable_fraction=_nan_to_none(r.get("detectable_fraction")),
                tp1_pct=_nan_to_none(r.get("tp1_pct")),
            )
            for r in df.to_dict("records")
        ]
        session.add_all(rows)
        session.commit()


def get_metrics_by_scene(run_id: int) -> pd.DataFrame:
    """
    Returns a run's `metrics_by_scene` rows
    """
    stmt = select(
        MetricsByScene.image_id, MetricsByScene.tile, MetricsByScene.date,
        MetricsByScene.cloud_pct, MetricsByScene.modal_background,
        MetricsByScene.tp, MetricsByScene.fp, MetricsByScene.fn, MetricsByScene.tp1,
        MetricsByScene.n_adsb_total, MetricsByScene.n_adsb_visible,
        MetricsByScene.precision, MetricsByScene.recall, MetricsByScene.f1,
        MetricsByScene.fpr, MetricsByScene.fnr,
        MetricsByScene.detectable_fraction, MetricsByScene.tp1_pct,
    ).where(MetricsByScene.run_id == run_id)
    return pd.read_sql(stmt, get_engine())
