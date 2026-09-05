"""
SQLAlchemy ORM models for the Postgres/PostGIS schema

Tables:
  pipeline_run           — one row per run; the run/job identity everything else is scoped by
  mask                    — Step 1 raster metadata, one row per exported mask
  candidate               — Step 2's evolving table, one row per candidate centroid
  chip                    — Step 3, one row per downloaded chip
  chip_cache              — global cache pointer to downloaded chip GeoTIFFs (not run-scoped)
  reflectance_centroid    — Step 4a, one row per per-band anomaly centroid
  collinearity_triplet    — Step 4b, one row per tested B2-B3-B4 triplet
  confirmed_detection     — Step 4c terminal table, one row per candidate's final result
  tile_scan_params        — global cache of Sentinel-2 acquisition timing (not run-scoped)
  adsb_query_cache        — global cache pointer to ADS-B parquet files (not run-scoped)
  evaluation_point        — Step 5, one row per pipeline/ADS-B/manual point
  metrics_by_background   — Step 5 terminal metrics, stratified by background type
  metrics_by_scene        — Step 5 terminal metrics, per scene

Design choices:

- `Candidate`: evolves through the whole Step 2 filter chain. Each
  filter sub-stage adds a boolean exclusion column.
- `EvaluationPoint`: one row per point (pipeline detection, ADS-B
  record, manual detection), initially populated by ADS-B cross-reference step, 
  optionally extended after manual review and annotation in QGIS. 
- `run_id`: every run-scoped table carries a `run_id` FK to
  `PipelineRun`, so multiple runs (even over same AOI) never mix.
- `TileScanParams`/`AdsbQueryCache`/`ChipCache`: not run-scoped, since
  acquisition timing, ADS-B records, and candidate positions (deterministic
  given the same input masks) are all independent of any one pipeline run.
- On-disk files (masks, chips, ADS-B parquet) stay on disk with `file_path`
  columns pointing to them.
- Ownership FKs (run_id, candidate_id) cascade on delete, i.e. deleting a 
  run cleans up everything under it in one statement. 
- Informational-only FKs (`Candidate.mask_id`, `EvaluationPoint.candidate_id`)
  use SET NULL instead (i.e. row will not be deleted)
- `Candidate` has a unique constraint on (run_id, image_id, col, row).
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from geoalchemy2 import Geometry
from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    SmallInteger,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class PipelineRun(Base):
    """
    One row per pipeline execution which scopes the run identitiy of all tables 
    """

    __tablename__ = "pipeline_run"

    id: Mapped[int] = mapped_column(primary_key=True)
    aoi_name: Mapped[str | None] = mapped_column(String, nullable=True)
    aoi_geom: Mapped[Any | None] = mapped_column(
        Geometry("POLYGON", srid=4326), nullable=True
    )
    lon_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    lat_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    lat_max: Mapped[float | None] = mapped_column(Float, nullable=True)
    start_date: Mapped[dt.date] = mapped_column(Date)
    end_date: Mapped[dt.date] = mapped_column(Date)
    tile_name: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="running")
    config_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    notes: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Mask(Base):
    """
    Step 1 raster metadata: one row per exported GeoTIFF mask
    """

    __tablename__ = "mask"
    __table_args__ = (
        UniqueConstraint("run_id", "image_id", name="uq_mask_run_image"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("pipeline_run.id", ondelete="CASCADE"), index=True
    )
    date: Mapped[dt.date] = mapped_column(Date)
    tile: Mapped[str] = mapped_column(String)
    image_id: Mapped[str] = mapped_column(String)
    file_path: Mapped[str] = mapped_column(String)
    epsg: Mapped[int | None] = mapped_column(Integer, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    candidate_pixel_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Candidate(Base):
    """
    Step 2's evolving table: 
    - one row per candidate pixel-blob centroid
    - carried through the full filter chain
    """

    __tablename__ = "candidate"
    __table_args__ = (
        Index("ix_candidate_run_image", "run_id", "image_id"),
        UniqueConstraint(
            "run_id", "image_id", "col", "row",
            name="uq_candidate_run_image_col_row",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("pipeline_run.id", ondelete="CASCADE"), index=True
    )
    mask_id: Mapped[int | None] = mapped_column(
        ForeignKey("mask.id", ondelete="SET NULL"), nullable=True
    )
    image_id: Mapped[str] = mapped_column(String)
    date: Mapped[dt.date] = mapped_column(Date)
    tile: Mapped[str] = mapped_column(String)
    col: Mapped[float] = mapped_column(Float)
    row: Mapped[float] = mapped_column(Float)
    geom: Mapped[Any] = mapped_column(Geometry("POINT", srid=4326))
    epsg: Mapped[int] = mapped_column(Integer)

    # One boolean per Step 2 filter sub-stage
    # A candidate is a final Step 2 survivor if none of these are true
    cluster_excluded: Mapped[bool] = mapped_column(Boolean, default=False)
    road_excluded: Mapped[bool] = mapped_column(Boolean, default=False)
    road_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    seam_excluded: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Chip(Base):
    """
    Step 3: one downloaded image chip per candidate
    """

    __tablename__ = "chip"

    id: Mapped[int] = mapped_column(primary_key=True)
    candidate_id: Mapped[int] = mapped_column(
        ForeignKey("candidate.id", ondelete="CASCADE"), unique=True
    )
    run_id: Mapped[int] = mapped_column(
        ForeignKey("pipeline_run.id", ondelete="CASCADE"), index=True
    )
    file_path: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String)  # 'ok' / 'failed' / 'missing'
    downloaded_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class ChipCache(Base):
    """
    Global cache pointer to downloaded Sentinel-2 chip GeoTIFFs

    Functionality:
    - Not run-scoped
    - Unique filepath from image_id, col, row --> a second write to the same path upserts
    the existing row 
    - checksum (sha256 of file bytes) lets a cache hit verify the file on
      disk hasn't been corrupted/replaced before reuse
    """

    __tablename__ = "chip_cache"
    __table_args__ = (
        Index("ix_chip_cache_tile_date", "tile", "date"),
        UniqueConstraint("file_path", name="uq_chip_cache_file_path"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    tile: Mapped[str] = mapped_column(String)
    date: Mapped[dt.date] = mapped_column(Date)
    image_id: Mapped[str] = mapped_column(String)
    col: Mapped[float] = mapped_column(Float)
    row: Mapped[float] = mapped_column(Float)
    file_path: Mapped[str] = mapped_column(String)
    checksum: Mapped[str] = mapped_column(String)
    cached_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ReflectanceCentroid(Base):
    """
    Step 4a:  one row per detected per-band anomaly centroid per candidate
    """

    __tablename__ = "reflectance_centroid"

    id: Mapped[int] = mapped_column(primary_key=True)
    candidate_id: Mapped[int] = mapped_column(
        ForeignKey("candidate.id", ondelete="CASCADE"), index=True
    )
    run_id: Mapped[int] = mapped_column(
        ForeignKey("pipeline_run.id", ondelete="CASCADE"), index=True
    )
    band: Mapped[str] = mapped_column(String)  # 'B2'/'B3'/'B4'
    col: Mapped[float] = mapped_column(Float)
    row: Mapped[float] = mapped_column(Float)
    area: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class CollinearityTriplet(Base):
    """
    Step 4b: One row per tested B2-B3-B4 centroid triplet per candidate
    """

    __tablename__ = "collinearity_triplet"

    id: Mapped[int] = mapped_column(primary_key=True)
    candidate_id: Mapped[int] = mapped_column(
        ForeignKey("candidate.id", ondelete="CASCADE"), index=True
    )
    run_id: Mapped[int] = mapped_column(
        ForeignKey("pipeline_run.id", ondelete="CASCADE"), index=True
    )
    c_b2_col: Mapped[float | None] = mapped_column(Float, nullable=True)
    c_b2_row: Mapped[float | None] = mapped_column(Float, nullable=True)
    c_b2_area: Mapped[float | None] = mapped_column(Float, nullable=True)
    c_b3_col: Mapped[float | None] = mapped_column(Float, nullable=True)
    c_b3_row: Mapped[float | None] = mapped_column(Float, nullable=True)
    c_b3_area: Mapped[float | None] = mapped_column(Float, nullable=True)
    c_b4_col: Mapped[float | None] = mapped_column(Float, nullable=True)
    c_b4_row: Mapped[float | None] = mapped_column(Float, nullable=True)
    c_b4_area: Mapped[float | None] = mapped_column(Float, nullable=True)
    angle_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    colinear_found: Mapped[bool] = mapped_column(Boolean, default=False)


class ConfirmedDetection(Base):
    """
    Step 4c: Pipeline's terminal per-candidate detections table
    """

    __tablename__ = "confirmed_detection"

    id: Mapped[int] = mapped_column(primary_key=True)
    candidate_id: Mapped[int] = mapped_column(
        ForeignKey("candidate.id", ondelete="CASCADE"), unique=True
    )
    run_id: Mapped[int] = mapped_column(
        ForeignKey("pipeline_run.id", ondelete="CASCADE"), index=True
    )
    confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    d_band2_3_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    d_band3_4_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    residual_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    angle_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    speed_kmh: Mapped[float | None] = mapped_column(Float, nullable=True)
    c_b2_col: Mapped[float | None] = mapped_column(Float, nullable=True)
    c_b2_row: Mapped[float | None] = mapped_column(Float, nullable=True)
    c_b3_col: Mapped[float | None] = mapped_column(Float, nullable=True)
    c_b3_row: Mapped[float | None] = mapped_column(Float, nullable=True)
    c_b4_col: Mapped[float | None] = mapped_column(Float, nullable=True)
    c_b4_row: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class TileScanParams(Base):
    """
    Global cache for S2 tile metadata, not run-scoped
    """

    __tablename__ = "tile_scan_params"
    __table_args__ = (
        UniqueConstraint("tile", "date", name="uq_tile_scan_params_tile_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    tile: Mapped[str] = mapped_column(String)
    date: Mapped[dt.date] = mapped_column(Date)
    t_top: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    t_bottom: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    t_query_start: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    t_query_end: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    west: Mapped[float] = mapped_column(Float)
    east: Mapped[float] = mapped_column(Float)
    south: Mapped[float] = mapped_column(Float)
    north: Mapped[float] = mapped_column(Float)
    cloud_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    fetched_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AdsbQueryCache(Base):
    """
    Global cache pointer to ADS-B parquet files
    - not run-scoped
    - parquet files stored on disk
    """

    __tablename__ = "adsb_query_cache"
    __table_args__ = (
        UniqueConstraint("tile", "date", name="uq_adsb_query_cache_tile_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    tile: Mapped[str] = mapped_column(String)
    date: Mapped[dt.date] = mapped_column(Date)
    image_id: Mapped[str] = mapped_column(String)
    file_path: Mapped[str] = mapped_column(String)
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    queried_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class EvaluationPoint(Base):
    """
    Step 5:
    — one row per point (pipeline detection, ADS-B record, or a manually-added QGIS point)
    - covers matching, ground truth, and annotation in a single table
    """

    __tablename__ = "evaluation_point"
    __table_args__ = (
        Index("ix_evaluation_point_run_image", "run_id", "image_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("pipeline_run.id", ondelete="CASCADE"), index=True
    )
    image_id: Mapped[str] = mapped_column(String)
    source: Mapped[str] = mapped_column(String)  # 'pipeline' / 'adsb' / 'manual'
    candidate_id: Mapped[int | None] = mapped_column(
        ForeignKey("candidate.id", ondelete="SET NULL"), nullable=True
    )
    geom: Mapped[Any] = mapped_column(Geometry("POINT", srid=4326))

    in_pipeline: Mapped[bool] = mapped_column(Boolean, default=False)
    in_adsb: Mapped[bool] = mapped_column(Boolean, default=False)
    matched: Mapped[bool] = mapped_column(Boolean, default=False)
    match_distance_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    matched_icao24: Mapped[str | None] = mapped_column(String, nullable=True)
    matched_callsign: Mapped[str | None] = mapped_column(String, nullable=True)

    icao24: Mapped[str | None] = mapped_column(String, nullable=True)
    callsign: Mapped[str | None] = mapped_column(String, nullable=True)
    baroaltitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    geoaltitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    heading: Mapped[float | None] = mapped_column(Float, nullable=True)
    speed_kmh: Mapped[float | None] = mapped_column(Float, nullable=True)
    angle_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    residual_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    outside_bounds: Mapped[bool] = mapped_column(Boolean, default=False)

    # Annotation fields: NULL until populated through manual annotation (e.g. in QGIS)
    inspected: Mapped[bool] = mapped_column(Boolean, default=False)
    is_visible: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_flying: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    seamline_duplicate: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    near_bounds: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    background_type: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    notes: Mapped[str | None] = mapped_column(String, nullable=True)
    note_code: Mapped[str | None] = mapped_column(String, nullable=True)
    cloud_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    annotated_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class MetricsByBackground(Base):
    """
    Step 5: terminal metrics after manual annotation stratified by background
    """

    __tablename__ = "metrics_by_background"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("pipeline_run.id", ondelete="CASCADE"), index=True
    )
    background_type: Mapped[int] = mapped_column(SmallInteger)
    background_label: Mapped[str] = mapped_column(String)
    tp: Mapped[int] = mapped_column(Integer)
    fp: Mapped[int] = mapped_column(Integer)
    fn: Mapped[int] = mapped_column(Integer)
    tp1: Mapped[int] = mapped_column(Integer)
    n_adsb_total: Mapped[int] = mapped_column(Integer)
    n_adsb_visible: Mapped[int] = mapped_column(Integer)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    fpr: Mapped[float | None] = mapped_column(Float, nullable=True)
    fnr: Mapped[float | None] = mapped_column(Float, nullable=True)
    detectable_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)
    tp1_pct: Mapped[float | None] = mapped_column(Float, nullable=True)


class MetricsByScene(Base):
    """
    Step 5: terminal metrics after manual annotation 
    """

    __tablename__ = "metrics_by_scene"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("pipeline_run.id", ondelete="CASCADE"), index=True
    )
    image_id: Mapped[str] = mapped_column(String)
    tile: Mapped[str | None] = mapped_column(String, nullable=True)
    date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    cloud_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    modal_background: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    tp: Mapped[int] = mapped_column(Integer)
    fp: Mapped[int] = mapped_column(Integer)
    fn: Mapped[int] = mapped_column(Integer)
    tp1: Mapped[int] = mapped_column(Integer)
    n_adsb_total: Mapped[int] = mapped_column(Integer)
    n_adsb_visible: Mapped[int] = mapped_column(Integer)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    fpr: Mapped[float | None] = mapped_column(Float, nullable=True)
    fnr: Mapped[float | None] = mapped_column(Float, nullable=True)
    detectable_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)
    tp1_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
