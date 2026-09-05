"""
Pipeline orchestration:
- entry point that ties Steps 1-4 together 
- candidate screening -> filtering -> chip download -> confirmation
- Step 5's ADS-B cross-reference is included optional fifth step
- Evaluation based on manually annotated files is not included since this orchestration 
script only includes full automated steps that run end-to-end.
"""

from __future__ import annotations

import ee

from pipeline.db import repository
from pipeline import config as pipeline_config
from pipeline.utils import io
from pipeline.s1_gee_candidates import aoi_selection, candidate_screening
from pipeline.s2_filter import transform, cluster_filter, road_filter, seamline_filter
from pipeline.s3_download import chip_download
from pipeline.s4_confirmation import reflectance_check, collinearity_check, displacement_check
from pipeline.s5_evaluation import evaluate, export


STEP_ORDER = ["step1", "step2", "step3", "step4", "step5"]

def _config_snapshot() -> dict:
    """
    Gets algorithm parameters for run
    """
    return {
        name: value
        for name, value in vars(pipeline_config).items()
        if name.isupper() and not name.startswith("_") and not callable(value)
    }


class Pipeline:
    """
    Orchestrates Steps 1-4 (core detection) plus an optional Step 5
    (ADS-B cross-reference) for one pipeline_run. 
    
    Each run defaults to self.run_id (set by start_run()) but accepts an explicit
    run_id override
    """

    def __init__(self):
        self.run_id: int | None = None
        self._aoi = None

    def start_run(
        self,
        aoi,
        start_date: str,
        end_date: str,
        tile_name: str | None = None,
        aoi_name: str | None = None,
        notes: str | None = None,
    ) -> int:
        """
        Creates a new pipeline_run and sets it as this instance's active run 
        """
        bbox = aoi_selection.get_bbox(aoi)
        self.run_id = repository.create_pipeline_run(
            start_date=start_date,
            end_date=end_date,
            aoi_name=aoi_name,
            tile_name=tile_name,
            notes=notes,
            #stores aoi so it can be reconstructed 
            lon_min=bbox["aoi_lon_min"], lon_max=bbox["aoi_lon_max"],
            lat_min=bbox["aoi_lat_min"], lat_max=bbox["aoi_lat_max"],
            config_snapshot=_config_snapshot(),
        )
        self._aoi = aoi
        return self.run_id

    def start_demo_run(
        self,
        mask_dir: str,
        start_date: str,
        end_date: str,
        tile_name: str | None = None,
        aoi_name: str | None = None,
        notes: str | None = None,
    ) -> int:
        """
        Creates a pipeline_run without requiring GEE, for exploring
        Steps 2-4 against a pre-supplied set of Step 1 masks. 

        - Derives box directly from every mask file's own embedded geotransform 
        - No ee.Geometry used 
        - Intended for demonstration purposes only without GEE access
        """
        mask_paths = io.list_files(mask_dir, "*_candidates.tif")
        bboxes = [io.get_mask_bbox_wgs84(p) for p in mask_paths]
        lon_min = min(b["west"] for b in bboxes)
        lon_max = max(b["east"] for b in bboxes)
        lat_min = min(b["south"] for b in bboxes)
        lat_max = max(b["north"] for b in bboxes)

        self.run_id = repository.create_pipeline_run(
            start_date=start_date,
            end_date=end_date,
            aoi_name=aoi_name,
            tile_name=tile_name,
            notes=notes,
            lon_min=lon_min, lon_max=lon_max,
            lat_min=lat_min, lat_max=lat_max,
            config_snapshot=_config_snapshot(),
        )
        self._aoi = None
        return self.run_id

    def _resolve_run_id(self, run_id: int | None) -> int:
        run_id = run_id if run_id is not None else self.run_id
        if run_id is None:
            raise ValueError(
                "No run_id available. Call start_run() first, or pass run_id explicitly."
            )
        return run_id

    # Pipeline steps
    def run_step1(self, run_id=None, aoi=None, start_date=None, end_date=None,
                  tile_name=None, mask_dir=None):
        """
        Runs Step 1 (GEE candidate screening)
        """
        run_id = self._resolve_run_id(run_id)
        run = repository.get_pipeline_run(run_id)
        if run is None:
            raise ValueError(f"No pipeline_run found for run_id={run_id}")

        aoi = aoi if aoi is not None else self._aoi
        if aoi is None and None not in (run["lon_min"], run["lon_max"], run["lat_min"], run["lat_max"]):
            aoi = ee.Geometry.Rectangle(
                [run["lon_min"], run["lat_min"], run["lon_max"], run["lat_max"]]
            )
        if aoi is None:
            raise ValueError(
                "run_step1() needs an aoi — pass one explicitly, or use a run_id "
                "whose start_run() call stored a bounding box."
            )

        start_date = start_date or run["start_date"].isoformat()
        end_date = end_date or run["end_date"].isoformat()
        tile_name = tile_name or run["tile_name"]

        return candidate_screening.run_candidate_screening(
            run_id, aoi, start_date, end_date, tile_name=tile_name, mask_dir=mask_dir
        )

    def run_step2(self, run_id=None, df=None):
        """
        Runs Step 2's four-stage filter (transform -> cluster -> road -> seamline)
        """
        run_id = self._resolve_run_id(run_id)
        df1 = transform.run_transformation(run_id) if df is None else df
        df2 = cluster_filter.run_cluster_exclusion(run_id, df1)
        df3 = road_filter.run_road_exclusion(run_id, df2)
        df4 = seamline_filter.run_seamline_exclusion(run_id, df3)
        return df4

    def run_step3(self, run_id=None, df=None):
        """
        Runs Step 3 (chip download)
        """
        run_id = self._resolve_run_id(run_id)
        survivors = df if df is not None else repository.get_candidates(run_id, survivors_only=True)
        chip_download.run_chip_download(run_id, survivors)
        return survivors

    def run_step4(self, run_id=None, df=None):
        """
        Runs Step 4's three-stage confirmation (reflectance -> collinearity -> displacement)
        """
        run_id = self._resolve_run_id(run_id)
        df5 = reflectance_check.run_reflectance_check(run_id, df)
        df6 = collinearity_check.run_collinearity_check(run_id, df5)
        df7 = displacement_check.run_displacement_check(run_id, df6)
        return df7

    def run_step5(self, run_id=None):
        """
        Runs Step 5's ADS-B cross-reference:

        Needs OpenSky credentials for (tile, date) scenes not already
        cached in adsb_query_cache, and GEE credentials for (tile, date)
        scenes not already cached in tile_scan_params.
        """
        run_id = self._resolve_run_id(run_id)
        results = evaluate.run_evaluation(run_id)
        export.run_export(run_id, results)
        return results

    #---main---
    def run(self, run_id=None, from_step: str = "step1", to_step: str = "step4", **step1_kwargs):
        """
        Runs steps from_step..to_step in sequence:
        - chains each stage's return value into the next so DB does not need to be requeried per step
        - Marks the run 'completed' or 'failed' when done
        """
        run_id = self._resolve_run_id(run_id)
        steps = {
            "step1": lambda df: self.run_step1(run_id=run_id, **step1_kwargs),
            "step2": lambda df: self.run_step2(run_id=run_id, df=df),
            "step3": lambda df: self.run_step3(run_id=run_id, df=df),
            "step4": lambda df: self.run_step4(run_id=run_id, df=df),
            "step5": lambda df: self.run_step5(run_id=run_id),
        }
        start_idx = STEP_ORDER.index(from_step)
        end_idx   = STEP_ORDER.index(to_step)

        df = None
        try:
            for step_name in STEP_ORDER[start_idx:end_idx + 1]:
                print(f"\n{'#' * 50}\n# {step_name}\n{'#' * 50}")
                df = steps[step_name](df)
            repository.update_run_status(run_id, "completed")
        except Exception:
            repository.update_run_status(run_id, "failed")
            raise

        return df
