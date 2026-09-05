"""
CLI wrapper over pipeline.Pipeline 
"""

import subprocess

import click
import ee

from pipeline.db import repository
from pipeline.pipeline import Pipeline, STEP_ORDER
from pipeline.utils import gee_auth

_STEP_CHOICES = click.Choice(STEP_ORDER)
_GEE_STEPS = {"step1", "step3", "step5"}


@click.group()
def main():
    """
    parallax: CLI for the parallax aircraft-detection pipeline

    - Steps 1-4 are core detection pipeline and need no ADS-B access
    - Step 5 (--to-step step5) cross-references confirmed detections against OpenSky ADS-B data 
    which is not ran by default
    """

@main.command("run")
@click.option("--tile", default=None, help="Sentinel-2 MGRS tile, e.g. 31UEU. Use this if you "
              "already know the exact tile.")
@click.option("--bbox", default=None,
              help="lon_min,lat_min,lon_max,lat_max. A rough bounding box from any map tool "
                   "(e.g. bboxfinder.com, geojson.io, or Google Maps' right-click 'What's here?'). "
                   "Alternative to --tile.")
@click.option("--start-date", required=True, help="YYYY-MM-DD")
@click.option("--end-date", required=True, help="YYYY-MM-DD")
@click.option("--aoi-name", default=None, help="Human-readable label for this run")
@click.option("--to-step", default="step4", type=_STEP_CHOICES, show_default=True,
              help="step5 additionally cross-references confirmed detections against ADS-B — "
                   "needs OpenSky credentials for any (tile, date) not already cached.")
def run_cmd(tile, bbox, start_date, end_date, aoi_name, to_step):
    """
    Runs the pipeline end-to-end for an area of interest. 
    - Requires GEE credentials. 
    - Specify the area via --tile or --bbox 
    """
    if not tile and not bbox:
        raise click.UsageError("Specify either --tile or --bbox.")

    gee_auth.gee_auth_init()

    if bbox:
        lon_min, lat_min, lon_max, lat_max = (float(v) for v in bbox.split(","))
        aoi = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
    else:
        image = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filter(ee.Filter.eq("MGRS_TILE", tile))
            .first()
        )
        aoi = image.geometry()

    pipeline = Pipeline()
    run_id = pipeline.start_run(aoi, start_date, end_date, tile_name=tile, aoi_name=aoi_name)
    click.echo(f"Started run_id={run_id}")
    pipeline.run(to_step=to_step)
    click.echo(f"Run {run_id} complete (through {to_step}).")


@main.command("resume")
@click.option("--run-id", required=True, type=int)
@click.option("--from-step", default="step1", type=_STEP_CHOICES, show_default=True)
@click.option("--to-step", default="step4", type=_STEP_CHOICES, show_default=True,
              help="step5 additionally cross-references confirmed detections against ADS-B — "
                   "needs OpenSky credentials for any (tile, date) not already cached.")
@click.option("--mask-dir", default=None,
              help="Override mask directory, e.g. the shared fixture masks (config.MASK_DIR) "
                   "instead of this run's own — only used if --from-step is step1.")
def resume_cmd(run_id, from_step, to_step, mask_dir):
    """
    Resumes an existing run from a given step 
    """
    start_idx = STEP_ORDER.index(from_step)
    end_idx = STEP_ORDER.index(to_step)
    if _GEE_STEPS & set(STEP_ORDER[start_idx:end_idx + 1]):
        gee_auth.gee_auth_init()

    pipeline = Pipeline()
    step1_kwargs = {"mask_dir": mask_dir} if mask_dir else {}
    pipeline.run(run_id=run_id, from_step=from_step, to_step=to_step, **step1_kwargs)
    click.echo(f"Run {run_id} resumed: {from_step} -> {to_step} complete.")


@main.group("runs")
def runs_group():
    """
    Inspect pipeline runs
    """


@runs_group.command("list")
def runs_list():
    """
    Lists all pipeline runs, most recent first
    """
    df = repository.list_pipeline_runs()
    if df.empty:
        click.echo("No pipeline runs yet.")
        return
    click.echo(df.to_string(index=False))


@runs_group.command("show")
@click.argument("run_id", type=int)
def runs_show(run_id):
    """
    Shows full detail for one pipeline run
    """
    run = repository.get_pipeline_run(run_id)
    if run is None:
        click.echo(f"No run found with id={run_id}")
        raise SystemExit(1)
    for key, value in run.items():
        click.echo(f"{key:20s}: {value}")


@main.group("db")
def db_group():
    """
    Database maintenance
    """


@db_group.command("upgrade")
def db_upgrade():
    """
    Runs `alembic upgrade head` to bring the schema up to date
    """
    subprocess.run(["alembic", "upgrade", "head"], check=True)


if __name__ == "__main__":
    main()
