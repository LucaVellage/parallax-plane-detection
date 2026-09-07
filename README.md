# parallax-plane-detection

A pipeline for detecting flying aircraft in Sentinel-2 multispectral imagery using inter-band parallax effects, reimplementing the method of [Liu et al. (2020)](https://www.sciencedirect.com/science/article/abs/pii/S0034425720302376?via%3Dihub) with extensions.

<p align="center">
  <img src="https://rawcdn.githack.com/LucaVellage/parallax-plane-detection/58ed182b7304a27cdd505b77c5046e6632030d2c/assets/LVellage_ParallaxPlaneDetection_ExampleDetections.png" width="100%"><br>
  <sub><em>Parallax Plane Detection: Example Detections. © 2026 Luca Vellage</em><sub>
</p>


Developed as part of my Master's thesis in the Master of Data Science for Public Policy at the Hertie School (Berlin).

📃 [View the MSc thesis poster here](https://rawcdn.githack.com/LucaVellage/parallax-plane-detection/28db9a370f966e3bd2f5b76c053c911d266c3fb2/assets/LVellage_ParallaxPlaneDetection_MScPoster_2026.pdf) 

⚙️ [View the pipeline overview here](https://rawcdn.githack.com/LucaVellage/parallax-plane-detection/28db9a370f966e3bd2f5b76c053c911d266c3fb2/assets/LVellage_ParallaxPlaneDetection_PipelineOverview.png) 

## Overview

The pipeline proceeds in five stages. Steps 1-4 are the core detection pipeline; Step 5 is an optional ADS-B cross-reference layer.

```
Step 1: GEE        : Band-3 reflectance anomaly detection → binary mask per image
Step 2: Local      : Sequential false positive filtering → candidate centroids
Step 3: GEE        : Image chip download around surviving candidates
Step 4: Local      : Aircraft confirmation via inter-band displacement
Step 5: Local      : (optional) ADS-B cross-reference of confirmed detections
```

Every run is identified by a `run_id`, stored in a Postgres/PostGIS database alongside every stage's output. Re-running the pipeline for a different area or date range does not mix results and any run can be inspected, resumed, or extended later using its `run_id`.

A full end-to-end demonstration of all pipeline steps is provided in `notebooks/01_pipeline_run.ipynb`. A second notebook, `notebooks/02_demo.ipynb`, runs the same pipeline end-to-end with no GEE account needed, using pre-supplied demo data (see "Demo Mode" below).

## Repository Structure

```
parallax-plane-detection-pipeline/
├── src/
│   └── pipeline/
│       ├── cli.py                      ← `parallax` CLI entry point
│       ├── pipeline.py                 ← Pipeline orchestration class
│       ├── settings.py                 ← runtime configuration
│       ├── config.py                   ← algorithm parameters
│       ├── db/
│       │   ├── models.py               ← SQLAlchemy/PostGIS schema
│       │   ├── base.py                 ← engine/session setup
│       │   └── repository.py           ← database reads/writes
│       ├── utils/
│       │   ├── gee_auth.py
│       │   ├── io.py
│       │   └── demo_seed.py            ← seeds caches for the no-credentials demo
│       ├── s1_gee_candidates/          ← Step 1: candidate screening
│       ├── s2_filter/                  ← Step 2: false positive filtering
│       │   ├── transform.py
│       │   ├── cluster_filter.py
│       │   ├── road_filter.py
│       │   └── seamline_filter.py
│       ├── s3_download/                ← Step 3: chip download
│       ├── s4_confirmation/            ← Step 4: aircraft confirmation
│       └── s5_evaluation/              ← Step 5: ADS-B evaluation (optional)
├── alembic/                            ← database migrations
├── tests/                              ← unit tests
├── notebooks/
│   ├── 01_pipeline_run.ipynb           ← full end-to-end demonstration
│   └── 02_demo.ipynb                   ← no-credentials demo 
├── docker-compose.yml                  ← Postgres/PostGIS service
├── .env.example                        ← copy to `.env` and fill in
├── alembic.ini
├── LICENSE
└── pyproject.toml
```


## Getting Started

### 1. Configure environment

```bash
cp .env.example .env
```
Open `.env` and set a `POSTGRES_PASSWORD` (required), your `GEE_PROJECT` id and `OPENSKY_USERNAME` (see Credentials below)

### 2. Start the database

Requires [Docker](https://docs.docker.com/get-docker/) installed and running.
```bash
docker compose up -d
```
Starts a Postgres/PostGIS container, which stores all of the pipeline's tabular output (candidates, chip metadata, confirmed detections, evaluation results). Rasters and image chips are stored on disk. Confirm it's healthy with `docker compose ps` before continuing which should show: `Up (healthy)`.

### 3. Install the package

Requires Python 3.11+. Recommended inside a virtual environment (`venv`, `conda`, etc.).
```bash
pip install -e ".[notebook]"
```
Omit `[notebook]` if you only need the CLI.

### 4. Apply the database schema

```bash
parallax db upgrade
```

### 5. Authenticate with Google Earth Engine

Needed for step 1 and 3:
```bash
earthengine authenticate
```

Setup completed, continue with running the pipeline.


## Running the Pipeline

### Via CLI

```bash
# core detection only (Steps 1-4) (no OpenSky account needed)
parallax run --tile 31UEU --start-date 2019-10-01 --end-date 2019-10-31

# specify an area by bounding box instead of a known Sentinel-2 tile
parallax run --bbox lon_min,lat_min,lon_max,lat_max --start-date 2019-10-01 --end-date 2019-10-31

# additionally cross-reference confirmed detections against ADS-B (Step 5)
parallax run --tile 31UEU --start-date 2019-10-01 --end-date 2019-10-31 --to-step step5
```

Other commands:
```bash
parallax resume --run-id ID --from-step stepN --to-step stepM # rerun part of an existing run
parallax runs list # list all runs
parallax runs show ID # full detail for one run
parallax db upgrade # apply migrations
```
Run `parallax --help` or `parallax run --help` for documentation.

### Via notebook

`notebooks/01_pipeline_run.ipynb` runs the same stages one at a time, with an inspection/plot after each step for debugging/inspection.

`notebooks/02_demo.ipynb` runs the same walkthrough with no GEE account needed.

## Demo Mode

`notebooks/02_demo.ipynb` runs the full pipeline (Steps 1-4) end-to-end with no GEE account needed. Step 1 uses pre-supplied masks, step 3 uses pre-supplied chip/ADS-B data. Step 2's filter logic, step 4's confirmation logic and step 3's matching logic run unmodified. Step 5 (ADS-B cross-reference) is not included in the demo run!

Demo data lives under `data/demo/`:
```
data/demo/
├── step1_masks/            ← pre-downloaded step 1 masks
├── step3_chip_cache/       ← pre-downloaded chips
```

## Credentials and Authentication

### Google Earth Engine

Steps 1 and 3 require a Google Earth Engine account with a registered cloud project. Accounts can be requested at [earthengine.google.com](https://earthengine.google.com). Once approved, create a cloud project at [console.cloud.google.com](https://console.cloud.google.com).

Store the project ID in `.env`:
```
GEE_PROJECT=your-gee-project-id
```

### OpenSky Network (ADS-B)

Step 5 (`--to-step step5`) requires an OpenSky Network research account with Trino API access. Accounts can be requested at [opensky-network.org](https://opensky-network.org). Trino access  must then be enabled by following [OpenSky's Trino setup guide](https://openskynetwork.github.io/opensky-api/trino.html). Once set up, store your username in `.env`:
```
OPENSKY_USERNAME=your-username
```

## Configuration

Algorithm parameters (thresholds, buffer sizes, kernel sizes, etc.) are defined in `src/pipeline/config.py`. Credentials, database connection, and data directory are handled separately by `src/pipeline/settings.py`, sourced from `.env`.

## Database Access

Pipeline runs are stored in a Postgres/PostGIS database, started via `docker compose up -d`. To inspect it or run a one-off query, open a `psql` shell inside the container:

```bash
docker compose exec db psql -U parallax -d parallax
```

This runs `psql` inside the running `db` container. Change `parallax`/`parallax` for your own `POSTGRES_USER`/`POSTGRES_DB` if you changed them from the defaults in `.env`.

## Database Maintenance

Common operations, run from the `psql` shell above:

### Cascade delete logic

Deleting a `pipeline_run` row cascades through every run-scoped table: `mask`, `candidate` (and everything keyed off `candidate.id`: `chip`, `reflectance_centroid`, `collinearity_triplet`, `confirmed_detection`), `evaluation_point`, `metrics_by_background`, `metrics_by_scene`:
```sql
DELETE FROM pipeline_run WHERE id = <run_id>;
-- to clear everything above a given run:
DELETE FROM pipeline_run WHERE id > <run_id>;
```

### Clearing global caches

`chip_cache`, `adsb_query_cache`, and `tile_scan_params` are not run-scoped and therefore not affected by the above cascade delete logic. Clear them independently if needed:
```sql
DELETE FROM chip_cache;
DELETE FROM adsb_query_cache;
DELETE FROM tile_scan_params;
```
To delete on-disk cached files:
```bash
rm -rf data/step3_chip_cache/*
rm -rf data/step5_adsb_cache/*
```


## Testing

Unit tests cover the database-independent functions (geometry, filtering thresholds, displacement checks). Install the `test` extra and run:
```bash
pip install -e ".[test]"
pytest
```
No database or credentials are required.

## Data Sources

| Data | Source | Access |
|---|---|---|
| Sentinel-2 imagery | ESA Copernicus via Google Earth Engine | GEE account required |
| ADS-B state vectors | OpenSky Network Trino API | Research account required |


## Sources
Liu, Y., Xu, B., Zhi, W., Hu, C., Dong, Y., Jin, S., Lu, Y., Chen, T., Xu, W., Liu, Y., Zhao, B., & Lu, W. (2020). Space eye on flying aircraft: From Sentinel-2 MSI parallax to hybrid computing. Remote Sensing of Environment, 246, 111867. https://doi.org/10.1016/j.rse.2020.111867

Schafer, M., Strohmeier, M., Lenders, V., Martinovic, I., & Wilhelm, M. (2014). Bringing up OpenSky: A large-scale ADS-B sensor network for research. IPSN-14 Proceedings of the 13th International Symposium on Information Processing in Sensor Networks, 83–94. https://doi.org/10.1109/IPSN.2014.6846743
