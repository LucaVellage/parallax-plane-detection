"""
Seeds the chip/ADS-B/tile-scan-params caches from files shipped under
data/demo/ for the no-credentials demo notebook. 
"""

import re
import hashlib
import datetime as dt
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from pipeline.db import repository
from pipeline.db.base import get_engine

_CHIP_NAME_RE = re.compile(r"^(\d{8})_([A-Za-z0-9]+)_(-?\d+\.\d+)_(-?\d+\.\d+)_chip\.tif$")
_ADSB_NAME_RE = re.compile(r"^(\d{8})_([A-Za-z0-9]+)_adsb_raw\.parquet$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_chip_cache(chip_dir: str | Path = "../data/demo/step3_chip_cache") -> int:
    """
    Registers every {date}_{tile}_{col}_{row}_chip.tif file in chip_dir into the chip_cache table
    """
    n = 0
    for f in Path(chip_dir).glob("*_chip.tif"):
        match = _CHIP_NAME_RE.match(f.name)
        if match is None:
            print(f"skipping {f.name}: does not match {{date}}_{{tile}}_{{col}}_{{row}}_chip.tif")
            continue
        date, tile, col, row = match.groups()
        repository.insert_chip_cache_entry(
            tile, date, f"{date}_{tile}", float(col), float(row),
            str(f.resolve()), _sha256(f),
        )
        n += 1
    print(f"Seeded {n} chip cache entr{'y' if n == 1 else 'ies'} from {chip_dir}")
    return n


def seed_adsb_cache(adsb_dir: str | Path = "../data/demo/step5_adsb_cache") -> int:
    """
    Registers every {date}_{tile}_adsb_raw.parquet file in adsb_dir into the adsb_query_cache table
    """
    n = 0
    for f in Path(adsb_dir).glob("*_adsb_raw.parquet"):
        match = _ADSB_NAME_RE.match(f.name)
        if match is None:
            print(f"skipping {f.name}: does not match {{date}}_{{tile}}_adsb_raw.parquet")
            continue
        date, tile = match.groups()
        date_dashed = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        row_count = len(pd.read_parquet(f))
        repository.insert_adsb_cache_entry(
            tile, date_dashed, f"{date}_{tile}", str(f.resolve()), row_count,
        )
        n += 1
    print(f"Seeded {n} adsb_query_cache entr{'y' if n == 1 else 'ies'} from {adsb_dir}")



def seed_tile_scan_params(csv_path: str | Path = "../data/demo/tile_scan_params.csv") -> int:
    """
    Loads tile_scan_params rows from a CSV previously written by export_tile_scan_params()
    """
    df = pd.read_csv(csv_path, dtype=str)
    for _, row in df.iterrows():
        params = row.to_dict()
        for c in ["t_top", "t_bottom", "t_query_start", "t_query_end"]:
            params[c] = dt.datetime.fromisoformat(params[c])
        cloud_pct = params.get("cloud_pct")
        params["cloud_pct"] = float(cloud_pct) if cloud_pct not in (None, "", "nan") else None
        repository.insert_tile_scan_params(params)
    print(f"Seeded {len(df)} tile_scan_params entr{'y' if len(df) == 1 else 'ies'} from {csv_path}")



def export_tile_scan_params(
    scenes: list[tuple[str, str]],
    out_path: str | Path = "../data/demo/tile_scan_params.csv",
) -> int:
    """
    Helper for demo setup: Exports tile_scan_params rows for the given (tile, date) scenes to a CSV
    """
    rows = []
    with get_engine().begin() as conn:
        for tile, date in scenes:
            r = conn.execute(text("""
                SELECT tile, date, t_top, t_bottom, t_query_start, t_query_end,
                       west, east, south, north, cloud_pct
                FROM tile_scan_params WHERE tile = :tile AND date = :date
            """), {"tile": tile, "date": date}).mappings().first()
            if r is None:
                print(f"WARNING: no tile_scan_params cached for {tile}/{date} yet — skipping")
                continue
            row = dict(r)
            for c in ["t_top", "t_bottom", "t_query_start", "t_query_end", "date"]:
                row[c] = row[c].isoformat()
            rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Exported {len(rows)} row(s) to {out_path}")
    return len(rows)
