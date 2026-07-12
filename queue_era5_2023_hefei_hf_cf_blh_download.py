# -*- coding: utf-8 -*-
"""ERA5 boundary-layer-height queue for the 2023 Hefei HF/CF study."""
from __future__ import annotations

import calendar
from pathlib import Path

import queue_era5_2025_hefei_science_island_600_1000_download as queue


YEAR = 2023
DATASET = "reanalysis-era5-single-levels"
ROOT = Path("E:/research/\u5782\u76f4\u4ea4\u6362") / "\u5408\u80a5\u53cc\u7ad9_ERA5\u4e0eWRF_2023"
OUT_DIR = ROOT / "ERA5_single_levels_BLH"
# Beijing time 08:00-16:00 corresponds to UTC 00:00-08:00.
TIMES = [f"{hour:02d}:00" for hour in range(9)]
AREA = [32.50, 116.75, 31.50, 117.75]
MONTHS_BY_ACCOUNT = {
    "main": [4, 6, 8, 10, 1, 3, 11],
    "secondary": [5, 7, 9, 2, 12],
}


def active_account_jobs(client: object) -> int:
    jobs = client.get_jobs(
        limit=100,
        sortby="-created",
        status=["accepted", "running"],
    ).json["jobs"]
    return len(jobs)


def state_file_for(account: str) -> Path:
    return OUT_DIR / f"era5_{YEAR}_hefei_hf_cf_blh_{account}_state.csv"


def period_label(month: int, start_day: int, end_day: int) -> str:
    return f"{YEAR}-{month:02d}-{start_day:02d}_to_{YEAR}-{month:02d}-{end_day:02d}"


def target_file(month: int, start_day: int, end_day: int) -> str:
    return f"era5_blh_hefei_hf_cf_{YEAR}{month:02d}_{start_day:02d}-{end_day:02d}.nc"


def month_chunks(month: int) -> list[tuple[int, int]]:
    last_day = calendar.monthrange(YEAR, month)[1]
    return [(1, 10), (11, 20), (21, last_day)]


def init_state(account: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for month in MONTHS_BY_ACCOUNT[account]:
        for start_day, end_day in month_chunks(month):
            rows.append(
                {
                    "period": period_label(month, start_day, end_day),
                    "month": f"{month:02d}",
                    "start_day": f"{start_day:02d}",
                    "end_day": f"{end_day:02d}",
                    "status": "pending",
                    "request_id": "",
                    "target_file": target_file(month, start_day, end_day),
                    "updated_at": "",
                    "note": f"{account} account; Hefei HF/CF; {YEAR}; hourly BLH.",
                }
            )
    return rows


def make_request(row: dict[str, str]) -> dict[str, object]:
    start_day = int(row["start_day"])
    end_day = int(row["end_day"])
    return {
        "product_type": ["reanalysis"],
        "variable": ["boundary_layer_height"],
        "year": [str(YEAR)],
        "month": [row["month"]],
        "day": [f"{day:02d}" for day in range(start_day, end_day + 1)],
        "time": TIMES,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": AREA,
    }


def main() -> None:
    queue.DATASET = DATASET
    queue.OUT_DIR = OUT_DIR
    queue.TIMES = TIMES
    queue.PRESSURE_LEVELS = ["single level"]
    queue.AREA = AREA
    queue.state_file_for = state_file_for
    queue.init_state = init_state
    queue.make_request = make_request
    queue.active_dataset_jobs = active_account_jobs
    queue.main()


if __name__ == "__main__":
    main()
