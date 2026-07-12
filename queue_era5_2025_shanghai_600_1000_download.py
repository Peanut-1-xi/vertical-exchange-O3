# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import calendar
import csv
import time
from datetime import datetime
from pathlib import Path

from ecmwf.datastores.client import Client


DATASET = "reanalysis-era5-pressure-levels"
BASE_DIR = Path("E:/research/????")
SHANGHAI_ROOT = BASE_DIR / "\u4e0a\u6d77_ERA5\u6570\u636e\u4e0e\u5904\u7406\u7ed3\u679c\u6c47\u603b"
OUT_DIR = SHANGHAI_ROOT / "\u4e0a\u6d77_ERA5_2025_pressure_levels_600_1000"

TIMES = [f"{hour:02d}:00" for hour in range(0, 9)]
PRESSURE_LEVELS = [
    "1000",
    "975",
    "950",
    "925",
    "900",
    "875",
    "850",
    "825",
    "800",
    "775",
    "750",
    "700",
    "650",
    "600",
]
VARIABLES = [
    "geopotential",
    "ozone_mass_mixing_ratio",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]
AREA = [31.5, 121.0, 30.75, 121.75]

FIELDNAMES = [
    "period",
    "month",
    "start_day",
    "end_day",
    "status",
    "request_id",
    "target_file",
    "updated_at",
    "note",
]
TERMINAL_BAD_STATUSES = {
    "rejected",
    "failed",
    "dismissed",
    "deleted",
    "cancelled",
    "canceled",
    "expired",
}


def state_file_for(account: str) -> Path:
    return OUT_DIR / f"era5_2025_shanghai_600_1000_{account}_state.csv"


def cdsapirc_for(account: str) -> Path:
    if account == "secondary":
        return Path.home() / ".cdsapirc_era5_new"
    return Path.home() / ".cdsapirc"


def month_range_for(account: str) -> tuple[int, int]:
    # Main account continues from the old queue's latest progress, while the
    # secondary account backfills from January.
    if account == "secondary":
        return 1, 7
    return 8, 12


def client_from_cdsapirc(path: Path) -> Client:
    cfg: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            cfg[key.strip()] = value.strip()
    return Client(url=cfg["url"], key=cfg["key"], sleep_max=60)


def period_label(month: int, start_day: int, end_day: int) -> str:
    return f"2025-{month:02d}-{start_day:02d}_to_2025-{month:02d}-{end_day:02d}"


def target_file(month: int, start_day: int, end_day: int) -> str:
    return f"era5_pressure_600_1000_shanghai_2025{month:02d}_{start_day:02d}-{end_day:02d}.nc"


def month_chunks(month: int) -> list[tuple[int, int]]:
    last_day = calendar.monthrange(2025, month)[1]
    return [(1, 10), (11, 20), (21, last_day)]


def init_state(account: str) -> list[dict[str, str]]:
    start_month, end_month = month_range_for(account)
    rows: list[dict[str, str]] = []
    for month in range(start_month, end_month + 1):
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
                    "note": f"{account} account redownload for 600-1000 hPa pressure levels.",
                }
            )
    return rows


def load_state(account: str) -> list[dict[str, str]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state_file = state_file_for(account)
    if not state_file.exists():
        rows = init_state(account)
        save_state(account, rows)
        return rows
    return list(csv.DictReader(state_file.open("r", encoding="utf-8-sig", newline="")))


def save_state(account: str, rows: list[dict[str, str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with state_file_for(account).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def mark_existing_downloads(rows: list[dict[str, str]]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for row in rows:
        if row["status"] == "downloaded":
            continue
        target = OUT_DIR / row["target_file"]
        if target.exists() and target.stat().st_size > 0:
            row["status"] = "downloaded"
            row["updated_at"] = now
            row["note"] = f"Found existing file: {target}"


def make_request(row: dict[str, str]) -> dict[str, object]:
    start_day = int(row["start_day"])
    end_day = int(row["end_day"])
    return {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": ["2025"],
        "month": [row["month"]],
        "day": [f"{day:02d}" for day in range(start_day, end_day + 1)],
        "time": TIMES,
        "pressure_level": PRESSURE_LEVELS,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": AREA,
    }


def active_dataset_jobs(client: Client) -> int:
    jobs = client.get_jobs(limit=100, sortby="-created", status=["accepted", "running"]).json["jobs"]
    return sum(1 for job in jobs if job.get("processID") == DATASET)


def update_existing_requests(client: Client, rows: list[dict[str, str]]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for row in rows:
        request_id = row.get("request_id", "").strip()
        if not request_id or row["status"] == "downloaded":
            continue

        try:
            remote = client.get_remote(request_id)
            status = remote.status
        except Exception as exc:
            row["status"] = "pending"
            row["request_id"] = ""
            row["updated_at"] = now
            row["note"] = f"Remote lookup failed and was reset for requeue: {exc}"
            continue

        row["status"] = status
        row["updated_at"] = now

        if status == "successful":
            target = OUT_DIR / row["target_file"]
            if target.exists() and target.stat().st_size > 0:
                row["status"] = "downloaded"
                row["note"] = f"Already downloaded to {target}"
                continue
            print(f"Downloading {row['period']} -> {target.name}")
            client.download_results(request_id, str(target))
            row["status"] = "downloaded"
            row["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row["note"] = f"Downloaded to {target}"
        elif status in TERMINAL_BAD_STATUSES:
            old_request_id = row["request_id"]
            try:
                client.delete(old_request_id)
            except Exception:
                pass
            row["status"] = "pending"
            row["request_id"] = ""
            row["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row["note"] = f"Previous request {old_request_id} ended as {status}; reset for requeue."


def submit_pending(client: Client, rows: list[dict[str, str]], max_active: int) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    active = active_dataset_jobs(client)
    print(f"Active accepted/running {DATASET} jobs on this account: {active}")

    for row in rows:
        if active >= max_active:
            break
        if row["status"] != "pending":
            continue

        target = OUT_DIR / row["target_file"]
        if target.exists() and target.stat().st_size > 0:
            row["status"] = "downloaded"
            row["updated_at"] = now
            row["note"] = f"Found existing file: {target}"
            continue

        print(f"Submitting {row['period']}")
        try:
            remote = client.submit(DATASET, make_request(row))
            row["request_id"] = remote.request_id
            row["status"] = remote.status
            row["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row["note"] = "Submitted by 600-1000 hPa queue script."
            active += 1
            print(f"Submitted {row['period']}: {remote.request_id} ({remote.status})")
        except Exception as exc:
            row["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row["note"] = f"Submit failed: {exc}"
            print(f"Submit failed for {row['period']}: {exc}")
            break


def all_done(rows: list[dict[str, str]]) -> bool:
    return all(row["status"] == "downloaded" for row in rows)


def run_once(account: str, max_active: int) -> bool:
    client = client_from_cdsapirc(cdsapirc_for(account))
    rows = load_state(account)
    mark_existing_downloads(rows)
    update_existing_requests(client, rows)
    mark_existing_downloads(rows)
    submit_pending(client, rows, max_active=max_active)
    mark_existing_downloads(rows)
    save_state(account, rows)
    return all_done(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", choices=["main", "secondary"], required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=180)
    parser.add_argument("--max-active", type=int, default=4)
    args = parser.parse_args()

    print(f"Using account={args.account}, config={cdsapirc_for(args.account)}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Pressure levels: {', '.join(PRESSURE_LEVELS)}")

    while True:
        done = run_once(args.account, max_active=args.max_active)
        if done:
            print(f"All 600-1000 hPa periods are downloaded for account={args.account}.")
            return
        if not args.watch:
            print(f"One pass complete. State file: {state_file_for(args.account)}")
            return
        print(f"Sleeping {args.interval} seconds before next check.")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
