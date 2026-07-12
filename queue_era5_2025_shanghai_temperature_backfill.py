# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

from ecmwf.datastores.client import Client


DATASET = "reanalysis-era5-pressure-levels"
BASE_DIR = Path("E:/research/????")
SHANGHAI_ROOT = BASE_DIR / "\u4e0a\u6d77_ERA5\u6570\u636e\u4e0e\u5904\u7406\u7ed3\u679c\u6c47\u603b"
OUT_DIR = SHANGHAI_ROOT / "\u4e0a\u6d77_ERA5_2025_temperature_backfill"
STATE_FILE = OUT_DIR / "era5_2025_shanghai_temperature_backfill_state.csv"
DEFAULT_CDSAPIRC = Path.home() / ".cdsapirc_era5_new"

TIMES = [f"{hour:02d}:00" for hour in range(0, 9)]
PRESSURE_LEVELS = ["1000", "900", "800", "700", "600", "500", "400", "300", "200", "100"]
AREA = [31.5, 121.0, 30.75, 121.75]
VARIABLES = ["temperature"]

# Existing files that were downloaded before temperature was added. Temperature
# only requests are small enough to use full months for the missing periods.
BACKFILL_PERIODS = [
    (1, 1, 15),
    (1, 16, 31),
    (2, 1, 28),
    (3, 1, 31),
    (4, 1, 30),
    (5, 1, 31),
    (8, 1, 15),
]

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


def read_cdsapirc(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"CDS API config not found: {path}\n"
            "Create this file with two lines: url: https://cds.climate.copernicus.eu/api and key: <your API token>."
        )
    cfg: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            cfg[key.strip()] = value.strip()
    if "url" not in cfg or "key" not in cfg:
        raise ValueError(f"{path} must contain both url and key entries.")
    return cfg


def client_from_cdsapirc(path: Path) -> Client:
    cfg = read_cdsapirc(path)
    return Client(url=cfg["url"], key=cfg["key"], sleep_max=60)


def period_label(month: int, start_day: int, end_day: int) -> str:
    return f"2025-{month:02d}-{start_day:02d}_to_2025-{month:02d}-{end_day:02d}"


def target_file(month: int, start_day: int, end_day: int) -> str:
    return f"era5_temperature_shanghai_2025{month:02d}_{start_day:02d}-{end_day:02d}.nc"


def init_state() -> list[dict[str, str]]:
    rows = []
    for month, start_day, end_day in BACKFILL_PERIODS:
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
                "note": "Temperature-only backfill for already downloaded ERA5 pressure-level files.",
            }
        )
    return rows


def load_state() -> list[dict[str, str]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.exists():
        rows = init_state()
        save_state(rows)
        return rows
    return list(csv.DictReader(STATE_FILE.open("r", encoding="utf-8-sig", newline="")))


def save_state(rows: list[dict[str, str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def make_request(row: dict[str, str]) -> dict[str, object]:
    start_day = int(row["start_day"])
    end_day = int(row["end_day"])
    days = [f"{day:02d}" for day in range(start_day, end_day + 1)]
    return {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": ["2025"],
        "month": [row["month"]],
        "day": days,
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
            row["note"] = f"Previous request {old_request_id} ended as {status}; reset for requeue."


def submit_pending(client: Client, rows: list[dict[str, str]], max_active: int) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    active = active_dataset_jobs(client)
    print(f"Active accepted/running {DATASET} jobs: {active}")

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
            row["note"] = "Submitted by temperature backfill script."
            active += 1
        except Exception as exc:
            row["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row["note"] = f"Submit failed: {exc}"
            print(f"Submit failed for {row['period']}: {exc}")
            break


def all_done(rows: list[dict[str, str]]) -> bool:
    return all(row["status"] == "downloaded" for row in rows)


def run_once(cdsapirc: Path, max_active: int) -> bool:
    client = client_from_cdsapirc(cdsapirc)
    rows = load_state()
    update_existing_requests(client, rows)
    submit_pending(client, rows, max_active=max_active)
    save_state(rows)
    return all_done(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cdsapirc", type=Path, default=DEFAULT_CDSAPIRC)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=180)
    parser.add_argument("--max-active", type=int, default=4)
    args = parser.parse_args()

    print(f"Using CDS API config: {args.cdsapirc}")
    while True:
        done = run_once(args.cdsapirc, max_active=args.max_active)
        if done:
            print("All temperature backfill periods are downloaded.")
            return
        if not args.watch:
            print(f"One pass complete. State file: {STATE_FILE}")
            return
        print(f"Sleeping {args.interval} seconds before next check.")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
