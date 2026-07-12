from __future__ import annotations

import time
from pathlib import Path

from queue_era5_2025_hefei_science_island_600_1000_download import (
    cdsapirc_for,
    client_from_cdsapirc,
)


DATASET = "reanalysis-era5-single-levels"
TARGET = Path(
    "E:/research/\u5782\u76f4\u4ea4\u6362/\u5408\u80a5\u53cc\u7ad9_ERA5\u4e0eWRF_2023/"
    "ERA5_single_levels_surface/era5_surface_geopotential_hefei_hf_cf_20230401_00.nc"
)
REQUEST = {
    "product_type": ["reanalysis"],
    "variable": ["geopotential"],
    "year": ["2023"],
    "month": ["04"],
    "day": ["01"],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [32.50, 116.75, 31.50, 117.75],
}


def main() -> None:
    if TARGET.exists() and TARGET.stat().st_size > 0:
        print(f"Existing file: {TARGET}", flush=True)
        return
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    client = client_from_cdsapirc(cdsapirc_for("secondary"))
    remote = client.submit(DATASET, REQUEST)
    request_id = remote.request_id
    print(f"Submitted {request_id}: {remote.status}", flush=True)
    while True:
        remote = client.get_remote(request_id)
        print(f"Status {request_id}: {remote.status}", flush=True)
        if remote.status == "successful":
            client.download_results(request_id, str(TARGET))
            print(f"Downloaded: {TARGET}", flush=True)
            return
        if remote.status in {"failed", "rejected", "dismissed", "deleted"}:
            raise RuntimeError(f"Surface geopotential request ended as {remote.status}")
        time.sleep(30)


if __name__ == "__main__":
    main()
