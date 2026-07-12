from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import netCDF4
import numpy as np


# Change this path if you want to inspect another ERA5 nc file.
NC_FILE = Path(
    r"E:\research\垂直交换\上海 24年十一月01-15"
    r"\50b3eed1d68342b5ec661f90f678975b.nc"
)


def open_nc_from_bytes(path: Path) -> netCDF4.Dataset:
    """Open NetCDF4 safely even when the Windows path contains Chinese text."""
    return netCDF4.Dataset("inmemory.nc", memory=path.read_bytes())


def print_coord(ds: netCDF4.Dataset, name: str) -> None:
    var = ds.variables[name]
    values = var[:]
    units = getattr(var, "units", "")
    print(f"{name}: units={units}, values={values.tolist()}")


def print_variable_stats(ds: netCDF4.Dataset, name: str) -> None:
    var = ds.variables[name]
    arr = np.ma.masked_invalid(var[:])
    long_name = getattr(var, "long_name", "")
    units = getattr(var, "units", "")
    print(
        f"{name}: {long_name}, units={units}, dims={var.dimensions}, "
        f"shape={arr.shape}, min={float(arr.min()):.6g}, "
        f"max={float(arr.max()):.6g}, mean={float(arr.mean()):.6g}"
    )


def main() -> None:
    if not NC_FILE.exists():
        raise FileNotFoundError(NC_FILE)

    ds = open_nc_from_bytes(NC_FILE)
    try:
        print(f"File: {NC_FILE}")
        print(f"Size: {NC_FILE.stat().st_size} bytes")

        print("\nDimensions")
        for name, dim in ds.dimensions.items():
            print(f"{name}: {len(dim)}")

        print("\nCoordinates")
        for name in ("valid_time", "pressure_level", "latitude", "longitude"):
            print_coord(ds, name)

        time_var = ds.variables["valid_time"]
        utc_times = netCDF4.num2date(
            time_var[:],
            time_var.units,
            getattr(time_var, "calendar", "standard"),
            only_use_cftime_datetimes=False,
        )
        bjt_times = [t + timedelta(hours=8) for t in utc_times]

        print("\nTime Range")
        print(f"UTC: {utc_times[0]} to {utc_times[-1]}, count={len(utc_times)}")
        print(f"BJT: {bjt_times[0]} to {bjt_times[-1]}, count={len(bjt_times)}")

        by_date: dict[str, list[int]] = defaultdict(list)
        for t in utc_times:
            by_date[t.date().isoformat()].append(t.hour)
        unique_utc_hours = sorted({hour for hours in by_date.values() for hour in hours})
        unique_bjt_hours = sorted({(hour + 8) % 24 for hour in unique_utc_hours})
        print(f"UTC hours in file: {unique_utc_hours}")
        print(f"Corresponding BJT hours: {unique_bjt_hours}")

        print("\nData Variables")
        for name in ds.variables:
            if name in {"number", "valid_time", "pressure_level", "latitude", "longitude", "expver"}:
                continue
            print_variable_stats(ds, name)

        print("\nVariable Name Reference")
        print("z = geopotential, q = specific humidity, t = temperature")
        print("u = u-component wind, v = v-component wind, w = vertical velocity")

        print("\nNote")
        print("ERA5 time is UTC. Beijing time = UTC + 8 hours.")
    finally:
        ds.close()


if __name__ == "__main__":
    main()
