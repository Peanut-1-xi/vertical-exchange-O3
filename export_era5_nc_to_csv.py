from __future__ import annotations

import csv
from datetime import timedelta
from pathlib import Path

import netCDF4


NC_FILE = Path(
    r"E:\research\垂直交换\上海 24年十一月01-15"
    r"\50b3eed1d68342b5ec661f90f678975b.nc"
)

OUT_DIR = Path(r"E:\research\垂直交换\O3 深度学习\codex1")
FULL_GRID_CSV = OUT_DIR / "era5_shanghai_20241101_1115_full_grid.csv"
NEAREST_POINT_CSV = OUT_DIR / "era5_shanghai_20241101_1115_nearest_point.csv"

# Approximate location of Shanghai Academy of Environmental Sciences.
TARGET_LAT = 31.17
TARGET_LON = 121.42
G0 = 9.80665


def open_nc_from_bytes(path: Path) -> netCDF4.Dataset:
    """Open NetCDF4 safely even when the Windows path contains Chinese text."""
    return netCDF4.Dataset("inmemory.nc", memory=path.read_bytes())


def nearest_index(values, target: float) -> int:
    return min(range(len(values)), key=lambda i: abs(float(values[i]) - target))


def export_rows(ds: netCDF4.Dataset, output_file: Path, lat_indices, lon_indices) -> int:
    time_var = ds.variables["valid_time"]
    utc_times = netCDF4.num2date(
        time_var[:],
        time_var.units,
        getattr(time_var, "calendar", "standard"),
        only_use_cftime_datetimes=False,
    )
    pressure_levels = ds.variables["pressure_level"][:]
    latitudes = ds.variables["latitude"][:]
    longitudes = ds.variables["longitude"][:]

    z = ds.variables["z"]
    q = ds.variables["q"]
    t = ds.variables["t"]
    u = ds.variables["u"]
    v = ds.variables["v"]
    w = ds.variables["w"]

    rows = 0
    with output_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "utc_time",
                "bjt_time",
                "pressure_level_hPa",
                "latitude",
                "longitude",
                "z_geopotential_m2_s2",
                "geopotential_height_m",
                "q_specific_humidity_kg_kg",
                "t_temperature_K",
                "t_temperature_C",
                "u_wind_m_s",
                "v_wind_m_s",
                "w_vertical_velocity_Pa_s",
            ]
        )

        for ti, utc_time in enumerate(utc_times):
            bjt_time = utc_time + timedelta(hours=8)
            for pi, pressure in enumerate(pressure_levels):
                for yi in lat_indices:
                    for xi in lon_indices:
                        z_value = float(z[ti, pi, yi, xi])
                        t_value = float(t[ti, pi, yi, xi])
                        writer.writerow(
                            [
                                utc_time.strftime("%Y-%m-%d %H:%M:%S"),
                                bjt_time.strftime("%Y-%m-%d %H:%M:%S"),
                                float(pressure),
                                float(latitudes[yi]),
                                float(longitudes[xi]),
                                z_value,
                                z_value / G0,
                                float(q[ti, pi, yi, xi]),
                                t_value,
                                t_value - 273.15,
                                float(u[ti, pi, yi, xi]),
                                float(v[ti, pi, yi, xi]),
                                float(w[ti, pi, yi, xi]),
                            ]
                        )
                        rows += 1
    return rows


def main() -> None:
    if not NC_FILE.exists():
        raise FileNotFoundError(NC_FILE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = open_nc_from_bytes(NC_FILE)
    try:
        latitudes = ds.variables["latitude"][:]
        longitudes = ds.variables["longitude"][:]
        nearest_lat_i = nearest_index(latitudes, TARGET_LAT)
        nearest_lon_i = nearest_index(longitudes, TARGET_LON)

        full_rows = export_rows(
            ds,
            FULL_GRID_CSV,
            range(len(latitudes)),
            range(len(longitudes)),
        )
        point_rows = export_rows(
            ds,
            NEAREST_POINT_CSV,
            [nearest_lat_i],
            [nearest_lon_i],
        )

        print(f"Full grid CSV: {FULL_GRID_CSV}")
        print(f"Full grid rows: {full_rows}")
        print(f"Nearest point CSV: {NEAREST_POINT_CSV}")
        print(f"Nearest point rows: {point_rows}")
        print(
            "Nearest point used: "
            f"lat={float(latitudes[nearest_lat_i])}, "
            f"lon={float(longitudes[nearest_lon_i])}"
        )
    finally:
        ds.close()


if __name__ == "__main__":
    main()
