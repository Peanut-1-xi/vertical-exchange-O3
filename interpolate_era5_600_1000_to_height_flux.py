# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd


BASE_DIR = Path("E:/research/????")
SHANGHAI_ROOT = BASE_DIR / "\u4e0a\u6d77_ERA5\u6570\u636e\u4e0e\u5904\u7406\u7ed3\u679c\u6c47\u603b"
DATA_DIR = SHANGHAI_ROOT / "\u4e0a\u6d77_ERA5_2025_pressure_levels_600_1000"
OUT_DIR = SHANGHAI_ROOT / "ERA5_600_1000_\u9ad8\u5ea6\u63d2\u503c\u4e0e\u901a\u91cf"
FILE_GLOB = "era5_pressure_600_1000_shanghai_2025*.nc"

SITE_LAT = 31.17
SITE_LON = 121.42

G0 = 9.80665
R_D = 287.05
M_AIR = 28.9647
M_O3 = 47.9982

# 近地面 0-4 km，和后续廓线/通量图更容易对接。
TARGET_HEIGHTS_M = np.arange(0, 4001, 100, dtype=float)

DERIVED_COLS = [
    "pressure_hPa",
    "geopotential_height_m",
    "temperature_K",
    "temperature_C",
    "specific_humidity_kg_kg",
    "air_density_kg_m3",
    "o3_mmr_kg_kg",
    "o3_ppbv_dry_approx",
    "o3_mass_ug_m3",
    "u_wind_m_s",
    "v_wind_m_s",
    "omega_Pa_s",
    "w_geometric_m_s",
    "o3_flux_u_ug_m2_s",
    "o3_flux_v_ug_m2_s",
    "o3_flux_h_mag_ug_m2_s",
    "o3_flux_w_ug_m2_s",
]


def open_nc(path: Path) -> netCDF4.Dataset:
    # Avoid Windows netCDF4 issues with Chinese paths.
    return netCDF4.Dataset("inmemory.nc", memory=path.read_bytes())


def nc_times(ds: netCDF4.Dataset) -> list[pd.Timestamp]:
    time_name = "valid_time" if "valid_time" in ds.variables else "time"
    time_var = ds.variables[time_name]
    values = netCDF4.num2date(
        time_var[:],
        time_var.units,
        getattr(time_var, "calendar", "standard"),
        only_use_cftime_datetimes=False,
    )
    return [pd.Timestamp(value) for value in values]


def ozone_var_name(ds: netCDF4.Dataset) -> str:
    if "o3" in ds.variables:
        return "o3"
    for name, var in ds.variables.items():
        if "ozone" in getattr(var, "long_name", "").lower():
            return name
    raise KeyError("No ozone mass mixing ratio variable found.")


def normalize_longitudes(longitudes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = ((np.asarray(longitudes, dtype=float) + 180.0) % 360.0) - 180.0
    order = np.argsort(lon)
    return lon[order], order


def read_file_to_long_table(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    with open_nc(path) as ds:
        times_utc = nc_times(ds)
        pressure = np.asarray(ds.variables["pressure_level"][:], dtype=float)
        lat = np.asarray(ds.variables["latitude"][:], dtype=float)
        lon, lon_order = normalize_longitudes(np.asarray(ds.variables["longitude"][:], dtype=float))

        z = np.asarray(ds.variables["z"][:], dtype=float)[:, :, :, lon_order]
        t = np.asarray(ds.variables["t"][:], dtype=float)[:, :, :, lon_order]
        q = np.asarray(ds.variables["q"][:], dtype=float)[:, :, :, lon_order]
        u = np.asarray(ds.variables["u"][:], dtype=float)[:, :, :, lon_order]
        v = np.asarray(ds.variables["v"][:], dtype=float)[:, :, :, lon_order]
        omega = np.asarray(ds.variables["w"][:], dtype=float)[:, :, :, lon_order]
        o3_mmr = np.asarray(ds.variables[ozone_var_name(ds)][:], dtype=float)[:, :, :, lon_order]

    nt, nz, ny, nx = z.shape
    pressure_pa = pressure.reshape(1, nz, 1, 1) * 100.0
    height = z / G0
    tv = t * (1.0 + 0.61 * np.nan_to_num(q, nan=0.0))
    rho = pressure_pa / (R_D * tv)
    o3_mass = o3_mmr * rho * 1.0e9
    o3_ppbv = o3_mmr * (M_AIR / M_O3) * 1.0e9
    w_geo = -omega / (rho * G0)

    flux_u = o3_mass * u
    flux_v = o3_mass * v
    flux_h = np.sqrt(flux_u**2 + flux_v**2)
    flux_w = o3_mass * w_geo

    ti, zi, yi, xi = np.indices(z.shape)
    utc_flat = [times_utc[i] for i in ti.ravel()]
    bjt_flat = [time + timedelta(hours=8) for time in utc_flat]

    df = pd.DataFrame(
        {
            "source_file": path.name,
            "utc_time": utc_flat,
            "bjt_time": bjt_flat,
            "bjt_date": [time.date().isoformat() for time in bjt_flat],
            "bjt_hour": [time.hour for time in bjt_flat],
            "month": [f"{time.month:02d}" for time in bjt_flat],
            "pressure_hPa": pressure[zi.ravel()],
            "latitude": lat[yi.ravel()],
            "longitude": lon[xi.ravel()],
            "geopotential_height_m": height.ravel(),
            "temperature_K": t.ravel(),
            "temperature_C": (t - 273.15).ravel(),
            "specific_humidity_kg_kg": q.ravel(),
            "air_density_kg_m3": rho.ravel(),
            "o3_mmr_kg_kg": o3_mmr.ravel(),
            "o3_ppbv_dry_approx": o3_ppbv.ravel(),
            "o3_mass_ug_m3": o3_mass.ravel(),
            "u_wind_m_s": u.ravel(),
            "v_wind_m_s": v.ravel(),
            "omega_Pa_s": omega.ravel(),
            "w_geometric_m_s": w_geo.ravel(),
            "o3_flux_u_ug_m2_s": flux_u.ravel(),
            "o3_flux_v_ug_m2_s": flux_v.ravel(),
            "o3_flux_h_mag_ug_m2_s": flux_h.ravel(),
            "o3_flux_w_ug_m2_s": flux_w.ravel(),
        }
    )

    info = {
        "file": path.name,
        "rows_pressure_grid": len(df),
        "time_utc_min": min(times_utc),
        "time_utc_max": max(times_utc),
        "time_bjt_min": min(time + timedelta(hours=8) for time in times_utc),
        "time_bjt_max": max(time + timedelta(hours=8) for time in times_utc),
        "latitudes": ", ".join(map(str, sorted(lat))),
        "longitudes": ", ".join(map(str, sorted(lon))),
        "pressure_levels_hPa": ", ".join(map(lambda x: str(int(x)), pressure)),
    }
    return df, info


def nearest_site(df: pd.DataFrame) -> tuple[float, float]:
    grid = df[["latitude", "longitude"]].drop_duplicates().copy()
    grid["distance2"] = (grid["latitude"] - SITE_LAT) ** 2 + (grid["longitude"] - SITE_LON) ** 2
    row = grid.sort_values("distance2").iloc[0]
    return float(row["latitude"]), float(row["longitude"])


def linear_interp_profile(group: pd.DataFrame, label: dict[str, object]) -> list[dict[str, object]]:
    group = group.sort_values("geopotential_height_m")
    heights = group["geopotential_height_m"].to_numpy(dtype=float)
    ok_h = np.isfinite(heights)
    if ok_h.sum() < 2:
        return []

    group = group.loc[ok_h].copy()
    heights = heights[ok_h]
    unique_h, unique_idx = np.unique(heights, return_index=True)
    if unique_h.size < 2:
        return []
    group = group.iloc[unique_idx]
    heights = unique_h

    rows: list[dict[str, object]] = []
    for target_h in TARGET_HEIGHTS_M:
        row = dict(label)
        row["height_m"] = target_h
        for col in DERIVED_COLS:
            values = group[col].to_numpy(dtype=float)
            ok = np.isfinite(values)
            if ok.sum() < 2:
                row[col] = np.nan
            else:
                row[col] = np.interp(target_h, heights[ok], values[ok], left=np.nan, right=np.nan)
        rows.append(row)
    return rows


def interpolate_area_mean(pressure_table: pd.DataFrame) -> pd.DataFrame:
    area_pressure = pressure_table.groupby(
        ["bjt_time", "bjt_date", "bjt_hour", "month", "pressure_hPa"], as_index=False
    )[DERIVED_COLS[1:]].mean()

    rows: list[dict[str, object]] = []
    for keys, group in area_pressure.groupby(["bjt_time", "bjt_date", "bjt_hour", "month"], sort=True):
        bjt_time, bjt_date, bjt_hour, month = keys
        rows.extend(
            linear_interp_profile(
                group,
                {
                    "profile_type": "area_mean",
                    "bjt_time": bjt_time,
                    "bjt_date": bjt_date,
                    "bjt_hour": bjt_hour,
                    "month": month,
                },
            )
        )
    return pd.DataFrame(rows)


def interpolate_site_point(pressure_table: pd.DataFrame) -> pd.DataFrame:
    site_lat, site_lon = nearest_site(pressure_table)
    site = pressure_table[
        (pressure_table["latitude"] == site_lat) & (pressure_table["longitude"] == site_lon)
    ].copy()

    rows: list[dict[str, object]] = []
    for keys, group in site.groupby(["bjt_time", "bjt_date", "bjt_hour", "month"], sort=True):
        bjt_time, bjt_date, bjt_hour, month = keys
        rows.extend(
            linear_interp_profile(
                group,
                {
                    "profile_type": "nearest_site_point",
                    "target_latitude": SITE_LAT,
                    "target_longitude": SITE_LON,
                    "nearest_latitude": site_lat,
                    "nearest_longitude": site_lon,
                    "bjt_time": bjt_time,
                    "bjt_date": bjt_date,
                    "bjt_hour": bjt_hour,
                    "month": month,
                },
            )
        )
    return pd.DataFrame(rows)


def add_flux_from_interpolated_products(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["o3_flux_u_from_interp_product_ug_m2_s"] = df["o3_mass_ug_m3"] * df["u_wind_m_s"]
    df["o3_flux_v_from_interp_product_ug_m2_s"] = df["o3_mass_ug_m3"] * df["v_wind_m_s"]
    df["o3_flux_h_from_interp_product_ug_m2_s"] = np.sqrt(
        df["o3_flux_u_from_interp_product_ug_m2_s"] ** 2
        + df["o3_flux_v_from_interp_product_ug_m2_s"] ** 2
    )
    df["o3_flux_w_from_interp_product_ug_m2_s"] = df["o3_mass_ug_m3"] * df["w_geometric_m_s"]
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_DIR.glob(FILE_GLOB))
    if not files:
        raise FileNotFoundError(f"No downloaded 600-1000 hPa ERA5 files found in {DATA_DIR}")

    frames: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, object]] = []
    for file in files:
        print(f"Reading {file.name}")
        frame, info = read_file_to_long_table(file)
        frames.append(frame)
        inventory_rows.append(info)

    pressure_table = pd.concat(frames, ignore_index=True)
    inventory = pd.DataFrame(inventory_rows)

    area_height = add_flux_from_interpolated_products(interpolate_area_mean(pressure_table))
    site_height = add_flux_from_interpolated_products(interpolate_site_point(pressure_table))

    # Compact summaries for plotting.
    area_daily_height = area_height.groupby(["bjt_date", "height_m"], as_index=False)[
        [
            "o3_mass_ug_m3",
            "u_wind_m_s",
            "v_wind_m_s",
            "w_geometric_m_s",
            "o3_flux_w_ug_m2_s",
            "o3_flux_w_from_interp_product_ug_m2_s",
        ]
    ].mean()
    area_month_height = area_height.groupby(["month", "height_m"], as_index=False)[
        [
            "o3_mass_ug_m3",
            "u_wind_m_s",
            "v_wind_m_s",
            "w_geometric_m_s",
            "o3_flux_w_ug_m2_s",
            "o3_flux_w_from_interp_product_ug_m2_s",
        ]
    ].mean()

    outputs = {
        "downloaded_inventory.csv": inventory,
        "pressure_level_full_grid.csv": pressure_table,
        "height_interp_area_mean.csv": area_height,
        "height_interp_site_point.csv": site_height,
        "height_daily_area_mean.csv": area_daily_height,
        "height_monthly_area_mean.csv": area_month_height,
    }
    for name, df in outputs.items():
        out = OUT_DIR / name
        print(f"Writing {out}")
        df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"Done. Output directory: {OUT_DIR}")
    print(f"Files processed: {len(files)}")
    print(f"Pressure rows: {len(pressure_table)}")
    print(f"Area height rows: {len(area_height)}")
    print(f"Site height rows: {len(site_height)}")


if __name__ == "__main__":
    main()
