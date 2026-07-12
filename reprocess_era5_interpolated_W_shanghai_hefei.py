# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd


BASE_DIR = Path("E:/research/垂直交换")

G0 = 9.80665
R_D = 287.05
M_AIR = 28.9647
M_O3 = 47.9982
TARGET_HEIGHTS_M = np.arange(0, 4001, 100, dtype=float)


@dataclass(frozen=True)
class CityJob:
    city: str
    site_lat: float
    site_lon: float
    data_dir: Path
    file_glob: str
    out_dir: Path


JOBS = [
    CityJob(
        city="上海",
        site_lat=31.17,
        site_lon=121.42,
        data_dir=BASE_DIR
        / "上海_ERA5数据与处理结果汇总"
        / "上海_ERA5_2025_pressure_levels_600_1000",
        file_glob="era5_pressure_600_1000_shanghai_2025*.nc",
        out_dir=BASE_DIR
        / "上海_ERA5数据与处理结果汇总"
        / "ERA5_600_1000_高度插值与W向下为正",
    ),
    CityJob(
        city="合肥科学岛",
        site_lat=31.91,
        site_lon=117.16,
        data_dir=BASE_DIR
        / "合肥科学岛_ERA5数据与处理结果汇总"
        / "合肥科学岛_ERA5_2025_pressure_levels_600_1000",
        file_glob="era5_pressure_600_1000_hefei_science_island_2025*.nc",
        out_dir=BASE_DIR
        / "合肥科学岛_ERA5数据与处理结果汇总"
        / "合肥科学岛_ERA5_600_1000_高度插值与W向下为正",
    ),
]


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
    "W_downward_m_s",
    "w_geometric_m_s",
    "o3_flux_u_ug_m2_s",
    "o3_flux_v_ug_m2_s",
    "o3_flux_h_mag_ug_m2_s",
    "TFw_downward_positive_ug_m2_s",
    "TFw_upward_positive_ug_m2_s",
]


def open_nc(path: Path) -> netCDF4.Dataset:
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
    geopotential_height = z / G0
    tv = t * (1.0 + 0.61 * np.nan_to_num(q, nan=0.0))
    rho = pressure_pa / (R_D * tv)

    o3_mass = o3_mmr * rho * 1.0e9
    o3_ppbv = o3_mmr * (M_AIR / M_O3) * 1.0e9

    # ERA5 omega > 0 means downward in pressure coordinates.
    W_downward = omega / (rho * G0)
    w_geometric = -W_downward

    flux_u = o3_mass * u
    flux_v = o3_mass * v
    flux_h = np.sqrt(flux_u**2 + flux_v**2)
    tfw_down = o3_mass * W_downward
    tfw_up = o3_mass * w_geometric

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
            "geopotential_height_m": geopotential_height.ravel(),
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
            "W_downward_m_s": W_downward.ravel(),
            "w_geometric_m_s": w_geometric.ravel(),
            "o3_flux_u_ug_m2_s": flux_u.ravel(),
            "o3_flux_v_ug_m2_s": flux_v.ravel(),
            "o3_flux_h_mag_ug_m2_s": flux_h.ravel(),
            "TFw_downward_positive_ug_m2_s": tfw_down.ravel(),
            "TFw_upward_positive_ug_m2_s": tfw_up.ravel(),
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


def nearest_site(df: pd.DataFrame, site_lat: float, site_lon: float) -> tuple[float, float]:
    grid = df[["latitude", "longitude"]].drop_duplicates().copy()
    grid["distance2"] = (grid["latitude"] - site_lat) ** 2 + (grid["longitude"] - site_lon) ** 2
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
        row["height_agl_m"] = target_h
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


def interpolate_site_point(pressure_table: pd.DataFrame, job: CityJob) -> pd.DataFrame:
    site_lat, site_lon = nearest_site(pressure_table, job.site_lat, job.site_lon)
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
                    "target_latitude": job.site_lat,
                    "target_longitude": job.site_lon,
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


def finalize_height_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["W_downward_from_interp_omega_rho_m_s"] = (
        df["omega_Pa_s"] / (df["air_density_kg_m3"] * G0)
    )
    # Use the post-interpolation formula result as the canonical W column.
    df["W_downward_m_s"] = df["W_downward_from_interp_omega_rho_m_s"]
    df["w_geometric_m_s"] = -df["W_downward_m_s"]
    df["TFw_downward_positive_ug_m2_s"] = df["o3_mass_ug_m3"] * df["W_downward_m_s"]
    df["TFw_upward_positive_ug_m2_s"] = -df["TFw_downward_positive_ug_m2_s"]
    return df


def build_summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_cols = [
        "o3_mass_ug_m3",
        "omega_Pa_s",
        "W_downward_m_s",
        "w_geometric_m_s",
        "TFw_downward_positive_ug_m2_s",
        "TFw_upward_positive_ug_m2_s",
        "u_wind_m_s",
        "v_wind_m_s",
        "o3_flux_h_mag_ug_m2_s",
    ]
    daily = df.groupby(["bjt_date", "height_m"], as_index=False)[value_cols].mean()
    monthly = df.groupby(["month", "height_m"], as_index=False)[value_cols].mean()
    return daily, monthly


def write_city_outputs(job: CityJob) -> None:
    job.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(job.data_dir.glob(job.file_glob))
    if not files:
        raise FileNotFoundError(f"No ERA5 netCDF files found for {job.city}: {job.data_dir}")

    frames: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, object]] = []
    print(f"\n=== {job.city}: {len(files)} files ===")
    for file in files:
        print(f"Reading {file.name}")
        frame, info = read_file_to_long_table(file)
        frames.append(frame)
        inventory_rows.append(info)

    pressure_table = pd.concat(frames, ignore_index=True)
    inventory = pd.DataFrame(inventory_rows)
    area_height = finalize_height_table(interpolate_area_mean(pressure_table))
    site_height = finalize_height_table(interpolate_site_point(pressure_table, job))
    site_daily, site_monthly = build_summaries(site_height)
    area_daily, area_monthly = build_summaries(area_height)

    outputs = {
        "downloaded_inventory.csv": inventory,
        "height_interp_site_point_with_W.csv": site_height,
        "height_interp_area_mean_with_W.csv": area_height,
        "height_daily_site_point_W.csv": site_daily,
        "height_monthly_site_point_W.csv": site_monthly,
        "height_daily_area_mean_W.csv": area_daily,
        "height_monthly_area_mean_W.csv": area_monthly,
    }
    for name, df in outputs.items():
        out = job.out_dir / name
        print(f"Writing {out.name} ({len(df)} rows)")
        df.to_csv(out, index=False, encoding="utf-8-sig")

    notes = pd.DataFrame(
        [
            {
                "item": "W_downward_m_s",
                "description": "Post-interpolation vertical velocity W. It is calculated after height interpolation as W = omega/(rho*g), positive downward and negative upward.",
            },
            {
                "item": "W_downward_from_interp_omega_rho_m_s",
                "description": "Same as W_downward_m_s; kept to make the post-interpolation formula explicit.",
            },
            {
                "item": "w_geometric_m_s",
                "description": "Geometric vertical velocity with positive upward. It equals -W_downward_m_s.",
            },
            {
                "item": "TFw_downward_positive_ug_m2_s",
                "description": "O3 vertical transport flux using the report convention: TFw = O3 * W, positive downward.",
            },
            {
                "item": "height_m / height_agl_m",
                "description": "Target interpolation heights from 0 to 4000 m at 100 m intervals, kept consistent with previous project outputs.",
            },
            {
                "item": "site_point",
                "description": f"Nearest ERA5 grid to target lat={job.site_lat}, lon={job.site_lon}.",
            },
        ]
    )
    with pd.ExcelWriter(job.out_dir / "W_downward_processing_summary.xlsx", engine="openpyxl") as writer:
        notes.to_excel(writer, sheet_name="notes", index=False)
        inventory.to_excel(writer, sheet_name="downloaded_inventory", index=False)
        site_monthly.to_excel(writer, sheet_name="site_monthly_W", index=False)
        area_monthly.to_excel(writer, sheet_name="area_monthly_W", index=False)

    print(f"Done {job.city}: {job.out_dir}")


def main() -> None:
    for job in JOBS:
        write_city_outputs(job)


if __name__ == "__main__":
    main()
