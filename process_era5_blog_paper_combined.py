# -*- coding: utf-8 -*-
from __future__ import annotations

import calendar
from datetime import timedelta
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd


BASE_DIR = Path("E:/research/????")
SHANGHAI_ROOT = BASE_DIR / "\u4e0a\u6d77_ERA5\u6570\u636e\u4e0e\u5904\u7406\u7ed3\u679c\u6c47\u603b"
DATA_DIR = SHANGHAI_ROOT / "\u4e0a\u6d77_ERA5_2025_pressure_levels"
OUT_DIR = SHANGHAI_ROOT / "ERA5_\u8bba\u6587\u65b9\u6cd5\u5904\u7406\u7ed3\u679c"
OUT_XLSX = OUT_DIR / "上海_ERA5_已有数据_CSDN处理结合论文方法.xlsx"
OUT_CSV_DIR = OUT_DIR / "上海_ERA5_已有数据_CSDN处理结合论文方法_csv"

SITE_NAME = "上海环境科学研究院附近最近ERA5格点"
SITE_LAT = 31.17
SITE_LON = 121.42
TARGET_HEIGHTS_M = np.arange(0, 5001, 200, dtype=float)

G0 = 9.80665
R_D = 287.05
M_AIR = 28.9647
M_O3 = 47.9982

MEAN_COLS = [
    "geopotential_height_m",
    "temperature_C",
    "specific_humidity_kg_kg",
    "air_density_kg_m3",
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


def open_nc_from_bytes(path: Path) -> netCDF4.Dataset:
    # Reading from memory avoids Windows/netCDF4 issues with Chinese paths.
    return netCDF4.Dataset("inmemory.nc", memory=path.read_bytes())


def canonical_dim(name: str) -> str:
    lowered = name.lower()
    if lowered in {"valid_time", "time"}:
        return "time"
    if lowered in {"pressure_level", "level", "isobaricinhpa"} or "pressure" in lowered:
        return "pressure"
    if lowered in {"latitude", "lat"}:
        return "latitude"
    if lowered in {"longitude", "lon"}:
        return "longitude"
    if lowered == "expver":
        return "expver"
    return lowered


def coord_values(ds: netCDF4.Dataset, names: tuple[str, ...]) -> tuple[str, np.ndarray]:
    for name in names:
        if name in ds.variables:
            return name, np.asarray(ds.variables[name][:])
    raise KeyError(f"None of these coordinates exist: {names}")


def combine_expver(data: np.ndarray, dims: list[str], ds: netCDF4.Dataset) -> tuple[np.ndarray, list[str], bool]:
    if "expver" not in [canonical_dim(dim) for dim in dims]:
        return data, dims, False

    axis = [canonical_dim(dim) for dim in dims].index("expver")
    expver_values = np.arange(data.shape[axis])
    if "expver" in ds.variables:
        expver_values = np.asarray(ds.variables["expver"][:])

    moved = np.moveaxis(data, axis, 0)

    def priority(idx: int) -> tuple[int, int]:
        value = str(expver_values[idx])
        if value in {"1", "1.0"}:
            return (0, idx)
        if value in {"5", "5.0"}:
            return (1, idx)
        return (2, idx)

    order = sorted(range(moved.shape[0]), key=priority)
    combined = np.asarray(moved[order[0]], dtype=float)
    for idx in order[1:]:
        candidate = np.asarray(moved[idx], dtype=float)
        combined = np.where(np.isfinite(combined), combined, candidate)

    new_dims = dims[:axis] + dims[axis + 1 :]
    return combined, new_dims, True


def read_4d(
    ds: netCDF4.Dataset,
    var_name: str,
    shape: tuple[int, int, int, int],
) -> tuple[np.ndarray, bool]:
    var = ds.variables[var_name]
    data = np.asarray(var[:], dtype=float)
    dims = list(var.dimensions)
    data, dims, used_expver = combine_expver(data, dims, ds)
    canonical = [canonical_dim(dim) for dim in dims]

    target_dims = ["time", "pressure", "latitude", "longitude"]
    present_axes = [canonical.index(dim) for dim in target_dims if dim in canonical]
    data = np.transpose(data, present_axes) if present_axes else data

    present_dims = [dim for dim in target_dims if dim in canonical]
    for axis, dim in enumerate(target_dims):
        if dim not in present_dims:
            data = np.expand_dims(data, axis=axis)
            present_dims.insert(axis, dim)

    data = np.broadcast_to(data, shape).astype(float, copy=False)
    return np.array(data, dtype=float, copy=True), used_expver


def find_ozone_variable(ds: netCDF4.Dataset) -> str:
    for name in ("o3", "o3_mmr"):
        if name in ds.variables:
            return name
    for name, var in ds.variables.items():
        long_name = getattr(var, "long_name", "").lower()
        standard_name = getattr(var, "standard_name", "").lower()
        if "ozone" in long_name or "ozone" in standard_name:
            return name
    raise KeyError("No ozone mass mixing ratio variable found.")


def nc_times(ds: netCDF4.Dataset) -> list[pd.Timestamp]:
    time_name, _ = coord_values(ds, ("valid_time", "time"))
    time_var = ds.variables[time_name]
    values = netCDF4.num2date(
        time_var[:],
        time_var.units,
        getattr(time_var, "calendar", "standard"),
        only_use_cftime_datetimes=False,
    )
    return [pd.Timestamp(value) for value in values]


def density_from_hydrostatic(pressure_levels_hpa: np.ndarray, heights_m: np.ndarray) -> np.ndarray:
    pressure_pa = pressure_levels_hpa * 100.0
    rho = np.full_like(heights_m, np.nan, dtype=float)
    nt, _, nlat, nlon = heights_m.shape

    for ti in range(nt):
        for yi in range(nlat):
            for xi in range(nlon):
                height_profile = np.asarray(heights_m[ti, :, yi, xi], dtype=float)
                ok = np.isfinite(height_profile)
                if ok.sum() < 2:
                    continue
                order = np.argsort(height_profile[ok])
                h_sorted = height_profile[ok][order]
                p_sorted = pressure_pa[ok][order]
                if np.unique(h_sorted).size < 2:
                    continue
                dp_dz = np.gradient(p_sorted, h_sorted)
                rho_sorted = -dp_dz / G0
                profile = np.full_like(height_profile, np.nan, dtype=float)
                ok_idx = np.where(ok)[0][order]
                profile[ok_idx] = rho_sorted
                rho[ti, :, yi, xi] = profile

    return np.where(rho > 0, rho, np.nan)


def longitudes_to_minus180_180(longitudes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized = ((np.asarray(longitudes, dtype=float) + 180.0) % 360.0) - 180.0
    order = np.argsort(normalized)
    return normalized[order], order


def file_period_from_name(name: str) -> str:
    stem = Path(name).stem
    parts = stem.split("_")
    ym = parts[-2]
    days = parts[-1]
    start_day, end_day = days.split("-")
    return f"{ym[:4]}-{ym[4:]}-{start_day}_to_{ym[:4]}-{ym[4:]}-{end_day}"


def read_one_file(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    ds = open_nc_from_bytes(path)
    try:
        times_utc = nc_times(ds)
        _, pressure_levels = coord_values(ds, ("pressure_level", "level", "isobaricInhPa"))
        _, latitudes = coord_values(ds, ("latitude", "lat"))
        _, longitudes_raw = coord_values(ds, ("longitude", "lon"))

        pressure_levels = np.asarray(pressure_levels, dtype=float)
        latitudes = np.asarray(latitudes, dtype=float)
        longitudes, lon_order = longitudes_to_minus180_180(longitudes_raw)
        shape = (len(times_utc), len(pressure_levels), len(latitudes), len(longitudes))

        expver_vars: set[str] = set()

        def read(name: str) -> np.ndarray:
            arr, used_expver = read_4d(ds, name, shape)
            if used_expver:
                expver_vars.add(name)
            return arr[:, :, :, lon_order]

        z = read("z")
        q = read("q") if "q" in ds.variables else np.full(shape, np.nan, dtype=float)
        t = read("t") if "t" in ds.variables else None
        u = read("u")
        v = read("v")
        omega = read("w")
        ozone_name = find_ozone_variable(ds)
        o3_mmr = read(ozone_name)

        geopotential_height_m = z / G0
        pressure_pa = pressure_levels.reshape(1, -1, 1, 1) * 100.0

        if t is not None and np.isfinite(t).any():
            q_for_tv = np.nan_to_num(q, nan=0.0)
            tv = t * (1.0 + 0.61 * q_for_tv)
            rho_air = pressure_pa / (R_D * tv)
            density_method = "ideal_gas_with_temperature"
            temperature_k = t
        else:
            rho_air = density_from_hydrostatic(pressure_levels, geopotential_height_m)
            density_method = "hydrostatic_from_pressure_and_geopotential_height"
            temperature_k = np.full(shape, np.nan, dtype=float)

        o3_ug_m3 = o3_mmr * rho_air * 1.0e9
        o3_ppbv = o3_mmr * (M_AIR / M_O3) * 1.0e9
        w_geo_m_s = -omega / (rho_air * G0)
        flux_u_ug_m2_s = o3_ug_m3 * u
        flux_v_ug_m2_s = o3_ug_m3 * v
        flux_h_mag_ug_m2_s = np.sqrt(flux_u_ug_m2_s**2 + flux_v_ug_m2_s**2)
        flux_w_ug_m2_s = o3_ug_m3 * w_geo_m_s

        time_idx, pressure_idx, lat_idx, lon_idx = np.indices(shape)
        utc_flat = [times_utc[i] for i in time_idx.ravel()]
        bjt_flat = [time + timedelta(hours=8) for time in utc_flat]

        data = pd.DataFrame(
            {
                "source_file": path.name,
                "utc_time": utc_flat,
                "bjt_time": bjt_flat,
                "bjt_date": [time.date().isoformat() for time in bjt_flat],
                "bjt_hour": [time.hour for time in bjt_flat],
                "bjt_year": [time.year for time in bjt_flat],
                "month": [f"{time.month:02d}" for time in bjt_flat],
                "pressure_hPa": pressure_levels[pressure_idx.ravel()],
                "latitude": latitudes[lat_idx.ravel()],
                "longitude": longitudes[lon_idx.ravel()],
                "geopotential_height_m": geopotential_height_m.ravel(),
                "density_method": density_method,
                "temperature_K": temperature_k.ravel(),
                "temperature_C": (temperature_k - 273.15).ravel(),
                "specific_humidity_kg_kg": q.ravel(),
                "air_density_kg_m3": rho_air.ravel(),
                "o3_mmr_kg_kg": o3_mmr.ravel(),
                "o3_ppbv_dry_approx": o3_ppbv.ravel(),
                "o3_mass_ug_m3": o3_ug_m3.ravel(),
                "u_wind_m_s": u.ravel(),
                "v_wind_m_s": v.ravel(),
                "omega_Pa_s": omega.ravel(),
                "w_geometric_m_s": w_geo_m_s.ravel(),
                "o3_flux_u_ug_m2_s": flux_u_ug_m2_s.ravel(),
                "o3_flux_v_ug_m2_s": flux_v_ug_m2_s.ravel(),
                "o3_flux_h_mag_ug_m2_s": flux_h_mag_ug_m2_s.ravel(),
                "o3_flux_w_ug_m2_s": flux_w_ug_m2_s.ravel(),
            }
        )

        variable_names = [
            name
            for name in ("z", "q", "t", "u", "v", "w", ozone_name)
            if name in ds.variables
        ]
        info = {
            "file": path.name,
            "period": file_period_from_name(path.name),
            "size_bytes": path.stat().st_size,
            "last_write_time": pd.Timestamp(path.stat().st_mtime, unit="s"),
            "n_time": len(times_utc),
            "n_pressure": len(pressure_levels),
            "n_latitude": len(latitudes),
            "n_longitude": len(longitudes),
            "time_utc_min": min(times_utc),
            "time_utc_max": max(times_utc),
            "time_bjt_min": min(time + timedelta(hours=8) for time in times_utc),
            "time_bjt_max": max(time + timedelta(hours=8) for time in times_utc),
            "lat_min": float(np.nanmin(latitudes)),
            "lat_max": float(np.nanmax(latitudes)),
            "lon_min": float(np.nanmin(longitudes)),
            "lon_max": float(np.nanmax(longitudes)),
            "pressure_levels_hPa": ", ".join(str(int(level)) for level in pressure_levels),
            "variables": ", ".join(variable_names),
            "has_temperature": "t" in ds.variables,
            "density_method": density_method,
            "expver_combined_vars": ", ".join(sorted(expver_vars)),
        }
        return data, info
    finally:
        ds.close()


def expected_periods() -> pd.DataFrame:
    rows = []
    for month in range(1, 13):
        last_day = calendar.monthrange(2025, month)[1]
        for start_day, end_day in ((1, 15), (16, last_day)):
            rows.append(
                {
                    "expected_period": f"2025-{month:02d}-{start_day:02d}_to_2025-{month:02d}-{end_day:02d}",
                    "month": f"{month:02d}",
                    "start_day": start_day,
                    "end_day": end_day,
                }
            )
    return pd.DataFrame(rows)


def nearest_site_table(data: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    grid = data[["latitude", "longitude"]].drop_duplicates().copy()
    grid["distance2"] = (grid["latitude"] - SITE_LAT) ** 2 + (grid["longitude"] - SITE_LON) ** 2
    nearest = grid.sort_values("distance2").iloc[0]
    lat = float(nearest["latitude"])
    lon = float(nearest["longitude"])
    table = data[(data["latitude"] == lat) & (data["longitude"] == lon)].copy()
    table.insert(0, "site_name", SITE_NAME)
    table.insert(1, "target_latitude", SITE_LAT)
    table.insert(2, "target_longitude", SITE_LON)
    return table, lat, lon


def interpolate_profiles_to_heights(
    profiles: pd.DataFrame,
    group_cols: list[str],
    label_cols: dict[str, object] | None = None,
) -> pd.DataFrame:
    label_cols = label_cols or {}
    rows = []
    interp_cols = ["pressure_hPa"] + MEAN_COLS

    for group_values, group in profiles.groupby(group_cols, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_dict = dict(zip(group_cols, group_values))
        group = group.sort_values("geopotential_height_m")
        heights = group["geopotential_height_m"].to_numpy(dtype=float)
        ok_height = np.isfinite(heights)
        if ok_height.sum() < 2:
            continue
        heights = heights[ok_height]
        keep = group.loc[ok_height].copy()

        unique_heights, unique_idx = np.unique(heights, return_index=True)
        if unique_heights.size < 2:
            continue
        keep = keep.iloc[unique_idx]
        heights = unique_heights

        for target_height in TARGET_HEIGHTS_M:
            row = dict(label_cols)
            row.update(group_dict)
            row["height_m"] = target_height
            for col in interp_cols:
                values = keep[col].to_numpy(dtype=float)
                ok = np.isfinite(values)
                if ok.sum() < 2:
                    row[col] = np.nan
                else:
                    row[col] = np.interp(
                        target_height,
                        heights[ok],
                        values[ok],
                        left=np.nan,
                        right=np.nan,
                    )
            rows.append(row)

    return pd.DataFrame(rows)


def write_csv_tables(sheets: dict[str, pd.DataFrame]) -> None:
    OUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in sheets.items():
        safe_name = name.replace("/", "_")
        df.to_csv(OUT_CSV_DIR / f"{safe_name}.csv", index=False, encoding="utf-8-sig")


def add_excel_formatting(writer: pd.ExcelWriter, sheets: dict[str, pd.DataFrame]) -> None:
    workbook = writer.book
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
    num_fmt = workbook.add_format({"num_format": "0.000"})
    date_fmt = workbook.add_format({"num_format": "yyyy-mm-dd hh:mm"})

    for sheet_name, df in sheets.items():
        worksheet = writer.sheets[sheet_name]
        worksheet.freeze_panes(1, 0)
        if len(df.columns) > 0:
            worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)
        for col_num, col_name in enumerate(df.columns):
            worksheet.write(0, col_num, col_name, header_fmt)
            width = min(max(len(str(col_name)) + 2, 12), 34)
            worksheet.set_column(col_num, col_num, width, num_fmt)
        for time_col in ("utc_time", "bjt_time", "time_utc_min", "time_utc_max", "time_bjt_min", "time_bjt_max"):
            if time_col in df.columns:
                idx = list(df.columns).index(time_col)
                worksheet.set_column(idx, idx, 20, date_fmt)

    if "monthly_pressure_mean" in writer.sheets and not sheets["monthly_pressure_mean"].empty:
        df = sheets["monthly_pressure_mean"]
        worksheet = writer.sheets["monthly_pressure_mean"]
        chart = workbook.add_chart({"type": "line"})
        pressure_col = df.columns.get_loc("pressure_hPa")
        ozone_col = df.columns.get_loc("o3_mass_ug_m3")
        for month in sorted(df["month"].unique()):
            rows = df.index[df["month"] == month].tolist()
            if not rows:
                continue
            first = rows[0] + 1
            last = rows[-1] + 1
            chart.add_series(
                {
                    "name": f"month {month}",
                    "categories": ["monthly_pressure_mean", first, pressure_col, last, pressure_col],
                    "values": ["monthly_pressure_mean", first, ozone_col, last, ozone_col],
                }
            )
        chart.set_title({"name": "Monthly mean O3 by pressure level"})
        chart.set_x_axis({"name": "Pressure level (hPa)"})
        chart.set_y_axis({"name": "O3 mass concentration (ug/m3)"})
        worksheet.insert_chart("N2", chart, {"x_scale": 1.5, "y_scale": 1.3})

    if "height_area_mean" in writer.sheets and not sheets["height_area_mean"].empty:
        df = sheets["height_area_mean"]
        worksheet = writer.sheets["height_area_mean"]
        chart = workbook.add_chart({"type": "line"})
        summary = df.groupby("height_m", as_index=False)["o3_mass_ug_m3"].mean()
        start_row = len(df) + 3
        worksheet.write(start_row, 0, "height_m", header_fmt)
        worksheet.write(start_row, 1, "mean_o3_mass_ug_m3", header_fmt)
        for row_idx, row in summary.iterrows():
            worksheet.write(start_row + 1 + row_idx, 0, row["height_m"])
            worksheet.write(start_row + 1 + row_idx, 1, row["o3_mass_ug_m3"])
        chart.add_series(
            {
                "name": "available mean",
                "categories": ["height_area_mean", start_row + 1, 0, start_row + len(summary), 0],
                "values": ["height_area_mean", start_row + 1, 1, start_row + len(summary), 1],
            }
        )
        chart.set_title({"name": "Area mean O3 by interpolated height"})
        chart.set_x_axis({"name": "Height (m)"})
        chart.set_y_axis({"name": "O3 mass concentration (ug/m3)"})
        worksheet.insert_chart("N2", chart, {"x_scale": 1.5, "y_scale": 1.3})


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_DIR.glob("era5_pressure_shanghai_2025*.nc"))
    if not files:
        raise FileNotFoundError(f"No ERA5 nc files found in {DATA_DIR}")

    frames: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, object]] = []
    for file in files:
        print(f"Reading {file.name}")
        frame, info = read_one_file(file)
        frames.append(frame)
        inventory_rows.append(info)

    data = pd.concat(frames, ignore_index=True)
    inventory = pd.DataFrame(inventory_rows)

    downloaded_periods = set(inventory["period"])
    coverage = expected_periods()
    coverage["downloaded_in_current_folder"] = coverage["expected_period"].isin(downloaded_periods)

    data_bounds = pd.DataFrame(
        [
            {
                "item": "downloaded_area",
                "north": data["latitude"].max(),
                "south": data["latitude"].min(),
                "west": data["longitude"].min(),
                "east": data["longitude"].max(),
                "unique_latitudes": ", ".join(map(str, sorted(data["latitude"].unique()))),
                "unique_longitudes": ", ".join(map(str, sorted(data["longitude"].unique()))),
                "site_target_latitude": SITE_LAT,
                "site_target_longitude": SITE_LON,
            }
        ]
    )

    group_time_pressure = ["bjt_time", "bjt_date", "bjt_hour", "bjt_year", "month", "pressure_hPa"]
    area_hourly_pressure = data.groupby(group_time_pressure, as_index=False)[MEAN_COLS].mean()

    site_data, nearest_lat, nearest_lon = nearest_site_table(data)
    site_hourly_pressure = site_data[
        [
            "site_name",
            "target_latitude",
            "target_longitude",
            "bjt_time",
            "bjt_date",
            "bjt_hour",
            "bjt_year",
            "month",
            "pressure_hPa",
            "latitude",
            "longitude",
        ]
        + MEAN_COLS
    ].copy()

    monthly_pressure_mean = area_hourly_pressure.groupby(["month", "pressure_hPa"], as_index=False)[MEAN_COLS].mean()
    daily_pressure_mean = area_hourly_pressure.groupby(["bjt_date", "month", "pressure_hPa"], as_index=False)[MEAN_COLS].mean()
    hour_level_mean = area_hourly_pressure.groupby(["bjt_hour", "pressure_hPa"], as_index=False)[MEAN_COLS].mean()
    yearly_pressure_mean = area_hourly_pressure.groupby(["bjt_year", "pressure_hPa"], as_index=False)[MEAN_COLS].mean()
    grid_summary = data.groupby(["latitude", "longitude"], as_index=False)[MEAN_COLS].mean()

    blog_style_variable_long = yearly_pressure_mean.melt(
        id_vars=["bjt_year", "pressure_hPa"],
        value_vars=MEAN_COLS,
        var_name="variable",
        value_name="available_year_mean",
    )
    blog_style_variable_long.insert(0, "region_id", "downloaded_area_mean")

    height_area_mean = interpolate_profiles_to_heights(
        area_hourly_pressure,
        ["bjt_time", "bjt_date", "bjt_hour", "bjt_year", "month"],
        {"region_id": "downloaded_area_mean"},
    )
    site_profiles = site_hourly_pressure.rename(
        columns={
            "latitude": "nearest_latitude",
            "longitude": "nearest_longitude",
        }
    )
    height_site_point = interpolate_profiles_to_heights(
        site_profiles,
        ["bjt_time", "bjt_date", "bjt_hour", "bjt_year", "month"],
        {
            "site_name": SITE_NAME,
            "target_latitude": SITE_LAT,
            "target_longitude": SITE_LON,
            "nearest_latitude": nearest_lat,
            "nearest_longitude": nearest_lon,
        },
    )

    full_grid_columns = [
        "source_file",
        "utc_time",
        "bjt_time",
        "bjt_date",
        "bjt_hour",
        "bjt_year",
        "month",
        "pressure_hPa",
        "latitude",
        "longitude",
        "density_method",
        "geopotential_height_m",
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
    full_grid = data[full_grid_columns].copy()

    method_notes = pd.DataFrame(
        [
            {
                "item": "CSDN_processing_adapted",
                "description": "借用博客最后处理节的思路：处理 expver 空值版本、修正经度坐标、做区域均值/时间聚合，并输出可直接用于表格和 GIS 统计的 CSV/Excel。",
            },
            {
                "item": "paper_processing_adapted",
                "description": "结合论文方法：由 ERA5 pressure-level 的 z/q/t/u/v/omega/O3 计算位势高度、空气密度、O3 质量浓度、几何垂直速度和 O3 水平/垂直输送通量。",
            },
            {
                "item": "temperature_note",
                "description": "若 nc 内含 t，则用虚温和理想气体方程算空气密度；旧已下载文件若无 t，则用静力关系 dp/dz=-rho*g 由 pressure 与 geopotential height 估算。",
            },
            {
                "item": "time_note",
                "description": "文件时间按 UTC 读取，并统一加 8 小时生成北京时间 bjt_time；所有 hourly/daily/monthly/yearly 聚合都基于北京时间。",
            },
            {
                "item": "height_interpolation",
                "description": "为贴近论文中垂直廓线处理，将区域均值和站点最近格点的压力层结果按位势高度插值到 0-5000 m、200 m 间隔。",
            },
            {
                "item": "site_point",
                "description": f"站点目标坐标约为 lat={SITE_LAT}, lon={SITE_LON}；当前 ERA5 网格最近点为 lat={nearest_lat}, lon={nearest_lon}。",
            },
            {
                "item": "limitation",
                "description": "当前结果只基于已存盘 ERA5 文件；如果后续月份或 temperature 新请求下载完成，重新运行本脚本即可更新覆盖范围和温度密度计算。",
            },
        ]
    )

    sheets = {
        "method_notes": method_notes,
        "file_inventory": inventory,
        "download_coverage": coverage,
        "data_bounds": data_bounds,
        "grid_summary": grid_summary,
        "area_hourly_pressure": area_hourly_pressure,
        "site_hourly_pressure": site_hourly_pressure,
        "height_area_mean": height_area_mean,
        "height_site_point": height_site_point,
        "monthly_pressure_mean": monthly_pressure_mean,
        "daily_pressure_mean": daily_pressure_mean,
        "hour_level_mean": hour_level_mean,
        "yearly_pressure_mean": yearly_pressure_mean,
        "blog_style_yearly_long": blog_style_variable_long,
        "full_grid_long": full_grid,
    }

    print(f"Writing CSV tables to {OUT_CSV_DIR}")
    write_csv_tables(sheets)

    print(f"Writing Excel workbook to {OUT_XLSX}")
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        add_excel_formatting(writer, sheets)

    print(f"Done: {OUT_XLSX}")
    print(f"Files processed: {len(files)}")
    print(f"Full-grid rows: {len(full_grid)}")
    print(f"Area height rows: {len(height_area_mean)}")
    print(f"Site height rows: {len(height_site_point)}")


if __name__ == "__main__":
    main()
