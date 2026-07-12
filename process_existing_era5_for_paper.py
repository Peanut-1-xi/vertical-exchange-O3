from __future__ import annotations

import calendar
from datetime import timedelta
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd


DATA_DIR = Path(r"E:\research\垂直交换\上海_ERA5_2025_pressure_levels")
OUT_DIR = Path(r"E:\research\垂直交换\ERA5_论文方法处理结果")
OUT_XLSX = OUT_DIR / "上海_ERA5_已有数据_论文方法处理与全局可视化.xlsx"

G0 = 9.80665
R_D = 287.05
M_AIR = 28.9647
M_O3 = 47.9982


def open_nc_from_bytes(path: Path) -> netCDF4.Dataset:
    # netCDF4 on this Windows environment can stumble on Chinese paths.
    return netCDF4.Dataset("inmemory.nc", memory=path.read_bytes())


def read_one_file(path: Path) -> pd.DataFrame:
    ds = open_nc_from_bytes(path)
    try:
        time_var = ds.variables["valid_time"]
        utc_times = netCDF4.num2date(
            time_var[:],
            time_var.units,
            getattr(time_var, "calendar", "standard"),
            only_use_cftime_datetimes=False,
        )
        pressure_levels = np.asarray(ds.variables["pressure_level"][:], dtype=float)
        latitudes = np.asarray(ds.variables["latitude"][:], dtype=float)
        longitudes = np.asarray(ds.variables["longitude"][:], dtype=float)

        z = np.asarray(ds.variables["z"][:], dtype=float)
        q = np.asarray(ds.variables["q"][:], dtype=float) if "q" in ds.variables else np.full_like(z, np.nan)
        t = np.asarray(ds.variables["t"][:], dtype=float) if "t" in ds.variables else None
        u = np.asarray(ds.variables["u"][:], dtype=float)
        v = np.asarray(ds.variables["v"][:], dtype=float)
        omega = np.asarray(ds.variables["w"][:], dtype=float)

        ozone_name = "o3" if "o3" in ds.variables else "o3_mmr"
        if ozone_name not in ds.variables:
            # CDS uses short_name=o3 for "ozone mass mixing ratio" in these files.
            candidates = [
                name
                for name, var in ds.variables.items()
                if "ozone" in getattr(var, "long_name", "").lower()
            ]
            if not candidates:
                raise KeyError(f"No ozone mass mixing ratio variable found in {path.name}")
            ozone_name = candidates[0]
        o3_mmr = np.asarray(ds.variables[ozone_name][:], dtype=float)

        nt, npres, nlat, nlon = z.shape
        pressure_pa = pressure_levels.reshape(1, npres, 1, 1) * 100.0
        geopotential_height_m = z / G0

        if t is not None:
            tv = t * (1.0 + 0.61 * q)
            rho_air = pressure_pa / (R_D * tv)
            density_method = "ideal_gas_with_temperature"
            temperature_k = t
        else:
            # Temperature was not selected in the current CDS requests. Estimate density
            # from the hydrostatic relation dp/dz = -rho*g using geopotential height.
            pressure_1d = pressure_levels * 100.0
            rho_air = np.full_like(z, np.nan, dtype=float)
            for ti in range(nt):
                for yi in range(nlat):
                    for xi in range(nlon):
                        height_profile = geopotential_height_m[ti, :, yi, xi]
                        dp_dz = np.gradient(pressure_1d, height_profile)
                        rho_air[ti, :, yi, xi] = -dp_dz / G0
            density_method = "hydrostatic_from_pressure_and_geopotential_height"
            temperature_k = np.full_like(z, np.nan, dtype=float)
        o3_ug_m3 = o3_mmr * rho_air * 1.0e9
        o3_ppbv = o3_mmr * (M_AIR / M_O3) * 1.0e9
        w_geo_m_s = -omega / (rho_air * G0)

        flux_u_ug_m2_s = o3_ug_m3 * u
        flux_v_ug_m2_s = o3_ug_m3 * v
        flux_h_mag_ug_m2_s = np.sqrt(flux_u_ug_m2_s**2 + flux_v_ug_m2_s**2)
        flux_w_ug_m2_s = o3_ug_m3 * w_geo_m_s

        rows = []
        for ti, utc_time in enumerate(utc_times):
            bjt_time = utc_time + timedelta(hours=8)
            for pi, pressure in enumerate(pressure_levels):
                for yi, lat in enumerate(latitudes):
                    for xi, lon in enumerate(longitudes):
                        rows.append(
                            {
                                "source_file": path.name,
                                "utc_time": utc_time,
                                "bjt_time": bjt_time,
                                "bjt_date": bjt_time.date().isoformat(),
                                "bjt_hour": bjt_time.hour,
                                "month": f"{bjt_time.month:02d}",
                                "pressure_hPa": pressure,
                                "latitude": lat,
                                "longitude": lon,
                                "geopotential_height_m": geopotential_height_m[ti, pi, yi, xi],
                                "density_method": density_method,
                                "temperature_K": temperature_k[ti, pi, yi, xi],
                                "temperature_C": temperature_k[ti, pi, yi, xi] - 273.15,
                                "specific_humidity_kg_kg": q[ti, pi, yi, xi],
                                "air_density_kg_m3": rho_air[ti, pi, yi, xi],
                                "o3_mmr_kg_kg": o3_mmr[ti, pi, yi, xi],
                                "o3_ppbv_dry_approx": o3_ppbv[ti, pi, yi, xi],
                                "o3_mass_ug_m3": o3_ug_m3[ti, pi, yi, xi],
                                "u_wind_m_s": u[ti, pi, yi, xi],
                                "v_wind_m_s": v[ti, pi, yi, xi],
                                "omega_Pa_s": omega[ti, pi, yi, xi],
                                "w_geometric_m_s": w_geo_m_s[ti, pi, yi, xi],
                                "o3_flux_u_ug_m2_s": flux_u_ug_m2_s[ti, pi, yi, xi],
                                "o3_flux_v_ug_m2_s": flux_v_ug_m2_s[ti, pi, yi, xi],
                                "o3_flux_h_mag_ug_m2_s": flux_h_mag_ug_m2_s[ti, pi, yi, xi],
                                "o3_flux_w_ug_m2_s": flux_w_ug_m2_s[ti, pi, yi, xi],
                            }
                        )
        return pd.DataFrame(rows)
    finally:
        ds.close()


def period_from_filename(name: str) -> str:
    # era5_pressure_shanghai_202501_01-15.nc -> 2025-01-01_to_2025-01-15
    stem = Path(name).stem
    parts = stem.split("_")
    ym = parts[-2]
    days = parts[-1]
    start_day, end_day = days.split("-")
    return f"{ym[:4]}-{ym[4:]}-{start_day}_to_{ym[:4]}-{ym[4:]}-{end_day}"


def build_inventory(files: list[Path]) -> pd.DataFrame:
    rows = []
    for file in files:
        rows.append(
            {
                "file": file.name,
                "period": period_from_filename(file.name),
                "size_bytes": file.stat().st_size,
                "last_write_time": pd.Timestamp(file.stat().st_mtime, unit="s"),
            }
        )
    return pd.DataFrame(rows)


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


def add_excel_formatting(writer: pd.ExcelWriter, sheets: dict[str, pd.DataFrame]) -> None:
    workbook = writer.book
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
    num_fmt = workbook.add_format({"num_format": "0.000"})
    date_fmt = workbook.add_format({"num_format": "yyyy-mm-dd hh:mm"})

    for sheet_name, df in sheets.items():
        ws = writer.sheets[sheet_name]
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, len(df), max(len(df.columns) - 1, 0))
        for col_num, col_name in enumerate(df.columns):
            ws.write(0, col_num, col_name, header_fmt)
            width = min(max(len(str(col_name)) + 2, 12), 32)
            ws.set_column(col_num, col_num, width, num_fmt)
        if "bjt_time" in df.columns:
            idx = list(df.columns).index("bjt_time")
            ws.set_column(idx, idx, 20, date_fmt)
        if "utc_time" in df.columns:
            idx = list(df.columns).index("utc_time")
            ws.set_column(idx, idx, 20, date_fmt)

    if "monthly_pressure_mean" in writer.sheets:
        ws = writer.sheets["monthly_pressure_mean"]
        df = sheets["monthly_pressure_mean"]
        if not df.empty:
            chart = workbook.add_chart({"type": "line"})
            # Plot one series per month for ozone concentration versus row order.
            # This is a quick-view chart; detailed contour plots can be made from the tables.
            for month in sorted(df["month"].unique())[:6]:
                month_rows = df.index[df["month"] == month].tolist()
                if not month_rows:
                    continue
                first = month_rows[0] + 1
                last = month_rows[-1] + 1
                col_y = df.columns.get_loc("o3_mass_ug_m3")
                chart.add_series(
                    {
                        "name": f"month {month}",
                        "categories": ["monthly_pressure_mean", first, df.columns.get_loc("pressure_hPa"), last, df.columns.get_loc("pressure_hPa")],
                        "values": ["monthly_pressure_mean", first, col_y, last, col_y],
                    }
                )
            chart.set_title({"name": "Monthly mean O3 mass concentration by pressure level"})
            chart.set_x_axis({"name": "Pressure level (hPa)"})
            chart.set_y_axis({"name": "O3 (ug/m3)"})
            ws.insert_chart("N2", chart, {"x_scale": 1.4, "y_scale": 1.3})


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_DIR.glob("era5_pressure_shanghai_2025*.nc"))
    if not files:
        raise FileNotFoundError(f"No ERA5 nc files found in {DATA_DIR}")

    frames = []
    for file in files:
        print(f"Reading {file.name}")
        frames.append(read_one_file(file))
    data = pd.concat(frames, ignore_index=True)

    inventory = build_inventory(files)
    downloaded_periods = set(inventory["period"])
    missing = expected_periods()
    missing["downloaded_in_current_folder"] = missing["expected_period"].isin(downloaded_periods)

    group_cols = ["bjt_time", "bjt_date", "bjt_hour", "month", "pressure_hPa"]
    mean_cols = [
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
    hourly_pressure_mean = data.groupby(group_cols, as_index=False)[mean_cols].mean()

    daily_pressure_mean = data.groupby(["bjt_date", "month", "pressure_hPa"], as_index=False)[mean_cols].mean()
    monthly_pressure_mean = data.groupby(["month", "pressure_hPa"], as_index=False)[mean_cols].mean()
    hour_level_mean = data.groupby(["bjt_hour", "pressure_hPa"], as_index=False)[mean_cols].mean()
    grid_summary = data.groupby(["latitude", "longitude"], as_index=False)[mean_cols].mean()

    # Compact full-grid table for Excel visualization; still includes all current rows.
    full_grid_columns = [
        "bjt_time",
        "bjt_date",
        "bjt_hour",
        "month",
        "pressure_hPa",
        "latitude",
        "longitude",
        "geopotential_height_m",
        "o3_mass_ug_m3",
        "o3_ppbv_dry_approx",
        "u_wind_m_s",
        "v_wind_m_s",
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
                "item": "data_used",
                "description": "已有 ERA5 pressure-level NetCDF 文件；包括 O3 mass mixing ratio, z, q, t, u, v, omega(w).",
            },
            {
                "item": "paper_alignment",
                "description": "按论文思路使用 ERA5 的温度、比湿、风场、垂直速度和 O3 垂直廓线计算输送通量；当前没有 MAX-DOAS NO2/HCHO，也没有 ERA5 single-level BLH，因此先做 ERA5-O3 的通量诊断。",
            },
            {
                "item": "o3_mass_concentration",
                "description": "ERA5 O3 mass mixing ratio(kg/kg) * air density(kg/m3) * 1e9 = ug/m3；air density = p/(Rd*Tv), Tv=T*(1+0.61q).",
            },
            {
                "item": "vertical_velocity",
                "description": "ERA5 omega 为 Pa/s；几何垂直速度 w_geo=-omega/(rho*g)，正值表示向上，负值表示向下。",
            },
            {
                "item": "transport_flux",
                "description": "水平通量 O3_flux_u/v = O3_mass(ug/m3)*u/v(m/s)；垂直通量 O3_flux_w = O3_mass*w_geo。单位均为 ug/m2/s。",
            },
            {
                "item": "limitation",
                "description": "论文主分析通常应与观测 O3/NO2/HCHO 垂直廓线、FNR、BLH 配合；本表是基于当前 ERA5 已下载数据的预处理和可视化数据底表。",
            },
        ]
    )

    sheets = {
        "method_notes": method_notes,
        "file_inventory": inventory,
        "download_coverage": missing,
        "grid_summary": grid_summary,
        "hour_level_mean": hour_level_mean,
        "monthly_pressure_mean": monthly_pressure_mean,
        "daily_pressure_mean": daily_pressure_mean,
        "hourly_pressure_mean": hourly_pressure_mean,
        "full_grid_long": full_grid,
    }

    print(f"Writing {OUT_XLSX}")
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        add_excel_formatting(writer, sheets)

    print(f"Done: {OUT_XLSX}")
    print(f"Files processed: {len(files)}")
    print(f"Full-grid rows: {len(full_grid)}")


if __name__ == "__main__":
    main()
