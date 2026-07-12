#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract compact HF/CF O3 and EXCH_H profiles from daily WRF post files."""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np


SITES = (
    ("HF", 31.78, 117.18),
    ("CF", 32.21, 117.18),
)
UTC_HOURS = tuple(range(9))


def iter_days(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def coordinate_2d(ds: nc.Dataset, name: str) -> np.ndarray:
    values = np.asarray(ds.variables[name][:], dtype=np.float64)
    while values.ndim > 2:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"{name} must resolve to 2-D, got shape {values.shape}")
    return values


def nearest_indices(
    xlat: np.ndarray,
    xlon: np.ndarray,
    sites: tuple[tuple[str, float, float], ...] = SITES,
) -> list[tuple[int, int]]:
    indices = []
    for _, target_lat, target_lon in sites:
        distance2 = (xlat - target_lat) ** 2 + (xlon - target_lon) ** 2
        indices.append(tuple(int(v) for v in np.unravel_index(np.argmin(distance2), distance2.shape)))
    return indices


def variable_units(ds: nc.Dataset, name: str) -> str:
    return str(getattr(ds.variables[name], "units", ""))


def read_profile(var: nc.Variable, hour_indices: list[int], iy: int, ix: int) -> np.ndarray:
    return np.asarray(var[hour_indices, :, iy, ix], dtype=np.float32)


def read_surface(var: nc.Variable, hour_indices: list[int], iy: int, ix: int) -> np.ndarray:
    return np.asarray(var[hour_indices, iy, ix], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/exports/d1/wrf_only_daily/wrfpost")
    parser.add_argument("--start", default="2023-04-01")
    parser.add_argument("--end", default="2023-10-31")
    parser.add_argument(
        "--output",
        default=str(Path.home() / "wrf_hf_cf_o3_exch_h_202304_202310.nc"),
    )
    parser.add_argument(
        "--report",
        default=str(Path.home() / "wrf_hf_cf_o3_exch_h_202304_202310_report.json"),
    )
    args = parser.parse_args()

    root = Path(args.root)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    output = Path(args.output)
    report_path = Path(args.report)

    records: list[dict[str, object]] = []
    missing: list[dict[str, str]] = []
    station_indices: list[tuple[int, int]] | None = None
    grid_lats: list[float] = []
    grid_lons: list[float] = []
    normal_mass_variable_names = (
        "o3",
        "height",
        "pressure",
        "tempc",
        "qv",
        "ua",
        "va",
        "wa",
    )
    physical_mass_variable_names = ("rh",)
    mass_variable_names = normal_mass_variable_names + physical_mass_variable_names
    source_units: dict[str, str] = {}
    source_descriptions: dict[str, str] = {}
    mass_levels = None
    staggered_levels = None

    for day in iter_days(start, end):
        month_dir = f"{day.year}M{day.month:02d}"
        stamp = day.strftime("%Y%m%d")
        normal_path = root / "normal_new" / month_dir / f"{stamp}00_normal.nc"
        physical_path = root / "physical_new" / month_dir / f"{stamp}00_physical.nc"

        if not normal_path.exists() or not physical_path.exists():
            missing.append(
                {
                    "date": day.isoformat(),
                    "normal": "present" if normal_path.exists() else "missing",
                    "physical": "present" if physical_path.exists() else "missing",
                }
            )
            continue

        with nc.Dataset(normal_path) as normal, nc.Dataset(physical_path) as physical:
            required_normal = {
                "xlat",
                "xlon",
                "height",
                "pressure",
                "o3",
                "tempc",
                "qv",
                "ua",
                "va",
                "wa",
                "pblh",
            }
            required_physical = {"exch_h", "rh"}
            absent = sorted(required_normal - set(normal.variables))
            absent += sorted(required_physical - set(physical.variables))
            if absent:
                missing.append({"date": day.isoformat(), "variables": ",".join(absent)})
                continue

            if station_indices is None:
                xlat = coordinate_2d(normal, "xlat")
                xlon = coordinate_2d(normal, "xlon")
                station_indices = nearest_indices(xlat, xlon)
                grid_lats = [float(xlat[iy, ix]) for iy, ix in station_indices]
                grid_lons = [float(xlon[iy, ix]) for iy, ix in station_indices]
                for name in normal_mass_variable_names + ("pblh",):
                    source_units[name] = variable_units(normal, name)
                    source_descriptions[name] = str(
                        getattr(normal.variables[name], "description", "")
                    )
                for name in physical_mass_variable_names:
                    source_units[name] = variable_units(physical, name)
                    source_descriptions[name] = str(
                        getattr(physical.variables[name], "description", "")
                    )
                source_units["exch_h"] = variable_units(physical, "exch_h")
                source_descriptions["exch_h"] = str(
                    getattr(physical.variables["exch_h"], "description", "")
                )
                mass_levels = int(normal.variables["o3"].shape[1])
                staggered_levels = int(physical.variables["exch_h"].shape[1])

            available_hours = min(
                *(
                    int(normal.variables[name].shape[0])
                    for name in normal_mass_variable_names
                ),
                *(
                    int(physical.variables[name].shape[0])
                    for name in physical_mass_variable_names
                ),
                int(normal.variables["pblh"].shape[0]),
                int(physical.variables["exch_h"].shape[0]),
            )
            hours = [hour for hour in UTC_HOURS if hour < available_hours]
            if not hours:
                missing.append({"date": day.isoformat(), "hours": "none"})
                continue

            site_mass_data = {name: [] for name in mass_variable_names}
            site_pblh = []
            site_exch_h = []
            for iy, ix in station_indices:
                for name in normal_mass_variable_names:
                    site_mass_data[name].append(
                        read_profile(normal.variables[name], hours, iy, ix)
                    )
                for name in physical_mass_variable_names:
                    site_mass_data[name].append(
                        read_profile(physical.variables[name], hours, iy, ix)
                    )
                site_pblh.append(read_surface(normal.variables["pblh"], hours, iy, ix))
                site_exch_h.append(
                    read_profile(physical.variables["exch_h"], hours, iy, ix)
                )

            records.append(
                {
                    "date": day,
                    "hours": hours,
                    "mass_data": {
                        name: np.stack(site_mass_data[name], axis=1)
                        for name in mass_variable_names
                    },
                    "pblh": np.stack(site_pblh, axis=1),
                    "exch_h": np.stack(site_exch_h, axis=1),
                }
            )

        if day.day == 1 or day == end:
            print(f"Processed through {day.isoformat()}", flush=True)

    if not records or station_indices is None or mass_levels is None or staggered_levels is None:
        raise RuntimeError("No matching WRF records were extracted.")

    total_times = sum(len(record["hours"]) for record in records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(output, "w", format="NETCDF4") as out:
        out.createDimension("time", total_times)
        out.createDimension("station", len(SITES))
        out.createDimension("mass_level", mass_levels)
        out.createDimension("staggered_level", staggered_levels)
        out.title = "WRF-Chem HF/CF station profiles: O3 and EXCH_H"
        out.source_root = str(root)
        out.time_note = "UTC 00:00-08:00, corresponding to Beijing time 08:00-16:00"

        time_var = out.createVariable("time", "i8", ("time",))
        time_var.units = "seconds since 1970-01-01 00:00:00 UTC"
        time_var.calendar = "standard"
        station_name = out.createVariable("station_name", str, ("station",))
        target_lat = out.createVariable("target_latitude", "f4", ("station",))
        target_lon = out.createVariable("target_longitude", "f4", ("station",))
        grid_lat = out.createVariable("grid_latitude", "f4", ("station",))
        grid_lon = out.createVariable("grid_longitude", "f4", ("station",))
        grid_y = out.createVariable("grid_y_index", "i4", ("station",))
        grid_x = out.createVariable("grid_x_index", "i4", ("station",))

        mass_output_variables = {}
        for name in mass_variable_names:
            var = out.createVariable(
                name,
                "f4",
                ("time", "station", "mass_level"),
                zlib=True,
                complevel=4,
                fill_value=np.float32(np.nan),
            )
            var.units = source_units[name]
            var.description = source_descriptions[name]
            mass_output_variables[name] = var

        pblh_var = out.createVariable(
            "pblh",
            "f4",
            ("time", "station"),
            zlib=True,
            complevel=4,
            fill_value=np.float32(np.nan),
        )
        pblh_var.units = source_units["pblh"]
        pblh_var.description = source_descriptions["pblh"]
        exch_h_var = out.createVariable(
            "exch_h",
            "f4",
            ("time", "station", "staggered_level"),
            zlib=True,
            complevel=4,
            fill_value=np.float32(np.nan),
        )
        exch_h_var.units = source_units["exch_h"]
        exch_h_var.description = source_descriptions["exch_h"]

        temperature_k_var = out.createVariable(
            "temperature_k",
            "f4",
            ("time", "station", "mass_level"),
            zlib=True,
            complevel=4,
            fill_value=np.float32(np.nan),
        )
        temperature_k_var.units = "K"
        temperature_k_var.description = "Temperature converted from source tempc"
        o3_mass_var = out.createVariable(
            "o3_mass_concentration",
            "f4",
            ("time", "station", "mass_level"),
            zlib=True,
            complevel=4,
            fill_value=np.float32(np.nan),
        )
        o3_mass_var.units = "ug m-3"
        o3_mass_var.description = "O3 mass concentration derived from ppmv, pressure, and temperature"
        density_var = out.createVariable(
            "air_density",
            "f4",
            ("time", "station", "mass_level"),
            zlib=True,
            complevel=4,
            fill_value=np.float32(np.nan),
        )
        density_var.units = "kg m-3"
        density_var.description = "Moist-air density derived from pressure, temperature, and qv"

        station_name[:] = np.asarray([site[0] for site in SITES], dtype=object)
        target_lat[:] = [site[1] for site in SITES]
        target_lon[:] = [site[2] for site in SITES]
        grid_lat[:] = grid_lats
        grid_lon[:] = grid_lons
        grid_y[:] = [index[0] for index in station_indices]
        grid_x[:] = [index[1] for index in station_indices]

        cursor = 0
        for record in records:
            day = record["date"]
            hours = record["hours"]
            count = len(hours)
            timestamps = [
                int(
                    datetime(
                        day.year,
                        day.month,
                        day.day,
                        hour,
                        tzinfo=timezone.utc,
                    ).timestamp()
                )
                for hour in hours
            ]
            sl = slice(cursor, cursor + count)
            time_var[sl] = timestamps
            mass_data = record["mass_data"]
            for name in mass_variable_names:
                mass_output_variables[name][sl, :, :] = mass_data[name]
            pblh_var[sl, :] = record["pblh"]
            exch_h_var[sl, :, :] = record["exch_h"]

            temperature_k = mass_data["tempc"] + np.float32(273.15)
            pressure_pa = mass_data["pressure"] * np.float32(100.0)
            o3_mass = (
                mass_data["o3"]
                * pressure_pa
                * np.float32(48.0)
                / (np.float32(8.314462618) * temperature_k)
            )
            virtual_temperature = temperature_k * (
                np.float32(1.0) + np.float32(0.61) * mass_data["qv"]
            )
            air_density = pressure_pa / (np.float32(287.05) * virtual_temperature)
            temperature_k_var[sl, :, :] = temperature_k
            o3_mass_var[sl, :, :] = o3_mass
            density_var[sl, :, :] = air_density
            cursor += count

    report = {
        "output": str(output),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "time_records": total_times,
        "utc_hours": list(UTC_HOURS),
        "beijing_hours": [hour + 8 for hour in UTC_HOURS],
        "sites": [
            {
                "name": site[0],
                "target_lat": site[1],
                "target_lon": site[2],
                "grid_lat": grid_lats[index],
                "grid_lon": grid_lons[index],
                "grid_y": station_indices[index][0],
                "grid_x": station_indices[index][1],
            }
            for index, site in enumerate(SITES)
        ],
        "mass_levels": mass_levels,
        "staggered_levels": staggered_levels,
        "source_units": source_units,
        "source_descriptions": source_descriptions,
        "derived_variables": {
            "temperature_k": "K",
            "o3_mass_concentration": "ug m-3",
            "air_density": "kg m-3",
        },
        "missing_count": len(missing),
        "missing": missing,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output} ({output.stat().st_size} bytes)", flush=True)
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
