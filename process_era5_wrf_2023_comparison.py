from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd


G0 = 9.80665
R_UNIVERSAL = 8.314462618
MOLAR_MASS_O3_G_MOL = 48.0
R_D = 287.05
TARGET_HEIGHTS_M = np.arange(0.0, 4000.1, 100.0)


def interp_profile(
    source_height_m: np.ndarray,
    source_value: np.ndarray,
    target_height_m: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate one profile without extrapolating."""
    heights = np.asarray(source_height_m, dtype=float)
    values = np.asarray(source_value, dtype=float)
    targets = np.asarray(target_height_m, dtype=float)
    valid = np.isfinite(heights) & np.isfinite(values)
    if valid.sum() < 2:
        return np.full(targets.shape, np.nan, dtype=float)

    heights = heights[valid]
    values = values[valid]
    order = np.argsort(heights)
    heights = heights[order]
    values = values[order]
    unique_height, unique_index = np.unique(heights, return_index=True)
    values = values[unique_index]
    if unique_height.size < 2:
        return np.full(targets.shape, np.nan, dtype=float)

    return np.interp(targets, unique_height, values, left=np.nan, right=np.nan)


def omega_to_w_geo(omega_pa_s: np.ndarray, density_kg_m3: np.ndarray) -> np.ndarray:
    """Convert ERA5 pressure velocity to geometric velocity, positive upward."""
    return -np.asarray(omega_pa_s, dtype=float) / (
        np.asarray(density_kg_m3, dtype=float) * G0
    )


def geopotential_to_agl(
    profile_geopotential_m2_s2: np.ndarray,
    surface_geopotential_m2_s2: float,
) -> np.ndarray:
    """Convert ERA5 profile geopotential to height above model ground."""
    return (
        np.asarray(profile_geopotential_m2_s2, dtype=float)
        - float(surface_geopotential_m2_s2)
    ) / G0


def wrf_o3_ppmv_to_ug_m3(
    o3_ppmv: np.ndarray,
    pressure_hpa: np.ndarray,
    temperature_k: np.ndarray,
) -> np.ndarray:
    """Convert WRF O3 volume mixing ratio in ppmv to mass concentration."""
    return (
        np.asarray(o3_ppmv, dtype=float)
        * np.asarray(pressure_hpa, dtype=float)
        * 100.0
        * MOLAR_MASS_O3_G_MOL
        / (R_UNIVERSAL * np.asarray(temperature_k, dtype=float))
    )


def nearest_grid_point(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    target_latitude: float,
    target_longitude: float,
) -> tuple[int, int, float, float]:
    """Return nearest rectilinear-grid indices and coordinates."""
    latitude = np.asarray(latitudes, dtype=float)
    longitude = np.asarray(longitudes, dtype=float)
    iy = int(np.nanargmin(np.abs(latitude - target_latitude)))
    ix = int(np.nanargmin(np.abs(longitude - target_longitude)))
    return iy, ix, float(latitude[iy]), float(longitude[ix])


def comparison_metrics(
    frame: pd.DataFrame,
    era5_column: str,
    wrf_column: str,
) -> dict[str, float | int]:
    """Calculate paired ERA5-minus-WRF comparison metrics."""
    paired = frame[[era5_column, wrf_column]].replace([np.inf, -np.inf], np.nan).dropna()
    if paired.empty:
        return {
            "n": 0,
            "bias_era5_minus_wrf": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "pearson_r": np.nan,
        }
    difference = paired[era5_column] - paired[wrf_column]
    correlation = (
        float(paired[era5_column].corr(paired[wrf_column]))
        if len(paired) >= 2
        else np.nan
    )
    return {
        "n": int(len(paired)),
        "bias_era5_minus_wrf": float(difference.mean()),
        "mae": float(difference.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "pearson_r": correlation,
    }


def interface_heights_from_mass_levels(mass_height_m: np.ndarray) -> np.ndarray:
    """Estimate WRF staggered interface heights from mass-level centers."""
    mass_height = np.asarray(mass_height_m, dtype=float)
    if mass_height.ndim != 1 or mass_height.size < 2:
        raise ValueError("At least two one-dimensional mass-level heights are required.")
    interfaces = np.empty(mass_height.size + 1, dtype=float)
    interfaces[0] = 0.0
    interfaces[1:-1] = (mass_height[:-1] + mass_height[1:]) / 2.0
    interfaces[-1] = mass_height[-1] + (mass_height[-1] - mass_height[-2]) / 2.0
    return interfaces


def open_netcdf(path: Path) -> netCDF4.Dataset:
    """Open a NetCDF file through memory to support non-ASCII Windows paths."""
    return netCDF4.Dataset("inmemory.nc", memory=Path(path).read_bytes())


def _station_names(variable: netCDF4.Variable) -> list[str]:
    values = variable[:]
    return [str(value) for value in values]


def _fill_near_surface(values: dict[str, np.ndarray]) -> list[str]:
    """Fill missing 0/100 m values from the nearest valid low target layer."""
    methods = ["interpolated"] * len(TARGET_HEIGHTS_M)
    reference = values.get("o3_mass_ug_m3", values.get("o3_ppmv"))
    if reference is None:
        reference = next(iter(values.values()))
    valid_index = np.flatnonzero(np.isfinite(reference))
    if valid_index.size == 0:
        return methods
    source_index = int(valid_index[0])
    for target_index in (0, 1, 2):
        if target_index >= source_index or target_index >= len(TARGET_HEIGHTS_M):
            continue
        for name, profile in values.items():
            if name == "exch_h_source_value":
                continue
            if np.isnan(profile[target_index]) and np.isfinite(profile[source_index]):
                profile[target_index] = profile[source_index]
        methods[target_index] = f"copied_from_{int(TARGET_HEIGHTS_M[source_index])}m"
    return methods


def process_wrf_file(path: Path, max_times: int | None = None) -> pd.DataFrame:
    """Interpolate WRF station profiles to 0-4 km AGL at 100 m intervals."""
    with open_netcdf(path) as ds:
        total_times = len(ds.dimensions["time"])
        count = total_times if max_times is None else min(max_times, total_times)
        seconds = np.asarray(ds.variables["time"][:count], dtype=np.int64)
        bjt_times = (
            pd.to_datetime(seconds, unit="s", utc=True)
            .tz_convert("Asia/Shanghai")
            .tz_localize(None)
        )
        stations = _station_names(ds.variables["station_name"])
        target_lat = np.asarray(ds.variables["target_latitude"][:], dtype=float)
        target_lon = np.asarray(ds.variables["target_longitude"][:], dtype=float)
        grid_lat = np.asarray(ds.variables["grid_latitude"][:], dtype=float)
        grid_lon = np.asarray(ds.variables["grid_longitude"][:], dtype=float)
        pblh = np.asarray(ds.variables["pblh"][:count], dtype=float)
        arrays = {
            name: np.asarray(ds.variables[name][:count], dtype=float)
            for name in (
                "height",
                "o3",
                "pressure",
                "temperature_k",
                "qv",
                "ua",
                "va",
                "wa",
                "rh",
                "exch_h",
            )
        }

    rows: list[dict[str, object]] = []
    for time_index, bjt_time in enumerate(bjt_times):
        for station_index, station in enumerate(stations):
            height = arrays["height"][time_index, station_index]
            profiles = {
                "o3_ppmv": interp_profile(
                    height, arrays["o3"][time_index, station_index], TARGET_HEIGHTS_M
                ),
                "temperature_K": interp_profile(
                    height,
                    arrays["temperature_k"][time_index, station_index],
                    TARGET_HEIGHTS_M,
                ),
                "specific_humidity_kg_kg": interp_profile(
                    height, arrays["qv"][time_index, station_index], TARGET_HEIGHTS_M
                ),
                "u_wind_m_s": interp_profile(
                    height, arrays["ua"][time_index, station_index], TARGET_HEIGHTS_M
                ),
                "v_wind_m_s": interp_profile(
                    height, arrays["va"][time_index, station_index], TARGET_HEIGHTS_M
                ),
                "w_geometric_m_s": interp_profile(
                    height, arrays["wa"][time_index, station_index], TARGET_HEIGHTS_M
                ),
                "relative_humidity_pct": interp_profile(
                    height, arrays["rh"][time_index, station_index], TARGET_HEIGHTS_M
                ),
            }
            log_pressure = interp_profile(
                height,
                np.log(arrays["pressure"][time_index, station_index]),
                TARGET_HEIGHTS_M,
            )
            profiles["pressure_hPa"] = np.exp(log_pressure)
            interface_height = interface_heights_from_mass_levels(height)
            profiles["exch_h_source_value"] = interp_profile(
                interface_height,
                arrays["exch_h"][time_index, station_index],
                TARGET_HEIGHTS_M,
            )
            profiles["o3_mass_ug_m3"] = wrf_o3_ppmv_to_ug_m3(
                profiles["o3_ppmv"],
                profiles["pressure_hPa"],
                profiles["temperature_K"],
            )
            virtual_temperature = profiles["temperature_K"] * (
                1.0 + 0.61 * profiles["specific_humidity_kg_kg"]
            )
            profiles["air_density_kg_m3"] = (
                profiles["pressure_hPa"] * 100.0 / (R_D * virtual_temperature)
            )
            fill_methods = _fill_near_surface(profiles)

            for level_index, target_height in enumerate(TARGET_HEIGHTS_M):
                row: dict[str, object] = {
                    "source": "WRF",
                    "station": station,
                    "target_latitude": target_lat[station_index],
                    "target_longitude": target_lon[station_index],
                    "grid_latitude": grid_lat[station_index],
                    "grid_longitude": grid_lon[station_index],
                    "bjt_time": bjt_time,
                    "bjt_date": bjt_time.date().isoformat(),
                    "bjt_hour": bjt_time.hour,
                    "month": bjt_time.month,
                    "height_agl_m": target_height,
                    "surface_fill_method": fill_methods[level_index],
                    "pblh_m": pblh[time_index, station_index],
                }
                for name, profile in profiles.items():
                    row[name] = profile[level_index]
                rows.append(row)
    return pd.DataFrame(rows)


def process_era5_files(
    paths: list[Path],
    stations: dict[str, tuple[float, float]],
    max_times_per_file: int | None = None,
    surface_geopotential_path: Path | None = None,
) -> pd.DataFrame:
    """Interpolate ERA5 nearest-grid profiles to a 0-4 km AGL grid."""
    surface_geopotential: dict[str, float] = {}
    if surface_geopotential_path is not None:
        with open_netcdf(surface_geopotential_path) as surface_ds:
            surface_latitude = np.asarray(surface_ds.variables["latitude"][:], dtype=float)
            surface_longitude = np.asarray(surface_ds.variables["longitude"][:], dtype=float)
            surface_z = np.asarray(surface_ds.variables["z"][:], dtype=float).squeeze()
        if surface_z.ndim != 2:
            raise ValueError(f"Surface geopotential must reduce to 2-D, got {surface_z.shape}")
        for station, (target_lat, target_lon) in stations.items():
            iy, ix, _, _ = nearest_grid_point(
                surface_latitude, surface_longitude, target_lat, target_lon
            )
            surface_geopotential[station] = float(surface_z[iy, ix])

    rows: list[dict[str, object]] = []
    for path in paths:
        with open_netcdf(path) as ds:
            time_name = "valid_time" if "valid_time" in ds.variables else "time"
            total_times = len(ds.dimensions[time_name])
            count = (
                total_times
                if max_times_per_file is None
                else min(max_times_per_file, total_times)
            )
            time_values = np.asarray(ds.variables[time_name][:count], dtype=np.int64)
            bjt_times = (
                pd.to_datetime(time_values, unit="s", utc=True)
                .tz_convert("Asia/Shanghai")
                .tz_localize(None)
            )
            pressure = np.asarray(ds.variables["pressure_level"][:], dtype=float)
            latitude = np.asarray(ds.variables["latitude"][:], dtype=float)
            longitude = np.asarray(ds.variables["longitude"][:], dtype=float)
            variable_names = {
                "geopotential": "z",
                "temperature_K": "t",
                "specific_humidity_kg_kg": "q",
                "o3_mmr_kg_kg": "o3",
                "u_wind_m_s": "u",
                "v_wind_m_s": "v",
                "omega_Pa_s": "w",
            }
            arrays = {
                output_name: np.asarray(ds.variables[source_name][:count], dtype=float)
                for output_name, source_name in variable_names.items()
            }

        station_grids = {
            station: nearest_grid_point(latitude, longitude, target_lat, target_lon)
            for station, (target_lat, target_lon) in stations.items()
        }
        pressure_pa = pressure * 100.0
        for time_index, bjt_time in enumerate(bjt_times):
            for station, (target_lat, target_lon) in stations.items():
                iy, ix, grid_lat, grid_lon = station_grids[station]
                profile_geopotential = arrays["geopotential"][time_index, :, iy, ix]
                height_msl = profile_geopotential / G0
                if station in surface_geopotential:
                    interpolation_height = geopotential_to_agl(
                        profile_geopotential, surface_geopotential[station]
                    )
                    height_reference = "exact_AGL_from_ERA5_surface_geopotential"
                else:
                    interpolation_height = height_msl
                    height_reference = "approx_AGL_using_MSL_zero_missing_surface_geopotential"
                temperature = arrays["temperature_K"][time_index, :, iy, ix]
                humidity = arrays["specific_humidity_kg_kg"][time_index, :, iy, ix]
                virtual_temperature = temperature * (1.0 + 0.61 * humidity)
                density = pressure_pa / (R_D * virtual_temperature)
                o3_mmr = arrays["o3_mmr_kg_kg"][time_index, :, iy, ix]
                o3_mass = o3_mmr * density * 1.0e9
                omega = arrays["omega_Pa_s"][time_index, :, iy, ix]
                w_geo = omega_to_w_geo(omega, density)
                source_profiles = {
                    "pressure_hPa": pressure,
                    "temperature_K": temperature,
                    "specific_humidity_kg_kg": humidity,
                    "air_density_kg_m3": density,
                    "o3_mmr_kg_kg": o3_mmr,
                    "o3_mass_ug_m3": o3_mass,
                    "u_wind_m_s": arrays["u_wind_m_s"][time_index, :, iy, ix],
                    "v_wind_m_s": arrays["v_wind_m_s"][time_index, :, iy, ix],
                    "omega_Pa_s": omega,
                    "w_geometric_m_s": w_geo,
                }
                above_ground = interpolation_height >= 0.0
                profiles = {
                    name: interp_profile(
                        interpolation_height,
                        np.where(above_ground, values, np.nan),
                        TARGET_HEIGHTS_M,
                    )
                    for name, values in source_profiles.items()
                }
                fill_methods = _fill_near_surface(profiles)
                for level_index, target_height in enumerate(TARGET_HEIGHTS_M):
                    row: dict[str, object] = {
                        "source": "ERA5",
                        "source_file": Path(path).name,
                        "station": station,
                        "target_latitude": target_lat,
                        "target_longitude": target_lon,
                        "grid_latitude": grid_lat,
                        "grid_longitude": grid_lon,
                        "bjt_time": bjt_time,
                        "bjt_date": bjt_time.date().isoformat(),
                        "bjt_hour": bjt_time.hour,
                        "month": bjt_time.month,
                        "height_agl_m": target_height,
                        "height_reference": height_reference,
                        "surface_fill_method": fill_methods[level_index],
                        "source_min_height_msl_m": float(np.nanmin(height_msl)),
                        "source_max_height_msl_m": float(np.nanmax(height_msl)),
                        "surface_geopotential_height_m": (
                            surface_geopotential.get(station, 0.0) / G0
                            if station in surface_geopotential
                            else np.nan
                        ),
                    }
                    for name, profile in profiles.items():
                        row[name] = profile[level_index]
                    rows.append(row)
    return pd.DataFrame(rows)


def read_era5_blh(
    paths: list[Path],
    stations: dict[str, tuple[float, float]],
    max_times_per_file: int | None = None,
) -> pd.DataFrame:
    """Read ERA5 BLH at the nearest grid point for each station."""
    rows: list[dict[str, object]] = []
    for path in paths:
        with open_netcdf(path) as ds:
            time_name = "valid_time" if "valid_time" in ds.variables else "time"
            total_times = len(ds.dimensions[time_name])
            count = total_times if max_times_per_file is None else min(max_times_per_file, total_times)
            time_values = np.asarray(ds.variables[time_name][:count], dtype=np.int64)
            bjt_times = (
                pd.to_datetime(time_values, unit="s", utc=True)
                .tz_convert("Asia/Shanghai")
                .tz_localize(None)
            )
            latitude = np.asarray(ds.variables["latitude"][:], dtype=float)
            longitude = np.asarray(ds.variables["longitude"][:], dtype=float)
            blh = np.asarray(ds.variables["blh"][:count], dtype=float)
        for station, (target_lat, target_lon) in stations.items():
            iy, ix, grid_lat, grid_lon = nearest_grid_point(
                latitude, longitude, target_lat, target_lon
            )
            for time_index, bjt_time in enumerate(bjt_times):
                rows.append(
                    {
                        "station": station,
                        "bjt_time": bjt_time,
                        "bjt_date": bjt_time.date().isoformat(),
                        "bjt_hour": bjt_time.hour,
                        "month": bjt_time.month,
                        "grid_latitude": grid_lat,
                        "grid_longitude": grid_lon,
                        "pblh_m": blh[time_index, iy, ix],
                    }
                )
    return pd.DataFrame(rows)


def read_era5_native_levels(
    paths: list[Path],
    stations: dict[str, tuple[float, float]],
    max_times_per_file: int | None = None,
    surface_geopotential_path: Path | None = None,
) -> pd.DataFrame:
    """Return ERA5 nearest-grid values on their original pressure levels."""
    surface_values: dict[str, float] = {}
    if surface_geopotential_path is not None:
        with open_netcdf(surface_geopotential_path) as surface_ds:
            lat_s = np.asarray(surface_ds.variables["latitude"][:], dtype=float)
            lon_s = np.asarray(surface_ds.variables["longitude"][:], dtype=float)
            z_s = np.asarray(surface_ds.variables["z"][:], dtype=float).squeeze()
        for station, (target_lat, target_lon) in stations.items():
            iy, ix, _, _ = nearest_grid_point(lat_s, lon_s, target_lat, target_lon)
            surface_values[station] = float(z_s[iy, ix])

    rows: list[dict[str, object]] = []
    for path in paths:
        with open_netcdf(path) as ds:
            time_name = "valid_time" if "valid_time" in ds.variables else "time"
            total_times = len(ds.dimensions[time_name])
            count = total_times if max_times_per_file is None else min(max_times_per_file, total_times)
            bjt_times = (
                pd.to_datetime(np.asarray(ds.variables[time_name][:count], dtype=np.int64), unit="s", utc=True)
                .tz_convert("Asia/Shanghai")
                .tz_localize(None)
            )
            pressure = np.asarray(ds.variables["pressure_level"][:], dtype=float)
            latitude = np.asarray(ds.variables["latitude"][:], dtype=float)
            longitude = np.asarray(ds.variables["longitude"][:], dtype=float)
            arrays = {
                name: np.asarray(ds.variables[name][:count], dtype=float)
                for name in ("z", "t", "q", "o3", "u", "v", "w")
            }
        for station, (target_lat, target_lon) in stations.items():
            iy, ix, grid_lat, grid_lon = nearest_grid_point(
                latitude, longitude, target_lat, target_lon
            )
            for time_index, bjt_time in enumerate(bjt_times):
                temperature = arrays["t"][time_index, :, iy, ix]
                humidity = arrays["q"][time_index, :, iy, ix]
                density = pressure * 100.0 / (
                    R_D * temperature * (1.0 + 0.61 * humidity)
                )
                height_msl = arrays["z"][time_index, :, iy, ix] / G0
                height_agl = (
                    geopotential_to_agl(
                        arrays["z"][time_index, :, iy, ix], surface_values[station]
                    )
                    if station in surface_values
                    else height_msl
                )
                o3_mmr = arrays["o3"][time_index, :, iy, ix]
                omega = arrays["w"][time_index, :, iy, ix]
                for level_index, pressure_level in enumerate(pressure):
                    rows.append(
                        {
                            "source": "ERA5",
                            "source_file": path.name,
                            "station": station,
                            "target_latitude": target_lat,
                            "target_longitude": target_lon,
                            "grid_latitude": grid_lat,
                            "grid_longitude": grid_lon,
                            "bjt_time": bjt_time,
                            "bjt_date": bjt_time.date().isoformat(),
                            "bjt_hour": bjt_time.hour,
                            "month": bjt_time.month,
                            "pressure_level_index": level_index,
                            "pressure_hPa": pressure_level,
                            "height_msl_m": height_msl[level_index],
                            "height_agl_m": height_agl[level_index],
                            "temperature_K": temperature[level_index],
                            "specific_humidity_kg_kg": humidity[level_index],
                            "air_density_kg_m3": density[level_index],
                            "o3_mmr_kg_kg": o3_mmr[level_index],
                            "o3_mass_ug_m3": o3_mmr[level_index] * density[level_index] * 1.0e9,
                            "u_wind_m_s": arrays["u"][time_index, level_index, iy, ix],
                            "v_wind_m_s": arrays["v"][time_index, level_index, iy, ix],
                            "omega_Pa_s": omega[level_index],
                            "w_geometric_m_s": omega_to_w_geo(
                                omega[level_index], density[level_index]
                            ),
                            "above_model_ground": bool(height_agl[level_index] >= 0.0),
                        }
                    )
    return pd.DataFrame(rows)


def read_wrf_native_levels(
    path: Path,
    max_times: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return WRF mass-level values and EXCH_H staggered-interface values."""
    with open_netcdf(path) as ds:
        total_times = len(ds.dimensions["time"])
        count = total_times if max_times is None else min(max_times, total_times)
        bjt_times = (
            pd.to_datetime(np.asarray(ds.variables["time"][:count], dtype=np.int64), unit="s", utc=True)
            .tz_convert("Asia/Shanghai")
            .tz_localize(None)
        )
        stations = _station_names(ds.variables["station_name"])
        target_lat = np.asarray(ds.variables["target_latitude"][:], dtype=float)
        target_lon = np.asarray(ds.variables["target_longitude"][:], dtype=float)
        grid_lat = np.asarray(ds.variables["grid_latitude"][:], dtype=float)
        grid_lon = np.asarray(ds.variables["grid_longitude"][:], dtype=float)
        pblh = np.asarray(ds.variables["pblh"][:count], dtype=float)
        arrays = {
            name: np.asarray(ds.variables[name][:count], dtype=float)
            for name in (
                "height",
                "o3",
                "pressure",
                "temperature_k",
                "qv",
                "ua",
                "va",
                "wa",
                "rh",
                "o3_mass_concentration",
                "air_density",
                "exch_h",
            )
        }

    mass_rows: list[dict[str, object]] = []
    exchange_rows: list[dict[str, object]] = []
    for time_index, bjt_time in enumerate(bjt_times):
        for station_index, station in enumerate(stations):
            height = arrays["height"][time_index, station_index]
            for level_index in range(height.size):
                mass_rows.append(
                    {
                        "source": "WRF",
                        "station": station,
                        "target_latitude": target_lat[station_index],
                        "target_longitude": target_lon[station_index],
                        "grid_latitude": grid_lat[station_index],
                        "grid_longitude": grid_lon[station_index],
                        "bjt_time": bjt_time,
                        "bjt_date": bjt_time.date().isoformat(),
                        "bjt_hour": bjt_time.hour,
                        "month": bjt_time.month,
                        "mass_level_index": level_index,
                        "height_agl_m": height[level_index],
                        "pressure_hPa": arrays["pressure"][time_index, station_index, level_index],
                        "temperature_K": arrays["temperature_k"][time_index, station_index, level_index],
                        "specific_humidity_kg_kg": arrays["qv"][time_index, station_index, level_index],
                        "air_density_kg_m3": arrays["air_density"][time_index, station_index, level_index],
                        "o3_ppmv": arrays["o3"][time_index, station_index, level_index],
                        "o3_mass_ug_m3": arrays["o3_mass_concentration"][time_index, station_index, level_index],
                        "u_wind_m_s": arrays["ua"][time_index, station_index, level_index],
                        "v_wind_m_s": arrays["va"][time_index, station_index, level_index],
                        "w_geometric_m_s": arrays["wa"][time_index, station_index, level_index],
                        "relative_humidity_pct": arrays["rh"][time_index, station_index, level_index],
                        "pblh_m": pblh[time_index, station_index],
                    }
                )
            interface_height = interface_heights_from_mass_levels(height)
            for level_index, interface_z in enumerate(interface_height):
                exchange_rows.append(
                    {
                        "source": "WRF",
                        "station": station,
                        "bjt_time": bjt_time,
                        "bjt_date": bjt_time.date().isoformat(),
                        "bjt_hour": bjt_time.hour,
                        "month": bjt_time.month,
                        "staggered_level_index": level_index,
                        "estimated_interface_height_agl_m": interface_z,
                        "exch_h_source_value": arrays["exch_h"][
                            time_index, station_index, level_index
                        ],
                        "source_unit_metadata": "-",
                    }
                )
    return pd.DataFrame(mass_rows), pd.DataFrame(exchange_rows)


def summarize_profiles(
    frame: pd.DataFrame,
    value_columns: list[str],
) -> dict[str, pd.DataFrame]:
    """Build daytime daily, monthly, and hour-of-day profile summaries."""
    valid_columns = [column for column in value_columns if column in frame.columns]
    return {
        "daily": frame.groupby(
            ["station", "bjt_date", "height_agl_m"], as_index=False
        )[valid_columns].mean(),
        "monthly": frame.groupby(
            ["station", "month", "height_agl_m"], as_index=False
        )[valid_columns].mean(),
        "hourly_climatology": frame.groupby(
            ["station", "bjt_hour", "height_agl_m"], as_index=False
        )[valid_columns].mean(),
    }


def build_comparison(era5: pd.DataFrame, wrf: pd.DataFrame) -> pd.DataFrame:
    """Pair ERA5 and WRF O3/w values on station, Beijing time, and height."""
    keys = ["station", "bjt_time", "height_agl_m"]
    selected = ["o3_mass_ug_m3", "w_geometric_m_s"]
    paired = era5[keys + selected].merge(
        wrf[keys + selected], on=keys, how="inner", suffixes=("_era5", "_wrf")
    )
    paired = paired.dropna(
        subset=[
            "o3_mass_ug_m3_era5",
            "o3_mass_ug_m3_wrf",
            "w_geometric_m_s_era5",
            "w_geometric_m_s_wrf",
        ]
    ).copy()
    paired["bjt_date"] = paired["bjt_time"].dt.date.astype(str)
    paired["bjt_hour"] = paired["bjt_time"].dt.hour
    paired["month"] = paired["bjt_time"].dt.month
    paired["o3_diff_era5_minus_wrf_ug_m3"] = (
        paired["o3_mass_ug_m3_era5"] - paired["o3_mass_ug_m3_wrf"]
    )
    paired["w_diff_era5_minus_wrf_m_s"] = (
        paired["w_geometric_m_s_era5"] - paired["w_geometric_m_s_wrf"]
    )
    paired["o3_relative_diff_pct"] = np.where(
        paired["o3_mass_ug_m3_wrf"].abs() > 1.0e-12,
        paired["o3_diff_era5_minus_wrf_ug_m3"]
        / paired["o3_mass_ug_m3_wrf"]
        * 100.0,
        np.nan,
    )
    return paired


def grouped_comparison_metrics(paired: pd.DataFrame) -> pd.DataFrame:
    """Create long-form O3 and vertical-velocity metrics at useful groupings."""
    variables = {
        "O3": ("o3_mass_ug_m3_era5", "o3_mass_ug_m3_wrf", "ug m-3"),
        "w_geometric": ("w_geometric_m_s_era5", "w_geometric_m_s_wrf", "m s-1"),
    }
    groupings: list[tuple[str, list[str]]] = [
        ("overall", []),
        ("station", ["station"]),
        ("station_month", ["station", "month"]),
        ("station_hour", ["station", "bjt_hour"]),
        ("station_height", ["station", "height_agl_m"]),
    ]
    rows: list[dict[str, object]] = []
    for grouping_name, keys in groupings:
        groups = [((), paired)] if not keys else paired.groupby(keys, dropna=False)
        for group_key, group in groups:
            key_values = group_key if isinstance(group_key, tuple) else (group_key,)
            labels = dict(zip(keys, key_values))
            for variable, (era5_col, wrf_col, unit) in variables.items():
                metrics = comparison_metrics(group, era5_col, wrf_col)
                rows.append(
                    {
                        "grouping": grouping_name,
                        **labels,
                        "variable": variable,
                        "unit": unit,
                        **metrics,
                        "era5_mean": float(group[era5_col].mean()),
                        "wrf_mean": float(group[wrf_col].mean()),
                    }
                )
    return pd.DataFrame(rows)


def file_inventory(paths: list[Path], dataset: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "modified_time": pd.Timestamp(path.stat().st_mtime, unit="s"),
            }
            for path in paths
        ]
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")


def write_dataset_outputs(
    directory: Path,
    prefix: str,
    hourly: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
    inventory: pd.DataFrame,
    notes: pd.DataFrame,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(directory / f"{prefix}_逐小时0-4km插值.csv", hourly)
    write_csv(directory / f"{prefix}_日均高度廓线.csv", summaries["daily"])
    write_csv(directory / f"{prefix}_月均高度廓线.csv", summaries["monthly"])
    write_csv(directory / f"{prefix}_08-16时综合小时平均.csv", summaries["hourly_climatology"])
    write_csv(directory / f"{prefix}_文件清单.csv", inventory)
    with pd.ExcelWriter(directory / f"{prefix}_统计摘要.xlsx", engine="openpyxl") as writer:
        notes.to_excel(writer, sheet_name="变量与方法说明", index=False)
        inventory.to_excel(writer, sheet_name="文件清单", index=False)
        summaries["daily"].to_excel(writer, sheet_name="日均高度廓线", index=False)
        summaries["monthly"].to_excel(writer, sheet_name="月均高度廓线", index=False)
        summaries["hourly_climatology"].to_excel(
            writer, sheet_name="08-16时小时平均", index=False
        )


def plot_time_height_comparison(
    comparison_hourly: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    output_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "O3": {
            "era5": "o3_mass_ug_m3_era5",
            "wrf": "o3_mass_ug_m3_wrf",
            "diff": "o3_diff_era5_minus_wrf_ug_m3",
            "unit": r"$\mu$g m$^{-3}$",
        },
        "Vertical_velocity": {
            "era5": "w_geometric_m_s_era5",
            "wrf": "w_geometric_m_s_wrf",
            "diff": "w_diff_era5_minus_wrf_m_s",
            "unit": r"m s$^{-1}$ (positive upward)",
        },
    }
    for station in sorted(comparison_hourly["station"].unique()):
        station_frame = comparison_hourly[comparison_hourly["station"] == station]
        for label, spec in specs.items():
            matrices = []
            for column in (spec["era5"], spec["wrf"], spec["diff"]):
                matrices.append(
                    station_frame.pivot(
                        index="height_agl_m", columns="bjt_hour", values=column
                    ).sort_index()
                )
            absolute_values = np.concatenate(
                [matrices[0].to_numpy().ravel(), matrices[1].to_numpy().ravel()]
            )
            absolute_values = absolute_values[np.isfinite(absolute_values)]
            if label == "O3":
                vmin = max(0.0, float(np.nanpercentile(absolute_values, 1)))
                vmax = float(np.nanpercentile(absolute_values, 99))
                cmap = "viridis"
                norm = None
            else:
                limit = float(np.nanpercentile(np.abs(absolute_values), 99))
                vmin, vmax = -limit, limit
                cmap = "RdBu_r"
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            diff_values = matrices[2].to_numpy()
            diff_limit = float(np.nanpercentile(np.abs(diff_values), 99))
            if not np.isfinite(diff_limit) or diff_limit == 0:
                diff_limit = 1.0
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
            for index, (axis, matrix, title) in enumerate(
                zip(axes, matrices, ("ERA5", "WRF", "ERA5 - WRF"))
            ):
                if index < 2:
                    image = axis.pcolormesh(
                        matrix.columns,
                        matrix.index / 1000.0,
                        matrix.to_numpy(),
                        shading="auto",
                        cmap=cmap,
                        vmin=None if norm is not None else vmin,
                        vmax=None if norm is not None else vmax,
                        norm=norm,
                    )
                else:
                    image = axis.pcolormesh(
                        matrix.columns,
                        matrix.index / 1000.0,
                        matrix.to_numpy(),
                        shading="auto",
                        cmap="RdBu_r",
                        norm=TwoSlopeNorm(
                            vmin=-diff_limit, vcenter=0.0, vmax=diff_limit
                        ),
                    )
                axis.set_title(title)
                axis.set_xlabel("Beijing time (hour)")
                axis.set_xticks(range(8, 17))
                fig.colorbar(image, ax=axis, label=spec["unit"])
            axes[0].set_ylabel("Height AGL (km)")
            fig.suptitle(f"{station}: {label.replace('_', ' ')} hourly climatology")
            fig.tight_layout()
            fig.savefig(output_dir / f"{station}_{label}_08-16时高度对比.png", dpi=220)
            plt.close(fig)


def plot_mean_profiles_and_scatter(paired: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharey=True)
    for row, station in enumerate(("HF", "CF")):
        station_frame = paired[paired["station"] == station]
        mean_profile = station_frame.groupby("height_agl_m", as_index=False)[
            [
                "o3_mass_ug_m3_era5",
                "o3_mass_ug_m3_wrf",
                "w_geometric_m_s_era5",
                "w_geometric_m_s_wrf",
            ]
        ].mean()
        height_km = mean_profile["height_agl_m"] / 1000.0
        axes[row, 0].plot(mean_profile["o3_mass_ug_m3_era5"], height_km, label="ERA5")
        axes[row, 0].plot(mean_profile["o3_mass_ug_m3_wrf"], height_km, label="WRF")
        axes[row, 1].plot(mean_profile["w_geometric_m_s_era5"], height_km, label="ERA5")
        axes[row, 1].plot(mean_profile["w_geometric_m_s_wrf"], height_km, label="WRF")
        axes[row, 1].axvline(0.0, color="0.5", linestyle="--", linewidth=1)
        axes[row, 0].set_ylabel(f"{station} height AGL (km)")
        axes[row, 0].grid(alpha=0.25)
        axes[row, 1].grid(alpha=0.25)
    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[1, 0].set_xlabel(r"O3 ($\mu$g m$^{-3}$)")
    axes[1, 1].set_xlabel(r"Geometric w (m s$^{-1}$, positive upward)")
    axes[0, 0].set_title("O3 mean profile")
    axes[0, 1].set_title("Vertical velocity mean profile")
    fig.tight_layout()
    fig.savefig(output_dir / "HF_CF_O3与垂直速度平均廓线对比.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    specs = [
        ("o3_mass_ug_m3_era5", "o3_mass_ug_m3_wrf", "O3", r"$\mu$g m$^{-3}$"),
        ("w_geometric_m_s_era5", "w_geometric_m_s_wrf", "Vertical velocity", "m s-1"),
    ]
    for row, station in enumerate(("HF", "CF")):
        station_frame = paired[paired["station"] == station]
        for column, (era5_col, wrf_col, label, unit) in enumerate(specs):
            sample = station_frame[[era5_col, wrf_col]].dropna()
            if len(sample) > 20000:
                sample = sample.sample(20000, random_state=42)
            axis = axes[row, column]
            axis.scatter(sample[wrf_col], sample[era5_col], s=4, alpha=0.15)
            low = float(min(sample.min()))
            high = float(max(sample.max()))
            axis.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
            r = sample[era5_col].corr(sample[wrf_col])
            axis.set_title(f"{station} {label}, r={r:.3f}")
            axis.set_xlabel(f"WRF ({unit})")
            axis.set_ylabel(f"ERA5 ({unit})")
            axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "HF_CF_O3与垂直速度一致性散点图.png", dpi=220)
    plt.close(fig)


def main() -> None:
    base = Path("E:/research/\u5782\u76f4\u4ea4\u6362/\u5408\u80a5\u53cc\u7ad9_ERA5\u4e0eWRF_2023")
    pressure_dir = base / "ERA5_pressure_levels_600_1000"
    blh_dir = base / "ERA5_single_levels_BLH"
    wrf_path = base / "WRF\u7ad9\u70b9\u62bd\u53d6\u7ed3\u679c" / "wrf_hf_cf_allvars_202304_202310.nc"
    surface_geopotential_path = (
        base
        / "ERA5_single_levels_surface"
        / "era5_surface_geopotential_hefei_hf_cf_20230401_00.nc"
    )
    output_root = base / "ERA5_WRF\u7edf\u4e00\u63d2\u503c\u4e0e\u5bf9\u6bd4\u7ed3\u679c"
    era5_dir = output_root / "ERA5\u6570\u636e\u8868"
    wrf_dir = output_root / "WRF\u6570\u636e\u8868"
    comparison_dir = output_root / "ERA5_WRF\u5bf9\u6bd4\u8868"
    figure_dir = output_root / "\u7efc\u5408\u5206\u6790\u56fe"
    stations = {"HF": (31.78, 117.18), "CF": (32.21, 117.18)}

    pressure_files = [
        path
        for month in range(4, 11)
        for path in sorted(pressure_dir.glob(f"*2023{month:02d}_*.nc"))
    ]
    blh_files = [
        path
        for month in range(4, 11)
        for path in sorted(blh_dir.glob(f"*2023{month:02d}_*.nc"))
    ]
    if len(pressure_files) != 21 or len(blh_files) != 21:
        raise RuntimeError(
            f"Expected 21 pressure and 21 BLH files for Apr-Oct; got "
            f"{len(pressure_files)} and {len(blh_files)}."
        )
    if not surface_geopotential_path.exists():
        raise FileNotFoundError(
            "ERA5 surface geopotential is required for exact AGL interpolation: "
            f"{surface_geopotential_path}"
        )

    print("Processing ERA5 pressure-level files...", flush=True)
    era5 = process_era5_files(
        pressure_files,
        stations,
        surface_geopotential_path=surface_geopotential_path,
    )
    era5_blh = read_era5_blh(blh_files, stations)
    era5 = era5.merge(
        era5_blh[["station", "bjt_time", "pblh_m"]],
        on=["station", "bjt_time"],
        how="left",
        validate="many_to_one",
    )
    era5_values = [
        "pressure_hPa",
        "temperature_K",
        "specific_humidity_kg_kg",
        "air_density_kg_m3",
        "o3_mmr_kg_kg",
        "o3_mass_ug_m3",
        "u_wind_m_s",
        "v_wind_m_s",
        "omega_Pa_s",
        "w_geometric_m_s",
        "pblh_m",
    ]
    era5_summaries = summarize_profiles(era5, era5_values)
    print("Exporting ERA5 original pressure levels...", flush=True)
    era5_native = read_era5_native_levels(
        pressure_files,
        stations,
        surface_geopotential_path=surface_geopotential_path,
    )
    era5_notes = pd.DataFrame(
        [
            {"item": "time", "description": "UTC 00:00-08:00 converted to Beijing time 08:00-16:00."},
            {"item": "grid", "description": "HF and CF use their nearest 0.25-degree ERA5 grid point."},
            {"item": "height_agl_m", "description": "Exact ERA5 model-grid AGL: (pressure-level geopotential - surface geopotential)/g."},
            {"item": "w_geometric_m_s", "description": "-omega/(rho*g), positive upward."},
            {"item": "surface_fill_method", "description": "Missing 0/100 m values copied from the nearest valid low target layer; method retained per row."},
        ]
    )
    write_dataset_outputs(
        era5_dir,
        "ERA5_HF_CF_202304-202310",
        era5,
        era5_summaries,
        pd.concat(
            [
                file_inventory(pressure_files, "ERA5 pressure"),
                file_inventory(blh_files, "ERA5 BLH"),
                file_inventory([surface_geopotential_path], "ERA5 surface geopotential"),
            ],
            ignore_index=True,
        ),
        era5_notes,
    )
    write_csv(era5_dir / "ERA5_HF_CF_202304-202310_原始压力层最近格点.csv", era5_native)

    print("Processing WRF station profiles...", flush=True)
    wrf = process_wrf_file(wrf_path)
    wrf_values = [
        "pressure_hPa",
        "temperature_K",
        "specific_humidity_kg_kg",
        "air_density_kg_m3",
        "o3_ppmv",
        "o3_mass_ug_m3",
        "u_wind_m_s",
        "v_wind_m_s",
        "w_geometric_m_s",
        "relative_humidity_pct",
        "exch_h_source_value",
        "pblh_m",
    ]
    wrf_summaries = summarize_profiles(wrf, wrf_values)
    print("Exporting WRF original mass and staggered levels...", flush=True)
    wrf_native, wrf_exchange_native = read_wrf_native_levels(wrf_path)
    wrf_notes = pd.DataFrame(
        [
            {"item": "time", "description": "UTC 00:00-08:00 converted to Beijing time 08:00-16:00; 1926 complete records."},
            {"item": "height_agl_m", "description": "WRF mass-level height is treated as AGL based on 12 m lowest centers and 0 m EXCH_H interface."},
            {"item": "w_geometric_m_s", "description": "WRF wa, positive upward."},
            {"item": "exch_h_source_value", "description": "45 staggered-interface source values mapped using estimated midpoint interface heights. Source unit metadata is '-', so no physical unit is asserted."},
            {"item": "surface_fill_method", "description": "Missing state at 0 m copied from 100 m; EXCH_H keeps its physical 0 m boundary value."},
        ]
    )
    write_dataset_outputs(
        wrf_dir,
        "WRF_HF_CF_202304-202310",
        wrf,
        wrf_summaries,
        file_inventory([wrf_path], "WRF extracted station profiles"),
        wrf_notes,
    )
    write_csv(wrf_dir / "WRF_HF_CF_202304-202310_原始质量层.csv", wrf_native)
    write_csv(wrf_dir / "WRF_HF_CF_202304-202310_EXCH_H原始交错层.csv", wrf_exchange_native)

    print("Building paired ERA5-WRF comparison...", flush=True)
    paired = build_comparison(era5, wrf)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    write_csv(comparison_dir / "ERA5_WRF_O3与垂直速度逐小时配对.csv", paired)
    comparison_values = [
        "o3_mass_ug_m3_era5",
        "o3_mass_ug_m3_wrf",
        "o3_diff_era5_minus_wrf_ug_m3",
        "w_geometric_m_s_era5",
        "w_geometric_m_s_wrf",
        "w_diff_era5_minus_wrf_m_s",
    ]
    comparison_summaries = summarize_profiles(paired, comparison_values)
    for name, frame in comparison_summaries.items():
        write_csv(comparison_dir / f"ERA5_WRF_{name}.csv", frame)
    metrics = grouped_comparison_metrics(paired)
    write_csv(comparison_dir / "ERA5_WRF_O3与垂直速度误差统计.csv", metrics)
    with pd.ExcelWriter(
        comparison_dir / "ERA5_WRF_O3与垂直速度对比摘要.xlsx", engine="openpyxl"
    ) as writer:
        pd.DataFrame(
            [
                {"item": "difference", "description": "All differences are ERA5 minus WRF."},
                {"item": "vertical_velocity", "description": "Both datasets use geometric vertical velocity positive upward."},
                {"item": "height", "description": "Both datasets use AGL. ERA5 subtracts the matching grid-cell surface geopotential; WRF uses native AGL height."},
            ]
        ).to_excel(writer, sheet_name="说明", index=False)
        metrics.to_excel(writer, sheet_name="误差统计", index=False)
        comparison_summaries["daily"].to_excel(writer, sheet_name="日均", index=False)
        comparison_summaries["monthly"].to_excel(writer, sheet_name="月均", index=False)
        comparison_summaries["hourly_climatology"].to_excel(
            writer, sheet_name="08-16时小时平均", index=False
        )

    plot_time_height_comparison(comparison_summaries["hourly_climatology"], figure_dir)
    plot_mean_profiles_and_scatter(paired, figure_dir)
    print(f"Done. Output: {output_root}", flush=True)
    print(f"ERA5 rows={len(era5)}, WRF rows={len(wrf)}, paired rows={len(paired)}", flush=True)


if __name__ == "__main__":
    main()
