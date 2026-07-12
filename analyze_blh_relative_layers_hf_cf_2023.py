from __future__ import annotations

import numpy as np
import pandas as pd


LAYER_ORDER = ["within_pbl", "pbl_top", "above_pbl"]
METRIC_NAMES = {
    "o3_mass_ug_m3": "o3",
    "w_geometric_m_s": "w",
    "fadv_ug_m2_s": "fadv",
    "tfh_ug_m2_s": "tfh",
    "fturb_ug_m2_s": "fturb",
    "ftotal_ug_m2_s": "ftotal",
    "pblh_m": "pblh",
}


def assign_blh_layer(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    height = result["height_agl_m"].to_numpy(dtype=float)
    pblh = result["pblh_m"].to_numpy(dtype=float)
    result["blh_layer"] = np.select(
        [height <= pblh, height <= pblh + 500.0],
        ["within_pbl", "pbl_top"],
        default="above_pbl",
    )
    return result


def calculate_diagnostics(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    source = source.upper()
    if source not in {"ERA5", "WRF"}:
        raise ValueError("source must be ERA5 or WRF")
    result = frame.copy()
    result["source"] = source
    result["fadv_ug_m2_s"] = (
        result["o3_mass_ug_m3"] * result["w_geometric_m_s"]
    )
    result["tfh_ug_m2_s"] = result["o3_mass_ug_m3"] * np.hypot(
        result["u_wind_m_s"], result["v_wind_m_s"]
    )
    if source == "ERA5":
        return result
    if "exch_h_source_value" not in result.columns:
        raise KeyError("WRF diagnostics require exch_h_source_value")

    result["o3_gradient_ug_m4"] = np.nan
    for _, group in result.groupby(["station", "bjt_time"], sort=False):
        ordered = group.sort_values("height_agl_m")
        if len(ordered) < 2:
            continue
        gradient = np.gradient(
            ordered["o3_mass_ug_m3"].to_numpy(dtype=float),
            ordered["height_agl_m"].to_numpy(dtype=float),
        )
        result.loc[ordered.index, "o3_gradient_ug_m4"] = gradient
    result["fturb_ug_m2_s"] = -(
        result["exch_h_source_value"] * result["o3_gradient_ug_m4"]
    )
    result["ftotal_ug_m2_s"] = (
        result["fadv_ug_m2_s"] + result["fturb_ug_m2_s"]
    )
    return result


def hourly_layer_means(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in METRIC_NAMES if column in frame.columns]
    keys = ["source", "station", "bjt_time", "bjt_hour", "blh_layer"]
    result = (
        frame.groupby(keys, as_index=False, observed=True)[metrics]
        .mean()
        .sort_values(["source", "station", "bjt_time", "blh_layer"])
        .reset_index(drop=True)
    )
    return result


def summarize_layers(hourly: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in METRIC_NAMES if column in hourly.columns]
    records: list[dict[str, object]] = []
    for (source, station, layer), group in hourly.groupby(
        ["source", "station", "blh_layer"], sort=False, observed=True
    ):
        record: dict[str, object] = {
            "source": source,
            "station": station,
            "blh_layer": layer,
            "valid_hours": int(len(group)),
        }
        for column in metrics:
            prefix = METRIC_NAMES[column]
            values = group[column].dropna().astype(float)
            count = int(values.count())
            std = float(values.std(ddof=1)) if count > 1 else 0.0
            record[f"{prefix}_mean"] = float(values.mean()) if count else np.nan
            record[f"{prefix}_std"] = std if count else np.nan
            record[f"{prefix}_sem"] = std / np.sqrt(count) if count else np.nan
            record[f"{prefix}_median"] = (
                float(values.median()) if count else np.nan
            )
            record[f"{prefix}_q25"] = (
                float(values.quantile(0.25)) if count else np.nan
            )
            record[f"{prefix}_q75"] = (
                float(values.quantile(0.75)) if count else np.nan
            )
        records.append(record)
    result = pd.DataFrame.from_records(records)
    result["blh_layer"] = pd.Categorical(
        result["blh_layer"], categories=LAYER_ORDER, ordered=True
    )
    return result.sort_values(["source", "station", "blh_layer"]).reset_index(
        drop=True
    )
