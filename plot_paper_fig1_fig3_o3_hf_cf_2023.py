from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_horizontal_flux(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["tfu_ug_m2_s"] = result["o3_mass_ug_m3"] * result["u_wind_m_s"]
    result["tfv_ug_m2_s"] = result["o3_mass_ug_m3"] * result["v_wind_m_s"]
    return result


def aggregate_hour_height(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(
            ["station", "bjt_hour", "height_agl_m"], as_index=False
        )
        .agg(
            o3_mean=("o3_mass_ug_m3", "mean"),
            tfu_mean=("tfu_ug_m2_s", "mean"),
            tfv_mean=("tfv_ug_m2_s", "mean"),
            pblh_mean=("pblh_m", "mean"),
        )
        .sort_values(["station", "height_agl_m", "bjt_hour"])
        .reset_index(drop=True)
    )
    grouped["tfh_mean"] = np.hypot(
        grouped["tfu_mean"], grouped["tfv_mean"]
    )
    return grouped


def aggregate_daytime_profile(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(["station", "height_agl_m"], as_index=False)
        .agg(
            tfu_mean=("tfu_ug_m2_s", "mean"),
            tfv_mean=("tfv_ug_m2_s", "mean"),
        )
        .sort_values(["station", "height_agl_m"])
        .reset_index(drop=True)
    )
    grouped["tfh_mean"] = np.hypot(
        grouped["tfu_mean"], grouped["tfv_mean"]
    )
    return grouped
