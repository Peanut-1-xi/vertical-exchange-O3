from __future__ import annotations

import numpy as np
import pandas as pd

from plot_paper_fig1_fig3_o3_hf_cf_2023 import (
    aggregate_daytime_profile,
    aggregate_hour_height,
    calculate_horizontal_flux,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": ["HF", "HF"],
            "bjt_time": pd.to_datetime(
                ["2023-04-01 08:00", "2023-04-02 08:00"]
            ),
            "bjt_hour": [8, 8],
            "height_agl_m": [200.0, 200.0],
            "o3_mass_ug_m3": [10.0, 30.0],
            "u_wind_m_s": [2.0, -1.0],
            "v_wind_m_s": [1.0, 3.0],
            "pblh_m": [500.0, 700.0],
        }
    )


def test_horizontal_flux_is_multiplied_before_averaging() -> None:
    flux = calculate_horizontal_flux(sample_frame())

    np.testing.assert_allclose(flux["tfu_ug_m2_s"], [20.0, -30.0])
    np.testing.assert_allclose(flux["tfv_ug_m2_s"], [10.0, 90.0])

    hourly = aggregate_hour_height(flux)
    assert hourly.loc[0, "tfu_mean"] == -5.0
    assert hourly.loc[0, "tfv_mean"] == 50.0
    assert hourly.loc[0, "tfh_mean"] == np.hypot(-5.0, 50.0)
    assert (
        hourly.loc[0, "tfu_mean"]
        != flux["o3_mass_ug_m3"].mean() * flux["u_wind_m_s"].mean()
    )


def test_daytime_profile_averages_vector_components_before_magnitude() -> None:
    profile = aggregate_daytime_profile(calculate_horizontal_flux(sample_frame()))

    assert profile.loc[0, "tfu_mean"] == -5.0
    assert profile.loc[0, "tfv_mean"] == 50.0
    assert profile.loc[0, "tfh_mean"] == np.hypot(-5.0, 50.0)
