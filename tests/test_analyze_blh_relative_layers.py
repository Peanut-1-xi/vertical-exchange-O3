from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from analyze_blh_relative_layers_hf_cf_2023 import (
    assign_blh_layer,
    calculate_diagnostics,
    hourly_layer_means,
    plot_layer_comparison,
    plot_wrf_flux_decomposition,
    summarize_layers,
)


def test_assign_blh_layer_respects_both_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "height_agl_m": [0.0, 500.0, 500.1, 1000.0, 1000.1, 3000.0],
            "pblh_m": [500.0] * 6,
        }
    )

    result = assign_blh_layer(frame)

    assert result["blh_layer"].tolist() == [
        "within_pbl",
        "within_pbl",
        "pbl_top",
        "pbl_top",
        "above_pbl",
        "above_pbl",
    ]


def test_wrf_diagnostics_use_pointwise_flux_formulas() -> None:
    frame = pd.DataFrame(
        {
            "source": ["WRF"] * 3,
            "station": ["HF"] * 3,
            "bjt_time": pd.to_datetime(["2023-04-01 08:00"] * 3),
            "bjt_hour": [8] * 3,
            "height_agl_m": [0.0, 100.0, 200.0],
            "pblh_m": [500.0] * 3,
            "o3_mass_ug_m3": [10.0, 20.0, 40.0],
            "w_geometric_m_s": [1.0, -1.0, 2.0],
            "u_wind_m_s": [3.0, 4.0, 0.0],
            "v_wind_m_s": [4.0, 3.0, 5.0],
            "exch_h_source_value": [2.0, 2.0, 2.0],
        }
    )

    result = calculate_diagnostics(frame, "WRF")

    expected_gradient = np.gradient([10.0, 20.0, 40.0], [0.0, 100.0, 200.0])
    np.testing.assert_allclose(result["fadv_ug_m2_s"], [10.0, -20.0, 80.0])
    np.testing.assert_allclose(result["tfh_ug_m2_s"], [50.0, 100.0, 200.0])
    np.testing.assert_allclose(result["o3_gradient_ug_m4"], expected_gradient)
    np.testing.assert_allclose(result["fturb_ug_m2_s"], -2.0 * expected_gradient)
    np.testing.assert_allclose(
        result["ftotal_ug_m2_s"],
        result["fadv_ug_m2_s"] + result["fturb_ug_m2_s"],
    )


def test_two_stage_average_gives_each_hour_equal_weight() -> None:
    frame = pd.DataFrame(
        {
            "source": ["ERA5"] * 4,
            "station": ["HF"] * 4,
            "bjt_time": pd.to_datetime(
                [
                    "2023-04-01 08:00",
                    "2023-04-01 08:00",
                    "2023-04-01 08:00",
                    "2023-04-01 09:00",
                ]
            ),
            "bjt_hour": [8, 8, 8, 9],
            "height_agl_m": [0.0, 100.0, 200.0, 0.0],
            "pblh_m": [500.0] * 4,
            "o3_mass_ug_m3": [10.0] * 4,
            "w_geometric_m_s": [0.0, 0.0, 0.0, 10.0],
            "u_wind_m_s": [0.0] * 4,
            "v_wind_m_s": [0.0] * 4,
        }
    )
    diagnosed = assign_blh_layer(calculate_diagnostics(frame, "ERA5"))

    hourly = hourly_layer_means(diagnosed)
    summary = summarize_layers(hourly)

    assert hourly["fadv_ug_m2_s"].tolist() == [0.0, 100.0]
    assert summary.loc[0, "fadv_mean"] == 50.0
    assert summary.loc[0, "valid_hours"] == 2


def synthetic_summary() -> pd.DataFrame:
    rows = []
    for source_index, source in enumerate(("ERA5", "WRF")):
        for station_index, station in enumerate(("HF", "CF")):
            for layer_index, layer in enumerate(
                ("within_pbl", "pbl_top", "above_pbl")
            ):
                base = 1.0 + source_index + station_index + layer_index
                rows.append(
                    {
                        "source": source,
                        "station": station,
                        "blh_layer": layer,
                        "valid_hours": 10,
                        "o3_mean": 70.0 + 5.0 * base,
                        "o3_sem": 1.0,
                        "w_mean": (base - 3.0) * 0.001,
                        "w_sem": 0.0002,
                        "fadv_mean": base - 3.0,
                        "fadv_sem": 0.2,
                        "tfh_mean": 50.0 * base,
                        "tfh_sem": 3.0,
                        "fturb_mean": 0.5 * (3.0 - base),
                        "fturb_sem": 0.1,
                        "ftotal_mean": 0.5 * (base - 3.0),
                        "ftotal_sem": 0.15,
                    }
                )
    return pd.DataFrame(rows)


def test_summary_plot_functions_create_nonblank_pngs(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.png"
    decomposition = tmp_path / "decomposition.png"

    plot_layer_comparison(synthetic_summary(), comparison)
    plot_wrf_flux_decomposition(synthetic_summary(), decomposition)

    for path in (comparison, decomposition):
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
            assert image.width >= 1800
            assert image.height >= 1000
            assert pixels.std() > 5.0
