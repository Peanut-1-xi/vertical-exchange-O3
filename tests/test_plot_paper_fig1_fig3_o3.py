from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

import plot_paper_fig1_fig3_o3_hf_cf_2023 as figure_module
from plot_paper_fig1_fig3_o3_hf_cf_2023 import (
    aggregate_daytime_profile,
    aggregate_hour_height,
    calculate_horizontal_flux,
    plot_figure1_o3,
    plot_figure3_o3,
    validate_frame,
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


def plot_frame() -> pd.DataFrame:
    rows = []
    for station, station_offset in (("HF", 0.0), ("CF", 15.0)):
        for day in (1, 2):
            for hour in (8, 9):
                for height in (0.0, 500.0, 1000.0):
                    rows.append(
                        {
                            "station": station,
                            "bjt_time": pd.Timestamp(2023, 4, day, hour),
                            "bjt_hour": hour,
                            "height_agl_m": height,
                            "o3_mass_ug_m3": (
                                60.0 + station_offset + hour + height / 100.0
                            ),
                            "u_wind_m_s": 1.0 + height / 1000.0,
                            "v_wind_m_s": -0.5 + (hour - 8) * 0.5,
                            "pblh_m": 400.0 + (hour - 8) * 150.0,
                        }
                    )
    return pd.DataFrame(rows)


def test_validate_frame_rejects_missing_required_wind() -> None:
    with pytest.raises(KeyError, match="v_wind_m_s"):
        validate_frame(sample_frame().drop(columns="v_wind_m_s"), "ERA5")


def test_plot_functions_create_nonblank_pngs(tmp_path: Path) -> None:
    frame = calculate_horizontal_flux(plot_frame())
    fig1 = tmp_path / "fig1.png"
    fig3 = tmp_path / "fig3.png"

    plot_figure1_o3(frame, "ERA5", fig1)
    plot_figure3_o3(frame, "ERA5", fig3)

    for path in (fig1, fig3):
        with Image.open(path) as image:
            assert image.width >= 1800
            assert image.height >= 1000
            assert np.asarray(image.convert("RGB")).std() > 5.0


def test_main_generates_exactly_four_source_consistent_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    era5_file = tmp_path / "era5.csv"
    wrf_file = tmp_path / "wrf.csv"
    output_dir = tmp_path / "output"
    plot_frame().assign(source="ERA5").to_csv(era5_file, index=False)
    plot_frame().assign(source="WRF").to_csv(wrf_file, index=False)
    monkeypatch.setattr(figure_module, "ERA5_FILE", era5_file)
    monkeypatch.setattr(figure_module, "WRF_FILE", wrf_file)
    monkeypatch.setattr(figure_module, "OUTPUT_DIR", output_dir)

    figure_module.main()

    assert sorted(path.name for path in output_dir.glob("*.png")) == [
        "ERA5_Fig1_O3_HF_CF时间高度分布.png",
        "ERA5_Fig3_O3水平输送通量_HF_CF.png",
        "WRF_Fig1_O3_HF_CF时间高度分布.png",
        "WRF_Fig3_O3水平输送通量_HF_CF.png",
    ]
