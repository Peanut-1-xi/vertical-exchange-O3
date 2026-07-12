import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from process_era5_wrf_2023_comparison import (
    _fill_near_surface,
    build_comparison,
    comparison_metrics,
    interface_heights_from_mass_levels,
    interp_profile,
    geopotential_to_agl,
    nearest_grid_point,
    omega_to_w_geo,
    process_era5_files,
    read_era5_native_levels,
    read_era5_blh,
    process_wrf_file,
    read_wrf_native_levels,
    summarize_profiles,
    wrf_o3_ppmv_to_ug_m3,
)


class ComparisonHelpersTest(unittest.TestCase):
    def test_interp_profile_is_linear_inside_source_range_and_nan_outside(self):
        source_height = np.array([100.0, 300.0, 500.0])
        source_value = np.array([10.0, 30.0, 50.0])
        target_height = np.array([0.0, 100.0, 200.0, 500.0, 600.0])

        result = interp_profile(source_height, source_value, target_height)

        np.testing.assert_allclose(result[1:4], [10.0, 20.0, 50.0])
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[4]))

    def test_interp_profile_ignores_missing_values_and_duplicate_heights(self):
        source_height = np.array([100.0, 200.0, 200.0, 300.0, np.nan])
        source_value = np.array([1.0, np.nan, 2.0, 3.0, 99.0])

        result = interp_profile(source_height, source_value, np.array([200.0, 250.0]))

        np.testing.assert_allclose(result, [2.0, 2.5])

    def test_positive_omega_becomes_negative_upward_geometric_velocity(self):
        omega = np.array([0.2, -0.2])
        density = np.array([1.0, 1.0])

        result = omega_to_w_geo(omega, density)

        np.testing.assert_allclose(result, [-0.2 / 9.80665, 0.2 / 9.80665])

    def test_wrf_o3_ppmv_to_mass_concentration_uses_ideal_gas_law(self):
        result = wrf_o3_ppmv_to_ug_m3(
            np.array([0.1]),
            np.array([1000.0]),
            np.array([298.15]),
        )

        expected = 0.1 * 100000.0 * 48.0 / (8.314462618 * 298.15)
        np.testing.assert_allclose(result, [expected], rtol=1e-12)

    def test_nearest_grid_point_returns_coordinate_and_indices(self):
        latitudes = np.array([32.25, 32.00, 31.75])
        longitudes = np.array([117.00, 117.25, 117.50])

        result = nearest_grid_point(latitudes, longitudes, 31.78, 117.18)

        self.assertEqual(result, (2, 1, 31.75, 117.25))

    def test_comparison_metrics_uses_era5_minus_wrf_and_pairwise_finite_rows(self):
        frame = pd.DataFrame(
            {
                "era5": [1.0, 2.0, 3.0, np.nan],
                "wrf": [0.0, 2.0, 4.0, 8.0],
            }
        )

        result = comparison_metrics(frame, "era5", "wrf")

        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["bias_era5_minus_wrf"], 0.0)
        self.assertAlmostEqual(result["mae"], 2.0 / 3.0)
        self.assertAlmostEqual(result["rmse"], np.sqrt(2.0 / 3.0))
        self.assertAlmostEqual(result["pearson_r"], 1.0)

    def test_interface_heights_use_ground_midpoints_and_top_extrapolation(self):
        mass_height = np.array([10.0, 30.0, 70.0])

        result = interface_heights_from_mass_levels(mass_height)

        np.testing.assert_allclose(result, [0.0, 20.0, 50.0, 90.0])

    def test_geopotential_to_agl_subtracts_surface_geopotential(self):
        result = geopotential_to_agl(
            np.array([980.665, 1961.33]),
            490.3325,
        )

        np.testing.assert_allclose(result, [50.0, 150.0], rtol=1e-12)

    def test_near_surface_fill_covers_zero_one_hundred_and_two_hundred_metres(self):
        profiles = {
            "o3_mass_ug_m3": np.array([np.nan, np.nan, np.nan, 30.0, 40.0]),
            "w_geometric_m_s": np.array([np.nan, np.nan, np.nan, 0.3, 0.4]),
        }

        methods = _fill_near_surface(profiles)

        np.testing.assert_allclose(profiles["o3_mass_ug_m3"][:4], [30.0, 30.0, 30.0, 30.0])
        self.assertEqual(methods[:3], ["copied_from_300m"] * 3)

    def test_process_wrf_file_interpolates_two_times_for_both_stations(self):
        root = Path("E:/research/\u5782\u76f4\u4ea4\u6362/\u5408\u80a5\u53cc\u7ad9_ERA5\u4e0eWRF_2023/WRF\u7ad9\u70b9\u62bd\u53d6\u7ed3\u679c")
        path = root / "wrf_hf_cf_allvars_202304_202310.nc"

        result = process_wrf_file(path, max_times=2)

        self.assertEqual(len(result), 2 * 2 * 41)
        self.assertEqual(set(result["station"]), {"HF", "CF"})
        self.assertEqual(set(result["bjt_hour"]), {8, 9})
        self.assertEqual(result.groupby(["bjt_time", "station"]).size().unique().tolist(), [41])
        self.assertTrue((result["height_agl_m"] >= 0).all())
        self.assertTrue((result["height_agl_m"] <= 4000).all())
        self.assertTrue(np.isfinite(result.loc[result["height_agl_m"] == 0, "o3_mass_ug_m3"]).all())

    def test_process_era5_file_uses_nearest_grid_and_converts_two_times(self):
        root = Path("E:/research/\u5782\u76f4\u4ea4\u6362/\u5408\u80a5\u53cc\u7ad9_ERA5\u4e0eWRF_2023/ERA5_pressure_levels_600_1000")
        path = root / "era5_pressure_600_1000_hefei_hf_cf_202304_01-10.nc"
        stations = {"HF": (31.78, 117.18), "CF": (32.21, 117.18)}

        result = process_era5_files([path], stations, max_times_per_file=2)

        self.assertEqual(len(result), 2 * 2 * 41)
        self.assertEqual(set(result["station"]), {"HF", "CF"})
        self.assertEqual(set(result["bjt_hour"]), {8, 9})
        self.assertEqual(result.groupby(["bjt_time", "station"]).size().unique().tolist(), [41])
        self.assertEqual(
            result.loc[result["station"] == "HF", ["grid_latitude", "grid_longitude"]]
            .drop_duplicates()
            .round(2)
            .values.tolist(),
            [[31.75, 117.25]],
        )
        self.assertTrue(np.isfinite(result.loc[result["height_agl_m"] == 0, "o3_mass_ug_m3"]).all())

    def test_read_era5_blh_returns_two_times_for_both_stations(self):
        root = Path("E:/research/\u5782\u76f4\u4ea4\u6362/\u5408\u80a5\u53cc\u7ad9_ERA5\u4e0eWRF_2023/ERA5_single_levels_BLH")
        path = root / "era5_blh_hefei_hf_cf_202304_01-10.nc"
        stations = {"HF": (31.78, 117.18), "CF": (32.21, 117.18)}

        result = read_era5_blh([path], stations, max_times_per_file=2)

        self.assertEqual(len(result), 4)
        self.assertEqual(set(result["bjt_hour"]), {8, 9})
        self.assertTrue(np.isfinite(result["pblh_m"]).all())

    def test_native_level_readers_preserve_source_level_counts(self):
        base = Path("E:/research/\u5782\u76f4\u4ea4\u6362/\u5408\u80a5\u53cc\u7ad9_ERA5\u4e0eWRF_2023")
        era5_path = base / "ERA5_pressure_levels_600_1000" / "era5_pressure_600_1000_hefei_hf_cf_202304_01-10.nc"
        wrf_path = base / "WRF\u7ad9\u70b9\u62bd\u53d6\u7ed3\u679c" / "wrf_hf_cf_allvars_202304_202310.nc"
        stations = {"HF": (31.78, 117.18), "CF": (32.21, 117.18)}

        era5_native = read_era5_native_levels([era5_path], stations, max_times_per_file=2)
        wrf_native, wrf_exch = read_wrf_native_levels(wrf_path, max_times=2)

        self.assertEqual(len(era5_native), 2 * 2 * 14)
        self.assertEqual(len(wrf_native), 2 * 2 * 44)
        self.assertEqual(len(wrf_exch), 2 * 2 * 45)

    def test_summarize_profiles_keeps_hour_dimension_for_hourly_climatology(self):
        frame = pd.DataFrame(
            {
                "station": ["HF"] * 4,
                "bjt_time": pd.to_datetime(
                    ["2023-04-01 08:00", "2023-04-02 08:00", "2023-04-01 09:00", "2023-04-02 09:00"]
                ),
                "bjt_date": ["2023-04-01", "2023-04-02", "2023-04-01", "2023-04-02"],
                "bjt_hour": [8, 8, 9, 9],
                "month": [4, 4, 4, 4],
                "height_agl_m": [100.0] * 4,
                "o3_mass_ug_m3": [1.0, 3.0, 2.0, 4.0],
            }
        )

        summaries = summarize_profiles(frame, ["o3_mass_ug_m3"])

        hourly = summaries["hourly_climatology"]
        self.assertEqual(hourly["bjt_hour"].tolist(), [8, 9])
        np.testing.assert_allclose(hourly["o3_mass_ug_m3"], [2.0, 3.0])

    def test_build_comparison_pairs_only_matching_station_time_and_height(self):
        keys = {
            "station": ["HF", "HF"],
            "bjt_time": pd.to_datetime(["2023-04-01 08:00", "2023-04-01 09:00"]),
            "height_agl_m": [100.0, 100.0],
        }
        era5 = pd.DataFrame(keys | {"o3_mass_ug_m3": [10.0, 20.0], "w_geometric_m_s": [0.1, 0.2]})
        wrf = pd.DataFrame(keys | {"o3_mass_ug_m3": [8.0, 25.0], "w_geometric_m_s": [0.0, 0.3]})

        result = build_comparison(era5, wrf)

        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result["o3_diff_era5_minus_wrf_ug_m3"], [2.0, -5.0])
        np.testing.assert_allclose(result["w_diff_era5_minus_wrf_m_s"], [0.1, -0.1])

    def test_build_comparison_drops_rows_without_complete_pair_for_both_variables(self):
        era5 = pd.DataFrame(
            {
                "station": ["HF", "HF"],
                "bjt_time": pd.to_datetime(["2023-04-01 08:00", "2023-04-01 09:00"]),
                "height_agl_m": [200.0, 200.0],
                "o3_mass_ug_m3": [np.nan, 20.0],
                "w_geometric_m_s": [np.nan, 0.2],
            }
        )
        wrf = pd.DataFrame(
            {
                "station": ["HF", "HF"],
                "bjt_time": pd.to_datetime(["2023-04-01 08:00", "2023-04-01 09:00"]),
                "height_agl_m": [200.0, 200.0],
                "o3_mass_ug_m3": [15.0, 25.0],
                "w_geometric_m_s": [0.1, 0.3],
            }
        )

        result = build_comparison(era5, wrf)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["bjt_hour"], 9)


if __name__ == "__main__":
    unittest.main()
