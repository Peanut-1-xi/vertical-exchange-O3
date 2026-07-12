import unittest

import numpy as np
import pandas as pd

from plot_three_vertical_flux_methods_hf_cf_2023 import (
    calculate_vertical_fluxes,
    daytime_profile,
    diurnal_matrix,
)


class VerticalFluxMethodsTest(unittest.TestCase):
    def test_wrf_uses_wrf_o3_for_advective_turbulent_and_total_flux(self):
        frame = pd.DataFrame(
            {
                "station": ["HF"] * 3,
                "bjt_time": pd.to_datetime(["2023-04-01 08:00"] * 3),
                "bjt_hour": [8] * 3,
                "height_agl_m": [0.0, 100.0, 200.0],
                "o3_mass_ug_m3": [10.0, 20.0, 40.0],
                "w_geometric_m_s": [1.0, 2.0, 3.0],
                "exch_h_source_value": [5.0, 5.0, 5.0],
            }
        )

        result = calculate_vertical_fluxes(frame, "WRF")

        expected_gradient = np.gradient([10.0, 20.0, 40.0], [0.0, 100.0, 200.0])
        np.testing.assert_allclose(result["f_adv_ug_m2_s"], [10.0, 40.0, 120.0])
        np.testing.assert_allclose(result["o3_gradient_ug_m4"], expected_gradient)
        np.testing.assert_allclose(result["f_turb_ug_m2_s"], -5.0 * expected_gradient)
        np.testing.assert_allclose(
            result["f_total_ug_m2_s"],
            result["f_adv_ug_m2_s"] + result["f_turb_ug_m2_s"],
        )

    def test_era5_uses_only_era5_o3_and_geometric_velocity(self):
        frame = pd.DataFrame(
            {
                "station": ["CF", "CF"],
                "bjt_time": pd.to_datetime(["2023-04-01 08:00"] * 2),
                "bjt_hour": [8, 8],
                "height_agl_m": [0.0, 100.0],
                "o3_mass_ug_m3": [50.0, 60.0],
                "w_geometric_m_s": [-0.01, 0.02],
            }
        )

        result = calculate_vertical_fluxes(frame, "ERA5")

        np.testing.assert_allclose(result["f_adv_ug_m2_s"], [-0.5, 1.2])
        self.assertNotIn("f_turb_ug_m2_s", result.columns)
        self.assertNotIn("f_total_ug_m2_s", result.columns)

    def test_diurnal_matrix_and_daytime_profile_average_requested_flux(self):
        frame = pd.DataFrame(
            {
                "station": ["HF"] * 8,
                "bjt_time": pd.to_datetime(
                    [
                        "2023-04-01 08:00",
                        "2023-04-01 08:00",
                        "2023-04-01 09:00",
                        "2023-04-01 09:00",
                        "2023-04-02 08:00",
                        "2023-04-02 08:00",
                        "2023-04-02 09:00",
                        "2023-04-02 09:00",
                    ]
                ),
                "bjt_hour": [8, 8, 9, 9, 8, 8, 9, 9],
                "height_agl_m": [0.0, 100.0] * 4,
                "flux": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            }
        )

        matrix = diurnal_matrix(frame, "flux", hours=[8, 9])
        profile = daytime_profile(frame, "flux")

        np.testing.assert_allclose(matrix.to_numpy(), [[3.0, 5.0], [4.0, 6.0]])
        np.testing.assert_allclose(profile["flux_mean"], [4.0, 5.0])


if __name__ == "__main__":
    unittest.main()
