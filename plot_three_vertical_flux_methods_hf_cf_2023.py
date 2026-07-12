from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


HOURS = list(range(8, 17))
HEIGHT_MAX_M = 4000.0
BASE_DIR = Path("E:/research/\u5782\u76f4\u4ea4\u6362/\u5408\u80a5\u53cc\u7ad9_ERA5\u4e0eWRF_2023")
RESULT_DIR = BASE_DIR / "ERA5_WRF\u7edf\u4e00\u63d2\u503c\u4e0e\u5bf9\u6bd4\u7ed3\u679c"
ERA5_FILE = RESULT_DIR / "ERA5\u6570\u636e\u8868" / "ERA5_HF_CF_202304-202310_\u9010\u5c0f\u65f60-4km\u63d2\u503c.csv"
WRF_FILE = RESULT_DIR / "WRF\u6570\u636e\u8868" / "WRF_HF_CF_202304-202310_\u9010\u5c0f\u65f60-4km\u63d2\u503c.csv"
OUTPUT_DIR = RESULT_DIR / "\u7efc\u5408\u5206\u6790\u56fe" / "\u4e09\u79cd\u5782\u76f4\u8f93\u9001\u65b9\u6cd5"


def calculate_vertical_fluxes(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Calculate source-consistent vertical ozone flux diagnostics."""
    source = source.upper()
    if source not in {"ERA5", "WRF"}:
        raise ValueError("source must be 'ERA5' or 'WRF'")
    result = frame.copy()
    result["f_adv_ug_m2_s"] = (
        result["o3_mass_ug_m3"] * result["w_geometric_m_s"]
    )
    if source == "ERA5":
        return result
    if "exch_h_source_value" not in result.columns:
        raise KeyError("WRF calculation requires exch_h_source_value")

    result["o3_gradient_ug_m4"] = np.nan
    for _, group in result.groupby(["station", "bjt_time"], sort=False):
        ordered = group.sort_values("height_agl_m")
        gradient = np.gradient(
            ordered["o3_mass_ug_m3"].to_numpy(dtype=float),
            ordered["height_agl_m"].to_numpy(dtype=float),
        )
        result.loc[ordered.index, "o3_gradient_ug_m4"] = gradient
    result["f_turb_ug_m2_s"] = -(
        result["exch_h_source_value"] * result["o3_gradient_ug_m4"]
    )
    result["f_total_ug_m2_s"] = (
        result["f_adv_ug_m2_s"] + result["f_turb_ug_m2_s"]
    )
    return result


def diurnal_matrix(
    frame: pd.DataFrame,
    value_column: str,
    hours: list[int] = HOURS,
) -> pd.DataFrame:
    matrix = frame.pivot_table(
        index="height_agl_m",
        columns="bjt_hour",
        values=value_column,
        aggfunc="mean",
    )
    return matrix.reindex(index=sorted(matrix.index), columns=hours)


def daytime_profile(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return (
        frame.groupby("height_agl_m", as_index=False)
        .agg(**{f"{value_column}_mean": (value_column, "mean")})
        .rename(columns={f"{value_column}_mean": "flux_mean"})
        .sort_values("height_agl_m")
    )


def set_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def read_source(path: Path, source: str) -> pd.DataFrame:
    columns = [
        "station",
        "bjt_time",
        "bjt_hour",
        "height_agl_m",
        "o3_mass_ug_m3",
        "w_geometric_m_s",
    ]
    if source.upper() == "WRF":
        columns.append("exch_h_source_value")
    frame = pd.read_csv(path, encoding="utf-8-sig", usecols=columns)
    frame["bjt_time"] = pd.to_datetime(frame["bjt_time"])
    frame = frame[
        frame["station"].isin(["HF", "CF"])
        & frame["bjt_hour"].isin(HOURS)
        & frame["height_agl_m"].between(0.0, HEIGHT_MAX_M)
    ].copy()
    return frame.dropna(subset=columns[2:])


def nice_symmetric_limit(frame: pd.DataFrame, value_column: str) -> float:
    values: list[np.ndarray] = []
    for station in ("HF", "CF"):
        matrix = diurnal_matrix(
            frame[frame["station"] == station], value_column
        ).to_numpy(dtype=float)
        finite = matrix[np.isfinite(matrix)]
        if finite.size:
            values.append(finite)
    if not values:
        raise ValueError(f"No finite values available for {value_column}")
    percentile = float(np.nanpercentile(np.abs(np.concatenate(values)), 98.0))
    if percentile <= 0.0:
        return 1.0
    magnitude = 10.0 ** np.floor(np.log10(percentile))
    step = magnitude / 5.0
    return float(np.ceil(percentile / step) * step)


def plot_four_panel(
    frame: pd.DataFrame,
    value_column: str,
    figure_title: str,
    x_label: str,
    colorbar_label: str,
    output_path: Path,
) -> None:
    set_chinese_font()
    limit = nice_symmetric_limit(frame, value_column)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(11.5, 7.6),
        dpi=300,
        gridspec_kw={"width_ratios": [1.45, 0.72]},
        sharey="row",
    )
    colors = {"HF": "#2f5f9f", "CF": "#b05a2a"}
    mesh = None
    for row, station in enumerate(("HF", "CF")):
        station_frame = frame[frame["station"] == station]
        matrix = diurnal_matrix(station_frame, value_column)
        hours = matrix.columns.to_numpy(dtype=float)
        height_km = matrix.index.to_numpy(dtype=float) / 1000.0

        axis = axes[row, 0]
        mesh = axis.pcolormesh(
            hours,
            height_km,
            matrix.to_numpy(dtype=float),
            shading="auto",
            cmap="RdBu_r",
            norm=norm,
        )
        axis.axhline(1.0, color="k", linewidth=0.7, alpha=0.25)
        axis.set_title(
            f"({chr(97 + row * 2)}) {station}：时间-高度分布",
            fontsize=14,
            fontweight="bold",
        )
        axis.set_ylabel("高度 (km)", fontsize=13, fontweight="bold")
        axis.set_xlabel("北京时间 (h)", fontsize=13, fontweight="bold")
        axis.set_xticks(HOURS)
        axis.set_ylim(0.0, HEIGHT_MAX_M / 1000.0)
        axis.tick_params(labelsize=11, direction="in")
        for spine in axis.spines.values():
            spine.set_linewidth(1.1)

        profile = daytime_profile(station_frame, value_column)
        profile_axis = axes[row, 1]
        profile_axis.plot(
            profile["flux_mean"],
            profile["height_agl_m"] / 1000.0,
            color=colors[station],
            linewidth=2.0,
            marker="o",
            markersize=3.0,
        )
        profile_axis.axvline(0.0, color="gray", linestyle="--", linewidth=1.2)
        profile_axis.axhline(1.0, color="k", linewidth=0.7, alpha=0.25)
        profile_axis.set_xlim(-limit, limit)
        profile_axis.set_ylim(0.0, HEIGHT_MAX_M / 1000.0)
        profile_axis.set_title(
            f"({chr(98 + row * 2)}) {station}：日间平均廓线",
            fontsize=14,
            fontweight="bold",
        )
        profile_axis.set_xlabel(x_label, fontsize=13, fontweight="bold")
        profile_axis.tick_params(labelsize=11, direction="in")
        for spine in profile_axis.spines.values():
            spine.set_linewidth(1.1)

    fig.suptitle(figure_title, fontsize=15, fontweight="bold", y=0.985)
    color_axis = fig.add_axes([0.91, 0.20, 0.018, 0.62])
    colorbar = fig.colorbar(mesh, cax=color_axis)
    colorbar.set_label(colorbar_label, fontsize=11.5)
    colorbar.ax.tick_params(labelsize=10.5)
    fig.subplots_adjust(
        left=0.075,
        right=0.885,
        top=0.92,
        bottom=0.075,
        hspace=0.42,
        wspace=0.25,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    wrf = calculate_vertical_fluxes(read_source(WRF_FILE, "WRF"), "WRF")
    era5 = calculate_vertical_fluxes(read_source(ERA5_FILE, "ERA5"), "ERA5")
    figures = [
        (
            wrf,
            "f_adv_ug_m2_s",
            "WRF垂直平流通量 F$_{adv}$（WRF O$_3$ × WRF wa）",
            "F$_{adv}$ (μg m$^{-2}$ s$^{-1}$)",
            "F$_{adv}$（+向上 / -向下）\n(μg m$^{-2}$ s$^{-1}$)",
            OUTPUT_DIR / "WRF_Fadv_HF_CF\u65f6\u95f4\u9ad8\u5ea6\u4e0e\u5e73\u5747\u5ed3\u7ebf.png",
        ),
        (
            wrf,
            "f_turb_ug_m2_s",
            "WRF湍流扩散通量 F$_{turb}$（EXCH_H按m$^2$ s$^{-1}$假定）",
            "F$_{turb}$ (μg m$^{-2}$ s$^{-1}$)",
            "F$_{turb}$（+向上 / -向下）\n(μg m$^{-2}$ s$^{-1}$)",
            OUTPUT_DIR / "WRF_Fturb_HF_CF\u65f6\u95f4\u9ad8\u5ea6\u4e0e\u5e73\u5747\u5ed3\u7ebf.png",
        ),
        (
            wrf,
            "f_total_ug_m2_s",
            "WRF综合垂直输送 F$_{total}$（EXCH_H按m$^2$ s$^{-1}$假定）",
            "F$_{total}$ (μg m$^{-2}$ s$^{-1}$)",
            "F$_{total}$（+向上 / -向下）\n(μg m$^{-2}$ s$^{-1}$)",
            OUTPUT_DIR / "WRF_Ftotal_HF_CF\u65f6\u95f4\u9ad8\u5ea6\u4e0e\u5e73\u5747\u5ed3\u7ebf.png",
        ),
        (
            era5,
            "f_adv_ug_m2_s",
            "ERA5垂直平流通量 F$_{adv}$（ERA5 O$_3$ × ERA5 w$_{geo}$）",
            "F$_{adv}$ (μg m$^{-2}$ s$^{-1}$)",
            "F$_{adv}$（+向上 / -向下）\n(μg m$^{-2}$ s$^{-1}$)",
            OUTPUT_DIR / "ERA5_Fadv_HF_CF\u65f6\u95f4\u9ad8\u5ea6\u4e0e\u5e73\u5747\u5ed3\u7ebf.png",
        ),
    ]
    for args in figures:
        plot_four_panel(*args)
        print(f"Wrote {args[-1]}")


if __name__ == "__main__":
    main()
