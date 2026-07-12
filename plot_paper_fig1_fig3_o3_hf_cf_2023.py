from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HOURS = list(range(8, 17))
HEIGHT_MAX_M = 3000.0
BASE_DIR = Path("E:/research/垂直交换/合肥双站_ERA5与WRF_2023")
RESULT_DIR = BASE_DIR / "ERA5_WRF统一插值与对比结果"
ERA5_FILE = (
    RESULT_DIR
    / "ERA5数据表"
    / "ERA5_HF_CF_202304-202310_逐小时0-4km插值.csv"
)
WRF_FILE = (
    RESULT_DIR
    / "WRF数据表"
    / "WRF_HF_CF_202304-202310_逐小时0-4km插值.csv"
)
OUTPUT_DIR = RESULT_DIR / "综合分析图" / "论文Fig1与Fig3_O3复现"
REQUIRED_COLUMNS = [
    "station",
    "bjt_time",
    "bjt_hour",
    "height_agl_m",
    "o3_mass_ug_m3",
    "u_wind_m_s",
    "v_wind_m_s",
    "pblh_m",
]


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


def validate_frame(frame: pd.DataFrame, source: str) -> None:
    source = source.upper()
    if source not in {"ERA5", "WRF"}:
        raise ValueError("source must be ERA5 or WRF")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(", ".join(missing))
    if "source" in frame.columns:
        observed = set(frame["source"].dropna().astype(str).str.upper().unique())
        if observed and observed != {source}:
            raise ValueError(
                f"{source} plot received rows labelled as {sorted(observed)}"
            )


def read_source(path: Path, source: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
    columns = REQUIRED_COLUMNS + (["source"] if "source" in header else [])
    frame = pd.read_csv(path, usecols=columns, encoding="utf-8-sig")
    validate_frame(frame, source)
    frame["bjt_time"] = pd.to_datetime(frame["bjt_time"])
    selected = frame[
        frame["station"].isin(["HF", "CF"])
        & frame["bjt_hour"].isin(HOURS)
        & frame["height_agl_m"].between(0.0, HEIGHT_MAX_M)
    ].copy()
    return selected.dropna(subset=REQUIRED_COLUMNS[2:])


def set_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "dejavusans"


def _station_hourly_pblh(frame: pd.DataFrame, station: str) -> pd.Series:
    unique_times = (
        frame[frame["station"] == station]
        .groupby(["bjt_time", "bjt_hour"], as_index=False)["pblh_m"]
        .first()
    )
    return unique_times.groupby("bjt_hour")["pblh_m"].mean().reindex(HOURS)


def _matrix(
    aggregated: pd.DataFrame, station: str, value: str
) -> pd.DataFrame:
    station_data = aggregated[aggregated["station"] == station]
    matrix = station_data.pivot(
        index="height_agl_m", columns="bjt_hour", values=value
    )
    return matrix.reindex(index=sorted(matrix.index), columns=HOURS)


def _rounded_upper_limit(values: np.ndarray, percentile: float = 98.0) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("No finite values are available for the color scale")
    upper = float(np.nanpercentile(finite, percentile))
    if upper <= 0.0:
        return 1.0
    magnitude = 10.0 ** np.floor(np.log10(upper))
    step = magnitude / 5.0
    return float(np.ceil(upper / step) * step)


def plot_figure1_o3(
    frame: pd.DataFrame, source: str, output_path: Path
) -> None:
    validate_frame(frame, source)
    set_chinese_font()
    aggregated = aggregate_hour_height(frame)
    matrices = {
        station: _matrix(aggregated, station, "o3_mean")
        for station in ("HF", "CF")
    }
    all_values = np.concatenate(
        [matrix.to_numpy(dtype=float).ravel() for matrix in matrices.values()]
    )
    finite = all_values[np.isfinite(all_values)]
    vmin = max(0.0, float(np.nanpercentile(finite, 2.0)))
    vmax = _rounded_upper_limit(finite)

    fig, axes = plt.subplots(
        1, 2, figsize=(12.2, 5.1), dpi=300, sharey=True, layout="constrained"
    )
    mesh = None
    titles = {"HF": "城市站 HF", "CF": "农村站 CF"}
    for axis, station, panel in zip(axes, ("HF", "CF"), ("a", "b")):
        matrix = matrices[station]
        mesh = axis.pcolormesh(
            matrix.columns.to_numpy(dtype=float),
            matrix.index.to_numpy(dtype=float) / 1000.0,
            matrix.to_numpy(dtype=float),
            shading="auto",
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        pblh = _station_hourly_pblh(frame, station)
        axis.plot(
            pblh.index,
            pblh.to_numpy(dtype=float) / 1000.0,
            color="black",
            linewidth=2.2,
            label="平均边界层高度",
        )
        axis.set_title(
            f"({panel}) {source.upper()} {titles[station]} O$_3$",
            fontsize=15,
            fontweight="bold",
        )
        axis.set_xlabel("北京时间 (h)", fontsize=13, fontweight="bold")
        axis.set_xticks(HOURS)
        axis.set_xlim(8, 16)
        axis.set_ylim(0, HEIGHT_MAX_M / 1000.0)
        axis.tick_params(labelsize=11, direction="in")
        axis.grid(color="white", alpha=0.22, linewidth=0.6)
        for spine in axis.spines.values():
            spine.set_linewidth(1.1)
    axes[0].set_ylabel("距地高度 (km)", fontsize=13, fontweight="bold")
    axes[1].legend(loc="upper left", fontsize=10, frameon=True)
    colorbar = fig.colorbar(mesh, ax=axes, shrink=0.88, pad=0.025)
    colorbar.set_label("O$_3$浓度 (μg m$^{-3}$)", fontsize=12)
    colorbar.ax.tick_params(labelsize=10.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _normalized_vectors(
    u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    speed = np.hypot(u, v)
    return (
        np.divide(u, speed, out=np.zeros_like(u), where=speed > 0),
        np.divide(v, speed, out=np.zeros_like(v), where=speed > 0),
    )


def plot_figure3_o3(
    frame: pd.DataFrame, source: str, output_path: Path
) -> None:
    validate_frame(frame, source)
    set_chinese_font()
    aggregated = aggregate_hour_height(frame)
    profile = aggregate_daytime_profile(frame)
    flux_matrices = {
        station: _matrix(aggregated, station, "tfh_mean")
        for station in ("HF", "CF")
    }
    limit = _rounded_upper_limit(
        np.concatenate(
            [matrix.to_numpy(dtype=float).ravel() for matrix in flux_matrices.values()]
        )
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.2, 8.2),
        dpi=300,
        sharey="row",
        gridspec_kw={"width_ratios": [1.6, 0.72]},
        layout="constrained",
    )
    titles = {"HF": "城市站 HF", "CF": "农村站 CF"}
    line_colors = {"HF": "#d62728", "CF": "#b22222"}
    mesh = None
    for row, station in enumerate(("HF", "CF")):
        magnitude = flux_matrices[station]
        u_matrix = _matrix(aggregated, station, "tfu_mean")
        v_matrix = _matrix(aggregated, station, "tfv_mean")
        hours = magnitude.columns.to_numpy(dtype=float)
        heights_km = magnitude.index.to_numpy(dtype=float) / 1000.0

        heat_axis = axes[row, 0]
        mesh = heat_axis.pcolormesh(
            hours,
            heights_km,
            magnitude.to_numpy(dtype=float),
            shading="auto",
            cmap="turbo",
            vmin=0.0,
            vmax=limit,
            rasterized=True,
        )
        stride = max(1, len(heights_km) // 11)
        arrow_u, arrow_v = _normalized_vectors(
            u_matrix.to_numpy(dtype=float)[::stride, :],
            v_matrix.to_numpy(dtype=float)[::stride, :],
        )
        arrow_hours, arrow_heights = np.meshgrid(
            hours, heights_km[::stride]
        )
        heat_axis.quiver(
            arrow_hours,
            arrow_heights,
            arrow_u,
            arrow_v,
            color="black",
            pivot="middle",
            angles="uv",
            scale_units="inches",
            scale=5.2,
            width=0.0042,
            headwidth=3.8,
            headlength=4.5,
        )
        pblh = _station_hourly_pblh(frame, station)
        heat_axis.plot(
            pblh.index,
            pblh.to_numpy(dtype=float) / 1000.0,
            color="black",
            linewidth=2.4,
        )
        heat_axis.set_title(
            f"({chr(97 + row * 2)}) {source.upper()} {titles[station]} O$_3$水平输送",
            fontsize=14,
            fontweight="bold",
        )
        heat_axis.set_xlabel("北京时间 (h)", fontsize=12.5, fontweight="bold")
        heat_axis.set_ylabel("距地高度 (km)", fontsize=12.5, fontweight="bold")
        heat_axis.set_xticks(HOURS)
        heat_axis.set_xlim(8, 16)
        heat_axis.set_ylim(0, HEIGHT_MAX_M / 1000.0)
        heat_axis.tick_params(labelsize=10.5, direction="in")
        for spine in heat_axis.spines.values():
            spine.set_linewidth(1.1)

        station_profile = profile[profile["station"] == station]
        profile_axis = axes[row, 1]
        x = station_profile["tfh_mean"].to_numpy(dtype=float)
        y = station_profile["height_agl_m"].to_numpy(dtype=float) / 1000.0
        profile_axis.plot(
            x,
            y,
            color=line_colors[station],
            linewidth=2.0,
            marker="o",
            markersize=3.2,
        )
        profile_u, profile_v = _normalized_vectors(
            station_profile["tfu_mean"].to_numpy(dtype=float),
            station_profile["tfv_mean"].to_numpy(dtype=float),
        )
        profile_axis.quiver(
            x[::2],
            y[::2],
            profile_u[::2],
            profile_v[::2],
            color="black",
            pivot="middle",
            angles="uv",
            scale_units="inches",
            scale=5.0,
            width=0.006,
        )
        profile_axis.set_title(
            f"({chr(98 + row * 2)}) 日间平均廓线",
            fontsize=14,
            fontweight="bold",
        )
        profile_axis.set_xlabel(
            "O$_3$水平输送通量\n(μg m$^{-2}$ s$^{-1}$)",
            fontsize=11.5,
            fontweight="bold",
        )
        profile_axis.set_xlim(0.0, limit * 1.08)
        profile_axis.set_ylim(0, HEIGHT_MAX_M / 1000.0)
        profile_axis.tick_params(labelsize=10.5, direction="in")
        profile_axis.grid(alpha=0.25, linewidth=0.6)
        for spine in profile_axis.spines.values():
            spine.set_linewidth(1.1)

    colorbar = fig.colorbar(mesh, ax=axes, shrink=0.84, pad=0.025)
    colorbar.set_label(
        "O$_3$水平输送通量 (μg m$^{-2}$ s$^{-1}$)", fontsize=11.5
    )
    colorbar.ax.tick_params(labelsize=10.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for source, path in (("ERA5", ERA5_FILE), ("WRF", WRF_FILE)):
        frame = calculate_horizontal_flux(read_source(path, source))
        plot_figure1_o3(
            frame,
            source,
            OUTPUT_DIR / f"{source}_Fig1_O3_HF_CF时间高度分布.png",
        )
        plot_figure3_o3(
            frame,
            source,
            OUTPUT_DIR / f"{source}_Fig3_O3水平输送通量_HF_CF.png",
        )


if __name__ == "__main__":
    main()
