from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LAYER_ORDER = ["within_pbl", "pbl_top", "above_pbl"]
LAYER_LABELS = {
    "within_pbl": "边界层内",
    "pbl_top": "边界层顶\n至+0.5 km",
    "above_pbl": "边界层以上\n至3 km",
}
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
OUTPUT_DIR = RESULT_DIR / "综合分析图" / "BLH相对分层统计"
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


def set_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _ordered_rows(summary: pd.DataFrame, source: str, station: str) -> pd.DataFrame:
    selected = summary[
        (summary["source"] == source) & (summary["station"] == station)
    ].copy()
    selected["blh_layer"] = pd.Categorical(
        selected["blh_layer"], categories=LAYER_ORDER, ordered=True
    )
    return selected.sort_values("blh_layer")


def plot_layer_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    set_chinese_font()
    specs = [
        ("o3", 1.0, "O$_3$浓度 (μg m$^{-3}$)", "(a) O$_3$浓度"),
        ("w", 1000.0, "几何垂直速度 (10$^{-3}$ m s$^{-1}$)", "(b) 垂直速度w"),
        ("fadv", 1.0, "Fadv (μg m$^{-2}$ s$^{-1}$)", "(c) 垂直平流通量"),
        ("tfh", 1.0, "水平输送强度 (μg m$^{-2}$ s$^{-1}$)", "(d) 水平输送强度"),
    ]
    styles = {
        ("ERA5", "HF"): ("#2166ac", "o", "ERA5-HF"),
        ("ERA5", "CF"): ("#67a9cf", "s", "ERA5-CF"),
        ("WRF", "HF"): ("#b2182b", "^", "WRF-HF"),
        ("WRF", "CF"): ("#ef8a62", "D", "WRF-CF"),
    }
    x = np.arange(len(LAYER_ORDER), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0), dpi=300)
    handles = []
    labels = []
    for axis, (prefix, scale, ylabel, title) in zip(axes.ravel(), specs):
        for (source, station), (color, marker, label) in styles.items():
            rows = _ordered_rows(summary, source, station)
            line = axis.errorbar(
                x,
                rows[f"{prefix}_mean"].to_numpy(dtype=float) * scale,
                yerr=rows[f"{prefix}_sem"].to_numpy(dtype=float) * scale,
                color=color,
                marker=marker,
                linewidth=2.0,
                markersize=7.0,
                capsize=3.5,
                label=label,
            )
            if len(handles) < len(styles):
                handles.append(line)
                labels.append(label)
        axis.set_title(title, fontsize=14, fontweight="bold")
        axis.set_ylabel(ylabel, fontsize=11.5, fontweight="bold")
        axis.set_xticks(x, [LAYER_LABELS[layer] for layer in LAYER_ORDER])
        axis.tick_params(labelsize=10.5, direction="in")
        axis.grid(axis="y", alpha=0.28, linewidth=0.7)
        if prefix in {"w", "fadv"}:
            axis.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        for spine in axis.spines.values():
            spine.set_linewidth(1.0)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=4,
        frameon=False,
        fontsize=11.5,
    )
    fig.suptitle(
        "2023年4-10月 HF/CF边界层相对分层对比（北京时间08:00-16:00）",
        fontsize=15,
        fontweight="bold",
        y=0.988,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.09,
        top=0.845,
        hspace=0.38,
        wspace=0.22,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_wrf_flux_decomposition(summary: pd.DataFrame, output_path: Path) -> None:
    set_chinese_font()
    x = np.arange(len(LAYER_ORDER), dtype=float)
    width = 0.23
    components = [
        ("fadv", "#377eb8", "Fadv"),
        ("fturb", "#e41a1c", "Fturb"),
        ("ftotal", "#4daf4a", "Ftotal"),
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(11.6, 5.0), dpi=300, sharey=True, layout="constrained"
    )
    for axis, station, panel in zip(axes, ("HF", "CF"), ("a", "b")):
        rows = _ordered_rows(summary, "WRF", station)
        for index, (prefix, color, label) in enumerate(components):
            offset = (index - 1) * width
            axis.bar(
                x + offset,
                rows[f"{prefix}_mean"].to_numpy(dtype=float),
                width=width,
                yerr=rows[f"{prefix}_sem"].to_numpy(dtype=float),
                color=color,
                edgecolor="black",
                linewidth=0.55,
                capsize=3.0,
                label=label,
            )
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.set_xticks(x, [LAYER_LABELS[layer] for layer in LAYER_ORDER])
        axis.set_title(
            f"({panel}) WRF {station}站", fontsize=14, fontweight="bold"
        )
        axis.set_xlabel("BLH相对高度层", fontsize=11.5, fontweight="bold")
        axis.tick_params(labelsize=10.5, direction="in")
        axis.grid(axis="y", alpha=0.25, linewidth=0.7)
        for spine in axis.spines.values():
            spine.set_linewidth(1.0)
    axes[0].set_ylabel("O$_3$通量 (μg m$^{-2}$ s$^{-1}$)", fontsize=12, fontweight="bold")
    axes[1].legend(loc="best", fontsize=10.5, frameon=True)
    fig.suptitle(
        "WRF边界层相对分层通量分解（EXCH_H按m$^2$ s$^{-1}$解释）",
        fontsize=15,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def read_source(path: Path, source: str) -> pd.DataFrame:
    base_columns = [
        "station",
        "bjt_time",
        "bjt_hour",
        "height_agl_m",
        "pblh_m",
        "o3_mass_ug_m3",
        "w_geometric_m_s",
        "u_wind_m_s",
        "v_wind_m_s",
    ]
    columns = base_columns + (
        ["exch_h_source_value"] if source.upper() == "WRF" else []
    )
    frame = pd.read_csv(path, usecols=columns, encoding="utf-8-sig")
    frame["bjt_time"] = pd.to_datetime(frame["bjt_time"])
    selected = frame[
        frame["station"].isin(["HF", "CF"])
        & frame["bjt_hour"].isin(HOURS)
        & frame["height_agl_m"].between(0.0, HEIGHT_MAX_M)
    ].copy()
    return selected.dropna(subset=columns[2:])


def main() -> None:
    hourly_parts = []
    for source, path in (("ERA5", ERA5_FILE), ("WRF", WRF_FILE)):
        diagnosed = calculate_diagnostics(read_source(path, source), source)
        layered = assign_blh_layer(diagnosed)
        hourly_parts.append(hourly_layer_means(layered))
    hourly = pd.concat(hourly_parts, ignore_index=True)
    summary = summarize_layers(hourly)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(
        OUTPUT_DIR / "BLH相对分层_逐小时层平均.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        OUTPUT_DIR / "BLH相对分层_统计摘要.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_layer_comparison(
        summary, OUTPUT_DIR / "BLH相对分层_O3_w_Fadv_水平通量对比.png"
    )
    plot_wrf_flux_decomposition(
        summary, OUTPUT_DIR / "WRF_BLH相对分层_通量分解.png"
    )


if __name__ == "__main__":
    main()
