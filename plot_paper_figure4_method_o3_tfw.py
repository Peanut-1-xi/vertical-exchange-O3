# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


BASE_DIR = Path("E:/research/????")
OUT_DIR = BASE_DIR / "\u8bba\u6587Figure4\u540c\u65b9\u6cd5\u590d\u73b0\u56fe"

HOURS = list(range(8, 17))
HEIGHT_MAX_M = 4000.0
DAYTIME_AVG_HOURS = list(range(8, 17))
LAYER_AVG_HOURS = list(range(10, 17))
LAYER_MIN_M = 200.0
LAYER_MAX_M = 1800.0
COMMON_START = "2025-03-01"
COMMON_END = "2025-12-31"
O3_THRESHOLD_UG_M3 = 160.0
NEAR_SURFACE_MAX_M = 100.0


CITY_FILES = [
    {
        "label": "Shanghai ERA5 O3",
        "short_label": "Shanghai",
        "path": BASE_DIR
        / "\u4e0a\u6d77_ERA5\u6570\u636e\u4e0e\u5904\u7406\u7ed3\u679c\u6c47\u603b"
        / "ERA5_\u7b49\u6548\u6e4d\u6d41\u7cfb\u6570_\u8fd1\u5730\u9762\u8865\u9f50"
        / "site_effective_turbulence_hourly.csv",
    },
    {
        "label": "Hefei ERA5 O3",
        "short_label": "Hefei",
        "path": BASE_DIR
        / "\u5408\u80a5\u79d1\u5b66\u5c9b_ERA5\u6570\u636e\u4e0e\u5904\u7406\u7ed3\u679c\u6c47\u603b"
        / "\u5408\u80a5\u79d1\u5b66\u5c9b_ERA5_\u7b49\u6548\u6e4d\u6d41\u7cfb\u6570_\u8fd1\u5730\u9762\u8865\u9f50"
        / "site_effective_turbulence_hourly.csv",
    },
]

HEFEI_ERA5_FILE = CITY_FILES[1]["path"]
HEFEI_LOCAL_O3_DIR = (
    BASE_DIR
    / "OzoneProfile_result_inter"
    / "OzoneProfile_result_inter"
    / "xgbr_AODplusOzonePorfile_pre_BJT_hourly_mean"
)


def read_era5_site(path: Path, label: str) -> pd.DataFrame:
    cols = [
        "bjt_time",
        "bjt_date",
        "bjt_hour",
        "height_agl_m",
        "o3_mass_ug_m3",
        "w_geometric_m_s",
    ]
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=cols)
    df["bjt_time"] = pd.to_datetime(df["bjt_time"])
    df["bjt_date"] = df["bjt_time"].dt.date.astype(str)
    df["bjt_hour"] = df["bjt_time"].dt.hour
    df["height_m"] = pd.to_numeric(df["height_agl_m"], errors="coerce")
    df["o3_concentration_ug_m3"] = pd.to_numeric(df["o3_mass_ug_m3"], errors="coerce")
    df["w_m_s"] = pd.to_numeric(df["w_geometric_m_s"], errors="coerce")
    df["tfw_upward_ug_m2_s"] = df["w_m_s"] * df["o3_concentration_ug_m3"]
    df["source"] = label
    return filter_common(df)


def read_hefei_local_o3_with_era5_w() -> pd.DataFrame:
    era5 = pd.read_csv(
        HEFEI_ERA5_FILE,
        encoding="utf-8-sig",
        usecols=["bjt_time", "bjt_hour", "height_agl_m", "w_geometric_m_s"],
    )
    era5["bjt_time"] = pd.to_datetime(era5["bjt_time"])
    era5["height_m"] = pd.to_numeric(era5["height_agl_m"], errors="coerce").round().astype("Int64")
    era5["w_m_s"] = pd.to_numeric(era5["w_geometric_m_s"], errors="coerce")
    era5 = era5[["bjt_time", "bjt_hour", "height_m", "w_m_s"]].dropna()
    era5["height_m"] = era5["height_m"].astype(int)

    frames: list[pd.DataFrame] = []
    files = sorted(HEFEI_LOCAL_O3_DIR.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No local O3 profile txt files found in {HEFEI_LOCAL_O3_DIR}")
    for path in files:
        wide = pd.read_csv(path, sep="\t")
        if wide.empty or "time" not in wide.columns:
            continue
        wide["bjt_time"] = pd.to_datetime(wide["time"], errors="coerce")
        wide = wide.drop(columns=["time"]).dropna(subset=["bjt_time"])
        value_cols = [col for col in wide.columns if col != "bjt_time"]
        long = wide.melt(id_vars="bjt_time", value_vars=value_cols, var_name="height_km", value_name="o3_concentration_ug_m3")
        long["height_km"] = pd.to_numeric(long["height_km"], errors="coerce")
        long["height_m"] = (long["height_km"] * 1000.0).round().astype("Int64")
        long["o3_concentration_ug_m3"] = pd.to_numeric(long["o3_concentration_ug_m3"], errors="coerce")
        long = long.dropna(subset=["height_m", "o3_concentration_ug_m3"])
        long["height_m"] = long["height_m"].astype(int)
        frames.append(long[["bjt_time", "height_m", "o3_concentration_ug_m3"]])

    o3 = pd.concat(frames, ignore_index=True)
    merged = o3.merge(era5, on=["bjt_time", "height_m"], how="inner")
    merged["bjt_date"] = merged["bjt_time"].dt.date.astype(str)
    merged["bjt_hour"] = merged["bjt_time"].dt.hour
    merged["tfw_upward_ug_m2_s"] = merged["w_m_s"] * merged["o3_concentration_ug_m3"]
    merged["source"] = "Hefei local O3 + ERA5 w"
    return filter_common(merged)


def filter_common(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[
        (out["bjt_time"] >= pd.Timestamp(COMMON_START))
        & (out["bjt_time"] <= pd.Timestamp(COMMON_END) + pd.Timedelta(days=1))
        & (out["bjt_hour"].isin(HOURS))
        & (out["height_m"] >= 0)
        & (out["height_m"] <= HEIGHT_MAX_M)
    ].copy()
    out = out.dropna(subset=["height_m", "bjt_hour", "tfw_upward_ug_m2_s", "o3_concentration_ug_m3", "w_m_s"])
    out["height_m"] = out["height_m"].astype(float)
    return out


def diurnal_matrix(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(index="height_m", columns="bjt_hour", values="tfw_upward_ug_m2_s", aggfunc="mean")
    heights = sorted([h for h in pivot.index if h <= HEIGHT_MAX_M])
    return pivot.reindex(index=heights, columns=HOURS)


def daytime_profile(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["bjt_hour"].isin(DAYTIME_AVG_HOURS)]
    return (
        sub.groupby("height_m", as_index=False)
        .agg(
            tfw_mean_ug_m2_s=("tfw_upward_ug_m2_s", "mean"),
            o3_mean_ug_m3=("o3_concentration_ug_m3", "mean"),
            w_mean_m_s=("w_m_s", "mean"),
            n=("tfw_upward_ug_m2_s", "count"),
        )
        .sort_values("height_m")
    )


def classify_polluted_days(df: pd.DataFrame) -> pd.DataFrame:
    near_surface = (
        df[df["height_m"] <= NEAR_SURFACE_MAX_M]
        .groupby("bjt_date")["o3_concentration_ug_m3"]
        .agg(["max", "mean"])
        .rename(
            columns={
                "max": "o3_near_surface_daily_max_ug_m3",
                "mean": "o3_near_surface_daily_mean_ug_m3",
            }
        )
    )
    daily = (
        df.groupby("bjt_date")
        .agg(
            o3_profile_max_ug_m3=("o3_concentration_ug_m3", "max"),
            records=("tfw_upward_ug_m2_s", "count"),
        )
        .join(near_surface, how="left")
        .reset_index()
    )
    daily["polluted"] = daily["o3_near_surface_daily_max_ug_m3"] > O3_THRESHOLD_UG_M3
    return daily


def select_polluted_or_fallback(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict[str, object]]:
    daily = classify_polluted_days(df)
    polluted_dates = set(daily[daily["polluted"]]["bjt_date"])
    using_fallback = False
    if polluted_dates:
        selected_dates = polluted_dates
    else:
        fallback = daily.sort_values("o3_near_surface_daily_max_ug_m3", ascending=False).iloc[0]
        selected_dates = {fallback["bjt_date"]}
        using_fallback = True
    selected = df[df["bjt_date"].isin(selected_dates)].copy()
    summary = {
        "source": label,
        "criterion": f"0-100 m daily max O3 > {O3_THRESHOLD_UG_M3:.0f} ug/m3",
        "strict_polluted_days": len(polluted_dates),
        "selected_days_used_for_polluted_plot": len(selected_dates),
        "using_fallback": using_fallback,
        "selected_dates": ",".join(sorted(selected_dates)),
        "max_near_surface_o3_ug_m3": daily["o3_near_surface_daily_max_ug_m3"].max(),
    }
    return selected, summary


def layer_summary(df: pd.DataFrame, label: str) -> dict[str, object]:
    sub = df[
        (df["bjt_hour"].isin(LAYER_AVG_HOURS))
        & (df["height_m"] >= LAYER_MIN_M)
        & (df["height_m"] <= LAYER_MAX_M)
    ].copy()
    tfw = sub["tfw_upward_ug_m2_s"]
    return {
        "source": label,
        "date_min": df["bjt_date"].min(),
        "date_max": df["bjt_date"].max(),
        "records": len(df),
        "unique_days": df["bjt_date"].nunique(),
        "height_min_m": df["height_m"].min(),
        "height_max_m": df["height_m"].max(),
        "hours": ",".join(map(str, sorted(df["bjt_hour"].unique()))),
        "layer_hours": "10-16",
        "layer_height_m": "200-1800",
        "layer_mean_tfw_upward_ug_m2_s": tfw.mean(),
        "layer_mean_downward_magnitude_all_ug_m2_s": -tfw.mean(),
        "layer_mean_downward_only_ug_m2_s": np.maximum(-tfw, 0).mean(),
        "layer_fraction_downward": (tfw < 0).mean(),
        "layer_mean_o3_ug_m3": sub["o3_concentration_ug_m3"].mean(),
        "layer_mean_w_m_s": sub["w_m_s"].mean(),
    }


def symmetric_norm(datasets: list[pd.DataFrame]) -> tuple[float, TwoSlopeNorm]:
    values = []
    for df in datasets:
        arr = df["tfw_upward_ug_m2_s"].to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(arr)
    all_values = np.concatenate(values)
    vmax = float(np.nanpercentile(np.abs(all_values), 98))
    vmax = max(vmax, 0.1)
    return vmax, TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def plot_comparison(datasets: list[tuple[str, pd.DataFrame]], output: Path, title: str) -> Path:
    vmax, norm = symmetric_norm([df for _, df in datasets])
    nrows = len(datasets)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=2,
        figsize=(11.4, 4.5 if nrows == 1 else 4.1 * nrows),
        dpi=260,
        gridspec_kw={"width_ratios": [1.35, 0.8]},
        sharey="row",
    )
    if len(datasets) == 1:
        axes = np.asarray(axes).reshape(1, 2)

    mesh = None
    for row, (label, df) in enumerate(datasets):
        matrix = diurnal_matrix(df)
        hours = matrix.columns.to_numpy(dtype=float)
        heights_km = matrix.index.to_numpy(dtype=float) / 1000.0
        values = matrix.to_numpy(dtype=float)

        ax = axes[row, 0]
        mesh = ax.pcolormesh(
            hours,
            heights_km,
            values,
            shading="auto",
            cmap="RdBu_r",
            norm=norm,
        )
        ax.axhspan(LAYER_MIN_M / 1000.0, LAYER_MAX_M / 1000.0, color="gray", alpha=0.08)
        ax.set_title(f"{label}: diurnal mean", fontsize=11, fontweight="bold")
        ax.set_xlabel("Beijing time (hour)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Height (km)", fontsize=10, fontweight="bold")
        ax.set_xticks(HOURS)
        ax.tick_params(labelsize=8, direction="in")
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)

        prof = daytime_profile(df)
        axp = axes[row, 1]
        axp.plot(prof["tfw_mean_ug_m2_s"], prof["height_m"] / 1000.0, color="#333333", linewidth=1.8, marker="o", markersize=2.6)
        axp.axvline(0, color="gray", linestyle="--", linewidth=1.2)
        axp.axhspan(LAYER_MIN_M / 1000.0, LAYER_MAX_M / 1000.0, color="gray", alpha=0.08)
        axp.set_xlim(-vmax, vmax)
        axp.set_ylim(0, HEIGHT_MAX_M / 1000.0)
        axp.set_title("Daytime mean profile", fontsize=11, fontweight="bold")
        axp.set_xlabel("TFw (+up / -down)\n(ug m-2 s-1)", fontsize=10, fontweight="bold")
        axp.tick_params(labelsize=8, direction="in")
        for spine in axp.spines.values():
            spine.set_linewidth(1.1)

    cax = fig.add_axes([0.92, 0.18, 0.015, 0.66])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("TFw = w x O3 (+up / -down)\n(ug m-2 s-1)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.035,
        "Formula follows the paper's vertical transport flux: TFw = w x C. Positive is upward; negative is downward. Gray band: 0.2-1.8 km.",
        ha="center",
        fontsize=8.5,
    )
    top = 0.84 if nrows == 1 else 0.90
    bottom = 0.18 if nrows == 1 else 0.11
    fig.subplots_adjust(left=0.08, right=0.90, top=top, bottom=bottom, hspace=0.40, wspace=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def write_summary(path: Path, summaries: list[dict[str, object]], polluted_summaries: list[dict[str, object]]) -> None:
    notes = pd.DataFrame(
        [
            {
                "item": "paper_formula",
                "description": "Vertical transport flux follows TFw = w x C, where w is geometric vertical velocity (m/s) and C is O3 mass concentration (ug/m3). Positive TFw is upward; negative TFw is downward.",
            },
            {
                "item": "ERA5_variables",
                "description": "Used ERA5 pressure-level vertical velocity converted from omega (Pa/s) to geometric w (m/s), plus pressure/temperature/humidity during preprocessing.",
            },
            {
                "item": "difference_from_previous_Feff",
                "description": "This is not F_eff = -K_eff x dO3/dz. Figure 4 in the paper uses vertical transport flux TFw = w x C.",
            },
            {
                "item": "polluted_day_definition",
                "description": f"Polluted days in the extra polluted-day plots are defined by near-surface 0-100 m daily maximum O3 > {O3_THRESHOLD_UG_M3:.0f} ug/m3. If no strict polluted day exists, the plot uses the max near-surface O3 day as fallback and marks it in the summary.",
            },
            {
                "item": "BLH",
                "description": "Boundary-layer height line is not overlaid because boundary_layer_height was not among the downloaded ERA5 pressure-level variables.",
            },
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        notes.to_excel(writer, sheet_name="notes", index=False)
        pd.DataFrame(summaries).to_excel(writer, sheet_name="layer_summary", index=False)
        pd.DataFrame(polluted_summaries).to_excel(writer, sheet_name="polluted_selection", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    era5_datasets: list[tuple[str, pd.DataFrame]] = []
    summaries: list[dict[str, object]] = []
    polluted_era5_datasets: list[tuple[str, pd.DataFrame]] = []
    polluted_summaries: list[dict[str, object]] = []
    for cfg in CITY_FILES:
        df = read_era5_site(cfg["path"], cfg["label"])
        era5_datasets.append((cfg["short_label"], df))
        summaries.append(layer_summary(df, cfg["label"]))
        selected, polluted_summary = select_polluted_or_fallback(df, cfg["label"])
        suffix = "fallback" if polluted_summary["using_fallback"] else f"polluted n={polluted_summary['strict_polluted_days']}"
        polluted_era5_datasets.append((f"{cfg['short_label']} {suffix}", selected))
        polluted_summaries.append(polluted_summary)

    main_out = OUT_DIR / "\u8bba\u6587Figure4\u540c\u65b9\u6cd5_O3\u5782\u76f4\u8f93\u9001\u901a\u91cf_\u4e0a\u6d77\u5408\u80a5ERA5\u5bf9\u6bd4.png"
    plot_comparison(
        era5_datasets,
        main_out,
        "Paper Figure 4 Method: O3 Vertical Transport Flux\nERA5 O3, common 2025-03 to 2025-12",
    )
    polluted_main_out = OUT_DIR / "\u8bba\u6587Figure4\u540c\u65b9\u6cd5_O3\u5782\u76f4\u8f93\u9001\u901a\u91cf_\u8fd1\u5730\u9762160\u6c61\u67d3\u65e5\u7248_\u4e0a\u6d77\u5408\u80a5ERA5\u5bf9\u6bd4.png"
    plot_comparison(
        polluted_era5_datasets,
        polluted_main_out,
        "Paper Figure 4 Method: O3 Vertical Transport Flux\nERA5 O3, polluted-day criterion: 0-100 m max > 160 ug/m3",
    )

    hefei_local = read_hefei_local_o3_with_era5_w()
    summaries.append(layer_summary(hefei_local, "Hefei local O3 profile + ERA5 w"))
    hefei_local_polluted, local_polluted_summary = select_polluted_or_fallback(
        hefei_local, "Hefei local O3 profile + ERA5 w"
    )
    polluted_summaries.append(local_polluted_summary)
    local_out = OUT_DIR / "\u8bba\u6587Figure4\u540c\u65b9\u6cd5_O3\u5782\u76f4\u8f93\u9001\u901a\u91cf_\u5408\u80a5\u672c\u5730O3\u7248.png"
    plot_comparison(
        [("Hefei local O3", hefei_local)],
        local_out,
        "Paper Figure 4 Method: O3 Vertical Transport Flux\nHefei local O3 profile + ERA5 w",
    )
    local_polluted_out = OUT_DIR / "\u8bba\u6587Figure4\u540c\u65b9\u6cd5_O3\u5782\u76f4\u8f93\u9001\u901a\u91cf_\u8fd1\u5730\u9762160\u6c61\u67d3\u65e5\u7248_\u5408\u80a5\u672c\u5730O3.png"
    local_suffix = (
        "fallback"
        if local_polluted_summary["using_fallback"]
        else f"polluted n={local_polluted_summary['strict_polluted_days']}"
    )
    plot_comparison(
        [(f"Hefei local O3 {local_suffix}", hefei_local_polluted)],
        local_polluted_out,
        "Paper Figure 4 Method: O3 Vertical Transport Flux\nHefei local O3 polluted-day criterion: 0-100 m max > 160 ug/m3",
    )

    summary_path = OUT_DIR / "\u8bba\u6587Figure4\u540c\u65b9\u6cd5_O3\u5782\u76f4\u8f93\u9001\u901a\u91cf_\u8bf4\u660e.xlsx"
    write_summary(summary_path, summaries, polluted_summaries)

    print(f"Wrote {main_out}")
    print(f"Wrote {polluted_main_out}")
    print(f"Wrote {local_out}")
    print(f"Wrote {local_polluted_out}")
    print(f"Wrote {summary_path}")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(polluted_summaries).to_string(index=False))


if __name__ == "__main__":
    main()
