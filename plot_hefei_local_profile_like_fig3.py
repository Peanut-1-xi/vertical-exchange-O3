# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


BASE_DIR = Path("E:/research/????")
PROFILE_DIR = (
    BASE_DIR
    / "OzoneProfile_result_inter"
    / "OzoneProfile_result_inter"
    / "xgbr_AODplusOzonePorfile_pre_BJT_hourly_mean"
)
ERA5_FILE = (
    BASE_DIR
    / "\u5408\u80a5\u79d1\u5b66\u5c9b_ERA5\u6570\u636e\u4e0e\u5904\u7406\u7ed3\u679c\u6c47\u603b"
    / "\u5408\u80a5\u79d1\u5b66\u5c9b_ERA5_\u7b49\u6548\u6e4d\u6d41\u7cfb\u6570_\u8fd1\u5730\u9762\u8865\u9f50"
    / "site_effective_turbulence_hourly.csv"
)
REPORT_DIR = BASE_DIR / "\u62a5\u544a\u56fe\u7247"
OUT_PNG = REPORT_DIR / "\u5408\u80a5\u672c\u5730\u5ed3\u7ebfO3_TFw\u65f6\u95f4\u9ad8\u5ea6\u5206\u5e03\u53ca\u65e5\u95f4\u5e73\u5747\u5ed3\u7ebf_\u7c7b\u56fe3.png"
OUT_XLSX = REPORT_DIR / "\u5408\u80a5\u672c\u5730\u5ed3\u7ebfO3_TFw\u7c7b\u56fe3\u7edf\u8ba1.xlsx"

HOURS = list(range(8, 17))
HEIGHT_MAX_M = 4000.0


def setup_matplotlib() -> None:
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


def read_local_o3() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    files = sorted(PROFILE_DIR.glob("SHHKY_*_OzoneProfile.txt"))
    if not files:
        raise FileNotFoundError(PROFILE_DIR)

    for path in files:
        wide = pd.read_csv(path, sep="\t")
        if "time" not in wide.columns:
            continue
        wide["bjt_time"] = pd.to_datetime(wide["time"], errors="coerce")
        wide = wide.drop(columns=["time"]).dropna(subset=["bjt_time"])

        height_cols: list[tuple[str, int]] = []
        for col in wide.columns:
            if col == "bjt_time":
                continue
            try:
                height_m = int(round(float(col) * 1000.0))
            except ValueError:
                continue
            if 0 <= height_m <= HEIGHT_MAX_M:
                height_cols.append((col, height_m))
        if not height_cols:
            continue

        long = wide.melt(
            id_vars="bjt_time",
            value_vars=[col for col, _ in height_cols],
            var_name="height_col",
            value_name="o3_local_ug_m3",
        )
        height_map = {col: height for col, height in height_cols}
        long["height_m"] = long["height_col"].map(height_map).astype(int)
        long["o3_local_ug_m3"] = pd.to_numeric(long["o3_local_ug_m3"], errors="coerce")
        long = long.dropna(subset=["o3_local_ug_m3"])
        frames.append(long[["bjt_time", "height_m", "o3_local_ug_m3"]])

    out = pd.concat(frames, ignore_index=True)
    out["bjt_date"] = out["bjt_time"].dt.date.astype(str)
    out["bjt_hour"] = out["bjt_time"].dt.hour
    return out[out["bjt_hour"].isin(HOURS)].copy()


def read_era5_w() -> pd.DataFrame:
    era5 = pd.read_csv(
        ERA5_FILE,
        encoding="utf-8-sig",
        usecols=["bjt_time", "height_agl_m", "w_geometric_m_s"],
    )
    era5["bjt_time"] = pd.to_datetime(era5["bjt_time"], errors="coerce")
    era5["height_m"] = pd.to_numeric(era5["height_agl_m"], errors="coerce").round()
    era5["w_geometric_m_s"] = pd.to_numeric(era5["w_geometric_m_s"], errors="coerce")
    era5 = era5.dropna(subset=["bjt_time", "height_m", "w_geometric_m_s"])
    era5["height_m"] = era5["height_m"].astype(int)
    return era5[["bjt_time", "height_m", "w_geometric_m_s"]]


def interpolate_w_to_local_grid(local: pd.DataFrame, era5: pd.DataFrame) -> pd.DataFrame:
    pivot = era5.pivot_table(
        index="bjt_time",
        columns="height_m",
        values="w_geometric_m_s",
        aggfunc="mean",
    ).sort_index()
    pivot = pivot.interpolate(axis=0, limit_direction="both")

    source_heights = np.asarray(pivot.columns, dtype=float)
    source_seconds = pivot.index.view("int64").astype(float) / 1e9
    target_heights = np.asarray(sorted(local["height_m"].unique()), dtype=float)
    target_times = pd.to_datetime(local["bjt_time"].drop_duplicates().sort_values())
    target_seconds = target_times.astype("int64").to_numpy(dtype=float) / 1e9

    rows: list[pd.DataFrame] = []
    for time_value, target_second in zip(target_times, target_seconds):
        values_at_time = []
        for height in pivot.columns:
            vals = pivot[height].to_numpy(dtype=float)
            ok = np.isfinite(vals)
            values_at_time.append(np.interp(target_second, source_seconds[ok], vals[ok]) if ok.sum() >= 2 else np.nan)
        values_at_time = np.asarray(values_at_time, dtype=float)
        ok = np.isfinite(values_at_time)
        if ok.sum() < 2:
            continue
        w_target = np.interp(target_heights, source_heights[ok], values_at_time[ok])
        rows.append(
            pd.DataFrame(
                {
                    "bjt_time": time_value,
                    "height_m": target_heights.astype(int),
                    "w_geometric_m_s": w_target,
                }
            )
        )

    w_grid = pd.concat(rows, ignore_index=True)
    merged = local.merge(w_grid, on=["bjt_time", "height_m"], how="inner")
    merged["tfw_ug_m2_s"] = merged["o3_local_ug_m3"] * merged["w_geometric_m_s"]
    return merged.dropna(subset=["tfw_ug_m2_s"])


def diurnal_matrix(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index="height_m",
        columns="bjt_hour",
        values="tfw_ug_m2_s",
        aggfunc="mean",
    )
    return pivot.reindex(index=sorted(pivot.index), columns=HOURS)


def daytime_profile(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("height_m", as_index=False)
        .agg(
            tfw_mean=("tfw_ug_m2_s", "mean"),
            tfw_median=("tfw_ug_m2_s", "median"),
            o3_mean=("o3_local_ug_m3", "mean"),
            w_mean=("w_geometric_m_s", "mean"),
            n=("tfw_ug_m2_s", "count"),
        )
        .sort_values("height_m")
    )


def plot_figure(df: pd.DataFrame) -> None:
    setup_matplotlib()
    matrix = diurnal_matrix(df)
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
    vmax = max(float(np.ceil(vmax * 10) / 10), 0.5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    hours = matrix.columns.to_numpy(dtype=float)
    heights_km = matrix.index.to_numpy(dtype=float) / 1000.0
    profile = daytime_profile(df)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(11.5, 4.25),
        dpi=300,
        gridspec_kw={"width_ratios": [1.55, 0.75]},
        sharey=True,
    )

    mesh = axes[0].pcolormesh(hours, heights_km, values, shading="auto", cmap="RdBu_r", norm=norm)
    axes[0].axhline(1.0, color="k", linewidth=0.7, alpha=0.25)
    axes[0].set_title("合肥科学岛：本地O$_3$廓线TFw时间-高度分布", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("北京时间 (h)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("高度 (km)", fontsize=13, fontweight="bold")
    axes[0].set_xticks(HOURS)
    axes[0].set_ylim(0, HEIGHT_MAX_M / 1000.0)
    axes[0].tick_params(labelsize=11, direction="in")

    axes[1].plot(
        profile["tfw_mean"],
        profile["height_m"] / 1000.0,
        color="#b05a2a",
        linewidth=2.2,
        marker="o",
        markersize=3.2,
    )
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=1.2)
    axes[1].axhline(1.0, color="k", linewidth=0.7, alpha=0.25)
    axes[1].set_xlim(-vmax, vmax)
    axes[1].set_title("日间平均廓线", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("TFw ($\\mu$g m$^{-2}$ s$^{-1}$)", fontsize=13, fontweight="bold")
    axes[1].tick_params(labelsize=11, direction="in")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(1.15)

    cax = fig.add_axes([0.91, 0.18, 0.018, 0.68])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("TFw\n(+向上 / -向下)\n($\\mu$g m$^{-2}$ s$^{-1}$)", fontsize=12)
    cb.ax.tick_params(labelsize=11)
    fig.subplots_adjust(left=0.075, right=0.885, top=0.88, bottom=0.16, wspace=0.24)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG)
    plt.close(fig)


def write_summary(df: pd.DataFrame) -> None:
    profile = daytime_profile(df)
    summary = pd.DataFrame(
        [
            {
                "数据源": "合肥科学岛本地O3小时均值廓线 + 合肥ERA5几何垂直速度w",
                "本地廓线目录": str(PROFILE_DIR),
                "ERA5_w文件": str(ERA5_FILE),
                "开始时间": df["bjt_time"].min(),
                "结束时间": df["bjt_time"].max(),
                "有效日期数": df["bjt_date"].nunique(),
                "有效小时": ",".join(map(str, sorted(df["bjt_hour"].unique()))),
                "高度范围_m": f"{df['height_m'].min():.0f}-{df['height_m'].max():.0f}",
                "记录数": len(df),
                "TFw平均_ug_m-2_s-1": df["tfw_ug_m2_s"].mean(),
                "TFw中位数_ug_m-2_s-1": df["tfw_ug_m2_s"].median(),
                "向上输送比例": float((df["tfw_ug_m2_s"] > 0).mean()),
            }
        ]
    )
    hourly = (
        df.groupby(["bjt_hour", "height_m"], as_index=False)
        .agg(
            tfw_mean=("tfw_ug_m2_s", "mean"),
            o3_mean=("o3_local_ug_m3", "mean"),
            w_mean=("w_geometric_m_s", "mean"),
            n=("tfw_ug_m2_s", "count"),
        )
        .sort_values(["bjt_hour", "height_m"])
    )
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="说明", index=False)
        profile.to_excel(writer, sheet_name="日间平均廓线", index=False)
        hourly.to_excel(writer, sheet_name="小时-高度矩阵长表", index=False)


def main() -> None:
    local = read_local_o3()
    era5 = read_era5_w()
    df = interpolate_w_to_local_grid(local, era5)
    df = df[
        (df["bjt_time"] >= era5["bjt_time"].min())
        & (df["bjt_time"] <= era5["bjt_time"].max())
        & (df["height_m"] <= HEIGHT_MAX_M)
        & (df["bjt_hour"].isin(HOURS))
    ].copy()
    if df.empty:
        raise RuntimeError("No common local O3 and ERA5 w data after filtering.")
    plot_figure(df)
    write_summary(df)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_XLSX}")
    print(
        f"Common period: {df['bjt_time'].min()} to {df['bjt_time'].max()}, "
        f"days={df['bjt_date'].nunique()}, records={len(df)}"
    )


if __name__ == "__main__":
    main()
