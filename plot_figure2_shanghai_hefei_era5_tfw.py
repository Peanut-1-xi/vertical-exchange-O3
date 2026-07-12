# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


BASE_DIR = Path("E:/research/垂直交换")
OUT_DIR = BASE_DIR / "报告图片"
OUT_PNG = OUT_DIR / "图2_上海与合肥科学岛ERA5臭氧垂直输送通量TFw对比.png"
OUT_XLSX = OUT_DIR / "图2_上海与合肥科学岛ERA5臭氧垂直输送通量TFw对比_说明.xlsx"

COMMON_START = "2025-03-01"
COMMON_END = "2025-12-31"
HOURS = list(range(8, 17))
HEIGHT_MAX_M = 4000.0
PROFILE_HOURS = HOURS

CITY_CONFIGS = [
    {
        "city": "上海",
        "site": "上海环境科学研究院附近 ERA5 站点格点",
        "file": BASE_DIR
        / "上海_ERA5数据与处理结果汇总"
        / "ERA5_等效湍流系数_近地面补齐"
        / "site_effective_turbulence_hourly.csv",
    },
    {
        "city": "合肥科学岛",
        "site": "合肥科学岛附近 ERA5 站点格点",
        "file": BASE_DIR
        / "合肥科学岛_ERA5数据与处理结果汇总"
        / "合肥科学岛_ERA5_等效湍流系数_近地面补齐"
        / "site_effective_turbulence_hourly.csv",
    },
]


def set_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def read_city(config: dict[str, object]) -> pd.DataFrame:
    cols = [
        "bjt_time",
        "bjt_date",
        "bjt_hour",
        "height_agl_m",
        "o3_mass_ug_m3",
        "w_geometric_m_s",
    ]
    df = pd.read_csv(config["file"], encoding="utf-8-sig", usecols=cols)
    df["bjt_time"] = pd.to_datetime(df["bjt_time"])
    df["bjt_date"] = df["bjt_time"].dt.date.astype(str)
    df["bjt_hour"] = df["bjt_time"].dt.hour
    df["height_m"] = pd.to_numeric(df["height_agl_m"], errors="coerce")
    df["o3_mass_ug_m3"] = pd.to_numeric(df["o3_mass_ug_m3"], errors="coerce")
    df["w_geometric_m_s"] = pd.to_numeric(df["w_geometric_m_s"], errors="coerce")
    df["tfw_ug_m2_s"] = df["w_geometric_m_s"] * df["o3_mass_ug_m3"]
    df["city"] = str(config["city"])
    df["site"] = str(config["site"])
    df = df[
        (df["bjt_time"] >= pd.Timestamp(COMMON_START))
        & (df["bjt_time"] <= pd.Timestamp(COMMON_END) + pd.Timedelta(days=1))
        & (df["bjt_hour"].isin(HOURS))
        & (df["height_m"] >= 0)
        & (df["height_m"] <= HEIGHT_MAX_M)
    ].copy()
    return df.dropna(subset=["height_m", "bjt_hour", "o3_mass_ug_m3", "w_geometric_m_s", "tfw_ug_m2_s"])


def diurnal_matrix(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index="height_m",
        columns="bjt_hour",
        values="tfw_ug_m2_s",
        aggfunc="mean",
    )
    return pivot.reindex(index=sorted(pivot.index), columns=HOURS)


def daytime_profile(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["bjt_hour"].isin(PROFILE_HOURS)]
    return (
        sub.groupby("height_m", as_index=False)
        .agg(
            tfw_mean=("tfw_ug_m2_s", "mean"),
            tfw_median=("tfw_ug_m2_s", "median"),
            o3_mean=("o3_mass_ug_m3", "mean"),
            w_mean=("w_geometric_m_s", "mean"),
            n=("tfw_ug_m2_s", "count"),
        )
        .sort_values("height_m")
    )


def matrix_scale(datasets: list[pd.DataFrame]) -> tuple[float, TwoSlopeNorm]:
    blocks = []
    for df in datasets:
        mat = diurnal_matrix(df).to_numpy(dtype=float)
        values = mat[np.isfinite(mat)]
        if values.size:
            blocks.append(values)
    all_values = np.concatenate(blocks)
    vmax = float(np.nanpercentile(np.abs(all_values), 98))
    vmax = max(vmax, 0.25)
    # Rounded scale makes the colorbar easier to read in a report.
    vmax = float(np.ceil(vmax * 10) / 10)
    return vmax, TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def plot_figure(datasets: list[pd.DataFrame]) -> None:
    set_chinese_font()
    vmax, norm = matrix_scale(datasets)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(11.5, 7.6),
        dpi=300,
        gridspec_kw={"width_ratios": [1.45, 0.72]},
        sharey="row",
    )

    mesh = None
    colors = {"上海": "#2f5f9f", "合肥科学岛": "#b05a2a"}
    for row, df in enumerate(datasets):
        city = str(df["city"].iloc[0])
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
        ax.axhline(1.0, color="k", linewidth=0.7, alpha=0.25)
        ax.set_title(f"({chr(97 + row * 2)}) {city}：TFw 时间-高度分布", fontsize=14, fontweight="bold")
        ax.set_ylabel("高度 (km)", fontsize=13, fontweight="bold")
        ax.set_xlabel("北京时间 (h)", fontsize=13, fontweight="bold")
        ax.set_xticks(HOURS)
        ax.set_ylim(0, HEIGHT_MAX_M / 1000.0)
        ax.tick_params(labelsize=11, direction="in")
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)

        profile = daytime_profile(df)
        axp = axes[row, 1]
        axp.plot(
            profile["tfw_mean"],
            profile["height_m"] / 1000.0,
            color=colors.get(city, "#333333"),
            linewidth=2.0,
            marker="o",
            markersize=3.0,
        )
        axp.axvline(0, color="gray", linestyle="--", linewidth=1.2)
        axp.axhline(1.0, color="k", linewidth=0.7, alpha=0.25)
        axp.set_xlim(-vmax, vmax)
        axp.set_ylim(0, HEIGHT_MAX_M / 1000.0)
        axp.set_title(f"({chr(98 + row * 2)}) {city}：日间平均廓线", fontsize=14, fontweight="bold")
        axp.set_xlabel("TFw (μg m$^{-2}$ s$^{-1}$)", fontsize=13, fontweight="bold")
        axp.tick_params(labelsize=11, direction="in")
        for spine in axp.spines.values():
            spine.set_linewidth(1.1)

    cax = fig.add_axes([0.91, 0.20, 0.018, 0.62])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("TFw = O$_3$ × w$_{geo}$\n(+上 / -下)\n(μg m$^{-2}$ s$^{-1}$)", fontsize=12)
    cb.ax.tick_params(labelsize=11)
    fig.subplots_adjust(left=0.075, right=0.885, top=0.94, bottom=0.075, hspace=0.42, wspace=0.25)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG)
    plt.close(fig)


def write_summary(datasets: list[pd.DataFrame]) -> None:
    rows = []
    for df in datasets:
        profile = daytime_profile(df)
        layer = df[(df["height_m"] >= 0) & (df["height_m"] <= HEIGHT_MAX_M) & df["bjt_hour"].isin(HOURS)]
        rows.append(
            {
                "城市": df["city"].iloc[0],
                "站点说明": df["site"].iloc[0],
                "开始日期": df["bjt_date"].min(),
                "结束日期": df["bjt_date"].max(),
                "有效天数": df["bjt_date"].nunique(),
                "记录数": len(df),
                "高度范围_m": f"{df['height_m'].min():.0f}-{df['height_m'].max():.0f}",
                "小时_BJT": ",".join(map(str, sorted(df["bjt_hour"].unique()))),
                "TFw平均_ug_m-2_s-1": layer["tfw_ug_m2_s"].mean(),
                "TFw中位数_ug_m-2_s-1": layer["tfw_ug_m2_s"].median(),
                "向上输送比例": (layer["tfw_ug_m2_s"] > 0).mean(),
                "日间平均廓线层数": len(profile),
            }
        )
    notes = pd.DataFrame(
        [
            {
                "项目": "公式",
                "说明": "TFw = O3 × w_geo；w_geo 为向上为正的几何垂直速度(m/s)，O3 为 ERA5 臭氧质量浓度(μg/m3)。",
            },
            {
                "项目": "符号",
                "说明": "TFw>0 表示向上输送；TFw<0 表示向下输送。",
            },
            {
                "项目": "时间范围",
                "说明": "为保证上海和合肥科学岛可比性，统一采用 2025-03-01 至 2025-12-31 的共同可用时段，北京时间 08:00-16:00。",
            },
        ]
    )
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        notes.to_excel(writer, sheet_name="说明", index=False)
        pd.DataFrame(rows).to_excel(writer, sheet_name="统计", index=False)


def main() -> None:
    datasets = [read_city(config) for config in CITY_CONFIGS]
    plot_figure(datasets)
    write_summary(datasets)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_XLSX}")


if __name__ == "__main__":
    main()
