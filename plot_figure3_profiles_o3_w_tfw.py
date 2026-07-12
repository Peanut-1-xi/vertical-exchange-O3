# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path("E:/research/垂直交换")
OUT_DIR = BASE_DIR / "报告图片"
OUT_PNG = OUT_DIR / "图3_上海与合肥不同高度层O3_w_TFw平均廓线对比.png"
OUT_XLSX = OUT_DIR / "图3_上海与合肥不同高度层O3_w_TFw平均廓线对比_说明.xlsx"

COMMON_START = "2025-03-01"
COMMON_END = "2025-12-31"
HOURS = list(range(8, 17))
HEIGHT_MIN_M = 0.0
HEIGHT_MAX_M = 4000.0
KEY_LAYER_MIN_M = 200.0
KEY_LAYER_MAX_M = 1800.0

CITY_CONFIGS = [
    {
        "city": "上海",
        "file": BASE_DIR
        / "上海_ERA5数据与处理结果汇总"
        / "ERA5_等效湍流系数_近地面补齐"
        / "site_effective_turbulence_hourly.csv",
        "color": "#2f5f9f",
    },
    {
        "city": "合肥科学岛",
        "file": BASE_DIR
        / "合肥科学岛_ERA5数据与处理结果汇总"
        / "合肥科学岛_ERA5_等效湍流系数_近地面补齐"
        / "site_effective_turbulence_hourly.csv",
        "color": "#b05a2a",
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
    df["tfw_ug_m2_s"] = df["o3_mass_ug_m3"] * df["w_geometric_m_s"]
    df["city"] = str(config["city"])
    df = df[
        (df["bjt_time"] >= pd.Timestamp(COMMON_START))
        & (df["bjt_time"] <= pd.Timestamp(COMMON_END) + pd.Timedelta(days=1))
        & (df["bjt_hour"].isin(HOURS))
        & (df["height_m"] >= HEIGHT_MIN_M)
        & (df["height_m"] <= HEIGHT_MAX_M)
    ].copy()
    return df.dropna(subset=["height_m", "o3_mass_ug_m3", "w_geometric_m_s", "tfw_ug_m2_s"])


def profile_mean(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("height_m", as_index=False)
        .agg(
            o3_mean_ug_m3=("o3_mass_ug_m3", "mean"),
            w_mean_m_s=("w_geometric_m_s", "mean"),
            tfw_mean_ug_m2_s=("tfw_ug_m2_s", "mean"),
            n=("tfw_ug_m2_s", "count"),
        )
        .sort_values("height_m")
    )


def layer_stats(df: pd.DataFrame) -> dict[str, object]:
    layer = df[(df["height_m"] >= KEY_LAYER_MIN_M) & (df["height_m"] <= KEY_LAYER_MAX_M)]
    full = df[(df["height_m"] >= HEIGHT_MIN_M) & (df["height_m"] <= HEIGHT_MAX_M)]
    return {
        "城市": df["city"].iloc[0],
        "日期范围": f"{df['bjt_date'].min()}至{df['bjt_date'].max()}",
        "时段_BJT": "08-16",
        "关键高度层": "0.2-1.8 km",
        "关键层TFw平均_ug_m-2_s-1": layer["tfw_ug_m2_s"].mean(),
        "关键层O3平均_ug_m-3": layer["o3_mass_ug_m3"].mean(),
        "关键层w平均_m_s-1": layer["w_geometric_m_s"].mean(),
        "0-4km_TFw平均_ug_m-2_s-1": full["tfw_ug_m2_s"].mean(),
        "0-4km_O3平均_ug_m-3": full["o3_mass_ug_m3"].mean(),
        "0-4km_w平均_m_s-1": full["w_geometric_m_s"].mean(),
        "有效天数": df["bjt_date"].nunique(),
        "记录数": len(df),
    }


def shade_key_layer(ax: plt.Axes) -> None:
    ax.axhspan(
        KEY_LAYER_MIN_M / 1000.0,
        KEY_LAYER_MAX_M / 1000.0,
        color="#d9d9d9",
        alpha=0.28,
        zorder=0,
    )


def plot_profiles(datasets: list[tuple[dict[str, object], pd.DataFrame]]) -> None:
    set_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.4), dpi=300, sharey=True)

    panel_defs = [
        {
            "title": "(a) O$_3$平均廓线",
            "x": "o3_mean_ug_m3",
            "xlabel": "O$_3$ (μg m$^{-3}$)",
            "zero": False,
            "scale": 1.0,
        },
        {
            "title": "(b) 几何垂直速度 w$_{geo}$（向上为正）",
            "x": "w_mean_m_s",
            "xlabel": "w (10$^{-3}$ m s$^{-1}$)",
            "zero": True,
            "scale": 1000.0,
        },
        {
            "title": "(c) 垂直输送通量 TFw",
            "x": "tfw_mean_ug_m2_s",
            "xlabel": "TFw (μg m$^{-2}$ s$^{-1}$)",
            "zero": True,
            "scale": 1.0,
        },
    ]

    for ax, panel in zip(axes, panel_defs):
        shade_key_layer(ax)
        for config, df in datasets:
            prof = profile_mean(df)
            y = prof["height_m"].to_numpy(dtype=float) / 1000.0
            x = prof[str(panel["x"])].to_numpy(dtype=float) * float(panel["scale"])
            ax.plot(
                x,
                y,
                color=str(config["color"]),
                linewidth=2.1,
                marker="o",
                markersize=3.0,
                label=str(config["city"]),
            )
        if panel["zero"]:
            ax.axvline(0, color="gray", linestyle="--", linewidth=1.2, zorder=1)
        ax.axhline(KEY_LAYER_MIN_M / 1000.0, color="gray", linewidth=0.7, alpha=0.7)
        ax.axhline(KEY_LAYER_MAX_M / 1000.0, color="gray", linewidth=0.7, alpha=0.7)
        ax.set_title(str(panel["title"]), fontsize=14, fontweight="bold")
        ax.set_xlabel(str(panel["xlabel"]), fontsize=13, fontweight="bold")
        ax.set_ylim(0, HEIGHT_MAX_M / 1000.0)
        ax.tick_params(labelsize=11, direction="in")
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    axes[0].set_ylabel("高度 (km)", fontsize=13, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.975, 0.985),
        ncol=2,
        frameon=False,
        fontsize=12,
        handlelength=2.6,
        columnspacing=1.2,
    )
    axes[2].text(
        0.98,
        0.50,
        "阴影：0.2-1.8 km",
        transform=axes[2].transAxes,
        rotation=90,
        ha="right",
        va="center",
        fontsize=11,
        color="#555555",
    )

    fig.subplots_adjust(left=0.075, right=0.98, top=0.80, bottom=0.15, wspace=0.22)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG)
    plt.close(fig)


def write_summary(datasets: list[tuple[dict[str, object], pd.DataFrame]]) -> None:
    stats = pd.DataFrame([layer_stats(df) for _, df in datasets])
    notes = pd.DataFrame(
        [
            {
                "项目": "数据",
                "说明": "统一采用上海与合肥科学岛共同可用的 2025年3-12月 ERA5 O3质量浓度与ERA5几何垂直速度；w_geo定义为向上为正。",
            },
            {
                "项目": "时段和高度",
                "说明": "北京时间08:00-16:00，垂直范围0-4 km；图中灰色阴影为0.2-1.8 km关键层。",
            },
            {
                "项目": "公式",
                "说明": "TFw = O3 × w_geo；TFw>0表示向上输送，TFw<0表示向下输送。",
            },
        ]
    )
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        notes.to_excel(writer, sheet_name="说明", index=False)
        stats.to_excel(writer, sheet_name="层平均统计", index=False)


def main() -> None:
    datasets = [(config, read_city(config)) for config in CITY_CONFIGS]
    plot_profiles(datasets)
    write_summary(datasets)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_XLSX}")


if __name__ == "__main__":
    main()
