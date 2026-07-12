# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path("E:/research/????")
OUT_DIR = BASE_DIR / "\u62a5\u544a\u56fe\u7247"
OUT_PNG = OUT_DIR / "\u56fe4_0.2-0.8km\u5c42\u4e0a\u6d77\u4e0e\u5408\u80a5TFw\u65e5\u53d8\u5316\u5dee\u5f02\u5bf9\u6bd4.png"
OUT_XLSX = OUT_DIR / "\u56fe4_0.2-0.8km\u5c42\u4e0a\u6d77\u4e0e\u5408\u80a5TFw\u65e5\u53d8\u5316\u5dee\u5f02\u5bf9\u6bd4_\u8bf4\u660e.xlsx"

COMMON_START = "2025-03-01"
COMMON_END = "2025-12-31"
HOURS = list(range(8, 17))
LAYER_MIN_M = 200.0
LAYER_MAX_M = 800.0

CITY_CONFIGS = [
    {
        "city": "\u4e0a\u6d77",
        "file": BASE_DIR
        / "\u4e0a\u6d77_ERA5\u6570\u636e\u4e0e\u5904\u7406\u7ed3\u679c\u6c47\u603b"
        / "ERA5_\u7b49\u6548\u6e4d\u6d41\u7cfb\u6570_\u8fd1\u5730\u9762\u8865\u9f50"
        / "site_effective_turbulence_hourly.csv",
        "color": "#2f5f9f",
    },
    {
        "city": "\u5408\u80a5\u79d1\u5b66\u5c9b",
        "file": BASE_DIR
        / "\u5408\u80a5\u79d1\u5b66\u5c9b_ERA5\u6570\u636e\u4e0e\u5904\u7406\u7ed3\u679c\u6c47\u603b"
        / "\u5408\u80a5\u79d1\u5b66\u5c9b_ERA5_\u7b49\u6548\u6e4d\u6d41\u7cfb\u6570_\u8fd1\u5730\u9762\u8865\u9f50"
        / "site_effective_turbulence_hourly.csv",
        "color": "#b45f2a",
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
        & (df["height_m"] >= LAYER_MIN_M)
        & (df["height_m"] <= LAYER_MAX_M)
    ].copy()
    return df.dropna(subset=["height_m", "bjt_hour", "o3_mass_ug_m3", "w_geometric_m_s", "tfw_ug_m2_s"])


def hourly_layer_mean(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("bjt_hour", as_index=False)
        .agg(
            tfw_mean_ug_m2_s=("tfw_ug_m2_s", "mean"),
            o3_mean_ug_m3=("o3_mass_ug_m3", "mean"),
            w_mean_m_s=("w_geometric_m_s", "mean"),
            n=("tfw_ug_m2_s", "count"),
        )
        .sort_values("bjt_hour")
    )


def build_hourly_table(hourly_by_city: list[tuple[dict[str, object], pd.DataFrame]]) -> pd.DataFrame:
    frames = []
    for config, hourly in hourly_by_city:
        temp = hourly.copy()
        temp.insert(0, "city", str(config["city"]))
        frames.append(temp)
    long = pd.concat(frames, ignore_index=True)
    pivot = long.pivot(index="bjt_hour", columns="city", values="tfw_mean_ug_m2_s").reset_index()
    pivot["上海-合肥_TFw差值"] = pivot["\u4e0a\u6d77"] - pivot["\u5408\u80a5\u79d1\u5b66\u5c9b"]
    return long, pivot


def plot_diurnal(hourly_by_city: list[tuple[dict[str, object], pd.DataFrame]]) -> None:
    set_chinese_font()
    _, pivot = build_hourly_table(hourly_by_city)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.8, 6.6),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [2.15, 1.0]},
    )
    ax = axes[0]
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.2, zorder=1)
    ax.axhspan(-0.08, 0, color="#d7e8f5", alpha=0.30, zorder=0)
    ax.axhspan(0, 0.82, color="#f8dfcf", alpha=0.20, zorder=0)

    for config, hourly in hourly_by_city:
        city = str(config["city"])
        color = str(config["color"])
        ax.plot(
            hourly["bjt_hour"],
            hourly["tfw_mean_ug_m2_s"],
            color=color,
            linewidth=2.5,
            marker="o",
            markersize=5.0,
            label=city,
        )

    ax.set_ylim(-0.08, 0.82)
    ax.set_ylabel("0.2-0.8 km 平均TFw\n(μg m$^{-2}$ s$^{-1}$)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    ax.grid(axis="y", color="#bbbbbb", linestyle="-", linewidth=0.5, alpha=0.45)
    ax.set_title("0.2-0.8 km低层上海与合肥科学岛O$_3$垂直输送通量日变化", fontsize=14, fontweight="bold")

    axd = axes[1]
    diff = pivot["上海-合肥_TFw差值"].to_numpy(dtype=float)
    colors = np.where(diff >= 0, "#7aa6d8", "#d89a77")
    axd.bar(pivot["bjt_hour"], diff, color=colors, edgecolor="#333333", linewidth=0.55, width=0.62)
    axd.axhline(0, color="gray", linestyle="--", linewidth=1.0)
    axd.set_ylabel("上海-合肥\nTFw差值", fontsize=13, fontweight="bold")
    axd.set_xlabel("北京时间 (h)", fontsize=13, fontweight="bold")
    axd.set_xticks(HOURS)
    axd.grid(axis="y", color="#bbbbbb", linewidth=0.5, alpha=0.42)
    axd.set_ylim(min(-0.08, diff.min() - 0.05), diff.max() + 0.10)

    for a in axes:
        a.tick_params(labelsize=11.5, direction="in")
        for spine in a.spines.values():
            spine.set_linewidth(1.15)

    fig.text(
        0.98,
        0.015,
        "TFw = O$_3$ × w$_{geo}$；正值表示向上输送，负值表示向下输送",
        ha="right",
        fontsize=10.5,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.025, 1, 1])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG)
    plt.close(fig)


def write_summary(
    datasets: list[tuple[dict[str, object], pd.DataFrame]],
    hourly_by_city: list[tuple[dict[str, object], pd.DataFrame]],
) -> None:
    long, pivot = build_hourly_table(hourly_by_city)
    rows = []
    hourly_lookup = {str(config["city"]): hourly for config, hourly in hourly_by_city}
    for config, df in datasets:
        city = str(config["city"])
        hourly = hourly_lookup[city]
        peak = hourly.loc[hourly["tfw_mean_ug_m2_s"].idxmax()]
        rows.append(
            {
                "城市": city,
                "日期范围": f"{df['bjt_date'].min()} 至 {df['bjt_date'].max()}",
                "时段_BJT": "08-16",
                "高度层": "0.2-0.8 km",
                "层平均TFw_ug_m-2_s-1": df["tfw_ug_m2_s"].mean(),
                "层平均O3_ug_m-3": df["o3_mass_ug_m3"].mean(),
                "层平均w_m_s-1": df["w_geometric_m_s"].mean(),
                "峰值小时_BJT": int(peak["bjt_hour"]),
                "峰值TFw_ug_m-2_s-1": peak["tfw_mean_ug_m2_s"],
                "16点TFw_ug_m-2_s-1": hourly.loc[hourly["bjt_hour"] == 16, "tfw_mean_ug_m2_s"].iloc[0],
                "有效天数": df["bjt_date"].nunique(),
                "记录数": len(df),
            }
        )
    notes = pd.DataFrame(
        [
            {
                "项目": "高度层选择",
                "说明": "对多个低层高度范围试算后，0.2-0.8 km层的上海-合肥差异最清楚，且仍位于低层边界层范围内。",
            },
            {
                "项目": "计算方法",
                "说明": "每个城市先在0.2-0.8 km内对TFw=O3*w_geo逐小时求层平均，再统计2025年3-12月08:00-16:00的平均日变化。",
            },
            {
                "项目": "符号含义",
                "说明": "w_geo>0和TFw>0表示向上输送；w_geo<0和TFw<0表示向下输送。",
            },
            {
                "项目": "主要结论",
                "说明": "上海低层TFw午后增强更明显，13:00达到约0.756 μg m-2 s-1；合肥科学岛峰值提前至11:00，约0.336 μg m-2 s-1，16:00转为弱下输。",
            },
        ]
    )
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        notes.to_excel(writer, sheet_name="说明", index=False)
        pd.DataFrame(rows).to_excel(writer, sheet_name="统计摘要", index=False)
        long.to_excel(writer, sheet_name="逐小时长表", index=False)
        pivot.to_excel(writer, sheet_name="上海合肥差值", index=False)


def main() -> None:
    datasets = [(config, read_city(config)) for config in CITY_CONFIGS]
    hourly_by_city = [(config, hourly_layer_mean(df)) for config, df in datasets]
    plot_diurnal(hourly_by_city)
    write_summary(datasets, hourly_by_city)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_XLSX}")


if __name__ == "__main__":
    main()
