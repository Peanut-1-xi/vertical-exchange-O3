# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


BASE_DIR = Path("E:/research/垂直交换")
REPORT_DIR = BASE_DIR / "报告图片"
PROFILE_DIR = (
    BASE_DIR
    / "OzoneProfile_result_inter"
    / "OzoneProfile_result_inter"
    / "xgbr_AODplusOzonePorfile_pre_BJT_hourly_mean"
)
ERA5_SITE_CSV = (
    BASE_DIR
    / "合肥科学岛_ERA5数据与处理结果汇总"
    / "合肥科学岛_ERA5_等效湍流系数_近地面补齐"
    / "site_effective_turbulence_hourly.csv"
)

FIG11 = REPORT_DIR / "图11_合肥科学岛典型污染日前后水平输送通量日变化对比.png"
FIG11A = REPORT_DIR / "图11a_合肥科学岛典型污染日前后TFh时间高度分布.png"
FIG12 = REPORT_DIR / "图12_合肥科学岛污染日与前后较好日水平输送通量垂直廓线对比.png"
SUMMARY_XLSX = REPORT_DIR / "4.3_合肥科学岛水平传输通量统计摘要.xlsx"

CASE_DATES = ["2025-03-06", "2025-03-07", "2025-03-08", "2025-03-10", "2025-03-11"]
POLLUTED_DATE = "2025-03-08"
PLOT_HOURS = list(range(8, 17))
PLOT_HOUR_TICKS = list(range(8, 17, 2))
TARGET_HEIGHTS_M = np.arange(0, 4000 + 1, 100, dtype=int)
KEY_MIN_M = 200
KEY_MAX_M = 1800


def set_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def interpolate_profiles_to_100m(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for bjt_time, profile in df.groupby("bjt_time"):
        profile = profile.sort_values("height_m")
        height = profile["height_m"].to_numpy(dtype=float)
        o3 = profile["o3_ug_m3"].to_numpy(dtype=float)
        ok = np.isfinite(height) & np.isfinite(o3)
        if ok.sum() < 2:
            continue
        height = height[ok]
        o3 = o3[ok]
        unique_h, unique_idx = np.unique(height, return_index=True)
        unique_o3 = o3[unique_idx]
        target = TARGET_HEIGHTS_M[(TARGET_HEIGHTS_M >= unique_h.min()) & (TARGET_HEIGHTS_M <= unique_h.max())]
        rows.append(
            pd.DataFrame(
                {
                    "bjt_time": pd.Timestamp(bjt_time),
                    "height_m": target.astype(int),
                    "o3_ug_m3": np.interp(target.astype(float), unique_h, unique_o3),
                }
            )
        )
    if not rows:
        raise RuntimeError("No local O3 profiles could be interpolated.")
    out = pd.concat(rows, ignore_index=True)
    out["bjt_date"] = out["bjt_time"].dt.date.astype(str)
    out["bjt_hour"] = out["bjt_time"].dt.hour
    return out


def read_local_o3() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(PROFILE_DIR.glob("*.txt")):
        wide = pd.read_csv(path, sep="\t")
        if wide.empty or "time" not in wide.columns:
            continue
        wide["bjt_time"] = pd.to_datetime(wide["time"], errors="coerce")
        wide = wide.drop(columns=["time"]).dropna(subset=["bjt_time"])
        height_cols = [col for col in wide.columns if col != "bjt_time"]
        long = wide.melt(id_vars="bjt_time", value_vars=height_cols, var_name="height_km", value_name="o3_ug_m3")
        long["height_km"] = pd.to_numeric(long["height_km"], errors="coerce")
        long["height_m"] = (long["height_km"] * 1000.0).round()
        long["o3_ug_m3"] = pd.to_numeric(long["o3_ug_m3"], errors="coerce")
        long = long.dropna(subset=["height_m", "o3_ug_m3"])
        long["height_m"] = long["height_m"].astype(int)
        frames.append(long[["bjt_time", "height_m", "o3_ug_m3"]])
    if not frames:
        raise FileNotFoundError(f"No local O3 profile txt files found in {PROFILE_DIR}")
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["height_m"] >= 0) & (df["height_m"] <= TARGET_HEIGHTS_M.max())].copy()
    return interpolate_profiles_to_100m(df)


def read_era5_wind() -> pd.DataFrame:
    cols = [
        "bjt_time",
        "bjt_hour",
        "height_agl_m",
        "u_wind_m_s",
        "v_wind_m_s",
        "w_geometric_m_s",
    ]
    era5 = pd.read_csv(ERA5_SITE_CSV, encoding="utf-8-sig", usecols=cols)
    era5["bjt_time"] = pd.to_datetime(era5["bjt_time"])
    era5["height_m"] = pd.to_numeric(era5["height_agl_m"], errors="coerce").round()
    era5["u_m_s"] = pd.to_numeric(era5["u_wind_m_s"], errors="coerce")
    era5["v_m_s"] = pd.to_numeric(era5["v_wind_m_s"], errors="coerce")
    era5["w_m_s"] = pd.to_numeric(era5["w_geometric_m_s"], errors="coerce")
    era5 = era5.dropna(subset=["height_m", "u_m_s", "v_m_s", "w_m_s"])
    era5["height_m"] = era5["height_m"].astype(int)
    return era5[["bjt_time", "bjt_hour", "height_m", "u_m_s", "v_m_s", "w_m_s"]]


def build_horizontal_flux() -> pd.DataFrame:
    local = read_local_o3()
    era5 = read_era5_wind()
    df = local.merge(era5, on=["bjt_time", "bjt_hour", "height_m"], how="left")
    df = df[df["bjt_date"].isin(CASE_DATES) & df["bjt_hour"].isin(PLOT_HOURS)].copy()
    df["wind_speed_m_s"] = np.sqrt(df["u_m_s"] ** 2 + df["v_m_s"] ** 2)
    df["TFu_ug_m2_s"] = df["o3_ug_m3"] * df["u_m_s"]
    df["TFv_ug_m2_s"] = df["o3_ug_m3"] * df["v_m_s"]
    df["TFh_ug_m2_s"] = np.sqrt(df["TFu_ug_m2_s"] ** 2 + df["TFv_ug_m2_s"] ** 2)
    df["TFw_ug_m2_s"] = df["o3_ug_m3"] * df["w_m_s"]
    return df


def key_layer(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["height_m"] >= KEY_MIN_M) & (df["height_m"] <= KEY_MAX_M)].copy()


def summarize_daily(df: pd.DataFrame) -> pd.DataFrame:
    layer = key_layer(df)
    out = (
        layer.groupby("bjt_date", as_index=False)
        .agg(
            o3_mean_ug_m3=("o3_ug_m3", "mean"),
            wind_speed_mean_m_s=("wind_speed_m_s", "mean"),
            TFu_mean_ug_m2_s=("TFu_ug_m2_s", "mean"),
            TFv_mean_ug_m2_s=("TFv_ug_m2_s", "mean"),
            TFh_mean_ug_m2_s=("TFh_ug_m2_s", "mean"),
            TFw_mean_ug_m2_s=("TFw_ug_m2_s", "mean"),
            records=("TFh_ug_m2_s", "count"),
        )
        .set_index("bjt_date")
        .reindex(CASE_DATES)
        .reset_index()
    )
    out["case_type"] = np.where(out["bjt_date"] == POLLUTED_DATE, "污染日", "较好日")
    out["abs_TFw_to_TFh_pct"] = out["TFw_mean_ug_m2_s"].abs() / out["TFh_mean_ug_m2_s"] * 100.0
    return out


def summarize_profiles(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    polluted = (
        df[df["bjt_date"] == POLLUTED_DATE]
        .groupby("height_m", as_index=False)
        .agg(
            o3_mean_ug_m3=("o3_ug_m3", "mean"),
            wind_speed_mean_m_s=("wind_speed_m_s", "mean"),
            TFu_mean_ug_m2_s=("TFu_ug_m2_s", "mean"),
            TFv_mean_ug_m2_s=("TFv_ug_m2_s", "mean"),
            TFh_mean_ug_m2_s=("TFh_ug_m2_s", "mean"),
            TFw_mean_ug_m2_s=("TFw_ug_m2_s", "mean"),
        )
    )
    good = (
        df[df["bjt_date"].isin([d for d in CASE_DATES if d != POLLUTED_DATE])]
        .groupby("height_m", as_index=False)
        .agg(
            o3_mean_ug_m3=("o3_ug_m3", "mean"),
            wind_speed_mean_m_s=("wind_speed_m_s", "mean"),
            TFu_mean_ug_m2_s=("TFu_ug_m2_s", "mean"),
            TFv_mean_ug_m2_s=("TFv_ug_m2_s", "mean"),
            TFh_mean_ug_m2_s=("TFh_ug_m2_s", "mean"),
            TFw_mean_ug_m2_s=("TFw_ug_m2_s", "mean"),
        )
    )
    return polluted, good


def plot_figure11(daily: pd.DataFrame) -> None:
    set_chinese_font()
    x = np.arange(len(daily))
    labels = pd.to_datetime(daily["bjt_date"]).dt.strftime("%m-%d").tolist()
    polluted_idx = CASE_DATES.index(POLLUTED_DATE)
    colors = ["#2b9c87" if d != POLLUTED_DATE else "#d75f2a" for d in daily["bjt_date"]]

    fig, axes = plt.subplots(3, 1, figsize=(9.4, 7.8), dpi=300, sharex=True)

    ax = axes[0]
    ax.axvspan(polluted_idx - 0.4, polluted_idx + 0.4, color="#f7c7c7", alpha=0.42, zorder=0)
    ax.bar(x, daily["TFh_mean_ug_m2_s"], color=colors, edgecolor="#333333", linewidth=0.7, width=0.62)
    ax.set_ylabel("TFh\n(µg m$^{-2}$ s$^{-1}$)", fontsize=12, fontweight="bold")
    ax.set_title("(a) 0.2−1.8 km层水平输送通量强度", loc="left", fontsize=13, fontweight="bold")

    ax = axes[1]
    ax.axvspan(polluted_idx - 0.4, polluted_idx + 0.4, color="#f7c7c7", alpha=0.42, zorder=0)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.9)
    ax.plot(x, daily["TFu_mean_ug_m2_s"], color="#3c6ca8", marker="o", linewidth=2.0, label="TFu 东西向")
    ax.plot(x, daily["TFv_mean_ug_m2_s"], color="#b45f2a", marker="o", linewidth=2.0, label="TFv 南北向")
    ax.set_ylabel("分量\n(µg m$^{-2}$ s$^{-1}$)", fontsize=12, fontweight="bold")
    ax.set_title("(b) 水平通量分量（正值分别为向东、向北）", loc="left", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=11, ncol=2)

    ax = axes[2]
    ax.axvspan(polluted_idx - 0.4, polluted_idx + 0.4, color="#f7c7c7", alpha=0.42, zorder=0)
    ax.plot(x, daily["wind_speed_mean_m_s"], color="#4a8f5a", marker="o", linewidth=2.0, label="水平风速")
    ax.set_ylabel("风速\n(m s$^{-1}$)", fontsize=12, fontweight="bold")
    ax2 = ax.twinx()
    ax2.plot(x, daily["o3_mean_ug_m3"], color="#d88c22", marker="s", linewidth=1.8, label="O$_3$")
    ax2.set_ylabel("O$_3$ (µg m$^{-3}$)", fontsize=12, fontweight="bold")
    ax.set_title("(c) 水平通量的风速与浓度背景", loc="left", fontsize=13, fontweight="bold")
    lines, line_labels = ax.get_legend_handles_labels()
    lines2, line_labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines + lines2,
        line_labels + line_labels2,
        loc="upper right",
        bbox_to_anchor=(0.86, 0.98),
        frameon=False,
        fontsize=11,
        ncol=2,
        borderaxespad=0.0,
    )

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, fontsize=12, fontweight="bold")
    axes[-1].set_xlabel("日期（2025年，北京时间08:00−16:00）", fontsize=12.5, fontweight="bold")

    for ax in axes:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis="y", color="#cccccc", linewidth=0.5, alpha=0.45)
        ax.tick_params(labelsize=11, direction="in")
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax2.tick_params(labelsize=11, direction="in")
    for spine in ax2.spines.values():
        spine.set_linewidth(1.0)

    fig.subplots_adjust(left=0.13, right=0.88, top=0.95, bottom=0.09, hspace=0.31)
    fig.savefig(FIG11)
    plt.close(fig)


def day_matrix(df: pd.DataFrame, date: str, value_col: str) -> pd.DataFrame:
    sub = df[(df["bjt_date"] == date) & (df["bjt_hour"].isin(PLOT_HOURS))]
    pivot = sub.pivot_table(index="height_m", columns="bjt_hour", values=value_col, aggfunc="mean")
    return pivot.reindex(index=TARGET_HEIGHTS_M, columns=PLOT_HOURS)


def plot_figure11a(df: pd.DataFrame) -> None:
    set_chinese_font()
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("white")
    extent = (min(PLOT_HOURS) - 0.5, max(PLOT_HOURS) + 0.5, -0.05, 4.05)

    fig, axes = plt.subplots(1, len(CASE_DATES), figsize=(13.8, 3.8), dpi=300, sharex=True, sharey=True)
    mesh = None
    for col, date in enumerate(CASE_DATES):
        matrix = day_matrix(df, date, "TFh_ug_m2_s")
        mesh = axes[col].imshow(
            matrix.to_numpy(dtype=float),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            cmap=cmap,
            vmin=0,
            vmax=1700,
        )
        axes[col].set_title(f"{pd.Timestamp(date).strftime('%m-%d')}{'*' if date == POLLUTED_DATE else ''}", fontsize=13, fontweight="bold")
        axes[col].set_ylim(0, 4)
        axes[col].set_xticks(PLOT_HOUR_TICKS)
        axes[col].set_xticklabels([f"{h:02d}" for h in PLOT_HOUR_TICKS], fontsize=11)
        axes[col].set_yticks([0, 1, 2, 3, 4])
        axes[col].set_yticklabels(["0", "1", "2", "3", "4"], fontsize=11)
        axes[col].axhspan(0.2, 1.8, color="none", ec="#333333", lw=0.65, alpha=0.6)
        axes[col].tick_params(direction="in", labelbottom=True, labelleft=True, labelsize=11)
        plt.setp(axes[col].get_xticklabels(), visible=True)
        plt.setp(axes[col].get_yticklabels(), visible=True)
        if date == POLLUTED_DATE:
            for spine in axes[col].spines.values():
                spine.set_color("#c62828")
                spine.set_linewidth(2.0)
        else:
            for spine in axes[col].spines.values():
                spine.set_linewidth(1.0)

    axes[0].set_ylabel("高度 (km)", fontsize=12.5, fontweight="bold")
    for ax in axes:
        ax.set_xlabel("北京时间 (h)", fontsize=11.5, fontweight="bold")

    fig.suptitle("(a) 水平输送通量强度TFh时间−高度分布", fontsize=14.5, fontweight="bold", y=0.98)
    cax = fig.add_axes([0.925, 0.23, 0.014, 0.60])
    cb = fig.colorbar(mesh, cax=cax, extend="max", ticks=[0, 400, 800, 1200, 1600])
    cb.set_label("TFh (µg m$^{-2}$ s$^{-1}$)", fontsize=11.5, fontweight="bold")
    cb.ax.tick_params(labelsize=10.5)
    fig.subplots_adjust(left=0.065, right=0.905, top=0.79, bottom=0.18, wspace=0.10)
    fig.savefig(FIG11A)
    plt.close(fig)


def plot_figure12(polluted: pd.DataFrame, good: pd.DataFrame) -> None:
    set_chinese_font()
    panels = [
        ("TFh水平通量强度", "TFh_mean_ug_m2_s", "TFh (µg m$^{-2}$ s$^{-1}$)"),
        ("TFu东西向分量", "TFu_mean_ug_m2_s", "TFu (µg m$^{-2}$ s$^{-1}$)"),
        ("TFv南北向分量", "TFv_mean_ug_m2_s", "TFv (µg m$^{-2}$ s$^{-1}$)"),
        ("水平风速", "wind_speed_mean_m_s", "风速 (m s$^{-1}$)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 5.2), dpi=300, sharey=True)
    for ax, (title, col, xlabel) in zip(axes, panels):
        ax.axhspan(0.2, 1.8, color="#f6d6bd", alpha=0.26, zorder=0)
        ax.axhspan(2.0, 4.0, color="#d8e7f2", alpha=0.22, zorder=0)
        ax.plot(
            good[col],
            good["height_m"] / 1000.0,
            color="#2b9c87",
            linewidth=2.0,
            marker="o",
            markersize=2.6,
            label="前后较好日平均",
        )
        ax.plot(
            polluted[col],
            polluted["height_m"] / 1000.0,
            color="#d75f2a",
            linewidth=2.1,
            marker="o",
            markersize=2.6,
            label="污染日 03-08",
        )
        if col in {"TFu_mean_ug_m2_s", "TFv_mean_ug_m2_s"}:
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.9)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9.8, fontweight="bold")
        ax.set_ylim(0, 4)
        ax.grid(color="#cccccc", linewidth=0.5, alpha=0.42)
        ax.tick_params(labelsize=8.5, direction="in")
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
    axes[0].set_ylabel("高度 (km)", fontsize=10.5, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10.5)
    fig.text(0.50, 0.89, "橙色阴影：0.2−1.8 km关键层；蓝色阴影：2−4 km上部层", ha="center", fontsize=9.3)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.12, wspace=0.26)
    fig.savefig(FIG12)
    plt.close(fig)


def write_summary(df: pd.DataFrame, daily: pd.DataFrame, polluted: pd.DataFrame, good: pd.DataFrame) -> None:
    notes = pd.DataFrame(
        [
            {
                "item": "formula",
                "description": "TFu=C*u, TFv=C*v, TFh=sqrt(TFu^2+TFv^2)=C*sqrt(u^2+v^2); C uses local O3 profile mass concentration, u/v use ERA5 wind components.",
            },
            {
                "item": "interpretation",
                "description": "TFh is used as an auxiliary horizontal ventilation/transport diagnostic. It is much larger than TFw because horizontal wind speed is usually orders of magnitude larger than geometric vertical velocity; the two fluxes describe transport through different planes.",
            },
            {
                "item": "window",
                "description": "BJT 08:00-16:00, 0-4 km vertical profiles, key layer 0.2-1.8 km; selected dates are 2025-03-06, 03-07, 03-08, 03-10 and 03-11.",
            },
        ]
    )
    with pd.ExcelWriter(SUMMARY_XLSX, engine="openpyxl") as writer:
        notes.to_excel(writer, sheet_name="notes", index=False)
        daily.to_excel(writer, sheet_name="key_layer_daily", index=False)
        polluted.to_excel(writer, sheet_name="polluted_profile", index=False)
        good.to_excel(writer, sheet_name="good_day_profile", index=False)
        df.to_excel(writer, sheet_name="hour_height_records", index=False)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_horizontal_flux()
    daily = summarize_daily(df)
    polluted, good = summarize_profiles(df)
    plot_figure11a(df)
    plot_figure11(daily)
    plot_figure12(polluted, good)
    write_summary(df, daily, polluted, good)
    print(f"Wrote {FIG11A}")
    print(f"Wrote {FIG11}")
    print(f"Wrote {FIG12}")
    print(f"Wrote {SUMMARY_XLSX}")
    print(daily.to_string(index=False))


if __name__ == "__main__":
    main()
