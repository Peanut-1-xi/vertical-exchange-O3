# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


BASE_DIR = Path("E:/research/????")
O3_THRESHOLD_UG_M3 = 160.0
NEAR_SURFACE_MAX_M = 100.0
BOX_HEIGHTS_M = list(range(100, 1001, 100))
POLLUTED_COLOR = "#ff8c00"
CLEAN_COLOR = "#22a88a"


CITY_CONFIGS = [
    {
        "label": "Shanghai",
        "input": BASE_DIR
        / "\u4e0a\u6d77_ERA5\u6570\u636e\u4e0e\u5904\u7406\u7ed3\u679c\u6c47\u603b"
        / "ERA5_\u7b49\u6548\u6e4d\u6d41\u7cfb\u6570_\u8fd1\u5730\u9762\u8865\u9f50"
        / "site_effective_turbulence_hourly.csv",
        "output_dir": BASE_DIR
        / "\u4e0a\u6d77_ERA5\u6570\u636e\u4e0e\u5904\u7406\u7ed3\u679c\u6c47\u603b"
        / "\u4e0a\u6d77_\u5178\u578b\u6c89\u964d\u4e0e\u5e74\u5c3a\u5ea6\u7edf\u8ba1\u56fe",
    },
    {
        "label": "Hefei Science Island",
        "input": BASE_DIR
        / "\u5408\u80a5\u79d1\u5b66\u5c9b_ERA5\u6570\u636e\u4e0e\u5904\u7406\u7ed3\u679c\u6c47\u603b"
        / "\u5408\u80a5\u79d1\u5b66\u5c9b_ERA5_\u7b49\u6548\u6e4d\u6d41\u7cfb\u6570_\u8fd1\u5730\u9762\u8865\u9f50"
        / "site_effective_turbulence_hourly.csv",
        "output_dir": BASE_DIR
        / "\u5408\u80a5\u79d1\u5b66\u5c9b_ERA5\u6570\u636e\u4e0e\u5904\u7406\u7ed3\u679c\u6c47\u603b"
        / "\u5408\u80a5\u79d1\u5b66\u5c9b_\u5178\u578b\u6c89\u964d\u4e0e\u5e74\u5c3a\u5ea6\u7edf\u8ba1\u56fe",
    },
]


def clean_box_arrays(arrays: list[np.ndarray]) -> list[np.ndarray]:
    cleaned = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size >= 4:
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                arr = arr[(arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)]
        cleaned.append(arr)
    return cleaned


def read_city(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "bjt_date",
        "height_agl_m",
        "o3_mass_ug_m3",
        "o3_advective_flux_downward_ug_m2_s",
    ]
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=cols)
    df["bjt_date"] = pd.to_datetime(df["bjt_date"]).dt.date.astype(str)
    df["height_agl_m"] = pd.to_numeric(df["height_agl_m"], errors="coerce")
    df["o3_advective_flux_upward_ug_m2_s"] = -pd.to_numeric(
        df["o3_advective_flux_downward_ug_m2_s"], errors="coerce"
    )
    daily = df.groupby("bjt_date").agg(o3_profile_max_ug_m3=("o3_mass_ug_m3", "max")).reset_index()
    near_surface = (
        df[df["height_agl_m"] <= NEAR_SURFACE_MAX_M]
        .groupby("bjt_date")["o3_mass_ug_m3"]
        .max()
        .rename("o3_near_surface_daily_max_ug_m3")
    )
    daily = daily.merge(near_surface, left_on="bjt_date", right_index=True, how="left")
    daily["polluted"] = daily["o3_near_surface_daily_max_ug_m3"] > O3_THRESHOLD_UG_M3
    return df, daily


def build_box_data(df: pd.DataFrame, polluted_dates: set[str], value_col: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    sub = df[df["height_agl_m"].isin(BOX_HEIGHTS_M)].copy()
    sub["is_polluted"] = sub["bjt_date"].isin(polluted_dates)
    polluted = []
    clean = []
    for height in BOX_HEIGHTS_M:
        hsub = sub[sub["height_agl_m"] == height]
        polluted.append(pd.to_numeric(hsub[hsub["is_polluted"]][value_col], errors="coerce").dropna().to_numpy())
        clean.append(pd.to_numeric(hsub[~hsub["is_polluted"]][value_col], errors="coerce").dropna().to_numpy())
    return clean_box_arrays(polluted), clean_box_arrays(clean)


def draw_boxplot(ax, y, data_p, data_c, xlabel, xlim=None, zero_line=False):
    offset = 0.18
    width = 0.30
    bp_p = ax.boxplot(
        data_p,
        positions=y + offset,
        widths=width,
        orientation="horizontal",
        patch_artist=True,
        showfliers=False,
        whis=(0, 100),
        manage_ticks=False,
    )
    bp_c = ax.boxplot(
        data_c,
        positions=y - offset,
        widths=width,
        orientation="horizontal",
        patch_artist=True,
        showfliers=False,
        whis=(0, 100),
        manage_ticks=False,
    )
    for box in bp_p["boxes"]:
        box.set(facecolor=POLLUTED_COLOR, edgecolor=POLLUTED_COLOR, alpha=0.30, linewidth=1.8)
    for item in bp_p["medians"] + bp_p["whiskers"] + bp_p["caps"]:
        item.set(color=POLLUTED_COLOR, linewidth=1.8)
    for box in bp_c["boxes"]:
        box.set(facecolor=CLEAN_COLOR, edgecolor=CLEAN_COLOR, alpha=0.30, linewidth=1.8)
    for item in bp_c["medians"] + bp_c["whiskers"] + bp_c["caps"]:
        item.set(color=CLEAN_COLOR, linewidth=1.8)
    if zero_line:
        ax.axvline(0, color="gray", linestyle="--", linewidth=1.5)
    ax.set_xlabel(xlabel, fontsize=18, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(BOX_HEIGHTS_M, fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", labelsize=12, direction="in")
    ax.tick_params(axis="y", direction="in")
    if xlim:
        ax.set_xlim(*xlim)
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)


def plot_city(config: dict[str, object]) -> Path:
    df, daily = read_city(config["input"])
    polluted_dates = set(daily[daily["polluted"]]["bjt_date"])
    flux_polluted, flux_clean = build_box_data(df, polluted_dates, "o3_advective_flux_upward_ug_m2_s")
    o3_polluted, o3_clean = build_box_data(df, polluted_dates, "o3_mass_ug_m3")

    fig, axes = plt.subplots(1, 2, figsize=(12, 7.2), dpi=260, sharey=True)
    y = np.arange(len(BOX_HEIGHTS_M))
    flux_all = np.concatenate([arr for arr in flux_polluted + flux_clean if len(arr)])
    flux_limit = max(1.0, np.nanpercentile(np.abs(flux_all), 98) * 1.25)
    draw_boxplot(
        axes[0],
        y,
        flux_polluted,
        flux_clean,
        "F_adv (+up / -down)\n(ug m-2 s-1)",
        xlim=(-flux_limit, flux_limit),
        zero_line=True,
    )
    o3_all = np.concatenate([arr for arr in o3_polluted + o3_clean if len(arr)])
    draw_boxplot(
        axes[1],
        y,
        o3_polluted,
        o3_clean,
        "O3 (ug/m3)",
        xlim=(0, np.nanpercentile(o3_all, 99) * 1.15),
    )
    axes[0].set_ylabel("Height above ground (m)", fontsize=20, fontweight="bold")

    total_days = daily["bjt_date"].nunique()
    handles = [
        Patch(facecolor=POLLUTED_COLOR, edgecolor=POLLUTED_COLOR, alpha=0.30, label=f"Polluted (n={len(polluted_dates)} d)"),
        Patch(facecolor=CLEAN_COLOR, edgecolor=CLEAN_COLOR, alpha=0.30, label=f"Clean (n={total_days - len(polluted_dates)} d)"),
    ]
    for ax in axes:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False, fontsize=12)

    fig.suptitle(
        f"Annual Signed Vertical Advective Flux by O3 Exceedance Days, {config['label']}\n"
        f"Polluted: 0-100 m daily maximum O3 > {O3_THRESHOLD_UG_M3:.0f} ug/m3",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.93])
    out = config["output_dir"] / "annual_signed_advective_flux_o3_boxplot.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    for config in CITY_CONFIGS:
        path = plot_city(config)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
