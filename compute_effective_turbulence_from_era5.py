# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


BASE_DIR = Path("E:/research/????")
SHANGHAI_ROOT = BASE_DIR / "\u4e0a\u6d77_ERA5\u6570\u636e\u4e0e\u5904\u7406\u7ed3\u679c\u6c47\u603b"
INPUT_DIR = SHANGHAI_ROOT / "ERA5_600_1000_\u9ad8\u5ea6\u63d2\u503c\u4e0e\u901a\u91cf"
OUT_DIR = SHANGHAI_ROOT / "ERA5_\u7b49\u6548\u6e4d\u6d41\u7cfb\u6570_\u8fd1\u5730\u9762\u8865\u9f50"

MIXING_LENGTH_M = 100.0
HEIGHT_COL = "height_agl_m"
TARGET_HEIGHTS_FOR_LINES = [0, 100, 200, 500, 1000, 2000, 3000, 4000]

VARIABLE_DESCRIPTIONS = [
    {
        "english_name": "profile_type",
        "中文含义": "剖面类型",
        "物理量/用途": "区分区域平均剖面 area_mean 或最近站点格点 nearest_site_point。",
        "单位": "-",
        "备注": "用于区分区域结果和站点结果。",
    },
    {
        "english_name": "bjt_time",
        "中文含义": "北京时间",
        "物理量/用途": "ERA5 UTC 时间加 8 小时后的北京时间。",
        "单位": "-",
        "备注": "本项目时间序列均按北京时间解释。",
    },
    {
        "english_name": "bjt_date",
        "中文含义": "北京日期",
        "物理量/用途": "北京时间对应日期。",
        "单位": "-",
        "备注": "用于日均统计。",
    },
    {
        "english_name": "bjt_hour",
        "中文含义": "北京小时",
        "物理量/用途": "北京时间小时。",
        "单位": "hour",
        "备注": "当前 ERA5 UTC 00-08 时对应北京时间 08-16 时。",
    },
    {
        "english_name": "month",
        "中文含义": "月份",
        "物理量/用途": "北京时间月份。",
        "单位": "-",
        "备注": "用于月均统计。",
    },
    {
        "english_name": "height_m",
        "中文含义": "原始插值高度",
        "物理量/用途": "原始高度插值脚本生成的固定高度层。",
        "单位": "m",
        "备注": "本次近地面补齐分析中不再按海拔高度解释，而转为 height_agl_m。",
    },
    {
        "english_name": "height_agl_m",
        "中文含义": "近地面相对高度",
        "物理量/用途": "从近地面 0 m 起算的相对高度层。",
        "单位": "m",
        "备注": "本次补齐结果使用该字段作为纵坐标。",
    },
    {
        "english_name": "near_ground_fill_method",
        "中文含义": "近地面补齐方式",
        "物理量/用途": "记录 0-200 m 低层数据是否由相邻高度复制补齐。",
        "单位": "-",
        "备注": "用于判断低层结果是否为原始插值或诊断补齐。",
    },
    {
        "english_name": "pressure_hPa",
        "中文含义": "气压",
        "物理量/用途": "插值高度处对应的气压。",
        "单位": "hPa",
        "备注": "由 ERA5 pressure levels 插值得到。",
    },
    {
        "english_name": "geopotential_height_m",
        "中文含义": "位势高度",
        "物理量/用途": "由位势除以重力加速度得到的高度。",
        "单位": "m",
        "备注": "原始 pressure-level 高度换算依据。",
    },
    {
        "english_name": "temperature_K",
        "中文含义": "温度",
        "物理量/用途": "空气温度。",
        "单位": "K",
        "备注": "用于空气密度计算。",
    },
    {
        "english_name": "temperature_C",
        "中文含义": "摄氏温度",
        "物理量/用途": "空气温度的摄氏表达。",
        "单位": "degC",
        "备注": "temperature_K - 273.15。",
    },
    {
        "english_name": "specific_humidity_kg_kg",
        "中文含义": "比湿",
        "物理量/用途": "单位质量湿空气中的水汽质量。",
        "单位": "kg/kg",
        "备注": "用于虚温和空气密度计算。",
    },
    {
        "english_name": "air_density_kg_m3",
        "中文含义": "空气密度",
        "物理量/用途": "单位体积空气质量。",
        "单位": "kg/m3",
        "备注": "由气压、温度、比湿近似计算。",
    },
    {
        "english_name": "o3_mmr_kg_kg",
        "中文含义": "O3 质量混合比",
        "物理量/用途": "臭氧质量与空气质量之比。",
        "单位": "kg/kg",
        "备注": "ERA5 ozone_mass_mixing_ratio。",
    },
    {
        "english_name": "o3_ppbv_dry_approx",
        "中文含义": "O3 体积分数近似值",
        "物理量/用途": "由质量混合比换算的干空气 ppbv 近似值。",
        "单位": "ppbv",
        "备注": "用于辅助理解，不是原始下载变量。",
    },
    {
        "english_name": "o3_mass_ug_m3",
        "中文含义": "O3 质量浓度",
        "物理量/用途": "单位体积空气中的臭氧质量。",
        "单位": "ug/m3",
        "备注": "由 O3 质量混合比乘空气密度换算。",
    },
    {
        "english_name": "u_wind_m_s",
        "中文含义": "东西向风速",
        "物理量/用途": "纬向风分量。",
        "单位": "m/s",
        "备注": "正值通常表示向东。",
    },
    {
        "english_name": "v_wind_m_s",
        "中文含义": "南北向风速",
        "物理量/用途": "经向风分量。",
        "单位": "m/s",
        "备注": "正值通常表示向北。",
    },
    {
        "english_name": "omega_Pa_s",
        "中文含义": "压力垂直速度",
        "物理量/用途": "压力坐标下的垂直运动速度 omega。",
        "单位": "Pa/s",
        "备注": "ERA5 原始 vertical_velocity；正值常表示下沉，负值表示上升。",
    },
    {
        "english_name": "w_geometric_m_s",
        "中文含义": "几何垂直速度",
        "物理量/用途": "由 omega 换算得到的高度坐标垂直速度。",
        "单位": "m/s",
        "备注": "w = -omega / (rho * g)，正值表示向上，负值表示向下。",
    },
    {
        "english_name": "vertical_exchange_intensity_m_s",
        "中文含义": "垂直交换强度速度项",
        "物理量/用途": "几何垂直速度绝对值，表示上升/下沉运动强度。",
        "单位": "m/s",
        "备注": "等于 abs(w_geometric_m_s)。",
    },
    {
        "english_name": "mixing_length_m",
        "中文含义": "代表性混合长度",
        "物理量/用途": "用于构造等效湍流交换系数的长度尺度。",
        "单位": "m",
        "备注": "本次统一取 100 m。",
    },
    {
        "english_name": "k_eff_m2_s",
        "中文含义": "等效湍流交换系数",
        "物理量/用途": "基于垂直速度构造的等效垂直交换强度。",
        "单位": "m2/s",
        "备注": "K_eff = abs(w_geometric_m_s) * 100 m；不是 WRF EXCH_H 的严格等价物。",
    },
    {
        "english_name": "k_eff_signed_m2_s",
        "中文含义": "带符号等效交换系数",
        "物理量/用途": "保留上升/下沉方向的等效交换指标。",
        "单位": "m2/s",
        "备注": "w_geometric_m_s * 100 m；正值偏上升，负值偏下沉。",
    },
    {
        "english_name": "o3_gradient_ug_m4",
        "中文含义": "O3 垂直浓度梯度",
        "物理量/用途": "O3 质量浓度随高度的变化率。",
        "单位": "ug/m4",
        "备注": "dO3/dz；用于计算等效湍流扩散通量。",
    },
    {
        "english_name": "o3_advective_flux_upward_ug_m2_s",
        "中文含义": "O3 垂直平流通量",
        "物理量/用途": "解析尺度垂直速度造成的 O3 垂直输送。",
        "单位": "ug/(m2 s)",
        "备注": "O3 * w；正值表示向上输送。",
    },
    {
        "english_name": "o3_advective_flux_downward_ug_m2_s",
        "中文含义": "O3 向下平流通量",
        "物理量/用途": "将向下输送定义为正的平流通量。",
        "单位": "ug/(m2 s)",
        "备注": "-O3 * w；正值表示向下输送。",
    },
    {
        "english_name": "o3_turbulent_flux_eff_upward_ug_m2_s",
        "中文含义": "O3 等效湍流扩散通量",
        "物理量/用途": "基于 K_eff 和 O3 垂直梯度计算的诊断型扩散通量。",
        "单位": "ug/(m2 s)",
        "备注": "F_eff = -K_eff * dO3/dz；正值表示向上扩散输送。",
    },
    {
        "english_name": "o3_turbulent_flux_eff_downward_ug_m2_s",
        "中文含义": "O3 向下等效扩散通量",
        "物理量/用途": "将向下扩散输送定义为正的诊断通量。",
        "单位": "ug/(m2 s)",
        "备注": "K_eff * dO3/dz；正值表示向下扩散输送。",
    },
    {
        "english_name": "o3_total_exchange_eff_upward_ug_m2_s",
        "中文含义": "O3 等效总垂直交换通量",
        "物理量/用途": "垂直平流通量与等效湍流扩散通量之和。",
        "单位": "ug/(m2 s)",
        "备注": "F_adv_upward + F_eff_upward；正值表示总效果偏向上输送。",
    },
]

BASE_COPY_COLS = [
    "pressure_hPa",
    "geopotential_height_m",
    "temperature_K",
    "temperature_C",
    "specific_humidity_kg_kg",
    "air_density_kg_m3",
    "o3_mmr_kg_kg",
    "o3_ppbv_dry_approx",
    "o3_mass_ug_m3",
    "u_wind_m_s",
    "v_wind_m_s",
    "omega_Pa_s",
    "w_geometric_m_s",
]


def read_height_csv(name: str) -> pd.DataFrame:
    path = INPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["bjt_time"] = pd.to_datetime(df["bjt_time"])
    df[HEIGHT_COL] = pd.to_numeric(df["height_m"], errors="coerce")
    return df


def copy_low_layer_from_100_200m(profile: pd.DataFrame) -> pd.DataFrame:
    """Fill near-ground rows from the 100-200 m layer as a diagnostic shortcut."""
    profile = profile.sort_values(HEIGHT_COL).copy()
    profile["near_ground_fill_method"] = "kept_or_original_interpolation"

    copy_cols = [col for col in BASE_COPY_COLS if col in profile.columns]

    def row_at(height: float) -> pd.Series | None:
        rows = profile.loc[np.isclose(profile[HEIGHT_COL], height)]
        if rows.empty:
            return None
        return rows.iloc[0]

    row_100 = row_at(100.0)
    row_200 = row_at(200.0)

    def has_key_values(row: pd.Series | None) -> bool:
        if row is None:
            return False
        return bool(np.isfinite(row.get("o3_mass_ug_m3", np.nan)) and np.isfinite(row.get("w_geometric_m_s", np.nan)))

    # If 200 m is still below the lowest reliable pressure-level information,
    # use the nearest valid level above it. This is intentionally diagnostic.
    if row_200 is not None and not has_key_values(row_200):
        idx_200 = profile.index[np.isclose(profile[HEIGHT_COL], 200.0)]
        candidates = profile[
            (profile[HEIGHT_COL] > 200.0)
            & np.isfinite(pd.to_numeric(profile["o3_mass_ug_m3"], errors="coerce"))
            & np.isfinite(pd.to_numeric(profile["w_geometric_m_s"], errors="coerce"))
        ].sort_values(HEIGHT_COL)
        if len(idx_200) > 0 and not candidates.empty:
            source = candidates.iloc[0]
            for col in copy_cols:
                profile.loc[idx_200, col] = source[col]
            profile.loc[idx_200, "near_ground_fill_method"] = f"200m_copied_from_{int(source[HEIGHT_COL])}m"
            row_200 = row_at(200.0)

    # If the 100 m row is missing key values, use 200 m to make the
    # 100-200 m layer usable. This follows the requested low-level fallback.
    if row_100 is not None and row_200 is not None:
        idx_100 = profile.index[np.isclose(profile[HEIGHT_COL], 100.0)]
        needs_100 = profile.loc[idx_100, ["o3_mass_ug_m3", "w_geometric_m_s"]].isna().any(axis=1)
        if bool(needs_100.any()):
            for col in copy_cols:
                profile.loc[idx_100[needs_100], col] = row_200[col]
            profile.loc[idx_100[needs_100], "near_ground_fill_method"] = "100m_copied_from_200m"
            row_100 = row_at(100.0)

    # Treat the 0-100 m layer as too uncertain for pressure-level ERA5 and
    # copy the nearest 100-200 m information into the 0 m representative row.
    idx_0 = profile.index[np.isclose(profile[HEIGHT_COL], 0.0)]
    if len(idx_0) > 0:
        source = row_100 if row_100 is not None else row_200
        if source is not None:
            for col in copy_cols:
                profile.loc[idx_0, col] = source[col]
            profile.loc[idx_0, "near_ground_fill_method"] = (
                "0_100m_layer_copied_from_100m"
                if row_100 is not None
                else "0_100m_layer_copied_from_200m"
            )

    return profile


def add_effective_turbulence(profile: pd.DataFrame) -> pd.DataFrame:
    profile = profile.sort_values(HEIGHT_COL).copy()

    o3 = pd.to_numeric(profile["o3_mass_ug_m3"], errors="coerce")
    w = pd.to_numeric(profile["w_geometric_m_s"], errors="coerce")
    z = pd.to_numeric(profile[HEIGHT_COL], errors="coerce").to_numpy(dtype=float)

    profile["vertical_exchange_intensity_m_s"] = w.abs()
    profile["mixing_length_m"] = MIXING_LENGTH_M
    profile["k_eff_m2_s"] = w.abs() * MIXING_LENGTH_M
    profile["k_eff_signed_m2_s"] = w * MIXING_LENGTH_M

    # Keep the resolved vertical advective flux for comparison.
    profile["o3_advective_flux_upward_ug_m2_s"] = o3 * w
    profile["o3_advective_flux_downward_ug_m2_s"] = -o3 * w

    # A diagnostic effective turbulent-diffusive flux based on K_eff.
    # Positive upward convention: F = -K * dC/dz.
    o3_interp = o3.interpolate(limit_direction="both")
    if o3_interp.notna().sum() >= 2 and np.isfinite(z).sum() >= 2:
        gradient = np.gradient(o3_interp.to_numpy(dtype=float), z)
    else:
        gradient = np.full(len(profile), np.nan)
    profile["o3_gradient_ug_m4"] = gradient
    profile["o3_turbulent_flux_eff_upward_ug_m2_s"] = -profile["k_eff_m2_s"] * profile["o3_gradient_ug_m4"]
    profile["o3_turbulent_flux_eff_downward_ug_m2_s"] = profile["k_eff_m2_s"] * profile["o3_gradient_ug_m4"]
    profile["o3_total_exchange_eff_upward_ug_m2_s"] = (
        profile["o3_advective_flux_upward_ug_m2_s"]
        + profile["o3_turbulent_flux_eff_upward_ug_m2_s"]
    )
    return profile


def process_profiles(df: pd.DataFrame) -> pd.DataFrame:
    groups = []
    group_cols = ["profile_type", "bjt_time"]
    for _, profile in df.groupby(group_cols, dropna=False, sort=True):
        profile = copy_low_layer_from_100_200m(profile)
        profile = add_effective_turbulence(profile)
        groups.append(profile)
    out = pd.concat(groups, ignore_index=True)
    out["bjt_date"] = out["bjt_time"].dt.date.astype(str)
    out["month"] = out["bjt_time"].dt.month.astype(str).str.zfill(2)
    out["bjt_hour"] = out["bjt_time"].dt.hour
    return out


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_cols = [
        "o3_mass_ug_m3",
        "w_geometric_m_s",
        "vertical_exchange_intensity_m_s",
        "k_eff_m2_s",
        "k_eff_signed_m2_s",
        "o3_gradient_ug_m4",
        "o3_advective_flux_upward_ug_m2_s",
        "o3_advective_flux_downward_ug_m2_s",
        "o3_turbulent_flux_eff_upward_ug_m2_s",
        "o3_turbulent_flux_eff_downward_ug_m2_s",
        "o3_total_exchange_eff_upward_ug_m2_s",
    ]
    daily = df.groupby(["bjt_date", HEIGHT_COL], as_index=False)[value_cols].mean()
    monthly = df.groupby(["month", HEIGHT_COL], as_index=False)[value_cols].mean()
    return daily, monthly


def matrix_by_month(monthly: pd.DataFrame, value_col: str) -> pd.DataFrame:
    pivot = monthly.pivot_table(index=HEIGHT_COL, columns="month", values=value_col, aggfunc="mean")
    pivot = pivot.sort_index().reset_index()
    pivot.columns = [HEIGHT_COL] + [f"month_{str(c).zfill(2)}" for c in pivot.columns[1:]]
    return pivot


def plot_pcolor(df: pd.DataFrame, value_col: str, out_path: Path, title: str, cmap: str, diverging: bool = False) -> None:
    pivot = df.pivot_table(index=HEIGHT_COL, columns="bjt_time", values=value_col, aggfunc="mean")
    pivot = pivot.sort_index()
    pivot = pivot.loc[pivot.index <= 4000]
    values = pivot.to_numpy(dtype=float)
    times = pd.to_datetime(pivot.columns)
    x = mdates.date2num(times.to_pydatetime())
    y = pivot.index.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=220)
    norm = None
    if diverging:
        abs_max = np.nanpercentile(np.abs(values), 98)
        if np.isfinite(abs_max) and abs_max > 0:
            norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    mesh = ax.pcolormesh(x, y, values, shading="auto", cmap=cmap, norm=norm)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=35, ha="right")
    ax.set_ylim(0, 4000)
    ax.set_ylabel("Height above ground (m)")
    ax.set_xlabel("Beijing time")
    ax.set_title(title)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(value_col)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_daily_lines(daily: pd.DataFrame, value_col: str, out_path: Path, title: str) -> None:
    selected = daily[daily[HEIGHT_COL].isin(TARGET_HEIGHTS_FOR_LINES)].copy()
    selected["bjt_date"] = pd.to_datetime(selected["bjt_date"])
    pivot = selected.pivot_table(index="bjt_date", columns=HEIGHT_COL, values=value_col, aggfunc="mean")
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(13, 5), dpi=220)
    for height in TARGET_HEIGHTS_FOR_LINES:
        if height in pivot.columns:
            ax.plot(pivot.index, pivot[height], linewidth=1.2, label=f"{height} m")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(value_col)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=4, fontsize=8)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_workbook(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    description_map = {
        item["english_name"]: item
        for item in VARIABLE_DESCRIPTIONS
    }
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for sheet, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
        wb = writer.book
        header_fmt = wb.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
        for sheet, df in sheets.items():
            ws = writer.sheets[sheet]
            ws.freeze_panes(1, 0)
            if len(df.columns):
                ws.autofilter(0, 0, len(df), len(df.columns) - 1)
            for col_idx, col in enumerate(df.columns):
                ws.write(0, col_idx, col, header_fmt)
                ws.set_column(col_idx, col_idx, min(max(len(str(col)) + 2, 12), 36))
                desc = description_map.get(str(col))
                if desc:
                    comment = (
                        f"{desc['中文含义']}\n"
                        f"物理量/用途: {desc['物理量/用途']}\n"
                        f"单位: {desc['单位']}\n"
                        f"备注: {desc['备注']}"
                    )
                    ws.write_comment(0, col_idx, comment, {"x_scale": 1.5, "y_scale": 1.5})


def process_dataset(source_name: str, csv_name: str, make_plots: bool) -> None:
    print(f"Reading {csv_name}")
    df = read_height_csv(csv_name)
    processed = process_profiles(df)
    daily, monthly = summarize(processed)

    processed_path = OUT_DIR / f"{source_name}_effective_turbulence_hourly.csv"
    daily_path = OUT_DIR / f"{source_name}_effective_turbulence_daily.csv"
    monthly_path = OUT_DIR / f"{source_name}_effective_turbulence_monthly.csv"
    processed.to_csv(processed_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    monthly.to_csv(monthly_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {processed_path}")
    print(f"Wrote {daily_path}")
    print(f"Wrote {monthly_path}")

    if make_plots:
        plot_pcolor(
            processed,
            "k_eff_m2_s",
            OUT_DIR / "area_k_eff_pcolor_hourly.png",
            "Effective vertical exchange coefficient K_eff = |w| x 100 m",
            "viridis",
        )
        plot_pcolor(
            processed,
            "o3_turbulent_flux_eff_upward_ug_m2_s",
            OUT_DIR / "area_effective_turbulent_flux_pcolor_hourly.png",
            "Effective O3 turbulent-diffusive flux, positive upward",
            "RdBu_r",
            diverging=True,
        )
        plot_daily_lines(
            daily,
            "k_eff_m2_s",
            OUT_DIR / "area_k_eff_daily_timeseries_selected_heights.png",
            "Daily K_eff at selected heights",
        )
        plot_daily_lines(
            daily,
            "o3_turbulent_flux_eff_upward_ug_m2_s",
            OUT_DIR / "area_effective_turbulent_flux_daily_timeseries_selected_heights.png",
            "Daily effective turbulent-diffusive O3 flux at selected heights",
        )

    return processed, daily, monthly


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    area, area_daily, area_monthly = process_dataset("area", "height_interp_area_mean.csv", make_plots=True)
    site, site_daily, site_monthly = process_dataset("site", "height_interp_site_point.csv", make_plots=False)

    notes = pd.DataFrame(
        [
            {
                "item": "height coordinate",
                "description": "The output height_agl_m is treated as near-ground relative height from 0 m, not altitude above sea level.",
            },
            {
                "item": "low-level filling",
                "description": "The 0-100 m representative row is copied from the 100 m row, or from 200 m if 100 m is unavailable. Missing 100 m values are copied from 200 m.",
            },
            {
                "item": "effective coefficient",
                "description": "K_eff = abs(w_geometric_m_s) * 100 m. This is a diagnostic substitute, not true WRF EXCH_H.",
            },
            {
                "item": "effective turbulent flux",
                "description": "F_eff = -K_eff * dO3/dz, positive upward. Downward-positive flux is also provided as K_eff * dO3/dz.",
            },
            {
                "item": "resolved advective flux",
                "description": "F_adv = O3 mass concentration * w_geometric_m_s, positive upward.",
            },
        ]
    )
    variable_descriptions = pd.DataFrame(VARIABLE_DESCRIPTIONS)

    sheets = {
        "notes": notes,
        "变量中文含义": variable_descriptions,
        "area_monthly": area_monthly,
        "site_monthly": site_monthly,
        "matrix_Keff_area": matrix_by_month(area_monthly, "k_eff_m2_s"),
        "matrix_Feff_area": matrix_by_month(area_monthly, "o3_turbulent_flux_eff_upward_ug_m2_s"),
        "matrix_Fadv_area": matrix_by_month(area_monthly, "o3_advective_flux_upward_ug_m2_s"),
        "area_daily_selected": area_daily[area_daily[HEIGHT_COL].isin(TARGET_HEIGHTS_FOR_LINES)].copy(),
    }
    workbook = OUT_DIR / "ERA5_effective_turbulence_summary.xlsx"
    write_workbook(workbook, sheets)
    print(f"Wrote {workbook}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Area rows: {len(area)}; Site rows: {len(site)}")


if __name__ == "__main__":
    main()
