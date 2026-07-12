# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT_DIR = Path("E:/research/????") / "\u4e0a\u6d77_ERA5\u6570\u636e\u4e0e\u5904\u7406\u7ed3\u679c\u6c47\u603b"
BASE_DIR = ROOT_DIR / "ERA5_600_1000_\u9ad8\u5ea6\u63d2\u503c\u4e0e\u901a\u91cf"
OUT_XLSX = BASE_DIR / "ERA5_600_1000_高度插值通量_汇总可视化.xlsx"


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(BASE_DIR / name, encoding="utf-8-sig")


def month_height_matrix(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    pivot = df.pivot_table(index="height_m", columns="month", values=value_col, aggfunc="mean")
    pivot = pivot.sort_index().reset_index()
    pivot.columns = ["height_m"] + [f"month_{str(col).zfill(2)}" for col in pivot.columns[1:]]
    return pivot


def month_height_valid_matrix(df: pd.DataFrame, value_col: str, mode: str) -> pd.DataFrame:
    grouped = df.groupby(["height_m", "month"], dropna=False)
    total = grouped.size().unstack("month").sort_index()
    valid = grouped[value_col].count().unstack("month").sort_index()
    months = sorted(total.columns)
    total = total.reindex(columns=months).fillna(0)
    valid = valid.reindex(index=total.index, columns=months).fillna(0)

    if mode == "count":
        matrix = valid.astype(int)
    elif mode == "fraction":
        matrix = valid.divide(total.where(total > 0))
    else:
        raise ValueError(f"Unsupported valid matrix mode: {mode}")

    matrix = matrix.reset_index()
    matrix.columns = ["height_m"] + [f"month_{str(col).zfill(2)}" for col in months]
    return matrix


def selected_height_daily_wide(df: pd.DataFrame) -> pd.DataFrame:
    selected_heights = [500, 1000, 2000, 3000, 4000]
    value_cols = ["o3_mass_ug_m3", "o3_flux_w_from_interp_product_ug_m2_s"]
    parts = []
    for value_col in value_cols:
        pivot = df[df["height_m"].isin(selected_heights)].pivot_table(
            index="bjt_date",
            columns="height_m",
            values=value_col,
            aggfunc="mean",
        )
        pivot = pivot.sort_index()
        pivot.columns = [f"{value_col}_{int(height)}m" for height in pivot.columns]
        parts.append(pivot)
    wide = pd.concat(parts, axis=1).reset_index()
    return wide


def add_basic_formatting(writer: pd.ExcelWriter, sheets: dict[str, pd.DataFrame]) -> None:
    wb = writer.book
    header_fmt = wb.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})
    num_fmt = wb.add_format({"num_format": "0.000"})
    time_fmt = wb.add_format({"num_format": "yyyy-mm-dd hh:mm"})

    for sheet, df in sheets.items():
        ws = writer.sheets[sheet]
        ws.freeze_panes(1, 0)
        if len(df.columns) > 0:
            ws.autofilter(0, 0, len(df), len(df.columns) - 1)
        for col_idx, col in enumerate(df.columns):
            ws.write(0, col_idx, col, header_fmt)
            width = min(max(len(str(col)) + 2, 12), 34)
            fmt = time_fmt if "time" in str(col).lower() else num_fmt
            ws.set_column(col_idx, col_idx, width, fmt)


def add_profile_chart(
    writer: pd.ExcelWriter,
    sheet: str,
    title: str,
    x_axis_name: str,
    first_value_col: int = 1,
) -> None:
    wb = writer.book
    ws = writer.sheets[sheet]
    df = writer._sheet_dfs[sheet]  # type: ignore[attr-defined]
    chart = wb.add_chart({"type": "scatter", "subtype": "straight"})
    height_col = 0
    for col in range(first_value_col, len(df.columns)):
        chart.add_series(
            {
                "name": [sheet, 0, col],
                "categories": [sheet, 1, col, len(df), col],
                "values": [sheet, 1, height_col, len(df), height_col],
                "marker": {"type": "none"},
            }
        )
    chart.set_title({"name": title})
    chart.set_x_axis({"name": x_axis_name})
    chart.set_y_axis({"name": "Height (m)", "min": 0, "max": 4000})
    chart.set_legend({"position": "right"})
    ws.insert_chart("J2", chart, {"x_scale": 1.7, "y_scale": 1.6})


def add_timeseries_chart(writer: pd.ExcelWriter, data_sheet: str, chart_sheet: str) -> None:
    wb = writer.book
    ws = writer.sheets[chart_sheet]
    df = writer._sheet_dfs[data_sheet]  # type: ignore[attr-defined]
    if df.empty:
        return
    cols = list(df.columns)
    date_col = cols.index("bjt_date")
    last = len(df)
    for idx, height in enumerate([500, 1000, 2000, 3000, 4000]):
        chart = wb.add_chart({"type": "line"})
        for value_col in [
            f"o3_mass_ug_m3_{height}m",
            f"o3_flux_w_from_interp_product_ug_m2_s_{height}m",
        ]:
            if value_col not in cols:
                continue
            chart.add_series(
                {
                    "name": [data_sheet, 0, cols.index(value_col)],
                    "categories": [data_sheet, 1, date_col, last, date_col],
                    "values": [data_sheet, 1, cols.index(value_col), last, cols.index(value_col)],
                }
            )
        chart.set_title({"name": f"Daily O3 and vertical flux at {height} m"})
        chart.set_x_axis({"name": "BJT date"})
        chart.set_y_axis({"name": "Value"})
        ws.insert_chart(2 + idx * 16, 10, chart, {"x_scale": 1.6, "y_scale": 1.2})


def main() -> None:
    inventory = read_csv("downloaded_inventory.csv")
    monthly_area = read_csv("height_monthly_area_mean.csv")
    daily_area = read_csv("height_daily_area_mean.csv")
    area = read_csv("height_interp_area_mean.csv")
    site = read_csv("height_interp_site_point.csv")

    monthly_site = site.groupby(["month", "height_m"], as_index=False)[
        [
            "o3_mass_ug_m3",
            "u_wind_m_s",
            "v_wind_m_s",
            "w_geometric_m_s",
            "o3_flux_w_ug_m2_s",
            "o3_flux_w_from_interp_product_ug_m2_s",
        ]
    ].mean()

    area_profile = area.groupby("height_m", as_index=False)[
        [
            "o3_mass_ug_m3",
            "u_wind_m_s",
            "v_wind_m_s",
            "w_geometric_m_s",
            "o3_flux_w_ug_m2_s",
            "o3_flux_w_from_interp_product_ug_m2_s",
        ]
    ].mean()
    site_profile = site.groupby("height_m", as_index=False)[
        [
            "o3_mass_ug_m3",
            "u_wind_m_s",
            "v_wind_m_s",
            "w_geometric_m_s",
            "o3_flux_w_ug_m2_s",
            "o3_flux_w_from_interp_product_ug_m2_s",
        ]
    ].mean()
    compare = area_profile.merge(site_profile, on="height_m", suffixes=("_area", "_site"))

    o3_matrix = month_height_matrix(monthly_area, "o3_mass_ug_m3")
    flux_matrix = month_height_matrix(monthly_area, "o3_flux_w_from_interp_product_ug_m2_s")
    w_matrix = month_height_matrix(monthly_area, "w_geometric_m_s")
    u_matrix = month_height_matrix(monthly_area, "u_wind_m_s")
    v_matrix = month_height_matrix(monthly_area, "v_wind_m_s")
    valid_count_matrix = month_height_valid_matrix(area, "o3_mass_ug_m3", "count")
    valid_fraction_matrix = month_height_valid_matrix(area, "o3_mass_ug_m3", "fraction")
    daily_selected = selected_height_daily_wide(daily_area)

    available_dates = sorted(area["bjt_date"].dropna().unique())
    sample_date = available_dates[0] if available_dates else None
    sample_hourly = area[area["bjt_date"] == sample_date].copy() if sample_date else area.head(0).copy()

    notes = pd.DataFrame(
        [
            {
                "item": "数据范围",
                "description": "当前工作簿只汇总已经成功下载的 600-1000 hPa ERA5 文件；后续新增 nc 后重新运行脚本即可更新。",
            },
            {
                "item": "高度插值范围",
                "description": "线性插值高度只取 0-4000 m，间隔 100 m；不做外推，所以低于最低有效 ERA5 高度的位置可能为空。",
            },
            {
                "item": "线性插值函数",
                "description": "对任一变量 x，在 h_i <= h <= h_(i+1) 时，x(h)=x_i+(x_(i+1)-x_i)*(h-h_i)/(h_(i+1)-h_i)。y 轴为 height_m，x 可为 O3、u、v、w_geo 等。",
            },
            {
                "item": "位势高度",
                "description": "height_m = geopotential / g，其中 g=9.80665 m s-2。",
            },
            {
                "item": "O3质量浓度",
                "description": "o3_mass_ug_m3 = ozone_mass_mixing_ratio * air_density * 1e9。",
            },
            {
                "item": "几何垂直速度",
                "description": "w_geometric_m_s = -omega / (air_density * g)。正值表示向上输送，负值表示向下输送。",
            },
            {
                "item": "垂直通量",
                "description": "推荐使用 o3_flux_w_from_interp_product_ug_m2_s = o3_mass_ug_m3 * w_geometric_m_s。单位 ug m-2 s-1。",
            },
            {
                "item": "另一种通量",
                "description": "o3_flux_w_ug_m2_s 是先在压力层计算通量再插值到高度层；两者都保留，便于比较。",
            },
            {
                "item": "有效样本矩阵",
                "description": "有效样本数统计每个月、每个高度有多少逐时区域剖面真正参与平均；有效比例=有效样本数/该月该高度总逐时剖面数。低层若比例很低，月均值只能作为参考。",
            },
        ]
    )

    sheets = {
        "说明": notes,
        "下载清单": inventory,
        "月均_区域高度剖面": monthly_area,
        "月均_站点高度剖面": monthly_site,
        "日均_区域高度剖面": daily_area,
        "全时段_区域站点对比": compare,
        "矩阵_O3_区域": o3_matrix,
        "矩阵_垂直通量_区域": flux_matrix,
        "矩阵_垂直速度_区域": w_matrix,
        "矩阵_U风_区域": u_matrix,
        "矩阵_V风_区域": v_matrix,
        "矩阵_有效样本数_区域": valid_count_matrix,
        "矩阵_有效比例_区域": valid_fraction_matrix,
        "日变化_选定高度": daily_selected,
        "示例逐时_区域": sample_hourly,
    }

    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
        writer._sheet_dfs = sheets  # type: ignore[attr-defined]
        for sheet, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
        add_basic_formatting(writer, sheets)
        add_profile_chart(writer, "矩阵_O3_区域", "Area Mean O3 Profiles by Month", "O3 (ug/m3)")
        add_profile_chart(
            writer,
            "矩阵_垂直通量_区域",
            "Area Mean Vertical O3 Flux Profiles by Month",
            "Vertical O3 flux (ug/m2/s)",
        )
        add_profile_chart(writer, "矩阵_垂直速度_区域", "Area Mean Geometric Vertical Velocity", "w (m/s)")
        add_timeseries_chart(writer, "日变化_选定高度", "日均_区域高度剖面")

    print(f"Wrote {OUT_XLSX}")
    print(f"Sheets: {len(sheets)}")


if __name__ == "__main__":
    main()
