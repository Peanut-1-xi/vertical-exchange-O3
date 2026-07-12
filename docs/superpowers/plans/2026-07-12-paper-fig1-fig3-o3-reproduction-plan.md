# Paper Figure 1 and Figure 3 O3 Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate ERA5 and WRF versions of the O3 panels from paper Figure 1 and Figure 3 for HF and CF during April-October 2023.

**Architecture:** Add one focused plotting module that loads the existing unified 100 m interpolation CSVs, validates source-consistent variables, computes pointwise horizontal O3 flux vectors, aggregates them by hour and height, and renders four publication-ready figures. Keep numerical transformations in pure functions so calculation order and vector direction can be unit tested independently from Matplotlib.

**Tech Stack:** Python 3.13, pandas, NumPy, Matplotlib, pytest, Pillow

## Global Constraints

- Use HF and CF data from 2023-04-01 through 2023-10-31, Beijing time 08:00-16:00, 0-3 km AGL.
- ERA5 figures must use only ERA5 O3, u, v, and BLH; WRF figures must use only WRF O3, u, v, and PBLH.
- Calculate `TFu=C*u` and `TFv=C*v` pointwise before any averaging.
- Figure 1 uses O3 in `μg m^-3`; Figure 3 uses horizontal flux in `μg m^-2 s^-1`.
- Missing values remain missing; do not interpolate across missing time periods.
- Output exactly four PNG files at 300 dpi or higher.

---

### Task 1: Horizontal Flux and Aggregation Functions

**Files:**
- Create: `plot_paper_fig1_fig3_o3_hf_cf_2023.py`
- Create: `tests/test_plot_paper_fig1_fig3_o3.py`

**Interfaces:**
- Consumes: DataFrames containing `station`, `bjt_time`, `bjt_hour`, `height_agl_m`, `o3_mass_ug_m3`, `u_wind_m_s`, `v_wind_m_s`, and `pblh_m`.
- Produces: `calculate_horizontal_flux(frame) -> DataFrame`, `aggregate_hour_height(frame) -> DataFrame`, and `aggregate_daytime_profile(frame) -> DataFrame`.

- [ ] **Step 1: Write failing tests for pointwise multiplication and aggregation order**

```python
import numpy as np
import pandas as pd

from plot_paper_fig1_fig3_o3_hf_cf_2023 import (
    aggregate_daytime_profile,
    aggregate_hour_height,
    calculate_horizontal_flux,
)


def sample_frame():
    return pd.DataFrame(
        {
            "station": ["HF", "HF"],
            "bjt_time": pd.to_datetime(["2023-04-01 08:00", "2023-04-02 08:00"]),
            "bjt_hour": [8, 8],
            "height_agl_m": [200.0, 200.0],
            "o3_mass_ug_m3": [10.0, 30.0],
            "u_wind_m_s": [2.0, -1.0],
            "v_wind_m_s": [1.0, 3.0],
            "pblh_m": [500.0, 700.0],
        }
    )


def test_horizontal_flux_is_multiplied_before_averaging():
    flux = calculate_horizontal_flux(sample_frame())
    np.testing.assert_allclose(flux["tfu_ug_m2_s"], [20.0, -30.0])
    np.testing.assert_allclose(flux["tfv_ug_m2_s"], [10.0, 90.0])

    hourly = aggregate_hour_height(flux)
    assert hourly.loc[0, "tfu_mean"] == -5.0
    assert hourly.loc[0, "tfv_mean"] == 50.0
    assert hourly.loc[0, "tfh_mean"] == np.hypot(-5.0, 50.0)
    assert hourly.loc[0, "tfu_mean"] != flux["o3_mass_ug_m3"].mean() * flux["u_wind_m_s"].mean()


def test_daytime_profile_averages_vector_components_before_magnitude():
    profile = aggregate_daytime_profile(calculate_horizontal_flux(sample_frame()))
    assert profile.loc[0, "tfu_mean"] == -5.0
    assert profile.loc[0, "tfv_mean"] == 50.0
    assert profile.loc[0, "tfh_mean"] == np.hypot(-5.0, 50.0)
```

- [ ] **Step 2: Run tests and verify that imports fail**

Run: `python -m pytest tests/test_plot_paper_fig1_fig3_o3.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'plot_paper_fig1_fig3_o3_hf_cf_2023'`.

- [ ] **Step 3: Implement the numerical functions**

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_horizontal_flux(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["tfu_ug_m2_s"] = result["o3_mass_ug_m3"] * result["u_wind_m_s"]
    result["tfv_ug_m2_s"] = result["o3_mass_ug_m3"] * result["v_wind_m_s"]
    return result


def aggregate_hour_height(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(["station", "bjt_hour", "height_agl_m"], as_index=False)
        .agg(
            o3_mean=("o3_mass_ug_m3", "mean"),
            tfu_mean=("tfu_ug_m2_s", "mean"),
            tfv_mean=("tfv_ug_m2_s", "mean"),
            pblh_mean=("pblh_m", "mean"),
        )
    )
    grouped["tfh_mean"] = np.hypot(grouped["tfu_mean"], grouped["tfv_mean"])
    return grouped


def aggregate_daytime_profile(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(["station", "height_agl_m"], as_index=False)
        .agg(
            tfu_mean=("tfu_ug_m2_s", "mean"),
            tfv_mean=("tfv_ug_m2_s", "mean"),
        )
    )
    grouped["tfh_mean"] = np.hypot(grouped["tfu_mean"], grouped["tfv_mean"])
    return grouped
```

- [ ] **Step 4: Run the numerical tests**

Run: `python -m pytest tests/test_plot_paper_fig1_fig3_o3.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit Task 1**

```powershell
git add plot_paper_fig1_fig3_o3_hf_cf_2023.py tests/test_plot_paper_fig1_fig3_o3.py
git commit -m "feat: calculate horizontal ozone flux vectors"
```

---

### Task 2: Source Validation and Figure Rendering

**Files:**
- Modify: `plot_paper_fig1_fig3_o3_hf_cf_2023.py`
- Modify: `tests/test_plot_paper_fig1_fig3_o3.py`

**Interfaces:**
- Consumes: Existing ERA5 and WRF interpolation CSV paths plus a source label `ERA5` or `WRF`.
- Produces: `read_source(path: Path, source: str) -> DataFrame`, `plot_figure1_o3(frame, source, output_path)`, and `plot_figure3_o3(frame, source, output_path)`.

- [ ] **Step 1: Add failing source-validation and plotting tests**

```python
from pathlib import Path

from PIL import Image
import pytest

from plot_paper_fig1_fig3_o3_hf_cf_2023 import (
    plot_figure1_o3,
    plot_figure3_o3,
    validate_frame,
)


def test_validate_frame_rejects_missing_required_wind():
    with pytest.raises(KeyError, match="v_wind_m_s"):
        validate_frame(sample_frame().drop(columns="v_wind_m_s"), "ERA5")


def test_plot_functions_create_nonblank_pngs(tmp_path: Path):
    frame = pd.concat(
        [sample_frame(), sample_frame().assign(station="CF", bjt_hour=9)],
        ignore_index=True,
    )
    frame = calculate_horizontal_flux(frame)
    fig1 = tmp_path / "fig1.png"
    fig3 = tmp_path / "fig3.png"
    plot_figure1_o3(frame, "ERA5", fig1)
    plot_figure3_o3(frame, "ERA5", fig3)
    for path in (fig1, fig3):
        with Image.open(path) as image:
            assert image.width >= 1800
            assert image.height >= 1000
            assert np.asarray(image.convert("RGB")).std() > 5.0
```

- [ ] **Step 2: Run tests and verify that new interfaces fail**

Run: `python -m pytest tests/test_plot_paper_fig1_fig3_o3.py -q`

Expected: FAIL because `validate_frame`, `plot_figure1_o3`, and `plot_figure3_o3` are not yet defined.

- [ ] **Step 3: Implement validation and data loading**

```python
from pathlib import Path

HOURS = list(range(8, 17))
HEIGHT_MAX_M = 3000.0
REQUIRED_COLUMNS = [
    "station", "bjt_time", "bjt_hour", "height_agl_m",
    "o3_mass_ug_m3", "u_wind_m_s", "v_wind_m_s", "pblh_m",
]


def validate_frame(frame: pd.DataFrame, source: str) -> None:
    if source.upper() not in {"ERA5", "WRF"}:
        raise ValueError("source must be ERA5 or WRF")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(", ".join(missing))


def read_source(path: Path, source: str) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=REQUIRED_COLUMNS, encoding="utf-8-sig")
    validate_frame(frame, source)
    frame["bjt_time"] = pd.to_datetime(frame["bjt_time"])
    selected = frame[
        frame["station"].isin(["HF", "CF"])
        & frame["bjt_hour"].isin(HOURS)
        & frame["height_agl_m"].between(0.0, HEIGHT_MAX_M)
    ].copy()
    return selected.dropna(subset=REQUIRED_COLUMNS[2:])
```

- [ ] **Step 4: Implement Figure 1 and Figure 3 plotting**

Implement `plot_figure1_o3` with two side-by-side `pcolormesh` axes sharing one O3 colorbar and overlaid station-hour mean PBLH. Implement `plot_figure3_o3` with two rows and two columns: heatmap plus quiver and PBLH on the left, daytime vector-magnitude profile plus direction arrows on the right. Use one shared flux colorbar and set all labels through `set_chinese_font()` using `Microsoft YaHei`, `SimHei`, then `DejaVu Sans` fallback.

The Figure 3 arrow inputs must be normalized only for display:

```python
speed = np.hypot(tfu, tfv)
arrow_u = np.divide(tfu, speed, out=np.zeros_like(tfu), where=speed > 0)
arrow_v = np.divide(tfv, speed, out=np.zeros_like(tfv), where=speed > 0)
axis.quiver(hours, heights_km, arrow_u, arrow_v, color="black", pivot="middle")
```

The heatmap values must remain the unnormalized `tfh_mean` values. Save with `dpi=300`, `bbox_inches="tight"`, then close each figure.

- [ ] **Step 5: Run all tests for the new module**

Run: `python -m pytest tests/test_plot_paper_fig1_fig3_o3.py -q`

Expected: `4 passed`.

- [ ] **Step 6: Commit Task 2**

```powershell
git add plot_paper_fig1_fig3_o3_hf_cf_2023.py tests/test_plot_paper_fig1_fig3_o3.py
git commit -m "feat: render paper-style O3 Figures 1 and 3"
```

---

### Task 3: Production Generation and Visual Verification

**Files:**
- Modify: `plot_paper_fig1_fig3_o3_hf_cf_2023.py`
- Create outputs under: `E:\research\垂直交换\合肥双站_ERA5与WRF_2023\ERA5_WRF统一插值与对比结果\综合分析图\论文Fig1与Fig3_O3复现\`

**Interfaces:**
- Consumes: `ERA5_HF_CF_202304-202310_逐小时0-4km插值.csv` and `WRF_HF_CF_202304-202310_逐小时0-4km插值.csv`.
- Produces: Four named PNG files defined in the design specification.

- [ ] **Step 1: Add the production entry point**

```python
def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source, path in (("ERA5", ERA5_FILE), ("WRF", WRF_FILE)):
        frame = calculate_horizontal_flux(read_source(path, source))
        plot_figure1_o3(
            frame,
            source,
            output_dir / f"{source}_Fig1_O3_HF_CF时间高度分布.png",
        )
        plot_figure3_o3(
            frame,
            source,
            output_dir / f"{source}_Fig3_O3水平输送通量_HF_CF.png",
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the complete test suite for the related data and plotting modules**

Run: `python -m pytest tests/test_plot_paper_fig1_fig3_o3.py tests/test_plot_three_vertical_flux_methods.py tests/test_process_era5_wrf_2023_comparison.py -q`

Expected: all tests PASS.

- [ ] **Step 3: Generate the four production figures**

Run: `python plot_paper_fig1_fig3_o3_hf_cf_2023.py`

Expected: process exits with code 0 and the output directory contains exactly the four specified PNG files.

- [ ] **Step 4: Verify dimensions, nonblank pixels, and output names**

```powershell
python -c "from pathlib import Path; from PIL import Image; import numpy as np; p=Path(r'E:\research\垂直交换\合肥双站_ERA5与WRF_2023\ERA5_WRF统一插值与对比结果\综合分析图\论文Fig1与Fig3_O3复现'); fs=sorted(p.glob('*.png')); assert len(fs)==4; [(lambda im: (print(f.name, im.size), (_ for _ in ()).throw(AssertionError('blank')) if np.asarray(im.convert('RGB')).std()<=5 else None))(Image.open(f)) for f in fs]"
```

Expected: four filenames and dimensions are printed; no assertion is raised.

- [ ] **Step 5: Visually inspect all four figures**

Open each PNG and confirm: HF/CF panels are present, 08-16 and 0-3 km axes are correct, BLH/PBLH curves are visible, Figure 3 arrows are coherent and not oversized, shared colorbars are readable, Chinese text is not garbled, and no labels overlap.

- [ ] **Step 6: Commit Task 3**

```powershell
git add plot_paper_fig1_fig3_o3_hf_cf_2023.py
git commit -m "feat: generate ERA5 and WRF O3 paper reproductions"
```
