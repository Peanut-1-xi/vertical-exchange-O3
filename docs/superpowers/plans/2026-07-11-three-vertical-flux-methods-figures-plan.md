# 三种垂直输送方法图 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 使用WRF独立变量生成F_adv、F_turb、F_total三张HF/CF四面板图，并使用ERA5独立变量生成F_adv四面板图。

**Architecture:** 新建独立绘图脚本，读取上午已生成的ERA5和WRF逐小时统一高度CSV。计算函数与图形函数分离，先通过小数组测试三个公式，再复用第四节报告原图三的2×2布局生成四张PNG。

**Tech Stack:** Python 3、NumPy、pandas、Matplotlib、标准库unittest。

## Global Constraints

- WRF三种方法只使用WRF的O3、wa和EXCH_H。
- ERA5 F_adv只使用ERA5的O3和w_geo。
- 时段为2023-04-01至2023-10-31，北京时间08:00–16:00。
- 高度为0–4 km，每100 m一层。
- 正值表示向上通量，负值表示向下通量。
- 只输出四张PNG，不输出CSV或Excel。
- WRF F_turb和F_total明确注明EXCH_H按m2 s-1假定。

---

### Task 1: 通量公式与梯度测试

**Files:**
- Create: `tests/test_plot_three_vertical_flux_methods.py`
- Create: `plot_three_vertical_flux_methods_hf_cf_2023.py`

**Interfaces:**
- Produces: `calculate_vertical_fluxes(frame, source)`，返回增加`f_adv_ug_m2_s`、`o3_gradient_ug_m4`、`f_turb_ug_m2_s`和`f_total_ug_m2_s`的DataFrame。

- [ ] **Step 1: 编写失败测试**

使用三层已知剖面，验证WRF公式：

```python
expected_adv = concentration * vertical_velocity
expected_gradient = np.gradient(concentration, height_m)
expected_turb = -exchange_coefficient * expected_gradient
expected_total = expected_adv + expected_turb
```

另用ERA5样本验证只生成`F_adv=C_ERA5*w_geo_ERA5`，不读取WRF列。

- [ ] **Step 2: 运行测试并确认因函数不存在而失败**

Run: `python -m unittest tests.test_plot_three_vertical_flux_methods -v`

Expected: FAIL，提示无法导入新模块或函数。

- [ ] **Step 3: 实现最小计算函数**

按`station,bjt_time`分组并按`height_agl_m`排序，使用实际米制高度计算梯度；WRF计算三种通量，ERA5只计算平流通量。

- [ ] **Step 4: 运行测试并确认通过**

Run: `python -m unittest tests.test_plot_three_vertical_flux_methods -v`

Expected: 所有测试PASS。

### Task 2: 四面板绘图与四张图片

**Files:**
- Modify: `plot_three_vertical_flux_methods_hf_cf_2023.py`
- Test: `tests/test_plot_three_vertical_flux_methods.py`

**Interfaces:**
- Consumes: `calculate_vertical_fluxes(frame, source)`。
- Produces: `diurnal_matrix`、`daytime_profile`、`plot_four_panel`和四张PNG。

- [ ] **Step 1: 增加统计矩阵测试**

构造两天08、09时数据，验证时间－高度矩阵按`station,bjt_hour,height_agl_m`求平均，日间廓线按`station,height_agl_m`求平均。

- [ ] **Step 2: 实现图形函数**

复用`plot_figure2_shanghai_hefei_era5_tfw.py`的`figsize=(11.5,7.6)`、`width_ratios=[1.45,0.72]`、`TwoSlopeNorm`、零线、字号与色标位置。色标范围取HF/CF共同有效值绝对值98百分位并向上取易读数。

- [ ] **Step 3: 生成四张PNG**

写入：

```text
E:/research/垂直交换/合肥双站_ERA5与WRF_2023/ERA5_WRF统一插值与对比结果/综合分析图/三种垂直输送方法
```

文件为WRF F_adv、WRF F_turb、WRF F_total和ERA5 F_adv四张图，不创建其他数据文件。

- [ ] **Step 4: 验证图片**

Run: `python plot_three_vertical_flux_methods_hf_cf_2023.py`

Expected: 只报告四个PNG路径且无异常。

检查每张PNG非空、像素尺寸一致、包含非白色绘图区；人工查看四图的轴、单位、色标、正负方向和EXCH_H假设说明。

### Task 3: 最终公式与产物核验

**Files:**
- Test: `tests/test_plot_three_vertical_flux_methods.py`

- [ ] **Step 1: 运行全部相关测试**

Run: `python -m unittest tests.test_plot_three_vertical_flux_methods tests.test_process_era5_wrf_2023_comparison -v`

Expected: 旧的16项插值测试及新增通量测试全部PASS。

- [ ] **Step 2: 抽查真实数据公式恒等关系**

从HF和CF各抽一个时次、高度，检查`F_adv-C*w`、`F_turb+Kz*dC/dz`、`F_total-F_adv-F_turb`的绝对误差小于`1e-10`。

- [ ] **Step 3: 检查输出范围**

确认横轴只有08–16，纵轴0–4 km，WRF三图使用WRF O3，ERA5图使用ERA5 O3，并确认输出目录没有新增表格文件。
