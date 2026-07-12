# ERA5 与 WRF 统一插值和对比 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 分别生成 ERA5、WRF 的 0–4 km 插值统计表，并生成共同期 O3 和几何垂直速度的独立对比表与综合分析图。

**Architecture:** 新建一个处理脚本，使用共享的高度插值和统计函数，但保持 ERA5、WRF 两条读取与单位统一流程独立。二者输出完成并通过质量检查后，再按站点、北京时间和高度进行内连接，计算 O3 和垂直速度差异及统计指标。

**Tech Stack:** Python 3、netCDF4、NumPy、pandas、openpyxl、Matplotlib、pytest。

## Global Constraints

- 共同对比期为 2023-04-01 至 2023-10-31。
- 北京时间 08:00–16:00，目标高度 0–4000 m，间隔 100 m。
- HF、CF 均使用各自最近格点。
- 几何垂直速度统一为正值向上。
- ERA5、WRF 和对比结果分别输出。
- 对比统计和图片仅包含 O3 与几何垂直速度。
- 当前目录不是 Git 仓库，每项任务以验证命令和产物检查作为完成门槛。

---

### Task 1: 共享物理换算和插值函数

**Files:**
- Create: `tests/test_process_era5_wrf_2023_comparison.py`
- Create: `process_era5_wrf_2023_comparison.py`

**Interfaces:**
- Produces: `interp_profile(height, values, targets)`, `omega_to_w_geo(omega, rho)`, `wrf_o3_ppmv_to_ug_m3(o3_ppmv, pressure_hpa, temperature_k)`。

- [ ] **Step 1: 编写插值、符号和臭氧换算测试**

测试线性剖面在 0–4000 m 每 100 m 的结果、`omega > 0` 转成 `w_geo < 0`、WRF ppmv 转质量浓度，并检查缺失值不参与插值。

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_process_era5_wrf_2023_comparison.py -v`

Expected: 因处理函数尚不存在而失败。

- [ ] **Step 3: 实现最小共享函数**

使用 `np.interp` 完成有效高度范围内插值；ERA5 使用 `-omega/(rho*9.80665)`；WRF O3 使用理想气体关系换算为 `ug m-3`。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_process_era5_wrf_2023_comparison.py -v`

Expected: 全部 PASS。

### Task 2: ERA5 最近格点逐小时插值和独立统计

**Files:**
- Modify: `process_era5_wrf_2023_comparison.py`
- Test: `tests/test_process_era5_wrf_2023_comparison.py`

**Interfaces:**
- Consumes: `interp_profile`, `omega_to_w_geo`。
- Produces: `process_era5_files(files, stations)`，返回原始最近格点表、逐小时插值表和文件清单。

- [ ] **Step 1: 增加ERA5样本文件读取测试**

断言时间转换为北京时间、HF/CF最近格点唯一、压力层数量正确、`w_geo` 符号满足换算公式。

- [ ] **Step 2: 实现ERA5读取与转换**

读取 `z,t,q,u,v,w,o3`，计算相对地面高度、湿空气密度、O3质量浓度和向上为正的 `w_geo`，按站点和每个北京时间小时插值。

- [ ] **Step 3: 生成ERA5独立输出**

写出文件清单、原始压力层、逐小时插值、日均、月均和08–16时综合小时平均CSV，并生成ERA5摘要Excel。

- [ ] **Step 4: 验证ERA5输出**

检查2023年4–10月文件完整、每个完整时刻每站理论上最多41个高度层、时间无重复、O3/温度/风速在合理范围内。

### Task 3: WRF模式层逐小时插值和独立统计

**Files:**
- Modify: `process_era5_wrf_2023_comparison.py`
- Test: `tests/test_process_era5_wrf_2023_comparison.py`

**Interfaces:**
- Consumes: `interp_profile`, `wrf_o3_ppmv_to_ug_m3`。
- Produces: `process_wrf_file(path)`，返回模式层统计、逐小时插值和变量单位核验表。

- [ ] **Step 1: 增加WRF样本读取测试**

断言1926个UTC时刻可转换为北京时间08–16时、HF/CF站点存在、`wa`单位为 `m s-1`、质量层高度单调。

- [ ] **Step 2: 确认WRF高度基准**

逐时检查最低 `height`；若为海拔高度则减去最低模式层代表的地面高度，若已是AGL则直接使用，并在输出说明表记录判定依据。

- [ ] **Step 3: 实现WRF插值与近地面质量标记**

对O3、温度、湿度、u、v、wa、rh逐质量层插值；将交错层 `exch_h` 映射到对应层高度后插值；0–100 m缺失按既定规则由100–200 m最近有效层补齐，并写入 `fill_method`。

- [ ] **Step 4: 生成并验证WRF独立输出**

写出逐小时、日均、月均、08–16时综合小时平均CSV和WRF摘要Excel；检查O3派生值与文件中 `o3_mass_concentration` 一致。

### Task 4: O3和垂直速度配对比较

**Files:**
- Modify: `process_era5_wrf_2023_comparison.py`
- Test: `tests/test_process_era5_wrf_2023_comparison.py`

**Interfaces:**
- Consumes: ERA5和WRF逐小时插值表。
- Produces: `build_comparison(era5, wrf)` 和 `comparison_metrics(pairs)`。

- [ ] **Step 1: 编写配对和统计测试**

使用小型已知数据验证内连接键为站点、北京时间、高度，偏差定义为 `ERA5-WRF`，并验证Bias、MAE、RMSE、Pearson r和样本数。

- [ ] **Step 2: 实现逐小时配对表**

只提取 O3 和向上为正的几何垂直速度，生成ERA5值、WRF值、差值和相对差值；缺失不填零。

- [ ] **Step 3: 生成分组统计表**

分别按总体、站点、月份、北京时间小时和高度输出统计指标，并写入独立对比Excel。

- [ ] **Step 4: 验证共同样本覆盖率**

报告各站理论记录数、实际配对数和缺失比例；随机抽查至少一个日期、小时和高度的原值与差值。

### Task 5: 综合分析图和最终核验

**Files:**
- Modify: `process_era5_wrf_2023_comparison.py`

**Interfaces:**
- Consumes: 综合小时平均表和对比统计表。
- Produces: O3与垂直速度的时间—高度图、平均廓线图和一致性图。

- [ ] **Step 1: 绘制O3综合分析图**

每站生成ERA5、WRF和差值三个面板，横轴为北京时间08–16时，纵轴为0–4 km，统一O3色标单位 `ug m-3`。

- [ ] **Step 2: 绘制垂直速度综合分析图**

保持正负号和以零为中心的对称色标，单位 `m s-1`，ERA5、WRF和差值分别显示。

- [ ] **Step 3: 绘制小时平均廓线与一致性图**

按HF、CF分别输出O3及垂直速度的08–16时平均廓线，并绘制ERA5对WRF散点、1:1线和相关系数。

- [ ] **Step 4: 运行完整测试和处理**

Run: `python -m pytest tests/test_process_era5_wrf_2023_comparison.py -v`

Run: `python process_era5_wrf_2023_comparison.py`

Expected: 测试全部PASS，三套数据目录和图片全部生成，无未捕获异常。

- [ ] **Step 5: 最终产物检查**

核对CSV编码、Excel工作表、图片文字和单位；汇总ERA5与WRF在O3及垂直速度上的主要差异，并说明分辨率、模式物理过程和观测代表性造成的限制。
