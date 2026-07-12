# Vertical Exchange O3

本仓库保存臭氧垂直交换研究中的核心数据下载、NetCDF读取、统一高度插值、ERA5/WRF对比、输送通量诊断和论文图复现代码。

## 研究范围

- 研究变量：O3、u、v、omega/几何垂直速度、温度、湿度、位势高度、BLH/PBLH和WRF `EXCH_H`
- 主要站点：上海、合肥科学岛，以及2023年HF城市站和CF农村站
- 统一高度范围：距地0-4 km，常用100 m间隔
- 常用时段：北京时间08:00-16:00

仓库不包含ERA5/WRF原始数据、处理后的CSV/Excel或本地观测文件。相关路径需要按照本机数据目录修改。

## 环境安装

```powershell
python -m pip install -r requirements.txt
```

CDS API凭据应保存在用户主目录的 `.cdsapirc` 中，不要写入脚本或提交到GitHub。

## 核心流程

### 1. ERA5下载

- `queue_era5_2023_hefei_hf_cf_600_1000_download.py`：2023年HF/CF压力层数据队列
- `queue_era5_2023_hefei_hf_cf_blh_download.py`：2023年HF/CF边界层高度数据队列
- `queue_era5_2025_shanghai_600_1000_download.py`：上海600-1000 hPa数据队列
- `queue_era5_2025_hefei_science_island_600_1000_download.py`：合肥科学岛数据队列
- `queue_era5_2025_shanghai_temperature_backfill.py`：温度补充下载
- `download_era5_surface_geopotential_hefei.py`：HF/CF地表位势下载

### 2. NetCDF读取与导出

- `read_era5_nc.py`：检查NetCDF维度、变量、单位和统计范围
- `export_era5_nc_to_csv.py`：将指定NetCDF内容导出为CSV
- `extract_wrf_hf_cf_2023_remote.py`：在服务器端从WRF日文件抽取HF/CF站点所需变量

### 3. 高度插值与数据整理

- `interpolate_era5_600_1000_to_height_flux.py`：ERA5压力层转高度层并计算基础输送量
- `reprocess_era5_interpolated_W_shanghai_hefei.py`：重新计算几何垂直速度并生成插值结果
- `process_era5_wrf_2023_comparison.py`：ERA5与WRF统一到0-4 km、100 m网格并输出配对数据
- `process_existing_era5_for_paper.py`：按论文分析要求整理已有ERA5数据
- `process_era5_blog_paper_combined.py`：融合早期处理流程与论文方法
- `make_era5_height_flux_result_workbook.py`：生成高度廓线与统计Excel结果

压力垂直速度转换为几何垂直速度采用：

```text
w_geo = -omega / (rho * g)
```

其中 `w_geo > 0` 表示向上，`w_geo < 0` 表示向下。

### 4. 通量诊断

- `compute_effective_turbulence_from_era5.py`：基于ERA5垂直速度估算等效湍流系数
- `plot_three_vertical_flux_methods_hf_cf_2023.py`：WRF平流、湍流扩散和总通量，以及ERA5平流通量
- `plot_paper_fig1_fig3_o3_hf_cf_2023.py`：论文Figure 1和Figure 3的O3部分
- `plot_paper_figure4_method_o3_tfw.py`：论文Figure 4方法的O3垂直输送通量

主要公式：

```text
Fadv  = C * w
Fturb = -Kh * dC/dz
Ftotal = Fadv + Fturb
TFu = C * u
TFv = C * v
```

所有平均图均先逐时、逐高度计算通量，再对通量求平均，即使用 `mean(C*w)`、`mean(C*u)` 和 `mean(C*v)`，不是分别平均后相乘。

### 5. 主要分析图

- `plot_figure2_shanghai_hefei_era5_tfw.py`：上海与合肥TFw时间-高度对比
- `plot_figure3_profiles_o3_w_tfw.py`：O3、w和TFw平均廓线
- `plot_figure4_low_layer_tfw_diurnal.py`：低层TFw日变化
- `plot_figures_11_12_hefei_horizontal_flux.py`：合肥水平输送通量
- `plot_annual_signed_advective_flux_o3.py`：年尺度有符号平流通量统计
- `plot_annual_signed_flux_o3_hourly_units.py`：小时单位年尺度统计
- `plot_hefei_local_profile_like_fig3.py`：合肥本地O3廓线统计图

## 测试

```powershell
python -m pytest tests -q
```

核心测试覆盖：

- omega至几何垂直速度的符号与单位转换
- ERA5/WRF统一高度插值
- O3质量浓度换算
- `Fadv`、`Fturb`、`Ftotal`公式
- 水平通量逐点计算后再平均
- Figure 1和Figure 3图片生成与非空检查

## 路径说明

部分历史脚本保留了本地绝对路径，例如 `E:/research/垂直交换/...`。在其他计算机运行前，应修改脚本顶部的 `BASE_DIR`、输入文件和输出目录常量。原始及处理数据均由 `.gitignore` 排除。

