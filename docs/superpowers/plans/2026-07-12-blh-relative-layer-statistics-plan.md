# BLH-Relative Layer Statistics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce equal-hour-weighted BLH-relative layer statistics and report-ready figures from the existing 2023 HF/CF ERA5 and WRF datasets.

**Architecture:** Implement pure functions for dynamic layer assignment, pointwise flux calculation, per-hour layer averaging, and summary statistics in one focused module. Render two figures from the summary outputs and write compact CSV files; no local profile data is read.

**Tech Stack:** Python, pandas, NumPy, Matplotlib, pytest, Pillow

## Global Constraints

- Use only ERA5 and WRF unified interpolation CSV files.
- Exclude all SHHKY/local profile inputs.
- Use Beijing time 08:00-16:00 and 0-3 km AGL.
- Average within each hour-layer first, then across hours.

---

### Task 1: Dynamic Layers and Flux Calculations

**Files:**
- Create: `analyze_blh_relative_layers_hf_cf_2023.py`
- Create: `tests/test_analyze_blh_relative_layers.py`

- [ ] Write failing tests for the three layer boundaries, Fadv/TFh/Fturb formulas, and equal-hour weighting.
- [ ] Run `python -m pytest tests/test_analyze_blh_relative_layers.py -q` and verify failure because the module is absent.
- [ ] Implement `assign_blh_layer`, `calculate_diagnostics`, `hourly_layer_means`, and `summarize_layers`.
- [ ] Run the tests and verify all numerical tests pass.

### Task 2: Figures and Production Outputs

**Files:**
- Modify: `analyze_blh_relative_layers_hf_cf_2023.py`
- Modify: `tests/test_analyze_blh_relative_layers.py`

- [ ] Add failing tests for nonblank figure creation and exact output names.
- [ ] Implement the 2x2 source/station comparison and WRF flux-decomposition figures.
- [ ] Add `main()` to read the two production CSV files, write two result CSVs, and generate two PNGs.
- [ ] Run all related tests.
- [ ] Run the production script and inspect all outputs.
- [ ] Commit and push the implementation, tests, design, and plan.
