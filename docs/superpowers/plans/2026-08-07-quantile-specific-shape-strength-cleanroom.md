# Quantile-Specific Shape-Strength Clean-Room Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 clean-room 演示项目的 `shape_strength` 更新为 train-only quantile-specific `H_q(x)` 与连续容量反解，同时保持各分位头独立且不做跨分位排序。

**Architecture:** `02_quantile_risk_model.ipynb` 用真实请求分位训练 direct quantile 模型，并以 direct 模型在 `x_adv=0` 的预测作为 `B_q(X)`，再从训练期残差构造按 `side × quantile` 的单调 `H_q(x)` 和 strength 模型。`03_capacity_inversion.ipynb` 在公开 45 点行动网格上按相同分位插值 `H_q`，生成 `B_q+A_qH_q` 的 absolute final quantile 曲线，只在单条 `x_adv` 曲线上 `cummax` 后分段线性反解 10bps 容量。

**Tech Stack:** Python 3.11、Jupyter Notebook、pandas、NumPy、LightGBM、joblib、unittest、nbconvert

---

### Task 1: Add Failing Clean-Room Contracts

**Files:**
- Modify: `tests/test_notebook_contracts.py`
- Test: `tests/test_notebook_contracts.py`

- [ ] **Step 1: Write the failing model contract test**

Add assertions requiring the new public behavior and rejecting the old proxy-tau implementation:

```python
def test_model_notebook_uses_train_only_quantile_specific_shape(self):
    source = notebook_source("02_quantile_risk_model.ipynb")
    for token in [
        "SHAPE_H_MODE = 'quantile_specific_train_curve'",
        "build_quantile_specific_shape_curves",
        "interpolate_quantile_shape",
        "base_prediction_at_x0",
        "02_shape_curve_diagnostics.csv",
        "cross_quantile_order_enforced': False",
    ]:
        self.assertIn(token, source)
    self.assertNotIn("RAW_TAU_MAP", source)
    self.assertNotIn("enforce_quantile_order", source)
```

- [ ] **Step 2: Write the failing capacity contract test**

```python
def test_capacity_notebook_uses_absolute_quantile_specific_continuous_curve(self):
    source = notebook_source("03_capacity_inversion.ipynb")
    for token in [
        "CAPACITY_RISK_MEASURE = 'absolute_final_quantile'",
        "quantile_specific_H_piecewise_linear",
        "continuous_prediction_source",
        "h_support_status",
        "shape_h_mode",
    ]:
        self.assertIn(token, source)
    self.assertNotIn("q99_final >= q95_final", source)
```

- [ ] **Step 3: Run the narrow test and verify RED**

Run:

```powershell
python -m unittest tests.test_notebook_contracts
```

Expected: both new tests fail because the required quantile-specific functions and capacity metadata are absent; existing tests continue to pass.

- [ ] **Step 4: Commit the failing tests**

```powershell
git add tests/test_notebook_contracts.py
git commit -m "test: require quantile-specific shape capacity"
```

### Task 2: Implement Quantile-Specific Shape Training

**Files:**
- Modify: `notebooks/02_quantile_risk_model.ipynb`
- Modify: `tests/test_notebook_contracts.py`
- Test: `tests/test_notebook_contracts.py`

- [ ] **Step 1: Replace proxy quantile mapping with true requested heads**

Keep one registry and pass the actual quantile to LightGBM:

```python
QUANTILE_REGISTRY = [0.50, 0.80, 0.85, 0.90, 0.95, 0.99]
SHAPE_H_MODE = 'quantile_specific_train_curve'

for quantile in QUANTILE_REGISTRY:
    model = LGBMRegressor(**model_parameters(quantile))
    model.fit(X_train, y_total_train)
    direct_models[quantile] = model
```

- [ ] **Step 2: Add side-aware train-only shape builders**

Add self-contained Notebook functions with this interface and behavior:

```python
def build_quantile_specific_shape_curves(
    train_frame: pd.DataFrame,
    base_prediction_at_x0: dict[float, np.ndarray],
    quantiles: list[float],
) -> tuple[dict[float, pd.DataFrame], pd.DataFrame]:
    rows = []
    curves = {}
    for quantile in quantiles:
        residual = np.maximum(
            pd.to_numeric(train_frame['total_bad_move_bps'], errors='coerce').to_numpy()
            - base_prediction_at_x0[quantile],
            0.0,
        )
        working = train_frame[['side', 'x_adv']].copy()
        working['increment_bad_bps'] = residual
        quantile_rows = []
        for side, side_frame in working.groupby('side', sort=True):
            curve = side_frame.groupby('x_adv', as_index=False)['increment_bad_bps'].quantile(quantile)
            curve = curve.sort_values('x_adv')
            curve['raw_q_bps'] = curve.pop('increment_bad_bps')
            curve['mono_q_bps'] = np.maximum.accumulate(np.maximum(curve['raw_q_bps'], 0.0))
            reference = float(np.interp(0.01, curve['x_adv'], curve['mono_q_bps']))
            scale = reference if np.isfinite(reference) and reference > 1e-9 else max(float(curve['mono_q_bps'].max()), 1.0)
            curve['h_value'] = curve['mono_q_bps'] / scale
            curve['side'] = side
            curve['quantile_level'] = quantile
            curve['shape_h_mode'] = SHAPE_H_MODE
            quantile_rows.append(curve)
        curves[quantile] = pd.concat(quantile_rows, ignore_index=True)
        rows.extend(curves[quantile].to_dict('records'))
    return curves, pd.DataFrame(rows)


def interpolate_quantile_shape(
    frame: pd.DataFrame,
    curve: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(len(frame), dtype=float)
    status = np.full(len(frame), 'missing_side_shape', dtype=object)
    for side, index in frame.groupby('side', sort=False).groups.items():
        side_curve = curve[curve['side'].eq(side)].sort_values('x_adv')
        if side_curve.empty:
            continue
        x = pd.to_numeric(frame.loc[index, 'x_adv'], errors='coerce').to_numpy(dtype=float)
        values[index] = np.interp(x, side_curve['x_adv'], side_curve['h_value'], left=0.0, right=float(side_curve['h_value'].iloc[-1]))
        status[index] = np.where(x > float(side_curve['x_adv'].max()), 'right_clipped', 'in_support')
    return values, status
```

- [ ] **Step 3: Train `B_q(X)+A_q(X)H_q(x)` without future information**

For every quantile, evaluate its direct model on a copy of training rows with `x_adv=0` to obtain `base_prediction_at_x0`; build train-only `H_q`; train the strength model on non-negative incremental labels divided by `max(H_q, 0.05)`. Prediction on calibration/test is:

```python
shape_raw = np.maximum(base_q_x0 + strength_q * h_q, 0.0)
```

Do not use calibration or test rows in the shape builder. Keep calibration buffers and causal history logic unchanged.

- [ ] **Step 4: Persist diagnostics and bundle metadata**

Write `outputs/02_shape_curve_diagnostics.csv`. Store `shape_h_mode`, `shape_reference_x_adv`, `shape_grids`, `cross_quantile_order_enforced=False`, and `final_month_learning_excluded=True` in the bundle/stage contract. Remove `raw_tau_map` from both outputs.

- [ ] **Step 5: Run the model contract test and verify GREEN**

```powershell
python -m unittest tests.test_notebook_contracts
```

Expected: all Notebook contract tests pass.

- [ ] **Step 6: Commit the model implementation**

```powershell
git add notebooks/02_quantile_risk_model.ipynb tests/test_notebook_contracts.py
git commit -m "feat: train quantile-specific shape curves"
```

### Task 3: Implement Continuous Absolute-Risk Capacity

**Files:**
- Modify: `notebooks/03_capacity_inversion.ipynb`
- Test: `tests/test_notebook_contracts.py`

- [ ] **Step 1: Declare the stable capacity contract**

```python
CAPACITY_RISK_MEASURE = 'absolute_final_quantile'
SHAPE_H_MODE = str(BUNDLE['shape_h_mode'])
CONTINUOUS_PREDICTION_SOURCE = 'quantile_specific_H_piecewise_linear'
```

- [ ] **Step 2: Generate per-quantile shape curves on the 45-point grid**

For each shape quantile, create an `x_adv=0` copy of the grid for `B_q(X)`, evaluate `A_q(X)` on condition features, interpolate the matching side-aware `H_q`, and compute:

```python
total_raw = np.maximum(base_q_x0 + strength_q * h_values, 0.0)
candidate['shape_h_mode'] = SHAPE_H_MODE
candidate['capacity_risk_measure'] = CAPACITY_RISK_MEASURE
candidate['continuous_prediction_source'] = CONTINUOUS_PREDICTION_SOURCE
candidate['h_support_status'] = h_support_status
```

Direct-model candidates remain evaluated on the requested grid. Both model families retain their independent quantile values.

- [ ] **Step 3: Keep monotonicity within each candidate curve only**

Group by one `anchor_id × model_family × quantile_level` curve, apply `cummax` only in increasing `x_adv`, and feed the monotone total/impact maximum into `invert_monotone_boundary`. Do not compare or reorder different quantile heads.

- [ ] **Step 4: Fail closed outside supported shape range**

Set `h_support_status='right_clipped'` above the largest train-supported `x_adv`. If the selected boundary equals the grid maximum while the final point is right-clipped, set `capacity_status='support_cap_reached'`; keep zero capacity when no-order risk exceeds 10bps.

- [ ] **Step 5: Persist stable metadata**

Add `shape_h_mode`, `capacity_risk_measure`, `continuous_prediction_source`, and aggregated `h_support_status` to `03_capacity_candidates.parquet`, the curve sample, and `03_stage_contract.json`. Preserve existing output filenames and skipped/fail-closed semantics.

- [ ] **Step 6: Run the capacity contract test and verify GREEN**

```powershell
python -m unittest tests.test_notebook_contracts
```

Expected: all Notebook contract tests pass, with no cross-quantile order token.

- [ ] **Step 7: Commit the capacity implementation**

```powershell
git add notebooks/03_capacity_inversion.ipynb
git commit -m "feat: invert continuous quantile-specific risk curves"
```

### Task 4: Execute Smoke Pipeline and Update Public Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/methodology.md`
- Modify: `reference/reference_run_manifest.json`
- Modify: `PROVENANCE.csv`
- Test: `tests/test_smoke_pipeline.py`
- Test: `tests/test_release_builder.py`

- [ ] **Step 1: Run the full smoke Notebook chain**

```powershell
$env:RUN_MODE='smoke'
python -m nbconvert --to notebook --execute notebooks/00_get_data.ipynb --output 00.executed.ipynb --output-dir $env:TEMP --ExecutePreprocessor.timeout=600
python -m nbconvert --to notebook --execute notebooks/01_build_panel_and_labels.ipynb --output 01.executed.ipynb --output-dir $env:TEMP --ExecutePreprocessor.timeout=600
python -m nbconvert --to notebook --execute notebooks/02_quantile_risk_model.ipynb --output 02.executed.ipynb --output-dir $env:TEMP --ExecutePreprocessor.timeout=1200
python -m nbconvert --to notebook --execute notebooks/03_capacity_inversion.ipynb --output 03.executed.ipynb --output-dir $env:TEMP --ExecutePreprocessor.timeout=1200
python -m nbconvert --to notebook --execute notebooks/04_optimization_and_report.ipynb --output 04.executed.ipynb --output-dir $env:TEMP --ExecutePreprocessor.timeout=1200
```

Expected: all five execute successfully and `04` remains `same-sample policy-development diagnostic`.

- [ ] **Step 2: Inspect smoke outputs**

Verify `02_model_metrics.csv`, `02_grouped_coverage.csv`, `02_shape_curve_diagnostics.csv`, `03_monotonicity_diagnostics.csv`, and `03_capacity_candidates.parquet`. Confirm all postprocessed curve drop counts are zero, capacity uses `absolute_final_quantile`, and no quantile-order enforcement field is true.

- [ ] **Step 3: Update documentation and run manifest**

Document quantile-specific train-only H, true requested quantile objectives, absolute-risk continuous capacity, and the no-cross-order policy. Update the smoke run manifest with the new model mode while retaining `full_run_status=not_run` and the same-sample policy disclosure.

- [ ] **Step 4: Refresh provenance hashes**

Compute SHA256 for every modified tracked file and update its existing `PROVENANCE.csv` row; add a row for this plan document. Do not add internal paths, source hashes, caches, or training artifacts.

- [ ] **Step 5: Run all verification**

```powershell
python -m unittest
python -m json.tool notebooks/02_quantile_risk_model.ipynb > $null
python -m json.tool notebooks/03_capacity_inversion.ipynb > $null
git diff --check
git status --short
```

Expected: all tests pass, Notebook JSON is valid, no forbidden paths are detected, and only intended tracked files are modified.

- [ ] **Step 6: Commit final documentation and provenance**

```powershell
git add README.md docs/methodology.md reference/reference_run_manifest.json PROVENANCE.csv docs/superpowers/plans/2026-08-07-quantile-specific-shape-strength-cleanroom.md
git commit -m "docs: record quantile-specific shape-strength reproduction"
```

### Task 5: Final Branch Verification and GitHub Delivery

**Files:**
- Verify only: entire repository

- [ ] **Step 1: Re-run the complete test suite from a clean status**

```powershell
python -m unittest
git status --short --branch
git log --oneline -5
```

Expected: all tests pass and the feature branch contains only the planned clean-room commits.

- [ ] **Step 2: Push the feature branch**

```powershell
git push -u origin codex/quantile-specific-shape-strength
```

- [ ] **Step 3: Integrate after final verification**

Use the finishing-a-development-branch workflow to present merge/push options. Do not alter the existing `data-v1.0.0` Release.
