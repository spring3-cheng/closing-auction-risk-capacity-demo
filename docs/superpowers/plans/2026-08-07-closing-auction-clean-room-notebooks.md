# Closing Auction Clean-room Notebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a physically isolated, publication-safe five-Notebook project that downloads an authorized real-data Release bundle and reproduces multi-quantile closing-auction risk, capacity, and same-sample December policy diagnostics. The final GitHub repository is private by user choice, so full-data acquisition also supports token-authenticated Release downloads.

**Architecture:** The Git repository contains only notebooks, tests, documentation, a deterministic smoke dataset, and manifests. A release builder sanitizes four authorized parquet inputs into one versioned ZIP; the notebooks consume relative paths and persist stable stage outputs. Each quantile is an independent policy candidate, while monotonicity is enforced only along that candidate's Cash/ADV curve.

**Tech Stack:** Python 3, JupyterLab, pandas, NumPy, PyArrow, scikit-learn, LightGBM, joblib, matplotlib, seaborn, nbformat, nbconvert, unittest.

**Execution policy:** Work only in the clean-room repository root. Do not initialize a remote, commit, or push during this plan. Do not copy legacy Notebook code or Git metadata.

---

## File Map

- `README.md`: new-device reproduction instructions and interpretation limits.
- `.gitignore`: excludes full data, Release ZIP, outputs, models, caches, and Notebook checkpoints.
- `DATA_LICENSE.md`: non-sensitive authorization and redistribution statement.
- `PROVENANCE.csv`: per-file provenance and allowed-use ledger.
- `environment.yml`, `requirements-lock.txt`: reproducible Python environment.
- `scripts/build_release_data.py`: local-only release/smoke data sanitation and manifest builder.
- `data/data_manifest.json`: public Release asset contract and hashes.
- `data/smoke/*.parquet`: deterministic authorized smoke inputs.
- `notebooks/00_get_data.ipynb`: download/extract/validate data.
- `notebooks/01_build_panel_and_labels.ipynb`: label and feature contract.
- `notebooks/02_quantile_risk_model.ipynb`: direct and shape-strength multi-quantile models.
- `notebooks/03_capacity_inversion.ipynb`: candidate grid scoring and monotone capacity inversion.
- `notebooks/04_optimization_and_report.ipynb`: B3-constrained policy selection and B1/B2/B3 reporting.
- `tests/test_release_builder.py`: sanitation and manifest tests.
- `tests/test_notebook_contracts.py`: notebook structure and clean-room tests.
- `tests/test_smoke_pipeline.py`: five-stage smoke execution and output contract tests.
- `.github/workflows/notebook-smoke.yml`: public CI smoke pipeline.

### Task 1: Scaffold public repository metadata

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `DATA_LICENSE.md`
- Create: `PROVENANCE.csv`
- Create: `environment.yml`
- Create: `requirements-lock.txt`

- [ ] **Step 1: Write `.gitignore` before creating data**

Exclude `data/raw/`, `data/processed/`, `release/`, `outputs/`, `models/`, `.ipynb_checkpoints/`, `.part`, caches, and Python build artifacts while explicitly allowing `data/smoke/*.parquet` and `data/data_manifest.json`.

- [ ] **Step 2: Pin the verified environment**

Use the locally verified versions: NumPy 2.4.6, pandas 2.3.3, PyArrow 24.0.0, scikit-learn 1.8.0, LightGBM 4.6.0, matplotlib 3.10.9, seaborn 0.13.2, nbformat 5.10.4, nbconvert 7.17.1, JupyterLab 4.6.1, joblib 1.5.3.

- [ ] **Step 3: Add legal and provenance boundaries**

Record that the public project starts from authorized derived data, does not contain the internal source system, and does not grant rights beyond the supplied redistribution authorization. Create exact provenance columns from the approved design.

### Task 2: Test release sanitation before implementation

**Files:**
- Create: `tests/test_release_builder.py`
- Create: `scripts/build_release_data.py`

- [ ] **Step 1: Write failing sanitation tests**

The tests create tiny temporary parquet frames and require these public functions:

```python
from scripts.build_release_data import sanitize_b3, sanitize_panel, sha256_file

def test_sanitize_panel_removes_internal_trace_columns():
    frame = pd.DataFrame({"source_file": ["chunk.parquet"], "_impact_sample_stratum": ["m|buy"], "x_adv": [0.01]})
    result = sanitize_panel(frame)
    assert "source_file" not in result
    assert "sample_stratum" in result

def test_sanitize_b3_removes_shard_trace_columns():
    frame = pd.DataFrame({"source_file": ["x"], "source_shard_id": [1], "b3_oracle_refined_x_safe": [0.01]})
    result = sanitize_b3(frame)
    assert "source_file" not in result
    assert "source_shard_id" not in result
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `python -m unittest tests.test_release_builder -v`

Expected: import or missing-function failure.

- [ ] **Step 3: Implement deterministic sanitation helpers**

Implement `sha256_file`, `sanitize_panel`, `sanitize_b3`, `write_parquet`, `build_smoke_inputs`, `build_release_bundle`, and `scan_forbidden_paths`. Use PyArrow ZSTD parquet output and stable JSON serialization.

- [ ] **Step 4: Re-run release builder unit tests**

Expected: all sanitation and hash tests pass.

### Task 3: Build sanitized full and smoke data

**Files:**
- Create: `release/closing_auction_demo_data_v1.zip`
- Create: `release/data-v1.0.0/*.parquet`
- Create: `data/data_manifest.json`
- Create: `data/smoke/*.parquet`

- [ ] **Step 1: Validate four exact source files**

Read only these filenames from the local directory supplied through the required `--source-dir` argument:

```text
total_experiment_counterfactual_grid.parquet
b1_zero_info_10bps_rematch.parquet
b3_adaptive_oracle_boundary_resume_refinement_zero_capacity_final.parquet
fill_safe_capacity_min0p950.parquet
```

Fail if a source is absent, has zero rows, or lacks required columns.

- [ ] **Step 2: Stream-write sanitized parquet files**

Drop or rename only the approved trace columns. Preserve source rows and business columns. Record row count, column count, date range, SHA256, byte count, Cash/ADV grid, sample mode, and evaluation scope.

- [ ] **Step 3: Generate deterministic smoke inputs**

Select a fixed-seed stratified panel sample across month, side, and `x_adv`; ensure December B3 keys, B1 rows, and fill rows overlap with selected panel anchors. Smoke data is execution-only and must carry `evaluation_scope=smoke_contract_only`.

- [ ] **Step 4: Package and hash the Release ZIP**

Store the four sanitized parquets plus the full manifest in `closing_auction_demo_data_v1.zip`. Write the ZIP SHA256 and Release metadata to repository `data/data_manifest.json`.

- [ ] **Step 5: Scan staged public assets**

Fail on absolute personal/company paths, credentials, old `.git` data, `source_file`, or `source_shard_id` in public data schemas and text files.

### Task 4: Write Notebook contract tests first

**Files:**
- Create: `tests/test_notebook_contracts.py`

- [ ] **Step 1: Require the five exact Notebook files**

Load every Notebook with `nbformat.read(..., as_version=4)` and assert ordered filenames `00` through `04` exist.

- [ ] **Step 2: Require self-contained business logic**

Assert notebooks do not import repository business modules, do not contain old absolute paths, and contain stage-specific output contracts.

- [ ] **Step 3: Require approved quantile semantics**

Assert Notebook 02 contains configurable quantile registry, raw-tau map, crossing diagnostics, and no cross-quantile cumulative-order enforcement. Assert Notebook 03 contains per-candidate Cash/ADV cumulative maximum and a 45-point scoring grid.

- [ ] **Step 4: Run and verify failure before Notebook creation**

Run: `python -m unittest tests.test_notebook_contracts -v`

Expected: missing Notebook failures.

### Task 5: Implement data acquisition and label Notebooks

**Files:**
- Create: `notebooks/00_get_data.ipynb`
- Create: `notebooks/01_build_panel_and_labels.ipynb`

- [ ] **Step 1: Implement `00_get_data.ipynb`**

Define repository-root discovery, `RUN_MODE`, atomic download, ZIP SHA validation, extraction, per-file validation, and `outputs/00_active_data.json`. Smoke mode uses tracked smoke parquets; full mode uses Release or local ZIP.

- [ ] **Step 2: Implement label reconstruction in Notebook 01**

Use:

```python
side_sign = np.where(panel["side"].str.startswith("sell"), -1.0, 1.0)
impact_rebuilt = np.maximum(side_sign * panel["impact_me_raw_bps"], 0.0)
total_rebuilt = np.maximum(side_sign * panel["total_move_bps"], 0.0)
```

Fail if the rebuilt/provided label mismatch exceeds numerical tolerance on non-null rows.

- [ ] **Step 3: Build explicit feature contract and anchors**

Use an allowlist of 14:59:59 numeric/context fields and explicitly block market-move/close/counterfactual/fill outcome fields. Write observed `model_panel.parquet` and one no-order `capacity_anchor_panel.parquet` row per anchor with `x_adv=0`, a `model_prediction_at_x0` source contract, and zero marginal impact; never inject the realized daily market move into model scoring.

- [ ] **Step 4: Re-run Notebook contract tests**

Expected: 00/01 contracts pass while later missing notebooks still fail.

### Task 6: Implement multi-quantile model Notebook

**Files:**
- Create: `notebooks/02_quantile_risk_model.ipynb`

- [ ] **Step 1: Add strict chronological splitting**

Resolve sorted trading dates into 70% train, 10% calibration, and 20% test. Persist resolved boundaries and assert zero date overlap. December remains evaluation-only at the model layer.

- [ ] **Step 2: Train direct LightGBM candidates**

Default operating quantiles are `[0.50, 0.80, 0.85, 0.90, 0.95, 0.99]`; raw tau map is `{0.50:0.50, 0.80:0.80, 0.85:0.85, 0.90:0.90, 0.95:0.80, 0.99:0.90}`. Train one total-risk model per candidate and one independent 0.95 impact guardrail model.

- [ ] **Step 3: Train shape-strength candidates**

Estimate train-only monotone `H_tau(x)` curves from incremental adverse risk, fit condition-only strength models, and predict `base_market_bad + strength * H_tau(x)`. Record each quantile's shape and strength identity.

- [ ] **Step 4: Apply calibration and causal floor**

Compute calibration-only residual buffers globally and by `side × ratio_bucket`. Build causal floor rows from earlier calibration folds only. Persist raw, calibrated, and final predictions without forcing cross-quantile order.

- [ ] **Step 5: Write metrics and crossing diagnostics**

Write overall and grouped coverage, pinball, calibration tables, final-month audit metrics, model bundle, feature map, and crossing rate/severity tables.

### Task 7: Implement capacity inversion Notebook

**Files:**
- Create: `notebooks/03_capacity_inversion.ipynb`

- [ ] **Step 1: Reconstruct the complete action grid**

Read the 44 positive ratios from `data_manifest.json`, prepend `0.0`, cross them with December B3-evaluable anchors, and score unobserved grid points as model predictions only.

- [ ] **Step 2: Score every candidate independently**

Generate direct and shape-strength total-risk predictions for every operating quantile, apply its calibration/floor, and apply the common 0.95 impact guardrail.

- [ ] **Step 3: Monotone each candidate along `x_adv`**

Within candidate and anchor, sort by `x_adv` and apply cumulative maximum separately to total-risk and impact-risk curves. Do not order candidates across quantiles.

- [ ] **Step 4: Invert the 10bps boundary**

Implement zero-capacity, interpolated crossing, all-grid-feasible, and invalid-curve statuses. Direct candidates use monotone linear interpolation; shape candidates include exact-H and grid diagnostic solvers.

- [ ] **Step 5: Apply fill capacity and write stable outputs**

Join `x_safe_fill`; set `x_safe_final=min(x_safe_price_risk,x_safe_fill)`; write candidate capacity, curve diagnostics, monotonicity diagnostics, and skipped outputs with stable schemas.

### Task 8: Implement optimization and report Notebook

**Files:**
- Create: `notebooks/04_optimization_and_report.ipynb`

- [ ] **Step 1: Join candidate capacity to B3 boundary**

Use `date × sym × side × quote_strategy`. Exclude missing boundary rows from risk and cash denominators and require candidate-family coverage at least 0.95.

- [ ] **Step 2: Compute formal constraints**

Use cash-weighted violation tolerance 0.05, unweighted tolerance 0.06, material-bucket tolerance 0.08, material bucket minimum 100 rows, and B3 boundary coverage minimum 0.95.

- [ ] **Step 3: Select operating point**

Among eligible candidates, maximize total price-only submitted cash. Persist selected candidate identity and boundary rows; fail closed if no candidate is eligible.

- [ ] **Step 4: Build B1/B2/B3 comparison**

Compare B1 and B3 realized outcomes separately from B2 boundary diagnostics. Never synthesize B2 same-x outcomes. Mark all December selection/evaluation results as same-sample development diagnostics.

- [ ] **Step 5: Generate report artifacts**

Write scorecards and charts for overall/bucket coverage, worst buckets, pinball, crossing, capacity, fill loss, B3 coverage, boundary violations, submitted cash, and oracle regret.

### Task 9: Add smoke integration tests and CI

**Files:**
- Create: `tests/test_smoke_pipeline.py`
- Create: `.github/workflows/notebook-smoke.yml`

- [ ] **Step 1: Write failing output-contract test**

Require each stage output and key fields, including `evaluation_scope`, `selection_evaluation_overlap`, `independent_out_of_sample_policy_evaluation`, candidate identity, monotonicity status, and B3 denominator policy.

- [ ] **Step 2: Execute notebooks in smoke mode**

Run each Notebook with `RUN_MODE=smoke` using nbconvert execute and a common repository working directory. Stop on first exception.

- [ ] **Step 3: Assert output contracts and business invariants**

Assert label agreement, date split isolation, final-month exclusion from learning, per-candidate ratio monotonicity, 10bps inversion statuses, B3 missing denominator policy, and same-sample disclosure.

- [ ] **Step 4: Add Windows/Linux-neutral CI**

Install `requirements-lock.txt` and execute only the smoke pipeline. Do not download the full Release in CI.

### Task 10: Document and verify the public handoff

**Files:**
- Create: `docs/data_dictionary.md`
- Create: `docs/methodology.md`
- Create: `docs/result_interpretation.md`
- Create: `reference/reference_run_manifest.json` after a clean run
- Modify: `README.md`

- [ ] **Step 1: Document exact reproduction commands**

Explain clone, environment creation, Jupyter execution order, Release fallback URL, output locations, and smoke/full distinction.

- [ ] **Step 2: Document limitations prominently**

State sample-level evaluation, sparse observed grids, model-scored 45-point capacity grid, December same-sample policy diagnostics, B3 future information, and no production guarantee.

- [ ] **Step 3: Run narrow tests then full smoke suite**

Run:

```powershell
python -m unittest tests.test_release_builder -v
python -m unittest tests.test_notebook_contracts -v
python -m unittest tests.test_smoke_pipeline -v
python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 4: Run public-asset scan**

Scan repository text, Notebook JSON, smoke schemas, Release ZIP members, and Git status. Confirm no internal paths, secrets, full data, old `.git`, or stale artifacts are publishable.

- [ ] **Step 5: Report unrun expensive validation honestly**

If the 2,000,000-row five-Notebook full run is not completed locally, mark `reference_run_manifest.json` as `full_run_status=not_run` and do not claim full numerical reproduction until it is executed on a suitable machine.
