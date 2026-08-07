# Ideal Hybrid Anchor-Strength Implementation Plan

> **For agentic workers:** Execute this plan inline with tests first. Do not copy internal source, data paths, caches, or Git history.

**Goal:** Replace row-wise `increment/H` strength labels with anchor-level curve-fit labels while retaining clean-room reproducibility and fail-closed capacity semantics.

**Architecture:** The published panel has no true `x_adv=0` outcome row, so `B_q(X)` remains the direct quantile model evaluated on an `x_adv=0` feature copy. For each train anchor, fit one `A_q` label from its observed curve using the train-only side-aware `H_q(x)`, then train the strength model on one row per anchor. Keep independent quantile heads and use the marginal impact model as the guardrail.

**Tech Stack:** Self-contained Jupyter Notebooks, LightGBM, pandas, NumPy, joblib, unittest.

---

### Task 1: Extend contracts for anchor-level strength

**Files:**
- Modify: `tests/test_notebook_contracts.py`
- Modify: `tests/test_smoke_pipeline.py`

- [ ] Add RED assertions for `anchor_strength_labels`, `fit_anchor_strength_label`, `anchor_level_strength_training`, and train-only provenance.
- [ ] Add runtime assertions for `strength_label_granularity='anchor_level_curve_fit'`, unchanged target/feature/split contracts, and fail-closed fill metadata.
- [ ] Run the narrow tests and observe expected failures before production edits.

### Task 2: Train anchor-level `A_q` labels

**Files:**
- Modify: `notebooks/02_quantile_risk_model.ipynb`

- [ ] Add `fit_anchor_strength_label(frame, h_values, base_prediction_at_x0, target_column)` using non-negative increments and one least-squares coefficient per `date × sym × side × quote_strategy` anchor.
- [ ] Build labels only from train rows; use anchor condition features from the first train row per anchor; train each `shape_strength_models[quantile]` on one row per anchor with true quantile alpha.
- [ ] Persist `anchor_strength_labels`, `strength_label_granularity`, and fit diagnostics; keep `base_prediction_at_x0`, independent heads, calibration buffer and causal history.
- [ ] Add near-zero H protection to the anchor fit and preserve `shape_h_support_status` diagnostics.

### Task 3: Consume the new bundle in capacity inversion

**Files:**
- Modify: `notebooks/03_capacity_inversion.ipynb`

- [ ] Evaluate `A_q` once per anchor condition frame and combine with direct `B_q(x=0)` and interpolated `H_q` on the action grid.
- [ ] Keep total-risk and impact guardrail curves independently monotone by `x_adv` only; preserve `absolute_final_quantile` and support-cap statuses.
- [ ] Make missing fill capacity fail closed by leaving `x_safe_final` missing rather than falling back to price-only capacity.

### Task 4: Preserve same-sample policy disclosure

**Files:**
- Modify: `notebooks/04_optimization_and_report.ipynb`
- Modify: `README.md`
- Modify: `docs/methodology.md`
- Modify: `docs/result_interpretation.md`
- Modify: `reference/reference_run_manifest.json`
- Modify: `PROVENANCE.csv`

- [ ] Add the anchor-level label granularity and no-true-x0 limitation to stage/report metadata.
- [ ] Keep December selection/evaluation marked `same-sample policy-development diagnostic`; do not claim independent policy holdout.
- [ ] Document the hybrid base-label boundary and update all provenance hashes.

### Task 5: Smoke, regression, and GitHub delivery

**Files:**
- Verify: all tracked files

- [ ] Execute notebooks `00→04` in smoke mode and inspect overall/grouped coverage, pinball, weakest buckets, monotonicity, support caps, and zero-capacity fail-closed counts.
- [ ] Run `python -m unittest`, JSON validation, `git diff --check`, and path/provenance checks.
- [ ] Commit, fast-forward `main`, push, and compare remote `main` SHA and key Notebook blobs.
