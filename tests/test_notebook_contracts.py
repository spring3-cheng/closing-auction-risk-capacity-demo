import json
import re
import unittest
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOK_NAMES = [
    "00_get_data.ipynb",
    "01_build_panel_and_labels.ipynb",
    "02_quantile_risk_model.ipynb",
    "03_capacity_inversion.ipynb",
    "04_optimization_and_report.ipynb",
]


def notebook_source(name: str) -> str:
    notebook = nbformat.read(NOTEBOOK_DIR / name, as_version=4)
    return "\n".join(cell.source for cell in notebook.cells)


class NotebookContractTests(unittest.TestCase):
    def test_all_code_cells_compile(self):
        for name in NOTEBOOK_NAMES:
            notebook = nbformat.read(NOTEBOOK_DIR / name, as_version=4)
            for cell_index, cell in enumerate(notebook.cells):
                if cell.cell_type == "code":
                    compile(cell.source, f"{name}:cell-{cell_index}", "exec")

    def test_five_ordered_notebooks_exist_and_are_valid(self):
        actual = sorted(path.name for path in NOTEBOOK_DIR.glob("*.ipynb"))
        self.assertEqual(actual, NOTEBOOK_NAMES)
        for name in NOTEBOOK_NAMES:
            raw_notebook = json.loads((NOTEBOOK_DIR / name).read_text(encoding="utf-8"))
            self.assertTrue(all(cell.get("id") for cell in raw_notebook["cells"]), msg=f"{name} has a cell without id")
            notebook = nbformat.read(NOTEBOOK_DIR / name, as_version=4)
            self.assertGreaterEqual(len(notebook.cells), 3)

    def test_notebooks_are_self_contained_and_path_clean(self):
        forbidden_imports = ["from scripts", "import scripts", "sys.path.append", "sys.path.insert"]
        forbidden_paths = [
            re.compile(r"/" + r"home/[^/\s]+/", re.IGNORECASE),
            re.compile(r"[A-Z]:\\", re.IGNORECASE),
        ]
        for name in NOTEBOOK_NAMES:
            source = notebook_source(name)
            for token in forbidden_imports:
                self.assertNotIn(token, source, msg=f"{name} contains {token}")
            for pattern in forbidden_paths:
                self.assertIsNone(pattern.search(source), msg=f"{name} contains an absolute path")

    def test_data_and_label_notebooks_declare_stable_contracts(self):
        data_source = notebook_source("00_get_data.ipynb")
        label_source = notebook_source("01_build_panel_and_labels.ipynb")
        for token in [
            "RUN_MODE", "sha256_file", "00_active_data.json", ".part",
            "GITHUB_TOKEN", "GH_TOKEN", "api.github.com/repos",
            "application/octet-stream", "download_with_headers", "release_parts",
            "Release part SHA256 mismatch", "github_release_asset_api_urls",
        ]:
            self.assertIn(token, data_source)
        for token in [
            "total_bad_move_bps",
            "impact_me_bad_bps",
            "capacity_anchor_panel.parquet",
            "01_feature_contract.csv",
            "14:59:59",
            "LEAKAGE_COLUMNS",
            "model_prediction_at_x0",
        ]:
            self.assertIn(token, label_source)
        self.assertNotIn("anchors['anchor_total_bad_bps'] = np.maximum", label_source)

    def test_model_notebook_uses_independent_quantile_candidates(self):
        source = notebook_source("02_quantile_risk_model.ipynb")
        for token in [
            "QUANTILE_REGISTRY",
            "RAW_TAU_MAP",
            "shape_strength",
            "crossing_rate",
            "side × ratio_bucket",
            "final_month",
            "shape_total_train",
        ]:
            self.assertIn(token, source)
        self.assertNotIn("market_bad_prediction + strength * h_values", source)
        self.assertNotIn("enforce_quantile_order", source)
        self.assertNotIn("enforce_quantile_stack_order", source)

    def test_capacity_notebook_orders_only_ratio_curves(self):
        source = notebook_source("03_capacity_inversion.ipynb")
        for token in [
            "ACTION_GRID",
            "45-point",
            "cummax",
            "x_safe_price_risk",
            "x_safe_final",
            "impact_guardrail_quantile",
            "exact_h_piecewise_linear_diagnostic",
        ]:
            self.assertIn(token, source)
        self.assertNotIn("q99_final >= q95_final", source)

    def test_optimization_notebook_discloses_same_sample_scope(self):
        source = notebook_source("04_optimization_and_report.ipynb")
        for token in [
            "same-sample policy-development diagnostic",
            "selection_evaluation_overlap",
            "independent_out_of_sample_policy_evaluation",
            "CASH_WEIGHTED_VIOLATION_TOLERANCE",
            "MIN_B3_BOUNDARY_COVERAGE_RATE",
            "excluded_from_risk_and_cash_denominators",
            "positive_capacity_required",
        ]:
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
