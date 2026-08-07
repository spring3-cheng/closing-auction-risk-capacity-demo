import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def read_json(name: str) -> dict:
    return json.loads((OUTPUTS / name).read_text(encoding="utf-8"))


class SmokePipelineTests(unittest.TestCase):
    def test_data_and_label_contracts(self):
        active = read_json("00_active_data.json")
        stage = read_json("01_stage_contract.json")
        audit = pd.read_csv(OUTPUTS / "01_label_audit.csv")
        features = pd.read_csv(OUTPUTS / "01_feature_contract.csv")

        self.assertEqual(active["run_mode"], "smoke")
        self.assertEqual(active["evaluation_scope"], "smoke_contract_only")
        self.assertEqual(stage["run_mode"], "smoke")
        self.assertEqual(float(audit.loc[0, "max_total_label_gap"]), 0.0)
        self.assertEqual(float(audit.loc[0, "max_impact_label_gap"]), 0.0)
        self.assertTrue(features["visible_at"].eq("14:59:59").all())
        forbidden_features = {
            "market_move_bps", "abs_market_move_bps", "p_close_hist", "p_close_cf",
            "total_move_bps", "total_bad_move_bps", "impact_me_raw_bps", "impact_me_bad_bps",
            "fill_ratio",
        }
        self.assertTrue(forbidden_features.isdisjoint(set(features["feature"])))
        self.assertEqual(stage["no_order_anchor_source"], "model_prediction_at_x0")

    def test_model_time_and_quantile_contracts(self):
        stage = read_json("02_stage_contract.json")
        split = pd.read_csv(OUTPUTS / "02_time_split.csv", parse_dates=[
            "train_min_date", "train_max_date", "calibration_min_date",
            "calibration_max_date", "test_min_date", "test_max_date",
        ])
        metrics = pd.read_csv(OUTPUTS / "02_model_metrics.csv")
        grouped = pd.read_csv(OUTPUTS / "02_grouped_coverage.csv")
        crossing = pd.read_csv(OUTPUTS / "02_crossing_diagnostics.csv")

        self.assertFalse(stage["cross_quantile_order_enforced"])
        self.assertTrue(stage["final_month_learning_excluded"])
        self.assertEqual(stage["shape_h_mode"], "quantile_specific_train_curve")
        self.assertTrue(stage["shape_h_side_aware"])
        self.assertEqual(stage["strength_label_granularity"], "anchor_level_curve_fit")
        self.assertEqual(
            stage["base_label_source"],
            "direct_quantile_model_at_x0_no_observed_outcome",
        )
        self.assertEqual(stage["quantile_registry"], [0.5, 0.8, 0.85, 0.9, 0.95, 0.99])
        row = split.iloc[0]
        self.assertLess(row["train_max_date"], row["calibration_min_date"])
        self.assertLess(row["calibration_max_date"], row["test_min_date"])
        self.assertIn("final_month", set(metrics["split"]))
        self.assertTrue({"coverage", "pinball_loss"}.issubset(metrics.columns))
        self.assertTrue({"side", "ratio_bucket", "coverage"}.issubset(grouped.columns))
        self.assertGreater(len(crossing), 0)
        anchor_labels = pd.read_csv(OUTPUTS / "02_anchor_strength_labels.csv")
        self.assertGreater(len(anchor_labels), 0)
        self.assertTrue(
            {"quantile", "anchor_id", "A_label", "n_curve_points"}.issubset(anchor_labels.columns)
        )
        self.assertTrue(anchor_labels["n_curve_points"].ge(1).all())

    def test_capacity_contracts(self):
        stage = read_json("03_stage_contract.json")
        candidates = pd.read_parquet(OUTPUTS / "03_capacity_candidates.parquet")
        monotonicity = pd.read_csv(OUTPUTS / "03_monotonicity_diagnostics.csv")

        self.assertEqual(stage["action_grid_count"], 45)
        self.assertFalse(stage["cross_quantile_order_enforced"])
        self.assertEqual(stage["ratio_curve_monotone_postprocess"], "cummax")
        self.assertEqual(stage["capacity_risk_measure"], "absolute_final_quantile")
        self.assertEqual(stage["continuous_prediction_source"], "quantile_specific_H_piecewise_linear")
        self.assertTrue({"h_support_status", "support_cap_status"}.issubset(candidates.columns))
        self.assertTrue(monotonicity["postprocess_total_drop_count"].eq(0).all())
        self.assertTrue(monotonicity["postprocess_impact_drop_count"].eq(0).all())
        formal = candidates[candidates["selection_eligible_solver"]]
        self.assertGreater(len(formal), 0)
        self.assertTrue(formal["boundary_solver"].eq("monotone_linear_interp").all())
        finite = formal.dropna(subset=["x_safe_price_risk", "x_safe_fill", "x_safe_final"])
        self.assertGreater(len(finite), 0)
        self.assertTrue((finite["x_safe_final"] <= finite["x_safe_price_risk"] + 1e-12).all())
        self.assertTrue((finite["x_safe_final"] <= finite["x_safe_fill"] + 1e-12).all())
        self.assertTrue(formal.loc[formal["x_safe_fill"].isna(), "x_safe_final"].isna().all())

    def test_policy_report_discloses_same_sample_diagnostic(self):
        report = read_json("04_report_summary.json")
        scorecard = pd.read_csv(OUTPUTS / "04_candidate_scorecard.csv")

        self.assertEqual(report["evaluation_scope"], "same-sample policy-development diagnostic")
        self.assertTrue(report["selection_evaluation_overlap"])
        self.assertFalse(report["independent_out_of_sample_policy_evaluation"])
        self.assertGreater(len(scorecard), 0)
        self.assertTrue(scorecard["selection_evaluation_overlap"].all())
        self.assertFalse(scorecard["independent_out_of_sample_policy_evaluation"].any())
        eligible = scorecard[scorecard["selection_eligible"]]
        self.assertTrue((eligible["submitted_cash_price_only"] > 0).all())
        self.assertTrue((eligible["n_positive_price_capacity"] > 0).all())


if __name__ == "__main__":
    unittest.main()
