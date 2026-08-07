import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.build_release_data import (
    build_smoke_inputs,
    sanitize_b3,
    sanitize_panel,
    scan_forbidden_paths,
    sha256_file,
    write_deterministic_zip,
)


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".csv", ".ipynb", ".json", ".md", ".py", ".txt", ".yml", ".yaml"}


def provenance_sha256_variants(path: Path) -> set[str]:
    payload = path.read_bytes()
    variants = {hashlib.sha256(payload).hexdigest()}
    if path.suffix.lower() in TEXT_SUFFIXES:
        normalized_lf = payload.replace(b"\r\n", b"\n")
        normalized_crlf = normalized_lf.replace(b"\n", b"\r\n")
        variants.add(hashlib.sha256(normalized_lf).hexdigest())
        variants.add(hashlib.sha256(normalized_crlf).hexdigest())
    return variants


class ReleaseBuilderTests(unittest.TestCase):
    def test_provenance_hashes_match_available_public_files(self):
        ledger = pd.read_csv(ROOT / "PROVENANCE.csv").fillna("")
        for record in ledger.to_dict("records"):
            relative_path = Path(record["file_path"])
            path = ROOT / relative_path
            if relative_path.parts[0] == "release" and not path.exists():
                continue
            self.assertTrue(path.is_file(), msg=f"missing provenance file: {relative_path.as_posix()}")
            self.assertIn(
                record["sha256"],
                provenance_sha256_variants(path),
                msg=f"stale provenance hash: {relative_path.as_posix()}",
            )

    def test_public_repository_has_no_forbidden_text_paths(self):
        self.assertEqual(scan_forbidden_paths(ROOT), [])

    def test_sanitize_panel_removes_internal_trace_columns(self):
        frame = pd.DataFrame(
            {
                "source_file": ["impact_chunk.parquet"],
                "_impact_sample_stratum": ["2024-12|buy|0.1-0.25%"],
                "x_adv": [0.001],
            }
        )

        result = sanitize_panel(frame)

        self.assertNotIn("source_file", result.columns)
        self.assertIn("sample_stratum", result.columns)
        self.assertEqual(result.loc[0, "sample_stratum"], "2024-12|buy|0.1-0.25%")

    def test_sanitize_b3_removes_trace_columns(self):
        frame = pd.DataFrame(
            {
                "source_file": ["part-01.parquet"],
                "source_shard_id": [1],
                "b3_oracle_refined_x_safe": [0.01],
            }
        )

        result = sanitize_b3(frame)

        self.assertNotIn("source_file", result.columns)
        self.assertNotIn("source_shard_id", result.columns)
        self.assertEqual(result.loc[0, "b3_oracle_refined_x_safe"], 0.01)

    def test_sha256_file_matches_hashlib(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.bin"
            path.write_bytes(b"closing-auction-demo")

            actual = sha256_file(path)

        expected = hashlib.sha256(b"closing-auction-demo").hexdigest()
        self.assertEqual(actual, expected)

    def test_scan_forbidden_paths_reports_absolute_internal_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "safe.md").write_text("relative/data/panel.parquet", encoding="utf-8")
            forbidden_path = "/" + "home/example/project/data"
            (root / "unsafe.md").write_text(f"internal server path: {forbidden_path}", encoding="utf-8")

            findings = scan_forbidden_paths(root)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["file"], "unsafe.md")

    def test_smoke_auxiliary_inputs_join_on_market_key_not_strategy_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = root / "staged"
            smoke = root / "smoke"
            staged.mkdir()
            common = {"date": ["2024-12-02"], "sym": ["000001.SZ"]}
            panel = pd.DataFrame(
                {
                    **common,
                    "side": ["buy"],
                    "quote_strategy": ["relative"],
                    "x_adv": [0.001],
                    "evaluation_scope": ["sample_level_evaluation"],
                }
            )
            b1 = pd.DataFrame(
                {**common, "side": ["buy"], "quote_strategy": ["10bps_protection_price"]}
            )
            b3 = pd.DataFrame(
                {**common, "side": ["buy"], "quote_strategy": ["relative"]}
            )
            fill = pd.DataFrame(
                {**common, "shock_side": ["buy"], "quote_strategy": ["cross_book"]}
            )
            staged_files = {}
            for kind, frame in {"panel": panel, "b1": b1, "b3": b3, "fill": fill}.items():
                path = staged / f"{kind}.parquet"
                frame.to_parquet(path, index=False)
                staged_files[kind] = path

            outputs = build_smoke_inputs(staged_files, smoke, rows_per_stratum=1)

            self.assertEqual(len(pd.read_parquet(outputs["b1"])), 1)
            self.assertEqual(len(pd.read_parquet(outputs["b3"])), 1)
            self.assertEqual(len(pd.read_parquet(outputs["fill"])), 1)

    def test_release_zip_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a.txt"
            second = root / "b.txt"
            first.write_text("alpha", encoding="utf-8")
            second.write_text("beta", encoding="utf-8")
            zip_one = root / "one.zip"
            zip_two = root / "two.zip"

            write_deterministic_zip(zip_one, [second, first])
            write_deterministic_zip(zip_two, [first, second])

            self.assertEqual(sha256_file(zip_one), sha256_file(zip_two))


if __name__ == "__main__":
    unittest.main()
