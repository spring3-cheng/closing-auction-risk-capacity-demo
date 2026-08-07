from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SOURCE_FILES = {
    "panel": {
        "source": "total_experiment_counterfactual_grid.parquet",
        "public": "closing_auction_panel_2024_sample2m_v1.parquet",
        "required": {
            "date",
            "sym",
            "side",
            "x_adv",
            "quote_strategy",
            "total_bad_move_bps",
            "impact_me_bad_bps",
            "impact_me_raw_bps",
            "total_move_bps",
            "market_move_bps",
        },
    },
    "b1": {
        "source": "b1_zero_info_10bps_rematch.parquet",
        "public": "b1_zero_info_10bps_rematch_v1.parquet",
        "required": {"date", "sym", "side", "quote_strategy", "submitted_x_adv"},
    },
    "b3": {
        "source": "b3_adaptive_oracle_boundary_resume_refinement_zero_capacity_final.parquet",
        "public": "b3_oracle_boundary_2024m12_v1.parquet",
        "required": {
            "date",
            "sym",
            "side",
            "quote_strategy",
            "b3_oracle_refined_x_safe",
            "b3_zero_capacity_fallback_flag",
        },
    },
    "fill": {
        "source": "fill_safe_capacity_min0p950.parquet",
        "public": "fill_safe_capacity_min0p950_v1.parquet",
        "required": {"date", "sym", "shock_side", "quote_strategy", "x_safe_fill"},
    },
}

TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".txt", ".yml", ".yaml", ".ipynb"}
FORBIDDEN_PATTERNS = {
    "unix_home": re.compile(r"/" + r"home/[^/\s]+/", re.IGNORECASE),
    "windows_user_home": re.compile(r"[A-Z]:\\Users\\", re.IGNORECASE),
    "legacy_workspace": re.compile(r"[A-Z]:\\predict(?:\\|\b)", re.IGNORECASE),
    "credential": re.compile(r"(?:ghp_|github_pat_|AKIA)[A-Za-z0-9_\-]{12,}"),
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_panel(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.drop(columns=["source_file"], errors="ignore").copy()
    if "_impact_sample_stratum" in result.columns:
        result = result.rename(columns={"_impact_sample_stratum": "sample_stratum"})
    return result


def sanitize_b3(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=["source_file", "source_shard_id"], errors="ignore").copy()


def scan_forbidden_paths(root: Path) -> list[dict[str, str]]:
    root = Path(root)
    findings: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern_name, pattern in FORBIDDEN_PATTERNS.items():
            match = pattern.search(text)
            if match:
                findings.append(
                    {
                        "file": path.relative_to(root).as_posix(),
                        "pattern": pattern_name,
                        "match": match.group(0),
                    }
                )
    return findings


def validate_source(path: Path, required_columns: set[str]) -> pq.ParquetFile:
    if not path.exists():
        raise FileNotFoundError(f"required source file not found: {path.name}")
    parquet_file = pq.ParquetFile(path)
    columns = set(parquet_file.schema_arrow.names)
    missing = sorted(required_columns - columns)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {missing}")
    if parquet_file.metadata.num_rows <= 0:
        raise ValueError(f"{path.name} has no rows")
    return parquet_file


def _sanitize_frame(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == "panel":
        return sanitize_panel(frame)
    if kind == "b3":
        return sanitize_b3(frame)
    return frame.copy()


def write_sanitized_parquet(source: Path, destination: Path, kind: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    parquet_file = pq.ParquetFile(source)
    writer: pq.ParquetWriter | None = None
    try:
        for batch in parquet_file.iter_batches(batch_size=100_000):
            frame = _sanitize_frame(batch.to_pandas(), kind)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError(f"no rows written for {source.name}")


def parquet_contract(path: Path) -> dict:
    parquet_file = pq.ParquetFile(path)
    columns = parquet_file.schema_arrow.names
    contract = {
        "file_name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": parquet_file.metadata.num_rows,
        "columns": len(columns),
        "column_names": list(columns),
    }
    if "date" in columns:
        dates = pd.to_datetime(pq.read_table(path, columns=["date"])["date"].to_pandas(), errors="coerce")
        contract["date_min"] = str(dates.min().date()) if dates.notna().any() else None
        contract["date_max"] = str(dates.max().date()) if dates.notna().any() else None
        contract["n_dates"] = int(dates.nunique())
    return contract


def _stable_smoke_panel(panel_path: Path, rows_per_stratum: int) -> pd.DataFrame:
    retained: pd.DataFrame | None = None
    parquet_file = pq.ParquetFile(panel_path)
    key_columns = ["date", "sym", "side", "x_adv"]
    for batch in parquet_file.iter_batches(batch_size=50_000):
        frame = batch.to_pandas()
        dates = pd.to_datetime(frame["date"], errors="coerce")
        frame["_smoke_month"] = dates.dt.to_period("M").astype(str)
        frame["_smoke_hash"] = pd.util.hash_pandas_object(frame[key_columns], index=False).astype("uint64")
        candidates = frame.sort_values("_smoke_hash").groupby(
            ["_smoke_month", "side", "x_adv"], dropna=False, sort=False
        ).head(rows_per_stratum)
        retained = candidates if retained is None else pd.concat([retained, candidates], ignore_index=True)
        retained = retained.sort_values("_smoke_hash").groupby(
            ["_smoke_month", "side", "x_adv"], dropna=False, sort=False
        ).head(rows_per_stratum)
    if retained is None or retained.empty:
        raise ValueError("unable to build smoke panel")
    retained = retained.drop(columns=["_smoke_month", "_smoke_hash"]).reset_index(drop=True)
    retained["evaluation_scope"] = "smoke_contract_only"
    return retained


def _normalize_key_columns(frame: pd.DataFrame, side_column: str = "side") -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["sym"] = result["sym"].astype(str)
    result[side_column] = result[side_column].astype(str).str.lower().str.strip()
    result["quote_strategy"] = result["quote_strategy"].astype(str).str.lower().str.strip()
    return result


def build_smoke_inputs(staged_files: dict[str, Path], smoke_dir: Path, rows_per_stratum: int) -> dict[str, Path]:
    smoke_dir.mkdir(parents=True, exist_ok=True)
    panel = _stable_smoke_panel(staged_files["panel"], rows_per_stratum=rows_per_stratum)
    panel = _normalize_key_columns(panel)
    key_columns = ["date", "sym", "side"]
    anchor_keys = panel[key_columns].drop_duplicates()

    outputs: dict[str, Path] = {}
    panel_out = smoke_dir / SOURCE_FILES["panel"]["public"]
    panel.to_parquet(panel_out, index=False, compression="zstd")
    outputs["panel"] = panel_out

    for kind in ["b1", "b3"]:
        frame = pd.read_parquet(staged_files[kind])
        frame = _normalize_key_columns(frame)
        frame = frame.merge(anchor_keys.assign(_keep=True), on=key_columns, how="inner")
        frame = frame.drop(columns=["_keep"])
        output = smoke_dir / SOURCE_FILES[kind]["public"]
        frame.to_parquet(output, index=False, compression="zstd")
        outputs[kind] = output

    fill = pd.read_parquet(staged_files["fill"])
    fill = fill.rename(columns={"shock_side": "side"})
    fill = _normalize_key_columns(fill)
    fill = fill.merge(anchor_keys.assign(_keep=True), on=key_columns, how="inner").drop(columns=["_keep"])
    fill = fill.rename(columns={"side": "shock_side"})
    fill_out = smoke_dir / SOURCE_FILES["fill"]["public"]
    fill.to_parquet(fill_out, index=False, compression="zstd")
    outputs["fill"] = fill_out
    return outputs


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_deterministic_zip(destination: Path, files: list[Path]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w") as archive:
        for path in sorted((Path(item) for item in files), key=lambda item: item.name):
            info = zipfile.ZipInfo(path.name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            with path.open("rb") as source, archive.open(info, "w") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)


def build_release_bundle(source_dir: Path, repo_root: Path, rows_per_stratum: int = 8) -> dict:
    source_dir = Path(source_dir).resolve()
    repo_root = Path(repo_root).resolve()
    stage_dir = repo_root / "release" / "data-v1.0.0"
    smoke_dir = repo_root / "data" / "smoke"
    stage_dir.mkdir(parents=True, exist_ok=True)

    staged_files: dict[str, Path] = {}
    source_hashes: dict[str, str] = {}
    for kind, spec in SOURCE_FILES.items():
        source = source_dir / spec["source"]
        validate_source(source, set(spec["required"]))
        destination = stage_dir / spec["public"]
        write_sanitized_parquet(source, destination, kind)
        staged_files[kind] = destination
        source_hashes[kind] = sha256_file(source)

    panel_grid = pd.to_numeric(
        pq.read_table(staged_files["panel"], columns=["x_adv"])["x_adv"].to_pandas(), errors="coerce"
    ).dropna()
    grid_values = sorted(panel_grid.astype(float).drop_duplicates().tolist())
    files_contract = {kind: parquet_contract(path) for kind, path in staged_files.items()}
    internal_manifest = {
        "data_version": "v1.0.0",
        "release_tag": "data-v1.0.0",
        "asset_name": "closing_auction_demo_data_v1.zip",
        "sample_mode": "stratified_cap",
        "sample_cap_rows": 2_000_000,
        "evaluation_scope": "sample_level_evaluation",
        "cash_adv_grid": grid_values,
        "cash_adv_grid_count": len(grid_values),
        "authorization_scope": "public redistribution for learning, job presentation, and independent demonstration",
        "source_sha256": source_hashes,
        "files": files_contract,
    }
    internal_manifest_path = stage_dir / "data_manifest.json"
    _write_json(internal_manifest_path, internal_manifest)

    smoke_files = build_smoke_inputs(staged_files, smoke_dir, rows_per_stratum=rows_per_stratum)
    smoke_contract = {kind: parquet_contract(path) for kind, path in smoke_files.items()}

    zip_path = repo_root / "release" / "closing_auction_demo_data_v1.zip"
    write_deterministic_zip(zip_path, [*staged_files.values(), internal_manifest_path])

    public_manifest = {
        **internal_manifest,
        "release_asset": {
            "file_name": zip_path.name,
            "bytes": zip_path.stat().st_size,
            "sha256": sha256_file(zip_path),
        },
        "smoke_files": smoke_contract,
    }
    _write_json(repo_root / "data" / "data_manifest.json", public_manifest)
    return public_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sanitized Release and smoke data")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--smoke-rows-per-stratum", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_rows_per_stratum < 1:
        raise ValueError("--smoke-rows-per-stratum must be >= 1")
    manifest = build_release_bundle(
        source_dir=args.source_dir,
        repo_root=args.repo_root,
        rows_per_stratum=args.smoke_rows_per_stratum,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "release_asset": manifest["release_asset"],
                "cash_adv_grid_count": manifest["cash_adv_grid_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
