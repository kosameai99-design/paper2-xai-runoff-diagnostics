#!/usr/bin/env python3
"""Validate the runoff-XAI software repository and optional results package."""

from __future__ import annotations

import argparse
import compileall
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


CODE_REQUIRED = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "environment.yml",
    "configs/cudalstm_epoch20.yml",
    "code/release_tools/run_integrated_gradients.py",
    "code/release_tools/reproduce_figure3.py",
    "code/release_tools/reproduce_figures_4_5.py",
    "code/figure_builders/build_manuscript_geospatial_figures.py",
    "code/figure_builders/build_si_figures.py",
]

DATA_REQUIRED = [
    "README.md",
    "LICENSE",
    "models/model_epoch020.pt",
    "models/ealstm_sensitivity_model_epoch019.pt",
    "derived_results/performance/test_metrics_epoch020.csv",
    "derived_results/attribution/groupA_global/global_feature_importance_aggregated.csv",
    "derived_results/validation/response_consistency/ig_vs_ablation_ranks.csv",
    "derived_results/validation/ig_determinism_summary.json",
    "figures/main/Figure1.png",
    "figures/main/Figure6.png",
    "docs/ARTIFACT_LINEAGE.csv",
    "docs/FIGURE_SOURCE_DATA.csv",
]

TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".json", ".csv", ".cff", ".txt"}
ABSOLUTE_PATTERNS = [
    re.compile("/" + "Users/"),
    re.compile(r"\b[A-Za-z]:\\(?:Python|Users|data|runs|NH)"),
]
FORBIDDEN_NAMES = ["regime", "shap_outputs", "metric_loss_sensitivity"]
PLACEHOLDERS = [
    "TO_" + "BE_ADDED",
    "TO_" + "BE_RESERVED",
    "[DATA" + " DOI]",
    "[SOFTWARE" + " DOI]",
]


def require_files(root: Path, required: list[str], errors: list[str], label: str) -> None:
    for relative in required:
        if not (root / relative).is_file():
            errors.append(f"{label} missing: {relative}")


def scan_text(root: Path, errors: list[str], allow_placeholders: bool) -> None:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern.search(text) for pattern in ABSOLUTE_PATTERNS):
            errors.append(f"absolute path in public text file: {path.relative_to(root)}")
        if not allow_placeholders and any(token in text for token in PLACEHOLDERS):
            errors.append(f"unresolved release placeholder: {path.relative_to(root)}")
    for path in root.rglob("*"):
        lower = str(path.relative_to(root)).lower()
        if any(name in lower for name in FORBIDDEN_NAMES):
            errors.append(f"forbidden follow-on regime/SHAP artifact: {path.relative_to(root)}")


def verify_code(root: Path, errors: list[str], allow_placeholders: bool) -> None:
    require_files(root, CODE_REQUIRED, errors, "code")
    if any(root.rglob("*.pt")):
        errors.append("model checkpoint present in GitHub software tree")
    if not compileall.compile_dir(root / "code", quiet=1):
        errors.append("Python source compilation failed")
    scan_text(root, errors, allow_placeholders)


def verify_dataset(root: Path, errors: list[str], allow_placeholders: bool) -> None:
    require_files(root, DATA_REQUIRED, errors, "dataset")
    metrics_path = root / "derived_results/performance/test_metrics_epoch020.csv"
    if metrics_path.is_file():
        metrics = pd.read_csv(metrics_path)
        if len(metrics) != 135:
            errors.append(f"expected 135 performance rows, found {len(metrics)}")
        if abs(metrics["NSE"].median() - 0.836881846) > 1e-8:
            errors.append("NSE median mismatch")
        if abs(metrics["KGE"].median() - 0.825047338) > 1e-8:
            errors.append("KGE median mismatch")

    daily_root = root / "derived_results/attribution/groupD_E_F_daily"
    basin_dirs = sorted(daily_root.glob("basin_*")) if daily_root.is_dir() else []
    attribution_csvs = [path for directory in basin_dirs for path in directory.glob("*.csv")]
    if len(basin_dirs) != 135 or len(attribution_csvs) != 405:
        errors.append(
            "expected 135 basin directories and 405 D/E/F CSVs, "
            f"found {len(basin_dirs)} and {len(attribution_csvs)}"
        )

    audit_path = root / "derived_results/validation/ig_determinism_summary.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit.get("release_ready"):
            errors.append("deterministic IG release gate did not pass")
    scan_text(root, errors, allow_placeholders)


def verify_sha256(root: Path, manifest: Path, errors: list[str]) -> int:
    checked = 0
    if not manifest.is_file():
        return checked
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        path = root / relative
        if not path.is_file():
            errors.append(f"checksum target missing: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            errors.append(f"checksum mismatch: {relative}")
        checked += 1
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument("--allow-placeholders", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    verify_code(args.repo_root.resolve(), errors, args.allow_placeholders)
    checked = 0
    if args.data_root is not None:
        data_root = args.data_root.resolve()
        verify_dataset(data_root, errors, args.allow_placeholders)
        checked = verify_sha256(data_root, data_root / "SHA256SUMS.txt", errors)
    elif not args.code_only:
        errors.append("provide --data-root for full verification or use --code-only")

    result = {
        "release_ready": not errors,
        "mode": "code-only" if args.code_only and args.data_root is None else "code+dataset",
        "dataset_sha256_entries_checked": checked,
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
