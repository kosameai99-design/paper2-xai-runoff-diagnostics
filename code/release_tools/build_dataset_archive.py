#!/usr/bin/env python3
"""Build the versioned Zenodo dataset archive from the validated public candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import zipfile
from pathlib import Path


VERSION = "1.0.0"
TITLE = "Model-consistent XAI diagnostics for runoff prediction across 135 Japanese basins: Derived results"
CREATORS = "Zhaoyu Zhang, Akiyuki Kawasaki, and Abdul Moiz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    shutil.copytree(source, destination)


def write_metadata(root: Path, doi: str) -> None:
    (root / "README.md").write_text(
        f"# {TITLE}\n\n"
        f"Version {VERSION}; DOI: {doi}.\n\n"
        "This dataset contains the trained Epoch 20 CudaLSTM checkpoint, the EA-LSTM sensitivity checkpoint, basin performance metrics, Group A-F Integrated Gradients summaries, closure/stability/response-consistency tables, rendered manuscript and Supporting Information figures, and source-data lineage manifests.\n\n"
        "The release supports verification of the reported post-processing, model-facing attribution checks, and central figures. Full model retraining requires provider-controlled source observations. Figures 1, 2, and 6 require third-party geospatial inputs that are not redistributed.\n\n"
        "The dataset does not include GRDC observations, station mappings, coordinates, basin polygons, QObs time series, third-party rasters, Paper 3 regime/event outputs, SHAP outputs, or metric-loss sensitivity outputs. See `docs/DATA_ACCESS.md` and `docs/EXCLUSIONS.md`.\n\n"
        "Derived results and released checkpoints are licensed under CC BY 4.0. Third-party source data remain under their providers' terms.\n",
        encoding="utf-8",
    )
    (root / "CITATION_DATASET.cff").write_text(
        f"""cff-version: 1.2.0
message: "If you use these derived results, please cite the Zenodo dataset."
title: "{TITLE}"
type: dataset
version: {VERSION}
date-released: 2026-07-10
doi: "{doi}"
license: CC-BY-4.0
authors:
  - family-names: Zhang
    given-names: Zhaoyu
    affiliation: The University of Tokyo
  - family-names: Kawasaki
    given-names: Akiyuki
    affiliation: The University of Tokyo
  - family-names: Moiz
    given-names: Abdul
    affiliation: University of California, San Diego
keywords:
  - hydrology
  - runoff prediction
  - explainable artificial intelligence
  - Integrated Gradients
  - CudaLSTM
  - Japan
""",
        encoding="utf-8",
    )


def write_manifest(root: Path) -> None:
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.name not in {"MANIFEST.csv", "SHA256SUMS.txt"})
    with (root / "MANIFEST.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "bytes", "sha256"])
        for path in paths:
            writer.writerow([path.relative_to(root).as_posix(), path.stat().st_size, sha256(path)])
    checksum_paths = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt")
    lines = [f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in checksum_paths]
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_archive(root: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, Path(root.name) / path.relative_to(root))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-public-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--doi", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.source_public_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output exists: {output}; pass --force to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    copy_tree(source / "derived_results", output / "derived_results")
    copy_tree(source / "figures", output / "figures")
    (output / "models").mkdir()
    shutil.copy2(source / "configs_models/model_epoch020.pt", output / "models/model_epoch020.pt")
    shutil.copy2(
        source / "configs_models/ealstm_sensitivity_model_epoch019.pt",
        output / "models/ealstm_sensitivity_model_epoch019.pt",
    )
    (output / "configs").mkdir()
    for name in ["cudalstm_epoch20.yml", "ealstm_sensitivity_epoch19.yml"]:
        shutil.copy2(source / "configs_models" / name, output / "configs" / name)

    (output / "docs").mkdir()
    for name in [
        "ARTIFACT_LINEAGE.csv",
        "DATA_ACCESS.md",
        "EXCLUSIONS.md",
        "FIGURE_SOURCE_DATA.csv",
        "IG_MODE_AUDIT.md",
        "THIRD_PARTY_NOTICES.md",
    ]:
        shutil.copy2(source / "docs" / name, output / "docs" / name)
    shutil.copy2(source / "docs/LICENSE_DERIVED_RESULTS_CC_BY_4.0.txt", output / "LICENSE")

    write_metadata(output, args.doi)
    write_manifest(output)
    build_archive(output, args.archive.resolve())
    print(output)
    print(args.archive.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
