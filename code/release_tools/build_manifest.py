#!/usr/bin/env python3
"""Write the public package manifest and SHA-256 checksum list."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.package_root
    manifest = root / "docs/PACKAGE_MANIFEST.csv"
    checksums = root / "docs/SHA256SUMS.txt"
    excluded = {manifest.resolve(), checksums.resolve()}
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.resolve() not in excluded
        and "__pycache__" not in path.parts
        and path.name != ".DS_Store"
    ]
    records = [(str(path.relative_to(root)), path.stat().st_size, sha256(path)) for path in sorted(files)]
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["relative_path", "size_bytes", "sha256"])
        writer.writerows(records)
    checksums.write_text(
        "".join(f"{digest}  {relative}\n" for relative, _, digest in records),
        encoding="utf-8",
    )
    print(f"files={len(records)} bytes={sum(size for _, size, _ in records)}")


if __name__ == "__main__":
    main()

