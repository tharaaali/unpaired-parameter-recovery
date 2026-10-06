"""Record relative paths, sizes, and SHA-256 checksums for distributed files."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "file_manifest.csv"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    paths = sorted(path for path in ROOT.rglob("*") if path.is_file() and path != DEST)
    with DEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("relative_path", "bytes", "sha256"))
        for path in paths:
            writer.writerow((path.relative_to(ROOT).as_posix(), path.stat().st_size, digest(path)))
    print(f"Manifested {len(paths)} files")


if __name__ == "__main__":
    main()
