"""Export the paper's calorimeter event archives as anonymous, minimal tables.

This script is for rebuilding the distributed data files from the original
simulation archives. The distributed files are already present in data/.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import zipfile
from pathlib import Path


FIELDS = (
    "cellid",
    "energy_raw_reference_mev",
    "aging_factor",
    "event",
    "x",
    "y",
    "z",
)
SOURCE_FIELDS = {
    "cellid": "cellid",
    "energy_raw_reference_mev": "E_cal_org",
    "aging_factor": "aging_factor",
    "event": "event",
    "x": "x",
    "y": "y",
    "z": "z",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_one(source: Path, destination: Path) -> tuple[int, int]:
    """Stream one prepared archive, retaining only raw simulation fields."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    event_ids: set[str] = set()
    rows_written = 0
    with zipfile.ZipFile(source) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1:
            raise ValueError(f"Expected one table in {source}")
        with archive.open(members[0]) as source_binary, destination.open("wb") as target_binary:
            with io.TextIOWrapper(source_binary, encoding="utf-8", newline="") as source_text:
                reader = csv.DictReader(source_text)
                missing = set(SOURCE_FIELDS.values()) - set(reader.fieldnames or ())
                if missing:
                    raise ValueError(f"Missing source fields in {source}: {sorted(missing)}")
                with gzip.GzipFile(fileobj=target_binary, mode="wb", filename="", mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as target_text:
                        writer = csv.DictWriter(target_text, fieldnames=FIELDS, lineterminator="\n")
                        writer.writeheader()
                        for row in reader:
                            writer.writerow({out: row[source_name] for out, source_name in SOURCE_FIELDS.items()})
                            event_ids.add(row["event"])
                            rows_written += 1
    return rows_written, len(event_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Original exp38 archive directory containing analysis/ and run folders")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "calorimeter")
    args = parser.parse_args()

    index = args.source_root / "analysis" / "canonical_prepared_runs.csv"
    with index.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != 16:
        raise ValueError(f"Expected 16 canonical archive conditions, found {len(records)}")

    manifest: list[dict[str, object]] = []
    for record in records:
        size = int(record["events_per_state"])
        pair = record["data_seed_pair"]
        run = args.source_root / record["run_name"] / "logs"
        for domain, original in (("reference", "train_prepared_new.zip"),
                                 ("changed_source", "train_prepared_old.zip")):
            filename = f"size_{size:06d}_pair_{pair}_{domain}.csv.gz"
            destination = args.output_dir / filename
            rows, events = export_one(run / original, destination)
            entry = {
                "archive_events_per_state": size,
                "data_seed_pair": pair,
                "domain": domain,
                "file": filename,
                "rows": rows,
                "events": events,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
            manifest.append(entry)
            print(f"{filename}: {rows} rows, {events} events", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    schema = {
        "format": "gzip-compressed UTF-8 CSV",
        "columns": list(FIELDS),
        "energy_unit": "MeV",
        "reference_domain": "uncensored nominal hits",
        "changed_source_domain": "nominal energies of aged-side hits retained after the 200 MeV aged-energy threshold",
        "raw_aged_energy": "energy_raw_reference_mev * aging_factor",
        "paired_events": False,
        "factor_truth_used_in_fitting": False,
    }
    (args.output_dir / "schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
