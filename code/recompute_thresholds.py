"""Recompute the counterfactual threshold audit from minimal raw CSV tables."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "calorimeter"
THRESHOLDS = (0.0, 175.0, 200.0, 250.0, 300.0)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit(path: Path) -> dict[float, dict[str, float]]:
    counts = {cut: 0 for cut in THRESHOLDS}
    lost_aged_energy = {cut: 0.0 for cut in THRESHOLDS}
    affected = {cut: set() for cut in THRESHOLDS}
    events: set[str] = set()
    total_hits = 0
    total_aged_energy = 0.0
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            event = row["event"]
            events.add(event)
            aged = float(row["energy_raw_reference_mev"]) * float(row["aging_factor"])
            if not math.isfinite(aged):
                raise ValueError(f"Nonfinite energy in {path.name}")
            total_hits += 1
            total_aged_energy += aged
            for cut in THRESHOLDS:
                if aged < cut:
                    counts[cut] += 1
                    lost_aged_energy[cut] += aged
                    affected[cut].add(event)
    return {
        cut: {
            "hits": total_hits,
            "censored_hits": counts[cut],
            "aged_energy": total_aged_energy,
            "censored_aged_energy": lost_aged_energy[cut],
            "affected_fraction": len(affected[cut]) / len(events),
        }
        for cut in THRESHOLDS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, choices=(50000, 90000, 150000, 250000),
                        help="Audit one archive size; default is all four")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA,
                        help="Directory containing the extracted raw tables and manifest.csv")
    args = parser.parse_args()
    manifest_file = args.data_dir / "manifest.csv"
    if not manifest_file.is_file():
        raise SystemExit("Raw calorimeter tables are not in this review supplement. "
                         "Provide an extracted archive with --data-dir PATH.")
    manifest = read_csv(manifest_file)
    saved = read_csv(ROOT / "results" / "exp38_event_level" / "threshold_summary.csv")
    sizes = (args.size,) if args.size else (50000, 90000, 150000, 250000)
    for size in sizes:
        records = [r for r in manifest if int(r["archive_events_per_state"]) == size
                   and r["domain"] == "reference"]
        if len(records) != 4:
            raise AssertionError(f"Expected four reference archives at size {size}")
        per_pair = [audit(args.data_dir / r["file"]) for r in records]
        for cut in THRESHOLDS:
            group = [pair[cut] for pair in per_pair]
            hit_fraction = sum(p["censored_hits"] for p in group) / sum(p["hits"] for p in group)
            energy_fraction = sum(p["censored_aged_energy"] for p in group) / sum(p["aged_energy"] for p in group)
            event_fraction = sum(p["affected_fraction"] for p in group) / len(group)
            expected = [r for r in saved if int(r["events_per_state"]) == size
                        and float(r["threshold_mev"]) == cut]
            if len(expected) != 1:
                raise AssertionError(f"Missing saved summary for {size}, {cut}")
            for field, value in (("censored_hit_fraction", hit_fraction),
                                 ("censored_aged_energy_fraction", energy_fraction),
                                 ("mean_affected_event_fraction", event_fraction)):
                if not math.isclose(value, float(expected[0][field]), abs_tol=1e-10):
                    raise AssertionError(f"{size} events, {cut:g} MeV, {field}: {value} != {expected[0][field]}")
            print(f"PASS {size:6d} events/state, {cut:3g} MeV: "
                  f"hits {100*hit_fraction:.3f}%, aged energy {100*energy_fraction:.3f}%, "
                  f"affected events {100*event_fraction:.3f}%")


if __name__ == "__main__":
    main()
