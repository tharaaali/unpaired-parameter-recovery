"""Quantify counterfactual aged-hit censoring in exp38 canonical archives.

The reference-side prepared CSV retains uncensored raw hit energies and the
fixed cell response factor. Applying the original 200 MeV rule to these
independent reference events estimates the changed-side hit and energy loss.
It is a measurement-mechanism audit, not an observed count of removed old rows.
"""
from __future__ import annotations

import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/exp38_fixed_cell_aging_reproducible"
OUT = ROOT / "results/exp38_event_level"
THRESHOLDS = (0.0, 175.0, 200.0, 250.0, 300.0)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_one(record: dict) -> list[dict]:
    archive_path = BASE / record["run_name"] / "logs/train_prepared_new.zip"
    counts = {threshold:0 for threshold in THRESHOLDS}
    aged_loss = {threshold:0.0 for threshold in THRESHOLDS}
    reference_loss = {threshold:0.0 for threshold in THRESHOLDS}
    affected_events = {threshold:set() for threshold in THRESHOLDS}
    events = set()
    raw_hits = 0
    aged_energy_total = 0.0
    reference_energy_total = 0.0
    mask_disagreements = 0
    with zipfile.ZipFile(archive_path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if len(names) != 1:
            raise ValueError(f"Expected one CSV in {archive_path}")
        with archive.open(names[0]) as binary:
            rows = csv.DictReader((line.decode("utf-8") for line in binary))
            for row in rows:
                event = int(row["event"])
                events.add(event)
                original_energy = float(row["E_cal_org"])
                factor = float(row["aging_factor"])
                aged_energy = original_energy * factor
                if not np.isfinite(aged_energy) or not np.isfinite(original_energy):
                    raise ValueError(f"Nonfinite hit in {archive_path}")
                raw_hits += 1
                aged_energy_total += aged_energy
                reference_energy_total += original_energy
                predicted_censored = aged_energy < 200.0
                archived_censored = not row["E_aged_cal"] or row["E_aged_cal"].lower() == "nan"
                mask_disagreements += predicted_censored != archived_censored
                for threshold in THRESHOLDS:
                    if aged_energy < threshold:
                        counts[threshold] += 1
                        aged_loss[threshold] += aged_energy
                        reference_loss[threshold] += original_energy
                        affected_events[threshold].add(event)
    if mask_disagreements:
        raise AssertionError(f"200 MeV mask mismatch in {archive_path}: {mask_disagreements}")
    output = []
    for threshold in THRESHOLDS:
        output.append(dict(events_per_state=int(record["events_per_state"]),
                           data_seed_pair=record["data_seed_pair"],
                           run_name=record["run_name"], threshold_mev=threshold,
                           events=len(events), raw_reference_hits=raw_hits,
                           censored_hits=counts[threshold],
                           censored_hit_fraction=counts[threshold]/raw_hits,
                           affected_event_fraction=len(affected_events[threshold])/len(events),
                           raw_reference_energy_mev=reference_energy_total,
                           total_counterfactual_aged_energy_mev=aged_energy_total,
                           censored_reference_energy_mev=reference_loss[threshold],
                           censored_aged_energy_mev=aged_loss[threshold],
                           censored_reference_energy_fraction=reference_loss[threshold]/reference_energy_total,
                           censored_aged_energy_fraction=aged_loss[threshold]/aged_energy_total,
                           archived_mask_disagreements_at_200mev=mask_disagreements))
    return output


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for events_per_state in sorted({r["events_per_state"] for r in rows}):
        for threshold in THRESHOLDS:
            group = [r for r in rows if r["events_per_state"] == events_per_state
                     and r["threshold_mev"] == threshold]
            total_hits = sum(r["raw_reference_hits"] for r in group)
            total_censored = sum(r["censored_hits"] for r in group)
            total_aged_energy = sum(r["total_counterfactual_aged_energy_mev"] for r in group)
            censored_aged_energy = sum(r["censored_aged_energy_mev"] for r in group)
            output.append(dict(events_per_state=events_per_state, threshold_mev=threshold,
                               independent_pairs=len(group), reference_hits=total_hits,
                               censored_hits=total_censored,
                               censored_hit_fraction=total_censored/total_hits,
                               censored_aged_energy_fraction=censored_aged_energy/total_aged_energy,
                               mean_affected_event_fraction=float(np.mean([r["affected_event_fraction"] for r in group])),
                               pair_sd_censored_hit_fraction=float(np.std([r["censored_hit_fraction"] for r in group],ddof=1)),
                               pair_sd_censored_aged_energy_fraction=float(np.std([r["censored_aged_energy_fraction"] for r in group],ddof=1))))
    return output


def plot(summary: list[dict]) -> None:
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.4),constrained_layout=True)
    colors={50_000:"#4b84ac",90_000:"#5b9b75",150_000:"#a66a92",250_000:"#d1904d"}
    for size,color in colors.items():
        group=sorted((r for r in summary if r["events_per_state"]==size),
                     key=lambda r:r["threshold_mev"])
        x=[r["threshold_mev"] for r in group]
        axes[0].plot(x,[100*r["censored_hit_fraction"] for r in group],"o-",
                     color=color,label=f"{size//1000}k events")
        axes[1].plot(x,[100*r["censored_aged_energy_fraction"] for r in group],"o-",
                     color=color,label=f"{size//1000}k events")
    for ax in axes:
        ax.axvline(200,color="black",ls="--",lw=1)
        ax.set_xlabel("Counterfactual aged-hit threshold (MeV)")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Potential hits censored (%)")
    axes[1].set_ylabel("Raw aged energy censored (%)")
    axes[0].legend(frameon=False,fontsize=8)
    fig.savefig(OUT / "threshold_sweep.png",dpi=200)
    fig.savefig(OUT / "threshold_sweep.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    canonical = read_csv(BASE / "analysis/canonical_prepared_runs.csv")
    if len(canonical) != 16:
        raise ValueError("Expected 16 canonical exp38 prepared pairs")
    rows = []
    for record in canonical:
        rows.extend(audit_one(record))
        print("audited", record["events_per_state"], record["data_seed_pair"],flush=True)
    summary = summarize(rows)
    write_csv(OUT / "threshold_pair_runs.csv",rows)
    write_csv(OUT / "threshold_summary.csv",summary)
    plot(summary)
    lines = ["# Threshold-induced censoring in exp38", "",
             "The training code sets the aged-side hit energy to missing when raw calibrated energy times the fixed aging factor is below 200 MeV, then drops those old-side rows. The independent reference archive retains uncensored raw `E_cal_org`, each hit's aging factor, and the archived 200 MeV missing-value mask. We apply thresholds to those reference hits as counterfactual changed measurements. This estimates censoring for the same event distribution; the removed old hits themselves are absent from the prepared old archives. The 200 MeV mask agrees with the archived reference-side mask on every audited hit.", "",
             "| Events/state | Threshold (MeV) | Censored hits | Hit fraction | Aged-energy fraction | Events with at least one censored hit |",
             "|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['events_per_state']:,} | {r['threshold_mev']:.0f} | {r['censored_hits']:,} / {r['reference_hits']:,} | {100*r['censored_hit_fraction']:.3f}% | {100*r['censored_aged_energy_fraction']:.3f}% | {100*r['mean_affected_event_fraction']:.3f}% |")
    lines += ["", "The 0/175/200/250/300 MeV rows are a measurement-threshold sweep on the same uncensored reference events. They do not retrain the published estimators. Censored energy is computed in raw MeV before quantile transformation; the event-level sliced-W2 estimator is fitted to the existing 200 MeV prepared archives.", ""]
    (OUT / "THRESHOLD_RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    (OUT / "threshold_metadata.json").write_text(json.dumps(dict(thresholds_mev=THRESHOLDS,
              canonical_pairs=len(canonical), source="canonical prepared reference archives",
              original_cut="aged raw energy < 200 MeV",
              limitation="counterfactual censoring on independent reference events; old removed rows unavailable"),indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
