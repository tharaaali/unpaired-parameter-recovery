"""Collect the exact saved experiment outputs and path-adjusted analysis code.

Run this only from a workspace containing the original AISTATS experiment tree.
The resulting material directory contains the files needed for reader inspection.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


MATERIAL = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[5]
SOURCE_RESULTS = WORKSPACE / "Publications" / "AISTATS_2027"
SOURCE_SCRIPTS = WORKSPACE / "scripts"

EXPERIMENT_SCRIPTS = (
    "aistats2027_nonlinear_transform_experiment.py",
    "aistats2027_nonlinear_curvature.py",
    "aistats2027_final_nonlinear_benchmark.py",
    "aistats2027_nonlinear_sensitivity.py",
    "aistats2027_rotation_separation.py",
    "aistats2027_misspecification_targets.py",
    "aistats2027_synthetic_benchmark.py",
    "aistats2027_joint_sliced_calo.py",
    "aistats2027_exp38_structured_sw.py",
    "aistats2027_exp38_projected_baselines.py",
    "aistats2027_exp38_profile_diagnostic.py",
    "aistats2027_exp38_threshold_audit.py",
)

RESULT_DIRS = (
    "nonlinear_transform",
    "rotation_separation",
    "misspecification_targets",
    "synthetic_n512",
    "synthetic_pilot",
    "exp38_event_level",
)

PAPER_FIGURES = (
    "main_figure.pdf",
    "curvature_vs_uncertainty.pdf",
    "separation_sweep.pdf",
    "pseudo_true_trajectories.pdf",
    "parameter_rmse_vs_sample_size.pdf",
    "parameter_rmse_vs_noise.pdf",
    "rotation_identifiability.pdf",
    "exp38_objective_profiles.pdf",
)


def anonymous_json(source: Path, destination: Path) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))

    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()
                    if key not in {"input", "fitted_field_source"}}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, str) and ("\\Users\\" in value or "/Users/" in value):
            return "[local path removed]"
        if isinstance(value, str) and "Publications\\AISTATS_2027\\" in value:
            return value.replace("Publications\\AISTATS_2027\\", "results\\")
        return value

    destination.write_text(json.dumps(clean(data), indent=2) + "\n", encoding="utf-8")


def copy_results() -> None:
    for directory in RESULT_DIRS:
        source_root = SOURCE_RESULTS / directory
        for source in source_root.rglob("*"):
            if not source.is_file() or source.suffix.lower() not in {".csv", ".json", ".pdf", ".png"}:
                continue
            # Only archived summary and run outputs from the paper; no pilot
            # directories or unrelated detector studies are traversed.
            destination = MATERIAL / "results" / directory / source.relative_to(source_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix.lower() == ".json":
                anonymous_json(source, destination)
            else:
                shutil.copyfile(source, destination)


def copy_code() -> None:
    destination_root = MATERIAL / "code" / "experiments"
    destination_root.mkdir(parents=True, exist_ok=True)
    for name in EXPERIMENT_SCRIPTS:
        original = (SOURCE_SCRIPTS / name).read_text(encoding="utf-8")
        # Path-only adjustments allow the synthetic scripts to use the local
        # material/results tree. The scientific calculations are unmodified.
        adjusted = original.replace(
            "Path(__file__).resolve().parents[1]",
            "Path(__file__).resolve().parents[2]",
        ).replace("Publications/AISTATS_2027/", "results/")
        (destination_root / name).write_text(adjusted, encoding="utf-8")


def copy_figures() -> None:
    source_root = SOURCE_RESULTS / "AISTAS-submission" / "figures"
    destination_root = MATERIAL / "figures"
    destination_root.mkdir(parents=True, exist_ok=True)
    for name in PAPER_FIGURES:
        shutil.copyfile(source_root / name, destination_root / name)


def write_published_context() -> None:
    destination = MATERIAL / "results" / "published_sensors_context.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("method", "archive_events_per_state", "train_events_per_state",
                         "factor_rmse_mean", "factor_rmse_pair_sd", "independent_pairs",
                         "configured_fits_per_pair", "matched_heldout_score", "source_doi"))
        writer.writerow(("Wasserstein adversarial", 150000, 150000, 0.0203, 0.0005,
                         4, 3, "unavailable", "10.3390/s26165024"))


def main() -> None:
    copy_results()
    copy_code()
    copy_figures()
    write_published_context()
    print("Copied experiment scripts, run outputs, aggregate results, and paper figures.")


if __name__ == "__main__":
    main()
