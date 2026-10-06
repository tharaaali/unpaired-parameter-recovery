"""Check manuscript numbers against the distributed, saved experiment outputs.

Uses Python's standard library only. This is an audit of reported results, not
a replacement for rerunning computationally intensive fits.
"""

from __future__ import annotations

import csv
import gzip
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
checks = 0


def read(relative: str) -> list[dict[str, str]]:
    with (RESULTS / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check(label: str, actual: float, printed: float, decimals: int) -> None:
    global checks
    tolerance = 0.5 * 10 ** (-decimals) + 1e-12
    if not math.isfinite(actual) or abs(actual - printed) > tolerance:
        raise AssertionError(f"{label}: saved {actual}, paper {printed} ({decimals} decimals)")
    checks += 1


def exact(label: str, actual: object, expected: object) -> None:
    global checks
    if actual != expected:
        raise AssertionError(f"{label}: saved {actual!r}, expected {expected!r}")
    checks += 1


def one(rows: list[dict[str, str]], **conditions: str) -> dict[str, str]:
    matches = [row for row in rows if all(row[key] == value for key, value in conditions.items())]
    if len(matches) != 1:
        raise AssertionError(f"Expected one row for {conditions}, found {len(matches)}")
    return matches[0]


def nonlinear() -> None:
    summary = read("nonlinear_transform/final_benchmark/summary.csv")
    expected = {
        "Moments": (0.0616, 0.00920),
        "MMD": (0.0538, 0.00909),
        "Energy": (0.0523, 0.00908),
        "Sliced W2": (0.0530, 0.00905),
    }
    for method, (rmse, alignment) in expected.items():
        row = one(summary, method=method)
        check(f"nonlinear {method} RMSE", float(row["parameter_rmse"]), rmse, 4)
        check(f"nonlinear {method} alignment", float(row["corrected_distribution_error"]), alignment, 5)
        exact(f"nonlinear {method} fits", int(row["total_fits"]), 150)
    check("nonlinear injected-parameter alignment", float(summary[0]["finite_sample_oracle_error"]), 0.00865, 5)
    check("nonlinear uncorrected alignment", float(summary[0]["uncorrected_distribution_error"]), 0.08294, 5)
    paired = read("nonlinear_transform/final_benchmark/paired_comparisons.csv")
    for comparator, difference in (("Moments", 0.0093), ("MMD", 0.0015), ("Sliced W2", 0.0007)):
        row = one(paired, comparator=comparator, reference="Energy")
        check(f"paired {comparator}-Energy RMSE", float(row["rmse_difference"]), difference, 4)
    sensitivity = read("nonlinear_transform/sensitivity/summary.csv")
    for variant, rmse in (("MMD 0.5x", 0.0526), ("MMD 1x", 0.0538),
                          ("MMD 2x", 0.0562), ("SW2 32", 0.0528),
                          ("SW2 64", 0.0530), ("SW2 128", 0.0514)):
        check(f"sensitivity {variant}", float(one(sensitivity, variant=variant)["parameter_rmse"]), rmse, 4)
    sizes = read("nonlinear_transform/sample_size_summary.csv")
    for size, value in (("64", 0.1289), ("512", 0.0558)):
        check(f"energy sample-size RMSE n={size}",
              float(one(sizes, method="Energy", n=size)["parameter_rmse"]), value, 4)
    noisy = [float(row["parameter_rmse"]) for row in read("nonlinear_transform/noise_summary.csv")
             if row["noise_std"] == "0.5"]
    exact("four noisy methods", len(noisy), 4)
    check("noise RMSE minimum", min(noisy), 0.1572, 4)
    check("noise RMSE maximum", max(noisy), 0.1886, 4)
    print("PASS nonlinear benchmark, paired contrasts, sensitivity, sample size, and noise")


def geometry_and_misspecification() -> None:
    curvature = read("nonlinear_transform/curvature/summary.csv")
    exact("curvature settings", len(curvature), 12)
    ratios = [float(row["predicted_to_holdout_sd_ratio"]) for row in curvature]
    check("curvature ratio minimum", min(ratios), 1.01, 2)
    check("curvature ratio maximum", max(ratios), 1.40, 2)
    for row in curvature:
        observed = float(row["holdout_sd"])
        if not (float(row["predicted_se_ci_low"]) <= observed <= float(row["predicted_se_ci_high"])):
            raise AssertionError("Observed SD outside predictor interval")
    for method, h, gradient, predicted, observed in (
        ("Moments", 1.240, 0.1077, 0.0868, 0.0620),
        ("Energy", 0.391, 0.0249, 0.0637, 0.0545),
    ):
        row = one(curvature, method=method, theta_true="0.5")
        for field, value, decimals in (("mean_curvature_truth", h, 3),
                                       ("gradient_sd", gradient, 4),
                                       ("predicted_se", predicted, 4),
                                       ("holdout_sd", observed, 4)):
            check(f"curvature {method} {field}", float(row[field]), value, decimals)

    rotation = read("rotation_separation/summary.csv")
    exact("rotation anisotropy levels", len(rotation), 7)
    for rho, rmse, width in (("0.0", 0.949, math.pi),
                             ("0.02", 0.798, math.pi),
                             ("1.0", 0.085, 0.063)):
        row = one(rotation, rho=rho)
        check(f"rotation RMSE rho={rho}", float(row["parameter_rmse_mod_pi"]), rmse, 3)
        if rho != "0.0":
            check(f"rotation width rho={rho}", float(row["near_optimal_width_radians"]), width, 3)

    target = read("misspecification_targets/summary.csv")
    for method, shift in (("Moments", 0.2385), ("MMD", 0.2486),
                          ("Energy", 0.2529), ("Sliced W2", 0.2403)):
        check(f"misspecification {method} shift",
              float(one(target, method=method, beta="0.2")["target_shift_from_beta0"]), shift, 4)
    energy = one(target, method="Energy", beta="0.2")
    check("misspecification population contrast", float(energy["excess_target_shift_vs_moments"]), 0.01435, 5)
    contrast = one(read("misspecification_targets/paired_shift_contrasts.csv"),
                   method="Energy", beta="0.2")
    for field, value in (("finite_mean_excess_shift", 0.01470),
                         ("finite_excess_shift_ci_low", 0.01399),
                         ("finite_excess_shift_ci_high", 0.01538)):
        check(f"misspecification finite {field}", float(contrast[field]), value, 5)
    print("PASS local geometry, rotational separation, and misspecification")


def calorimeter() -> None:
    projected = read("exp38_event_level/projected_baselines/summary.csv")
    sliced = read("exp38_event_level/structured_sw_summary.csv")
    expected = {
        "moments": (0.0320, 0.0087, 2.513),
        "mmd": (0.0395, 0.0223, 2.512),
        "energy": (0.0422, 0.0151, 2.520),
    }
    for method, (rmse, sd, alignment_scaled) in expected.items():
        row = one(projected, events_per_state="150000", method_key=method)
        check(f"calorimeter {method} RMSE", float(row["rmse_mean"]), rmse, 4)
        check(f"calorimeter {method} SD", float(row["rmse_std"]), sd, 4)
        check(f"calorimeter {method} alignment", float(row["heldout_sliced_w2_squared_mean"]) * 1e6,
              alignment_scaled, 3)
        exact(f"calorimeter {method} pairs", int(row["independent_pairs"]), 4)
    row = one(sliced, events_per_state="150000")
    check("calorimeter sliced W2 RMSE", float(row["rmse_mean"]), 0.0283, 4)
    check("calorimeter sliced W2 SD", float(row["rmse_std"]), 0.0037, 4)
    check("calorimeter sliced W2 alignment", float(row["heldout_sliced_w2_squared_mean"]) * 1e6, 2.483, 3)
    check("calorimeter injected-factor alignment", float(row["heldout_oracle_sliced_w2_squared_mean"]), 2.465e-6, 9)
    check("calorimeter uncorrected alignment", float(row["heldout_uncorrected_sliced_w2_squared_mean"]), 2.125e-5, 8)
    exact("calorimeter new fitted runs", len(read("exp38_event_level/projected_baselines/pair_metrics.csv"))
          + len(read("exp38_event_level/structured_sw_pair_metrics.csv")), 64)

    threshold = read("exp38_event_level/threshold_summary.csv")
    for cutoff, hits, energy, events in (("0.0", 0.0, 0.0, 0.0),
                                         ("175.0", 0.975, 0.072, 6.020),
                                         ("200.0", 4.933, 0.396, 26.100),
                                         ("250.0", 13.141, 1.196, 52.845),
                                         ("300.0", 19.858, 1.996, 65.968)):
        item = one(threshold, events_per_state="150000", threshold_mev=cutoff)
        check(f"threshold {cutoff} hit percentage", float(item["censored_hit_fraction"]) * 100, hits, 3)
        check(f"threshold {cutoff} energy percentage", float(item["censored_aged_energy_fraction"]) * 100, energy, 3)
        check(f"threshold {cutoff} event percentage", float(item["mean_affected_event_fraction"]) * 100, events, 3)

    profiles = read("exp38_event_level/objective_profiles/key_points.csv")
    for direction, amplitude, data, sd, penalty in (
        ("global_decrease", -0.02, 0.010483, 0.003368, 0.0),
        ("layer_16", 0.02, -0.000625, 0.001070, 0.0),
        ("smooth_xy_layer_16", 0.02, -0.00000013, 0.0000465, 0.000552),
    ):
        item = next(row for row in profiles if row["direction"] == direction
                    and math.isclose(float(row["amplitude"]), amplitude, abs_tol=1e-12))
        check(f"profile {direction} data", float(item["heldout_data_change_mean"]), data, 6 if direction != "smooth_xy_layer_16" else 7)
        check(f"profile {direction} SD", float(item["heldout_data_change_sd"]), sd, 6 if direction != "smooth_xy_layer_16" else 7)
        check(f"profile {direction} penalty", float(item["spatial_penalty_change"]), penalty, 6)
    published = one(read("published_sensors_context.csv"), method="Wasserstein adversarial")
    check("published Sensors RMSE", float(published["factor_rmse_mean"]), 0.0203, 4)
    check("published Sensors pair SD", float(published["factor_rmse_pair_sd"]), 0.0005, 4)
    print("PASS calorimeter four-method table, saved threshold summary, profiles, and cited context")


def affine() -> None:
    runs = read("synthetic_n512/runs.csv")
    for method, mean, sd in (("moments", 0.034, 0.008),
                             ("rff_mmd", 0.032, 0.012),
                             ("energy", 0.031, 0.012),
                             ("sliced_w2", 0.032, 0.006)):
        values = [float(row["parameter_rmse"]) for row in runs if row["method"] == method]
        exact(f"affine {method} seeds", len(values), 5)
        check(f"affine {method} mean", statistics.mean(values), mean, 3)
        check(f"affine {method} SD", statistics.stdev(values), sd, 3)
    print("PASS affine implementation check")


def data_and_figures() -> None:
    manifest = ROOT / "data" / "calorimeter" / "manifest.csv"
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as handle:
            files = list(csv.DictReader(handle))
        exact("minimal calorimeter data files", len(files), 32)
        for entry in files:
            data_file = manifest.parent / entry["file"]
            exact(f"data file exists {entry['file']}", data_file.is_file(), True)
            exact(f"data event count {entry['file']}",
                  int(entry["events"]), int(entry["archive_events_per_state"]))
            with gzip.open(data_file, "rt", newline="", encoding="utf-8") as handle:
                exact(f"data columns {entry['file']}",
                      next(csv.reader(handle)),
                      ["cellid", "energy_raw_reference_mev", "aging_factor", "event", "x", "y", "z"])
        print("PASS optional calorimeter data manifest and table headers")
    else:
        print("INFO raw calorimeter event tables are omitted from the review supplement")
    figures = (
        "main_figure.pdf", "curvature_vs_uncertainty.pdf", "separation_sweep.pdf",
        "pseudo_true_trajectories.pdf", "parameter_rmse_vs_sample_size.pdf",
        "parameter_rmse_vs_noise.pdf", "rotation_identifiability.pdf",
        "exp38_objective_profiles.pdf",
    )
    for name in figures:
        exact(f"paper figure {name}", (ROOT / "figures" / name).is_file(), True)
    print("PASS all manuscript figures")


if __name__ == "__main__":
    nonlinear()
    geometry_and_misspecification()
    calorimeter()
    affine()
    data_and_figures()
    print(f"PASS {checks} checks")
