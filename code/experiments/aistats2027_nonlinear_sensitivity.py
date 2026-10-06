"""MMD bandwidth and sliced-W2 projection sensitivity on the frozen 50-seed toy.

Every variant uses the same n=512 unpaired train and test arrays, scalar
optimizer, inverse transformation, and common held-out energy evaluation.
The 64-projection draw exactly reproduces the original main benchmark; 32 is
a subset of its directions and 128 adds 64 independently drawn directions.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist, pdist

from aistats2027_final_nonlinear_benchmark import bootstrap_indices, test_data
from aistats2027_nonlinear_curvature import train_data
from aistats2027_nonlinear_transform_experiment import (
    THETAS, empirical_energy_distance, inverse_transform,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/nonlinear_transform"
OUT = BASE / "sensitivity"
MMD_MULTIPLIERS = (0.5, 1.0, 2.0)
SW_PROJECTIONS = (32, 64, 128)
VARIANTS = ("Moments", "MMD 0.5x", "MMD 1x", "MMD 2x", "Energy",
            "SW2 32", "SW2 64", "SW2 128")
SEEDS = range(50)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_variants(reference: np.ndarray, observed: np.ndarray, scale: np.ndarray,
                 seed: int) -> list[dict]:
    ref = reference / scale
    n, d = ref.shape
    bandwidth = max(float(np.median(pdist(ref[:min(n, 256)]))), 1e-6)
    ref_sqdist = cdist(ref, ref, metric="sqeuclidean")
    rng = np.random.default_rng(seed + 71_003)
    first_64 = rng.normal(size=(d, 64))
    first_64 /= np.linalg.norm(first_64, axis=0)
    extra_64 = rng.normal(size=(d, 64))
    extra_64 /= np.linalg.norm(extra_64, axis=0)
    directions_128 = np.concatenate((first_64, extra_64), axis=1)
    output = []

    for multiplier in MMD_MULTIPLIERS:
        bandwidth_used = multiplier * bandwidth
        denominator = 2 * bandwidth_used**2
        reference_kernel = float(np.exp(-ref_sqdist / denominator).mean())

        def objective(theta: float) -> float:
            corrected = inverse_transform(observed, theta) / scale
            corrected_kernel = float(np.exp(-cdist(corrected, corrected,
                                                   metric="sqeuclidean") / denominator).mean())
            cross_kernel = float(np.exp(-cdist(ref, corrected,
                                               metric="sqeuclidean") / denominator).mean())
            return max(reference_kernel + corrected_kernel - 2 * cross_kernel, 0.0)

        start = time.perf_counter()
        result = minimize_scalar(objective, bounds=(0.0, 1.2), method="bounded",
                                 options={"xatol":1e-5, "maxiter":80})
        output.append(dict(variant=f"MMD {multiplier:g}x", family="MMD",
                           hyperparameter="bandwidth_multiplier", value=multiplier,
                           actual_bandwidth=bandwidth_used, theta_estimate=float(result.x),
                           training_objective=float(result.fun),
                           objective_evaluations=int(result.nfev),
                           fit_runtime_seconds=time.perf_counter()-start,
                           success=bool(result.success)))

    for count in SW_PROJECTIONS:
        directions = directions_128[:, :count]
        projected_reference = np.sort(ref @ directions, axis=0)

        def objective(theta: float) -> float:
            corrected = inverse_transform(observed, theta) / scale
            projected_corrected = np.sort(corrected @ directions, axis=0)
            return float(np.mean((projected_reference-projected_corrected)**2))

        start = time.perf_counter()
        result = minimize_scalar(objective, bounds=(0.0, 1.2), method="bounded",
                                 options={"xatol":1e-5, "maxiter":80})
        output.append(dict(variant=f"SW2 {count}", family="Sliced W2",
                           hyperparameter="projections", value=count,
                           actual_bandwidth=float("nan"), theta_estimate=float(result.x),
                           training_objective=float(result.fun),
                           objective_evaluations=int(result.nfev),
                           fit_runtime_seconds=time.perf_counter()-start,
                           success=bool(result.success)))
    return output


def run_fits() -> tuple[list[dict], dict]:
    main = read_csv(BASE / "final_benchmark/runs.csv")
    saved = {(float(r["theta_true"]), int(r["seed"]), r["method"]):r for r in main}
    expected = {(theta, seed, method) for theta in THETAS for seed in SEEDS
                for method in ("Moments", "MMD", "Energy", "Sliced W2")}
    if set(saved) != expected:
        raise ValueError("The saved 50-seed main benchmark is incomplete")
    output = []
    max_baseline_theta_difference = 0.0
    max_baseline_alignment_difference = 0.0
    failures = 0
    for theta in THETAS:
        for seed in SEEDS:
            reference, observed, scale = train_data(theta, seed, 3)
            ref_test, obs_test, test_scale = test_data(theta, seed)
            if not np.array_equal(scale, test_scale):
                raise AssertionError("Training and held-out scaling protocols differ")
            standardized_test = ref_test / scale
            fitted = fit_variants(reference, observed, scale, seed)
            for variant in fitted:
                estimate = variant["theta_estimate"]
                alignment = empirical_energy_distance(
                    standardized_test, inverse_transform(obs_test, estimate) / scale)
                variant.update(theta_true=theta, seed=seed,
                               parameter_error=estimate-theta,
                               absolute_parameter_error=abs(estimate-theta),
                               heldout_energy=alignment)
                original_name = ("MMD" if variant["variant"] == "MMD 1x" else
                                 "Sliced W2" if variant["variant"] == "SW2 64" else None)
                if original_name:
                    saved_row = saved[(theta, seed, original_name)]
                    theta_difference = abs(estimate-float(saved_row["theta_estimate"]))
                    alignment_difference = abs(alignment-float(saved_row["corrected_distribution_error"]))
                    max_baseline_theta_difference = max(max_baseline_theta_difference,
                                                        theta_difference)
                    max_baseline_alignment_difference = max(max_baseline_alignment_difference,
                                                            alignment_difference)
                    if theta_difference > 1e-6 or alignment_difference > 1e-8:
                        raise AssertionError(f"Baseline reproduction failed: {theta}, {seed}, {original_name}, "
                                             f"theta difference {theta_difference}, alignment {alignment_difference}")
                failures += not variant["success"]
                output.append(variant)
            if seed % 10 == 9:
                write_csv(OUT / "variant_runs.csv", output)
                print(f"completed theta={theta:.1f}, seed={seed}, fits={len(output)}", flush=True)
    write_csv(OUT / "variant_runs.csv", output)
    return output, dict(fits=len(output), failures=failures,
                        max_baseline_theta_difference=max_baseline_theta_difference,
                        max_baseline_alignment_difference=max_baseline_alignment_difference)


def summarize(variant_runs: list[dict]) -> tuple[list[dict], list[dict]]:
    saved = read_csv(BASE / "final_benchmark/runs.csv")
    anchors = []
    for row in saved:
        if row["method"] in ("Moments", "Energy"):
            anchors.append(dict(variant=row["method"], family=row["method"],
                                hyperparameter="fixed", value="",
                                actual_bandwidth="", theta_estimate=float(row["theta_estimate"]),
                                training_objective="", objective_evaluations="",
                                fit_runtime_seconds="", success=True,
                                theta_true=float(row["theta_true"]), seed=int(row["seed"]),
                                parameter_error=float(row["parameter_error"]),
                                absolute_parameter_error=float(row["absolute_parameter_error"]),
                                heldout_energy=float(row["corrected_distribution_error"])))
    all_rows = anchors + variant_runs
    indices = bootstrap_indices(50)
    lookup = {(float(r["theta_true"]), int(r["seed"]), r["variant"]):r for r in all_rows}
    keys = [(theta, seed) for theta in THETAS for seed in SEEDS]
    if set(lookup) != {(theta,seed,variant) for theta,seed in keys for variant in VARIANTS}:
        raise AssertionError("A paired sensitivity dataset is missing or duplicated")
    moments_error = np.array([float(lookup[(theta,seed,"Moments")]["parameter_error"])
                              for theta,seed in keys])
    moments_alignment = np.array([float(lookup[(theta,seed,"Moments")]["heldout_energy"])
                                  for theta,seed in keys])
    summary = []
    for variant in VARIANTS:
        records = [lookup[(theta,seed,variant)] for theta,seed in keys]
        errors = np.array([float(r["parameter_error"]) for r in records])
        alignments = np.array([float(r["heldout_energy"]) for r in records])
        rmse_boot = np.sqrt(np.mean(errors[indices]**2, axis=1))
        alignment_boot = np.mean(alignments[indices], axis=1)
        rmse_diff_boot = rmse_boot - np.sqrt(np.mean(moments_error[indices]**2,axis=1))
        align_diff_boot = np.mean((alignments-moments_alignment)[indices], axis=1)
        summary.append(dict(variant=variant, family=records[0]["family"],
                            hyperparameter=records[0]["hyperparameter"], value=records[0]["value"],
                            datasets=len(records), parameter_rmse=float(np.sqrt(np.mean(errors**2))),
                            parameter_rmse_ci_low=float(np.quantile(rmse_boot,.025)),
                            parameter_rmse_ci_high=float(np.quantile(rmse_boot,.975)),
                            mean_absolute_parameter_error=float(np.mean(np.abs(errors))),
                            heldout_energy=float(np.mean(alignments)),
                            heldout_energy_ci_low=float(np.quantile(alignment_boot,.025)),
                            heldout_energy_ci_high=float(np.quantile(alignment_boot,.975)),
                            rmse_difference_vs_moments=float(np.sqrt(np.mean(errors**2))-
                                                              np.sqrt(np.mean(moments_error**2))),
                            rmse_diff_ci_low=float(np.quantile(rmse_diff_boot,.025)),
                            rmse_diff_ci_high=float(np.quantile(rmse_diff_boot,.975)),
                            alignment_difference_vs_moments=float(np.mean(alignments-moments_alignment)),
                            alignment_diff_ci_low=float(np.quantile(align_diff_boot,.025)),
                            alignment_diff_ci_high=float(np.quantile(align_diff_boot,.975))))
    return summary, all_rows


def plot(summary: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), constrained_layout=True)
    colors = ("#547fa8", "#9d648e")
    for axis, metric, lo, hi, ylabel, fixed in (
        (axes[0], "parameter_rmse", "parameter_rmse_ci_low", "parameter_rmse_ci_high",
         "Parameter RMSE", next(r for r in summary if r["variant"] == "Moments")["parameter_rmse"]),
        (axes[1], "heldout_energy", "heldout_energy_ci_low", "heldout_energy_ci_high",
         "Held-out energy distance", next(r for r in summary if r["variant"] == "Moments")["heldout_energy"])):
        for offset, family, variants, label in ((-.09,"MMD",("MMD 0.5x","MMD 1x","MMD 2x"),"MMD bandwidth"),
                                                (.09,"Sliced W2",("SW2 32","SW2 64","SW2 128"),"SW2 projections")):
            selected = [next(r for r in summary if r["variant"] == variant) for variant in variants]
            x = np.arange(3)+offset
            y = np.array([r[metric] for r in selected])
            error = np.array([[r[metric]-r[lo] for r in selected],
                              [r[hi]-r[metric] for r in selected]])
            axis.errorbar(x,y,yerr=error,fmt="o-",capsize=3,label=label,
                          color=colors[0 if family=="MMD" else 1])
        axis.axhline(fixed,color="black",ls="--",lw=1,label="Moments")
        axis.set_xticks(range(3),["low\n0.5x / 32","default\n1x / 64","high\n2x / 128"])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y",alpha=.2)
    axes[0].set_title("A  Parameter recovery")
    axes[1].set_title("B  Common held-out alignment")
    axes[0].legend(frameon=False,fontsize=8,loc="upper center",
                   bbox_to_anchor=(0.5,1.27),ncol=3)
    fig.savefig(OUT / "sensitivity.png",dpi=200,bbox_inches="tight")
    fig.savefig(OUT / "sensitivity.pdf",bbox_inches="tight")
    plt.close(fig)


def report(summary: list[dict], validation: dict) -> None:
    lines = ["# MMD and sliced-W2 hyperparameter sensitivity", "",
             "The six variants are refitted on exactly the 50 paired datasets at each of three true parameters in the main nonlinear benchmark (`n=512`, zero changed-side noise). MMD changes only the median-distance RBF bandwidth multiplier; sliced W2 changes only the number of fixed random projections. For each seed, the 32 directions are a subset of the original 64, and 128 adds 64 more. Every variant uses the same bounded scalar optimizer and common fresh held-out energy-distance evaluation. Moments and Energy are unchanged anchors from the main benchmark.", "",
             "Intervals use 20,000 stratified paired bootstrap draws, preserving methods and variants on each dataset. Differences are variant minus Moments. A negative RMSE difference favors the variant; the alignment differences measure the much smaller changes in a common held-out metric.", "",
             "| Variant | Parameter RMSE [95% CI] | Held-out energy [95% CI] | Paired RMSE difference vs Moments [95% CI] | Paired held-out difference vs Moments [95% CI] |",
             "|---|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        r = next(row for row in summary if row["variant"] == variant)
        lines.append(f"| {variant} | {r['parameter_rmse']:.4f} [{r['parameter_rmse_ci_low']:.4f}, {r['parameter_rmse_ci_high']:.4f}] | {r['heldout_energy']:.5f} [{r['heldout_energy_ci_low']:.5f}, {r['heldout_energy_ci_high']:.5f}] | {r['rmse_difference_vs_moments']:+.4f} [{r['rmse_diff_ci_low']:+.4f}, {r['rmse_diff_ci_high']:+.4f}] | {r['alignment_difference_vs_moments']:+.5f} [{r['alignment_diff_ci_low']:+.5f}, {r['alignment_diff_ci_high']:+.5f}] |")
    lines += ["", "## Checks", "",
              f"- {validation['fits']} refits; {validation['failures']} optimizer failures.",
              f"- Default-setting reproduction against the frozen main benchmark: maximum parameter difference {validation['max_baseline_theta_difference']:.3g}; maximum held-out energy difference {validation['max_baseline_alignment_difference']:.3g}.",
              "- All variant estimates and common held-out evaluations are in `variant_runs.csv`; unchanged anchors appear in `paired_runs.csv`. The figure `sensitivity.pdf` shows both metrics.", "",
              "The toy benchmark has no trained critic among its four compared estimators. Critic regularization is therefore not varied here; changing a separate gas-sensor or calorimeter critic would answer a different sensitivity question and should not be presented as evidence for this controlled four-method phenomenon.", ""]
    (OUT / "RESULTS.md").write_text("\n".join(lines),encoding="utf-8")


def main(summarize_only: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if summarize_only:
        variant_runs = read_csv(OUT / "variant_runs.csv")
        metadata = json.loads((OUT / "metadata.json").read_text(encoding="utf-8"))
        validation = {key:metadata[key] for key in ("fits", "failures",
                      "max_baseline_theta_difference", "max_baseline_alignment_difference")}
    else:
        variant_runs, validation = run_fits()
    summary, paired_runs = summarize(variant_runs)
    write_csv(OUT / "summary.csv", summary)
    write_csv(OUT / "paired_runs.csv", paired_runs)
    plot(summary)
    report(summary, validation)
    (OUT / "metadata.json").write_text(json.dumps(dict(n=512,noise_std=0,dimension=3,
                        theta_true=THETAS,seeds=list(SEEDS),mmd_multipliers=MMD_MULTIPLIERS,
                        sw_projections=SW_PROJECTIONS,
                        sw_direction_protocol="first 64 exactly match original; 32 subset; 128 adds 64",
                        bootstrap="20000 stratified paired draws by theta and seed",
                        common_evaluation="biased empirical energy distance on n_test=512",
                        **validation),indent=2),encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize-only", action="store_true", help="Reuse saved raw fits")
    main(parser.parse_args().summarize_only)
