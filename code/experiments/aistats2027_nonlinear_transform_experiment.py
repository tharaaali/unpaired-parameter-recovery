"""Unpaired recovery for Y = X + theta*tanh(X) with matched objectives.

The experiment uses independent samples from an asymmetric multivariate
mixture.  It runs the requested sample-size and changed-side noise sweeps,
evaluates every fitted parameter with the same held-out energy distance, and
writes publication-ready CSV tables and plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist, pdist

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = ("Moments", "MMD", "Energy", "Sliced W2")
THETAS = (0.2, 0.5, 0.8)
SAMPLE_SIZES = (64, 256, 512)
NOISE_LEVELS = (0.0, 0.1, 0.35, 0.5)


def sample_asymmetric_mixture(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    """Sample a correlated, asymmetric three-component Gaussian mixture."""
    component = rng.choice(3, size=n, p=(0.58, 0.29, 0.13))
    axis = np.arange(d, dtype=float)
    means = np.stack((
        -0.25 + 0.15 * axis,
        1.10 - 0.18 * axis,
        -1.35 + 0.28 * np.sin(axis + 0.4),
    ))
    scales = 0.48 + 0.17 * axis
    independent = rng.normal(size=(n, d)) * scales
    common = rng.normal(size=(n, 1)) * (0.18 + 0.05 * axis)
    return means[component] + independent + common


def forward_transform(x: np.ndarray, theta: float) -> np.ndarray:
    return x + theta * np.tanh(x)


def inverse_transform(y: np.ndarray, theta: float) -> np.ndarray:
    """Invert x + theta*tanh(x) using stable vectorized Newton iterations."""
    x = y.copy()
    for _ in range(12):
        tanh_x = np.tanh(x)
        residual = x + theta * tanh_x - y
        derivative = 1.0 + theta * (1.0 - tanh_x**2)
        x -= residual / derivative
    return x


def empirical_energy_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Biased empirical energy distance, including zero diagonal terms."""
    left_self = float(pdist(left).mean()) * (len(left) - 1) / len(left)
    right_self = float(pdist(right).mean()) * (len(right) - 1) / len(right)
    value = 2.0 * float(cdist(left, right).mean()) - left_self - right_self
    return max(value, 0.0)


def make_objective(method: str, reference: np.ndarray, observed: np.ndarray,
                   scale: np.ndarray, seed: int):
    reference = reference / scale
    n, d = reference.shape

    if method == "Moments":
        target_mean = reference.mean(axis=0)
        target_covariance = np.cov(reference, rowvar=False).reshape(d, d)

        def objective(theta: float) -> float:
            corrected = inverse_transform(observed, theta) / scale
            mean_error = np.mean((corrected.mean(axis=0) - target_mean) ** 2)
            covariance_error = np.mean(
                (np.cov(corrected, rowvar=False).reshape(d, d) - target_covariance) ** 2
            )
            return float(mean_error + covariance_error)

        details = {"objective": "squared mean plus covariance discrepancy"}

    elif method == "MMD":
        bandwidth_sample = reference[: min(n, 256)]
        bandwidth = max(float(np.median(pdist(bandwidth_sample))), 1e-6)
        denominator = 2.0 * bandwidth**2
        reference_kernel_mean = float(
            np.exp(-cdist(reference, reference, metric="sqeuclidean") / denominator).mean()
        )

        def objective(theta: float) -> float:
            corrected = inverse_transform(observed, theta) / scale
            corrected_kernel_mean = float(
                np.exp(-cdist(corrected, corrected, metric="sqeuclidean") / denominator).mean()
            )
            cross_kernel_mean = float(
                np.exp(-cdist(reference, corrected, metric="sqeuclidean") / denominator).mean()
            )
            return max(reference_kernel_mean + corrected_kernel_mean - 2 * cross_kernel_mean, 0.0)

        details = {"objective": "biased RBF MMD squared", "bandwidth": bandwidth}

    elif method == "Energy":
        reference_self = float(pdist(reference).mean()) * (n - 1) / n

        def objective(theta: float) -> float:
            corrected = inverse_transform(observed, theta) / scale
            corrected_self = float(pdist(corrected).mean()) * (n - 1) / n
            value = 2 * float(cdist(reference, corrected).mean()) - reference_self - corrected_self
            return max(value, 0.0)

        details = {"objective": "biased empirical energy distance"}

    elif method == "Sliced W2":
        rng = np.random.default_rng(seed + 71_003)
        directions = rng.normal(size=(d, 64))
        directions /= np.linalg.norm(directions, axis=0)
        projected_reference = np.sort(reference @ directions, axis=0)

        def objective(theta: float) -> float:
            corrected = inverse_transform(observed, theta) / scale
            projected_corrected = np.sort(corrected @ directions, axis=0)
            return float(np.mean((projected_reference - projected_corrected) ** 2))

        details = {"objective": "squared sliced Wasserstein-2", "projections": 64}

    else:
        raise ValueError(method)
    return objective, details


def fit_method(method: str, reference: np.ndarray, observed: np.ndarray,
               reference_test: np.ndarray, observed_test: np.ndarray,
               theta_true: float, scale: np.ndarray, seed: int) -> dict:
    start = time.perf_counter()
    objective, details = make_objective(method, reference, observed, scale, seed)
    result = minimize_scalar(
        objective,
        bounds=(0.0, 1.2),
        method="bounded",
        options={"xatol": 1e-5, "maxiter": 80},
    )
    fit_runtime = time.perf_counter() - start
    estimate = float(result.x)
    corrected_test = inverse_transform(observed_test, estimate) / scale
    standardized_reference_test = reference_test / scale
    initial_test = inverse_transform(observed_test, 0.0) / scale
    oracle_test = inverse_transform(observed_test, theta_true) / scale
    error = estimate - theta_true
    return {
        "method": method,
        "theta_true": theta_true,
        "theta_estimate": estimate,
        "parameter_error": error,
        "absolute_parameter_error": abs(error),
        "squared_parameter_error": error**2,
        "relative_parameter_error": abs(error) / abs(theta_true),
        "squared_relative_parameter_error": (error / theta_true) ** 2,
        "corrected_distribution_error": empirical_energy_distance(
            standardized_reference_test, corrected_test
        ),
        "initial_distribution_error": empirical_energy_distance(
            standardized_reference_test, initial_test
        ),
        "oracle_distribution_error": empirical_energy_distance(
            standardized_reference_test, oracle_test
        ),
        "training_objective": float(result.fun),
        "objective_evaluations": int(result.nfev),
        "fit_runtime_seconds": fit_runtime,
        "success": bool(result.success),
        "optimizer_message": str(result.message),
        "objective_details": json.dumps(details, sort_keys=True),
    }


def bootstrap_rmse(values: np.ndarray, seed: int, draws: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    bootstrapped = np.sqrt(np.mean(values[indices] ** 2, axis=1))
    return float(np.quantile(bootstrapped, 0.025)), float(np.quantile(bootstrapped, 0.975))


def aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for group_index, (values, group) in enumerate(sorted(groups.items())):
        errors = np.array([float(row["parameter_error"]) for row in group])
        relative_errors = errors / np.array([float(row["theta_true"]) for row in group])
        rmse_low, rmse_high = bootstrap_rmse(errors, 80_000 + group_index)
        relative_low, relative_high = bootstrap_rmse(
            relative_errors, 90_000 + group_index
        )
        result = {key: value for key, value in zip(keys, values)}
        result.update({
            "runs": len(group),
            "parameter_rmse": float(np.sqrt(np.mean(errors**2))),
            "parameter_rmse_ci_low": rmse_low,
            "parameter_rmse_ci_high": rmse_high,
            "relative_parameter_rmse": float(np.sqrt(np.mean(relative_errors**2))),
            "relative_parameter_rmse_ci_low": relative_low,
            "relative_parameter_rmse_ci_high": relative_high,
            "mean_absolute_parameter_error": float(np.mean(np.abs(errors))),
            "corrected_distribution_error_mean": float(np.mean([
                row["corrected_distribution_error"] for row in group
            ])),
            "corrected_distribution_error_std": float(np.std([
                row["corrected_distribution_error"] for row in group
            ], ddof=1)),
            "initial_distribution_error_mean": float(np.mean([
                row["initial_distribution_error"] for row in group
            ])),
            "oracle_distribution_error_mean": float(np.mean([
                row["oracle_distribution_error"] for row in group
            ])),
            "objective_evaluations_mean": float(np.mean([
                row["objective_evaluations"] for row in group
            ])),
            "fit_runtime_seconds_mean": float(np.mean([
                row["fit_runtime_seconds"] for row in group
            ])),
            "failed_runs": int(sum(not row["success"] for row in group)),
        })
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_sweep(summary: list[dict], x_key: str, xlabel: str, output_stem: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    for method in METHODS:
        selected = sorted((row for row in summary if row["method"] == method),
                          key=lambda row: float(row[x_key]))
        x = np.array([float(row[x_key]) for row in selected])
        y = np.array([row["parameter_rmse"] for row in selected])
        lower = y - np.array([row["parameter_rmse_ci_low"] for row in selected])
        upper = np.array([row["parameter_rmse_ci_high"] for row in selected]) - y
        axis.errorbar(x, y, yerr=np.vstack((lower, upper)), marker="o",
                      linewidth=1.8, capsize=3, label=method)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(r"Parameter RMSE")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_main_table(path: Path, summary: list[dict]) -> None:
    lines = [
        "| Method | Parameter RMSE [95% CI] ↓ | Relative parameter RMSE [95% CI] ↓ | "
        "Corrected distribution error ↓ | Evaluations ↓ | Runtime (s) ↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        row = next(item for item in summary if item["method"] == method)
        lines.append(
            f"| {method} | {row['parameter_rmse']:.4f} "
            f"[{row['parameter_rmse_ci_low']:.4f}, {row['parameter_rmse_ci_high']:.4f}] | "
            f"{row['relative_parameter_rmse']:.4f} "
            f"[{row['relative_parameter_rmse_ci_low']:.4f}, "
            f"{row['relative_parameter_rmse_ci_high']:.4f}] | "
            f"{row['corrected_distribution_error_mean']:.5f} ± "
            f"{row['corrected_distribution_error_std']:.5f} | "
            f"{row['objective_evaluations_mean']:.1f} | "
            f"{row['fit_runtime_seconds_mean']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_theta_table(path: Path, theta_summary: list[dict],
                      main_summary: list[dict], relative: bool = False) -> None:
    if relative:
        metric = "relative_parameter_rmse"
        low_key = "relative_parameter_rmse_ci_low"
        high_key = "relative_parameter_rmse_ci_high"
        title = "Relative parameter RMSE [95% CI]"
        multiplier = 100.0
        suffix = "%"
    else:
        metric = "parameter_rmse"
        low_key = "parameter_rmse_ci_low"
        high_key = "parameter_rmse_ci_high"
        title = "Parameter RMSE [95% CI]"
        multiplier = 1.0
        suffix = ""
    lines = [
        f"| Method | $\\theta^\\star=0.2$ {title} | "
        f"$\\theta^\\star=0.5$ {title} | $\\theta^\\star=0.8$ {title} | Overall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        cells = []
        for theta in THETAS:
            row = next(item for item in theta_summary
                       if item["method"] == method and float(item["theta_true"]) == theta)
            cells.append(
                f"{multiplier * row[metric]:.2f}{suffix} "
                f"[{multiplier * row[low_key]:.2f}, "
                f"{multiplier * row[high_key]:.2f}]"
                if relative else
                f"{row[metric]:.4f} [{row[low_key]:.4f}, {row[high_key]:.4f}]"
            )
        overall = next(item for item in main_summary if item["method"] == method)
        if relative:
            overall_cell = (
                f"{100 * overall[metric]:.2f}% "
                f"[{100 * overall[low_key]:.2f}, {100 * overall[high_key]:.2f}]"
            )
        else:
            overall_cell = (
                f"{overall[metric]:.4f} "
                f"[{overall[low_key]:.4f}, {overall[high_key]:.4f}]"
            )
        lines.append(f"| {method} | " + " | ".join(cells + [overall_cell]) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def paired_comparisons(rows: list[dict], bootstrap_draws: int = 10_000) -> list[dict]:
    """Compare each method to Energy using paired dataset resampling."""
    selected = [row for row in rows if row["n"] == 512 and row["noise_std"] == 0.0]
    output = []
    for theta_group in ("overall", *THETAS):
        group = selected if theta_group == "overall" else [
            row for row in selected if float(row["theta_true"]) == theta_group
        ]
        datasets: dict[tuple[float, int], dict[str, float]] = defaultdict(dict)
        for row in group:
            key = (float(row["theta_true"]), int(row["seed"]))
            datasets[key][row["method"]] = float(row["parameter_error"])
        ordered = [datasets[key] for key in sorted(datasets)]
        if not all(set(item) == set(METHODS) for item in ordered):
            raise RuntimeError("Paired comparison requires every method on every dataset")
        energy_errors = np.array([item["Energy"] for item in ordered])
        for method_index, comparator in enumerate(method for method in METHODS if method != "Energy"):
            comparator_errors = np.array([item[comparator] for item in ordered])
            absolute_differences = np.abs(comparator_errors) - np.abs(energy_errors)
            rmse_difference = (
                np.sqrt(np.mean(comparator_errors**2))
                - np.sqrt(np.mean(energy_errors**2))
            )
            rng = np.random.default_rng(
                130_000 + method_index + 100 * (0 if theta_group == "overall" else THETAS.index(theta_group) + 1)
            )
            indices = rng.integers(0, len(ordered), size=(bootstrap_draws, len(ordered)))
            mean_difference_bootstrap = np.mean(absolute_differences[indices], axis=1)
            comparator_bootstrap = comparator_errors[indices]
            energy_bootstrap = energy_errors[indices]
            rmse_difference_bootstrap = (
                np.sqrt(np.mean(comparator_bootstrap**2, axis=1))
                - np.sqrt(np.mean(energy_bootstrap**2, axis=1))
            )
            output.append({
                "theta_group": theta_group,
                "comparator": comparator,
                "reference": "Energy",
                "paired_runs": len(ordered),
                "mean_absolute_error_improvement": float(np.mean(absolute_differences)),
                "mean_absolute_error_improvement_ci_low": float(np.quantile(mean_difference_bootstrap, 0.025)),
                "mean_absolute_error_improvement_ci_high": float(np.quantile(mean_difference_bootstrap, 0.975)),
                "rmse_improvement": float(rmse_difference),
                "rmse_improvement_ci_low": float(np.quantile(rmse_difference_bootstrap, 0.025)),
                "rmse_improvement_ci_high": float(np.quantile(rmse_difference_bootstrap, 0.975)),
                "energy_win_fraction": float(np.mean(np.abs(energy_errors) < np.abs(comparator_errors))),
            })
    return output


def write_paired_table(path: Path, paired: list[dict]) -> None:
    overall = [row for row in paired if row["theta_group"] == "overall"]
    lines = [
        "Positive differences favor Energy. Intervals use paired bootstrap resampling of the same datasets.",
        "",
        "| Comparator vs Energy | Mean paired absolute-error improvement [95% CI] | "
        "RMSE improvement [95% CI] | Energy win fraction |",
        "| --- | ---: | ---: | ---: |",
    ]
    for comparator in ("Moments", "MMD", "Sliced W2"):
        row = next(item for item in overall if item["comparator"] == comparator)
        lines.append(
            f"| {comparator} | {row['mean_absolute_error_improvement']:.4f} "
            f"[{row['mean_absolute_error_improvement_ci_low']:.4f}, "
            f"{row['mean_absolute_error_improvement_ci_high']:.4f}] | "
            f"{row['rmse_improvement']:.4f} "
            f"[{row['rmse_improvement_ci_low']:.4f}, "
            f"{row['rmse_improvement_ci_high']:.4f}] | "
            f"{100 * row['energy_win_fraction']:.1f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("results/nonlinear_transform"))
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of integer seeds beginning at zero")
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--test-n", type=int, default=512)
    args = parser.parse_args()
    if args.seeds < 2 or args.dimension < 1 or args.test_n < 3:
        parser.error("Require at least two seeds, positive dimension and test-n >= 3")
    args.output.mkdir(parents=True, exist_ok=True)

    conditions = [(n, 0.0) for n in SAMPLE_SIZES]
    conditions.extend((512, noise) for noise in NOISE_LEVELS if noise != 0.0)
    rows: list[dict] = []
    total = len(THETAS) * args.seeds * len(conditions) * len(METHODS)
    completed = 0
    experiment_start = time.perf_counter()
    for theta_index, theta_true in enumerate(THETAS):
        for seed in range(args.seeds):
            rng = np.random.default_rng(1_000_003 + 10_007 * seed + 1_009 * theta_index)
            reference_full = sample_asymmetric_mixture(rng, max(SAMPLE_SIZES), args.dimension)
            changed_source_full = sample_asymmetric_mixture(rng, max(SAMPLE_SIZES), args.dimension)
            noise_full = rng.normal(size=changed_source_full.shape)
            reference_test = sample_asymmetric_mixture(rng, args.test_n, args.dimension)
            changed_source_test = sample_asymmetric_mixture(rng, args.test_n, args.dimension)
            noise_test = rng.normal(size=changed_source_test.shape)
            for n, noise_std in conditions:
                reference = reference_full[:n]
                observed = forward_transform(changed_source_full[:n], theta_true)
                observed = observed + noise_std * noise_full[:n]
                observed_test = forward_transform(changed_source_test, theta_true)
                observed_test = observed_test + noise_std * noise_test
                scale = np.maximum(reference.std(axis=0, ddof=1), 0.1)
                for method in METHODS:
                    row = fit_method(
                        method, reference, observed, reference_test, observed_test,
                        theta_true, scale, seed,
                    )
                    row.update({
                        "seed": seed,
                        "n": n,
                        "noise_std": noise_std,
                        "dimension": args.dimension,
                        "test_n": args.test_n,
                    })
                    rows.append(row)
                    completed += 1
                print(
                    f"completed {completed:3d}/{total}: theta={theta_true:.1f}, "
                    f"seed={seed}, n={n}, noise={noise_std:.2f}",
                    flush=True,
                )

    write_csv(args.output / "runs.csv", rows)
    sample_rows = [row for row in rows if row["noise_std"] == 0.0]
    noise_rows = [row for row in rows if row["n"] == 512]
    sample_summary = aggregate(sample_rows, ("method", "n"))
    noise_summary = aggregate(noise_rows, ("method", "noise_std"))
    theta_summary = aggregate(
        [row for row in rows if row["n"] == 512 and row["noise_std"] == 0.0],
        ("method", "theta_true"),
    )
    main_summary = aggregate(
        [row for row in rows if row["n"] == 512 and row["noise_std"] == 0.0],
        ("method",),
    )
    write_csv(args.output / "sample_size_summary.csv", sample_summary)
    write_csv(args.output / "noise_summary.csv", noise_summary)
    write_csv(args.output / "theta_summary.csv", theta_summary)
    write_csv(args.output / "main_method_summary.csv", main_summary)
    write_main_table(args.output / "main_method_table.md", main_summary)
    write_theta_table(args.output / "theta_method_table.md", theta_summary, main_summary)
    write_theta_table(
        args.output / "theta_relative_method_table.md",
        theta_summary,
        main_summary,
        relative=True,
    )
    paired = paired_comparisons(rows)
    write_csv(args.output / "paired_comparisons.csv", paired)
    write_paired_table(args.output / "paired_comparisons.md", paired)
    plot_sweep(sample_summary, "n", "Samples per distribution",
               args.output / "parameter_rmse_vs_sample_size")
    plot_sweep(noise_summary, "noise_std", "Changed-side noise standard deviation",
               args.output / "parameter_rmse_vs_noise")
    metadata = {
        "model": "Y = X + theta*tanh(X) + epsilon",
        "theta_values": THETAS,
        "sample_sizes": SAMPLE_SIZES,
        "noise_levels": NOISE_LEVELS,
        "seeds": list(range(args.seeds)),
        "dimension": args.dimension,
        "test_n": args.test_n,
        "unpaired": "reference and changed samples are independent draws",
        "correction": "numerical inverse of x + theta*tanh(x)",
        "common_corrected_distribution_metric": (
            "biased empirical energy distance on fresh held-out samples after "
            "feature-wise scaling by the training reference standard deviation"
        ),
        "relative_error": "abs(theta_hat-theta_true)/abs(theta_true); table reports its RMS",
        "runtime": "objective setup plus bounded scalar optimization; held-out evaluation excluded",
        "total_experiment_seconds": time.perf_counter() - experiment_start,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
