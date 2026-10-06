"""CPU pilot for unpaired transformation recovery with matched objectives.

Example:
    python scripts/aistats2027_synthetic_benchmark.py --seeds 0 1 2

This intentionally excludes WGAN and Sinkhorn. It is a small, fully unpaired
feasibility benchmark, not a substitute for the paper's full comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist
from scipy.special import logsumexp


CASES = {
    "shift_2d": (2, "shift"),
    "scale_2d": (2, "scale"),
    "shift_scale_2d": (2, "shift_scale"),
    "shift_10d": (10, "shift"),
    "scale_10d": (10, "scale"),
}
METHODS = ("moments", "rff_mmd", "energy", "sliced_w2", "sinkhorn")


def source_sample(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    """Asymmetric mixture with correlations and unequal feature scales."""
    component = rng.choice(3, size=n, p=(0.55, 0.30, 0.15))
    axis = np.arange(d, dtype=float)
    means = np.stack((
        0.15 * np.sin(axis + 1),
        0.7 + 0.25 * np.cos(axis * 0.7),
        -0.9 + 0.2 * np.sin(axis * 0.9),
    ))
    scales = 0.55 + 0.35 * (axis % 4) / 3
    independent = rng.normal(size=(n, d)) * scales
    common = rng.normal(size=(n, 1)) * (0.20 + 0.10 * np.sin(axis + 1))
    return means[component] + independent + common


def truth(d: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(d, dtype=float)
    a = 0.78 + 0.40 * ((axis % 5) / 4)
    b = 0.28 * np.cos(axis * 0.8 + 0.3)
    return a, b


def make_objective(method: str, x: np.ndarray, scale: np.ndarray, seed: int,
                   feature_reference: np.ndarray | None = None):
    """Return an objective that only receives corrected changed samples."""
    x = x / scale
    n, d = x.shape
    if method == "moments":
        mean_x = x.mean(axis=0)
        cov_x = np.cov(x, rowvar=False).reshape(d, d)

        def objective(y: np.ndarray) -> float:
            y = y / scale
            return float(np.mean((mean_x - y.mean(axis=0)) ** 2)
                         + np.mean((cov_x - np.cov(y, rowvar=False).reshape(d, d)) ** 2))

    elif method == "rff_mmd":
        rng = np.random.default_rng(seed + 10_000)
        bandwidth_source = x if feature_reference is None else feature_reference / scale
        median = float(np.median(pdist(bandwidth_source[: min(len(bandwidth_source), 256)])))
        bandwidth = max(median, 1e-6)
        omega = rng.normal(size=(d, 256)) / bandwidth
        phase = rng.uniform(0, 2 * np.pi, size=256)
        mean_feature_x = np.cos(x @ omega + phase).mean(axis=0)

        def objective(y: np.ndarray) -> float:
            mean_feature_y = np.cos((y / scale) @ omega + phase).mean(axis=0)
            return float(2 * np.mean((mean_feature_x - mean_feature_y) ** 2))

    elif method == "energy":
        xx = float(pdist(x).mean()) * (n - 1) / n

        def objective(y: np.ndarray) -> float:
            y = y / scale
            yy = float(pdist(y).mean()) * (len(y) - 1) / len(y)
            return float(2 * cdist(x, y).mean() - xx - yy)

    elif method == "sliced_w2":
        rng = np.random.default_rng(seed + 20_000)
        projections = rng.normal(size=(d, 48))
        projections /= np.linalg.norm(projections, axis=0)
        projected_x = np.sort(x @ projections, axis=0)

        def objective(y: np.ndarray) -> float:
            projected_y = np.sort((y / scale) @ projections, axis=0)
            return float(np.mean((projected_x - projected_y) ** 2))

    elif method == "sinkhorn":
        epsilon_source = x if feature_reference is None else feature_reference / scale
        epsilon = max(0.2 * float(np.median(pdist(epsilon_source) ** 2)), 1e-5)

        def entropic_ot(left: np.ndarray, right: np.ndarray) -> float:
            cost = cdist(left, right, metric="sqeuclidean")
            log_kernel = -cost / epsilon
            count_left, count_right = len(left), len(right)
            log_u = np.zeros(count_left)
            log_v = np.zeros(count_right)
            for _ in range(60):
                log_u = -np.log(count_left) - logsumexp(log_kernel + log_v[None, :], axis=1)
                log_v = -np.log(count_right) - logsumexp(log_kernel + log_u[:, None], axis=0)
            log_plan = log_u[:, None] + log_kernel + log_v[None, :]
            plan = np.exp(log_plan)
            entropy = np.sum(plan * (log_plan + np.log(count_left * count_right)))
            return float(np.sum(plan * cost) + epsilon * entropy)

        self_x = entropic_ot(x, x)

        def objective(y: np.ndarray) -> float:
            y = y / scale
            return entropic_ot(x, y) - 0.5 * self_x - 0.5 * entropic_ot(y, y)

    else:
        raise ValueError(method)
    return objective


def fit_case(case: str, method: str, seed: int, n: int, test_n: int,
             maxfev: int, noise_std: float = 0.0) -> dict:
    d, family = CASES[case]
    rng = np.random.default_rng(seed + 417)
    a_true, b_true = truth(d)
    if family == "shift":
        a_true = np.ones(d)
    elif family == "scale":
        b_true = np.zeros(d)
    # Separate source draws guarantee no event-level correspondence.
    x = source_sample(rng, n, d)
    y = source_sample(rng, n, d) * a_true + b_true
    x_test = source_sample(rng, test_n, d)
    y_test = source_sample(rng, test_n, d) * a_true + b_true
    if noise_std:
        y += rng.normal(scale=noise_std, size=y.shape)
        y_test += rng.normal(scale=noise_std, size=y_test.shape)
    scale = np.maximum(x.std(axis=0), 0.1)
    objective = make_objective(method, x, scale, seed)
    test_objective = make_objective(method, x_test, scale, seed, feature_reference=x)

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = params[:d] if family != "shift" else np.ones(d)
        b = params[-d:] if family != "scale" else np.zeros(d)
        return a, b

    def corrected(data: np.ndarray, params: np.ndarray) -> np.ndarray:
        a, b = unpack(params)
        return (data - b) / a

    bounds = []
    if family != "shift":
        bounds.extend([(0.55, 1.45)] * d)
    if family != "scale":
        bounds.extend([(-0.65, 0.65)] * d)
    init = np.array([1.0] * (d if family != "shift" else 0)
                    + [0.0] * (d if family != "scale" else 0))
    true_params = np.concatenate((a_true if family != "shift" else np.empty(0),
                                  b_true if family != "scale" else np.empty(0)))
    start = time.perf_counter()
    result = minimize(lambda p: objective(corrected(y, p)), init,
                      method="Powell", bounds=bounds,
                      options={"maxfev": maxfev, "xtol": 1e-4, "ftol": 1e-6})
    elapsed = time.perf_counter() - start
    estimate = np.asarray(result.x)
    return {
        "case": case, "method": method, "seed": seed, "noise_std": noise_std, "n_train": n,
        "n_test": test_n, "dimension": d, "parameter_count": len(init),
        "parameter_rmse": float(np.sqrt(np.mean((estimate - true_params) ** 2))),
        "train_objective": float(result.fun),
        "test_objective_initial": test_objective(corrected(y_test, init)),
        "test_objective_estimate": test_objective(corrected(y_test, estimate)),
        "test_objective_truth": test_objective(corrected(y_test, true_params)),
        "elapsed_seconds": elapsed, "success": bool(result.success),
        "optimizer_message": str(result.message), "function_evaluations": result.nfev,
        "true_parameters": json.dumps(true_params.tolist()),
        "estimated_parameters": json.dumps(estimate.tolist()),
    }


def rotation_identifiability(seed: int, n: int) -> list[dict]:
    """Profile a known nonidentifiable family against an identifiable source."""
    rng = np.random.default_rng(seed + 91_000)
    angle_true = 0.62
    angles = np.linspace(-np.pi, np.pi, 73)
    projection_rng = np.random.default_rng(seed + 92_000)
    directions = projection_rng.normal(size=(2, 64))
    directions /= np.linalg.norm(directions, axis=0)

    def rotate(points: np.ndarray, angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return points @ np.array([[c, -s], [s, c]]).T

    rows = []
    for source_name in ("isotropic_gaussian", "asymmetric_mixture"):
        if source_name == "isotropic_gaussian":
            x = rng.normal(size=(n, 2))
            z = rng.normal(size=(n, 2))
        else:
            x = source_sample(rng, n, 2)
            z = source_sample(rng, n, 2)
        y = rotate(z, angle_true)
        sorted_x = np.sort(x @ directions, axis=0)
        for candidate in angles:
            corrected = rotate(y, -candidate)
            score = np.mean((sorted_x - np.sort(corrected @ directions, axis=0)) ** 2)
            rows.append({"source": source_name, "seed": seed, "n": n,
                         "true_angle": angle_true, "candidate_angle": float(candidate),
                         "sliced_w2_squared": float(score)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=[method for method in METHODS if method != "sinkhorn"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--test-n", type=int, default=256)
    parser.add_argument("--maxfev", type=int, default=5000)
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="Extra Gaussian measurement noise in changed data only")
    parser.add_argument("--rotation-study", action="store_true",
                        help="Also profile rotation of isotropic and asymmetric distributions")
    parser.add_argument("--output", type=Path, default=Path("results/synthetic_pilot"))
    args = parser.parse_args()
    if min(args.n, args.test_n) < 3 or args.maxfev < 10 or args.noise_std < 0:
        parser.error("n and test-n must be >= 3, and maxfev must be >= 10")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in args.cases:
        for method in args.methods:
            for seed in args.seeds:
                row = fit_case(case, method, seed, args.n, args.test_n,
                               args.maxfev, args.noise_std)
                rows.append(row)
                print(f"{case:16} {method:10} seed={seed} "
                      f"RMSE={row['parameter_rmse']:.4f} "
                      f"success={row['success']} seconds={row['elapsed_seconds']:.2f}", flush=True)
    with (args.output / "runs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "command_options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source": "independent draws from an asymmetric three-component Gaussian mixture",
        "changed_model": "Y = a * Z + b, with Z independent of reference X",
        "correction": "T_(a,b)(Y) = (Y-b)/a",
        "optimizer": "scipy Powell with common bounds and evaluation budget",
        "objectives": {
            "moments": "squared mean and covariance differences",
            "rff_mmd": "squared RBF MMD estimated using 256 fixed random Fourier features",
            "energy": "biased empirical energy distance using Euclidean norm",
            "sliced_w2": "mean squared sorted-projection difference over 48 fixed directions",
            "sinkhorn": "debiased entropic OT with squared Euclidean cost, 60 log-domain iterations and fixed train-reference epsilon",
        },
        "limitations": "CPU pilot; WGAN and Sinkhorn are absent; no hyperparameter tuning or large-scale replication",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.rotation_study:
        rotation_rows = [row for seed in args.seeds
                         for row in rotation_identifiability(seed, args.test_n)]
        with (args.output / "rotation_profiles.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rotation_rows[0].keys())
            writer.writeheader()
            writer.writerows(rotation_rows)


if __name__ == "__main__":
    main()
