"""Continuous rotation-identifiability audit for the AISTATS manuscript.

The covariance criterion has population form 2*rho**2*sin(theta-theta*)**2.
Angles are identified modulo pi for every centered anisotropic Gaussian.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/rotation_separation"
RHOS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
THETA = 0.62
N = 512
SEEDS = range(50)
EPSILON = 0.002


def rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def periodic_error(estimate: float) -> float:
    return float((estimate - THETA + np.pi / 2) % np.pi - np.pi / 2)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_rmse(errors: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(errors), size=(10000, len(errors)))
    values = np.sqrt(np.mean(errors[indices] ** 2, axis=1))
    return tuple(float(v) for v in np.quantile(values, (0.025, 0.975)))


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    grid = np.linspace(-np.pi / 2, np.pi / 2, 721, endpoint=False)
    for rho in RHOS:
        for seed in SEEDS:
            rng = np.random.default_rng(1_800_001 + seed * 1009 + RHOS.index(rho) * 100_003)
            scales = np.array([1.0, np.sqrt(1.0 + rho)])
            reference = rng.normal(size=(N, 2)) * scales
            changed_source = rng.normal(size=(N, 2)) * scales
            observed = changed_source @ rotation(THETA).T
            ref_cov = np.cov(reference, rowvar=False)
            obs_cov = np.cov(observed, rowvar=False)

            def objective(angle: float) -> float:
                corrected_cov = rotation(-angle) @ obs_cov @ rotation(-angle).T
                return float(np.sum((ref_cov - corrected_cov) ** 2))

            grid_values = np.array([objective(float(v)) for v in grid])
            best = int(np.argmin(grid_values))
            step = float(grid[1] - grid[0])
            result = minimize_scalar(objective, bounds=(float(grid[best] - step),
                                                       float(grid[best] + step)),
                                     method="bounded", options={"xatol": 1e-10})
            estimate = float((result.x + np.pi / 2) % np.pi - np.pi / 2)
            offsets = np.linspace(-0.04, 0.04, 13)
            fit = np.polynomial.polynomial.polyfit(
                offsets, [objective(THETA + float(t)) for t in offsets], 2)
            rows.append(dict(rho=rho, seed=seed, theta_true=THETA,
                             theta_estimate=estimate, error_mod_pi=periodic_error(estimate),
                             absolute_error_mod_pi=abs(periodic_error(estimate)),
                             empirical_curvature_truth=float(2 * fit[2]),
                             population_curvature=4 * rho**2,
                             objective_at_truth=objective(THETA),
                             objective_at_estimate=float(result.fun),
                             evaluations=int(len(grid) + result.nfev), success=bool(result.success)))
        print(f"rho={rho}: {len(SEEDS)} fits", flush=True)
    write_csv(OUT / "runs.csv", rows)

    summary: list[dict] = []
    for i, rho in enumerate(RHOS):
        group = [row for row in rows if row["rho"] == rho]
        errors = np.array([row["error_mod_pi"] for row in group])
        empirical_h = np.array([row["empirical_curvature_truth"] for row in group])
        ci = bootstrap_rmse(errors, 6_000 + i)
        if rho == 0 or 2 * rho**2 <= EPSILON:
            width = np.pi
        else:
            width = 2 * np.arcsin(np.sqrt(EPSILON / (2 * rho**2)))
        summary.append(dict(rho=rho, n=N, replicates=len(group),
                            parameter_rmse_mod_pi=float(np.sqrt(np.mean(errors**2))),
                            parameter_rmse_ci_low=ci[0], parameter_rmse_ci_high=ci[1],
                            mean_absolute_error_mod_pi=float(np.mean(np.abs(errors))),
                            mean_empirical_curvature=float(empirical_h.mean()),
                            empirical_curvature_sd=float(empirical_h.std(ddof=1)),
                            population_curvature=4 * rho**2,
                            near_optimal_width_radians=float(width),
                            near_optimal_threshold=EPSILON))
    write_csv(OUT / "summary.csv", summary)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.35), constrained_layout=True)
    x = np.arange(len(RHOS))
    rmse = np.array([r["parameter_rmse_mod_pi"] for r in summary])
    low = np.array([r["parameter_rmse_ci_low"] for r in summary])
    high = np.array([r["parameter_rmse_ci_high"] for r in summary])
    axes[0].errorbar(x, rmse, yerr=[rmse-low, high-rmse], fmt="o-", capsize=3)
    axes[0].set(xlabel=r"Anisotropy $\rho$", ylabel=r"Angle RMSE modulo $\pi$ (rad)", title="A  Parameter recovery")
    axes[1].plot(x, [r["population_curvature"] for r in summary], "o-", label=r"Population $4\rho^2$")
    axes[1].plot(x, [r["mean_empirical_curvature"] for r in summary], "s--", label="Mean empirical")
    axes[1].set(xlabel=r"Anisotropy $\rho$", ylabel="Local curvature", title="B  Objective separation")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].plot(x, [r["near_optimal_width_radians"] for r in summary], "o-", color="#905095")
    axes[2].set(xlabel=r"Anisotropy $\rho$", ylabel="Near-optimal width (rad)", title="C  Population ambiguity")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_xticks(x, [f"{rho:g}" for rho in RHOS])
    fig.savefig(OUT / "separation_sweep.png", dpi=200)
    fig.savefig(OUT / "separation_sweep.pdf")
    plt.close(fig)

    lines = ["# Continuous rotation-identifiability sweep", "",
             "Independent 2D Gaussian samples have covariance `diag(1, 1+rho)`. The changed sample is rotated by 0.62 radians, and a covariance-matching correction estimates its inverse angle from 512 unpaired observations per domain. There are 50 paired dataset replicates for each anisotropy. Errors use circular distance modulo pi, because centered Gaussian covariance has an exact 180-degree ambiguity even when rho>0.", "",
             "The population squared Frobenius covariance discrepancy is `2 rho² sin²(theta-theta*)`; its local curvature is `4 rho²`. The reported near-optimal width is the full circular arc where excess population discrepancy is below 0.002.", "",
             "| rho | RMSE modulo pi [95% bootstrap] | Mean empirical curvature | Population curvature | Near-optimal width (rad) |",
             "|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['rho']:.2f} | {r['parameter_rmse_mod_pi']:.3f} [{r['parameter_rmse_ci_low']:.3f}, {r['parameter_rmse_ci_high']:.3f}] | {r['mean_empirical_curvature']:.4f} | {r['population_curvature']:.4f} | {r['near_optimal_width_radians']:.3f} |")
    lines += ["", "At rho=0 the population distribution is rotation invariant, so error has no meaningful identifiable target. Increasing rho improves separation, but finite samples can still dominate when anisotropy is weak.", ""]
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "metadata.json").write_text(json.dumps(dict(theta_true=THETA, sample_size=N, seeds=list(SEEDS),
                                                         rhos=RHOS, near_optimal_threshold=EPSILON,
                                                         criterion="squared Frobenius covariance discrepancy",
                                                         angle_equivalence="modulo pi"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
