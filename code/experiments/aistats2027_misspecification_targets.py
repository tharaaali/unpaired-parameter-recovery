"""Discrepancy-specific pseudo-true parameters under an omitted nonlinear term."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import pdist

from aistats2027_nonlinear_transform_experiment import (
    METHODS, forward_transform, inverse_transform, make_objective, sample_asymmetric_mixture,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/misspecification_targets"
THETA = 0.5
BETAS = (0.0, 0.05, 0.1, 0.2)
DIMENSION = 3
MC_SIZE = 100_000
MC_PAIRS = 1_000_000
MC_REPLICATES = 2
FINITE_SEEDS = range(20)
N = 512
PROJECTION_SEED = 12345


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def large_mc_data(replicate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
    rng = np.random.default_rng(9_300_001 + replicate * 210_109)
    reference = sample_asymmetric_mixture(rng, MC_SIZE, DIMENSION)
    source = sample_asymmetric_mixture(rng, MC_SIZE, DIMENSION)
    scale = np.maximum(reference.std(axis=0, ddof=1), 0.1)
    ref_idx = rng.integers(MC_SIZE, size=MC_PAIRS)
    src_idx_1 = rng.integers(MC_SIZE, size=MC_PAIRS)
    src_idx_2 = rng.integers(MC_SIZE, size=MC_PAIRS)
    return reference, source, scale, (ref_idx, src_idx_1, src_idx_2)


def population_objective(method: str, reference: np.ndarray, observed: np.ndarray,
                         scale: np.ndarray, pairs: tuple):
    ref = reference / scale
    ref_idx, src_idx_1, src_idx_2 = pairs
    ref_pair = ref[ref_idx]
    if method == "Moments":
        target_mean = ref.mean(axis=0)
        target_cov = np.cov(ref, rowvar=False)

        def objective(theta: float) -> float:
            z = inverse_transform(observed, theta) / scale
            return float(np.mean((z.mean(axis=0)-target_mean)**2) +
                         np.mean((np.cov(z, rowvar=False)-target_cov)**2))

    elif method in ("MMD", "Energy"):
        if method == "MMD":
            bandwidth = max(float(np.median(pdist(ref[:2048]))), 1e-6)
            denominator = 2 * bandwidth**2

        def objective(theta: float) -> float:
            z = inverse_transform(observed, theta) / scale
            z1 = z[src_idx_1]
            z2 = z[src_idx_2]
            zz = np.sum((z1-z2)**2, axis=1)
            rz = np.sum((ref_pair-z1)**2, axis=1)
            if method == "MMD":
                return float(np.exp(-zz/denominator).mean() -
                             2*np.exp(-rz/denominator).mean())
            return float(2*np.sqrt(rz).mean() - np.sqrt(zz).mean())

    elif method == "Sliced W2":
        rng = np.random.default_rng(PROJECTION_SEED + 71_003)
        directions = rng.normal(size=(DIMENSION, 64))
        directions /= np.linalg.norm(directions, axis=0)
        # A large empirical quantile grid approximates the projected population criterion.
        ref_projected = np.sort(ref @ directions, axis=0)

        def objective(theta: float) -> float:
            z = inverse_transform(observed, theta) / scale
            return float(np.mean((ref_projected - np.sort(z @ directions, axis=0))**2))
    else:
        raise ValueError(method)
    return objective


def monte_carlo_targets() -> list[dict]:
    rows = []
    for replicate in range(MC_REPLICATES):
        reference, source, scale, pairs = large_mc_data(replicate)
        for beta in BETAS:
            observed = forward_transform(source, THETA) + beta*np.tanh(2*source)
            for method in METHODS:
                objective = population_objective(method, reference, observed, scale, pairs)
                result = minimize_scalar(objective, bounds=(0, 1.2), method="bounded",
                                         options={"xatol": 1e-5, "maxiter": 80})
                rows.append(dict(mc_replicate=replicate, beta=beta, method=method,
                                 pseudo_true_theta=float(result.x), objective_minimum=float(result.fun),
                                 objective_evaluations=int(result.nfev), success=bool(result.success)))
                print("target", replicate, beta, method, f"{result.x:.5f}", flush=True)
    return rows


def finite_sample_fits() -> list[dict]:
    rows = []
    for seed in FINITE_SEEDS:
        rng = np.random.default_rng(3_700_001 + seed*1009)
        reference = sample_asymmetric_mixture(rng, N, DIMENSION)
        source = sample_asymmetric_mixture(rng, N, DIMENSION)
        scale = np.maximum(reference.std(axis=0, ddof=1), .1)
        for beta in BETAS:
            observed = forward_transform(source, THETA) + beta*np.tanh(2*source)
            for method in METHODS:
                objective, _ = make_objective(method, reference, observed, scale, PROJECTION_SEED)
                result = minimize_scalar(objective, bounds=(0, 1.2), method="bounded",
                                         options={"xatol": 1e-5, "maxiter": 80})
                rows.append(dict(seed=seed, beta=beta, method=method, theta_hat=float(result.x),
                                 error_from_injected_theta=float(result.x-THETA),
                                 objective_evaluations=int(result.nfev), success=bool(result.success)))
        print("finite", seed, flush=True)
    return rows


def summarize(targets: list[dict], finite: list[dict]) -> list[dict]:
    rows = []
    for method in METHODS:
        for beta in BETAS:
            target_values = np.array([float(r["pseudo_true_theta"]) for r in targets
                                      if r["method"] == method and r["beta"] == beta])
            estimates = np.array([float(r["theta_hat"]) for r in finite
                                  if r["method"] == method and r["beta"] == beta])
            rng = np.random.default_rng(8_900_001 + 100*METHODS.index(method) + BETAS.index(beta))
            idx = rng.integers(len(estimates), size=(10000, len(estimates)))
            means = estimates[idx].mean(axis=1)
            target = float(target_values.mean())
            rows.append(dict(method=method, beta=beta, injected_theta=THETA,
                             pseudo_true_theta=target,
                             mc_replicate_0=float(target_values[0]),
                             mc_replicate_1=float(target_values[1]),
                             mc_replicate_disagreement=float(abs(target_values[1]-target_values[0])),
                             finite_replicates=len(estimates), finite_mean_theta=float(estimates.mean()),
                             finite_mean_ci_low=float(np.quantile(means, .025)),
                             finite_mean_ci_high=float(np.quantile(means, .975)),
                             finite_sd_theta=float(estimates.std(ddof=1)),
                             finite_rmse_around_pseudo_true=float(np.sqrt(np.mean((estimates-target)**2))),
                             pseudo_true_shift_from_injected=float(target-THETA)))
    lookup = {(r["method"], r["beta"]): r for r in rows}
    mc_lookup = {(r["method"], float(r["beta"]), int(r["mc_replicate"])):
                 float(r["pseudo_true_theta"]) for r in targets}
    for row in rows:
        method, beta = row["method"], row["beta"]
        row["target_shift_from_beta0"] = row["pseudo_true_theta"] - lookup[(method, 0.0)]["pseudo_true_theta"]
        row["excess_target_shift_vs_moments"] = (
            row["target_shift_from_beta0"] -
            lookup[("Moments", beta)]["pseudo_true_theta"] +
            lookup[("Moments", 0.0)]["pseudo_true_theta"])
        replicate_contrasts = [
            (mc_lookup[(method,beta,k)]-mc_lookup[(method,0.0,k)]) -
            (mc_lookup[("Moments",beta,k)]-mc_lookup[("Moments",0.0,k)])
            for k in range(MC_REPLICATES)]
        row["excess_shift_mc_replicate_disagreement"] = float(abs(replicate_contrasts[1]-replicate_contrasts[0]))
    return rows


def figure(summary: list[dict], contrasts: list[dict]) -> None:
    colors = {"Moments":"#4776b9", "MMD":"#d27b3d", "Energy":"#528b68", "Sliced W2":"#985791"}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    for method in METHODS:
        group = [r for r in summary if r["method"] == method]
        x = np.array(BETAS)
        targets = np.array([r["pseudo_true_theta"] for r in group])
        means = np.array([r["finite_mean_theta"] for r in group])
        low = np.array([r["finite_mean_ci_low"] for r in group])
        high = np.array([r["finite_mean_ci_high"] for r in group])
        axes[0].plot(x, targets, "o-", color=colors[method], label=method)
        axes[1].errorbar(x, means, yerr=[means-low, high-means], fmt="o-", capsize=2,
                         color=colors[method], label=method)
        axes[1].plot(x, targets, ":", color=colors[method], alpha=.6)
    for ax in axes[:2]:
        ax.axhline(THETA, color="black", ls="--", lw=1)
        ax.set(xlabel=r"Omitted-term strength $\beta$", ylabel=r"Fitted parameter $\theta$", xticks=BETAS)
        ax.grid(alpha=.2)
    axes[0].set_title("A  Large-MC pseudo-true targets")
    axes[1].set_title("B  Finite-sample means (20 runs)")
    axes[0].legend(frameon=False, fontsize=8)
    for method in METHODS[1:]:
        group = [r for r in contrasts if r["method"] == method]
        x = np.array([r["beta"] for r in group])
        means = np.array([r["finite_mean_excess_shift"] for r in group])
        low = np.array([r["finite_excess_shift_ci_low"] for r in group])
        high = np.array([r["finite_excess_shift_ci_high"] for r in group])
        axes[2].errorbar(x, means, yerr=[means-low, high-means], fmt="o-", capsize=2,
                         color=colors[method], label=method)
        axes[2].plot(x, [r["mc_mean_excess_shift"] for r in group], ":", color=colors[method], alpha=.65)
    axes[2].axhline(0, color="black", ls="--", lw=1)
    axes[2].set(xlabel=r"Omitted-term strength $\beta$", ylabel="Excess shift vs Moments", xticks=BETAS[1:])
    axes[2].set_title(r"C  Paired shifts from $\beta=0$")
    axes[2].grid(alpha=.2)
    fig.savefig(OUT / "pseudo_true_trajectories.png", dpi=200)
    fig.savefig(OUT / "pseudo_true_trajectories.pdf")
    plt.close(fig)


def paired_shift_contrasts(targets: list[dict], finite: list[dict]) -> list[dict]:
    fitted = {(r["method"], float(r["beta"]), int(r["seed"])):float(r["theta_hat"]) for r in finite}
    target = {(r["method"], float(r["beta"]), int(r["mc_replicate"])):
              float(r["pseudo_true_theta"]) for r in targets}
    rows = []
    for method in METHODS[1:]:
        for beta in BETAS[1:]:
            values = np.array([
                (fitted[(method,beta,seed)]-fitted[(method,0.0,seed)]) -
                (fitted[("Moments",beta,seed)]-fitted[("Moments",0.0,seed)])
                for seed in FINITE_SEEDS])
            mc_values = np.array([
                (target[(method,beta,k)]-target[(method,0.0,k)]) -
                (target[("Moments",beta,k)]-target[("Moments",0.0,k)])
                for k in range(MC_REPLICATES)])
            rng = np.random.default_rng(7_200_001 + 100*METHODS.index(method) + BETAS.index(beta))
            idx = rng.integers(len(values), size=(20000, len(values)))
            boot = values[idx].mean(axis=1)
            rows.append(dict(method=method, beta=beta, finite_replicates=len(values),
                             finite_mean_excess_shift=float(values.mean()),
                             finite_excess_shift_ci_low=float(np.quantile(boot,.025)),
                             finite_excess_shift_ci_high=float(np.quantile(boot,.975)),
                             finite_excess_shift_sd=float(values.std(ddof=1)),
                             mc_mean_excess_shift=float(mc_values.mean()),
                             mc_excess_shift_replicate_difference=float(abs(mc_values[1]-mc_values[0]))))
    return rows


def run(summarize_only: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if summarize_only:
        with (OUT / "mc_targets.csv").open(newline="", encoding="utf-8") as handle:
            targets = [{**r, "beta":float(r["beta"])} for r in csv.DictReader(handle)]
        with (OUT / "finite_fits.csv").open(newline="", encoding="utf-8") as handle:
            finite = [{**r, "beta":float(r["beta"])} for r in csv.DictReader(handle)]
    else:
        targets = monte_carlo_targets()
        write_csv(OUT / "mc_targets.csv", targets)
        finite = finite_sample_fits()
        write_csv(OUT / "finite_fits.csv", finite)
    summary = summarize(targets, finite)
    write_csv(OUT / "summary.csv", summary)
    contrasts = paired_shift_contrasts(targets, finite)
    write_csv(OUT / "paired_shift_contrasts.csv", contrasts)
    figure(summary, contrasts)
    lines = ["# Discrepancy-dependent targets under omitted nonlinear structure", "",
             "The true changed-data generator is `Y = X' + 0.5 tanh(X') + beta tanh(2X')`; fitting uses only `g_theta(x)=x+theta tanh(x)` and its inverse. Four discrepancies share the same source/reference arrays and bounded scalar optimizer. Thus beta=0 is correctly specified and beta>0 is structurally misspecified. There is no observation noise in this audit.", "",
             f"Pseudo-true targets are independently approximated twice with {MC_SIZE:,} source and reference observations each. MMD and Energy use {MC_PAIRS:,} fixed random pair draws per replicate; Sliced W2 uses 64 fixed projections shared with the finite-sample study. The average target is plotted, and replicate disagreement measures Monte Carlo resolution. Finite-sample means use {len(FINITE_SEEDS)} paired datasets at n={N} per distribution. Error bars are bootstrap intervals for the mean, not for an individual fit.", "",
             "| Method | beta | Pseudo-true theta | MC replicate difference | Finite mean [95% bootstrap] | Finite RMSE around pseudo-true |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['method']} | {r['beta']:.2f} | {r['pseudo_true_theta']:.4f} | {r['mc_replicate_disagreement']:.4f} | {r['finite_mean_theta']:.4f} [{r['finite_mean_ci_low']:.4f}, {r['finite_mean_ci_high']:.4f}] | {r['finite_rmse_around_pseudo_true']:.4f} |")
    lines += ["", "## Target shifts relative to beta=0", "",
              "Using the same Monte Carlo draws at all beta values cancels most target noise. The second column is the target shift from beta=0 to beta=0.2; the third subtracts the Moments shift. The final column is disagreement between the two independent Monte Carlo estimates of that difference-in-differences.", "",
              "| Method | Target shift, beta 0 to 0.2 | Excess shift vs Moments | MC disagreement in excess shift |",
              "|---|---:|---:|---:|"]
    for r in summary:
        if r["beta"] == 0.2:
            lines.append(f"| {r['method']} | {r['target_shift_from_beta0']:.5f} | {r['excess_target_shift_vs_moments']:+.5f} | {r['excess_shift_mc_replicate_disagreement']:.5f} |")
    lines += ["", "## Paired finite-sample contrasts", "",
              "For each finite dataset, the contrast subtracts the method's beta=0 estimate and the Moments shift on that same dataset. Confidence intervals resample paired datasets, preserving all method and beta values within a seed.", "",
              "| Method | beta | Finite excess shift vs Moments [95% paired CI] | Large-MC excess shift |",
              "|---|---:|---:|---:|"]
    for r in contrasts:
        lines.append(f"| {r['method']} | {r['beta']:.2f} | {r['finite_mean_excess_shift']:+.5f} [{r['finite_excess_shift_ci_low']:+.5f}, {r['finite_excess_shift_ci_high']:+.5f}] | {r['mc_mean_excess_shift']:+.5f} |")
    lines += ["", "The Energy and MMD excess shifts exceed the corresponding Monte Carlo disagreement. Marginal finite-sample mean intervals overlap, but the paired shift contrasts resolve the small method differences because the same datasets are used throughout. The plotted finite-sample target markers are large-MC approximations rather than known analytic values.", ""]
    (OUT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "metadata.json").write_text(json.dumps(dict(injected_theta=THETA, betas=BETAS,
                        mc_size_per_domain=MC_SIZE, mc_pairs=MC_PAIRS, mc_replicates=MC_REPLICATES,
                        finite_sample_size_per_domain=N, finite_seeds=list(FINITE_SEEDS),
                        projection_seed=PROJECTION_SEED, noise_std=0), indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize-only", action="store_true", help="Reuse saved raw fits")
    run(parser.parse_args().summarize_only)
