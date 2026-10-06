"""Joint event-level sliced-W2 calorimeter factor estimator (CPU pilot).

Reads the same prepared quantile-domain event files and common active cells as
an existing WGAN run. Fits one multiplicative correction factor per cell from
unpaired event vectors. This script needs NumPy and SciPy, not PyTorch.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, csr_matrix


def load_sparse_events(path: Path, cell_index: dict[int, int],
                       value_column: str) -> tuple[csr_matrix, int]:
    event_index: dict[int, int] = {}
    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1:
            raise ValueError(f"Expected exactly one table in {path}")
        with archive.open(members[0]) as binary:
            reader = csv.DictReader((line.decode("utf-8") for line in binary))
            for record in reader:
                event = int(record["event"])
                if event not in event_index:
                    event_index[event] = len(event_index)
                cell = cell_index.get(int(record["cellid"]))
                if cell is None or not record[value_column]:
                    continue
                value = float(record[value_column])
                if not np.isfinite(value):
                    continue
                row_indices.append(event_index[event])
                col_indices.append(cell)
                values.append(value)
    matrix = coo_matrix((np.asarray(values, dtype=np.float32),
                         (row_indices, col_indices)),
                        shape=(len(event_index), len(cell_index))).tocsr()
    return matrix, len(values) - matrix.nnz


def select_rows(matrix: csr_matrix, seed: int, n_train: int,
                n_val: int) -> tuple[csr_matrix, csr_matrix]:
    if n_train + n_val > matrix.shape[0]:
        raise ValueError(f"Requested {n_train}+{n_val} events, found {matrix.shape[0]}")
    order = np.random.default_rng(seed).permutation(matrix.shape[0])
    return matrix[order[:n_train]], matrix[order[n_train:n_train + n_val]]


def sliced_loss_and_grad(a: np.ndarray, reference_sorted: np.ndarray,
                         aged: csr_matrix, directions: np.ndarray,
                         loss_scale: float) -> tuple[float, np.ndarray]:
    corrected_projection = np.asarray(aged @ (directions / a[:, None]))
    order = np.argsort(corrected_projection, axis=0)
    sorted_corrected = np.take_along_axis(corrected_projection, order, axis=0)
    residual = sorted_corrected - reference_sorted
    loss = loss_scale * float(np.mean(residual ** 2))
    derivative = np.empty_like(residual)
    columns = np.arange(residual.shape[1])[None, :]
    derivative[order, columns] = (2 * loss_scale / residual.size) * residual
    per_cell_projection = np.asarray(aged.T @ derivative)
    gradient = -np.sum(per_cell_projection * directions, axis=1) / (a ** 2)
    return loss, gradient.astype(float)


def sliced_loss(a: np.ndarray, reference: csr_matrix, aged: csr_matrix,
                directions: np.ndarray) -> float:
    projected_reference = np.sort(np.asarray(reference @ directions), axis=0)
    projected_aged = np.sort(np.asarray(aged @ (directions / a[:, None])), axis=0)
    return float(np.mean((projected_reference - projected_aged) ** 2))


def xy_neighbor_edges(cells: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    by_coord = {(int(row["z"]), int(row["x"]), int(row["y"])): i
                for i, row in enumerate(cells)}
    left, right = [], []
    for (z, x, y), i in by_coord.items():
        for neighbor in ((z, x + 1, y), (z, x, y + 1)):
            j = by_coord.get(neighbor)
            if j is not None:
                left.append(i)
                right.append(j)
    return np.asarray(left, dtype=int), np.asarray(right, dtype=int)


def regularized_objective(a: np.ndarray, reference_sorted: np.ndarray,
                          aged: csr_matrix, directions: np.ndarray,
                          loss_scale: float, smoothness: float,
                          left: np.ndarray, right: np.ndarray) -> tuple[float, np.ndarray]:
    loss, gradient = sliced_loss_and_grad(a, reference_sorted, aged, directions, loss_scale)
    if smoothness:
        delta = a[left] - a[right]
        loss += smoothness * float(np.mean(delta ** 2))
        edge_gradient = (2 * smoothness / len(left)) * delta
        np.add.at(gradient, left, edge_gradient)
        np.add.at(gradient, right, -edge_gradient)
    return loss, gradient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--train-events", type=int, default=20_000)
    parser.add_argument("--val-events", type=int, default=5_000)
    parser.add_argument("--projections", type=int, default=32)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--smoothness", type=float, default=0.0,
                        help="XY-neighbor squared-difference penalty in scaled objective units")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if min(args.train_events, args.val_events, args.projections, args.maxiter) < 1 or args.smoothness < 0:
        parser.error("Train/validation counts, projections and maxiter must be positive")
    output = args.output or Path("results/joint_sliced_calo") / args.result_dir.name
    with (args.result_dir / "mean_af_results" / "per_cell_mean_energy_ratio.csv").open(
            newline="", encoding="utf-8") as file:
        cells = list(csv.DictReader(file))
    cell_index = {int(row["cellid"]): i for i, row in enumerate(cells)}
    truth = np.array([float(row["true_aging_factor"]) for row in cells])
    start = time.perf_counter()
    reference, reference_duplicates = load_sparse_events(
        args.result_dir / "logs" / "train_prepared_new.zip", cell_index, "E_cal")
    aged, aged_duplicates = load_sparse_events(
        args.result_dir / "logs" / "train_prepared_old.zip", cell_index, "E_aged_cal")
    if reference_duplicates or aged_duplicates:
        raise ValueError("Duplicate event-cell rows would be summed; inspect input before comparing to WGAN")
    reference_train, reference_val = select_rows(reference, args.seed, args.train_events, args.val_events)
    aged_train, aged_val = select_rows(aged, args.seed + 1, args.train_events, args.val_events)
    rng = np.random.default_rng(args.seed + 2)
    directions = rng.normal(size=(len(cells), args.projections)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=0)
    reference_sorted = np.sort(np.asarray(reference_train @ directions), axis=0)
    edge_left, edge_right = xy_neighbor_edges(cells)
    initial = np.ones(len(cells), dtype=float)
    initial_train_loss = sliced_loss(initial, reference_train, aged_train, directions)
    initial_val_loss = sliced_loss(initial, reference_val, aged_val, directions)
    # The projection scale is small because each event has few active cells.
    loss_scale = 10_000.0
    optimization_start = time.perf_counter()
    fit = minimize(regularized_objective, initial,
                   args=(reference_sorted, aged_train, directions, loss_scale,
                         args.smoothness, edge_left, edge_right),
                   jac=True, method="L-BFGS-B", bounds=[(0.5, 1.0)] * len(cells),
                   options={"maxiter": args.maxiter, "ftol": 1e-10})
    optimization_seconds = time.perf_counter() - optimization_start
    estimate = np.asarray(fit.x)
    metrics = {
        "result_dir": str(args.result_dir), "train_events_per_state": args.train_events,
        "val_events_per_state": args.val_events, "n_cells": len(cells),
        "reference_events_available": reference.shape[0],
        "aged_events_available": aged.shape[0],
        "projections": args.projections, "seed": args.seed,
        "smoothness": args.smoothness, "xy_edges": len(edge_left),
        "maxiter": args.maxiter, "iterations": int(fit.nit),
        "function_evaluations": int(fit.nfev), "success": bool(fit.success),
        "optimizer_message": str(fit.message),
        "parameter_rmse": float(np.sqrt(np.mean((estimate - truth) ** 2))),
        "parameter_mae": float(np.mean(np.abs(estimate - truth))),
        "initial_train_sliced_w2_squared": initial_train_loss,
        "fitted_train_sliced_w2_squared": sliced_loss(estimate, reference_train, aged_train, directions),
        "initial_val_sliced_w2_squared": initial_val_loss,
        "fitted_val_sliced_w2_squared": sliced_loss(estimate, reference_val, aged_val, directions),
        "truth_val_sliced_w2_squared": sliced_loss(truth, reference_val, aged_val, directions),
        "optimization_seconds": optimization_seconds,
        "total_seconds": time.perf_counter() - start,
        "note": "Event-level joint SW baseline on quantile-transformed observed data; truth used only for evaluation",
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("cellid", "z", "x", "y", "true_aging_factor", "predicted_aging_factor"))
        for row, prediction in zip(cells, estimate):
            writer.writerow((row["cellid"], row["z"], row["x"], row["y"],
                             row["true_aging_factor"], prediction))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
