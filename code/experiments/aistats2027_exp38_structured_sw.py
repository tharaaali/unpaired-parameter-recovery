"""Joint event-level sliced-W2 recovery on all 16 exp38 prepared dataset pairs.

Fits a 2,048-cell correction on unpaired sparse event vectors. XY-neighbor
regularization was fixed from a separate earlier pilot, not tuned on exp38.
The exp38 original data-efficiency labels denote archive size; this CPU study
uses stated increasing training subsamples and disjoint 5k validation events.
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
from scipy.optimize import minimize

from aistats2027_joint_sliced_calo import (
    load_sparse_events, regularized_objective, select_rows, sliced_loss,
    xy_neighbor_edges,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/exp38_fixed_cell_aging_reproducible"
OUT = ROOT / "results/exp38_event_level"
FIT_OUT = OUT / "structured_sw"
TRAIN_BUDGET = {50_000:20_000, 90_000:30_000, 150_000:40_000, 250_000:50_000}
VAL_EVENTS = 5_000
PROJECTIONS = 32
EVAL_PROJECTIONS = 64
SMOOTHNESS = 1000.0
MAXITER = 300
REFINE_MAXITER = 700
SEED = 2027


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def directions(count: int, seed: int, n_cells: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = rng.normal(size=(n_cells,count)).astype(np.float32)
    output /= np.linalg.norm(output,axis=0)
    return output


def fit_one(record: dict, cells: list[dict], refine: bool = False) -> dict:
    size = int(record["events_per_state"])
    pair = record["data_seed_pair"]
    result_dir = BASE / record["run_name"]
    condition_out = FIT_OUT / f"{size}_{pair}"
    metrics_file = condition_out / "metrics.json"
    previous = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else None
    if previous is not None and (not refine or previous["success"]):
        return previous
    condition_out.mkdir(parents=True,exist_ok=True)
    start = time.perf_counter()
    cell_index = {int(row["cellid"]):i for i,row in enumerate(cells)}
    truth = np.array([float(row["true_aging_factor"]) for row in cells])
    if len(cells) != 2048 or len(cell_index) != len(cells):
        raise ValueError("Expected 2,048 unique active cells")
    saved_truth = np.load(result_dir / "real_aging_factors.npy")
    for i,row in enumerate(cells):
        indexed = float(saved_truth[int(row["z"]),int(row["x"]),int(row["y"])])
        if abs(indexed-truth[i]) > 1e-6:
            raise AssertionError(f"True field mismatch for cell {row['cellid']} in {record['run_name']}")

    reference, ref_duplicates = load_sparse_events(
        result_dir / "logs/train_prepared_new.zip",cell_index,"E_cal")
    aged, aged_duplicates = load_sparse_events(
        result_dir / "logs/train_prepared_old.zip",cell_index,"E_aged_cal")
    if ref_duplicates or aged_duplicates:
        raise ValueError("Prepared archive contains duplicate event-cell rows")
    n_train = TRAIN_BUDGET[size]
    ref_train, ref_val = select_rows(reference,SEED,n_train,VAL_EVENTS)
    aged_train, aged_val = select_rows(aged,SEED+1,n_train,VAL_EVENTS)
    fit_directions = directions(PROJECTIONS,SEED+2,len(cells))
    eval_directions = directions(EVAL_PROJECTIONS,SEED+3,len(cells))
    ref_sorted = np.sort(np.asarray(ref_train @ fit_directions),axis=0)
    edge_left,edge_right = xy_neighbor_edges(cells)
    if previous is None:
        initial = np.ones(len(cells))
    else:
        saved_predictions = read_csv(condition_out / "predictions.csv")
        if [row["cellid"] for row in saved_predictions] != [row["cellid"] for row in cells]:
            raise AssertionError("Saved prediction order differs from active-cell table")
        initial = np.array([float(row["predicted_aging_factor"]) for row in saved_predictions])
    iteration_budget = REFINE_MAXITER if previous is not None else MAXITER
    fit_start = time.perf_counter()
    result = minimize(regularized_objective,initial,
                      args=(ref_sorted,aged_train,fit_directions,10_000.0,
                            SMOOTHNESS,edge_left,edge_right),
                      jac=True,method="L-BFGS-B",bounds=[(.5,1.0)]*len(cells),
                      options={"maxiter":iteration_budget,"ftol":1e-10})
    fit_seconds = time.perf_counter()-fit_start
    estimate = np.asarray(result.x)
    errors = estimate-truth
    rmse=float(np.sqrt(np.mean(errors**2)))
    r2=float(1-np.sum(errors**2)/np.sum((truth-truth.mean())**2))
    initial_eval=sliced_loss(np.ones(len(cells)),ref_val,aged_val,eval_directions)
    fitted_eval=sliced_loss(estimate,ref_val,aged_val,eval_directions)
    truth_eval=sliced_loss(truth,ref_val,aged_val,eval_directions)
    metrics=dict(events_per_state=size,data_seed_pair=pair,run_name=record["run_name"],
                 method="Joint event-level sliced W2 + XY smoothness",
                 train_events_per_state=n_train,val_events_per_state=VAL_EVENTS,
                 reference_events_available=int(reference.shape[0]),
                 aged_events_available=int(aged.shape[0]),n_cells=len(cells),
                 projections=PROJECTIONS,independent_eval_projections=EVAL_PROJECTIONS,
                 smoothness=SMOOTHNESS,maxiter=iteration_budget,seed=SEED,
                 rmse=rmse,r2=r2,mae=float(np.mean(np.abs(errors))),
                 initial_val_sliced_w2_squared=initial_eval,
                 fitted_val_sliced_w2_squared=fitted_eval,
                 truth_val_sliced_w2_squared=truth_eval,
                 iterations=int(result.nit),function_evaluations=int(result.nfev),
                 success=bool(result.success),optimizer_message=str(result.message),
                 fit_seconds=fit_seconds,total_seconds=time.perf_counter()-start)
    if previous is not None:
        metrics["refinement_rounds"] = int(previous.get("refinement_rounds",0)) + 1
        metrics["previous_rmse"] = previous["rmse"]
        metrics["previous_val_sliced_w2_squared"] = previous["fitted_val_sliced_w2_squared"]
        metrics["iterations"] += previous["iterations"]
        metrics["function_evaluations"] += previous["function_evaluations"]
        metrics["maxiter"] += previous["maxiter"]
        metrics["fit_seconds"] += previous["fit_seconds"]
        metrics["total_seconds"] += previous["total_seconds"]
    with (condition_out / "predictions.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.writer(handle)
        writer.writerow(("cellid","z","x","y","true_aging_factor","predicted_aging_factor"))
        for row,pred in zip(cells,estimate):
            writer.writerow((row["cellid"],row["z"],row["x"],row["y"],
                             row["true_aging_factor"],float(pred)))
    metrics_file.write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    return metrics


def summarize(metrics: list[dict]) -> list[dict]:
    output=[]
    for size in TRAIN_BUDGET:
        group=[row for row in metrics if row["events_per_state"]==size]
        if len(group)!=4:
            raise ValueError(f"Expected four pairs at {size}: {len(group)}")
        rmse=np.array([row["rmse"] for row in group])
        r2=np.array([row["r2"] for row in group])
        eval_initial=np.array([row["initial_val_sliced_w2_squared"] for row in group])
        eval_error=np.array([row["fitted_val_sliced_w2_squared"] for row in group])
        eval_oracle=np.array([row["truth_val_sliced_w2_squared"] for row in group])
        output.append(dict(events_per_state=size,method=group[0]["method"],
                           train_events_per_state=TRAIN_BUDGET[size],
                           val_events_per_state=VAL_EVENTS,independent_pairs=len(group),
                           rmse_mean=float(rmse.mean()),rmse_std=float(rmse.std(ddof=1)),
                           r2_mean=float(r2.mean()),r2_std=float(r2.std(ddof=1)),
                           heldout_uncorrected_sliced_w2_squared_mean=float(eval_initial.mean()),
                           heldout_sliced_w2_squared_mean=float(eval_error.mean()),
                           heldout_oracle_sliced_w2_squared_mean=float(eval_oracle.mean()),
                           fit_seconds_mean=float(np.mean([row["fit_seconds"] for row in group])),
                           function_evaluations_mean=float(np.mean([row["function_evaluations"] for row in group])),
                           converged_fits=sum(row["success"] for row in group)))
    return output


def combine_table(summary: list[dict]) -> list[dict]:
    original=read_csv(BASE / "analysis/main_comparison_summary.csv")
    output=[]
    for size in TRAIN_BUDGET:
        for row in original:
            if int(row["events_per_state"])==size:
                output.append(dict(events_per_state=size,method=row["method"],
                                   train_events_per_state=(0 if row["method"]=="Global-mean predictor" else size),
                                   independent_pairs=int(row["independent_pairs"]),
                                   rmse_mean=float(row["rmse_mean"]),rmse_std=float(row["rmse_std"]),
                                   r2_mean=float(row["r2_mean"]),r2_std=float(row["r2_std"]),
                                   comparison_note=("Oracle constant equal to the injected field mean; no fitted events"
                                       if row["method"]=="Global-mean predictor" else
                                       "Published exp38 analysis using nominal archive; adversarial result averages model fits within pair")))
        sw=next(row for row in summary if row["events_per_state"]==size)
        output.append(dict(events_per_state=size,method=sw["method"],
                           train_events_per_state=sw["train_events_per_state"],
                           independent_pairs=4,rmse_mean=sw["rmse_mean"],
                           rmse_std=sw["rmse_std"],r2_mean=sw["r2_mean"],
                           r2_std=sw["r2_std"],
                           comparison_note="New CPU fit on one canonical prepared split per pair; increasing train subsamples"))
    return output


def plot(table: list[dict]) -> None:
    colors={"Global-mean predictor":"#777777","Mean-energy ratio":"#c77c42",
            "WS-only":"#9a6496","Wasserstein adversarial":"#4d80b1",
            "Joint event-level sliced W2 + XY smoothness":"#40936a"}
    fig,ax=plt.subplots(figsize=(7.4,4.4),constrained_layout=True)
    for method,color in colors.items():
        selected=sorted((row for row in table if row["method"]==method),
                        key=lambda row:row["events_per_state"])
        x=np.array([row["events_per_state"] for row in selected])/1000
        y=np.array([row["rmse_mean"] for row in selected])
        err=np.array([row["rmse_std"] for row in selected])
        label="Event-level SW2 (20–50k fit)" if method.startswith("Joint") else method
        ax.errorbar(x,y,yerr=err,marker="o",capsize=3,color=color,label=label)
    ax.set(xlabel="Events per state in prepared archive (thousands)",
           ylabel="Factor RMSE across 2,048 cells")
    ax.set_xticks(np.array(list(TRAIN_BUDGET))/1000)
    ax.grid(alpha=.2)
    ax.legend(frameon=False,fontsize=8)
    fig.savefig(OUT / "exp38_event_level_comparison.png",dpi=200)
    fig.savefig(OUT / "exp38_event_level_comparison.pdf")
    plt.close(fig)


def report(summary: list[dict], table: list[dict]) -> None:
    lines=["# Event-level sliced-W2 calibration on exp38", "",
           "The estimator fits one correction factor per active cell from **unpaired sparse event vectors**, using 32 random projections of complete events and the same quantile-transformed prepared archives as exp38. A squared-difference penalty connects XY-neighbor cells within each detector layer. Its weight 1000 was fixed from a separate earlier CPU pilot and was not selected on these exp38 outcomes. Candidate factors are constrained to [0.5,1]. No true factor is used by the fit; ground truth is read only for evaluation.", "",
           "For computational feasibility the new CPU method trains on increasing subsamples (20k/30k/40k/50k per state from archives labeled 50k/90k/150k/250k) and holds out 5k independent events per state. The archived fitted baselines used their full nominal archives; their compute and training sample counts are therefore **not matched** to this new baseline. The global-mean predictor is an infeasible oracle constant equal to the injected factor-field mean and fits no events. Each row aggregates four disjoint data-seed pairs; the adversarial baseline first averages its model-seed fits within each pair as in the original exp38 audit. All methods are scored against the same fixed 2,048-cell factor field.", "",
           "| Archive events/state | Method | Train events/state | RMSE mean ± pair SD | R² mean ± pair SD |",
           "|---:|---|---:|---:|---:|"]
    for row in table:
        budget="—" if row["method"]=="Global-mean predictor" else f"{row['train_events_per_state']:,}"
        lines.append(f"| {row['events_per_state']:,} | {row['method']} | {budget} | {row['rmse_mean']:.4f} ± {row['rmse_std']:.4f} | {row['r2_mean']:.3f} ± {row['r2_std']:.3f} |")
    lines += ["", "## New method's disjoint-event validation", "",
              "Held-out squared sliced-W2 uses 64 random directions independent of the 32 training directions. These values are descriptive and are not directly comparable with the published methods, for which a matched held-out objective was not saved.", "",
              "| Archive events/state | SW² uncorrected | SW² after correction | SW² at true factors | Mean fit seconds | Mean objective evaluations | Converged fits / 4 |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['events_per_state']:,} | {row['heldout_uncorrected_sliced_w2_squared_mean']:.3e} | {row['heldout_sliced_w2_squared_mean']:.3e} | {row['heldout_oracle_sliced_w2_squared_mean']:.3e} | {row['fit_seconds_mean']:.1f} | {row['function_evaluations_mean']:.1f} | {row['converged_fits']} |")
    lines += ["", "Raw per-pair results and 2,048 factor predictions for each fit are under `structured_sw/`. The combined CSV retains source-specific sample budgets and comparison caveats. The threshold audit is in `THRESHOLD_RESULTS.md`.", ""]
    (OUT / "STRUCTURED_SW_RESULTS.md").write_text("\n".join(lines),encoding="utf-8")


def main(summarize_only: bool, refine: bool = False) -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    canonical=read_csv(BASE / "analysis/canonical_prepared_runs.csv")
    cells=read_csv(BASE / "analysis/active_cells_and_occupancy.csv")
    metrics=[]
    for record in canonical:
        path=FIT_OUT / f"{record['events_per_state']}_{record['data_seed_pair']}" / "metrics.json"
        if summarize_only:
            metrics.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            metrics.append(fit_one(record,cells,refine=refine))
        print("fit",record["events_per_state"],record["data_seed_pair"],
              "rmse",f"{metrics[-1]['rmse']:.5f}",flush=True)
    summary=summarize(metrics)
    table=combine_table(summary)
    write_csv(OUT / "structured_sw_pair_metrics.csv",metrics)
    write_csv(OUT / "structured_sw_summary.csv",summary)
    write_csv(OUT / "exp38_combined_comparison.csv",table)
    plot(table)
    report(summary,table)
    (OUT / "structured_sw_metadata.json").write_text(json.dumps(dict(
        canonical_pairs=len(canonical),cells=len(cells),train_budgets=TRAIN_BUDGET,
        val_events=VAL_EVENTS,fit_projections=PROJECTIONS,
        independent_eval_projections=EVAL_PROJECTIONS,smoothness=SMOOTHNESS,
        initial_maxiter=MAXITER,additional_refine_maxiter=REFINE_MAXITER,
        all_fits_converged=all(row["success"] for row in metrics),seed=SEED,
        limitation="New baseline uses training subsets; published methods use full nominal archives"),indent=2),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize-only",action="store_true")
    parser.add_argument("--refine",action="store_true",help="Continue fits that hit the iteration limit")
    args=parser.parse_args()
    main(args.summarize_only,args.refine)
