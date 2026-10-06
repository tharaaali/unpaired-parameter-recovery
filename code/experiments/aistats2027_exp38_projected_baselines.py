"""Projected moments, RFF-MMD, and energy calibration on canonical exp38 pairs.

All three estimators use the same unpaired event splits, 32 event projections,
2,048 bounded cell factors, and XY-neighbor penalty as the saved sliced-W2
study. The raw data objectives are normalized to 0.2 at the identity map so
the fixed smoothness weight has a stated, comparable initial scale. No factor
truth is used in fitting or hyperparameter selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
try:
    from numba import njit
except ImportError:
    njit = None

from aistats2027_joint_sliced_calo import (
    load_sparse_events, select_rows, sliced_loss, xy_neighbor_edges,
)
from aistats2027_exp38_structured_sw import (
    BASE, OUT, TRAIN_BUDGET, VAL_EVENTS, PROJECTIONS, EVAL_PROJECTIONS,
    SMOOTHNESS, SEED, directions, read_csv, write_csv,
)

METHODS = ("moments", "mmd", "energy")
LABELS = {
    "moments": "Projected event moments + XY smoothness",
    "mmd": "Projected RFF-MMD + XY smoothness",
    "energy": "Sliced energy distance + XY smoothness",
}
FIT_OUT = OUT / "projected_baselines"
MAXITER = 300
REFINE_MAXITER = 300
RFF_PER_PROJECTION = 4


def moment_loss(z: np.ndarray, reference: np.ndarray, scales: tuple[np.ndarray, np.ndarray]) -> tuple[float, np.ndarray]:
    n, k = z.shape
    first_ref, second_ref = reference
    first_scale, second_scale = scales
    d1 = z.mean(axis=0) - first_ref
    d2 = np.mean(z*z, axis=0) - second_ref
    loss = float(np.mean((d1/first_scale)**2 + (d2/second_scale)**2))
    dz = (2*d1/first_scale**2 + 4*z*d2/second_scale**2)/(n*k)
    return loss, dz


def rff_mmd_loss(z: np.ndarray, reference: tuple[np.ndarray, np.ndarray], freqs: np.ndarray) -> tuple[float, np.ndarray]:
    n, k = z.shape
    cos_ref, sin_ref = reference
    angle = z[:,:,None] * freqs[None,:,:]
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    dc = cos_angle.mean(axis=0) - cos_ref
    ds = sin_angle.mean(axis=0) - sin_ref
    loss = float(np.mean(dc*dc + ds*ds))
    dz = np.sum((-dc[None,:,:]*sin_angle + ds[None,:,:]*cos_angle)
                * freqs[None,:,:], axis=2) * (2/(n*freqs.size))
    return loss, dz


def energy_loss(z: np.ndarray, reference_sorted: np.ndarray,
                reference_self: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean projected 1D energy distance and an exact a.e. subgradient."""
    n, k = z.shape
    loss = 0.0
    dz = np.empty_like(z)
    ranks = np.arange(n)
    for j in range(k):
        x = reference_sorted[:,j]
        y = z[:,j]
        order = np.argsort(y,kind="stable")
        ys = y[order]
        prefix = np.r_[0.0,np.cumsum(x)]
        count = np.searchsorted(x,y,side="right")
        cross = np.sum(y*(2*count-n) + prefix[n] - 2*prefix[count])
        self_y = 2*np.dot(2*ranks-n+1,ys)
        loss += 2*cross/n**2 - reference_self[j] - self_y/n**2
        # The all-pairs within-Y derivative has weight 2/n^2.
        sorted_grad = 2*(2*count[order]-n)/n**2 - 2*(2*ranks-n+1)/n**2
        dz[order,j] = sorted_grad/k
    return float(loss/k), dz


if njit is not None:
    @njit(cache=True)
    def _energy_loss_fast(z: np.ndarray, reference_sorted_t: np.ndarray,
                          reference_prefix_t: np.ndarray,
                          reference_self: np.ndarray) -> tuple[float, np.ndarray]:
        n,k=z.shape
        inv=1.0/(n*n)
        dz=np.empty((n,k),dtype=np.float64)
        total=0.0
        for col in range(k):
            xs=reference_sorted_t[col]
            prefix=reference_prefix_t[col]
            order=np.argsort(z[:,col])
            cross=0.0
            within=0.0
            for i in range(n):
                value=z[i,col]
                lo=0
                hi=n
                while lo<hi:
                    mid=(lo+hi)//2
                    if xs[mid]<=value:
                        lo=mid+1
                    else:
                        hi=mid
                count=lo
                cross+=value*(2*count-n)+prefix[n]-2*prefix[count]
                dz[i,col]=2*(2*count-n)*inv/k
            for rank in range(n):
                index=order[rank]
                weight=2*rank-n+1
                within+=2*weight*z[index,col]
                dz[index,col]-=2*weight*inv/k
            total+=2*cross*inv-reference_self[col]-within*inv
        return total/k,dz


def make_context(method: str, reference_z: np.ndarray, initial_z: np.ndarray) -> dict:
    if method == "moments":
        scales = (np.maximum(reference_z.std(axis=0),1e-7),
                  np.maximum((reference_z**2).std(axis=0),1e-7))
        reference = (reference_z.mean(axis=0),np.mean(reference_z**2,axis=0))
        raw,_ = moment_loss(initial_z,reference,scales)
        return dict(method=method,reference=reference,scales=scales,raw_initial=raw)
    if method == "mmd":
        rng = np.random.default_rng(SEED+4)
        n,k=reference_z.shape
        i=rng.integers(0,n,size=(2048,k))
        j=rng.integers(0,n,size=(2048,k))
        differences=np.abs(np.take_along_axis(reference_z,i,axis=0)-
                           np.take_along_axis(reference_z,j,axis=0))
        bandwidth=np.maximum(np.median(differences,axis=0),1e-6)
        freqs=rng.normal(size=(k,RFF_PER_PROJECTION))/bandwidth[:,None]
        angle=reference_z[:,:,None]*freqs[None,:,:]
        reference=(np.cos(angle).mean(axis=0),np.sin(angle).mean(axis=0))
        raw,_=rff_mmd_loss(initial_z,reference,freqs)
        return dict(method=method,reference=reference,freqs=freqs,
                    bandwidth=bandwidth,raw_initial=raw)
    if method == "energy":
        x=np.sort(reference_z,axis=0)
        n=len(x)
        reference_self=np.array([2*np.dot(2*np.arange(n)-n+1,x[:,j])/n**2
                                 for j in range(x.shape[1])])
        x_t=np.ascontiguousarray(x.T)
        prefix_t=np.concatenate((np.zeros((x_t.shape[0],1)),np.cumsum(x_t,axis=1)),axis=1)
        if njit is not None:
            raw,_=_energy_loss_fast(initial_z,x_t,prefix_t,reference_self)
        else:
            raw,_=energy_loss(initial_z,x,reference_self)
        return dict(method=method,reference=x,reference_self=reference_self,
                    reference_t=x_t,reference_prefix_t=prefix_t,
                    raw_initial=raw)
    raise ValueError(method)


def raw_loss(z: np.ndarray, context: dict) -> tuple[float,np.ndarray]:
    method=context["method"]
    if method=="moments":
        return moment_loss(z,context["reference"],context["scales"])
    if method=="mmd":
        return rff_mmd_loss(z,context["reference"],context["freqs"])
    if method=="energy" and njit is not None:
        return _energy_loss_fast(z,context["reference_t"],
                                 context["reference_prefix_t"],context["reference_self"])
    return energy_loss(z,context["reference"],context["reference_self"])


def objective(a: np.ndarray, aged, fit_directions: np.ndarray, context: dict,
              scale: float, left: np.ndarray, right: np.ndarray) -> tuple[float,np.ndarray]:
    z=np.asarray(aged @ (fit_directions/a[:,None]),dtype=np.float64)
    loss,dz=raw_loss(z,context)
    gradient=-np.sum(np.asarray(aged.T @ dz)*fit_directions,axis=1)/a**2
    delta=a[left]-a[right]
    loss=scale*loss + SMOOTHNESS*float(np.mean(delta**2))
    gradient=scale*gradient
    edge_gradient=(2*SMOOTHNESS/len(left))*delta
    np.add.at(gradient,left,edge_gradient)
    np.add.at(gradient,right,-edge_gradient)
    return loss,gradient


def fit_method(method: str, record: dict, cells: list[dict], reference, aged,
               truth: np.ndarray, refine: bool) -> dict:
    size=int(record["events_per_state"])
    pair=record["data_seed_pair"]
    dest=FIT_OUT/f"{size}_{pair}"/method
    metrics_path=dest/"metrics.json"
    previous=json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    if previous is not None and (not refine or previous["success"]):
        return previous
    dest.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    ntrain=TRAIN_BUDGET[size]
    ref_train,ref_val=select_rows(reference,SEED,ntrain,VAL_EVENTS)
    aged_train,aged_val=select_rows(aged,SEED+1,ntrain,VAL_EVENTS)
    fit_directions=directions(PROJECTIONS,SEED+2,len(cells))
    eval_directions=directions(EVAL_PROJECTIONS,SEED+3,len(cells))
    reference_z=np.asarray(ref_train@fit_directions,dtype=np.float64)
    initial_z=np.asarray(aged_train@fit_directions,dtype=np.float64)
    context=make_context(method,reference_z,initial_z)
    scale=0.2/max(context["raw_initial"],1e-12)
    left,right=xy_neighbor_edges(cells)
    if previous:
        saved=read_csv(dest/"predictions.csv")
        if [r["cellid"] for r in saved]!=[r["cellid"] for r in cells]:
            raise AssertionError("Saved factor order mismatch")
        start=np.array([float(r["predicted_aging_factor"]) for r in saved])
    else:
        start=np.ones(len(cells))
    fit_started=time.perf_counter()
    result=minimize(objective,start,args=(aged_train,fit_directions,context,
                                        scale,left,right),jac=True,
                    method="L-BFGS-B",bounds=[(.5,1.)]*len(cells),
                    options={"maxiter":REFINE_MAXITER if previous else MAXITER,
                             "ftol":1e-8 if previous else 1e-10})
    fit_seconds=time.perf_counter()-fit_started
    estimate=np.asarray(result.x)
    error=estimate-truth
    metrics=dict(events_per_state=size,data_seed_pair=pair,run_name=record["run_name"],
                 method=LABELS[method],method_key=method,train_events_per_state=ntrain,
                 validation_events_per_state=VAL_EVENTS,n_cells=len(cells),
                 fit_projections=PROJECTIONS,eval_projections=EVAL_PROJECTIONS,
                 smoothness=SMOOTHNESS,initial_raw_loss=context["raw_initial"],
                 data_loss_scale=scale,
                 rmse=float(np.sqrt(np.mean(error**2))),
                 mae=float(np.mean(np.abs(error))),
                 r2=float(1-np.sum(error**2)/np.sum((truth-truth.mean())**2)),
                 initial_val_sliced_w2_squared=sliced_loss(np.ones(len(cells)),ref_val,aged_val,eval_directions),
                 fitted_val_sliced_w2_squared=sliced_loss(estimate,ref_val,aged_val,eval_directions),
                 truth_val_sliced_w2_squared=sliced_loss(truth,ref_val,aged_val,eval_directions),
                 iterations=int(result.nit),function_evaluations=int(result.nfev),
                 success=bool(result.success),optimizer_message=str(result.message),
                 fit_seconds=fit_seconds,total_seconds=time.perf_counter()-started)
    if method=="mmd":
        metrics["rff_per_projection"]=RFF_PER_PROJECTION
        metrics["bandwidth_median"]=float(np.median(context["bandwidth"]))
    if previous:
        metrics["previous_rmse"]=previous["rmse"]
        metrics["previous_val_sliced_w2_squared"]=previous["fitted_val_sliced_w2_squared"]
        for key in ("iterations","function_evaluations"):
            metrics[key]+=previous[key]
        metrics["fit_seconds"]+=previous["fit_seconds"]
        metrics["refinement_rounds"]=int(previous.get("refinement_rounds",0))+1
    with (dest/"predictions.csv").open("w",newline="",encoding="utf-8") as file:
        writer=csv.writer(file)
        writer.writerow(("cellid","true_aging_factor","predicted_aging_factor"))
        for cell,factor in zip(cells,estimate):
            writer.writerow((cell["cellid"],cell["true_aging_factor"],float(factor)))
    metrics_path.write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    return metrics


def summarize(metrics: list[dict]) -> list[dict]:
    output=[]
    for size in TRAIN_BUDGET:
        for method in METHODS:
            group=[r for r in metrics if r["events_per_state"]==size and r["method_key"]==method]
            if len(group)!=4:
                raise ValueError(f"Expected four pairs: {size}, {method}, got {len(group)}")
            rmse=np.array([r["rmse"] for r in group])
            val=np.array([r["fitted_val_sliced_w2_squared"] for r in group])
            output.append(dict(events_per_state=size,method=LABELS[method],
                               method_key=method,train_events_per_state=TRAIN_BUDGET[size],
                               independent_pairs=4,rmse_mean=float(rmse.mean()),
                               rmse_std=float(rmse.std(ddof=1)),
                               r2_mean=float(np.mean([r["r2"] for r in group])),
                               heldout_sliced_w2_squared_mean=float(val.mean()),
                               fit_seconds_mean=float(np.mean([r["fit_seconds"] for r in group])),
                               objective_evaluations_mean=float(np.mean([r["function_evaluations"] for r in group])),
                               converged_fits=sum(r["success"] for r in group)))
    return output


def write_expanded_report(summary: list[dict]) -> None:
    archived=read_csv(OUT/"exp38_combined_comparison.csv")
    sw=read_csv(OUT/"structured_sw_summary.csv")
    combined=[]
    method_order=("Global-mean predictor","Mean-energy ratio","WS-only",
                  "Wasserstein adversarial",LABELS["moments"],LABELS["mmd"],
                  LABELS["energy"],"Joint event-level sliced W2 + XY smoothness")
    for size in TRAIN_BUDGET:
        combined.extend(r for r in archived if int(r["events_per_state"])==size)
        for row in summary:
            if row["events_per_state"]==size:
                combined.append(dict(events_per_state=size,method=row["method"],
                                     train_events_per_state=row["train_events_per_state"],
                                     independent_pairs=4,rmse_mean=row["rmse_mean"],
                                     rmse_std=row["rmse_std"],r2_mean=row["r2_mean"],
                                     comparison_note="New event-level projected fit; same training and validation events as sliced W2"))
    combined.sort(key=lambda r:(int(r["events_per_state"]),method_order.index(r["method"])))
    write_csv(OUT/"exp38_expanded_comparison.csv",combined)
    lines=["# Expanded event-level comparison on exp38", "",
           "All new fits use the same canonical independent reference/aged event pairs, the same 20k/30k/40k/50k fitting events per state, 32 random event projections, bounds [0.5,1], and XY-neighbor penalty as the joint sliced-W2 fit. Projected Moments matches the first and second projected moments (scaled by the reference spread); MMD uses four fixed random Fourier frequencies per projection and a reference-only median bandwidth; Energy uses the exact one-dimensional energy statistic on each projected distribution. No true factor enters these objectives. Each discrepancy is scaled to 0.2 at the identity correction before adding the fixed neighbor penalty. This normalization is a prespecified computational convention, not hyperparameter optimization. The same 5k held-out events per state and 64 additional random projections provide a common validation score.", "",
           "The archived WGAN, per-cell WS, and mean-ratio rows use the full nominal archive sizes, whereas the four new event-level rows fit smaller subsamples. The global-mean row uses the injected field mean and is an infeasible oracle constant. The adversarial row averages its model fits within each data-seed pair first; some model fits have different realized event splits within a nominal pair. RFF-MMD uses 128 frequencies in total and is an approximation, while the Energy row averages 1D energy distances over projections rather than computing full multivariate energy distance. Their bandwidth, feature count, and regularization were not separately tuned for this cohort. Results are descriptive, not a controlled ranking of discrepancies.", "",
           "| Archive events/state | Estimator | Fit events/state | Factor RMSE mean ± pair SD | R² mean |",
           "|---:|---|---:|---:|---:|"]
    for row in combined:
        fit=int(row["train_events_per_state"])
        fit_label="oracle" if fit==0 else f"{fit:,}"
        lines.append(f"| {int(row['events_per_state']):,} | {row['method']} | {fit_label} | {float(row['rmse_mean']):.4f} ± {float(row['rmse_std']):.4f} | {float(row['r2_mean']):.3f} |")
    lines.extend(["", "## Common held-out alignment for new event-level fits", "",
                  "These four rows at each archive size are directly comparable on the same validation events and independent projections. The finite-sample oracle applies the injected factor field; its score need not be zero.", "",
                  "| Archive events/state | Estimator | Held-out sliced-W2² | Mean fit seconds | Mean objective evaluations | Converged / 4 |",
                  "|---:|---|---:|---:|---:|---:|"])
    for size in TRAIN_BUDGET:
        for method in METHODS:
            row=next(r for r in summary if r["events_per_state"]==size and r["method_key"]==method)
            lines.append(f"| {size:,} | {row['method']} | {row['heldout_sliced_w2_squared_mean']:.3e} | {row['fit_seconds_mean']:.1f} | {row['objective_evaluations_mean']:.1f} | {row['converged_fits']} |")
        row=next(r for r in sw if int(r["events_per_state"])==size)
        lines.append(f"| {size:,} | Joint event-level sliced W2 + XY smoothness | {float(row['heldout_sliced_w2_squared_mean']):.3e} | {float(row['fit_seconds_mean']):.1f} | {float(row['function_evaluations_mean']):.1f} | {row['converged_fits']} |")
        lines.append(f"| {size:,} | Injected factors (oracle score) | {float(row['heldout_oracle_sliced_w2_squared_mean']):.3e} | — | — | — |")
    lines.extend(["", "Individual factor estimates and run metrics are under `projected_baselines/{archive_size}_{seed_pair}/{method}/`. Runtime is local CPU fit time and should not be compared with archived GPU training. The threshold audit is in `THRESHOLD_RESULTS.md`.", ""])
    (OUT/"EXPANDED_RESULTS.md").write_text("\n".join(lines),encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    FIT_OUT.mkdir(parents=True,exist_ok=True)
    canonical=read_csv(BASE/"analysis/canonical_prepared_runs.csv")
    cells=read_csv(BASE/"analysis/active_cells_and_occupancy.csv")
    index={int(c["cellid"]):i for i,c in enumerate(cells)}
    truth=np.array([float(c["true_aging_factor"]) for c in cells])
    metrics=[]
    for record in canonical:
        size=int(record["events_per_state"])
        pair=record["data_seed_pair"]
        if args.size and size!=args.size:
            continue
        if args.pair and pair!=args.pair:
            continue
        if args.summarize_only:
            for method in METHODS:
                path=FIT_OUT/f"{size}_{pair}"/method/"metrics.json"
                metrics.append(json.loads(path.read_text()))
            continue
        selected=(args.method,) if args.method else METHODS
        if args.refine:
            paths=[FIT_OUT/f"{size}_{pair}"/method/"metrics.json" for method in selected]
            if all(path.exists() and json.loads(path.read_text())["success"] for path in paths):
                metrics.extend(json.loads(path.read_text()) for path in paths)
                continue
        run=BASE/record["run_name"]
        saved_truth=np.load(run/"real_aging_factors.npy")
        for i,c in enumerate(cells):
            if abs(saved_truth[int(c["z"]),int(c["x"]),int(c["y"])]-truth[i])>1e-6:
                raise AssertionError("True factor field differs across exp38 archives")
        reference,dup_ref=load_sparse_events(run/"logs/train_prepared_new.zip",index,"E_cal")
        aged,dup_aged=load_sparse_events(run/"logs/train_prepared_old.zip",index,"E_aged_cal")
        if dup_ref or dup_aged:
            raise ValueError("Duplicate event-cell rows")
        for method in METHODS:
            if args.method and method!=args.method:
                continue
            result=fit_method(method,record,cells,reference,aged,truth,args.refine)
            metrics.append(result)
            print(size,pair,method,f"rmse={result['rmse']:.5f}",
                  f"converged={result['success']}",flush=True)
    if args.size or args.pair or args.method:
        return
    summary=summarize(metrics)
    write_csv(FIT_OUT/"pair_metrics.csv",metrics)
    write_csv(FIT_OUT/"summary.csv",summary)
    write_expanded_report(summary)
    (FIT_OUT/"metadata.json").write_text(json.dumps(dict(
        method_labels=LABELS,fit_projections=PROJECTIONS,
        mmd_rff_per_projection=RFF_PER_PROJECTION,
        train_budgets=TRAIN_BUDGET,val_events=VAL_EVENTS,
        smoothness=SMOOTHNESS,initial_data_loss_target=0.2,
        initial_maxiter=MAXITER,refine_maxiter=REFINE_MAXITER,
        initial_ftol=1e-10,refine_ftol=1e-8,
        all_fits_converged=all(r["success"] for r in metrics),
        limitation="Projected approximations; fixed regularization; archived baselines have unmatched fitting budgets"
    ),indent=2),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size",type=int,choices=tuple(TRAIN_BUDGET))
    parser.add_argument("--pair",choices=("0-1","2-3","4-5","6-7"))
    parser.add_argument("--method",choices=METHODS)
    parser.add_argument("--refine",action="store_true")
    parser.add_argument("--summarize-only",action="store_true")
    main(parser.parse_args())
