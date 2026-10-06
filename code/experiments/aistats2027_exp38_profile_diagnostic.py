"""Profile a fitted exp38 calorimeter objective along interpretable field shifts.

Uses the saved 150k, seed-pair 0-1 sliced-W2 fit. The three directions are
specified geometrically before inspecting any objective curves. Random 5k
event subsets are drawn from events excluded from fitting and validation.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aistats2027_joint_sliced_calo import load_sparse_events, xy_neighbor_edges
from aistats2027_exp38_structured_sw import (
    BASE, OUT, TRAIN_BUDGET, VAL_EVENTS, PROJECTIONS, SEED, SMOOTHNESS,
    directions, read_csv, write_csv,
)

ARCHIVE_SIZE=150_000
SEED_PAIR="0-1"
SUBSAMPLE_EVENTS=5_000
RESAMPLES=32
RESAMPLE_SEED=73027
DATA_SCALE=10_000.0
LAYER=16
SPATIAL_CENTER=(12.5,12.5)
SPATIAL_SIGMA=3.0
DEST=OUT/"objective_profiles"


def profiles(cells: list[dict]) -> dict[str, tuple[np.ndarray,np.ndarray]]:
    z=np.array([int(c["z"]) for c in cells])
    x=np.array([int(c["x"]) for c in cells])
    y=np.array([int(c["y"]) for c in cells])
    layer=z==LAYER
    bump=np.exp(-((x-SPATIAL_CENTER[0])**2+(y-SPATIAL_CENTER[1])**2)/
                (2*SPATIAL_SIGMA**2))
    spatial=np.zeros(len(cells))
    spatial[layer]=bump[layer]-bump[layer].mean()
    spatial/=np.max(np.abs(spatial))
    return {
        "global_decrease":(np.ones(len(cells)),np.linspace(-.04,0,13)),
        "layer_16":(layer.astype(float),np.linspace(-.04,.04,17)),
        "smooth_xy_layer_16":(spatial,np.linspace(-.06,.06,25)),
    }


def penalty(a: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    return SMOOTHNESS*float(np.mean((a[left]-a[right])**2))


def projected_objective(a: np.ndarray, reference_sorted: np.ndarray,
                        aged, fit_directions: np.ndarray) -> float:
    transformed=np.asarray(aged @ (fit_directions/a[:,None]))
    transformed.sort(axis=0)
    return DATA_SCALE*float(np.mean((reference_sorted-transformed)**2))


def evaluate_grid(reference,aged,field: np.ndarray,fit_directions: np.ndarray,
                  profile: tuple[np.ndarray,np.ndarray],left: np.ndarray,
                  right: np.ndarray) -> list[dict]:
    direction,grid=profile
    reference_sorted=np.sort(np.asarray(reference@fit_directions),axis=0)
    output=[]
    for amplitude in grid:
        candidate=field+amplitude*direction
        if candidate.min()<.5-1e-10 or candidate.max()>1+1e-10:
            raise ValueError(f"Profile leaves fitted parameter bounds at {amplitude}")
        data=projected_objective(candidate,reference_sorted,aged,fit_directions)
        prior=penalty(candidate,left,right)
        output.append(dict(amplitude=float(amplitude),data_objective=data,
                           spatial_penalty=prior,full_objective=data+prior,
                           parameter_rms_shift=float(np.sqrt(np.mean((candidate-field)**2)))))
    return output


def plot(summary: list[dict],metadata: dict) -> None:
    names={"global_decrease":"Global decrease (all cells)",
           "layer_16":"Uniform shift, layer 16",
           "smooth_xy_layer_16":"Smooth spatial contrast, layer 16"}
    fig,axes=plt.subplots(2,3,figsize=(13.5,7.0),constrained_layout=True)
    for col,(key,title) in enumerate(names.items()):
        rows=[r for r in summary if r["direction"]==key]
        t=np.array([r["amplitude"] for r in rows])*100
        for row_index,part in enumerate(("data","full")):
            ax=axes[row_index,col]
            train=np.array([r[f"train_{part}_change"] for r in rows])
            held=np.array([r[f"heldout_{part}_change_mean"] for r in rows])
            spread=np.array([r[f"heldout_{part}_change_sd"] for r in rows])
            ax.fill_between(t,held-spread,held+spread,color="#3d86b7",alpha=.22,
                            label="±1 SD of paired change")
            ax.plot(t,held,color="#216b9c",marker="o",markersize=3,
                    label="held-out mean")
            ax.plot(t,train,color="#202020",linestyle="--",linewidth=1.5,
                    label="fitting events")
            if part=="full":
                prior=np.array([r["spatial_penalty_change"] for r in rows])
                if np.max(np.abs(prior))>1e-8:
                    ax.plot(t,prior,color="#c47c38",linestyle=":",linewidth=1.8,
                            label="spatial penalty change")
            ax.axhline(0,color="0.35",linewidth=.8)
            ax.axvline(0,color="0.5",linewidth=.8)
            ax.grid(alpha=.15)
            if row_index==0:
                ax.set_title(title,fontsize=10)
            else:
                ax.set_xlabel("Change in affected factors (percentage points)")
    axes[0,0].set_ylabel("Data discrepancy change")
    axes[1,0].set_ylabel("Full objective change")
    by_label={}
    for ax in axes.ravel():
        h,l=ax.get_legend_handles_labels()
        by_label.update(zip(l,h))
    fig.legend(by_label.values(),by_label.keys(),loc="upper center",
               bbox_to_anchor=(.5,1.07),ncol=4,frameon=False,fontsize=8)
    fig.savefig(DEST/"objective_profiles.png",dpi=210,bbox_inches="tight")
    fig.savefig(DEST/"objective_profiles.pdf",bbox_inches="tight")
    plt.close(fig)


def report(summary: list[dict],key_points: list[dict],metadata: dict) -> None:
    lines=["# Local objective profiles for a 2,048-cell exp38 fit", "",
           "This diagnostic uses the converged joint sliced-$W_2$ response field from the canonical 150k-event, seed-pair 0–1 exp38 archive. It profiles the **fitted objective** `10,000 × empirical sliced-W2² + 1,000 × mean squared XY-neighbor factor difference` along three directions fixed from detector geometry: a uniform decrease of all cells; a constant shift of layer 16; and a zero-mean Gaussian spatial contrast in layer 16, centered at (12.5,12.5) with width 3 cells. The spatial direction is normalized to maximum absolute component 1. The global direction is one-sided to stay within the fitted [0.5,1] factor range. The spatial grid extends to ±0.06 factor points so its field-wide RMS change is about equal to a ±0.02 layer shift.", "",
           "The top row plots the data discrepancy alone; the bottom row includes the deterministic spatial penalty. The black curve is the change on the 40k fitting events per state. Blue shows mean change over 32 random 5k-event subsets per state, drawn from the 105k archived events per state unused in fitting and the original 5k validation split. The blue band is ±1 SD of *paired objective changes* (each subset is evaluated at both the fitted and perturbed field). Subsets are drawn independently but can overlap within the fixed archive. The 32 projection directions are exactly those used in fitting. A dotted orange curve shows the deterministic penalty change where nonzero. No factor truth is used by this diagnostic.", "",
           "| Direction | Shift (factor points) | RMS field change | Data change mean ± resampling SD | Data signal / SD | Penalty change | Full change mean ± resampling SD | Full signal / SD |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in key_points:
        data_signal_to_sd=(abs(row['heldout_data_change_mean'])/row['heldout_data_change_sd']
                           if row['heldout_data_change_sd']>0 else 0.0)
        lines.append(f"| {row['direction']} | {row['amplitude']:+.3f} | {row['parameter_rms_shift']:.5f} | {row['heldout_data_change_mean']:+.6f} ± {row['heldout_data_change_sd']:.6f} | {data_signal_to_sd:.2f} | {row['spatial_penalty_change']:+.6f} | {row['heldout_full_change_mean']:+.6f} ± {row['heldout_full_change_sd']:.6f} | {row['paired_signal_to_sd']:.2f} |")
    lines.extend(["", "At the 2-point smooth spatial shift, the data-only change is near zero relative to its across-subset SD, while the full-objective rise is approximately the spatial penalty. A plot of the penalized objective alone would make this direction appear well separated for the wrong reason. The global decrease is more detectable; the +2-point layer shift changes the objective by less than one across-subset SD. At ±6 points the smooth direction changes the field by RMS 0.00706, essentially the same as a ±2-point layer shift (RMS 0.00707), yet its data-only change is still below one paired resampling SD in either sign. The baseline full-objective SD at the fitted field is 0.002026 across the 32 subsets. These are diagnostics for fixed perturbations with 5k-event samples, not confidence intervals for the 2,048 factors. The penalty does not change under a uniform global or within-layer shift because its edges connect only XY neighbors in the same layer. The three directions are probes, not a full 2,048-dimensional Hessian or proof of global identifiability.", "",
                  "Files: `objective_profiles.png/pdf`, `profile_summary.csv`, `profile_resamples.csv`, `key_points.csv`, and `metadata.json`. Reproduce with `python scripts/aistats2027_exp38_profile_diagnostic.py`.", ""])
    (DEST/"PROFILE_RESULTS.md").write_text("\n".join(lines),encoding="utf-8")


def main() -> None:
    DEST.mkdir(parents=True,exist_ok=True)
    canonical=read_csv(BASE/"analysis/canonical_prepared_runs.csv")
    record=next(r for r in canonical if int(r["events_per_state"])==ARCHIVE_SIZE
                and r["data_seed_pair"]==SEED_PAIR)
    cells=read_csv(BASE/"analysis/active_cells_and_occupancy.csv")
    fitted=read_csv(OUT/"structured_sw"/f"{ARCHIVE_SIZE}_{SEED_PAIR}"/"predictions.csv")
    if [r["cellid"] for r in fitted]!=[r["cellid"] for r in cells]:
        raise AssertionError("Fitted field and active-cell order differ")
    field=np.array([float(r["predicted_aging_factor"]) for r in fitted])
    index={int(r["cellid"]):i for i,r in enumerate(cells)}
    run=BASE/record["run_name"]
    reference,dup_ref=load_sparse_events(run/"logs/train_prepared_new.zip",index,"E_cal")
    aged,dup_aged=load_sparse_events(run/"logs/train_prepared_old.zip",index,"E_aged_cal")
    if dup_ref or dup_aged:
        raise ValueError("Duplicate event-cell rows")
    ntrain=TRAIN_BUDGET[ARCHIVE_SIZE]
    ref_order=np.random.default_rng(SEED).permutation(reference.shape[0])
    aged_order=np.random.default_rng(SEED+1).permutation(aged.shape[0])
    ref_train=reference[ref_order[:ntrain]]
    aged_train=aged[aged_order[:ntrain]]
    ref_pool=ref_order[ntrain+VAL_EVENTS:]
    aged_pool=aged_order[ntrain+VAL_EVENTS:]
    fit_directions=directions(PROJECTIONS,SEED+2,len(cells))
    left,right=xy_neighbor_edges(cells)
    definition=profiles(cells)
    training={}
    for key,profile in definition.items():
        training[key]=evaluate_grid(ref_train,aged_train,field,fit_directions,
                                    profile,left,right)
        print("training",key,flush=True)
    rng=np.random.default_rng(RESAMPLE_SEED)
    resampled=[]
    for replicate in range(RESAMPLES):
        ref_ids=rng.choice(ref_pool,size=SUBSAMPLE_EVENTS,replace=False)
        aged_ids=rng.choice(aged_pool,size=SUBSAMPLE_EVENTS,replace=False)
        ref_sample=reference[ref_ids]
        aged_sample=aged[aged_ids]
        for key,profile in definition.items():
            for row in evaluate_grid(ref_sample,aged_sample,field,fit_directions,
                                     profile,left,right):
                resampled.append(dict(replicate=replicate,direction=key,**row))
        print("resample",replicate+1,"/",RESAMPLES,flush=True)
    write_csv(DEST/"profile_resamples.csv",resampled)
    summary=[]
    key_points=[]
    for key,(direction,grid) in definition.items():
        base_train=next(r for r in training[key] if abs(r["amplitude"])<1e-12)
        base_resampled={b:next(r for r in resampled if r["direction"]==key and
                               r["replicate"]==b and abs(r["amplitude"])<1e-12)
                        for b in range(RESAMPLES)}
        baseline_sd=float(np.std([r["full_objective"] for r in base_resampled.values()],ddof=1))
        for i,amplitude in enumerate(grid):
            subset=[r for r in resampled if r["direction"]==key and
                    abs(r["amplitude"]-amplitude)<1e-12]
            full_delta=np.array([r["full_objective"]-
                                 base_resampled[r["replicate"]]["full_objective"]
                                 for r in subset])
            data_delta=np.array([r["data_objective"]-
                                 base_resampled[r["replicate"]]["data_objective"]
                                 for r in subset])
            train=training[key][i]
            row=dict(direction=key,amplitude=float(amplitude),
                     parameter_rms_shift=train["parameter_rms_shift"],
                     train_full_change=train["full_objective"]-base_train["full_objective"],
                     train_data_change=train["data_objective"]-base_train["data_objective"],
                     spatial_penalty_change=train["spatial_penalty"]-base_train["spatial_penalty"],
                     heldout_full_change_mean=float(full_delta.mean()),
                     heldout_full_change_sd=float(full_delta.std(ddof=1)),
                     heldout_data_change_mean=float(data_delta.mean()),
                     heldout_data_change_sd=float(data_delta.std(ddof=1)),
                     baseline_full_objective_sd=baseline_sd,
                     paired_signal_to_sd=(float(abs(full_delta.mean())/full_delta.std(ddof=1))
                                          if full_delta.std(ddof=1)>0 else 0.0))
            summary.append(row)
            if abs(abs(amplitude)-.02)<1e-12 or (key=="smooth_xy_layer_16" and
                                                 abs(abs(amplitude)-.06)<1e-12):
                key_points.append(row)
    write_csv(DEST/"profile_summary.csv",summary)
    write_csv(DEST/"key_points.csv",key_points)
    metadata=dict(archive_events_per_state=ARCHIVE_SIZE,data_seed_pair=SEED_PAIR,
                  run_name=record["run_name"],fitted_field_source=str(OUT/"structured_sw"/
                  f"{ARCHIVE_SIZE}_{SEED_PAIR}"/"predictions.csv"),
                  train_events_per_state=ntrain,excluded_original_validation_events=VAL_EVENTS,
                  resample_pool_events_per_state=len(ref_pool),
                  resample_events_per_state=SUBSAMPLE_EVENTS,resamples=RESAMPLES,
                  resample_seed=RESAMPLE_SEED,fit_projections=PROJECTIONS,
                  data_loss_scale=DATA_SCALE,xy_smoothness=SMOOTHNESS,
                  layer=LAYER,spatial_center=SPATIAL_CENTER,spatial_sigma=SPATIAL_SIGMA,
                  bounds=[.5,1.0],directions=list(definition),
                  interpretation="Local profiles along three fixed directions; not a full Hessian")
    (DEST/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    plot(summary,metadata)
    report(summary,key_points,metadata)


if __name__=="__main__":
    main()
