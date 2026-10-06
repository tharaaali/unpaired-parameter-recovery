# Supplementary material: Distribution Matching for Interpretable Parameter Recovery from Unpaired Data

This review supplement links each empirical claim in the paper to its analysis code, saved run-level outputs, aggregate tables, and figures. Start with `python code/verify_reported.py`; it checks the printed numbers against the CSV files using only the Python standard library. The large simulated calorimeter event tables are omitted from this compact review package. No names, affiliations, email addresses, or personal filesystem paths are included.

## Folder guide

| Location | Contents |
|---|---|
| `code/verify_reported.py` | Fast, independent audit of the numerical values reported in the paper. |
| `code/recompute_thresholds.py` | Recomputes the counterfactual threshold rates when the raw reference tables are available. |
| `code/experiments/` | The experiment scripts used to generate the results. Paths alone were adjusted for this folder; numerical calculations are unchanged. |
| `code/export_calorimeter.py` | Rebuilds the minimal calorimeter tables from the original simulation archives, if those archives are available. |
| `results.zip` | Compressed copy of `results/` for convenient download. |
| `figures/` | The eight PDF figures included in the manuscript. |
| `data/README.md` | Documents the omitted calorimeter event archivee and the request procedure. |
| `requirements.txt` | Tested scientific Python versions for the experiment scripts. The fast verifier and data exporter use only the standard library. |

Run the audit from this folder:

```text
python code/verify_reported.py
```

The audit reads saved outputs; it does not rerun optimization. For file integrity, `file_manifest.csv` records relative paths, byte counts, and SHA-256 hashes.

Each source script has a specific role:

| Script in `code/experiments/` | Purpose |
|---|---|
| `aistats2027_synthetic_benchmark.py` | Affine implementation check and two rotational profiles. |
| `aistats2027_nonlinear_transform_experiment.py` | Nonlinear data generation, discrepancy fits, sample-size and noise sweeps. |
| `aistats2027_nonlinear_curvature.py` | Local objective curvature and independent held-out variation. |
| `aistats2027_final_nonlinear_benchmark.py` | Combines the nonlinear runs into the final four-method comparison. |
| `aistats2027_nonlinear_sensitivity.py` | Bandwidth and projection-count sensitivity runs. |
| `aistats2027_rotation_separation.py` | Continuous rotation family and separation-width analysis. |
| `aistats2027_misspecification_targets.py` | Population targets and finite-sample fits for omitted-term misspecification. |
| `aistats2027_joint_sliced_calo.py` | Common calorimeter event representation and initial joint sliced-Wasserstein fit. |
| `aistats2027_exp38_structured_sw.py` | Structured 2,048-factor sliced-Wasserstein fitting. |
| `aistats2027_exp38_projected_baselines.py` | Projected-discrepancy comparison methods and saved fitted factors. |
| `aistats2027_exp38_profile_diagnostic.py` | Fixed-direction objective profiles and held-out event-subset variation. |
| `aistats2027_exp38_threshold_audit.py` | Original counterfactual censoring calculations. |

## Paper-to-material map

| Paper result | Code | Saved evidence |
|---|---|---|
| Nonlinear four-discrepancy comparison, held-out alignment, paired RMSE contrasts | `aistats2027_nonlinear_transform_experiment.py`, `aistats2027_nonlinear_curvature.py`, `aistats2027_final_nonlinear_benchmark.py` | `results/nonlinear_transform/final_benchmark/runs.csv`, `summary.csv`, `theta_summary.csv`, `paired_comparisons.csv`; `figures/main_figure.pdf` |
| Bandwidth and projection-count sensitivity | `aistats2027_nonlinear_sensitivity.py` | `results/nonlinear_transform/sensitivity/variant_runs.csv`, `paired_runs.csv`, `summary.csv` |
| Local curvature, gradient variation, independent held-out SD | `aistats2027_nonlinear_curvature.py` | `results/nonlinear_transform/curvature/runs.csv`, `holdout_fits.csv`, `summary.csv`; `figures/curvature_vs_uncertainty.pdf` |
| Sample-size and changed-side-noise sweeps | `aistats2027_nonlinear_transform_experiment.py` | `results/nonlinear_transform/runs.csv`, `sample_size_summary.csv`, `noise_summary.csv`; two corresponding PDFs in `figures/` |
| Continuous rotation and near-optimal profile width | `aistats2027_rotation_separation.py` | `results/rotation_separation/runs.csv`, `summary.csv`; `figures/separation_sweep.pdf` |
| Isotropic/asymmetric rotation counterexample | `aistats2027_synthetic_benchmark.py` | `results/synthetic_pilot/rotation_profiles.csv`; `figures/rotation_identifiability.pdf` |
| Omitted nonlinear term and discrepancy-dependent targets | `aistats2027_misspecification_targets.py` | `results/misspecification_targets/mc_targets.csv`, `finite_fits.csv`, `paired_shift_contrasts.csv`, `summary.csv`; `figures/pseudo_true_trajectories.pdf` |
| Affine implementation check | `aistats2027_synthetic_benchmark.py` | `results/synthetic_n512/runs.csv` (five seeds per method) |
| Four new joint calorimeter fits, 2,048 cell factors, held-out alignment | `aistats2027_joint_sliced_calo.py`, `aistats2027_exp38_structured_sw.py`, `aistats2027_exp38_projected_baselines.py` | `results/exp38_event_level/structured_sw/` and `projected_baselines/` contain 64 fit-level `predictions.csv` and `metrics.json` files; aggregate CSVs are in their parent folders |
| Directional objective diagnostic | `aistats2027_exp38_profile_diagnostic.py` | `results/exp38_event_level/objective_profiles/profile_resamples.csv`, `profile_summary.csv`, `key_points.csv`; `figures/exp38_objective_profiles.pdf` |
| Counterfactual threshold-censoring audit | `aistats2027_exp38_threshold_audit.py` | `results/exp38_event_level/threshold_pair_runs.csv`, `threshold_summary.csv`; the raw reference tables used to generate these outputs are omitted from this review supplement |
| Previously published adversarial RMSE in the main calorimeter table | Prior Sensors study, DOI `10.3390/s26165024` | `results/published_sensors_context.csv`; this is a cited contextual number, not a new fit or a matched held-out comparison |

## Calorimeter data availability

The simulated event-level calorimeter archive is kept outside this review supplement and is not publicly archived because of its size. The code, recorded configuration metadata, fit-level predictions and metrics, aggregate results, and numerical checks are included here. A seven-column raw-event export is retained by the authors and may be obtained upon reasonable request after anonymous review. The event tables are not available to reviewers in this package.

The retained export has 32 gzip-compressed CSV tables: four independent data-seed pairs at each of four available event counts (50k, 90k, 150k, 250k), with a `reference` and a `changed_source` table for each condition. Each event table has exactly these seven columns:

| Column | Meaning |
|---|---|
| `cellid` | Original active-cell identifier, equivalent to the requested cell column. |
| `energy_raw_reference_mev` | Untransformed nominal hit energy in MeV (`E_cal_org` in the original simulation output). In a `changed_source` file, this is the nominal energy underlying a retained aged-side hit. |
| `aging_factor` | Injected cell response factor. It is supplied for simulation evaluation and the threshold audit; the unsupervised fit does not use it. |
| `event` | Simulated event identifier. Reference and changed-source events are independent and must **not** be paired by row or identifier. |
| `x`, `y`, `z` | Integer detector-cell coordinates. |

For a changed-source row, raw aged energy is `energy_raw_reference_mev * aging_factor`. The prepared aged-side archive excludes hits whose raw aged energy is below 200 MeV; the `reference` table retains uncensored hits and supports the counterfactual threshold audit. Reference and changed-source events are independent. Missing aged-side hits remain missing.

The exact new fitting runs used a quantile transform fitted in the original preparation pipeline **before** the independent event split. The seven-column raw export supports independent inspection of raw events and reproduction of the threshold calculations, but it does not by itself guarantee an exact refit of the historical calorimeter optimization. The saved fit-level predictions, metrics, and source code in this supplement support auditing the reported parameter and held-out-score results now.

After the raw export is released, a table can be read without additional packages:

```python
import csv
import gzip

with gzip.open("data/calorimeter/size_150000_pair_0-1_reference.csv.gz", "rt", newline="", encoding="utf-8") as stream:
    rows = csv.DictReader(stream)
    first_hit = next(rows)
    print(first_hit)
```

The external archive includes a manifest of row and event counts, file sizes, and hashes. To recompute the reported counterfactual threshold fractions after obtaining that archive, extract it into `data/calorimeter/` and run:

```text
python code/recompute_thresholds.py
```

Use `--size 150000` to check only the canonical archive size, or `--data-dir PATH` to read an extracted archive elsewhere. This independent calculation compares every reported cutoff with `results/exp38_event_level/threshold_summary.csv`. It cannot run from the review supplement alone because the raw rows are omitted.

## Running the source analyses

The synthetic experiments need only the packages in `requirements.txt`. From this folder, the intended sequence for a complete rerun is:

```text
python code/experiments/aistats2027_nonlinear_transform_experiment.py --output results/nonlinear_transform
python code/experiments/aistats2027_nonlinear_curvature.py
python code/experiments/aistats2027_final_nonlinear_benchmark.py
python code/experiments/aistats2027_nonlinear_sensitivity.py
python code/experiments/aistats2027_rotation_separation.py
python code/experiments/aistats2027_misspecification_targets.py
```

The original ten nonlinear datasets are generated first; the curvature audit generates forty additional independent datasets, which the final benchmark combines. The sensitivity and misspecification scripts also offer `--summarize-only` to rebuild summaries from their saved run-level CSVs. Complete optimization is substantially more expensive than `verify_reported.py`.

The calorimeter fitting scripts retain their original preparation interface: they expect the quantile-transformed sparse event archives used by the study. The seven-column raw tables are a different representation and do not supply those prepared arrays. The published Sensors adversarial result used all nominal 150k events per state and lacks a score on the new held-out projection set; it is separated from the four new fits in the manuscript table.

## Statistical interpretation

The nonlinear datasets are independent across seeds and true parameters, with methods paired on the same data within each condition. The calorimeter table's four data-seed pairs are the independent units; cells and configured fits are not additional independent replicates. The profile bands are one SD of paired changes across 32 subsets of one fixed archive, not confidence intervals for the 2,048 factors. Published and new calorimeter methods have different fitting budgets, so their RMSE values are contextual rather than a controlled ranking.
