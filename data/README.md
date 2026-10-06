# Calorimeter event data

The simulated raw-event tables are omitted from this review supplement to keep the package compact. The paper's aggregate results, per-fit predictions and metrics, and analysis code are in `../results/` and `../code/`.

The large archive is not publicly hosted because of its size. A seven-column raw-event export is retained by the authors and may be obtained upon reasonable request after anonymous review. It contains 32 gzip-compressed CSV tables covering four event counts and four independent seed pairs, with separate reference and changed-source tables. Each table contains only `cellid`, `energy_raw_reference_mev`, `aging_factor`, `event`, `x`, `y`, and `z`, plus a separate file manifest. The reference and changed-source events are unpaired.

The threshold audit can be rerun after the archive is available by extracting the tables and manifest into `data/calorimeter/` and running `python code/recompute_thresholds.py` from the supplement root. The historical calorimeter fits additionally used pre-split quantile preparation; the seven raw columns alone do not guarantee exact refits.
