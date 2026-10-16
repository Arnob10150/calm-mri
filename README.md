# CALM-MRI

Research code for **"CALM-MRI: A Leakage-Controlled, Calibrated, and Deferral-Aware
Framework for Brain Tumor MRI Classification."**

Brain tumor MRI classifiers routinely report accuracies above 98%. This repository
contains the experiments behind a paper arguing that those numbers come from
evaluation protocols that would not survive clinical deployment, and a framework
designed for the corrected protocol.

## What this code establishes

| Finding | Result |
|---|---|
| Slice-level splitting inflates accuracy | **+3.92 points** on average across 6 architectures, data/schedule/seeds held fixed |
| Public cohorts overlap each other | **72.2%** of one cohort and **52.7%** of another are pixel-confirmed duplicates of a third |
| CALM-MRI under patient-grouped evaluation | **94.90%** accuracy, **94.20** macro-F1, **0.9877** AUROC, **0.0167** AURC, vs 12 baselines |
| Conformal deferral gate (alpha = 0.1) | reports **90.7%** of cases automatically at **97.6%** accuracy; referred cases carry a 32.2% error rate |

The headline accuracy is *lower* than the comparable literature. That is the point:
it is measured with patients, not slices, separated across folds.

## Repository layout

```
src/
  data.py          cohort loading, patient-grouped splitting, CLAHE + augmentation
  models.py        trunk + multi-scale fusion head, baseline backbones.
                   Note: the class default is fusion="gated", but the reported model
                   uses fusion="concat" (see results/final_model.json); the gated
                   variant is one of the ablations.
  train.py         training loop, label smoothing, EMA weight averaging
  metrics.py       accuracy, macro-F1, AUROC, ECE, Brier, NLL, AURC, conformal sets
  xai.py           Grad-CAM, deletion/insertion faithfulness curves

scripts/
  dl_par.sh            download the four public cohorts
  prepare_cheng.py     build the primary cohort with patient identifiers
  prepare_external.py  build the three external test cohorts
  verify_duplicates.py difference-hash + pixel-correlation contamination audit
  tune.py              recipe search on inner validation patients only
  run_experiments.py   all experiments (E0-E9); see --help
  finalize.py          assemble the final ensemble from tuned members
  analyze.py           aggregate results into the reported statistics

results/               every JSON/NPZ the paper's numbers are computed from
```

`results/` is committed deliberately. Every number in the paper is generated from
these files rather than typed by hand, so the tables and figures can be regenerated
and checked without re-running training.

## Reproducing

```bash
pip install -r requirements.txt     # see the note about the CUDA-matched torch build
bash scripts/dl_par.sh              # ~3 GB of public cohorts; not tracked in git
python scripts/prepare_cheng.py
python scripts/prepare_external.py
python scripts/verify_duplicates.py # contamination audit (runs before any training)
python scripts/run_experiments.py --help
```

Experiments are five-fold cross-validated and repeated over three seeds. Reported
values are the mean and standard deviation across seeds of the out-of-fold aggregate.
Scalar metrics are computed **per seed and then averaged**; pooling seeds' predictions
before scoring is a different estimator and does not reproduce the tables.

Full training is GPU-hours expensive. To inspect the findings without retraining,
read `results/` directly — `E1_benchmark.json` holds the main comparison and
`oof_grouped_<model>_s<seed>.npz` / `final/oof_s<seed>.npz` hold per-case calibrated
probabilities and labels for every model.

## Data

Four public cohorts, reduced to the three classes they share (glioma, meningioma,
pituitary tumor). None is redistributed here; `scripts/dl_par.sh` fetches them.

- **Cheng** (figshare, CC BY 4.0) — 3,064 T1-w slices from 233 patients. The only
  public brain tumor MRI cohort distributing a patient identifier per slice, which is
  what makes patient-grouped evaluation possible. Training cohort for every experiment.
- **Sartaj**, **Hemg** (Kaggle compilations) and **PMRAM** (Mendeley Data, CC BY 4.0) —
  external test sets only. None distributes patient identifiers.

All four are public, de-identified, and used under their stated licenses. No new
patient data were collected and no attempt was made to re-identify any subject.

## Intended use

This is research code supporting a methodological argument. The framework is decision
support with an explicit referral pathway, not an autonomous diagnostic device. It
classifies single 2D slices across three tumor types; normal studies, other lesions,
and study-level decisions are outside its scope. Prospective, multi-institution
validation with radiologist comparison would be required before any clinical use.

## Citation

Citation details will be added here once the paper is published.

## License

Code released under the MIT License (see `LICENSE`). The image cohorts are governed by
their own licenses at the sources linked above.
