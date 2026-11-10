# YAIB AKI reproduction on the eICU and MIMIC-III demo cohorts

This repository holds the code and outputs behind our reproduction of the acute kidney injury (AKI) task of Yet Another ICU Benchmark (YAIB; van de Water et al., ICLR 2024). We reran the task from the public [YAIB](https://github.com/rvandewater/YAIB) code and the eICU and MIMIC-III demo cohorts shipped with it. We then traced the differences from the paper to specific code versions and adapted the task to moderate-to-severe AKI (KDIGO stage ≥2).

Manuscript: Pham TT, et al. *Reproducing an open intensive care machine-learning benchmark for acute kidney injury: software version drift, undocumented label design, and adaptation to moderate-to-severe AKI.* In preparation.

## Contents

| Path | What it is |
| --- | --- |
| `notebooks/YAIB_AKI_reproduction_colab.ipynb` | The Google Colab notebook we ran, with its outputs |
| `cells/` | The five supplementary notebook cells (o1–o5) as standalone scripts, matching notebook cells 25 and 27–30 |

Per-fold test metrics for every run will be added under `results/`.

The notebook's headings and comments are in Vietnamese. The code, commands and printed outputs are as run, and the table below maps each cell to the manuscript.

## Where each result comes from

| Manuscript item | Notebook cell | Script | Code version and settings |
| --- | --- | --- | --- |
| Table 1: six models on both demo cohorts | 12–16 (table printed by cell 15) | — | YAIB `c61fad0`, default configurations, seed 1111 |
| Table 2: cohorts by KDIGO threshold | 20–24 (table printed by cell 23) | — | YAIB-cohorts `74ac699`, ricu `0969a52` with the patches below |
| Table 3: all hours vs hours before onset; eICU → MIMIC-III | 25 | `cells/o2_pre_onset_and_external_validation.py` | Models from the Table 1 and Table 2 runs, no retraining |
| Figure 1: LightGBM with the code of May 2023 | 27 | `cells/o3_code_and_libraries_may_2023.py` | YAIB `221b00f4` (26 May 2023) with the libraries of the paper's Appendix G.2 |
| Figure 1: GRU with the 2023 search space | 8 | — | YAIB `c61fad0`; only three GRU search ranges changed (below) |
| Seed variability of LightGBM (seeds 2222, 3333) | 28 | `cells/o1_lightgbm_three_seeds.py` | YAIB `c61fad0` |
| Table 4: LightGBM after the count-feature fix (#200) | 29 | `cells/o4_lightgbm_after_fixes.py` | YAIB `26b682d7` and `6c9c19f4` |
| Table 4: GRU after the class-weight and checkpoint fixes (#243, #244) | 30 | `cells/o5_gru_after_fixes.py` | YAIB `6c9c19f4` |

## Environment

- Google Colab, Python 3.12, with PyTorch 2.6.0, scikit-learn 1.5.2, LightGBM 4.6.0 and Optuna 4.0.0 as pinned by YAIB `c61fad0`. Deep models ran on an NVIDIA T4 GPU.
- Seed 1111 unless stated otherwise. 30 tuning trials, then 5 × 5 repeated cross-validation split by ICU stay.
- May 2023 rerun (cell 27): Python 3.10, PyTorch 2.0.1, LightGBM 3.3.5, scikit-learn 1.2.2 and scikit-optimize 0.9.0, with 50 Gaussian-process trials on 3 folds, as in that code. We patched two lines of scikit-optimize for NumPy compatibility (`np.int` → `int`). It ran on a CPU.

## Changes we made to run the code

1. **Missing sex in the eICU demo.** Two stays (1648745 and 258915) have no value for `sex`, which stops cross-validation. Every cell fills them with the most frequent value before training (YAIB #251).
2. **Completed runs.** `accumulated_test_metrics.json` is rewritten after every fold. We therefore treated a run as complete only when it had 25 `repetition_*/fold_*/test_metrics.json` files.
3. **GRU 2023 search space** (cell 8), passed as `-hp` bindings:
   - `optimizer/hyperparameter.lr = (1e-5, 3e-4)`; the current value is `(1e-6, 1e-4, "log")`.
   - `model/hyperparameter.hidden_dim = (32, 256, "log")`; the current value is `(32, 512, "log")`.
   - `model/hyperparameter.layer_dim = (1, 3)`; the current value is `(1, 10)`.
4. **YAIB-cohorts with current R packages** (cells 21–22):
   - In `R/src/obs_time.R`, we renamed the column `...time` to `stop_time_`, because data.table ≥1.15 treats the `..` prefix specially.
   - We added the `ignore.attr` argument that data.table now passes to ricu's `rbind_id_tbl`.
   - We installed ricu's suggested packages `units`, `progress`, `pillar` and `withr`.
5. **KDIGO ≥2 cohorts.** The only change to the cohort code is `threshold = 2L` in `aki()` of `ricu-extensions/callbacks/callback-kdigo.R`.
6. **External validation.** `--eval` fails for classical models, so cell 25 scores the saved fold models with YAIB's own `preprocess_data` and `PredictionPolarsDataset`.

## Rerunning

Open the notebook in Colab and run Steps 1–3 (GPU check, Google Drive, code and environment), then the cell you need. Results are written to Google Drive under `MyDrive/YAIB_logs*`. Long cells resume from saved tuning trials if a session stops.

## Data

The repository contains no patient data. The demo cohorts come from the YAIB repository and are derived from the open-access eICU-CRD and MIMIC-III demo datasets on PhysioNet.

## License

MIT, as for YAIB. Questions and corrections are welcome as GitHub issues.
