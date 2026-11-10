# Results

Test metrics of every run we made, exported from the YAIB log folders on Google Drive. The export cell copied no models, checkpoints, tuning databases or cohort data.

## Files

| File | Content |
| --- | --- |
| `per_fold_test_metrics.csv` | One row per test fold: 828 folds from 37 runs. `AUC` = AUROC, `PR` = AUPRC, `loss` = test loss, as written by YAIB in `repetition_*/fold_*/test_metrics.json` |
| `runs_summary.csv` | One row per run: number of folds, whether all 25 folds finished, and mean ± standard error (%) of AUROC and AUPRC. The SE is computed as YAIB does it: population SD ÷ √n |
| `summaries/` | The summary tables saved by the notebook cells, in the folder where each cell saved them (Vietnamese headers; glossary below) |

The columns `log_root`, `dataset`, `task`, `model` and `run` give the YAIB log path: `<log_root>/<dataset>/<task>/<model>/<run>/`.

Six runs are incomplete (`complete = False`). They stopped early and were restarted as new runs. The four eICU LightGBM and logistic-regression runs that stopped after 10 folds were hit by the two stays with missing sex, which fall in the test fold of the third repetition with seed 1111 (manuscript Table 4). Only complete runs are reported in the manuscript.

## Runs behind each manuscript item

| Manuscript item | `log_root` | `task` | Summary table |
| --- | --- | --- | --- |
| Table 1 | `YAIB_logs` | `AKI` (complete runs, one per model and dataset) | `summaries/YAIB_logs/nv1_bang11_aki.csv` |
| Table 2, KDIGO ≥1 regenerated | `YAIB_logs` | `AKI_KDIGO1_sinhlai` | `summaries/YAIB_logs/nv3_kdigo2.csv`, `nv3_co_mau.csv` |
| Table 2, KDIGO ≥2 regenerated | `YAIB_logs` | `AKI_KDIGO2` | same as above; cohort attrition in `summaries/YAIB_logs/nv3_cohort/` |
| Table 3 | models of Tables 1 and 2 | — | `summaries/YAIB_logs/bs2_truoc_khoi_phat_kiem_dinh_ngoai.csv`; per fold in the `.json` of the same name |
| Figure 1, LightGBM and logistic regression with the code of 26 May 2023 | `YAIB_logs/kiemchung_052023` | `AKI` | `summaries/YAIB_logs/kiemchung_052023/kiemchung_052023.csv` |
| Figure 1, GRU with the 2023 search space | `YAIB_logs_gru_v03` | `AKI` | — |
| LightGBM seeds 2222 and 3333 | `YAIB_logs_seeds/seed2222`, `YAIB_logs_seeds/seed3333` | `AKI`, `AKI_KDIGO2` | `summaries/YAIB_logs_seeds/ba_seed_lgbm.csv` |
| Table 4, LightGBM after #200 | `YAIB_logs_sualoi/26b682d7`, `YAIB_logs_sualoi/6c9c19f4` | `AKI` | `summaries/YAIB_logs_sualoi/sua_loi_lgbm.csv` |
| Table 4, GRU after #243 and #244 | `YAIB_logs_gru_sualoi/hien_tai` (current search space), `YAIB_logs_gru_sualoi/kg2023` (2023 search space) | `AKI` | `summaries/YAIB_logs_gru_sualoi/gru_sua_loi.csv` |

`summaries/YAIB_logs/nv2_kiem_dinh_ngoai.csv` is an earlier external validation with a single eICU model. Table 3 replaces it, scoring all 25 paired fold models.

In `bs2_truoc_khoi_phat_kiem_dinh_ngoai.json`, each key is `label|repetition|fold`. Each scenario holds three sets of scores:

- `yaib`: the score stored by YAIB;
- `all`: our recomputation on all hours;
- `pre`: hours before onset only.

Each set gives `AUC`, `PR`, the number of hours `n` and the positive share `pos`.

## Glossary of Vietnamese headers

| Vietnamese | English |
| --- | --- |
| bộ dữ liệu | dataset |
| mô hình | model |
| số fold | number of folds |
| lượt tinh chỉnh | tuning trials |
| chạy lại | this study (rerun) |
| bài báo (Bảng 11) | paper (Table 11) |
| phiên bản; mã | version; code |
| hiện tại | current |
| sửa lỗi | after the fix |
| không gian (tìm kiếm) | search space |
| demo gốc | shipped demo cohort |
| sinh lại | regenerated cohort |
| số lượt ICU | ICU stays |
| ca dương | stays with AKI |
| giờ nhãn 1; giờ dương | positive hours |
| mọi giờ | all hours |
| trước khởi phát | before onset |
| kịch bản | scenario |
| chênh lệch theo cặp | paired difference |
| trung bình 3 seed (SD) | mean of the three seeds (SD) |
| lượt | run |
