# BỔ SUNG, ô 2/3. Hai phân tích trên các mô hình LightGBM đã huấn luyện (seed 1111), không huấn luyện lại:
#  (a) Chấm riêng các giờ TRƯỚC khởi phát. Nhãn của YAIB đánh 1 cho 6 giờ trước VÀ 6 giờ sau khởi phát; ở đây bỏ các
#      giờ từ lúc khởi phát trở đi, để đo đúng khả năng báo trước. Khởi phát = giờ có nhãn 1 đầu tiên + 6 giờ.
#  (b) Kiểm định ngoài eICU → MIMIC-III cho KDIGO ≥ 2 (và làm lại cho KDIGO ≥ 1): mô hình eICU của từng fold (lần lặp r,
#      fold f) chấm trên tập kiểm tra MIMIC-III cùng chỉ số, nên dùng cả 25 mô hình eICU thay vì 1 như Nhiệm vụ 2.
# Ô này cũng kiểm tra nhãn có khớp đúng từng giờ với dự đoán không: YAIB lấy nhãn theo thứ tự bảng kết cục, còn đặc
# trưng thì sắp theo (lượt ICU, giờ); ở đây nhãn được ghép lại theo đúng (lượt ICU, giờ) rồi so với điểm YAIB đã lưu.
# Không cần GPU. Khoảng 30–40 phút. Bị ngắt thì chạy lại ô này: các fold đã chấm được lưu trên Drive và bỏ qua.
import os, json, subprocess, shutil, time, statistics, math
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/YAIB_logs'
OUT = f'{LOGDIR}/bs2_truoc_khoi_phat_kiem_dinh_ngoai.json'
os.environ['MPLBACKEND'] = 'Agg'
JOBS = [['KDIGO ≥ 1, demo gốc', 'AKI', 'demo_data/aki/eicu_demo', 'demo_data/aki/mimic_demo'],
        ['KDIGO ≥ 2, sinh lại', 'AKI_KDIGO2', 'demo_data_nv3/aki_kdigo2/eicu_demo', 'demo_data_nv3/aki_kdigo2/mimic_demo']]

def gio_vn():
    return time.strftime('%H:%M', time.gmtime(time.time() + 7 * 3600))

def sh(cmd, logname, cwd=None):
    logf = os.path.join(LOGDIR, logname)
    with open(logf, 'w') as f:
        p = subprocess.run(cmd, shell=True, executable='/bin/bash', cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        print(f'LỖI (mã {p.returncode}) ở bước {logname}. 30 dòng cuối của log:')
        print(''.join(open(logf).readlines()[-30:]))
        raise RuntimeError('Dừng ở bước trên. Gửi thông báo lỗi này cho người hướng dẫn.')

# Bước 1. YAIB c61fad0, cohort sinh lại, điền 2 ca thiếu giới tính (như lúc huấn luyện)
print(f'{gio_vn()} 1/3 Cài YAIB (vài phút)...', flush=True)
if not os.path.isdir('/content/YAIB/.git'):
    shutil.rmtree('/content/YAIB', ignore_errors=True)
sh('cd /content && ([ -d YAIB ] || git clone -q -b development https://github.com/rvandewater/YAIB.git) && cd YAIB && '
   'git checkout -q c61fad0f70813cdbd815c955038ea22416c46ef5 && pip install -q uv && uv python install 3.12 && '
   'uv sync --python 3.12 && uv run icu-benchmarks -h', 'bs2_cai_YAIB.log')
os.chdir('/content/YAIB')
shutil.copytree(f'{LOGDIR}/nv3_cohort/aki_kdigo2', 'demo_data_nv3/aki_kdigo2', dirs_exist_ok=True)
for job in JOBS:
    for d in job[2:]:
        sta = pd.read_parquet(f'{d}/sta.parquet')
        if sta['sex'].isna().any():
            sta['sex'] = sta['sex'].fillna(sta['sex'].mode()[0])
            sta.to_parquet(f'{d}/sta.parquet', index=False)

# Bước 2. Chấm lại bằng chính hàm tiền xử lý và hàm lấy dữ liệu của YAIB
SCRIPT = r'''
import sys, os, glob, json, logging, warnings
from pathlib import Path
import numpy as np, pandas as pd
logging.disable(logging.WARNING)
warnings.filterwarnings('ignore')
sys.path.insert(0, '/content/YAIB')
import gin
import icu_benchmarks.run  # đăng ký mọi cấu hình gin giống khi chạy YAIB thật
gin.parse_config_file('configs/tasks/BinaryClassification.gin', skip_unknown=True)
gin.parse_config_files_and_bindings(['configs/prediction_models/LGBMClassifier.gin',
                                     'configs/tasks/BinaryClassification.gin'], None,
                                    finalize_config=False, skip_unknown=True)
from joblib import load
from sklearn.metrics import roc_auc_score, average_precision_score
from icu_benchmarks.data.split_process_data import preprocess_data
from icu_benchmarks.data.loader import PredictionPolarsDataset
from icu_benchmarks.data.constants import DataSplit, DataSegment

LOGDIR = '/content/drive/MyDrive/YAIB_logs'
JOBS, OUT = json.loads(sys.argv[1]), sys.argv[2]

def so_fold(run):
    return len(glob.glob(os.path.join(run, 'repetition_*', 'fold_*', 'test_metrics.json')))

def final_run(name, task):
    runs = [os.path.dirname(p) for p in glob.glob(os.path.join(LOGDIR, name, task, 'LGBMClassifier', '*', 'accumulated_test_metrics.json'))]
    runs = [r for r in runs if so_fold(r) == 25]
    if not runs:
        raise SystemExit(f'Không tìm thấy lần chạy đủ 25 fold: {name}/{task}')
    return max(runs, key=os.path.getmtime)

def test_rows(data_dir, r, f):
    data = preprocess_data(Path(data_dir), seed=1111, cv_repetitions=5, repetition_index=r, cv_folds=5,
                           fold_index=f, load_cache=True, generate_cache=True)
    ds = PredictionPolarsDataset(data, split=DataSplit.test, ram_cache=False)
    X, y_yaib, _ = ds.get_data_and_labels()
    # Hàng của X được sắp theo (lượt ICU, giờ): ghép nhãn theo đúng cặp đó
    feat = data[DataSplit.test][DataSegment.features].select(['stay_id', 'time']).sort(['stay_id', 'time'])
    outc = data[DataSplit.test][DataSegment.outcome].select(['stay_id', 'time', 'label'])
    lab = feat.with_row_index('i').join(outc, on=['stay_id', 'time'], how='left').sort('i')
    # Chuyển sang pandas qua danh sách Python: to_pandas() cần pyarrow, mà pyarrow của YAIB không chạy với numpy 1.26
    lab = pd.DataFrame({c: lab[c].to_list() for c in ['stay_id', 'time', 'label']})
    if not (len(lab) == len(y_yaib) == X.shape[0]):
        raise SystemExit(f'Số hàng không khớp: {len(lab)}, {len(y_yaib)}, {X.shape[0]}')
    t = lab['time']
    h = (t.dt.total_seconds() / 3600 if pd.api.types.is_timedelta64_dtype(t) else t.astype(float)).to_numpy()
    y = lab['label'].astype(float).to_numpy()
    first = pd.Series(h, index=lab['stay_id'].to_numpy())[y == 1].groupby(level=0).min()
    onset = lab['stay_id'].map(first + 6).to_numpy(dtype=float)
    pre = np.isnan(onset) | (h < onset)
    return X, y_yaib, y, pre

def metr(yt, p):
    return {'AUC': float(roc_auc_score(yt, p)), 'PR': float(average_precision_score(yt, p)),
            'n': int(len(yt)), 'pos': float(np.mean(yt))}

def evaluate(model, X, y_yaib, y, pre):
    if X.shape[1] != model.n_features_in_:
        raise SystemExit(f'Số đặc trưng không khớp: {X.shape[1]} so với {model.n_features_in_} lúc huấn luyện')
    p = model.predict_proba(X)[:, 1]
    ok = ~np.isnan(y)
    return {'yaib': metr(y_yaib, p), 'all': metr(y[ok], p[ok]), 'pre': metr(y[ok & pre], p[ok & pre]),
            'lech_nhan': float(np.mean(y_yaib[ok] != y[ok])), 'thieu_nhan': int((~ok).sum())}

res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ten, task, edir, mdir in JOBS:
    erun, mrun = final_run('eicu_demo', task), final_run('mimic_demo', task)
    for r in range(5):
        for f in range(5):
            key = f'{ten}|{r}|{f}'
            if key in res:
                continue
            fold = f'repetition_{r}/fold_{f}'
            em, mm = load(f'{erun}/{fold}/last.joblib'), load(f'{mrun}/{fold}/last.joblib')
            E, M = test_rows(edir, r, f), test_rows(mdir, r, f)
            res[key] = {'eICU → eICU': evaluate(em, *E), 'MIMIC → MIMIC': evaluate(mm, *M),
                        'eICU → MIMIC': evaluate(em, *M),
                        'luu_eicu': json.load(open(f'{erun}/{fold}/test_metrics.json'))['AUC'],
                        'luu_mimic': json.load(open(f'{mrun}/{fold}/test_metrics.json'))['AUC']}
            json.dump(res, open(OUT, 'w'))  # lưu sau mỗi fold để chạy tiếp được
            print(f"{ten} lặp {r} fold {f}: eICU {100*res[key]['eICU → eICU']['all']['AUC']:.1f}, "
                  f"trước khởi phát {100*res[key]['eICU → eICU']['pre']['AUC']:.1f}", flush=True)
print('XONG', flush=True)
'''
open('/content/YAIB/bs2_eval.py', 'w').write(SCRIPT)
logf = os.path.join(LOGDIR, f'bs2_eval_{time.strftime("%m%d-%H%M")}.log')
print(f'{gio_vn()} 2/3 Chấm lại 2 × 25 fold (mỗi fold khoảng 1 phút)...', flush=True)
with open(logf, 'w') as f:
    p = subprocess.Popen(['uv', 'run', 'python', 'bs2_eval.py', json.dumps(JOBS, ensure_ascii=False), OUT],
                         cwd='/content/YAIB', stdout=f, stderr=subprocess.STDOUT)
    while True:
        try:
            p.wait(timeout=300)
            break
        except subprocess.TimeoutExpired:
            done = len(json.load(open(OUT))) if os.path.exists(OUT) else 0
            print(f'    {gio_vn()} đã chấm {done}/50 fold', flush=True)
if p.returncode != 0:
    print(f'LỖI (mã {p.returncode}). 30 dòng cuối của log {logf}:')
    print(''.join(open(logf).readlines()[-30:]))
    raise RuntimeError('Dừng ở bước trên. Gửi thông báo lỗi này cho người hướng dẫn.')

# Bước 3. Bảng kết quả (trung bình ± sai số chuẩn qua 25 fold)
res = json.load(open(OUT))
def fmt(v):
    return f'{statistics.mean(v):.1f} ± {statistics.pstdev(v) / math.sqrt(len(v)):.1f}'
rows, kiem = [], []
for ten, task, edir, mdir in JOBS:
    keys = [f'{ten}|{r}|{f}' for r in range(5) for f in range(5) if f'{ten}|{r}|{f}' in res]
    for kb in ['eICU → eICU', 'MIMIC → MIMIC', 'eICU → MIMIC']:
        v = [res[k][kb] for k in keys]
        rows.append({'nhãn': ten, 'kịch bản': kb, 'số fold': len(v),
                     'AUROC mọi giờ': fmt([100 * x['all']['AUC'] for x in v]),
                     'AUROC trước khởi phát': fmt([100 * x['pre']['AUC'] for x in v]),
                     'AUPRC mọi giờ': fmt([100 * x['all']['PR'] for x in v]),
                     'AUPRC trước khởi phát': fmt([100 * x['pre']['PR'] for x in v]),
                     'giờ dương: mọi giờ / trước khởi phát': f"{statistics.mean([100 * x['all']['pos'] for x in v]):.1f}% / "
                                                              f"{statistics.mean([100 * x['pre']['pos'] for x in v]):.1f}%"})
    for kind in ['all', 'pre']:
        d = [100 * (res[k]['eICU → MIMIC'][kind]['AUC'] - res[k]['MIMIC → MIMIC'][kind]['AUC']) for k in keys]
        rows.append({'nhãn': ten, 'kịch bản': f'Chênh lệch theo cặp ({"mọi giờ" if kind == "all" else "trước khởi phát"})',
                     'số fold': len(d), 'AUROC mọi giờ' if kind == 'all' else 'AUROC trước khởi phát': f'{fmt(d)}'})
    for k in keys:
        kiem.append(abs(res[k]['eICU → eICU']['yaib']['AUC'] - res[k]['luu_eicu']))
        kiem.append(abs(res[k]['MIMIC → MIMIC']['yaib']['AUC'] - res[k]['luu_mimic']))
    lech = statistics.mean([res[k][kb]['lech_nhan'] for k in keys for kb in ['eICU → eICU', 'MIMIC → MIMIC']])
    print(f'Kiểm tra {ten}: tỷ lệ giờ có nhãn YAIB khác nhãn ghép theo (lượt ICU, giờ) = {100 * lech:.2f}%')
print(f'Kiểm tra cách tính: chênh lệch AUROC lớn nhất so với YAIB đã lưu = {100 * max(kiem):.3f} điểm (phải gần 0)')
bang = pd.DataFrame(rows).fillna('')
bang.to_csv(os.path.join(LOGDIR, 'bs2_truoc_khoi_phat_kiem_dinh_ngoai.csv'), index=False)
print(f'\n{gio_vn()} 3/3 Kết quả LightGBM (%, trung bình ± sai số chuẩn qua 25 fold):')
print(bang.to_string(index=False))
