# BỔ SUNG, ô 1/3. Chạy lại LightGBM với thêm 2 seed (2222, 3333) để biết kết quả thay đổi bao nhiêu khi đổi cách
# chia fold và tinh chỉnh. Seed 1111 đã có từ Nhiệm vụ 1 và 3. Bốn cohort: eICU và MIMIC-III, nhãn KDIGO ≥ 1 (demo gốc)
# và KDIGO ≥ 2 (sinh lại). Cấu hình gốc của YAIB (c61fad0). Không cần GPU. Khoảng 1 giờ 40 phút.
# Kết quả lưu riêng ở MyDrive/YAIB_logs_seeds để không lẫn với các lần chạy cũ.
# Bị ngắt giữa chừng thì chạy lại ô này: máy chạy tiếp từ các lượt tinh chỉnh đã lưu.
import os, glob, json, sqlite3, subprocess, shutil, time, statistics
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/YAIB_logs'          # seed 1111 (Nhiệm vụ 1 và 3)
SEEDDIR = '/content/drive/MyDrive/YAIB_logs_seeds'   # seed 2222, 3333
SEEDS = [2222, 3333]
MODEL = 'LGBMClassifier'
COHORTS = [('mimic_demo', 'AKI', 'demo_data/aki', 'KDIGO ≥ 1, demo gốc'),     # MIMIC trước vì chạy nhanh
           ('mimic_demo', 'AKI_KDIGO2', 'demo_data_nv3/aki_kdigo2', 'KDIGO ≥ 2, sinh lại'),
           ('eicu_demo', 'AKI', 'demo_data/aki', 'KDIGO ≥ 1, demo gốc'),
           ('eicu_demo', 'AKI_KDIGO2', 'demo_data_nv3/aki_kdigo2', 'KDIGO ≥ 2, sinh lại')]
os.makedirs(SEEDDIR, exist_ok=True)
os.environ['MPLBACKEND'] = 'Agg'

def gio_vn():
    return time.strftime('%H:%M', time.gmtime(time.time() + 7 * 3600))

def sh(cmd, logname, cwd=None):
    logf = os.path.join(SEEDDIR, logname)
    with open(logf, 'w') as f:
        p = subprocess.run(cmd, shell=True, executable='/bin/bash', cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        print(f'LỖI (mã {p.returncode}) ở bước {logname}. 30 dòng cuối của log:')
        print(''.join(open(logf).readlines()[-30:]))
        raise RuntimeError('Dừng ở bước trên. Gửi thông báo lỗi này cho người hướng dẫn.')

# Bước 1. YAIB c61fad0 như các ô trước; chép cohort sinh lại từ Drive; điền 2 ca thiếu giới tính như Nhiệm vụ 1
print(f'{gio_vn()} 1/3 Cài YAIB (vài phút)...', flush=True)
if not os.path.isdir('/content/YAIB/.git'):
    shutil.rmtree('/content/YAIB', ignore_errors=True)
sh('cd /content && ([ -d YAIB ] || git clone -q -b development https://github.com/rvandewater/YAIB.git) && cd YAIB && '
   'git checkout -q c61fad0f70813cdbd815c955038ea22416c46ef5 && pip install -q uv && uv python install 3.12 && '
   'uv sync --python 3.12 && uv run icu-benchmarks -h', 'bs1_cai_YAIB.log')
os.chdir('/content/YAIB')
shutil.copytree(f'{LOGDIR}/nv3_cohort/aki_kdigo2', 'demo_data_nv3/aki_kdigo2', dirs_exist_ok=True)
for d in ['demo_data/aki/eicu_demo', 'demo_data/aki/mimic_demo',
          'demo_data_nv3/aki_kdigo2/eicu_demo', 'demo_data_nv3/aki_kdigo2/mimic_demo']:
    sta = pd.read_parquet(f'{d}/sta.parquet')
    n = int(sta['sex'].isna().sum())
    if n:
        sta['sex'] = sta['sex'].fillna(sta['sex'].mode()[0])
        sta.to_parquet(f'{d}/sta.parquet', index=False)
        shutil.rmtree(f'{d}/cache', ignore_errors=True)
        shutil.rmtree(f'{d}/preproc', ignore_errors=True)
    print(f'    {d}: điền {n} ca thiếu giới tính')

# Bước 2. Huấn luyện từng seed
def so_fold(run):
    return len(glob.glob(os.path.join(run, 'repetition_*', 'fold_*', 'test_metrics.json'))) if run else 0

def final_run(root, name, task):
    runs = [os.path.dirname(p) for p in glob.glob(os.path.join(root, name, task, MODEL, '*', 'accumulated_test_metrics.json'))]
    runs = [r for r in runs if so_fold(r) == 25]
    return max(runs, key=os.path.getmtime) if runs else None

def best_db(base):
    best, best_n = None, 0
    for db in glob.glob(os.path.join(base, '*', 'hyperparameter_tuning_logs.db')):
        try:
            n = sqlite3.connect(db).execute("select count(*) from trials where state='COMPLETE'").fetchone()[0]
        except Exception:
            n = 0
        if best is None or n > best_n:
            best, best_n = db, n
    return best, best_n

def chay(name, task, data_dir, seed):
    root = f'{SEEDDIR}/seed{seed}'
    base = os.path.join(root, name, task, MODEL)
    db, n = best_db(base)
    cmd = ['uv', 'run', 'icu-benchmarks', 'train', '-d', f'{data_dir}/{name}', '-n', name,
           '-t', 'BinaryClassification', '-tn', task, '-m', MODEL, '--seed', str(seed),
           '-l', root, '--tune', '-gc', '-lc']
    if db:
        cmd += ['--hp-checkpoint', os.path.relpath(db, base)]
    print(f'{gio_vn()} 2/3 {name} / {task} / seed {seed}: bắt đầu, đã có {n}/30 lượt tinh chỉnh', flush=True)
    logf = os.path.join(SEEDDIR, f'bs1_{name}_{task}_seed{seed}_{time.strftime("%m%d-%H%M")}.log')
    with open(logf, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        while True:
            try:
                p.wait(timeout=600)
                break
            except subprocess.TimeoutExpired:
                runs = sorted(glob.glob(os.path.join(base, '20*')))
                print(f'    {gio_vn()} tinh chỉnh {best_db(base)[1]}/30 lượt, '
                      f'huấn luyện chính {so_fold(runs[-1]) if runs else 0}/25 fold', flush=True)
    if p.returncode != 0:
        print(f'    LỖI (mã {p.returncode}). 30 dòng cuối của log {logf}:')
        print(''.join(open(logf).readlines()[-30:]))
        return False
    print(f'    {gio_vn()} Xong: {so_fold(final_run(root, name, task))}/25 fold', flush=True)
    return True

for name, task, data_dir, ten in COHORTS:
    for seed in SEEDS:
        for lan in range(3):   # thử lại tối đa 3 lần nếu Drive lỗi tạm thời
            if final_run(f'{SEEDDIR}/seed{seed}', name, task) or chay(name, task, data_dir, seed):
                break

# Bước 3. Bảng so sánh 3 seed (± trong từng seed là sai số chuẩn qua 25 fold; cột cuối là trung bình và độ lệch chuẩn giữa 3 seed)
rows = []
for name, task, data_dir, ten in COHORTS:
    row = {'bộ dữ liệu': name, 'cohort': ten}
    auc, pr = [], []
    for seed, root in [(1111, LOGDIR)] + [(s, f'{SEEDDIR}/seed{s}') for s in SEEDS]:
        run = final_run(root, name, task)
        if not run:
            row[f'seed {seed}'] = 'chưa có'
            continue
        m = json.load(open(os.path.join(run, 'accumulated_test_metrics.json')))
        auc.append(100 * m['avg']['AUC']); pr.append(100 * m['avg']['PR'])
        row[f'seed {seed}'] = f"{auc[-1]:.1f} ± {100*m['std']['AUC']:.1f} / {pr[-1]:.1f} ± {100*m['std']['PR']:.1f}"
    if len(auc) == 3:
        row['trung bình 3 seed (SD)'] = (f'{statistics.mean(auc):.1f} ({statistics.stdev(auc):.1f}) / '
                                         f'{statistics.mean(pr):.1f} ({statistics.stdev(pr):.1f})')
    rows.append(row)
bang = pd.DataFrame(rows)
bang.to_csv(os.path.join(SEEDDIR, 'ba_seed_lgbm.csv'), index=False)
print(f'\n{gio_vn()} 3/3 LightGBM, AUROC / AUPRC (%) theo seed:')
print(bang.to_string(index=False))
