# BỔ SUNG, ô 4. KIỂM TRA: các lỗi YAIB sửa sau lượt chạy của nhóm (tháng 10/2026) có làm đổi kết quả LightGBM trên eICU không?
# Lượt A: commit 26b682d7 (23/10/2026) = c61fad0 + bản sửa biến đếm (issue #200) + một thay đổi chỉ dành cho bài hồi quy.
# Lượt B: commit 6c9c19f4 (29/10/2026), bản mới nhất: thêm các bản sửa sắp xếp dữ liệu theo lượt nằm và giờ, xử lý NaN,
#         giới tính thiếu (#251) và hai lỗi chỉ của mô hình sâu (#243, #244); lỗi thang log khi tinh chỉnh (#240) chưa sửa.
# Lưu ý: mã 05/2023 cũng tính biến đếm sau khi điền giá trị thiếu, giống c61fad0, nên #200 không giải thích được chênh lệch với
# bài báo. Ô này trả lời câu hỏi khác: số liệu LightGBM của nhóm (76,9) có bị các lỗi sửa sau này làm lệch không.
# Giữ nguyên như Nhiệm vụ 1: dữ liệu demo, seed 1111, tinh chỉnh 30 lượt rồi 25 fold, điền 2 ca thiếu giới tính.
# Không cần GPU. Khoảng 1–1,5 giờ. Bị ngắt giữa chừng thì chạy lại ô này: máy chạy tiếp từ các lượt tinh chỉnh đã lưu.
import os, glob, json, sqlite3, subprocess, shutil, time
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
LOGDIR = '/content/drive/MyDrive/YAIB_logs'                    # Nhiệm vụ 1 (c61fad0), seed 1111
KC2023 = '/content/drive/MyDrive/YAIB_logs/kiemchung_052023'   # ô 3: mã 26/05/2023
FIXDIR = '/content/drive/MyDrive/YAIB_logs_sualoi'             # ô này
COMMITS = [('26b682d7e9aff507f7f5e0e2dcde85b5a706fc99', 'A: c61fad0 + sửa #200 (23/10/2026)'),
           ('6c9c19f4d8e8f98addf062ea0171e97df82e745e', 'B: bản mới nhất (29/10/2026)')]
MODEL, NAME, TASK, SEED = 'LGBMClassifier', 'eicu_demo', 'AKI', 1111
os.makedirs(FIXDIR, exist_ok=True)
os.environ['MPLBACKEND'] = 'Agg'

def gio_vn():
    return time.strftime('%H:%M', time.gmtime(time.time() + 7 * 3600))

def sh(cmd, logname, cwd=None):
    logf = os.path.join(FIXDIR, logname)
    with open(logf, 'w') as f:
        p = subprocess.run(cmd, shell=True, executable='/bin/bash', cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        print(f'LỖI (mã {p.returncode}) ở bước {logname}. 30 dòng cuối của log:')
        print(''.join(open(logf).readlines()[-30:]))
        raise RuntimeError('Dừng ở bước trên. Gửi thông báo lỗi này cho người hướng dẫn.')

# Bước 1. Mỗi commit một thư mục riêng (để bộ nhớ đệm tiền xử lý không lẫn nhau); điền 2 ca thiếu giới tính như Nhiệm vụ 1
for sha, ten in COMMITS:
    repo = f'/content/YAIB_{sha[:8]}'
    print(f'{gio_vn()} 1/3 Cài YAIB {sha[:8]} ({ten}), vài phút...', flush=True)
    if not os.path.isdir(f'{repo}/.git'):
        shutil.rmtree(repo, ignore_errors=True)
    sh(f'cd /content && ([ -d {repo} ] || git clone -q -b development https://github.com/rvandewater/YAIB.git {repo}) && '
       f'cd {repo} && git checkout -q {sha} && pip install -q uv && uv python install 3.12 && uv sync --python 3.12 && '
       'uv run icu-benchmarks -h', f'sl1_cai_{sha[:8]}.log')
    d = f'{repo}/demo_data/aki/eicu_demo'
    sta = pd.read_parquet(f'{d}/sta.parquet')
    n = int(sta['sex'].isna().sum())
    if n:
        sta['sex'] = sta['sex'].fillna(sta['sex'].mode()[0])
        sta.to_parquet(f'{d}/sta.parquet', index=False)
        shutil.rmtree(f'{d}/cache', ignore_errors=True)
        shutil.rmtree(f'{d}/preproc', ignore_errors=True)
    print(f'    điền {n} ca thiếu giới tính (0 nghĩa là đã điền từ trước)')

# Bước 2. Tinh chỉnh rồi chạy 25 fold cho từng commit
def so_fold(run):
    return len(glob.glob(os.path.join(run, 'repetition_*', 'fold_*', 'test_metrics.json'))) if run else 0

def final_run(root):
    runs = [os.path.dirname(p) for p in glob.glob(os.path.join(root, NAME, TASK, MODEL, '*', 'accumulated_test_metrics.json'))]
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

def chay(sha):
    repo, root = f'/content/YAIB_{sha[:8]}', f'{FIXDIR}/{sha[:8]}'
    base = os.path.join(root, NAME, TASK, MODEL)
    db, n = best_db(base)
    cmd = ['uv', 'run', 'icu-benchmarks', 'train', '-d', f'demo_data/aki/{NAME}', '-n', NAME,
           '-t', 'BinaryClassification', '-tn', TASK, '-m', MODEL, '--seed', str(SEED),
           '-l', root, '--tune', '-gc', '-lc']
    if db:
        cmd += ['--hp-checkpoint', os.path.relpath(db, base)]
    print(f'{gio_vn()} 2/3 {sha[:8]}: bắt đầu, đã có {n}/30 lượt tinh chỉnh', flush=True)
    logf = os.path.join(FIXDIR, f'sl2_{sha[:8]}_{time.strftime("%m%d-%H%M")}.log')
    with open(logf, 'w') as f:
        p = subprocess.Popen(cmd, cwd=repo, stdout=f, stderr=subprocess.STDOUT)
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
    print(f'    {gio_vn()} Xong: {so_fold(final_run(root))}/25 fold', flush=True)
    return True

for sha, ten in COMMITS:
    for lan in range(3):   # thử lại tối đa 3 lần nếu Drive lỗi tạm thời
        if final_run(f'{FIXDIR}/{sha[:8]}') or chay(sha):
            break

# Bước 3. Bảng so sánh (± là sai số chuẩn qua 25 fold) và siêu tham số được chọn
def ket_qua(root):
    run = final_run(root)
    if not run:
        return None
    m = json.load(open(os.path.join(run, 'accumulated_test_metrics.json')))
    return [100 * m['avg']['AUC'], 100 * m['std']['AUC'], 100 * m['avg']['PR'], 100 * m['std']['PR']]

def sieu_tham_so(root):
    db, n = best_db(os.path.join(root, NAME, TASK, MODEL))
    if not db:
        return '(không có cơ sở dữ liệu Optuna)'
    con = sqlite3.connect(db)
    tid = con.execute("select t.trial_id from trials t join trial_values v on t.trial_id = v.trial_id "
                      "where t.state='COMPLETE' order by v.value limit 1").fetchone()[0]
    ps = con.execute('select param_name, param_value from trial_params where trial_id = ?', (tid,)).fetchall()
    return f'{n} lượt; ' + ', '.join(f'{k}={v:.3g}' for k, v in ps)

rows = [{'phiên bản': 'Bài báo (Bảng 11)', 'AUROC': '72.6 ± 0.3', 'AUPRC': '23.8 ± 0.4', 'Δ AUROC so với c61fad0': ''}]
goc = ket_qua(LOGDIR)
for ten, root in [('Mã 26/05/2023 (ô 3)', KC2023), ('c61fad0 (Nhiệm vụ 1)', LOGDIR)] + \
                 [(t, f'{FIXDIR}/{s[:8]}') for s, t in COMMITS]:
    kq = ket_qua(root)
    rows.append({'phiên bản': ten,
                 'AUROC': f'{kq[0]:.1f} ± {kq[1]:.1f}' if kq else 'chưa có',
                 'AUPRC': f'{kq[2]:.1f} ± {kq[3]:.1f}' if kq else 'chưa có',
                 'Δ AUROC so với c61fad0': f'{kq[0] - goc[0]:+.1f}' if kq and goc else ''})
bang = pd.DataFrame(rows)
bang.to_csv(os.path.join(FIXDIR, 'sua_loi_lgbm.csv'), index=False)
print(f'\n{gio_vn()} 3/3 LightGBM, eICU demo, AKI, seed 1111 (AUROC / AUPRC %):')
print(bang.to_string(index=False))
print('\nSiêu tham số tốt nhất:')
for ten, root in [('c61fad0', LOGDIR)] + [(s[:8], f'{FIXDIR}/{s[:8]}') for s, t in COMMITS]:
    print(f'  {ten}: {sieu_tham_so(root)}')
