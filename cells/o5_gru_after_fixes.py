# BỔ SUNG, ô 5. KIỂM TRA: GRU 54,5 (thấp gần ngẫu nhiên) và 63,7 (khôi phục không gian tìm kiếm 2023) có còn đúng khi đã sửa
# hai lỗi chỉ của mô hình học sâu không: #243 (trọng số lớp bị đảo khoảng một nửa số lần chạy) và #244 (chấm bằng trọng số
# epoch cuối thay vì epoch tốt nhất). Cả hai đã sửa trong commit mới nhất 6c9c19f4 (29/10/2026).
# Lượt C: 6c9c19f4, không gian tìm kiếm hiện tại   → so với c61fad0 cùng không gian: 54,5 ± 1,4
# Lượt D: 6c9c19f4, không gian tìm kiếm năm 2023   → so với c61fad0 cùng không gian: 63,7 ± 1,0
# Cách đọc: C vẫn thấp và D vẫn gần 63 thì kết luận "không gian tìm kiếm gây ra chênh lệch" giữ nguyên;
# C tăng lên gần 63 thì một phần chênh lệch là do hai lỗi trên.
# Giữ nguyên như các lần trước: dữ liệu demo eICU, seed 1111, 30 lượt tinh chỉnh × 5 fold rồi 5×5 fold, điền 2 ca thiếu giới tính.
# CẦN GPU T4 (Thời gian chạy → Thay đổi loại thời gian chạy). Khoảng 3–4 giờ cho cả hai lượt. Bị ngắt thì chạy lại ô này:
# máy chạy tiếp từ các lượt tinh chỉnh đã lưu. Log ghi vào máy Colab rồi chép sang Drive mỗi 10 phút (tránh lỗi ghi .ckpt lên Drive).
import os, re, glob, json, sqlite3, subprocess, shutil, time
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
SHA = '6c9c19f4d8e8f98addf062ea0171e97df82e745e'
REPO = f'/content/YAIB_{SHA[:8]}'                       # dùng chung thư mục với ô 4 nếu đã chạy trong phiên này
DRIVE = '/content/drive/MyDrive/YAIB_logs_gru_sualoi'
LOCAL = '/content/yaib_logs_gru_sualoi'
KHONG_GIAN_2023 = ['optimizer/hyperparameter.lr = (1e-5, 3e-4)',          # hiện tại: (1e-6, 1e-4, "log")
                   'model/hyperparameter.hidden_dim = (32, 256, "log")',  # hiện tại: (32, 512, "log")
                   'model/hyperparameter.layer_dim = (1, 3)']             # hiện tại: (1, 10)
LUOT = [('hien_tai', 'C: không gian hiện tại', [], (54.5, 1.4, 11.9, 0.6)),
        ('kg2023', 'D: không gian năm 2023', KHONG_GIAN_2023, (63.7, 1.0, 15.8, 0.5))]
NAME, TASK, MODEL = 'eicu_demo', 'AKI', 'GRU'
os.environ['MPLBACKEND'] = 'Agg'
subprocess.run(['pkill', '-f', 'icu-benchmarks train'])

gpu = subprocess.run('nvidia-smi --query-gpu=name --format=csv,noheader', shell=True, capture_output=True, text=True).stdout.strip()
assert gpu, 'CHƯA CÓ GPU: chọn Thời gian chạy → Thay đổi loại thời gian chạy → T4 GPU, rồi chạy lại ô này.'
print('GPU:', gpu)

def gio_vn():
    return time.strftime('%H:%M', time.gmtime(time.time() + 7 * 3600))

# Bước 1. Cài YAIB 6c9c19f4 (bỏ qua nếu ô 4 đã cài trong phiên này); điền 2 ca thiếu giới tính
print(f'{gio_vn()} 1/3 Cài YAIB {SHA[:8]}, vài phút...', flush=True)
os.makedirs(DRIVE, exist_ok=True)
if not os.path.isdir(f'{REPO}/.git'):
    shutil.rmtree(REPO, ignore_errors=True)
logf = os.path.join(DRIVE, 'gs1_cai.log')
with open(logf, 'w') as f:
    p = subprocess.run(f'cd /content && ([ -d {REPO} ] || git clone -q -b development https://github.com/rvandewater/YAIB.git {REPO}) && '
                       f'cd {REPO} && git checkout -q {SHA} && pip install -q uv && uv python install 3.12 && '
                       'uv sync --python 3.12 && uv run icu-benchmarks -h',
                       shell=True, executable='/bin/bash', stdout=f, stderr=subprocess.STDOUT)
if p.returncode != 0:
    print(''.join(open(logf).readlines()[-30:]))
    raise RuntimeError('Cài đặt lỗi. Gửi thông báo lỗi này cho người hướng dẫn.')
d = f'{REPO}/demo_data/aki/{NAME}'
sta = pd.read_parquet(f'{d}/sta.parquet')
n = int(sta['sex'].isna().sum())
if n:
    sta['sex'] = sta['sex'].fillna(sta['sex'].mode()[0])
    sta.to_parquet(f'{d}/sta.parquet', index=False)
    shutil.rmtree(f'{d}/cache', ignore_errors=True)
    shutil.rmtree(f'{d}/preproc', ignore_errors=True)
print(f'    điền {n} ca thiếu giới tính (0 nghĩa là đã điền từ trước)')

# Bước 2. Hai lượt GRU, mỗi lượt một thư mục log riêng
def dong_bo(src, dst, ghi_de=True):
    """Chép kết quả từ src sang dst, bỏ file mô hình .ckpt và thư mục tạm. Trả về số file chép lỗi."""
    loi = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [x for x in dirs if not x.startswith('tmp')]
        out = os.path.join(dst, os.path.relpath(root, src))
        os.makedirs(out, exist_ok=True)
        for fn in files:
            if fn.endswith('.ckpt'):
                continue
            s, t = os.path.join(root, fn), os.path.join(out, fn)
            try:
                if os.path.exists(t) and (not ghi_de or (not fn.endswith('.db') and os.path.getsize(t) == os.path.getsize(s)
                                                         and os.path.getmtime(t) >= os.path.getmtime(s) - 2)):
                    continue
                if fn.endswith('.db'):  # file SQLite (có thể đang được ghi): sao lưu qua API của SQLite
                    a, b = sqlite3.connect(s), sqlite3.connect('/tmp/_ban_sao.db')
                    a.backup(b)
                    a.close(); b.close()
                    shutil.copyfile('/tmp/_ban_sao.db', t)
                else:
                    shutil.copy2(s, t)
            except Exception:
                loi += 1
    return loi

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

def khoang(db):
    """Khoảng tốc độ học và số tầng lớn nhất đã thử, để kiểm tra khoảng tìm kiếm có được áp dụng đúng."""
    try:
        con = sqlite3.connect(db)
        lo, hi = con.execute("select min(param_value), max(param_value) from trial_params where param_name like '%lr'").fetchone()
        tang = con.execute("select max(param_value) from trial_params where param_name like '%layer_dim'").fetchone()[0]
        return f'(đã thử lr {lo:.1e}–{hi:.1e}, tối đa {int(tang)} tầng)'
    except Exception:
        return ''

def chay(khoa, bindings):
    local, drv = f'{LOCAL}/{khoa}', f'{DRIVE}/{khoa}'
    base, base_drive = os.path.join(local, NAME, TASK, MODEL), os.path.join(drv, NAME, TASK, MODEL)
    dong_bo(base_drive, base, ghi_de=False)        # lấy các lượt đã có trên Drive về máy Colab
    db, n = best_db(base)
    cmd = ['uv', 'run', 'icu-benchmarks', 'train', '-d', f'demo_data/aki/{NAME}', '-n', NAME,
           '-t', 'BinaryClassification', '-tn', TASK, '-m', MODEL, '--seed', '1111',
           '-l', local, '--tune', '-gc', '-lc']
    if db:
        cmd += ['--hp-checkpoint', os.path.relpath(db, base)]
    if bindings:
        cmd += ['-hp', *bindings]
    print(f'{gio_vn()} 2/3 {khoa}: bắt đầu, đã có {n}/30 lượt tinh chỉnh', flush=True)
    os.makedirs(local, exist_ok=True)
    logf = os.path.join(local, f'gs2_{khoa}_{time.strftime("%m%d-%H%M")}.log')
    with open(logf, 'w') as f:
        p = subprocess.Popen(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    p.wait(timeout=600)
                    break
                except subprocess.TimeoutExpired:
                    dong_bo(base, base_drive)          # chép sang Drive mỗi 10 phút
                    runs = sorted(glob.glob(os.path.join(base, '20*')))
                    db_now = best_db(base)
                    print(f'    {gio_vn()} tinh chỉnh {db_now[1]}/30 lượt, '
                          f'huấn luyện chính {so_fold(runs[-1]) if runs else 0}/25 fold {khoang(db_now[0])}', flush=True)
        except KeyboardInterrupt:
            subprocess.run(['pkill', '-f', 'icu-benchmarks train'])
            dong_bo(base, base_drive)
            raise
    for _ in range(5):                                 # chép lần cuối, thử lại nếu Drive báo lỗi
        if dong_bo(base, base_drive) == 0:
            break
        time.sleep(30)
    try:
        shutil.copy2(logf, drv)
    except Exception:
        pass
    if p.returncode != 0:
        print(f'    LỖI (mã {p.returncode}). 30 dòng cuối của log:')
        print(''.join(open(logf).readlines()[-30:]))
        return False
    print(f'    {gio_vn()} Xong: {so_fold(final_run(drv))}/25 fold đã lưu trên Drive', flush=True)
    return True

for khoa, ten, bindings, _ in LUOT:
    for lan in range(5):   # thử lại nếu Drive lỗi tạm thời
        if final_run(f'{DRIVE}/{khoa}'):
            print(f'{gio_vn()} 2/3 {khoa}: đã xong 25/25 fold, bỏ qua')
            break
        if chay(khoa, bindings):
            break
        print(f'    {gio_vn()} chờ 1 phút rồi chạy tiếp...', flush=True)
        time.sleep(60)

# Bước 3. Bảng so sánh (± là sai số chuẩn qua 25 fold) và cấu hình được chọn
rows = [{'lượt': 'Bài báo (Bảng 11)', 'AUROC': '63.0 ± 0.8', 'AUPRC': '17.3 ± 0.6', 'cùng không gian, mã c61fad0': ''}]
for khoa, ten, bindings, cu in LUOT:
    run = final_run(f'{DRIVE}/{khoa}')
    m = json.load(open(os.path.join(run, 'accumulated_test_metrics.json'))) if run else None
    rows.append({'lượt': ten + ' (6c9c19f4)',
                 'AUROC': f"{100*m['avg']['AUC']:.1f} ± {100*m['std']['AUC']:.1f}" if m else 'chưa có',
                 'AUPRC': f"{100*m['avg']['PR']:.1f} ± {100*m['std']['PR']:.1f}" if m else 'chưa có',
                 'cùng không gian, mã c61fad0': f'{cu[0]} ± {cu[1]} / {cu[2]} ± {cu[3]}'})
bang = pd.DataFrame(rows)
bang.to_csv(os.path.join(DRIVE, 'gru_sua_loi.csv'), index=False)
print(f'\n{gio_vn()} 3/3 GRU, eICU demo, AKI, seed 1111 (AUROC / AUPRC %):')
print(bang.to_string(index=False))
for khoa, ten, bindings, _ in LUOT:
    run = final_run(f'{DRIVE}/{khoa}')
    if not run:
        continue
    cfg = open(os.path.join(run, 'repetition_0', 'fold_0', 'train_config.gin')).read().splitlines()
    print(f'\n{ten}, cấu hình được chọn {khoang(best_db(os.path.join(DRIVE, khoa, NAME, TASK, MODEL))[0])}:')
    print('\n'.join('    ' + l for l in cfg if re.search(r'hidden_dim|layer_dim|\blr\b', l)))
