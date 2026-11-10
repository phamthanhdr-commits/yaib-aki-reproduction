# BỔ SUNG, ô 3/3. KIỂM CHỨNG: LR và LightGBM trên eICU cao hơn bài báo có phải do khác phiên bản mã và thư viện không?
# Chạy lại 2 mô hình này bằng mã YAIB ngày 26/05/2023 (commit 221b00f4) với đúng thư viện ghi ở phụ lục G.2 của bài báo
# (Python 3.10, PyTorch 2.0, LightGBM 3.3.5, scikit-learn 1.2.2). Vì sao chọn bản này: Bảng 11 có từ bản arXiv đầu tiên
# (08/06/2023) và dùng sai số chuẩn (mã có từ 08/05/2023); từ 30/05/2023 mã chấm LR/LightGBM bằng nhãn 0/1 nên AUROC
# chỉ quanh 50 (lỗi này có trong bản phát hành v0.3.0). Bản 26/05/2023 là bản cuối cùng trước lỗi đó.
# Giữ nguyên như Nhiệm vụ 1: dữ liệu demo (giống hệt từng byte), seed 1111, tinh chỉnh rồi 25 fold, điền 2 ca thiếu giới tính.
# Chỉ khác: phiên bản mã và thư viện. Đã chạy thử trên máy Mac ngày 28/10/2026: LightGBM 73,6 ± 0,5; LR 49,8 ± 0,8.
# Không cần GPU. Khoảng 1–1,5 giờ. Bị ngắt giữa chừng thì chạy lại ô này: máy chạy tiếp từ các lượt tinh chỉnh đã lưu.
import os, glob, json, subprocess, time, datetime
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')

LOGDIR = '/content/drive/MyDrive/YAIB_logs/kiemchung_052023'  # kết quả mã 05/2023, để riêng
NV1DIR = '/content/drive/MyDrive/YAIB_logs'                    # kết quả Nhiệm vụ 1 (mã hiện tại, c61fad0)
COMMIT = '221b00f4'
REPO, ENV = f'/content/YAIB_{COMMIT}', '/content/env2023'
PY = f'{ENV}/bin/python'
MODELS = ['LGBMClassifier', 'LogisticRegression']             # LightGBM trước vì là phép kiểm chứng chính
N_CALLS, N_INIT = 50, 10                                       # mã 2023: 50 lượt tinh chỉnh, 10 lượt đầu bốc ngẫu nhiên
os.makedirs(LOGDIR, exist_ok=True)
ENVVARS = dict(os.environ, WANDB_MODE='disabled', MPLBACKEND='Agg')

def gio_vn():
    return time.strftime('%H:%M', time.gmtime(time.time() + 7 * 3600))

def sh(cmd, logname, cwd=None):
    """Chạy lệnh bash, ghi log ra Drive; lỗi thì in 30 dòng cuối rồi dừng."""
    logf = os.path.join(LOGDIR, logname)
    with open(logf, 'w') as f:
        p = subprocess.run(cmd, shell=True, executable='/bin/bash', cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        print(f'LỖI ở bước {logname}. 30 dòng cuối của log:')
        print(''.join(open(logf).readlines()[-30:]))
        raise RuntimeError('Dừng ở bước trên. Gửi thông báo lỗi này cho người hướng dẫn.')
    return open(logf).read()

def py(code):
    """Chạy một đoạn Python trong môi trường 2023, trong thư mục mã 05/2023."""
    p = subprocess.run([PY, '-c', code], cwd=REPO, capture_output=True, text=True, env=ENVVARS)
    if p.returncode != 0:
        print(p.stderr[-3000:])
        raise RuntimeError('Dừng ở bước trên. Gửi thông báo lỗi này cho người hướng dẫn.')
    return p.stdout.strip()

# Bước 1. Python 3.10 và thư viện năm 2023 (khoảng 5 phút, mỗi phiên Colab làm một lần)
print(f'{gio_vn()} 1/4 Cài Python 3.10 và thư viện như bài báo...', flush=True)
if not os.path.exists(f'{ENV}/.xong'):
    sh(f'pip -q install uv && uv venv -q --python 3.10 {ENV} && '
       f'uv pip install -q --python {PY} --extra-index-url https://download.pytorch.org/whl/cpu '
       '--index-strategy unsafe-best-match "torch==2.0.1+cpu" numpy==1.24.3 pandas==2.0.0 pyarrow==11.0.0 '
       'scikit-learn==1.2.2 lightgbm==3.3.5 scipy==1.10.1 scikit-optimize==0.9.0 gin-config==0.5.0 '
       'pytorch-lightning==2.0.1 torchmetrics==0.11.4 pytorch-ignite==0.4.11 tensorboard==2.12.2 '
       'wandb==0.15.0 tqdm==4.64.1 matplotlib==3.7.1 setuptools==67.8.0 && '
       # recipys: thư viện tiền xử lý của tác giả, bản 05/05/2023 (bản mới nhất vào 26/05/2023)
       f'uv pip install -q --python {PY} --no-deps '
       '"recipys @ git+https://github.com/rvandewater/ReciPies.git@391129cd209b4446e9a1803d60e15bf1b3dd7fe7" && '
       # scikit-optimize 0.9.0 còn dùng np.int, đã bị bỏ từ numpy 1.24: đổi thành int (cùng nghĩa)
       f'T=$({PY} -c "import skopt, os; print(os.path.dirname(skopt.__file__))")/space/transformers.py && '
       f'sed -i "s/astype(np\\.int)/astype(int)/g" $T && touch {ENV}/.xong', 'kc_1_cai_dat.log')

# Bước 2. Mã YAIB ngày 26/05/2023; điền 2 ca thiếu giới tính giống Nhiệm vụ 1
print(f'{gio_vn()} 2/4 Tải mã YAIB ngày 26/05/2023 (commit {COMMIT})...', flush=True)
sh(f'[ -d {REPO} ] || (curl -sSL https://codeload.github.com/rvandewater/YAIB/tar.gz/{COMMIT} | tar -xz -C /content '
   f'&& mv /content/YAIB-{COMMIT}* {REPO})', 'kc_2_tai_ma.log')
print('   ', py("import sys, torch, lightgbm, sklearn, pandas; print('Python', sys.version.split()[0], '| PyTorch', "
                "torch.__version__, '| LightGBM', lightgbm.__version__, '| scikit-learn', sklearn.__version__, "
                "'| pandas', pandas.__version__)"))
print('   ', py("import pandas as pd\n"
                "f = 'demo_data/aki/eicu_demo/sta.parquet'\n"
                "sta = pd.read_parquet(f); n = int(sta['sex'].isna().sum())\n"
                "if n:\n"
                "    sta['sex'] = sta['sex'].fillna(sta['sex'].mode()[0]); sta.to_parquet(f, index=False)\n"
                "print(f'eicu_demo: điền {n} ca thiếu giới tính (0 nghĩa là đã điền từ trước)')"))

# Bước 3. Tinh chỉnh rồi chạy 25 fold cho từng mô hình
def thu_muc(model):
    return os.path.join(LOGDIR, 'eicu_demo', 'AKI', model)

def cac_lan_chay(model):
    return sorted(glob.glob(os.path.join(thu_muc(model), '20*')))

def so_fold(run):
    return len(glob.glob(os.path.join(run, 'repetition_*', 'fold_*', 'test_metrics.json')))

def so_luot(run):
    try:
        return len(json.load(open(os.path.join(run, 'hyperparameter_tuning_logs.json')))['x_iters'])
    except Exception:
        return 0

for model in MODELS:
    runs = cac_lan_chay(model)
    if any(so_fold(r) == 25 for r in runs):
        print(f'{gio_vn()} 3/4 {model}: đã xong 25 fold, bỏ qua')
        continue
    cmd = [PY, '-m', 'icu_benchmarks.run', 'train', '-d', 'demo_data/aki/eicu_demo', '-n', 'eicu_demo',
           '-t', 'BinaryClassification', '-tn', 'AKI', '-m', model, '-s', '1111', '-l', LOGDIR, '--tune', '-gc', '-lc']
    cu = max(runs, key=so_luot, default=None)
    if cu and so_luot(cu) > 0:
        con = N_CALLS - so_luot(cu)
        cmd += ['--checkpoint', os.path.basename(cu)]
        if 0 < con < N_INIT:   # mã 2023 báo lỗi nếu số lượt còn lại ít hơn số lượt bốc ngẫu nhiên ban đầu
            cmd += ['-hp', f'tune_hyperparameters.n_initial_points={con}']
        print(f'{gio_vn()} 3/4 {model}: chạy tiếp, {so_luot(cu)}/{N_CALLS} lượt tinh chỉnh đã xong', flush=True)
    else:
        print(f'{gio_vn()} 3/4 {model}: bắt đầu ({N_CALLS} lượt tinh chỉnh, rồi 25 fold)', flush=True)
    logf = os.path.join(LOGDIR, f'kc_3_{model}.log')
    with open(logf, 'a') as f:
        p = subprocess.Popen(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, env=ENVVARS)
        while p.poll() is None:
            time.sleep(600)   # 10 phút báo một lần
            runs = cac_lan_chay(model)
            if runs:
                print(f'    {gio_vn()} {model}: {max(map(so_luot, runs))}/{N_CALLS} lượt tinh chỉnh, '
                      f'{so_fold(runs[-1])}/25 fold', flush=True)
    if p.returncode != 0:
        print(f'LỖI (mã {p.returncode}). 30 dòng cuối của log:')
        print(''.join(open(logf).readlines()[-30:]))
        raise RuntimeError('Dừng ở bước trên. Gửi thông báo lỗi này cho người hướng dẫn.')
    print(f'{gio_vn()}     {model}: xong', flush=True)

# Bước 4. Bảng so sánh: bài báo (Bảng 11), mã 05/2023, mã hiện tại (Nhiệm vụ 1). ± là sai số chuẩn
PAPER = {'LGBMClassifier': (72.6, 0.3, 23.8, 0.4), 'LogisticRegression': (61.3, 0.6, 16.8, 0.4)}

def ket_qua(goc, model):
    runs = [r for r in glob.glob(os.path.join(goc, 'eicu_demo', 'AKI', model, '20*')) if so_fold(r) == 25]
    if not runs:
        return None
    m = json.load(open(os.path.join(max(runs), 'accumulated_test_metrics.json')))
    return [100 * m['avg']['AUC'], 100 * m['std']['AUC'], 100 * m['avg']['PR'], 100 * m['std']['PR']]

rows = []
for model in MODELS:
    for ten, kq in [('Bài báo (Bảng 11)', list(PAPER[model])),
                    ('Mã 26/05/2023, Colab', ket_qua(LOGDIR, model)),
                    ('Mã hiện tại (c61fad0)', ket_qua(NV1DIR, model))]:
        if kq:
            rows.append({'mô hình': model, 'phiên bản': ten,
                         'AUROC': f'{kq[0]:.1f} ± {kq[1]:.1f}', 'AUPRC': f'{kq[2]:.1f} ± {kq[3]:.1f}',
                         'Δ AUROC so với bài báo': round(kq[0] - PAPER[model][0], 1)})
bang = pd.DataFrame(rows)
bang.to_csv(os.path.join(LOGDIR, 'kiemchung_052023.csv'), index=False)
print(f'\n{gio_vn()} 4/4 Kết quả (AUROC/AUPRC %, eICU demo, AKI):')
print(bang.to_string(index=False))
