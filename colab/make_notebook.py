"""Generate colab/train.ipynb.  Usage: python colab/make_notebook.py colab/train.ipynb"""
import json
import sys
from pathlib import Path

cells = []


def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=text.strip('\n').splitlines(keepends=True)))


def code(text):
    cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                      source=text.strip('\n').splitlines(keepends=True)))


md(r"""
# HMF — training tahan-codec di Google Colab

Notebook ini melatih ulang CNN deteksi musik AI **tanpa** membiarkan codec/kualitas file menjadi penentu label.
Pekerjaannya dibagi dua dan **bisa dicicil lintas sesi**:

**A. Sesi CPU (boleh berkali-kali)** — tidak butuh GPU:
1. salin MP3 SONICS + FMA dari Drive ke `/content`, unduh subset MUSEBENCH,
2. buat manifest multi-sumber, lalu `prepare` (augmentasi codec untuk **kedua kelas**),
3. setiap ±`CHUNK_TRACKS` lagu, cache fitur disimpan ke Drive (`HMF/cache/<RUN_NAME>/`).
   Kalau sesi putus, jalankan notebook lagi dari atas: lagu yang sudah selesai tidak diproses ulang,
   dan audio hanya diunduh untuk lagu yang belum selesai.

**B. Sesi GPU (setelah prepare selesai)**: jalankan dari atas lagi; notebook memulihkan cache dari Drive,
melewati unduhan audio, lalu `train` → tes robustness → evaluasi OOD.

Ubah **hanya sel konfigurasi**. Selalu jalankan sel berurutan dari atas.
""")

code(r"""
# ====== KONFIGURASI (ubah di sini) ======
DRIVE = '/content/drive/MyDrive/HMF'
CODE_DIR = f'{DRIVE}/ai-music-detector-colab.zip'  # zip kode (dist/ai-music-detector-colab.zip) atau folder proyek
# Manifest lama (path Windows, dipetakan lewat AUDIO_SOURCES); sudah ikut di dalam zip kode.
SONICS_MANIFEST = '/content/ai-music-detector/data/manifest-28000.csv'

# Awalan path Windows di manifest -> lokasi audio di Drive (folder atau .zip).
AUDIO_SOURCES = {
    r'C:\Users\makiv\OneDrive\Documents\Coding\projek\dataset\fake_songs': f'{DRIVE}/fake_songs.zip',  # 20.000 MP3 AI SONICS (zip atau folder)
    r'C:\Users\makiv\Music\FMA\fma_small': f'{DRIVE}/fma_small.zip',                                    # 8.000 MP3 FMA-small (zip atau folder)
}

USE_MUSEBENCH = True
MUSEBENCH_INDEX = f'{DRIVE}/musebench_index.csv'  # dibuat sekali (lama), lalu dipakai ulang dari Drive
# Folder tingkat atas MUSEBENCH -> jumlah file yang diambil acak (None = semua); dikecilkan otomatis jika disk tidak cukup.
# Cek nama persisnya di output sel "ringkasan indeks"; suno/udio tidak perlu (sama dengan SONICS).
MUSEBENCH_AI = {
    'Musicgen_Generated_Music': 800,
    'audioldm_generated_music': 800,
    'riffusion_generated_music': 800,
    'magenta_generated_music': 800,
    'jasco_generated_music': 800,
    'stable_audio_generated_music': 800,
    'MagNet_Generated_Music': 800,
}
JAMENDO_EXTRA = 1000   # lagu Jamendo tanpa pasangan AI (selain pasangan generasi Jamedo yang terpilih)
OOD_FMA = 800          # lagu manusia dari folder 'OOD FMA'

HOLDOUT = []           # mis. ['musicgen'] untuk leave-one-generator-out; [] = split biasa

AUGMENT_VARIANTS = 3   # varian codec acak per lagu train (val/test tidak diaugmentasi); 2 = prepare ~30% lebih cepat
MAX_SEGMENTS = 6       # jendela 5 detik pertama per lagu per varian (6 = 30 detik)
FMAX = 7000            # batas atas mel (Hz); AI SONICS kosong di atas ~7,3 kHz, jadi pita itu = petunjuk codec
CHUNK_TRACKS = 1000    # simpan cache ke Drive setiap N lagu (sesi putus = paling banyak N lagu diulang)
EPOCHS = 30
EXTRA_CACHES = []      # cache tambahan di Drive (folder atau .zip), mis. [f'{DRIVE}/cache-feedback-1.zip'] dari lagu umpan balik
SKIP_TRAIN_EVAL = True  # lewati penilaian ulang seluruh data train tiap epoch: epoch ±2x lebih cepat (train_acc kosong)
BATCH_SIZE = 64
RUN_NAME = 'hmf-aug-v1' + (f"-logo-{'-'.join(HOLDOUT)}" if HOLDOUT else '')
CACHE_DRIVE = f'{DRIVE}/cache/{RUN_NAME}'       # cache fitur bertahap (±20–25 GB saat selesai)
OLD_RUN = f'{DRIVE}/runs/hmf-28000-9947e77d'   # model lama untuk perbandingan (boleh tidak ada)
OOD_AI = f'{DRIVE}/ood/ai'                      # lagu AI baru (mis. Suno v4/v5, Udio terbaru)
OOD_HUMAN = f'{DRIVE}/ood/human'                # lagu manusia di luar FMA
ROBUST_SAMPLE = 100    # lagu test per kelas untuk tes robustness re-encode
SEED = 42
""")

md("## 1. Drive, kode, status cache")
code(r"""
import os, shutil, subprocess, sys, json, random, csv
from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')

LOCAL = Path('/content/work'); LOCAL.mkdir(exist_ok=True)
DATA = LOCAL / 'data'; DATA.mkdir(exist_ok=True)
MB = LOCAL / 'musebench'
CACHE = LOCAL / f'cache-{RUN_NAME}'
CACHE_DRIVE = Path(CACHE_DRIVE)
PARTS = CACHE_DRIVE / 'parts'
ROBUST_DRIVE = CACHE_DRIVE / 'robust'
RUN = Path(DRIVE) / 'runs' / RUN_NAME
FINAL_FILES = ('index.json', 'audio_config.json', 'prepare_config.json', 'skipped.json', 'duplicates.json')

def disk():
    total, used, free = shutil.disk_usage('/content')
    print(f'Disk /content: {free / 1e9:.0f} GB kosong dari {total / 1e9:.0f} GB')
    return free

def sh(cmd, ok=(0,)):
    # Teruskan output subprocess baris per baris; tanpa ini Colab hanya menampilkan CalledProcessError.
    print('$', cmd, flush=True)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    for line in process.stdout:
        print(line, end='', flush=True)
    if process.wait() not in ok:
        raise RuntimeError(f'Perintah gagal (exit {process.returncode}); lihat pesan di atas.')
    return process.returncode

disk()
!nvidia-smi --query-gpu=name,memory.total --format=csv || echo 'Tanpa GPU (cukup untuk prepare; training butuh GPU)'
print('CPU:', os.cpu_count(), 'core')

CODE = Path('/content/ai-music-detector')
if CODE.exists():
    shutil.rmtree(CODE)
if CODE_DIR.endswith('.zip'):
    shutil.unpack_archive(CODE_DIR, CODE)
else:
    shutil.copytree(CODE_DIR, CODE, ignore=shutil.ignore_patterns('data', 'runs', '.venv', 'backups', '__pycache__'))
    manifest = Path(CODE_DIR, 'data', 'manifest-28000.csv')
    if manifest.exists():
        (CODE / 'data').mkdir(exist_ok=True)
        shutil.copy(manifest, CODE / 'data' / manifest.name)
assert Path(SONICS_MANIFEST).exists(), f'Manifest tidak ditemukan: {SONICS_MANIFEST}'
assert '--max-new-tracks' in (CODE / 'hmf' / 'prepare.py').read_text(), 'Zip kode di Drive versi lama; unggah ulang'
assert '--resume' in (CODE / 'hmf' / 'train.py').read_text(), 'Zip kode di Drive versi lama (tanpa resume training); unggah ulang'
os.chdir(CODE)
sys.path.insert(0, str(CODE))
!pip -q install "librosa>=0.10.2,<1" "soundfile>=0.12,<1" "imageio-ffmpeg>=0.6,<1"
""")
code(r"""
# Pulihkan cache yang sudah tersimpan di Drive (dari sesi sebelumnya).
PARTS.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
restored_log = CACHE / '.restored'
restored = set(restored_log.read_text().split()) if restored_log.exists() else set()
parts = sorted(PARTS.glob('batch-*.tar'))
todo = [p for p in parts if p.name not in restored]
for i, part in enumerate(todo, 1):
    sh(f'tar -xf "{part}" -C "{CACHE}"')
    with restored_log.open('a') as f:
        f.write(part.name + '\n')
    if i % 20 == 0:
        print(f'{i}/{len(todo)} bagian dipulihkan', flush=True)
for name in ('plan.json',) + FINAL_FILES:
    if (CACHE_DRIVE / name).exists() and not (CACHE / name).exists():
        shutil.copy(CACHE_DRIVE / name, CACHE / name)

PLAN = json.loads((CACHE / 'plan.json').read_text()) if (CACHE / 'plan.json').exists() else None
PREPARE_DONE = (CACHE / 'index.json').exists()
BATCH = 100
def marker(n):
    return CACHE / f'batch-{n // BATCH:05d}' / f'{n:06d}.json'
if PLAN:
    PENDING = [t['path'] for n, t in enumerate(PLAN['tracks']) if not marker(n).exists()]
    print(f"Cache: {len(PLAN['tracks']) - len(PENDING)}/{len(PLAN['tracks'])} lagu selesai, "
          f'{len(parts)} bagian di Drive' + (' — PREPARE SELESAI' if PREPARE_DONE else ''))
else:
    PENDING = None
    print('Cache belum ada: sesi pertama, semua audio akan disiapkan.')

def needed(prefix):
    # Apakah masih ada lagu belum selesai yang audionya di bawah prefix ini?
    return not PREPARE_DONE and (PENDING is None or any(p.startswith(str(prefix)) for p in PENDING))
_, _, drive_free = shutil.disk_usage('/content/drive/MyDrive')
print(f'Sisa ruang Drive: {drive_free / 1e9:.0f} GB (cache butuh ±20–25 GB)')
disk()
""")

md("## 2. Salin audio SONICS + FMA dari Drive ke `/content` (dilewati jika tidak diperlukan)")
code(r"""
PATH_MAP = []
with open(SONICS_MANIFEST, encoding='utf-8-sig', newline='') as f:
    manifest_rows = list(csv.DictReader(f))

def find_source(source):
    # Terima 'x.zip' atau folder 'x', mana pun yang ada di Drive.
    source = Path(source)
    base = source.with_suffix('') if source.suffix == '.zip' else source
    for candidate in (source, base.with_name(base.name + '.zip'), base):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Tidak ditemukan {base} atau {base}.zip. Isi {base.parent}: '
                            f'{sorted(p.name for p in base.parent.iterdir())}')

for old, source in AUDIO_SOURCES.items():
    source = find_source(source)
    target = DATA / source.name.removesuffix('.zip')
    if not needed(target):
        print(f'Lewati {source.name}: semua lagunya sudah ada di cache.')
        continue
    if not target.exists():
        if source.suffix == '.zip':
            # Salin zip ke disk lokal dulu: unzip langsung dari Drive sangat lambat.
            local_zip = LOCAL / source.name
            sh(f'cp "{source}" "{local_zip}"')
            sh(f'unzip -q "{local_zip}" -d "{target}"')
            local_zip.unlink()
        else:
            sh(f'cp -r "{source}" "{target}"')
    # Cari folder yang cocok dengan path di manifest (zip bisa berisi folder induk, bertingkat).
    prefix = old.replace('\\', '/').casefold() + '/'
    sample = next(r['path'].replace('\\', '/')[len(prefix):] for r in manifest_rows
                  if r['path'].replace('\\', '/').casefold().startswith(prefix))
    roots = [target] + sorted(d for d in target.rglob('*') if d.is_dir() and len(d.relative_to(target).parts) <= 3)
    match = next((d for d in roots if (d / sample).exists()), None)
    assert match, f'{sample} tidak ditemukan di dalam {source.name}; cek isi zip'
    target = match
    count = sum(1 for _ in target.rglob('*.mp3'))
    PATH_MAP.append(f'{old}={target}')
    print(old, '->', target, f'({count} mp3)')
    assert count, f'Tidak ada MP3 di {target}; cek isi zip {source.name}'
disk()
""")

md("""
## 3. MUSEBENCH: indeks, pilih subset, unduh

Indeks dibuat sekali lewat HTTP Range (tidak mengunduh tar 10 GB) dan disimpan di Drive. Pilihan lagu disimpan
di Drive juga, sehingga sesi lanjutan hanya mengunduh lagu yang belum selesai diproses.
Generasi di folder `.../Jamedo/` berpasangan dengan lagu `Jamendo/` bernama sama; pasangan itu ikut diunduh.
Folder perturbasi (aac/eq/noise/loudness/pitch) dan `ai_human_co-created_music` tidak dipakai karena hanya
menyentuh lagu AI (akan menjadi petunjuk label baru) atau labelnya ambigu.
""")
code(r"""
if USE_MUSEBENCH and not Path(MUSEBENCH_INDEX).exists():
    sh(f'python scripts/musebench_index.py index --out /content/musebench_index.csv --workers 32')
    shutil.copy('/content/musebench_index.csv', MUSEBENCH_INDEX)
if USE_MUSEBENCH and PLAN is None:
    sh(f'python scripts/musebench_index.py summary --index "{MUSEBENCH_INDEX}" --depth 2')
""")
code(r"""
from collections import defaultdict
from hmf.manifest import base_name
SLACK_GB = 6  # cadangan untuk file sementara ffmpeg, index.json, salinan tes robustness
SELECTION_DRIVE = CACHE_DRIVE / 'musebench_selection.csv'
selection = LOCAL / 'musebench_selection.csv'

def cache_gb(tracks):
    # float16 64x313 = ~42 KB per jendela; train ~75% dengan AUGMENT_VARIANTS varian, val/test 1 versi.
    return tracks * (0.75 * AUGMENT_VARIANTS + 0.25) * MAX_SEGMENTS * 42e3 / 1e9

def write_selection(rows, path):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, ['repo', 'tar', 'path', 'offset', 'size'], extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)

if USE_MUSEBENCH and PLAN is not None:
    # Sesi lanjutan: unduh hanya lagu MUSEBENCH yang belum selesai.
    with open(SELECTION_DRIVE, encoding='utf-8', newline='') as f:
        saved = list(csv.DictReader(f))
    wanted = {p for p in PENDING if p.startswith(str(MB) + '/')} if not PREPARE_DONE else set()
    chosen = [r for r in saved if str(MB / r['path'].strip('./')) in wanted]
    write_selection(chosen, selection)
    print(f'MUSEBENCH yang masih perlu diunduh: {len(chosen)} file, {sum(int(r["size"]) for r in chosen) / 1e9:.1f} GB')
elif USE_MUSEBENCH:
    with open(MUSEBENCH_INDEX, encoding='utf-8', newline='') as f:
        index = list(csv.DictReader(f))
    with open(SONICS_MANIFEST, encoding='utf-8-sig', newline='') as f:
        sonics_tracks = sum(1 for _ in csv.DictReader(f))
    by_top = defaultdict(list)
    for r in index:
        by_top[r['path'].strip('./').split('/')[0].casefold()].append(r)
    name_of = lambda r: base_name(r['path'].split('/')[-1])
    jamendo = by_top['jamendo']
    ood_all = [r for key, rows in by_top.items() if key.startswith('ood') for r in rows]

    def select(scale):
        # Pilihan acak yang sama untuk scale yang sama; semua jumlah dikali scale.
        rng = random.Random(SEED)
        chosen, paired, counts = [], set(), {}
        for folder, limit in MUSEBENCH_AI.items():
            rows = [r for r in by_top.get(folder.casefold(), []) if r['path'].endswith(('.wav', '.mp3', '.flac'))]
            if not rows:
                counts[folder] = 'TIDAK ADA di indeks; cek nama di ringkasan'
                continue
            n = len(rows) if limit is None else min(len(rows), max(1, int(limit * scale)))
            rows = rng.sample(rows, n)
            chosen += rows
            paired |= {name_of(r) for r in rows if '/jamedo/' in r['path'].casefold()}
            counts[folder] = n
        pairs = [r for r in jamendo if name_of(r) in paired]
        rest = [r for r in jamendo if name_of(r) not in paired]
        extra = rng.sample(rest, min(int(JAMENDO_EXTRA * scale), len(rest)))
        ood = rng.sample(ood_all, min(int(OOD_FMA * scale), len(ood_all)))
        counts.update({'Jamendo berpasangan': len(pairs), 'Jamendo tambahan': len(extra), 'OOD FMA': len(ood)})
        return chosen + pairs + extra + ood, counts

    free_gb = disk() / 1e9
    scale = 1.0
    for _ in range(40):  # kecilkan 15% per langkah sampai unduhan + cache muat di disk
        chosen, counts = select(scale)
        size_gb = sum(int(r['size']) for r in chosen) / 1e9
        need_gb = size_gb + cache_gb(sonics_tracks + len(chosen)) + SLACK_GB
        if need_gb <= free_gb:
            break
        scale *= 0.85
    for key, value in counts.items():
        print(f'{key}: {value}')
    print(f'Skala jumlah: {scale:.2f} | unduhan {size_gb:.1f} GB + cache ~{cache_gb(sonics_tracks + len(chosen)):.1f} GB'
          f' + cadangan {SLACK_GB} GB = {need_gb:.1f} GB dari {free_gb:.0f} GB kosong')
    assert need_gb <= free_gb, 'Disk tetap tidak cukup; turunkan AUGMENT_VARIANTS atau MAX_SEGMENTS'
    write_selection(chosen, selection)
    shutil.copy(selection, SELECTION_DRIVE)
""")
code(r"""
if USE_MUSEBENCH and sum(1 for _ in open(selection, encoding='utf-8')) > 1:
    sh(f'python scripts/musebench_index.py fetch --index "{selection}" --prefix "" --dest "{MB}" --workers 16')
    disk()
""")

md("## 4. Manifest multi-sumber (+ opsional leave-one-generator-out)")
code(r"""
path_map_args = ' '.join(f'"{m}"' for m in PATH_MAP)
if PLAN is not None:
    MANIFESTS = PLAN['manifests']  # plan.json sudah memuat daftar lagu; manifest tidak dibaca ulang
    print('Plan sudah ada; manifest tidak dibuat ulang.')
else:
    MANIFESTS = []
    if USE_MUSEBENCH:
        # Group FMA yang juga ada di manifest SONICS mewarisi split-nya. MUSEBENCH di urutan pertama agar
        # lagunya diproses lebih dulu (sesi lanjutan tidak perlu mengunduhnya lagi).
        sh(f'python -m hmf.manifest musebench --root "{MB}" --seed {SEED} --reference-manifest "{SONICS_MANIFEST}"'
           f' --output "{MB}/manifest-musebench.csv"')
        MANIFESTS.append(f'{MB}/manifest-musebench.csv')
    MANIFESTS.append(SONICS_MANIFEST)
    if HOLDOUT:
        manifest_list = ' '.join(f'"{m}"' for m in MANIFESTS)
        sh(f'python -m hmf.manifest logo --manifest {manifest_list} --holdout {" ".join(HOLDOUT)}'
           f' --path-map {path_map_args} --skip-missing --output "{LOCAL}/manifest-logo.csv"')
        MANIFESTS = [f'{LOCAL}/manifest-logo.csv']
    (CACHE_DRIVE / 'manifests').mkdir(exist_ok=True)
    for m in MANIFESTS:
        shutil.copy(m, CACHE_DRIVE / 'manifests' / Path(m).name)
""")

md("""
## 5. Prepare (dicicil, disimpan ke Drive)

Setiap lagu **train** mendapat `AUGMENT_VARIANTS` rantai acak: MP3/AAC/Opus/PCM, bitrate 32–320 kbps,
resample 16–48 kHz, mono/stereo, gain −12…+4 dB, lowpass acak 4–7,9 kHz (40%). Rantainya ditentukan oleh
seed + sha256 file, jadi hasilnya sama walau dicicil lintas sesi. Val/test satu versi bersih.
`--loudness-norm` menormalkan RMS per jendela (juga saat inferensi). `--fmax 7000` membuang pita di atas 7 kHz:
lagu AI SONICS tidak punya energi di sana, jadi pita itu petunjuk codec yang tidak bisa dihapus dengan augmentasi.

Sel ini memproses `CHUNK_TRACKS` lagu per putaran, lalu menyimpan setiap folder 100 lagu yang lengkap ke Drive
(`parts/batch-XXXXX.tar`). Jika sesi putus, jalankan notebook lagi dari atas.
""")
code(r"""
import time
workers = os.cpu_count()
manifest_list = ' '.join(f'"{m}"' for m in MANIFESTS)

def sync_to_drive():
    plan = json.loads((CACHE / 'plan.json').read_text())
    if not (CACHE_DRIVE / 'plan.json').exists():
        shutil.copy(CACHE / 'plan.json', CACHE_DRIVE / 'plan.json')
    total, saved = len(plan['tracks']), 0
    for b in range((total + BATCH - 1) // BATCH):
        part = PARTS / f'batch-{b:05d}.tar'
        if part.exists():
            continue
        if not all(marker(n).exists() for n in range(b * BATCH, min(total, (b + 1) * BATCH))):
            continue
        partial = PARTS / f'batch-{b:05d}.tar.part'
        sh(f'tar -cf "{partial}" -C "{CACHE}" "batch-{b:05d}"')
        partial.rename(part)
        with restored_log.open('a') as f:
            f.write(part.name + '\n')
        saved += 1
    for name in FINAL_FILES:
        if (CACHE / name).exists():
            shutil.copy(CACHE / name, CACHE_DRIVE / name)
    print(f'Drive: {saved} bagian baru disimpan, total {len(list(PARTS.glob("batch-*.tar")))} bagian', flush=True)

def save_robust_sample():
    # Pilih lagu test untuk tes robustness selagi audionya masih ada (sesi pertama), simpan ke Drive.
    if (ROBUST_DRIVE / 'done').exists():
        return
    plan = json.loads((CACHE / 'plan.json').read_text())
    rng = random.Random(SEED)
    for label, name in (('1', 'ai'), ('0', 'human')):
        paths = sorted(t['path'] for t in plan['tracks'] if t['split'] == 'test' and t['label'] == label
                       and Path(t['path']).exists())
        (ROBUST_DRIVE / name).mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(rng.sample(paths, min(ROBUST_SAMPLE, len(paths)))):
            shutil.copy(p, ROBUST_DRIVE / name / f'{i:04d}_{Path(p).name}')
    (ROBUST_DRIVE / 'done').write_text('ok')
    print('Sampel tes robustness disimpan ke', ROBUST_DRIVE)

started = time.time()
while not (CACHE / 'index.json').exists():
    code = sh(f'python -m hmf.prepare --manifest {manifest_list} --path-map {path_map_args} --output "{CACHE}"'
              f' --augment-variants {AUGMENT_VARIANTS} --max-segments {MAX_SEGMENTS} --loudness-norm --fmax {FMAX}'
              f' --float16 --workers {workers} --seed {SEED} --skip-unreadable --skip-missing --drop-duplicates'
              f' --resume --max-new-tracks {CHUNK_TRACKS}', ok=(0, 3))
    save_robust_sample()
    sync_to_drive()
    print(f'Waktu sesi ini: {(time.time() - started) / 3600:.1f} jam', flush=True)
    disk()
sync_to_drive()
print('PREPARE SELESAI. Lanjutkan di runtime GPU: jalankan notebook ini lagi dari atas.')
""")

md("## 6. Training (butuh GPU; hasil langsung ke Drive)")
code(r"""
import torch
assert (CACHE / 'index.json').exists(), 'Prepare belum selesai; jalankan sel prepare (boleh di runtime CPU) dulu'
assert torch.cuda.is_available(), ('Runtime ini tanpa GPU. Cache sudah aman di Drive; ganti ke runtime GPU '
                                   '(Runtime > Change runtime type) lalu jalankan notebook lagi dari atas.')
# Cache tambahan (mis. lagu umpan balik) dipakai bersama cache utama, tanpa prepare ulang cache utama.
extra = []
for i, source in enumerate(EXTRA_CACHES, 1):
    source, target = Path(source), LOCAL / f'extra-cache-{i}'
    if not target.exists():
        if source.suffix == '.zip':
            shutil.unpack_archive(source, target)
        else:
            shutil.copytree(source, target)
    found = [p.parent for p in target.rglob('index.json')]  # the zip may hold a parent folder
    assert len(found) == 1, f'{source}: harus berisi tepat satu cache (index.json)'
    extra.append(found[0])
caches = ' '.join(f'"{c}"' for c in [CACHE, *extra])
# Run yang terhenti di tengah (punya history.json) dilanjutkan otomatis dari epoch terakhir yang tersimpan.
resume = (RUN / 'history.json').exists()
assert resume or not RUN.exists(), f'{RUN} sudah ada tanpa history.json; ganti RUN_NAME atau pindahkan folder lama'
sh(f'python -m hmf.train --cache {caches} --output "{RUN}" --epochs {EPOCHS} --batch-size {BATCH_SIZE}'
   f' --model frequency --monitor loss --patience 5 --balance sampler --balance-by generator --workers 2 --seed {SEED}'
   + (' --resume' if resume else '') + (' --skip-train-eval' if SKIP_TRAIN_EVAL else ''))
for name in FINAL_FILES:
    if (CACHE / name).exists():
        shutil.copy(CACHE / name, RUN / f'cache_{name}')
for m in (CACHE_DRIVE / 'manifests').glob('*.csv'):
    shutil.copy(m, RUN / m.name)
report = json.loads((RUN / 'report.json').read_text())
print(json.dumps({k: report['test'][k] for k in ('accuracy', 'balanced_accuracy', 'macro_f1', 'roc_auc')}, indent=2))
print(json.dumps(report['test']['per_generator'], indent=2))
""")

md("""
## 7. Tes robustness (test split) dan evaluasi OOD

`clean` = file apa adanya; kondisi lain = file dikompres ulang (MP3 32k mono 16 kHz, MP3 128k, AAC 128k, Opus 64k).
Model yang baik: `|Δ|` kecil dan `flip` ≈ 0. Model lama gagal di `mp3_32k_mono16k` (FMA 0,00 → 0,86).
""")
code(r"""
ROBUST = LOCAL / 'robust'
if not ROBUST.exists():
    shutil.copytree(ROBUST_DRIVE, ROBUST)
runs = f'"{RUN}"' + (f' "{OLD_RUN}"' if Path(OLD_RUN, 'best.pt').exists() else '')
sh(f'python -m hmf.evaluate_folder --run {runs} --ai "{ROBUST}/ai" --human "{ROBUST}/human"'
   f' --max-segments {MAX_SEGMENTS} --output "{RUN}/robustness-test-split"')
""")
code(r"""
if Path(OOD_AI).exists() or Path(OOD_HUMAN).exists():
    OOD = LOCAL / 'ood'
    if not OOD.exists():
        shutil.copytree(Path(OOD_AI).parent, OOD)
    args = (f' --ai "{OOD / Path(OOD_AI).name}"' if Path(OOD_AI).exists() else '') + \
           (f' --human "{OOD / Path(OOD_HUMAN).name}"' if Path(OOD_HUMAN).exists() else '')
    sh(f'python -m hmf.evaluate_folder --run {runs}{args} --output "{RUN}/ood-eval"')
else:
    print(f'Lewati OOD: siapkan {OOD_AI} (lagu AI baru) dan {OOD_HUMAN} (lagu manusia non-FMA) di Drive.')
""")
code(r"""
print('Hasil tersimpan di', RUN)
for p in sorted(RUN.rglob('*')):
    if p.is_file() and p.suffix in ('.json', '.csv', '.pt'):
        print(f'{p.stat().st_size / 1e6:8.2f} MB  {p.relative_to(RUN)}')
""")

nb = dict(cells=cells, metadata=dict(
    accelerator='GPU', colab=dict(provenance=[], gpuType='T4'),
    kernelspec=dict(display_name='Python 3', name='python3'), language_info=dict(name='python')),
    nbformat=4, nbformat_minor=5)
for i, cell in enumerate(nb['cells']):
    cell['id'] = f'cell-{i:02d}'
out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print('written', out)
