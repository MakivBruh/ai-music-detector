import argparse
import subprocess
import csv
import errno
import hashlib
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path, PureWindowsPath
import soundfile as sf
import numpy as np
from .audio import AudioConfig, features, sample_augmentation

BATCH = 100  # tracks per cache folder (and per Drive sync part)


def dataset_index(root):
    if not root.is_dir():
        raise ValueError(f'Folder audio tidak ditemukan: {root}')
    names = defaultdict(list)
    for path in root.rglob('*'):
        if path.suffix.lower() in {'.wav', '.mp3', '.flac', '.ogg', '.aif', '.aiff', '.opus'} and path.is_file():
            names[path.name.casefold()].append(path)
            names[path.stem.casefold()].append(path)
    return names


def parse_path_map(items):
    r"""['C:\old\dir=/content/new', ...] -> [(old, new)] for prefix rewriting."""
    pairs = []
    for item in items or []:
        old, sep, new = item.partition('=')
        if not sep or not old:
            raise ValueError(f'path-map harus berbentuk LAMA=BARU: {item}')
        pairs.append((old.replace('\\', '/').rstrip('/'), new.rstrip('/\\')))
    return pairs


def map_path(raw, path_map):
    """Rewrite a manifest path made on another machine (e.g. Windows -> Colab)."""
    text = raw.replace('\\', '/')
    for old, new in path_map or []:
        if text.casefold() == old.casefold() or text.casefold().startswith(old.casefold() + '/'):
            return new + text[len(old):]
    return raw


def is_absolute(raw):
    return Path(raw).is_absolute() or PureWindowsPath(raw).is_absolute()


def read_rows(manifests, audio_root, available_only=False, path_map=None):
    missing = 0
    rows = []
    indexes = {}
    for manifest in manifests:
        manifest = Path(manifest).resolve()
        with manifest.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or [])
            native = {'path', 'group_id', 'label', 'split'} <= fields
            if not native and not {'filename', 'id', 'target', 'label', 'split'} <= fields:
                raise ValueError(f'Header CSV tidak dikenali: {manifest}')
            root = Path(audio_root).resolve() if audio_root else manifest.parent
            for row in reader:
                if native:
                    raw = map_path(row['path'], path_map)
                    if is_absolute(raw) and not Path(raw).is_absolute():
                        raise ValueError(f'Path Windows tidak dikenali di sistem ini, pakai --path-map: {raw}')
                    row['path'] = str((manifest.parent / raw).resolve())
                    if available_only and not Path(row['path']).is_file():
                        missing += 1
                        continue
                else:
                    filename = (row['filename'] or '').strip()
                    target = (row['target'] or '').strip()
                    description = (row['label'] or '').strip().lower()
                    expected = {'full fake': '1', 'real': '0', 'human': '0'}.get(description)
                    if target not in ('0', '1') or (expected is not None and target != expected):
                        raise ValueError(f'Label/target tidak sesuai: {filename} ({description}, {target})')
                    path = root / filename
                    if not path.is_file():
                        if root not in indexes:
                            indexes[root] = dataset_index(root)
                        matches = indexes[root].get(Path(filename).name.casefold(), [])
                        if not matches and available_only:
                            missing += 1
                            continue
                        if len(matches) != 1:
                            raise ValueError(f'{filename}: ditemukan {len(matches)} file audio di {root}')
                        path = matches[0]
                    row = dict(path=str(path), group_id=(row['id'] or '').strip(),
                               label=target, split=(row['split'] or '').strip(),
                               source=row.get('source', ''), generator=row.get('algorithm', ''))
                if row['split'] == 'valid':
                    row['split'] = 'val'
                rows.append(row)
    if missing:
        print(f'Dilewati: {missing} baris tanpa audio lokal.')
    return rows


def sample_tracks(rows, limit, seed):
    if limit < 6:
        raise ValueError('Minimal 6 lagu untuk dua kelas di tiga split.')
    buckets = defaultdict(list)
    for row in rows:
        buckets[row['label'], row['split']].append(row)
    rng = random.Random(seed)
    keys = list(buckets)
    sizes = {key: len(buckets[key]) for key in keys}
    counts = dict.fromkeys(keys, 1)
    class_counts = {'0': 3, '1': 3}
    for _ in range(min(limit, len(rows)) - len(keys)):
        available = [key for key in keys if counts[key] < sizes[key]]
        key = min(available, key=lambda k: (class_counts[k[0]], counts[k] / sizes[k]))
        counts[key] += 1
        class_counts[key[0]] += 1
    return [row for key in keys for row in rng.sample(buckets[key], counts[key])]


def track_variants(row, digest, augment_variants, seed):
    """Train tracks get K random codec chains; val/test stay one clean version."""
    if row['split'] != 'train' or augment_variants < 1:
        return [None]
    return [sample_augmentation(f'{seed}:{digest}:{v}') for v in range(augment_variants)]


def batch_folder(number):
    return f'batch-{number // BATCH:05d}'


def marker_path(output, number):
    """Per-track result file; its presence means the track is finished (used by --resume)."""
    return output / batch_folder(number) / f'{number:06d}.json'


def process_track(job):
    """Write all feature windows of one track; runs in a worker process."""
    number, row, path, digest, variants, output, config, max_segments, skip_unreadable, float16 = job
    folder = batch_folder(number)
    (output / folder).mkdir(exist_ok=True)
    entries, failure = [], None
    try:
        for variant, augment in enumerate(variants):
            count = 0
            for segment, feature in enumerate(features(path, config, max_segments=max_segments, augment=augment)):
                name = f'{folder}/{number:06d}_v{variant}_{segment:05d}.npy'
                np.save(output / name, feature.astype('float16' if float16 else 'float32'))
                entries.append(dict(row, file=name, track_id=str(number), segment=segment, sha256=digest,
                                    variant=variant, augment=json.dumps(augment) if augment else ''))
                count += 1
            if not count:
                raise ValueError(f'Empty audio: {path}')
    except Exception as error:
        if isinstance(error, OSError) and error.errno == errno.ENOSPC:
            raise  # disk penuh bukan masalah file ini; jangan lewati semua lagu satu per satu
        readable_error = (sf.LibsndfileError, subprocess.CalledProcessError, subprocess.TimeoutExpired)
        if not skip_unreadable:
            if isinstance(error, readable_error):
                raise ValueError(f'Gagal membaca audio: {path}') from error
            raise
        for entry in entries:
            (output / entry['file']).unlink()
        entries, failure = [], dict(path=str(path), split=row['split'], label=row['label'], error=str(error))
    marker = marker_path(output, number)
    partial = marker.with_suffix('.json.part')
    partial.write_text(json.dumps(dict(entries=entries, failure=failure)), encoding='utf-8')
    partial.replace(marker)  # atomic: a half-written marker never counts as finished
    return number, len(entries), failure


def build_plan(manifests, audio_root, max_tracks, seed, skip_missing, path_map, drop_duplicates, partial_splits=False):
    """Validate the manifests and fix the ordered track list (with sha256) once."""
    rows = read_rows(manifests, audio_root, available_only=max_tracks is not None or skip_missing,
                     path_map=path_map)
    if not rows:
        raise ValueError('Empty manifest')
    groups, hashes = {}, {}
    conflicting = set()
    resolved = []
    for row in rows:
        split, group, label = row['split'], row['group_id'].strip(), row['label']
        if split not in ('train', 'val', 'test') or label not in ('0', '1') or not group:
            raise ValueError(f'Invalid row: {row}')
        if group in groups and groups[group] != split:
            if max_tracks is None:
                raise ValueError(f'Group leakage: {group}')
            conflicting.add(group)
        groups[group] = split
    if conflicting:
        before = len(rows)
        rows = [r for r in rows if r['group_id'].strip() not in conflicting]
        print(f'Dilewati: {len(conflicting)} grup lintas split ({before - len(rows)} lagu).')
    if not partial_splits:
        for split in ('train', 'val', 'test'):
            if {r['label'] for r in rows if r['split'] == split} != {'0', '1'}:
                raise ValueError(f'{split} must contain both labels')
    if max_tracks is not None:
        rows = sample_tracks(rows, max_tracks, seed)
    duplicates = []
    for row in rows:
        split = row['split']
        path = Path(row['path'])
        with path.open('rb') as audio_file:
            digest = hashlib.file_digest(audio_file, 'sha256').hexdigest()
        if digest in hashes:
            if not drop_duplicates:
                raise ValueError(f'Duplicate audio bytes: {path} (sama dengan {hashes[digest][1]}); '
                                 f'pakai --drop-duplicates untuk menyimpan salinan pertama saja')
            # Same bytes from two sources (e.g. MUSEBENCH "OOD FMA" vs FMA-small): keep the first copy,
            # so the song cannot appear in two splits.
            duplicates.append(dict(path=str(path), split=split, kept=str(hashes[digest][1]),
                                   kept_split=hashes[digest][0]))
            continue
        hashes[digest] = (split, path)
        resolved.append(dict(row, sha256=digest))
    if duplicates:
        print(f'Dilewati: {len(duplicates)} file duplikat byte-identik (salinan pertama disimpan).', flush=True)
    if not partial_splits:
        for split in ('train', 'val', 'test'):
            if {r['label'] for r in resolved if r['split'] == split} != {'0', '1'}:
                raise ValueError(f'{split} must contain both labels')
    if not resolved:
        raise ValueError('Tidak ada lagu yang bisa diproses.')
    return resolved, duplicates


def prepare(manifest, output, audio_root=None, max_tracks=None, seed=42, skip_unreadable=False,
            max_segments=None, canonical_mp3=False, augment_variants=0, loudness_norm=False,
            workers=1, float16=False, path_map=None, skip_missing=False, fmax=None, drop_duplicates=False,
            resume=False, max_new_tracks=None, partial_splits=False):
    """Build the feature cache. Returns True when every track is done.

    With resume=True the work is split into sessions: plan.json fixes the track list,
    every finished track leaves a marker file, and a rerun only processes the rest
    (at most max_new_tracks per call). index.json is written once all tracks are done.
    """
    if max_segments is not None and max_segments < 1:
        raise ValueError("max-segments harus positif")
    if augment_variants < 0 or workers < 1:
        raise ValueError('augment-variants >= 0 dan workers >= 1')
    if max_new_tracks is not None and max_new_tracks < 1:
        raise ValueError('max-new-tracks harus positif')
    output = Path(output).resolve()
    manifests = [manifest] if isinstance(manifest, (str, Path)) else manifest
    config = AudioConfig(canonical_mp3=canonical_mp3, loudness_norm=loudness_norm, fmax=fmax)
    settings = dict(seed=seed, max_tracks=max_tracks, max_segments=max_segments, augment_variants=augment_variants,
                    float16=float16, skip_unreadable=skip_unreadable, audio_config=asdict(config))
    plan_file = output / 'plan.json'
    fresh = not output.exists() or (resume and not any(output.iterdir()))
    if not fresh and not (resume and plan_file.exists()):
        raise ValueError('Output must be a new directory; avoid stale cached features. '
                         '(Gunakan --resume untuk melanjutkan cache yang punya plan.json.)')
    if plan_file.exists():
        plan = json.loads(plan_file.read_text(encoding='utf-8'))
        if plan['settings'] != settings:
            raise ValueError(f'Pengaturan berbeda dari plan.json yang ada: {plan["settings"]} vs {settings}')
        print(f"Melanjutkan plan {plan_file} ({len(plan['tracks'])} lagu).", flush=True)
    else:
        tracks, duplicates = build_plan(manifests, audio_root, max_tracks, seed, skip_missing, path_map,
                                        drop_duplicates, partial_splits)
        output.mkdir(parents=True, exist_ok=True)
        plan = dict(settings=settings, manifests=[str(m) for m in manifests], path_map=path_map or [],
                    tracks=tracks, duplicates=duplicates)
        partial = output / 'plan.json.part'
        partial.write_text(json.dumps(plan), encoding='utf-8')
        partial.replace(plan_file)
        if duplicates:
            (output / 'duplicates.json').write_text(json.dumps(duplicates, indent=2), encoding='utf-8')
    tracks = plan['tracks']
    pending = [n for n in range(len(tracks)) if not marker_path(output, n).exists()]
    todo = pending[:max_new_tracks] if max_new_tracks else pending
    if todo:
        print(f'{len(tracks) - len(pending)}/{len(tracks)} lagu sudah selesai; memproses {len(todo)} lagu.',
              flush=True)
    jobs = [(n, {k: v for k, v in tracks[n].items() if k != 'sha256'}, Path(tracks[n]['path']),
             tracks[n]['sha256'], track_variants(tracks[n], tracks[n]['sha256'], augment_variants, seed),
             output, config, max_segments, skip_unreadable, float16) for n in todo]
    pool = ProcessPoolExecutor(workers) if workers > 1 and len(jobs) > 1 else None
    finished = len(tracks) - len(pending)
    try:
        results = pool.map(process_track, jobs, chunksize=4) if pool else map(process_track, jobs)
        for done, (_, _, failure) in enumerate(results, 1):
            if failure:
                print(f"Dilewati: {failure['path']} ({failure['error']})", flush=True)
            if done % 500 == 0:
                print(f'{finished + done}/{len(tracks)} lagu selesai', flush=True)
    except OSError as error:
        if error.errno == errno.ENOSPC:
            raise SystemExit(f'Disk penuh saat menulis cache di {output}. Kosongkan ruang (mis. hapus audio yang '
                             f'sudah tidak dipakai) atau kurangi --augment-variants / --max-segments.') from error
        raise
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    remaining = len(pending) - len(todo)
    if remaining:
        print(f'Sebagian selesai: {len(tracks) - remaining}/{len(tracks)} lagu. '
              f'Jalankan lagi dengan --resume untuk {remaining} lagu sisanya.', flush=True)
        return False
    index, skipped = [], []
    for n in range(len(tracks)):
        result = json.loads(marker_path(output, n).read_text(encoding='utf-8'))
        index.extend(result['entries'])
        if result['failure']:
            skipped.append(result['failure'])
    if skipped:
        (output / 'skipped.json').write_text(json.dumps(skipped, indent=2), encoding='utf-8')
        if not partial_splits and len({(item['split'], item['label']) for item in index}) != 6:
            raise ValueError('Setelah melewati audio rusak, ada split yang kekurangan kelas.')
    (output / 'index.json').write_text(json.dumps(index, indent=2), encoding='utf-8')
    (output / 'audio_config.json').write_text(json.dumps(asdict(config), indent=2), encoding='utf-8')
    (output / 'prepare_config.json').write_text(json.dumps(dict(
        manifests=plan['manifests'], seed=seed, max_tracks=max_tracks, max_segments=max_segments,
        augment_variants=augment_variants, float16=float16, path_map=plan['path_map']), indent=2), encoding='utf-8')
    print(f'Prepared {len(index)} segments from {len(tracks) - len(skipped)} tracks')
    return True

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', nargs='+', required=True)
    p.add_argument('--audio-root', help='Folder induk audio untuk CSV dataset asli')
    p.add_argument('--output', required=True)
    p.add_argument('--max-tracks', type=int, help='Batas total lagu dari semua manifest')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--skip-unreadable', action='store_true', help='Lewati lagu yang gagal didekode dan catat path')
    p.add_argument('--skip-missing', action='store_true', help='Lewati baris manifest yang file audionya tidak ada')
    p.add_argument('--drop-duplicates', action='store_true',
                   help='Simpan hanya salinan pertama file byte-identik (mis. OOD FMA MUSEBENCH = FMA-small)')
    p.add_argument('--max-segments', type=int, help='Maksimum segmen pertama per lagu (per varian)')
    p.add_argument('--canonical-mp3', action='store_true', help='Terapkan MP3 mono 16 kHz 32 kbps untuk semua kelas dan inferensi')
    p.add_argument('--augment-variants', type=int, default=0,
                   help='K varian codec acak (MP3/AAC/Opus/PCM, bitrate, resample, gain, lowpass) per lagu train; val/test tidak diaugmentasi')
    p.add_argument('--loudness-norm', action='store_true', help='Normalisasi RMS per jendela 5 detik (juga dipakai saat inferensi)')
    p.add_argument('--fmax', type=float, help='Batas atas mel dalam Hz (disarankan 7000 untuk SONICS; default Nyquist)')
    p.add_argument('--workers', type=int, default=1, help='Proses paralel untuk ffmpeg dan log-mel')
    p.add_argument('--float16', action='store_true', help='Simpan fitur float16 (separuh ruang disk)')
    p.add_argument('--partial-splits', action='store_true',
                   help='Cache tambahan: boleh hanya berisi sebagian split/label (mis. lagu umpan balik, split train)')
    p.add_argument('--resume', action='store_true',
                   help='Lanjutkan cache yang sudah punya plan.json (lagu yang selesai tidak diproses ulang)')
    p.add_argument('--max-new-tracks', type=int,
                   help='Proses paling banyak N lagu baru lalu berhenti (dicicil; pakai dengan --resume)')
    p.add_argument('--path-map', nargs='*', default=[], metavar='LAMA=BARU',
                   help='Ganti awalan path manifest, mis. "C:\\Users\\me\\dataset=/content/data"')
    a = p.parse_args()
    complete = prepare(a.manifest, a.output, a.audio_root, a.max_tracks, a.seed, a.skip_unreadable, a.max_segments,
            a.canonical_mp3, a.augment_variants, a.loudness_norm, a.workers, a.float16,
            parse_path_map(a.path_map), a.skip_missing, a.fmax, a.drop_duplicates, a.resume, a.max_new_tracks, a.partial_splits)
    raise SystemExit(0 if complete else 3)  # 3 = sebagian selesai, jalankan lagi dengan --resume
