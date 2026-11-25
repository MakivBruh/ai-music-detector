"""Pembuat manifest multi-sumber dan split leave-one-generator-out (LOGO).

Subperintah:
  musebench  pindai folder hasil `scripts/musebench_index.py fetch`, beri label
             dari nama folder, isi source/generator, dan turunkan group_id dari
             nama lagu dasar sehingga lagu Jamendo dan semua generasi AI yang
             memakai nama file yang sama berada di satu group dan satu split.
  logo       gabungkan manifest apa pun (mis. SONICS + MUSEBENCH) lalu pindahkan
             satu/lebih generator ke test saja.

Contoh:
  python -m hmf.manifest musebench --root /content/musebench \
      --reference-manifest data/manifest-28000.csv
  python -m hmf.manifest logo --manifest data/manifest-28000.csv \
      /content/musebench/manifest-musebench.csv --holdout musicgen \
      --output /content/manifest-logo-musicgen.csv --path-map "C:\\Users\\me=/content/data"
"""
import argparse
import csv
import hashlib
import html
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.opus', '.aac'}
FIELDS = ['path', 'group_id', 'label', 'split', 'source', 'generator', 'subset']
# Folders that only perturb AI audio (codec/EQ/noise/loudness/pitch). Including them
# would make "perturbed" an AI-only cue again, so they are skipped by default.
PERTURBATION_FOLDERS = {'aac_generated_music', 'eq_generated_music', 'gaussion_noised_music',
                        'gaussian_noised_music', 'loudness_generation_music', 'pitch_time_generated_music'}
AMBIGUOUS_FOLDERS = {'ai_human_co-created_music'}
GENERATOR_NAMES = {'musicgen': 'musicgen', 'audioldm': 'audioldm', 'riffusion': 'riffusion',
                   'magnet': 'magnet', 'jasco': 'jasco', 'magenta': 'magenta',
                   'stable_audio': 'stable_audio', 'suno': 'suno', 'udio': 'udio'}
GENERATED = re.compile(r'(.+?)_generated_music', re.IGNORECASE)


def base_name(filename):
    """'Song - Artist.mp3.wav' -> 'song - artist' (shared by the human original)."""
    name = filename
    for _ in range(2):  # generations are saved as <original>.mp3.wav
        stem, ext = os.path.splitext(name)
        if ext.lower() not in AUDIO_EXTENSIONS:
            break
        name = stem
    name = unicodedata.normalize('NFC', html.unescape(name))
    return ' '.join(name.split()).casefold()


def family_of(subset):
    key = subset.casefold()
    if key in ('jamendo', 'jamedo'):
        return 'jamendo'
    if key.startswith('genius'):
        return 'genius'
    return key or 'root'


def generator_of(folder):
    match = GENERATED.fullmatch(folder)
    if not match:
        return None
    raw = match.group(1).casefold()
    return GENERATOR_NAMES.get(raw, raw)


def classify(parts, include_perturbed=False):
    """Return (label, source, generator, subset) or (None, reason, ...) for a path under the root."""
    top = parts[0]
    key = top.casefold()
    if key == 'jamendo':
        return '0', 'musebench-jamendo', '', 'Jamendo'
    if key.startswith('ood'):
        return '0', 'musebench-ood-fma', '', top
    if key in AMBIGUOUS_FOLDERS:
        return None, 'ai+human (ambigu)', None, None
    if key in PERTURBATION_FOLDERS:
        if not include_perturbed or len(parts) < 3:
            return None, f'perturbasi {top}', None, None
        generator = generator_of(parts[1]) or parts[1].casefold()
        subset = parts[2] if len(parts) > 3 else ''
        return '1', f'musebench-{key}', generator, subset
    generator = generator_of(top)
    if generator is None:
        return None, f'folder tidak dikenal {top}', None, None
    return '1', 'musebench', generator, parts[1] if len(parts) > 2 else ''


def group_for(parts, label, subset):
    name = parts[-1]
    if parts[0].casefold().startswith('ood'):
        match = re.match(r'(\d+)_', name)
        if match:  # same id format as FMA rows in manifest-28000 (fma_000615)
            return f'fma_{int(match.group(1)):06d}'
    family = 'jamendo' if label == '0' and parts[0].casefold() == 'jamendo' else family_of(subset)
    return f'mb-{family}:{base_name(name)}'


def scan_musebench(root, include_perturbed=False, exclude_generators=('suno', 'udio')):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f'Folder MUSEBENCH tidak ditemukan: {root}')
    excluded = {g.casefold() for g in exclude_generators or ()}
    rows, skipped = [], Counter()
    for path in sorted(root.rglob('*')):
        if path.suffix.lower() not in AUDIO_EXTENSIONS or not path.is_file() or path.name.endswith('.part'):
            continue
        parts = path.relative_to(root).parts
        if len(parts) < 2:
            skipped['file di akar folder'] += 1
            continue
        label, source, generator, subset = classify(parts, include_perturbed)
        if label is None:
            skipped[source] += 1
            continue
        if generator and generator in excluded:
            skipped[f'generator dikecualikan {generator}'] += 1
            continue
        rows.append(dict(path=str(path), group_id=group_for(parts, label, subset), label=label, split='',
                         source=source, generator=generator, subset=subset))
    return rows, skipped


def split_for(group, seed, val_ratio, test_ratio):
    value = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:12], 16) / 16 ** 12
    if value < test_ratio:
        return 'test'
    if value < test_ratio + val_ratio:
        return 'val'
    return 'train'


def read_reference(paths):
    """group_id -> split from existing manifests, so shared groups (e.g. FMA ids) agree."""
    reference = {}
    for path in paths or []:
        with open(path, encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                split = {'valid': 'val'}.get(row['split'], row['split'])
                group = (row.get('group_id') or row.get('id') or '').strip()
                if group:
                    reference[group] = split
    return reference


def assign_splits(rows, seed=42, val_ratio=0.15, test_ratio=0.15, reference=None):
    reference = reference or {}
    for row in rows:
        group = row['group_id']
        row['split'] = reference.get(group) or split_for(group, seed, val_ratio, test_ratio)
    return rows


def leave_one_generator_out(rows, holdout, drop_seen_test=True):
    """Move every group that contains a held-out generator to test.

    Train/val then never see the held-out generator nor its paired human songs.
    With drop_seen_test, AI rows of seen generators are removed from test so the
    headline test score measures only the unseen generator against human audio.
    """
    held = {h.casefold() for h in holdout}
    found = {(r.get('generator') or '').casefold() for r in rows if r['label'] == '1'}
    missing = held - found
    if missing:
        raise ValueError(f'Generator tidak ada di manifest: {sorted(missing)}; tersedia: {sorted(found)}')
    test_groups = {r['group_id'] for r in rows if (r.get('generator') or '').casefold() in held}
    out, dropped = [], 0
    for row in rows:
        row = dict(row)
        is_held = (row.get('generator') or '').casefold() in held
        if row['group_id'] in test_groups:
            row['split'] = 'test'
        if drop_seen_test and row['label'] == '1' and not is_held and row['split'] == 'test':
            dropped += 1
            continue
        out.append(row)
    if dropped:
        print(f'LOGO: {dropped} lagu AI dari generator yang terlihat dibuang dari test.')
    return out


def check_manifest(rows):
    splits = {}
    for row in rows:
        previous = splits.setdefault(row['group_id'], row['split'])
        if previous != row['split']:
            raise ValueError(f"Group lintas split: {row['group_id']} ({previous}/{row['split']})")
    for split in ('train', 'val', 'test'):
        labels = {r['label'] for r in rows if r['split'] == split}
        if labels != {'0', '1'}:
            raise ValueError(f'Split {split} tidak memiliki kedua label (ada: {sorted(labels)})')


def summarize(rows):
    counts = Counter((r['split'], r['label'], r.get('generator') or r.get('source') or '-') for r in rows)
    print(f"{'split':6} {'label':6} {'generator/source':28} {'lagu':>7}")
    for (split, label, name), n in sorted(counts.items()):
        print(f"{split:6} {'ai' if label == '1' else 'human':6} {name:28} {n:7d}")
    print(f'TOTAL {len(rows)} lagu, {len({r["group_id"] for r in rows})} group')


def write_manifest(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, FIELDS, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            row = dict(row)
            try:  # relative paths keep the manifest valid when the folder is moved
                row['path'] = os.path.relpath(row['path'], output.parent.resolve())
            except ValueError:  # different Windows drive
                pass
            writer.writerow({k: row.get(k, '') for k in FIELDS})
    print(f'Manifest ditulis: {output}')


def musebench_command(a):
    rows, skipped = scan_musebench(a.root, a.include_perturbed, a.exclude_generator)
    for reason, n in sorted(skipped.items()):
        print(f'Dilewati {n:6d}: {reason}')
    if not rows:
        raise ValueError('Tidak ada audio MUSEBENCH yang bisa dipakai.')
    assign_splits(rows, a.seed, a.val_ratio, a.test_ratio, read_reference(a.reference_manifest))
    if a.holdout:
        rows = leave_one_generator_out(rows, a.holdout, not a.keep_seen_test)
    if not a.allow_incomplete:
        check_manifest(rows)
    summarize(rows)
    write_manifest(rows, a.output or Path(a.root) / 'manifest-musebench.csv')


def logo_command(a):
    from .prepare import parse_path_map, read_rows
    rows = read_rows(a.manifest, None, available_only=a.skip_missing, path_map=parse_path_map(a.path_map))
    rows = leave_one_generator_out(rows, a.holdout, not a.keep_seen_test)
    check_manifest(rows)
    summarize(rows)
    write_manifest(rows, a.output)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('musebench', help='buat manifest dari folder MUSEBENCH')
    m.add_argument('--root', required=True, help='folder tujuan fetch (berisi Jamendo/, OOD FMA/, *_generated_music/)')
    m.add_argument('--output', help='default: <root>/manifest-musebench.csv')
    m.add_argument('--seed', type=int, default=42)
    m.add_argument('--val-ratio', type=float, default=0.15)
    m.add_argument('--test-ratio', type=float, default=0.15)
    m.add_argument('--exclude-generator', nargs='*', default=['suno', 'udio'],
                   help='default suno udio (sudah ada di SONICS)')
    m.add_argument('--include-perturbed', action='store_true',
                   help='ikutkan folder perturbasi AI-saja (aac/eq/noise/loudness/pitch); tidak disarankan')
    m.add_argument('--reference-manifest', nargs='*', default=[],
                   help='manifest lain; group_id yang sama mewarisi split-nya (mis. FMA di manifest-28000)')
    m.add_argument('--holdout', nargs='*', default=[], help='generator yang hanya muncul di test (LOGO)')
    m.add_argument('--keep-seen-test', action='store_true', help='jangan buang AI generator lain dari test saat LOGO')
    m.add_argument('--allow-incomplete', action='store_true',
                   help='izinkan split tanpa salah satu label (mis. hanya mengunduh AI); gabungkan dengan manifest lain')
    m.set_defaults(func=musebench_command)
    g = sub.add_parser('logo', help='gabungkan manifest lalu terapkan leave-one-generator-out')
    g.add_argument('--manifest', nargs='+', required=True)
    g.add_argument('--holdout', nargs='+', required=True)
    g.add_argument('--output', required=True)
    g.add_argument('--path-map', nargs='*', default=[], metavar='LAMA=BARU')
    g.add_argument('--skip-missing', action='store_true')
    g.add_argument('--keep-seen-test', action='store_true')
    g.set_defaults(func=logo_command)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == '__main__':
    main()
