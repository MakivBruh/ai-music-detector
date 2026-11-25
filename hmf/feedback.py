"""Umpan balik pengguna studio: simpan, ringkas, dan ekspor untuk evaluasi.

Setiap umpan balik = satu baris JSON di <folder>/feedback.jsonl (hanya ditambah,
tidak pernah ditimpa). Untuk lagu (sha256) dan model (run) yang sama, baris
terbaru yang dipakai, sehingga pengguna bisa mengoreksi jawabannya.

Ini adalah *uji lapangan*: lagu dipilih pengguna, bukan sampel acak, dan
labelnya berasal dari pengakuan pengguna. Jangan dicampur dengan angka test
resmi, dan jangan menguji model dengan lagu yang ikut dipakai melatihnya.

Contoh:
  python -m hmf.feedback report                      # ringkasan per model
  python -m hmf.feedback export --run hmf-aug-v1 --only-wrong --ood data/ood
  python -m hmf.feedback export --manifest data/feedback/manifest.csv --split train
"""
import argparse
import csv
import json
import math
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parents[1] / 'data' / 'feedback'
LABEL_SOURCES = {
    'dibuat_sendiri': 'Saya yang membuat atau merekamnya',
    'label_platform': 'Label resmi dari platform atau penerbit',
    'sumber_dataset': 'Dari dataset dengan asal-usul jelas',
    'pembuat_mengonfirmasi': 'Pembuatnya mengonfirmasi',
    'lainnya': 'Lainnya (jelaskan di catatan)',
}
ANSWERS = ('correct', 'wrong', 'unknown')
SHA = re.compile(r'[0-9a-f]{64}')


def safe_name(name):
    """Keep a readable, harmless file name (no folders, no odd characters)."""
    name = Path(str(name)).name
    name = re.sub(r'[^\w.\- ()\[\]]+', '_', name, flags=re.UNICODE).strip(' .')
    return name[:120] or 'audio'


def validate(entry):
    """Check a feedback entry from the studio; returns a clean record or raises ValueError."""
    if not isinstance(entry, dict):
        raise ValueError('Umpan balik harus berupa objek JSON.')
    answer = entry.get('answer')
    if answer not in ANSWERS:
        raise ValueError('Jawaban harus correct, wrong, atau unknown.')
    verdict = entry.get('verdict')
    if verdict not in ('AI', 'Human'):
        raise ValueError('Vonis model tidak dikenali.')
    sha = str(entry.get('sha256', ''))
    if not SHA.fullmatch(sha):
        raise ValueError('sha256 tidak valid; analisis lagu ulang lalu beri umpan balik.')
    source = entry.get('label_source') or None
    if answer != 'unknown' and source not in LABEL_SOURCES:
        raise ValueError('Pilih dari mana kamu yakin akan label lagu ini.')
    score = float(entry.get('ai_score'))
    threshold = float(entry.get('decision_threshold', 0.5))
    if not (0 <= score <= 1 and 0 <= threshold <= 1):
        raise ValueError('Skor di luar rentang 0–1.')
    scores = [float(s) for s in entry.get('segment_scores') or []][:2000]
    model_label = 'ai' if verdict == 'AI' else 'human'
    true_label = None
    if answer == 'correct':
        true_label = model_label
    elif answer == 'wrong':
        true_label = 'human' if model_label == 'ai' else 'ai'
    return dict(
        id=uuid.uuid4().hex, time=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        filename=safe_name(entry.get('filename', 'audio')), sha256=sha, run=str(entry.get('run', ''))[:200],
        model=str(entry.get('model', ''))[:100], verdict=verdict, ai_score=score, decision_threshold=threshold,
        segments_analyzed=len(scores), segment_scores=scores, answer=answer, true_label=true_label,
        label_source=source if answer != 'unknown' else None,
        note=str(entry.get('note', ''))[:500], audio_saved=False)


def append(folder, record):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'feedback.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    return record


def load(folder):
    """Latest record per (sha256, run); older answers for the same song and model are corrections."""
    path = Path(folder) / 'feedback.jsonl'
    if not path.exists():
        return []
    latest = {}
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a half-written line from a crash should not break the panel
            key = (record.get('sha256'), record.get('run'))
            if record.get('kind') == 'audio':
                if key in latest:
                    latest[key] = dict(latest[key], audio_saved=True, audio_path=record['audio_path'])
                continue
            previous = latest.get(key)
            if previous and previous.get('audio_saved'):  # a corrected answer keeps the copy already saved
                record = dict(record, audio_saved=True, audio_path=previous.get('audio_path'))
            latest[key] = record
    return sorted(latest.values(), key=lambda r: r['time'])


def wilson(successes, total, z=1.96):
    """95% Wilson interval; honest for the small samples typical of feedback."""
    if not total:
        return None
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def summarize(records, run=None):
    """Field-test metrics for one model (run); records answered "unknown" are counted but not scored."""
    mine = [r for r in records if run is None or r.get('run') == run]
    known = [r for r in mine if r.get('true_label')]
    cell = lambda truth, said: sum(1 for r in known if r['true_label'] == truth and r['verdict'].lower() == said)
    confusion = dict(human_as_human=cell('human', 'human'), human_as_ai=cell('human', 'ai'),
                     ai_as_ai=cell('ai', 'ai'), ai_as_human=cell('ai', 'human'))
    correct = confusion['human_as_human'] + confusion['ai_as_ai']
    humans = confusion['human_as_human'] + confusion['human_as_ai']
    ais = confusion['ai_as_ai'] + confusion['ai_as_human']
    wrong = [dict(filename=r['filename'], time=r['time'], ai_score=r['ai_score'], verdict=r['verdict'],
                  true_label=r['true_label'], label_source=r['label_source'], note=r.get('note', ''),
                  audio_saved=bool(r.get('audio_saved')))
             for r in reversed(known) if r['answer'] == 'wrong']
    others = sorted({r.get('run') for r in records if run is not None and r.get('run') != run})
    return dict(run=run, total=len(mine), scored=len(known), unknown=len(mine) - len(known), correct=correct,
                accuracy=correct / len(known) if known else None, accuracy_ci95=wilson(correct, len(known)),
                false_accusation_rate=confusion['human_as_ai'] / humans if humans else None,
                missed_ai_rate=confusion['ai_as_human'] / ais if ais else None,
                mean_score_wrong=(sum(r['ai_score'] for r in known if r['answer'] == 'wrong') / len(wrong)
                                  if wrong else None),
                confusion=confusion, wrong=wrong[:50], other_runs=others,
                other_runs_total=sum(1 for r in records if run is not None and r.get('run') != run))


def save_audio(folder, record, data, original_name):
    """Keep a copy of the song, filed by its true label, only after the user agreed to it."""
    label = record.get('true_label')
    if label not in ('ai', 'human'):
        raise ValueError('Audio hanya disimpan untuk lagu yang labelnya diketahui.')
    folder = Path(folder)
    target = folder / 'audio' / label / f"{record['sha256'][:12]}_{safe_name(original_name)}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    append(folder, dict(kind='audio', sha256=record['sha256'], run=record['run'],
                        audio_path=str(target.relative_to(folder)).replace('\\', '/'),
                        time=datetime.now(timezone.utc).isoformat(timespec='seconds')))
    return target


def report_command(a):
    records = load(a.dir)
    if not records:
        print(f'Belum ada umpan balik di {a.dir}.')
        return
    for run in sorted({r['run'] for r in records}):
        s = summarize(records, run)
        acc = '-' if s['accuracy'] is None else f"{s['accuracy']:.1%}"
        ci = '' if not s['accuracy_ci95'] else f" (95%: {s['accuracy_ci95'][0]:.0%}–{s['accuracy_ci95'][1]:.0%})"
        c = s['confusion']
        print(f"\n== {run} ==\n{s['scored']} lagu dinilai, {s['unknown']} 'tidak tahu' | akurasi {acc}{ci}")
        print(f"manusia→human {c['human_as_human']}, manusia→AI {c['human_as_ai']} (salah tuduh), "
              f"AI→AI {c['ai_as_ai']}, AI→human {c['ai_as_human']} (lolos)")
        for w in s['wrong']:
            print(f"  SALAH  {w['filename']}  skor {w['ai_score']:.3f}  seharusnya {w['true_label']}  ({w['label_source']})")
    if a.csv:
        path = Path(a.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ['time', 'run', 'filename', 'sha256', 'verdict', 'ai_score', 'answer', 'true_label', 'label_source',
                  'note', 'audio_saved']
        with path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(records)
        print(f'\nCSV: {path}')


def export_command(a):
    records = [r for r in load(a.dir) if r.get('true_label') and r.get('audio_saved')
               and (a.run is None or r['run'] == a.run) and (not a.only_wrong or r['answer'] == 'wrong')]
    if not records:
        print('Tidak ada lagu dengan audio tersimpan yang cocok dengan filter ini.')
        return
    folder = Path(a.dir)
    if a.ood:
        for r in records:
            source = folder / r['audio_path']
            target = Path(a.ood) / r['true_label'] / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        print(f'{len(records)} lagu disalin ke {a.ood}/ai dan {a.ood}/human (untuk hmf.evaluate_folder).')
    if a.manifest:
        out = Path(a.manifest)
        out.parent.mkdir(parents=True, exist_ok=True)
        seen = set()
        with out.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, ['path', 'group_id', 'label', 'split', 'source', 'generator'])
            writer.writeheader()
            for r in records:
                if r['sha256'] in seen:  # same song judged under two models = one training example
                    continue
                seen.add(r['sha256'])
                path = (folder / r['audio_path']).resolve()
                writer.writerow(dict(path=str(path), group_id=f"feedback_{r['sha256'][:16]}",
                                     label='1' if r['true_label'] == 'ai' else '0', split=a.split,
                                     source='feedback', generator=''))
        print(f'Manifest {len(seen)} lagu ditulis ke {out} (split={a.split}).')
        print('Buat cache kecilnya dengan pengaturan yang SAMA seperti cache utama, misalnya:')
        print(f'  python -m hmf.prepare --manifest "{out}" --output data/cache-feedback-1 --augment-variants 3 '
              '--max-segments 6 --loudness-norm --fmax 7000 --float16 --seed 42 --partial-splits --drop-duplicates')
        print('lalu latih dengan: python -m hmf.train --cache <cache-utama> data/cache-feedback-1 ...')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dir', default=str(DEFAULT_DIR), help='folder umpan balik (default data/feedback)')
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('report', help='ringkasan akurasi uji lapangan per model')
    r.add_argument('--csv', help='tulis semua umpan balik ke CSV')
    r.set_defaults(func=report_command)
    e = sub.add_parser('export', help='salin lagu berlabel ke folder OOD dan/atau tulis manifest training')
    e.add_argument('--run', help='hanya umpan balik untuk run ini')
    e.add_argument('--only-wrong', action='store_true', help='hanya lagu yang salah dinilai model')
    e.add_argument('--ood', help='folder tujuan berisi ai/ dan human/, mis. data/ood')
    e.add_argument('--manifest', help='tulis manifest CSV untuk hmf.prepare')
    e.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    e.set_defaults(func=export_command)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == '__main__':
    main()
