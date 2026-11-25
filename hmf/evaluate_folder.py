"""Evaluasi OOD per lagu dan tes robustness re-encode.

Folder `--ai` berisi lagu AI asli (label 1), `--human` berisi lagu manusia
(label 0). Setiap lagu diskor dalam kondisi `clean` (file apa adanya) lalu
dikompres ulang dengan beberapa codec. Model yang memakai codec sebagai
penentu label akan menunjukkan selisih skor besar dan prediksi yang berbalik.

Contoh:
  python -m hmf.evaluate_folder --run runs/hmf-28000 runs/hmf-aug \
      --ai data/ood/ai --human data/ood/human --max-segments 12
"""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import soundfile as sf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from .audio import _ffmpeg
from .predict import load_run, segment_scores

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.opus', '.aac', '.aif', '.aiff', '.wma', '.webm'}
CONDITIONS = {
    # The exact chain that pushed FMA songs from 0.00 to 0.86 on runs/hmf-28000.
    'mp3_32k_mono16k': dict(codec='mp3', bitrate=32, sample_rate=16000, channels=1, gain_db=0.0, lowpass=None),
    'mp3_128k': dict(codec='mp3', bitrate=128, sample_rate=44100, channels=2, gain_db=0.0, lowpass=None),
    'aac_128k': dict(codec='aac', bitrate=128, sample_rate=44100, channels=2, gain_db=0.0, lowpass=None),
    'opus_64k': dict(codec='opus', bitrate=64, sample_rate=48000, channels=2, gain_db=0.0, lowpass=None),
}


def list_audio(folder, label, limit=None):
    if not folder:
        return []
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f'Folder tidak ditemukan: {folder}')
    files = sorted(p for p in folder.rglob('*') if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file())
    return [(p, label) for p in files[:limit]]


def readable(path, folder):
    """libsndfile cannot open m4a/opus/wma; decode those once to WAV."""
    try:
        with sf.SoundFile(path):
            return path
    except sf.LibsndfileError:
        decoded = Path(folder) / 'clean.wav'
        _ffmpeg(['-i', str(path), '-map', '0:a:0', '-map_metadata', '-1', '-c:a', 'pcm_f32le', str(decoded)])
        return decoded


def score_track(model, config, path, conditions, max_segments):
    with TemporaryDirectory(prefix='hmf-eval-') as folder:
        source = readable(path, folder)
        scores = {'clean': float(np.mean(segment_scores(model, config, source, max_segments)))}
        for name in conditions:
            scores[name] = float(np.mean(segment_scores(model, config, source, max_segments, CONDITIONS[name])))
    return scores


def condition_metrics(labels, scores, clean):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    pred = (scores >= 0.5).astype(int)
    out = dict(tracks=len(labels), accuracy=float(accuracy_score(labels, pred)),
               mean_p_ai_human=float(scores[labels == 0].mean()) if (labels == 0).any() else None,
               mean_p_ai_ai=float(scores[labels == 1].mean()) if (labels == 1).any() else None)
    if len(set(labels.tolist())) == 2:
        out.update(balanced_accuracy=float(balanced_accuracy_score(labels, pred)),
                   roc_auc=float(roc_auc_score(labels, scores)))
    if clean is not None:
        delta = scores - np.asarray(clean)
        out.update(mean_delta=float(delta.mean()), mean_abs_delta=float(np.abs(delta).mean()),
                   max_abs_delta=float(np.abs(delta).max()),
                   flip_rate=float((pred != (np.asarray(clean) >= 0.5)).mean()))
        for label, name in ((0, 'human'), (1, 'ai')):
            if (labels == label).any():
                out[f'mean_delta_{name}'] = float(delta[labels == label].mean())
    return out


def evaluate_run(run, tracks, conditions, max_segments, output):
    model, config, variant = load_run(run)
    rows = []
    for number, (path, label) in enumerate(tracks, 1):
        try:
            scores = score_track(model, config, path, conditions, max_segments)
        except Exception as error:  # one broken file should not stop the report
            print(f'GAGAL {path}: {error}', flush=True)
            continue
        row = dict(file=str(path), label=label, **{f'p_ai_{k}': v for k, v in scores.items()})
        for name in conditions:
            row[f'delta_{name}'] = scores[name] - scores['clean']
        rows.append(row)
        changes = ' '.join(f"{n}={scores[n]:.2f}" for n in conditions)
        print(f"[{number}/{len(tracks)}] {'AI' if label else 'HU'} clean={scores['clean']:.2f} {changes} | {path.name}",
              flush=True)
    if not rows:
        raise ValueError('Tidak ada lagu yang berhasil dievaluasi.')
    labels = [r['label'] for r in rows]
    clean = [r['p_ai_clean'] for r in rows]
    summary = dict(run=str(run), model=variant, audio_config=config.__dict__, max_segments=max_segments,
                   conditions={'clean': condition_metrics(labels, clean, None)})
    for name in conditions:
        summary['conditions'][name] = dict(condition_metrics(labels, [r[f'p_ai_{name}'] for r in rows], clean),
                                           chain=CONDITIONS[name])
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'per_track.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'\n== {run} ==')
    print(f"{'kondisi':16} {'acc':>6} {'bal_acc':>7} {'auc':>6} {'p_ai(hu)':>9} {'p_ai(ai)':>9} {'|Δ|':>6} {'flip':>6}")
    for name, m in summary['conditions'].items():
        def fmt(key, width):
            value = m.get(key)
            return f'{value:{width}.3f}' if value is not None else ' ' * (width - 1) + '-'
        print(f"{name:16} {fmt('accuracy', 6)} {fmt('balanced_accuracy', 7)} {fmt('roc_auc', 6)} "
              f"{fmt('mean_p_ai_human', 9)} {fmt('mean_p_ai_ai', 9)} {fmt('mean_abs_delta', 6)} {fmt('flip_rate', 6)}")
    print(f'Hasil: {output}')
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', nargs='+', required=True, help='satu atau lebih folder run untuk dibandingkan')
    p.add_argument('--ai', help='folder lagu AI (label 1), mis. data/ood/ai')
    p.add_argument('--human', help='folder lagu manusia (label 0), mis. data/ood/human')
    p.add_argument('--conditions', nargs='*', default=list(CONDITIONS), choices=list(CONDITIONS),
                   help='kondisi re-encode; kosongkan (--conditions) untuk hanya clean')
    p.add_argument('--max-segments', type=int, help='hanya N jendela 5 detik pertama per lagu (lebih cepat)')
    p.add_argument('--limit', type=int, help='maksimum lagu per kelas')
    p.add_argument('--output', help='default: <run>/ood-eval; dengan beberapa run: <output>/<nama run>')
    a = p.parse_args(argv)
    tracks = list_audio(a.ai, 1, a.limit) + list_audio(a.human, 0, a.limit)
    if not tracks:
        raise ValueError('Berikan --ai dan/atau --human yang berisi audio.')
    print(f"{sum(t[1] for t in tracks)} lagu AI, {sum(1 - t[1] for t in tracks)} lagu human, "
          f"kondisi: clean + {a.conditions}", flush=True)
    results = []
    for run in a.run:
        run = Path(run)
        if a.output:
            output = Path(a.output) / run.name if len(a.run) > 1 else Path(a.output)
        else:
            output = run / 'ood-eval'
        results.append(evaluate_run(run, tracks, a.conditions, a.max_segments, output))
    return results


if __name__ == '__main__':
    main()
