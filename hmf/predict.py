import argparse
import base64
import json
from time import perf_counter
from pathlib import Path
import numpy as np
import torch
from .audio import AudioConfig, features
from .model import CNN


def load_run(run):
    """Return (model, AudioConfig, variant) for a training run folder."""
    run = Path(run)
    report = json.loads((run / 'report.json').read_text())
    config = AudioConfig(**report['audio_config'])
    variant = report['args'].get('model', 'baseline')
    model = CNN(variant).eval()
    model.load_state_dict(torch.load(run / 'best.pt', map_location='cpu', weights_only=True))
    return model, config, variant


@torch.no_grad()
def segment_scores(model, config, audio, max_segments=None, augment=None, batch_size=32, return_windows=False):
    device = next(model.parameters()).device
    windows = [f.astype('float32') for f in features(audio, config, max_segments=max_segments, augment=augment)]
    scores = []
    for start in range(0, len(windows), batch_size):
        x = torch.from_numpy(np.stack(windows[start:start + batch_size]))[:, None].to(device)
        scores.extend(model(x).softmax(1)[:, 1].cpu().tolist())
    return (scores, windows) if return_windows else scores


def spectrogram_preview(windows, config, frames_per_segment=40, max_segments=180):
    """The log-mel windows the CNN saw, shrunk for display: uint8, mel bins x time, base64."""
    shown = windows[:max_segments]
    parts = []
    for w in shown:  # w: (n_mels, frames); average neighbouring frames
        edges = np.linspace(0, w.shape[1], frames_per_segment + 1).astype(int)
        parts.append(np.stack([w[:, a:b].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1))
    image = np.concatenate(parts, axis=1)  # features are scaled to [-1, 1]
    pixels = np.clip((image + 1) * 127.5, 0, 255).astype(np.uint8)
    return dict(mels=int(pixels.shape[0]), frames_per_segment=frames_per_segment, segments=len(shown),
                fmin=0, fmax=config.fmax or config.sample_rate / 2,
                db_range=[-80, 20], data=base64.b64encode(pixels.tobytes()).decode())


def analyze(run, audio, *, filename=None, include_segments=False, include_spectrogram=False):
    started = perf_counter()
    model, config, variant = load_run(run)
    scores, windows = segment_scores(model, config, audio, return_windows=True)
    if not scores:
        raise ValueError('Empty audio')
    score = sum(scores) / len(scores)
    threshold = 0.5
    result = dict(verdict='AI' if score >= threshold else 'Human',
                  ai_score=score, decision_threshold=threshold,
                  aggregation='mean_segment_score', segment_seconds=config.seconds,
                  segments_analyzed=len(scores))
    if include_segments:
        result['segment_scores'] = scores
    extra = {}
    if include_spectrogram:
        extra['visual'] = dict(spectrogram=spectrogram_preview(windows, config))
    return dict(**extra,status='completed', filename=filename or Path(audio).name,
                analysis_level='track_classification', result=result,
                metadata=dict(model=f'{variant}_cnn', models_used=1, run=Path(run).name,
                              processing_time_seconds=round(perf_counter() - started, 3)))


def predict(run, audio):
    result = analyze(run, audio)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--audio', required=True)
    a = p.parse_args()
    predict(a.run, a.audio)


def model_summary(run):
    """Training facts for the studio's model panel (works for old reports too)."""
    run = Path(run)
    report = json.loads((run / 'report.json').read_text())
    test = report.get('test', {})
    keys = ('accuracy', 'balanced_accuracy', 'macro_f1', 'roc_auc', 'loss')
    history = [dict(epoch=h['epoch'], train_loss=h.get('train_loss'),
                    val_loss=h.get('val', {}).get('loss'), val_macro_f1=h.get('val', {}).get('macro_f1'))
               for h in report.get('history', [])]
    labels = {}
    for p in report.get('predictions', []):
        labels[p['label']] = labels.get(p['label'], 0) + 1
    return dict(run=run.name, model=f"{report['args'].get('model', 'baseline')}_cnn",
                audio_config=report.get('audio_config', {}), best_epoch=report.get('best_epoch'),
                epochs_completed=report.get('epochs_completed', len(history)), stop_reason=report.get('stop_reason'),
                balance=report['args'].get('balance', 'none'), monitor=report['args'].get('monitor'),
                test={k: test[k] for k in keys if k in test}, confusion_matrix=test.get('confusion_matrix'),
                test_tracks=dict(human=labels.get(0, 0), ai=labels.get(1, 0)),
                per_generator=test.get('per_generator'), history=history)
