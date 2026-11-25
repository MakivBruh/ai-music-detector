import argparse
import json
import random
import shutil
from pathlib import Path
import numpy as np
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score, classification_report
from .model import CNN
from .monitor import TrainingMonitor


def as_caches(cache):
    return [Path(c) for c in ([cache] if isinstance(cache, (str, Path)) else cache)]


def load_caches(cache):
    """Rows of one or more feature caches, ready to train on together.

    The first cache is the main one. A song (sha256) already in an earlier cache is
    skipped in later ones, so e.g. a test song sent back as feedback cannot leak
    into training. All caches must share the same audio settings.
    """
    caches = as_caches(cache)
    configs = [json.loads((c / 'audio_config.json').read_text()) for c in caches]
    for c, config in zip(caches[1:], configs[1:]):
        if config != configs[0]:
            raise ValueError(f'Pengaturan audio {c} berbeda dari {caches[0]}: {config} vs {configs[0]}')
    rows, owner = [], {}
    for i, c in enumerate(caches):
        skipped = set()
        for r in json.loads((c / 'index.json').read_text()):
            sha = r.get('sha256')
            if sha and owner.setdefault(sha, i) != i:
                skipped.add(sha)
                continue
            r['_cache'] = i
            if i:  # track ids restart at 0 in every cache
                r['track_id'] = f"c{i}:{r['track_id']}"
            rows.append(r)
        if skipped:
            print(f'{c}: {len(skipped)} lagu dilewati karena sudah ada di cache sebelumnya.', flush=True)
    return caches, rows, configs[0]


class Segments(Dataset):
    def __init__(self, root, split, rows=None):
        self.roots = as_caches(root)
        self.root = self.roots[0]
        if rows is None:
            rows = load_caches(self.roots)[1]
        self.rows = [r for r in rows if r['split'] == split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = np.load(self.roots[r.get('_cache', 0)] / r['file']).astype(np.float32, copy=False)  # cache may be float16
        return torch.from_numpy(x)[None], int(r['label']), r['track_id']


def balance_key(row, by):
    if by == 'label':
        return row['label']
    return row['label'], row.get('generator') or row.get('source') or ''


def sample_weights(rows, by='label'):
    """Per-segment weights: each label gets half the mass; with by='generator'
    every (label, generator) bucket inside a label also gets an equal share."""
    keys = [balance_key(r, by) for r in rows]
    counts = Counter(keys)
    label_of = (lambda k: k) if by == 'label' else (lambda k: k[0])
    buckets_per_label = Counter(label_of(k) for k in counts)
    labels = len(buckets_per_label)
    return [1.0 / (counts[k] * buckets_per_label[label_of(k)] * labels) for k in keys]


def class_weights(rows):
    counts = Counter(int(r['label']) for r in rows)
    total = sum(counts.values())
    return torch.tensor([total / (2 * counts[c]) if counts[c] else 0.0 for c in (0, 1)], dtype=torch.float32)


def group_metrics(predictions, key):
    """Accuracy per generator/source; for AI rows this is the detection rate."""
    out = {}
    for p in predictions:
        name = f"{'ai' if p['label'] else 'human'}:{p.get(key) or '-'}"
        bucket = out.setdefault(name, dict(tracks=0, correct=0, mean_p_ai=0.0))
        bucket['tracks'] += 1
        bucket['correct'] += int((p['p_ai'] >= 0.5) == bool(p['label']))
        bucket['mean_p_ai'] += p['p_ai']
    for bucket in out.values():
        bucket['accuracy'] = bucket['correct'] / bucket['tracks']
        bucket['mean_p_ai'] /= bucket['tracks']
    return dict(sorted(out.items()))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tracks = {}
    for x, y, ids in loader:
        probs = model(x.to(device)).softmax(1)[:, 1].cpu().tolist()
        for key, label, prob in zip(ids, y.tolist(), probs):
            tracks.setdefault(key, {'label': label, 'scores': []})['scores'].append(prob)
    predictions = [{'track_id': k, 'label': v['label'], 'p_ai': float(np.mean(v['scores']))} for k, v in tracks.items()]
    y = [r['label'] for r in predictions]
    score = [r['p_ai'] for r in predictions]
    pred = [int(s >= 0.5) for s in score]
    clipped = np.clip(np.asarray(score, dtype=np.float64), 1e-7, 1 - 1e-7)
    targets = np.asarray(y)
    track_loss = float(-np.mean(targets * np.log(clipped) + (1 - targets) * np.log1p(-clipped)))
    metrics = dict(loss=track_loss, accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                   macro_f1=f1_score(y, pred, average='macro', zero_division=0),
                   confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
                   roc_auc=roc_auc_score(y, score),
                   per_class=classification_report(y, pred, labels=[0, 1],
                       target_names=['human', 'ai'], output_dict=True, zero_division=0))
    return metrics, predictions


def main(a):
    if a.epochs < 1 or a.batch_size < 1:
        raise ValueError('epochs and batch-size must be positive')
    monitor = TrainingMonitor(a.monitor, a.patience, a.min_delta)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() and not a.cpu else 'cpu')
    output = Path(a.output)
    resuming = a.resume and (output / 'history.json').is_file()
    if resuming and (output / 'report.json').exists():
        # A report made by hmf.finalize is only a stand-in for an interrupted run; it is replaced at the end.
        if not json.loads((output / 'report.json').read_text()).get('finalized_from_checkpoint'):
            raise ValueError(f'{output} sudah selesai (report.json ada); tidak ada yang bisa dilanjutkan.')
        print('report.json hasil finalize akan ditimpa setelah training lanjutan selesai.', flush=True)
    if a.resume and not resuming and output.exists() and any(output.iterdir()):
        raise ValueError(f'{output} tidak punya history.json; tidak ada epoch yang bisa dilanjutkan.')
    output.mkdir(parents=True, exist_ok=a.resume)  # without --resume an existing folder is still refused
    caches, all_rows, audio_config = load_caches(a.cache)
    for i, cache in enumerate(caches):  # keep a copy of every index the run was trained on
        copy = output / ('data_index.json' if i == 0 else f'data_index_{i}.json')
        if resuming and not copy.exists():
            raise ValueError('Jumlah cache berbeda dari run awal; lanjutkan dengan cache yang sama.')
        if not copy.exists():
            shutil.copy2(cache / 'index.json', copy)
        elif resuming and copy.stat().st_size != (cache / 'index.json').stat().st_size:
            raise ValueError(f'Cache fitur {cache} berbeda dari yang dipakai run ini (ukuran index.json tidak sama).')
    if resuming and (output / f'data_index_{len(caches)}.json').exists():
        raise ValueError('Jumlah cache berbeda dari run awal; lanjutkan dengan cache yang sama.')
    datasets = {s: Segments(caches, s, all_rows) for s in ('train', 'val', 'test')}
    for s, d in datasets.items():
        if {r['label'] for r in d.rows} != {'0', '1'}:
            raise ValueError(f'Split {s} gabungan cache tidak memiliki kedua label.')
    train_rows = datasets['train'].rows
    sampler = None
    if a.balance == 'sampler':
        sampler = WeightedRandomSampler(sample_weights(train_rows, a.balance_by), num_samples=len(train_rows),
                                        replacement=True, generator=torch.Generator().manual_seed(a.seed))
    loss_weight = class_weights(train_rows).to(device) if a.balance == 'weight' else None
    loaders = {s: DataLoader(d, batch_size=a.batch_size, shuffle=s == 'train' and sampler is None,
                            sampler=sampler if s == 'train' else None, num_workers=a.workers,
                            pin_memory=device.type == 'cuda') for s, d in datasets.items()}
    print('Train segmen per kelas:', dict(Counter(r['label'] for r in train_rows)),
          f'| balance={a.balance} ({a.balance_by})', flush=True)
    train_eval_loader = DataLoader(datasets['train'], batch_size=a.batch_size, shuffle=False,
                                  num_workers=a.workers)
    model = CNN(a.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    history, start = [], 0
    args_file = output / 'train_args.json'
    if resuming:
        if args_file.exists():
            before = json.loads(args_file.read_text())
            changed = {k: (before[k], getattr(a, k)) for k in ('model', 'balance', 'balance_by', 'monitor', 'seed', 'patience')
                       if k in before and before[k] != getattr(a, k)}
            if changed:
                raise ValueError(f'Pengaturan berbeda dari run awal (awal, sekarang): {changed}')
        else:
            print('train_args.json tidak ada (run dibuat sebelum fitur resume); pengaturan tidak bisa dibandingkan.')
        history = json.loads((output / 'history.json').read_text())
        last = output / 'last.pt'
        state = torch.load(last, map_location=device, weights_only=True) if last.is_file() else None
        if state is not None and state['epoch'] == len(history):
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scaler.load_state_dict(state['scaler'])
            how = 'last.pt (bobot dan optimizer)'
        else:
            saved = [h['epoch'] for h in history if h.get('checkpoint_saved')]
            if not saved:
                raise ValueError('history.json tidak menandai epoch yang disimpan; tidak ada checkpoint untuk dilanjutkan.')
            history = [h for h in history if h['epoch'] <= saved[-1]]  # best.pt holds the last saved epoch
            model.load_state_dict(torch.load(output / 'best.pt', map_location=device, weights_only=True))
            how = f'best.pt epoch {saved[-1]} (optimizer dimulai baru)'
        for h in history:  # replaying the history restores best epoch and the early-stopping counter
            monitor.update(h['val'], h['epoch'])
        start = len(history)
        sampler_seed = a.seed + start  # do not replay the first epochs' sample order
        torch.manual_seed(sampler_seed)
        if sampler is not None:
            sampler.generator.manual_seed(sampler_seed)
        print(f'Melanjutkan dari epoch {start} lewat {how}; epoch terbaik sejauh ini {monitor.best_epoch}.', flush=True)
    else:
        args_file.write_text(json.dumps(vars(a), indent=2), encoding='utf-8')
    stop_reason = 'early_stopping' if resuming and monitor.should_stop else 'max_epochs'
    for epoch in ([] if stop_reason == 'early_stopping' else range(start, a.epochs)):
        model.train()
        total = 0.0
        for x, y, _ in loaders['train']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                loss = torch.nn.functional.cross_entropy(model(x), y, weight=loss_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(y)
        metrics, _ = evaluate(model, loaders['val'], device)
        train_metrics = None if a.skip_train_eval else evaluate(model, train_eval_loader, device)[0]
        improved = monitor.update(metrics, epoch + 1)
        history.append(dict(epoch=epoch + 1, train_loss=total / len(datasets['train']),
                            train_eval=train_metrics, val=metrics, checkpoint_saved=improved))
        train_acc = f"{train_metrics['accuracy']:.3f}" if train_metrics else '-'
        print(f"Epoch {epoch + 1}/{a.epochs} | train_loss={history[-1]['train_loss']:.4f} "
              f"| train_acc={train_acc} | val_loss={metrics['loss']:.4f} "
              f"| val_f1={metrics['macro_f1']:.3f} | best_epoch={monitor.best_epoch}", flush=True)
        (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        if improved:
            torch.save(model.state_dict(), output / 'best.pt')
        partial = output / 'last.pt.part'  # everything needed to continue exactly where this epoch ended
        torch.save(dict(epoch=epoch + 1, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict()), partial)
        partial.replace(output / 'last.pt')
        if monitor.should_stop:
            stop_reason = 'early_stopping'
            break
    model.load_state_dict(torch.load(output / 'best.pt', map_location=device, weights_only=True))
    metrics, predictions = evaluate(model, loaders['test'], device)
    info = {r['track_id']: r for r in datasets['test'].rows}
    for p in predictions:
        row = info[p['track_id']]
        p.update(path=row['path'], source=row.get('source', ''), generator=row.get('generator', ''))
    metrics['per_generator'] = group_metrics(predictions, 'generator')
    metrics['per_source'] = group_metrics(predictions, 'source')
    report = dict(args=vars(a), device=str(device), torch_version=torch.__version__, history=history,
                  best_epoch=monitor.best_epoch, best_monitor_value=monitor.best_value,
                  epochs_completed=len(history), stop_reason=stop_reason,
                  resumed_from_epoch=start if resuming else None,
                  test=metrics, predictions=predictions,
                  audio_config=audio_config)
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Finished: {stop_reason}; selected epoch {monitor.best_epoch} using val {a.monitor}.')
    print('Test:', metrics)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache', nargs='+', required=True,
                   help='Satu atau lebih cache fitur; yang pertama cache utama, sisanya tambahan (mis. lagu umpan balik)')
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model', choices=['baseline', 'frequency'], default='baseline', help='frequency preserves frequency position in pooling')
    p.add_argument('--patience', type=int, default=5, help='Epochs without improvement; 0 disables early stopping')
    p.add_argument('--monitor', choices=['macro_f1', 'loss'], default='macro_f1', help='Validation metric for checkpoint selection')
    p.add_argument('--min-delta', type=float, default=0.0, help='Minimum strict improvement of monitored metric')
    p.add_argument('--balance', choices=['sampler', 'weight', 'none'], default='sampler',
                   help='sampler: WeightedRandomSampler seimbang; weight: class weight di loss; none: seperti versi lama')
    p.add_argument('--balance-by', choices=['label', 'generator'], default='label',
                   help='generator: tiap generator/sumber dalam satu kelas juga diberi porsi sama')
    p.add_argument('--workers', type=int, default=0, help='DataLoader num_workers (Colab: 2-4)')
    p.add_argument('--resume', action='store_true',
                   help='Lanjutkan run yang terhenti di folder --output (dari last.pt, atau best.pt untuk run lama)')
    p.add_argument('--skip-train-eval', action='store_true',
                   help='Lewati penilaian ulang seluruh data train tiap epoch (epoch sekitar dua kali lebih cepat; train_acc kosong)')
    p.add_argument('--cpu', action='store_true')
    main(p.parse_args())
