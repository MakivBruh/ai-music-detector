"""Lengkapi run yang terhenti di tengah training.

`hmf.train` menyimpan best.pt dan history.json setiap epoch, tetapi report.json
(yang dibutuhkan studio web, predict, dan evaluate_folder) baru ditulis di akhir.
Skrip ini mengevaluasi best.pt pada split test lalu menulis report.json dengan
stop_reason "interrupted", sehingga model sementara bisa dipakai.

Contoh:
  python -m hmf.finalize --cache /content/work/cache-hmf-aug-v1 \
      --run /content/drive/MyDrive/HMF/runs/hmf-aug-v1 --model frequency --balance sampler \
      --balance-by generator --monitor loss --batch-size 64 --seed 42
"""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .model import CNN
from .train import Segments, evaluate, group_metrics, load_caches


def finalize(cache, run, model_name='frequency', batch_size=64, workers=2, force=False, **recorded):
    run = Path(run)
    if (run / 'report.json').exists() and not force:
        raise ValueError(f'{run / "report.json"} sudah ada; training ini sudah selesai (pakai --force untuk menimpa).')
    for name in ('best.pt', 'history.json'):
        if not (run / name).is_file():
            raise ValueError(f'{run / name} tidak ditemukan; training belum menyelesaikan satu epoch pun.')
    history = json.loads((run / 'history.json').read_text())
    saved = [h for h in history if h.get('checkpoint_saved')]
    if not saved:
        raise ValueError('history.json tidak menandai epoch yang disimpan.')
    best = saved[-1]  # best.pt is overwritten at every improvement, so it holds the last saved epoch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN(model_name).to(device)
    model.load_state_dict(torch.load(run / 'best.pt', map_location=device, weights_only=True))
    caches, rows, audio_config = load_caches(cache)
    dataset = Segments(caches, 'test', rows)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    metrics, predictions = evaluate(model, loader, device)
    info = {r['track_id']: r for r in dataset.rows}
    for p in predictions:
        row = info[p['track_id']]
        p.update(path=row['path'], source=row.get('source', ''), generator=row.get('generator', ''))
    metrics['per_generator'] = group_metrics(predictions, 'generator')
    metrics['per_source'] = group_metrics(predictions, 'source')
    args = dict(cache=[str(c) for c in caches], output=str(run), model=model_name, batch_size=batch_size, **recorded)
    report = dict(args=args, device=str(device), torch_version=torch.__version__, history=history,
                  best_epoch=best['epoch'], best_monitor_value=best['val'].get(recorded.get('monitor', 'loss')),
                  epochs_completed=len(history), stop_reason='interrupted', finalized_from_checkpoint=True,
                  test=metrics, predictions=predictions,
                  audio_config=audio_config)
    (run / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"report.json dibuat: epoch terpilih {best['epoch']} dari {len(history)} epoch yang selesai.")
    print('Test:', {k: round(metrics[k], 4) for k in ('accuracy', 'balanced_accuracy', 'macro_f1', 'roc_auc')})
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache', nargs='+', required=True, help='cache yang sama (dan urutan sama) seperti saat training')
    p.add_argument('--run', required=True)
    p.add_argument('--model', choices=['baseline', 'frequency'], default='frequency')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--monitor', choices=['macro_f1', 'loss'], default='loss')
    p.add_argument('--balance', choices=['sampler', 'weight', 'none'], default='sampler')
    p.add_argument('--balance-by', choices=['label', 'generator'], default='generator')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    finalize(a.cache, a.run, a.model, a.batch_size, a.workers, a.force,
             monitor=a.monitor, balance=a.balance, balance_by=a.balance_by, seed=a.seed)
