import argparse
import json
import random
import shutil
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score, classification_report
from .model import CNN


class Segments(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        self.rows = [r for r in json.loads((self.root / 'index.json').read_text()) if r['split'] == split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return torch.from_numpy(np.load(self.root / r['file']))[None], int(r['label']), r['track_id']


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
    metrics = dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                   macro_f1=f1_score(y, pred, average='macro', zero_division=0),
                   confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
                   roc_auc=roc_auc_score(y, score),
                   per_class=classification_report(y, pred, labels=[0, 1],
                       target_names=['human', 'ai'], output_dict=True, zero_division=0))
    return metrics, predictions


def main(a):
    if a.epochs < 1 or a.batch_size < 1:
        raise ValueError('epochs and batch-size must be positive')
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() and not a.cpu else 'cpu')
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(a.cache) / 'index.json', output / 'data_index.json')
    datasets = {s: Segments(a.cache, s) for s in ('train', 'val', 'test')}
    loaders = {s: DataLoader(d, batch_size=a.batch_size, shuffle=s == 'train', num_workers=0,
                            pin_memory=device.type == 'cuda') for s, d in datasets.items()}
    model = CNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    best, stale, history = -1, 0, []
    for epoch in range(a.epochs):
        model.train()
        total = 0.0
        for x, y, _ in loaders['train']:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                loss = torch.nn.functional.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(y)
        metrics, _ = evaluate(model, loaders['val'], device)
        history.append(dict(epoch=epoch + 1, train_loss=total / len(datasets['train']), val=metrics))
        print(json.dumps(history[-1]))
        if metrics['macro_f1'] > best:
            best, stale = metrics['macro_f1'], 0
            torch.save(model.state_dict(), output / 'best.pt')
        else:
            stale += 1
        if stale >= 5:
            break
    model.load_state_dict(torch.load(output / 'best.pt', map_location=device, weights_only=True))
    metrics, predictions = evaluate(model, loaders['test'], device)
    report = dict(args=vars(a), device=str(device), torch_version=torch.__version__, history=history,
                  test=metrics, predictions=predictions,
                  audio_config=json.loads((Path(a.cache) / 'audio_config.json').read_text()))
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Test:', metrics)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu', action='store_true')
    main(p.parse_args())
