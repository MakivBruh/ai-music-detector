"""Synthetic plumbing test only: these tones are NOT forensic training data."""
import argparse
import csv
from pathlib import Path
import numpy as np
import soundfile as sf


def generate(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(12)
    rows = []
    sr = 22050
    t = np.arange(sr * 6) / sr
    for split in ('train', 'val', 'test'):
        for label in (0, 1):
            for i in range(3):
                name = f'{split}_{label}_{i}.wav'
                frequency = rng.uniform(150, 450) + label * 700
                signal = .2 * np.sin(2 * np.pi * frequency * t) + rng.normal(0, .01, len(t))
                sf.write(root / name, signal, sr)
                rows.append(dict(path=name, label=label, group_id=name, split=split,
                                 source='synthetic-test-only', generator='tone'))
    with (root / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    generate(p.parse_args().output)
