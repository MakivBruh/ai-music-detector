import csv
import json
import numpy as np
import pytest
import torch
import soundfile as sf
from hmf.audio import AudioConfig, features
from hmf.demo import generate
from hmf.model import CNN
from hmf.prepare import prepare
from hmf.train import Segments, evaluate
from torch.utils.data import DataLoader


@pytest.mark.parametrize('variant', ['baseline', 'frequency'])
def test_audio_training_and_aggregation(tmp_path, variant):
    generate(tmp_path / 'demo')
    prepare(tmp_path / 'demo/manifest.csv', tmp_path / 'cache')
    dataset = Segments(tmp_path / 'cache', 'train')
    assert len(dataset) == 12  # six tracks, two windows each, last zero-padded
    x, y, _ = dataset[0]
    assert x.shape == (1, 64, 313)
    assert torch.isfinite(x).all()
    model = CNN(variant)
    optimizer = torch.optim.Adam(model.parameters())
    before = model.head[-1].weight.detach().clone()
    loss = torch.nn.functional.cross_entropy(model(x[None]), torch.tensor([y]))
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.head[-1].weight)
    metrics, predictions = evaluate(model, DataLoader(dataset, batch_size=4), 'cpu')
    assert len(predictions) == 6
    assert np.isfinite(metrics['macro_f1'])


def test_reject_group_leakage(tmp_path):
    generate(tmp_path / 'demo')
    manifest = tmp_path / 'demo/manifest.csv'
    rows = list(csv.DictReader(manifest.open()))
    rows[6]['group_id'] = rows[0]['group_id']
    with manifest.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError, match='Group leakage'):
        prepare(manifest, tmp_path / 'cache')


def test_silent_stereo_resample(tmp_path):
    path = tmp_path / 'silence.wav'
    sf.write(path, np.zeros((44100, 2), dtype=np.float32), 44100)
    result = list(features(path, AudioConfig()))
    assert len(result) == 1
    assert result[0].shape == (64, 313)
    assert np.isfinite(result[0]).all()


def test_reject_duplicate_bytes(tmp_path):
    generate(tmp_path / 'demo')
    manifest = tmp_path / 'demo/manifest.csv'
    rows = list(csv.DictReader(manifest.open()))
    rows[1]['path'] = rows[0]['path']
    with manifest.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError, match='Duplicate audio'):
        prepare(manifest, tmp_path / 'cache')


def test_loss_uses_track_averages():
    # Unequal segment counts must not give track A more evaluation weight.
    samples = [
        (torch.tensor([.2, .8]).log(), 1, 'A'),
        (torch.tensor([.2, .8]).log(), 1, 'A'),
        (torch.tensor([.6, .4]).log(), 0, 'B'),
    ]
    metrics, predictions = evaluate(torch.nn.Identity(), DataLoader(samples, batch_size=2), 'cpu')
    assert len(predictions) == 2
    assert metrics['loss'] == pytest.approx((-np.log(.8) - np.log(.6)) / 2, abs=1e-6)
