import argparse
import json
from pathlib import Path
import torch
from .audio import AudioConfig, features
from .model import CNN


def predict(run, audio):
    run = Path(run)
    report = json.loads((run / 'report.json').read_text())
    config = AudioConfig(**report['audio_config'])
    model = CNN().eval()
    model.load_state_dict(torch.load(run / 'best.pt', map_location='cpu', weights_only=True))
    scores = []
    with torch.no_grad():
        for feature in features(audio, config):
            scores.append(model(torch.from_numpy(feature.astype('float32'))[None, None]).softmax(1)[0, 1].item())
    if not scores:
        raise ValueError('Empty audio')
    score = sum(scores) / len(scores)
    print(json.dumps(dict(p_ai=score, label=int(score >= .5), segments=len(scores),
                          note='Uncalibrated model score; not proof of origin'), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--audio', required=True)
    a = p.parse_args()
    predict(a.run, a.audio)
