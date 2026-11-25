import csv
import hashlib
import json
import threading
import urllib.error
import urllib.request
import pytest
from hmf import feedback

SHA_A = 'a' * 64
SHA_B = 'b' * 64


def entry(**over):
    base = dict(answer='correct', verdict='AI', sha256=SHA_A, ai_score=0.8, decision_threshold=0.5,
                segment_scores=[0.7, 0.9], filename='lagu.mp3', model='frequency_cnn', label_source='dibuat_sendiri',
                note='', run='run-x')
    return {**base, **over}


def test_validate_derives_true_label_and_rejects_bad_input():
    assert feedback.validate(entry())['true_label'] == 'ai'
    assert feedback.validate(entry(answer='wrong'))['true_label'] == 'human'
    assert feedback.validate(entry(verdict='Human', answer='wrong'))['true_label'] == 'ai'
    unknown = feedback.validate(entry(answer='unknown', label_source=None))
    assert unknown['true_label'] is None and unknown['label_source'] is None
    with pytest.raises(ValueError, match='sumber|Pilih'):
        feedback.validate(entry(label_source=None))
    with pytest.raises(ValueError, match='sha256'):
        feedback.validate(entry(sha256='../../etc'))
    with pytest.raises(ValueError, match='Jawaban'):
        feedback.validate(entry(answer='maybe'))
    with pytest.raises(ValueError, match='rentang'):
        feedback.validate(entry(ai_score=1.5))
    assert feedback.validate(entry(filename='../../secret/x?.mp3'))['filename'] == 'x_.mp3'


def test_latest_answer_wins_and_broken_lines_are_skipped(tmp_path):
    feedback.append(tmp_path, dict(feedback.validate(entry(answer='wrong')), run='run-x'))
    (tmp_path / 'feedback.jsonl').open('a').write('{broken\n')
    feedback.append(tmp_path, dict(feedback.validate(entry(answer='correct')), run='run-x'))
    feedback.append(tmp_path, dict(feedback.validate(entry(answer='correct')), run='run-y'))
    records = feedback.load(tmp_path)
    assert len(records) == 2
    assert {r['run']: r['answer'] for r in records} == {'run-x': 'correct', 'run-y': 'correct'}


def test_summary_counts_confusion_and_interval():
    rows = []
    for i in range(10):  # 8 correct: 5 AI found, 3 humans kept; 2 wrong: 1 human accused, 1 AI missed
        answer = 'wrong' if i in (8, 9) else 'correct'
        verdict = 'AI' if i < 5 or i == 8 else 'Human'
        rows.append(dict(feedback.validate(entry(sha256=f'{i:064x}', verdict=verdict, answer=answer)), run='r'))
    rows.append(dict(feedback.validate(entry(sha256='f' * 64, answer='unknown', label_source=None)), run='r'))
    rows.append(dict(feedback.validate(entry(sha256='e' * 64)), run='other'))
    s = feedback.summarize(rows, 'r')
    assert (s['total'], s['scored'], s['unknown'], s['correct']) == (11, 10, 1, 8)
    assert s['confusion'] == dict(human_as_human=3, human_as_ai=1, ai_as_ai=5, ai_as_human=1)
    assert s['accuracy'] == pytest.approx(0.8)
    low, high = s['accuracy_ci95']
    assert low == pytest.approx(0.490, abs=0.002) and high == pytest.approx(0.943, abs=0.002)
    assert s['false_accusation_rate'] == pytest.approx(0.25) and s['missed_ai_rate'] == pytest.approx(1 / 6)
    assert len(s['wrong']) == 2 and s['other_runs'] == ['other'] and s['other_runs_total'] == 1
    assert feedback.summarize([], 'r')['accuracy'] is None


@pytest.fixture
def server(tmp_path):
    from hmf.web import make_server
    run = tmp_path / 'run-web'
    run.mkdir()
    srv = make_server(run, 0, tmp_path / 'fb')
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{srv.server_address[1]}', tmp_path / 'fb'
    srv.shutdown()
    srv.server_close()


def call(url, data=None, headers=None):
    request = urllib.request.Request(url, data=data, headers=headers or {}, method='POST' if data is not None else 'GET')
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def test_web_feedback_flow_and_audio_checks(server):
    base, folder = server
    status, empty = call(base + '/feedback')
    assert status == 200 and empty['scored'] == 0 and 'dibuat_sendiri' in empty['label_sources']
    audio = b'fake audio bytes'
    sha = hashlib.sha256(audio).hexdigest()
    headers = {'X-Sha256': sha, 'X-Audio-Filename': 'lagu%20saya.mp3'}
    status, text = call(base + '/feedback/audio', audio, headers)
    assert status == 400 and 'umpan balik dulu' in text  # no answer recorded yet
    body = json.dumps(entry(sha256=sha, answer='wrong', run='ignored-by-server')).encode()
    status, saved = call(base + '/feedback', body, {'Content-Type': 'application/json'})
    assert status == 200 and saved['true_label'] == 'human'
    assert saved['summary']['run'] == 'run-web' and saved['summary']['confusion']['human_as_ai'] == 1
    status, text = call(base + '/feedback/audio', b'other bytes', headers)
    assert status == 400 and 'berbeda' in text
    status, done = call(base + '/feedback/audio', audio, headers)
    assert status == 200 and done['summary']['wrong'][0]['audio_saved'] is True
    assert (folder / 'audio/human' / f'{sha[:12]}_lagu saya.mp3').read_bytes() == audio
    status, text = call(base + '/feedback', b'{not json', {'Content-Type': 'application/json'})
    assert status == 400


def test_export_to_ood_and_manifest(tmp_path):
    folder = tmp_path / 'fb'
    for i, answer in enumerate(('wrong', 'correct')):
        audio = f'song {i}'.encode()
        record = dict(feedback.validate(entry(sha256=hashlib.sha256(audio).hexdigest(), answer=answer)), run='r')
        feedback.append(folder, record)
        feedback.save_audio(folder, record, audio, f'song{i}.mp3')
    feedback.main(['--dir', str(folder), 'export', '--run', 'r', '--only-wrong', '--ood', str(tmp_path / 'ood'),
                   '--manifest', str(tmp_path / 'm.csv')])
    assert [p.name.split('_', 1)[1] for p in (tmp_path / 'ood/human').iterdir()] == ['song0.mp3']
    assert not (tmp_path / 'ood/ai').exists()
    rows = list(csv.DictReader((tmp_path / 'm.csv').open()))
    assert len(rows) == 1 and rows[0]['label'] == '0' and rows[0]['group_id'].startswith('feedback_')
