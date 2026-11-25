import csv
import json
from collections import Counter
import numpy as np
import pytest
import torch
from hmf.audio import AudioConfig, CODECS, features, sample_augmentation
from hmf.demo import generate
from hmf.manifest import (assign_splits, base_name, check_manifest, leave_one_generator_out,
                          scan_musebench, write_manifest)
from hmf.prepare import map_path, parse_path_map, prepare, read_rows
from hmf.train import Segments, class_weights, sample_weights
from torch.utils.data import WeightedRandomSampler


# --- augmentation -----------------------------------------------------------

def test_augmentation_params_deterministic_per_key():
    first = [sample_augmentation(f'42:abc:{v}') for v in range(20)]
    again = [sample_augmentation(f'42:abc:{v}') for v in range(20)]
    other_seed = [sample_augmentation(f'7:abc:{v}') for v in range(20)]
    assert first == again
    assert first != other_seed
    for params in first:
        assert params['codec'] in CODECS
        assert params['bitrate'] in CODECS[params['codec']][2]
        assert -12 <= params['gain_db'] <= 4
    assert len({p['codec'] for p in first}) > 1


@pytest.mark.parametrize('codec', ['mp3', 'aac', 'opus', 'pcm'])
def test_augmented_features_deterministic(tmp_path, codec):
    generate(tmp_path / 'demo')
    path = next((tmp_path / 'demo').glob('*.wav'))
    params = dict(codec=codec, bitrate=CODECS[codec][2][0], sample_rate=22050, channels=1,
                  gain_db=-3.0, lowpass=6000)
    config = AudioConfig(loudness_norm=True)
    a = list(features(path, config, augment=params))
    b = list(features(path, config, augment=params))
    clean = list(features(path, config))
    assert len(a) == len(clean) and a[0].shape == clean[0].shape
    assert all(np.allclose(x, y) for x, y in zip(a, b))
    assert not np.allclose(a[0], clean[0])  # the chain really changed the audio


def test_loudness_norm_removes_gain(tmp_path):
    import soundfile as sf
    t = np.arange(16000 * 5) / 16000
    tone = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(tmp_path / 'loud.wav', tone, 16000, subtype='FLOAT')
    sf.write(tmp_path / 'quiet.wav', tone * 0.1, 16000, subtype='FLOAT')
    config = AudioConfig(loudness_norm=True)
    loud = next(features(tmp_path / 'loud.wav', config))
    quiet = next(features(tmp_path / 'quiet.wav', config))
    assert np.allclose(loud, quiet, atol=1e-3)


def test_fmax_hides_band_above_cutoff(tmp_path):
    import soundfile as sf
    t = np.arange(16000 * 5) / 16000
    high = (0.3 * np.sin(2 * np.pi * 7600 * t)).astype(np.float32)  # only above 7 kHz
    sf.write(tmp_path / 'high.wav', high, 16000, subtype='FLOAT')
    full = next(features(tmp_path / 'high.wav', AudioConfig()))
    cut = next(features(tmp_path / 'high.wav', AudioConfig(fmax=7000)))
    assert full.shape == cut.shape == (64, 313)
    assert full.max() - cut.max() > 0.6  # >30 dB weaker: only window leakage remains below 7 kHz


def prepared(tmp_path, name, **kwargs):
    if not (tmp_path / 'demo').exists():
        generate(tmp_path / 'demo')
    prepare(tmp_path / 'demo/manifest.csv', tmp_path / name, augment_variants=2, loudness_norm=True, **kwargs)
    return json.loads((tmp_path / name / 'index.json').read_text())


def test_prepare_augments_both_train_classes_not_eval(tmp_path):
    index = prepared(tmp_path, 'cache')
    train = [r for r in index if r['split'] == 'train']
    held = [r for r in index if r['split'] != 'train']
    assert {r['variant'] for r in train} == {0, 1}
    assert {r['label'] for r in train if r['augment']} == {'0', '1'}
    assert all(r['augment'] for r in train)
    assert held and all(r['variant'] == 0 and r['augment'] == '' for r in held)
    # Two variants of one train track use different chains.
    track = train[0]['track_id']
    chains = {r['augment'] for r in train if r['track_id'] == track}
    assert len(chains) == 2
    # Eval windows are the plain features, so val/test are untouched.
    row = held[0]
    clean = list(features(row['path'], AudioConfig(loudness_norm=True)))[row['segment']]
    assert np.allclose(np.load(tmp_path / 'cache' / row['file']), clean)


def test_prepare_deterministic_per_seed_and_parallel(tmp_path):
    first = prepared(tmp_path, 'a')
    second = prepared(tmp_path, 'b', workers=2, float16=True)
    assert [r['augment'] for r in first] == [r['augment'] for r in second]
    for x, y in zip(first, second):
        a = np.load(tmp_path / 'a' / x['file'])
        b = np.load(tmp_path / 'b' / y['file'])
        assert b.dtype == np.float16
        assert np.allclose(a, b.astype(np.float32), atol=2e-3)
    dataset = Segments(tmp_path / 'b', 'train')
    assert dataset[0][0].dtype == torch.float32
    third = prepared(tmp_path, 'c', seed=7)
    assert [r['augment'] for r in first] != [r['augment'] for r in third]


def test_path_map_rewrites_windows_prefix(tmp_path):
    audio = tmp_path / 'colab' / 'songs'
    audio.mkdir(parents=True)
    (audio / 'a.wav').write_bytes(b'x')
    manifest = tmp_path / 'm.csv'
    manifest.write_text('path,group_id,label,split\nC:\\Users\\me\\Songs\\a.wav,g1,0,train\n', encoding='utf-8')
    pairs = parse_path_map([f'C:\\Users\\me\\songs={audio}'])
    assert map_path('C:\\Users\\me\\Songs\\a.wav', pairs) == f'{audio}/a.wav'
    rows = read_rows([manifest], None, path_map=pairs)
    assert rows[0]['path'] == str((audio / 'a.wav').resolve())


# --- balancing ----------------------------------------------------------------

def rows_for(counts):
    return [dict(label=label, generator=gen) for (label, gen), n in counts.items() for _ in range(n)]


def test_balanced_sampler_gives_equal_class_mass():
    rows = rows_for({('1', 'suno'): 900, ('1', 'udio'): 100, ('0', ''): 250})
    weights = sample_weights(rows, 'label')
    mass = Counter()
    for row, w in zip(rows, weights):
        mass[row['label']] += w
    assert mass['0'] == pytest.approx(mass['1'])
    sampler = WeightedRandomSampler(weights, 20000, replacement=True, generator=torch.Generator().manual_seed(0))
    drawn = Counter(rows[i]['label'] for i in sampler)
    assert abs(drawn['1'] / 20000 - 0.5) < 0.02


def test_balance_by_generator_equalizes_generators():
    rows = rows_for({('1', 'suno'): 900, ('1', 'udio'): 100, ('0', ''): 250})
    weights = sample_weights(rows, 'generator')
    mass = Counter()
    for row, w in zip(rows, weights):
        mass[row['label'], row['generator']] += w
    assert mass['1', 'suno'] == pytest.approx(mass['1', 'udio'])
    assert mass['1', 'suno'] + mass['1', 'udio'] == pytest.approx(mass['0', ''])


def test_class_weights_inverse_frequency():
    weights = class_weights(rows_for({('1', 'x'): 300, ('0', ''): 100}))
    assert weights[0] / weights[1] == pytest.approx(3.0)


# --- MUSEBENCH manifest -------------------------------------------------------

def touch(root, relative):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(relative.encode())  # unique bytes per file
    return path


@pytest.fixture
def musebench(tmp_path):
    root = tmp_path / 'mb'
    songs = [f'Song {i} - Artist {i}' for i in range(40)]
    for song in songs:
        touch(root, f'Jamendo/{song}.mp3')
        touch(root, f'Musicgen_Generated_Music/Jamedo/{song}.mp3.wav')
        touch(root, f'audioldm_generated_music/Jamedo/{song}.mp3.wav')
    for i in range(20):
        touch(root, f'riffusion_generated_music/Genius-Lyrics/Lyric {i}_Band_abc{i}.mp3.wav')
        touch(root, f'stable_audio_generated_music/Genius-Lyrics/Lyric {i}_Band_abc{i}.mp3.wav')
        touch(root, f'OOD FMA/{1000 + i}_Title_{i}.mp3')
    touch(root, 'Jamendo/&amp; Rock &amp; Roll - Band.mp3')
    touch(root, 'magenta_generated_music/Jamedo/& Rock & Roll - Band.mp3.wav')
    touch(root, 'eq_generated_music/suno_generated_music/root/fake_1_suno_0.mp3')
    touch(root, 'gaussion_noised_music/MagNet_Generated_Music/Genius-Lyrics/x.mp3.wav')
    touch(root, 'ai_human_co-created_music/Jamendo/Song 1 - Artist 1_edited.wav')
    touch(root, 'suno_generated_music/root/fake_2_suno_1.mp3')
    touch(root, 'Caption/captions.csv')
    return root


def test_base_name_pairs_generation_with_original():
    assert base_name('Song A - Artist.mp3.wav') == base_name('Song A - Artist.mp3')
    assert base_name('&amp; Rock - X.mp3') == base_name('& Rock - X.mp3.wav')


def test_musebench_labels_and_pairing(musebench):
    rows, skipped = scan_musebench(musebench)
    by_path = {r['path'].replace('\\', '/').split('/mb/')[1]: r for r in rows}
    human = by_path['Jamendo/Song 3 - Artist 3.mp3']
    musicgen = by_path['Musicgen_Generated_Music/Jamedo/Song 3 - Artist 3.mp3.wav']
    audioldm = by_path['audioldm_generated_music/Jamedo/Song 3 - Artist 3.mp3.wav']
    assert (human['label'], musicgen['label'], audioldm['label']) == ('0', '1', '1')
    assert (musicgen['generator'], audioldm['generator']) == ('musicgen', 'audioldm')
    assert human['group_id'] == musicgen['group_id'] == audioldm['group_id']
    assert by_path['Jamendo/&amp; Rock &amp; Roll - Band.mp3']['group_id'] == \
        by_path['magenta_generated_music/Jamedo/& Rock & Roll - Band.mp3.wav']['group_id']
    assert by_path['OOD FMA/1005_Title_5.mp3']['group_id'] == 'fma_001005'
    riff = by_path['riffusion_generated_music/Genius-Lyrics/Lyric 2_Band_abc2.mp3.wav']
    stable = by_path['stable_audio_generated_music/Genius-Lyrics/Lyric 2_Band_abc2.mp3.wav']
    assert riff['group_id'] == stable['group_id'] != human['group_id']
    # Perturbed AI-only folders, co-created songs and SONICS generators are skipped.
    assert not any('eq_generated' in p or 'gaussion' in p or 'co-created' in p or 'suno' in p for p in by_path)
    assert sum(skipped.values()) == 4

    assign_splits(rows, seed=1)
    check_manifest(rows)
    splits = {}
    for row in rows:
        assert splits.setdefault(row['group_id'], row['split']) == row['split']


def test_reference_manifest_split_wins(musebench):
    rows, _ = scan_musebench(musebench)
    assign_splits(rows, seed=1, reference={'fma_001005': 'val'})
    assert {r['split'] for r in rows if r['group_id'] == 'fma_001005'} == {'val'}


def test_leave_one_generator_out(musebench, tmp_path):
    rows, _ = scan_musebench(musebench)
    assign_splits(rows, seed=3)
    logo = leave_one_generator_out(rows, ['musicgen'])
    check_manifest(logo)
    assert all(r['split'] == 'test' for r in logo if r['generator'] == 'musicgen')
    assert not any(r['generator'] == 'musicgen' for r in logo if r['split'] != 'test')
    test_ai = {r['generator'] for r in logo if r['split'] == 'test' and r['label'] == '1'}
    assert test_ai == {'musicgen'}
    # Paired human originals of held-out generations moved to test with them.
    held_groups = {r['group_id'] for r in logo if r['generator'] == 'musicgen'}
    assert {r['split'] for r in logo if r['group_id'] in held_groups} == {'test'}
    assert any(r['label'] == '0' for r in logo if r['group_id'] in held_groups)
    with pytest.raises(ValueError, match='tidak ada'):
        leave_one_generator_out(rows, ['nonexistent'])

    out = tmp_path / 'manifest.csv'
    write_manifest(logo, out)
    back = read_rows([out], None)
    assert len(back) == len(logo)
    assert all(Path_exists(r['path']) for r in back)


def Path_exists(path):
    from pathlib import Path
    return Path(path).is_file()


def write_rows(manifest, rows):
    with manifest.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def test_drop_duplicates_keeps_first_copy(tmp_path):
    generate(tmp_path / 'demo')
    manifest = tmp_path / 'demo/manifest.csv'
    rows = list(csv.DictReader(manifest.open()))
    rows[1]['path'] = rows[0]['path']  # same bytes listed twice (e.g. OOD FMA == FMA-small)
    write_rows(manifest, rows)
    prepare(manifest, tmp_path / 'cache', drop_duplicates=True)
    index = json.loads((tmp_path / 'cache/index.json').read_text())
    assert len({r['track_id'] for r in index}) == len(rows) - 1
    dropped = json.loads((tmp_path / 'cache/duplicates.json').read_text())
    assert len(dropped) == 1


def test_skip_unreadable_also_skips_empty_audio(tmp_path):
    import soundfile as sf
    generate(tmp_path / 'demo')
    manifest = tmp_path / 'demo/manifest.csv'
    rows = list(csv.DictReader(manifest.open()))
    sf.write(tmp_path / 'demo/empty.wav', np.zeros(0, dtype=np.float32), 16000)
    rows.append(dict(rows[0], path='empty.wav', group_id='empty'))
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match='Empty audio'):
        prepare(manifest, tmp_path / 'strict')
    prepare(manifest, tmp_path / 'cache', skip_unreadable=True)
    skipped = json.loads((tmp_path / 'cache/skipped.json').read_text())
    assert [Path_name(s['path']) for s in skipped] == ['empty.wav']


def Path_name(path):
    from pathlib import Path
    return Path(path).name


def test_resume_in_chunks_matches_single_run(tmp_path):
    generate(tmp_path / 'demo')
    manifest = tmp_path / 'demo/manifest.csv'
    options = dict(augment_variants=2, loudness_norm=True, float16=True)
    assert prepare(manifest, tmp_path / 'once', **options) is True
    calls = 0
    while not prepare(manifest, tmp_path / 'chunks', resume=True, max_new_tracks=5, **options):
        calls += 1
        assert not (tmp_path / 'chunks/index.json').exists()
    assert calls == 3  # 18 tracks, 5 per call
    once = json.loads((tmp_path / 'once/index.json').read_text())
    chunks = json.loads((tmp_path / 'chunks/index.json').read_text())
    assert once == chunks
    for row in once:
        assert np.array_equal(np.load(tmp_path / 'once' / row['file']), np.load(tmp_path / 'chunks' / row['file']))
    # A finished cache resumes instantly, without needing the audio any more.
    assert prepare(manifest, tmp_path / 'chunks', resume=True, **options) is True
    with pytest.raises(ValueError, match='Pengaturan berbeda'):
        prepare(manifest, tmp_path / 'chunks', resume=True, augment_variants=3, loudness_norm=True, float16=True)
    with pytest.raises(ValueError, match='new directory'):
        prepare(manifest, tmp_path / 'chunks', **options)


def test_finalize_rebuilds_report_of_interrupted_run(tmp_path):
    from argparse import Namespace
    from hmf import train
    from hmf.finalize import finalize
    generate(tmp_path / 'demo')
    prepare(tmp_path / 'demo/manifest.csv', tmp_path / 'cache')
    args = Namespace(cache=str(tmp_path / 'cache'), output=str(tmp_path / 'run'), epochs=2, batch_size=8, seed=1,
                     model='frequency', patience=5, monitor='loss', min_delta=0.0, balance='sampler',
                     balance_by='label', workers=0, cpu=True, resume=False, skip_train_eval=False)
    train.main(args)
    original = json.loads((tmp_path / 'run/report.json').read_text())
    with pytest.raises(ValueError, match='sudah ada'):
        finalize(tmp_path / 'cache', tmp_path / 'run', batch_size=8, workers=0)
    (tmp_path / 'run/report.json').unlink()  # what an interrupted run looks like
    rebuilt = finalize(tmp_path / 'cache', tmp_path / 'run', 'frequency', batch_size=8, workers=0, monitor='loss')
    assert rebuilt['stop_reason'] == 'interrupted'
    assert rebuilt['best_epoch'] == original['best_epoch']
    assert rebuilt['test']['accuracy'] == pytest.approx(original['test']['accuracy'])
    assert rebuilt['test']['loss'] == pytest.approx(original['test']['loss'])
    from hmf.predict import model_summary
    assert model_summary(tmp_path / 'run')['stop_reason'] == 'interrupted'


def run_args(tmp_path, epochs, **extra):
    from argparse import Namespace
    base = dict(cache=str(tmp_path / 'cache'), output=str(tmp_path / 'run'), epochs=epochs, batch_size=8, seed=1,
                model='frequency', patience=0, monitor='loss', min_delta=0.0, balance='sampler', balance_by='label',
                workers=0, cpu=True, resume=False, skip_train_eval=False)
    return Namespace(**{**base, **extra})


@pytest.fixture
def interrupted(tmp_path):
    """A run that trained 2 epochs and then stopped before writing report.json."""
    from hmf import train
    generate(tmp_path / 'demo')
    prepare(tmp_path / 'demo/manifest.csv', tmp_path / 'cache')
    train.main(run_args(tmp_path, 2))
    (tmp_path / 'run/report.json').unlink()
    return tmp_path


def test_resume_continues_from_last_checkpoint(interrupted):
    from hmf import train
    tmp_path = interrupted
    before = json.loads((tmp_path / 'run/history.json').read_text())
    assert (tmp_path / 'run/last.pt').is_file() and (tmp_path / 'run/train_args.json').is_file()
    train.main(run_args(tmp_path, 4, resume=True))
    after = json.loads((tmp_path / 'run/history.json').read_text())
    assert [h['epoch'] for h in after] == [1, 2, 3, 4]
    assert after[:2] == before  # finished epochs are kept untouched
    report = json.loads((tmp_path / 'run/report.json').read_text())
    assert report['resumed_from_epoch'] == 2 and report['epochs_completed'] == 4
    import torch
    assert torch.load(tmp_path / 'run/last.pt', weights_only=True)['epoch'] == 4


def test_resume_from_best_pt_when_run_has_no_last_pt(interrupted):
    from hmf import train
    tmp_path = interrupted  # same situation as a run made before --resume existed
    (tmp_path / 'run/last.pt').unlink()
    (tmp_path / 'run/train_args.json').unlink()
    train.main(run_args(tmp_path, 3, resume=True, skip_train_eval=True))
    after = json.loads((tmp_path / 'run/history.json').read_text())
    assert [h['epoch'] for h in after] == [1, 2, 3]
    assert after[2]['train_eval'] is None and after[0]['train_eval'] is not None
    assert json.loads((tmp_path / 'run/report.json').read_text())['epochs_completed'] == 3


def test_resume_refuses_wrong_setups(interrupted):
    from hmf import train
    tmp_path = interrupted
    with pytest.raises(ValueError, match='Pengaturan berbeda'):
        train.main(run_args(tmp_path, 4, resume=True, model='baseline'))
    with pytest.raises(FileExistsError):
        train.main(run_args(tmp_path, 4))  # without --resume an existing run folder is still refused
    train.main(run_args(tmp_path, 3, resume=True))
    with pytest.raises(ValueError, match='sudah selesai'):
        train.main(run_args(tmp_path, 5, resume=True))


def test_resume_after_finalize_replaces_stand_in_report(interrupted):
    from hmf import train
    from hmf.finalize import finalize
    tmp_path = interrupted
    finalize(tmp_path / 'cache', tmp_path / 'run', 'frequency', batch_size=8, workers=0, monitor='loss')
    assert json.loads((tmp_path / 'run/report.json').read_text())['stop_reason'] == 'interrupted'
    train.main(run_args(tmp_path, 3, resume=True))
    report = json.loads((tmp_path / 'run/report.json').read_text())
    assert report['stop_reason'] == 'max_epochs' and report['epochs_completed'] == 3
    assert not report.get('finalized_from_checkpoint')


def extra_cache(tmp_path, name='extra', loudness_norm=False, duplicate_of=None):
    """A small one-split cache, like songs exported from studio feedback."""
    import shutil
    import soundfile as sf
    folder = tmp_path / f'{name}-audio'
    folder.mkdir()
    rows = []
    for i, label in enumerate('0101'):
        t = np.arange(16000 * 6) / 16000
        sf.write(folder / f'fb{i}.wav', (0.2 * np.sin(2 * np.pi * (500 + 70 * i) * t)).astype(np.float32), 16000)
        rows.append(dict(path=f'fb{i}.wav', group_id=f'fb{i}', label=label, split='train', source='feedback'))
    if duplicate_of is not None:  # a song that already sits in the main cache
        shutil.copy(duplicate_of, folder / 'dup.wav')
        rows.append(dict(path='dup.wav', group_id='dup', label='0', split='train', source='feedback'))
    write_rows(folder / 'manifest.csv', rows)
    with pytest.raises(ValueError, match='both labels'):
        prepare(folder / 'manifest.csv', tmp_path / f'{name}-strict', loudness_norm=loudness_norm)
    prepare(folder / 'manifest.csv', tmp_path / name, loudness_norm=loudness_norm, partial_splits=True)
    return tmp_path / name


def test_train_on_main_and_extra_cache(tmp_path):
    from hmf import train
    from hmf.finalize import finalize
    generate(tmp_path / 'demo')
    prepare(tmp_path / 'demo/manifest.csv', tmp_path / 'cache')
    main_rows = json.loads((tmp_path / 'cache/index.json').read_text())
    test_song = next(r['path'] for r in main_rows if r['split'] == 'test')
    extra = extra_cache(tmp_path, duplicate_of=test_song)
    caches = [str(tmp_path / 'cache'), str(extra)]
    _, rows, _ = train.load_caches(caches)
    added = [r for r in rows if r['_cache'] == 1]
    assert {r['track_id'].split(':')[0] for r in added} == {'c1'}
    assert len({r['track_id'] for r in added}) == 4  # the copied test song was dropped from the extra cache
    assert not {r['sha256'] for r in added} & {r['sha256'] for r in main_rows}
    train.main(run_args(tmp_path, 1, cache=caches))
    assert (tmp_path / 'run/data_index_1.json').is_file()
    report = json.loads((tmp_path / 'run/report.json').read_text())
    assert report['args']['cache'] == caches
    (tmp_path / 'run/report.json').unlink()
    with pytest.raises(ValueError, match='Jumlah cache'):
        train.main(run_args(tmp_path, 2, cache=caches[:1], resume=True))
    rebuilt = finalize(caches, tmp_path / 'run', 'frequency', batch_size=8, workers=0, monitor='loss')
    assert rebuilt['test']['accuracy'] == pytest.approx(report['test']['accuracy'])


def test_caches_with_different_audio_settings_are_refused(tmp_path):
    from hmf import train
    generate(tmp_path / 'demo')
    prepare(tmp_path / 'demo/manifest.csv', tmp_path / 'cache')
    extra = extra_cache(tmp_path, loudness_norm=True)
    with pytest.raises(ValueError, match='Pengaturan audio'):
        train.load_caches([tmp_path / 'cache', extra])
