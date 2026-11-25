from dataclasses import asdict, dataclass
import random
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path
import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    seconds: float = 5.0
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 64
    canonical_mp3: bool = False
    # RMS normalization per 5 s window. Default False keeps old reports readable.
    loudness_norm: bool = False
    # Upper mel frequency. None = Nyquist (old runs). SONICS MP3s have no energy above
    # ~7.3 kHz, so any band above that is a pure codec cue; 7000 hides it from the CNN.
    fmax: float | None = None


LOUDNESS_TARGET_DB = -20.0
# codec name -> (ffmpeg encoder, container extension, allowed bitrates in kbps)
CODECS = {
    'mp3': ('libmp3lame', 'mp3', (32, 48, 64, 96, 128, 160, 192, 256, 320)),
    'aac': ('aac', 'm4a', (32, 48, 64, 96, 128, 160, 192, 256, 320)),
    'opus': ('libopus', 'ogg', (32, 48, 64, 96, 128, 160, 192, 256)),
    'pcm': ('pcm_s16le', 'wav', (None,)),
}
CODEC_WEIGHTS = {'mp3': 0.35, 'aac': 0.3, 'opus': 0.2, 'pcm': 0.15}
SAMPLE_RATES = (16000, 22050, 24000, 32000, 44100, 48000)


def sample_augmentation(key):
    """Random codec chain, fully determined by `key` (e.g. seed:sha256:variant).

    The same distribution is used for human and AI tracks, so codec, bitrate,
    bandwidth and level stop being label cues.
    """
    rng = random.Random(str(key))
    codec = rng.choices(list(CODEC_WEIGHTS), weights=list(CODEC_WEIGHTS.values()))[0]
    lowpass = None
    if rng.random() < 0.4:
        # Covers the ~7.3 kHz cutoff of low-quality generator MP3s.
        lowpass = rng.randrange(4000, 7900, 100)
    return dict(codec=codec, bitrate=rng.choice(CODECS[codec][2]),
                sample_rate=rng.choice(SAMPLE_RATES), channels=rng.choice((1, 2)),
                gain_db=round(rng.uniform(-12.0, 4.0), 2), lowpass=lowpass)


def _ffmpeg(args):
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-v', 'error', '-y'] + args,
                   check=True, capture_output=True, timeout=600)


def transcode(path, folder, params, max_seconds=None):
    """Apply gain/lowpass/resample/codec, then decode back to float WAV."""
    encoder, extension, _ = CODECS[params['codec']]
    encoded = Path(folder) / f'encoded.{extension}'
    decoded = Path(folder) / 'augmented.wav'
    filters = [f"volume={params['gain_db']}dB"]
    if params.get('lowpass'):
        filters.append(f"lowpass=f={params['lowpass']}")
    filters.append(f"aresample={params['sample_rate']}")
    command = ['-i', str(path), '-map', '0:a:0', '-map_metadata', '-1']
    if max_seconds is not None:
        command += ['-t', str(max_seconds)]
    command += ['-af', ','.join(filters), '-ac', str(params['channels']), '-c:a', encoder]
    if params['codec'] != 'opus':
        # libopus only accepts 8/12/16/24/48 kHz; ffmpeg upsamples after aresample.
        command += ['-ar', str(params['sample_rate'])]
    if params['bitrate']:
        command += ['-b:a', f"{params['bitrate']}k"]
    _ffmpeg(command + [str(encoded)])
    _ffmpeg(['-i', str(encoded), '-c:a', 'pcm_f32le', str(decoded)])
    return decoded


CANONICAL_MP3 = dict(codec='mp3', bitrate=32, sample_rate=16000, channels=1, gain_db=0.0, lowpass=None)


def features(path, config, max_segments=None, augment=None):
    """Yield log-mel windows. `augment` is a dict from sample_augmentation."""
    params = augment or (CANONICAL_MP3 if config.canonical_mp3 else None)
    if params is None:
        yield from raw_features(path, config, max_segments)
        return
    max_seconds = None if max_segments is None else max_segments * config.seconds
    with TemporaryDirectory(prefix='hmf-audio-') as folder:
        if augment and config.canonical_mp3:
            # Keep the canonical chain as the final step so train matches inference.
            path = transcode(path, folder, augment, max_seconds)
            inner = Path(folder) / 'canonical'
            inner.mkdir()
            decoded = transcode(path, inner, CANONICAL_MP3)
        else:
            decoded = transcode(path, folder, params, max_seconds)
        yield from raw_features(decoded, config, max_segments)


def normalize_loudness(x, target_db=LOUDNESS_TARGET_DB):
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    if rms < 1e-4:  # leave near-silence alone instead of amplifying noise
        return x
    return (x * (10 ** (target_db / 20) / rms)).astype(np.float32)


def raw_features(path, config, max_segments=None):
    # Stream fixed windows rather than decoding a whole long recording into RAM.
    with sf.SoundFile(path) as f:
        size = max(1, round(f.samplerate * config.seconds))
        count = 0
        while max_segments is None or count < max_segments:
            x = f.read(size, dtype='float32', always_2d=True)
            if len(x) == 0:
                break
            x = x.mean(axis=1)
            if not np.isfinite(x).all():
                raise ValueError(f'Non-finite audio: {path}')
            x = librosa.resample(x, orig_sr=f.samplerate, target_sr=config.sample_rate)
            length = round(config.sample_rate * config.seconds)
            if config.loudness_norm:
                x = normalize_loudness(x)
            x = librosa.util.fix_length(x, size=length)
            mel = librosa.feature.melspectrogram(y=x, sr=config.sample_rate,
                n_fft=config.n_fft, hop_length=config.hop_length, n_mels=config.n_mels,
                fmax=config.fmax or config.sample_rate / 2, power=2)
            count += 1
            # Fixed dB scale, no dataset-wide normalization or peak normalization.
            yield (librosa.power_to_db(mel, ref=1.0, top_db=None).clip(-80, 20) + 30) / 50
