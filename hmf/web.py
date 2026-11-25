from argparse import ArgumentParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote
import hashlib
import json
import subprocess

import imageio_ffmpeg
import soundfile as sf

from . import feedback
from .predict import analyze, model_summary


def readable_audio(path):
    try:
        with sf.SoundFile(path):
            return path
    except sf.LibsndfileError:
        pass

    decoded = path.with_suffix('.wav')
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-v', 'error',
        '-protocol_whitelist', 'file',
        '-format_whitelist', 'mov,mp3,aac,ogg,flac,wav,asf,matroska,webm,aiff,amr,ac3,eac3,dts,ape,wv,tta,caf,au,mpc,mpc8,dsf,dff,oma,flv,mpeg,mpegts,w64,spdif,tak',
        '-i', str(path), '-map', '0:a:0', '-map_metadata', '-1',
        '-c:a', 'pcm_f32le', '-fs', '536870912', str(decoded)
    ], check=True, timeout=120, stderr=subprocess.DEVNULL)
    if decoded.stat().st_size >= 536870912:
        raise ValueError('Audio terlalu panjang setelah didekode.')
    return decoded


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, data, mime):
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def reply_json(self, status, value):
        self.reply(status, json.dumps(value, ensure_ascii=False).encode(), 'application/json; charset=utf-8')

    def feedback_summary(self):
        records = feedback.load(self.server.feedback_dir)
        return dict(feedback.summarize(records, Path(self.server.run_path).name),
                    label_sources=feedback.LABEL_SOURCES)

    def read_body(self, limit):
        size = int(self.headers.get('Content-Length', 0))
        if not 0 < size <= limit:
            raise ValueError(f'Ukuran data harus 1 byte sampai {limit // 1024} KB.')
        return self.rfile.read(size)

    def do_GET(self):
        if self.path == '/feedback':
            return self.reply_json(200, self.feedback_summary())
        if self.path == '/model':
            try:
                summary = model_summary(self.server.run_path)
            except Exception:
                return self.reply(500, b'Info model tidak bisa dibaca dari report.json.', 'text/plain')
            return self.reply(200, json.dumps(summary).encode(), 'application/json')
        if self.path != '/':
            return self.send_error(404)
        self.reply(200, Path(__file__).with_name('studio.html').read_bytes(), 'text/html; charset=utf-8')

    def do_POST(self):
        if self.path == '/feedback':
            return self.post_feedback()
        if self.path == '/feedback/audio':
            return self.post_feedback_audio()
        if self.path != '/analyze':
            return self.send_error(404)
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 32 * 1024**2:
                raise ValueError()
            with TemporaryDirectory() as folder:
                path = Path(folder) / 'upload'
                data = self.rfile.read(size)
                path.write_bytes(data)
                try:
                    source = readable_audio(path)
                except (subprocess.SubprocessError, ValueError):
                    return self.reply(400, b'Audio gagal dibaca. Format tidak didukung, file rusak, atau rekaman terlalu panjang.', 'text/plain')
                filename = Path(unquote(self.headers.get('X-Audio-Filename', 'upload'))).name
                result = analyze(self.server.run_path, source, filename=filename, include_segments=True,
                                 include_spectrogram=True)
                # Identifies the song for feedback without keeping the audio.
                result['metadata']['sha256'] = hashlib.sha256(data).hexdigest()
        except Exception:
            return self.reply(400, b'Analisis gagal. Periksa file audio dan folder model.', 'text/plain')
        self.reply(200, json.dumps(result).encode(), 'application/json')

    def post_feedback(self):
        try:
            entry = json.loads(self.read_body(256 * 1024))
            record = feedback.validate(entry)
            record['run'] = Path(self.server.run_path).name  # the server knows which model answered
            feedback.append(self.server.feedback_dir, record)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            return self.reply(400, str(error).encode(), 'text/plain; charset=utf-8')
        self.reply_json(200, dict(saved=record['id'], true_label=record['true_label'], summary=self.feedback_summary()))

    def post_feedback_audio(self):
        try:
            data = self.read_body(32 * 1024**2)
            sha = hashlib.sha256(data).hexdigest()
            if sha != self.headers.get('X-Sha256', ''):
                raise ValueError('File berbeda dari lagu yang dianalisis; audio tidak disimpan.')
            run = Path(self.server.run_path).name
            record = next((r for r in feedback.load(self.server.feedback_dir)
                           if r['sha256'] == sha and r['run'] == run), None)
            if record is None:
                raise ValueError('Simpan umpan balik dulu sebelum menyimpan audionya.')
            name = unquote(self.headers.get('X-Audio-Filename', record['filename']))
            feedback.save_audio(self.server.feedback_dir, record, data, name)
        except ValueError as error:
            return self.reply(400, str(error).encode(), 'text/plain; charset=utf-8')
        self.reply_json(200, dict(audio_saved=True, summary=self.feedback_summary()))


def make_server(run, port=8765, feedback_dir=feedback.DEFAULT_DIR):
    # Windows SO_REUSEADDR lets a second server bind the same port silently; requests
    # then reach whichever server Windows picks (possibly with another model). Refuse instead.
    ThreadingHTTPServer.allow_reuse_address = False
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.run_path = Path(run)
    server.feedback_dir = Path(feedback_dir)
    return server


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--run', type=Path)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--feedback-dir', type=Path, default=feedback.DEFAULT_DIR,
                        help='folder umpan balik pengguna (default data/feedback)')
    args = parser.parse_args()
    if args.run is None:
        runs = Path(__file__).resolve().parents[1] / 'runs'
        completed = [p for p in runs.glob('*/best.pt') if (p.parent / 'report.json').is_file()]
        if not completed:
            parser.error('Tidak ada run lengkap dengan best.pt dan report.json.')
        args.run = max(completed, key=lambda p: p.stat().st_mtime_ns).parent
    try:
        server = make_server(args.run, args.port, args.feedback_dir)
    except OSError:
        parser.error(f'Port {args.port} sudah dipakai (mungkin studio lain masih berjalan). '
                     f'Hentikan dengan Ctrl+C di terminalnya atau pakai --port lain.')
    print(f'Model aktif: {args.run.resolve()}', flush=True)
    print(f'Umpan balik disimpan di: {args.feedback_dir.resolve()}', flush=True)
    print(f'Buka http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()

