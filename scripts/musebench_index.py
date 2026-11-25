"""Petakan isi tar MUSEBENCH di Hugging Face tanpa mengunduh seluruh tar.

Tar menyimpan header 512 byte sebelum setiap file. Skrip ini membaca header itu
dengan HTTP Range, lalu melompat ke header berikutnya, sehingga hanya beberapa
kilobyte per file yang diunduh. Hasilnya bisa dipakai untuk mengunduh file
tertentu saja (subperintah `fetch`) tanpa mengambil tar 10 GB.

Contoh:
  python scripts/musebench_index.py index --out musebench_index.csv
  python scripts/musebench_index.py summary --index musebench_index.csv
  python scripts/musebench_index.py fetch --index musebench_index.csv \
      --prefix Jamendo/ --limit 500 --dest /content/musebench
"""
import argparse
import csv
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

OWNER = 'Anonymousv22222'
PARTS = [f'MuseBench-part{i}' for i in range(1, 6)]
BLOCK = 512
FIELDS = ['repo', 'tar', 'path', 'offset', 'size']
_local = threading.local()


def session():
    if not hasattr(_local, 's'):
        s = requests.Session()
        token = os.environ.get('HF_TOKEN')
        if token:
            s.headers['Authorization'] = f'Bearer {token}'
        _local.s = s
    return _local.s


def list_tars(repo):
    r = session().get(f'https://huggingface.co/api/datasets/{OWNER}/{repo}/tree/main', timeout=60)
    r.raise_for_status()
    return [(f['path'], f['size']) for f in r.json() if f['path'].endswith('.tar')]


def cdn_url(repo, tar):
    """Resolve once; the signed CDN URL avoids the hub resolver rate limit."""
    url = f'https://huggingface.co/datasets/{OWNER}/{repo}/resolve/main/{tar}'
    r = session().head(url, allow_redirects=False, timeout=60)
    if r.status_code in (301, 302, 307, 308):
        return r.headers['Location']
    r.raise_for_status()
    return url


def read_range(url, start, length, retries=5):
    for attempt in range(retries):
        try:
            r = session().get(url, headers={'Range': f'bytes={start}-{start + length - 1}'}, timeout=60)
            if r.status_code == 206:
                return r.content
            if r.status_code == 200:
                raise RuntimeError('Server mengabaikan Range; dibatalkan agar tidak mengunduh seluruh tar.')
            r.raise_for_status()
        except (requests.RequestException, RuntimeError) as error:
            if isinstance(error, RuntimeError) or attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def _str(raw):
    return raw.split(b'\0', 1)[0].decode('utf-8', 'replace')


def _size(raw):
    if raw[0] & 0x80:  # GNU base-256 for large files
        return int.from_bytes(raw[1:], 'big')
    text = raw.split(b'\0', 1)[0].strip()
    return int(text, 8) if text else 0


def _padded(n):
    return (n + BLOCK - 1) // BLOCK * BLOCK


def walk_tar(url, total, limit=None):
    """Yield (path, data_offset, size) for every regular file in the tar."""
    pos, pending_name, count = 0, None, 0
    while pos + BLOCK <= total:
        header = read_range(url, pos, BLOCK)
        if header == b'\0' * BLOCK:
            break
        size, kind = _size(header[124:136]), header[156:157]
        data = pos + BLOCK
        if kind in (b'L', b'x', b'g'):
            body = read_range(url, data, size) if size else b''
            if kind == b'L':
                pending_name = _str(body)
            elif kind == b'x':
                match = re.search(rb'\d+ path=([^\n]*)\n', body)
                if match:
                    pending_name = match.group(1).decode('utf-8', 'replace')
        else:
            name = pending_name
            if name is None:
                prefix = _str(header[345:500])
                name = (prefix + '/' if prefix else '') + _str(header[0:100])
            pending_name = None
            if kind in (b'0', b'\0'):
                yield name, data, size
                count += 1
                if limit and count >= limit:
                    return
        pos = data + _padded(size)


def index(args):
    out = Path(args.out)
    done = set()
    if out.exists():
        with out.open(encoding='utf-8', newline='') as f:
            done = {(r['repo'], r['tar']) for r in csv.DictReader(f)}
    jobs = []
    for repo in args.parts:
        for tar, size in list_tars(repo):
            if (repo, tar) in done and not args.limit:
                print(f'lewati (sudah ada): {repo}/{tar}')
                continue
            jobs.append((repo, tar, size))
    print(f'{len(jobs)} tar akan dipetakan dengan {args.workers} worker', flush=True)
    lock = threading.Lock()
    new = not out.exists()
    with out.open('a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, FIELDS)
        if new:
            writer.writeheader()

        def run(job):
            repo, tar, size = job
            url = cdn_url(repo, tar)
            rows = [dict(repo=repo, tar=tar, path=p, offset=o, size=s)
                    for p, o, s in walk_tar(url, size, args.limit)]
            with lock:  # write a whole tar at once so resume never sees half a tar
                writer.writerows(rows)
                f.flush()
            return repo, tar, len(rows)

        with ThreadPoolExecutor(args.workers) as pool:
            for future in as_completed(pool.submit(run, j) for j in jobs):
                try:
                    repo, tar, n = future.result()
                    print(f'selesai {repo}/{tar}: {n} file', flush=True)
                except Exception as error:  # keep other tars going; rerun resumes
                    print(f'GAGAL: {error}', file=sys.stderr, flush=True)


def load(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def folder(path, depth):
    parts = path.strip('./').split('/')
    return '/'.join(parts[:depth]) if len(parts) > depth else '/'.join(parts[:-1]) or '.'


def summary(args):
    rows = load(args.index)
    files, sizes, tars = Counter(), Counter(), defaultdict(set)
    for r in rows:
        key = folder(r['path'], args.depth)
        files[key] += 1
        sizes[key] += int(r['size'])
        tars[key].add(f"{r['repo'][-5:]}/{r['tar']}")
    print(f'{"folder":60} {"file":>8} {"GB":>8}  tar')
    for key in sorted(files):
        t = sorted(tars[key])
        shown = ', '.join(t[:4]) + (f' (+{len(t) - 4})' if len(t) > 4 else '')
        print(f'{key:60} {files[key]:8d} {sizes[key] / 1e9:8.2f}  {shown}')
    print(f'TOTAL {len(rows)} file, {sum(int(r["size"]) for r in rows) / 1e9:.1f} GB')


def fetch(args):
    rows = [r for r in load(args.index) if r['path'].strip('./').startswith(args.prefix)]
    rows.sort(key=lambda r: r['path'])
    if args.limit:
        rows = rows[:args.limit]
    dest = Path(args.dest)
    total = sum(int(r['size']) for r in rows)
    print(f'{len(rows)} file, {total / 1e9:.2f} GB -> {dest}', flush=True)
    urls, url_lock = {}, threading.Lock()

    def url_for(repo, tar):
        with url_lock:
            if (repo, tar) not in urls:
                urls[repo, tar] = cdn_url(repo, tar)
            return urls[repo, tar]

    def run(r):
        target = dest / r['path'].strip('./')
        size = int(r['size'])
        if target.exists() and target.stat().st_size == size:
            return 0
        target.parent.mkdir(parents=True, exist_ok=True)
        data = read_range(url_for(r['repo'], r['tar']), int(r['offset']), size) if size else b''
        if len(data) != size:
            raise ValueError(f'ukuran tidak cocok: {target}')
        tmp = target.with_suffix(target.suffix + '.part')
        tmp.write_bytes(data)
        tmp.replace(target)
        return size

    got = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for i, future in enumerate(as_completed(pool.submit(run, r) for r in rows), 1):
            try:
                got += future.result()
            except Exception as error:
                print(f'GAGAL: {error}', file=sys.stderr, flush=True)
            if i % 100 == 0 or i == len(rows):
                print(f'{i}/{len(rows)} file, {got / 1e9:.2f} GB baru', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('index', help='petakan isi tar lewat HTTP Range')
    a.add_argument('--out', default='musebench_index.csv')
    a.add_argument('--parts', nargs='+', default=PARTS)
    # Satu tar dibaca berurutan (header berikutnya bergantung pada ukuran file),
    # jadi paralelisme hanya antar tar; ada 75 tar total.
    a.add_argument('--workers', type=int, default=32)
    a.add_argument('--limit', type=int, help='maksimum file per tar (untuk uji coba)')
    a.set_defaults(func=index)
    b = sub.add_parser('summary', help='ringkas jumlah file dan ukuran per folder')
    b.add_argument('--index', default='musebench_index.csv')
    b.add_argument('--depth', type=int, default=2)
    b.set_defaults(func=summary)
    c = sub.add_parser('fetch', help='unduh hanya file dengan awalan path tertentu')
    c.add_argument('--index', default='musebench_index.csv')
    c.add_argument('--prefix', required=True, help='mis. Jamendo/ atau Musicgen_generated_music/')
    c.add_argument('--dest', required=True)
    c.add_argument('--limit', type=int)
    c.add_argument('--workers', type=int, default=8)
    c.set_defaults(func=fetch)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
