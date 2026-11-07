"""Download the pinned Linux/Python 3.10 CUDA wheel with parallel HTTP ranges."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import os
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request

def download():
    root = Path(os.environ.get('VIDEOFAKE_HOME', '/opt/videofake')) / 'wheels'
    root.mkdir(parents=True, exist_ok=True)
    name = 'torch-2.0.1+cu118-cp310-cp310-linux_x86_64.whl'
    target = root / name
    index = 'https://download.pytorch.org/whl/cu118/torch/'
    with urllib.request.urlopen(index, timeout=60) as response:
        page = response.read().decode()
    links = re.findall(r'href="([^"]+)"', page)
    link = next(html.unescape(link) for link in links if urllib.parse.unquote(urllib.parse.urlsplit(link).path).endswith('/' + name))
    parsed = urllib.parse.urlsplit(urllib.parse.urljoin(index, link))
    expected = urllib.parse.parse_qs(parsed.fragment)['sha256'][0]
    url = urllib.parse.urlunsplit(parsed._replace(fragment=''))
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()
    if target.is_file() and digest(target) == expected:
        print('PyTorch già scaricato e verificato.', flush=True)
        return
    with urllib.request.urlopen(urllib.request.Request(url, headers={'Range': 'bytes=0-0'}), timeout=60) as response:
        if response.status != 206:
            raise RuntimeError('Il server non supporta il download parallelo.')
        total = int(response.headers['Content-Range'].split('/')[-1])
    partial = target.with_suffix('.whl.part')
    start = 0
    if len(sys.argv) > 1:
        seed = Path(sys.argv[1])
        if seed.is_file() and seed.stat().st_size < total:
            shutil.copyfile(seed, partial)
            start = seed.stat().st_size
    with partial.open('ab') as stream:
        stream.truncate(total)
    fd = os.open(partial, os.O_WRONLY)
    chunk = 32 * 1024 * 1024
    def segment(bounds):
        begin, end = bounds
        for attempt in range(4):
            try:
                request = urllib.request.Request(url, headers={'Range': f'bytes={begin}-{end}'})
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {begin}-{end}/'):
                        raise RuntimeError('Intervallo HTTP inatteso.')
                    position = begin
                    while position <= end:
                        block = response.read(min(1024 * 1024, end + 1 - position))
                        if not block:
                            raise RuntimeError('Download incompleto.')
                        written = 0
                        while written < len(block):
                            written += os.pwrite(fd, block[written:], position + written)
                        position += len(block)
                return end - begin + 1
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)
    print('Scarico PyTorch con 16 connessioni; verifico SHA256 ufficiale.', flush=True)
    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(segment, [(begin, min(begin + chunk - 1, total - 1)) for begin in range(start, total, chunk)]))
    finally:
        os.close(fd)
    if digest(partial) != expected:
        raise RuntimeError('SHA256 PyTorch non valido: file non utilizzato.')
    partial.replace(target)
    print('PyTorch scaricato e verificato.', flush=True)

if __name__ == '__main__':
    download()
