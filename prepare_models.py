"""Download the public model assets once per Pod, without processing a video."""
from concurrent.futures import ThreadPoolExecutor
import os
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile
import time


def prepare_models():
    root = Path(os.environ.get('VIDEOFAKE_HOME', '/opt/videofake'))
    models = root / 'app/roop/models'
    models.mkdir(parents=True, exist_ok=True)
    face_models = root / 'insightface/models/buffalo_l'
    expected = {'1k3d68.onnx', '2d106det.onnx', 'det_10g.onnx', 'genderage.onnx', 'w600k_r50.onnx'}
    jobs = [
        ('https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx', models / 'inswapper_128.onnx', 554253681),
        ('https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth', models / 'GFPGANv1.4.pth', 348632874),
    ]
    need_face_models = any(not (face_models / name).is_file() or not (face_models / name).stat().st_size for name in expected)
    if need_face_models:
        jobs.append(('https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip', root / 'buffalo_l.zip', 288621354))

    def download(job):
        url, target, expected_size = job
        if target.is_file() and target.stat().st_size == expected_size:
            print('Modello già presente:', target.name, flush=True)
            return
        part = target.with_suffix(target.suffix + '.part')
        print('Scarico:', target.name, flush=True)
        request = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
        with urllib.request.urlopen(request, timeout=120) as response:
            ranged = response.status == 206 and response.headers.get('Content-Range', '').endswith('/' + str(expected_size))
        if ranged:
            resume = part.with_suffix(part.suffix + '.json')
            start = json.loads(resume.read_text())['prefix'] if resume.is_file() else (part.stat().st_size if part.exists() else 0)
            if not 0 <= start < expected_size:
                start = 0
            resume.write_text(json.dumps({'prefix': start}))
            with part.open('ab') as output:
                output.truncate(expected_size)
            def segment(bounds):
                begin, end = bounds
                for attempt in range(4):
                    try:
                        request = urllib.request.Request(url, headers={'Range': f'bytes={begin}-{end}'})
                        with urllib.request.urlopen(request, timeout=120) as response, part.open('r+b') as output:
                            if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {begin}-{end}/'):
                                raise RuntimeError('Intervallo HTTP inatteso: ' + target.name)
                            output.seek(begin)
                            remaining = end - begin + 1
                            while remaining:
                                block = response.read(min(1024 * 1024, remaining))
                                if not block:
                                    raise RuntimeError('Download interrotto: ' + target.name)
                                output.write(block)
                                remaining -= len(block)
                        return
                    except Exception:
                        if attempt == 3:
                            raise
                        time.sleep(attempt + 1)
            chunk = 16 * 1024 * 1024
            with ThreadPoolExecutor(max_workers=16) as ranges:
                list(ranges.map(segment, [(p, min(p + chunk - 1, expected_size - 1)) for p in range(start, expected_size, chunk)]))
            resume.unlink(missing_ok=True)
        else:
            with urllib.request.urlopen(url, timeout=120) as response, part.open('wb') as output:
                shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
        if part.stat().st_size != expected_size:
            raise RuntimeError('Download incompleto o modello cambiato: ' + target.name)
        part.replace(target)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, jobs))
    if need_face_models:
        face_models.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(root / 'buffalo_l.zip') as archive:
            for info in archive.infolist():
                name = Path(info.filename).name
                if name in expected:
                    part = face_models / (name + '.part')
                    with archive.open(info) as source, part.open('wb') as target:
                        shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
                    part.replace(face_models / name)
        (root / 'buffalo_l.zip').unlink()
    if any(not (face_models / name).is_file() for name in expected):
        raise RuntimeError('Archivio InsightFace incompleto.')
    print('MODELLI PRONTI', flush=True)


if __name__ == '__main__':
    prepare_models()
