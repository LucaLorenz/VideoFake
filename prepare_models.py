"""Download the public model assets once per Pod, without processing a video."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import urllib.request
import zipfile


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
