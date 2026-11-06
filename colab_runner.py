"""Reloaded from GitHub for each Colab step; no browser or GitHub token needed."""
import hashlib
import html
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request

ROOT = Path(os.environ.get('VIDEOFAKE_HOME', '/content/videofake-live'))
REPO = ROOT / 'app'
PROJECT = REPO / 'roop'
ENV = ROOT / 'env'
PYTHON = str(ENV / 'bin/python')
UPSTREAM = '1b4f9f43d42de0b4a2d16ff58075464f676ba8dc'
REQUIREMENTS = '''numpy==1.24.3
opencv-python==4.8.0.76
opencv-python-headless==4.8.0.76
onnx==1.14.0
insightface==0.7.3
psutil==5.9.5
pillow==10.0.0
onnxruntime-gpu==1.15.1
tensorflow-cpu==2.13.0
protobuf==4.23.4
tqdm==4.65.0
gfpgan==1.3.8
basicsr==1.4.2
albumentations==1.3.1
scipy==1.11.4
scikit-image==0.21.0
scikit-learn==1.3.2
torch==2.0.1+cu118
torchvision==0.15.2+cu118
nvidia-cublas-cu11==11.11.3.6
nvidia-cudnn-cu11==8.9.6.50
nvidia-cuda-runtime-cu11==11.8.89
nvidia-cufft-cu11==10.9.0.58
nvidia-curand-cu11==10.3.0.86
'''
INSTALL_ID = hashlib.sha256(('python3.10-build1\n' + REQUIREMENTS).encode()).hexdigest()
LOG = None


def live_command(args, stream, **kwargs):
    """Stream frame progress to Jupyter or terminal, preserving the full log."""
    handle = None
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
        if get_ipython() is not None:
            handle = display(HTML('<b>Elaborazione avviata. Attendo il primo avanzamento…</b>'), display_id=True)
    except ImportError:
        pass
    last_update = 0
    stage = 'Preparazione'
    pattern = re.compile(r'(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([^\]]+)\]')
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding='utf-8', errors='replace', **kwargs)
    pending = ''
    def emit(line):
        nonlocal last_update, stage
        if not line.strip():
            return
        if line.startswith('[ROOP.'):
            stage = line.strip()
        match = pattern.search(line)
        if not match and not line.startswith('[ROOP.'):
            return
        now = time.monotonic()
        if match and now - last_update < 1 and match.group(1) != '100':
            return
        last_update = now
        if handle is not None:
            content = '<b>' + html.escape(stage) + '</b><br>'
            if match:
                percent, done, total, timing = match.groups()
                content += (f'<progress value="{percent}" max="100" style="width:95%;height:24px"></progress>'
                            f'<br>{percent}% · {done}/{total} · {html.escape(timing)}')
            else:
                content += html.escape(line.strip())
            handle.update(HTML(content))
        else:
            print(line.strip(), flush=True)
    try:
        for char in iter(lambda: process.stdout.read(1), ''):
            stream.write(char)
            if char in '\r\n':
                stream.flush()
                emit(pending)
                pending = ''
            else:
                pending += char
        if pending:
            emit(pending)
        code = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        process.stdout.close()
    if code:
        raise subprocess.CalledProcessError(code, args)


def checked(label, args, live=False, **kwargs):
    """Full output stays on the runtime; the notebook receives only error tails."""
    print(label, flush=True)
    with LOG.open('a') as stream:
        stream.write('\n--- ' + label + ' ---\n')
        stream.flush()
        try:
            command = [str(a) for a in args]
            if live:
                live_command(command, stream, **kwargs)
            else:
                subprocess.run(command, check=True, stdout=stream,
                               stderr=subprocess.STDOUT, **kwargs)
        except subprocess.CalledProcessError as error:
            # Do not echo a whole inline Python probe into the short report.
            cause = next((line.strip() for line in reversed(tail(LOG)) if line.strip()), '')
            raise RuntimeError(f'{label}: processo terminato con codice {error.returncode}. {cause}') from None


def run_env():
    site = ENV / 'lib/python3.10/site-packages'
    libs = [str(p) for p in sorted((site / 'nvidia').glob('*/lib'))] + [str(site / 'torch/lib')]
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = ':'.join(libs + [env.get('LD_LIBRARY_PATH', '')])
    env['PATH'] = str(ENV / 'bin') + ':' + env.get('PATH', '')
    env['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # Colab's inline backend belongs to its kernel, not the isolated CLI env.
    env['MPLBACKEND'] = 'Agg'
    env['PYTHONUNBUFFERED'] = '1'
    env['VIDEOFAKE_MODEL_ROOT'] = str(ROOT / 'insightface')
    return env


def prepare():
    if not (REPO / '.git').is_dir():
        if REPO.exists():
            raise RuntimeError(f'{REPO} esiste ma non è un checkout. Nessun file cancellato.')
        checked('Scarico applicazione originale',
                ['git', 'clone', '--no-checkout', 'https://github.com/neuromodern/VideoFake.git', REPO])
        checked('Seleziono versione verificata', ['git', '-C', REPO, 'checkout', '--detach', UPSTREAM])
    head = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    if head != UPSTREAM:
        raise RuntimeError('Checkout inatteso: usa una nuova cartella VIDEOFAKE_HOME.')
    (PROJECT / 'models').mkdir(parents=True, exist_ok=True)
    analyser = PROJECT / 'roop/face_analyser.py'
    source = analyser.read_text()
    old = "insightface.app.FaceAnalysis(name='buffalo_l', providers="
    if old in source:
        source = 'import os\n' + source.replace(old,
            "insightface.app.FaceAnalysis(name='buffalo_l', root=os.environ.get('VIDEOFAKE_MODEL_ROOT', '~/.insightface'), providers=")
        compile(source, str(analyser), 'exec')
        analyser.write_text(source)


def patched_core(source):
    # CLI must not import Tk; preload PyTorch before ONNX Runtime for CUDA.
    if '\nimport roop.ui as ui\n' not in source:
        if '# videofake-colab-headless' in source:
            return source
        raise RuntimeError('core.py inatteso: patch non applicata.')
    source = source.replace('\nimport onnxruntime\n', '\nimport torch\nimport onnxruntime\n')
    source = source.replace('\nimport roop.ui as ui\n', '\nui = None  # videofake-colab-headless\n')
    source = source.replace('def run() -> None:\n', 'def run() -> None:\n    global ui\n')
    source = source.replace('        window = ui.init(start, destroy)',
                            '        import roop.ui as ui\n        window = ui.init(start, destroy)')
    compile(source, 'roop/core.py', 'exec')
    return source


def verify():
    probe = '''
import os, ctypes
require_gpu = os.environ.get('VIDEOFAKE_REQUIRE_GPU', '1') == '1'
if require_gpu:
    for library in ['libcudart.so.11.0', 'libcublasLt.so.11', 'libcublas.so.11',
                    'libcudnn.so.8', 'libcufft.so.10', 'libcurand.so.10']:
        try:
            ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
        except OSError as error:
            raise RuntimeError(f'Libreria CUDA non caricabile: {library}: {error}') from None
import torch, numpy as np, onnx, onnxruntime as ort
from onnx import helper, TensorProto
import roop.core
from roop.processors.frame.core import get_frame_processors_modules
get_frame_processors_modules(['face_swapper', 'face_enhancer'])
if require_gpu and not torch.cuda.is_available():
    raise RuntimeError('GPU non disponibile: scegli runtime T4 e riprova.')
provider = 'CUDAExecutionProvider' if require_gpu else 'CPUExecutionProvider'
x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [2, 2])
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2])
graph = helper.make_graph([helper.make_node('MatMul', ['x', 'x'], ['y'])], 'check', [x], [y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
model.ir_version = 8
session = ort.InferenceSession(model.SerializeToString(), providers=[provider])
if session.get_providers()[0] != provider:
    raise RuntimeError('ONNX Runtime non riesce ad attivare ' + provider)
np.testing.assert_allclose(session.run(None, {'x': np.eye(2, dtype=np.float32)})[0], np.eye(2))
print('Python:', __import__('sys').version.split()[0], 'Provider:', provider)
'''
    checked('Controllo dipendenze', [PYTHON, '-m', 'pip', 'check'])
    checked('Controllo import e inferenza ONNX', [PYTHON, '-c', probe], env=run_env(), cwd=PROJECT)
    checked('Controllo avvio CLI', [PYTHON, 'run.py', '--help'], env=run_env(), cwd=PROJECT)


def setup():
    (ROOT / 'installed.json').unlink(missing_ok=True)
    prepare()
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise RuntimeError('ffmpeg/ffprobe mancanti nel runtime.')
    tools = ROOT / 'tools'
    if not (tools / 'uv').is_dir():
        checked('Installo gestore Python separato', [sys.executable, '-m', 'pip', 'install',
                '--target', tools, 'uv==0.9.5'])
    tool_env = os.environ.copy()
    tool_env['PYTHONPATH'] = str(tools)
    if not (ENV / 'bin/python').exists():
        checked('Creo ambiente Python 3.10', [sys.executable, '-m', 'uv', 'venv',
                '--python', '3.10', '--seed', ENV], env=tool_env)
    checked('Installo strumenti di compilazione', [PYTHON, '-m', 'pip', 'install',
            'pip==24.3.1', 'setuptools==68.2.2', 'wheel==0.41.3',
            'Cython==0.29.36', 'numpy==1.24.3'])
    checked('Installo PyTorch CUDA compatibile', [PYTHON, '-m', 'pip', 'install',
            'torch==2.0.1+cu118', 'torchvision==0.15.2+cu118',
            '--index-url', 'https://download.pytorch.org/whl/cu118'])
    req = ROOT / 'requirements-colab.txt'
    req.write_text(REQUIREMENTS)
    checked('Installo dipendenze applicazione', [PYTHON, '-m', 'pip', 'install',
            '--no-build-isolation', '-r', req])
    core = PROJECT / 'roop/core.py'
    core.write_text(patched_core(core.read_text()))
    verify()
    (ROOT / 'installed.json').write_text(json.dumps({'install_id': INSTALL_ID}))


def arguments(config):
    args = [PYTHON, 'run.py', '--source', config['source'], '--target', config['target'],
            '--output', config['output'], '--execution-provider', config.get('Device', 'cuda'),
            '--frame-processor', *config.get('Processor', 'face_swapper face_enhancer').split(),
            '--output-video-encoder', config.get('VideoEncoder', 'libx264'),
            '--output-video-quality', str(config.get('VideoQuality', '18'))]
    for enabled, flag in [(config.get('KeepFPS', True), '--keep-fps'),
                          (not config.get('KeepAudio', True), '--skip-audio'),
                          (config.get('KeepFrames', False), '--keep-frames'),
                          (config.get('ManyFaces', True), '--many-faces')]:
        if enabled:
            args.append(flag)
    return args


def process(config):
    marker = ROOT / 'installed.json'
    if not marker.is_file() or json.loads(marker.read_text()).get('install_id') != INSTALL_ID:
        raise RuntimeError('Ambiente assente o dipendenze cambiate: riesegui step 3.')
    for name in ('source', 'target'):
        if not Path(config[name]).is_file():
            raise FileNotFoundError(f'{name}: {config[name]}. Carica il file e correggi step 4.')
    if Path(config['output']).resolve() in (Path(config['source']).resolve(), Path(config['target']).resolve()):
        raise ValueError('Output deve essere diverso dai file di input.')
    prepare()
    Path(config['output']).parent.mkdir(parents=True, exist_ok=True)
    # Frame extraction creates many files. Keep them on the local container
    # disk, rather than on Runpod's persistent network filesystem.
    with tempfile.TemporaryDirectory(prefix='videofake-') as scratch:
        scratch = Path(scratch)
        local = dict(config)
        for key in ('source', 'target'):
            destination = scratch / (key + Path(config[key]).suffix)
            shutil.copy2(config[key], destination)
            local[key] = str(destination)
        local['output'] = str(scratch / ('result' + Path(config['output']).suffix))
        checked('Elaborazione in corso — frame sul disco locale', arguments(local), live=True, env=run_env(), cwd=PROJECT)
        result = Path(local['output'])
        if not result.is_file() or result.stat().st_size == 0:
            raise RuntimeError('Nessun output creato: consulta il rapporto DEBUG.')
        shutil.copy2(result, config['output'])
        if config.get('KeepFrames', False) and (scratch / 'temp').is_dir():
            kept = Path(config['output']).parent / (scratch.name + '-frames')
            shutil.copytree(scratch / 'temp', kept)
            print('Frame conservati:', kept)
    print('Output:', config['output'])


def tail(path, lines=24, chars=5000):
    if not path.is_file():
        return []
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 24000))
        text = stream.read().decode('utf-8', errors='replace')
    # Keep the failing command's output, without unrelated pip messages.
    if '\n--- ' in text:
        text = text.rsplit('\n--- ', 1)[1]
    return text[-chars:].splitlines()[-lines:]


def present_status(report):
    failed = report['status'] == 'error'
    number = {'diagnose': '0', 'repo': '1', 'folders': '2', 'setup': '3', 'run': '5'}.get(report['step'], report['step'])
    title = f"STEP {number} {'FALLITO — FERMATI QUI' if failed else 'COMPLETATO'}"
    detail = report.get('error', 'Puoi passare allo step successivo.')
    color, background = ('#fca5a5', '#450a0a') if failed else ('#86efac', '#052e16')
    markup = (f'<div role="alert" style="border:3px solid {color};background:{background};'
              f'color:{color};padding:16px;margin:12px 0;border-radius:8px;font-size:16px">'
              f'<strong style="font-size:22px">{html.escape(title)}</strong>'
              f'<pre style="white-space:pre-wrap;color:inherit;margin-bottom:0">{html.escape(detail)}</pre></div>')
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
        shell = get_ipython()
        if shell is not None:
            # Keep step 6 accurate even when an older bootstrap is still open.
            shell.user_ns['VF_DEBUG'] = report
            display(HTML(markup))
            return
    except ImportError:
        pass
    ansi = '\033[1;31m' if failed else '\033[1;32m'
    print(ansi + title + '\n' + detail + '\033[0m', file=sys.stderr, flush=True)


def run_step(step, revision, config=None):
    global LOG
    started = time.monotonic()
    ROOT.mkdir(parents=True, exist_ok=True)
    LOG = ROOT / f'{step}-{time.time_ns()}.log'
    LOG.touch()
    report = {'revision': revision, 'step': step, 'status': 'ok',
              'kernel_python': sys.version.split()[0]}
    print(f'GitHub {revision[:12]} | step {step}', flush=True)
    try:
        if step == 'diagnose':
            print('Kernel Colab:', sys.version.split()[0], '| applicazione: Python 3.10 separato')
            if shutil.which('nvidia-smi'):
                checked('Controllo GPU', ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'])
            print('Nessuna reinstallazione CUDA di sistema necessaria.')
        elif step in ('repo', 'folders'):
            prepare()
        elif step == 'setup':
            setup()
        elif step == 'run':
            process(config or {})
        else:
            raise ValueError('Step sconosciuto: ' + step)
    except Exception as error:
        report.update(status='error', error=f'{type(error).__name__}: {error}',
                      traceback=traceback.format_exc().splitlines()[-8:], log_tail=tail(LOG))
    report['log_file'] = str(LOG)
    report['elapsed_seconds'] = round(time.monotonic() - started, 2)
    (ROOT / 'debug.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    present_status(report)
    print('DEBUG_BEGIN\n' + json.dumps(report, ensure_ascii=False, indent=2) + '\nDEBUG_END')
    if report['status'] == 'error':
        print('Fermati qui. Condividi soltanto il blocco DEBUG oppure salva il notebook con output su GitHub.')
        # Older notebook bootstraps catch Exception. SystemExit still marks the
        # cell as failed, without requiring the user to reload their notebook.
        raise SystemExit(f"STEP {step} FALLITO: {report['error']}")
    return report
