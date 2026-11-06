"""Run the shared VideoFake pipeline on a persistent Runpod workspace."""
import argparse
import os
from pathlib import Path
import subprocess

os.environ.setdefault('VIDEOFAKE_HOME', '/workspace/videofake-live')
os.environ.setdefault('UV_PYTHON_INSTALL_DIR', '/workspace/videofake-live/python')
os.environ.setdefault('UV_CACHE_DIR', '/workspace/videofake-live/uv-cache')
os.environ.setdefault('PIP_NO_CACHE_DIR', '1')

import colab_runner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=['diagnose', 'repo', 'folders', 'setup', 'run'])
    parser.add_argument('--source', default='/workspace/videofake-live/app/face.jpg')
    parser.add_argument('--target', default='/workspace/videofake-live/app/input_video.mp4')
    parser.add_argument('--output', default='/workspace/output/out.mp4')
    parser.add_argument('--enhancer', action='store_true')
    parser.add_argument('--many-faces', action='store_true')
    parser.add_argument('--no-enhancer', dest='enhancer', action='store_false', help=argparse.SUPPRESS)
    parser.add_argument('--single-face', dest='many_faces', action='store_false', help=argparse.SUPPRESS)
    parser.set_defaults(enhancer=False, many_faces=False)
    args = parser.parse_args()
    try:
        revision = subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parent),
                                            'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'local'
    colab_runner.run_step(args.step, revision, {
        'source': args.source, 'target': args.target, 'output': args.output,
        'Processor': 'face_swapper face_enhancer' if args.enhancer else 'face_swapper',
        'ManyFaces': args.many_faces,
    })
