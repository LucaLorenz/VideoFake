import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('runner', HERE / 'colab_runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ColabTests(unittest.TestCase):
    def test_lean_profile_removes_only_unused_tensorflow_dependency(self):
        with patch.dict(os.environ, {'VIDEOFAKE_INSTALL_PROFILE': 'lean'}):
            fresh_spec = importlib.util.spec_from_file_location('lean_runner', HERE / 'colab_runner.py')
            lean = importlib.util.module_from_spec(fresh_spec)
            fresh_spec.loader.exec_module(lean)
        self.assertNotIn('tensorflow-cpu', lean.REQUIREMENTS)
        self.assertIn('onnxruntime-gpu==1.15.1', lean.REQUIREMENTS)
        self.assertIn('gfpgan==1.3.8', lean.REQUIREMENTS)
        self.assertIn('tensorflow-cpu', runner.REQUIREMENTS)

    def test_model_preparation_reuses_complete_downloads(self):
        fresh_spec = importlib.util.spec_from_file_location('assets', HERE / 'prepare_models.py')
        assets = importlib.util.module_from_spec(fresh_spec)
        fresh_spec.loader.exec_module(assets)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            models = root / 'app/roop/models'
            models.mkdir(parents=True)
            for name, size in [('inswapper_128.onnx', 554253681), ('GFPGANv1.4.pth', 348632874)]:
                with (models / name).open('wb') as stream:
                    stream.truncate(size)
            face_models = root / 'insightface/models/buffalo_l'
            face_models.mkdir(parents=True)
            for name in ('1k3d68.onnx', '2d106det.onnx', 'det_10g.onnx', 'genderage.onnx', 'w600k_r50.onnx'):
                (face_models / name).write_bytes(b'model')
            with patch.dict(os.environ, {'VIDEOFAKE_HOME': folder}), patch('urllib.request.urlopen', side_effect=AssertionError('unexpected download')):
                with contextlib.redirect_stdout(io.StringIO()):
                    assets.prepare_models()

    def test_notebook_code_is_valid_python(self):
        notebook = json.loads((HERE / 'VideoFake_colab.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
                self.assertEqual(cell['outputs'], [])

    def test_audio_and_paths_are_passed_correctly(self):
        config = {'source': '/tmp/a face.jpg', 'target': '/tmp/a video.mp4', 'output': '/tmp/out.mp4'}
        args = runner.arguments(config)
        self.assertNotIn('--skip-audio', args)
        self.assertIn('/tmp/a face.jpg', args)
        self.assertIn('--output-video-encoder', args)
        self.assertNotIn('--many-faces', args)
        processors = args[args.index('--frame-processor') + 1:args.index('--output-video-encoder')]
        self.assertEqual(processors, ['face_swapper'])
        config['KeepAudio'] = False
        self.assertIn('--skip-audio', runner.arguments(config))

    def test_headless_patch_is_valid_and_idempotent(self):
        source = ('import onnxruntime\nimport roop.ui as ui\n'
                  'def run() -> None:\n    if True:\n        pass\n'
                  '    else:\n        window = ui.init(start, destroy)\n')
        patched = runner.patched_core('\n' + source)
        ast.parse(patched)
        self.assertLess(patched.index('import torch'), patched.index('import onnxruntime'))
        self.assertEqual(patched, runner.patched_core(patched))
        self.assertNotIn('\nimport roop.ui as ui\n', patched)

    def test_error_report_is_short_and_saved(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(runner, 'ROOT', Path(folder)):
            def fail():
                runner.LOG.write_text(('verbose build output\n' * 10000) + 'ROOT CAUSE\n')
                raise RuntimeError('install failed')
            with patch.object(runner, 'setup', fail), contextlib.redirect_stdout(io.StringIO()) as output:
                with contextlib.redirect_stderr(io.StringIO()) as error_output:
                    with self.assertRaises(SystemExit):
                        runner.run_step('setup', 'a' * 40)
                report = json.loads((Path(folder) / 'debug.json').read_text())
            self.assertIn('STEP 3 FALLITO', error_output.getvalue())
            self.assertIn('\033[1;31m', error_output.getvalue())
            self.assertEqual(report['status'], 'error')
            self.assertEqual(report['log_tail'][-1], 'ROOT CAUSE')
            self.assertLessEqual(len(report['log_tail']), 24)
            self.assertLess(len(output.getvalue()), 7000)
            self.assertEqual(json.loads((Path(folder) / 'debug.json').read_text()), report)

    def test_missing_environment_blocks_processing(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(runner, 'ROOT', Path(folder)):
            with self.assertRaisesRegex(RuntimeError, 'step 3'):
                runner.process({})

    def test_colab_inline_backend_does_not_leak_into_cli(self):
        backend = 'module://matplotlib_inline.backend_inline'
        with patch.dict(os.environ, {'MPLBACKEND': backend}):
            env = runner.run_env()
            self.assertEqual(env['MPLBACKEND'], 'Agg')
            self.assertEqual(os.environ['MPLBACKEND'], backend)

    def test_process_error_does_not_echo_inline_code(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(runner, 'LOG', Path(folder) / 'process.log'):
                with self.assertRaisesRegex(RuntimeError, 'probe: processo terminato con codice 1') as caught:
                    runner.checked('probe', [__import__('sys').executable, '-c',
                                            'raise RuntimeError("long inline code")'])
                self.assertNotIn('raise RuntimeError(', str(caught.exception))

    def test_live_progress_handles_carriage_returns_and_keeps_log(self):
        code = ('import sys; '
                'sys.stdout.write("[ROOP.FACE-SWAPPER] Progressing...\\n"); '
                'sys.stdout.write("Processing: 100%|##########| 2/2 [00:01<00:00, 2frame/s]\\r"); '
                'sys.stdout.flush()')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'live.log'
            with patch.object(runner, 'LOG', path), contextlib.redirect_stdout(io.StringIO()) as output:
                runner.checked('live', [__import__('sys').executable, '-c', code], live=True)
            self.assertIn('2/2', output.getvalue())
            self.assertIn('Progressing...', output.getvalue())
            self.assertIn('2frame/s', path.read_text())

    def test_processing_uses_local_frames_and_persists_only_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'installed.json').write_text(json.dumps({'install_id': runner.INSTALL_ID}))
            face, video, output = root / 'face.jpg', root / 'video.mp4', root / 'out.mp4'
            face.write_bytes(b'face')
            video.write_bytes(b'video')
            config = {'source': str(face), 'target': str(video), 'output': str(output)}
            def execute(label, args, **kwargs):
                local_video = Path(args[args.index('--target') + 1])
                self.assertNotEqual(local_video, video)
                self.assertEqual(local_video.read_bytes(), b'video')
                Path(args[args.index('--output') + 1]).write_bytes(b'result')
            with patch.object(runner, 'ROOT', root), patch.object(runner, 'prepare'), patch.object(runner, 'checked', execute):
                with contextlib.redirect_stdout(io.StringIO()):
                    runner.process(config)
            self.assertEqual(output.read_bytes(), b'result')
            self.assertEqual(video.read_bytes(), b'video')

    def test_fps_uses_average_instead_of_90000_timing_base(self):
        fixture = 'def detect_fps(target_path: str) -> float:\n    return 90000\n\ndef extract_frames():\n    pass\n'
        patched = runner.patched_utilities(fixture)
        self.assertEqual(patched, runner.patched_utilities(patched))
        namespace = {'subprocess': __import__('subprocess')}
        exec(patched, namespace)
        cases = [({'avg_frame_rate': '30/1', 'r_frame_rate': '90000/1'}, 30),
                 ({'avg_frame_rate': '30000/1001', 'r_frame_rate': '90000/1'}, 30000/1001),
                 ({'avg_frame_rate': '0/0', 'r_frame_rate': '90000/1', 'nb_frames': '450', 'duration': '15'}, 30)]
        for stream, expected in cases:
            with patch('subprocess.check_output', return_value=json.dumps({'streams': [stream]})):
                self.assertAlmostEqual(namespace['detect_fps']('video.mp4'), expected)
        with patch('subprocess.check_output', return_value=json.dumps({'streams': [{'r_frame_rate': '90000/1'}]})):
            with self.assertRaisesRegex(ValueError, 'Frame rate non valido'):
                namespace['detect_fps']('video.mp4')

    def test_launcher_fetches_new_revision_on_every_call(self):
        notebook = json.loads((HERE / 'VideoFake_colab.ipynb').read_text())
        tree = ast.parse(''.join(notebook['cells'][1]['source']))
        tree.body.pop()  # Do not execute the initial diagnose call.
        namespace = {}
        exec(compile(tree, '<bootstrap>', 'exec'), namespace)
        requests = []
        def response(request, timeout):
            url = request.full_url if hasattr(request, 'full_url') else request
            requests.append(url)
            if 'api.github.com' in url:
                revision = 'a' * 40 if len(requests) == 1 else 'b' * 40
                return io.BytesIO(json.dumps({'object': {'sha': revision}}).encode())
            return io.BytesIO(b'def run_step(step, revision, config):\n return {"revision": revision}\n')
        with patch('urllib.request.urlopen', response):
            first = namespace['_vf_step']('setup')
            second = namespace['_vf_step']('setup')
        self.assertEqual(first['revision'], 'a' * 40)
        self.assertEqual(second['revision'], 'b' * 40)
        self.assertIn('/' + 'a' * 40 + '/colab_runner.py', requests[1])
        self.assertIn('/' + 'b' * 40 + '/colab_runner.py', requests[3])


if __name__ == '__main__':
    unittest.main()
