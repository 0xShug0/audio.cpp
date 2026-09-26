"""Compare upstream/updated single-slot inference on the same backend.

Also checks the updated pool's queue, unload/reload and unsupported-slot recovery.
Requires the model files and reference audio; uses Python's standard library.
"""

import argparse
import base64
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request


class Server:
    def __init__(self, exe, models, backend, device, out):
        self.exe, self.models, self.backend, self.device, self.out = exe, models, backend, device, out

    def __enter__(self):
        self.out.mkdir(parents=True, exist_ok=True)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        self.url = f'http://127.0.0.1:{port}'
        config = {'host': '127.0.0.1', 'port': port, 'backend': self.backend,
                  'threads': 8, 'device': self.device, 'lazy_load': True, 'max_loaded_models': 1,
                  'models': self.models}
        path = self.out / 'config.json'
        path.write_text(json.dumps(config, indent=2), encoding='utf-8')
        self.log = (self.out / 'server.log').open('w', encoding='utf-8')
        try:
            self.proc = subprocess.Popen(
                [str(self.exe), '--config', str(path), '--no-ui'],
                cwd=self.exe.parent, stdout=self.log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        except BaseException:
            self.log.close()
            raise
        try:
            deadline = time.monotonic() + 120
            while True:
                if self.proc.poll() is not None:
                    raise RuntimeError(f'server exited; see {self.out / "server.log"}')
                try:
                    if self.api('/health')[0] == 200:
                        return self
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(.05)
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if self.proc.poll() is None:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=30)
        finally:
            self.log.close()

    def api(self, path, body=None):
        request = urllib.request.Request(self.url + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def status(self, model='higgs'):
        return next(m for m in self.api('/v1/models')[1]['data'] if m['id'] == model)

    def run(self, request, name, model='higgs', timeout=None):
        body = {'model': model, 'request': request}
        if timeout is not None:
            body['busy_timeout_ms'] = timeout
        started = time.perf_counter()
        code, result = self.api('/v1/tasks/run', body)
        elapsed = time.perf_counter() - started
        audios = {}
        if 'audio' in result:
            audios['audio'] = base64.b64decode(result['audio'])
        for audio in result.get('named_audio_outputs', []):
            audios[audio['id']] = base64.b64decode(audio['audio'])
        for key, audio in audios.items():
            (self.out / f'{name}-{key}.wav').write_bytes(audio)
        record = {'name': name, 'status': code, 'http_s': elapsed, 'timing': result.get('timing'),
                  'sha256': {k: hashlib.sha256(v).hexdigest() for k, v in audios.items()}}
        if code != 200:
            record['error'] = result
        (self.out / f'{name}.json').write_text(json.dumps(record, indent=2))
        return record, audios


def wait_for(predicate):
    deadline = time.monotonic() + 90
    while not predicate():
        if time.monotonic() > deadline:
            raise RuntimeError('status condition timed out')
        time.sleep(.02)


def same(actual, expected):
    assert actual[0]['status'] == expected[0]['status'], (actual[0], expected[0])
    assert actual[1] == expected[1], (actual[0], expected[0])
    if actual[0]['status'] != 200:
        assert actual[0]['error'] == expected[0]['error'], (actual[0], expected[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['before-server', 'after-server', 'model', 'spec', 'reference', 'output-dir']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--backend', choices=['vulkan', 'cpu'], default='vulkan',
                        help='backends currently advertising one Higgs execution slot')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--separator-model', type=Path)
    parser.add_argument('--separator-spec', type=Path)
    parser.add_argument('--input-audio', type=Path)
    args = parser.parse_args()
    if any([args.separator_model, args.separator_spec, args.input_audio]) and not all([
            args.separator_model, args.separator_spec, args.input_audio]):
        parser.error('separator model, spec and input audio must be supplied together')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    english = ('The model checkpoint contains the diffusion transformer and layer-fusion weights. '
               'The MLLM encoder and VAE are loaded from separate files at runtime, so missing '
               'text_encoder.* keys during checkpoint loading are expected.')
    polish = ('Ptaszki ćwierkają, że Apple chce wrócić na rynek serwerów dostarczając chłonnemu '
              'rynkowi AI swoje własne rozwiązania bazujące na zmodyfikowanych układach z serii M.')
    request = {'text': polish, 'voice_ref': str(args.reference), 'seed': 1234, 'max_tokens': 1024,
        'reference_text': "Some call me nature. Others call me Mother Nature. I've been here for over 4.5 billion years. 22,500 times longer than you."}
    cases = [('polish_cold', request), ('polish_warm', request),
        ('english_same_reference', {**request, 'text': english, 'seed': 42}),
        ('small_cache', {**request, 'text': 'Hello, this is a memory allocation test.', 'seed': 7, 'max_tokens': 128}),
        ('english_after_resize', {**request, 'text': english, 'seed': 123}),
        ('unconditioned_cold', {'text': english, 'seed': 42, 'max_tokens': 256}),
        ('unconditioned_repeat', {'text': english, 'seed': 42, 'max_tokens': 256}),
        ('reference_after_unconditioned', {**request, 'seed': 4321}),
        ('long_generation', {**request, 'text': english + ' ' + english, 'seed': 42, 'max_tokens': 1536}),
        ('polish_after_long', request)]
    higgs = {'id': 'higgs', 'family': 'higgs_audio_tts', 'task': 'tts', 'mode': 'offline',
             'path': str(args.model), 'model_spec_override': str(args.spec)}
    models = [higgs]
    if args.separator_model:
        models.append({'id': 'roformer', 'family': 'bs_roformer', 'task': 'sep', 'mode': 'offline',
                       'path': str(args.separator_model), 'model_spec_override': str(args.separator_spec)})
    checks, records, expected, expected_queue = [], {}, {}, []
    def passed(message):
        checks.append(message)
        print('PASS:', message, flush=True)

    with Server(args.before_server, models, args.backend, args.device, args.output_dir / 'before') as server:
        records['before'] = []
        for name, body in cases:
            value = server.run(body, name)
            if value[0]['status'] != 200:
                assert name.startswith('unconditioned_') and 'reached max_tokens' in json.dumps(value[0]['error']), value[0]
            expected[name] = value
            records['before'].append(value[0])
            print('before', name, value[0]['status'], round(value[0]['http_s'], 3), flush=True)
        for i in range(4):
            expected_queue.append(server.run(request, f'queue-{i}'))
        if args.separator_model:
            expected['roformer'] = server.run({'audio': str(args.input_audio)}, 'roformer', 'roformer')
            assert expected['roformer'][0]['status'] == 200, expected['roformer'][0]
    passed('unmodified same-backend references generated')

    # This harness specifically tests backends that currently advertise one slot.
    updated_models = [{**m, 'slots': 1} for m in models] + [{**higgs, 'id': 'unsupported', 'slots': 2}]
    with Server(args.after_server, updated_models, args.backend, args.device, args.output_dir / 'after') as server:
        records['after'] = []
        for name, body in cases:
            value = server.run(body, name)
            same(value, expected[name])
            records['after'].append(value[0])
            assert server.status()['max_parallel_slots'] == 1 and server.status()['active_slots'] == 0
            print('after', name, value[0]['status'], round(value[0]['http_s'], 3), 'exact', flush=True)
        passed('all ten inference/error cases match upstream on the same backend')
        with cf.ThreadPoolExecutor(max_workers=5) as pool:
            barrier = threading.Barrier(5)
            def run_queued(i):
                barrier.wait(timeout=10)
                return server.run(request, f'queue-{i}')
            jobs = [pool.submit(run_queued, i) for i in range(4)]
            barrier.wait(timeout=10)
            wait_for(lambda: server.status()['queued_requests'] == 3)
            assert server.status()['active_slots'] == 1
            assert server.run(request, 'timeout', timeout=10)[0]['status'] == 503
            results = [job.result() for job in jobs]
            # Identical requests may lease in a different order; compare the
            # full byte sequences as a multiset, preserving cache-state effects.
            assert sorted(v[1]['audio'] for v in results) == sorted(v[1]['audio'] for v in expected_queue)
            assert server.status()['active_slots'] == 0 and server.status()['queued_requests'] == 0
            records['queue'] = [v[0] for v in results]
            passed('four requests serialize safely through one slot; queue, timeout and WAVs match')
            job = pool.submit(server.run, request, 'before_unload')
            wait_for(lambda: server.status()['active_slots'] == 1)
            unload = pool.submit(server.api, '/v1/tasks/unload_all_models', {})
            time.sleep(.05)
            assert not unload.done(), 'unload skipped active inference'
            same(job.result(), expected_queue[-1])
            assert unload.result()[0] == 200
            assert not server.status()['loaded'] and server.status()['max_parallel_slots'] is None
            same(server.run(request, 'reload'), expected['polish_cold'])
            passed('unload drains inference; reload restores identical cold output')
        if args.separator_model:
            value = server.run({'audio': str(args.input_audio)}, 'roformer', 'roformer')
            same(value, expected['roformer'])
            assert not server.status()['loaded'] and server.status('roformer')['max_parallel_slots'] == 1
            records['roformer'] = value[0]
            passed('idle eviction and legacy BS-RoFormer output unchanged')
        failure = server.run(request, 'unsupported', 'unsupported')[0]
        assert failure['status'] >= 400 and 'capacity=1' in json.dumps(failure['error']), failure
        assert not server.status('unsupported')['loaded'] and server.status('unsupported')['active_slots'] == 0
        same(server.run(request, 'after_failure'), expected['polish_cold'])
        assert server.run({**request, 'max_tokens': -1}, 'invalid')[0]['status'] >= 400
        assert server.status()['active_slots'] == 0
        passed('unsupported multiple slots reject cleanly; load and inference errors release leases')
        for endpoint, body in [('/v1/tasks/unload_models', {'model_ids': ['higgs']}), ('/v1/tasks/unload_all_models', {})]:
            assert server.api('/v1/tasks/unload_all_models', {})[0] == 200
            with cf.ThreadPoolExecutor(max_workers=2) as pool:
                job = pool.submit(server.run, request, 'lazy_' + endpoint.rsplit('/', 1)[-1])
                wait_for(lambda: server.status()['active_slots'] == 1 and not server.status()['loaded'])
                unload = pool.submit(server.api, endpoint, body)
                time.sleep(.05)
                assert not unload.done(), 'unload skipped first load'
                same(job.result(), expected['polish_cold'])
                assert 'higgs' in unload.result()[1]['unloaded']
                assert not server.status()['loaded'] and server.status()['max_parallel_slots'] is None
        passed('targeted/all-model unload wait for lazy first load and inference')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'results.json').write_text(json.dumps({'backend': args.backend, 'device': args.device,
        'passed': checks, 'records': records}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
