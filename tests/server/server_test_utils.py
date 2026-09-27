"""Standard-library HTTP test server lifecycle helper."""

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
