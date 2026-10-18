#!/usr/bin/env python3
"""Exercise native multilingual presets, Unicode chunking and voice cloning.

Requires NumPy and SoundFile for validation only. Uses local
assets and a temporary localhost server; never downloads weights or runs ASR.
"""
import argparse
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--reference-model', type=Path, required=True)
    parser.add_argument('--voices', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    voices = json.loads(args.voices.read_text(encoding='utf-8'))
    raw = json.loads((args.reference_model / 'voices/voices.json').read_text(encoding='utf-8'))
    cases = json.loads(Path(__file__).with_name('multilingual_cases.json').read_text(encoding='utf-8'))
    env = {key.upper(): value for key, value in os.environ.items()}
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    server_config = output / 'server.json'
    server_config.write_text(json.dumps({
        'host': '127.0.0.1', 'port': port, 'backend': args.backend, 'threads': args.threads,
        'lazy_load': True, 'models': [{'id': 'kitten-tts2', 'family': 'kitten_tts2',
            'path': str(args.model.resolve()), 'task': 'tts', 'mode': 'offline',
            'voice_presets': {name: {'voice_id': name} for name in voices},
            'default_voice_preset': {'voice_id': 'Bruno'}}],
    }), encoding='utf-8')
    base = f'http://127.0.0.1:{port}'

    def get(path):
        with urllib.request.urlopen(base + path, timeout=5) as response:
            return json.load(response)

    def synthesize(name, voice, text, extra=None):
        payload = {'model': 'kitten-tts2', 'input': text, 'voice': voice, 'seed': 1234}
        payload.update(extra or {})
        started = time.perf_counter()
        request = urllib.request.Request(base + '/v1/audio/speech',
            json.dumps(payload, ensure_ascii=False).encode('utf-8'), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=600) as response:
            wav = response.read()
        elapsed = time.perf_counter() - started
        samples, rate = sf.read(io.BytesIO(wav), dtype='float32')
        assert rate == 24000 and samples.ndim == 1 and samples.size >= 2400, name
        rms = float(np.sqrt(np.mean(samples**2)))
        assert np.isfinite(samples).all() and rms > 1e-4, name
        (output / (name + '.wav')).write_bytes(wav)
        result = {'voice': voice, 'text': text, 'audio_seconds': samples.size / rate,
                  'seconds': elapsed, 'rms': rms}
        print(json.dumps({'case': name, **result}, ensure_ascii=True), flush=True)
        return result

    report = {'backend': args.backend, 'cases': {}}
    with (output / 'server.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen([str(args.server.resolve()), '--config', str(server_config)],
            env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            for _ in range(150):
                if process.poll() is not None:
                    raise RuntimeError('server exited; inspect server.log')
                try:
                    get('/health')
                    break
                except OSError:
                    time.sleep(.2)
            else:
                raise RuntimeError('server did not become ready')
            advertised = get('/v1/audio/voices?model=kitten-tts2')['voices']
            assert set(advertised) == set(voices), 'server voice registration mismatch'
            report['registered_voices'] = len(advertised)
            for case in cases:
                name = case['voice']
                assert voices[name]['transcript'] == raw[name]['transcript'], name
                with np.load(args.reference_model / 'voices' / raw[name]['artifacts'], allow_pickle=False) as artifacts:
                    assert voices[name]['reference_tokens'] == artifacts['reference_tokens'].tolist(), name
                report['cases'][name] = synthesize(name, name, case['text'])
            # Three complete Unicode sentences, each under the 32-codepoint chunk limit.
            sentence = next(case['text'] for case in cases if case['voice'] == 'Chinese').split('。', 1)[1]
            report['cases']['Chinese_chunks'] = synthesize('Chinese_chunks', 'Chinese', sentence * 3,
                {'options': {'text_chunk_size': 32}})
            assert report['cases']['Chinese_chunks']['audio_seconds'] > report['cases']['Chinese']['audio_seconds'] * 2
            for name in ('German', 'Chinese'):
                text = next(case['text'] for case in cases if case['voice'] == name)
                report['cases'][name + '_clone'] = synthesize(name + '_clone', name, text, {
                    'voice_ref': {'type': 'path', 'path': str((args.reference_model / 'voices' / raw[name]['reference']).resolve())},
                    'reference_text': raw[name]['transcript'],
                })
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('PASS: language presets, prepared reference tokens, Unicode chunking and multilingual cloning.', flush=True)


if __name__ == '__main__':
    main()
