"""Compare slot-scheduler busy rejection before/after a fix using real Canary ASR.

Requires two server executables, a Canary GGUF/spec and a PCM WAV fixture.
Uses the standard library. Reports HTTP rejection latency separately from
inference time and verifies outputs, unload/reload and queue draining.
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import statistics
import threading
import time
import wave

from server_test_utils import Server


def wait_active(server, count):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        state = server.status('canary')
        if state['active_slots'] == count:
            return state
        time.sleep(.001)
    raise AssertionError(f'Expected {count} active slots: {state}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before-server', 'after-server', 'model', 'spec', 'audio', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--parallel-jobs', action='store_true', help='enable the parallel runtime in both compared builds')
    parser.add_argument('--backend', choices=('cuda', 'vulkan'), required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--slots', type=int, nargs='+', default=[1], choices=(1, 2, 3, 4),
                        help='use 1 for the standalone framework; larger counts require model support')
    args = parser.parse_args()
    if not args.parallel_jobs and args.slots != [1]:
        parser.error("multiple slots require --parallel-jobs")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # A longer fixture keeps the warm inference active throughout the rejection
    # probe even on fast GPUs. Preserve all PCM samples; do not resample.
    audio = args.output_dir / 'speech-repeated.wav'
    with wave.open(str(args.audio), 'rb') as source:
        params, frames = source.getparams(), source.readframes(source.getnframes())
    with wave.open(str(audio), 'wb') as dest:
        dest.setparams(params)
        dest.writeframes(frames * 4)
    payload = {'audio': str(audio), 'language': 'en', 'max_tokens': 512}
    model = {'id': 'canary', 'family': 'canary_asr', 'task': 'asr', 'mode': 'offline',
             'path': str(args.model), 'model_spec_override': str(args.spec)}
    records = []
    result_path = args.output_dir / 'results.json'
    metadata = {'backend': args.backend, 'device': args.device, 'timeout_ms': 15,
                'input_duration_s': params.nframes * 4 / params.framerate,
                'model': str(args.model), 'model_spec': str(args.spec),
                'source_audio': str(args.audio),
                'executables': {label: {'path': str(exe), 'sha256': hashlib.sha256(exe.read_bytes()).hexdigest()}
                                for label, exe in [('before', args.before_server), ('after', args.after_server)]}}

    def save():
        result_path.write_text(json.dumps({'metadata': metadata, 'records': records}, indent=2), encoding='utf-8')

    def timed(server, body):
        started = time.perf_counter()
        status, response = server.api('/v1/tasks/run', body)
        elapsed = (time.perf_counter() - started) * 1000
        timing = response.pop('timing', None)
        return {'status': status, 'response': response, 'timing': timing, 'http_ms': elapsed}

    references = {}
    for label, exe in [('before', args.before_server), ('after', args.after_server)]:
        for slots in args.slots:
            folder = args.output_dir / f'{label}-{slots}'
            with Server(exe, [{**model, 'slots': slots}] if args.parallel_jobs else [model],
                        args.backend, args.device, folder, parallel_jobs=args.parallel_jobs) as server:
                body = {'model': 'canary', 'request': payload}
                cold = timed(server, body)
                warm = timed(server, body)
                assert cold['status'] == warm['status'] == 200, (cold, warm)
                assert cold['response'] == warm['response'], ('Cold/warm Canary reference differs', cold, warm)
                if label == 'before':
                    references[slots] = warm['response']
                assert warm['response'] == references[slots]
                records.append({'build': label, 'slots': slots, 'test': 'reference', 'cold': cold, 'warm': warm})
                save()
                # Full pools, plus a partially occupied pool blocked by unload.
                scenarios = [(False, slots), (True, slots)]
                if slots > 1:
                    scenarios.append((True, 1))
                with cf.ThreadPoolExecutor(max_workers=slots + 2) as pool:
                    for management_waiting, occupied in scenarios:
                        for repeat in range(args.repeats):
                            barrier = threading.Barrier(occupied + 1)

                            def running():
                                barrier.wait(timeout=30)
                                return timed(server, body)

                            jobs = [pool.submit(running) for _ in range(occupied)]
                            barrier.wait(timeout=30)
                            wait_active(server, occupied)
                            # Give every occupied lease time to exceed 15 ms.
                            time.sleep(.025)
                            unload = None
                            if management_waiting:
                                unload = pool.submit(server.api, '/v1/tasks/unload_all_models', {})
                                time.sleep(.01)
                                assert not unload.done(), 'Unload finished before rejection probe'
                            assert all(not job.done() for job in jobs), 'Fixture inference too short'
                            rejection = timed(server, {**body, 'busy_timeout_ms': 15})
                            assert rejection['status'] == 503, rejection
                            error = json.dumps(rejection['response'])
                            expected_error = ('timed out waiting for a slot' if label == 'before' and management_waiting
                                              else 'slots exceeded busy_timeout_ms')
                            assert expected_error in error, (label, management_waiting, rejection)
                            outputs = [job.result() for job in jobs]
                            assert all(x['status'] == 200 and x['response'] == references[slots] for x in outputs), outputs
                            if unload:
                                assert unload.result()[0] == 200
                                assert not server.status('canary')['loaded']
                                reload = timed(server, body)
                                assert reload['status'] == 200 and reload['response'] == references[slots], reload
                                # Keep each rejection probe on a warm server.
                                assert timed(server, body)['response'] == references[slots]
                            state = server.status('canary')
                            assert state['active_slots'] == state['queued_requests'] == 0, state
                            record = {'build': label, 'slots': slots, 'occupied': occupied,
                                      'management_waiting': management_waiting, 'repeat': repeat,
                                      'test': 'busy_rejection', 'rejection': rejection,
                                      'running_outputs': outputs, 'unload_reload_exact': bool(unload)}
                            records.append(record)
                            save()
                        values = [r['rejection']['http_ms'] for r in records if r.get('test') == 'busy_rejection'
                                  and r['build'] == label and r['slots'] == slots and r['occupied'] == occupied
                                  and r['management_waiting'] == management_waiting]
                        print(f'{args.backend} {label} slots={slots} active={occupied} unload={management_waiting}: '
                              f'median rejection={statistics.median(values):.3f} ms, exact outputs', flush=True)
    print('PASS real HTTP overdue rejection, concurrent parity and unload/reload', flush=True)


if __name__ == '__main__':
    main()
