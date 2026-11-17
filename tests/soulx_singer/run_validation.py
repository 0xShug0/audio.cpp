#!/usr/bin/env python3
"""Sequential public-server lifecycle and process-memory validation."""
import argparse
import copy
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/server'))
sys.path.insert(0, str(ROOT / 'tools/audiocpp_cli'))
from server_test_utils import Server
from run_audiocpp_cli_path_tests import ResourceMonitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--task', choices=['svc', 'gen'], required=True)
    parser.add_argument('--backend', choices=['cuda', 'vulkan', 'cpu'], required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--long-request', type=Path)
    parser.add_argument('--alternate-reference', type=Path)
    parser.add_argument('--errors-only', action='store_true')
    parser.add_argument('--log', action='store_true', required=True)
    args = parser.parse_args()
    if bool(args.long_request) != bool(args.alternate_reference):
        parser.error('--long-request and --alternate-reference must be supplied together')
    request = json.loads(args.request.read_text())
    request.setdefault('options', {})
    request['options'].update(seed=2026, guidance_scale=3, auto_pitch_shift=True)
    if args.task == 'gen': request['options'].setdefault('control', 'score')
    models = [{'id': 'singer', 'family': 'soulx_singer', 'path': str(args.model.resolve()),
               'task': args.task, 'mode': 'offline'}]
    records = []
    rejected = []
    with Server((ROOT / 'build/debug/bin/audiocpp_server').resolve(), models,
                args.backend, 0, args.output.resolve(), extra_args=('--log',)) as server:
        monitor = ResourceMonitor(server.proc.pid, 0, 20)
        stop = threading.Event()

        def sample():
            while not stop.wait(.02):
                monitor.sample()

        sampling = threading.Thread(target=sample)
        sampling.start()
        try:
            cases = [('warmup', 17, 3, 2), ('first', 2026, 3, 32), ('other_seed', 1234, 3, 32),
                     ('repeat', 2026, 3, 32), ('cfg_off', 2026, 0, 32), ('cfg_restored', 2026, 3, 32)]
            if args.long_request:
                cases = [(name, 2026, 3, 32) for name in
                         ('first', 'long', 'short_restored', 'other_reference', 'reference_restored')]
            if args.errors_only:
                cases = []
            for name, seed, cfg, steps in cases:
                current = copy.deepcopy(request)
                if name == 'long':
                    current = json.loads(args.long_request.read_text())
                elif name == 'other_reference':
                    current['voice_ref'] = str(args.alternate_reference.resolve())
                current.setdefault('options', {})
                current['options'].update(seed=seed, guidance_scale=cfg, num_steps=steps, auto_pitch_shift=True)
                record, audio = server.run(current, name, model='singer')
                if record['status'] != 200 or not audio:
                    raise RuntimeError(f'{name}: {record}')
                monitor.sample()
                record['resident_vram_mib'] = monitor.process_tree_vram(monitor.process_tree_pids()) / 2**20
                records.append(record)
            if args.long_request and not args.errors_only:
                if not records[0]['sha256'] == records[2]['sha256'] == records[4]['sha256']:
                    raise RuntimeError('Shape or reference restoration changed output')
                if records[0]['sha256'] == records[3]['sha256']:
                    raise RuntimeError('Reference change did not affect output')
            elif not args.errors_only:
                first = records[1]['sha256']
                if not first == records[3]['sha256'] == records[5]['sha256']:
                    raise RuntimeError('Repeated seed or CFG restoration changed output')
                if first == records[2]['sha256'] or first == records[4]['sha256']:
                    raise RuntimeError('Seed or CFG did not affect output')
            invalid = [('num_steps', 0), ('guidance_scale', -1), ('pitch_shift', 25),
                       ('seed', -2), ('unknown_option', True), ('seed', [1, 2])]
            if args.task == 'gen':
                invalid.append(('control', 'typo'))
            for key, value in invalid:
                current = copy.deepcopy(request)
                current['options'][key] = value
                code, result = server.api('/v1/tasks/run', {'model': 'singer', 'request': current})
                rejected.append({'option': key, 'value': value, 'status': code})
                if code != 400:
                    raise RuntimeError(f'Invalid {key} should return HTTP 400: {code} {result}')
        finally:
            stop.set()
            sampling.join()
            monitor.sample()
            metrics = monitor.metrics.to_json()
            monitor.close()
            args.output.joinpath('summary.json').write_text(json.dumps(
                {'family': 'soulx_singer', 'backend': args.backend, 'requests': records,
                 'resources': metrics, 'rejected_requests': rejected}, indent=2))


if __name__ == '__main__':
    main()
