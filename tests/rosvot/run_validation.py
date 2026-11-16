#!/usr/bin/env python3
"""Sequential public-server lifecycle and process-memory checks for ROSVOT."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tests/server'))
sys.path.insert(0, str(ROOT/'tools/audiocpp_cli'))
from server_test_utils import Server
from run_audiocpp_cli_path_tests import ResourceMonitor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--audio', type=Path, required=True)
    p.add_argument('--other-audio', type=Path, required=True)
    p.add_argument('--backend', choices=['cuda', 'vulkan', 'cpu'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--log', action='store_true', required=True)
    p.add_argument('--errors-only', action='store_true')
    p.add_argument('--benchmark-only', action='store_true')
    p.add_argument('--server-exe', type=Path, default=ROOT/'build/debug/bin/audiocpp_server')
    p.add_argument('--measured-requests', type=int, default=2)
    p.add_argument('--warmup-requests', type=int, default=2)
    a = p.parse_args()
    if a.warmup_requests < 1: p.error('--warmup-requests must be positive')
    if a.measured_requests < 2: p.error('--measured-requests must be at least two')
    model = {'id':'rosvot', 'family':'rosvot', 'path':str(a.model.resolve()), 'task':'midi', 'mode':'offline'}
    records = []
    with Server(a.server_exe.resolve(), [model], a.backend, 0, a.output.resolve(), extra_args=('--log',)) as server:
        monitor = ResourceMonitor(server.proc.pid, 0, 10)
        stop = threading.Event()
        def sample():
            while not stop.wait(.01): monitor.sample()
        worker = threading.Thread(target=sample); worker.start()
        try:
            first = None
            for name in (['first'] if a.errors_only else [f'warmup{i}' for i in range(a.warmup_requests)] + ['first', 'second'] + [f'measured{i}' for i in range(2, a.measured_requests)] if a.benchmark_only else ['warmup', 'first', 'second', 'other', 'restored', 'midi']):
                audio = a.other_audio if name == 'other' else a.audio
                request = {'audio':str(audio.resolve()),'options':{'output_format':'midi' if name == 'midi' else 'json'}}
                start = time.perf_counter()
                code, result = server.api('/v1/tasks/run', {'model':'rosvot', 'request':request})
                elapsed = (time.perf_counter()-start)*1000
                (a.output/f'{name}.json').write_text(json.dumps(result, indent=2))
                if code != 200: raise RuntimeError(f'{name}: HTTP {code}: {result}')
                text = result.get('text','')
                if name == 'first': first = text
                if (name in ['second', 'restored'] or name.startswith('measured')) and text != first: raise RuntimeError(f'{name}: repeated output changed')
                monitor.sample()
                records.append({'name':name,'http_ms':elapsed,'timing':result.get('timing'),
                    'resident_vram_mib':monitor.process_tree_vram(monitor.process_tree_pids())/2**20})
            for option, value in ([] if a.benchmark_only else [('note_boundary_threshold',-0.1),('word_boundary_threshold',1.1),('typo',True),('output_format','bad')]):
                code, result = server.api('/v1/tasks/run', {'model':'rosvot','request':{'audio':str(a.audio.resolve()),'options':{option:value}}})
                if code != 400: raise RuntimeError(f'Invalid {option} returned HTTP {code}: {result}')
            if not a.benchmark_only:
                code, result = server.api('/v1/tasks/run', {'model':'rosvot','request':{'audio':str(a.audio.resolve()),'options':{'output_format':'json'}}})
                if code != 200 or result.get('text') != first: raise RuntimeError('Valid request after rejection changed output')
                (a.output/'recovered.json').write_text(json.dumps(result, indent=2))
        finally:
            stop.set(); worker.join(); monitor.sample(); monitor.close()
            (a.output/'summary.json').write_text(json.dumps({'backend':a.backend,'requests':records,'memory':monitor.metrics.to_json()},indent=2))


if __name__ == '__main__': main()
