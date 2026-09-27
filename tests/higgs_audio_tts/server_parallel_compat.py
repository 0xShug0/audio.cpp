"""Check mixed concurrent Higgs requests against independent single-slot references.

Uses real model inference, fixed seeds, default reference caching, and exact WAV
comparisons with the same request history per session. Cache growth can change
whether upstream reuses reference KV state, so a generic "warm" baseline is not
sufficient when mixing prompt lengths. Run alongside server_slots_bench.py for
timing and VRAM measurements.
"""

import argparse
import concurrent.futures as cf
import json
from pathlib import Path
import threading
import time

from server_backend_compat import Server, same, wait_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before-server', 'after-server', 'model', 'spec', 'request-json', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--backend', choices=['vulkan', 'cuda'], default='vulkan')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--slots', type=int, default=4)
    args = parser.parse_args()
    if not 2 <= args.slots <= 4:
        parser.error('this mixed-request fixture supports two through four slots')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    request = json.loads(args.request_json.read_text(encoding='utf-8-sig'))
    assert 'seed' in request and 'voice_ref' in request
    english = ('The model checkpoint contains the diffusion transformer and layer-fusion weights. '
               'The MLLM encoder and VAE are loaded from separate files at runtime, so missing '
               'text_encoder.* keys during checkpoint loading are expected.')
    requests = [request, {**request, 'text': english, 'seed': 42},
                {**request, 'text': 'Hello. This request runs alongside other voices.', 'seed': 7},
                {**request, 'text': english + ' ' + english, 'seed': 123, 'max_tokens': 1536}][:args.slots]
    model = {'id': 'higgs', 'family': 'higgs_audio_tts', 'task': 'tts', 'mode': 'offline',
             'path': str(args.model), 'model_spec_override': str(args.spec)}
    offsets = [0, 0, 1, 2, 3]
    sequences = [[requests[(slot + offset) % args.slots] for offset in offsets]
                 for slot in range(args.slots)]
    expected, checks, records = [], [], {}

    def passed(message):
        checks.append(message)
        print('PASS:', message, flush=True)

    with Server(args.before_server, [model], args.backend, args.device, args.output_dir / 'before') as server:
        for i, sequence in enumerate(sequences):
            assert server.api('/v1/tasks/unload_all_models', {})[0] == 200
            values = [server.run(body, f'slot-{i}-step-{step}') for step, body in enumerate(sequence)]
            assert all(value[0]['status'] == 200 for value in values), [v[0] for v in values]
            expected.append(values)
            print('reference slot', i, 'sequence exact inputs generated', flush=True)
    passed('independent single-slot references generated for each complete slot history')

    with Server(args.after_server, [{**model, 'slots': args.slots}], args.backend, args.device,
                args.output_dir / 'after') as server, cf.ThreadPoolExecutor(max_workers=args.slots + 4) as pool:
        def launch(bodies, label, ordered=False):
            if ordered:
                # The scheduler chooses the lowest free slot. Wait for each
                # admission before starting the next; assert nobody finished
                # early, so the histories are known while inference overlaps.
                jobs = []
                for i, body in enumerate(bodies):
                    jobs.append(pool.submit(server.run, body, f'{label}-{i}'))
                    wait_for(lambda: server.status()['active_slots'] == i + 1)
                    assert not any(job.done() for job in jobs), 'request finished before all slots were admitted'
                return jobs
            barrier = threading.Barrier(len(bodies) + 1)

            def run(i, body):
                barrier.wait(timeout=30)
                return server.run(body, f'{label}-{i}')

            jobs = [pool.submit(run, i, body) for i, body in enumerate(bodies)]
            barrier.wait(timeout=30)
            return jobs

        for repeat in range(len(offsets)):
            started = time.perf_counter()
            jobs = launch([sequence[repeat] for sequence in sequences], f'mixed-{repeat}', ordered=True)
            wait_for(lambda: server.status()['active_slots'] == args.slots and server.status()['loaded'])
            values = [job.result() for job in jobs]
            for i, value in enumerate(values):
                same(value, expected[i][repeat])
            status = server.status()
            assert status['active_slots'] == status['queued_requests'] == 0
            records[f'mixed-{repeat}'] = {'whole_batch_s': time.perf_counter() - started,
                                         'results': [v[0] for v in values]}
            passed(f'{args.slots} mixed requests step {repeat}: exact same-history single-slot WAVs; leases drained')

        # Give every slot the same history before testing nondeterministic queue
        # admission. Separate cold and warm upstream outputs are intentional.
        assert server.api('/v1/tasks/unload_all_models', {})[0] == 200
        for job in launch([request] * args.slots, 'queue-prime'):
            same(job.result(), expected[0][0])

        jobs = launch([request] * (args.slots + 2), 'queued')
        wait_for(lambda: server.status()['queued_requests'] == 2)
        assert server.status()['active_slots'] == args.slots
        assert server.run(request, 'timeout', timeout=10)[0]['status'] == 503
        values = [job.result() for job in jobs]
        for value in values:
            same(value, expected[0][1])
        records['queue'] = [v[0] for v in values]
        passed('overflow requests queue, busy timeout returns 503, and every queued WAV is exact')

        jobs = launch([request] * args.slots, 'unload-active')
        wait_for(lambda: server.status()['active_slots'] == args.slots)
        unload = pool.submit(server.api, '/v1/tasks/unload_all_models', {})
        time.sleep(.05)
        assert not unload.done(), 'unload did not wait for active inference'
        for job in jobs:
            same(job.result(), expected[0][1])
        assert unload.result()[0] == 200
        assert not server.status()['loaded']
        passed('unload waits for every active slot and then releases the shared model')

        jobs = launch(requests, 'reload')
        wait_for(lambda: server.status()['active_slots'] == args.slots and server.status()['loaded'])
        for i, job in enumerate(jobs):
            same(job.result(), expected[i][0])
        # Restore uniform histories before the failure recovery comparison.
        assert server.api('/v1/tasks/unload_all_models', {})[0] == 200
        for job in launch([request] * args.slots, 'error-prime'):
            same(job.result(), expected[0][0])
        assert server.run({**request, 'max_tokens': -1}, 'invalid')[0]['status'] >= 400
        same(server.run(request, 'after-error'), expected[0][1])
        assert server.status()['active_slots'] == server.status()['queued_requests'] == 0
        passed('parallel reload restores cold outputs; invalid request releases its lease; later inference succeeds')

    (args.output_dir / 'results.json').write_text(json.dumps({
        'backend': args.backend, 'device': args.device, 'slots': args.slots,
        'passed': checks, 'records': records}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
