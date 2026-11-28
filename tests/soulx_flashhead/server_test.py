#!/usr/bin/env python3
"""Exercise real FlashHead video artifacts through offline HTTP and streaming SSE."""
import argparse
import base64
import json
from pathlib import Path
import sys
import time
import urllib.request
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'server'))
from server_test_utils import Server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--portrait', type=Path, required=True)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    models = [{'id': mode, 'family': 'soulx_flashhead', 'path': str(args.model.resolve()),
               'task': 'gen', 'mode': mode} for mode in ('offline', 'streaming')]
    request = {'audio': str(args.audio.resolve()), 'options': {'reference_image_file': str(args.portrait.resolve()),
               'height': 256, 'width': 256, 'seed': 42}}
    with wave.open(str(args.audio), 'rb') as audio:
        assert audio.getframerate() == 16000 and audio.getnchannels() == 1
        expected_chunks = (audio.getnframes() + 15359) // 15360
    streamed = []
    events = []
    with Server(args.server.resolve(), models, 'cuda', 0, out,
                extra_args=('--log', '--log-file', str(out / 'timing.log'))) as server:
        started = time.perf_counter()
        body = {'model': 'streaming', 'stream_format': 'sse', 'include_artifacts': True, 'request': request}
        http = urllib.request.Request(server.url + '/v1/tasks/stream', data=json.dumps(body).encode(),
                                      headers={'Content-Type': 'application/json'})
        completed = False
        done_result = None
        with urllib.request.urlopen(http, timeout=180) as response:
            assert response.status == 200
            assert response.headers['Content-Type'].startswith('text/event-stream')
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                payload = line[6:].strip()
                if payload == b'[DONE]':
                    completed = True
                    break
                event = json.loads(payload)
                assert event.get('type') != 'error', event
                events.append({'type': event['type'], 'http_ms': (time.perf_counter()-started)*1000})
                if event['type'] == 'task.stream.event':
                    assert done_result is None, 'video event arrived after completion'
                    streamed.extend(event['event'].get('artifacts', []))
                elif event['type'] == 'task.stream.done':
                    done_result = event['result']
        assert completed and done_result is not None
        assert done_result['timing']['ttft_ms'] is not None, 'artifact-only output has no TTFA'
        assert not done_result.get('artifacts'), 'video chunks duplicated in the final result'
        assert len(streamed) == expected_chunks, (len(streamed), expected_chunks)
        assert events[0]['type'] == 'task.stream.event' and events[-1]['type'] == 'task.stream.done'
        status, offline = server.api('/v1/tasks/run', {'model': 'offline', 'request': request})
        assert status == 200, offline
        assert len(offline['artifacts']) == len(streamed)
        artifacts_dir = out / 'artifacts'
        artifacts_dir.mkdir(exist_ok=True)
        for index, (actual, expected) in enumerate(zip(streamed, offline['artifacts'])):
            data = base64.b64decode(actual['payload'], validate=True)
            assert data == base64.b64decode(expected['payload'], validate=True), 'HTTP offline/streaming parity failed'
            meta = actual['meta']
            assert int(meta['chunk_index']) == index and int(meta['frames']) == 24
            assert len(data) == int(meta['width'])*int(meta['height'])*int(meta['frames'])*3
            artifact = {'id': actual['id'], 'kind': actual['kind'], 'meta': meta, 'payload_hex': data.hex()}
            (artifacts_dir / f'{actual["id"]}.json').write_text(json.dumps(artifact))
        report = {'events': events, 'chunks': len(streamed), 'ttfa_ms': done_result['timing']['ttft_ms'],
                  'offline_streaming_byte_identical': True}
        (out / 'results.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
