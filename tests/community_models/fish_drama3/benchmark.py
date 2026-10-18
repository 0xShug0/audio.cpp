"""Warm first-audio benchmark using a running native Drama 3 server.

Run on the server machine. Uses only the standard library and generates its
own reference. Saved WAVs, exact JSON bodies and CSV rows permit replay.
"""
import argparse
import base64
import csv
import http.client
import json
import math
import statistics
import time
import urllib.request
import urllib.error
import wave
from pathlib import Path
from urllib.parse import urlparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base-url', default='http://127.0.0.1:8035')
parser.add_argument('--output-directory', type=Path, required=True)
parser.add_argument('--count', type=int, default=100)
args = parser.parse_args()
if args.count <= 0:
    parser.error('--count must be positive')
args.output_directory.mkdir(parents=True, exist_ok=True)

def request(path, body):
    encoded = json.dumps(body).encode('utf-8')
    try:
        with urllib.request.urlopen(urllib.request.Request(args.base_url + path, data=encoded,
                                     headers={'Content-Type': 'application/json'}), timeout=900) as response:
            assert response.status == 200
            return response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode('utf-8', errors='replace')
        (args.output_directory / 'http-error.txt').write_text(detail)
        raise RuntimeError(f'{path}: HTTP {error.code}: {detail}') from error

reference_text = 'Drama Three is running through the native audio cpp server.'
reference_wav = args.output_directory.resolve() / 'reference.wav'
reference_wav.write_bytes(request('/v1/audio/speech', {'model': 'drama-3', 'input': reference_text,
                          'seed': 123, 'max_tokens': 150, 'response_format': 'wav'}))
prepared = json.loads(request('/v1/tasks/run', {'model': 'drama-3', 'request': {
    'text': 'Prepare the reference.', 'voice_ref': str(reference_wav),
    'options': {'reference_text': reference_text, 'encode_reference_only': True}}}))
artifact = next(x for x in prepared['artifacts'] if x['id'] == 'drama3_reference_0')
reference = json.loads(base64.b64decode(artifact['payload']))
clone = {'model': 'drama-3', 'input': 'This voice reference came from the model itself.',
         'seed': 123, 'max_tokens': 200, 'voice_ref': str(reference_wav), 'response_format': 'wav',
         'options': {'reference_text': reference_text}}
from_path = request('/v1/audio/speech', clone)
uploaded = json.loads(json.dumps(clone))
uploaded['voice_ref'] = {'type': 'base64', 'data': base64.b64encode(reference_wav.read_bytes()).decode()}
from_upload = request('/v1/audio/speech', uploaded)
assert from_path == from_upload, 'Uploaded and server-path voice cloning disagree'
(args.output_directory / 'uploaded-reference-clone.wav').write_bytes(from_upload)
(args.output_directory / 'clone-requests.json').write_text(json.dumps([
    {'path': '/v1/audio/speech', 'body': body} for body in (clone, uploaded)], indent=2))
texts = ['Hello there.', 'The careful narrator paused, then spoke clearly.',
         'We can now test a longer request with a different number of tokens.',
         'Please check the window. I think it is still open.',
         'The morning train will arrive in fifteen minutes.']
bodies = [{'model': 'drama-3', 'input': text, 'stream': True, 'response_format': 'pcm', 'stream_format': 'audio',
           'max_tokens': 500, 'seed': 123, 'temperature': 1.0, 'top_p': .9, 'top_k': 30,
           'options': {'references': json.dumps([reference])}} for text in texts]
(args.output_directory / 'requests.json').write_text(json.dumps(bodies, indent=2))
parsed = urlparse(args.base_url)
assert parsed.scheme == 'http', 'This localhost benchmark expects HTTP'
connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=900)
expected = {}

def trial(index):
    case = index % len(bodies)
    encoded = json.dumps(bodies[case]).encode('utf-8')
    started = time.perf_counter()
    connection.request('POST', (parsed.path.rstrip('/') + '/v1/audio/speech'), encoded,
                       {'Content-Type': 'application/json'})
    response = connection.getresponse()
    assert response.status == 200, response.read().decode(errors='replace')
    chunks = []
    first = None
    while True:
        chunk = response.read1(8192)
        if not chunk:
            break
        if first is None:
            first = (time.perf_counter() - started) * 1000
        chunks.append(chunk)
    elapsed = time.perf_counter() - started
    pcm = b''.join(chunks)
    assert first is not None and len(pcm) % 2 == 0
    if case not in expected:
        expected[case] = pcm
        with wave.open(str(args.output_directory / f'case-{case + 1}.wav'), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(44100)
            wav.writeframes(pcm)
    assert expected[case] == pcm, f'Seeded PCM changed in case {case}'
    seconds = len(pcm) / 88200
    return {'case': case, 'ttfa_ms': first, 'total_seconds': elapsed, 'audio_seconds': seconds,
            'rtf': elapsed / seconds, 'pcm_exact': True}

try:
    for index in range(10):
        trial(index)
    rows = [trial(index) for index in range(args.count)]
finally:
    connection.close()
times = sorted(row['ttfa_ms'] for row in rows)
summary = {'count': len(rows), 'warmups_excluded': 10, 'median_ms': statistics.median(times),
           'p95_ms': times[math.ceil(.95 * len(times)) - 1],
           'median_rtf': statistics.median(row['rtf'] for row in rows),
           'all_repeated_pcm_exact': True, 'streams_fully_drained': True,
           'base64_upload_and_server_path_cloning_wav_exact': True,
           'cold_load_and_reference_preparation_excluded': True,
           'p95_method': 'nearest rank', 'timing': 'HTTP request write to first nonempty PCM chunk'}
(args.output_directory / 'summary.json').write_text(json.dumps(summary, indent=2))
with (args.output_directory / 'measured.csv').open('w', newline='') as output:
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(summary, indent=2))
