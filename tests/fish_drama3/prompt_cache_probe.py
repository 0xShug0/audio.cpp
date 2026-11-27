"""Measure offline prompt reuse on a running Drama 3 server (Python 3, standard library)."""

import argparse
import base64
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path


FIRST = 'The lamps came on just before dusk.'
TURNS = (
    'A narrow path led to the gate.',
    'The air smelled of rain.',
    'Across the road, a window was open.',
    'A bicycle leaned against the wall.',
    'The last bus turned the corner.',
    'Soon the street was quiet again.',
)
LONG = (
    'A warm breeze moved through the open door. Small drops shone on the leaves beside the window. '
    'I put the book on the table and listened to the birds outside. '
    'For a moment, I stayed still and enjoyed the peaceful morning.'
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8490')
    parser.add_argument('--model', default='drama-3')
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reference-text-file', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--history-directory', type=Path,
                        help='Use WAVs from the baseline run for identical narration conditioning.')
    parser.add_argument('--suite', choices=('full', 'takes', 'prefix'), default='full')
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    history = (args.history_directory or args.output_directory).resolve()
    reference = str(args.reference.resolve())
    transcript = args.reference_text_file.read_text(encoding='utf-8').strip()
    records = {}

    def report(name, record):
        records[name] = record
        (args.output_directory / 'results.json').write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
        print(name, json.dumps(record), flush=True)

    def request(text, seed=42, previous=None, **options):
        payload = {'model': args.model, 'request': {'text': text, 'seed': seed, 'max_tokens': 240,
                   'voice_ref': reference, 'options': {'reference_text': transcript, **options}}}
        if previous is not None:
            payload['request']['options']['previous_takes'] = json.dumps([previous])
        body = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(args.base_url + '/v1/tasks/run', body,
                                     headers={'Content-Type': 'application/json'})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=600) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            raise RuntimeError(error.read().decode('utf-8', 'replace')) from error
        return json.loads(raw), time.perf_counter() - started

    def speak(name, text, seed=42, previous=None, **options):
        result, seconds = request(text, seed, previous, text_chunk_size=options.get('text_chunk_size', 128))
        payload = base64.b64decode(result['audio'])
        with wave.open(io.BytesIO(payload), 'rb') as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (44100, 1, 2)
            pcm = wav.readframes(wav.getnframes())
        assert pcm
        (args.output_directory / (name + '.wav')).write_bytes(payload)
        generation = result['custom_schema_output']['data']
        report(name, {'seconds': seconds, 'audio_seconds': len(pcm) / (44100 * 2), 'text': text, 'seed': seed,
                      'pcm_sha256': hashlib.sha256(pcm).hexdigest(), 'frames': generation['frames'],
                      'reached_limit': generation['reached_limit']})

    speak('first', FIRST)
    if args.suite != 'prefix':
        speak('take-2', FIRST, 43)
        speak('take-3', FIRST)
    if args.suite == 'takes':
        return

    previous_name, previous_text = 'first', FIRST
    for index, text in enumerate(TURNS if args.suite == 'full' else TURNS[:1], 1):
        name = f'turn-{index}'
        speak(name, text, 42 + index, {'text': previous_text, 'audio_path': str(history / (previous_name + '.wav'))})
        previous_name, previous_text = name, text
    if args.suite == 'full':
        speak('first-again', FIRST)
        speak('chunked', LONG, text_chunk_size=60)

    result, seconds = request(FIRST, encode_reference_only=True,
                              references=json.dumps([{'text': FIRST, 'audio_path': str(history / 'first.wav')}]))
    artifacts = [json.loads(base64.b64decode(item['payload']))['vq_codes']
                 for item in result['artifacts'] if item['id'].startswith('drama3_reference_')]
    assert len(artifacts) == 2 and all(len(codes) == 10 for codes in artifacts)
    report('vq-codes', {'seconds': seconds, 'frames': [len(codes[0]) for codes in artifacts],
                        'codes_sha256': [hashlib.sha256(json.dumps(codes).encode('utf-8')).hexdigest() for codes in artifacts]})


if __name__ == '__main__':
    main()
