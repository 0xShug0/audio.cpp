"""Check a running Drama 3 server using generated audio and standard task APIs.

Uses the Python standard library. Start the server described in the model guide,
then pass --base-url and an output directory. No reference dataset is needed.
"""

import argparse
import base64
import io
import json
import math
import struct
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--output-directory', type=Path, required=True)
parser.add_argument('--base-url', default='http://127.0.0.1:8035')
parser.add_argument('--timeout', type=float, default=600, help='Per-request timeout, including a cold model load')
arguments = parser.parse_args()
arguments.output_directory.mkdir(parents=True, exist_ok=True)
report = {'base_url': arguments.base_url, 'family': 'fish_drama3', 'checks': {}}


def request(path, body=None):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode('utf-8')
    return urllib.request.urlopen(urllib.request.Request(
        arguments.base_url + path, data=data,
        headers={'Content-Type': 'application/json'}), timeout=arguments.timeout)


def json_request(path, body=None):
    with request(path, body) as response:
        assert response.status == 200
        return json.load(response)


def wav_info(payload):
    with wave.open(io.BytesIO(payload), 'rb') as audio:
        rate = audio.getframerate()
        channels = audio.getnchannels()
        width = audio.getsampwidth()
        count = audio.getnframes()
        pcm = audio.readframes(count)
    assert rate == 44100 and channels == 1 and width == 2 and count > 0
    samples = struct.unpack('<' + 'h' * (len(pcm) // 2), pcm)
    rms = math.sqrt(sum((sample / 32768) ** 2 for sample in samples) / len(samples))
    assert rms > 0.0001
    return {'sample_rate': rate, 'channels': channels, 'seconds': count / rate, 'rms': rms}, pcm


def stream(path, body, audio_type, done_type):
    started = time.perf_counter()
    first_audio = None
    events = []
    complete = False
    with request(path, body) as response:
        assert response.status == 200
        assert 'text/event-stream' in response.headers.get('Content-Type', '')
        for line in response:
            text = line.decode('utf-8').strip()
            if not text.startswith('data: '):
                continue
            data = text[6:]
            if data == '[DONE]':
                complete = True
                break
            event = json.loads(data)
            assert event.get('type') != 'error', event
            if event.get('type') == audio_type and first_audio is None:
                if audio_type == 'speech.audio.delta' or event.get('event', {}).get('audio'):
                    first_audio = time.perf_counter() - started
            events.append(event)
    assert complete and first_audio is not None
    assert any(event.get('type') == done_type for event in events)
    return events, {'first_audio_ms': first_audio * 1000,
                    'wall_seconds': time.perf_counter() - started,
                    'event_count': len(events), 'completed': complete}


try:
    report['checks']['health'] = json_request('/health')
    report['checks']['models_before'] = json_request('/v1/models?include_session_options=true')

    speech = {'model': 'drama-3', 'input': 'Drama Three is running through the native audio cpp server.',
              'max_tokens': 150, 'seed': 123, 'response_format': 'wav'}
    original_speech = dict(speech)
    started = time.perf_counter()
    with request('/v1/audio/speech', speech) as response:
        assert response.status == 200 and 'audio/wav' in response.headers.get('Content-Type', '')
        payload = response.read()
    info, first_pcm = wav_info(payload)
    info['wall_seconds_including_lazy_load'] = time.perf_counter() - started
    (arguments.output_directory / 'drama3-http.wav').write_bytes(payload)
    report['checks']['wav_response'] = info

    speech['response_format'] = 'pcm'
    speech['stream_format'] = 'sse'
    events, timing = stream('/v1/audio/speech', speech, 'speech.audio.delta', 'speech.audio.done')
    pcm = b''.join(base64.b64decode(event['audio']) for event in events
                   if event.get('type') == 'speech.audio.delta')
    assert pcm == first_pcm, 'WAV and SSE PCM disagree for identical seeded requests.'
    timing['matches_wav_samples'] = True
    timing['audio_bytes'] = len(pcm)
    report['checks']['streaming_speech'] = timing

    take_body = {'model': 'drama-3', 'stream_format': 'sse',
                 'request': {'text': 'The report is due on Wednesday.', 'max_tokens': 150, 'seed': 123}}
    events, timing = stream('/v1/tasks/stream', take_body, 'task.stream.event', 'task.stream.done')
    result = next(event['result'] for event in events if event.get('type') == 'task.stream.done')
    artifacts = result['artifacts']
    artifact = next(item for item in artifacts if item['id'] == 'drama3_take')
    assert artifact['kind'] == 'acoustic_tokens' and artifact['meta']['family'] == 'fish_drama3'
    take = json.loads(base64.b64decode(artifact['payload']))
    assert len(take['vq_codes']) == 10 and len(take['vq_codes'][0]) > 0
    (arguments.output_directory / 'drama3-server-previous-take.json').write_text(
        json.dumps(take), encoding='utf-8')
    timing['artifact_id'] = artifact['id']
    timing['codebook_count'] = len(take['vq_codes'])
    timing['acoustic_frames'] = len(take['vq_codes'][0])
    report['checks']['streaming_take_artifact'] = timing

    edit_body = {'model': 'drama-3',
                 'request': {'text': 'The report is due on Thursday.',
                             'max_tokens': 150, 'seed': 123, 'artifacts': artifacts}}
    started = time.perf_counter()
    edit = json_request('/v1/tasks/run', edit_body)
    edit_audio = base64.b64decode(edit['audio'])
    info, _ = wav_info(edit_audio)
    info['wall_seconds'] = time.perf_counter() - started
    assert any(item['id'] == 'drama3_take' for item in edit['artifacts'])
    (arguments.output_directory / 'drama3-http-edit.wav').write_bytes(edit_audio)
    report['checks']['continuation_artifact_input'] = info

    # Reuse our own generated sample as the voice reference, with its known text.
    reference_path = (arguments.output_directory / 'drama3-http.wav').resolve()
    reference_request = {'text': 'Prepare the reference.', 'voice_ref': str(reference_path),
                         'options': {'reference_text': speech['input'], 'encode_reference_only': True}}
    encoded = json_request('/v1/tasks/run', {'model': 'drama-3', 'request': reference_request})
    assert not encoded.get('audio')
    reference_artifact = next(item for item in encoded['artifacts'] if item['id'] == 'drama3_reference_0')
    reference = json.loads(base64.b64decode(reference_artifact['payload']))
    assert len(reference['vq_codes']) == 10 and len(reference['vq_codes'][0]) > 0
    report['checks']['reference_encoding'] = {'codebooks': 10, 'frames': len(reference['vq_codes'][0])}

    cloned_request = {'text': 'This voice reference came from the model itself.', 'voice_ref': str(reference_path),
                      'seed': 123, 'max_tokens': 200, 'options': {'reference_text': speech['input']}}
    cloned = json_request('/v1/tasks/run', {'model': 'drama-3', 'request': cloned_request})
    cloned_audio = base64.b64decode(cloned['audio'])
    info, _ = wav_info(cloned_audio)
    assert not cloned['custom_schema_output']['data']['reached_limit']
    (arguments.output_directory / 'drama3-http-clone.wav').write_bytes(cloned_audio)
    preencoded_request = {'text': cloned_request['text'], 'seed': 123, 'max_tokens': 200,
                          'options': {'references': json.dumps([{'text': speech['input'], 'vq_codes': reference['vq_codes']}])}}
    preencoded = json_request('/v1/tasks/run', {'model': 'drama-3', 'request': preencoded_request})
    assert cloned_audio == base64.b64decode(preencoded['audio']), 'Inline reference and prepared-code cloning disagree.'
    info['prepared_reference_matches_inline_audio'] = True
    report['checks']['voice_cloning'] = info

    long_text = (
        'The garden was quiet after the rain. Small drops shone on the leaves beside the window. '
        'A warm breeze moved through the open door. I put the book on the table and listened to the birds outside. '
        'There was nothing else I needed to do. For a moment, I stayed still and enjoyed the peaceful morning.')
    long_form = json_request('/v1/tasks/run', {'model': 'drama-3', 'request': {
        'text': long_text, 'seed': 123, 'max_tokens': 150, 'options': {'text_chunk_size': 80}}})
    long_audio = base64.b64decode(long_form['audio'])
    info, _ = wav_info(long_audio)
    generation = long_form['custom_schema_output']['data']
    assert not generation['reached_limit'], 'A long-form chunk reached its frame limit.'
    assert generation['frames'] > 150, 'Long-form request did not exceed the per-chunk budget.'
    (arguments.output_directory / 'drama3-http-long-form.wav').write_bytes(long_audio)
    info['frames'] = generation['frames']
    info['per_chunk_max_tokens'] = 150
    report['checks']['long_form_chunking'] = info

    with request('/v1/audio/speech', original_speech) as response:
        assert response.read() == payload, 'A repeated seeded request changed after cloning and continuation.'
    report['checks']['repeated_request_after_mixed_tasks'] = {'wav_exact': True}
    report['checks']['models_after'] = json_request('/v1/models?include_session_options=true')
    report['passed'] = True
finally:
    (arguments.output_directory / 'drama3-server-check.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')

print(json.dumps(report, indent=2))
