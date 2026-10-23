"""Summarize warmed, unprofiled Qwen3 runs separately from kernel traces."""
import array
import csv
import json
import math
import os
import re
import statistics
import wave
from collections import defaultdict
from pathlib import Path

root = Path('results')
variants = ('off-a1', 'on-b1', 'vulkan-c1', 'vulkan-c2', 'on-b2', 'off-a2')
rows = []
for variant in variants:
    pending = defaultdict(float)
    stages, metrics = {}, defaultdict(dict)
    for line in (root/f'qwen-warm-{variant}.log').read_text().splitlines():
        timing = re.match(r'^\[TIMING[^\]]*\] (qwen3_tts\.[\w.]+) ([\d.eE+-]+)$', line)
        if timing:
            pending[timing[1]] += float(timing[2])
        metric = re.match(r'^metrics\[([^\]]+)\]\.([^=]+)=(.*)$', line)
        if metric:
            name, key, value = metric.groups()
            if name not in stages:
                stages[name] = dict(pending)
                pending.clear()
            metrics[name][key] = value
    for name, metric in metrics.items():
        if name.endswith('warmup'):
            continue
        assert 'qwen3_tts.speech_decoder.graph.compute_ms' in stages[name]
        rows.append(dict(variant=variant, mode=variant.split('-')[0], request=name,
                         rtf=float(metric['wall_ms'])/float(metric['audio_duration_ms']),
                         stages=stages[name]))


def pcm(path):
    with wave.open(str(path), 'rb') as wav:
        assert wav.getsampwidth() == 2
        return (wav.getframerate(), wav.getnchannels(), array.array('h', wav.readframes(wav.getnframes())))


parity = []
for length in ('short', 'medium'):
    ref = pcm(root/'qwen-warm-off-a1'/f'{length}_1.wav')
    assert any(ref[2]), 'Silent reference output'
    for variant in variants:
        if variant.startswith('vulkan'):
            continue  # Backend sampling drift is assessed by ASR, not waveform parity.
        for repeat in (1, 2):
            out = pcm(root/f'qwen-warm-{variant}'/f'{length}_{repeat}.wav')
            assert ref[:2] == out[:2] and len(ref[2]) == len(out[2]), 'HIP audio shape changed'
            error = sum((a-b)**2 for a, b in zip(ref[2], out[2]))
            energy = sum(a*a for a in ref[2])
            snr = 10*math.log10(energy/error) if error else None
            peak = max(abs(a-b) for a, b in zip(ref[2], out[2]))/32768
            parity.append(dict(length=length, variant=variant, repeat=repeat,
                               byte_exact=error == 0, snr_db=snr, peak_error=peak))
            assert error == 0 or (snr >= 60 and peak <= 0.001), parity[-1]


def wer(reference, actual):
    previous = list(range(len(actual)+1))
    for i, word in enumerate(reference, 1):
        current = [i]
        for j, other in enumerate(actual, 1):
            current.append(min(previous[j]+1, current[-1]+1, previous[j-1]+(word != other)))
        previous = current
    return previous[-1]/len(reference)


requests = json.loads(Path('.github/tests/halo_requests.json').read_text())
reference = re.findall('[a-z]+', next(r['text'] for r in requests if r['id']=='medium_2').lower())
asr = {}
for variant in ('off-a2', 'on-b2', 'vulkan-c2'):
    text = (root/f'qwen-asr-{variant}.txt').read_text().strip()
    error = wer(reference, re.findall('[a-z]+', text.lower()))
    asr[variant] = dict(text=text, wer=error)
    assert error <= 0.2, asr[variant]

summary = {}
for length in ('short', 'medium'):
    summary[length] = {}
    for mode in ('off', 'on', 'vulkan'):
        group = [r for r in rows if r['mode']==mode and r['request'].startswith(length)]
        assert len(group) == 4, (length, mode, len(group))
        summary[length][mode] = dict(rtf=statistics.median(r['rtf'] for r in group),
            decoder_ms=statistics.median(r['stages']['qwen3_tts.speech_decoder.graph.compute_ms'] for r in group),
            talker_ms=statistics.median(r['stages']['qwen3_tts.talker_ms'] for r in group))

kernel_totals = {}
for mode in ('off', 'on'):
    totals = defaultdict(float)
    for path in (root/f'qwen-profile-{mode}').rglob('*kernel_trace.csv'):
        for r in csv.DictReader(path.open()):
            totals[r['Kernel_Name']] += (int(r['End_Timestamp'])-int(r['Start_Timestamp']))/1e6
    kernel_totals[mode] = sorted(totals.items(), key=lambda x: -x[1])
assert any('halo_sgemm' in name for name, _ in kernel_totals['on']), 'Model did not select dedicated kernel'
assert not any('halo_sgemm' in name for name, _ in kernel_totals['off'])

result = dict(summary=summary, requests=rows, parity=parity, asr=asr, kernel_totals=kernel_totals)
(root/'qwen-analysis.json').write_text(json.dumps(result, indent=2)+'\n')
report = json.dumps(dict(summary=summary, parity=parity, asr=asr), indent=2)+'\n'
print(report)
(root/'qwen-summary.txt').write_text(report)
if os.environ.get('GITHUB_STEP_SUMMARY'):
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as handle:
        handle.write('```json\n'+report+'```\n')
