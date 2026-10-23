import csv
import hashlib
import json
import os
import re
import statistics
import wave
from collections import defaultdict
from pathlib import Path

root = Path('results')
variants = ('off-a1','on-b1','on-b2','off-a2')
requests = json.loads(Path('.github/tests/halo_requests.json').read_text())
data = {}
hashes = defaultdict(set)
for variant in variants:
    log = (root / f'{variant}.log').read_text()
    pending = defaultdict(float)
    stages = {}
    metrics = defaultdict(dict)
    for line in log.splitlines():
        timing = re.match(r'^\[TIMING[^\]]*\] (kokoro\.[\w.]+) ([\d.eE+-]+)$',line)
        if timing: pending[timing[1]] += float(timing[2])
        metric = re.match(r'^metrics\[([^\]]+)\]\.([^=]+)=([^\n]+)$',line)
        if metric:
            request,key,value = metric.groups()
            if request not in stages:
                stages[request] = dict(pending)
                pending.clear()
            metrics[request][key] = value
    rows = []
    for request in requests:
        name = request['id']
        wav = root / variant / f'{name}.wav'
        hashes[name].add(hashlib.sha256(wav.read_bytes()).hexdigest())
        with wave.open(str(wav),'rb') as audio:
            assert audio.getnframes() > 0, 'Empty audio'
        if name.endswith('warmup'): continue
        assert 'kokoro.decoder.generator_ms' in stages[name], 'Missing generator component timing'
        rows.append({'request':name,
                     'rtf':float(metrics[name]['wall_ms'])/float(metrics[name]['audio_duration_ms']),
                     'reported_rtf':float(metrics[name]['rtf']),'stages':stages[name]})
    data[variant] = rows
assert all(len(values)==1 for values in hashes.values()), 'Model WAV parity regression'
reference = 'and so my fellow americans ask not what your country can do for you ask what you can do for your country'.split()
for mode in ('off','on'):
    assert re.findall('[a-z]+',(root/f'{mode}-jfk.txt').read_text().lower()) == reference, f'{mode}: ASR accuracy mismatch'
kernel_log = (root/'kernel-test.log').read_text()
assert 'PASS cases=' in kernel_log
report = ['Halo IM2COL: original-kernel and independent scalar byte parity passed.',
          'All 32 Kokoro WAVs are byte-identical across off/on runs; Moonshine ASR matches reference.']
summary = {}
for length in ('short','medium'):
    by_mode = {}
    for mode in ('off','on'):
        rows = [r for variant in variants if variant.startswith(mode) for r in data[variant] if r['request'].startswith(length)]
        values = {'rtf':statistics.median(r['rtf'] for r in rows)}
        for key in rows[0]['stages']:
            values[key] = statistics.median(r['stages'][key] for r in rows)
        by_mode[mode] = values
    summary[length] = by_mode
    report.append(f'{length}: ' + json.dumps(by_mode))
for mode in ('off','on'):
    totals = defaultdict(float)
    for path in (root/f'profile-{mode}').glob('*kernel_trace.csv'):
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if 'Kernel_Name' in row and 'Start_Timestamp' in row and 'End_Timestamp' in row:
                    totals[row['Kernel_Name']] += (int(row['End_Timestamp'])-int(row['Start_Timestamp']))/1e6
    if totals:
        report.append(f'{mode} profiled kernel totals ms (includes cold start): ' + json.dumps(sorted(totals.items(),key=lambda x:-x[1])[:15]))
result = '\n'.join(report)+'\n'
print(result)
(root/'summary.txt').write_text(result)
(root/'analysis.json').write_text(json.dumps({'requests':data,'summary':summary},indent=2)+'\n')
if os.environ.get('GITHUB_STEP_SUMMARY'):
    with open(os.environ['GITHUB_STEP_SUMMARY'],'a') as handle: handle.write('```text\n'+result+'```\n')
