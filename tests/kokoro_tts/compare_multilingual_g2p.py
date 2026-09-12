"""Compare native pronunciation with upstream eSpeak and Misaki frontends."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import espeakng_loader
from phonemizer.backend.espeak.wrapper import EspeakWrapper
EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
from misaki import espeak, ja, zh

p = argparse.ArgumentParser()
p.add_argument('--probe', type=Path, required=True)
p.add_argument('--resources', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
cases_path = Path(__file__).with_name('multilingual_cases.json')
cases = json.loads(cases_path.read_text(encoding='utf-8'))
frontends = {k: espeak.EspeakG2P(language=v) for k,v in
             {'e':'es', 'f':'fr-fr', 'h':'hi', 'i':'it', 'p':'pt-br'}.items()}
frontends['j'], frontends['z'] = ja.JAG2P(), zh.ZHG2P()
native = subprocess.run([str(a.probe.resolve()), str(a.resources.resolve()), str(cases_path.resolve())],
                        capture_output=True, encoding='utf-8', check=True)
lines = native.stdout.splitlines()
if len(lines) != len(cases): raise RuntimeError(native.stdout + native.stderr)
results = []
for case, actual in zip(cases, lines):
    expected = frontends[case['language']](case['text'])[0]
    results.append({**case, 'reference':expected, 'native':actual, 'match':expected == actual})
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
for i, r in enumerate(results): print(i, r['language'], r['match'], flush=True)
print('Report:', a.output)
