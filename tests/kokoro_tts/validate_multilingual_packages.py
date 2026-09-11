"""Synthesize one UTF-8 sample for every Kokoro language from both standalone GGUFs."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import wave
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--cli', type=Path, required=True)
p.add_argument('--models', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--types', nargs='+', default=['q8_0', 'bf16'])
a = p.parse_args()
cases = json.loads(Path(__file__).with_name('multilingual_cases.json').read_text(encoding='utf-8'))[:7]
cases = [
    {'language':'en-us','voice':'af_heart','text':'Hello, this is a native Kokoro TTS test. Good morning to everyone.'},
    {'language':'en-gb','voice':'bf_emma','text':'Hello, this is a native Kokoro TTS test. Good morning to everyone.'}
] + cases
codes = {'e':'es', 'f':'fr-fr', 'h':'hi', 'i':'it', 'p':'pt-br', 'j':'ja', 'z':'zh'}
a.output.mkdir(parents=True, exist_ok=True)
report = []
for precision in a.types:
    model = a.models.resolve() / ('kokoro-v1.0-' + precision + '.gguf')
    for case in cases:
        lang = codes.get(case['language'], case['language'])
        dest = a.output.resolve() / (precision + '-' + lang)
        dest.mkdir(exist_ok=True)
        textfile = dest / 'text.txt'
        textfile.write_text(case['text'] + '\n', encoding='utf-8')
        wav = a.output.resolve() / (precision + '-' + lang + '.wav')
        command = [str(a.cli.resolve()), '--task', 'tts', '--family', 'kokoro_tts',
            '--model', str(model), '--backend', 'cpu', '--threads', '8', '--seed', '1234',
            '--voice-id', case['voice'], '--language', lang, '--batch-text-file', str(textfile),
            '--batch-merge-audio', 'concat', '--out', str(wav)]
        start = time.perf_counter()
        run = subprocess.run(command, capture_output=True, encoding='utf-8', errors='replace', timeout=180)
        elapsed = time.perf_counter() - start
        (dest / 'console.log').write_text(run.stdout + run.stderr, encoding='utf-8')
        result = {'precision':precision, **case, 'exit_code':run.returncode, 'process_seconds':elapsed}
        if run.returncode == 0:
            with wave.open(str(wav), 'rb') as f:
                assert f.getsampwidth() == 2
                pcm = np.frombuffer(f.readframes(f.getnframes()), dtype='<i2').astype(np.float64) / 32768
                result.update(sample_rate=f.getframerate(), channels=f.getnchannels(),
                    seconds=f.getnframes()/f.getframerate(), rms=float(np.sqrt(np.mean(pcm * pcm))),
                    peak=float(np.abs(pcm).max()), file=str(wav))
                result['valid_audio'] = result['seconds'] > 1 and result['rms'] > 0.002
        else:
            result['error'] = (run.stdout + run.stderr)[-2000:]
        report.append(result)
        (a.output / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(precision, lang, 'PASS' if result.get('valid_audio') else result.get('error', 'INVALID AUDIO'), flush=True)
if any(not r.get('valid_audio') for r in report): raise SystemExit(1)
