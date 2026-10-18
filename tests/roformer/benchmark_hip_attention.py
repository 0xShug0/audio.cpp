"""Retain a bounded separation run and optionally compare PCM16 stems.

Run baseline and candidate sequentially with the same model and input. A timeout
is reported as a lower bound, never as a measured completed baseline duration.
"""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import wave


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pcm(path):
    with wave.open(str(path)) as stream:
        if stream.getsampwidth() != 2:
            raise ValueError('Comparison expects PCM16 WAV output')
        signature = (stream.getnchannels(), stream.getframerate(), stream.getnframes())
        data = array.array('h', stream.readframes(stream.getnframes()))
        if sys.byteorder != 'little':
            data.byteswap()
        return signature, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'model', 'audio', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    command = [str(args.binary.resolve()), '--family', 'bs_roformer', '--task', 'sep',
               '--backend', 'hip', '--model', str(args.model.resolve()),
               '--audio', str(args.audio.resolve()), '--out-dir', str(args.out.resolve()), '--log']
    result = dict(command=command, binary_sha256=digest(args.binary), input_sha256=digest(args.audio))
    result['environment'] = {key: value for key, value in os.environ.items()
                             if key.startswith(('GGML_', 'HIP_', 'ROCBLAS_', 'HIPBLASLT_'))}
    started = time.monotonic()
    with (args.out / 'run.log').open('w') as log:
        try:
            result['returncode'] = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                                  timeout=args.timeout).returncode
        except subprocess.TimeoutExpired:
            result['timed_out'] = True
    result['seconds'] = time.monotonic() - started
    if args.reference and result.get('returncode') == 0:
        result['comparison'] = {}
        for stem in ('vocals', 'instrumental'):
            shape, candidate = pcm(args.out / (stem + '.wav'))
            ref_shape, reference = pcm(args.reference / (stem + '.wav'))
            if shape != ref_shape:
                raise ValueError('Reference and candidate WAV dimensions differ')
            errors = [int(a) - int(b) for a, b in zip(candidate, reference)]
            result['comparison'][stem] = dict(
                identical=not any(errors),
                max_abs_pcm16_steps=max(map(abs, errors), default=0),
                rms_difference=math.sqrt(sum(e * e for e in errors) / max(1, len(errors))) / 32768)
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result.get('returncode') == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
