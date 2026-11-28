#!/usr/bin/env python3
"""Run the pinned official Real-time RE-USE model without changing its math."""
import argparse
import json
import logging
from pathlib import Path
import statistics
import sys
import time

import soundfile as sf
import torch
from safetensors.torch import load_model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--audio', type=Path, required=True)
parser.add_argument('--out-dir', type=Path, required=True)
parser.add_argument('--exit-layer', type=int, default=8)
parser.add_argument('--lookahead-frames', type=int, default=0)
parser.add_argument('--repeats', type=int, default=2)
parser.add_argument('--threads', type=int, default=8)
parser.add_argument('--no-tf32', action='store_true', help='Separate strict-F32 parity diagnostic; not the ordinary performance reference')
parser.add_argument('--mode', choices=['offline', 'streaming', 'both'], default='both')
parser.add_argument('--reference-file', type=Path, help='Export aligned network inputs and the final waveform after timing')
parser.add_argument('--log', type=Path, required=True)
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)
args.log.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(args.log, mode='w'), logging.StreamHandler()])
sys.path.insert(0, str(args.source.resolve()))
from models.streaming_generator_SEMamba_time_d1_random_layer_ahead_sep_conv import SEMamba_decoder_list
from models.stfts import mag_phase_stft, mag_phase_istft
from online_inference import StreamingSEMamba

if not 3 <= args.exit_layer <= 12 or not 0 <= args.lookahead_frames <= 2:
    parser.error('exit-layer must be 3-12 and lookahead-frames 0-2')
torch.set_num_threads(args.threads)
if args.no_tf32:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
config = json.loads((args.source / 'config.json').read_text())
model = SEMamba_decoder_list(cfg=config)
load_model(model, str(args.source / 'model.safetensors'), strict=True)
model = model.cuda().eval()
streaming = StreamingSEMamba(model).eval()
audio, sr = sf.read(args.audio, dtype='float32', always_2d=True)
if audio.shape[1] != 1:
    raise ValueError('reference requires mono audio')
waveform = torch.from_numpy(audio.T.copy()).cuda()
n_fft = 320 * sr // 8000
hop = 160 * sr // 8000
n_fft += n_fft % 2
hop += hop % 2
modes = ['offline', 'streaming'] if args.mode == 'both' else [args.mode]
results = {}
with torch.inference_mode():
    for mode in modes:
        timings = []
        for iteration in range(args.repeats + 1):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            mag, phase, _ = mag_phase_stft(waveform, n_fft, hop, n_fft, 'relu_log1p', center=True, addeps=False)
            if mode == 'offline':
                enhanced_mag, enhanced_phase, _ = model(mag, phase, args.exit_layer, args.lookahead_frames)
                zeros = (torch.expm1(torch.relu(enhanced_mag)) == 0).sum(1) / enhanced_mag.shape[1]
                enhanced_mag[:, :, (zeros > .5)[0]] = 0
            else:
                streaming.reset(mag.shape[1] // 2 + 1)
                mag = torch.nn.functional.pad(mag, (0, args.lookahead_frames))
                phase = torch.nn.functional.pad(phase, (0, args.lookahead_frames))
                enhanced_mag, enhanced_phase = streaming(mag[:, :, :args.lookahead_frames + 1], phase[:, :, :args.lookahead_frames + 1], args.exit_layer, args.lookahead_frames)
                magnitudes, phases = [enhanced_mag], [enhanced_phase]
                for frame in range(args.lookahead_frames + 1, mag.shape[-1]):
                    enhanced_mag, enhanced_phase = streaming(mag[:, :, frame:frame+1], phase[:, :, frame:frame+1], args.exit_layer, args.lookahead_frames)
                    if ((torch.expm1(torch.relu(enhanced_mag)) == 0).sum(1) / enhanced_mag.shape[1]).item() > .5:
                        enhanced_mag = 0 * enhanced_mag
                    magnitudes.append(enhanced_mag)
                    phases.append(enhanced_phase)
                enhanced_mag = torch.stack(magnitudes, -1)
                enhanced_phase = torch.stack(phases, -1)
            output = mag_phase_istft(enhanced_mag, enhanced_phase, n_fft, hop, n_fft, 'relu_log1p')
            output = torch.nn.functional.pad(output, (0, waveform.shape[-1] - output.shape[-1]), value=1e-8)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            logging.info('session.wall_ms %.9f mode=%s iteration=%d', elapsed, mode, iteration)
            timings.append(elapsed)
        sf.write(args.out_dir / f'python-{mode}.wav', output.cpu().numpy().T, sr, subtype='FLOAT')
        results[mode] = dict(warm_wall_ms=timings[1:], warm_rtf=statistics.mean(timings[1:]) / 1000 / (len(audio) / sr),
                             peak_allocated_mib=torch.cuda.max_memory_allocated()/1048576,
                             peak_reserved_mib=torch.cuda.max_memory_reserved()/1048576)
(args.out_dir / 'python-results.json').write_text(json.dumps(results, indent=2))
logging.info('%s', json.dumps(results))

if args.reference_file:
    from safetensors.torch import save_file
    with torch.inference_mode():
        mag, phase, _ = mag_phase_stft(waveform, n_fft, hop, n_fft, 'relu_log1p', center=True, addeps=False)
        enhanced_mag, enhanced_phase, _ = model(mag, phase, args.exit_layer, args.lookahead_frames)
        zeros = (torch.expm1(torch.relu(enhanced_mag)) == 0).sum(1) / enhanced_mag.shape[1]
        enhanced_mag[:, :, (zeros > .5)[0]] = 0
        expected = mag_phase_istft(enhanced_mag, enhanced_phase, n_fft, hop, n_fft, 'relu_log1p')
        expected = torch.nn.functional.pad(expected, (0, waveform.shape[-1] - expected.shape[-1]), value=1e-8)
        save_file({'features': torch.cat([mag.transpose(1, 2), phase.transpose(1, 2)], 0).contiguous().cpu(),
                   'waveform': waveform.flatten().cpu(), 'expected': expected.flatten().cpu()}, str(args.reference_file),
                  metadata={'sample_rate': str(sr), 'exit_layer': str(args.exit_layer), 'lookahead_frames': str(args.lookahead_frames)})
