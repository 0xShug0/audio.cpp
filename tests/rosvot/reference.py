#!/usr/bin/env python3
"""Run the official ROSVOT/RWBD graphs and export component parity fixtures."""
import argparse
import json
import logging
import os
import threading
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml
from safetensors.torch import save_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--audio', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backend', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--warmup-requests', type=int, default=1)
    p.add_argument('--measured-requests', type=int, default=2)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--word-durations-file', type=Path)
    p.add_argument('--controlled', action='store_true', help='Disable TF32 only for mathematical parity')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(a.log), logging.StreamHandler()])
    torch.set_num_threads(a.threads)
    if a.controlled:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(a.source.resolve()))
    from modules.rosvot.rosvot import MidiExtractor, WordbdExtractor, regulate_boundary
    from modules.pe.rmvpe import RMVPE
    from utils.audio.mel import MelNet
    from utils.audio.pitch_utils import norm_interp_f0, denorm_f0, f0_to_coarse, boundary2Interval
    import librosa
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools/audiocpp_cli'))
    from run_audiocpp_cli_path_tests import ResourceMonitor
    monitor = ResourceMonitor(os.getpid(), 0, 10)
    stop = threading.Event()
    def sample():
        while not stop.wait(.01): monitor.sample()
    sampling = threading.Thread(target=sample, daemon=True); sampling.start()
    device = torch.device(a.backend)
    configs = {n: yaml.safe_load((a.checkpoint/n/'config.yaml').read_text()) for n in ['rosvot', 'rwbd']}
    models = {}
    for n, cls in [('rosvot', MidiExtractor), ('rwbd', WordbdExtractor)]:
        models[n] = cls(configs[n]).eval().to(device)
        models[n].load_state_dict(torch.load(a.checkpoint/n/'model.pt', map_location='cpu', weights_only=False)['state_dict']['model'])
    models['rosvot'].note_bd_threshold = 0.85
    models['rwbd'].word_bd_threshold = 0.5
    pe = RMVPE(str(a.checkpoint/'rmvpe/model.pt'), device=device)
    melnet = MelNet(configs['rosvot']); melnet.to(device)
    wav, _ = librosa.load(a.audio, sr=24000)
    real = int(np.ceil(len(wav)/128))
    word_durations = None
    if a.word_durations_file:
        raw = json.loads(a.word_durations_file.read_text())
        word_durations = []
        for i, d in enumerate(raw):
            if d < .02:
                if i == 0 and len(raw)>1: raw[i+1] += d
                elif word_durations: word_durations[-1] += d
            else: word_durations.append(d)
        real = min(real, int(sum(word_durations)*24000/128+.5))
    frames = int(np.ceil(real/16)*16)
    wav_t = torch.from_numpy(np.pad(wav, (0, max(0,frames*128-len(wav))))).unsqueeze(0).to(device)
    fixtures = {}
    outputs = {}
    timings = []
    pitch_timings = []
    with torch.inference_mode():
        for run in range(a.warmup_requests + a.measured_requests):
            if device.type == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter()
            f0, _ = pe.get_pitch_batch(wav_t, 24000, 128, [real], fmin=50, fmax=900)
            if device.type == 'cuda': torch.cuda.synchronize()
            pitch_ms = (time.perf_counter()-start)*1000
            pitch_timings.append(pitch_ms)
            logging.info('rosvot.pitch_extraction_ms %.6f', pitch_ms)
            f, uv = norm_interp_f0(f0[0])
            f = torch.from_numpy(np.pad(f, (0, frames-real))).unsqueeze(0).to(device)
            uv = torch.from_numpy(np.pad(uv.astype(np.int64), (0, frames-real))).unsqueeze(0).to(device)
            pitch = f0_to_coarse(denorm_f0(f, uv))
            mel = melnet(wav_t)[:, :frames]
            mel[:, real:] = 0
            keep = mel.abs().sum(-1)>0
            w = models['rwbd'](mel=mel[:,:,:40], pitch=pitch, uv=uv, non_padding=keep, train=False)
            if word_durations is not None:
                wb = torch.zeros(1,frames,dtype=torch.long,device=device)
                offset = 0
                for d in word_durations[:-1]:
                    offset += d
                    frame = int(offset*24000/128+.5)
                    if 0<frame<real: wb[0,frame]=1
                w['word_bd_pred'] = wb
            x = models['rosvot'](mel=mel[:,:,:40], pitch=pitch, uv=uv, word_bd=w['word_bd_pred'], non_padding=keep, train=False)
            if device.type == 'cuda': torch.cuda.synchronize()
            elapsed = (time.perf_counter()-start)*1000
            timings.append(elapsed)
            logging.info('session.wall_ms %.6f rtf %.6f', elapsed, elapsed/1000/(len(wav)/24000))
        # Freeze process-memory measurements before validation-only fixture work.
        stop.set(); sampling.join(); monitor.sample(); monitor.close()
        memory = monitor.metrics.to_json()
        feat = models['rosvot'].run_encoder(mel[:,:,:40],w['word_bd_pred'],pitch,uv,keep)
        feat = models['rosvot'].net(feat)
        attn = models['rosvot'].pitch_decoder.multihead_dot_attn(feat).sigmoid().mean(-1)
        fixtures = {'mel': mel[:,:,:40].contiguous(), 'pitch': pitch.int(), 'uv': uv.int(),
                    'word_boundary': w['word_bd_pred'].int(), 'word_logits': w['word_bd_logits'],
                    'features': feat, 'note_logits': x['note_logits'], 'note_boundary': x['note_bd_pred'].int(),
                    'boundary_logits': x['note_bd_logits'], 'attention': attn, 'note_pitch': x['note_pred'].int(),
                    'waveform': torch.from_numpy(wav).float()}
        intervals = boundary2Interval(x['note_bd_pred'][0,:real]).cpu().numpy()
        pitches = x['note_pred'][0,:len(intervals)].cpu().tolist()
        if word_durations is not None:
            from tasks.rosvot.rosvot_utils import regulate_real_note_itv, regulate_ill_slur
            intervals, mapping = regulate_real_note_itv(intervals, x['note_bd_pred'][0,:real].cpu().numpy(),
                w['word_bd_pred'][0,:real].cpu().numpy(), np.asarray(word_durations,dtype=np.float32), 128, 24000)
            pitches, intervals, mapping = regulate_ill_slur(np.asarray(pitches), intervals, mapping)
            pitches = pitches.tolist()
            intervals = intervals*24000/128
        outputs = {'real_frames':real,'frames':frames,'pitches':pitches,'intervals':(intervals*128/24000).tolist()}
    save_file({k:v.detach().cpu().contiguous() for k,v in fixtures.items()}, a.output/'reference.safetensors')
    (a.output/'notes.json').write_text(json.dumps(outputs, indent=2)+'\n')
    logging.info('notes %s', outputs)
    (a.output/'metrics.json').write_text(json.dumps({'wall_ms':timings,'pitch_ms':pitch_timings,'duration_sec':len(wav)/24000,'memory':memory},indent=2))


if __name__ == '__main__':
    main()
