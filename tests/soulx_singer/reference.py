#!/usr/bin/env python3
"""Export aligned official FP32 component fixtures, or run normal Python SVC."""
import argparse
import importlib.util
import logging
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--rmvpe',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--audio',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['control','performance'],default='control')
    p.add_argument('--seconds',type=float,default=0)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--steps',type=int,default=32)
    p.add_argument('--repeat',type=int,default=3,help='Measured performance requests after a short warmup')
    p.add_argument('--log',action='store_true',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(a.upstream))
    import torch
    if a.mode == 'performance':
        import os
        import threading
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/audiocpp_cli'))
        from run_audiocpp_cli_path_tests import ResourceMonitor
        monitor=ResourceMonitor(os.getpid(),0,20)
        stop=threading.Event()
        def sample():
            while not stop.wait(.02): monitor.sample()
        sampler=threading.Thread(target=sample,daemon=True);sampler.start()
    import numpy as np
    import soundfile as sf
    from safetensors.torch import save_file
    from cli.inference_svc import build_model
    from soulxsinger.utils.file_utils import load_config
    from soulxsinger.utils.audio_utils import load_wav
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    logging.basicConfig(level=logging.INFO)
    if a.mode=='control':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    spec=importlib.util.spec_from_file_location('official_f0',a.upstream/'preprocess/tools/f0_extraction.py')
    f0=importlib.util.module_from_spec(spec);spec.loader.exec_module(f0)
    extractor=f0.F0Extractor(str(a.rmvpe),device='cuda')
    reference=load_wav(str(a.reference),24000).cuda()
    audio=load_wav(str(a.audio),24000).cuda()
    if a.seconds:audio=audio[:,:int(a.seconds*24000)]
    sf.write(a.output/'input.wav',audio[0].cpu().numpy(),24000,subtype='FLOAT')
    reference_f0=torch.from_numpy(extractor.process(str(a.reference))).float().unsqueeze(0).cuda()
    source_f0=torch.from_numpy(extractor.process(str(a.output/'input.wav'))).float().unsqueeze(0).cuda()
    if a.mode == 'control':
        del extractor;torch.cuda.empty_cache()
    config=load_config(str(a.upstream/'soulxsinger/config/soulxsinger.yaml'))
    model=build_model(str(a.checkpoint),config,'cuda',a.mode=='performance')
    torch.manual_seed(a.seed);tensors={};handles=[]
    if a.mode=='control':
        from unittest.mock import patch
        original_randn=torch.randn
        def random(*args,**kwargs):
            value=original_randn(*args,**kwargs)
            tensors['noise']=value.transpose(1,2).detach().float().cpu().contiguous()
            return value
        def vocoder_hook(module,args):tensors['mel']=args[0].detach().float().cpu().contiguous()
        handles.append(model.vocoder.register_forward_pre_hook(vocoder_hook))
        capture=patch('torch.randn',side_effect=random)
    else:
        from contextlib import nullcontext
        capture=nullcontext()
    timings=[]
    with torch.no_grad(),capture:
        if a.mode == 'performance':
            # Warm the same session; this output is excluded from measurement.
            model.infer(reference,audio[:,:24000],reference_f0,source_f0[:,:50],auto_shift=True,n_steps=2,cfg=3,use_fp16=True)
            torch.cuda.synchronize()
        for _ in range(a.repeat if a.mode == 'performance' else 1):
            torch.manual_seed(a.seed)
            torch.cuda.synchronize();started=time.perf_counter()
            if a.mode == 'performance':
                source_f0=torch.from_numpy(extractor.process(str(a.output/'input.wav'))).float().unsqueeze(0).cuda()
            output,shift=model.infer(reference,audio,reference_f0,source_f0,auto_shift=True,n_steps=a.steps,cfg=3,use_fp16=a.mode=='performance')
            torch.cuda.synchronize()
            timings.append((time.perf_counter()-started)*1000)
            logging.info('session.wall_ms %.6f',timings[-1])
    if a.mode == 'performance':
        import json
        stop.set();sampler.join();monitor.sample()
        resources=monitor.metrics.to_json();monitor.close()
        (a.output/'timing.json').write_text(json.dumps({'wall_ms':timings,'audio_seconds':audio.shape[-1]/24000,'rtf':[t/1000/(audio.shape[-1]/24000) for t in timings],'resources':resources},indent=2))
    for h in handles:h.remove()
    sf.write(a.output/'python.wav',output.flatten().float().cpu().numpy(),24000,subtype='FLOAT')
    if a.mode=='control':
        with torch.no_grad():
            tensors.update(reference_waveform=reference.cpu(),source_waveform=audio.cpu(),reference_f0=reference_f0.cpu(),source_f0=source_f0.cpu(),waveform=output.cpu())
            tensors['reference_mel']=model.mel(reference).cpu().contiguous()
            tensors['reference_content']=model.whisper_encoder.encode(reference,sr=24000).cpu().contiguous()
            tensors['source_content']=model.whisper_encoder.encode(audio,sr=24000).cpu().contiguous()
        save_file(tensors,str(a.output/'reference.safetensors'),metadata={'pitch_shift':str(shift),'steps':str(a.steps)})
        print('fixture shapes', {k:tuple(v.shape) for k,v in tensors.items()},flush=True)


if __name__=='__main__':main()
