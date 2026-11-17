#!/usr/bin/env python3
"""Official SVS reference for score and melody control, with aligned fixtures."""
import argparse
import copy
import json
import logging
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--control',choices=['score','melody'],default='score')
    p.add_argument('--example',choices=['zh','en','yue'],default='zh')
    p.add_argument('--mode',choices=['control','performance'],default='control')
    p.add_argument('--repeat',type=int,default=3)
    p.add_argument('--log',action='store_true',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(a.upstream))
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
    from cli.inference import build_model
    from soulxsinger.utils.file_utils import load_config
    from soulxsinger.utils.data_processor import DataProcessor
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    logging.basicConfig(level=logging.INFO)
    if a.mode=='control':
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=load_config(str(a.upstream/'soulxsinger/config/soulxsinger.yaml'))
    processor=DataProcessor(480,24000,str(a.upstream/'soulxsinger/utils/phoneme/phone_set.json'),'cuda')
    prompt_name='en_prompt' if a.example == 'en' else 'zh_prompt'
    target_name={'zh':'music','en':'en_target','yue':'yue_target'}[a.example]
    pm=json.loads((a.upstream/f'example/audio/{prompt_name}.json').read_text())[0]
    tm=json.loads((a.upstream/f'example/audio/{target_name}.json').read_text())[0]
    # Keep the first eight seconds of the official melody and its aligned notes.
    ds=list(map(float,tm['duration'].split()));size=0;elapsed=0
    for d in ds:
        size+=1;elapsed+=d
        if elapsed>=8:break
    ds=ds[:size];ds[-1]-=max(0,elapsed-8)
    tm['duration']=' '.join(map(str,ds))
    for key in ['phoneme','note_pitch','note_type']:tm[key]=' '.join(tm[key].split()[:size])
    tm['f0']=' '.join(tm['f0'].split()[:400]);tm['time']=[0,round(min(elapsed,8)*1000)]
    (a.output/'reference_metadata.json').write_text(json.dumps([pm],ensure_ascii=False))
    (a.output/'target_metadata.json').write_text(json.dumps([tm],ensure_ascii=False))
    prompt=processor.process(copy.deepcopy(pm),str(a.upstream/f'example/audio/{prompt_name}.mp3'))
    target=processor.process(copy.deepcopy(tm))
    model=build_model(str(a.checkpoint),config,'cuda',a.mode=='performance')
    tensors={};handles=[]
    if a.mode=='control':
        from unittest.mock import patch
        randn=torch.randn
        def random(*args,**kwargs):
            value=randn(*args,**kwargs);tensors['noise']=value.transpose(1,2).detach().float().cpu().contiguous();return value
        def mel_hook(module,args):tensors['mel']=args[0].detach().float().cpu().contiguous()
        def features_hook(module,args,output):tensors['note_content']=output.detach().float().cpu().contiguous()
        handles=[model.vocoder.register_forward_pre_hook(mel_hook),model.preflow.register_forward_hook(features_hook)]
        context=patch('torch.randn',side_effect=random)
    else:
        from contextlib import nullcontext
        context=nullcontext()
    torch.manual_seed(2026)
    timings=[]
    with torch.no_grad(),context:
        if a.mode == 'performance':
            model.infer({'prompt':prompt,'target':target},auto_shift=True,n_steps=2,cfg=3,control=a.control,use_fp16=True)
            torch.cuda.synchronize()
        for _ in range(a.repeat if a.mode == 'performance' else 1):
            torch.manual_seed(2026)
            torch.cuda.synchronize();started=time.perf_counter()
            wave=model.infer({'prompt':prompt,'target':target},auto_shift=True,n_steps=32,cfg=3,control=a.control,use_fp16=a.mode=='performance')
            torch.cuda.synchronize();timings.append((time.perf_counter()-started)*1000)
            logging.info('session.wall_ms %.6f',timings[-1])
    if a.mode == 'performance':
        stop.set();sampler.join();monitor.sample()
        resources=monitor.metrics.to_json();monitor.close()
        seconds=target['mel2note'].shape[1]*480/24000
        (a.output/'timing.json').write_text(json.dumps({'wall_ms':timings,'audio_seconds':seconds,'rtf':[t/1000/seconds for t in timings],'resources':resources},indent=2))
    for h in handles:h.remove()
    wave=wave.flatten().float().cpu();sf.write(a.output/'python.wav',wave.numpy(),24000,subtype='FLOAT')
    if a.mode=='control':
        prefix=int(prompt['mel2note'].shape[1]);frames=int(target['mel2note'].shape[1])
        joined=torch.cat([prompt['mel2note'],target['mel2note']+prompt['phoneme'].shape[1]],dim=1).cpu()
        expanded=torch.gather(tensors.pop('note_content'),1,joined[:,:,None].expand(-1,-1,512))
        with torch.no_grad():reference_mel=model.mel(prompt['waveform']).float().cpu()
        tensors.update(reference_waveform=prompt['waveform'].float().cpu(),source_waveform=torch.zeros(1,frames*480),reference_mel=reference_mel,reference_content=expanded[:,:prefix].contiguous(),source_content=expanded[:,prefix:].contiguous(),waveform=wave)
        for side,item in [('reference',prompt),('source',target)]:
            tensors[side+'_f0']=item['f0'].float().cpu() if a.control=='melody' else torch.zeros(1,item['mel2note'].shape[1])
            for key in ['phoneme','note_pitch','note_type','mel2note']:tensors[side+'_'+key]=item[key].to('cpu',torch.int32).contiguous()
        save_file({k:v.contiguous() for k,v in tensors.items()},str(a.output/'reference.safetensors'),metadata={'control':a.control,'steps':'32'})
        print('fixture shapes',{k:tuple(v.shape) for k,v in tensors.items()},flush=True)


if __name__=='__main__':main()
