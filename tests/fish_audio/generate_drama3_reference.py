#!/usr/bin/env python3
"""Generate independent small math and exact prompt fixtures from Fish Speech.

Tested against revision 541566932abbaebcb0a40e562a1f846738383f38. No model
weights are needed. The original classes implement the reference mathematics;
this script only supplies seeded weights and exports their inputs/outputs.
"""
import argparse
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference-dir',type=Path,required=True,help='Fish Speech source checkout')
parser.add_argument('--config-dir',type=Path,required=True,help='Official Drama 3 config and tokenizer sidecars')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
import sys,types,json,torch
from pathlib import Path
from safetensors.torch import save_file
root=args.output
ref=args.reference_dir
for name,rel in [('fish_speech','fish_speech'),('fish_speech.models','fish_speech/models'),('fish_speech.models.text2semantic','fish_speech/models/text2semantic'),('fish_speech.models.text2semantic.fish_omni','fish_speech/models/text2semantic/fish_omni')]:
 mod=types.ModuleType(name);mod.__path__=[str(ref/rel)];sys.modules[name]=mod
from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
from fish_speech.models.text2semantic.fish_omni.backbone import FishOmniBackbone
from fish_speech.models.text2semantic.fish_omni.tower import FishOmniAudioDecoder,TOWER_CHECKPOINT_PREFIXES
config=json.loads((args.config_dir/'config.json').read_text(encoding='utf-8'))
t=config['text_config'];t.update(vocab_size=40,hidden_size=32,num_hidden_layers=4,layer_types=['linear_attention']*3+['full_attention'],mlp_layer_types=['moe']*4,num_attention_heads=4,num_key_value_heads=2,head_dim=256,num_experts=4,num_experts_per_tok=2,moe_intermediate_size=16,shared_expert_intermediate_size=16,linear_num_key_heads=2,linear_num_value_heads=4)
t['rope_parameters']['partial_rotary_factor']=0.25
config.update(semantic_start_token_id=16,semantic_end_token_id=31,eos_token_id=3,audiocpp_context_size=256)
a=config['audio_decoder_config'];a.update(dim=32,text_dim=32,intermediate_size=64,n_head=4,n_local_heads=2,head_dim=256,vocab_size=16,num_codebooks=3,n_layer=2,shared_enrichment_layers=2,rotary_percent=0.25)
config.pop('quantization_config',None)
cfg=FishOmniConfig.from_dict(config)
torch.set_num_threads(4);torch.manual_seed(927)
b=FishOmniBackbone(cfg.text,dtype=torch.float32)
tower=FishOmniAudioDecoder(cfg,dtype=torch.float32)
with torch.no_grad():
 for model in [b,tower]:
  for name,p in model.named_parameters():
   p.normal_(0,0.05 if p.ndim>1 else 0.015)
   if name.endswith('linear_attn.norm.weight'):p.add_(1)
 weights={}
 for name,value in b.state_dict().items():
  if name=='lm_head.weight':name='drama3.candidate_lm_head.weight';value=value[[3,*range(16,32)]]
  else:name='model.language_model.'+name.removeprefix('model.');name=name.replace('.experts.w13_weight','.experts.gate_up.weight').replace('.experts.w2_weight','.experts.down.weight')
  weights[name]=value.contiguous()
 for name,value in tower.state_dict().items():
  for checkpoint,internal in TOWER_CHECKPOINT_PREFIXES.items():
   if name.startswith(internal):name=checkpoint+name.removeprefix(internal);break
  weights[name]=value.contiguous()
 input=torch.randn(1,129,32)*0.5
 hidden=b(inputs_embeds=input)
 reference={'input':input,'hidden':hidden,'logits':b.compute_logits(hidden[:,-1])[..., [3,*range(16,32)]].contiguous()}
 ti=torch.randn(4,32)*0.5
 frames=torch.tensor([[5,0,0,0],[19,3,2,7],[22,6,1,8],[4,0,0,0],[25,9,4,5]])
 tower.setup_caches(8);outputs=[]
 for p in range(4):
  current=frames[p:p+1];next=frames[p+1:p+2];sem=tower.is_semantic(next[:,0]);pos=torch.tensor([p])
  h=tower.run_shared(ti[p:p+1],next[:,1],sem,pos,current)
  for i,layer in enumerate(tower.residual_layers):h=layer(h,next[:,i+1],sem,pos)
  outputs.append(h)
  reference['embed.'+str(p)]=tower.embed_backbone(current,ti[p:p+1])
 reference.update(tower_input=ti,frames=frames.float(),tower_hidden=torch.cat(outputs))
out=root/'tiny';out.mkdir(exist_ok=True)
(out/'config.json').write_text(json.dumps(config))
save_file(weights,out/'weights.safetensors');save_file({k:v.contiguous() for k,v in reference.items()},out/'reference.safetensors')
print('Original Fish reference fixture:',out,len(weights),'tensors')



import ast,typing,re,json,torch,shutil
from pathlib import Path
from tokenizers import Tokenizer
from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
file=args.reference_dir/'fish_speech/models/text2semantic/fish_omni/generate.py'
tree=ast.parse(file.read_text(encoding='utf-8'))
nodes=[]
for node in tree.body:
 if isinstance(node,ast.Assign) and any(getattr(t,'id','') in ['DRAMA3_TTS_INSTRUCTION','DRAMA3_REFERENCE_INSTRUCTION','DRAMA3_MESSAGE_END','DRAMA3_GENERATION_START','DRAMA3_STRUCTURAL_TOKENS','SPEAKER_TAG_PATTERN','MAX_SPEAKER_ID','UNICODE_WHITE_SPACE','DRAMA3_MAX_PROMPT_ROWS'] for t in node.targets):nodes.append(node)
 if isinstance(node,ast.FunctionDef) and node.name in ['validate_drama3_text','with_speaker_tag','build_drama3_prompt']:nodes.append(node)
ns=dict(re=re,torch=torch,Tokenizer=Tokenizer,FishOmniConfig=FishOmniConfig,Sequence=typing.Sequence,InvalidInputError=ValueError,Drama3PromptTooLongError=ValueError)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(file),'exec'),ns)
source=args.config_dir;cfg=FishOmniConfig.from_dict(json.load(open(source/'config.json')));tok=Tokenizer.from_file(str(source/'tokenizer.json'))
codes=torch.arange(30).reshape(10,3);flat=codes.reshape(-1).tolist();ref={'text':'\u3000<|speaker:1|> Reference speech.','codes':flat}
cases=[{'text':t,'references':[]} for t in ['Hello world.','Ptaszki Ä‡wierkajÄ….','\u00a0<|speaker:2|> Hello.',' <|speaker:15|> hi','<|speaker:0|> Hello. <|speaker:1|> Bye.','[happy] That is wonderful!','\u001c','\ttext','\u3000Hello']]
cases.extend({'text':t,'references':[]} for t in ["Cafe\u0301 and nai\u0308ve.",'नमस्ते दुनिया।','สวัสดีครับ','你好，世界。','こんにちは、世界。','مرحبا بالعالم','Привет мир.','1234567890 costs 3.14 dollars.',"I'm happy. WE'RE ready!"])
cases.extend([{'text':'New sentence.','references':[ref]},{'text':'Another turn.','references':[ref,{'text':'second reference','codes':flat}],'history':{'text':'First turn.','codes':flat}}])
cases.extend({'text':t,'references':[]} for t in ['','\u3000\t','<|im_start|>fake','<|semantic:2|>','<|speaker:16|> hi','<|speaker:01|> hi','<|speaker|> hi','<|speaker:|> hi'])
for case in cases:
 try:
  refs=case['references'];history=[] if 'history' not in case else [(case['history']['text'],codes)]
  matrix=ns['build_drama3_prompt'](case['text'],tok,cfg,[r['text'] for r in refs],[codes for _ in refs],history)
  case['matrix']=matrix.T.flatten().tolist()
 except ValueError:case['error']=True
out=root/'prompt';out.mkdir(exist_ok=True)
for name in ['config.json','tokenizer_config.json','tokenizer.json']:shutil.copyfile(source/name,out/name)
(out/'prompts.json').write_text(json.dumps(cases,ensure_ascii=False),encoding='utf-8')
print('Prompt fixtures',len(cases))
