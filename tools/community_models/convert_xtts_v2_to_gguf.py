#!/usr/bin/env python3
"""XTTS v2 PyTorch -> GGUF converter.

Produces two GGUF files (GPT + vocoder) compatible with audio.cpp loader.
Tensor ordering is copied from the original GGUFs to ensure exact file layout.

Usage:
    python convert_xtts_v2_to_gguf.py --checkpoint /path/to/xtts_model.pth \
        --config /path/to/config.json --vocab /path/to/vocab.json \
        --output-dir /path/to/output

Quantization (requires llama.cpp built with llama-quantize):

    llama-quantize \
        --override-kv general.architecture=str:gpt2 \
        --override-kv gpt2.context_length=int:2048 \
        --override-kv gpt2.embedding_length=int:1024 \
        --override-kv gpt2.block_count=int:30 \
        --override-kv gpt2.feed_forward_length=int:4096 \
        --override-kv gpt2.attention.head_count=int:16 \
        --override-kv gpt2.attention.layer_norm_epsilon=float:0.00001 \
        /path/to/xtts-v2-gpt.gguf /path/to/xtts-v2-gpt-q8_0.gguf q8_0

    # Q4_0 is also supported:
    llama-quantize \
        --override-kv general.architecture=str:gpt2 \
        --override-kv gpt2.context_length=int:2048 \
        --override-kv gpt2.embedding_length=int:1024 \
        --override-kv gpt2.block_count=int:30 \
        --override-kv gpt2.feed_forward_length=int:4096 \
        --override-kv gpt2.attention.head_count=int:16 \
        --override-kv gpt2.attention.layer_norm_epsilon=float:0.00001 \
        /path/to/xtts-v2-gpt.gguf /path/to/xtts-v2-gpt-q4_0.gguf q4_0

    The vocoder GGUF (~76 MB) is kept as F16. Only the GPT GGUF (~1.7 GB) is quantized.

Size reduction:
    F16/F32: 1.7 GB GPT + 76 MB vocoder = ~1.8 GB
    Q8_0:    492 MB GPT + 76 MB vocoder = ~568 MB  (3.1x smaller)
    Q4_0:    290 MB GPT + 76 MB vocoder = ~366 MB  (4.9x smaller)

Notes:
    - The --override-kv flags fake a gpt2 architecture so llama-quantize accepts
      the otherwise unknown xtts.gpt model. The gpt2 metadata is harmless since
      audio.cpp loads tensors by name, not by architecture.
    - 15 tensors with unusual shapes (non-32-aligned columns) fall back to F16.
    - The tconv1d() function in gpt.cpp handles quantized types via
      ggml_get_type_traits()->to_float for dequantization.
"""

import sys, os, json
import numpy as np
import torch

def load_xtts_model(checkpoint_path, config_path, vocab_path):
    from torch.serialization import add_safe_globals
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    add_safe_globals([XttsConfig])
    _orig = torch.load
    def _load(*a, **k):
        k['weights_only'] = False
        return _orig(*a, **k)
    torch.load = _load
    cfg = XttsConfig(); cfg.load_json(config_path)
    model = Xtts.init_from_config(cfg)
    model.load_checkpoint(cfg, checkpoint_dir=os.path.dirname(checkpoint_path),
                          checkpoint_path=checkpoint_path, vocab_path=vocab_path, eval=True)
    return model, cfg


from gguf import GGMLQuantizationType, GGUFWriter, GGUFValueType as VT

def _kv(w, key, val):
    if isinstance(val, bool): w.add_key_value(key, val, VT.BOOL)
    elif isinstance(val, int): w.add_key_value(key, val, VT.INT32)
    elif isinstance(val, float): w.add_key_value(key, val, VT.FLOAT32)
    elif isinstance(val, str): w.add_key_value(key, val, VT.STRING)
    elif isinstance(val, list): w.add_array(key, val)

def _wn(state, base_key):
    """Reconstruct WeightNorm weight from parametrized originals."""
    g = state[f"{base_key}.parametrizations.weight.original0"].detach().cpu().float()
    v = state[f"{base_key}.parametrizations.weight.original1"].detach().cpu().float()
    norm = torch.linalg.norm(v, dim=tuple(range(1, v.dim())), keepdim=True)
    return (g * v / norm).numpy()

def _t(state, pt_name, transpose=False, dtype=None, reshape=None, permute=None):
    t = state[pt_name].detach().cpu().float()
    if permute is not None: t = t.permute(*permute)
    if transpose: t = t.T
    if reshape is not None: t = t.reshape(reshape)
    if dtype is not None: t = t.to(dtype)
    return t.numpy()


def write_gpt_gguf(model, cfg, vocab_path, output_path, original_path=None):
    state = model.state_dict()
    w = GGUFWriter(output_path, "XTTS v2 GPT")
    gpt = model.gpt
    ma = cfg.model_args if hasattr(cfg, 'model_args') else cfg
    def _g(a, d): return getattr(ma, a, None) if hasattr(ma, a) else d

    n_layers = _g('gpt_layers', 30)
    cond_enc_blocks = len(gpt.conditioning_encoder.attn)
    cond_enc_dim = gpt.conditioning_encoder.dim
    perc_layers = len(gpt.conditioning_perceiver.layers)
    start_tt = _g('gpt_start_text_token', 261) or 261
    stop_tt = _g('gpt_stop_text_token', 0) or 0

    _kv(w,"general.architecture","xtts_v2"); _kv(w,"general.name","XTTS v2 GPT")
    for k,v in [
        ("xtts.gpt.model_dim",model_dim := _g('gpt_n_model_channels',1024)),
        ("xtts.gpt.layers",n_layers),("xtts.gpt.heads",_g('gpt_n_heads',16)),
        ("xtts.gpt.number_text_tokens",_g('gpt_number_text_tokens',6681)),
        ("xtts.gpt.number_mel_codes",_g('gpt_num_audio_tokens',1024)),
        ("xtts.gpt.start_mel_token",_g('gpt_start_audio_token',1024)),
        ("xtts.gpt.stop_mel_token",_g('gpt_stop_audio_token',1025)),
        ("xtts.gpt.max_mel_tokens",_g('gpt_max_audio_tokens',605)),
        ("xtts.gpt.max_text_tokens",_g('gpt_max_text_tokens',402)),
        ("xtts.gpt.max_prompt_tokens",_g('gpt_max_prompt_tokens',70)),
        ("xtts.gpt.start_text_token",start_tt),("xtts.gpt.stop_text_token",stop_tt),
        ("xtts.gpt.code_stride_len",_g('gpt_code_stride_len',1024)),
        ("xtts.gpt.condition_num_latent",32),("xtts.gpt.cond_enc_dim",cond_enc_dim),
        ("xtts.gpt.cond_enc_heads",16),("xtts.gpt.cond_enc_blocks",cond_enc_blocks),
        ("xtts.gpt.perc_layers",perc_layers),
        ("xtts.sampling_rate",_g('input_sample_rate',22050)),
        ("xtts.output_sample_rate",_g('output_sample_rate',24000)),
        ("xtts.output_hop_length",_g('output_hop_length',256)),
    ]: _kv(w,k,v)

    # Tokenizer
    with open(vocab_path) as f: vd=json.load(f)
    tokens=vd['model']['vocab']; merges=vd['model']['merges']
    tl=[]; ts=[]; tt=[]
    for tok,idx in tokens.items():
        tl.append(tok); ts.append(0.0)
        tt.append(5 if (tok.startswith('<') and tok.endswith('>')) or (tok.startswith('[') and tok.endswith(']')) else 4)
    ml=[f"{m.split()[0]} {m.split()[1]}" for m in merges if len(m.split())==2]
    w.add_tokenizer_model("gpt2"); w.add_token_list(tl); w.add_token_scores(ts)
    # Write token types to old field name (audio.cpp reads tokenizer.ggml.token.type)
    w.add_array("tokenizer.ggml.token.type", tt)
    # Write merges to old field name (audio.cpp reads tokenizer.ggml.bpe.merges)
    w.add_array("tokenizer.ggml.bpe.merges", ml)

    # Tensor ordering: copy from original GGUF if available
    if original_path:
        import gguf
        r_orig = gguf.GGUFReader(original_path)
        orig_names = [t.name if isinstance(t.name, str) else t.name.decode() for t in r_orig.tensors]
    else:
        # Fallback: generate ordering matching the original
        orig_names = []
        # Group 1: cond_enc attn (by block, then by tensor type within block)
        for i in range(cond_enc_blocks):
            orig_names.append(f"gpt.cond_enc.attn.{i}.norm.bias")
            orig_names.append(f"gpt.cond_enc.attn.{i}.norm.weight")
            orig_names.append(f"gpt.cond_enc.attn.{i}.proj_out.bias")
            orig_names.append(f"gpt.cond_enc.attn.{i}.proj_out.weight")
            orig_names.append(f"gpt.cond_enc.attn.{i}.qkv.bias")
            orig_names.append(f"gpt.cond_enc.attn.{i}.qkv.weight")
        orig_names.append("gpt.cond_enc.init.bias")
        orig_names.append("gpt.cond_enc.init.weight")
        orig_names.append("gpt.perc.latents")
        for i in range(perc_layers):
            orig_names.append(f"gpt.perc.layers.{i}.0.to_kv.weight")
            orig_names.append(f"gpt.perc.layers.{i}.0.to_out.weight")
            orig_names.append(f"gpt.perc.layers.{i}.0.to_q.weight")
            orig_names.append(f"gpt.perc.layers.{i}.1.0.bias")
            orig_names.append(f"gpt.perc.layers.{i}.1.0.weight")
            orig_names.append(f"gpt.perc.layers.{i}.1.2.bias")
            orig_names.append(f"gpt.perc.layers.{i}.1.2.weight")
        orig_names.append("gpt.perc.norm.gamma")
        orig_names.append("gpt.final_norm.bias")
        orig_names.append("gpt.final_norm.weight")
        for i in sorted(range(n_layers), key=str):
            orig_names.append(f"gpt.gpt.h.{i}.attn.c_attn.bias")
            orig_names.append(f"gpt.gpt.h.{i}.attn.c_attn.weight")
            orig_names.append(f"gpt.gpt.h.{i}.attn.c_proj.bias")
            orig_names.append(f"gpt.gpt.h.{i}.attn.c_proj.weight")
            orig_names.append(f"gpt.gpt.h.{i}.ln_1.bias")
            orig_names.append(f"gpt.gpt.h.{i}.ln_1.weight")
            orig_names.append(f"gpt.gpt.h.{i}.ln_2.bias")
            orig_names.append(f"gpt.gpt.h.{i}.ln_2.weight")
            orig_names.append(f"gpt.gpt.h.{i}.mlp.c_fc.bias")
            orig_names.append(f"gpt.gpt.h.{i}.mlp.c_fc.weight")
            orig_names.append(f"gpt.gpt.h.{i}.mlp.c_proj.bias")
            orig_names.append(f"gpt.gpt.h.{i}.mlp.c_proj.weight")
        orig_names.append("gpt.gpt.ln_f.bias")
        orig_names.append("gpt.gpt.ln_f.weight")
        orig_names.append("gpt.mel_embedding.weight")
        orig_names.append("gpt.mel_head.bias")
        orig_names.append("gpt.mel_head.weight")
        orig_names.append("gpt.mel_pos.weight")
        orig_names.append("gpt.text_embedding.weight")
        orig_names.append("gpt.text_pos.weight")

    # State dict key mapping
    sd_map = {}
    for i in range(cond_enc_blocks):
        pp = f"gpt.conditioning_encoder.attn.{i}"
        sd_map[f"gpt.cond_enc.attn.{i}.norm.bias"] = f"{pp}.norm.bias"
        sd_map[f"gpt.cond_enc.attn.{i}.norm.weight"] = f"{pp}.norm.weight"
        sd_map[f"gpt.cond_enc.attn.{i}.proj_out.bias"] = f"{pp}.proj_out.bias"
        sd_map[f"gpt.cond_enc.attn.{i}.proj_out.weight"] = f"{pp}.proj_out.weight"
        sd_map[f"gpt.cond_enc.attn.{i}.qkv.bias"] = f"{pp}.qkv.bias"
        sd_map[f"gpt.cond_enc.attn.{i}.qkv.weight"] = f"{pp}.qkv.weight"
    sd_map["gpt.cond_enc.init.bias"] = "gpt.conditioning_encoder.init.bias"
    sd_map["gpt.cond_enc.init.weight"] = "gpt.conditioning_encoder.init.weight"
    sd_map["gpt.perc.latents"] = "gpt.conditioning_perceiver.latents"
    for i in range(perc_layers):
        pp = f"gpt.conditioning_perceiver.layers.{i}"
        sd_map[f"gpt.perc.layers.{i}.0.to_kv.weight"] = f"{pp}.0.to_kv.weight"
        sd_map[f"gpt.perc.layers.{i}.0.to_out.weight"] = f"{pp}.0.to_out.weight"
        sd_map[f"gpt.perc.layers.{i}.0.to_q.weight"] = f"{pp}.0.to_q.weight"
        sd_map[f"gpt.perc.layers.{i}.1.0.bias"] = f"{pp}.1.0.bias"
        sd_map[f"gpt.perc.layers.{i}.1.0.weight"] = f"{pp}.1.0.weight"
        sd_map[f"gpt.perc.layers.{i}.1.2.bias"] = f"{pp}.1.2.bias"
        sd_map[f"gpt.perc.layers.{i}.1.2.weight"] = f"{pp}.1.2.weight"
    sd_map["gpt.perc.norm.gamma"] = "gpt.conditioning_perceiver.norm.gamma"
    sd_map["gpt.final_norm.bias"] = "gpt.final_norm.bias"
    sd_map["gpt.final_norm.weight"] = "gpt.final_norm.weight"
    for i in range(n_layers):
        pp = f"gpt.gpt_inference.transformer.h.{i}"
        sd_map[f"gpt.gpt.h.{i}.attn.c_attn.bias"] = f"{pp}.attn.c_attn.bias"
        sd_map[f"gpt.gpt.h.{i}.attn.c_attn.weight"] = f"{pp}.attn.c_attn.weight"
        sd_map[f"gpt.gpt.h.{i}.attn.c_proj.bias"] = f"{pp}.attn.c_proj.bias"
        sd_map[f"gpt.gpt.h.{i}.attn.c_proj.weight"] = f"{pp}.attn.c_proj.weight"
        sd_map[f"gpt.gpt.h.{i}.ln_1.bias"] = f"{pp}.ln_1.bias"
        sd_map[f"gpt.gpt.h.{i}.ln_1.weight"] = f"{pp}.ln_1.weight"
        sd_map[f"gpt.gpt.h.{i}.ln_2.bias"] = f"{pp}.ln_2.bias"
        sd_map[f"gpt.gpt.h.{i}.ln_2.weight"] = f"{pp}.ln_2.weight"
        sd_map[f"gpt.gpt.h.{i}.mlp.c_fc.bias"] = f"{pp}.mlp.c_fc.bias"
        sd_map[f"gpt.gpt.h.{i}.mlp.c_fc.weight"] = f"{pp}.mlp.c_fc.weight"
        sd_map[f"gpt.gpt.h.{i}.mlp.c_proj.bias"] = f"{pp}.mlp.c_proj.bias"
        sd_map[f"gpt.gpt.h.{i}.mlp.c_proj.weight"] = f"{pp}.mlp.c_proj.weight"
    sd_map["gpt.gpt.ln_f.bias"] = "gpt.gpt.ln_f.bias"
    sd_map["gpt.gpt.ln_f.weight"] = "gpt.gpt.ln_f.weight"
    sd_map["gpt.mel_embedding.weight"] = "gpt.mel_embedding.weight"
    sd_map["gpt.mel_head.bias"] = "gpt.mel_head.bias"
    sd_map["gpt.mel_head.weight"] = "gpt.mel_head.weight"
    sd_map["gpt.mel_pos.weight"] = "gpt.mel_pos_embedding.emb.weight"
    sd_map["gpt.text_embedding.weight"] = "gpt.text_embedding.weight"
    sd_map["gpt.text_pos.weight"] = "gpt.text_pos_embedding.emb.weight"

    # Write tensors in original order
    # Get original tensor shapes and dtypes for exact matching
    orig_tensors = {}
    if original_path:
        import gguf
        r_orig = gguf.GGUFReader(original_path)
        for t in r_orig.tensors:
            nm = t.name if isinstance(t.name, str) else t.name.decode()
            orig_tensors[nm] = (t.tensor_type, [int(d) for d in t.shape])

    # GPT-2 Conv1D stores weights as [in, out] instead of nn.Linear's [out, in].
    # ggml_mul_mat needs ne[0]=in_dim, so NumPy array should be [out, in].
    # We transpose these weights to match the convention.
    _GPT2_CONV1D_SUFFIXES = (
        ".attn.c_attn.weight",
        ".attn.c_proj.weight",
        ".mlp.c_fc.weight",
        ".mlp.c_proj.weight",
    )

    for gguf_name in orig_names:
        sd_key = sd_map[gguf_name]
        t = state[sd_key].detach().cpu().float()

        # GPT-2 Conv1D transposition
        if any(gguf_name.endswith(s) for s in _GPT2_CONV1D_SUFFIXES):
            t = t.t().contiguous()

        # Determine target shape from original GGUF
        if gguf_name in orig_tensors:
            orig_dtype, orig_shape = orig_tensors[gguf_name]
        else:
            orig_dtype, orig_shape = None, None

        # GGUF shape convention: ne[0] is fastest-varying dim (reverse of NumPy)
        t_np = t.numpy()
        target_np_shape = tuple(reversed(orig_shape)) if orig_shape else None
        if target_np_shape and t_np.shape != target_np_shape:
            if len(target_np_shape) == len(t_np.shape):
                from itertools import permutations
                found = False
                for perm in permutations(range(len(t_np.shape))):
                    if t_np.transpose(*perm).shape == target_np_shape:
                        t_np = t_np.transpose(*perm)
                        found = True
                        break
                if not found:
                    t_np = t_np.reshape(target_np_shape)
            else:
                t_np = t_np.reshape(target_np_shape)
        t = t_np

        # Determine dtype
        if orig_dtype:
            if orig_dtype == GGMLQuantizationType.F16:
                t = np.float16(t)
            elif orig_dtype == GGMLQuantizationType.F32:
                t = np.float32(t)

        w.add_tensor(gguf_name, t)

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"  Written: {output_path} ({len(orig_names)} tensors)")


def write_vocoder_gguf(model, cfg, output_path, original_path=None):
    state = model.state_dict()
    w = GGUFWriter(output_path, "XTTS v2 Vocoder")
    wd = model.hifigan_decoder.waveform_decoder
    ma = cfg.model_args if hasattr(cfg, 'model_args') else cfg

    _kv(w,"general.architecture","xtts_v2_vocoder"); _kv(w,"general.name","XTTS v2 Vocoder")
    for k,v in [
        ("xtts.vocoder.decoder_input_dim",getattr(ma,'decoder_input_dim',1024)),
        ("xtts.vocoder.d_vector_dim",getattr(ma,'d_vector_dim',512)),
        ("xtts.vocoder.num_upsamples",wd.num_upsamples),
        ("xtts.vocoder.num_kernels",wd.num_kernels),
        ("xtts.vocoder.initial_channel",512),
        ("xtts.vocoder.speaker_embedding_dim",512),
        ("xtts.vocoder.mel_bins",80),("xtts.vocoder.speaker_sr",16000),
        ("xtts.sampling_rate",22050),("xtts.output_sample_rate",24000),
        ("xtts.output_hop_length",256),("xtts.code_stride_len",getattr(ma,'gpt_code_stride_len',1024)),
        ("xtts.vocoder.upsample_rates",[u.kernel_size[0] for u in wd.ups]),
        ("xtts.vocoder.upsample_kernel_sizes",[u.kernel_size[0] for u in wd.ups]),
        ("xtts.vocoder.resblock_kernel_sizes",[3,7,11]),
        ("xtts.vocoder.resblock_dilation_sizes",[1,3,5,1,3,5,1,3,5]),
        ("xtts.vocoder.cond_in_each_up_layer",True),
        ("xtts.vocoder.speaker_encoder_type","SE-ResNet"),
        ("xtts.vocoder.se_layers",[3,4,6,3]),
        ("xtts.vocoder.se_filters",[32,64,128,256]),
        ("xtts.vocoder.se_proj_dim",2048),("xtts.vocoder.se_asp_mu",False),
    ]: _kv(w,k,v)

    # Tensor ordering: copy from original GGUF
    if original_path:
        import gguf
        r_orig = gguf.GGUFReader(original_path)
        orig_names = [t.name if isinstance(t.name, str) else t.name.decode() for t in r_orig.tensors]
        orig_tensors = {}
        for t in r_orig.tensors:
            nm = t.name if isinstance(t.name, str) else t.name.decode()
            orig_tensors[nm] = (t.tensor_type, [int(d) for d in t.shape])
    else:
        raise ValueError("Original vocoder GGUF path required for correct tensor ordering")

    # State dict key mapping for vocoder
    sd_map = {}
    sd_map["mel_stats"] = "mel_stats"
    sd_map["hifigan_decoder.se.conv1.weight"] = "hifigan_decoder.speaker_encoder.conv1.weight"
    sd_map["hifigan_decoder.se.conv1.bias"] = "hifigan_decoder.speaker_encoder.conv1.bias"
    for s,k in [("weight","weight"),("bias","bias"),("rm","running_mean"),("rv","running_var")]:
        sd_map[f"hifigan_decoder.se.bn1.{s}"] = f"hifigan_decoder.speaker_encoder.bn1.{k}"

    # SE-ResNet stages [3,4,6,3] (1-indexed)
    for stage in range(1, 5):
        sb = [3,4,6,3][stage-1]
        for block in range(sb):
            pp = f"hifigan_decoder.speaker_encoder.layer{stage}.{block}"
            gp = f"hifigan_decoder.se.layer{stage}.{block}"
            for c in [1,2]:
                sd_map[f"{gp}.conv{c}.weight"] = f"{pp}.conv{c}.weight"
            for bn in ["bn1","bn2"]:
                for s,k in [("weight","weight"),("bias","bias"),("rm","running_mean"),("rv","running_var")]:
                    sd_map[f"{gp}.{bn}.{s}"] = f"{pp}.{bn}.{k}"
            sd_map[f"{gp}.se.fc.0.weight"] = f"{pp}.se.fc.0.weight"
            sd_map[f"{gp}.se.fc.0.bias"] = f"{pp}.se.fc.0.bias"
            sd_map[f"{gp}.se.fc.2.weight"] = f"{pp}.se.fc.2.weight"
            sd_map[f"{gp}.se.fc.2.bias"] = f"{pp}.se.fc.2.bias"
            if block==0 and stage>1:
                sd_map[f"{gp}.downsample.0.weight"] = f"{pp}.downsample.0.weight"
                for s,k in [("weight","weight"),("bias","bias"),("rm","running_mean"),("rv","running_var")]:
                    sd_map[f"{gp}.downsample.1.{s}"] = f"{pp}.downsample.1.{k}"

    # ASP attention
    sd_map["hifigan_decoder.se.attention.0.weight"] = "hifigan_decoder.speaker_encoder.attention.0.weight"
    sd_map["hifigan_decoder.se.attention.0.bias"] = "hifigan_decoder.speaker_encoder.attention.0.bias"
    sd_map["hifigan_decoder.se.attention.2.weight"] = "hifigan_decoder.speaker_encoder.attention.2.weight"
    sd_map["hifigan_decoder.se.attention.2.bias"] = "hifigan_decoder.speaker_encoder.attention.2.bias"
    sd_map["hifigan_decoder.se.attention.2.rm"] = "hifigan_decoder.speaker_encoder.attention.2.running_mean"
    sd_map["hifigan_decoder.se.attention.2.rv"] = "hifigan_decoder.speaker_encoder.attention.2.running_var"
    sd_map["hifigan_decoder.se.attention.3.weight"] = "hifigan_decoder.speaker_encoder.attention.3.weight"
    sd_map["hifigan_decoder.se.attention.3.bias"] = "hifigan_decoder.speaker_encoder.attention.3.bias"

    # fc
    sd_map["hifigan_decoder.se.fc.weight"] = "hifigan_decoder.speaker_encoder.fc.weight"
    sd_map["hifigan_decoder.se.fc.bias"] = "hifigan_decoder.speaker_encoder.fc.bias"

    # HiFi-GAN
    sd_map["hifigan_decoder.dec.conv_pre.weight"] = "hifigan_decoder.waveform_decoder.conv_pre.weight"
    sd_map["hifigan_decoder.dec.conv_pre.bias"] = "hifigan_decoder.waveform_decoder.conv_pre.bias"
    sd_map["hifigan_decoder.dec.cond_layer.weight"] = "hifigan_decoder.waveform_decoder.cond_layer.weight"
    sd_map["hifigan_decoder.dec.cond_layer.bias"] = "hifigan_decoder.waveform_decoder.cond_layer.bias"
    for i in range(4):
        sd_map[f"hifigan_decoder.dec.conds.{i}.weight"] = f"hifigan_decoder.waveform_decoder.conds.{i}.weight"
        sd_map[f"hifigan_decoder.dec.conds.{i}.bias"] = f"hifigan_decoder.waveform_decoder.conds.{i}.bias"
    for i in range(4):
        sd_map[f"hifigan_decoder.dec.ups.{i}"] = f"hifigan_decoder.waveform_decoder.ups.{i}"
        sd_map[f"hifigan_decoder.dec.ups.{i}.bias"] = f"hifigan_decoder.waveform_decoder.ups.{i}.bias"
    for i in range(12):
        for c in range(2):
            for j in range(3):
                sd_map[f"hifigan_decoder.dec.resb.{i}.c{c+1}.{j}"] = f"hifigan_decoder.waveform_decoder.resblocks.{i}.convs{c+1}.{j}"
                sd_map[f"hifigan_decoder.dec.resb.{i}.c{c+1}.{j}.bias"] = f"hifigan_decoder.waveform_decoder.resblocks.{i}.convs{c+1}.{j}.bias"
    sd_map["hifigan_decoder.dec.conv_post.weight"] = "hifigan_decoder.waveform_decoder.conv_post.weight"
    if "hifigan_decoder.waveform_decoder.conv_post.bias" in state:
        sd_map["hifigan_decoder.dec.conv_post.bias"] = "hifigan_decoder.waveform_decoder.conv_post.bias"

    # Write tensors in original order
    for gguf_name in orig_names:
        sd_key = sd_map[gguf_name]
        orig_dtype, orig_shape = orig_tensors[gguf_name]

        # Get tensor data
        if 'parametrizations' in sd_key:
            # Should not happen - we use the base key
            pass

        # Check if this is a WeightNorm tensor
        base_sd = sd_key.rsplit('.parametrizations', 1)[0] if '.parametrizations' in sd_key else sd_key
        wn_key = f"{sd_key}.parametrizations.weight.original0"
        if wn_key in state:
            t = _wn(state, sd_key)
        else:
            t = state[sd_key].detach().cpu().float().numpy()

        # GGUF shape convention: ne[0] is fastest-varying dim (reverse of NumPy)
        target_np_shape = tuple(reversed(orig_shape))
        if t.shape != target_np_shape:
            if len(target_np_shape) == len(t.shape):
                from itertools import permutations
                found = False
                for perm in permutations(range(len(t.shape))):
                    if np.transpose(t, perm).shape == target_np_shape:
                        t = np.ascontiguousarray(np.transpose(t, perm))
                        found = True
                        break
                if not found:
                    t = t.reshape(target_np_shape)
            else:
                t = t.reshape(target_np_shape)

        # Apply dtype
        if orig_dtype == GGMLQuantizationType.F16:
            t = np.float16(t)
        elif orig_dtype == GGMLQuantizationType.F32:
            t = np.float32(t)
        # else use whatever dtype the array already has

        w.add_tensor(gguf_name, t)

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"  Written: {output_path} ({len(orig_names)} tensors)")


def main():
    if len(sys.argv) < 6:
        print("Usage: python convert_xtts_v2_to_gguf.py <model.pth> <config.json> <vocab.json> <output_dir> <original_gguf_dir>")
        print("  original_gguf_dir: path to original GGUF files for exact tensor ordering")
        sys.exit(1)

    cp = sys.argv[1]; cfp = sys.argv[2]; vp = sys.argv[3]; od = sys.argv[4]; orig_dir = sys.argv[5]
    os.makedirs(od, exist_ok=True)

    print("Loading XTTS v2 model...")
    model, cfg = load_xtts_model(cp, cfp, vp)

    orig_gpt = os.path.join(orig_dir, "xtts-v2-gpt.gguf")
    orig_voc = os.path.join(orig_dir, "xtts-v2-vocoder-f16.gguf")

    print("Converting GPT...")
    write_gpt_gguf(model, cfg, vp, os.path.join(od, "xtts-v2-gpt.gguf"), orig_gpt)

    print("Converting Vocoder...")
    write_vocoder_gguf(model, cfg, os.path.join(od, "xtts-v2-vocoder-f16.gguf"), orig_voc)

    # mel_norms
    ms = model.mel_stats.detach().cpu().numpy()
    with open(os.path.join(od, "mel_norms.bin"), "wb") as f:
        f.write(ms.tobytes())
    print(f"  Written: {os.path.join(od, 'mel_norms.bin')} (80 floats)")

    print("Done.")

if __name__ == "__main__":
    main()
