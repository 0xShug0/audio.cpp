#!/usr/bin/env python3
"""Build the fixed-length RKNN generator step-graph for the OmniVoice family.

The runtime side (``src/models/omnivoice/rknn_generator.cpp``, session option
``omnivoice.exec_plan`` ``{"generator": {"impl": "rknn", ...}}``) runs one compiled RKNN step in
place of the ggml ``Qwen3DiffusionForwardGraph``, inside the same diffusion-LM unmasking loop.
Nothing in this repository produces that graph, so this script is its only producer: it
recompiles the checkpoint's own generator (the Qwen3 backbone + audio embeddings/heads inside
``model.safetensors`` of ``k2-fsa/OmniVoice``) for one fixed padded sequence length and for a
RKNPU2 target.

The step is the whole bidirectional 28-layer forward, exactly as the reference port runs it
(``omnivoice/models/omnivoice.py::_generate_iterative`` and ``src/models/omnivoice/generator.cpp``):
batch 2 is the classifier-free-guidance pair ``[cond, uncond]``, there is no timestep input (the
diffusion "time" lives only in the host schedule), and every step recomputes the full sequence
(no KV cache). Six inputs:

- ``input_ids [2,8,S]`` int64: 8 codebook rows; row 0 carries the text tokens (the reference
  repeats them across codebooks, the mask makes only row 0 count at text positions).
- ``audio_mask [2,S]`` fp32: 1.0 at audio frames, 0.0 at text.
- ``attn_mask [2,1,S,S]`` fp32 additive: 0.0 keep, -10000 mask. One block-diagonal row pair:
  cond attends to ``[0,c_len)``, uncond to ``[0,u_len)``, pads attend to themselves (the
  diagonal is always kept, so no softmax row is all-masked).
- ``rope_cos [2,S,head_dim/2]`` / ``rope_sin`` fp32: the per-row RoPE tables. The reference uses
  ``arange(S)`` per row; the packed-CFG C++ layout passes its own flat positions. The angles
  (``pos * inv_freq`` mod 2*pi) and their cos/sin are computed by the HOST in float64/float32 and
  enter as inputs: computing ``Cos``/``Sin`` of raw angles up to ``S`` radians inside the graph
  quantizes the ARGUMENT to fp16 on the NPU (ulp ~0.5 rad at 1e3), which destroys attention -
  invisible on the fp32-emulating simulator, and the qwen3 vocoder export already sets the
  host-tables-as-inputs precedent.
- ``guidance [1]`` fp32: the CFG scale. The graph returns the guided logits the ggml graph's
  ``build_guided_logits`` returns, ``cond + guidance * (cond - uncond)`` on RAW logits (the
  C++ port's scoring, not the HF reference's log-probability-space combine), and the host keeps
  the log_softmax/argmax/top-k unmasking, so this is a drop-in for the ggml graph's ``logits``
  output.

One output: ``logits [1,8,S,1025]`` fp32, the cond row's guided logits (the uncond row is a
placeholder inside the combine). The sidecar crops the target block
``[c_len-u_len, c_len)`` and hands it to the existing scoring path.

RKNN has no dynamic shapes, so the runtime compiles a ladder of lengths and pads a shorter
utterance up to the smallest engine that fits: ``input_ids`` filled with ``audio_mask_id`` in the
pad, ``audio_mask`` 0 there, ``attn_mask`` diagonal-only there, outputs cropped back. That
convention is what the padding check measures on the host: a padded run at ``S`` against an exact
run at ``c_len``, per length, before any of it is paid for on a board (``--skip-check`` skips
it). ``--verify-sim`` goes one step further and runs the fp16 graph on the PC simulator against
the fp32 ONNX - the engine's own accuracy, measured without a board.

Requirements (host, x86_64, dedicated venv -- the RKNN toolkit pins its own numpy/onnx):
  pip install onnx onnxruntime numpy safetensors torch --index-url https://download.pytorch.org/whl/cpu
  pip install rknn-toolkit2          # PC-side converter, x86_64 only
On Python 3.12 the 2.3.2 converter additionally needs ``setuptools<81`` and an ``onnx/mapping.py``
shim inside the venv's onnx package; see docs/build/rockchip-npu.md.

Usage:
  python3 tools/rknn/prepare_omnivoice_rknn_generator.py --model-dir models/OmniVoice \
      --seq-len 256 --seq-len 512 --output-dir models/OmniVoice-RKNN
  python3 tools/rknn/prepare_omnivoice_rknn_generator.py --model-dir models/OmniVoice \
      --seq-len 64 --check-c-len 40 --check-c-len 63 --verify-sim
  python3 tools/rknn/prepare_omnivoice_rknn_generator.py --model-dir models/OmniVoice \
      --seq-len 256 --skip-rknn          # ONNX export + padding gate only
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

INPUT_NAMES = ["input_ids", "audio_mask", "attn_mask", "rope_cos", "rope_sin", "guidance"]
MASK_VALUE = -10000.0  # additive-mask "off" value, fp16-safe (exp(0 - 10000 - max) == 0)


class Qwen3Layer(torch.nn.Module):
    """One Qwen3 (qk-norm variant) decoder layer, bidirectional under a supplied additive mask.

    ``cos``/``sin`` are the per-row RoPE tables, ``[B,S,head_dim]``, doubled from the
    ``rope_cos``/``rope_sin`` graph inputs by the owning module.
    """

    def __init__(self, sd: dict, i: int, cfg: dict):
        super().__init__()
        lc = cfg["llm_config"]
        h, hd = lc["hidden_size"], lc["head_dim"]
        self.n_heads, self.n_kv, self.head_dim = lc["num_attention_heads"], lc["num_key_value_heads"], hd
        self.eps = lc["rms_norm_eps"]
        self.kv_rep = self.n_heads // self.n_kv
        p = f"llm.layers.{i}."
        self.register_buffer(
            "kv_idx",
            torch.arange(self.n_heads) // self.kv_rep if self.kv_rep > 1 else torch.arange(self.n_heads),
            persistent=False,
        )
        self.input_norm = torch.nn.Parameter(sd[p + "input_layernorm.weight"], requires_grad=False)
        self.post_norm = torch.nn.Parameter(sd[p + "post_attention_layernorm.weight"], requires_grad=False)
        self.q_norm = torch.nn.Parameter(sd[p + "self_attn.q_norm.weight"], requires_grad=False)
        self.k_norm = torch.nn.Parameter(sd[p + "self_attn.k_norm.weight"], requires_grad=False)
        for name, key in [("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"),
                          ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj"),
                          ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"),
                          ("down", "mlp.down_proj")]:
            lin = torch.nn.Linear(h, h, bias=False)  # shapes come from the assigned weights
            lin.weight = torch.nn.Parameter(sd[f"{p}{key}.weight"], requires_grad=False)
            setattr(self, name, lin)

    def _rms(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    def forward(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attn_mask: torch.Tensor) -> torch.Tensor:
        b, s, _ = h.shape
        x = self._rms(h, self.input_norm)
        q = self.q(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, s, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, s, self.n_kv, self.head_dim).transpose(1, 2)
        q = self._rms(q, self.q_norm)
        k = self._rms(k, self.k_norm)
        half = self.head_dim // 2
        rot_q = torch.cat([-q[..., half:], q[..., :half]], dim=-1)
        rot_k = torch.cat([-k[..., half:], k[..., :half]], dim=-1)
        q = q * cos.unsqueeze(1) + rot_q * sin.unsqueeze(1)
        k = k * cos.unsqueeze(1) + rot_k * sin.unsqueeze(1)
        if self.kv_rep > 1:
            # GQA by a constant head-selection Gather: expand+reshape exports a
            # Shape/ConstantOfShape ladder, 5D matmul dies in the converter's swap passes -
            # a Gather on the head axis is fully static (same toolchain lesson as the qwen3
            # vocoder's kv-selection matmuls).
            k = torch.index_select(k, 1, self.kv_idx)
            v = torch.index_select(v, 1, self.kv_idx)
        attn = torch.softmax(
            q @ k.transpose(2, 3) * (1.0 / math.sqrt(self.head_dim)) + attn_mask.to(q.dtype),
            dim=-1,
        )
        h = h + self.o((attn @ v).transpose(1, 2).reshape(b, s, self.n_heads * self.head_dim))
        x = self._rms(h, self.post_norm)
        return h + self.down(F.silu(self.gate(x)) * self.up(x))


class Qwen3StepForward(torch.nn.Module):
    """The OmniVoice generator forward, hand-rolled from the checkpoint weights.

    Mirrors ``OmniVoice.forward`` (embedding mix + Qwen3 backbone + audio heads, bidirectional
    attention under the caller-supplied additive mask, RoPE from the cos/sin table inputs) followed
    by
    the C++ port's CFG combine (``build_guided_logits``: raw-logit
    ``cond + g*(cond - uncond)``, guidance as a graph input). Written against the tensors in
    ``model.safetensors`` directly so the export does not depend on a transformers version that
    registers the ``omnivoice`` architecture (none does).
    """

    def __init__(self, sd: dict[str, torch.Tensor], cfg: dict):
        super().__init__()
        lc = cfg["llm_config"]
        self.hidden = lc["hidden_size"]
        self.head_dim = lc["head_dim"]
        self.eps = lc["rms_norm_eps"]
        self.n_cb = cfg["num_audio_codebook"]
        self.avocab = cfg["audio_vocab_size"]
        self.mask_id = cfg["audio_mask_id"]

        self.embed = torch.nn.Embedding(lc["vocab_size"], self.hidden)
        self.embed.weight = torch.nn.Parameter(sd["llm.embed_tokens.weight"], requires_grad=False)
        self.layers = torch.nn.ModuleList(
            [Qwen3Layer(sd, i, cfg) for i in range(lc["num_hidden_layers"])]
        )
        self.final_norm = torch.nn.Parameter(sd["llm.norm.weight"], requires_grad=False)
        self.audio_embed = torch.nn.Embedding(self.n_cb * self.avocab, self.hidden)
        self.audio_embed.weight = torch.nn.Parameter(sd["audio_embeddings.weight"], requires_grad=False)
        self.audio_heads = torch.nn.Linear(self.hidden, self.n_cb * self.avocab, bias=False)
        self.audio_heads.weight = torch.nn.Parameter(sd["audio_heads.weight"], requires_grad=False)
        offsets = sd.get("codebook_layer_offsets")
        if offsets is None:
            offsets = torch.arange(self.n_cb) * self.avocab
        self.register_buffer("cb_offsets", offsets.view(1, -1, 1).long(), persistent=False)

        self.rope_theta = lc["rope_parameters"]["rope_theta"]

    def _rms(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    def forward(self, input_ids: torch.Tensor, audio_mask: torch.Tensor,
                attn_mask: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor,
                guidance: torch.Tensor) -> list[torch.Tensor]:
        b, _, s = input_ids.shape
        ids = input_ids.long()
        m = audio_mask.unsqueeze(1).long()  # [B,1,S] -> broadcast over codebooks
        text = self.embed(ids[:, 0, :])
        audio = self.audio_embed((ids * m) + self.cb_offsets).sum(dim=1)
        am = audio_mask.unsqueeze(-1).to(audio.dtype)
        h = am * audio + (1.0 - am) * text

        # The RoPE tables arrive precomputed (host, float64 angles mod 2*pi -> float32 cos/sin);
        # only the doubling to head_dim and the cast to h's dtype stay in the graph.
        cos = torch.cat([rope_cos, rope_cos], dim=-1).to(h.dtype)
        sin = torch.cat([rope_sin, rope_sin], dim=-1).to(h.dtype)

        for layer in self.layers:
            h = layer(h, cos, sin, attn_mask)

        h = self._rms(h, self.final_norm)
        logits = self.audio_heads(h).view(b, s, self.n_cb, self.avocab).permute(0, 2, 1, 3)
        # build_guided_logits, raw-logit space, cond row only (the C++ port's scoring stays on
        # the host: log_softmax/argmax over these guided logits).
        cond, uncond = logits[:1], logits[b // 2 : b // 2 + 1]
        g = guidance.view(1, 1, 1, 1)
        return [cond + g * (cond - uncond)]


class Qwen3FrontShard(torch.nn.Module):
    """The first ``k`` blocks of ``Qwen3StepForward``; output ``hidden [2,S,hidden]``.

    The ``--front-k`` engine: the embedding mix and the RoPE doubling are the full module's, the
    CFG tail (final RMS norm + audio heads + the guided combine) moves to the fp32 host. Measured
    reason (PC simulator, real step-0 feed): the full graph's fp16 audio heads put the uncond
    logits at 26.9 dB SNR, past the decoding knife-edge, while the 28-layer fp16 backbone leaves
    the hidden state at 67 dB - the heads are the whole noise, not the backbone.
    """

    def __init__(self, full: Qwen3StepForward, k: int):
        super().__init__()
        self.full = full
        self.k = k

    def forward(self, input_ids: torch.Tensor, audio_mask: torch.Tensor,
                attn_mask: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor):
        f = self.full
        ids = input_ids.long()
        m = audio_mask.unsqueeze(1).long()
        text = f.embed(ids[:, 0, :])
        audio = f.audio_embed((ids * m) + f.cb_offsets).sum(dim=1)
        am = audio_mask.unsqueeze(-1).to(audio.dtype)
        h = am * audio + (1.0 - am) * text
        cos = torch.cat([rope_cos, rope_cos], dim=-1).to(h.dtype)
        sin = torch.cat([rope_sin, rope_sin], dim=-1).to(h.dtype)
        for layer in f.layers[:self.k]:
            h = layer(h, cos, sin, attn_mask)
        return h


def load_generator(model_dir: Path) -> tuple[dict, dict]:
    sd = load_file(str(model_dir / "model.safetensors"))
    import json
    cfg = json.loads((model_dir / "config.json").read_text())
    return sd, cfg


def build_model(model_dir: Path, fuse: bool = False,
                sd_cfg: tuple[dict, dict] | None = None) -> Qwen3StepForward:
    sd, cfg = sd_cfg if sd_cfg is not None else load_generator(model_dir)
    model = Qwen3StepForward(sd, cfg).eval().float()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def export_onnx(model_dir: Path, seq_len: int, guidance: float, out: Path,
                sd_cfg: tuple[dict, dict] | None = None, verbose: bool = True,
                dtype: torch.dtype = torch.float32, front_k: int = 0) -> Path:
    model: torch.nn.Module = build_model(model_dir, sd_cfg=sd_cfg)
    if front_k:
        model = Qwen3FrontShard(model, front_k)
    if dtype is not torch.float32:
        # fp16 weights for the RKNN stage: the 28-layer fp32 initializers (2.45 GB) make the
        # converter's constant-fold/fuse copies OOM-abort (measured: 8 layers peak 9.3 GB and
        # build clean, 28 layers SIGABRT in fold_constant). The graph-input dtype stays fp32 for
        # everything except nothing: inputs pass through Cast, so the C++ sidecar keeps feeding
        # float arrays.
        model = model.half()
    rng = np.random.default_rng(7)
    example = make_inputs(seq_len, max(1, seq_len // 2), max(1, seq_len // 2), 1025, rng,
                          guidance, model.full.head_dim if front_k else model.head_dim,
                          model.full.rope_theta if front_k else model.rope_theta)
    inputs = [torch.from_numpy(a) for a in (example[:5] if front_k else example)]
    out.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"export {out} (seq_len {seq_len}, weights {str(dtype).replace('torch.','')})")
    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(inputs),
            str(out),
            input_names=INPUT_NAMES[:5] if front_k else INPUT_NAMES,
            output_names=["hidden"] if front_k else ["logits"],
            opset_version=17,
            # Fold at export: with do_constant_folding=False the converter's own FoldConstant
            # pass dies at ~3161/3194 nodes on the full 28-layer graph (SIGABRT; 8 layers
            # build clean, so it is the fold count, not an op).
            do_constant_folding=True,
        )
    return out


def input_shapes(seq_len: int, head_dim: int = 128) -> list[list[int]]:
    return [[2, 8, seq_len], [2, seq_len], [2, 1, seq_len, seq_len],
            [2, seq_len, head_dim // 2], [2, seq_len, head_dim // 2], [1]]


def rope_tables(seq_len: int, head_dim: int, rope_theta: float) -> tuple[np.ndarray, np.ndarray]:
    """The ``[2,seq_len,head_dim/2]`` fp32 cos/sin RoPE tables for ``arange(seq_len)`` per row.

    Angles are formed in float64 and reduced mod 2*pi before the trig: the graph must never see
    raw positions (fp16 Cos/Sin of a ~1e3-rad argument is what makes the board render noise).
    """
    inv = rope_theta ** (
        -np.arange(0, head_dim, 2, dtype=np.float64) / head_dim
    )  # [hd/2]
    ang = (np.arange(seq_len, dtype=np.float64)[:, None] * inv[None, :]) % (2.0 * np.pi)
    cos = np.tile(np.cos(ang).astype(np.float32), (2, 1, 1))
    sin = np.tile(np.sin(ang).astype(np.float32), (2, 1, 1))
    return cos, sin


def make_inputs(seq_len: int, c_len: int, u_len: int, avocab: int, rng: np.random.Generator,
                guidance: float = 2.0, head_dim: int = 128, rope_theta: float = 1000000.0):
    """A reference-shaped input set for a cond block of ``c_len`` and uncond of ``u_len``.

    Cond row: text tokens on ``[0, c_len-u_len)``, audio grid on ``[c_len-u_len, c_len)`` (the
    no-reference-audio layout). Uncond row: audio on ``[0, u_len)``, pads after. RoPE tables are
    ``arange`` per row (the HF reference layout); pads attend to themselves only.
    """
    text_len = c_len - u_len
    ids = np.zeros((2, 8, seq_len), dtype=np.int64)
    ids[0, :, text_len:c_len] = rng.integers(0, avocab, size=(8, u_len))
    ids[0, :, :text_len] = rng.integers(0, 100000, size=(8, text_len))
    ids[1, :, :u_len] = rng.integers(0, avocab, size=(8, u_len))
    ids[0, :, c_len:] = 1024
    ids[1, :, u_len:] = 1024
    amask = np.zeros((2, seq_len), dtype=np.float32)
    amask[0, text_len:c_len] = 1.0
    amask[1, :u_len] = 1.0
    attn = np.full((2, 1, seq_len, seq_len), MASK_VALUE, dtype=np.float32)
    attn[0, 0, :c_len, :c_len] = 0.0
    attn[1, 0, :u_len, :u_len] = 0.0
    np.fill_diagonal(attn[0, 0][c_len:, c_len:], 0.0)
    np.fill_diagonal(attn[1, 0][u_len:, u_len:], 0.0)
    cos, sin = rope_tables(seq_len, head_dim, rope_theta)
    return [ids, amask, attn, cos, sin, np.array([guidance], dtype=np.float32)]


def snr_db(reference: np.ndarray, other: np.ndarray) -> float:
    power = float((reference.astype(np.float64) ** 2).sum())
    err = float(((reference.astype(np.float64) - other.astype(np.float64)) ** 2).sum())
    return 10.0 * np.log10(max(power, 1e-30) / max(err, 1e-30))


def crop_valid(outs: list[np.ndarray], c_len: int, u_len: int) -> list[np.ndarray]:
    """Crop a run's output to the cond target block ``[c_len-u_len, c_len)`` the host reads."""
    return [o[:, :, c_len - u_len : c_len, :] if o.ndim == 4 else o[:, c_len - u_len : c_len]
            for o in outs]


def check_padding(model_dir: Path, onnx_path: Path, seq_len: int, cases: list[tuple[int, int]],
                  guidance: float, seed: int, front_k: int = 0) -> None:
    """Padded run at ``seq_len`` vs exact run at ``c_len``: the padding gate (onnxruntime fp32).

    Compared on the cropped valid region only: the pad's outputs are placeholders by
    construction.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    sd_cfg = load_generator(model_dir)
    hd = sd_cfg[1]["llm_config"]["head_dim"]
    theta = sd_cfg[1]["llm_config"]["rope_parameters"]["rope_theta"]
    rng = np.random.default_rng(seed)
    print(f"check {seq_len}: padded run vs exact run (onnxruntime CPU, fp32)")
    for c_len, u_len in cases:
        if c_len > seq_len:
            continue
        base = make_inputs(c_len, c_len, u_len, 1025, rng, guidance, hd, theta)  # the exact run's inputs
        padded = pad_to(base, seq_len, hd, theta)
        # The exact run needs its own graph (RKNN graphs are fixed-shape; the ONNX too).
        exact_path = export_onnx(model_dir, c_len, guidance,
                                 Path("/tmp") / f"ov_exact_seq{c_len}.onnx", sd_cfg, verbose=False,
                                 front_k=front_k)
        exact_sess = (
            sess
            if c_len == seq_len
            else ort.InferenceSession(str(exact_path), opts, providers=["CPUExecutionProvider"])
        )
        feed_p = dict(zip(INPUT_NAMES, padded[:5] if front_k else padded))
        feed_e = dict(zip(INPUT_NAMES, base[:5] if front_k else base))
        out_p = crop_valid(sess.run(None, feed_p), c_len, u_len)[0]
        out_e = crop_valid(exact_sess.run(None, feed_e), c_len, u_len)[0]
        tail = min(32, u_len)
        tail_of = lambda o: o[:, :, -tail:] if o.ndim == 4 else o[:, -tail:]
        print(
            f"  c_len={c_len:4d} u_len={u_len:4d} pad={seq_len - c_len:4d} "
            f"snr={snr_db(out_e, out_p):6.1f} dB tail_snr={snr_db(tail_of(out_e), tail_of(out_p)):6.1f} dB "
            f"max_abs={np.abs(out_e - out_p).max():.4f}"
        )


def pad_to(inputs: list[np.ndarray], seq_len: int, head_dim: int = 128,
           rope_theta: float = 1000000.0) -> list[np.ndarray]:
    ids, amask, attn, _, _, g = inputs
    n = seq_len - ids.shape[-1]
    if n < 0:
        raise SystemExit(f"inputs longer ({ids.shape[-1]}) than engine ({seq_len})")
    if n == 0:
        return inputs
    ids = np.concatenate([ids, np.full((2, 8, n), 1024, dtype=ids.dtype)], axis=-1)
    amask = np.concatenate([amask, np.zeros((2, n), dtype=np.float32)], axis=-1)
    attn = np.pad(attn, [(0, 0), (0, 0), (0, n), (0, n)], constant_values=MASK_VALUE)
    pad = np.arange(seq_len - n, seq_len)
    attn[0, 0, pad, pad] = 0.0
    attn[1, 0, pad, pad] = 0.0
    # The tables are pure arange functions, so padding is just recomputing them at the new length.
    cos, sin = rope_tables(seq_len, head_dim, rope_theta)
    return [ids, amask, attn, cos, sin, g]


def verify_fp16(model_dir: Path, onnx_path: Path, rknn_onnx_path: Path, seq_len: int,
                c_len: int, u_len: int, guidance: float, target: str, opt_level: int,
                seed: int, front_k: int = 0) -> None:
    """The fp16 cost of the compiled graph, measured on the PC simulator.

    The fp32 ONNX is the reference; the engine is built from the fp16-weights export, which is
    what the board runs.
    """
    import onnxruntime as ort
    from rknn.api import RKNN

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rknn_build_log import rknn_build_gate

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    lc = load_generator(model_dir)[1]["llm_config"]
    rng = np.random.default_rng(seed)
    inputs = make_inputs(seq_len, c_len, u_len, 1025, rng, guidance,
                         lc["head_dim"], lc["rope_parameters"]["rope_theta"])
    feed = inputs[:5] if front_k else inputs
    reference = sess.run(None, dict(zip(INPUT_NAMES, feed)))[0]
    del sess

    rknn = RKNN(verbose=False)
    rknn.config(target_platform=target, optimization_level=opt_level)
    if rknn.load_onnx(model=str(rknn_onnx_path)) != 0:
        raise SystemExit("load_onnx failed")
    with rknn_build_gate(f"verify {rknn_onnx_path.name}", rknn_onnx_path.with_suffix(".verify.build.log")):
        if rknn.build(do_quantization=False) != 0:
            raise SystemExit("build failed")
    if rknn.init_runtime() != 0:
        raise SystemExit("init_runtime (PC simulator) failed")
    # The 4-D attention mask's declared layout is converter-decided (the 28-layer shard graph
    # comes out NHWC-flagged, and seq64 flags the full graph NHWC too, while the seq768 full
    # graph stays NCHW), and the simulator default is 'nhwc'. Declare the fed layout as NCHW
    # instead: the simulator relabels/transposes to each input's native layout, so the same
    # raw feed works for every graph. C=1, so the relabel moves no data.
    got = np.asarray(rknn.inference(inputs=feed, data_format='nchw')[0])
    rknn.release()
    got = got.reshape(reference.shape)
    tail = min(32, u_len)
    tail_of = lambda o: o[:, :, -tail:] if o.ndim == 4 else o[:, -tail:]
    print(
        f"verify fp16 (seq {seq_len}, PC simulator): snr={snr_db(reference, got):6.1f} dB "
        f"tail_snr={snr_db(tail_of(reference), tail_of(got)):6.1f} dB "
        f"max_abs={np.abs(reference - got).max():.4f} "
        f"argmax_match={float((reference.argmax(-1) == got.argmax(-1)).mean()):.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="k2-fsa/OmniVoice HF snapshot dir (model.safetensors + config.json)")
    parser.add_argument("--seq-len", type=int, action="append", default=None,
                        help="padded sequence length to compile (repeatable); the runtime pads "
                             "up to the smallest fit")
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--opt-level", type=int, default=0,
                        help="rknn optimization_level; 0 for this graph: the full 28-layer "
                             "generator builds clean at 0 (measured, 13 min)")
    parser.add_argument("--output-dir", type=Path, default=Path("models/OmniVoice-RKNN"))
    parser.add_argument("--guidance", type=float, default=2.0,
                        help="CFG scale used by the gates (the graph takes it as an input)")
    parser.add_argument("--check-c-len", type=int, action="append", default=None,
                        help="cond lengths to check the padding at, per engine (u_len defaults "
                             "to 3/4 of c_len; default: half and all of the engine length)")
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--skip-export", action="store_true", help="reuse the exported ONNX")
    parser.add_argument("--skip-rknn", action="store_true", help="export + check only")
    parser.add_argument("--verify-sim", action="store_true",
                        help="build the first --seq-len and run it on the PC simulator against "
                             "the fp32 ONNX; slow, the simulator runs the whole 28-layer step")
    parser.add_argument("--front-k", type=int, default=0,
                        help="export the front shard (embedding + the first k transformer "
                             "blocks, output hidden [2,S,hidden], no guidance input) instead of "
                             "the full step graph; the CFG tail (final norm + audio heads) then "
                             "runs on the fp32 host")
    args = parser.parse_args()
    seq_lens = args.seq_len or [256, 384, 512, 768, 1024]
    sd_cfg = load_generator(args.model_dir)
    lc = sd_cfg[1]["llm_config"]
    if args.front_k and not 0 < args.front_k <= lc["num_hidden_layers"]:
        raise SystemExit(f"--front-k must be in [1, {lc['num_hidden_layers']}]")
    stem = (f"omnivoice_generator{'_front' + str(args.front_k) if args.front_k else ''}"
            f"_seq")

    if args.verify_sim:
        seq_len = seq_lens[0]
        onnx_path = args.output_dir / f"{stem}{seq_len}.onnx"
        rknn_onnx_path = args.output_dir / f"{stem}{seq_len}_w16.onnx"
        if not args.skip_export:
            export_onnx(args.model_dir, seq_len, args.guidance, onnx_path, sd_cfg,
                        front_k=args.front_k)
            export_onnx(args.model_dir, seq_len, args.guidance, rknn_onnx_path, sd_cfg,
                        dtype=torch.float16, front_k=args.front_k)
        c_len = (args.check_c_len or [max(1, seq_len // 2)])[0]
        verify_fp16(args.model_dir, onnx_path, rknn_onnx_path, seq_len, c_len, c_len * 3 // 4,
                    args.guidance, args.target, args.opt_level, args.seed, args.front_k)
        return 0

    for seq_len in seq_lens:
        onnx_path = args.output_dir / f"{stem}{seq_len}.onnx"
        rknn_onnx_path = args.output_dir / f"{stem}{seq_len}_w16.onnx"
        if not args.skip_export:
            export_onnx(args.model_dir, seq_len, args.guidance, onnx_path, sd_cfg,
                        front_k=args.front_k)
            export_onnx(args.model_dir, seq_len, args.guidance, rknn_onnx_path, sd_cfg,
                        dtype=torch.float16, front_k=args.front_k)
        if not args.skip_check:
            cases = args.check_c_len or [max(1, seq_len // 2), seq_len]
            check_padding(args.model_dir, onnx_path, seq_len,
                          [(c, max(1, c * 3 // 4)) for c in cases], args.guidance, args.seed,
                          args.front_k)
        if args.skip_rknn:
            continue
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from rknn_build import build_engine, report_engine

        out = args.output_dir / f"{stem}{seq_len}_fp16.rknn"
        build_engine(rknn_onnx_path, out, args.target, opt_level=args.opt_level,
                     gate_label=f"build {out.name}")
        report_engine(out, f"seq_len {seq_len}", args.target,
                      f"optimization_level {args.opt_level}")
        onnx_path.unlink(missing_ok=True)  # 2.5 GB per bucket; the engine is the artefact
        rknn_onnx_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
