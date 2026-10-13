#!/usr/bin/env python3
"""Build the stateful RKNN speech decoder (vocoder) for the Qwen3-TTS family.

The runtime side (``src/models/qwen3_tts/rknn_speech_decoder.cpp``, session options
``qwen3_tts.rknn_decoder`` / ``qwen3_tts.rknn_decoder_first``) runs one compiled RKNN step graph
in place of the ggml speech tokenizer decoder. Nothing in this repository produces that graph, so
this script is its only producer: it re-traces the WavTokenizer-v2 decoder from the
``Qwen/Qwen3-TTS-12Hz-0.6B-Base`` Hugging Face checkpoint (the ``speech_tokenizer`` subdirectory,
i.e. the same weights the GGUF package was converted from) as a fixed-shape streaming step, checks
it against a full decode, and compiles it for the NPU.

The exported step graph is stateful: every causal conv's tail and the pre-transformer's
sliding-window KV history travel through the graph I/O as one flat fp32 cache, so a step decodes
from continuous state and there are no chunk seams. Four inputs
``audio_codes [1,16,B] int64`` / ``freqs_in [B,head_dim/2] fp32`` / ``cache_valid [1] fp32`` /
``cache_in [N] fp32``, two outputs ``audio_values`` / ``cache_out``, ``1920`` samples per token at
24 kHz. ``N`` does not depend on ``B``, so a small-B step's cache feeds a large-B step directly —
that is what ``qwen3_tts.rknn_decoder_first`` uses.

Requirements (host, x86_64, dedicated venv — the RKNN toolkit pins its own numpy/onnx/torch):
  pip install torch onnx onnxruntime numpy
  pip install rknn-toolkit2          # PC-side converter, x86_64 only
On Python 3.12 the 2.3.2 converter additionally needs ``setuptools<81`` and an ``onnx/mapping.py``
shim inside the venv's onnx package; see docs/build/rockchip-npu.md.

Usage:
  python3 tools/prepare_qwen3_tts_rknn_vocoder.py --model-dir ~/.cache/huggingface/hub/\\
      models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/<sha> --codes-length 64
  python3 tools/prepare_qwen3_tts_rknn_vocoder.py --model-dir <dir> --codes-length 16
  python3 tools/prepare_qwen3_tts_rknn_vocoder.py --model-dir <dir> --skip-rknn   # ONNX only
"""

import argparse
import math
import os
import sys


def conv_cache_len(kernel_size, dilation):
    return (kernel_size - 1) * dilation


def convT_cache_len(kernel_size, stride):
    return math.ceil((kernel_size - 1) / stride)


def build_stateful_decoder(dec, B):
    """Wrap ``dec`` (the checkpoint's decoder, eval mode, eager attention) as a step graph:
    B tokens + flat cache in, B*samples_per_token samples + flat cache out.

    ``forward(codes [1,16,B] int64, freqs [B,head_dim/2] fp32, cache_valid [1] fp32,
    cache [N] fp32)``. The host-supplied choices below are NPU-driven and load-bearing; each is
    measured, and re-simplifying one costs 30-62 dB of output quality.
    """
    import torch

    class Step(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dec = dec
            self.B = B
            cfg = dec.config
            self.Lc = cfg.sliding_window - 1        # KV history slots
            self.n_layers = cfg.num_hidden_layers
            self.n_heads = cfg.num_attention_heads
            self.head_dim = cfg.head_dim
            self.scaling = self.head_dim ** -0.5

            specs = [("pre_conv", conv_cache_len(3, 1) * cfg.codebook_dim)]
            specs.append(
                ("kv", self.n_layers * 2 * self.n_heads * self.Lc * self.head_dim)
            )
            for bi in range(len(cfg.upsampling_ratios)):
                specs.append((f"up_convT{bi}", convT_cache_len(2, 2) * cfg.latent_dim))
                specs.append((f"up_dw{bi}", conv_cache_len(7, 1) * cfg.latent_dim))
            specs.append(("dec0", conv_cache_len(7, 1) * cfg.latent_dim))
            for i, r in enumerate(cfg.upsample_rates):
                in_dim = cfg.decoder_dim // 2 ** i
                out_dim = cfg.decoder_dim // 2 ** (i + 1)
                specs.append((f"b{i}_convT", convT_cache_len(2 * r, r) * in_dim))
                for d in (1, 3, 9):
                    specs.append((f"b{i}_r{d}", conv_cache_len(7, d) * out_dim))
            out_dim = cfg.decoder_dim // 2 ** len(cfg.upsample_rates)
            specs.append(("final", conv_cache_len(7, 1) * out_dim))
            self.specs = specs
            self.cache_numel = sum(n for _, n in specs)

            # KV history joins the step's k/v through constant-selection matmuls, not Concat:
            # the RKNN Concat path corrupts sliding-window attention (measured: step-0 audio SNR
            # 9.5 dB with kv zeroed, vs 56.8 dB for the concat-free graph); matmul and add are
            # first-class NPU ops.
            a = torch.zeros(self.Lc + B, self.Lc)
            a[: self.Lc, :] = torch.eye(self.Lc)
            p = torch.zeros(self.Lc + B, B)
            p[self.Lc :, :] = torch.eye(B)
            self.register_buffer("kv_sel_hist", a)
            self.register_buffer("kv_sel_cur", p)
            # Sliding-window causal mask on the RELATIVE grid (depends only on qp-kp), as a graph
            # constant at -100 (exp(-100) is exactly 0 in fp16; -3e4 leaks through the NPU
            # softmax). The training mask's "no history before token 0" guard is not expressible
            # on a relative grid, and feeding an absolute mask as an input together with in-graph
            # Cos/Sin segfaults the board runtime, so the cache slots carry a per-slot bias built
            # in forward() from the host-supplied count of real slots.
            qp = torch.arange(B, dtype=torch.float32).view(B, 1)
            kp = (torch.arange(self.Lc + B, dtype=torch.float32) - self.Lc).view(1, self.Lc + B)
            m = torch.where(
                (kp <= qp) & (kp > qp - self.Lc - 1),
                torch.zeros(()),
                torch.full((), -100.0),
            )
            self.register_buffer("attn_mask", m.view(1, 1, B, self.Lc + B))
            self.register_buffer(
                "slot_idx",
                torch.arange(self.Lc, dtype=torch.float32).view(1, 1, 1, self.Lc),
            )

        def _slot_bias(self, n_valid):
            # -100 on the Lc - n_valid never-written slots, 0 on the real ones. Elementwise on a
            # [1,1,1,Lc] grid: no comparison/Where, and no reduction over the 1.3 MB cache tensor
            # (whose Abs/ReduceSum/Clip version costs 30 dB of NPU fp16 accuracy).
            phantom = torch.clamp(self.Lc - n_valid - self.slot_idx, 0.0, 1.0)
            bias = -100.0 * phantom
            cur = torch.zeros(1, 1, 1, self.B)
            return torch.cat([bias, cur], dim=-1)

        def _split(self, flat):
            out, off = {}, 0
            for name, n in self.specs:
                out[name] = flat[off : off + n]
                off += n
            return out

        def _conv_step(self, cnet, x, cache):
            cl = x.shape[1]
            n = conv_cache_len(cnet.conv.kernel_size[0], cnet.conv.dilation[0])
            cs = cache.reshape(1, cl, n) if n else cache.reshape(1, cl, 0)
            xin = torch.cat([cs, x], dim=-1)
            y = cnet.conv(xin)
            return y, xin[..., -n:].reshape(-1) if n else x[..., :0].reshape(-1)

        def _convT_step(self, tnet, x, cache):
            cl, L = x.shape[1], x.shape[2]
            n = convT_cache_len(tnet.conv.kernel_size[0], tnet.conv.stride[0])
            cs = cache.reshape(1, cl, n) if n else cache.reshape(1, cl, 0)
            xin = torch.cat([cs, x], dim=-1)
            s = tnet.conv.stride[0]
            y = tnet.conv(xin)
            total = xin.shape[-1]
            y = y[..., : total * s]
            y = y[..., (total - L) * s :]
            return y, x[..., -n:].reshape(-1) if n else x[..., :0].reshape(-1)

        @staticmethod
        def _rot_half(t):
            d = t.shape[-1]
            a, b = t[..., : d // 2], t[..., d // 2 :]
            return torch.cat([-b, a], dim=-1)

        def forward(self, codes, freqs, cache_valid, flat_cache):
            # freqs = (pos * inv_freq) reduced mod 2*pi on the host, cos/sin computed in-graph:
            # cos/sin as RKNN inputs explodes the fp16 pipeline, and raw angles near pos 1e3 land
            # on an fp16 ulp of 0.5.
            # cache_valid = min(pos, Lc), the COUNT of real cache slots. The cache is a rolling
            # window, so the real slots are always the last ones; never-written slots hold
            # all-zero keys, whose score is 0 (not -inf), so they steal softmax mass. A boolean
            # "any history" flag is only true at step 0, so every later step leaks Lc - B phantom
            # keys into the softmax: measured 10.2 dB at B=16 (40 of 87 keys phantom) against
            # 48.8 dB at B=64 (7 of 79).
            dec = self.dec
            B = self.B
            c = self._split(flat_cache)
            parts = []

            hidden = dec.quantizer.decode(codes)
            hidden, cache = self._conv_step(dec.pre_conv, hidden, c["pre_conv"])
            parts.append(cache)
            hidden = hidden.transpose(1, 2)

            t = dec.pre_transformer
            hidden = t.input_proj(hidden)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos()
            sin = emb.sin()

            kv = c["kv"].reshape(self.n_layers, 2, self.n_heads, self.Lc, self.head_dim)
            for i, layer in enumerate(t.layers):
                res = hidden
                hs = layer.input_layernorm(hidden)[0]
                a = layer.self_attn
                q = a.q_proj(hs).view(B, self.n_heads, self.head_dim).transpose(0, 1)
                k = a.k_proj(hs).view(B, self.n_heads, self.head_dim).transpose(0, 1)
                v = a.v_proj(hs).view(B, self.n_heads, self.head_dim).transpose(0, 1)
                q = q * cos + self._rot_half(q) * sin
                k = k * cos + self._rot_half(k) * sin
                k_full = torch.matmul(self.kv_sel_hist, kv[i, 0]) + torch.matmul(
                    self.kv_sel_cur, k
                )
                v_full = torch.matmul(self.kv_sel_hist, kv[i, 1]) + torch.matmul(
                    self.kv_sel_cur, v
                )
                parts.append(k_full[:, -self.Lc :].reshape(-1))
                parts.append(v_full[:, -self.Lc :].reshape(-1))
                att = torch.softmax(
                    q[None] @ k_full[None].transpose(2, 3) * self.scaling
                    + self.attn_mask
                    + self._slot_bias(cache_valid),
                    dim=-1,
                )
                o = (att @ v_full[None]).transpose(1, 2).reshape(1, B, -1)
                hidden = res + layer.self_attn_layer_scale(a.o_proj(o))
                hidden = hidden + layer.mlp_layer_scale(
                    layer.mlp(layer.post_attention_layernorm(hidden))
                )
            hidden = t.norm(hidden)
            hidden = t.output_proj(hidden).transpose(1, 2)

            for bi, blocks in enumerate(dec.upsample):
                hidden, cache = self._convT_step(blocks[0], hidden, c[f"up_convT{bi}"])
                parts.append(cache)
                nb = blocks[1]
                res = hidden
                hidden, cache = self._conv_step(nb.dwconv, hidden, c[f"up_dw{bi}"])
                parts.append(cache)
                hs = hidden.permute(0, 2, 1)
                hs = nb.pwconv2(nb.act(nb.pwconv1(nb.norm(hs))))
                hidden = res + (nb.gamma * hs).permute(0, 2, 1)

            hidden, cache = self._conv_step(dec.decoder[0], hidden, c["dec0"])
            parts.append(cache)
            for bi in range(len(dec.config.upsample_rates)):
                blk = dec.decoder[1 + bi]
                hidden = blk.block[0](hidden)
                hidden, cache = self._convT_step(blk.block[1], hidden, c[f"b{bi}_convT"])
                parts.append(cache)
                for unit, d in zip(blk.block[2:], (1, 3, 9)):
                    res = hidden
                    hidden = unit.act1(hidden)
                    hidden, cache = self._conv_step(unit.conv1, hidden, c[f"b{bi}_r{d}"])
                    parts.append(cache)
                    hidden = unit.act2(hidden)
                    hidden = unit.conv2(hidden) + res
            hidden = dec.decoder[-2](hidden)
            hidden, cache = self._conv_step(dec.decoder[-1], hidden, c["final"])
            parts.append(cache)

            wav = hidden.clamp(min=-1, max=1).reshape(-1)
            return wav, torch.cat([p.reshape(-1) for p in parts])

    return Step().eval()


def export_onnx(args):
    import numpy as np
    import torch

    tokenizer_dir = os.path.join(args.model_dir, "speech_tokenizer")
    if not os.path.isdir(tokenizer_dir):
        sys.exit(f"ERROR: {tokenizer_dir} is not a Qwen3-TTS speech_tokenizer directory")
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Model,
    )

    full = Qwen3TTSTokenizerV2Model.from_pretrained(tokenizer_dir, local_files_only=True)
    dec = full.decoder.eval()
    dec.config._attn_implementation = "eager"

    B = args.codes_length
    step = build_stateful_decoder(dec, B)
    print(f"step B={B} cache_numel={step.cache_numel} "
          f"({step.cache_numel * 4 / 1e6:.1f} MB fp32) head_dim={step.head_dim} kv_len={step.Lc}")

    torch.manual_seed(0)
    n = 3 * B
    codes = torch.randint(0, 2048, (1, 16, n), dtype=torch.int64)
    inv_freq = dec.pre_transformer.rotary_emb.inv_freq.float()

    def rope(i, L):
        pos = torch.arange(i, i + L, dtype=torch.float32)
        return torch.remainder(torch.outer(pos, inv_freq), 2 * math.pi)

    with torch.no_grad():
        ref = dec(codes).reshape(-1)
        flat = torch.zeros(step.cache_numel, dtype=torch.float32)
        outs = []
        for i in range(0, n, B):
            L = min(B, n - i)
            wav, flat = step(codes[:, :, i : i + L], rope(i, L), np.float32(min(i, step.Lc)), flat)
            outs.append(wav)
        stepped = torch.cat(outs)

    L = min(len(ref), len(stepped))
    snr = 10 * torch.log10(
        ref[:L].pow(2).sum() / ((ref[:L] - stepped[:L]).pow(2).sum() + 1e-12)
    )
    print(f"torch stepwise vs full decode: len {len(stepped)}/{len(ref)} SNR {snr:.1f} dB")
    # fp32 self-check: semantic errors land below 20 dB; the kv-selection matmuls and the
    # mask broadcast add fp32 accumulation noise (measured 103-111 dB). Do not raise this to
    # 80: it aborts before torch.onnx.export and leaves no ONNX for the converter to load.
    if snr < 40:
        d = (ref[:L] - stepped[:L]).abs()
        print("max diff by 10k-sample blocks:",
              [round(float(d[i : i + 10000].max()), 5) for i in range(0, L, 10000)][:12])
        sys.exit("stateful decode does not match full decode")
    if args.check_only:
        return None

    out = args.output or f"vocoder_stateful_{B}.onnx"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    dummy_c = torch.randint(0, 2048, (1, 16, B), dtype=torch.int64)
    dummy_cache = torch.zeros(step.cache_numel, dtype=torch.float32)
    print(f"exporting {out} ...")
    # cache_valid is traced as a dynamic input; trace it mid-range so no branch of the
    # phantom-slot clamp degenerates into a constant.
    torch.onnx.export(
        step,
        (dummy_c, rope(0, B), torch.full((1,), float(B)), dummy_cache),
        out,
        input_names=["audio_codes", "freqs_in", "cache_valid", "cache_in"],
        output_names=["audio_values", "cache_out"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"saved {out} ({os.path.getsize(out) / 1e6:.1f} MB)")

    # Preflight for the converter stage: rknn-toolkit2 2.3.2's fold_constant aborts (core dump,
    # no Python traceback) on the decomposed LayerNorm/Softmax patterns a recent torch emits.
    # Measured with this graph: torch 2.4.0 + onnx 1.22 exports 1064 nodes (Gemm linears, no
    # Pad/Exp/Shape/ConstantOfShape) and converts byte-identically to the reference .rknn;
    # torch 2.14 + onnx 1.23 exports 2653 nodes and aborts. Say it here, at export time,
    # instead of losing a 20-minute converter run to a core dump.
    import onnx

    nodes = onnx.load(out, load_external_data=False).graph.node
    risky = sorted({n.op_type for n in nodes} & {"Pad", "Exp", "ConstantOfShape", "Shape", "Split"})
    print(f"graph: {len(nodes)} nodes, risky ops for rknn-toolkit2: {risky or 'none'}")
    if risky:
        print("WARNING: the RKNN stage will most likely abort in fold_constant with this graph; "
              "re-export it in the converter venv (toolkit-pinned torch/onnx) and pass that file "
              "with --onnx.")

    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(out, sess_options=opts, providers=["CPUExecutionProvider"])
    flat = np.zeros(step.cache_numel, dtype=np.float32)
    outs = []
    for i in range(0, n, B):
        L = min(B, n - i)
        w, flat = sess.run(None, {
            "audio_codes": codes[:, :, i : i + L].numpy(),
            "freqs_in": rope(i, L).numpy(),
            "cache_valid": np.full(1, min(i, step.Lc), dtype=np.float32),
            "cache_in": flat,
        })
        outs.append(np.asarray(w).reshape(-1)[: L * args.samples_per_token])
    onnx_step = np.concatenate(outs)
    ref_np = ref[:L].numpy()
    snr = 10 * np.log10(
        np.sum(ref_np**2) / (np.sum((ref_np - onnx_step[:L]) ** 2) + 1e-12)
    )
    print(f"onnxruntime stepwise vs full decode: SNR {snr:.1f} dB")
    if snr < 40:
        sys.exit("exported ONNX does not reproduce the torch step graph")
    return out, step.cache_numel, step.head_dim, step.Lc


def graph_contract(onnx_path):
    """Read (B, cache_numel, head_dim) back out of an exported ONNX, so the RKNN stage can run
    alone in the converter venv without restating the contract."""
    import onnx

    graph = onnx.load(onnx_path, load_external_data=False).graph
    dims = {
        i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in graph.input
    }
    for name in ("audio_codes", "freqs_in", "cache_in"):
        if name not in dims:
            sys.exit(f"ERROR: {onnx_path} is not a stateful step export (no {name} input)")
    codes, freqs, cache = dims["audio_codes"], dims["freqs_in"], dims["cache_in"]
    if len(codes) != 3 or len(freqs) != 2:
        sys.exit(f"ERROR: {onnx_path} does not have the stateful step shapes")
    return codes[2], cache[0], freqs[1] * 2


def convert_rknn(args, onnx_path, cache_numel, head_dim, kv_len):
    from rknn.api import RKNN

    out = args.output or (
        os.path.splitext(os.path.basename(onnx_path))[0] + f"_{args.target}.rknn"
    )
    print(f"=== {onnx_path} -> {out} ({args.target}) ===")
    rknn = RKNN(verbose=True)
    rknn.config(target_platform=args.target, optimization_level=args.opt_level)
    ret = rknn.load_onnx(
        model=onnx_path,
        inputs=["audio_codes", "freqs_in", "cache_valid", "cache_in"],
        input_size_list=[
            [1, 16, args.codes_length],
            [args.codes_length, head_dim // 2],
            [1],
            [cache_numel],
        ],
    )
    if ret != 0:
        sys.exit(f"ERROR: rknn load_onnx failed: {ret}")
    # FP16 (w16a16), not quantized: the Snake activation makes Q8/INT8 audibly noisy
    # (SNR 9.5-13.4 dB) for 1.75x the speed, while fp16 measures lossless (56.8 dB).
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rknn_build_log import rknn_build_gate

    with rknn_build_gate(f"build {os.path.basename(out)}", out + ".build.log"):
        ret = rknn.build(do_quantization=False)
    if ret != 0:
        sys.exit(f"ERROR: rknn build failed: {ret}")
    ret = rknn.export_rknn(out)
    if ret != 0:
        sys.exit(f"ERROR: rknn export failed: {ret}")
    rknn.release()
    print(f"saved {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    print(
        f"\nDeploy (per-SoC file, not portable between chips):\n"
        f"  scp {out} <board>:/root/tts-rknn/vocoder/\n"
        f"then run with\n"
        f"  --session-option qwen3_tts.rknn_decoder=/root/tts-rknn/vocoder/"
        f"{os.path.basename(out)}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--model-dir",
        default=os.environ.get("QWEN3_TTS_MODEL", ""),
        help="Qwen/Qwen3-TTS-12Hz-0.6B-Base snapshot directory (env QWEN3_TTS_MODEL)",
    )
    ap.add_argument("--codes-length", type=int, default=64,
                    help="tokens per compiled step (B); one compiled graph per value")
    ap.add_argument("--output", default=None)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--target", default="rk3588",
                    choices=["rk3588", "rk3576", "rk3562", "rk3566", "rk3568"])
    ap.add_argument("--opt-level", type=int, default=1, help="RKNN graph optimization level")
    ap.add_argument("--samples-per-token", type=int, default=1920,
                    help="24 kHz / 12.5 Hz token duration")
    ap.add_argument("--check-only", action="store_true", help="run the self-check, export nothing")
    ap.add_argument("--skip-rknn", action="store_true", help="export ONNX only")
    ap.add_argument("--onnx", default=None,
                    help="compile an existing stateful ONNX step (the converter-venv-only path); "
                         "B, cache_numel and head_dim are read back out of the graph")
    args = ap.parse_args()

    if args.onnx:
        B, cache_numel, head_dim = graph_contract(args.onnx)
        args.codes_length = B
        print(f"onnx {args.onnx}: B={B} cache_numel={cache_numel} head_dim={head_dim}")
        convert_rknn(args, args.onnx, cache_numel, head_dim, 0)
        return

    if not args.model_dir:
        sys.exit("ERROR: --model-dir (or QWEN3_TTS_MODEL) is required")
    exported = export_onnx(args)
    if exported is None:
        return
    if args.skip_rknn:
        return
    convert_rknn(args, *exported)


if __name__ == "__main__":
    main()
