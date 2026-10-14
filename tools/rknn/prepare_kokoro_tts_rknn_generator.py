#!/usr/bin/env python3
"""Compile the Kokoro iSTFTNet generator for a Rockchip NPU.

The seam is the generator graph of `src/models/kokoro_tts/decoder.cpp`, the stage that
is 83% of the family's runtime on RK3588: the predictor latent `x [1,512,F]`, the
harmonic-source STFT conditioning `har [1,22,60F+1]` (11 magnitude rows, 11 phase
rows) and the style vector `style [1,128]` in, the pre-iSTFT head
`spec_phase [1,22,60W+1]` out. The source-signal synthesis, the forward STFT and the
inverse STFT stay on the CPU, so the engine is conv/AdaIN/Snake only.

RKNN has no dynamic shapes, so the engine is compiled for a fixed window of `W` latent
frames. AdaIN normalizes over the time axis, so a padded window is only exact if the
padding is excluded from the statistics: the frame mask (1 real, 0 pad) is a graph
input and the statistics are `sum(x*keep)/sum(keep)` per channel, which is identical to
the runtime's unmasked norm when the mask is all ones. With real context on both sides
(`--padding`, the measured receptive field) an interior window then reproduces a
full-length decode exactly, and the seams are stitched in the spectrogram domain: the
runtime writes each window's emitted columns into one full-length spectrogram and runs
the iSTFT once over it, so there is no waveform crossfade and no depop.

The torch module below re-implements the runtime's generator graph, not the upstream
StyleTTS2 module: it matches the ggml graph the board runs (per-channel instance norm
with eps 1e-5, Snake `x + sin(a*x)^2/a`, conv_transpose with the centered
`(k-s)//2` padding, reflect-pad 1 on the last stage, resblock sum / 3). The export
gate below checks the exported ONNX graph against this torch module, and the
padded-window gate checks the windowed decode against the full-length one.

Two venvs: the export stage needs torch, the conversion stage needs rknn-toolkit2.

    <export-venv>/bin/python tools/rknn/prepare_kokoro_tts_rknn_generator.py \
      --checkpoint kokoro-v1_0.pth --config config.json --frames 512 --skip-rknn
    <converter-venv>/bin/python tools/rknn/prepare_kokoro_tts_rknn_generator.py \
      --onnx generator_w512.onnx --target rk3588
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

EPS = 1.0e-5  # AdaIN eps, assets.h:109
HARMONIC_DIMS = 9  # harmonic_num 8 + the fundamental
SAMPLING_RATE = 24000
SINE_AMP = 0.1
NOISE_STD = 0.003
INPUT_NAMES = ["x", "har", "style", "mask_mid", "mask_out"]
OUTPUT_NAME = "spec_phase"


def flatten(prefix: str, value, out: dict | None = None) -> dict:
    out = {} if out is None else out
    if torch.is_tensor(value):
        out[prefix] = value.detach().float().contiguous()
    elif isinstance(value, dict):
        for key, item in value.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), item, out)
    return out


def fold_weight_norm(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """torch's weight_norm on a conv: g * v / ||v||, norms over every dim but the output channel."""
    dims = list(range(1, v.dim()))
    return g * v / torch.linalg.vector_norm(v, ord=2, dim=dims, keepdim=True)


class Generator(torch.nn.Module):
    """The runtime's Kokoro generator graph, compiled at one fixed window width."""

    def __init__(self, state: dict, config: dict):
        super().__init__()
        istftnet = config["istftnet"]
        self.upsample_rates = [int(v) for v in istftnet["upsample_rates"]]
        self.upsample_kernels = [int(v) for v in istftnet["upsample_kernel_sizes"]]
        self.resblock_kernels = [int(v) for v in istftnet["resblock_kernel_sizes"]]
        self.resblock_dilations = [[int(v) for v in d] for d in istftnet["resblock_dilation_sizes"]]
        self.n_bins = int(istftnet["gen_istft_n_fft"]) // 2 + 1
        self.stages = len(self.upsample_rates)
        self.scale = int(math.prod(self.upsample_rates))
        # noise_convs: stage i < last is Conv1d(22, C, k=2*stride, stride, (stride+1)//2) with
        # stride = prod(upsample_rates[i+1:]); the last stage is a 1x1 (istftnet.py:280-287).
        self.noise_conv = []
        self.noise_res_kernel = []
        for i in range(self.stages):
            if i + 1 < self.stages:
                stride = int(math.prod(self.upsample_rates[i + 1:]))
                self.noise_conv.append((2 * stride, stride, (stride + 1) // 2))
                self.noise_res_kernel.append(7)
            else:
                self.noise_conv.append((1, 1, 0))
                self.noise_res_kernel.append(11)
        # Fold weight-norm once, at load: leaving `vector_norm` inside the traced graph puts 51
        # ReduceL2 nodes in the ONNX, and rknn.build aborts (core dump, no traceback) on the
        # decomposition the exporter builds around it.
        folded = {}
        for key, value in state.items():
            if key.endswith(".weight_g"):
                base = key[: -len(".weight_g")]
                folded[base + ".weight"] = fold_weight_norm(value, state[base + ".weight_v"])
            elif key.endswith(".weight_v"):
                continue
            else:
                folded[key] = value
        self.w = {key: value.float().contiguous() for key, value in folded.items()}
        # None evaluates `sin` exactly; a list of the minimax coefficients (fit on
        # [-clip, clip], powers 1,3,5,7,9) is the exported form.
        self.sin_form = None
        # With a coefficient list, True folds the argument into [-pi, pi] before the
        # polynomial (rewrite_sin_range_reduced) instead of clipping it (the shipped form).
        self.sin_reduce = False
        # --max-axis: run the decode in-graph as windows of (max_axis-1)//scale latent
        # frames with chunk_padding frames of context per side (0 = single window, the
        # shipping export).
        self.max_axis = 0
        self.chunk_padding = 0

    # --- blocks -------------------------------------------------------------

    def adain(self, prefix: str, index: int, x, style, keep):
        fc = f"{prefix}.{index}.fc"
        affine = F.linear(style, self.w[f"{fc}.weight"], self.w[f"{fc}.bias"])
        gamma, beta = affine.chunk(2, dim=1)
        # Masked instance norm through GlobalAveragePool, the way the reference Kokoro RKNN
        # port rewrites InstanceNorm: `rknn.build` lowers a ReduceSum over the time axis to a
        # register field capped at 8191, and at a 512-frame window the fine rate is 30720
        # columns, so the engine builds with `REGTASK: bit width of field value exceeds the
        # limit` and voices noise. GAP gives the mean over the padded length; times
        # length/count turns it into the mean over the real frames.
        count = keep.sum(dim=2, keepdim=True)
        scale = float(x.shape[-1]) / count
        mean = F.adaptive_avg_pool1d(x * keep, 1) * scale
        centered = x - mean
        variance = F.adaptive_avg_pool1d(centered * centered * keep, 1) * scale
        normalized = centered / torch.sqrt(variance + EPS)
        # Zero the padding on the way out, not just in the statistics: the convolutions
        # downstream read the pad frames, and a padded frame carrying the affine's bias
        # (or a conv bias, two levels deep) leaks into the last real frames. The runtime's
        # masked AdaIN does the same (decoder.cpp build_masked_adain_bct).
        return (normalized * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)) * keep

    def snake_conv(self, prefix: str, branch: str, index: int, x, dilation: int):
        """Snake1d feeding its convolution, with the per-channel 1/alpha folded into
        the convolution weights.

        `u + sin^2(alpha*u)/alpha` into `conv(W, .)` is exactly
        `conv(W/alpha, alpha*u + sin^2(alpha*u))`, which drops one full-tensor
        multiply and one full-tensor divide per Snake at the rates where the
        tensors are longest. Measured on the board, elementwise ops are 84% of the
        window's NPU time (Mul alone 63%) and the convolutions 15%, so removing
        DMA passes is where the wall clock is. The reference Kokoro RKNN port does
        the same as ONNX surgery (`rewrite_snake_alpha_fold`); the graph here is
        ours, so it is traced in this form. Exact algebra, not an approximation.
        """
        alpha = self.w[f"{prefix}.alpha{branch}.{index}"].reshape(1, -1, 1)
        argument = alpha * x
        wave = self.sin(argument)
        return self.conv(
            f"{prefix}.convs{branch}.{index}",
            argument + wave * wave,
            dilation,
            alpha.reciprocal(),
        )

    def sin(self, argument, clip: float = 3.14159):
        """`sin`, in the form the exported graph uses (self.sin_form).

        The exported graph cannot keep `Sin` (librknnrt runs it on CPU, 41% of the
        window), so the shipped form is the clipped minimax polynomial. The torch
        reference has to be able to evaluate the same function, otherwise the
        export gate measures the approximation instead of the export."""
        if self.sin_form is None:
            return torch.sin(argument)
        coeffs = self.sin_form
        if self.sin_reduce:
            # The fold of rewrite_sin_range_reduced: k = floor(x/pi + 0.5),
            # xr = x - k*pi. Snake consumes sin(x)^2, which is pi-periodic, so folding by
            # pi is exact for this graph's use and lands the argument in [-pi/2, pi/2],
            # the domain the shipped fit is minimax on.
            clipped = argument - math.pi * torch.floor(
                argument * (1.0 / math.pi) + 0.5
            )
        else:
            clipped = argument.clamp(-clip, clip)
        square = clipped * clipped
        acc = clipped.new_tensor(coeffs[-1])
        for coefficient in reversed(coeffs[:-1]):
            acc = acc * square + coefficient
        return clipped * acc

    def conv(self, prefix: str, x, dilation: int, scale=None):
        # Weight-norm folded at load time (assets.cpp:44-62), plain conv1d in the graph.
        # `scale` is a per-input-channel weight scaling, shaped (1, Cin, 1) to
        # broadcast over (Cout, Cin, kernel); constant at trace time.
        weight = self.w[f"{prefix}.weight"]
        if scale is not None:
            weight = weight * scale
        kernel = int(weight.shape[-1])
        return F.conv1d(x, weight, self.w[f"{prefix}.bias"], padding=(kernel * dilation - dilation) // 2, dilation=dilation)

    def resblock(self, prefix: str, x, style, keep):
        for index in range(len(self.resblock_dilations)):
            current = self.snake_conv(
                prefix, "1", index, self.adain(f"{prefix}.adain1", index, x, style, keep),
                self.resblock_dilations[index][index],
            )
            current = self.snake_conv(
                prefix, "2", index, self.adain(f"{prefix}.adain2", index, current, style, keep), 1,
            )
            x = current + x
        return x

    def upsample(self, index: int, x):
        """The transposed convolution as traced.

        The stuff + flipped-stride-1-conv + crop form the reference Kokoro RKNN port uses is
        arithmetically identical (verified to 1e-4), and was tried here: under the tracer it
        becomes Reshape/Slice with shape-derived bounds, and `rknn.build` then aborts inside
        `fold_constant` with a core dump and no traceback. `ConvTranspose` compiles. If the
        board shows the NPU falling back to CPU on these two ops, do the rewrite as ONNX
        graph surgery with constant shapes, the way the reference port does.
        """
        return F.conv_transpose1d(
            x,
            self.w[f"ups.{index}.weight"],
            self.w[f"ups.{index}.bias"],
            stride=self.upsample_rates[index],
            padding=(self.upsample_kernels[index] - self.upsample_rates[index]) // 2,
        )

    # --- in-graph chunked decode (diagnostic, --max-axis) ------------------------

    def forward_fine_sliced(self, x, har, style, mask_mid, mask_out):
        """The runtime's window walk moved inside the graph: the latent axis is split into
        windows of `(max_axis - 1) // scale` frames with `chunk_padding` frames of context
        per side, each window runs the full unsliced graph, and the emitted columns are
        concatenated (the seam column takes the later window's value, the way
        decode_windowed overwrites it). Every window's fine-rate axis is at most
        `max_axis` columns, so no op encoding field exceeds its 13 bits. The AdaIN
        statistics are per window (the runtime's semantics), so the stitched result is a
        valid synthesis, not the full-length one: the chunked-vs-monolithic gate measures
        that gap (~135 dB per window, ~10 dB stitched, same as decode_windowed)."""
        frames = x.shape[-1]
        scale = self.scale
        window = (self.max_axis - 1) // scale
        padding = self.chunk_padding
        emitted = window - 2 * padding
        outs = []
        start = 0
        while True:
            tail = start + emitted + padding >= frames
            lo = max(0, min(start - padding, frames - window))
            hi = frames if tail else min(frames, start + emitted + padding)
            count = hi - lo
            # cat against a constant zeros tensor, not F.pad: the legacy exporter turns
            # F.pad into a Shape/Reshape/Transpose/Cast/ConstantOfShape chain, and
            # rknn.build core-dumps in fold_constant on exactly that (the upsample()
            # docstring's Reshape/Slice abort, same optimizer pass).
            xw = torch.cat([x[:, :, lo:hi],
                            torch.zeros(1, x.shape[1], window - count)], dim=2)
            hw = torch.cat([har[:, :, scale * lo:scale * lo + scale * count + 1],
                            torch.zeros(1, har.shape[1], scale * (window - count))], dim=2)
            midw = torch.cat([mask_mid[:, :, scale // self.upsample_rates[-1] * lo
                                      :scale // self.upsample_rates[-1] * (lo + count)],
                              torch.zeros(1, 1, scale // self.upsample_rates[-1]
                                          * (window - count))], dim=2)
            outw = torch.cat([mask_out[:, :, scale * lo:scale * lo + scale * count + 1],
                              torch.zeros(1, 1, scale * (window - count))], dim=2)
            y = self._forward_full(xw, hw, style, midw, outw)
            end = frames if tail else min(frames, start + emitted)
            crop = y[:, :, scale * (start - lo):scale * (end - lo) + 1]
            # Bounded slice, not `crop[:, :, 1:]`: an open-ended slice exports an
            # INT64_MAX sentinel and rknn's fold_constant core-dumps on it.
            outs.append(crop if not outs else
                        crop[:, :, 1:scale * (end - start) + 1])
            if tail:
                break
            start += emitted
        return torch.cat(outs, dim=2)

    # --- graph --------------------------------------------------------------

    def _forward_full(self, x, har, style, mask_mid, mask_out):
        for index in range(self.stages):
            # The mask at the rate this stage works at: W*up0 for stage 0, W*up0*up1 + 1 for
            # stage 1 (the +1 is the reflect pad the last stage prepends).
            keep = mask_mid if index == 0 else mask_out
            x = F.leaky_relu(x, 0.1)
            kernel, stride, padding = self.noise_conv[index]
            source = F.conv1d(
                har,
                self.w[f"noise_convs.{index}.weight"],
                self.w[f"noise_convs.{index}.bias"],
                stride=stride,
                padding=padding,
            )
            source = self.resblock(f"noise_res.{index}", source, style, keep)
            x = self.upsample(index, x)
            if index + 1 == self.stages:
                # ReflectionPad1d((1, 0)): half-sample symmetric, so the pad sample is x[1],
                # not x[0] (ggml_pad_reflect_1d does the same, ggml-cpu/ops.cpp:8187).
                x = torch.cat([x[:, :, 1:2], x], dim=2)
            # Zero the padding at every stage boundary, not just inside AdaIN. AdaIN masks its
            # own output, but the conv that follows it reads real frames within the kernel
            # radius and writes their sum into the pad, and conv_transpose (and conv_post at
            # the end) reads that nonzero pad where a full-length decode reads the implicit
            # zeros past the tensor end. Measured: without this the last ~35 latent frames of a
            # padded window land 20 dB off; with it they match.
            x = (x + source) * keep
            total = 0.0
            for kernel_index in range(len(self.resblock_kernels)):
                total = total + self.resblock(
                    f"resblocks.{index * len(self.resblock_kernels) + kernel_index}",
                    x,
                    style,
                    keep,
                )
            x = total / len(self.resblock_kernels) * keep
        x = F.leaky_relu(x, 0.01)
        x = F.conv1d(x, self.w["conv_post.weight"], self.w["conv_post.bias"], padding=3)
        return x

    def forward(self, x, har, style, mask_mid, mask_out):
        if self.max_axis and mask_out.shape[-1] > self.max_axis:
            return self.forward_fine_sliced(x, har, style, mask_mid, mask_out)
        return self._forward_full(x, har, style, mask_mid, mask_out)


def masks_for(keep: torch.Tensor, model: Generator):
    """The frame mask at the two rates the graph normalizes at: W*up0 and W*up0*up1 + 1
    (the +1 is the reflect pad the last stage prepends)."""
    mid = None
    fine = keep
    for index in range(model.stages):
        fine = fine.repeat_interleave(model.upsample_rates[index], dim=2)
        if index == 0:
            mid = fine
    return mid, torch.cat([fine[:, :, :1], fine], dim=2)


def decode(model: Generator, x, har, style, keep):
    mid, mask_out = masks_for(keep, model)
    return model(x, har, style, mid, mask_out)


def ones_mask(batch: int, frames: int):
    return torch.ones(batch, 1, frames)


def receptive_field(config: dict) -> int:
    """Generator receptive field in latent frames: conv_transpose scales it by the stride and
    adds kernel-stride, each resblock conv adds (kernel-1)*dilation at its own rate."""
    field = 1
    for index, (stride, kernel) in enumerate(zip(config["istftnet"]["upsample_rates"], config["istftnet"]["upsample_kernel_sizes"])):
        field = field * stride + (kernel - stride)
        for kernel_size, dilations in zip(
            config["istftnet"]["resblock_kernel_sizes"], config["istftnet"]["resblock_dilation_sizes"]
        ):
            field += sum((kernel_size - 1) * d for d in dilations) + 3 * (kernel_size - 1)
    field += 6  # conv_post, kernel 7, dilation 1 (istftnet.py:289)
    return math.ceil(field / int(math.prod(config["istftnet"]["upsample_rates"])))


def synth_conditioning(frames: int, scale: int, hop: int, n_fft: int, seed: int):
    """A plausible `har`: the STFT of a harmonic source over a random f0 contour, the same
    source-signal recipe the runtime's conditioning stage runs (decoder.cpp:759-882)."""
    rng = torch.Generator().manual_seed(seed)
    contour = torch.exp(torch.randn(frames, generator=rng).cumsum(0) * 0.03)
    f0 = (contour * 120.0).clamp(50.0, 380.0) * (torch.rand(frames, generator=rng) > 0.18)
    source = torch.zeros(frames * scale * hop)
    phase = 2.0 * math.pi * torch.cumsum(f0.repeat_interleave(scale * hop), 0) / SAMPLING_RATE
    for harmonic in range(1, HARMONIC_DIMS + 1):
        source = source + torch.sin(phase * harmonic) * SINE_AMP / harmonic
    source = torch.tanh(source + torch.randn(source.shape, generator=rng) * NOISE_STD)
    spec = torch.stft(
        source,
        n_fft,
        hop,
        n_fft,
        window=torch.hann_window(n_fft),
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    # The stft of a 1-D source is [bins, frames], so the two halves stack on the channel
    # axis (dim=0 here, the batch axis is added by the unsqueeze): [2*bins, frames] ->
    # [1, 2*bins, frames], which is the `har` layout the graph takes.
    return torch.cat([spec.abs(), spec.angle()], dim=0).unsqueeze(0)


def snr(reference: torch.Tensor, windowed: torch.Tensor) -> tuple[float, float]:
    error = windowed - reference
    power = float((reference * reference).sum())
    noise = float((error * error).sum())
    return (
        10.0 * math.log10(power / noise) if noise > 0.0 else float("inf"),
        float(error.abs().max()),
    )


def decode_windowed(
    model: Generator,
    x: torch.Tensor,
    har: torch.Tensor,
    style: torch.Tensor,
    window: int,
    padding: int,
) -> tuple[torch.Tensor, int, float]:
    """The runtime's window walk: `window - 2*padding` latent frames emitted per step, real
    context on each side, zero-padded and mask-zeroed where the latent has run out, and the
    window that reaches the end of the latent extends its left context to fill the shape.

    Returns the stitched spectrogram, the window count, and the worst per-window SNR of the
    emitted columns against the same frames decoded unpadded at full length — the check that
    the padding, the masks and the stitching are right, independent of what the per-window
    AdaIN statistics do to the sound (see `main`)."""
    frames = x.shape[-1]
    scale = model.scale
    bins = model.n_bins * 2
    emitted = window - 2 * padding
    if emitted <= 0:
        raise SystemExit(f"window {window} is not wider than 2*padding {2 * padding}")
    out = torch.zeros(1, bins, scale * frames + 1)
    windows = 0
    worst = float("inf")
    for start in range(0, frames, emitted):
        tail = start + emitted + padding >= frames
        lo = max(0, min(start - padding, frames - window))
        hi = frames if tail else min(frames, start + emitted + padding)
        count = hi - lo
        xw = torch.zeros(1, x.shape[1], window)
        xw[:, :, :count] = x[:, :, lo:hi]
        hw = torch.zeros(1, har.shape[1], scale * window + 1)
        columns = scale * count + 1
        hw[:, :, :columns] = har[:, :, scale * lo: scale * lo + columns]
        keep = torch.zeros(1, 1, window)
        keep[:, :, :count] = 1.0
        mid, mask_out = masks_for(keep, model)
        y = model(xw, hw, style, mid, mask_out)
        end = frames if tail else min(frames, start + emitted)
        first = scale * (start - lo)
        last = scale * (end - lo) + 1
        # Stitch at the global offset, as the runtime does (rknn_generator.cpp:
        # `out + scale * emitted`). Writing at the window-local offset put every interior
        # window's output 28 frames early, which made the stitched-vs-full number
        # meaningless (it read as ~0 dB for the wrong reason).
        out[:, :, scale * start: scale * end + 1] = y[:, :, first:last]
        reference = model(
            x[:, :, lo:hi],
            har[:, :, scale * lo: scale * lo + columns],
            style,
            *masks_for(torch.ones(1, 1, count), model),
        )
        worst = min(worst, snr(reference[:, :, first:last], y[:, :, first:last])[0])
        windows += 1
        if tail:
            break
    return out, windows, worst


def check_onnx(
    model: Generator,
    x: torch.Tensor,
    har: torch.Tensor,
    style: torch.Tensor,
    window: int,
    padding: int,
    onnx_path: Path,
    sin_degree: int = 7,
    clip: float = 3.14159,
    sin_reduce: bool = True,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Run the exported graph on the last window of the runtime's walk and compare it with
    the same window in torch. This is the gate for the export path itself — the tracer, the
    mask plumbing and the ONNX-level rewrites — which the torch-side checks above cannot see.
    The compiled engine can only be checked on a board.

    Returns (SNR against the exact-sin reference, SNR against the reference evaluated in
    the exported sin form). The first number is the shipped graph's distance from the true
    Snake1d, i.e. the price of the sine approximation; the second is the export's distance
    from the graph it was built from, which is what has to be tight."""
    import onnxruntime as ort

    frames = x.shape[-1]
    scale = model.scale
    emitted = window - 2 * padding
    start = ((frames - 1) // emitted) * emitted
    lo = max(0, min(start - padding, frames - window))
    count = frames - lo
    columns = scale * count + 1
    xw = torch.zeros(1, x.shape[1], window)
    xw[:, :, :count] = x[:, :, lo:frames]
    hw = torch.zeros(1, har.shape[1], scale * window + 1)
    hw[:, :, :columns] = har[:, :, scale * lo: scale * lo + columns]
    keep = torch.zeros(1, 1, window)
    keep[:, :, :count] = 1.0
    mid, mask_out = masks_for(keep, model)
    with torch.no_grad():
        exact = model(xw, hw, style, mid, mask_out)
        model.sin_form = (
            # The folded form fits on the folded domain, so the reference has to use
            # the same coefficients the ONNX rewrite embeds.
            sin_polynomial_coeffs(
                sin_degree, math.pi / 2.0 if sin_reduce else clip
            )
            if sin_degree >= 3
            else None
        )
        model.sin_reduce = sin_reduce
        reference = model(xw, hw, style, mid, mask_out)
        model.sin_form = None
        model.sin_reduce = False
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {
        name: tensor.numpy()
        for name, tensor in zip(INPUT_NAMES, [xw, hw, style, mid, mask_out])
    }
    with torch.no_grad():
        got = torch.from_numpy(session.run(None, feeds)[0])
    first, last = scale * (start - lo), scale * (frames - lo) + 1
    return (
        snr(exact[:, :, first:last], got[:, :, first:last]),
        snr(reference[:, :, first:last], got[:, :, first:last]),
    )


def sin_polynomial_coeffs(degree: int, clip: float) -> list[float]:
    """Minimax (Chebyshev-projection) odd-polynomial fit of sin on [-clip, +clip],
    coefficients for the powers 1, 3, 5, 7, 9 that are <= degree.

    Minimax is ~50-300x tighter than Taylor at the same degree (degree 7 on
    [-pi, pi]: max error 4.5e-3 against Taylor's 7.5e-2) because it spreads the
    error uniformly. Shared by the torch reference form and the ONNX rewrite so
    the two are the same function, which is what makes the export gate a check of
    the export rather than of the approximation."""
    import numpy as np
    from numpy.polynomial import chebyshev

    t = np.cos(np.linspace(0, np.pi, 4096))  # Chebyshev nodes in [-1, 1]
    poly = chebyshev.Chebyshev.fit(t, np.sin(clip * t), degree)
    poly = poly.convert(kind=np.polynomial.Polynomial).coef.copy()
    poly[::2] = 0.0  # enforce odd symmetry
    return [float(poly[power] / clip**power) for power in (1, 3, 5, 7, 9) if power <= degree]


def horner_sin_nodes(base: str, argument: str, target: str, powers, coeffs, constant) -> list:
    """The Horner chain of the odd polynomial on `argument`, writing the product into
    `target`. Shared by the two sin rewrites so they are the same polynomial."""
    from onnx import helper

    x2 = f"{base}/x2"
    new = [helper.make_node("Mul", [argument, argument], [x2], name=f"{base}/x2")]
    # Horner from the top coefficient down, one running accumulator, so the
    # fine-rate tensor is written once per step instead of materialising x^3/x^5/x^7.
    acc = constant(f"{base}/c{powers[-1]}", coeffs[-1])
    for index in range(len(powers) - 2, -1, -1):
        power = powers[index]
        mul = f"{base}/h_mul_{power}"
        new.append(helper.make_node("Mul", [x2, acc], [mul], name=f"{base}/H_Mul_{power}"))
        acc = f"{base}/h_add_{power}"
        new.append(
            helper.make_node(
                "Add",
                [constant(f"{base}/c{power}", coeffs[index]), mul],
                [acc],
                name=f"{base}/H_Add_{power}",
            )
        )
    new.append(helper.make_node("Mul", [argument, acc], [target], name=f"{base}/H_x"))
    return new


def rewrite_sin_polynomial(model, degree: int = 7, clip: float = 3.14159) -> int:
    """Snake1d's `sin` -> clipped minimax odd polynomial, in ONNX.

    librknnrt has no NPU `Sin`: measured on the board, the 48 Sin nodes of the
    128-frame window ran on CPU and were 41% of the window's total time (350 ms
    of 846 ms, `RKNN_LOG_LEVEL=5` operator ranking). The replacement is
    Clip + 5 Mul + 3 Add, all NPU-native.

    The clip is the part that bites: Snake's argument is alpha*x with a learned
    per-channel alpha, so |alpha*x| runs well past pi, and saturating it there
    moves the generator output by ~30 dB SNR (measured, peak |d| 3.4 on the
    spectrogram). The reference Kokoro RKNN port ships the same pass and calls
    the residue inaudible - Snake squares the sine and adds x, so the error is
    bounded - which is why the torch reference applies the identical form and the
    export gate measures the export, not the approximation.
    """
    import numpy as np
    from onnx import helper, numpy_helper

    nodes = [node for node in list(model.graph.node) if node.op_type == "Sin"]
    if not nodes:
        return 0
    powers = [power for power in (1, 3, 5, 7, 9) if power <= degree]
    if not powers:
        return 0
    coeffs = sin_polynomial_coeffs(degree, clip)

    def constant(name: str, value) -> str:
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(value, np.float32), name)
        )
        return name

    for node in nodes:
        source, target = node.input[0], node.output[0]
        base = node.name or target
        lo = constant(f"{base}/lo", -clip)
        hi = constant(f"{base}/hi", clip)
        clipped = f"{base}/clip"
        new = [helper.make_node("Clip", [source, lo, hi], [clipped], name=f"{base}/Clip")]
        new += horner_sin_nodes(base, clipped, target, powers, coeffs, constant)
        position = list(model.graph.node).index(node)
        model.graph.node.remove(node)
        for offset, created in enumerate(new):
            model.graph.node.insert(position + offset, created)
    return len(nodes)


def rewrite_sin_range_reduced(model, degree: int = 7, clip: float = 3.14159) -> int:
    """Snake1d's `sin` -> range-reduced minimax odd polynomial, in ONNX (`--sin-reduce`).

    The clipped rewrite clips its argument at ±pi, and the clip, not the fit, is the
    30.4 dB floor: measured over the 49M Snake arguments of a 200-frame decode, 1.6%
    are past pi (max 20.5), and degrees 5, 7 and 9 all measure the same 17.6 dB
    sine-level SNR. This form folds the argument with
    k = Floor(x/pi + 0.5); xr = x - k*pi (Mul/Add/Floor/Mul/Sub) and evaluates the
    minimax fit of sin on [-pi/2, pi/2], where it is valid. The fold period is pi, not
    2*pi, because Snake consumes sin(x)^2, which is pi-periodic: folding by pi is exact
    for this graph's use, costs the identical 4 extra full-tensor elementwise ops per
    Snake, and halves the fitted interval, which is where the accuracy comes from.
    `Floor` is not free: the 2.3.2 compiler lowers it to CPU (`Floor FLOAT16 CPU` at all
    48 sine sites of a folded engine, against `Clip FLOAT16 NPU` for the clipped form),
    so each fold site is a full-tensor NPU->CPU->NPU round trip and the folded engine
    measures +36% generator time on RK3588.
    """
    import numpy as np
    from onnx import helper, numpy_helper

    nodes = [node for node in list(model.graph.node) if node.op_type == "Sin"]
    if not nodes:
        return 0
    powers = [power for power in (1, 3, 5, 7, 9) if power <= degree]
    if not powers:
        return 0
    # Fit on the folded domain, not on the clip domain.
    coeffs = sin_polynomial_coeffs(degree, math.pi / 2.0)

    def constant(name: str, value) -> str:
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(value, np.float32), name)
        )
        return name

    for node in nodes:
        source, target = node.input[0], node.output[0]
        base = node.name or target
        k = f"{base}/fold_k"
        reduced = f"{base}/fold_xr"
        new = [
            helper.make_node(
                "Mul",
                [source, constant(f"{base}/inv_pi", 1.0 / math.pi)],
                [f"{base}/fold_scaled"],
                name=f"{base}/Fold_Mul",
            ),
            helper.make_node(
                "Add",
                [f"{base}/fold_scaled", constant(f"{base}/half", 0.5)],
                [f"{base}/fold_shift"],
                name=f"{base}/Fold_Add",
            ),
            helper.make_node("Floor", [f"{base}/fold_shift"], [k], name=f"{base}/Fold_Floor"),
            helper.make_node(
                "Mul",
                [k, constant(f"{base}/pi", math.pi)],
                [f"{base}/fold_kpi"],
                name=f"{base}/Fold_K",
            ),
            helper.make_node("Sub", [source, f"{base}/fold_kpi"], [reduced], name=f"{base}/Fold_Sub"),
        ]
        new += horner_sin_nodes(base, reduced, target, powers, coeffs, constant)
        position = list(model.graph.node).index(node)
        model.graph.node.remove(node)
        for offset, created in enumerate(new):
            model.graph.node.insert(position + offset, created)
    return len(nodes)


def export_onnx(
    model: Generator,
    window: int,
    output: Path,
    sin_degree: int = 7,
    sin_reduce: bool = True,
) -> None:
    scale = model.scale
    args = (
        torch.zeros(1, 512, window),
        torch.zeros(1, model.n_bins * 2, scale * window + 1),
        torch.zeros(1, 128),
        torch.ones(1, 1, window * model.upsample_rates[0]),
        torch.ones(1, 1, scale * window + 1),
    )
    with torch.no_grad():
        torch.onnx.export(
            model,
            args,
            str(output),
            input_names=INPUT_NAMES,
            output_names=[OUTPUT_NAME],
            opset_version=17,
            do_constant_folding=True,
        )
    import onnx

    # torch 2.14's exporter spills the initializers to a `<name>.data` sidecar. rknn.load_onnx
    # reads the graph proto only, so merge the weights back into one self-contained file.
    sidecar = output.with_suffix(output.suffix + ".data")
    if sidecar.exists():
        graph = onnx.load(str(output), load_external_data=True)
        onnx.save(graph, str(output))
        sidecar.unlink()
    graph = onnx.load(str(output))
    if sin_degree >= 3:
        rewritten = (
            rewrite_sin_range_reduced(graph, sin_degree)
            if sin_reduce
            else rewrite_sin_polynomial(graph, sin_degree)
        )
    else:
        rewritten = 0
    if rewritten:
        onnx.save(graph, str(output))
    onnx.checker.check_model(str(output))
    print(
        f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB, window {window}, "
        f"{rewritten} sin -> polynomial)"
    )


def convert_rknn(onnx_path: Path, output: Path, target: str) -> None:
    import onnx

    from rknn.api import RKNN

    graph = onnx.load(str(onnx_path))
    names = [value.name for value in graph.graph.input]
    sizes = [
        [dim.dim_value if dim.HasField("dim_value") else 1 for dim in value.type.tensor_type.shape.dim]
        for value in graph.graph.input
    ]
    rknn = RKNN(verbose=False)
    # Not single_core_mode: the engine stays multi-core and rknn_set_core_mask in the runtime
    # decides how many NPU cores a call runs on.
    rknn.config(target_platform=target)
    if rknn.load_onnx(model=str(onnx_path), inputs=names, input_size_list=sizes) != 0:
        raise SystemExit("load_onnx failed")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rknn_build_log import rknn_build_gate

    with rknn_build_gate(f"{output.name}", output.with_suffix(".build.log")):
        built = rknn.build(do_quantization=False)
    if built != 0:
        raise SystemExit("build failed (see the rknn log above)")
    if rknn.export_rknn(str(output)) != 0:
        raise SystemExit("export_rknn failed")
    rknn.release()
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB, inputs {names})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, help="kokoro-v1_0.pth (export stage)")
    parser.add_argument("--config", type=Path, help="Kokoro config.json (export stage)")
    parser.add_argument("--onnx", type=Path, help="exported generator ONNX to compile")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--prefix", default="generator")
    # The window width is capped by the NPU, not by the model: an op encoding field on
    # the time axis is 13 bits, so the longest axis in the graph, the fine rate
    # `columns_per_frame * W + 1` = 60W + 1, has to stay under 8192. At 512 the compiler
    # emits ~800k `REGTASK: The bit width of field value exceeds the limit, ... limit:
    # 0x1fff, value: 0x7800` lines and the engine voices clipped noise (measured: rms
    # 32764/32767, SNR -27 dB against the CPU decode). 128 is the widest power of two
    # inside the limit.
    parser.add_argument("--frames", type=int, nargs="*", default=[128], help="window widths to compile")
    parser.add_argument("--padding", type=int, default=0, help="context frames per side (default: measured receptive field)")
    parser.add_argument("--target", default="rk3588")
    parser.add_argument(
        "--sin-degree",
        type=int,
        default=5,
        help="odd-polynomial degree for Snake1d's sin in the exported graph (0 keeps Sin, "
             "which librknnrt runs on CPU). With the default range reduction the fit is valid "
             "over the whole folded argument, so degree 5 already measures 93.2 dB against the "
             "exact sine at the export - above the compiled engine's own fp16 floor (~50 dB at "
             "the generator output) - and the extra terms buy nothing. Past the clip form "
             "(--no-sin-reduce) all degrees plateau at 30.4 dB",
    )
    parser.add_argument(
        "--sin-reduce",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="range-reduce Snake1d's sin (fold to [-pi/2,pi/2] with Floor, no Clip); the "
             "default. Clipping at ±pi is what sets the 30.4 dB sine floor (measured: 1.6%% of "
             "the Snake arguments of a 200-frame decode are past pi, max 20.5). The fold period "
             "is pi, not 2*pi, because Snake consumes sin(x)^2, which is pi-periodic: folding by "
             "pi is exact for this graph's use and halves the fitted interval, which is where the "
             "accuracy is - the exported graph measures 93.2 dB against the exact sine, against "
             "the 2*pi fold's 50.0 dB. The fold is a quality default, not a speed one: engines "
             "from one producer revision, interleaved 3x per utterance on RK3588, measure RTF "
             "1.046 against the clipped engine's 0.867 (+21%%), and the build log says why - the "
             "2.3.2 compiler lowers every rounding op off the NPU ('Floor FLOAT16 CPU' at all 48 "
             "sine sites of a folded engine, against 'Clip FLOAT16 NPU' for the clipped form), so "
             "each fold site is a full-tensor NPU->CPU->NPU round trip. --no-sin-reduce restores "
             "the clipped export byte for byte",
    )
    parser.add_argument(
        "--max-axis",
        type=int,
        default=0,
        help="run the decode in-graph as windows of (max_axis-1)//scale latent frames with "
             "--padding frames of context per side, stitched like decode_windowed (0 = single "
             "window, the shipping export, byte-identical). Diagnostic for wide windows: at "
             "W=256 the unsliced build emits 403 587 REGTASK register-field lines (the 13-bit "
             "op-encoding field caps at 8191 fine columns); chunking to 8191 takes it to 0. "
             "Each window normalizes with its OWN AdaIN statistics, so the stitched result is "
             "a valid per-window synthesis, not the monolithic one (the chunked-vs-monolithic "
             "gate measures that gap, same numbers as decode_windowed).",
    )
    parser.add_argument("--skip-rknn", action="store_true", help="export and check only")
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()

    if args.onnx is not None:
        convert_rknn(args.onnx, args.output_dir / f"{args.prefix}_{args.onnx.stem}.rknn", args.target)
        return 0
    if args.checkpoint is None or args.config is None:
        raise SystemExit("--checkpoint and --config are needed to export (or pass --onnx to compile)")

    config = json.loads(args.config.read_text())
    state = flatten("", torch.load(args.checkpoint, map_location="cpu"))
    prefix = "decoder.module.generator."
    generator_state = {
        key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)
    }
    if not generator_state:
        raise SystemExit(f"no {prefix}* tensors in {args.checkpoint}")
    print(f"generator: {len(generator_state)} tensors, "
          f"{sum(int(v.numel()) for v in generator_state.values()) / 1e6:.1f}M params")

    padding = args.padding or receptive_field(config)
    print(f"receptive field: {padding} latent frames of context per side")
    model = Generator(generator_state, config)
    model.eval()
    model.max_axis = args.max_axis
    model.chunk_padding = padding
    scale = model.scale
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_check:
        frames = 200
        x = torch.randn(1, 512, frames) * 0.5
        har = synth_conditioning(frames, scale, config["istftnet"]["gen_istft_hop_size"],
                                config["istftnet"]["gen_istft_n_fft"], seed=7)
        style = torch.randn(1, 128) * 0.5
        with torch.no_grad():
            full = decode(model, x, har, style, ones_mask(1, x.shape[-1]))
            if args.max_axis:
                # The chunking gate: the in-graph windowed module against the unsliced
                # one on the same full-length decode. Per-window AdaIN statistics, so this
                # is the same gap decode_windowed reports (~135 dB per window, ~10 dB
                # stitched), not exactness. The gates below use the unsliced decode.
                model.max_axis = 0
                full_unsliced = decode(model, x, har, style, ones_mask(1, x.shape[-1]))
                model.max_axis = args.max_axis
                sliced_snr, sliced_peak = snr(full_unsliced, full)
                frames_ = x.shape[-1]
                window_ = (args.max_axis - 1) // scale
                emitted_ = window_ - 2 * padding
                windows_ = 1 + max(0, -(-(frames_ - window_) // emitted_))
                full = full_unsliced
                print(f"chunked vs monolithic (full decode, {windows_} "
                      f"windows of {window_} at padding {padding}): "
                      f"{sliced_snr:.1f} dB (peak |d| {sliced_peak:.6f})")
        model.max_axis = 0  # the out-of-graph window gates run the unsliced module
        for window in args.frames:
            if window <= 2 * padding:
                raise SystemExit(f"window {window} leaves no emitted frames at padding {padding}")
            with torch.no_grad():
                windowed, windows, worst = decode_windowed(
                    model, x, har, style, window, padding)
            global_snr, peak = snr(full, windowed)
            print(f"window {window} (emitted {window - 2 * padding}, {windows} windows): "
                  f"per-window SNR {worst:.1f} dB, vs the full decode {global_snr:.1f} dB "
                  f"(peak |d| {peak:.4f})")
            if worst < 60.0:
                raise SystemExit("the windowed decode does not reproduce the same frames decoded "
                                 "unpadded — the padding, masks or stitching are wrong")
            if windows > 1:
                # AdaIN normalizes per window, so a multi-window decode is a different (valid)
                # synthesis, not an approximation of the full-length one: the stitched decode
                # lands ~10 dB from the full-length one on the 486-frame dump (worst window
                # 0.4 dB), and the shipping kokoro-server engine (T_fix=50, CTX=2, STEP=46,
                # 480-sample crossfade, 11 windows here) measures -0.7 dB against the same
                # full-length reference. The gate is per-window exactness above; this number is
                # reported so a regression in the seam is visible, and the audio is judged on
                # the board.
                print(f"  note: {windows} windows re-estimate the AdaIN statistics per window; "
                      f"the gap to the full decode is that, not the stitching")

    model.max_axis = args.max_axis  # the export traces the in-graph windowed module
    for window in args.frames:
        onnx_path = args.output_dir / f"{args.prefix}_w{window}.onnx"
        export_onnx(model, window, onnx_path, args.sin_degree, args.sin_reduce)
        if not args.skip_check:
            with torch.no_grad():
                exact, form = check_onnx(
                    model, x, har, style, window, padding, onnx_path,
                    args.sin_degree, sin_reduce=args.sin_reduce,
                )
            print(
                f"  exported graph, last window: {form[0]:.1f} dB against the graph it was "
                f"built from (peak |d| {form[1]:.5f}), {exact[0]:.1f} dB against the exact "
                f"sin (peak |d| {exact[1]:.4f}"
                + (", the price of the sine polynomial" if args.sin_degree >= 3 else "")
                + ")"
            )
            if form[0] < 60.0:
                raise SystemExit("the exported ONNX graph does not reproduce the torch graph")
        if not args.skip_rknn:
            convert_rknn(onnx_path, args.output_dir / f"{args.prefix}_w{window}.rknn", args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
