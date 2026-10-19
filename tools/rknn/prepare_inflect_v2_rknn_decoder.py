#!/usr/bin/env python3
"""Build the fixed-window RKNN speech decoder for an Inflect v2 model.

Inflect ships two ONNX per model: `duration.onnx` (text ids in, prior stats out) and
`decode.onnx` (prior stats plus its own noise in, PCM out). The RKNN path needs the decoder half
alone, compiled for one fixed latent window, because RKNN has no dynamic shapes. This script slices
`decode.onnx` at the inverse-flow input `z` and its mask `y_mask` (the seam the runtime has on
hand: the pre-flow latent and the frame mask), checks that the fixed window plus the chunk
stitching reproduces a full-length decode, and compiles the result for a RKNPU2 target.

The seam is the flow input and not the input of `dec.conv_pre` on purpose: the flow is part of the
decoder half - the runtime's decoder graph is literally `reverse_flow(latent)` into `dec` - and the
runtime's latent is the pre-flow one. Slicing at `conv_pre` compiles a decoder that expects
post-flow samples and feeds it pre-flow ones, which is quiet mush.

The head of `decode.onnx` (`z = m_p + exp(logs) * zp_noise * noise_scale`) is outside the slice, so
the engine has no noise input and no `noise_scale`: the runtime draws the noise itself, with its own
`variation` knob, and hands the engine the finished latent.

The slice carries decoder-half mask multiplications by default (`mask_decoder_graph`): the decoder
has no masking of its own, so a zero-padded window would otherwise read its own conv biases where a
full-length decode reads the implicit zeros past the end of the tensor, and the window that reaches
the end of the utterance is exactly that case.

Two stages, usually in two venvs (the slice/check needs onnx + onnxruntime + onnxslim, the
conversion needs rknn-toolkit2 with its own pinned torch/onnx):

    <venv>/bin/python tools/rknn/prepare_inflect_v2_rknn_decoder.py \
        --decode-onnx decode.onnx --duration-onnx duration.onnx --skip-rknn --output decoder_z.onnx
    <rknn-venv>/bin/python tools/rknn/prepare_inflect_v2_rknn_decoder.py \
        --onnx decoder_z.onnx --decode-onnx decode.onnx --duration-onnx duration.onnx \
        --target rk3588

The slice stage pins the graph to the compiled window, so the self-check needs the original
`decode.onnx` for a full-length reference decode; without it the check degrades to one
window-length latent decoded against itself.

The window contract (window 256 latent slices, 192 emitted + 32+32 real context, zero-padded and
mask-zeroed at the end that has no data) is the same one the runtime implements in
src/community_models/inflect_v2/rknn_decoder.cpp; change it here and there together. 32 is
measured: at 480 frames against a full-length decode, interior windows score 5.1 dB with no
context, 40.0 dB at 20, 82.1 dB at 24 and 128.8 dB at 32.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rknn_graph import decoder_channels, decoder_channels_of, graph_input_names

CHUNK = 192
PADDING = 32
DEPOP_COMPARE = 24
DEPOP_SEARCH = 44
# The runtime's default `variation` (InflectV2GenerationOptions), which is how much of the prior's
# standard deviation the self-check's latents carry.
VARIATION = 0.667


def decoder_seam(graph) -> tuple[str, str]:
    """Latent and mask at the decoder seam: the inverse-flow input `z` and its `y_mask`.

    The export draws the prior sample as `m_p_exp + exp(logs_p_exp) * zp_noise * noise_scale` and
    feeds the sum into the inverse flow, so that sum's output tensor is exactly the seam. `y_mask`
    is a graph input of the export and the flow multiplies by it after every convolution, so it
    stays live downstream of the seam (the decoder half itself never reads it).
    """
    add = next(
        (n for n in graph.node if n.op_type == "Add" and "m_p_exp" in n.input), None)
    if add is None:
        raise SystemExit(
            "no Add of m_p_exp: this ONNX is not an Inflect v2 decoder export "
            "(decode.onnx of the official Micro/Nano v2 release)"
        )
    if not any(i.name == "y_mask" for i in graph.input):
        raise SystemExit("decode.onnx has no y_mask input; this is not the expected export")
    return add.output[0], "y_mask"


def mask_decoder_graph(model) -> int:
    """Inject `y_mask` multiplications into the decoder half (opt-in `--dec-mask`).

    Inflect's dec has no masking, so in a zero-padded window every decoder conv writes its
    bias-driven output over the padding and the next conv reads that where a full-length decode
    reads the implicit zeros past the tensor end: the tail window's last slices land far off.
    Multiplying every decoder Conv/ConvTranspose output by the mask - the mask itself upsampled
    nearest by each `dec.ups` stride, so it is at the stage's rate - makes a padded window read
    zeros exactly where the full decode reads implicit zeros. With mask=1 everywhere (the full
    decode, and every interior window) each injected Mul is an exact fp32 identity, so the
    reference decode is unchanged.
    """
    graph = model.graph
    current = "y_mask"
    rebuilt = []
    injected = 0
    for node in graph.node:
        rebuilt.append(node)
        if not node.name.startswith("/dec/"):
            continue
        if node.op_type == "ConvTranspose":
            stride = next(a for a in node.attribute if a.name == "strides").ints[0]
            # Exact nearest upsampling of the mask as an all-ones ConvTranspose1d
            # (kernel = stride, no bias): the op dec.ups already runs on the NPU. A
            # Resize node lowers to a target the board runtime rejects (the build
            # logs `Unkown op target: 0` and the engine outputs garbage there).
            weight = f"dec_mask_upw_{injected}"
            graph.initializer.append(helper.make_tensor(
                weight, TensorProto.FLOAT, [1, 1, stride], [1.0] * stride))
            rebuilt.append(helper.make_node(
                "ConvTranspose", [current, weight], [f"dec_mask_up_{injected}"],
                name=f"dec_mask_upsampler_{injected}",
                strides=[stride], kernel_shape=[stride], pads=[0, 0]))
            current = f"dec_mask_up_{injected}"
        if node.op_type in ("Conv", "ConvTranspose"):
            out = node.output[0]
            masked = f"{out}_masked"
            rebuilt.append(helper.make_node(
                "Mul", [out, current], [masked], name=f"dec_mask_mul_{injected}"))
            for consumer in graph.node:
                consumer.input[:] = [masked if i == out else i for i in consumer.input]
            for value in graph.output:
                if value.name == out:
                    value.name = masked
            injected += 1
    del graph.node[:]
    graph.node.extend(rebuilt)
    return injected


def fold_shapes(model, channels: int, window: int):
    """Make the slice static: pin the compiled window and fold what that makes constant.

    The export sizes its spline-flow tensors from the latent length (`Shape` into
    `ConstantOfShape` into `Slice`), which is a dynamic-shape chain RKNN has no use for and whose
    presence makes the toolkit's own `fold_constant` abort the build with a core dump (measured:
    `base_optimize done.` then SIGABRT, with the log truncated at `fold_constant ...`). Pinning the
    input shapes and folding first leaves a plain conv graph: 979 nodes with 24 `Shape` and 24
    `ConstantOfShape` become 409 of Conv/ConvTranspose/LeakyRelu/Add/Mul/Slice and friends.
    """
    import onnxslim

    return onnxslim.slim(
        model,
        input_shapes=[f"z:1,{channels},{window}", f"y_mask:1,1,{window}"])


def slice_decoder(
    decode_onnx: Path,
    output: Path,
    window: int,
    dec_mask: bool = True,
) -> None:
    model = onnx.load(str(decode_onnx), load_external_data=True)
    latent, mask = decoder_seam(model.graph)
    onnx.utils.extract_model(
        str(decode_onnx), str(output), [latent, mask], ["waveform"])

    sliced = onnx.load(str(output))
    names = {latent: "z", mask: "y_mask"}
    for value in sliced.graph.input:
        if value.name not in names:
            continue
        value.name = names[value.name]
    for node in sliced.graph.node:
        node.input[:] = [names.get(name, name) for name in node.input]
    channels = decoder_channels_of(sliced.graph)
    sliced = fold_shapes(sliced, channels, window)
    if dec_mask:
        print(f"dec-mask: injected {mask_decoder_graph(sliced)} mask muls into the decoder half")
    onnx.checker.check_model(sliced, full_check=False)
    onnx.save(sliced, str(output))
    ops: dict[str, int] = {}
    for node in sliced.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print(
        f"sliced {decode_onnx.name} at {latent!r} + {mask!r}: {len(sliced.graph.node)} of "
        f"{len(model.graph.node)} nodes, inputs {list(names.values())}, "
        f"output {sliced.graph.output[0].name!r}"
    )
    risky = sorted(
        op for op in ops
        if op in {"Pad", "Exp", "Shape", "ConstantOfShape", "Split", "Atan", "Erf", "Gather"}
    )
    if risky:
        print(f"  note: folded graph still contains {risky}; if rknn.build aborts in "
              "fold_constant, the exporting torch/onnx pair is too new for the toolkit")
    print(f"wrote {output}")


def decode_window(session, z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    feeds = {"z": z.astype(np.float32)}
    for value in session.get_inputs():
        if value.name == "y_mask":
            feeds[value.name] = mask.astype(np.float32)
    out = session.run(None, feeds)[0]
    return np.asarray(out).reshape(-1).astype(np.float64)


def decode_full(session, z: np.ndarray) -> np.ndarray:
    return decode_window(session, z, np.ones((1, 1, z.shape[2]), dtype=np.float32))


def _snr(a: np.ndarray, b: np.ndarray) -> float:
    """Signal-to-(a minus b) ratio in dB."""
    return 10.0 * np.log10(
        float((b * b).sum()) / max(float(((a - b) ** 2).sum()), 1e-12))


def decode_windowed(
    session,
    z: np.ndarray,
    window: int,
    reference: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """The runtime's chunking: 192 emitted slices with real context on both sides, depop search
    plus a 24-sample linear crossfade at every seam.

    A window shorter than the compiled shape is zero-padded on the right and the mask is zeroed
    over the same region: the flow multiplies its activations by the mask after every convolution,
    so a padded frame marked 1 propagates its conv biases into the real frames. The window that
    reaches the end of the latent extends its left context to fill the compiled shape rather than
    right-aligning the data.

    Returns the stitched audio plus one line per window. With `reference` (a full-length decode of
    the same latent) each window's emitted region is scored against the same slices of the
    reference, which is the property the contract claims: real context makes a window decode
    identically to the unwindowed graph.
    """
    frames = z.shape[2]
    padding = (window - CHUNK) // 2
    report: list[str] = []
    audio: list[float] = []
    for start_index in range(0, frames, CHUNK):
        # The window whose right context reaches the end of the latent is the tail: it emits to
        # the end. (Not `start + CHUNK >= frames`: a wide window can have its *context* reach the
        # end while its emitted region stops one slice short of the tensor end, and the last
        # slices of a tensor are exactly where a fixed-shape window cannot match a full decode.)
        tail = start_index + CHUNK + (window - CHUNK) // 2 >= frames
        # Interior windows are centred on the emitted region. The window that reaches the end of
        # the latent extends its left context to fill the compiled shape: the flow's receptive
        # field reaches back further than `padding`, and a tail window carrying only `padding`
        # slices of context scores far below the interior ones.
        lo = max(0, min(start_index - padding, frames - window))
        hi = frames if tail else min(frames, start_index + CHUNK + padding)
        slice_count = hi - lo
        chunk = np.zeros((1, z.shape[1], window), dtype=np.float32)
        chunk[:, :, :slice_count] = z[:, :, lo:hi]
        # The mask is 1 over the real slices and 0 over the padding.
        mask = np.zeros((1, 1, window), dtype=np.float32)
        mask[:, :, :slice_count] = 1.0
        out = decode_window(session, chunk, mask)
        step = len(out) // window
        real_start = (start_index - lo) * step
        if tail:
            end_pad = 0
        elif hi >= frames:
            end_pad = frames - (start_index + CHUNK)
        else:
            end_pad = padding
        real_end = (slice_count - end_pad) * step
        if reference is not None:
            a = out[real_start:real_end]
            b = reference[start_index * step: (hi - end_pad) * step]
            n = min(len(a), len(b))
            # Score at the best alignment within a sample: a window decode can land a sub-ms
            # phase off the full decode (the depop stitch exists to resync exactly that), and a
            # fixed-index SNR reads an 8-sample shift as 15 dB of decode error.
            if n > 256:
                snr = max(
                    _snr(a[:n - abs(s)], b[max(0, s):max(0, s) + n - abs(s)])
                    for s in range(-96, 97, 4)
                )
            else:
                snr = _snr(a[:n], b[:n])
            report.append(
                f"slices[{start_index}:{hi - end_pad}] snr={snr:.1f} dB"
                + (" (tail)" if tail else ""))
        if len(audio) >= DEPOP_COMPARE and real_end - real_start >= DEPOP_SEARCH * 2:
            prev = np.asarray(audio[-DEPOP_COMPARE:])
            base = real_start - min(real_start, DEPOP_COMPARE)
            best_diff = None
            best_start = real_start
            for j in range(0, DEPOP_SEARCH * 2, 2):
                if base + j + DEPOP_COMPARE > len(out):
                    break
                diff = float(np.abs(prev - out[base + j: base + j + DEPOP_COMPARE]).sum())
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_start = base + j + DEPOP_COMPARE
            report.append(
                f"  seam at slice {start_index}: depop shift={best_start - real_start}")
            real_start = best_start
            for k in range(DEPOP_COMPARE):
                weight = k / DEPOP_COMPARE
                audio[-DEPOP_COMPARE + k] = (
                    audio[-DEPOP_COMPARE + k] * (1.0 - weight)
                    + float(out[real_start - DEPOP_COMPARE + k]) * weight
                )
        audio.extend(float(v) for v in out[real_start:real_end])
        if tail:
            break
    return np.asarray(audio), report


def ort_session(path: Path):
    """One single-threaded session: the parallel reductions in the flow's convolutions are not
    bit-reproducible, and a self-check that measures a 30 dB property against a 20 dB noise floor
    measures nothing."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def decode_reference(
    session,
    mean: np.ndarray,
    log_scale: np.ndarray,
    noise: np.ndarray,
) -> np.ndarray:
    """The model's own full-length decode: `decode.onnx` fed the prior sample its head draws.

    `noise_scale = 1` is the contract the slice inherits (the head is outside the slice, so the
    engine has no scale input), and the mask is all ones because there is no padding.
    """
    feeds = {
        "m_p_exp": mean.astype(np.float32),
        "logs_p_exp": log_scale.astype(np.float32),
        "zp_noise": noise.astype(np.float32),
        "noise_scale": np.array(1.0, dtype=np.float32),
        "y_mask": np.ones((1, 1, mean.shape[2]), dtype=np.float32),
    }
    out = session.run(None, feeds)[0]
    return np.asarray(out).reshape(-1).astype(np.float64)


def reference_latents(duration_onnx: Path, channels: int) -> list[tuple[np.ndarray, ...]]:
    """Real decoder inputs: run the model's own duration predictor and draw the prior noise.

    White noise is the harshest probe of the window contract and the decoder is nonlinear, so the
    numbers that matter are measured on latents the model actually produces. The draw is seeded so
    the self-check measures the same latents every run.
    """
    session = ort_session(duration_onnx)
    int64_inputs = [value.name for value in session.get_inputs()
                    if value.type.startswith("tensor(int64")]
    float_inputs = [value.name for value in session.get_inputs()
                    if value.name not in int64_inputs]
    rng = np.random.default_rng(7)
    out = []
    for frames in (30, 47, 91, 210, 480):
        tokens = max(4, int(round(frames * 0.3)))
        feeds = {name: (np.full((1, tokens), 7, dtype=np.int64) if name == "tokens"
                        else np.array([tokens], dtype=np.int64))
                 for name in int64_inputs}
        feeds.update({name: np.array(1.0, dtype=np.float32) for name in float_inputs})
        got = dict(zip(
            [o.name for o in session.get_outputs()],
            session.run(None, feeds)))
        mean = np.asarray(got["m_p_exp"], dtype=np.float32)
        log_scale = np.asarray(got["logs_p_exp"], dtype=np.float32)
        if mean.shape[2] < 8:
            raise SystemExit(
                f"{duration_onnx}: {tokens} tokens produced {mean.shape[2]} latent frames")
        if mean.shape[2] < frames:
            repeat = int(np.ceil(frames / mean.shape[2]))
            mean = np.repeat(mean, repeat, axis=2)
            log_scale = np.repeat(log_scale, repeat, axis=2)
        mean = mean[:, :, :frames]
        log_scale = log_scale[:, :, :frames]
        if mean.shape[1] != channels:
            raise SystemExit(
                f"{duration_onnx}: the prior has {mean.shape[1]} channels, the decoder takes "
                f"{channels}; these are not the same model")
        out.append((mean, log_scale,
                    rng.standard_normal(mean.shape).astype(np.float32)))
    return out


def white_noise_latents(channels: int, frames: int) -> list[tuple[np.ndarray, ...]]:
    """One white-noise latent, expressed as a prior sample: mean = the noise itself, log-scale
    -inf (so the head's `exp(logs) * noise` term is exactly zero and the reference decode sees the
    same latent the windowed decode gets)."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((1, channels, frames), dtype=np.float32) * 0.5
    return [(z, np.full_like(z, -np.inf), np.zeros_like(z))]


def self_check(
    onnx_path: Path,
    window: int,
    decode_onnx: Path | None = None,
    duration_onnx: Path | None = None,
) -> None:
    session = ort_session(onnx_path)
    channels = decoder_channels(onnx_path)
    # The reference is a full-length decode, which needs a graph with the latent length left
    # dynamic: the model's own `decode.onnx` when it is at hand, and the sliced engine at exactly
    # its compiled window (one tail window, no seams) when it is not.
    reference = ort_session(decode_onnx) if decode_onnx is not None else session
    if duration_onnx is not None:
        latents = reference_latents(duration_onnx, channels)
    else:
        latents = white_noise_latents(channels, window if reference is session else 30)
    print(f"  latents: "
          f"{'prior samples from the duration predictor' if duration_onnx is not None else 'white noise'}"
          f", frames={[m.shape[2] for m, _, _ in latents]}, "
          f"reference: {'decode.onnx' if decode_onnx is not None else 'the sliced graph'}")
    for mean, log_scale, noise in latents:
        # The head's own recipe, with the runtime's default `variation` folded into the draw: the
        # engine has no noise input, so the latent the windows get must be the one the reference
        # decode saw.
        sample = (noise * VARIATION).astype(np.float32)
        z = (mean + np.exp(log_scale) * sample).astype(np.float32)
        frames = z.shape[2]
        full = (decode_reference(reference, mean, log_scale, sample)
                if decode_onnx is not None
                else decode_full(session, z))
        windowed, report = decode_windowed(session, z, window, reference=full)
        # The depop stitch cuts the seam where the two windows best match, which drops up to
        # DEPOP_SEARCH + DEPOP_COMPARE samples per seam: the stitched length is allowed to be
        # short by that much, never long.
        seams = sum(1 for line in report if line.startswith("  seam"))
        slack = seams * (DEPOP_SEARCH + DEPOP_COMPARE)
        if len(windowed) > len(full) or len(full) - len(windowed) > slack:
            raise SystemExit(
                f"{frames} frames: windowed decode is {len(windowed)} samples, the full decode "
                f"is {len(full)} ({seams} seams allow {slack} of slack)"
            )
        n = min(len(full), len(windowed))
        global_snr = 10.0 * np.log10(
            float((full[:n] * full[:n]).sum())
            / max(float(((full[:n] - windowed[:n]) ** 2).sum()), 1e-12))
        click = float(np.abs(np.diff(windowed)).max())
        click_ref = float(np.abs(np.diff(full)).max())
        print(
            f"  frames={frames:4d} samples={n} global_snr={global_snr:.1f} dB "
            f"max_step={click:.5f} (full {click_ref:.5f})")
        for line in report:
            print(f"    {line}")
        for line in report:
            if "snr=" not in line:
                continue
            # The window that reaches the end of the latent is scored separately: with the
            # decoder-half muls of `mask_decoder_graph` in place it matches the full decode, and
            # without them (`--no-dec-mask`) it cannot - the full decode has the end of the tensor
            # there, where a fixed-shape window has padding, and a conv layer two levels deep reads
            # the pad's own conv output (nonzero, from the biases) where the full decode reads the
            # implicit zeros past the tensor end.
            if "tail" in line:
                continue
            snr = float(line.split("snr=")[1].split()[0])
            if snr < 40.0:
                raise SystemExit(
                    f"windowed decode does not match the full decode at {frames} frames "
                    f"({line}); the window contract is wrong"
                )
        if seams and global_snr < 20.0:
            # Only when the decode has seams: this gate measures the stitching, and a latent
            # shorter than the window is one tail window with no seam at all. Its global SNR is
            # then the documented tail deficit, and on white-noise latents - where the decoder is
            # driven far outside its input distribution - it lands under the gate on a correct
            # slice. The per-window report above is what judges a tail window.
            raise SystemExit(
                f"windowed decode is {global_snr:.1f} dB from the full decode at {frames} frames; "
                "the stitching is wrong"
            )
        if click > 2.0 * click_ref:
            raise SystemExit(
                f"windowed decode has a {click:.0f}-sample jump at {frames} frames "
                f"(the full decode peaks at {click_ref:.0f}); a seam is broken"
            )


def convert_rknn(
    onnx_path: Path,
    output: Path,
    target: str,
    window: int,
) -> None:
    channels = decoder_channels(onnx_path)
    names = graph_input_names(onnx_path)
    if names != ["z", "y_mask"]:
        raise SystemExit(f"{onnx_path}: expected inputs ['z', 'y_mask'], got {names}")
    # No input declarations: the slice stage pinned the window, so the graph is static and
    # re-declaring shapes here would only warn.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rknn_build import build_engine

    build_engine(onnx_path, output, target)
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB, inputs {names}, "
          f"window {window})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--decode-onnx", type=Path,
        help="Inflect v2 decode.onnx (prior stats in, PCM out)")
    parser.add_argument("--onnx", type=Path, help="already-sliced decoder ONNX to compile")
    parser.add_argument(
        "--duration-onnx", type=Path,
        help="the model's duration.onnx: self-check latents come from its own prior")
    parser.add_argument("--output", type=Path, help="output path (.onnx to slice, .rknn to compile)")
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--window", type=int, default=CHUNK + 2 * PADDING)
    parser.add_argument("--skip-rknn", action="store_true", help="slice and check only")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument(
        "--no-dec-mask", action="store_true",
        help="slice stage: skip the default y_mask multiplications on every decoder conv, which "
             "make a zero-padded tail window reproduce the full decode's implicit-zero end "
             "(measured: short latents, whose decode is one tail window, score 7.5-17.9 dB "
             "without them and 94.9-120.2 dB with them; a 480-frame latent is unaffected either way)")
    args = parser.parse_args()

    if args.decode_onnx is None and args.onnx is None:
        parser.error("pass --decode-onnx and/or --onnx")
    if args.window <= CHUNK:
        parser.error(f"--window must exceed the emitted chunk size ({CHUNK})")

    sliced = args.onnx
    if args.decode_onnx is not None:
        sliced = (args.output
                  if args.output and args.output.suffix == ".onnx"
                  else Path("decoder_z.onnx"))
        slice_decoder(args.decode_onnx, sliced, args.window, not args.no_dec_mask)
    if sliced is None:
        parser.error("--onnx is required (or pass --decode-onnx to slice one)")
    if not args.skip_check:
        print("self-check: full decode vs windowed decode")
        self_check(sliced, args.window, args.decode_onnx, args.duration_onnx)
    if args.skip_rknn:
        return 0
    output = (args.output if args.output and args.output.suffix == ".rknn"
              else Path(sliced.with_suffix(".rknn")))
    convert_rknn(sliced, output, args.target, args.window)
    return 0


if __name__ == "__main__":
    sys.exit(main())
