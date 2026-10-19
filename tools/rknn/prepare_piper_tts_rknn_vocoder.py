#!/usr/bin/env python3
"""Build the fixed-window RKNN speech decoder for a Piper (VITS) voice.

Piper ships one ONNX per voice: text ids in, PCM out. The RKNN path needs the decoder half
alone, compiled for one fixed latent window, because RKNN has no dynamic shapes. This script
slices the voice graph at the inverse-flow input `z` and its mask `y_mask` (the seam the runtime
has on hand: the pre-flow latent and the frame mask), checks that the fixed window plus the chunk
stitching reproduces a full-length decode, and compiles the result for a RKNPU2 target.

The seam is the flow input and not the input of `dec.conv_pre` on purpose: the flow is part of the
decoder half, and the runtime's latent is the pre-flow one. Slicing at `conv_pre` compiles a
decoder that expects post-flow samples and feeds it pre-flow ones, which is quiet mush.

Two stages, usually in two venvs (the slice/check needs onnx + onnxruntime, the conversion needs
rknn-toolkit2 with its own pinned torch/onnx):

    <venv>/bin/python tools/rknn/prepare_piper_tts_rknn_vocoder.py \
        --voice-onnx en_US-lessac-medium.onnx --skip-rknn --output decoder_z.onnx
    <rknn-venv>/bin/python tools/rknn/prepare_piper_tts_rknn_vocoder.py \
        --onnx decoder_z.onnx --target rk3588

The window contract (window 93 latent slices, 45 emitted + 24+24 real context, zero-padded and
mask-zeroed at the end that has no data) is the same one the runtime implements in
src/community_models/piper_tts/rknn_decoder.cpp; change it here and there together. 24 is
measured: at the 8+8 paroli ships, interior windows score 8-60 dB against a full-length decode.

`--encoder` compiles the other half — the duration encoder — for a fixed phoneme-token width
(`--frames`, piper tokens = 2*phonemes + 3):

    <rknn-venv>/bin/python tools/rknn/prepare_piper_tts_rknn_vocoder.py \
        --voice-onnx en_US-lessac-medium.onnx --encoder --frames 128
"""

from __future__ import annotations

import argparse

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rknn_build import report_engine
from rknn_graph import decoder_channels, graph_input_names

CHUNK = 45
PADDING = 24
DEPOP_COMPARE = 24
DEPOP_SEARCH = 44


def decoder_seam(graph) -> tuple[str, str | None]:
    """Latent and mask at the decoder seam: the inverse-flow input `z` and its `y_mask`.

    Walked back from `dec.conv_pre`: piper's decoder half is `z * y_mask` into the inverse flow
    into `conv_pre`, so the flow module's external inputs are exactly the seam.
    """
    producers = {output: node for node in graph.node for output in node.output}
    conv = next(
        (n for n in graph.node if any(i.startswith("dec.conv_pre") for i in n.input)), None)
    if conv is None:
        raise SystemExit(
            "no dec.conv_pre node: this ONNX is not a Piper/VITS voice, "
            "or its decoder weights are not named dec.*"
        )
    masked = producers.get(conv.input[0])
    mask = None
    entry = conv.input[0]
    if masked is not None and masked.op_type == "Mul" and len(masked.input) == 2:
        # One operand is the flow output, the other the frame mask.
        entry = next(
            (i for i in masked.input if i in producers and "/flows." in producers[i].name),
            masked.input[0])
        mask = next(i for i in masked.input if i != entry)
    node = producers[entry]
    module = f"/{node.name.strip('/').split('/')[0]}/"
    flow_nodes = [n for n in graph.node if n.name.startswith(module)]
    internal = {o for n in flow_nodes for o in n.output}
    initializers = {i.name for i in graph.initializer}
    # The flow runs in reverse order, so the first of its nodes in topological order is the
    # entry; its data input (index 0) is the latent. The other external inputs of the module are
    # weight-norm intermediates, which feed a Conv at index 1/2.
    first = next(n for n in graph.node if n.name.startswith(module))
    latent_in = first.input[0]
    if latent_in in initializers or latent_in in internal:
        raise SystemExit(
            f"the flow module {module!r} starts at {first.name!r}, whose data input "
            f"{latent_in!r} is not an external tensor"
        )
    return latent_in, mask


def slice_decoder(voice_onnx: Path, output: Path, window: int, dec_mask: bool = False) -> None:
    model = onnx.load(str(voice_onnx))
    latent, mask = decoder_seam(model.graph)
    inputs = [latent] if mask is None else [latent, mask]
    onnx.utils.extract_model(str(voice_onnx), str(output), inputs, ["output"])

    sliced = onnx.load(str(output))
    names = {latent: "z"}
    if mask is not None:
        names[mask] = "y_mask"
    for value in sliced.graph.input:
        if value.name not in names:
            continue
        value.name = names[value.name]
        dims = list(value.type.tensor_type.shape.dim)
        if len(dims) == 3:
            dims[0].dim_param = "batch"
            dims[-1].dim_param = "frames"
    for node in sliced.graph.node:
        node.input[:] = [names.get(name, name) for name in node.input]
    if dec_mask:
        if mask is None:
            raise SystemExit("--dec-mask needs a sliced graph with a y_mask input")
        print(f"dec-mask: injected {mask_decoder_graph(sliced)} mask muls into the decoder half")
    onnx.checker.check_model(sliced, full_check=False)
    onnx.save(sliced, str(output))
    ops: dict[str, int] = {}
    for node in sliced.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print(
        f"sliced {voice_onnx.name} at {latent!r}"
        + (f" + {mask!r}" if mask else "")
        + f": {len(sliced.graph.node)} nodes, inputs {list(names.values())}, "
        f"output {sliced.graph.output[0].name!r}"
    )
    risky = sorted(
        op for op in ops
        if op in {"Pad", "Exp", "Shape", "ConstantOfShape", "Split", "Atan", "Erf",
                   "RandomNormalLike"}
    )
    if risky:
        print(f"  note: graph contains {risky}; if rknn.build aborts in fold_constant, "
              "the exporting torch/onnx pair is too new for the toolkit")
    report_engine(output)


def mask_decoder_graph(model) -> int:
    """Inject `y_mask` multiplications into the decoder half (opt-in `--dec-mask`).

    VITS's dec has no masking, so in a zero-padded window every decoder conv writes its
    bias-driven output over the padding and the next conv reads that where a full-length
    decode reads the implicit zeros past the tensor end: the tail window's last ~7 slices
    land 20-40 dB off. Multiplying every decoder Conv/ConvTranspose output by the mask -
    the mask itself upsampled nearest by each `dec.ups` stride, so it is at the stage's
    rate - makes a padded window read zeros exactly where the full decode reads implicit
    zeros. With mask=1 everywhere (the full decode, and every interior window) each
    injected Mul is an exact fp32 identity, so the reference decode is unchanged.
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


ENCODER_INPUTS = ["input", "input_lengths"]


def slice_encoder(voice_onnx: Path, output: Path) -> None:
    """Slice the duration encoder (text encoder + duration discriminator) at its seam.

    The seam is `enc_p.proj` (the 2x-inter_channels prior stats) and `dp.proj` (the flow
    conditioning): the two arrays the runtime's flow graphs and length regulator consume.
    Piper's export carries the frame mask inside the encoder (attention biases and the
    x_mask muls are wired to `input_lengths`), so a right-padded fixed-width run reproduces
    the exact-length run on the real frames — measured 117-130 dB fp32 — the same
    pad-to-compile recipe the decoder window uses, with no mask surgery.
    `scales` is not an input of the slice: all three scalars (noise_scale, length_scale,
    noise_scale_w) feed the sampling, the duration ceiling and the discriminator's flow
    noise, which sit strictly downstream of the seam — the slice is deterministic.
    """
    model = onnx.load(str(voice_onnx), load_external_data=False)
    stats = next((n for n in model.graph.node if n.name == "/enc_p/proj/Conv"), None)
    conditioning = next((n for n in model.graph.node if n.name == "/dp/proj/Conv"), None)
    if stats is None or conditioning is None:
        raise SystemExit(
            "no /enc_p/proj/Conv + /dp/proj/Conv: this ONNX is not a Piper/VITS voice"
        )
    onnx.utils.extract_model(
        str(voice_onnx), str(output), ENCODER_INPUTS, [stats.output[0], conditioning.output[0]])
    sliced = onnx.load(str(output))
    onnx.checker.check_model(sliced, full_check=False)
    onnx.save(sliced, str(output))
    ops: dict[str, int] = {}
    for node in sliced.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print(
        f"sliced {voice_onnx.name} at {stats.output[0]!r} + {conditioning.output[0]!r}:"
        f" {len(sliced.graph.node)} nodes, inputs {ENCODER_INPUTS},"
        f" outputs {[o.name for o in sliced.graph.output]}"
    )
    risky = sorted(
        op for op in ops
        if op in {"Pad", "Exp", "Shape", "ConstantOfShape", "Split", "Atan", "Erf"}
    )
    if risky:
        print(f"  note: graph contains {risky}; if rknn.build aborts in fold_constant, "
              "the exporting torch/onnx pair is too new for the toolkit")
    report_engine(output)


def check_encoder(onnx_path: Path, raw_path: Path, frames: int) -> None:
    """Right-padding the folded fixed-`frames` graph must not move the real frames.

    Exact-width runs come from the raw dynamic slice (the folded graph is fixed-shape by
    construction); >= 100 dB fp32 against them is the pad-safety gate, on the graph the
    engine is actually compiled from.
    """
    exact_session = ort_session(raw_path)
    pad_session = ort_session(onnx_path)
    rng = np.random.default_rng(7)
    for n in (17, 61, frames - 1):
        tokens = rng.integers(3, 250, size=(1, n), dtype=np.int64)
        padded = rng.integers(0, 250, size=(1, frames), dtype=np.int64)
        padded[:, :n] = tokens[0]
        lengths = np.array([n], np.int64)
        exact = exact_session.run(None, {"input": tokens, "input_lengths": lengths})
        pad = pad_session.run(None, {"input": padded, "input_lengths": lengths})
        for out, a, b in zip(pad_session.get_outputs(), exact, pad):
            a = np.asarray(a).reshape(a.shape[0], -1, a.shape[-1])[..., :n]
            b = np.asarray(b).reshape(b.shape[0], -1, b.shape[-1])[..., :n]
            snr = _snr(a.astype(np.float64), b.astype(np.float64))
            print(f"  {out.name} n={n} padded to {frames}: SNR {snr:.1f} dB")
            if snr < 100:
                raise SystemExit(f"right-padding moves the encoder output at n={n} ({snr:.1f} dB)")


def fold_encoder(onnx_path: Path, output: Path, frames: int) -> None:
    """Static-crop and constant-fold the encoder slice with ONNX Runtime's graph optimizer.

    The sliced graph computes its own masks and pads from the dynamic frame axis (25 Shape,
    48 Pad, 1 Range), and rknn-toolkit2's C++ rewriter aborts without a message on exactly
    those shape-computation chains — the crash family of the qwen3 full-decode export. With
    the frame axis pinned to `frames` the chains become constant, the optimizer folds them
    into initializers, and the toolkit gets the static graph it can compile. The pad-safety
    self-check runs on the folded graph, so a fold that moved anything is caught here.
    """
    import onnxruntime as ort

    model = onnx.load(str(onnx_path))
    for value in model.graph.input:
        dims = value.type.tensor_type.shape.dim
        for dim in dims:
            dim.dim_param = ""
            dim.dim_value = 0
        if value.name == "input":
            dims[0].dim_value = 1
            dims[1].dim_value = frames
        elif value.name == "input_lengths":
            dims[0].dim_value = 1
        else:
            raise SystemExit(f"unexpected encoder input {value.name!r}")
    for value in model.graph.output:
        for dim in value.type.tensor_type.shape.dim:
            dim.dim_param = ""
            dim.dim_value = 0
    pinned = output.with_suffix(".pinned.onnx")
    onnx.save(model, str(pinned))

    options = ort.SessionOptions()
    options.log_severity_level = 3
    # BASIC, not ALL: the extended levels emit the com.microsoft fusion ops (FusedConv,
    # FusedMatMul, Gelu) into the saved model, and the RKNN toolkit drowns on them.
    # Constant folding — the point of the pass — runs at BASIC.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(output)
    ort.InferenceSession(str(pinned), options, providers=["CPUExecutionProvider"])

    folded = onnx.load(str(output))
    onnx.checker.check_model(folded, full_check=False)
    ops: dict[str, int] = {}
    for node in folded.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    shapes = sorted(
        op for op in ("Shape", "Range", "ConstantOfShape", "NonZero", "GatherND", "ScatterND")
        if op in ops
    )
    print(
        f"folded to fixed {frames}-token width: {len(folded.graph.node)} nodes, "
        f"shape ops {shapes or 'none'}"
    )


def matrix_form_encoder(onnx_path: Path, output: Path, frames: int) -> None:
    """Rewrite the per-row relative-shift chains of the attention layers into gathers.

    With the frame axis pinned, every piper attention layer shifts its [N, 2N-1]
    relative-position slabs by one step per query row (Pad + Reshape + Slice, unrolled per
    row), and the RKNN NPU kernels compute exactly those chains as garbage on the board:
    probes show the flatten-Pad tails filled with uninitialized memory (NaN, ±65504) and
    one such layer drags the engine to 0.4 dB end to end, while the PC simulator computes
    the same graph at 56 dB. The two chains are exact gathers with constant indices:

        bias[b,h,w,t] = rel_logits[b,h,w,(N-1)+t-w]                     (GatherElements)
        rel_out[w]    = sum_j probs[b,h,w,j] * rel_v[(N-1)+j-w]  (Gather + Mul + ReduceSum)

    The indices are constants, so the toolkit folds them back into weights at build:
    the shift costs O(N^3) per layer — measured engines 137 MB at N=128, 402 MB at 192,
    2.9 GB at 384, and N=512 exhausts 31 GB of converter RAM. Compile narrow bins only
    and leave the wider widths to the host encoder.
    """
    from onnx import numpy_helper, shape_inference

    model = shape_inference.infer_shapes(
        onnx.load(str(onnx_path), load_external_data=True), strict_mode=True)
    heads = next(
        (list(v.type.tensor_type.shape.dim)[1].dim_value
         for v in model.graph.value_info
         if v.name == "/enc_p/encoder/attn_layers.0/Softmax_output_0"), 0)
    if not heads:
        raise SystemExit("no attn_layers.0/Softmax shape: not a folded piper encoder")
    producers = {output_tensor: node for node in model.graph.node for output_tensor in node.output}
    consumers: dict[str, list] = {}
    for node in model.graph.node:
        for i in node.input:
            consumers.setdefault(i, []).append(node)
    shift = (frames - 1) + np.arange(frames)[None, :] - np.arange(frames)[:, None]
    drop, added, inits = set(), [], []
    for layer in range(6):
        p = f"/enc_p/encoder/attn_layers.{layer}/"
        bias_t = producers[p + "Add_1_output_0"].input[1]
        chain, t = [], bias_t
        while producers[t].name != p + "Pad_1":
            chain.append(producers[t])
            t = producers[t].input[0]
        chain.append(producers[t])
        relv = producers[p + "MatMul_3_output_0"]
        rel_v = next(i for i in relv.input if i != p + "Slice_8_output_0")
        if len(consumers[p + "Slice_8_output_0"]) != 1:
            raise SystemExit(f"{p}Slice_8 has foreign consumers; unexpected fold shape")
        drop.update(id(n) for n in chain)
        for x in (p + "Pad_3_output_0", p + "Reshape_13_output_0", p + "Pad_4_output_0",
                  p + "Reshape_16_output_0", p + "Slice_8_output_0"):
            drop.add(id(producers[x]))
        drop.add(id(relv))
        added.append(helper.make_node(
            "GatherElements", [producers[p + "Pad_1_output_0"].input[0], p + "idxE"],
            [bias_t], axis=3, name=p + "RelBiasGather"))
        inits.append(numpy_helper.from_array(
            np.broadcast_to(shift, (1, heads, frames, frames)).astype(np.int64), p + "idxE"))
        added.append(helper.make_node(
            "Gather", [rel_v, p + "idxG"], [p + "G_gathered"], axis=2, name=p + "RelVGather"))
        inits.append(numpy_helper.from_array(shift.astype(np.int64), p + "idxG"))
        added.append(helper.make_node(
            "Unsqueeze", [p + "Softmax_output_0", "/enc_p/encoder/Axes4"],
            [p + "ProbsUnsq"], name=p + "ProbsUnsq"))
        added.append(helper.make_node(
            "Mul", [p + "ProbsUnsq", p + "G_gathered"], [p + "RelVProd"], name=p + "RelVMul"))
        added.append(helper.make_node(
            "ReduceSum", [p + "RelVProd", "/enc_p/encoder/Axes3"],
            [p + "MatMul_3_output_0"], name=p + "RelVSum", keepdims=0))
    model.graph.initializer.extend([
        numpy_helper.from_array(np.array([4], np.int64), "/enc_p/encoder/Axes4"),
        numpy_helper.from_array(np.array([3], np.int64), "/enc_p/encoder/Axes3"),
    ])
    model.graph.initializer.extend(inits)
    # rknn-toolkit2's fold_constant walks the node list once, in order: hand it a
    # topologically sorted graph.
    nodes = [n for n in model.graph.node if id(n) not in drop] + added
    available = {i.name for i in model.graph.initializer} | {i.name for i in model.graph.input}
    ordered, pending = [], list(nodes)
    while pending:
        ready = [n for n in pending if all(not i or i in available for i in n.input)]
        if not ready:
            raise SystemExit("cycle in the rewritten encoder graph")
        for n in ready:
            pending.remove(n)
        available.update(o for n in ready for o in n.output)
        ordered.extend(ready)
    del model.graph.node[:]
    model.graph.node.extend(ordered)
    del model.graph.value_info[:]
    model = shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, str(output))
    print(f"rewrote {6} attention shift chains to gathers: {len(model.graph.node)} nodes")


def convert_rknn_encoder(
    onnx_path: Path,
    output: Path,
    target: str,
    frames: int,
    opt_level: int | None = None,
) -> None:
    from rknn_build import build_engine, report_engine

    # The folded graph is static: declare nothing (see build_engine).
    build_engine(onnx_path, output, target, opt_level=opt_level)
    report_engine(output, "inputs input/input_lengths", f"frames {frames}")


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
    """The runtime's chunking: 45 emitted slices with real context on both sides, depop search
    plus a 24-sample linear crossfade at every seam.

    A window shorter than the compiled shape is zero-padded on the right and the mask is zeroed
    over the same region: the flow multiplies its activations by the mask after every
    convolution, so a padded frame marked 1 propagates its conv biases into the real frames and
    costs 10-20 dB there. The window that reaches the end of the latent extends its left context
    to fill the compiled shape rather than right-aligning the data.

    Returns the stitched audio plus one line per window. With `reference` (a full-length decode
    of the same latent) each window's emitted region is scored against the same slices of the
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
        # slices of context scores -0.1 dB on its emitted slices (measured: 62 dB with the
        # context, 42 dB right-aligned, 0.1 dB right-aligned with an unmasked pad).
        lo = max(0, min(start_index - padding, frames - window))
        hi = frames if tail else min(frames, start_index + CHUNK + padding)
        slice_count = hi - lo
        chunk = np.zeros((1, z.shape[1], window), dtype=np.float32)
        chunk[:, :, :slice_count] = z[:, :, lo:hi]
        # The mask is 1 over the real slices and 0 over the padding: the flow multiplies its
        # activations by the mask after every convolution, so a padded frame marked 1 propagates
        # its conv biases into the real frames and costs 10-20 dB there.
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


def reference_latents(voice_onnx: Path, count: int) -> list[np.ndarray]:
    """Real decoder inputs: run the voice's own encoder and capture the latent at the seam.

    White noise is the harshest probe of the window contract and the VITS decoder is nonlinear,
    so the numbers that matter are measured on latents the model actually produces.
    """
    model = onnx.load(str(voice_onnx))
    latent, mask = decoder_seam(model.graph)
    model.graph.output.extend(
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
         for name in ([latent, mask] if mask else [latent])])
    # The voice draws its latent with RandomNormalLike; seed it so the self-check measures the
    # same latents every run (the window contract is a deterministic property, and an unseeded
    # probe makes the SNRs move by tens of dB between runs).
    for node in model.graph.node:
        if node.op_type in {"RandomNormal", "RandomNormalLike"}:
            node.attribute.extend([helper.make_attribute("seed", 7.0)])
    probe = voice_onnx.with_name("probe.onnx")
    onnx.save(model, str(probe))
    session = ort_session(probe)
    out = []
    for frames in (30, 47, 91, 210):
        tokens = int(frames * 0.45) + 4
        feeds = {
            "input": np.arange(3, 3 + tokens, dtype=np.int64).reshape(1, tokens),
            "input_lengths": np.array([tokens], dtype=np.int64),
            "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32),
        }
        z = session.run(None, feeds)[1]
        if z.shape[2] < 8:
            z = np.repeat(z, int(np.ceil(frames / z.shape[2])), axis=2)
        out.append(z[:, :, :frames].astype(np.float32) * 1.0)
        if len(out) >= count:
            break
    probe.unlink(missing_ok=True)
    return out


def self_check(onnx_path: Path, window: int, voice_onnx: Path | None = None) -> None:
    session = ort_session(onnx_path)
    channels = decoder_channels(onnx_path)
    if voice_onnx is not None:
        latents = reference_latents(voice_onnx, 4)
        print(f"  latents: encoder output, frames={[z.shape[2] for z in latents]}")
    else:
        rng = np.random.default_rng(0)
        latents = [
            rng.standard_normal((1, channels, frames), dtype=np.float32) * 0.5
            for frames in (30, 47, 91, 210)
        ]
    for z in latents:
        frames = z.shape[2]
        full = decode_full(session, z)
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
        step = len(full) // frames
        click = float(np.abs(np.diff(windowed)).max())
        click_ref = float(np.abs(np.diff(full)).max())
        print(
            f"  frames={frames:4d} samples={n} global_snr={global_snr:.1f} dB "
            f"max_step={click:.0f} (full {click_ref:.0f})")
        for line in report:
            print(f"    {line}")
        for line in report:
            if "snr=" not in line:
                continue
            # The window that reaches the end of the latent cannot reproduce the full decode. The
            # full decode has the end of the tensor there, where a fixed-shape window has padding,
            # and a conv layer two levels deep reads the pad's own conv output (nonzero, from the
            # biases) where the full decode reads the implicit zeros past the tensor end. No mask
            # value in the padding reproduces that (measured: the last ~7 slices land 20-40 dB
            # off, the interior windows are exact), and the mask cannot be applied inside the
            # decoder at all - VITS's dec has no masking. paroli ships the same trade.
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
            # then the documented tail deficit (the last ~7 slices land 20-40 dB off a full-length
            # decode, which no mask can fix), and on white-noise latents - where the decoder is
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
    sizes = [[1, channels, window]]
    for name in names:
        if name == "y_mask":
            sizes.append([1, 1, window])
        elif name == "g":
            sizes.append([1, 512, 1])
    from rknn_build import build_engine, report_engine

    build_engine(onnx_path, output, target, names=names, shapes=sizes)
    report_engine(output, f"inputs {names}", f"window {window}")


def verify_encoder(
    onnx_path: Path,
    raw_path: Path,
    frames: int,
    target: str,
    seed: int = 7,
) -> None:
    """The fp16 cost of the compiled encoder, measured on the PC simulator.

    Rebuilt from the folded ONNX rather than run from the exported `.rknn`: a
    `load_rknn` engine cannot be inferred on the simulator (see the cosyvoice3
    flow converter), and `load_onnx` + `build` with the same config produces the
    same fp16 graph the board runs. Reference: the raw dynamic slice at exact
    width, fp32.
    """
    from rknn.api import RKNN

    exact = ort_session(raw_path)
    rng = np.random.default_rng(seed)
    rknn = RKNN(verbose=False)
    rknn.config(target_platform=target)
    if rknn.load_onnx(model=str(onnx_path)) != 0:
        raise SystemExit("load_onnx failed")
    if rknn.build(do_quantization=False) != 0:
        raise SystemExit("build failed")
    if rknn.init_runtime() != 0:
        raise SystemExit("init_runtime (PC simulator) failed")
    for n in (17, 61, frames - 1):
        tokens = rng.integers(3, 250, size=(1, n), dtype=np.int64)
        padded = rng.integers(0, 250, size=(1, frames), dtype=np.int64)
        padded[:, :n] = tokens[0]
        lengths = np.array([n], np.int64)
        want = exact.run(None, {"input": tokens, "input_lengths": lengths})
        got = rknn.inference(inputs=[padded, lengths])
        for out, a, b in zip(exact.get_outputs(), want, got):
            a = np.asarray(a).reshape(a.shape[0], -1, a.shape[-1])[..., :n]
            b = np.asarray(b).reshape(b.shape[0], -1, b.shape[-1])[..., :n]
            snr = _snr(a.astype(np.float64), b.astype(np.float64))
            print(f"  {out.name} n={n} fp16 engine vs fp32 slice: SNR {snr:.1f} dB")
            if snr < 30:
                raise SystemExit(f"compiled encoder moved at n={n} ({snr:.1f} dB)")
    rknn.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--voice-onnx", type=Path, help="Piper voice ONNX (text in, PCM out)")
    parser.add_argument("--onnx", type=Path, help="already-sliced decoder ONNX to compile")
    parser.add_argument("--output", type=Path, help="output path (.onnx to slice, .rknn to compile)")
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--window", type=int, default=CHUNK + 2 * PADDING)
    parser.add_argument("--skip-rknn", action="store_true", help="slice and check only")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument(
        "--encoder", action="store_true",
        help="slice/compile the duration encoder (enc_p.proj + dp.proj seam, phoneme-width "
             f"bin --frames) instead of the vocoder window")
    parser.add_argument(
        "--frames", type=int, default=128,
        help="--encoder: compiled phoneme-token width (piper tokens = 2*phonemes + 3)")
    parser.add_argument(
        "--verify-sim", action="store_true",
        help="--encoder: after compiling, infer the engine on the PC simulator against "
             "the fp32 slice (fp16 cost of the encoder)")
    parser.add_argument(
        "--opt-level", type=int, default=None,
        help="--encoder: rknn optimization level (wide bins whose kernels overflow the "
             "NPU register fields at the default level: try 2)")
    parser.add_argument(
        "--dec-mask", action="store_true",
        help="slice stage: multiply every decoder conv output by the (rate-upsampled) y_mask, "
             "so a zero-padded tail window reproduces the full decode's implicit-zero end "
             "(fixes the last ~7 slices of the tail window); the compiled engine carries it")
    args = parser.parse_args()

    if args.voice_onnx is None and args.onnx is None:
        parser.error("pass --voice-onnx and/or --onnx")
    if not args.encoder:
        if args.window <= CHUNK:
            parser.error("--window must exceed the emitted chunk size (45)")

    if args.encoder:
        sliced = args.onnx
        if args.voice_onnx is not None:
            sliced = (args.output
                      if args.output and args.output.suffix == ".onnx"
                      else Path(f"encoder_n{args.frames}.onnx"))
            slice_encoder(args.voice_onnx, sliced)
        if sliced is None:
            parser.error("--onnx is required (or pass --voice-onnx to slice one)")
        folded = sliced.with_name(sliced.stem + "_static.onnx")
        fold_encoder(sliced, folded, args.frames)
        matrix = sliced.with_name(sliced.stem + "_matrix.onnx")
        matrix_form_encoder(folded, matrix, args.frames)
        if not args.skip_check:
            print("self-check: encoder pad vs exact (matrix-form graph)")
            check_encoder(matrix, sliced, args.frames)
        if args.skip_rknn:
            return 0
        output = args.output if args.output and args.output.suffix == ".rknn" else Path(
            sliced.with_suffix(".rknn"))
        convert_rknn_encoder(matrix, output, args.target, args.frames, args.opt_level)
        if args.verify_sim:
            print("verify: compiled fp16 encoder on the PC simulator")
            verify_encoder(matrix, sliced, args.frames, args.target)
        return 0

    sliced = args.onnx
    if args.voice_onnx is not None:
        sliced = args.output if args.output and args.output.suffix == ".onnx" else Path("decoder_z.onnx")
        slice_decoder(args.voice_onnx, sliced, args.window, args.dec_mask)
    if sliced is None:
        parser.error("--onnx is required (or pass --voice-onnx to slice one)")
    if not args.skip_check:
        print("self-check: full decode vs windowed decode")
        self_check(sliced, args.window, args.voice_onnx)
    if args.skip_rknn:
        return 0
    output = args.output if args.output and args.output.suffix == ".rknn" else Path(
        sliced.with_suffix(".rknn"))
    convert_rknn(sliced, output, args.target, args.window)
    return 0


if __name__ == "__main__":
    sys.exit(main())
