#!/usr/bin/env python3
"""Build the fixed-length RKNN Flow DiT estimator for the CosyVoice3 family.

The runtime side (``src/models/cosyvoice3/rknn_flow.cpp``, session option
``cosyvoice3.rknn_flow``) runs one compiled RKNN estimator step in place of the ggml DiT graph,
inside the same flow-matching ODE loop. Nothing in this repository produces that graph, so this
script is its only producer: it compiles the checkpoint's own
``flow.decoder.estimator.fp32.onnx`` (from ``FunAudioLLM/Fun-CosyVoice3-0.5B-2512``, i.e. the same
weights the GGUF package was converted from) for one fixed sequence length and for a RKNPU2 target.

The estimator is the flow decoder's ODE step: batch 2 is the classifier-free-guidance pair
``[cond, uncond]`` (the uncond branch carries zeroed ``mu``/``cond``/``spks``), and one call returns
``dphi_dt`` for both branches. Six inputs ``x [2,80,S]`` / ``mask [2,1,S]`` / ``mu [2,80,S]`` /
``t [2]`` (the raw timestep; the time-embedding MLP is inside the graph) / ``spks [2,80]`` /
``cond [2,80,S]``, one output ``estimator_out [2,80,S]``, fp32 in and out. The runtime's ggml graph
has no ``mask`` and takes the timestep embedding instead of ``t``; the sidecar supplies both.

RKNN has no dynamic shapes, so the runtime compiles a ladder of lengths and pads a shorter
utterance up to the smallest engine that fits: ``x``/``mu``/``cond`` zero-padded on the right,
``mask`` 1 on the real frames and 0 in the pad, output cropped back. That convention is what the
padding check measures on the host: a padded run at ``S`` against an exact run at ``frames``, per
length, before any of it is paid for on a board (``--skip-check`` skips it). ``--verify-sim`` goes
one step further and runs the fp16 graph on the PC simulator against the fp32 ONNX, which is the
engine's own accuracy, measured without a board.

Requirements (host, x86_64, dedicated venv -- the RKNN toolkit pins its own numpy/onnx):
  pip install onnx onnxruntime numpy
  pip install rknn-toolkit2          # PC-side converter, x86_64 only
On Python 3.12 the 2.3.2 converter additionally needs ``setuptools<81`` and an ``onnx/mapping.py``
shim inside the venv's onnx package; see docs/build/rockchip-npu.md.

Usage:
  python3 tools/rknn/prepare_cosyvoice3_rknn_flow.py --estimator-onnx flow.decoder.estimator.fp32.onnx \
      --seq-len 200 --seq-len 500 --seq-len 1000 --output-dir models/CosyVoice3-RKNN
  python3 tools/rknn/prepare_cosyvoice3_rknn_flow.py --estimator-onnx <onnx> --seq-len 200 --skip-rknn
  python3 tools/rknn/prepare_cosyvoice3_rknn_flow.py --estimator-onnx <onnx> --seq-len 200 \
      --check-frames 106 --verify-sim
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx

MEL_CHANNELS = 80
CFG_BATCH = 2
# Graph order, as exported by the checkpoint. `mask` is the frame mask, `t` the raw timestep.
INPUT_NAMES = ["x", "mask", "mu", "t", "spks", "cond"]


def input_shapes(seq_len: int) -> list[list[int]]:
    return [
        [CFG_BATCH, MEL_CHANNELS, seq_len],
        [CFG_BATCH, 1, seq_len],
        [CFG_BATCH, MEL_CHANNELS, seq_len],
        [CFG_BATCH],
        [CFG_BATCH, MEL_CHANNELS],
        [CFG_BATCH, MEL_CHANNELS, seq_len],
    ]


def graph_input_names(onnx_path: Path) -> list[str]:
    model = onnx.load(str(onnx_path), load_external_data=False)
    names = [i.name for i in model.graph.input]
    # Initializers can appear as graph inputs in older exports; only the real feeders count.
    feed = [n for n in names if n not in {i.name for i in model.graph.initializer}]
    if feed != INPUT_NAMES:
        raise SystemExit(
            f"{onnx_path.name}: expected inputs {INPUT_NAMES} in graph order, got {feed}; "
            "this is not the checkpoint's flow.decoder.estimator export"
        )
    return feed


def pad_to(inputs: list[np.ndarray], seq_len: int) -> list[np.ndarray]:
    """Right-pad a length-carrying input set to `seq_len`: zeros everywhere, mask 0 in the pad."""
    out = []
    for name, arr in zip(INPUT_NAMES, inputs):
        if name in ("t", "spks"):
            out.append(arr)
            continue
        axis = arr.ndim - 1
        pad = seq_len - arr.shape[axis]
        if pad < 0:
            raise SystemExit(f"{name} is longer ({arr.shape[axis]}) than the engine ({seq_len})")
        # Zero pad: for x/mu/cond that is the reference port's `zero` mode, and for mask it is the
        # 0 that switches the pad frames off.
        out.append(arr if pad == 0 else np.pad(arr, [(0, 0), (0, 0), (0, pad)]))
    return out


def random_inputs(seq_len: int, rng: np.random.Generator) -> list[np.ndarray]:
    inputs = []
    for name, shape in zip(INPUT_NAMES, input_shapes(seq_len)):
        if name == "t":
            # The ODE schedule evaluates the estimator at the start of each interval, t in [0, 1).
            inputs.append(rng.random(shape).astype(np.float32))
        elif name == "mask":
            inputs.append(np.ones(shape, dtype=np.float32))
        else:
            inputs.append(rng.standard_normal(shape).astype(np.float32))
    return inputs


def run(session, inputs: list[np.ndarray]) -> np.ndarray:
    feeds = {name: arr for name, arr in zip(INPUT_NAMES, inputs)}
    return session.run(None, feeds)[0]


def snr_db(reference: np.ndarray, other: np.ndarray) -> float:
    power = float((reference * reference).sum())
    err = float(((reference - other) ** 2).sum())
    return 10.0 * np.log10(max(power, 1e-30) / max(err, 1e-30))


def check_padding(onnx_path: Path, seq_len: int, frames: list[int], seed: int) -> None:
    """Padded run at `seq_len` vs exact run at `frames`: the padding convention's own gate."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    print(f"check {seq_len}: padded run vs exact run (onnxruntime CPU, fp32)")
    for length in frames:
        if length > seq_len:
            continue
        inputs = random_inputs(length, rng)
        exact = run(session, inputs)
        padded = run(session, pad_to(inputs, seq_len))[:, :, :length]
        # The frames nearest the pad are the ones the pad can contaminate: the estimator's causal
        # conv-pos-embed pads on the left only, so a right pad reaches the tail through attention.
        tail = padded[:, :, -min(60, length):]
        tail_ref = exact[:, :, -min(60, length):]
        print(
            f"  frames={length:4d} pad={seq_len - length:4d} "
            f"snr={snr_db(exact, padded):6.1f} dB tail_snr={snr_db(tail_ref, tail):6.1f} dB "
            f"max_abs={np.abs(exact - padded).max():.4f}"
        )


def verify_fp16(
    onnx_path: Path,
    seq_len: int,
    frames: int,
    target: str,
    opt_level: int,
    seed: int,
) -> None:
    """The fp16 cost of the compiled graph, measured on the PC simulator.

    Built from the ONNX rather than run from the exported `.rknn`: a `load_rknn` engine cannot be
    inferred on the simulator at all (`init_runtime: RKNN model that loaded by 'load_rknn' not
    support inference on the simulator, please set 'target' first!`), and `load_onnx` + `build`
    with the same config produces the same fp16 graph the board runs.

    Worth measuring before a board is asked to judge it: the whole utterance's mel comes out of ten
    of these steps, so a per-step bias is what the reference port compensates with a hand-tuned mel
    offset.
    """
    import onnxruntime as ort
    from rknn.api import RKNN

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    inputs = pad_to(random_inputs(frames, rng), seq_len)
    reference = run(session, inputs)

    rknn = RKNN(verbose=False)
    rknn.config(target_platform=target, optimization_level=opt_level)
    if rknn.load_onnx(model=str(onnx_path), inputs=INPUT_NAMES, input_size_list=input_shapes(seq_len)) != 0:
        raise SystemExit("load_onnx failed")
    if rknn.build(do_quantization=False) != 0:
        raise SystemExit("build failed")
    if rknn.init_runtime() != 0:
        raise SystemExit("init_runtime (PC simulator) failed")
    got = rknn.inference(inputs=inputs)[0]
    rknn.release()
    got = np.asarray(got).reshape(reference.shape)
    # The frames nearest the pad are where an fp16 engine's error concentrates, same as the
    # padding gate above, and the ODE loop feeds its own output back in ten times.
    tail = min(60, frames)
    print(
        f"verify fp16 (seq {seq_len}) vs fp32 ONNX (PC simulator): frames={frames} pad={seq_len - frames} "
        f"snr={snr_db(reference, got):6.1f} dB "
        f"tail_snr={snr_db(reference[:, :, -tail:], got[:, :, -tail:]):6.1f} dB "
        f"max_abs={np.abs(reference - got).max():.4f} "
        f"mean={reference.mean():+.5f}/{got.mean():+.5f}"
    )


def convert_rknn(
    onnx_path: Path,
    output: Path,
    target: str,
    opt_level: int,
    seq_len: int,
) -> None:
    from rknn.api import RKNN

    names = graph_input_names(onnx_path)
    rknn = RKNN(verbose=False)
    # FP16 with no quantization: the flow estimator is the stage the reference port keeps on the NPU,
    # and quantizing it (int8) is what pushes the mel off enough to need a hand-tuned offset.
    rknn.config(target_platform=target, optimization_level=opt_level)
    if rknn.load_onnx(model=str(onnx_path), inputs=names, input_size_list=input_shapes(seq_len)) != 0:
        raise SystemExit("load_onnx failed")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rknn_build_log import rknn_build_gate

    with rknn_build_gate(f"build {output.name}", output.with_suffix(".build.log")):
        built = rknn.build(do_quantization=False)
    if built != 0:
        raise SystemExit("build failed (see the rknn log above)")
    if rknn.export_rknn(str(output)) != 0:
        raise SystemExit("export_rknn failed")
    rknn.release()
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB, seq_len {seq_len}, "
          f"{target}, optimization_level {opt_level})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--estimator-onnx", type=Path, required=True,
        help="flow.decoder.estimator.fp32.onnx from the Fun-CosyVoice3 checkpoint")
    parser.add_argument(
        "--seq-len", type=int, action="append", default=None,
        help="sequence length to compile (repeatable); the runtime pads up to the smallest fit")
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--opt-level", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("models/CosyVoice3-RKNN"))
    parser.add_argument(
        "--check-frames", type=int, action="append", default=None,
        help="utterance lengths to check the padding at, per engine (default: a short and a "
             "near-full length)")
    parser.add_argument("--seed", type=int, default=1986, help="matches the runtime's flow seed")
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--skip-rknn", action="store_true", help="check only")
    parser.add_argument(
        "--verify-sim", action="store_true",
        help="build the first --seq-len and run it on the PC simulator against the fp32 ONNX at "
             "the first --check-frames; slow, the simulator runs the whole estimator")
    args = parser.parse_args()

    seq_lens = args.seq_len or [200, 500, 1000]
    if args.verify_sim:
        seq_len = seq_lens[0]
        frames = (args.check_frames or [seq_len // 2])[0]
        verify_fp16(args.estimator_onnx, seq_len, frames, args.target, args.opt_level, args.seed)
        return 0
    if not args.skip_check:
        for seq_len in seq_lens:
            frames = args.check_frames or [max(1, seq_len // 2), max(1, seq_len - 1)]
            check_padding(args.estimator_onnx, seq_len, frames, args.seed)
    if args.skip_rknn:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seq_len in seq_lens:
        convert_rknn(
            args.estimator_onnx,
            args.output_dir / f"flow_estimator_seq{seq_len}.rknn",
            args.target,
            args.opt_level,
            seq_len,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
