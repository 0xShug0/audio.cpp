#!/usr/bin/env python3
"""Engine-build plumbing shared by the `tools/rknn/prepare_*` converters.

Every converter ends in the same five calls on the RKNPU2 Python API, and the compiler-log gate
(`rknn_build_log.rknn_build_gate`) has to wrap `rknn.build` in every one of them, so the sequence
lives here. Everything that produces its arguments stays in the converter: the ONNX surgery, the
input names and shapes, the per-engine file naming, and the simulator-side verification.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence


def build_engine(
    onnx_path: Path,
    output: Path,
    target: str,
    *,
    names: Optional[Sequence[str]] = None,
    shapes: Optional[Sequence[Sequence[int]]] = None,
    opt_level: Optional[int] = None,
    verbose: bool = False,
    gate_label: Optional[str] = None,
    gate_log: Optional[Path] = None,
) -> None:
    """Compile ``onnx_path`` into ``output`` for ``target``: config, load, gated build, export.

    ``names``/``shapes`` are declared to ``load_onnx`` only for a graph that is still dynamic
    there. A converter that cropped the sequence axis into a fixed window compiles a static graph
    and declares nothing: re-declaring shapes on a static graph only warns.

    Not ``single_core_mode``: the engine stays multi-core, and ``rknn_set_core_mask`` in the
    runtime decides how many NPU cores a call actually runs on.
    """
    from rknn.api import RKNN

    output = Path(output)
    onnx_path = Path(onnx_path)

    rknn = RKNN(verbose=verbose)
    config: dict = {"target_platform": target}
    if opt_level is not None:
        config["optimization_level"] = opt_level
    rknn.config(**config)
    if names is not None and shapes is not None:
        ret = rknn.load_onnx(model=str(onnx_path), inputs=list(names),
                             input_size_list=[list(s) for s in shapes])
    else:
        ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise SystemExit("load_onnx failed")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rknn_build_log import rknn_build_gate

    with rknn_build_gate(gate_label or output.name, gate_log or output.with_suffix(".build.log")):
        built = rknn.build(do_quantization=False)
    if built != 0:
        raise SystemExit("build failed (see the rknn log above)")
    if rknn.export_rknn(str(output)) != 0:
        raise SystemExit("export_rknn failed")
    rknn.release()
