#!/usr/bin/env python3
"""The ONNX graph reads the `tools/rknn/prepare_*` converters share.

The Piper vocoder and the Inflect v2 decoder converters each carried a private copy of the same
two reads: the input-name list of the sliced graph, and the latent channel count read back from
its `z`. One copy lives here now. The same-named function in the cosyvoice3 converter is not a
reader -- it validates the export's feeder set against the expected ladder -- and it stays there.

`onnx` is imported at module level: the only importers are the converters, and a converter only
runs in the toolkit venv, which pins `onnx==1.16.1`.
"""

from __future__ import annotations

from pathlib import Path

import onnx


def graph_input_names(onnx_path: Path) -> list[str]:
    return [value.name for value in onnx.load(str(onnx_path)).graph.input]


def decoder_channels_of(graph, *, where: str = "graph") -> int:
    """Latent channel count read back from an already-loaded graph."""
    for value in graph.input:
        if value.name == "z":
            dims = [d.dim_value for d in value.type.tensor_type.shape.dim]
            if len(dims) == 3 and dims[1] > 0:
                return int(dims[1])
    raise SystemExit(f"{where}: input z must be [1, channels, frames]")


def decoder_channels(onnx_path: Path) -> int:
    return decoder_channels_of(onnx.load(str(onnx_path)).graph, where=str(onnx_path))
