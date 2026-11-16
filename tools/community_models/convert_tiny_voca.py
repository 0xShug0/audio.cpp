#!/usr/bin/env python3
"""Pack tiny-voca as one GGUF file for audio.cpp.

    python tools/community_models/convert_tiny_voca.py \
        --source /path/to/tiny-voca --output models/tiny-voca-GGUF/tiny-voca-orig.gguf

``--source`` is a copy of https://huggingface.co/n4ze3m/tiny-voca: it needs
``tiny-voca.pt``, ``config.json`` and the ``phonemizer`` folder. Nothing is
changed on the way: the weights of the model and of the phonemizer's
unknown-word model are stored as float32, bit for bit, and the phonemizer's
data files are stored as they are. ``--check`` reads the GGUF back and compares.

Needs torch, numpy and safetensors, and a built ``audiocpp_gguf``.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from safetensors.numpy import save_file

REPO = Path(__file__).resolve().parents[2]
PHONEMIZER = ("lexicon.tsv", "align.json", "g2p.json", "homographs.json", "LICENSE.cmudict", "LICENSE.homograph-data")


def model_tensors(checkpoint):
    import torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    tensors = {}
    for name, value in state["model"].items():
        array = value.numpy()
        if array.dtype != np.float32:
            raise ValueError(f"{name} is {array.dtype}; a float32 checkpoint is expected")
        tensors[name] = np.ascontiguousarray(array)
    return tensors


def write_model_directory(source, root):
    """The files audio.cpp reads, in a folder: it can be loaded as it is, too."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "phonemizer").mkdir(exist_ok=True)
    shutil.copyfile(source / "config.json", root / "config.json")
    for name in PHONEMIZER:
        shutil.copyfile(source / "phonemizer" / name, root / "phonemizer" / name)
    tensors = model_tensors(source / "tiny-voca.pt")
    save_file(tensors, str(root / "model.safetensors"))
    with np.load(source / "phonemizer" / "g2p.npz") as arrays:
        g2p = {name: np.ascontiguousarray(arrays[name]) for name in arrays.files}
    save_file(g2p, str(root / "g2p.safetensors"))
    return tensors, g2p


def check(path, tensors, g2p):
    """Every tensor in the GGUF equals the source, byte for byte."""
    import gguf
    reader = gguf.GGUFReader(str(path))
    stored = {tensor.name: tensor for tensor in reader.tensors}
    wrong = 0
    for prefix, group in (("model", tensors), ("g2p", g2p)):
        for name, array in group.items():
            found = next((stored[key] for key in (f"{prefix}/{name}", f"{prefix}.{name}", name) if key in stored), None)
            if found is None or np.asarray(found.data).tobytes() != array.tobytes():
                wrong += 1
                print("DIFFERENT" if found is not None else "MISSING", prefix, name)
    print(f"{len(tensors) + len(g2p)} tensors checked, {wrong} wrong; {len(stored)} tensors in the file")
    return wrong == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="A copy of the Hugging Face repository n4ze3m/tiny-voca")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--converter", type=Path, default=REPO / "build/bin/audiocpp_gguf")
    parser.add_argument("--directory", type=Path, help="Also keep the unpacked model folder here (loadable as it is)")
    parser.add_argument("--check", action="store_true", help="Read the GGUF back and compare every tensor (needs the gguf package)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tiny-voca-", dir=args.output.parent) as temporary:
        root = args.directory or Path(temporary) / "tiny-voca"
        tensors, g2p = write_model_directory(args.source, root)
        command = [str(args.converter.resolve()), "--input", f"model={root / 'model.safetensors'}",
                   "--input", f"g2p={root / 'g2p.safetensors'}", "--root", str(root), "--family", "tiny_voca",
                   "--model-spec", str(REPO / "model_specs/tiny_voca.json"), "--type", "orig",
                   "--output", str(args.output.resolve())]
        if args.overwrite:
            command.append("--overwrite")
        subprocess.run(command, check=True)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB): {len(tensors)} model tensors, "
          f"{sum(v.size for v in tensors.values()):,} parameters; {len(g2p)} phonemizer tensors")
    if args.check and not check(args.output, tensors, g2p):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
