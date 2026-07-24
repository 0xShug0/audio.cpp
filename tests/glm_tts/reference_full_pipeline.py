#!/usr/bin/env python3
"""Launch the upstream GLM-TTS pipeline on platforms without WeText/Pynini.

The Windows PyPI index does not provide the Pynini wheel required by
WeTextProcessing. For parity inputs that contain no numbers or abbreviations,
an identity normalizer is sufficient and leaves the model path unchanged.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
import types
from pathlib import Path

import torch


class IdentityNormalizer:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def normalize(self, text: str) -> str:
        return text


def install_wetext_identity_stub() -> None:
    modules = {
        "tn": types.ModuleType("tn"),
        "tn.chinese": types.ModuleType("tn.chinese"),
        "tn.chinese.normalizer": types.ModuleType("tn.chinese.normalizer"),
        "tn.english": types.ModuleType("tn.english"),
        "tn.english.normalizer": types.ModuleType("tn.english.normalizer"),
    }
    modules["tn.chinese.normalizer"].Normalizer = IdentityNormalizer
    modules["tn.english.normalizer"].Normalizer = IdentityNormalizer
    sys.modules.update(modules)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("--data", required=True)
    parser.add_argument("--exp-name", default="_audio_cpp_parity")
    args = parser.parse_args()

    root = args.reference_root.resolve()
    entry = root / "glmtts_inference.py"
    if not entry.is_file():
        raise FileNotFoundError(entry)
    install_wetext_identity_stub()
    if not hasattr(torch, "npu"):
        torch.npu = types.SimpleNamespace(is_available=lambda: False)
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.argv = [
        str(entry),
        "--data",
        args.data,
        "--exp_name",
        args.exp_name,
    ]
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
