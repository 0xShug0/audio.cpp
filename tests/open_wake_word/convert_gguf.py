#!/usr/bin/env python3
"""Convert the three official OpenWakeWord Alexa ONNX checkpoints into one GGUF."""

import argparse
from pathlib import Path
import sys

import numpy as np
from conversion import write_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_repo.resolve()))
    from openwakeword.model import Model
    # The official initializer computes features from random startup noise. Cache a
    # fixed initialization so backend A/B runs have identical state and no startup inference.
    np.random.seed(0)
    model = Model(wakeword_models=[str(args.models / "alexa_v0.1.onnx")],
                  melspec_model_path=str(args.models / "melspectrogram.onnx"),
                  embedding_model_path=str(args.models / "embedding_model.onnx"),
                  inference_framework="onnx", ncpu=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_model(args.models, args.output, model.preprocessor.feature_buffer)


if __name__ == "__main__":
    main()
