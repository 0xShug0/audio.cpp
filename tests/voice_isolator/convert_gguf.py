#!/usr/bin/env python3
"""Package one Voice Isolator checkpoint as a self-contained GGUF.

The released safetensors files carry their architecture as "config" metadata;
the GGUF embeds it as config.json beside the model spec.
"""

import argparse
import hashlib
import logging
from pathlib import Path
import subprocess
import tempfile

from safetensors import safe_open

SHA256 = {
    "denoiser.safetensors": "647335a6aeb4dc706b895532c209d51e0e84a4849d17b339f00d4392a050ae53",
    "denoiser_realtime.safetensors": "76598f05523c0669e76562a8b2f31c9d395fca316e80c907ee11775e7497ae0f",
}


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="denoiser.safetensors or denoiser_realtime.safetensors")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--type", choices=("f32", "f16", "bf16"), default="f32")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audiocpp-gguf", type=Path, default=root / "build/debug/bin/audiocpp_gguf")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    expected = SHA256.get(args.source.name)
    if expected is None:
        raise ValueError(f"Expected one of {sorted(SHA256)}, got {args.source.name}")
    with args.source.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(f"{args.source.name} is not the inworld/voice-isolator release (sha256 {digest})")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch, safe_open(args.source, framework="np") as source:
        (Path(scratch) / "config.json").write_text(source.metadata()["config"])
        command = [str(args.audiocpp_gguf.resolve()), "--input", f"weights={args.source.resolve()}",
                   "--output", str(args.output.resolve()), "--type", args.type, "--family", "voice_isolator",
                   "--model-spec", str(root / "model_specs/voice_isolator.json"), "--root", scratch]
        if args.overwrite:
            command.append("--overwrite")
        # Vectors and the Mamba state matrix stay F32, as the runtime keeps them.
        for name in source.keys():
            if len(source.get_slice(name).get_shape()) == 1 or name.endswith(".A_log"):
                command.extend(["--keep-type", f"weights/{name}=f32"])
        logging.info("command=%s", command)
        subprocess.run(command, check=True)
    logging.info("wrote %s from %s (sha256 %s)", args.output, args.source.name, digest)


if __name__ == "__main__":
    main()
