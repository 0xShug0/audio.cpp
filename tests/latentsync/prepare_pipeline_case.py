"""Prepare a short real-media case for native LatentSync pipeline integration."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from safetensors.numpy import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--face-fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference.resolve()))
    from latentsync.whisper.whisper.audio import load_audio

    capture = cv2.VideoCapture(str(args.reference / "assets/demo1_video.mp4"))
    frames = []
    for _ in range(2):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Cannot read demo frames")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32))
    capture.release()
    mask = cv2.imread(str(args.reference / "latentsync/utils/mask.png"))
    if mask is None:
        raise RuntimeError("Cannot read official mask")
    mask = cv2.cvtColor(cv2.resize(mask, (512, 512), interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_BGR2RGB)
    audio = load_audio(str(args.reference / "assets/demo1_audio.wav"))[:10240]
    args.output.mkdir(parents=True, exist_ok=True)
    save_file({"frames": np.stack(frames), "audio": audio,
               "mask": np.ascontiguousarray(mask.transpose(2, 0, 1), dtype=np.float32) / 255},
              args.output / "input.safetensors")
    sources = {name: str((args.weights / f"{name}.safetensors").resolve()) for name in ["unet", "vae", "whisper"]}
    for name, directory in [("detector", "scrfd"), ("landmarks", "landmarks")]:
        sources[name] = str((args.face_fixtures / directory / "weights.safetensors").resolve())
    sources["detector_config"] = str((args.face_fixtures / "scrfd/config.json").resolve())
    sources["landmark_config"] = str((args.face_fixtures / "landmarks/config.json").resolve())
    (args.output / "config.json").write_text(json.dumps(sources, indent=2) + "\n")
    print(f"Prepared {len(frames)} frames and {len(audio) / 16000} seconds of audio")


if __name__ == "__main__":
    main()
