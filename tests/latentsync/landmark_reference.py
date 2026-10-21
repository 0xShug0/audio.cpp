"""Compare the native landmark network against the original ONNX checkpoint."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from safetensors.numpy import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture = cv2.VideoCapture(str(args.video))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("Cannot read reference video")
    rgb = cv2.cvtColor(cv2.resize(frame, (192, 192)), cv2.COLOR_BGR2RGB)
    image = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    model = ort.InferenceSession(str(args.checkpoint), sess_options=options, providers=["CPUExecutionProvider"])
    output = model.run(None, {model.get_inputs()[0].name: image})[0]
    args.output.mkdir(parents=True, exist_ok=True)
    save_file({"input": image, "output": np.ascontiguousarray(output)}, args.output / "reference.safetensors")
    print(f"Original landmark graph: {image.shape} -> {output.shape}")


if __name__ == "__main__":
    main()
