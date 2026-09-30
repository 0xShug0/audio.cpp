"""Reference SCRFD heads using the original detector and its image preprocessing."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from safetensors.numpy import save_file
from insightface.model_zoo.scrfd import SCRFD


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
    height, width = frame.shape[:2]
    ratio = 512 / max(height, width)
    resized = cv2.resize(frame, (int(width * ratio), int(height * ratio)))
    padded = np.zeros((512, 512, 3), dtype=np.uint8)
    padded[:resized.shape[0], :resized.shape[1]] = resized
    image = cv2.dnn.blobFromImage(padded, 1 / 128, (512, 512), (127.5,) * 3, swapRB=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    model = ort.InferenceSession(str(args.checkpoint), sess_options=options, providers=["CPUExecutionProvider"])
    heads = model.run(None, {model.get_inputs()[0].name: image})
    args.output.mkdir(parents=True, exist_ok=True)
    tensors = {"input": image, "output": np.concatenate([head.ravel() for head in heads])}
    detector = SCRFD(session=model)
    detector.prepare(ctx_id=-1, input_size=(512, 512))
    boxes, landmarks = detector.detect(frame)
    tensors["detections"] = np.concatenate([boxes, landmarks.reshape(-1, 10)], axis=1)
    tensors["resize_scale"] = np.array([resized.shape[0] / height], dtype=np.float32)
    print(f"Detected faces: {boxes}")
    for i, head in enumerate(heads):
        tensors[f"head_{i}"] = np.ascontiguousarray(head)
        print(f"Head {i}: {head.shape}, range {head.min()} .. {head.max()}")
    save_file(tensors, args.output / "reference.safetensors")


if __name__ == "__main__":
    main()
