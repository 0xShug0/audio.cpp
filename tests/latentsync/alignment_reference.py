"""Export official face-alignment transforms over consecutive video frames."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import kornia
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from insightface.app import FaceAnalysis
from safetensors.numpy import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--face-root", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference_dir.resolve()))
    from latentsync.utils.affine_transform import AlignRestore

    detector = FaceAnalysis(root=str(args.face_root), allowed_modules=["detection", "landmark_2d_106"],
                            providers=["CPUExecutionProvider"])
    detector.prepare(ctx_id=-1, det_size=(512, 512))
    aligner = AlignRestore(resolution=512, device="cpu", dtype=torch.float32)
    capture = cv2.VideoCapture(str(args.video))
    points, matrices, landmarks = [], [], []
    warp_tensors = {}
    bias = None
    for _ in range(8):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Reference video has fewer than eight frames")
        faces = detector.get(frame)
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        rounded = np.round(face.landmark_2d_106)
        three = np.round([rounded[[43, 48, 49, 51, 50]].mean(0), rounded[101:106].mean(0),
                          rounded[[74, 77, 83, 86]].mean(0)])
        matrix, bias = aligner.transformation_from_points(three, aligner.face_template, True, bias)
        if not warp_tensors:
            image = torch.from_numpy(frame.transpose(2, 0, 1).copy()).float()[None]
            warped = kornia.geometry.transform.warp_affine(image, torch.from_numpy(matrix)[None],
                (aligner.face_size[1], aligner.face_size[0]), mode="bilinear", padding_mode="fill",
                fill_value=torch.tensor([127., 127., 127.]))
            warp_tensors = {"image": image.numpy(), "warped": warped.numpy()}
            resized = cv2.resize(warped[0].numpy().transpose(1, 2, 0), (512, 512), interpolation=cv2.INTER_LANCZOS4)
            warp_tensors["resized"] = np.ascontiguousarray(resized.transpose(2, 0, 1)[None])
            aligned = cv2.resize(warped[0].numpy().transpose(1, 2, 0).astype(np.uint8), (512, 512), interpolation=cv2.INTER_LANCZOS4)
            warp_tensors["rgb"] = np.ascontiguousarray(image.numpy()[:, ::-1])
            warp_tensors["aligned"] = np.ascontiguousarray(aligned[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 127.5 - 1
            changed_face = torch.from_numpy(warp_tensors["aligned"][0].copy())
            changed_face[0, 250:450, 100:400] += 0.15
            resized_face = resize(changed_face, (560, 420), interpolation=InterpolationMode.BICUBIC, antialias=True)
            restored = aligner.restore_img(frame[:, :, ::-1].copy(), resized_face, matrix)
            warp_tensors["restore_face"] = changed_face.numpy()
            warp_tensors["restore_resized"] = resized_face.numpy()
            warp_tensors["restored"] = np.ascontiguousarray(restored.transpose(2, 0, 1), dtype=np.float32)
        landmarks.append(face.landmark_2d_106)
        points.append(three)
        matrices.append(matrix)
    capture.release()
    args.output.mkdir(parents=True, exist_ok=True)
    save_file({**warp_tensors, "landmarks": np.array(landmarks, dtype=np.float32),
               "points": np.array(points, dtype=np.float32),
               "matrices": np.array(matrices, dtype=np.float32)}, args.output / "reference.safetensors")
    print(f"Exported {len(matrices)} official transforms; crop size {aligner.face_size}")


if __name__ == "__main__":
    main()
