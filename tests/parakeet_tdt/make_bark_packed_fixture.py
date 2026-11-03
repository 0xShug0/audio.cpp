#!/usr/bin/env python3
"""Regenerate the small native decoder fixture using constriction==0.5.0.

Only fixture generation needs Python/constriction. The CTest reads fixed bytes.
"""
import json
import lzma
from pathlib import Path
import sys
import numpy as np
import constriction

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/community_models"))
from convert_phonon2 import compensate_batch_norm, compensate_projection


def main():
    entries, blobs, tensors = [], [], {}

    def add(name, shape, kind, data, **metadata):
        entry = dict(n=name, shape=shape, k=kind, **metadata)
        for field, value in data.items():
            entry[field] = len(value)
            blobs.append(value)
        entries.append(entry)

    for name, values, compressed in [
        ("raw.weight", [[0.5, -2, 0], [3, -0., 0.125]], False),
        ("encoder.subsampling.linear.weight", [[1, -2, 0.5], [3, 2, -4]], True),
        ("encoder.subsampling.linear.bias", [1, -0.5], True),
        ("encoder.layers.0.conv.norm.weight", [0.25, 2], False),
        ("encoder.layers.0.conv.norm.running_var", [0., 0.5], True),
    ]:
        value = np.asarray(values, dtype="<f2")
        data = value.tobytes()
        add(name, list(value.shape), "raw", {"b": lzma.compress(data) if compressed else data},
            dtype="torch.float32", zr=compressed)
        tensors[name] = value.astype(np.float32)
    add("counter", [], "raw", {"b": np.array(-7, dtype="<i8").tobytes()}, dtype="torch.int64", zr=False)

    for levels_count in (2, 3):
        rows, cols = 3, 7
        classes = np.array([0, 1, 15, 0, 4, 1, 15], np.uint8)
        levels = np.array([[0., 0.5, 2], [0.1, 0.4, 1], [0.25, 0.125, 3]], dtype="<f2")[:, :levels_count].copy()
        codes = np.arange(rows * cols, dtype=np.int32).reshape(rows, cols) % (2 * levels_count + 1)
        counts = np.ones((16, 2 * levels_count + 1), dtype="<u4")
        for group in range(16):
            counts[group] += np.bincount(codes[:, classes == group].ravel(), minlength=counts.shape[1]).astype(np.uint32)
        coder = constriction.stream.stack.AnsCoder()
        for group in reversed(range(16)):
            probabilities = counts[group].astype(np.float64)
            model = constriction.stream.model.Categorical(probabilities / probabilities.sum(), perfect=False)
            coder.encode_reverse(codes[:, classes == group].T.copy().ravel(), model)
        packed_classes = bytes((int(classes[c]) << 4) | (int(classes[c + 1]) if c + 1 < cols else 0)
                               for c in range(0, cols, 2))
        add(f"levels{levels_count}.weight", [rows, cols], "levels2",
            {"lev": lzma.compress(levels.tobytes()), "meta": packed_classes + counts.tobytes(),
             "b": coder.get_compressed().astype("<u4").tobytes()}, K=levels_count, zl=True)
        values = np.zeros((rows, cols), np.float32)
        for row in range(rows):
            for col in range(cols):
                code = codes[row, col]
                value = np.float32(levels[row, (code - 1) % levels_count]) if code else np.float32(0)
                values[row, col] = -value if code > levels_count else value
        tensors[f"levels{levels_count}.weight"] = values
    for bits in (5, 6):
        symbols, rows, cols = 2 ** bits - 1, 3, 7
        codes = (np.arange(rows * cols, dtype=np.int32) * 7 % symbols).reshape(rows, cols)
        counts = np.ones(symbols, dtype="<u4") + np.bincount(codes.ravel(), minlength=symbols).astype("<u4")
        probabilities = counts.astype(np.float64)
        model = constriction.stream.model.Categorical(probabilities / probabilities.sum(), perfect=False)
        coder = constriction.stream.stack.AnsCoder()
        coder.encode_reverse(codes.ravel(), model)
        scales = np.array([0.1, 0.25, 0.3], dtype="<f2")
        add(f"int{bits}.weight", [rows, cols], f"ansint{bits}",
            {"s": lzma.compress(scales.tobytes()), "meta": counts.tobytes(),
             "b": coder.get_compressed().astype("<u4").tobytes()}, zs=True)
        tensors[f"int{bits}.weight"] = (codes - symbols // 2).astype(np.float32) * scales.astype(np.float32)[:, None]
    # Last blob is XZ, so the corruption test also exercises its checksum/footer.
    data = np.arange(32, dtype="<f2")
    add("xz.weight", [32], "raw", {"b": lzma.compress(data.tobytes())}, dtype="torch.float32", zr=True)
    tensors["xz.weight"] = data.astype(np.float32)
    header = json.dumps(dict(format="bark-levels-v3", config="{}", index=entries)).encode()
    target = Path(__file__).resolve().parent / "assets/bark-packed"
    target.mkdir(exist_ok=True)
    (target / "model.bark").write_bytes(len(header).to_bytes(8, "little") + header + b"".join(blobs))
    expected = {name: compensate_batch_norm(name, compensate_projection(name, value), tensors).ravel().tolist()
                for name, value in tensors.items()}
    (target / "expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
