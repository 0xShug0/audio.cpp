#!/usr/bin/env python3

import argparse
import json

import onnx
import numpy as np
import onnxruntime
import torch
import torchaudio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_model")
    parser.add_argument("reference")
    parser.add_argument("--boundaries", action="store_true")
    args = parser.parse_args()

    waveform, sample_rate = torchaudio.load(args.reference)
    waveform = waveform.mean(dim=0, keepdim=True)
    waveform = torchaudio.functional.resample(
        waveform, sample_rate, 16000
    )
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=80,
        dither=0.0,
        sample_frequency=16000,
    )
    fbank = fbank - fbank.mean(dim=0, keepdim=True)

    model_input = args.onnx_model
    boundary_names = {}
    if args.boundaries:
        model = onnx.load(args.onnx_model)
        nodes = list(model.graph.node)

        def output_of(node_name: str) -> str:
            return next(
                node.output[0] for node in nodes if node.name == node_name
            )

        boundary_names = {
            "head": output_of("/head/Relu_1"),
            "tdnn": output_of("/xvector/tdnn/nonlinear/relu/Relu"),
            "layer1.h": output_of(
                "/xvector/block1/tdnnd1/nonlinear2/relu/Relu"
            ),
            "layer1.local": output_of(
                "/xvector/block1/tdnnd1/cam_layer/linear_local/Conv"
            ),
            "layer1.global": output_of(
                "/xvector/block1/tdnnd1/cam_layer/ReduceMean"
            ),
            "layer1.avg": output_of(
                "/xvector/block1/tdnnd1/cam_layer/Slice"
            ),
            "layer1.context": output_of(
                "/xvector/block1/tdnnd1/cam_layer/Add"
            ),
            "layer1.gate": output_of(
                "/xvector/block1/tdnnd1/cam_layer/relu/Relu"
            ),
            "layer1.sigmoid": output_of(
                "/xvector/block1/tdnnd1/cam_layer/sigmoid/Sigmoid"
            ),
            "layer1.out": output_of(
                "/xvector/block1/tdnnd1/cam_layer/Mul_1"
            ),
            "transit1": output_of("/xvector/transit1/linear/Conv"),
            "transit2": output_of("/xvector/transit2/linear/Conv"),
            "transit3": output_of("/xvector/transit3/linear/Conv"),
            "out_nonlinear": output_of(
                "/xvector/out_nonlinear/relu/Relu"
            ),
            "dense": "output",
        }
        for block in range(1, 4):
            candidates = [
                node for node in nodes
                if node.op_type == "Concat"
                and node.name.startswith(f"/xvector/block{block}/")
            ]
            boundary_names[f"block{block}"] = candidates[-1].output[0]
        stats_candidates = [
            node for node in nodes
            if node.op_type == "Concat"
            and node.name.startswith("/xvector/stats/")
        ]
        boundary_names["stats"] = stats_candidates[-1].output[0]

        known = {
            value.name: value
            for value in (
                list(onnx.shape_inference.infer_shapes(model).graph.value_info)
                + list(model.graph.output)
            )
        }
        existing = {value.name for value in model.graph.output}
        for tensor_name in boundary_names.values():
            if tensor_name not in existing:
                model.graph.output.append(known[tensor_name])
        model_input = model.SerializeToString()

    session = onnxruntime.InferenceSession(
        model_input,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    result = session.run(
        list(boundary_names.values()) if args.boundaries else None,
        {input_name: fbank.unsqueeze(0).cpu().numpy()},
    )
    embedding_index = (
        list(boundary_names).index("dense") if args.boundaries else 0
    )
    embedding = result[embedding_index].astype(np.float32).reshape(-1)
    boundaries = {}
    if args.boundaries:
        for name, values in zip(boundary_names, result):
            flat = values.astype(np.float32).reshape(-1)
            boundaries[name] = {
                "shape": list(values.shape),
                "size": int(flat.size),
                "sum": float(flat.astype(np.float64).sum()),
                "minimum": float(flat.min()),
                "maximum": float(flat.max()),
                "samples": [
                    float(flat[index])
                    for index in sample_point_indices(flat.size)
                ],
            }
    print(
        json.dumps(
            {
                "frames": int(fbank.shape[0]),
                "embedding": embedding.tolist(),
                "boundaries": boundaries,
            }
        )
    )


def sample_point_indices(count: int) -> list[int]:
    if count <= 40:
        return list(range(count))
    points = []
    for begin, end, samples in (
        (0, count // 3, 14),
        (count // 3, count * 2 // 3, 12),
        (count * 2 // 3, count, 14),
    ):
        span = end - begin
        for index in range(samples):
            offset = int(index / (samples - 1) * (span - 1))
            value = begin + offset
            if not points or points[-1] != value:
                points.append(value)
    if points[0] != 0:
        points.insert(0, 0)
    if points[-1] != count - 1:
        points.append(count - 1)
    return points


if __name__ == "__main__":
    main()
