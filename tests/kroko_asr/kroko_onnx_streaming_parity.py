#!/usr/bin/env python3
"""Compare the native two-chunk Kroko encoder probe with the source ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def initial_states(config: dict[str, object]) -> list[np.ndarray]:
    states: list[np.ndarray] = []
    for stack, layers in enumerate(config["num_encoder_layers"]):
        channels = config["encoder_dims"][stack]
        heads = config["num_heads"][stack]
        query = config["query_head_dims"][stack]
        value = config["value_head_dims"][stack]
        left = config["left_context_len"][stack]
        conv = config["cnn_module_kernels"][stack] // 2
        for _ in range(layers):
            states.extend(
                [
                    np.zeros((left, 1, heads * query), np.float32),
                    np.zeros((1, 1, left, 3 * channels // 4), np.float32),
                    np.zeros((left, 1, heads * value), np.float32),
                    np.zeros((left, 1, heads * value), np.float32),
                    np.zeros((1, channels, conv), np.float32),
                    np.zeros((1, channels, conv), np.float32),
                ]
            )
    states.append(np.zeros((1, 128, 3, 19), np.float32))
    states.append(np.zeros((1,), np.int64))
    return states


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    ref = reference.astype(np.float64).ravel()
    got = actual.astype(np.float64).ravel()
    delta = got - ref
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "cosine": float(np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("encoder_onnx", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("native_output", type=Path)
    parser.add_argument("--native-embedding", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    chunk_size = config["chunk_size"]
    chunk_shift = config["chunk_shift"]
    embedded_frames = chunk_shift // 2
    encoded_frames = chunk_shift // 4
    model = onnx.load(args.encoder_onnx)
    model.graph.output.append(
        helper.make_tensor_value_info(
            "/Slice_3_output_0",
            TensorProto.FLOAT,
            [embedded_frames, 1, 192],
        )
    )
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    state_values = initial_states(config)
    state_names = [item.name for item in session.get_inputs()[1:]]
    total_frames = chunk_shift + chunk_size
    indexes = np.arange(total_frames * 80, dtype=np.float32)
    features = (
        np.sin(indexes * np.float32(0.0013)) * np.float32(0.7)
        + np.cos(indexes * np.float32(0.0007)) * np.float32(0.2)
    ).reshape(total_frames, 80)

    reference_chunks: list[np.ndarray] = []
    reference_embeddings: list[np.ndarray] = []
    for chunk in range(2):
        inputs = {
            "x": features[
                chunk * chunk_shift : chunk * chunk_shift + chunk_size
            ][None]
        }
        inputs.update(dict(zip(state_names, state_values)))
        outputs = session.run(None, inputs)
        reference_chunks.append(outputs[0])
        reference_embeddings.append(outputs[117])
        state_values = outputs[1:117]

    native = np.fromfile(args.native_output, dtype=np.float32).reshape(
        2, encoded_frames, 1, 512
    )
    report = {
        f"chunk_{index + 1}": metrics(reference_chunks[index], native[index])
        for index in range(2)
    }
    if args.native_embedding is not None:
        native_embedding = np.fromfile(
            args.native_embedding, dtype=np.float32
        ).reshape(2, embedded_frames, 1, 192)
        report["embedding_chunk_1"] = metrics(
            reference_embeddings[0], native_embedding[0]
        )
        report["embedding_chunk_2"] = metrics(
            reference_embeddings[1], native_embedding[1]
        )
        layer_inputs = [item.name for item in model.graph.input[1:-2]]
        processed_input = model.graph.input[-1].name
        layer_outputs = [
            item.name for item in model.graph.output[1:115]
        ]
        retained_nodes = [
            node
            for node in model.graph.node
            if "/Slice_3_output_0" not in node.output
        ]
        del model.graph.node[:]
        model.graph.node.extend(retained_nodes)
        zipformer_model = onnx.utils.Extractor(model).extract_model(
            [
                "x",
                "/Slice_3_output_0",
                *layer_inputs,
                "embed_states",
                processed_input,
            ],
            [
                model.graph.output[0].name,
                *layer_outputs,
            ],
        )
        zipformer_session = ort.InferenceSession(
            zipformer_model.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        zipformer_states = initial_states(config)
        processed = zipformer_states[-1]
        zipformer_states = zipformer_states[:-2]
        zipformer_names = [
            item.name
            for item in zipformer_session.get_inputs()
            if item.name
            not in {
                "x",
                "/Slice_3_output_0",
                "embed_states",
                processed_input,
            }
        ]
        for chunk in range(2):
            inputs = {
                "/Slice_3_output_0": native_embedding[chunk],
                "x": features[
                    chunk * chunk_shift :
                    chunk * chunk_shift + chunk_size
                ][None],
                "embed_states": np.zeros(
                    (1, 128, 3, 19), np.float32
                ),
                processed_input: processed,
            }
            inputs.update(dict(zip(zipformer_names, zipformer_states)))
            outputs = zipformer_session.run(None, inputs)
            report[f"zipformer_chunk_{chunk + 1}"] = metrics(
                outputs[0], native[chunk]
            )
            zipformer_states = outputs[1:]
            processed = processed + embedded_frames
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
