#!/usr/bin/env python3
"""Generate deterministic Fun-ASR-Nano projector/adaptor checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

TRANSFORMERS_COMMIT = "f9966442ac24fff57060774ce22e1884760f4a3b"
MODEL_REVISION = "854d88f94205cd17d2afdb24332130d86fbe654a"
MODEL_CONFIG_SHA256 = "c7c4a30316929631ac5fabc5fb3c0dd3278dcc9809670720c5920186285d004a"
MODEL_SAFETENSORS_SHA256 = (
    "335ca3e74917f1156690400e2c344350112950165789cf78ce3d0a367affd821"
)
TORCH_VERSION = "2.11.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformers-src", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    transformers_src = args.transformers_src.resolve()
    model_dir = args.model_dir.resolve()
    sys.path.insert(0, str(transformers_src))

    import torch
    import transformers
    from safetensors import safe_open
    from transformers import FunAsrNanoConfig
    from transformers.models.fun_asr_nano import modeling_fun_asr_nano as modeling

    imported_module = Path(transformers.__file__).resolve()
    if transformers_src not in imported_module.parents:
        raise RuntimeError(
            f"expected Transformers from {transformers_src}, imported {imported_module}"
        )
    transformers_root = transformers_src.parent
    actual_commit = subprocess.check_output(
        ["git", "-C", str(transformers_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != TRANSFORMERS_COMMIT:
        raise RuntimeError(
            f"expected Transformers {TRANSFORMERS_COMMIT}, found {actual_commit}"
        )
    modeling_path = (
        transformers_src / "transformers/models/fun_asr_nano/modeling_fun_asr_nano.py"
    )
    modeling_relative = modeling_path.relative_to(transformers_root)
    modeling_status = subprocess.check_output(
        [
            "git",
            "-C",
            str(transformers_root),
            "status",
            "--porcelain",
            "--",
            str(modeling_relative),
        ],
        text=True,
    ).strip()
    if modeling_status:
        raise RuntimeError(
            f"Fun-ASR-Nano modeling source has uncommitted changes: {modeling_status}"
        )
    if torch.__version__.split("+", maxsplit=1)[0] != TORCH_VERSION:
        raise RuntimeError(f"expected torch {TORCH_VERSION}, found {torch.__version__}")

    model_path = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"
    actual_config_sha256 = sha256_file(config_path)
    actual_model_sha256 = sha256_file(model_path)
    if actual_config_sha256 != MODEL_CONFIG_SHA256:
        raise RuntimeError(
            f"expected config SHA-256 {MODEL_CONFIG_SHA256}, found {actual_config_sha256}"
        )
    if actual_model_sha256 != MODEL_SAFETENSORS_SHA256:
        raise RuntimeError(
            f"expected model SHA-256 {MODEL_SAFETENSORS_SHA256}, found {actual_model_sha256}"
        )

    config = FunAsrNanoConfig.from_pretrained(model_dir, local_files_only=True)
    config.encoder_config._attn_implementation = "eager"
    projector = modeling.FunAsrNanoMultiModalProjector(config).eval()
    adaptor = modeling.FunAsrNanoAdaptor(config).eval()
    projector_state = {}
    adaptor_state = {}
    catalog_lines = []
    with safe_open(model_path, framework="pt", device="cpu") as source:
        for name in source.keys():
            if not name.startswith("model.multi_modal_projector."):
                continue
            tensor_slice = source.get_slice(name)
            catalog_lines.append(
                json.dumps(
                    {
                        "name": name,
                        "dtype": tensor_slice.get_dtype(),
                        "shape": list(tensor_slice.get_shape()),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            if ".blocks." in name:
                target = name.removeprefix("model.multi_modal_projector.")
                adaptor_state[target] = source.get_tensor(name).float()
            else:
                target = name.removeprefix("model.multi_modal_projector.")
                projector_state[target] = source.get_tensor(name).float()
    if len(projector_state) != 4 or len(adaptor_state) != 32:
        raise RuntimeError(
            f"expected 4 projector and 32 adaptor tensors, found {len(projector_state)} and {len(adaptor_state)}"
        )
    projector.load_state_dict(projector_state, strict=True, assign=True)
    adaptor.load_state_dict(adaptor_state, strict=True, assign=True)

    torch.set_num_threads(1)
    encoder_embeddings = torch.linspace(
        -0.75, 0.75, steps=2 * 4 * config.encoder_config.d_model, dtype=torch.float32
    ).reshape(2, 4, config.encoder_config.d_model)
    encoder_embeddings[1, 2:] = torch.linspace(
        10.0, 20.0, steps=2 * config.encoder_config.d_model, dtype=torch.float32
    ).reshape(2, config.encoder_config.d_model)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.int64)

    def forward(
        values: torch.Tensor, keep_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        linear_1 = projector.linear_1(values)
        linear_2 = projector.linear_2(projector.act(linear_1))
        attention_mask = modeling._prepare_4d_attention_mask(keep_mask, linear_2.dtype)
        block_0 = adaptor.blocks[0](linear_2, attention_mask)
        block_1 = adaptor.blocks[1](block_0, attention_mask)
        return {
            "linear_1": linear_1,
            "linear_2": linear_2,
            "block_0": block_0,
            "block_1": block_1,
            "packed_valid": block_1[keep_mask.bool()],
        }

    with torch.no_grad():
        batched = forward(encoder_embeddings, mask)
        valid_mask = mask.bool()
        checkpoints = {
            name: tensor if name == "packed_valid" else tensor[valid_mask]
            for name, tensor in batched.items()
        }
        individual = []
        for batch_index, valid_frames in enumerate(mask.sum(dim=1).tolist()):
            values = encoder_embeddings[batch_index : batch_index + 1, :valid_frames]
            keep = torch.ones((1, valid_frames), dtype=torch.int64)
            individual.append(forward(values, keep)["packed_valid"])
        torch.testing.assert_close(
            checkpoints["packed_valid"],
            torch.cat(individual, dim=0),
            atol=2.0e-4,
            rtol=2.0e-4,
        )

    binary_data = bytearray()

    def store_tensor(tensor: torch.Tensor) -> dict[str, object]:
        array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        descriptor = {
            "shape": list(array.shape),
            "offset_f32": len(binary_data) // 4,
            "count": int(array.size),
        }
        binary_data.extend(array.reshape(-1).tobytes())
        return descriptor

    binary_path = args.output.with_suffix(".bin")
    catalog_text = "\n".join(sorted(catalog_lines)) + "\n"
    payload = {
        "schema_version": 1,
        "transformers_commit": TRANSFORMERS_COMMIT,
        "model_revision": MODEL_REVISION,
        "modeling_sha256": hashlib.sha256(modeling_path.read_bytes()).hexdigest(),
        "model_config_sha256": actual_config_sha256,
        "model_safetensors_sha256": actual_model_sha256,
        "weight_catalog_count": len(catalog_lines),
        "weight_catalog_sha256": hashlib.sha256(catalog_text.encode()).hexdigest(),
        "torch_version": torch.__version__,
        "data_file": binary_path.name,
        "data_format": "little-endian-float32",
        "input": store_tensor(encoder_embeddings),
        "mask": store_tensor(mask.to(torch.float32)),
        "checkpoints": {
            name: store_tensor(tensor) for name, tensor in checkpoints.items()
        },
    }
    binary_bytes = bytes(binary_data)
    payload["data_sha256"] = hashlib.sha256(binary_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    binary_path.write_bytes(binary_bytes)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
