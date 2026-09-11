#!/usr/bin/env python3
"""Reference PT2 warmbench for ABR Niagara batch ASR."""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers.audio_utils import mel_filter_bank


DTYPE_MAP = {
    4: torch.int32,
    5: torch.int64,
    7: torch.float32,
    12: torch.bool,
}


def split_csv(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def model_files(model: Path) -> tuple[Path, Path]:
    if model.is_file() and model.suffix == ".pt2":
        root = model.parent
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        return model, root / str(config["tokenizer_file"])
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    pt2 = model / f"{config['exported_model_file']}-cpu.pt2"
    return pt2, model / str(config["tokenizer_file"])


def load_raw_tensor(archive: zipfile.ZipFile, path: str, dtype_id: int, shape: list[int]) -> torch.Tensor:
    dtype = DTYPE_MAP[dtype_id]
    data = archive.read(path)
    return torch.frombuffer(bytearray(data), dtype=dtype).clone().reshape(shape)


def tensor_shape(meta: dict[str, Any]) -> list[int]:
    return [int(item["as_int"]) for item in meta.get("sizes", [])]


def load_graph_inputs(pt2: Path, input_tensor: torch.Tensor) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    with zipfile.ZipFile(pt2) as archive:
        base = archive.namelist()[0].split("/")[0]
        model = json.loads(archive.read(f"{base}/models/model.json"))
        constants = json.loads(archive.read(f"{base}/data/constants/model_constants_config.json"))["config"]
        input_specs = model["graph_module"]["signature"]["input_specs"]
        tensor_values = model["graph_module"]["graph"]["tensor_values"]

        values: dict[str, torch.Tensor] = {}
        parameters = [item["parameter"] for item in input_specs if "parameter" in item]
        for index, parameter in enumerate(parameters):
            name = parameter["arg"]["name"]
            meta = tensor_values[name]
            values[name] = load_raw_tensor(
                archive,
                f"{base}/data/weights/weight_{index}",
                int(meta["dtype"]),
                tensor_shape(meta),
            )

        for item in input_specs:
            if "tensor_constant" in item:
                constant = item["tensor_constant"]
                name = constant["arg"]["name"]
                info = constants[constant["tensor_constant_name"]]
                meta = info["tensor_meta"]
                values[name] = load_raw_tensor(
                    archive,
                    f"{base}/data/constants/{info['path_name']}",
                    int(meta["dtype"]),
                    tensor_shape(meta),
                )
            elif "user_input" in item:
                values[item["user_input"]["arg"]["as_tensor"]["name"]] = input_tensor
    return model, values


def arg_value(arg: dict[str, Any], values: dict[str, torch.Tensor], symbols: dict[str, int]) -> Any:
    if "as_tensor" in arg:
        return values[arg["as_tensor"]["name"]]
    if "as_tensors" in arg:
        return [values[item["name"]] for item in arg["as_tensors"]]
    if "as_int" in arg:
        return int(arg["as_int"])
    if "as_ints" in arg:
        return [int(item) for item in arg["as_ints"]]
    if "as_float" in arg:
        return float(arg["as_float"])
    if "as_bool" in arg:
        return bool(arg["as_bool"])
    if "as_none" in arg:
        return None
    if "as_string" in arg:
        return str(arg["as_string"])
    if "as_scalar_type" in arg:
        return DTYPE_MAP[int(arg["as_scalar_type"])]
    if "as_sym_int" in arg:
        sym = arg["as_sym_int"]
        if "as_int" in sym:
            return int(sym["as_int"])
        return symbols[sym["as_name"]]
    if "as_name" in arg:
        return symbols[arg["as_name"]]
    if "as_sym_ints" in arg:
        return [arg_value(item, values, symbols) for item in arg["as_sym_ints"]]
    if "as_device" in arg or "as_layout" in arg or "as_memory_format" in arg:
        return None
    raise RuntimeError(f"unsupported Niagara reference arg: {arg}")


def node_args(node: dict[str, Any], values: dict[str, torch.Tensor], symbols: dict[str, int]) -> tuple[list[Any], dict[str, Any]]:
    positional: list[Any] = []
    keyword: dict[str, Any] = {}
    for item in node.get("inputs", []):
        value = arg_value(item["arg"], values, symbols)
        if item.get("kind") == 1:
            positional.append(value)
        elif value is not None and item["name"] not in {"device", "layout", "memory_format"}:
            keyword[item["name"]] = value
    return positional, keyword


def set_output(node: dict[str, Any], result: Any, values: dict[str, torch.Tensor], symbols: dict[str, int]) -> None:
    output = node["outputs"][0]
    if "as_tensor" in output:
        values[output["as_tensor"]["name"]] = result
    elif "as_tensors" in output:
        for spec, tensor in zip(output["as_tensors"], result):
            values[spec["name"]] = tensor
    elif "as_sym_int" in output:
        symbols[output["as_sym_int"]["as_name"]] = int(result)
    else:
        raise RuntimeError(f"unsupported Niagara reference output: {output}")


def replay(pt2: Path, input_tensor: torch.Tensor) -> torch.Tensor:
    model, values = load_graph_inputs(pt2, input_tensor)
    symbols: dict[str, int] = {}
    with torch.no_grad():
        for node in model["graph_module"]["graph"]["nodes"]:
            target = node["target"]
            positional, keyword = node_args(node, values, symbols)
            if target == "torch.ops.aten._assert_tensor_metadata.default":
                continue
            if target.startswith("torch.ops.aten.to."):
                dtype = next((arg_value(item["arg"], values, symbols) for item in node["inputs"] if "as_scalar_type" in item["arg"]), None)
                result = positional[0].to(dtype=dtype) if dtype else positional[0]
            elif target == "torch.ops.aten.sym_size.int":
                result = positional[0].shape[positional[1]]
            elif target == "torch.ops.aten.lift_fresh_copy.default":
                result = positional[0].clone()
            elif target == "torch.ops.aten.not_equal.Tensor":
                result = torch.not_equal(positional[0], positional[1])
            elif target == "torch.ops.aten.all.dim":
                result = torch.all(positional[0], dim=positional[1], keepdim=positional[2])
            elif target == "torch.ops.aten.squeeze.dim":
                result = torch.squeeze(positional[0], dim=positional[1])
            elif target == "torch.ops.aten.unsqueeze.default":
                result = torch.unsqueeze(positional[0], dim=positional[1])
            elif target == "torch.ops.aten.logical_not.default":
                result = torch.logical_not(positional[0])
            elif target == "torch.ops.aten.permute.default":
                result = positional[0].permute(positional[1])
            elif target == "torch.ops.aten.max_pool2d.default":
                result = F.max_pool2d(*positional, **keyword)
            elif target == "torch.ops.aten.reshape.default":
                result = torch.reshape(positional[0], tuple(positional[1]))
            elif target == "torch.ops.aten.conv2d.default":
                result = F.conv2d(*positional, **keyword)
            elif target == "torch.ops.aten.matmul.default":
                result = torch.matmul(positional[0], positional[1])
            elif target in {"torch.ops.aten.add.Tensor", "torch.ops.aten.add_.Tensor"}:
                result = positional[0] + positional[1]
            elif target in {"torch.ops.aten.sub.Tensor", "torch.ops.aten.sub_.Tensor", "torch.ops.aten.subtract.Tensor"}:
                result = positional[0] - positional[1]
            elif target == "torch.ops.aten.abs.default":
                result = torch.abs(positional[0])
            elif target == "torch.ops.aten.mean.dim":
                result = torch.mean(positional[0], dim=positional[1], keepdim=positional[2])
            elif target == "torch.ops.aten.div.Tensor":
                result = positional[0] / positional[1]
            elif target in {"torch.ops.aten.multiply.Tensor", "torch.ops.aten.mul.Tensor", "torch.ops.aten.mul_.Tensor"}:
                result = positional[0] * positional[1]
            elif target == "torch.ops.aten.logical_or.default":
                result = torch.logical_or(positional[0], positional[1])
            elif target == "torch.ops.aten.einsum.default":
                result = torch.einsum(positional[0], positional[1])
            elif target == "torch.ops.aten.sigmoid.default":
                result = torch.sigmoid(positional[0])
            elif target == "torch.ops.aten.pad.default":
                result = F.pad(positional[0], positional[1])
            elif target == "torch.ops.aten.__and__.Tensor":
                result = positional[0] & positional[1]
            elif target == "torch.ops.aten.silu.default":
                result = F.silu(positional[0])
            elif target == "torch.ops.aten.sqrt.default":
                result = torch.sqrt(positional[0])
            elif target == "torch.ops.aten.softmax.int":
                result = F.softmax(positional[0], dim=positional[1])
            elif target == "torch.ops.aten.flip.default":
                result = torch.flip(positional[0], positional[1])
            elif target == "torch.ops.aten.split.Tensor":
                result = torch.split(positional[0], positional[1], dim=positional[2])
            elif target == "torch.ops.aten.contiguous.default":
                result = positional[0].contiguous()
            elif target == "torch.ops.aten.rsqrt_.default":
                result = torch.rsqrt(positional[0])
            elif target == "torch.ops.aten.select.int":
                result = torch.select(positional[0], positional[1], positional[2])
            elif target == "torch.ops.aten.slice.Tensor":
                tensor = positional[0]
                slices = [slice(None)] * tensor.ndim
                end = positional[3] if len(positional) > 3 else None
                if isinstance(end, int) and end > 10**15:
                    end = None
                slices[positional[1]] = slice(positional[2] if len(positional) > 2 else None, end, positional[4] if len(positional) > 4 else 1)
                result = tensor[tuple(slices)]
            elif target == "_operator.sub":
                result = positional[0] - positional[1]
            elif target == "_operator.add":
                result = positional[0] + positional[1]
            elif target == "_operator.mul":
                result = positional[0] * positional[1]
            else:
                raise NotImplementedError(target)
            set_output(node, result, values, symbols)
    out_name = model["graph_module"]["signature"]["output_specs"][0]["user_output"]["arg"]["as_tensor"]["name"]
    return values[out_name]


def features(wav: np.ndarray, sample_rate: int) -> tuple[torch.Tensor, int]:
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    signal = torch.from_numpy(wav.astype("float32"))[None, :]
    win = round(sample_rate * 25 / 1000)
    hop = round(sample_rate * 10 / 1000)
    if signal.shape[1] < win:
        signal = F.pad(signal, (0, win - signal.shape[1]))
    spec = torch.stft(
        signal,
        n_fft=win,
        hop_length=hop,
        win_length=win,
        window=torch.hann_window(win),
        center=False,
        return_complex=True,
    ).abs()
    mel = torch.from_numpy(mel_filter_bank(
        num_frequency_bins=win // 2 + 1,
        num_mel_filters=80,
        min_frequency=0,
        max_frequency=8000,
        sampling_rate=sample_rate,
    ).astype("float32"))
    result = torch.matmul(spec.transpose(1, 2), mel)
    result = torch.log(torch.clamp(result, min=1e-12))
    pad = (-result.shape[1]) % 4
    if pad:
        result = F.pad(result, (0, 0, 0, pad), value=1000.0)
    valid_frames = (len(wav) - win) // hop + 1 if len(wav) >= win else 1
    return result, valid_frames


def decode(logits: torch.Tensor, tokenizer: spm.SentencePieceProcessor, valid_input_frames: int) -> str:
    output_frames = valid_input_frames
    for _ in range(2):
        output_frames = max(0, (output_frames - 4) // 2 + 1)
    pred = logits[:, :output_frames, :].argmax(dim=-1)[0].tolist()
    pieces: list[int] = []
    previous: int | None = None
    blank = tokenizer.GetPieceSize()
    for item in pred:
        if item == previous:
            continue
        previous = item
        if item != blank and item > 0:
            pieces.append(int(item))
    return str(tokenizer.Decode(pieces)).strip()


def transcribe(pt2: Path, tokenizer: spm.SentencePieceProcessor, audio_path: Path) -> str:
    wav, sample_rate = sf.read(audio_path)
    if sample_rate != 16000:
        raise RuntimeError(f"Niagara reference expects 16 kHz WAV input, got {sample_rate}: {audio_path}")
    input_features, valid_frames = features(wav, sample_rate)
    logits = replay(pt2, input_features)
    return decode(logits, tokenizer, valid_frames)


def step_json(index: int, audio_path: Path, text: str, wall_ms: float) -> dict[str, Any]:
    return {
        "request_index": index,
        "audio": str(audio_path),
        "text_output": text,
        "word_timestamps": [],
        "metrics": {"wall_ms": wall_ms},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/niagara-19m-batch.en"))
    parser.add_argument("--audio", type=Path, default=Path("resources/sample_16k.wav"))
    parser.add_argument("--warmup-audio", type=Path)
    parser.add_argument("--audio-sequence", default="")
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--device", default="0")
    parser.add_argument("--threads", default="8")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--timing-file", type=Path, default=Path("/tmp/niagara_asr_python_warm_bench_timing.log"))
    args = parser.parse_args()

    if args.backend != "cpu":
        raise RuntimeError("Niagara PT2 reference warmbench supports only CPU")

    pt2, tokenizer_path = model_files(args.model)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(tokenizer_path))

    warmup_audio = args.warmup_audio or args.audio
    for _ in range(args.warmup):
        transcribe(pt2, tokenizer, warmup_audio)

    request_paths = [Path(item) for item in split_csv(args.audio_sequence)] or [args.audio]
    steps: list[dict[str, Any]] = []
    timing_lines: list[str] = []
    for index, audio_path in enumerate(request_paths):
        last_text = ""
        total_ms = 0.0
        for _ in range(max(1, args.iterations)):
            started = time.time()
            last_text = transcribe(pt2, tokenizer, audio_path)
            total_ms += (time.time() - started) * 1000.0
        wall_ms = total_ms / max(1, args.iterations)
        print(f"average[{index}]")
        print(f"niagara_asr.python_wall_ms={wall_ms}")
        steps.append(step_json(index, audio_path, last_text, wall_ms))
        timing_lines.append(f"niagara_asr.request{index}.python_wall_ms {wall_ms:.6f}")

    if args.timing_file.parent:
        args.timing_file.parent.mkdir(parents=True, exist_ok=True)
    args.timing_file.write_text("\n".join(timing_lines) + "\n", encoding="utf-8")
    print("summary_json=" + json.dumps({
        "family": "niagara_asr",
        "backend": args.backend,
        "sequence_steps": steps,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
