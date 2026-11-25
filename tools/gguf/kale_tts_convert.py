#!/usr/bin/env python3
"""Pack kale-tts-1-tiny or kale-tts-1-nano as one GGUF file for audio.cpp.

    python tools/gguf/kale_tts_convert.py \
        --source /path/to/kale-tts-1-nano --output models/kale-tts-1-GGUF/kale-tts-1-nano-f32.gguf

``--source`` is a copy of https://huggingface.co/KaleComputer/kale-tts-1-nano (or -tiny). The weights come from its
float32 graph (``kale-tts-1-nano.fp32.onnx`` / ``kale-tts-1-tiny.fp32.onnx``); every weight is stored as float32 under its PyTorch module name and layout (LSTM
gates reordered from ONNX's i, o, f, c to PyTorch's i, f, g, o). The voices (style table + per-voice values), the word
lists and, for tiny, the Ettin tokenizer are stored as they are. kale-tts-1-base is plain Kokoro: use
``tools/gguf/kokoro_tts_convert.py`` (see docs/models/kale_tts.md).

Needs numpy, onnx and safetensors, and a built ``audiocpp_gguf``.
"""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

REPO = Path(__file__).resolve().parents[2]
WORDS = ("lexicon.tsv", "override.public.tsv", "terms.tsv")
LSTM_GATES = [0, 2, 3, 1]  # ONNX i, o, f, c -> PyTorch i, f, g, o
ALBERT_PARTS = {"query", "key", "value", "dense", "ffn", "ffn_output"}
NAMED = {
    "ettin": [("et.", "ettin."), ("st.", "student."), ("pred.", "predictor."), ("te.", "text_encoder."), ("dec.", "decoder.")],
    "albert": [("rt.predictor.", "predictor."), ("rt.bert_encoder.", "bert_encoder."), ("rt.bert.", "bert."),
               ("ra.te.", "text_encoder."), ("ra.dec.", "decoder.")],
}
NODE_EXACT = {"lstm": "predictor.lstm", "shared": "predictor.shared", "lstm_1": "text_encoder.lstm", "bert_encoder": "bert_encoder"}
NODE_PREFIX = [("lstms.", "predictor.text_encoder.lstms."), ("duration_proj.", "predictor.duration_proj."), ("head.", "decoder.head."),
               ("bert.", "bert."), ("et.", "ettin."), ("st.", "student.")]


def rename(name, table):
    for src, dst in table:
        if name.startswith(src):
            return dst + name[len(src):]
    return None


def node_paths(graph):
    """anonymous initializer -> (module path, op, input slot) of the node that reads it"""
    readers = {}
    for node in graph.node:
        for slot, value in enumerate(node.input):
            readers.setdefault(value, []).append((node, slot))

    def follow(value, depth=0):
        for node, slot in readers.get(value, []):
            if node.op_type in ("Identity", "Cast", "Transpose") and depth < 3:
                yield from follow(node.output[0], depth + 1)
            else:
                yield node, slot

    out = {}
    for init in graph.initializer:
        if not init.name.startswith("onnx::"):
            continue
        hits = list(follow(init.name))
        if not hits:
            continue
        node, slot = hits[0]
        parts = [p for p in node.name.removeprefix("/g").strip("/").split("/")[:-1] if p]
        parts = [re.sub(r"_\d+$", "", p) if re.sub(r"_\d+$", "", p) in ALBERT_PARTS else p for p in parts]
        parts = [p for i, p in enumerate(parts) if not (i + 1 < len(parts) and parts[i + 1].startswith(p + "."))]
        exact = ".".join(parts)
        out[init.name] = (NODE_EXACT.get(exact) or rename(exact, NODE_PREFIX), node.op_type, slot)
    return out


def model_tensors(onnx_path):
    import onnx
    from onnx import numpy_helper

    graph = onnx.load(str(onnx_path)).graph
    text_side = "ettin" if any(i.name == "ettin_ids" for i in graph.input) else "albert"
    anonymous = node_paths(graph)
    tensors = {}
    for init in graph.initializer:
        array = numpy_helper.to_array(init)
        if array.dtype != np.float32:
            continue
        if init.name in anonymous:
            module, op, slot = anonymous[init.name]
            if module is None:
                continue
            if op == "MatMul":
                tensors[f"{module}.weight"] = array.T
            elif op == "LSTM":
                for direction, suffix in enumerate(("", "_reverse")):
                    if slot in (1, 2):
                        hidden = array.shape[1] // 4
                        weight = array[direction].reshape(4, hidden, -1)[LSTM_GATES].reshape(4 * hidden, -1)
                        tensors[f"{module}.weight_{'ih' if slot == 1 else 'hh'}_l0{suffix}"] = weight
                    else:
                        hidden = array.shape[1] // 8
                        bias = array[direction].reshape(2, 4, hidden)[:, LSTM_GATES].reshape(2, 4 * hidden)
                        tensors[f"{module}.bias_ih_l0{suffix}"] = bias[0]
                        tensors[f"{module}.bias_hh_l0{suffix}"] = bias[1]
            continue
        name = rename(init.name, NAMED[text_side])
        if name is None or "/" in init.name:
            continue
        if ".cnn." in name:
            name = re.sub(r"\.gamma$", ".weight", name)
        name = re.sub(r"\.beta$", ".bias", name)
        tensors[name] = array
    tensors = {name: np.ascontiguousarray(value, dtype=np.float32) for name, value in tensors.items()}
    return text_side, graph, tensors


def albert_passes(graph):
    return len({m.group(1) for n in graph.node if (m := re.search(r"attention/query(_\d+)?/MatMul", n.name))})


def model_config(source, text_side, graph, tensors):
    voices = json.loads((source / "voices.json").read_text())
    vocab = json.loads((source / "tokenizer.json").read_text())["model"]["vocab"]
    t = tensors
    config = {
        "model": {
            "sample_rate": 24000,
            "text_side": text_side,
            "n_token": int(t["text_encoder.embedding.weight"].shape[0]),
            "hidden_dim": int(t["predictor.duration_proj.linear_layer.weight"].shape[1]),
            "style_dim": int(t["predictor.text_encoder.lstms.1.fc.weight"].shape[1]),
            "n_layer": len({k.split(".")[2] for k in t if k.startswith("text_encoder.cnn.")}),
            "max_dur": int(t["predictor.duration_proj.linear_layer.weight"].shape[0]),
            "decoder_width": int(t["decoder.encode.conv1.weight"].shape[0]),
            "decoder_up_channels": int(t["decoder.decode.3.conv1.weight"].shape[0]),
            "voice_sound_dim": int(t["decoder.encode.norm1.fc.weight"].shape[1]),
            "head_dim": int(t["decoder.head.inp.weight"].shape[0]),
            "head_blocks": len({k.split(".")[3] for k in t if k.startswith("decoder.head.blocks.")}),
            "head_points": 80,
            "head_burst_points": 24,
            "head_harmonics": 64,
            "head_mod": 2,
            "max_tokens": 510,
        },
        "phonemes": voices.get("phonemes", "ipa"),
        "vocab": vocab,
        "voices": [{k: v[k] for k in ("id", "name", "accent", "description") if k in v} for v in voices["voices"]],
    }
    m = config["model"]
    if text_side == "albert":
        mapping = t["bert.encoder.embedding_hidden_mapping_in.weight"]
        m["albert"] = {"hidden_size": int(mapping.shape[0]), "embedding_size": int(mapping.shape[1]), "num_attention_heads": 4,
                       "intermediate_size": int(t["bert.encoder.albert_layer_groups.0.albert_layers.0.ffn.weight"].shape[0]),
                       "num_hidden_layers": albert_passes(graph) or 6, "max_position_embeddings": int(t["bert.embeddings.position_embeddings.weight"].shape[0])}
    else:
        m["ettin"] = {"hidden_size": int(t["ettin.embeddings.tok_embeddings.weight"].shape[1]),
                      "intermediate_size": int(t["ettin.layers.0.mlp.Wo.weight"].shape[1]),
                      "num_hidden_layers": len({k.split(".")[2] for k in t if k.startswith("ettin.layers.")}),
                      "num_attention_heads": 4, "global_attn_every_n_layers": 3, "local_attention": 128, "rope_theta": 160000.0}
        m["student"] = {"channels": int(t["student.emb.weight"].shape[1]), "heads": 4,
                        "conv_blocks": len({k.split(".")[2] for k in t if k.startswith("student.convs.")}),
                        "transformer_blocks": len({k.split(".")[2] for k in t if k.startswith("student.tf.")})}
    return config, voices


def voice_tensors(source, voices):
    """voices/<id>.bin = style table [510, 256]; voices/<id>.extra.bin = the graph's voice inputs, in voices.json order"""
    out = {}
    for voice in voices["voices"]:
        style = np.fromfile(source / f"voices/{voice['id']}.bin", np.float32).reshape(-1, 256)
        extra = np.fromfile(source / f"voices/{voice['id']}.extra.bin", np.float32)
        out[f"{voice['id']}.style"] = np.ascontiguousarray(style)
        offset = 0
        for item in voices["voice_inputs"]:
            size = int(np.prod(item["dims"]))
            key = item["name"].removeprefix("voice_")
            if key != "prosody_col":
                out[f"{voice['id']}.{key}"] = np.ascontiguousarray(extra[offset:offset + size])
            offset += size
    return out


def float32_graph(source):
    """The release's float32 graph: kale-tts-1-nano.fp32.onnx or kale-tts-1-tiny.fp32.onnx"""
    found = sorted(source.glob("*.fp32.onnx"))
    if not found:
        raise FileNotFoundError(f"no *.fp32.onnx in {source}; --source must be a copy of kale-tts-1-nano or -tiny")
    return found[0]


def write_model_directory(source, root):
    root.mkdir(parents=True, exist_ok=True)
    text_side, graph, tensors = model_tensors(float32_graph(source))
    config, voices = model_config(source, text_side, graph, tensors)
    (root / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1))
    save_file(tensors, str(root / "model.safetensors"))
    save_file(voice_tensors(source, voices), str(root / "voices.safetensors"))
    for sub in ("", "en-gb"):
        (root / "words" / sub).mkdir(parents=True, exist_ok=True)
        for name in WORDS:
            if (source / "frontend/words" / sub / name).exists():
                shutil.copyfile(source / "frontend/words" / sub / name, root / "words" / sub / name)
    ettin = source / "ettin-tokenizer.json"
    if ettin.exists():
        shutil.copyfile(ettin, root / "ettin-tokenizer.json")
    else:
        (root / "ettin-tokenizer.json").write_text("{}")
    return text_side, tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="A copy of KaleComputer/kale-tts-1-nano or kale-tts-1-tiny")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--converter", type=Path, default=REPO / "build/debug/bin/audiocpp_gguf")
    parser.add_argument("--directory", type=Path, help="Also keep the unpacked model folder here (loadable as it is)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".kale-tts-", dir=args.output.parent) as temporary:
        root = args.directory or Path(temporary) / "kale-tts"
        text_side, tensors = write_model_directory(args.source, root)
        command = [str(args.converter.resolve()), "--input", f"model={root / 'model.safetensors'}",
                   "--input", f"voices={root / 'voices.safetensors'}", "--root", str(root), "--family", "kale_tts",
                   "--model-spec", str(REPO / "model_specs/kale_tts.json"), "--type", "f32", "--output", str(args.output.resolve())]
        if args.overwrite:
            command.append("--overwrite")
        subprocess.run(command, check=True)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB): {text_side} text side, {len(tensors)} tensors, "
          f"{sum(v.size for v in tensors.values()):,} parameters")


if __name__ == "__main__":
    main()
