#!/usr/bin/env python3
"""Package Reson8/Rumble-2-Flash as an F32 qwen3_asr GGUF.

Download the original HF checkpoint, then run:
  python tests/qwen3_asr/convert_rumble2_flash.py --source /path/to/Rumble-2-Flash \
      --converter build/debug/bin/audiocpp_gguf --output /path/to/rumble-2-flash-f32.gguf

Inference uses the existing qwen3_asr family. Pass
  --text '<digits><nofiller><verbatim>'
to reproduce the upstream default system controls.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--converter", type=Path, default=repo / "build/debug/bin/audiocpp_gguf")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "qwen3_asr" or "text_config" not in config:
        parser.error("expected the HF-format Reson8/Rumble-2-Flash checkpoint")
    weights = source / "model.safetensors"
    if not weights.is_file():
        parser.error(f"missing checkpoint: {weights}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rumble2-flash-") as temp:
        root = Path(temp)
        for name in ("config.json", "generation_config.json", "tokenizer_config.json",
                     "tokenizer.json", "vocab.json", "merges.txt", "processor_config.json",
                     "preprocessor_config.json", "chat_template.json", "chat_template.jinja",
                     "added_tokens.json", "special_tokens_map.json", "LICENSE", "README.md"):
            if (source / name).is_file():
                shutil.copy2(source / name, root / name)
        command = [str(args.converter.resolve()), "--input", str(weights), "--root", str(root),
                   "--family", "qwen3_asr", "--model-spec", str(repo / "model_specs/qwen3_asr.json"),
                   "--type", "f32", "--output", str(args.output.resolve())]
        if args.overwrite:
            command.append("--overwrite")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
