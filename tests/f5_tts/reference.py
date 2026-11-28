#!/usr/bin/env python3
"""Generate external FP32 DiT/CFG/tokenizer references using pinned official F5-TTS."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

UPSTREAM_REVISION = "600e5fdfbf319f84735557aca80a7aaf60d487e6"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens-only", action="store_true")
    args = parser.parse_args()
    if not args.tokens_only and args.checkpoint is None:
        parser.error("--checkpoint is required unless --tokens-only is set")
    upstream = args.upstream.resolve()
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise RuntimeError(f"expected upstream {UPSTREAM_REVISION}, found {revision}")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--", "src/f5_tts/model"], text=True).strip():
        raise RuntimeError("upstream model sources have local changes")
    sys.path.insert(0, str(upstream / "src"))
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from f5_tts.model.backbones.dit import DiT
    from f5_tts.model.utils import list_str_to_idx

    if not Path(inspect.getfile(DiT)).resolve().is_relative_to(upstream):
        raise RuntimeError("F5-TTS imported from outside --upstream")
    args.output.mkdir(parents=True, exist_ok=True)
    vocab = {char: i for i, char in enumerate(args.vocab.read_text(encoding="utf-8").splitlines()) if char}
    # Explicit expected prompt strings: no call into the C++ formatter under test.
    cases = [
        ("UNK", "", "Hello.", "⓪〈Hello.〉"),
        ("MSA", "Hello.", "Again.", "①〈Hello. Again.〉"),
        ("IRQ", "مرحبا", "أهلا", "⑤〈مرحباأهلا〉"),
        ("unknown", "", "🙂", "⓪〈🙂〉"),
    ]
    token_dir = args.output / "tokens"
    token_dir.mkdir(exist_ok=True)
    for index, (dialect, reference, text, assembled) in enumerate(cases):
        prefix = token_dir / f"case{index}"
        for suffix, value in (("dialect", dialect), ("ref", reference), ("gen", text)):
            prefix.with_suffix("." + suffix).write_text(value, encoding="utf-8")
        ids = list_str_to_idx([assembled], vocab)[0].cpu().numpy().astype("<i4")
        ids.tofile(prefix.with_suffix(".ids"))
    (token_dir / "count").write_text(str(len(cases)))
    if not args.tokens_only:
        torch.set_num_threads(8)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = DiT(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2,
                    mel_dim=100, text_num_embeds=2730, text_dim=512, conv_layers=4,
                    text_mask_padding=True).eval()
        state = load_file(str(args.checkpoint))
        prefix = "ema_model.transformer."
        model.load_state_dict({key[len(prefix):]: value for key, value in state.items()
                               if key.startswith(prefix)}, strict=True)
        model = model.to(device=args.device, dtype=torch.float32)
        x = torch.randn(1, 64, 100)
        cond = torch.randn(1, 64, 100)
        ids = torch.randint(0, 2730, (1, 24))
        x.numpy().astype("<f4").tofile(args.output / "input_x.bin")
        cond.numpy().astype("<f4").tofile(args.output / "input_cond.bin")
        ids.numpy().astype("<f4").tofile(args.output / "input_ids.bin")
        x.numpy().astype("<f4").tofile(args.output / "cfg_input_x.bin")
        cond.numpy().astype("<f4").tofile(args.output / "cfg_input_cond.bin")
        ids.numpy().astype("<i4").tofile(args.output / "cfg_input_ids.bin")
        with torch.no_grad():
            arguments = dict(x=x.to(args.device), cond=cond.to(args.device),
                             text=ids.to(args.device), time=torch.tensor(0.42, device=args.device))
            model(**arguments).cpu().numpy().astype("<f4").tofile(args.output / "08_final_out.bin")
            conditional, unconditional = model(**arguments, cfg_infer=True).chunk(2, dim=0)
            conditional.cpu().numpy().astype("<f4").tofile(args.output / "cfg_out_cond.bin")
            unconditional.cpu().numpy().astype("<f4").tofile(args.output / "cfg_out_null.bin")
    # Keep provenance outside Git beside the generated fixtures.
    manifest = {"upstream_revision": revision, "torch": torch.__version__, "precision": "tokenizer_i32" if args.tokens_only else "fp32",
                "device": args.device, "seed": 42, "files": {}}
    for path in ([args.checkpoint] if args.checkpoint is not None else []) + [args.vocab]:
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            manifest[path.name] = digest.hexdigest()
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["files"][str(path.relative_to(args.output))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
