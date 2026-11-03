#!/usr/bin/env python3
"""Package original Bark 3.0 Standard/Plus .bark weights for native C++ loading.

No weights are decoded and no GGUF, safetensors, PyTorch or constriction runtime
is needed. Python/numpy are preparation-only dependencies of the reused metadata
validation helpers. Inference uses the C++ Parakeet community-model adapter.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType

from convert_phonon2 import read_exact, sha256, validate_sidecars
from convert_underdog_bark3 import package_spec as pro_package_spec, validate_config


@dataclass(frozen=True)
class PackedProfile:
    source_repo: str
    source_revision: str
    source_sha256: str
    variant: str
    display_name: str


PROFILES = MappingProxyType({
    "standard": PackedProfile("ConwayResearch/Underdog-Bark-3.0",
        "85db2ad18ab41e534a222ee253c29191d77ed919",
        "c18de9418d73cbeb80b8c1287b392d480a56e191ef4a81d1baaa1b02e66d92ed",
        "underdog_bark3_standard", "Underdog Bark 3.0"),
    "plus": PackedProfile("ConwayResearch/Underdog-Bark-3.0-Plus",
        "d4fab2f8b95681e5ad3cfb4e8c5f4e0c009b3deb",
        "fdd9f1052e4fdead07d71e8cd2138813eb0c209fd6ff8b1b6f47737e1e501857",
        "underdog_bark3_plus", "Underdog Bark 3.0 Plus"),
})
BASE_REPO = "nvidia/parakeet-tdt-0.6b-v3"
BASE_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
BASE_SHA256 = MappingProxyType({
    "config.json": "e747b85e1bdfd300c8b8ac63bac8dd5221f8fe9bc275b48d06c735fcd6971b6e",
    "processor_config.json": "8346a93a3b987fa1dec57a78f045cd0817d21786589a5a096b41a57a446fd1d7",
    "tokenizer.json": "bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91",
    "tokenizer_config.json": "0b2fe0037599ee335f0b972fa682bf0ece74e4ccfec755cb7daa3405d3d3e874",
    "generation_config.json": "b141de6ec6d7f982ece13f98f604e3fe1807ea9c0e839185d0ab7064604209d0",
})
SIDECARS = ("processor_config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")


def native_package_spec(repository, profile):
    spec = pro_package_spec(repository)
    spec.update(display_name=profile.display_name, packages=[],
                description=f"{profile.display_name}: English Parakeet-TDT ASR, original compressed bark-levels-v3 checkpoint, native C++ decoding and 250 ms reference prefix.")
    spec["runtime"]["tags"] = ["bark", "stream"]
    spec["ui"]["docs"] = ["docs/models/bark_packed_loading.md"]
    spec["sources"] = [{"format": "bark", "roots": {"model": "."},
                        "files": {"config": "model:config.json", "processor_config": "model:processor_config.json",
                                  "tokenizer_json": "model:tokenizer.json"},
                        "tensors": {"weights": "model:model.bark"}}]
    return spec


def prepare_native_package(source, target, profile):
    """Preserve pinned compressed bytes; publish the verified package atomically."""
    if target.exists():
        raise FileExistsError("native package already exists; choose a new output directory")
    if sha256(source / "model.bark") != profile.source_sha256:
        raise ValueError("Bark container differs from the pinned revision for " + profile.variant)
    for name, expected in BASE_SHA256.items():
        if sha256(source / name) != expected:
            raise ValueError("base sidecar differs from the pinned revision: " + name)
    with (source / "model.bark").open("rb") as stream:
        size = int.from_bytes(read_exact(stream, 8), "little")
        if not 0 < size <= min(4 * 1024 * 1024, (source / "model.bark").stat().st_size - 8):
            raise ValueError("invalid Bark header size")
        header = json.loads(read_exact(stream, size))
    if header.get("format") != "bark-levels-v3" or len(header.get("index", [])) != 723:
        raise ValueError("unexpected pinned Bark container")
    config = json.loads(header["config"])
    validate_config(config)
    base = json.loads((source / "config.json").read_text(encoding="utf-8"))
    for key in ("vocab_size", "blank_token_id", "decoder_hidden_size", "num_decoder_layers", "durations"):
        if config[key] != base[key]:
            raise ValueError("base sidecars and Bark container disagree: " + key)
    validate_sidecars(source, json.loads((source / "mlx/config.json").read_text(encoding="utf-8")))
    config["variant"] = profile.variant
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bark-native-", dir=target.parent) as temporary:
        package = Path(temporary) / "package"
        package.mkdir()
        shutil.copyfile(source / "model.bark", package / "model.bark")
        for name in (*SIDECARS, "NOTICE", "README.md"):
            shutil.copyfile(source / name, package / name)
        shutil.copyfile(source / "coreml/LICENSE", package / "LICENSE")
        (package / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        spec = native_package_spec(Path(__file__).resolve().parents[2], profile)
        (package / "model_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        provenance = dict(source=profile.source_repo, revision=profile.source_revision,
            model_sha256=profile.source_sha256, variant=profile.variant, storage="bark-levels-v3",
            base_repo=BASE_REPO, base_revision=BASE_REVISION, input_prefix_samples=4000,
            expanded_checkpoint=False, compressed_inference=False,
            sidecar_sha256={name: sha256(source / name) for name in SIDECARS})
        (package / "conversion.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        package.rename(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=tuple(PROFILES))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare_native_package(args.source.resolve(), args.output.resolve(), PROFILES[args.variant])


if __name__ == "__main__":
    main()
