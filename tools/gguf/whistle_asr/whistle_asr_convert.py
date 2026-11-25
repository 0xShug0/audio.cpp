"""Repackage the official Cactus Whistle checkpoint as an experimental audio.cpp GGUF.

Pin Cactus-Compute/whistle at b358ddadd89b7a713b5aa131f23032d3cca1b251.
The checkpoint supplies FP32 weights; whistle.cact supplies the custom tokenizer,
mel filterbank, and Hadamard permutations for the native whistle_asr family.
The output embeds these assets, the upstream LICENSE, and the schema-v1 spec.
"""

import argparse
import hashlib
import subprocess
from pathlib import Path

from extract_assets import extract

SOURCE = {
    "whistle.safetensors": (
        220618620,
        "5fd58c246f522ecee2b568b306598a3befdc274c225e3a7376abce277ab3d107",
    ),
    "config.json": (
        603,
        "ac2dbcb98a945eff2de98a03104c468dc9d68c06b9ff791539327f90298a3d43",
    ),
    "LICENSE": (
        11358,
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    ),
}
ARCHIVE = (
    16919407,
    "b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb",
)
ASSETS = {
    "hadamard_perm1.f32",
    "hadamard_perm2.f32",
    "mel_filterbank_80.f32",
    "tokenizer.blob",
}
SPEC = Path(__file__).resolve().parents[3] / "model_specs" / "whistle_asr.json"


def verify(path: Path, expected: tuple[int, str]) -> None:
    size, sha256 = expected
    if path.stat().st_size != size:
        raise ValueError(f"Wrong length for {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        raise ValueError(f"SHA-256 mismatch for {path}")


def convert(source: Path, archive: Path, converter: Path, output: Path, overwrite: bool = False) -> None:
    source = source.resolve()
    archive = archive.resolve()
    converter = converter.resolve()
    output = output.resolve()
    if output.is_relative_to(source):
        raise ValueError("The output must not be inside the sidecar directory")
    for name, expected in SOURCE.items():
        verify(source / name, expected)
    verify(archive, ARCHIVE)
    extract(archive, source)
    unexpected = {path.name for path in source.iterdir()} - set(SOURCE) - ASSETS
    if unexpected:
        raise ValueError(f"Unexpected source files would become GGUF sidecars: {sorted(unexpected)}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Pass --overwrite to replace the existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(converter), "--input", str(source / "whistle.safetensors"),
        "--root", str(source), "--family", "whistle_asr", "--model-spec", str(SPEC),
        "--type", "orig", "--output", str(output)
    ]
    if overwrite:
        command.append("--overwrite")
    subprocess.run(command, check=True)
    inspection = subprocess.run(
        [str(converter), "--inspect", str(output)],
        check=True, text=True, capture_output=True
    ).stdout.splitlines()
    for field in ("tensors=116", "embedded_sidecar_count=6", "model_spec_family=whistle_asr"):
        if field not in inspection:
            raise RuntimeError(f"GGUF inspection is missing {field}")
    print(f"Converted Whistle weights: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Directory with official safetensors, config, LICENSE")
    parser.add_argument("--cact", required=True, type=Path, help="Official whistle.cact for tokenizer and frontend assets")
    parser.add_argument("--converter", required=True, type=Path, help="audiocpp_gguf executable")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert(args.source, args.cact, args.converter, args.output, args.overwrite)
