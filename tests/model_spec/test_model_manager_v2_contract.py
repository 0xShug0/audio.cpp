#!/usr/bin/env python3
"""Conformance check for resolved package metadata in `list --json`."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manager", required=True, type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="audiocpp-model-spec-") as temp:
        specs = Path(temp)
        (specs / "integrator_fixture.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "family": "integrator_fixture",
                    "package_defaults": {
                        "download": {
                            "kind": "huggingface_snapshot",
                            "repo": "audio-cpp/integrator-fixture",
                            "revision": "fixture-revision",
                            "gated": False,
                            "checksums": {"fixture.gguf": "sha256:abc"},
                        }
                    },
                    "packages": [
                        {
                            "id": "integrator_fixture_q8",
                            "display_name": "Integrator Fixture Q8",
                            "format": "gguf",
                            "precision": "q8_0",
                            "target_directory": "Integrator-Fixture-GGUF",
                            "files": ["weights/fixture.gguf"],
                            "strip_prefix": "weights",
                            "default": True,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(args.manager),
                "--specs-dir",
                str(specs),
                "list",
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        rows = json.loads(completed.stdout)
        if len(rows) != 1:
            raise AssertionError(f"expected one package, got {len(rows)}")
        row = rows[0]
        assert row["family"] == "integrator_fixture"
        assert row["files"] == ["weights/fixture.gguf"]
        assert row["strip_prefix"] == "weights"
        assert row["access_status"] == "public"
        assert row["download"]["revision"] == "fixture-revision"
        assert row["download"]["checksums"]["fixture.gguf"] == "sha256:abc"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
