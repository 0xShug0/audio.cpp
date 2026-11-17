#!/usr/bin/env python3
"""tiny-voca reads English only: the real command-line tool and the real server
must take English and no language, and turn any other language away.

    python tests/community_models/tiny_voca/language_test.py \
        --cli build/bin/audiocpp_cli --server build/bin/audiocpp_server \
        --model models/tiny-voca-GGUF/tiny-voca-orig.gguf --out /tmp/tiny_voca_language

It needs the model file, so it is not part of ctest; the rule itself is covered
without model files by tiny_voca_unit_test. The server is checked in its
default mode and with --parallel-jobs.
"""

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "server"))
from server_test_utils import Server  # noqa: E402

TEXT = "Hello! This is a short test."
ACCEPTED = [None, "en", "en-us", "English"]
REFUSED = ["ja", "de", "en-gb"]
failures = []


def check(ok, what, detail=""):
    print(("ok    " if ok else "FAIL  ") + what + (("  " + detail) if detail and not ok else ""))
    if not ok:
        failures.append(what)


def cli(args, name, extra):
    out = args.out / (name + ".wav")
    out.unlink(missing_ok=True)
    command = [str(args.cli), "--task", "tts", "--family", "tiny_voca", "--model", str(args.model), "--backend", "cpu",
               "--text", TEXT, "--out", str(out)] + extra
    done = subprocess.run(command, capture_output=True, text=True)
    return done.returncode, done.stdout + done.stderr, out.exists()


def check_cli(args):
    def ways(language):
        return (("--language", ["--language", language]), ("--request-option", ["--request-option", "language=" + language]))

    code, text, wrote = cli(args, "cli_ok", [])
    check(code == 0 and wrote, "cli no language: speaks", text[-300:])
    for language in ACCEPTED[1:]:
        for label, extra in ways(language):
            code, text, wrote = cli(args, "cli_ok", extra)
            check(code == 0 and wrote, f"cli {label} {language}: speaks", text[-300:])
    for language in REFUSED:
        for label, extra in ways(language):
            code, text, wrote = cli(args, "cli_refused", extra)
            check(code != 0 and not wrote and "English only" in text and language in text,
                  f"cli {label} {language}: refused with a clear error", f"exit {code} {text[-300:]}")


def check_server(args, parallel):
    mode = "parallel" if parallel else "default"
    model = {"id": "tiny", "family": "tiny_voca", "task": "tts", "mode": "offline", "path": str(args.model)}
    with Server(args.server, [model], "cpu", 0, args.out / ("server_" + mode), parallel_jobs=parallel) as server:
        def speech(language, as_option):
            body = {"model": "tiny", "input": TEXT, "response_format": "wav"}
            if language is not None:
                if as_option:
                    body["options"] = {"language": language}
                else:
                    body["language"] = language
            return raw(server, "/v1/audio/speech", body)

        def task(language, as_option):
            request = {"text": TEXT}
            if language is not None:
                if as_option:
                    request["options"] = {"language": language}
                else:
                    request["language"] = language
            return raw(server, "/v1/tasks/run", {"model": "tiny", "request": request})

        for name, call in (("/v1/audio/speech", speech), ("/v1/tasks/run", task)):
            for as_option in (False, True):
                where = "options.language" if as_option else "language"
                for language in ACCEPTED:
                    status, text = call(language, as_option)
                    check(status == 200, f"server {mode} {name} {where}={language or 'none'}: 200", f"{status} {text[:200]}")
                for language in REFUSED:
                    status, text = call(language, as_option)
                    check(status == 400 and "English only" in text and "invalid_request_error" in text,
                          f"server {mode} {name} {where}={language}: 400 with a clear error", f"{status} {text[:200]}")
        # The server still speaks after the refusals.
        status, text = speech(None, False)
        check(status == 200, f"server {mode}: speaks after the refusals", f"{status} {text[:200]}")


def raw(server, path, body):
    import json
    import urllib.error
    import urllib.request
    request = urllib.request.Request(server.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, response.read()[:200].decode("latin-1")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.cli, args.server, args.model, args.out = (p.resolve() for p in (args.cli, args.server, args.model, args.out))
    args.out.mkdir(parents=True, exist_ok=True)
    check_cli(args)
    check_server(args, parallel=False)
    check_server(args, parallel=True)
    print(f"{len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
