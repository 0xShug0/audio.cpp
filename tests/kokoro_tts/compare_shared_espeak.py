"""Compare a pre-migration Kokoro probe with shared dynamic/static eSpeak."""
import argparse
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--baseline", type=Path, required=True)
p.add_argument("--current", type=Path, required=True)
p.add_argument("--resources", type=Path, required=True)
p.add_argument("--library", type=Path, required=True)
p.add_argument("--mecab", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--dynamic-only", action="store_true", help="Check a build without static eSpeak")
a = p.parse_args()
cases_path = Path(__file__).with_name("multilingual_cases.json")
cases = json.loads(cases_path.read_text(encoding="utf-8"))
env = dict(os.environ, AUDIOCPP_MECAB_LIBRARY=str(a.mecab.resolve()))
env.pop("AUDIOCPP_ESPEAK_DATA", None)

def run(exe, overrides):
    result = subprocess.run([str(exe.resolve()), str(a.resources.resolve()),
                             str(cases_path.resolve())],
                            env=overrides, capture_output=True, encoding="utf-8", check=True)
    lines = result.stdout.splitlines()
    if len(lines) != len(cases) or any(line.startswith("ERROR:") for line in lines):
        raise RuntimeError(result.stdout + result.stderr)
    return lines

dynamic_env = dict(env, AUDIOCPP_ESPEAK_LIBRARY=str(a.library.resolve()))
baseline = run(a.baseline, dynamic_env)
dynamic = run(a.current, dynamic_env)
env.pop("AUDIOCPP_ESPEAK_LIBRARY", None)
static = run(a.current, env) if not a.dynamic_only else [None] * len(cases)
report = [{**case, "baseline": old, "shared_dynamic": dyn, "shared_static": sta,
           "dynamic_match": old == dyn, "static_match": old == sta if sta is not None else None}
          for case, old, dyn, sta in zip(cases, baseline, dynamic, static)]
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
for mode in (("dynamic",) if a.dynamic_only else ("dynamic", "static")):
    count = sum(item[mode + "_match"] for item in report)
    print(f"{mode}: {count}/{len(report)} identical pronunciation sequences")
if not all(item["dynamic_match"] and (a.dynamic_only or item["static_match"]) for item in report):
    raise SystemExit(1)
