#!/usr/bin/env python3
"""Multilingual eval metrics for granite_speech_2b (E2 validation gate).

Pinned rules (do not "improve" without recording a tolerance adjustment):
- Unicode normalization: NFKC on both reference and hypothesis, always.
- WER (space-delimited languages: en, fr, de, es, pt): after NFKC, uppercase,
  hyphen -> space, drop every code point that is not a Unicode letter,
  an ASCII apostrophe, or a space; split on whitespace. Edit distance on the
  resulting word sequences.
- CER (Japanese): after NFKC, delete all whitespace; edit distance on the
  resulting character sequences (punctuation and kana/kanji kept).
- Punctuation/case pass: NFKC only, no other folding; exact match flag plus
  character-level edit distance on the raw strings, reported separately from
  WER/CER. Never mixed into the gate numbers.
- Edit counts + denominators are always reported (sub/del/ins + N words/chars)
  so small slices stay interpretable.

tools/asr_wer.py remains the English smoke tool; this module extends scoring to
the manifest languages without rewriting that tool.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import unicodedata

WER_LANGS = ("en", "fr", "de", "es", "pt")
CER_LANGS = ("ja",)


def nfkc(text):
    return unicodedata.normalize("NFKC", text)


def normalize_wer_words(text):
    text = nfkc(text).upper().replace("-", " ")
    kept = "".join(
        ch if (ch == "'" or ch == " " or (ch != "'" and ch.isalpha())) else " "
        for ch in text
    )
    return kept.split()


def normalize_cer_chars(text):
    return [ch for ch in nfkc(text) if not ch.isspace()]


def edit_ops(a, b):
    """Levenshtein backtrace: returns (distance, subs, dels, inss)."""
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    # full matrix needed for the backtrace; slices here are tiny (<1k units)
    table = [list(range(m + 1))]
    for i in range(1, n + 1):
        row = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost)
        table.append(row)
        prev = row
    subs = dels = inss = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and table[i][j] == table[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and table[i][j] == table[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            inss += 1
            j -= 1
    return table[n][m], subs, dels, inss


def score_pair(reference, hypothesis, language):
    """Score one reference/hypothesis pair. Returns a result dict."""
    if language in WER_LANGS:
        metric, unit = "WER", "words"
        ref = normalize_wer_words(reference)
        hyp = normalize_wer_words(hypothesis)
    elif language in CER_LANGS:
        metric, unit = "CER", "chars"
        ref = normalize_cer_chars(reference)
        hyp = normalize_cer_chars(hypothesis)
    else:
        raise ValueError(f"unsupported eval language: {language!r}")
    dist, subs, dels, inss = edit_ops(ref, hyp)
    raw_ref, raw_hyp = nfkc(reference), nfkc(hypothesis)
    raw_dist, _, _, _ = edit_ops(list(raw_ref), list(raw_hyp))
    denom = len(ref)
    return {
        "metric": metric,
        "unit": unit,
        "errors": dist,
        "denominator": denom,
        "rate_pct": 100.0 * dist / denom if denom else (0.0 if not hyp else 100.0),
        "subs": subs,
        "dels": dels,
        "inss": inss,
        "empty_hypothesis": len(hyp) == 0,
        "punct_case_exact": raw_ref == raw_hyp,
        "punct_case_char_errors": raw_dist,
    }


def aggregate(results):
    """Aggregate scored rows sharing one metric. Returns summary dict."""
    by_metric = {}
    for row in results:
        by_metric.setdefault(row["metric"], []).append(row)
    summary = {}
    for metric, rows in by_metric.items():
        errors = sum(r["errors"] for r in rows)
        denom = sum(r["denominator"] for r in rows)
        summary[metric] = {
            "errors": errors,
            "denominator": denom,
            "rate_pct": 100.0 * errors / denom if denom else 0.0,
            "clips": len(rows),
            "empty_outputs": sum(1 for r in rows if r["empty_hypothesis"]),
        }
    return summary


def run_cli_transcribe(cli, family, model, audio, backend, threads, language,
                       timeout=1800):
    cmd = [cli, "--task", "asr", "--family", family, "--model", model,
           "--backend", backend, "--threads", str(threads), "--audio", audio]
    if language:
        cmd += ["--request-option", f"language={language}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"cli exit {proc.returncode}: {(proc.stderr.strip() or proc.stdout.strip())[-500:]}")
    collecting = False
    parts = []
    for line in proc.stdout.splitlines():
        if line.startswith("text_output="):
            collecting = True
            parts.append(line[len("text_output="):])
        elif collecting:
            if "=" in line.split(" ")[0] and not line.startswith(" "):
                break
            parts.append(line)
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True,
                        help="eval_manifest.jsonl path")
    parser.add_argument("--cli", required=True)
    parser.add_argument("--family", default="granite_speech_2b")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--threads", default="8")
    parser.add_argument("--label", default="")
    parser.add_argument("--out", default="",
                        help="optional path to write the JSON result rows")
    args = parser.parse_args()

    rows = []
    for line in pathlib.Path(args.manifest).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    ready = [r for r in rows if r.get("status") == "ready"]
    pending = [r for r in rows if r.get("status") != "ready"]

    scored = []
    for row in ready:
        ref_path = pathlib.Path(row["reference_path"])
        reference = ref_path.read_text(encoding="utf-8").strip()
        hypothesis = run_cli_transcribe(
            args.cli, args.family, args.model, row["audio_path"],
            args.backend, args.threads, row.get("request_language", ""))
        result = score_pair(reference, hypothesis, row["language"])
        result.update({"clip_id": row["clip_id"], "language": row["language"],
                       "hypothesis": hypothesis, "reference": reference})
        scored.append(result)
        print(f"{row['clip_id']:44s} {result['metric']} "
              f"{result['rate_pct']:6.2f}%  {result['errors']:3d}/"
              f"{result['denominator']:<4d} s/d/i="
              f"{result['subs']}/{result['dels']}/{result['inss']}"
              f"{'   <- NO OUTPUT' if result['empty_hypothesis'] else ''}")

    summary = aggregate(scored)
    for metric, agg in summary.items():
        unit = {"WER": "words", "CER": "chars"}.get(metric, "units")
        print(f"\n{metric}: {agg['rate_pct']:.2f}%  ({agg['errors']} errors over "
              f"{agg['denominator']} {unit}, "
              f"{agg['clips']} clips, {agg['empty_outputs']} empty)")
    if pending:
        print(f"\npending ({len(pending)} rows, not scored):")
        for row in pending:
            print(f"  {row['clip_id']:42s} {row.get('pending_reason', '')}")
    if args.label:
        print(f"\n[{args.label}]")
    if args.out:
        pathlib.Path(args.out).write_text(
            json.dumps({"label": args.label, "model": args.model,
                        "rows": scored, "summary": summary,
                        "pending": [r["clip_id"] for r in pending]},
                       ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
