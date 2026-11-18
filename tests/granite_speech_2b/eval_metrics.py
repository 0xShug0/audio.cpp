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


def normalize_kw(text):
    """Keyword matching normalization: NFKC + casefold (no WER-style
    punctuation/number folding — keyword identity is literal text)."""
    return nfkc(text).casefold()


def kw_match(term, text):
    """True when a keyword term occurs in text. Single ASCII words
    ([A-Za-z0-9']+) need word boundaries so "tent" does not match "tents";
    multiword or non-ASCII terms (Japanese, accented Latin) match as
    substrings, since word boundaries are meaningless there."""
    import re
    norm_term = normalize_kw(term)
    norm_text = normalize_kw(text)
    if not norm_term:
        return False
    if re.fullmatch(r"[A-Za-z0-9']+", norm_term):
        return re.search(r"(?<![A-Za-z0-9'])" + re.escape(norm_term) +
                         r"(?![A-Za-z0-9'])", norm_text) is not None
    return norm_term in norm_text


def score_keywords(present_terms, absent_terms, hypothesis, reference):
    """Paired keyword-biasing stats. Present terms: hit when they occur in
    the hypothesis (recall); also checked against the reference so a
    mislabeled fixture (term not in the transcript at all) shows up as
    in_ref=False instead of silently counting as a model miss. Absent terms:
    any hypothesis occurrence is an insertion (must be zero)."""
    hits, misses, insertions = [], [], []
    for term in present_terms:
        if kw_match(term, hypothesis):
            hits.append(term)
        else:
            misses.append(term)
    for term in absent_terms:
        if kw_match(term, hypothesis):
            insertions.append(term)
    in_ref = {term: kw_match(term, reference) for term in present_terms}
    return {
        "present": list(present_terms),
        "absent": list(absent_terms),
        "hits": hits,
        "misses": misses,
        "insertions": insertions,
        "in_ref": in_ref,
        "ref_transcript_words": len(normalize_wer_words(reference)),
    }


def aggregate_keyword_stats(rows):
    """Binary keyword decisions over rows: predicted = occurs in hyp,
    labeled = occurs in ref (present terms) or never (absent terms).
    Only present terms with in_ref=True join the labeled set; mislabeled
    fixtures are reported separately, never folded into P/R."""
    tp = fp = fn = 0
    mislabeled = []
    for row in rows:
        stats = row["keyword_stats"]
        ref = row["reference"]
        for term in stats["present"]:
            if kw_match(term, ref):
                if term in stats["hits"]:
                    tp += 1
                else:
                    fn += 1
            else:
                mislabeled.append((row["clip_id"], term))
        fp += len(stats["insertions"])
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision_pct": 100.0 * precision,
            "recall_pct": 100.0 * recall, "f1_pct": 100.0 * f1,
            "mislabeled": mislabeled}


def score_translation(reference, hypothesis):
    """chrF + BLEU via sacrebleu (lazy import: the ASR path has no new deps).
    Defaults: chrF (char order 6, no word order = chrF), BLEU 13a — both
    case- and punctuation-sensitive, matching the human authored refs."""
    try:
        import sacrebleu
    except ImportError:
        raise RuntimeError("translation scoring requires sacrebleu "
                           "(build/granite-2b-handoff/venv has it)")
    chrf = sacrebleu.sentence_chrf(hypothesis, [reference]).score
    bleu = sacrebleu.sentence_bleu(hypothesis, [reference]).score
    return {"chrf": chrf, "bleu": bleu}


def aggregate_translation(rows):
    """Corpus chrF/BLEU over the scored rows (same sacrebleu defaults)."""
    import sacrebleu
    hyps = [r["hypothesis"] for r in rows]
    refs = [[r["reference"] for r in rows]]
    return {"chrf": sacrebleu.corpus_chrf(hyps, refs).score,
            "bleu": sacrebleu.corpus_bleu(hyps, refs).score,
            "clips": len(rows)}


def normalize_wer_words(text):
    text = nfkc(text).upper().replace("-", " ")
    kept = "".join(
        ch if (ch == "'" or ch == " " or (ch != "'" and ch.isalpha())) else " "
        for ch in text
    )
    return kept.split()


def normalize_cer_chars(text):
    return [ch for ch in nfkc(text) if not ch.isspace()]


# CLI stdout is `key=value` lines, but hypothesis text is multi-line and may
# itself contain '=' (e.g. a transcript line starting with "a=b").
# Terminate text_output ONLY at a line starting with a KNOWN key; any other
# `key=`-looking line is a hypothesis continuation. Observed CLI keys
# (order varies): family, task, mode, text_output.
KNOWN_CLI_KEYS = frozenset({"family", "task", "mode", "text_output"})

_KEY_RE = None


def _key_re():
    global _KEY_RE
    if _KEY_RE is None:
        import re
        _KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
    return _KEY_RE


def parse_cli_text_output(stdout):
    """Extract the text_output value from CLI stdout.

    Collects the remainder of the `text_output=` line plus following lines
    until a line starting with a KNOWN key (family/task/mode/text_output).
    Lines starting with unknown `key=` text are hypothesis continuations.
    """
    collecting = False
    parts = []
    for line in stdout.splitlines():
        m = _key_re().match(line)
        if m and m.group(1) in KNOWN_CLI_KEYS:
            if m.group(1) == "text_output":
                collecting = True
                parts.append(m.group(2))
            else:
                if collecting and parts:
                    break  # a known field after the transcript: value ended
                collecting = False
            continue
        if collecting:
            parts.append(line)
    return " ".join(parts)


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
                       extra_options=(), timeout=1800, text=""):
    cmd = [cli, "--task", "asr", "--family", family, "--model", model,
           "--backend", backend, "--threads", str(threads), "--audio", audio]
    if language:
        cmd += ["--request-option", f"language={language}"]
    for opt in extra_options:
        cmd += ["--request-option", opt]
    if text:
        cmd += ["--text", text]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"cli exit {proc.returncode}: {(proc.stderr.strip() or proc.stdout.strip())[-500:]}")
    return parse_cli_text_output(proc.stdout)


def check_manifest(manifest_path):
    """Manifest sha self-check: every row carrying an audio_path must agree
    with the file on disk, and all rows sharing one audio_path must carry the
    identical sha256. Returns nonzero on any mismatch."""
    import hashlib
    failures = []
    seen = {}
    for lineno, line in enumerate(pathlib.Path(manifest_path).read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        audio = row.get("audio_path")
        if not audio:
            continue
        try:
            actual = hashlib.sha256(pathlib.Path(audio).read_bytes()).hexdigest()
        except OSError as exc:
            failures.append(f"line {lineno} {row.get('clip_id')}: unreadable audio: {exc}")
            continue
        claimed = row.get("audio_sha256")
        if claimed != actual:
            failures.append(f"line {lineno} {row.get('clip_id')}: sha {claimed} != file {actual}")
        if audio in seen and seen[audio] != claimed:
            failures.append(f"line {lineno} {row.get('clip_id')}: sha differs from an earlier "
                            f"row for {audio}")
        seen.setdefault(audio, claimed)
    if failures:
        print("MANIFEST CHECK FAILURES:")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"manifest check ok ({len(seen)} audio files, "
          f"{sum(1 for _ in pathlib.Path(manifest_path).read_text().splitlines() if _.strip())} rows)")
    return 0


def self_test():
    """Regression checks for the parser + scoring rules. Exit nonzero on failure."""
    failures = []
    passed = [0]

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        else:
            passed[0] += 1

    # '=' inside the hypothesis must not truncate (finding 3).
    check("eq-first-token",
          parse_cli_text_output("family=g\ntask=asr\nmode=offline\n"
                                "text_output=first line here\na=b then more\nlast line\n"),
          "first line here a=b then more last line")
    check("eq-key-order",
          parse_cli_text_output("family=g\nmode=offline\ntask=asr\n"
                                "text_output=Concord returned,\n"),
          "Concord returned,")
    check("unknown-key-continues",
          parse_cli_text_output("text_output=line one\nnewfield=kept as text\n"),
          "line one newfield=kept as text")
    check("known-key-terminates",
          parse_cli_text_output("text_output=hyp text\nmode=offline\n"),
          "hyp text")
    check("empty", parse_cli_text_output("family=g\ntask=asr\n"), "")
    # scoring spot checks (pinned rules).
    check("wer-exact", score_pair("A B C", "a b c!", "en")["errors"], 0)
    check("wer-sub", score_pair("I AM HERE", "I am there", "en")["errors"], 1)
    check("cer-exact", score_pair("日本語。", "日本語。", "ja")["errors"], 0)
    check("digits-dropped",
          normalize_wer_words("call 911 now"), ["CALL", "NOW"])
    # Pinned dropping rule: digits and non-ASCII punctuation are dropped.
    # NFKC does NOT fold curly quotes to ASCII (U+2018 has no decomposition),
    # so '\u2018tis' loses the quote; only Unicode letters + ASCII ' survive.
    check("curly-quote-dropped", normalize_wer_words("\u2018tis"), ["TIS"])
    try:
        score_pair("x", "y", "xx")
        failures.append("xx-lang: no ValueError raised")
    except ValueError:
        pass
    # keyword matching rules (F): casefold + NFKC, word boundaries for
    # single ASCII words, substring for multiword / non-ASCII.
    check("kw-basic", kw_match("Concord", "Concord returned."), True)
    check("kw-case", kw_match("concord", "CONCORD RETURNED"), True)
    check("kw-boundary", kw_match("tent", "amidst the tents"), False)
    check("kw-boundary-ok", kw_match("tents", "amidst the tents."), True)
    check("kw-multiword", kw_match("returned to its place", "X returned to its place Y"), True)
    check("kw-accent", kw_match("café", "un CAFÉ noir"), True)
    check("kw-accent-ne", kw_match("têntes", "amidst the tents"), False)
    check("kw-ja", kw_match("日本語", "日本語で書かれた"), True)
    check("kw-empty", kw_match("", "anything"), False)
    stats = score_keywords(["quetzal", "brzezinski"], ["xylophone"],
                           "notch ital quetzal arrived", "XOCHITL QUETZAL ARRIVED")
    check("kw-hits", stats["hits"], ["quetzal"])
    check("kw-misses", stats["misses"], ["brzezinski"])
    check("kw-insert", stats["insertions"], [])
    check("kw-in-ref", stats["in_ref"],
          {"quetzal": True, "brzezinski": False})
    agg = aggregate_keyword_stats([
        {"clip_id": "a", "reference": "X QUETZAL Y",
         "keyword_stats": score_keywords(["quetzal"], ["xylophone"], "quetzal here", "X QUETZAL Y")},
        {"clip_id": "b", "reference": "plain MISSING text",
         "keyword_stats": score_keywords(["missing"], [], "plain text", "plain MISSING text")},
        {"clip_id": "c", "reference": "plain text",
         "keyword_stats": score_keywords(["ghost"], [], "plain text", "plain text")},
    ])
    check("kw-agg-tp", agg["tp"], 1)
    check("kw-agg-fn", agg["fn"], 1)
    check("kw-agg-fp", agg["fp"], 0)
    check("kw-agg-mislabeled", agg["mislabeled"], [("c", "ghost")])
    try:
        tr = score_translation("Concord est retourné.", "Concord est retourné.")
        check("tr-ident-chrf", round(tr["chrf"], 1), 100.0)
        check("tr-ident-bleu", round(tr["bleu"], 1), 100.0)
    except RuntimeError:
        print("  (sacrebleu missing: translation self-checks skipped)")
    if failures:
        print("SELF-TEST FAILURES:")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"self-test ok ({passed[0]} checks)")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="",
                        help="eval_manifest.jsonl path (required unless --self-test)")
    parser.add_argument("--cli", default="")
    parser.add_argument("--family", default="granite_speech_2b")
    parser.add_argument("--model", default="",
                        help="model path (required unless --self-test)")
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--threads", default="8")
    parser.add_argument("--label", default="")
    parser.add_argument("--out", default="",
                        help="optional path to write the JSON result rows")
    parser.add_argument("--self-test", action="store_true",
                        help="run parser/scoring regression checks, no CLI needed")
    parser.add_argument("--check-manifest", default="",
                        help="verify every manifest row's audio_sha256 against the file "
                        "on disk (all rows per path share one sha); no CLI needed")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.check_manifest:
        return check_manifest(args.check_manifest)
    for need in ("manifest", "cli", "model"):
        if not getattr(args, need):
            parser.error(f"--{need} is required unless --self-test is given")

    rows = []
    for line in pathlib.Path(args.manifest).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    ready = [r for r in rows if r.get("status") == "ready"]
    pending = [r for r in rows if r.get("status") != "ready"]

    scored = []
    kw_rows = []
    tr_rows = []
    for row in ready:
        ref_path = pathlib.Path(row["reference_path"])
        reference = ref_path.read_text(encoding="utf-8").strip()
        task = row.get("task", "asr")
        hypothesis = run_cli_transcribe(
            args.cli, args.family, args.model, row["audio_path"],
            args.backend, args.threads, row.get("request_language", ""),
            row.get("request_options", []), text=row.get("text", ""))
        if task == "translation":
            result = {"metric": "chrF/BLEU", "unit": "segment",
                      "language": row.get("target_language", "")}
            result.update(score_translation(reference, hypothesis))
            tr_rows.append({**result, "clip_id": row["clip_id"],
                           "hypothesis": hypothesis, "reference": reference,
                           "target_language": row.get("target_language", "")})
            print(f"{row['clip_id']:44s} chrF "
                  f"{result['chrf']:6.2f}  BLEU {result['bleu']:6.2f}  hyp={hypothesis[:80]!r}")
        elif task == "keyword":
            kw = row.get("keywords", {})
            result = score_pair(reference, hypothesis, row["language"])
            kw_stats = score_keywords(kw.get("present", []), kw.get("absent", []),
                                      hypothesis, reference)
            print(f"{row['clip_id']:44s} {result['metric']} "
                  f"{result['rate_pct']:6.2f}%  {result['errors']:3d}/"
                  f"{result['denominator']:<4d} s/d/i="
                  f"{result['subs']}/{result['dels']}/{result['inss']}  "
                  f"kw hits={len(kw_stats['hits'])}/{len(kw_stats['present'])} "
                  f"misses={kw_stats['misses']} insertions={kw_stats['insertions']}"
                  f"{'   <- NO OUTPUT' if result['empty_hypothesis'] else ''}")
            kw_rows.append({"clip_id": row["clip_id"], "language": row["language"],
                            "hypothesis": hypothesis, "reference": reference,
                            "wer_cer": result, "keyword_stats": kw_stats})
            result.update({"clip_id": row["clip_id"], "language": row["language"],
                           "hypothesis": hypothesis, "reference": reference,
                           "keyword_stats": kw_stats})
            scored.append(result)
        else:
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
    if kw_rows:
        kw_agg = aggregate_keyword_stats(kw_rows)
        print(f"\nkeywords: P {kw_agg['precision_pct']:.1f}%  "
              f"R {kw_agg['recall_pct']:.1f}%  F1 {kw_agg['f1_pct']:.1f}%  "
              f"(tp={kw_agg['tp']} fp={kw_agg['fp']} fn={kw_agg['fn']})")
        if kw_agg["mislabeled"]:
            print(f"mislabeled fixtures (term not in reference, excluded): "
                  f"{kw_agg['mislabeled']}")
        summary["keywords"] = kw_agg
    if tr_rows:
        tr_agg = aggregate_translation(tr_rows)
        print(f"\ntranslation: chrF {tr_agg['chrf']:.2f}  BLEU {tr_agg['bleu']:.2f}  "
              f"({tr_agg['clips']} clips)")
        summary["translation"] = tr_agg
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
