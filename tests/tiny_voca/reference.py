#!/usr/bin/env python3
"""The Python side of the checks for the dizione C++ port.

Each command writes what Python gives; ``dizione_probe`` reads the same input
and must give the same. Run it with a Python that has dizione on its path
(``PYTHONPATH=/path/to/dizione/src``) and ``DIZIONE_DATA`` set to the built
phonemizer files.

    reference.py unicode  OUT              string methods and normalisation
    reference.py regex    TEXTS OUT        every pattern dizione uses, on the texts it meets
    reference.py ids      TEXTS OUT        phoneme ids and stress, one line per text
    reference.py tokens   TEXTS OUT        the normaliser's words and marks
    reference.py units    TEXTS OUT        each word with its phones and where they came from
    reference.py words    WORDS OUT        single words: phones and source

``TEXTS`` holds one text per line.
"""

import argparse
import json
import random
import re
import sys
import unicodedata

LIMIT = 0x110000


def codes(text):
    return " ".join(f"{ord(c):X}" for c in text)


# ---- string methods ---------------------------------------------------------

def unicode_cases(out):
    random.seed(7)
    write = out.write
    for code in range(LIMIT):
        c = chr(code)
        bits = (c.isalpha() | c.isdecimal() << 1 | c.isdigit() << 2 | c.isnumeric() << 3 | c.isspace() << 4 | c.isupper() << 5
                | c.islower() << 6 | c.isalnum() << 7 | (re.fullmatch(r"\w", c) is not None) << 8
                | (re.fullmatch(r"\d", c) is not None) << 9 | (re.fullmatch(r"\s", c) is not None) << 10)
        write(f"P\t{code:X}\t{bits}\t{unicodedata.combining(c)}\n")
    interesting = [chr(c) for c in list(range(0x20, 0x250)) + list(range(0x2B0, 0x370)) + list(range(0x370, 0x400))
                   + list(range(0x1100, 0x1200)) + list(range(0xAC00, 0xAC80)) + list(range(0x1E00, 0x2000))
                   + list(range(0x2000, 0x2200)) + list(range(0xFB00, 0xFB10)) + list(range(0xFF00, 0xFF60))
                   + [0x3A3, 0x3C3, 0x3C2, 0x130, 0x131, 0xDF, 0x1E9E, 0x345, 0x1D400, 0x1F600, 0x10400, 0x10428]]
    texts = [chr(code) for code in range(LIMIT)]
    for _ in range(200000):
        size = random.randint(2, 9)
        texts.append("".join(random.choice(interesting) for _ in range(size)))
    for text in texts:
        for name, value in (("L", text.lower()), ("U", text.upper()), ("C", unicodedata.normalize("NFKC", text)),
                            ("D", unicodedata.normalize("NFKD", text))):
            if len(text) > 1 or value != text:
                write(f"{name}\t{codes(text)}\t{codes(value)}\n")
        if len(text) > 1:
            write(f"u\t{codes(text)}\t{int(text.isupper())}\n")
            write(f"l\t{codes(text)}\t{int(text.islower())}\n")
            write(f"S\t{codes(text)}\t{'|'.join(codes(part) for part in text.split())}\n")
            write(f"T\t{codes(text)}\t{codes(text.strip())}\n")
    arabic, wide = "".join(map(chr, (0x661, 0x662, 0x663))), "".join(map(chr, (0xFF11, 0xFF12)))
    for text in ("0", "007", " 42 ", "1_000", arabic, "-5", "+7", "", "12a", "_1", "1__0", wide):
        try:
            value = str(int(text))
        except ValueError:
            value = "ValueError"
        write(f"I\t{codes(text)}\t{value}\n")


# ---- regular expressions ----------------------------------------------------

class Recorder:
    """Stands in for ``re._compile``: every pattern dizione compiles is kept
    with a sample of the texts it was used on."""

    def __init__(self, keep=250):
        self.keep = keep
        self.real = re._compile
        self.seen = {}

    def install(self):
        recorder = self

        class Proxy:
            def __init__(self, compiled):
                self._compiled = compiled
                key = (compiled.pattern, compiled.flags & (re.IGNORECASE | re.ASCII))
                self._texts = recorder.seen.setdefault(key, ({}, {}))

            def _note(self, text):
                hit, miss = self._texts
                bucket = hit if self._compiled.search(text) else miss
                if len(bucket) < (recorder.keep if bucket is hit else recorder.keep // 3):
                    bucket[text] = None

            def __getattr__(self, name):
                return getattr(self._compiled, name)

        def method(name):
            def call(self, *args, **kwargs):
                text = args[1] if name in ("sub", "subn") else args[0]
                self._note(text)
                return getattr(self._compiled, name)(*args, **kwargs)
            return call

        for name in ("sub", "subn", "search", "match", "fullmatch", "finditer", "findall", "split"):
            setattr(Proxy, name, method(name))

        def from_dizione():
            frame = sys._getframe(2)
            while frame is not None and frame.f_globals.get("__name__", "").split(".")[0] == "re":
                frame = frame.f_back
            return frame is not None and frame.f_globals.get("__name__", "").split(".")[0] == "dizione"

        def compile_(pattern, flags):
            if isinstance(pattern, Proxy):
                return pattern
            compiled = recorder.real(pattern, flags)
            return Proxy(compiled) if isinstance(compiled.pattern, str) and from_dizione() else compiled

        re._compile = compile_

    def uninstall(self):
        re._compile = self.real


def spans(match):
    return [list(match.span(group)) for group in range(match.re.groups + 1)]


def expected(compiled, text):
    found = [spans(match) for match in compiled.finditer(text)]
    first = compiled.match(text)
    whole = compiled.fullmatch(text)
    return {"all": found, "match": spans(first) if first else None, "full": spans(whole) if whole else None,
            "split": compiled.split(text) if compiled.groups == 0 else None}


def mutate(text, alphabet):
    chars = list(text)
    for _ in range(random.randint(1, 3)):
        kind = random.random()
        place = random.randint(0, len(chars))
        if kind < 0.4 or not chars:
            chars.insert(place, random.choice(alphabet))
        elif kind < 0.7:
            del chars[min(place, len(chars) - 1)]
        elif kind < 0.9:
            chars[min(place, len(chars) - 1)] = random.choice(alphabet)
        else:
            piece = chars[max(0, place - 4):place]
            chars[place:place] = piece
    return "".join(chars)


def regex_cases(texts, out, keep=250, changed=3):
    recorder = Recorder(keep)
    recorder.install()
    import dizione
    from dizione.engine import Phonemizer
    phonemizer = Phonemizer()
    for text in texts:
        try:
            phonemizer.pieces(text)
        except ValueError:
            pass
    recorder.uninstall()
    random.seed(11)
    # Signs the rules look for, letters whose case is special, the marks the
    # normaliser puts in the text, a digit and a letter of other scripts, an emoji.
    alphabet = list("abcdeAZxk019 .,;:!?'\"()[]{}<>=+-*/^%$#@&_~\n\t") + [chr(code) for code in (
        0x2019, 0x2013, 0x2014, 0x2026, 0xA3, 0x20AC, 0xBD, 0xB2, 0x2265, 0x2264, 0x2260, 0xE9, 0xDF, 0x212A, 0x17F,
        0xE000, 0xE001, 0xE006, 0x661, 0x4E2D, 0x1F600)]
    count = 0
    for (pattern, flags), (hit, miss) in recorder.seen.items():
        compiled = recorder.real(pattern, flags)
        subjects = dict.fromkeys(list(hit) + list(miss))
        for text in list(hit)[:keep // 4] + list(miss)[:keep // 12]:
            for _ in range(changed):
                subjects[mutate(text, alphabet)] = None
        for text in subjects:
            out.write(json.dumps({"p": pattern, "f": flags, "s": text, **expected(compiled, text)}) + "\n")
            count += 1
    print(f"{len(recorder.seen)} patterns, {count} cases", file=sys.stderr)
    return [{"p": pattern, "f": flags} for pattern, flags in recorder.seen]


# ---- phonemes ---------------------------------------------------------------

def show_phones(phones):
    return " ".join(f"{phone}/{stress}" for phone, stress in phones) if phones is not None else "-"


def ids(texts, out):
    import dizione
    for text in texts:
        try:
            result = dizione.phonemize(text)
            out.write(" ".join(map(str, result.ids)) + "\t" + " ".join(map(str, result.stress)) + "\n")
        except Exception as error:      # noqa: BLE001 - the kind of error is part of the answer
            out.write(f"ERROR {type(error).__name__}\n")


def tokens(texts, out):
    from dizione import get, normalize
    phonemizer = get()
    for text in texts:
        try:
            parts = []
            for token in phonemizer.normalizer.tokens(text):
                if isinstance(token, normalize.Punct):
                    parts.append("P:" + token.mark)
                else:
                    parts.append(f"W:{token.text}|{int(token.attach)}|{token.raw}|{int(token.spell)}|{show_phones(token.phones)}")
            out.write("\x1f".join(parts) + "\n")
        except Exception as error:      # noqa: BLE001
            out.write(f"ERROR {type(error).__name__}\n")


def units(texts, out):
    from dizione import get
    phonemizer = get()
    for text in texts:
        try:
            parts = []
            for kind, value in phonemizer.analyse(text):
                if kind == "punct":
                    parts.append("P:" + value)
                else:
                    parts += [f"U:{unit.word}|{unit.kind}|{int(unit.attach)}|{unit.source}|{show_phones(unit.phones)}" for unit in value]
            out.write("\x1f".join(parts) + "\n")
        except Exception as error:      # noqa: BLE001
            out.write(f"ERROR {type(error).__name__}\n")


def words(texts, out):
    from dizione import get
    phonemizer = get()
    for word in texts:
        try:
            phones, source = phonemizer.word(word)
            out.write(f"{source}\t{show_phones(phones)}\n")
        except Exception as error:      # noqa: BLE001
            out.write(f"ERROR {type(error).__name__}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("unicode", "regex", "ids", "tokens", "units", "words"))
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--keep", type=int, default=250, help="regex: texts kept for each pattern")
    parser.add_argument("--changed", type=int, default=3, help="regex: altered copies made of a kept text")
    args = parser.parse_args()
    if args.command == "unicode":
        with open(args.paths[0], "w", encoding="utf-8") as out:
            unicode_cases(out)
        return
    with open(args.paths[0], encoding="utf-8") as source:
        texts = [line.rstrip("\n") for line in source]
    with open(args.paths[1], "w", encoding="utf-8") as out:
        if args.command == "regex":
            patterns = regex_cases(texts, out, args.keep, args.changed)
            if len(args.paths) > 2:
                with open(args.paths[2], "w", encoding="utf-8") as listing:
                    json.dump(patterns, listing, ensure_ascii=True, indent=0)
        else:
            {"ids": ids, "tokens": tokens, "units": units, "words": words}[args.command](texts, out)


if __name__ == "__main__":
    main()
