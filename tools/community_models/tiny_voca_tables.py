#!/usr/bin/env python3
"""Write the two generated files of the dizione C++ port.

    python tools/community_models/tiny_voca_tables.py --dizione /path/to/dizione

``unicode_tables.inc`` holds the Unicode data the port needs to behave like
Python: character classes, case mappings, NFKC/NFKD and the case rules of the
``re`` module. It is read from the Python that runs this script, so the C++
code agrees with that Python version.

``tables_data.inc`` holds dizione's hand-written tables (``data/*.txt``) as
they are in the checkout given with ``--dizione``.
"""

import argparse
from pathlib import Path
import re
import subprocess
import sys
import unicodedata

import _sre
from re import _casefix

LIMIT = 0x110000
HANGUL_FIRST, HANGUL_LAST = 0xAC00, 0xD7A3
SIGMA, FINAL_SIGMA = chr(0x3A3), chr(0x3C2)
OUTPUT = Path(__file__).resolve().parents[2] / "src/community_models/tiny_voca/dizione"


def ranges(test):
    """Inclusive ranges of the code points for which ``test`` holds."""
    out, start = [], None
    for code in range(LIMIT):
        if test(code):
            if start is None:
                start = code
        elif start is not None:
            out.append((start, code - 1))
            start = None
    if start is not None:
        out.append((start, LIMIT - 1))
    return out


def numbers(values, width=14):
    values = [str(value) for value in values]
    return "\n".join("    " + ", ".join(values[i:i + width]) + "," for i in range(0, len(values), width))


def array(name, values, kind="uint32_t"):
    values = list(values)
    body = numbers(values) if values else "    0,"
    return f"static const {kind} {name}[] = {{\n{body}\n}};\nstatic const size_t {name}Count = {len(values)};\n"


def flat(pairs):
    return [value for pair in pairs for value in pair]


def unicode_tables():
    char = chr
    parts = []
    properties = (
        ("kAlpha", lambda c: char(c).isalpha()),
        ("kDecimal", lambda c: char(c).isdecimal()),
        ("kDigit", lambda c: char(c).isdigit()),
        ("kNumeric", lambda c: char(c).isnumeric()),
        ("kSpace", lambda c: char(c).isspace()),
        ("kUpper", lambda c: char(c).isupper()),
        ("kLower", lambda c: char(c).islower()),
        ("kTitle", lambda c: char(c).istitle() and not char(c).isupper()),
        # The two properties str.lower() consults for a capital sigma. A sigma is
        # final after a cased letter and before none; letters that are
        # "case-ignorable" are skipped on both sides.
        ("kCaseIgnorable", lambda c: ("a" + char(c) + SIGMA).lower()[-1] == FINAL_SIGMA
                                     and ("a" + SIGMA + char(c)).lower()[1] == FINAL_SIGMA),
        ("kCasedNotIgnorable", lambda c: (char(c) + SIGMA).lower()[-1] == FINAL_SIGMA),
        ("kSreCased", _sre.unicode_iscased),
    )
    for name, test in properties:
        parts.append(array(name, flat(ranges(test))))

    # Decimal digits come in runs of ten; the zero of each run is enough.
    zeros = [c for c in range(LIMIT) if char(c).isdecimal() and unicodedata.decimal(char(c)) == 0]
    for c in range(LIMIT):
        if char(c).isdecimal():
            value = unicodedata.decimal(char(c))
            assert c - value in zeros, hex(c)
    parts.append(array("kDecimalZeros", zeros))

    # Case mappings. The simple lower-case mapping is the one the re module uses.
    simple_lower = [(c, _sre.unicode_tolower(c)) for c in range(LIMIT) if _sre.unicode_tolower(c) != c]
    parts.append(array("kSimpleLower", flat(simple_lower)))
    special_lower = []
    for c in range(LIMIT):
        if c == 0x3A3:
            continue                                   # the sigma is handled in code
        full = char(c).lower()
        if full != char(_sre.unicode_tolower(c)):
            special_lower += [c, len(full)] + [ord(x) for x in full]
    assert SIGMA.lower() == chr(0x3C3)
    parts.append(array("kSpecialLower", special_lower))
    simple_upper, special_upper = [], []
    for c in range(LIMIT):
        full = char(c).upper()
        if full == char(c):
            continue
        if len(full) == 1:
            simple_upper += [c, ord(full)]
        else:
            special_upper += [c, len(full)] + [ord(x) for x in full]
    parts.append(array("kSimpleUpper", simple_upper))
    parts.append(array("kSpecialUpper", special_upper))
    extra = []
    for low, others in sorted(_casefix._EXTRA_CASES.items()):
        extra += [low, len(others)] + list(others)
    parts.append(array("kExtraCases", extra))

    # Normalisation: the full compatibility decomposition of every character
    # that has one (Hangul syllables are computed), combining classes, and the
    # pairs that compose.
    keys, offsets, pool = [], [0], []
    for c in range(LIMIT):
        if HANGUL_FIRST <= c <= HANGUL_LAST:
            continue
        decomposed = unicodedata.normalize("NFKD", char(c))
        if decomposed != char(c):
            keys.append(c)
            pool += [ord(x) for x in decomposed]
            offsets.append(len(pool))
    parts.append(array("kNfkdKeys", keys))
    parts.append(array("kNfkdOffsets", offsets))
    parts.append(array("kNfkdPool", pool))
    classes, start = [], None
    for c in range(LIMIT + 1):
        value = unicodedata.combining(char(c)) if c < LIMIT else 0
        if start is not None and value != start[1]:
            classes += [start[0], c - 1, start[1]]
            start = None
        if value and start is None:
            start = (c, value)
    parts.append(array("kCombining", classes))
    compose = []
    for c in range(LIMIT):
        if HANGUL_FIRST <= c <= HANGUL_LAST:
            continue
        mapping = unicodedata.decomposition(char(c))
        if not mapping or mapping.startswith("<"):
            continue
        codes = [int(code, 16) for code in mapping.split()]
        if len(codes) == 2 and unicodedata.normalize("NFC", char(codes[0]) + char(codes[1])) == char(c):
            compose.append((codes[0], codes[1], c))
    compose.sort()
    parts.append(array("kCompose", flat(compose)))

    header = (
        "// Generated by tools/community_models/tiny_voca_tables.py. Do not edit.\n"
        f"// Unicode {unicodedata.unidata_version}, read from CPython {sys.version.split()[0]}.\n"
        "// Ranges are inclusive pairs; maps are flat lists as described in unicode.cpp.\n\n"
    )
    return header + "\n".join(parts)


def literal(line):
    """One line of a table as a C string literal, the newline included."""
    out = []
    for byte in line.encode("utf-8") + b"\n":
        if byte == 0x0A:
            out.append("\\n")
        elif byte in (0x22, 0x5C, 0x3F):                # quote, backslash, question mark (trigraphs)
            out.append("\\" + chr(byte))
        elif 0x20 <= byte < 0x7F:
            out.append(chr(byte))
        else:
            out.append(f"\\{byte:03o}")
    return '"' + "".join(out) + '"'


def data_tables(dizione):
    data = dizione / "src/dizione/data"
    commit = subprocess.run(["git", "-C", str(dizione), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    parts = [
        "// Generated by tools/community_models/tiny_voca_tables.py. Do not edit.\n"
        f"// dizione's hand-written tables (src/dizione/data/*.txt) at commit {commit or 'unknown'}.\n"
    ]
    names = []
    for path in sorted(data.glob("*.txt")):
        name = re.sub(r"[^A-Za-z0-9]", "_", path.stem)
        names.append((path.name, name))
        lines = path.read_text(encoding="utf-8").split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        parts.append(f"static const char kTable_{name}[] =\n" + "\n".join("    " + literal(line) for line in lines) + ";\n")
    parts.append("static const EmbeddedTable kEmbeddedTables[] = {\n"
                 + "\n".join(f'    {{"{file}", kTable_{name}, sizeof(kTable_{name}) - 1}},' for file, name in names) + "\n};\n")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dizione", type=Path, help="A checkout of https://github.com/n4ze3m/dizione (for the hand-written tables)")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "unicode_tables.inc").write_text(unicode_tables(), encoding="ascii")
    print("wrote", args.out / "unicode_tables.inc")
    if args.dizione:
        (args.out / "tables_data.inc").write_text(data_tables(args.dizione), encoding="ascii")
        print("wrote", args.out / "tables_data.inc")


if __name__ == "__main__":
    main()
