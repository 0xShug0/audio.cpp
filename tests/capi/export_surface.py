#!/usr/bin/env python3
"""Assert the shared library exports exactly the C ABI and nothing else.

Three platforms reach this guarantee three different ways -- a version script on
ELF, an exported-symbols list on Mach-O, and on Windows merely the absence of
any dependency that annotates its own symbols, because __declspec(dllexport) is
additive and there is no allowlist equivalent. The Windows case has no mechanism
enforcing it at all: a newly vendored library that exports its own symbols would
silently widen the surface. That happened once already -- cJSON put the first
Windows build at 147 exports against the 69 intended.

So the invariant is checked here rather than left to a comment.

Exits 0 when the surface matches, 1 when it does not, and 77 (the CTest skip
code) when the library or the platform's symbol tool is unavailable.
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

SKIP = 77


def find_dumpbin(library, explicit=None):
    """Locate dumpbin, which is not on PATH outside a Developer Command Prompt.

    Three layers, most reliable first:

    1. Whatever CMake passed in. dumpbin ships beside the cl.exe that built the
       library, so CMake already knows exactly where the right one is -- no
       searching, no version guessing, and guaranteed to match the toolset that
       produced the binary being inspected.
    2. PATH, for a Developer Command Prompt.
    3. vswhere, for running this script standalone. Deliberately WITHOUT
       -latest: that returns a single install, which would leave nothing to fall
       back to when the newest one is registered but incomplete -- a real Windows
       state, and one this project hit during testing.

    Candidates are probed with the operation actually needed rather than with
    help text. `dumpbin /?` exits 1100, not 0, so a returncode==0 check against
    it rejects every candidate on every machine.
    """
    if explicit and os.path.exists(explicit):
        return explicit

    found = shutil.which("dumpbin")
    if found:
        return found
    if os.name != "nt":
        return None

    program_files = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = os.path.join(program_files, "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.exists(vswhere):
        return None

    try:
        completed = subprocess.run(
            [vswhere, "-products", "*", "-property", "installationPath"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None

    def toolset_version(path):
        # .../VC/Tools/MSVC/14.44.35207/bin/... -> (14, 44, 35207) for ordering.
        match = re.search(r"MSVC[\\/]+([0-9.]+)[\\/]+", path)
        if not match:
            return ()
        return tuple(int(part) for part in match.group(1).split(".") if part.isdigit())

    candidates = []
    for install in completed.stdout.splitlines():
        install = install.strip()
        if not install:
            continue
        for host in ("Hostx64", "Hostx86"):
            for arch in ("x64", "x86"):
                candidates.extend(glob.glob(os.path.join(
                    install, "VC", "Tools", "MSVC", "*", "bin", host, arch, "dumpbin.exe")))

    for candidate in sorted(candidates, key=toolset_version, reverse=True):
        try:
            probe = subprocess.run([candidate, "/EXPORTS", library],
                                   capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0:
            return candidate

    if candidates:
        print(f"vswhere found {len(candidates)} dumpbin candidate(s), none of which ran:")
        for candidate in sorted(candidates, key=toolset_version, reverse=True)[:5]:
            print(f"  {candidate}")
    return None


def find_library(build_dir):
    """Locate the built shared library, whatever this platform calls it.

    Multi-config generators -- Visual Studio, Xcode, Ninja Multi-Config -- put
    outputs in a per-configuration subdirectory, so bin/ alone is not enough on
    any platform. Getting this wrong means skipping on a tree where the library
    is sitting right there, which reads as "not applicable" and passes.

    Returns (kind, path, searched) so a skip can say where it looked.
    """
    names = (
        ("elf", "libaudiocpp.so"),
        ("macho", "libaudiocpp.dylib"),
        ("pe", "audiocpp.dll"),
    )
    configs = ("", "Release", "RelWithDebInfo", "MinSizeRel", "Debug")

    searched = []
    for config in configs:
        for kind, name in names:
            path = os.path.join(build_dir, "bin", config, name)
            searched.append(path)
            if os.path.exists(path):
                return kind, os.path.realpath(path), searched
    return None, None, searched


def declared_symbols(header_path):
    """Every audiocpp_* entry point the header declares."""
    with open(header_path, encoding="utf-8") as handle:
        source = handle.read()
    # Strip comments so a name mentioned in prose is not mistaken for an entry point.
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return set(re.findall(r"AUDIOCPP_API[^;]*?\b(audiocpp_[a-z0-9_]+)\s*\(", source, flags=re.S))


def exported_symbols(kind, library, symbol_tool=None):
    """Every symbol the built library actually exports."""
    if kind == "elf":
        if not shutil.which("nm"):
            return None
        out = subprocess.run(["nm", "-D", "--defined-only", library],
                             capture_output=True, text=True, check=True).stdout
        names = set()
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[1] in ("T", "W"):
                names.add(parts[2].split("@@")[0])
        return names

    if kind == "macho":
        if not shutil.which("nm"):
            return None
        out = subprocess.run(["nm", "-gU", library],
                             capture_output=True, text=True, check=True).stdout
        # Mach-O prefixes C symbols with exactly one underscore. Stripping only
        # that one, rather than all leading underscores, so a name that legitimately
        # begins with one survives intact.
        def unprefix(name):
            return name[1:] if name.startswith("_") else name

        return {unprefix(line.split()[-1]) for line in out.splitlines() if line.split()}

    if kind == "pe":
        dumpbin = find_dumpbin(library, symbol_tool)
        if not dumpbin:
            return None
        environment = dict(os.environ)
        environment["PATH"] = os.path.dirname(dumpbin) + os.pathsep + environment.get("PATH", "")
        out = subprocess.run([dumpbin, "/EXPORTS", library], env=environment,
                             capture_output=True, text=True, check=True).stdout
        names = set()
        # Rows are: ordinal hint RVA name
        for line in out.splitlines():
            match = re.match(r"\s*\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)", line)
            if match:
                names.add(match.group(1))
        return names

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--header", required=True)
    parser.add_argument("--symbol-tool", default=None,
                        help="Path to the platform's symbol tool. CMake passes dumpbin's "
                             "location on MSVC, since it sits beside the compiler.")
    parser.add_argument("--require-tool", action="store_true",
                        help="Fail instead of skipping when no symbol tool is found. Set on "
                             "MSVC builds: the tool ships with the compiler, so its absence "
                             "means the invariant goes unchecked on the one platform with no "
                             "allowlist to fall back on.")
    args = parser.parse_args()

    # Checked here rather than inside the PE branch so it is reported on every
    # platform. Falling through to discovery is right -- a working tool beats no
    # tool -- but doing it silently would mean reporting results from a tool the
    # caller did not choose, which is the same reads-right-but-isn't shape this
    # test exists to catch.
    if args.symbol_tool and not os.path.exists(args.symbol_tool):
        print(f"warning: --symbol-tool {args.symbol_tool} does not exist; "
              "falling back to discovery")

    kind, library, searched = find_library(args.build_dir)
    if library is None:
        print(f"no shared library under {args.build_dir}; build the audiocpp target first.")
        print("Looked for:")
        for path in searched:
            print(f"  {path}")
        print("Skipping.")
        return SKIP

    exported = exported_symbols(kind, library, args.symbol_tool)
    if exported is None:
        hint = (" (not on PATH, and vswhere offered nothing usable)" if kind == "pe" else "")
        if args.require_tool:
            print(f"no symbol tool for {kind}{hint}, and --require-tool was set.")
            print("Skipping here would report a green run with the export surface")
            print("unchecked, which is the gap this test exists to close.")
            if kind == "pe":
                print("Windows has no export allowlist to fall back on, so this is the")
                print("platform where that matters most. Run from a Developer Command")
                print("Prompt, or pass --symbol-tool with dumpbin's path.")
            else:
                print("Install the platform's symbol tool, or pass --symbol-tool.")
            return 1
        print(f"no symbol tool available for {kind} on this platform{hint}; skipping")
        return SKIP

    declared = declared_symbols(args.header)
    if not declared:
        print(f"parsed no declarations out of {args.header}; refusing to pass vacuously")
        return 1

    foreign = {name for name in exported if not name.startswith("audiocpp_")}
    missing = declared - exported
    extra = {name for name in exported if name.startswith("audiocpp_")} - declared

    print(f"library:  {library} ({kind})")
    print(f"declared: {len(declared)}   exported: {len(exported)}")

    if not foreign and not missing and not extra:
        print(f"export surface matches the header exactly ({len(declared)} symbols)")
        return 0

    if foreign:
        print(f"\n{len(foreign)} symbol(s) leaked from a dependency:")
        for name in sorted(foreign)[:40]:
            print(f"  {name}")
        if len(foreign) > 40:
            print(f"  ... and {len(foreign) - 40} more")
        print("\nOn ELF and Mach-O, add them to src/capi/audiocpp.map or audiocpp.symbols.")
        print("On Windows there is no allowlist: find the dependency annotating its")
        print("own symbols and build it with its equivalent of CJSON_HIDE_SYMBOLS.")
    if missing:
        print(f"\n{len(missing)} declared but not exported: {sorted(missing)}")
    if extra:
        print(f"\n{len(extra)} exported but not declared: {sorted(extra)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
