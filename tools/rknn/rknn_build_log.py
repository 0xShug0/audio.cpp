#!/usr/bin/env python3
"""Gate on the RKNN compiler's register-field diagnostics.

`rknn.build` reports an op encoding whose field exceeds the DPU's 13-bit register width by
printing

    REGTASK: The bit width of field value exceeds the limit, ... limit: 0x1fff, value: 0x7800

from the C++ compiler on file descriptors 1 and 2, not through Python logging, so a gate on
those lines has to redirect the descriptors rather than `sys.stdout`.

The lines are not by themselves a failure. The RK3566 builds of the three shipping engines log 352
(Kokoro w128 generator), 355 (Piper w93 vocoder) and 6570 (Qwen3 B=64 vocoder) of them and measure
clean on the board; the RK3588 build of the Piper w93 vocoder logs none. What correlates with a
dead engine is the magnitude of the count: the Kokoro W=512 build, whose engine voices clipped
noise (measured rms 32764 of 32767, SNR -27 dB against the CPU decode), logs 403 587 of them, with
the peak value 0x7800 = 30720, nearly four times past the 0x1fff field. Hence a threshold between
the two regimes, not a zero tolerance.
"""
from __future__ import annotations

import contextlib
import os
import re
import sys
from pathlib import Path

REGTASK_FATAL_LINES = 100_000

_VALUE = re.compile(r"REGTASK.*?value: 0x([0-9a-fA-F]+)")


@contextlib.contextmanager
def rknn_build_gate(label: str, log_path: Path, fatal: int = REGTASK_FATAL_LINES):
    """Run the block (the `rknn.build` call) with fds 1/2 captured to `log_path`, then count
    the register-field overflow lines and abort past `fatal` of them. The whole build log is
    kept at `log_path` for inspection."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w+") as log:
        saved = (os.dup(1), os.dup(2))
        try:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            yield log
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
        log.flush()
        log.seek(0)
        text = log.read()

    values = [int(v, 16) for v in _VALUE.findall(text)]
    if not values:
        bare = text.count("REGTASK")
        if bare:
            print(f"{label}: {bare} register-field overflow lines (no value field parsed) "
                  f"-> {log_path}")
        return
    print(f"{label}: {len(values)} register-field overflow lines "
          f"(peak value 0x{max(values):x}, limit 0x1fff) -> {log_path}")
    if len(values) >= fatal:
        raise SystemExit(
            f"{label}: {len(values)} register-field overflow lines is the signature of a build "
            f"whose engine voices saturated output (measured: 403 587 lines at the Kokoro W=512 "
            f"window, rms 32764 of 32767, SNR -27 dB against the CPU decode; the shipping builds "
            f"log 352 to 6570). Shrink the window, or split the axis that carries the long "
            f"dimension so no single op sees more than 8191 of it."
        )


def _self_test() -> int:
    """Both regimes, and that the capture really sees C-level writes to fds 1/2."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        line = b"REGTASK: The bit width of field value exceeds the limit, limit: 0x1fff, value: 0x7800\n"
        benign = Path(tmp) / "benign.log"
        with rknn_build_gate("benign", benign):
            os.write(1, line * 352)
            os.write(2, b"plain build noise\n")
        text = benign.read_text()
        assert text.count("REGTASK") == 352 and "plain build noise" in text, text[:120]
        fatal = Path(tmp) / "fatal.log"
        try:
            with rknn_build_gate("fatal", fatal):
                os.write(1, line * REGTASK_FATAL_LINES)
        except SystemExit as exit_:
            assert "100000 register-field" in str(exit_), str(exit_)[:120]
        else:
            raise AssertionError("a fatal build did not abort the gate")
        os.write(1, b"fds restored\n")
    print("rknn_build_log self-test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
