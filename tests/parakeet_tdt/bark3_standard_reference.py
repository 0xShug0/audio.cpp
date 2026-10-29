#!/usr/bin/env python3
"""Run standard Bark 3.0 through its pinned, unmodified official loader.

Uses the same reference procedure as Plus, with the standard checkpoint's own
immutable source profile. No audio.cpp staged weights are loaded.
"""
from bark3_plus_reference import main
from convert_underdog_bark3_standard import STANDARD_PROFILE


if __name__ == "__main__":
    main(STANDARD_PROFILE, __doc__)
