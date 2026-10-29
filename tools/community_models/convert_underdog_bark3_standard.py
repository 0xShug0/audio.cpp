#!/usr/bin/env python3
"""Convert the pinned standard Underdog Bark 3.0 checkpoint to native GGUF.

Reuses the packed Bark reader and precision policy; the immutable profile keeps
standard/Plus checkpoints, variants and provenance separate. Python/ANS is only
needed for conversion, never native inference.
"""
from convert_underdog_bark3_plus import BarkPackedProfile, main

SOURCE_REPO = "ConwayResearch/Underdog-Bark-3.0"
SOURCE_REVISION = "85db2ad18ab41e534a222ee253c29191d77ed919"
SOURCE_SHA256 = "c18de9418d73cbeb80b8c1287b392d480a56e191ef4a81d1baaa1b02e66d92ed"
VARIANT = "underdog_bark3_standard"
STANDARD_PROFILE = BarkPackedProfile(SOURCE_REPO, SOURCE_REVISION, SOURCE_SHA256,
                                    VARIANT, "Underdog Bark 3.0", "docs/models/underdog_bark3_standard.md")


if __name__ == "__main__":
    main(STANDARD_PROFILE, __doc__)
