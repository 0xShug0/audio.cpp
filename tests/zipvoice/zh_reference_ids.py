#!/usr/bin/env python3
"""Dump reference EmiliaTokenizer token ids for the zh-token parity test.

Run with the upstream ZipVoice Python environment:
  /path/to/ZipVoice/.venv/bin/python tests/zipvoice/zh_reference_ids.py \
      --tokenizer-repo /path/to/ZipVoice \
      --model-dir /models/ZipVoice/zipvoice_distill \
      --output /tmp/zipvoice/zh_reference_ids.json
"""
import argparse
import json
import sys
from pathlib import Path


CORPUS = [
    "今夜的月光如此清亮，不做些什么真是浪费。随我一同去月下漫步吧，不许拒绝。",
    "你好，世界。",
    "老虎养殖场里养着几只老虎。",
    "一点诚意都没有，不见不散。",
    "一直走，不要停，第一个路口右转。",
    "我在银行门口的长凳上休息了很久。",
    "重庆的重量单位和别处不一样吗？",
    "2024年3月14日，第2名，50元。",
    "我用 audio.cpp 合成语音，效果不错！",
    "This is English text inside.",
    "<zhong1> <guo2>人最棒。",
    "他拿着画，走向画室。",
    "买东西的时候记得带伞。",
    "地得工工作很努力。",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-repo", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import logging
    logging.disable(logging.CRITICAL)
    sys.path.insert(0, str(args.tokenizer_repo.resolve()))
    from zipvoice.tokenizer.tokenizer import EmiliaTokenizer

    tokenizer = EmiliaTokenizer(token_file=str(args.model_dir / "tokens.txt"))
    out = {}
    for text in CORPUS:
        ids = tokenizer.texts_to_token_ids([text])[0]
        tokens = tokenizer.texts_to_tokens([text])[0]
        out[text] = {"ids": ids, "tokens": tokens}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.output} ({len(out)} cases)")


if __name__ == "__main__":
    main()
