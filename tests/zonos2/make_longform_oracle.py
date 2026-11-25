"""Continuation prompts and word-packed chunks from Zyphra/ZONOS2's own code (tts/longform.py, tts/prompt.py), for
zonos2_prompt_test.

    python make_longform_oracle.py <out_dir> [text_file]      (run with the ZONOS2 checkout's interpreter)

Without a text file, the split oracle uses a built-in multilingual paragraph.

The speaker and marker rows are the scheduler's (_with_speaker_frames): slot, background marker, accurate marker.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from zonos2.tts.longform import build_continuation_prompt, split_text
from zonos2.tts.prompt import (TTSPromptBuilder, TTSPromptConfig, accurate_mode_token_id, conditioned_text_vocab_size,
                               speaker_background_token_id)

out = Path(sys.argv[1])
COUNTS = (12, 12, 12, 8, 8, 8)
text_vocab = conditioned_text_vocab_size(8, sum(COUNTS), 2, 1)
assert text_vocab == 519, text_vocab
builder = TTSPromptBuilder(TTSPromptConfig(
    n_codebooks=9, audio_pad_id=1025, text_vocab=text_vocab, speaking_rate_num_buckets=8,
    quality_bucket_counts=COUNTS, speaker_background_num_buckets=2, accurate_mode_num_buckets=1))

rng = np.random.default_rng(7)
anchor = rng.integers(0, 1024, size=(40, 9))
recent = rng.integers(0, 1024, size=(55, 9))
recent[-1, 3] = 1024  # an EOA code inside a prefix must pass through unchanged
np.save(out / "longform_anchor_rows.npy", anchor.astype(np.float32))
np.save(out / "longform_recent_rows.npy", recent.astype(np.float32))
texts = ["The harbor was quiet.", "Maria counted nine crates.", "Nobody noticed the stranger."]
(out / "longform_texts.txt").write_text("\n".join(texts))

def marker(token):
    return torch.tensor([[1025] * 9 + [token]], dtype=torch.int32)

def save(name, context_text, codes, quality, rate=None, speaker=False):
    prompt = build_continuation_prompt(builder, context_text, texts[2], codes.tolist(), text_vocab=text_vocab,
                                       speaking_rate_bucket=rate, quality_buckets=quality,
                                       speaker_slot=builder.speaker_slot() if speaker else None)
    if speaker:  # noisy background, accurate mode
        prompt = torch.cat([prompt[:1],
                            marker(speaker_background_token_id(text_vocab, 8, COUNTS, False, 2, 1)),
                            marker(accurate_mode_token_id(text_vocab, 8, COUNTS, 2, 1)), prompt[1:]], dim=0)
    np.save(out / name, prompt.numpy().astype(np.float32))
    print(name, tuple(prompt.shape))

final_q = [None, None, None, None, None, 3]      # the release default, trailing_silence_s:3
open_q = [None] * 6                               # a chunk that another chunk continues
save("longform_cont_anchor.npy", f"{texts[0]} {texts[1]}", np.concatenate([anchor, recent]), final_q)
save("longform_cont_recent.npy", texts[1], recent, open_q)
save("longform_cont_spk.npy", f"{texts[0]} {texts[1]}", np.concatenate([anchor, recent]), final_q, rate=3, speaker=True)

# Word packing at the default 150-character budget. Characters are code points in Python, as in the C++ port.
if len(sys.argv) > 2:
    paragraph = Path(sys.argv[2]).read_text(encoding="utf-8")
else:
    paragraph = (
        "When the ferry finally left the pier, the fog had already swallowed the far shore, and the only sound was "
        "the slow beat of the engine under the deck.\nThomas stood at the rail with a paper cup of coffee,  counting "
        "the gulls that followed the boat. Un café très serré, s'il vous plaît, et une déclaration "
        "extraordinairement-longue-sans-aucune-espace-pour-voir-ce-que-fait-le-découpage-quand-un-seul-mot-dépasse-"
        "largement-la-limite-de-cent-cinquante-caractères-fixée-par-défaut-dans-le-modèle. 港は静かだった。 "
        "Nobody on the island ever seemed to be in a hurry.\t The end.")
chunks = split_text(paragraph, 150, "word")
(out / "longform_split_input.txt").write_text(paragraph, encoding="utf-8")
(out / "longform_split_chunks.txt").write_text("\n".join(chunks), encoding="utf-8")
print("longform_split_chunks.txt", [len(c) for c in chunks])
