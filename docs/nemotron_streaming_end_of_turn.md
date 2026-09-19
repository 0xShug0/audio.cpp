# Nemotron streaming: end-of-turn latency — verified analysis (reference-validated)

All findings below were established against the official reference
implementation (transformers `nemotron3_5_asr`, f32, NVIDIA-contributed)
running the same audio through its own encoder, its own decode loop, and
its own official `generate()` streaming path.

## The end-of-turn latency budget (x2 CPU, per phrase)

EOT->final = flush encode (~100-175 ms) +, for pocketed utterances, the
padded-offline fallback (~250-330 ms). The flush encode is the encoder
running one 64-mel window; it is exact vs the reference (cos 0.999,
frame-for-frame, verified on multiple shapes).

## Trailing-token mechanism (verified)

The RNNT fires a word's trailing tokens only after a per-utterance amount
of silence follows the word in the encoded stream. The silence CONTENT is
irrelevant (digital zeros = room tone = dither; verified). What matters is
duration, and it interacts with the cut position:

- hello.wav cut at 0.65 s: decodes in streaming at the 64-mel window.
- hi.wav cut at 0.67 s: decodes in streaming ('Hi. ').
- hi.wav cut at 0.57 s: does NOT decode in streaming AT ALL.

## The hi.wav 0.57 s cut: streaming-mode failure, not a bug

Exhaustive shape matrix against the reference, same audio (hi.wav[:9120]),
la0, prompt aligned:

    (8,32)            5 f -> ''
    (8,32,32)         9 f -> ''
    (8,32,64)        13 f -> ''
    (8,32,96)        17 f -> ''
    (8,32,32,32)     13 f -> ''
    (8,32,32,32,32)  17 f -> ''
    (8,32x7)         25 f -> ''

Every streaming shape fails, including 1.6 s of trailing synthetic silence.
The reference's own official streaming `generate()` also returns ''.
The OFFLINE encoder on the same audio + >=0.5 s of padding decodes 'Hi. '.

Conclusion: the streaming causal-conv frame grid fails this utterance at
this cut regardless of windowing; the offline grid succeeds. audio.cpp's
padded-offline fallback is the only correct mechanism and matches the
model's own semantics. It is not a workaround to remove - it is the fix.

## Dead ends (do not retry)

- Longer single flush windows (64/72/96 mel): the 0.57 s hi cut fails at
  every length; longer windows only add encode time.
- Iterative flush chunks: same frames as the long window, same failure.
- Silence dither instead of zero padding: identical results.
- Per-sample-max energy gates for the speculative trigger: decay
  transients count as speech; use 10 ms windowed RMS.

## Remaining latency levers (ranked)

1. Speculative offline fallback: when the speculative flush's decode comes
   back blank (decode is cheap, ~10 ms), run the padded-offline encode
   during the tail wait too. Requires snapshot/restore of the decoder RNNT
   state alongside the encoder state. Saves the fallback's ~250-330 ms from
   the EOT->final path for pocketed utterances.
2. q4_K GGUF regeneration: the encoder is weight-bandwidth-bound; halves
   the per-chunk cost on CPU.
3. Vulkan: the flush hides entirely; EOT->final 60-90 ms already.

## Verified harness artifacts

- bench_tmp/ref_nemotron_stream.py: reference runner (offline/official
  streaming/manual chunked with dumps).
- bench_tmp/dump/ref96/*: reference frames for the shape matrix above.
- NEMOTRON_DUMP_CHUNKS / NEMOTRON_DUMP_CACHES: audio.cpp-side dumps.
