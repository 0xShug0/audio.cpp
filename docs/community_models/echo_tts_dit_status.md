# Echo-TTS DiT: implementation status

## What exists

| Component | State | Verification |
| --- | --- | --- |
| Byte tokenizer + WhisperD normalisation | complete | executed, output checked by hand |
| PCA forward / inverse | complete | executed, cross-checked against numpy |
| Flattening-point crop | complete | executed |
| Euler dual-CFG sampler | complete | executed, matches a numpy transcription of `inference.py` to 6.6e-07 |
| Timestep embedding | complete | matches `model.py` exactly (0.0 diff) |
| Attention mask construction | complete | layout checked against upstream `cat()` semantics and ggml constraints |
| DiT graph (encoders, joint attention, adaLN, blocks) | written | compiles against real headers; **never executed** |
| Weight loading (1,117 tensors) | written | compiles; **tensor names unconfirmed against a real checkpoint** |
| Conditioning / denoiser graph execution | written | compiles; **never executed** |
| Fish codec `z_q` seam | not started | — |
| Session integration | not started | — |

The distinction in that last column is the important one. Everything above the
line was run and compared against a reference. Everything below it has only been
type-checked. A clean compile here means the framework APIs are used correctly;
it says nothing about whether the numbers are right.

## Design decisions worth reviewing

### Flash attention in the DiT joint attention

`joint_attention` uses `ggml_flash_attn_ext`, which never materialises the
`(lanes, heads, seq, keys)` scores tensor. That tensor was the largest
per-request allocation in the model:

| Case | Keys | Scores tensor removed |
| --- | ---: | ---: |
| Typical (64 text bytes, 10 s reference) | 793 | 97 MB per attention |
| Long text, 30 s reference | 1569 | 193 MB per attention |
| Worst case (768 text, 5 min reference) | 3008 | 370 MB per attention |

Live across 24 blocks with `ggml_gallocr` reuse, the practical saving is a few
hundred MB to over a gigabyte, and flash attention is also faster.

This was initially written with the explicit lowering on the belief that the
speaker-unconditional CFG lane produces fully masked rows, which would make
`-inf` softmax to NaN. That was wrong: `make_denoiser_mask` leaves the self block
of every row unmasked, so a query always attends to at least its own 640
positions and no row can be fully masked.

Two details the flash path requires. The mask must be F16, so the masked value
is `-65000` rather than `-1e9`; the latter converts to `-inf` in F16, which would
reintroduce exactly the NaN hazard the explicit path was chosen to avoid. And
`q->ne[2] % mask->ne[2]` and `q->ne[3] % mask->ne[3]` must both be zero, which
holds because the mask carries a singleton head axis and matches the lane count.

Set `AUDIOCPP_ECHO_TTS_NO_FLASH=1` to fall back to the explicit lowering and F32
mask, for A/B comparison without a rebuild.

The two encoders still use the explicit lowering. Their sequences are short (a
few hundred tokens at most) so the scores tensors are small, and the speaker
encoder is causal with no explicit mask, which the flash path rejects.

### Speaker references are encoded in chunks

`encode_speaker` splits the reference into ~29.7 s chunks (640 latents x 2048
samples), zero-pads the last one, and concatenates the per-chunk latents,
following `inference.py::get_speaker_latent_and_mask`. Upstream's comment calls
that the longest chunk seen in training, so this is a fidelity matter as much as
a memory one -- encoding several minutes in a single pass is a different
computation from what the model saw.

The memory difference is large, because the Fish encoder's first stages run at
the full 44.1 kHz rate. A single 64-channel activation is 0.34 GB for one chunk
against 3.04 GB for a 4m29s reference encoded in one pass, and several such
tensors are live at once. Fixed-size chunks also mean one encode graph is built
and reused across all chunks.

After chunking, the dominant per-request allocation at long reference lengths is
the persistent KV cache: 0.59 GB at 4m29s and 0.65 GB at the 297 s maximum,
stored F32. Halving it to F16 is the obvious next step if that ever matters.

### KV cache as a separate backend buffer

The conditioning encoders and the denoiser are separate graphs so the encoders
run once per request rather than once per sampler step. They share the cached
projections through tensors allocated in their own `ggml_context` and backend
buffer, referenced as leaves by both graphs. `ggml_gallocr` leaves
already-allocated tensors alone, so the conditioning graph writes into them with
`ggml_cpy` and the denoiser graph reads them directly.

Consequence: changing text length or speaker length invalidates the cache and
every graph built against it. `prepare_conditioning` tears all of it down and
rebuilds, which is correct but means a request with new conditioning pays full
graph construction. Acceptable given that a 40-step sample dominates.

### Speaker KV scaling round-trips through the host

`scale_speaker_kv` reads the cached tensors back, scales, and re-uploads, because
the cache has no graph attached. This runs at most twice per request (once to
apply, once to undo at the threshold) and touches at most 24 layers x 2 tensors.
It is not on the per-step path. If it ever shows up in a profile, the fix is a
tiny scaling graph rather than a host round trip.

## Things most likely to be wrong

Listed in rough order of how much damage they would do and how hard they would
be to spot without a parity run:

1. **Tensor names.** Derived from `model.py`'s module structure, corroborated by
   a parameter count matching the published file size to ten digits, but never
   resolved against an actual checkpoint. `convert_echo_tts.py --model-dir ...`
   settles this in seconds and prints exactly what is wrong if anything is.
2. **Half-head RoPE.** Heads 0-7 rotate, 8-15 do not. Implemented as
   slice/rope/concat on the head axis. Wrong here means plausible-sounding but
   incorrect audio, with no shape error.
3. **Rotary pairing convention.** `GGML_ROPE_TYPE_NORMAL` (interleaved), matching
   upstream's complex view of adjacent pairs. The in-tree `rf_dit.cpp` uses NEOX,
   so copying from it would be wrong.
4. **Speaker patchify reshape.** Folding `patch_size` frames into the feature
   axis assumes row-major frame-then-channel ordering. A transposed reading would
   still produce correct shapes.
5. **adaLN chunk order.** `shift, scale, gate` from `cond_embed.chunk(3, -1)`.
   A permutation here is silent.

Items 2-5 are all caught by the per-block parity dumps from
`tools/community_models/echo_tts/echo_tts_reference.py`, which is why that script dumps
per-block activations at a fixed timestep rather than only the final output.

## Next steps

1. Run `convert_echo_tts.py` against the real checkpoint to confirm item 1.
2. Build on a machine with a GPU and run the parity comparison per block.
3. Split `fish_audio/codec.cpp`'s `build_decode_quantizer` at the `post_module`
   boundary and expose the summed `quantized` term from
   `build_encode_quantizer`, giving Echo decode and native speaker encoding.
4. Wire the session: tokenize, encode speaker, sample, PCA inverse, decode, crop.
