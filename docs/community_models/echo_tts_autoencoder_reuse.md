# Echo-TTS: autoencoder reuse

Status: verified against checkpoint sizes and upstream source. No weights were
downloaded to reach these conclusions; every number below is reproducible from
`autoencoder.py` plus the file sizes Hugging Face reports.

> The "Integration seam" section below was written when the codec still lived
> inside the `fish_audio` model. Upstream has since promoted it to a framework
> runtime (`engine::codecs::FishDacCodecRuntime`, #310), which exposes the two
> seams this document argued for -- `encode_latents` and `decode_latents` --
> directly. Echo now calls that runtime and touches no `fish_audio` code at all.
> The analysis is kept because the reasoning about *which* weights Echo needs
> still holds.

## Summary

Echo-TTS depends on the Fish S1-DAC autoencoder, and **audio.cpp already
implements that exact autoencoder** -- today in
`src/framework/codecs/fish_dac_codec_runtime.cpp`, at the time of writing inside
the `fish_audio` model. The Echo port does not need a new decoder,
encoder, quantiser, or window-limited transformer. It needs a `z_q` seam on the
existing one.

This changes the cost of milestones M1 and M2 substantially relative to the
original PR plan, which scoped "Fish decode" and "native speaker encoding
(Fish encoder + RVQ)" as separate pieces of work.

## Evidence

### Configuration

`jordand/fish-s1-dac-min/config.json` reports:

    sample_rate 44100, encoder_dim 64, encoder_rates [2,4,8,8], latent_dim 1024,
    decoder_dim 1536, decoder_rates [8,8,4,2], n_codebooks 9, codebook_size 1024,
    codebook_dim 8, semantic_codebook_size 4096, causal true

Every one of these matches the constants already compiled into the shared
codec: `kCodecDim` 1024, semantic codebook 4096, nine residual
quantisers of 1024, codebook dim 8, a final decoder snake at 96 channels
(= 1536 / 2^4), and causal convolutions throughout.

### Parameter budget

Deriving the parameter count from `autoencoder.py` and comparing against the
1.87 GB Hugging Face reports for `pytorch_model.safetensors`:

| Component | Parameters |
| --- | ---: |
| Encoder | 76,851,328 |
| Decoder | 54,102,722 |
| Quantiser (incl. pre/post transformers) | 260,475,040 |
| **Total** | **391,429,090** |

At F32 that is 1.566 GB. The `Transformer` base class registers two buffers per
instance — a `freqs_cis` table and a `block_size^2` boolean `causal_mask` — which
for the three surviving transformer instances (encoder block 3 at block_size
16384, quantiser pre/post at 4096) comes to 305 MB. Together: **1.871 GB**,
against the 1.87 GB reported. This also reproduces the 303.6 MB
"regenerable buffers" figure noted on the PR.

The match only holds once the decoder is counted **without** a transformer, which
leads to the next point.

### The decoder has no transformer

`build_ae` passes `decoder_transformer_layers=[4, 0, 0, 0]`, which reads as though
decoder block 0 carries a 4-layer transformer. It does not. `DecoderBlock.__init__`
constructs `transformer_module` into a local variable and then builds
`self.block = nn.Sequential(Snake1d, conv_trans, ResidualUnit x3)` without it.
The module is never assigned to `self`, so it is not a submodule, has no
parameters, and is absent from the checkpoint. `EncoderBlock`, by contrast, does
include `transformer_module` in its `Sequential`.

Two independent checks agree:

1. The 1.87 GB file size only reconciles when the decoder transformer is excluded
   (including it predicts 2.05 GB, and adds a second 16384x16384 mask buffer that
   would break the 303.6 MB figure).
2. The codec implementation already loads the encoder transformer conditionally
   at `block_index == 3` and loads no transformer anywhere in the decoder path.

The C++ was evidently written against the real checkpoint, and it agrees with
the source reading. Worth knowing before anyone "fixes" the apparent omission.

## Integration seam

Echo needs continuous `z_q` where `fish_audio` uses discrete codes. Both seams
sit at existing boundaries in the codec, and the framework runtime now exposes
them as `decode_latents` and `encode_latents`:

**Decode.** `DAC.decode_zq` is `post_module -> upsample -> decoder`.
`build_decode_quantizer` already performs exactly that chain; it just derives its
input by looking up codebook entries first:

    latent = build_quantizer_out(semantic) + sum(build_quantizer_out(residual_i))
    latent = build_window_transformer(..., post_module, 128)   <- Echo enters here
    for stage in upsample: ...

Echo supplies `latent` directly from the PCA inverse and runs from the
`post_module` line onward -- which is what `decode_latents` does.

**Encode.** `DAC.encode_zq` quantises and then sums the dequantised results:
`z_q = z_q_semantic + z_q_residual`. The encode graph already computes each
`quantized` term internally on the way to emitting code indices; `encode_latents`
returns that sum instead of discarding it. That gives native speaker encoding
without new model code, which is most of milestone M2.

## Consequences for packaging

The *implementation* is shared; the *weights* are not.

`fish_audio` ships Fish Audio S2 Pro. Echo is trained against the Fish S1 DAC
(`jordand/fish-s1-dac-min`, a mirror of `fishaudio/openaudio-s1-mini`), and its
PCA basis is fitted to that codec's latent space. The S2 technical report says S2
retains S1's RVQ codec, and the shapes line up (10 codebooks, ~21 Hz), but
"retains the codec" in a report can mean the architecture rather than identical
weights -- and a retrained-but-isomorphic codec would yield wrong audio with no
error raised anywhere. That is not a risk worth taking to save a download.

`convert_echo_tts.py` therefore packages the S1 codec into Echo's GGUF under the
`codec_weights` prefix, folding weight normalisation and dropping the 305 MB of
regenerable buffers. Echo hands that tensor source to the framework runtime with
a `FishDacCodecConfig` that differs from the defaults in only four fields
(`sample_rate`, `frame_length`, `total_codebooks`, `quantizer_codebooks`) -- the
rest already describe S1-DAC.

Verified against the real checkpoint manifest: the folded output supplies all 220
tensor names the codec loads, and the 541 stored tensors resolve to 455 after
folding and buffer removal.

## Caveat

Everything above is derived from source reading plus file-size arithmetic. The
parameter total agreeing with the reported size to three significant figures is
strong evidence, but it is not the same as having loaded the tensors. The
tensor-name check in `convert_echo_tts.py` and a parity run against
`echo_tts_reference.py` remain the gates before any of this is claimed as done.
