# GGUF Models

audio.cpp can load audio.cpp-native GGUF checkpoints for model families that have a
package spec. GGUF is a container for tensors and sidecar files; it is not a universal
adapter for arbitrary llama.cpp or whisper.cpp GGUF files. The tensor names and embedded
metadata still have to match the selected `--family`.

Package specs are maintained as `model_specs/*.json`. New GGUFs contain the selected
spec in `audiocpp.model_spec.*` metadata, so a standalone GGUF does not depend on a
`model_specs` directory or on the binary having that family compiled into its spec
catalog.

Runtime resolution is deterministic:

```text
explicit --model-spec-override
              |
              v
package spec embedded in the selected GGUF
              |
              v
compiled catalog (AUDIOCPP_DEPLOYMENT_BUILD=ON)
              |
              v
external model_specs/<family>.json discovery
```

An explicit override is useful for testing a modified layout without rebuilding or
reconverting:

```bash
audiocpp_cli --inspect --family qwen3_asr --model /path/to/model.gguf \
  --model-spec-override /path/to/qwen3_asr.json
```

The override may also be a directory containing `<family>.json`. The server supports
the same command-line option and a `model_spec_override` field either at the top level
or inside an individual model entry. A per-model field takes precedence over the
server-wide value.

## Support And Test Status

Status labels:

| Label | Meaning |
|---|---|
| `Done` | Package-spec refactor is in place for this family. |
| `No` | Package-spec refactor is not done, or the tested format is not usable. |
| `Skip (...)` | Package-spec refactor is intentionally skipped. |
| `Pass` | Covered by the path-test matrix with acceptable output. |
| `Pass (TTS + clone)` | Both no-reference TTS and reference-audio voice cloning run successfully. |
| `Pass (drift)` | Loads and runs, with known acceptable output drift. |
| `Pass (bit-identical)` | Loads and runs, reproducing the F32/F16 reference timestamps exactly (0-sample boundary diff). |
| `Pass (ASR match, drift)` | TTS output has similarity/frame drift but ASR transcript remains usable. |
| `No (...)` | Known unsupported, failing, or too much output drift. |
| `---` | Not tested in the current GGUF path-test matrix. |

| Family | Package-spec refactor | Safetensors tested after refactor | `orig` GGUF tested | 16-bit GGUF tested | `q8_0` GGUF tested |
|---|---|---|---|---|---|
| `ace_step` | Done | Pass | --- | Pass (drift) | No (planner sampling can fail) |
| `bs_roformer` | Done | Pass | --- | --- | Pass |
| `chatterbox` | Done | Pass | --- | Pass (ASR match, drift) | Pass (ASR match, drift) |
| `citrinet_asr` | Done | Pass | --- | --- | Pass |
| `fish_audio` | Done | Pass | --- | Pass | Pass |
| `fun_asr_nano` | Done | Pass | --- | Pass | Pass |
| `glm_tts` | Done | Pass (TTS + clone) | --- | --- | Pass (ASR match, drift) |
| `heartmula` | Done | Pass | --- | Pass (drift) | Pass (drift) |
| `higgs_audio_stt` | Done | Pass | --- | Pass | Pass |
| `higgs_audio_tts` | Done | Pass | --- | Pass | Pass |
| `htdemucs` | Done | Pass | --- | Pass | Pass (drift) |
| `hviske_asr` | Done | Pass | --- | --- | Pass |
| `inflect_v2` | Done | Pass | Pass | --- | --- |
| `index_tts2` | Done (v2 + v2.5 variant) | Pass | Pass | Pass (drift) | Pass (ASR match, drift) |
| `irodori_tts` | Done | Pass | --- | Pass | Pass (ASR match, drift) |
| `kroko_asr` | Done | Pass | --- | --- | Pass |
| `magpie_tts` | Done | --- | Pass | --- | Pass |
| `marblenet_vad` | Bundled (tiny model) | Pass | --- | --- | --- |
| `meanvc2` | Done | --- | --- | Pass | --- |
| `mel_band_roformer` | Done | Pass | --- | Pass (drift) | Pass (drift) |
| `miocodec` | Done | Pass | Pass | Pass (drift) | Pass (drift) |
| `miotts` | Done | Pass | Pass | Pass (drift) | Pass (ASR match, drift) |
| `mms_forced_aligner` | Done | Pass | --- | Pass | Pass (bit-identical) |
| `moss_tts_local` | Done | Pass | --- | Pass | Pass (ASR match, drift) |
| `moss_tts_nano` | Done | Pass | --- | Pass | Pass (ASR match, drift) |
| `muscriptor` | Done | Pass | Pass | --- | --- |
| `nemotron_asr` | Done | Pass | --- | Pass | Pass (minor filler drift) |
| `neutts` | Done | Pass | --- | Pass | --- |
| `omnivoice` | Done | Pass | --- | Pass (drift) | Pass (drift) |
| `outetts` | Done | Pass (TTS + clone) | --- | --- | Pass (TTS + clone) |
| `parakeet_tdt` | Done | Pass | Pass | Pass | Pass |
| `personaplex` | Done | --- | --- | --- | Pass |
| `pocket_tts` | Done | Pass | --- | Pass | Pass (drift) |
| `qwen3_asr` | Done | Pass | --- | Pass | Pass |
| `qwen3_forced_aligner` | Done | Pass | --- | Pass | Pass |
| `qwen3_tts` base | Done | Pass | Pass | Pass (ASR match, drift) | Pass (ASR match, drift) |
| `qwen3_tts` custom voice | Done | Pass | --- | Pass (ASR match, drift) | Pass (ASR match, drift) |
| `qwen3_tts` voice design | Done | Pass | --- | Pass (ASR match, drift) | Pass (ASR match, drift) |
| `rvc` | Done | --- | --- | Pass | --- |
| `seed_vc` | Done | Pass | --- | Pass (drift) | Pass (drift) |
| `soprano_tts` | Done | Pass | --- | Pass | Pass (drift) |
| `silero_vad` | Skip (tiny model) | --- | --- | --- | --- |
| `sortformer_diar` | Done | Pass | --- | Pass | Pass |
| `stable_audio` | Done | Pass | --- | Pass (drift) | Pass (drift) |
| `supertonic` | Done | Pass | Pass | Pass | No (Q8 blockers unresolved) |
| `vevo2` | Done | Pass | Pass | Pass (drift) | No (mixed route drift; speech ASR match) |
| `vibevoice` | Done | Pass | --- | Pass | Pass (drift) |
| `vibevoice_asr` | Done | Pass | --- | Pass | Pass |
| `voxcpm2` | Done | Pass | Pass | Pass (ASR match, drift) | Pass (ASR match, drift) |
| `voxtral_realtime` | Done | Pass | --- | Pass | Pass |

Additional lower-bit checks:

| Family | Format | Tested |
|---|---|---|
| `meanvc2` | `q4_k` | Pass |
| `personaplex` | `q4_k` | Pass |
| `voxtral_realtime` | `q4_k` | Pass (quick CUDA check; transcripts match Q8 except one capitalization-only difference) |

**What the matrix does not cover.** The three tables above test `orig`, 16-bit and
`q8_0`, plus the three `q4_k` spot checks listed here. That is the whole of the recorded
validation. In particular:

- `minimax_music3`, `minimax_h3` and `audiosr` appear in **no row of any table**. Nothing
  about those families has been recorded through this process.
- There is no 4-bit column in the main table, so no family's 4-bit package is validated
  against its own 16-bit or `orig` output.
- Every "Pass" label is a load-and-run judgement plus, for ASR, a transcript comparison.
  There is no perceptual metric anywhere in the repo — no MOS, PESQ, SI-SDR or
  mel-cepstral distortion — so "Pass (drift)" is not quantified and cannot be compared
  between packages. Timbre, prosody and speaker-identity drift, which is what
  quantization actually costs a TTS or music model, is not measured at all.

`minimax_music3` and `personaplex` shipped a 4-bit package as their default while a
`q8_0` package sat published and unused. Both now default to `q8_0`; the 4-bit packages
remain installable by id. `minimax_h3` still defaults to `q4_k` because no
higher-precision package is published for it.

| Family | Old default | New default | Package size |
|---|---|---|---|
| `minimax_music3` | `minimax_music3_q4_0` | `minimax_music3_q8_0` | 8045 MiB -> 12902 MiB (`minimax_music3_bf16` would be 22547 MiB) |
| `personaplex` | `personaplex_7b_v1_q4_k` | `personaplex_7b_v1_q8_0` | roughly 4 GB -> 7 GB for a 7B model |
| `minimax_h3` | `minimax_h3_q4_k` | unchanged | only 4-bit packages are published |

The MiniMax Music 3 figures are the sum of the five component GGUFs, computed from the
installed `q4_0` package's tensor shapes; the `q4_0` total reproduces the installed files
to within 1 MiB. Re-cutting the `q8_0` package with the tensor policy below would take it
to 13634 MiB.

`q4_0`, which `minimax_music3` used, is a legacy format: one scale per 32-element block
and no zero point. `q4_k` carries per-super-block scales and mins at essentially the same
bitrate and is what every other 4-bit user in the repo is already on. If a 4-bit MiniMax
Music 3 package is re-cut, cut it as `q4_k`.

Q8 packaging notes:

- `chatterbox` Q8 is intentionally mixed type. Graph-sensitive scalar, norm,
  bias, and side tensors stay in non-Q8 types while matmul-compatible weights
  are quantized.
- `pocket_tts` Q8 keeps the four `flow_lm.flow_net.time_embed.*.mlp.{0,2}.weight`
  tensors in Q8 in addition to the default converter selection. `conditioner.embed`,
  `cond_embed`, and Mimi conv tensors are not forced to Q8 because tested outputs
  drifted or the current conv path casts quantized conv weights back to F32.
- `dots_tts` Q8 keeps the vocoder in 16-bit storage and folds vocoder
  weight-norm conv tensors at conversion time. Use
  `--keep-type 'vocoder/*=f16' --fold-weight-norm 'vocoder/*'` for both SOAR
  and MeanFlow Q8 conversion so the flow/LLM path is quantized while the
  vocoder stays in the tested dtype with direct conv weights.
- `qwen3_tts` Q8 should keep speaker-sensitive components in their original
  16-bit type. The tested Base Q8 package quantizes the talker transformer and
  projections, talker code-predictor heads, and speech-tokenizer encoder/decoder
  projection or linear weights, while leaving the speaker encoder, lookup, and
  codebook-sensitive tensors unquantized. Quantizing those speaker-side tensors
  can produce long-form quality problems such as large silence.
- `supertonic` F16 is intentionally mixed type. Keep duration predictor,
  vocoder, non-weight tensors, embeddings, codebook tensors, and norm tensors in
  F32; convert only compatible projection weights to F16. Supertonic Q8 is not
  currently supported: broader Q8 packages still hit CUDA Q8 copy/layout
  blockers in text/vector graph paths, so the Q8 blocker is not fully solved.
- `voxtral_realtime` also has a tested `q4_k` package. In a quick CUDA path
  check it was smaller and faster than Q8_0, while transcript output matched
  Q8_0 except for one capitalization-only difference.

## Which Tensors Get Quantized

Two rules decide the storage type of every tensor the converter writes. Both live in
`convert_tensor_sources_to_gguf` (`src/framework/assets/tensor_source.cpp`).

**1. The shape rule.** A tensor is a candidate for the requested quantized type only when
it is a float source, its name ends in `.weight`, it is exactly rank 2, and its last
dimension divides the target's block size (32 for `q8_0`, 256 for the `_k` formats).
Everything else keeps its source dtype. This is why biases and rank-1 norms come out F32,
and why conv kernels do: a Conv1d kernel is rank 3 and its row length is the kernel size
(3, 7, 11, 16), never a multiple of 32.

**2. The audible-tensor rule.** The shape rule is blind to what a tensor *does*. It
protects conv-stack vocoders by accident of rank and protects nothing at all once a
vocoder, a codec decoder or a norm is written as a matmul. The converter therefore also
holds back tensors by name, and stores them at F16 instead — the same fallback embedding
and codebook tables already use.

A tensor belongs on that list when quantization error in it reaches the waveform with no
remaining layer to absorb it. That is the whole test. Mid-stack attention and FFN weights
fail it and stay quantized.

| Group | Patterns | Why |
|---|---|---|
| iSTFT and spectrogram heads | `*istft*`, `*stft_head*`, `*spec_head*`, `*spectrogram_head*`, `*mag_head*`, `*phase_head*` | The layer's output *is* the magnitude/phase spectrum, exponentiated on the way into the iSTFT. `redae/decoder.istft_head.out.weight` shipped as `Q8_0 [896,1922]` in FireRedTTS3-Instruct and FireRedAudio. |
| Waveform vocoders | `*vocoder*`, `*bigvgan*`, `*hifigan*`, `*hifi_gan*`, `hift.*`, `hift/*`, `*.hift.*`, `*/hift.*`, `*waveform_decoder*`, `*wave_decoder*` | Last stage before the WAV. Conv stacks already survived on rank; this makes it deliberate and covers the rank-2 layers inside the same stacks. The HiFT patterns are anchored because `shift` contains `hift`. |
| Codec and latent-to-audio decoders | `*acoustic_decoder*`, `*audio_decoder*`, `*codec_decoder*`, `*codec.decoder.*`, `*codec/decoder.*`, `*_tokenizer.decoder.*`, `*_tokenizer/decoder.*`, `*mel_decoder*`, `*s2mel*` | VibeVoice-7B shipped 52 Q8_0 FFN matmuls inside `model.acoustic_tokenizer.decoder`; IndexTTS2.5 shipped its whole s2mel flow decoder at Q8_0 while OmniVoice's conv vocoder stayed F32 for free. Encoder-side names are deliberately absent: analysis error is absorbed by the decoder that follows it. |
| Normalisation projections | `*norm*`, `*adaln*` | Norms survive today only because they are rank 1. Written as a learned rank-2 projection — `*_norm.project_layer.weight`, an adaLN modulation — a norm is quantized like any other matmul, and its error is multiplied across every channel it scales. |
| Output heads and final projections | `*head.weight`, `*heads.weight`, `*proj_out.weight`, `*final_proj.weight`, `*final_layer.linear.weight` | The head emits the distribution the sampler draws audio tokens from, with no later layer to absorb the error. In every package inspected the head has exactly the shape of the embedding table, which the converter already stores at F16 — so keeping the head at F16 is what makes the two ends of the model agree. |

Matching is case-insensitive over the whole logical tensor name, `*` stands for any run of
characters, and the first matching rule wins. The list is
`gguf_audible_tensor_rules()` in `src/framework/assets/tensor_source.cpp`; each entry
carries the reason it exists. Add to it there, and add a case to
`tests/unittests/test_quantization_policy.cpp` in the same change.

Note what is deliberately **not** on the list. `decoder` on its own is not a pattern: a
name like `redae/decoder.qwen2.layers.*` is a 24-layer transformer that still has the
whole rest of the decoder downstream, and excluding it would move gigabytes to F16 for
nothing. Only the synthesis-side decoders named above are protected.

### Overriding the policy

`--keep-type <tensor-prefix>*=<type>` still wins over everything, including this list, so
a packager who knows better can quantize a protected tensor deliberately:

```bash
# keep the policy, but quantize this one head anyway
audiocpp_gguf ... --type q8_0 --keep-type 'lm_head.weight=q8_0'
```

The whole policy can also be turned off from a programmatic caller by setting
`GgufConversionOptions::quantize_audible_tensors`, which reproduces the pre-policy output
byte for byte. Prefer the per-tensor override: it records which tensor was quantized and
at what type, instead of turning the protection off wholesale.

### What this costs

The exclusion list makes packages bigger. Measured against the shipped packages installed
on the development machine, by re-deciding every tensor with the new policy at the same
requested type:

| Package | Size | Growth | Tensors moved to F16 |
|---|---|---|---|
| `dramabox-q8_0.gguf` | 18027 MiB | +34 MiB (+0.2%) | 2 |
| `parakeet-tdt-0.6b-v3-q8_0.gguf` | 872 MiB | +5 MiB (+0.5%) | 1 |
| `omnivoice-q8_0.gguf` | 1277 MiB | +8 MiB (+0.6%) | 1 |
| `fireredtts3-instruct-q8_0.gguf` | 3943 MiB | +98 MiB (+2.5%) | 15 |
| `index-tts2_5-q8_0.gguf` | 3324 MiB | +138 MiB (+4.2%) | 106 |
| `firered-audio-q8_0.gguf` | 13308 MiB | +1006 MiB (+7.6%) | 15 |
| `vibevoice-7b-q8_0.gguf` | 10080 MiB | +918 MiB (+9.1%) | 59 |
| `MiniMax-Music3 language_model_q4_0.gguf` | 5729 MiB | +1123 MiB (+19.6%) | 1 |

Almost all of the growth in the last three rows is a single `lm_head.weight`, which is a
`[248320, 4096]`, `[152064, 3584]` and `[200000, 4096]` matrix respectively. If a package
needs that back, `--keep-type '<name>=q8_0'` restores the old size exactly.

`orig`, `f16` and `bf16` conversions are unaffected: no quantization happens on those
paths, so the policy is inert.

### This does not change GGUF files that already exist

The policy runs at conversion time. It has no effect on any GGUF already on disk or
already published — an installed `index-tts2_5-q8_0.gguf` keeps the Q8_0 s2mel decoder it
was built with. For a package to benefit, it has to be converted again from the original
safetensors weights with a converter built from this change, and re-uploaded. Until then
the only thing that changes for a user is which *package* a spec's default points at.

## Build The Converter

```bash
cmake --build build/debug --parallel --target audiocpp_gguf
```

Normal builds leave `AUDIOCPP_DEPLOYMENT_BUILD` off. Enable it when one binary must also
carry fallback specs for safetensors packages or legacy GGUFs that predate embedded spec
metadata:

```bash
cmake -S . -B build/deploy -DAUDIOCPP_DEPLOYMENT_BUILD=ON
cmake --build build/deploy --parallel
```

`audiocpp_gguf` always carries the conversion catalog. This is separate from the optional
CLI/server deployment catalog, and keeps a copied converter executable usable when the
source checkout and its `model_specs` directory are not present.

Check the converter interface:

```bash
audiocpp_gguf --help
```

Current shape:

```bash
audiocpp_gguf --input [namespace=]<weights> [--input namespace=<weights> ...] \
  --output <weights.gguf> \
  --type <orig|f16|bf16|q8_0|q2_k|q3_k|q4_k|q5_k|q6_k> \
  [--family <family>] \
  [--model-spec <json-or-directory>] \
  [--root <model-dir>] \
  [--sidecar <source>=<destination>] \
  [--bnb-nf4-type q8_0] \
  [--exclude-prefix <logical-prefix>] \
  [--keep-type <tensor-prefix>*=<type>] \
  [--overwrite] \
  [--no-sidecars] \
  [--allow-missing-model-spec]

audiocpp_gguf --inspect <model.gguf>
```

## Convert A Single Tensor Source

Standalone conversion is the default. The converter embeds non-weight files recursively
from the first tensor source's directory, or from `--root` when it is supplied. Use
`--root` when the model has tokenizer, config, processor, or other non-weight files in
a different model root. It also finds, validates, and embeds the package spec. Conversion
fails before writing when the tensor namespaces or required sidecars do not match that
spec.

```bash
audiocpp_gguf \
  --input /path/to/model/model.safetensors \
  --root /path/to/model \
  --output /path/to/model-gguf/model.gguf \
  --type f16 \
  --overwrite
```

Safetensors shard indexes are accepted directly:

```bash
audiocpp_gguf \
  --input /path/to/model/model.safetensors.index.json \
  --root /path/to/model \
  --output /path/to/model-gguf/model.gguf \
  --type q8_0 \
  --overwrite
```

## Convert A Multi-Component Model

Use repeated namespaced `--input` entries when a model has multiple tensor components.
The namespace must match the model's package spec.

```bash
audiocpp_gguf \
  --input model_weights=/path/to/model/model.safetensors \
  --input codec_weights=/path/to/model/codec/model.safetensors \
  --root /path/to/model \
  --output /path/to/model-gguf/model.gguf \
  --type f16 \
  --overwrite
```

## Convert BitsAndBytes NF4 Sources

Some upstream packages store tensors as BitsAndBytes NF4 data in U8 safetensors plus
helper tensors. Use `--bnb-nf4-type q8_0` for these sources. The converter decodes the
NF4 payload, uses the quant-state shape for GGUF metadata, re-quantizes the decoded
weights to GGML Q8_0, and skips the BNB helper tensors from the output.

`--keep-type` only overrides the GGUF output type for normal tensors. It does not decode
raw BNB NF4 U8 tensors by itself.

Use `--exclude-prefix <logical-prefix>` to omit a tensor subtree that the audio.cpp model
does not load, for example an unused vision tower in a shared language-model checkpoint.

```bash
audiocpp_gguf \
  --input audio=models/Dramabox/dramabox-audio-components.safetensors \
  --input dit=models/Dramabox/dramabox-dit-v1.safetensors \
  --input gemma=models/gemma-3-12b-it-bnb-4bit/model.safetensors.index.json \
  --input silence=models/Dramabox/assets/silence_latent_frame.safetensors \
  --root build/debug/dramabox_gguf_sidecars_spec \
  --output models/Dramabox-GGUF/dramabox-q8_0.gguf \
  --type q8_0 \
  --bnb-nf4-type q8_0 \
  --exclude-prefix gemma/vision_tower \
  --family dramabox \
  --model-spec model_specs/dramabox.json \
  --overwrite
```

## Add External Sidecars

Use `--sidecar <source>=<destination>` when a runtime file is needed but does not live
under `--root`, or when it should be embedded at a different path inside the GGUF.

```bash
audiocpp_gguf \
  --input /path/to/model/model.safetensors.index.json \
  --root /path/to/model \
  --sidecar /path/to/shared/preprocessor_config.json=preprocessor_config.json \
  --output /path/to/model-gguf/model.gguf \
  --type q8_0 \
  --overwrite
```

If the default pipeline cannot find any sidecars, conversion fails instead of silently
creating a tensor-only file. Supply the correct `--root` and any required external
`--sidecar` mappings. Pass `--no-sidecars` only when you intentionally want a tensor-only
container; place that GGUF and all package-spec-required sidecars together in one model
directory when loading it. `--no-sidecars` does not remove the embedded package spec and
does not disable build-time validation.

## Package Spec Discovery During Conversion

The converter selects the first valid source at the highest available priority:

1. `--model-spec <json-or-directory>` (also accepted as `--model-spec-override`).
2. A spec object, JSON string, or relative path in the model's `config.json`.
3. `model_spec.json` or `model_specs/*.json` below the model root.
4. A discovered `model_specs/*.json` directory from the working directory upward.
5. The converter's bundled source catalog.

Higher-priority inputs are authoritative. If an explicit override, model-config spec, or
local spec is present but does not match the tensor namespaces and required files, the
converter reports that error instead of silently falling back to a lower-priority layout.

Use `--family <family>` to disambiguate models whose configuration does not identify the
audio.cpp family. A model configuration can declare it directly:

```json
{
  "audiocpp_family": "qwen3_asr",
  "audiocpp_model_spec": "model_spec.json"
}
```

`audiocpp_model_spec` may instead be a JSON string or a path relative to `config.json`.
The nested forms `audiocpp.family`, `audiocpp.model_spec`, and
`audiocpp.package_spec` are also accepted. The converter additionally recognizes known
upstream `model_type` values.

`--allow-missing-model-spec` is an explicit escape hatch for creating a tensor archive
that audio.cpp is not expected to load as a model. It is not recommended for deployable
GGUFs.

## Inspect And Run

Inspect the finished package before using it:

```bash
audiocpp_gguf --inspect /path/to/model-gguf/model.gguf
```

If the GGUF embeds all required sidecars, it can be passed directly as `--model`:

```bash
audiocpp_cli --task asr --family qwen3_asr --model /path/to/model-gguf/model.gguf --backend cuda --audio speech.wav
```

A directory is also accepted by supported package specs. It resolves to `model.gguf` when that
name is present, otherwise to the single `*.gguf` inside it — so a downloaded package directory
works under its release name without renaming anything:

```bash
audiocpp_cli --task tts --family qwen3_tts --model /path/to/model-gguf --backend cuda --text "Hello." --out out.wav
audiocpp_cli --task vc --family vevo2 --model models/Vevo2-GGUF --backend cuda --audio source.wav --voice-ref target.wav --out converted.wav
```

A directory holding several GGUFs and no `model.gguf` is ambiguous and is rejected with the
candidates listed; pass one of them directly as `--model`, or keep a single GGUF per directory.

Compatibility summary:

| Format | Where its package spec comes from | Other model files |
|---|---|---|
| Safetensors | Override, compiled deployment catalog, or external discovery | Required |
| New standalone GGUF | Embedded in GGUF | None |
| New tensor-only GGUF (`--no-sidecars`) | Embedded in GGUF | Required sidecars |
| Legacy GGUF without embedded spec | Compiled deployment catalog or external discovery | Depends on embedded sidecars |

Compatibility with older binaries:

`After PR #53` refers to tag `release-0.3-gguf-v2`, commit
`bf1ac678758aee4caafa7bb25fc0e6db9c25228f`.

| Build | GGUF package | Runtime context | Result |
|---|---|---|---|
| Before PR #53 (`14e9258`) | Legacy GGUF without embedded spec | Repo checkout or external `model_specs` visible | Pass |
| Before PR #53 (`14e9258`) | New standalone GGUF | Repo checkout or external `model_specs` visible | Pass |
| After PR #53 (`release-0.3-gguf-v2`), normal build | Legacy GGUF without embedded spec | Repo checkout or external `model_specs` visible | Pass |
| After PR #53 (`release-0.3-gguf-v2`), normal build | New standalone GGUF | Repo checkout or external `model_specs` visible | Pass |
| Before PR #53 (`14e9258`) | Legacy GGUF without embedded spec | No `model_specs` visible | Fail |
| Before PR #53 (`14e9258`) | New standalone GGUF | No `model_specs` visible | Fail |
| After PR #53 (`release-0.3-gguf-v2`), normal build | Legacy GGUF without embedded spec | No `model_specs` visible | Fail; use `--model-spec-override`, a deployment build, or external specs |
| After PR #53 (`release-0.3-gguf-v2`), normal build | New standalone GGUF | No `model_specs` visible | Pass |
| After PR #53 (`release-0.3-gguf-v2`), deployment build | Legacy GGUF without embedded spec | No `model_specs` visible | Pass through compiled package specs |

## Type Notes

| Type | Meaning |
|---|---|
| `orig` | Preserve the original safetensors storage type where possible. |
| `f16` | Convert eligible tensors to FP16. |
| `bf16` | Convert eligible tensors to BF16. Useful for BF16 source models. |
| `q8_0` | Quantize eligible tensors to Q8_0; unsupported tensors remain in a backend-safe type. |
| `q2_k`/`q3_k`/`q4_k`/`q5_k`/`q6_k` | Lower-bit quantized formats. Treat as experimental per model and backend. |

Quantized GGUF support is model- and route-specific. A model may load successfully but
still drift in length, waveform similarity, or recognized text, so validate the exact
route you plan to ship.

For measured 16-bit vs Q8 speed and peak VRAM results, see
[GGUF Q8 performance](reports/gguf_q8_performance.md).
