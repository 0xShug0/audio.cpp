# VieNeu-TTS v3 Turbo

[VieNeu-TTS v3 Turbo](https://github.com/pnnbao97/VieNeu-TTS) is an on-device bilingual (Vietnamese / English) TTS model with instant voice cloning by Phạm Nguyễn Ngọc Bảo ([pnnbao97](https://github.com/pnnbao97)): a 12-layer Qwen3-style backbone, a small acoustic decoder that emits the 16 residual-VQ codes of each 80 ms frame, and the MOSS-Audio-Tokenizer-Nano codec (48 kHz). audio.cpp runs it torch-free on CPU and CUDA. This family is maintained in audio.cpp by the model author.

| Field | Value |
|---|---|
| Family | `vieneu_v3_turbo` (alias: `vietneu_tts`, the name of the first community port) |
| Model directory | `models/VieNeu-TTS-v3-Turbo-GGUF` |
| Task | `tts`, `clon` |
| Modes | `offline` |
| Languages | `vi`, `en` |
| Text input | **SEA-G2P phonemes** (see below), not raw text |
| Voice input | Reference WAV, or pre-encoded reference codes + speaker embedding |
| Output | stereo 48 kHz WAV |

## Model package

GGUF packages published by the model author in the model's own repo, [pnnbao-ump/VieNeu-TTS-v3-Turbo](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo) under `gguf/`, built from its `update/` weights with `audiocpp_gguf` (the model spec, `config.json` and tokenizer sidecars are embedded, so one file is enough):

| File | Precision | Size |
|---|---|---|
| `vieneu-v3-turbo-q8_0.gguf` | Q8_0 matmuls, bf16 norms / embeddings, f16 codec | 188 MB |
| `vieneu-v3-turbo-bf16.gguf` | bf16 | 292 MB |
| `voices/<id>/{ref_codes.txt,speaker.emb.txt}` | — | the 25 preset voices of the Python SDK as packaged voices (`voices/manifest.json`) |

```bash
python tools/model_manager_v2.py install vieneu_v3_turbo          # q8_0 + the default voice (minh_quan_pro)
python tools/model_manager_v2.py install vieneu_v3_turbo_bf16
```

The original HF layout also loads directly (`--model <dir>` with `model.safetensors`, `config.json`, `tokenizer.json` and `speech_tokenizer/{config.json,model.safetensors}` = `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`).

The codec stays at f16 in **both** packages, and it is the only tensor group
that does. Quantising it to Q8_0 costs 4 dB of SNR against the reference decoder
on identical codes (32.2 vs 36.0 dB) — audible as a faint haze — for 16 MB and no
speed, because the codec runs once per chunk rather than once per frame. bf16 is
the wrong shape for it in the other direction: the same 2 bytes, but 7 bits of
mantissa against f16's 10, and the encoder is sensitive enough to show it. On a
4.6 s clip, reference codes from the f16 codec agree with the fp32 reference
encoder on 96.3% of the 912 codes and the bf16 codec on 82.4%, so a bf16 package
keeps a bf16 backbone and an f16 codec. f32 measures the same as f16.

To repack from a safetensors directory:

```bash
audiocpp_gguf --input model_weights=VieNeu-TTS-v3-Turbo/model.safetensors \
  --input speech_tokenizer_weights=VieNeu-TTS-v3-Turbo/speech_tokenizer/model.safetensors \
  --root VieNeu-TTS-v3-Turbo --family vieneu_v3_turbo --type q8_0 \
  --keep-type "speech_tokenizer_weights*=f16" \
  --output vieneu-v3-turbo-q8_0.gguf
```

The bf16 package is the same command with `--type bf16`, and it keeps the same
`--keep-type`.

## Text input

The model reads SEA-G2P phonemes, and the front end that produces them —
[sea-g2p](https://github.com/pnnbao97/sea-g2p): Vietnamese normalisation of
numbers, dates, units and abbreviations, English code-switching, emotion cues —
is a Rust library with a C ABI. Point the session at it and `--text` takes
ordinary text:

```bash
# once: build the library (Rust toolchain, nothing added to the audio.cpp build)
git clone https://github.com/pnnbao97/sea-g2p && cd sea-g2p
cargo build --release --no-default-features --features capi
# target/release/{libsea_g2p_rs.so | sea_g2p_rs.dll | libsea_g2p_rs.dylib}

audiocpp_cli --task tts --family vieneu_v3_turbo --model .../vieneu-v3-turbo-q8_0.gguf   --backend cpu --text "Tỉ lệ giải ngân đầu tư công đạt 68,5% kế hoạch năm."   --session-option vieneu_v3_turbo.g2p_dict=/path/to/sea_g2p.bin   --request-option reference_codes_file=.../voices/minh_quan_pro/ref_codes.txt   --request-option speaker_embedding_file=.../voices/minh_quan_pro/speaker.emb.txt   --out out.wav
```

The library is found by name (next to the binary, or on the search path) unless
`vieneu_v3_turbo.g2p_library` names it. `sea_g2p.bin` is the dictionary, ~50 MB,
published with the Python package.

Without those options nothing changes: `--text` is then the phoneme string, which
is what every earlier version expected.

```python
from vieneu_utils.phonemize_text import phonemize_text_with_emotions   # pip install vieneu
print(phonemize_text_with_emotions("Xin chào thế giới. Đây là bản thử nghiệm."))
# sˈin tʃˈaː2w tˈeɜ zˈəːɜj. ɗˈəɪ lˌaː2 bˈaː4n tˈy4 ŋˈiɛ6m.
```

Both routes run the same code, so they agree character for character — checked on
dates, times, money, percentages and English words.

## Voice

Every request needs a voice. Two ways:

1. **Reference WAV** (`--voice-ref`): the codec encodes it into reference codes (mono, resampled to 48 kHz, first 8 s). A 192-d speaker embedding is also required: the CAM++ speaker encoder is not ported yet, so pass one with `speaker_embedding_file=` (192 comma-separated floats, produced by the Python engine's `extract_speaker_emb`). Without it the embedding is all zeros and the voice will not match.
2. **Packaged voice** (no audio): `reference_codes_file=` (one frame per line, 16 integers — `numpy.savetxt(codes, fmt="%d")` of the Python `ref_codes`) plus `speaker_embedding_file=`. This is what the Python preset voices ship, and it skips the encoder pass.
3. **Speaker embedding only**: `speaker_embedding_file=` with no reference audio and no codes (`x_vector_only_mode` is then implied). The voice is weaker than with reference codes, because the model only gets the speaker anchor, not the in-context reference frames.

```bash
audiocpp_cli --task tts --family vieneu_v3_turbo \
  --model models/VieNeu-TTS-v3-Turbo-GGUF/vieneu-v3-turbo-q8_0.gguf --backend cpu \
  --text "sˈin tʃˈaː2w tˈeɜ zˈəːɜj. ɗˈəɪ lˌaː2 bˈaː4n tˈy4 ŋˈiɛ6m." \
  --request-option reference_codes_file=models/VieNeu-TTS-v3-Turbo-GGUF/voices/minh_quan_pro/ref_codes.txt \
  --request-option speaker_embedding_file=models/VieNeu-TTS-v3-Turbo-GGUF/voices/minh_quan_pro/speaker.emb.txt \
  --out out.wav
```

Reference audio instead of packaged codes:

```bash
audiocpp_cli --task clone --family vieneu_v3_turbo \
  --model models/VieNeu-TTS-v3-Turbo-GGUF/vieneu-v3-turbo-q8_0.gguf --backend cpu \
  --text "<phonemes>" --voice-ref voice/ref.wav \
  --request-option speaker_embedding_file=voice/speaker.emb.txt --out out.wav
```

## Enrolling a voice

Adding a voice to an application is not a synthesis: what it needs is the
reference codes, kept on disk, so that every later request is a packaged voice
(option 2 above) and the clip and the encoder are never touched again.

`encode_reference_only=true` runs the encoder pass and stops. The codes come
back as an `acoustic_tokens` artifact whose payload is the text
`reference_codes_file` reads — one frame per line, `code_groups` integers each —
so the file the CLI writes goes straight back in:

```bash
# once per voice: the clip becomes codes
audiocpp_cli --task tts --family vieneu_v3_turbo \
  --model models/VieNeu-TTS-v3-Turbo-GGUF/vieneu-v3-turbo-q8_0.gguf --backend cpu \
  --voice-ref voice/ref.wav --request-option encode_reference_only=true \
  --out-dir voice/
# voice/vieneu_v3_turbo_reference_codes.txt

# from then on: no clip, and no encoder pass
audiocpp_cli --task tts --family vieneu_v3_turbo \
  --model models/VieNeu-TTS-v3-Turbo-GGUF/vieneu-v3-turbo-q8_0.gguf --backend cpu \
  --text "<phonemes>" \
  --request-option reference_codes_file=voice/vieneu_v3_turbo_reference_codes.txt \
  --request-option speaker_embedding_file=voice/speaker.emb.txt --out out.wav
```

`frames`, `code_groups` and `sample_rate` are in the artifact's metadata, for a
caller reading it through the API rather than off disk.

A normal request that was given `--voice-ref` returns the same artifact
alongside its audio, so a caller can keep the voice it just paid to derive.
Codes passed in through `reference_codes` are not echoed back.

The speaker embedding still has to come from elsewhere: the CAM++ encoder is
not ported (see [Not yet ported](#not-yet-ported)).

## Options

Defaults follow `Vieneu.infer()` in the Python package. Use `--request-option name=value`; `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty` and `--max-tokens` map to the same names.

| Option | Default | Meaning |
|---|---:|---|
| `temperature` / `top_k` / `top_p` | `0.8` / `25` / `0.95` | One sampler for all 16 codebooks of the acoustic decoder. |
| `repetition_penalty` | `1.2` | Penalty on codes seen in the recent window of each codebook. |
| `repetition_window` | `64` | Frames each codebook remembers (~5 s); `0` = unbounded. |
| `do_sample` | `true` | `false` = greedy (argmax) decoding. |
| `max_tokens` | `300` | Frame budget per chunk (80 ms each). |
| `frame_cap` | `true` | Also cap the budget by the phoneme count of the chunk (Python `max_expected_frames`), which stops runaway generation when EOS is missed. |
| `seed` | random | Sampling seed. |
| `reference_codes_file` / `speaker_embedding_file` / `speaker_embedding` | — | Voice inputs, see above. |
| `x_vector_only_mode` | `false` | Ignore reference codes and clone from the speaker embedding alone. |
| `encode_reference_only` | `false` | Encode `--voice-ref` into reference codes, return them as an artifact and generate nothing. |
| `text_chunk_size` | `200` | Character budget per chunk of the phoneme string. The cut follows paragraphs, then sentences, then minor punctuation, then whitespace. |
| `text_chunk_min` | `20` | Chunks shorter than this join a neighbour — alone they read as a stutter. |
| `babble_retries` | `2` | Re-generations for a short chunk that kept talking past its text (`0` disables the guard). |
| `subtalker_temperature` / `subtalker_top_k` / `subtalker_top_p` | = main | Acoustic decoder overrides. |
| `codes_dump_file` | — | Parity debugging: appends prompt ids, reference codes and generated codes as text. |

Session options: `vieneu_v3_turbo.weight_type` (`native|f32|f16|bf16|q8_0`), `vieneu_v3_turbo.mem_saver`, `vieneu_v3_turbo.voice_prompt_cache_slots`. Options written with the old `vietneu_tts.` prefix are still accepted (mapped to the current prefix, with a deprecation note in the log); an unknown option is rejected rather than ignored.

## Parity and performance

Checked against fp32 references with identical prompt inputs (CPU, `weight_type=f32`): backbone prefill hidden state max |diff| 6e-7; acoustic-decoder logits identical to a numpy fp32 implementation of the safetensors weights; codec decoder 83 dB SNR on the same codes. With the GGUF packages the acoustic logits deviate by 0.007 (bf16) / 0.02–0.03 (q8_0) on average — smaller than the int8 acoustic graph the Python CPU path ships.

Speed on an Intel Core i5-12400F (6 P-cores, `--threads 6`, q8_0): a 28 s utterance in 8.0 s wall including process start and model load, RTF ≈ 0.29 (0.34 with `--threads 4`); the Python CPU path on the same machine is RTF 0.55–0.62 (fp32 ONNX) / 0.35 (int8 ONNX, needs VNNI).

## Chunking, pauses and the babble guard

A long text is generated chunk by chunk, and the seams are what make the pieces
sound like one utterance:

- **the cut** follows paragraphs, then sentences (`. ! ? …`), then minor
  punctuation, then whitespace, packing up to `text_chunk_size` characters of
  the phoneme string; a chunk under `text_chunk_min` joins a neighbour;
- **the pause** between two chunks is a minimum, not an insertion: the silence
  the model already left (tail of one, lead of the next) is measured and only
  the shortfall is padded with zeros — 0.70 s after a paragraph, 0.50 s after a
  sentence, 0.30 s otherwise. A longer natural pause is kept;
- **a chunk that ran to its frame ceiling** never emitted EOS, so its tail is
  missing; with sampling on it is generated again (up to twice) and the shorter
  result kept;
- **the babble guard** catches a short chunk that kept talking past its text
  ("Được." coming out as "Được không?"): it compares the energy bursts in the
  decoded audio with the chunk's syllable count, and re-generates a suspect
  chunk, keeping the best attempt. Measured on the Python engine: ~5% of
  one-syllable chunks, 8 cases in 480 chunks down to 0.

The text-level rules of the Python engine that need the normaliser — cutting at
connectives, never between two number words — are not here: they belong with a
text front end rather than as a second copy.

## Not yet ported

- CAM++ speaker encoder — pass `speaker_embedding_file`.
- Reference denoiser used at enrollment by the Python engine.
- Streaming; the enrollment denoiser.

## Credits

- Model, training data and reference implementation: Phạm Nguyễn Ngọc Bảo ([pnnbao97](https://github.com/pnnbao97), [pnnbao-ump on Hugging Face](https://huggingface.co/pnnbao-ump)) — also the maintainer of this audio.cpp family.
- First audio.cpp port (`vietneu_tts`, PR #80): Phuoc Nguyen ([phuocnguyen90](https://github.com/phuocnguyen90)).
- Audio codec: MOSS-Audio-Tokenizer-Nano (OpenMOSS-Team).
