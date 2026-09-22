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

GGUF packages published by the model author at [pnnbao-ump/VieNeu-TTS-v3-Turbo-GGUF](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo-GGUF), built from the current `update/` weights of [pnnbao-ump/VieNeu-TTS-v3-Turbo](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo) with `audiocpp_gguf` (the model spec, `config.json` and tokenizer sidecars are embedded, so one file is enough):

| File | Precision | Size |
|---|---|---|
| `vieneu-v3-turbo-q8_0.gguf` | Q8_0 matmuls, bf16 norms / embeddings | 170 MB |
| `vieneu-v3-turbo-bf16.gguf` | bf16 | 292 MB |
| `voices/<id>/{ref_codes.txt,speaker.emb.txt}` | — | the 25 preset voices of the Python SDK as packaged voices (`voices/manifest.json`) |

```bash
python tools/model_manager_v2.py install vieneu_v3_turbo          # q8_0 + the default voice (minh_quan_pro)
python tools/model_manager_v2.py install vieneu_v3_turbo_bf16
```

The original HF layout also loads directly (`--model <dir>` with `model.safetensors`, `config.json`, `tokenizer.json` and `speech_tokenizer/{config.json,model.safetensors}` = `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`).

To repack from a safetensors directory:

```bash
audiocpp_gguf --input model_weights=VieNeu-TTS-v3-Turbo/model.safetensors \
  --input speech_tokenizer_weights=VieNeu-TTS-v3-Turbo/speech_tokenizer/model.safetensors \
  --root VieNeu-TTS-v3-Turbo --family vieneu_v3_turbo --type q8_0 \
  --output vieneu-v3-turbo-q8_0.gguf
```

## Text input: phonemes

The Python package turns text into phonemes with [sea-g2p](https://github.com/pnnbao97/sea-g2p) (Vietnamese normalisation, English code-switching, emotion tags) before the model sees it. That front end is not part of audio.cpp yet, so `--text` must already be phonemes:

```python
from vieneu_utils.phonemize_text import phonemize_text_with_emotions   # pip install vieneu
print(phonemize_text_with_emotions("Xin chào thế giới. Đây là bản thử nghiệm."))
# sˈin tʃˈaː2w tˈeɜ zˈəːɜj. ɗˈəɪ lˌaː2 bˈaː4n tˈy4 ŋˈiɛ6m.
```

## Voice

Every request needs a voice. Two ways:

1. **Reference WAV** (`--voice-ref`): the codec encodes it into reference codes (mono, resampled to 48 kHz, first 8 s). A 192-d speaker embedding is also required: the CAM++ speaker encoder is not ported yet, so pass one with `speaker_embedding_file=` (192 comma-separated floats, produced by the Python engine's `extract_speaker_emb`). Without it the embedding is all zeros and the voice will not match.
2. **Packaged voice** (no audio): `reference_codes_file=` (one frame per line, 16 integers — `numpy.savetxt(codes, fmt="%d")` of the Python `ref_codes`) plus `speaker_embedding_file=`. This is what the Python preset voices ship, and it skips the encoder pass.

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
audiocpp_cli --task clon --family vieneu_v3_turbo \
  --model models/VieNeu-TTS-v3-Turbo-GGUF/vieneu-v3-turbo-q8_0.gguf --backend cpu \
  --text "<phonemes>" --voice-ref voice/ref.wav \
  --request-option speaker_embedding_file=voice/speaker.emb.txt --out out.wav
```

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
| `text_chunk_size` / `text_chunk_mode` | `200` / `default` | Framework chunking of the phoneme string. The Python engine chunks the *text* by sentence (≤ 256 characters) and joins chunks with short pauses; for best results chunk in Python and call once per chunk. |
| `subtalker_temperature` / `subtalker_top_k` / `subtalker_top_p` | = main | Acoustic decoder overrides. |
| `codes_dump_file` | — | Parity debugging: appends prompt ids, reference codes and generated codes as text. |

Session options: `vieneu_v3_turbo.weight_type` (`native|f32|f16|bf16|q8_0`), `vieneu_v3_turbo.mem_saver`, `vieneu_v3_turbo.voice_prompt_cache_slots`.

## Parity and performance

Checked against fp32 references with identical prompt inputs (CPU, `weight_type=f32`): backbone prefill hidden state max |diff| 6e-7; acoustic-decoder logits identical to a numpy fp32 implementation of the safetensors weights; codec decoder 83 dB SNR on the same codes. With the GGUF packages the acoustic logits deviate by 0.007 (bf16) / 0.02–0.03 (q8_0) on average — smaller than the int8 acoustic graph the Python CPU path ships.

Speed on an Intel Core i5-12400F (6 P-cores, `--threads 6`, q8_0): a 12 s two-sentence utterance in 4.0 s wall including process start and model load, RTF ≈ 0.33; the Python CPU path on the same machine is RTF 0.55–0.62 (fp32 ONNX) / 0.35 (int8 ONNX, needs VNNI).

## Not yet ported

- Text front end (sea-g2p normalisation + phonemisation) — pass phonemes.
- CAM++ speaker encoder — pass `speaker_embedding_file`.
- Reference denoiser used at enrollment by the Python engine.
- Sentence-based chunking with pause insertion, babble guard / retries, streaming.

## Credits

- Model, training data and reference implementation: Phạm Nguyễn Ngọc Bảo ([pnnbao97](https://github.com/pnnbao97), [pnnbao-ump on Hugging Face](https://huggingface.co/pnnbao-ump)) — also the maintainer of this audio.cpp family.
- First audio.cpp port (`vietneu_tts`, PR #80): Phuoc Nguyen ([phuocnguyen90](https://github.com/phuocnguyen90)).
- Audio codec: MOSS-Audio-Tokenizer-Nano (OpenMOSS-Team).
