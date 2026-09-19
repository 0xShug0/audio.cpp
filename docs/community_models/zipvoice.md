# ZipVoice (community model)

[ZipVoice](https://github.com/k2-fsa/ZipVoice) (k2-fsa) is a zero-shot voice-cloning TTS built on a
TTSZipformer flow-matching backbone with a Vocos mel-24kHz vocoder. `zipvoice_distill` is the
distilled variant (8 Euler steps, guidance-scale embedding); the base model uses batched
classifier-free guidance. Both share one architecture and one GGUF packaging.

**Status: inference complete.** Text encoder, duration/ratio conditioning, flow-matching decoder,
and the Vocos vocoder are verified against the reference PyTorch implementation with golden
fixtures (`tests/zipvoice/zipvoice_parity_main.cpp`): per-layer and per-submodule encoder taps,
text conditioning, sampled features, single- and batched velocity fields (odd/even/full lengths,
both timesteps), fbank, and vocos audio — all at cosine 1.0. The Chinese/English text frontend
(`emilia` mode) is verified token-for-token against the upstream `EmiliaTokenizer`
(`tests/zipvoice/zipvoice_zh_tokens_main.cpp`).

## Highlights

- Zero-shot voice cloning from a short reference clip + transcript (zh and en)
- Duration prediction from the prompt speaking rate (`speed` option scales it)
- Distill model: 8 Euler steps with guidance-scale embedding; base model: batched CFG
- Chinese + English mixed text via the `emilia` frontend (jieba + pypinyin tables + espeak-ng),
  inline pinyin overrides (`<zhong1> <guo2>`)
- Ready-made GGUF package hosted at [davidxifeng/zipvoice-gguf](https://huggingface.co/davidxifeng/zipvoice-gguf) (Apache-2.0 upstream); local re-conversion with the tools below also works

## Conversion

A ready-made, self-contained package is hosted at
[davidxifeng/zipvoice-gguf](https://huggingface.co/davidxifeng/zipvoice-gguf) — the model
manager downloads it directly (`zipvoice-distill-orig.gguf`, flow-matching model + bundled
Vocos + embedded frontend sidecars). The Q8_0 package includes the same resources:

```bash
python3 tools/model_manager_v2.py install zipvoice_distill_q8_0
```

To rebuild it locally:

```bash
# 1. stage the Chinese frontend tables (requires the upstream ZipVoice python env
#    for pypinyin; writes zh_chars.tsv / zh_phrases.tsv / zh_syllables.tsv and
#    downloads the pinned jieba dictionaries next to tokens.txt)
python3 tools/community_models/export_zipvoice_zh_dict.py \
    --output-dir /models/ZipVoice/zipvoice_distill

# 2. flatten + package (torch checkpoint -> safetensors -> GGUF with model.* and vocos.*)
python3 tools/community_models/convert_zipvoice.py \
    --model-dir /models/ZipVoice/zipvoice_distill \
    --vocos /models/vocos-mel-24khz/vocos.safetensors \
    --converter build/bin/audiocpp_gguf
```

The converter needs `tokens.txt` + `model.json` + `model.pt` in `--model-dir` (the HF
`k2-fsa/ZipVoice` `zipvoice_distill` snapshot layout). When the directory was staged by
`export_zipvoice_zh_dict.py`, the `zh_*` frontend sidecars are embedded into the GGUF alongside
`tokens.txt` / `model.json`, so a converted package is a single self-sufficient file (loose
copies are still staged next to it for the directory layout). With `--safetensors-only` it
stops at the development format (`zipvoice-orig.safetensors` + config + vocab in one
directory) — an intermediate for tooling and the direct synthesis API; the CLI loads GGUF
packages.

## CLI usage

```bash
# English cloning (espeak frontend, the default)
audiocpp_cli --task clon --family zipvoice \
    --model /models/ZipVoice-Distill-GGUF/zipvoice-distill-orig.gguf \
    --session-option zipvoice.espeak_library_path=/opt/homebrew/lib/libespeak-ng.dylib \
    --voice-ref prompt.wav --reference-text "Reference transcript." \
    --text "Text to synthesize." --out out.wav

# Chinese / mixed text (emilia frontend, built in; the zh_* tables are embedded in the
# GGUF, or staged by export_zipvoice_zh_dict.py next to it)
audiocpp_cli --task clon --family zipvoice \
    --model ... \
    --session-option zipvoice.espeak_library_path=/opt/homebrew/lib/libespeak-ng.dylib \
    --voice-ref prompt.wav --reference-text "参考文本。" \
    --text "要合成的文本。" --out out.wav
```

The text frontend is fixed to the EmiliaTokenizer pipeline (zh/en/mixed; the upstream
default): `tokenizer` is no longer a session option. Session options: `zipvoice.vocos_path`
(only for safetensors checkpoints without a bundled vocoder), `zipvoice.espeak_library_path`,
`zipvoice.espeak_data_path`, `zipvoice.num_inference_steps`, `zipvoice.guidance_scale`,
`zipvoice.t_shift`. Requests accept `reference_text` (required), `guidance_scale`,
`num_inference_steps`, `t_shift`, `speed`, `feat_scale`, `target_rms`, `seed`, `lang` (espeak
voice for English segments), and `token_ids`/`prompt_token_ids` to bypass the frontend entirely
(direct API callers can also still select the espeak/simple frontends through
`ZipVoiceSynthesisRequest::tokenizer`).

## Frontend details (emilia mode)

The upstream default `EmiliaTokenizer` pipeline is reproduced exactly:
Chinese text normalization (framework `ChineseTextNormalizer`) → punctuation mapping → jieba
segmentation (a model-local port of the Jieba maximum-probability DAG and four-state BMES HMM,
`src/community_models/zipvoice/jieba_segmenter.cpp`, same dictionaries as python jieba) → pypinyin
readings from baked tables (TONE3 syllables split into initial+`0` / final+tone tokens) → tone sandhi
(3rd-tone runs, 一/不) applied per jieba word → token mapping with OOV skipping; English runs are
phonemized with espeak-ng. Verification compares C++ token ids against the upstream tokenizer
run in the ZipVoice python environment (`tests/zipvoice/zh_reference_ids.py`).

The jieba segmentation port is derived from [cppjieba](https://github.com/yanyiwu/cppjieba)
(Copyright (c) 2013 Yanyi Wu) and [jieba](https://github.com/fxsjy/jieba) (Copyright (c) 2012 Sun
Junyi), both MIT licensed; the `zh_jieba_dict.txt` / `zh_hmm_model.txt` resources are the jieba
dictionaries and are embedded in the GGUF.

## Performance

The measured ZipVoice-Distill ggml Metal runs achieve **RTF < 1**, the target
for community models. RTF is synthesis wall time divided by generated audio
duration; lower is better.

Measured on **Apple M3**, **Metal** backend, on **2026-09-18**, with a
Release build and 8 CPU threads, using
`zipvoice-distill-orig.gguf`, 8 Euler steps, guidance scale 3, time shift 0.5,
speed 1, seed 666, and the 24 kHz Vocos vocoder. Each reference pair was tested
with one discarded warmup followed by three timed calls in the same process.
Wall time includes tokenization, reference preprocessing, feature extraction,
text encoding, the solver, and Vocos. Model loading and initial compilation
are excluded by warmup; reference-file reading and output WAV writing are
outside the timer.

| Reference audio / transcript | Generated audio | Mean synthesis wall time | Mean RTF | Meets RTF < 1 |
|---|---:|---:|---:|---|
| `zh-male-ref.wav` / `zh-male-ref.txt` | 19.115 s | 4.697 s | **0.2457** | Yes |
| `zh-ref.wav` / `zh-ref-text.txt` | 12.843 s | 2.332 s | **0.1816** | Yes |

The individual measured RTFs were `0.245406`, `0.245910`, `0.245799` for
the male reference and `0.181118`, `0.181592`, `0.182100` for the female
reference. Both used this exact target text, including the space in “宁 静”:

> 风声渐息，落叶不再沙沙作响。小狐狸缓缓合上双眼，在平缓绵长的呼吸声中，安然步入宁 静的梦境。

The measured implementation includes the session-owned runtime, framework Vocos graph, and
model-local Jieba implementation. It measures the direct synthesis API with
one persistent `ZipVoiceComputeDevice` per reference pair; session-level
long-text chunking is not exercised. The two benchmarks ran sequentially.
The discarded first calls took 6.744 s (male) and 2.772 s (female), including
lazy model loading and graph/kernel setup but excluding resource-bundle
resolution and input-file reading.

Both cases meet the community-model target. Results cover these two inputs
on one machine; CPU and CUDA performance are not established by this measurement.
Local benchmark source, full logs, and output WAVs are saved in
`outputs/zipvoice-metal-current-20260918/` (not distributed with the model).

## Parity

Select `--backend metal` for Apple GPU inference or `--backend cpu` for the
CPU reference path. The text encoder, flow decoder and Vocos backbone run on
the selected backend; feature extraction and the final ISTFT run on the host.
The direct C++ API defaults to `BestAvailable`, while sessions honor the requested
backend, including an explicit CPU selection.

CPU/Metal parity is tested on Apple M3 with the Distill GGUF, including odd
sequence lengths, batched velocity evaluation, Vocos and eight-step synthesis.
ZipVoice requests F32 matrix products to avoid FP16 staging error accumulating
through the sampler. CUDA uses the same graphs but has not been verified on
hardware in this GPU validation run.

```bash
# golden fixtures: tests/zipvoice/build_reference.py (upstream env) + parity harness
build/bin/zipvoice_parity reference.npz <model-path> reference.npz vocos.safetensors out.wav
build/bin/zipvoice_zh_tokens zh_reference_ids.json <model-dir>
```

Weights and runtime graphs are owned by the `ZipVoiceComputeDevice` runtime (one
per session), so callers injecting a borrowed backend must release the device
runtime (for example with `zipvoice_clear_runtime(device)`) before destroying
that backend, after all ZipVoice calls using it have finished.
