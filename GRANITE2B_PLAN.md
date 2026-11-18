# Add IBM Granite Speech 5.0 2B to audio.cpp

## Current support

**`ibm-granite/granite-speech-5.0-2b` is unsupported in this checkout (`a6e32214`).** A full CMake configure generated one Granite loader, `make_granite5asr_loader`, and `python3 tools/model_manager_v2.py list --json` listed only its two TurboCTC packages. The `granite_speech` build alias also resolves to `granite5asr`. This was verified through the generated registry, package catalog, and source inspection. Model inference was outside this verification. [CMake registration](CMakeLists.txt#L2522) · [Package spec](model_specs/granite5asr.json) · [Current model guide](docs/models/granite5asr.md)

| IBM checkpoint | Inference architecture | Current support |
| --- | --- | --- |
| Granite Speech 5.0 470M TurboCTC | Conformer and non-autoregressive greedy CTC | `granite5asr`, with native Safetensors and audio.cpp GGUF packages |
| Granite Speech 4.1 2B NAR | CTC hypothesis editing by a bidirectional LLM | No dedicated loader or package in this checkout. [IBM NAR model card](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar) |
| Granite Speech 5.0 2B | Conformer, Q-Former projector, and autoregressive Granite LLM | No dedicated loader or package in this checkout. [IBM 5.0 2B model card](https://huggingface.co/ibm-granite/granite-speech-5.0-2b) |

TurboCTC consumes 320-wide mel-plus-delta features and decodes frame-level CTC logits. The 5.0 2B checkpoint has `model_type: granite_speech` and `GraniteSpeechForConditionalGeneration`, consumes 160-wide stacked mel features, and projects the encoder's hidden states into a causal decoder. The model needs a new inference path. [TurboCTC frontend](src/community_models/granite5asr/frontend.cpp) · [IBM 2B configuration](https://huggingface.co/ibm-granite/granite-speech-5.0-2b/blob/main/config.json)

## Proposed release scope

Add `granite_speech_2b` as a community family under `src/community_models/` and `include/engine/community_models/`, with a model spec v1. The family boundary follows the distinct autoregressive runtime. Preserve the existing `granite5asr` loader, packages, and aliases. Reuse framework components where their semantics match. [Model contribution guidelines](CONTRIBUTING.md) · [Community model requirements](https://github.com/0xShug0/audio.cpp/issues/54)

Start with offline multilingual ASR in English, French, German, Spanish, Portuguese, and Japanese through the CLI and legacy server. Load IBM's native checkpoint for numerical parity, then release a standalone audio.cpp GGUF package. Include punctuation and capitalization, optional source-language hints, and the framework's offline chunking. Follow with keyword biasing and speech translation. Defer live ingest, partial decoding, diarization, and word timestamps until there is a concrete implementation and validation for each capability. [IBM model card](https://huggingface.co/ibm-granite/granite-speech-5.0-2b)

## Work sequence and deliverables

| Milestone | Depends on | Deliverables | Exit evidence |
| --- | --- | --- | --- |
| A. Freeze the reference | IBM snapshot and Python environment | Checkpoint inventory, reference generator, fixture manifest, first trace | IBM sample and two local languages run end to end on pinned Transformers. Tensor inventory identifies every live weight and sidecar. |
| B. Port the acoustic model | A | Native asset parser, frontend, Conformer, Q-Former, C++ stage probe | CPU matches the pinned FP32 reference through the projected audio vectors at clip, window, and block boundaries. |
| C. Port the decoder and prompt | A. Can proceed alongside B using saved projector vectors. | Granite decoder settings, tokenizer, prompt builder, generation and cache tests | CPU matches mixed-input prefill and forced cached logits. Short golden clips match greedy token sequences. |
| D. Integrate offline ASR | B and C | Loader, spec, reusable session, chunking, CLI and legacy server path coverage | Native checkpoint runs from normal entry points. Error recovery, concurrent request isolation, and TurboCTC regression checks pass. |
| E. Ship GGUF | D | Standalone conversion, package, docs, catalog, quality and performance report | Clean install and outside-tree smoke pass. Quantized quality and target-backend checks meet the criteria below. |
| F. Extend prompting | D. Use E's published artifact for release checks. | Keyword biasing and speech translation with separate option and test coverage | Paired biasing and bidirectional translation fixtures pass, along with the ordinary ASR regression set. |

Land or review B and C independently where practical. Keep optional math added to the shared decoder in a focused change with existing caller regressions. Bring the model up as `experimental` with an empty `packages` array until E. Use `community` once an installable artifact and runtime validation are available. [Community contribution rules](CONTRIBUTING.md) · [Spec and package rules](docs/maintainers/model_specs.md)

### Implementation map

| Planned path | Implementation responsibility | Starting point |
| --- | --- | --- |
| `src/community_models/granite_speech_2b/assets.cpp` and matching `assets.h` | Family and variant identification, typed checkpoint config, resource bundle, tensor inventory, and early validation | [Audio8 ASR assets](src/community_models/audio8_asr/assets.cpp) |
| `frontend.cpp` and `frontend.h` | PCM normalization, 80-bin logmel, even-frame pairing, and shared length calculations | [Shared mel frontend](src/framework/audio/nemo_mel_frontend.cpp) |
| `encoder.cpp`, `projector.cpp`, and corresponding headers | Graph construction, weight binding, CPU/GPU execution, and stage probe access | [TurboCTC encoder](src/community_models/granite5asr/encoder.cpp) · [Audio8 ASR projector](src/community_models/audio8_asr/projector.cpp) |
| `decoder.cpp`, `prompt.cpp`, and corresponding headers | Bind Granite weights into shared decoder, implement prompt and length rules, and decode token streams | [Audio8 thinker](src/community_models/audio8_asr/thinker.cpp) · [Greedy runtime](src/framework/runtime/greedy_causal_decoder.cpp) |
| `session.cpp` and `session.h` | Loader factory, loaded model, backend state, chunking, task input/output, and cancellation checkpoints where available | [Audio8 ASR session](src/community_models/audio8_asr/session.cpp) · [TurboCTC session](src/community_models/granite5asr/session.cpp) |
| `model_specs/granite_speech_2b.json` and `CMakeLists.txt` | Typed model contract, embedded packaging, registry, and conditional test targets | [Model spec v1](docs/maintainers/model_specs.md) · [CMake model block](CMakeLists.txt#L2522) |
| `tests/granite_speech_2b/` and the shared CLI case file | `reference.py`, `compare_trace.py`, tensor/shape unit tests, native parity probe, path cases, and optional warm benchmark | [ASR validation assets](assets/asr_validation/README.md) · [CLI case file](tools/audiocpp_cli/audiocpp_cli_path_cases.json) |

## A. Freeze the reference and enumerate the assets

Use checkpoint revision `ec49ca6b1a5e7e9176127e09cb303ab78cbacdf7`. The pinned repository contains one 4.63 GB primary weight file and a separate `out_llm.safetensors`. The native generation source calls the encoder, projector, and LLM using model weights loaded by `from_pretrained`. Confirm which files were opened by running the local model with the explicit list below and checking load warnings. Inventory the auxiliary file separately if its role needs confirmation. Include auxiliary weights in the package only if ordinary reference generation uses them. [IBM snapshot metadata](https://huggingface.co/api/models/ibm-granite/granite-speech-5.0-2b) · [IBM checkpoint files](https://huggingface.co/ibm-granite/granite-speech-5.0-2b/tree/main) · [Transformers model](https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/granite_speech/modeling_granite_speech.py)

Run from the repository root. Use Python 3.11 or 3.12 for this starting environment. Start with CPU and FP32 model execution for a stable reference. If the proposed environment needs an adjustment, record the resolved working versions and rerun the fixture before generating goldens.

```bash
export GRANITE2B_WORK="$PWD/build/granite-2b-handoff"
export GRANITE2B_HF="$GRANITE2B_WORK/hf"
export GRANITE2B_REV=ec49ca6b1a5e7e9176127e09cb303ab78cbacdf7
mkdir -p "$GRANITE2B_WORK"
python3 -m venv "$GRANITE2B_WORK/venv"
. "$GRANITE2B_WORK/venv/bin/activate"
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.8.0 torchaudio==2.8.0
python -m pip install 'transformers==4.57.0' safetensors soundfile
hf download ibm-granite/granite-speech-5.0-2b \
  config.json generation_config.json processor_config.json preprocessor_config.json \
  tokenizer.json tokenizer_config.json special_tokens_map.json added_tokens.json \
  chat_template.jinja vocab.json merges.txt model.safetensors multilingual_sample.wav \
  --revision "$GRANITE2B_REV" --local-dir "$GRANITE2B_HF"
python -m pip freeze > "$GRANITE2B_WORK/python-versions.txt"
(cd "$GRANITE2B_HF" && sha256sum *.json *.jinja merges.txt *.safetensors *.wav) \
  > "$GRANITE2B_WORK/checkpoint.sha256"
```

IBM's published SHA-256 for `model.safetensors` is `4151269881d2dd825d45ca75d6552b2a20e6e39d0934ed84c4db826af8bf6d88`. To inventory header metadata without materializing the tensors, read the first eight bytes as an unsigned little-endian length, then decode that many bytes as JSON. Emit a sorted row per tensor with name, source file, source shape in PyTorch order, dtype, and byte range. On the reference load, inspect missing/unexpected/mismatched keys with `output_loading_info=True`; record which exceptions are justified by the upstream graph. Require 40 text layers, 16 encoder layers, two Q-Former layers, and both CTC feedback projections in the inventory. Use the inventory to check the key families in B and C before finalizing their bindings. [Published weight file](https://huggingface.co/ibm-granite/granite-speech-5.0-2b/blob/main/model.safetensors)

Runnable reference smoke after those setup commands. It uses the shipped sample, the stock chat template, and the generation configuration loaded from the snapshot.

```bash
python - <<'PY'
import json
import os
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

root = Path(os.environ["GRANITE2B_HF"])
torch.set_num_threads(8)
processor = AutoProcessor.from_pretrained(root, local_files_only=True)
model, load_info = AutoModelForSpeechSeq2Seq.from_pretrained(
    root, torch_dtype=torch.float32, local_files_only=True,
    output_loading_info=True, attn_implementation="eager",
)
model.eval()
samples, rate = sf.read(root / "multilingual_sample.wav", dtype="float32")
assert rate == 16000 and samples.ndim == 1
user = "<|audio|>transcribe the speech with proper punctuation and capitalization."
prompt = processor.tokenizer.apply_chat_template(
    [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
)
inputs = processor(prompt, torch.from_numpy(samples)[None], device="cpu", return_tensors="pt")
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=512, do_sample=False, num_beams=1)
ids = output[0, inputs["input_ids"].shape[-1]:].tolist()
print(json.dumps({
    "load_info": load_info,
    "generation_config": model.generation_config.to_diff_dict(),
    "prompt": prompt,
    "audio_shape": list(inputs["input_features"].shape),
    "input_ids": inputs["input_ids"][0].tolist(),
    "generated_ids": ids,
    "text": processor.tokenizer.decode(ids, skip_special_tokens=True),
}, ensure_ascii=False))
PY
```

Turn that smoke into `tests/granite_speech_2b/reference.py --model PATH --audio PATH --out-dir DIR [--language CODE] [--max-tokens N] [--force-ids FILE]`. `--force-ids` feeds the same known continuation through both decoders for numerical comparison even if an earlier greedy token differs. Define the probe interface before B and C start. The C++ target `granite_speech_2b_parity_probe` should take the same core arguments plus `--backend`, `--threads`, and `--through frontend|encoder|projector|prefill|generate`; add a direct projector-fixture input for decoder work in parallel.

| Trace output | Storage and contents |
| --- | --- |
| `trace.json` | Revision, hashes, reference/runtime versions, backend, dtype, input/output lengths, sample rate, prompt UTF-8, IDs including observed stop in the reference, audio span indices, model config values, and emitted text. Include tensor file names and shapes. Use the same keys in C++ and Python. |
| `*.f32` | Little-endian, contiguous, time-major float dumps. Capture mel, stacked features, input projection, encoder block 1, layer 8 before/after feedback, last encoder layer, projected vectors, merged/scaled prefill, last-position logits, and each of the first eight forced cached steps. Size-check files using recorded shapes before comparing. |
| `reference_manifest.jsonl` | Per fixture, record ID, audio file path, SHA-256, source dataset and license, language, request options, and optional human reference. Keep corpus audio and full traces in artifact storage or local build paths. Commit tiny synthetic input/golden fixtures appropriate for ordinary CI. |

For FP32 isolate math first by capturing the reference with `model.eval()`, eager attention, and fixed inputs. Record native-BF16 and official llama.cpp Q8_0 separately for quality comparisons. Establish the stage tolerances with B and C, using the initial limits in the verification section.

Once the planned trace tools exist, the first acoustic parity check should be repeatable with the following commands. Define `compare_trace.py --reference DIR --candidate DIR` to use the candidate's declared `--through` stage as its cutoff, require every trace entry through that stage, and fail on the first mismatch in pipeline order with the shape, error metrics, and worst coordinate. Run it again through `generate` after C.

```bash
GRANITE2B_FIXTURE="$PWD/assets/asr_validation/common_voice_ja/common_voice_ja_20461197.wav"
python tests/granite_speech_2b/reference.py --model "$GRANITE2B_HF" \
  --audio "$GRANITE2B_FIXTURE" --out-dir "$GRANITE2B_WORK/trace/ja-hf"
"$GRANITE2B_WORK/bin/granite_speech_2b_parity_probe" --model "$GRANITE2B_HF" \
  --audio "$GRANITE2B_FIXTURE" --backend cpu --threads 8 --through projector \
  --out-dir "$GRANITE2B_WORK/trace/ja-cpp"
python tests/granite_speech_2b/compare_trace.py --reference "$GRANITE2B_WORK/trace/ja-hf" \
  --candidate "$GRANITE2B_WORK/trace/ja-cpp"
```

## B. Implement assets, frontend, encoder, and projector

### B1. Resource and weight binding

In the source-layout spec, give native resources a `model:.` root and a `weights` tensor entry targeting `model:model.safetensors`. For standalone GGUF, bind the same logical `weights` entry to `weights:` with root `"$gguf"` and resolve files by the same embedded destination names. Use the `audio8_asr` spec for the exact schema and keep the native and converted tensor names identical. Declare `config.json`, `preprocessor_config.json`, `processor_config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, `special_tokens_map.json`, and `chat_template.jinja` in the staged package. Check whether `vocab.json`, `merges.txt`, and `added_tokens.json` contribute anything beyond the embedded `tokenizer.json` before requiring copies. [Resource bundle example](model_specs/audio8_asr.json) · [Model spec resolver](include/engine/framework/model_spec/package.h)

Use `load_resource_bundle()` and `open_tensor_source("weights")`. Parse metadata once into model-local encoder, projector, text, and preprocessing configs. Validate the outer architecture, all nested model types, no external LoRA requirement on this snapshot, tensor shape relationships, context/window bounds, and vocabulary IDs. In `can_load()`, distinguish `granite_speech` conditional generation from TurboCTC and `granite_speech_nar`; a family hint filters probes but does not substitute for checkpoint identity. Keep `can_load()` and `inspect()` metadata-only. On explicit `load()`, report family, component, tensor or file name, and actual versus expected shape where available. Check all required bindings before uploading large weights. Use `BackendWeightStore` and existing `TensorSource`/shared decoder materialization paths. Retain a reopenable source until sessions that need it can load, and release host tensor storage only after all consumers of that read have finished. [Tensor source](include/engine/framework/assets/tensor_source.h) · [Backend weight store](include/engine/framework/core/backend_weight_store.h)

Use the following inventory expectations in PyTorch dimension order. Derive exact names from A's header, including the Q-Former subtree and text LM head, before binding. Let framework loaders handle ggml dimension ordering and known quantized layouts.

| Component | Expected bindings or dimensions |
| --- | --- |
| Encoder input and conditioning | `encoder.input_linear` [1024,160], `encoder.out` [348,1024], `encoder.out_mid` [1024,348], with matching biases. Reuse the same character projection at block 8. |
| `encoder.layers.{i}` | `ff1`/`ff2` 1024→4096→1024, pre/post norms, `attn.to_q` [1024,1024], `attn.to_kv` [2048,1024], `attn.rel_pos_emb` [1025,128], `attn.to_out`, `conv.up_conv` [4096,1024,1], `conv.depth_conv.conv` [2048,1,15], BN parameters and running statistics, `conv.down_conv` [1024,2048,1]. |
| `projector` | `query` [1,3,1024], `qformer` embedding norm and two layers each with 16 heads of width 64, self/cross attention, query FFN 1024→4096→1024, and final `linear` [2048,1024] plus bias. |
| `language_model` | Token table and separate LM head [100353,2048], final norm [2048], 40 layers with Q [2048,2048], K/V [512,2048] each, output [2048,2048], gate/up [4096,2048] each, down [2048,4096], and two RMSNorm weights [2048] per layer. |

### B2. Frontend and length contract

Validate interleaved `AudioBuffer` shape, positive input rate/channel count, and finite amplitudes before DSP. Downmix using the existing framework path and resample to 16 kHz. Use the post-resample count to calculate lengths. Feed `NemoMelFrontend` with a centered periodic Hann window, reflect padding, HTK filters with no mel normalization, 512 FFT / 400 window / 160 hop, power spectrum, and `DensePowerLog10` with a `1e-10` floor. Select time-major layout, `StftFrames`, `stack_frames=1`, and no deltas. Apply the reference's per-clip maximum floor across **all** raw frames, including an odd trailing frame, then `(value / 4) + 1`. Pair adjacent frames into 160-dimensional rows by discarding the final row when odd. This avoids the shared stack path's extension of the waveform for even padding. Check exact framework window/filterbank and log operations against the upstream intermediate tensors before enabling any optimized path. [Shared mel implementation](src/framework/audio/nemo_mel_frontend.cpp) · [Pinned feature extractor](https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/granite_speech/feature_extraction_granite_speech.py)

For a single valid clip with `N` mono samples at 16 kHz, calculate the following in one helper and assert observed counts at each stage. Use checked integer arithmetic for output sizes.

```text
M = floor(N / 160) + 1              # raw mel frames
T = floor(M / 2)                    # encoder steps, width 160 at input
W = ceil(T / 15)                     # projector windows
A = 3 * W                            # audio placeholders, each width 2048
```

With the reference reflection pad width of 256, use an empty transcript for a valid zero-sample buffer and a clear input error for a nonempty clip of 256 samples or fewer after resampling. Have the chunk planner avoid producing such a standalone tail by merging it into a neighboring in-bounds span. Test 257 and 320 samples, `T=14/15/16`, and `T=199/200/201`. For each target `T`, test `N=320*T-160` and `N=320*T` to produce the same number of encoder frames from even and odd mel frames, respectively. Also compare impulse, silence, and resampled/stereo inputs. Use identical resampled PCM for intermediate parity and compare end-to-end resampling independently.

### B3. Conformer graph

Implement with model-local scheduling and shared linear/normalization/convolution operations. Build input and output tensors as contiguous rows of width 160 and 1,024. Validate the first block before cloning it across all 16. Each block does a pre-norm Macaron FFN with 0.5 residual weight, Shaw attention plus residual, convolution plus residual, a second FFN with 0.5 residual weight, and post-layer norm. Use the upstream FFN activation and epsilon, including module defaults. For the convolution, check the GLU split order, symmetric padding of seven samples, BN eps and inference statistics, SiLU, and final pointwise projection. Run convolution over the original contiguous time sequence across attention block boundaries. If folding BN, verify the unfolded and folded CPU output before quantization. [Pinned Conformer implementation](https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/granite_speech/modeling_granite_speech.py) · [Existing depthwise/attention code](src/community_models/granite5asr/encoder.cpp)

For attention, apply pre-norm before zero-padding the final 200-frame block. Split packed KV into K followed by V. For valid query/key indices `i,j` in a block, use `(Q_i · K_j + Q_i · R[512 + i - j]) / sqrt(128)`, blocking invalid keys in the last block and cropping padded queries. Materialize scores only inside each block and retain the time-major order when reassembling frames. At the end of block 8, apply `h += out_mid(softmax(out(h)))` independently for each real frame. Feed raw block-16 states to the projector. Compare per-block activations around the final tail and conditioning before debugging any downstream mismatch.

### B4. Q-Former graph

Right-zero-pad the encoder state matrix to `W * 15` rows and reshape into independent 15-row windows. Start each window from the same three learned query vectors. Apply the query input LayerNorm, then two layers of query self-attention, cross-attention from the three queries to that window's 15 acoustic rows, and query-only GELU FFN. Use the pinned reference's dense/bias tensors, post-residual LayerNorm placement, and `1e-12` Q-Former epsilon. The upstream projector passes an absent encoder-attention mask, so the zero-padded rows participate in the final cross-attention window. Flatten queries in window-major, query-major order and apply the final 1024→2048 projection. Keep padded encoder slots confined to this stage. Test that changing one window changes only its three outputs in a projector-only fixture. [Pinned Q-Former implementation](https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/blip_2/modeling_blip_2.py) · [Pinned Granite projector](https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/granite_speech/modeling_granite_speech.py)

## C. Implement the prompt and text decoder

### C1. Tokenizer and prompt contract

Use `LlamaBpeTokenizerSpec` with the checkpoint's `tokenizer.json` and `tokenizer_config.json` as the starting route. Inspect the serialized normalizer, split regex, byte-level options, merge settings, postprocessor, and added-token whitespace flags. Prove equivalence before selecting a pretokenizer preset. If the preset differs, use the model-owned splitter hook or an additive preset with a test. The framework's `HuggingFaceTokenizerJson` class supplies decode only. Use whole-sequence decode to handle tokens that split a UTF-8 code point. [Encoding and decoding APIs](include/engine/framework/tokenizers/llama_bpe.h) · [Decode-only API](include/engine/framework/tokenizers/hf_tokenizer_json.h)

1. Build a single user turn with the model-owned audio marker ID `100352` and a transcription instruction. Use the pinned chat template, including the space preceding the assistant cue. Match the Python processor's whole-string tokenization and explicit or implicit BOS behavior. Demonstrate parity for a one-marker prompt and for the expanded marker run. Keep the template in the package and verify its supported single-turn shape during asset validation.
2. Implement language hints using a fixed `en/fr/de/es/pt/ja → English/French/German/Spanish/Portuguese/Japanese` mapping and the source-language instruction pattern from IBM. Keep the default prompt identical to the Python smoke. Store exact prompt/ID fixtures for no hint, `en`, and `ja`. Apply the same hint to each chunk. Omit a language result on an unspecified source and use the provided normalized code when present. [IBM prompt examples](https://huggingface.co/ibm-granite/granite-speech-5.0-2b#usage)
3. Build the expanded ID vector plus `GreedyCausalDecoderRuntime::Injection` with a sorted, contiguous range of position indices and `A * 2048` floats. Compare the position indices and count with the HF input mask. Validate that the range is within the prompt and that the values and positions have matching lengths. Text controls added in F must reject user-supplied reserved audio markers before expansion.
4. Use generation config and tokenizer precedence from the Python trace for EOS/BOS/pad. The text config uses `100257` for BOS/EOS and `100256` for pad, which should agree with the resolved generation settings. Keep EOS handling separate from display filtering. Decode the generated sequence and skip special tokens with the reference tokenizer semantics. Python generation can return the EOS token while the shared C++ runtime excludes it, so preserve the observed EOS in the trace and normalize both token sequences when comparing visible output. [Pinned speech processor](https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/granite_speech/processing_granite_speech.py) · [IBM text configuration](https://huggingface.co/ibm-granite/granite-speech-5.0-2b/blob/main/config.json)

### C2. Decoder math and reuse strategy

Start with `GreedyCausalDecoderRuntime::generate(..., reuse_graphs=true)`. This path already supports host lookup, audio injection, cache-sized prefill blocks, and token-by-token decode through `CausalDecoderRuntime`. It avoids the original path's host KV export/import. Use one complete prompt per inference call and zero retained prefix steps. Use the existing non-flash attention policy for the initial CPU reference. Find the direct and static-cache code paths in `DecoderLayerModule` and `CausalDecoderModule` before adding optional configuration fields. If a path cannot implement a setting correctly, make that path unavailable for the Granite configuration until covered by math and tests. [Greedy decoder](src/framework/runtime/greedy_causal_decoder.cpp) · [Decoder layer](src/framework/modules/transformers/decoder.cpp) · [Causal decoder graph](src/framework/modules/transformers/causal_decoder.cpp)

| Behavior | Granite implementation and parity check |
| --- | --- |
| Weight layout | Bind `language_model.model.embed_tokens`, 40 `language_model.model.layers`, final RMS norm, and untied `language_model.lm_head`, subject to the inventory from A. Use separate Q/K/V, bias-free text attention and MLP, no Q/K norm, RMS epsilon `1e-5`, SiLU-gated MLP, and grouped-query attention with four query heads per KV head. |
| Embedding scale | Multiply the merged sequence by `12` **after** substituting projected audio rows. Apply the same factor exactly once to each generated-token lookup. Place the opt-in scaling at a common graph boundary reached by both embedded prefill and token lookup decode. Use a fixture with distinct token and audio rows so a double scale or an audio-only omission is observable. |
| Attention scale | Apply `0.0078125` to Q·K before softmax with the ordinary causal mask. Supply this value to both plain and any accelerated attention path that is selected. Usual `1/sqrt(128)` stays the default for existing callers. |
| Layer residual | Change both branches to `residual + 0.22 * branch_output`, once after the attention output projection and once after the MLP down projection. Preserve accumulation/cast ordering in reference mode. |
| RoPE | Use 128 rotary channels in NeoX half-pair layout, base 10,000, factor 4, original positions 4,096 and model limit 16,384. Match upstream YaRN beta defaults 32/1, inverse-frequency interpolation and cos/sin magnitude factor (about 1.13863 for the published defaults). Export the reference's 64 inverse frequencies and magnitude factor. Test Q/K at 0, 1, 4095, 4096, 4097, and 16383 before testing full decode. If the ggml parameters differ numerically, use reference-derived frequency divisors and one separate Q/K magnitude scaling step. Apply scaling once and preserve absolute positions across prefill blocks. |
| Logits | Divide the final projection by `8`. Compare all 100,353 last-position logits on the short FP32 fixtures and the top IDs and margins at each forced decode step. Stop on EOS or the requested budget. |
| Shared protections | Use neutral defaults for opt-in scale fields. Retain current attention modes, RoPE behavior, layouts, and logits for callers that omit them. Run shared decoder tests and an Audio8 ASR and Qwen3 ASR parity/smoke case after the shared change. |

Use upstream math as the authority for the scaling and reference constants. Make cache tests exercise both block prefill and single-token paths with the same forced continuation. Compare against a full-prefix recompute at the same positions. For positions above 4,096, use a decoder-only fixture and the pinned checkpoint so coverage is independent of the audio chunk cap. [Granite decoder source](https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/granite/modeling_granite.py) · [YaRN calculation](https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/modeling_rope_utils.py)

## D. Integrate a reusable offline ASR session

Implement the `IVoiceModelLoader` factory and match the model capabilities and metadata in `advertised_capabilities()`, `inspect()`, and the typed spec. The loaded model owns checkpoint metadata and reopenable tensor assets. The session owns its execution context and mutable graph/cache state. Use the model's usual `RuntimeSessionBase` prepare/run contract. Only share backend weights if lifetimes and backend/device keys match existing framework ownership. Guard or serialize access to a single session according to the existing server and runtime contract. Never carry a previous request's prompt or cache state into the next one. Republish the current prompt, absolute positions and masks on each call, and reset state after errors. [Session contract](include/engine/framework/runtime/session.h) · [Offline loader/session example](src/community_models/audio8_asr/session.cpp)

### D1. Initial options and error behavior

| Scope and key | Behavior to implement |
| --- | --- |
| Request `language` | Optional enum or typed string with validated `en/fr/de/es/pt/ja` values. Prefer the explicit request option, then `TaskRequest::text_input.language`. Trim and lowercase ASCII code strings. A blank value requests automatic recognition without a hint. Reject unsupported codes. |
| Request `max_tokens` | Integer, default 512 generated tokens per chunk. Require 1 through 16,383, then enforce the tighter processor-derived context bound. A deliberate budget returns the generated prefix under the existing result convention. Expose cap/EOS in the trace for debugging. |
| Request `audio_chunk_mode` | `auto` (default), `fixed`, `vad`, or `none`. In `auto`, keep a short input whole, use configured VAD for long input, and use fixed chunking if VAD is unavailable or errors. Explicit `vad` requires a valid VAD model; fail with its actionable configuration or inference error. |
| Request `audio_chunk_duration_sec` | Positive finite value at most 120. Initial default 120 seconds, subject to the quality gate before release. Ignore the chunk duration override in `none` mode. Until validated otherwise, one model pass has a 120-second cap. |
| Session `granite_speech_2b.vad_model_path` | Optional existing VAD weights file. Resolve using the existing VAD loader rather than hardcoding a VAD family. |
| Session or load precision | Use native storage by default. For parity, provide a way to load the BF16 snapshot into FP32 execution, and document any user-facing weight override only once both graphs and specs implement it. Read GGUF mixed types directly. Advertise only supported combinations. |

Build the prompt budget before allocating the acoustic or text graphs. Calculate `A` from normalized samples, then derive text overhead from the actual token IDs. Require `expanded_prompt_length + max_tokens <= 16384`. Fail on an oversized prompt or disabled chunking with oversized audio instead of silently truncating the waveform. With an FP16 KV cache, budget about 80 KiB/token from 40 layers × 4 KV heads × 128 elements × K/V × 2 bytes. Full capacity is 1.25 GiB before graph and allocator overhead. Record actual cache precision and capacity in the benchmark. Keep allocation reuse bounded and verify that changing audio lengths does not leak previously cached positions.

### D2. Chunk planning and request flow

1. Validate the request and produce mono 16 kHz PCM, or an empty transcript if the valid buffer is empty. Use one sample clock throughout chunk planning and inference. Reuse `parse_audio_chunk_mode()`, `parse_audio_chunk_seconds_override()`, `plan_audio_chunks()`, `plan_vad_audio_chunks()`, and existing VAD loading. Cap each delivered chunk. Join tiny tails when they fit and rebalance the last two neighboring chunks when they do not. Use zero overlap initially because the first release has no alignment metadata to remove duplicated text. When successful VAD yields no speech spans, return an empty transcript. [Audio chunk helpers](include/engine/framework/audio/chunking.h) · [VAD integration example](src/community_models/granite5asr/session.cpp)
2. For each span, compute features, encoder, projector, prompt, and greedy continuation. Release temporary per-clip buffers before moving on. Trim outer whitespace and join nonempty text in order with a single separator at word boundaries. For Japanese neighbors use script-aware boundary handling to avoid injected spaces while still preserving separations around Latin words. Unit-test punctuation and mixed-script boundaries.
3. Return `TaskResult::text_output` using the normalized hint as the language when known. Surface decoder/weight failures through the current CLI/server error convention and complete a new valid request afterward on the same lease. Keep the model registered for `asr/offline` with a plain `Transcript`. Advertise VAD only when it works through the modeled options. Leave word timings, speaker labels, and streaming capability fields unset.
4. Register the new factory with the exact family ID `granite_speech_2b`. Keep all existing Granite aliases bound to their current family. Verify native auto-detection with a checkpoint directory and explicit `--family` on both supported source types. Verify registry, spec, and manager agreement in a full composite, and confirm a custom composite links only the selected families.

### D3. Runnable path verification

Once the planned targets are added, run from the repository root. Set `TURBOCTC_MODEL` to a local existing TurboCTC model path.

```bash
export GRANITE2B_BIN="$GRANITE2B_WORK/bin"
export GRANITE2B_AUDIO="$PWD/assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav"
cmake -S . -B "$GRANITE2B_WORK" -DCMAKE_BUILD_TYPE=Release \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=granite_speech_2b,granite5asr \
  -DENGINE_BUILD_TESTS=ON -DENGINE_BUILD_MODEL_TESTS=ON
cmake --build "$GRANITE2B_WORK" --parallel 8 --target audiocpp_cli audiocpp_server audiocpp_gguf \
  granite_speech_2b_unit_test granite_speech_2b_parity_probe \
  granite5asr_ctc_timestamps_test test_granite5asr_golden_transcription
ctest --test-dir "$GRANITE2B_WORK" -R '^(granite_speech_2b_unit_test|granite5asr_ctc_timestamps_test)$' \
  --output-on-failure
"$GRANITE2B_BIN/audiocpp_cli" --task asr --family granite_speech_2b \
  --model "$GRANITE2B_HF" --audio "$GRANITE2B_AUDIO" --backend cpu --threads 8 \
  --request-option language=en --text-out "$GRANITE2B_WORK/cli.txt"
"$GRANITE2B_BIN/test_granite5asr_golden_transcription" --model "$TURBOCTC_MODEL" --backend cpu
jq -n --arg path "$GRANITE2B_HF" '{
  host:"127.0.0.1", port:18080, backend:"cpu",
  models:[{id:"granite-2b",family:"granite_speech_2b",path:$path,task:"asr",mode:"offline"}]
}' > "$GRANITE2B_WORK/server.json"
"$GRANITE2B_BIN/audiocpp_server" --config "$GRANITE2B_WORK/server.json" --no-ui
```

In a second terminal with those environment variables available, compare both responses with `cli.txt` after applying the same request language. Save response bodies for failures. Also cover missing input, malformed WAV, unsupported language, and serial A/B/A requests across chunk and context boundaries.

```bash
curl --fail-with-body http://127.0.0.1:18080/v1/models
curl --fail-with-body http://127.0.0.1:18080/v1/audio/transcriptions \
  -F model=granite-2b -F language=en -F "file=@$GRANITE2B_AUDIO"
jq -n --arg path "$GRANITE2B_AUDIO" '{model:"granite-2b",audio:$path,language:"en"}' |
  curl --fail-with-body http://127.0.0.1:18080/v1/audio/transcriptions \
    -H 'Content-Type: application/json' --data-binary @-
python3 tools/audiocpp_cli/run_audiocpp_cli_path_tests.py --family granite_speech_2b \
  --backend cpu --audiocpp-cli-bin "$GRANITE2B_BIN/audiocpp_cli" --model-path "$GRANITE2B_HF"
```

Add distinct shared CLI case IDs for short English, Japanese with a hint, and fixed chunking with an input longer than the requested chunk limit. Keep the current legacy HTTP API shape. Test response language and model selection in addition to transcript text. [HTTP routes](app/server/api.md#post-v1audiotranscriptions)

## E. Build, validate, and publish a standalone GGUF

### E1. Conversion and installation

Use the repository's converter and preserve native tensor keys. Stage just the files consumed by the loader so the converter embeds the right sidecars. First produce an `orig` artifact as a format-control build. Then quantize eligible language-model matrices to Q8_0, keeping the encoder, Q-Former, embedding table and output head in BF16 for the first quality run. Let the converter retain appropriate high precision for one-dimensional tensors. If weight type changes are needed for a backend, compare them one component at a time. With `--keep-type`, use quoted patterns to avoid shell expansion and verify the actual stored types before using the file. [GGUF conversion guide](docs/gguf.md#convert-a-single-tensor-source) · [Converter flags](app/gguf/main.cpp)

```bash
GRANITE2B_STAGE="$GRANITE2B_WORK/convert-input"
mkdir -p "$GRANITE2B_STAGE" "$GRANITE2B_WORK/dist"
for f in config.json generation_config.json processor_config.json preprocessor_config.json \
  tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.jinja; do
  cp "$GRANITE2B_HF/$f" "$GRANITE2B_STAGE/$f"
done
ln -sfn "$GRANITE2B_HF/model.safetensors" "$GRANITE2B_STAGE/model.safetensors"
"$GRANITE2B_BIN/audiocpp_gguf" --family granite_speech_2b \
  --input "$GRANITE2B_STAGE/model.safetensors" --root "$GRANITE2B_STAGE" \
  --type orig --output "$GRANITE2B_WORK/dist/granite-speech-5.0-2b-orig.gguf"
"$GRANITE2B_BIN/audiocpp_gguf" --family granite_speech_2b \
  --input "$GRANITE2B_STAGE/model.safetensors" --root "$GRANITE2B_STAGE" \
  --type q8_0 --keep-type 'encoder.*=bf16' --keep-type 'projector.*=bf16' \
  --keep-type 'language_model.model.embed_tokens.weight=bf16' \
  --keep-type 'language_model.lm_head.weight=bf16' \
  --output "$GRANITE2B_WORK/dist/granite-speech-5.0-2b-q8_0.gguf"
"$GRANITE2B_BIN/audiocpp_gguf" --inspect "$GRANITE2B_WORK/dist/granite-speech-5.0-2b-q8_0.gguf"
```

Adjust the staging list and matched keys to match the output from A. Verify the embedded spec, resolved sources, all sidecar names, Q8_0/BF16 tensor totals, and layout. Test with only the GGUF and a WAV in a temporary directory outside the checkout. Use a non-deployment build and `--model-spec-override embedded` to force the embedded spec, then repeat without the override to test ordinary discovery. Compare native versus `orig` through trace and transcript before attributing divergences to quantization. [Embedded package behavior](docs/gguf.md#inspect-and-run)

The published package should be `granite_speech_2b_q8_0` if it passes the gates below, with one exact remote filename, pinned revision and recorded checksum. If the mixed artifact fails the quality gate, ship the verified higher-precision artifact under an accurately named package and keep quantization experimental. Use IBM's official `granite-speech-5.0-2b-Q8_0.gguf` plus `mmproj-model-f16.gguf` in a pinned llama.cpp build as an independent comparison. For direct IBM GGUF import, treat the two-file schema and its tensor/metadata mapping as follow-up work. The audio.cpp release uses its native standalone package. [IBM GGUF files](https://huggingface.co/ibm-granite/granite-speech-5.0-2b-GGUF/tree/main) · [IBM llama.cpp usage](https://huggingface.co/ibm-granite/granite-speech-5.0-2b#usage)

Before release, set the status, default package, tested backend tags, and docs together. Update `README.md`, `docs/asr.md`, `docs/models/granite_speech_2b.md`, `docs/model_licenses.md` (Apache 2.0), and the WebUI catalog if exposed there. Report measured hardware usage and the real maximum per-chunk length. Test install and inspect in both managers with separate empty model roots. From a full build, pipe `audiocpp_cli --list-loaders --json` into `python3 tools/check_loader_catalog_sync.py --loader-json -`. Use `python3 tools/model_manager_v2.py info granite_speech_2b`, then `install granite_speech_2b --models-root <empty-root>`. Build the native manager with `-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON` and repeat with `audiocpp_model_manager info` and `install`. Run the CLI from each installed artifact. [Package and loader synchronization](docs/maintainers/loader_and_catalog.md) · [WebUI catalog](webui/configs/models_catalog.json)

### E2. Verification matrix and exit thresholds

These are proposed engineering gates. Record any adjustment, the reason, and the observed results before promoting the model.

| Area | Cases and gate |
| --- | --- |
| Unit and shape tests | Derive exact mel, encoder, projector, prompt and cache lengths for boundaries in B2. Compare discrete IDs and masks exactly. Reject incompatible model families, bad tensor shapes, malformed tokenizer/processor assets, missing output head, bad PCM, out-of-range languages, requests outside the chunk limit, and prompts over context capacity. Include a no-weight test for every shape or option rule. |
| CPU FP32 intermediate parity | Use `max_abs`, RMS difference divided by reference RMS (with small zero guard), and worst coordinate. Start with max absolute error `2e-4` for normalized frontend outputs, normalized RMS error `1e-3` for encoder/projector activations, and logits max absolute error `2e-2` plus normalized RMS error `1e-3`. Treat nonfinite data as failure. Investigate the first failing stage and record backend and FFT sources before changing a tolerance. |
| Decoder and cache correctness | Match top-1 where the reference top-1 margin exceeds twice the allowed logit error. Require exact visible greedy sequences on the short English/Japanese stable-margin fixtures. Compare eight teacher-forced steps, one prefill spanning more than one block, a text-only probe across 4,096 positions, and the longest supported budget edge. Prove cache state resets with A/B/A, including a failed request between A and B. |
| End-to-end quality | Use IBM's sample for processor/generation smoke and the checked-in LibriSpeech and Japanese Common Voice fixtures for immediate local tests. For release, pin a held-out public evaluation manifest for each of the six ASR languages. Record source revision, clip IDs, reference text and audio checksum. Include short clips, natural clips longer than one minute where available, and recordings that require chunking. Synthetic concatenations cover joins and limits. Score word error rate on space-delimited languages, Japanese character error rate, and punctuation/casing separately on reference transcripts that preserve them. Record edit counts and denominators so small slices can be interpreted. |
| GGUF quality | Compare the native reference, GGUF `orig`, mixed Q8_0, and IBM's official Q8_0 pair using the same PCM, prompts, budgets, and scoring. Require exact stable-margin outputs for native versus `orig`. For the release Q8_0 candidate, start with an absolute limit of +1 percentage point aggregate and +2 points for any language or long-audio slice versus `orig`, and review empty, repetitive, or truncated outputs individually. Compare WER and CER as separate metrics. |
| Runtime integration | Run CLI native and embedded GGUF, legacy-server JSON and multipart, a valid request after a failure, and two separate sessions with different clips and options. Compare each result to sequential inference. Test custom/full CMake registration, existing ASR callers affected by framework changes, and TurboCTC golden and timestamp tests. Add the existing `causal_decoder_chunked_prefill_test` and `decoder_packed_projection_test` to the shared-math regression run. |
| Performance and memory | Measure CPU, then CUDA and Metal independently if those targets will be claimed. Use fixed thread/device settings. Record load time, peak RSS and VRAM, graph/weight residency, per-stage time, first-token latency, decode tokens/s, and wall time divided by audio duration. Report cold and warm runs, three warm-up requests then at least 20 alternating short/long calls on one session. Check that peak allocation settles and reused requests match fresh sessions. Compare GPU end-to-end RTF with the community guideline of below 1.0 on the declared test hardware and provide actual CPU numbers. [Community requirements](https://github.com/0xShug0/audio.cpp/issues/54) |

Use `tools/asr_wer.py` for the existing English smoke. Extend only the new evaluation harness for multilingual metrics, with pinned Unicode normalization, WER/CER rules, and a separate punctuation/case pass. Record hardware, OS, commit, CMake flags, model revision and artifact digest, input manifest, prompt text, exact run commands, reference and C++ dumps, timing rows, and known parity deviations in one validation report. [Existing WER tool](tools/asr_wer.py) · [CLI path runner](tools/audiocpp_cli/run_audiocpp_cli_path_tests.py)

## F. Add keyword biasing and speech translation

### F1. Keyword biasing

Use `TaskRequest::text_input.text` as recognition context for CLI `--text`, legacy multipart `prompt`, or JSON `text`. Treat it as a list of comma-separated terms initially. Trim entries, skip empty items, preserve UTF-8 and multiword spellings, and reject reserved control markers. Construct the task-specific keyword instruction described by IBM, with a model-owned audio marker. Use the expanded token budget to enforce a usable maximum. IBM cites up to 2,000 keywords, so test the larger list sizes separately and document where the runtime's actual prompt budget caps them. For structured API support, add a `string_list` option only when needed and define its precedence relative to existing text context. [Legacy multipart mapping](app/server/runtime.cpp#L1348) · [IBM prompting guide](https://huggingface.co/ibm-granite/granite-speech-5.0-2b#usage)

Gate the feature on paired runs for clips containing rare names and clips where the same list should produce no insertions. Include Japanese and accented Latin terms, whitespace-only input, repeated terms, a multiword term, and an over-context prompt. Report keyword precision, recall, and F1 alongside transcript WER/CER. Match reference prompt token IDs before debugging quality deltas.

### F2. Speech translation

Add a model request key `target_language` via the existing `asr/offline` contract as a first exposure. Resolve the ASR request when absent and build an explicit translation instruction when present. Keep `language` for a source hint and label output with the normalized target code. Use the verified direction matrix from IBM's evaluation and publish only tested directions. Validate English ↔ French/German/Spanish/Portuguese/Japanese as applicable to the pinned reference and evaluate English → Italian and Mandarin separately. For language pairs where the card's prompt table and evaluation claims differ, document evidence from the pinned model before advertising the direction. JSON can pass the option through `options`. The current legacy multipart parser has a fixed field list, so multipart translation would require accepting and dispatching a new field. A dedicated translation endpoint is a separate API change. [IBM model card and supported directions](https://huggingface.co/ibm-granite/granite-speech-5.0-2b) · [Legacy multipart parser](app/server/runtime.cpp#L1344)

For each advertised direction, add short and long-form fixtures with human target references. Compare the same pinned model in Python and C++, report corpus chrF or BLEU with the tokenizer and normalization settings, and examine wrong-language outputs. Exercise source-language hints, unsupported targets, ordinary ASR with no target, and keyword biasing with translation if both options are enabled together.

## Handoff checklist

- [ ] Reference revision and load manifest recorded, with a successful smoke and matching trace interfaces.
- [ ] CPU acoustic and decoder stage parity demonstrated independently and then through the normal ASR path.
- [ ] Native/staged GGUF request, error recovery, backend and shared-caller checks recorded with exact commands.
- [ ] Mixed GGUF passes quality gates or the verified higher-precision package is selected and named accordingly.
- [ ] Installer, embedded spec, docs, license entry, and release catalog identify the same tested artifact.
- [ ] Keyword biasing and translation have their own prompt, quality, and API validation before being advertised.
