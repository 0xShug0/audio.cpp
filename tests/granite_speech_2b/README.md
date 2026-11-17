# granite_speech_2b reference freeze (Milestone A)

Pinned reference tooling for IBM Granite Speech 5.0 2B
(`ibm-granite/granite-speech-5.0-2b`, autoregressive Conformer + Q-Former +
Granite LLM). This directory owns the Milestone A scaffold; milestones B/C
parallelize against the traces produced here.

## Pinned reference

- Revision: `ec49ca6b1a5e7e9176127e09cb303ab78cbacdf7`
- `model.safetensors` SHA-256:
  `4151269881d2dd825d45ca75d6552b2a20e6e39d0934ed84c4db826af8bf6d88`
- Python 3.11/3.12; `torch==2.8.0 torchaudio==2.8.0` (CPU index
  `https://download.pytorch.org/whl/cpu`), `transformers==4.57.0`,
  `safetensors`, `soundfile`.
- 13 HF files: `config.json generation_config.json processor_config.json
  preprocessor_config.json tokenizer.json tokenizer_config.json
  special_tokens_map.json added_tokens.json chat_template.jinja vocab.json
  merges.txt model.safetensors multilingual_sample.wav`.
- Reference run: CPU, FP32, `model.eval()`, `attn_implementation="eager"`,
  `torch.set_num_threads(8)`, prompt
  `<|audio|>transcribe the speech with proper punctuation and
  capitalization.` via `apply_chat_template` (single user turn + generation
  prompt), `generate(max_new_tokens=512, do_sample=False, num_beams=1)`.

Work dirs: `GRANITE2B_WORK=$PWD/build/granite-2b-handoff`,
`GRANITE2B_HF=$GRANITE2B_WORK/hf`.

## `out_llm.safetensors` note

The sibling file `out_llm.safetensors` (single F32 head `weight [100353,
1024]` + `bias [100353]`, resolved remotely via header range request) is a
CTC/BPE classification head over LLM hidden states. It is **not** referenced
by `transformers` 4.57.0 `modeling_granite_speech.py`, never appears in
`output_loading_info` (clean load: no missing/unexpected/mismatched keys, no
errors), and ordinary `generate()` never opens it. It is excluded from the
13-file download and must NOT be packaged unless a later task proves the
reference generation path uses it.

## Inventory

`$GRANITE2B_WORK/inventory/tensors.csv`: 954 sorted rows
(name, source file, PyTorch shape, dtype, byte range) from the safetensors
header (first 8 bytes u64-LE length + JSON). Verified: 40 text layers, 16
encoder layers, 2 Q-Former layers, both CTC feedback projections
(`encoder.out`, `encoder.out_mid`).

## Running the reference

```bash
. "$GRANITE2B_WORK/venv/bin/activate"
python tests/granite_speech_2b/reference.py --model "$GRANITE2B_HF" \
  --audio "$GRANITE2B_FIXTURE" --out-dir "$GRANITE2B_WORK/trace/<name>-hf" \
  [--language CODE] [--max-tokens N] [--force-ids FILE] \
  [--fixture-id ID --dataset NAME --license NAME --human-ref-txt FILE]
```

`--language` (`en/fr/de/es/pt/ja`) prepends the IBM source-language
instruction (`The speech is in <Name>. Transcribe it exactly; do not
translate.`) to the default prompt; without it the prompt is identical to the
smoke. `--force-ids` takes a JSON list of token IDs as the teacher-forced
continuation (default: this run's greedy IDs, observed stop included).

Each out-dir holds `trace.json` (revision, hashes, runtime versions,
backend/dtype, lengths, sample rate, UTF-8 prompt, input IDs, generated IDs
including the observed stop, audio span indices, model config values,
emitted text, tensor file names/shapes, `through: generate`), `*.f32`
dumps (mel, stacked, input proj, encoder block 1, layer 8 pre/post feedback,
last encoder layer, projected vectors, merged/scaled prefill, last-position
logits, first 8 forced cached steps), and a one-row
`reference_manifest.jsonl`.

First traces (all default prompt, `max_tokens=512`):

| out-dir | fixture | language |
| --- | --- | --- |
| `trace/multilingual-hf` | HF `multilingual_sample.wav` | (none) |
| `trace/librispeech-0000-hf` | `assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav` (CC BY 4.0, en) | (none) |
| `trace/ja-hf` | `assets/asr_validation/common_voice_ja/common_voice_ja_20461197.wav` (CC0 1.0, ja) | (none) |

## Comparing a candidate trace

```bash
python tests/granite_speech_2b/compare_trace.py \
  --reference "$GRANITE2B_WORK/trace/ja-hf" \
  --candidate "$GRANITE2B_WORK/trace/ja-cpp"
```

The candidate's `trace.json` declares its `through` stage as the cutoff.
Every entry through that cutoff is required with an exactly matching shape;
comparison stops at the first mismatch reporting shape, `max_abs`,
`rel_rms`, and worst coordinate. Exit 0 iff all stages through the cutoff
pass.

## Future C++ probe interface (definition only, NOT implemented)

Milestones B/C implement `granite_speech_2b_parity_probe` satisfying this
contract (documented here so both tracks build against the same surface):

```bash
granite_speech_2b_parity_probe --model DIR --audio WAV \
  --backend cpu --threads 8 \
  --through frontend|encoder|projector|prefill|generate \
  --out-dir DIR [--projector-input FILE]
```

- `--model`: native checkpoint dir (same 13 files) or staged equivalent.
- `--audio`: 16 kHz mono WAV clip.
- `--backend`/`--threads`: execution backend and CPU thread count.
- `--through`: pipeline cutoff; the probe must emit every trace entry
  through that stage and declare it as `"through"` in its `trace.json`:
  - `frontend`: `mel.f32`, `stacked.f32`
  - `encoder`: + `input_proj.f32`, `encoder_block1.f32`,
    `encoder_layer8_pre_feedback.f32`,
    `encoder_layer8_post_feedback.f32`, `encoder_last.f32`
  - `projector`: + `projected.f32`
  - `prefill`: + `merged_prefill.f32`, `scaled_prefill.f32`,
    `last_pos_logits.f32`
  - `generate`: + `forced_step{k}.f32` (teacher-forced on the reference
    continuation; obtain IDs from the reference trace's `generated_ids` or
    `--force-ids`), full `trace.json` with emitted text.
- `--projector-input FILE`: optional direct projector fixture (an
  `encoder_last`-shaped `*.f32`) letting decoder work (C) proceed
  independently of the encoder (B); when given, `--audio` still provides
  span/length metadata and `--through` is limited to
  `projector|prefill|generate`.
- Output `trace.json` uses the same keys as the Python reference
  (`revision`, hashes, `runtime` versions, `backend`, `dtype`, lengths,
  prompt, IDs, text, tensor file names/shapes). `compare_trace.py` is the
  acceptance gate; stage order and thresholds are authoritative from the
  reference `trace.json`. The candidate `trace.json` must also carry the
  same `cutoffs` phase→stage map so `through` resolves to a stage cutoff.
