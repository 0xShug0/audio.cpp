# F5-TTS component checks

Build only the required optional targets in the existing Debug tree:

```sh
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug -DENGINE_BUILD_MODEL_TESTS=ON
cmake --build build/debug --parallel 12 --target f5_parity f5_cfg_parity f5_tokenizer audiocpp_cli
```

## Independent reference generation

Use a clean checkout of the official [F5-TTS](https://github.com/SWivid/F5-TTS)
repository at `600e5fdfbf319f84735557aca80a7aaf60d487e6`. Install its dependencies
in the model environment. `reference.py` checks the revision and imported source
location before running the official PyTorch DiT; it does not reproduce the C++
network in Python. It expects the Habibi/F5-v1 architecture used by these probes
(1024 hidden, 22 layers, 16 heads, 2731 embedding rows), an original EMA
safetensors checkpoint, and its matching `vocab.txt`.

```sh
conda run -n qwen3-tts python tests/f5_tts/reference.py \
  --upstream /path/to/F5-TTS --checkpoint /path/to/model.safetensors \
  --vocab /path/to/vocab.txt --output /path/to/external/f5-reference --device cpu
```

This saves one controlled FP32 input, conditional/final and batched-CFG outputs,
and tokenizer cases. The final-output probe reads token IDs stored as F32;
the CFG probe reads the same IDs stored as I32. Randomness is fixed at seed 42;
CUDA TF32 is disabled in the reference. A manifest records checkpoint/vocabulary
hashes, upstream revision, Torch version, precision, device and output hashes.
Use `--tokens-only` when only prompt construction changed; it does not load or
require a DiT checkpoint. A small explicit vocabulary can exercise those
formatting branches without model weights. Generated tensors remain outside the source tree.

The tokenizer fixture uses explicit expected assembled strings for an empty
reference, an ASCII-ended reference, an Arabic-ended reference and an unknown
character/dialect. Official `list_str_to_idx` produces the expected IDs.

The generator supplies no calibrated amplitude tolerance. Both velocity probes
therefore require `--max-abs-error`: a finite, non-negative absolute-error bound
chosen from reviewed evidence for the checkpoint, backend and precision being
validated. There is no default. Record its provenance alongside the saved log;
do not select a bound merely to make the candidate pass. The same bound applies
to both CFG halves. Cosine must also remain at least 0.999; non-finite values,
empty outputs and shape mismatches fail.

```sh
# Set this to an evidence-backed bound before running either velocity probe.
: "${F5_MAX_ABS_ERROR:?set a reviewed absolute-error acceptance bound}"
conda run --no-capture-output -n qwen3-tts build/debug/bin/f5_tokenizer \
  /path/to/external/f5-reference/tokens /path/to/model.safetensors --log \
  > f5-tokenizer.log 2>&1
F5_CUDA=1 conda run --no-capture-output -n qwen3-tts build/debug/bin/f5_parity \
  /path/to/model.safetensors /path/to/external/f5-reference \
  --max-abs-error "$F5_MAX_ABS_ERROR" 0 --log > f5-final.log 2>&1
F5_CUDA=1 conda run --no-capture-output -n qwen3-tts build/debug/bin/f5_cfg_parity \
  /path/to/model.safetensors /path/to/external/f5-reference \
  --max-abs-error "$F5_MAX_ABS_ERROR" 0 --log > f5-cfg.log 2>&1
```

Run commands sequentially. The checkpoint's parent directory must contain its
`vocab.txt`. CPU is the probes' default; `F5_CUDA=1` selects CUDA and the optional
positional device defaults to zero. Each probe logs `session.wall_ms` on success.
Final-forward, batched conditional/null CFG, and tokenizer checks retain their
independent upstream oracles. Historical unwired intermediate taps are removed.

## End-to-end and long text through the production CLI

Use the normal session path with a model directory containing the original
`model_200000.safetensors` and `vocab.txt`, plus an explicit Vocos checkpoint:

```sh
conda run --no-capture-output -n qwen3-tts build/debug/bin/audiocpp_cli \
  --task tts --family f5_tts --model /path/to/f5-model-directory \
  --backend cuda --threads 8 --voice-ref /path/to/reference.wav \
  --request-option 'reference_text=Transcript of the reference recording' \
  --request-option dialect=UNK --request-option num_inference_steps=16 \
  --request-option guidance_scale=2 --request-option seed=42 \
  --session-option f5_tts.vocos_path=/path/to/vocos.safetensors \
  --text 'أهلا، هذا نص عربي تجريبي.' --out f5-output.wav \
  --log --log-file f5-cli.log > f5-cli-console.log 2>&1

conda run --no-capture-output -n qwen3-tts build/debug/bin/audiocpp_cli \
  --task tts --family f5_tts --model /path/to/f5-model-directory \
  --backend cuda --threads 8 --voice-ref /path/to/reference.wav \
  --request-option 'reference_text=Transcript of the reference recording' \
  --request-option dialect=UNK --request-option num_inference_steps=16 \
  --request-option guidance_scale=2 --request-option seed=42 \
  --session-option f5_tts.vocos_path=/path/to/vocos.safetensors \
  --session-option f5_tts.frame_budget=512 \
  --text "$(cat /path/to/long-arabic-text.txt)" --out f5-long.wav \
  --log --log-file f5-long.log > f5-long-console.log 2>&1
```

`reference_text`, `dialect`, `num_inference_steps`, `guidance_scale` and `seed`
are request options; `f5_tts.vocos_path` and `f5_tts.frame_budget` are session
options declared in `model_specs/f5_tts.json`. Long text is chunked automatically
according to the frame budget; supply enough text to exceed one pass and inspect
the saved log for chunking. Use a short reference and its matching transcript
that fit the selected budget. These commands exercise synthesis and session
integration; listening or generating a WAV alone does not establish waveform
parity or transcription accuracy.
