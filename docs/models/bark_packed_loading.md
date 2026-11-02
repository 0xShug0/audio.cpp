# Original compressed Bark loading

Standard and Plus can load their original `bark-levels-v3` checkpoint directly
through the native Parakeet community loader. The 163/180 MB `model.bark` stays
compressed on disk; C++ decodes one tensor at a time and uploads ordinary native
weights. Inference needs no Python, expanded checkpoint, or temporary GGUF.
RAM/VRAM still includes expanded weights and inference buffers. This is not
packed GPU execution. Pro uses ordinary BF16 safetensors and keeps its GGUF path.

This route builds on the merged Pro runtime and is independent of the separate
Standard/Plus GGUF-conversion ports. It reuses the publisher's 250 ms prefix,
full-context default, timestamps, offline VAD, native hotwords and buffered
streaming. English only; streaming is not native cache-aware inference.

## Download and prepare

Download only the pinned weights, metadata and licenses. Preparation needs
Python/numpy for existing metadata helpers, but no PyTorch, constriction, Rust
or GGUF tool. It never executes the publisher's Python loader.

```bash
hf download ConwayResearch/Underdog-Bark-3.0 \
  --revision 85db2ad18ab41e534a222ee253c29191d77ed919 \
  --include model.bark mlx/config.json NOTICE README.md coreml/LICENSE \
  --local-dir models/Underdog-Bark-3.0-source
hf download nvidia/parakeet-tdt-0.6b-v3 \
  --revision 541d1f99c6b0c3cd0b11a95167540bb8edefd82b \
  --include config.json processor_config.json tokenizer.json tokenizer_config.json generation_config.json \
  --local-dir models/Underdog-Bark-3.0-source
python tools/community_models/prepare_bark_packed.py --variant standard \
  --source models/Underdog-Bark-3.0-source --output models/Underdog-Bark-3.0-Native

hf download ConwayResearch/Underdog-Bark-3.0-Plus \
  --revision d4fab2f8b95681e5ad3cfb4e8c5f4e0c009b3deb \
  --include model.bark mlx/config.json NOTICE README.md coreml/LICENSE \
  --local-dir models/Underdog-Bark-3.0-Plus-source
hf download nvidia/parakeet-tdt-0.6b-v3 \
  --revision 541d1f99c6b0c3cd0b11a95167540bb8edefd82b \
  --include config.json processor_config.json tokenizer.json tokenizer_config.json generation_config.json \
  --local-dir models/Underdog-Bark-3.0-Plus-source
python tools/community_models/prepare_bark_packed.py --variant plus \
  --source models/Underdog-Bark-3.0-Plus-source --output models/Underdog-Bark-3.0-Plus-Native
```

Each package retains the unchanged checkpoint, config, tokenizer, English spec
v1, conversion/source provenance, NOTICE and full CC-BY-4.0 license. An existing
output directory is never overwritten. No unpublished model-manager download is
advertised. Keep the package files together.

## CLI and server

Follow the [Parakeet/Bark build instructions](underdog_bark3.md#convert), building
`audiocpp_cli` and `audiocpp_server` with the `parakeet_tdt` model set. The optional
validation targets are `parakeet_warm_bench`, `bark_packed_probe` and
`bark_packed_test`. CPU-only builds disable both GPU backend flags.

```bash
build/bin/audiocpp_cli --family parakeet_tdt --task asr \
  --model models/Underdog-Bark-3.0-Native --audio sample.wav \
  --backend cuda --threads 8 --text-out transcript.txt --words-out words.json
```

Use the Plus package path for Plus. Passing the `model.bark` file itself also
works with these sidecars next to it. Do not pass `--model-spec-override embedded`;
the spec is the adjacent JSON file. Server entries use the same model path,
`family: parakeet_tdt`, `task: asr` and offline or streaming mode. Existing GGUF
routes are retained. CPU, CUDA and Vulkan use the same compressed reader.
Native weight storage defaults retain decoded values; do not override weight
precision when checking exact source fidelity.

Preparation checks the pinned checkpoint and base-sidecar SHA256 hashes and
vocabulary/config consistency. Loading bounds shape/lengths, rejects duplicate
names, validates entropy metadata/XZ checksums/finite values and checks package
configuration against the container. The compressed buffer is released after
upload; a new session may reread it. Do not replace loaded weight files.

## Validation

```bash
ctest --test-dir build -R bark_packed_test --output-on-failure
python -m unittest discover -s tests/parakeet_tdt -p test_bark_packed_package.py
build/bin/bark_packed_probe models/Underdog-Bark-3.0-source/model.bark tensor-dump
python tests/parakeet_tdt/validate_bark3.py --bin-dir build/bin \
  --model models/Underdog-Bark-3.0-Native --backend cuda \
  --reference original-results.json --output-dir outputs/bark-native-parity
```

The probe's `.bin`/`.json` dump is optional parity-test output; inference does not
create it. Both original checkpoints were checked across all 699 inference
tensors against the independent official/Python reconstruction. CPU, CUDA and
Vulkan are validated; Metal/HIP are not runtime-validated.

[Storage, timing, memory and validation report](../validation/bark_packed_loading.md).
