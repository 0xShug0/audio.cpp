# Fish Audio S2 Pro - Official GGUF Quantization Recipe (Q6_K & Q4_K)

> **Status**: Verified & Production-Ready  
> **Target Engine**: `audio.cpp` C++ GGML runtime  
> **Tested Models**: `Q6_K` & `Q4_K` Mixed-Precision  
> **Memory Savings**: Up to **1.82 GB VRAM saved** (4.50 GB down from 6.32 GB `Q8_0`)  
> **Audio Quality**: 100% crystal clear speech synthesis and zero-shot voice cloning with clean `<|im_end|>` termination.

---

## 1. Background & The Long-Standing Community Roadblock

Until now, the consensus in the open-source audio community was that Fish Audio S2 Pro could only run reliably in `Q8_0`, and that lower quantizations (`Q6_K`, `Q4_K`) suffered from catastrophic degradation: producing abrasive looping static, missing sentence-boundary `<|im_end|>` stop tokens, and running until hard 47.5-second generation timeouts.

Through forensic disassembly of the GGUF container format and `audio.cpp` engine internals, we uncovered the **two root causes** that caused every previous quantization attempt to fail:

### Root Cause 1: Silent Namespace Override Failures in `audiocpp_gguf.exe`
Fish Audio S2 Pro is a **Dual-AR architecture**:
1. **Slow AR Backbone (36 layers, 2560 dim)**: Processes text prompts and speaker reference codes, emitting high-level semantic tokens.
2. **Fast AR Decoder (`fast_layers`, 4 layers + `fast_output`)**: Autoregressively generates **7 acoustic codebooks** per semantic step to drive the audio codec.

In `audio.cpp`, tensors are strictly organized under namespaces (`model_weights/` and `codec_weights/`). The GGUF converter (`audiocpp_gguf.exe`) matches type overrides using exact string prefixes (`name.substr(0, pattern.size()) == pattern`). When quantizing with `--keep-type 'fast_layers*=orig'`, **zero tensors matched** because the actual tensor names are `model_weights/fast_layers...`.

Consequently:
* The 4-layer fast acoustic decoder was completely crushed into low-bit integers, corrupting the codebook predictions on every single audio frame from step 1.
* By adding the `model_weights/` prefix (`--keep-type 'model_weights/fast_*=orig'`), the fast acoustic decoder stays at original precision, preserving crystal clear audio synthesis.

### Root Cause 2: Corrupted Token Metadata in Extracted Sidecars
`audio.cpp` enforces semantic token filtering during sampling via `apply_semantic_bias`. It sets all logits outside `[semantic_start_token_id, semantic_end_token_id]` and `eos_token_id` to $-\infty$.

In broken distributions, `config.json` contained bogus token ranges (`100000`–`102047`, which map to random Chinese characters). This mathematically barred the model from emitting valid acoustic codebook indices (`151678`–`155773`) or the real `<|im_end|>` token (`151645`), trapping the generator in an infinite timeout loop.

---

## 2. Prerequisites

Ensure the following tools and assets are available:
* **CUDA Toolkit**: v12.x or v13.x installed.
* **`audio.cpp` Windows CUDA build**:
  - `audiocpp_gguf.exe` (located in `build/windows-cuda-release/bin/`)
  - `audiocpp_cli.exe` or `audiocpp_server.exe`
* **Node.js**: v18+ (for sidecar extraction).
* **Base Model**: Official `fish-audio-s2-pro-q8_0.gguf` downloaded into `models/Fish-Audio-S2-Pro-GGUF/` (from [audio-cpp/audio.cpp-gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf)).

---

## 3. Step-by-Step Quantization Recipe

### Step 1: Extract Authentic Metadata Sidecars Directly from Q8_0
To guarantee 100% authentic token configurations, extract the embedded JSON sidecars directly from the binary payload of `fish-audio-s2-pro-q8_0.gguf`:

```javascript
// extract_sidecars.js
const fs = require('fs');
const fd = fs.openSync('models/Fish-Audio-S2-Pro-GGUF/fish-audio-s2-pro-q8_0.gguf', 'r');
let pos = 0;
function readBytes(len) { const b = Buffer.alloc(len); fs.readSync(fd, b, 0, len, pos); pos += len; return b; }
function readU32() { const b = readBytes(4); return b.readUInt32LE(0); }
function readU64() { const b = readBytes(8); return Number(b.readBigUInt64LE(0)); }
function readStr() { const len = readU64(); const b = readBytes(len); return b.toString('utf8'); }

readBytes(4); // magic
readU32();    // version
readU64();    // tensorCount
const kvCount = readU64();

function skipValue(valType) {
  if (valType <= 1) pos += 1;
  else if (valType <= 3) pos += 2;
  else if (valType <= 6) pos += 4;
  else if (valType === 7) pos += 1;
  else if (valType === 8) { const len = readU64(); pos += len; }
  else if (valType === 9) {
    const elemType = readU32();
    const arrLen = readU64();
    if (elemType <= 7 || elemType >= 10) {
      const sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8};
      pos += arrLen * sizes[elemType];
    } else if (elemType === 8) {
      for (let i = 0; i < arrLen; i++) { const l = readU64(); pos += l; }
    }
  } else if (valType >= 10) pos += 8;
}

let names = [], offsets = [], data = null;
for (let i = 0; i < kvCount; i++) {
  const key = readStr();
  const valType = readU32();
  if (key === 'audiocpp.embedded_files.names') {
    const elemType = readU32();
    const arrLen = readU64();
    for (let j = 0; j < arrLen; j++) names.push(readStr());
  } else if (key === 'audiocpp.embedded_files.offsets') {
    const elemType = readU32();
    const arrLen = readU64();
    for (let j = 0; j < arrLen; j++) offsets.push(readU64());
  } else if (key === 'audiocpp.embedded_files.data') {
    const elemType = readU32();
    const arrLen = readU64();
    data = readBytes(arrLen);
  } else {
    skipValue(valType);
  }
}

for (let i = 0; i < names.length; i++) {
  if (names[i].endsWith('.json')) {
    const fileBuf = data.subarray(offsets[i], offsets[i+1]);
    fs.writeFileSync('models/Fish-Audio-S2-Pro-GGUF/' + names[i], fileBuf);
    console.log(`Extracted authentic sidecar: ${names[i]} (${fileBuf.length} bytes)`);
  }
}
```

Run extraction:
```bash
node extract_sidecars.js
```

Verify that `models/Fish-Audio-S2-Pro-GGUF/config.json` contains:
```json
"audio_pad_token_id": 151677,
"eos_token_id": 151645,
"pad_token_id": 151669,
"semantic_end_token_id": 155773,
"semantic_start_token_id": 151678
```

---

### Step 2: Execute Mixed-Precision Quantization

#### Option A: Build `Q6_K` (5.44 GB, Saves ~880 MB VRAM)
```powershell
.\build\windows-cuda-release\bin\audiocpp_gguf.exe `
  --input "models\Fish-Audio-S2-Pro-GGUF\fish-audio-s2-pro-q8_0.gguf" `
  --output "models\Fish-Audio-S2-Pro-GGUF\fish-audio-s2-pro-q6_k.gguf" `
  --type q6_k `
  --keep-type 'model_weights/fast_*=orig' `
  --keep-type 'model_weights/codebook_embeddings*=orig' `
  --keep-type 'model_weights/embeddings*=orig' `
  --keep-type 'model_weights/norm*=orig' `
  --keep-type 'codec_weights*=orig' `
  --family fish_audio `
  --sidecar "models/Fish-Audio-S2-Pro-GGUF/config.json=config.json" `
  --sidecar "models/Fish-Audio-S2-Pro-GGUF/tokenizer_config.json=tokenizer_config.json" `
  --sidecar "models/Fish-Audio-S2-Pro-GGUF/tokenizer.json=tokenizer.json" `
  --sidecar "models/Fish-Audio-S2-Pro-GGUF/special_tokens_map.json=special_tokens_map.json" `
  --allow-missing-model-spec `
  --overwrite
```

#### Option B: Build `Q4_K` (4.50 GB, Saves ~1.82 GB VRAM)
```powershell
.\build\windows-cuda-release\bin\audiocpp_gguf.exe `
  --input "models\Fish-Audio-S2-Pro-GGUF\fish-audio-s2-pro-q8_0.gguf" `
  --output "models\Fish-Audio-S2-Pro-GGUF\fish-audio-s2-pro-q4_k.gguf" `
  --type q4_k `
  --keep-type 'model_weights/fast_*=orig' `
  --keep-type 'model_weights/codebook_embeddings*=orig' `
  --keep-type 'model_weights/embeddings*=orig' `
  --keep-type 'model_weights/norm*=orig' `
  --keep-type 'codec_weights*=orig' `
  --family fish_audio `
  --sidecar "models/Fish-Audio-S2-Pro-GGUF/config.json=config.json" `
  --sidecar "models/Fish-Audio-S2-Pro-GGUF/tokenizer_config.json=tokenizer_config.json" `
  --sidecar "models/Fish-Audio-S2-Pro-GGUF/tokenizer.json=tokenizer.json" `
  --sidecar "models/Fish-Audio-S2-Pro-GGUF/special_tokens_map.json=special_tokens_map.json" `
  --allow-missing-model-spec `
  --overwrite
```

#### Tensor Breakdown:
* **Total Tensors**: 939
* **Preserved at Native / Orig (543 tensors)**:
  - 501 `codec_weights` (acoustic VQ codec & decoder)
  - 36 `fast_layers` (4 layers × 9 projections of the fast acoustic generator)
  - 1 `fast_output` projection
  - 1 `fast_embeddings`
  - 2 Text & Codebook embedding matrices
  - 2 Final RMSNorm scale tensors
* **Quantized (396 tensors)**:
  - 36 transformer backbone layers (`layers.0` through `layers.35`).

---

## 4. Benchmark & Performance Matrix

Tested on NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM) with CUDA graphs enabled:

| Quantization | File / VRAM Size | Savings vs Q8_0 | Peak RTF | Avg RTF | Real-Time Speed | Audio Quality |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Official `Q8_0`** | 6.32 GB | Baseline | ~1.20 | ~1.25 | ~0.80x | Pristine |
| **Recipe `Q6_K`** | 5.44 GB | **-880 MB** | ~1.10 | ~1.15 | ~0.87x | Pristine |
| **Recipe `Q4_K`** | 4.50 GB | **-1.82 GB** | **1.05** | ~1.14 - 1.18 | **~0.88x - 0.95x** | Pristine |

### Analysis:
* **VRAM Benefit**: `Q4_K` cuts model footprint from 6.32 GB down to 4.50 GB, allowing Fish Audio S2 Pro to fit comfortably into 6 GB and 8 GB consumer GPUs alongside other services.
* **Latency / RTF**: Because Fish Audio S2 Pro is dual-autoregressive with 7 fast codebook forward passes per token, compute latency scales with decoding steps rather than purely memory bandwidth. Real-time factor (RTF) remains around **~1.05–1.18** even at Q4_K.

---

## 5. Using with `airi-audio-server`

To use the new `Q4_K` or `Q6_K` model in `airi-audio-server`, simply update your `config.json`:

```json
{
  "tts_model": {
    "id": "fish-audio-tts",
    "family": "fish_audio",
    "model_path": "models/Fish-Audio-S2-Pro-GGUF/fish-audio-s2-pro-q4_k.gguf",
    "working_dir": "../audio.cpp"
  }
}
```
