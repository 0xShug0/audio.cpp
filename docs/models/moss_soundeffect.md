# MOSS-SoundEffect v2

MOSS-SoundEffect v2 generates sound effects from English or Chinese descriptions.
It uses a Qwen3 text encoder, an audio diffusion transformer, and a continuous DAC
decoder. It produces mono audio at 48 kHz. This is a separate architecture from
MOSS-TTS and MOSS-SoundEffect v1.

## Quick Start

```bash
audiocpp_cli --model MOSS-SoundEffect-v2-GGUF/moss-soundeffect-v2-q8_0.gguf \
  --family moss_soundeffect --task gen --backend cuda \
  --text "Steady rain pattering on a wooden roof, with distant thunder." \
  --request-option duration_sec=10 --out rain.wav --log
```

Download a package from
[audio-cpp/audio.cpp-gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/MOSS-SoundEffect-v2-GGUF).

The GGUF contains all three components, tokenizer files, configuration, and the
v1 model contract. Python is needed only for checkpoint conversion, not inference.
The upstream model supports offline generation; this family does not advertise
streaming.

## Model

| Field | Value |
|---|---|
| Family | `moss_soundeffect` |
| Task | `sfx` (`gen` route) |
| Mode | `offline` |
| Prompt languages | English, Chinese |
| Output | 48 kHz mono audio |
| Packages | Q8_0, BF16 |

## Request Options

Pass these through `--request-option name=value`.

| Option | Default | Meaning |
|---|---|---|
| `duration_sec` | `10` | Output duration, 0.1–30 seconds, rounded to one decimal. |
| `num_steps` | `100` | Number of Euler denoising steps. |
| `guidance_scale` | `4` | Classifier-free guidance; `1` uses only the positive branch. |
| `sigma_shift` | `5` | Positive flow-schedule shift. |
| `seed` | `0` | Initial-noise seed. |
| `negative_prompt` | Empty | Negative sound description. |
| `append_duration_suffix` | `true` | Append the trained duration instruction to the prompt. |

Denoising retains the trained 30-second latent window. CFG and Euler updates
keep their latent state on the backend between steps. The DAC decoder processes
only the requested output span in bounded tiles, including the convolution
context required from the surrounding latent window. Shorter requests reduce
decoder work, but do not shorten the DiT canvas. Text conditioning is limited
to 512 tokens; its hidden-state graphs use cached token-length buckets and
process only the required prefix plus bucket padding. Seed values
are reproducible within the runtime; audio.cpp and PyTorch use different random
number generators, so equal seeds do not imply identical starting noise.

## Session Options

| Option | Default | Meaning |
|---|---|---|
| `moss_soundeffect.fused_epilogues` | `true` | Enable the denoiser runtime's normalization-affine, bias-activation and gated-residual fusions. Set `false` to use ordinary kernels. |
| `moss_soundeffect.attention` | `auto` | DiT attention: `auto`, `flash`, or `eager`. Eager retains F32 attention arithmetic and uses more workspace. |
| `moss_soundeffect.weight_type` | `native` | Storage override for text-encoder and DiT weights. The DAC decoder retains its packaged precision. |

Set session options with `--session-option name=value`. The large inference
graphs and projected text-conditioning caches belong to the session and are
reused across requests. Each request starts from fresh noise and conditioning.
On CUDA with native weight storage, BF16 DiT packages use the existing FP16 projection path for the FFN
intermediates; attention remains standard flash attention, and modulation and
residual arithmetic remain F32. The text encoder and DAC retain their packaged
precision. Explicit weight-storage overrides retain their requested type.

With native BF16 packages, Vulkan retains BF16 projection operands and uses
GGML's existing BF16 matrix kernels with F32 accumulation. Matrix-core support
requires both a compatible GPU and a shader compiler that supports the relevant
extensions. Use a recent Vulkan SDK's `glslc` when configuring the build; older
shader compilers can leave these projections on substantially slower scalar
kernels even when the GPU supports matrix cores.

Prompts use Unicode width, quote, HTML, whitespace, and NFC normalization.
Malformed UTF-8 is rejected; Python’s repair of legacy-encoding mojibake is not implemented.

## Conversion

Download the [official checkpoint](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0)
at revision `e35df4d82fbe87fcd5d14e5d100e349c0c3c076d`, then run:

```bash
python tools/models/moss_soundeffect/convert_gguf.py \
  --model-dir MOSS-SoundEffect-v2.0 \
  --output-dir MOSS-SoundEffect-v2-GGUF --type q8_0
```

Use `--type bf16` for the BF16 package. Both published packages follow upstream
BF16 text encoding and projection autocast, with F32 timestep embedding,
modulation, normalization, guidance accumulation, and DAC decoding. Conversion folds DAC weight normalization
and keeps the decoder in F32, matching the upstream decoder's compute path.
Only the decoder portion of the DAC checkpoint is included.

## References and License

- [Upstream implementation](https://github.com/OpenMOSS/MOSS-TTS/tree/4cf2dab88fc8711aef93089b3b75c0ae07373e47/moss_soundeffect_v2)
- [Official model](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0)
- The upstream repository and checkpoint are distributed under Apache-2.0.
