# Granite Speech 2B validation

## Scope

IBM `granite-speech-5.0-2b`, revision
`ec49ca6b1a5e7e9176127e09cb303ab78cbacdf7`. Python reference: Transformers
4.57.0 and PyTorch 2.11.0+cu128, controlled FP32 eager execution.

The performance baseline is the original PR implementation rebased onto the
same main revision (`4de392e03`), versus the cleanup implementation. Main does
not contain this family, so it cannot provide a model-performance baseline.

Reference audio, tensor dumps and generated logits are external test inputs;
none are stored in the repository. See the [parity test instructions](../../tests/granite_speech5/README.md).

## Correctness

- Six model test executables pass. Coverage includes frontend lengths, encoder
  and projector math, independent projector windows, prompt/token composition,
  decoder scaling, neutral shared-module defaults, cache reuse and errors.
- Controlled English CPU, English CUDA and Japanese CUDA comparisons pass all
  19 intermediate/teacher-forced stages. English CUDA injects the Python
  projected embeddings at the AR boundary. Generated token IDs and text match.
- CUDA controls disable TF32 and use eager attention; production does neither.
  Teacher-forced logit maximum errors are below `4.6e-5`. Independent YaRN checks
  include positions 4095, 4096, 4097 and 16383.
- Production English/Japanese outputs match the original PR across all measured
  backends. CUDA and Vulkan also match through fixed chunking, repeated
  requests, short/long/short shape reuse and recovery after an invalid language.
- Mixed Q8 loads and produces matching English/Japanese transcripts on CUDA,
  Vulkan and CPU. Standalone GGUF auto-detection also works outside the checkout.
  All 16 integer BatchNorm counters remain unchanged; no tensor-source changes
  are needed for conversion.

Strict eager-reference logit parity is **not** established for production flash
attention: its raw LM-head maximum error was 0.06055 against the 0.02 gate
(relative RMS `2.57e-4`), despite matching checked transcripts. Gates were not
relaxed. Controlled mathematical parity and production transcript checks are
separate evidence.

## Warmed performance

RTX 5090, Ryzen 9 7900X, eight threads, Debug build, native BF16 checkpoint.
Input: 14.225 seconds of English LibriSpeech. Sequential A→B→B→A runs; each
server receives an unrelated warmup and one first-use request before three
measured requests. RTF uses canonical session wall time; loading and warmup
are excluded. Memory is the peak for the server process, including first use.

| Backend | Original RTF | Cleanup RTF | Time reduction | Peak VRAM, original → cleanup |
|---|---:|---:|---:|---:|
| CUDA | 0.02218 | 0.01810 | 18.4% | 6,148 → 5,424 MiB |
| Vulkan | 0.02646 | 0.01968 | 25.6% | 5,580 → 5,121 MiB |
| CPU | 0.55969 | 0.35374 | 36.8% | N/A |

CPU peak process RAM is **6,138 → 5,424 MiB**. Results are medians of six
warmed requests per implementation and backend. Fixed-chunk and longer-input
checks validate reuse; they are not mixed into this timing table.

The cleanup batches independent Q-Former windows, reuses session-owned graphs
and weight bindings, removes normal-path intermediate captures, and uses the
shared greedy decoder directly. Vulkan retains manual prefill after flash
prefill caused device loss; single-token decoding uses the existing flash path.
The adjusted Vulkan path passes both the lifecycle sequence and repeated
benchmark runs. No GGML/kernel changes are included.

## Existing callers and limits

Default CUDA sessions were also compared with actual main: Qwen3-ASR transcripts
and Audio8 TTS WAVs match exactly across changed-length/repeated requests.
Warmed times were 70.08 → 69.88 ms and 635.67 → 634.56 ms, respectively;
peak VRAM was 3,198 → 3,198 MiB and 2,872 → 2,866 MiB. This is targeted
existing-caller coverage, not an exhaustive audit of every decoder consumer.

The mixed-Q8 catalog URL still returns 404; uploading the validated package
remains a release prerequisite. Metal, natural long-form quality across all
supported languages, translation quality and VAD integration were not newly
validated. The family is offline-only, without timestamps or speaker labels.
