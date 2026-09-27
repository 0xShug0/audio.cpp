# Higgs v3 CUDA fixed-seed cold/warm repeatability

Windows/MSVC, RTX 3090 (24 GiB), 2026-09-27. Current source: PR #706
commit `f69d0f92fd06fe722aef58885ce25d985a3deab2`. CUDA device 0, eight CPU
threads, Higgs Audio v3 TTS 4B Q8_0, offline TTS, `max_tokens=1024`.

**Result:** the cold/warm difference reproduces in both the current and saved
older builds. For these fixtures, parallel requests match their corresponding
serial cold or warm references. A diagnostic full-prefill control removes the
difference, isolating reference KV prefix reuse and its different prefill
computation as the trigger. No production inference change is proposed here.

## Fixtures and method

- Prompt A: `This is a test of two independent audio requests.` Seed: `1234`.
- Prompt B: `A second request should keep its own voice and seed.` Seed: `5678`.
- Reference audio: `assets/resources/a.wav`.
- Reference transcript: `This little work was finished in the year eighteen o three, and intended for immediate publication.`
- Model: `higgs-audio-v3-tts-4b-q8_0.gguf`.

For each normal build, start three fresh one-slot servers. Repeat A within each
server; on the first server also switch A/B/A and unload/reload the model.
Separately start two- and four-slot servers and submit barrier-synchronized
waves: cold A, warm A twice, mixed A/B, then warm A again. Poll active slots to
confirm the configured concurrency. Compare decoded PCM SHA-256, WAV SHA-256,
frame count, sample rate, and normalized response data.

Additional controls repeat A without reference audio in both builds, disable
the encoded-reference cache in the current build, trace four serial requests,
and temporarily force full prefill in a separate diagnostic executable.
Production source and the normal server are restored before running that
diagnostic executable. Finally, retest the restored normal server.

## Results

| Check | Result |
|---|---|
| Three fresh servers per normal build | Cold A repeats exactly; warm A repeats exactly |
| Switch A/B/A | Returning to A preserves its warm output |
| Unload/reload model | Restores cold A, then warm A |
| Saved older build versus current | 49 byte-identical output pairs |
| Two and four slots, cold/warm and mixed prompts | 60 outputs match their serial phase references; every wave reached the configured concurrency |
| No reference audio, both builds | Cold and warm outputs are byte-identical |
| Encoded-reference cache disabled, capacity zero | Cold/warm difference remains |
| Diagnostic reference KV reuse disabled | Repeated A is byte-identical to normal cold A; 30 parallel outputs match their serial diagnostic references |
| Logging enabled | WAV hashes match the corresponding untraced requests |
| Restored production server | Original cold/warm hashes reproduced |

All **152 requests** succeeded, with no VRAM guard termination. Counts overlap:
the 49 old/current pairs include outputs counted in parallel validation.

| Normal prompt A output | PCM samples | Sample rate | Duration | WAV SHA-256 |
|---|---:|---:|---:|---|
| Cold | 90,240 | 24,000 Hz | 3.760 s | `8473d0e70ef49c13a64ba4074fd60fce64fb3c35c30bd7794fb60b4506f07f8d` |
| Warm | 88,320 | 24,000 Hz | 3.680 s | `15ad618730d45107be228598866b5d957b4edd4588f5657a5ab58532f098909d` |

This is a generated-content difference, not just a WAV header difference.

## Trace and causal control

Normal cold prefill processes all **189 prompt steps**. Warm requests reuse the
first **177 reference-prefix steps** and prefill the remaining **12 steps**.
The sampled prefill logits differ between these paths, and generated codec
code arrays have **94 versus 92 frames**. Each request resets the sampler seed;
the warm repeats and their trace samples remain stable.

The diagnostic changes only the reference KV hit condition in
`src/models/higgs_audio_tts/generator.cpp`:

```diff
 const bool reference_kv_cache_hit =
-    reference_cache_hit && reference_kv_ready_ &&
+    false && reference_cache_hit && reference_kv_ready_ &&
     ar_kv_cache_->valid_steps() >= prepared.prefix_steps;
```

Forcing full prefill removes the cold/warm difference, including with two and
four concurrent slots. Model weights, reference audio, seeds, CUDA kernels,
graph reuse, and sampler remain unchanged. Disabling only the session's
encoded-reference cache (`higgs_audio_tts.reference_cache_slots=0`) does not
disable generator KV prefix reuse and does not remove the difference.

The exact CUDA operation responsible for the numerical difference was not
identified. These tests do not establish stale state, a quality loss, or
determinism for every prompt and seed. Full prefill is a more expensive
diagnostic control, not a proposed production fix. The results support keeping
this investigation separate from parallel-slot regressions.

## Build provenance and local artifacts

The saved older executable is `audiocpp_server_parity_audit.exe` from the earlier
validation pass. It already includes slots and earlier changes; it is **not**
claimed to be pristine upstream. Its exact source revision was not recorded.

| Executable | SHA-256 |
|---|---|
| Saved older build | `72a9a263a59c594999a10035f0382477c876aa7e47a373c3c0da7748e7fa99c2` |
| Current build before diagnostic rebuild | `eb239233804efc0cae87655efc316c7c859abd2efa214a4f41c212492233de81` |
| Separate full-prefill diagnostic | `cb66723cf2b4f8152eaa55d45776541614bba9ee96b9ab995d1e592b87fa3473` |
| Restored normal build | `d664161463097576229aaca6f8cc3fd39c5817d0f49965179c019190dbf65434` |

Per-request WAVs, normalized responses, PCM hashes, configs, trace logs,
diagnostic patch and build logs are retained locally under
`outputs/model-two-slot-audit/higgs-cuda-repeatability/`, outside the repository.
The local harnesses are `test_higgs_repeatability.py`,
`test_higgs_full_prefill_control.py`, and `analyze_higgs_repeatability.py` in the
parent audit directory. Strict assertions over the saved results pass.
The full local artifacts and diagnostic executable are not included in this
documentation commit. Production source and the normal build were restored;
this report changes no inference implementation or slot eligibility.
