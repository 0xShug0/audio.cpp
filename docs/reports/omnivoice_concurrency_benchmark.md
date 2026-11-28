# OmniVoice Concurrency Benchmark (RTX 4090)

This report measures how much OmniVoice voice-cloning throughput one RTX 4090 can deliver, comparing
the C++ runtime with the Python OmniVoice package. Every variant runs as 1, 2, 3, ... separate
instances (processes, one model each) at batch size 1, 2, 3, ... until the GPU is saturated, in both
f16 and bf16.

- **Single request:** C++ `perf_mode=flash_attention` is the fastest at every sentence length
  (106–270 ms for 5–30 words, f16).
- **Peak throughput:** Python + FlashInfer is the highest, but only with 5–6 processes and large
  batches (19.9–22.5 GiB of GPU memory). For C++, a second instance lowers the best throughput
  averaged over the input sets by 5–7%, so one process is its saturation point.
- **bf16 vs f16 (best throughput per input set):** C++ is 3–12% slower in bf16, Python (PyTorch)
  changes by -0.1% to +2.5%, Python + CUDA graphs by -6.8% to +2.6%. Plain FlashInfer cannot run
  bf16 (the upstream patch plans attention as f16).

Notes:

- Each row of the data is one timed round after 5 warm-up rounds of the same shape, so single rows
  carry a few percent of run-to-run noise.
- C++ and Python estimate the output duration differently. For the same sentence in the same round
  shape, C++ output is from 29% shorter to 30% longer than Python's (median 0.7% longer), so the work
  per request is not identical. The audio length of every output is in the data.
- bf16 means the whole model in Python, but only the stored weights of the generator and audio
  tokenizer in C++; most C++ activations are f32 in both cases.

## Setup

- **GPU:** NVIDIA GeForce RTX 4090, 24 GiB, driver 560.35.05, CUDA 12.6, nothing else running on it.
- **Model:** `models/OmniVoice` SafeTensors, the same files for C++ and Python.
- **Task:** voice cloning in Bengali, one reference clip with its transcript, the same for every request.
- **Sampling:** 16 steps, guidance scale 2.0, t_shift 0.1, class_temperature 0.0, position_temperature
  5.0, layer_penalty_factor 5.0; denoise, preprocess_prompt and postprocess_output on.
- **Seed:** C++ 42, Python `torch.manual_seed(0)` before each call.
- **Inputs:** Bengali customer-care sentences grouped by length: 5, 10, 15, 20, 25 or 30 words, and a
  mixed list repeating 5, 15, 25, 30, 10, 20, 5, 30 words. Instance i gets sentences i x B to
  i x B + B - 1 of the list; a list is reused from the start when it runs out (33 sentences of 5
  words, 54 of 10 words).
- **C++:** OmniVoice code as merged in #1208, CUDA build, sm_89, default session options (`packed_cfg`,
  `device_scoring`, `packed_projections` on); `omnivoice.perf_mode=off` (default) or `flash_attention`;
  precision via `omnivoice.generator_weight_type` and `omnivoice.audio_tokenizer_weight_type`.
- **Python:** OmniVoice package, torch 2.8.0+cu128, model loaded as `torch.float16` or
  `torch.bfloat16`; FlashInfer = upstream `apply_flashinfer(model)` (flashinfer 0.6.15.post1); CUDA
  graphs = `apply_flashinfer(model, cuda_graph_buckets=[3, 5, 8, 12, 16, 20])`.
- **Calls:** batch 1: `run()` in C++, one string to `generate()` in Python. Batch 2+: `run_batch()` in
  C++, a list to `generate()` in Python.

## Method

- **Round:** all instances start at the same moment; each makes one call with *batch size* different
  sentences. Concurrent requests = instances x batch size.
- **Time:** from the common start until the last output of all instances is ready; each instance's
  own finish time is in brackets.
- **Throughput:** concurrent requests / time.
- **Batch size** goes 1, 2, 3, ... for each instance count and input set until 3 sizes in a row give
  no new best throughput, or a call runs out of memory (cap 64). A failed round ends that sweep.
- **Instances** go 1, 2, 3, ... until one more instance no longer raises the best throughput
  (best batch per input set, averaged over the sets), or the instances no longer fit in memory (cap 8).
- **GPU columns:** NVML sampled every 50 ms during the timed round. Utilization is the share of time a
  kernel was running; memory is the peak of all instances above the idle baseline, including cached
  allocations.

Saturation point per variant:

- Python + FlashInfer, f16: 6 instances gave -0.4% best throughput over fewer instances, so 5 is the saturation point.
- Python + FlashInfer, bf16: did not run (bf16 not supported: upstream `apply_flashinfer` plans attention as f16 only).
- Python + FlashInfer + CUDA graphs, f16: 4 instances gave -2.3% best throughput over fewer instances, so 3 is the saturation point.
- Python + FlashInfer + CUDA graphs, bf16: 3 instances gave -0.3% best throughput over fewer instances, so 2 is the saturation point.
- C++ (`perf_mode=flash_attention`), f16: 2 instances gave -5.2% best throughput over fewer instances, so 1 is the saturation point.
- C++ (`perf_mode=flash_attention`), bf16: 2 instances gave -5.7% best throughput over fewer instances, so 1 is the saturation point.
- C++ (`perf_mode=off`), f16: 2 instances gave -7.1% best throughput over fewer instances, so 1 is the saturation point.
- C++ (`perf_mode=off`), bf16: 2 instances gave -7.3% best throughput over fewer instances, so 1 is the saturation point.
- Python (PyTorch), f16: 4 instances gave -1.1% best throughput over fewer instances, so 3 is the saturation point.
- Python (PyTorch), bf16: 4 instances gave -0.8% best throughput over fewer instances, so 3 is the saturation point.

## All rounds

1511 rounds: 1496 measured and 15 failed (marked "failed" with the reason). Every round of the run, in run order per input set and variant. The same table, row for row and
cell for cell, is in [`data/omnivoice_concurrency_rtx4090.csv`](data/omnivoice_concurrency_rtx4090.csv).
Audio out: total, then each output's length; `+` between outputs of one instance, `,` between
instances.

| Input set | Instances | Batch size | Concurrent requests | Precision | Variant | Time, ms (start until all outputs of all instances are ready; per instance in brackets) | Throughput, req/s | GPU util | GPU mem peak | Audio out, s (individual) |
|---|---:|---:|---:|---|---|---|---:|---:|---:|---|
| Sentence of 5 words | 1 | 1 | 1 | f16 | Python + FlashInfer | 481 | 2.1 | 18% | 3.3 GB | 2.4 (2.4) |
| Sentence of 5 words | 1 | 2 | 2 | f16 | Python + FlashInfer | 507 | 3.9 | 26% | 3.3 GB | 4.6 (2.4 + 2.1) |
| Sentence of 5 words | 1 | 3 | 3 | f16 | Python + FlashInfer | 509 | 5.9 | 32% | 3.3 GB | 6.4 (2.2 + 2.1 + 2.1) |
| Sentence of 5 words | 1 | 4 | 4 | f16 | Python + FlashInfer | 557 | 7.2 | 37% | 3.3 GB | 9.3 (2.4 + 2.1 + 2.0 + 2.7) |
| Sentence of 5 words | 1 | 5 | 5 | f16 | Python + FlashInfer | 560 | 8.9 | 43% | 3.3 GB | 11.3 (2.4 + 2.1 + 2.1 + 2.7 + 2.0) |
| Sentence of 5 words | 1 | 6 | 6 | f16 | Python + FlashInfer | 568 | 10.6 | 46% | 3.3 GB | 13.5 (2.4 + 2.1 + 2.1 + 2.3 + 2.2 + 2.4) |
| Sentence of 5 words | 1 | 7 | 7 | f16 | Python + FlashInfer | 843 | 8.3 | 35% | 3.3 GB | 15.8 (2.4 + 2.1 + 1.9 + 2.7 + 2.2 + 2.4 + 2.1) |
| Sentence of 5 words | 1 | 8 | 8 | f16 | Python + FlashInfer | 771 | 10.4 | 55% | 3.3 GB | 18.2 (2.4 + 2.1 + 2.1 + 2.7 + 2.0 + 2.4 + 2.1 + 2.4) |
| Sentence of 5 words | 1 | 9 | 9 | f16 | Python + FlashInfer | 633 | 14.2 | 74% | 3.3 GB | 20.6 (2.1 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1 + 2.7 + 2.2) |
| Sentence of 5 words | 1 | 10 | 10 | f16 | Python + FlashInfer | 733 | 13.6 | 68% | 3.4 GB | 22.4 (2.4 + 2.0 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1 + 2.4 + 2.2 + 2.0) |
| Sentence of 5 words | 1 | 11 | 11 | f16 | Python + FlashInfer | 711 | 15.5 | 74% | 3.4 GB | 24.8 (2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.2 + 1.7 + 2.4 + 2.2 + 1.9 + 2.9) |
| Sentence of 5 words | 1 | 12 | 12 | f16 | Python + FlashInfer | 735 | 16.3 | 72% | 3.5 GB | 28.5 (2.4 + 2.1 + 2.1 + 2.6 + 2.2 + 2.4 + 2.1 + 2.7 + 2.2 + 2.0 + 3.2 + 2.5) |
| Sentence of 5 words | 1 | 13 | 13 | f16 | Python + FlashInfer | 857 | 15.2 | 71% | 3.6 GB | 29.5 (2.4 + 2.1 + 2.1 + 2.5 + 2.2 + 2.4 + 2.1 + 2.7 + 2.2 + 2.0 + 2.9 + 2.1 + 1.8) |
| Sentence of 5 words | 1 | 14 | 14 | f16 | Python + FlashInfer | 870 | 16.1 | 75% | 3.6 GB | 31.4 (2.1 + 2.1 + 2.1 + 2.4 + 2.2 + 2.1 + 2.1 + 2.3 + 2.2 + 1.6 + 3.2 + 2.5 + 1.8 + 2.7) |
| Sentence of 5 words | 1 | 15 | 15 | f16 | Python + FlashInfer | 1066 | 14.1 | 64% | 3.7 GB | 35.4 (2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 1.6 + 2.4 + 2.1 + 2.0 + 3.2 + 2.5 + 1.8 + 2.7 + 3.1) |
| Sentence of 5 words | 2 | 1 | 2 | f16 | Python + FlashInfer | 559 (501, 559) | 3.6 | 87% | 6.7 GB | 4.6 (2.4, 2.1) |
| Sentence of 5 words | 2 | 2 | 4 | f16 | Python + FlashInfer | 618 (618, 608) | 6.5 | 86% | 6.7 GB | 9.0 (2.4 + 2.1, 2.1 + 2.3) |
| Sentence of 5 words | 2 | 3 | 6 | f16 | Python + FlashInfer | 639 (616, 639) | 9.4 | 90% | 6.7 GB | 13.2 (2.2 + 2.1 + 2.1, 2.4 + 2.0 + 2.4) |
| Sentence of 5 words | 2 | 4 | 8 | f16 | Python + FlashInfer | 642 (642, 572) | 12.5 | 89% | 6.7 GB | 18.4 (2.4 + 2.1 + 2.0 + 2.7, 2.2 + 2.4 + 1.7 + 2.7) |
| Sentence of 5 words | 2 | 5 | 10 | f16 | Python + FlashInfer | 652 (652, 639) | 15.3 | 89% | 6.7 GB | 21.8 (2.4 + 2.1 + 2.1 + 2.7 + 2.0, 2.4 + 2.1 + 2.2 + 2.2 + 1.6) |
| Sentence of 5 words | 2 | 6 | 12 | f16 | Python + FlashInfer | 749 (704, 749) | 16.0 | 88% | 6.7 GB | 27.6 (2.4 + 2.1 + 2.1 + 2.3 + 2.2 + 2.4, 1.7 + 2.7 + 2.2 + 2.0 + 2.9 + 2.6) |
| Sentence of 5 words | 2 | 7 | 14 | f16 | Python + FlashInfer | 839 (822, 839) | 16.7 | 89% | 6.7 GB | 32.1 (2.4 + 2.1 + 1.9 + 2.7 + 2.2 + 2.4 + 2.1, 2.3 + 2.2 + 1.6 + 3.2 + 2.2 + 2.1 + 2.7) |
| Sentence of 5 words | 2 | 8 | 16 | f16 | Python + FlashInfer | 1004 (949, 1004) | 15.9 | 78% | 6.7 GB | 38.7 (2.4 + 2.1 + 2.1 + 2.7 + 2.0 + 2.4 + 2.1 + 2.4, 2.2 + 2.0 + 3.2 + 2.5 + 2.1 + 3.1 + 2.8 + 2.6) |
| Sentence of 5 words | 2 | 9 | 18 | f16 | Python + FlashInfer | 1089 (1023, 1089) | 16.5 | 86% | 6.8 GB | 42.9 (2.1 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1 + 2.7 + 2.2, 1.7 + 2.9 + 2.5 + 2.1 + 3.0 + 2.6 + 2.6 + 2.3 + 2.6) |
| Sentence of 5 words | 2 | 10 | 20 | f16 | Python + FlashInfer | 1235 (1187, 1235) | 16.2 | 85% | 6.9 GB | 46.9 (2.4 + 2.0 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1 + 2.4 + 2.2 + 2.0, 2.7 + 2.6 + 1.7 + 2.7 + 2.6 + 2.3 + 2.3 + 2.6 + 2.4 + 2.6) |
| Sentence of 5 words | 3 | 1 | 3 | f16 | Python + FlashInfer | 752 (752, 703, 647) | 4.0 | 76% | 10.0 GB | 6.6 (2.4, 2.1, 2.1) |
| Sentence of 5 words | 3 | 2 | 6 | f16 | Python + FlashInfer | 848 (573, 848, 592) | 7.1 | 87% | 10.0 GB | 13.6 (2.4 + 2.1, 2.1 + 2.3, 2.2 + 2.4) |
| Sentence of 5 words | 3 | 3 | 9 | f16 | Python + FlashInfer | 670 (660, 670, 659) | 13.4 | 92% | 10.0 GB | 20.2 (2.2 + 2.1 + 2.1, 2.4 + 2.0 + 2.4, 2.1 + 2.7 + 2.2) |
| Sentence of 5 words | 3 | 4 | 12 | f16 | Python + FlashInfer | 800 (768, 800, 783) | 15.0 | 90% | 10.0 GB | 28.0 (2.4 + 2.1 + 2.0 + 2.7, 2.2 + 2.4 + 1.7 + 2.7, 2.2 + 2.0 + 2.9 + 2.6) |
| Sentence of 5 words | 3 | 5 | 15 | f16 | Python + FlashInfer | 886 (858, 858, 886) | 16.9 | 92% | 10.0 GB | 34.7 (2.4 + 2.1 + 2.1 + 2.7 + 2.0, 2.4 + 2.1 + 2.2 + 2.2 + 1.6, 3.1 + 2.2 + 2.0 + 3.1 + 2.6) |
| Sentence of 5 words | 3 | 6 | 18 | f16 | Python + FlashInfer | 1047 (950, 1047, 1007) | 17.2 | 94% | 10.0 GB | 42.8 (2.4 + 2.1 + 2.1 + 2.3 + 2.2 + 2.4, 1.7 + 2.7 + 2.2 + 2.0 + 2.9 + 2.6, 2.1 + 3.1 + 2.5 + 2.6 + 2.3 + 2.6) |
| Sentence of 5 words | 3 | 7 | 21 | f16 | Python + FlashInfer | 1212 (1130, 1212, 1177) | 17.3 | 95% | 10.0 GB | 49.7 (2.4 + 2.1 + 1.9 + 2.7 + 2.2 + 2.4 + 2.1, 2.3 + 2.2 + 1.6 + 3.2 + 2.2 + 2.1 + 2.7, 2.6 + 2.6 + 1.8 + 3.1 + 2.2 + 2.9 + 2.4) |
| Sentence of 5 words | 3 | 8 | 24 | f16 | Python + FlashInfer | 1415 (1309, 1415, 1381) | 17.0 | 94% | 10.0 GB | 59.6 (2.4 + 2.1 + 2.1 + 2.7 + 2.0 + 2.4 + 2.1 + 2.4, 2.2 + 2.0 + 3.2 + 2.5 + 2.1 + 3.1 + 2.8 + 2.6, 2.3 + 2.3 + 2.7 + 2.9 + 2.4 + 2.7 + 2.8 + 2.9) |
| Sentence of 5 words | 3 | 9 | 27 | f16 | Python + FlashInfer | 1570 (1409, 1505, 1570) | 17.2 | 94% | 10.2 GB | 66.6 (2.1 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1 + 2.7 + 2.2, 1.7 + 2.9 + 2.5 + 2.1 + 3.0 + 2.6 + 2.6 + 2.3 + 2.6, 2.7 + 2.8 + 2.4 + 2.7 + 2.6 + 2.5 + 2.6 + 2.8 + 2.5) |
| Sentence of 5 words | 3 | 10 | 30 | f16 | Python + FlashInfer | 1692 (1676, 1673, 1692) | 17.7 | 95% | 10.3 GB | 72.5 (2.4 + 2.0 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1 + 2.4 + 2.2 + 2.0, 2.7 + 2.6 + 1.7 + 2.7 + 2.6 + 2.3 + 2.3 + 2.6 + 2.4 + 2.6, 2.4 + 2.7 + 2.5 + 2.7 + 2.6 + 2.2 + 2.5 + 2.6 + 2.6 + 2.6) |
| Sentence of 5 words | 3 | 11 | 33 | f16 | Python + FlashInfer | 1882 (1751, 1873, 1882) | 17.5 | 95% | 10.5 GB | 81.0 (2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.2 + 1.7 + 2.4 + 2.2 + 1.9 + 2.9, 2.6 + 2.1 + 2.8 + 3.0 + 2.3 + 1.9 + 2.6 + 2.3 + 2.9 + 2.4 + 2.7, 2.8 + 3.0 + 2.2 + 2.1 + 2.5 + 2.1 + 2.7 + 2.6 + 2.4 + 3.2 + 3.2) |
| Sentence of 5 words | 3 | 12 | 36 | f16 | Python + FlashInfer | 2002 (1978, 2002, 1997) | 18.0 | 93% | 10.7 GB | 89.5 (2.4 + 2.1 + 2.1 + 2.6 + 2.2 + 2.4 + 2.1 + 2.7 + 2.2 + 2.0 + 3.2 + 2.5, 2.0 + 2.8 + 2.5 + 2.6 + 2.3 + 2.4 + 2.7 + 2.9 + 2.4 + 2.7 + 2.8 + 3.0, 2.6 + 2.2 + 2.5 + 2.5 + 2.8 + 2.6 + 2.4 + 3.2 + 2.8 + 2.1 + 2.1 + 2.0) |
| Sentence of 5 words | 3 | 13 | 39 | f16 | Python + FlashInfer | 2180 (2158, 2171, 2180) | 17.9 | 94% | 10.9 GB | 96.3 (2.4 + 2.1 + 2.1 + 2.5 + 2.2 + 2.4 + 2.1 + 2.7 + 2.2 + 2.0 + 2.9 + 2.1 + 1.8, 3.1 + 2.6 + 2.4 + 2.3 + 3.1 + 2.7 + 2.9 + 2.4 + 2.5 + 2.8 + 2.7 + 2.6 + 2.2, 2.5 + 2.6 + 2.8 + 2.6 + 2.4 + 2.9 + 3.0 + 2.4 + 2.1 + 2.1 + 2.7 + 2.0 + 2.4) |
| Sentence of 5 words | 3 | 14 | 42 | f16 | Python + FlashInfer | 2328 (2262, 2326, 2328) | 18.0 | 95% | 11.2 GB | 102.0 (2.1 + 2.1 + 2.1 + 2.4 + 2.2 + 2.1 + 2.1 + 2.3 + 2.2 + 1.6 + 3.2 + 2.5 + 1.8 + 2.7, 2.6 + 2.6 + 2.2 + 2.9 + 2.7 + 2.7 + 2.4 + 2.7 + 2.8 + 3.0 + 2.3 + 2.6 + 2.5 + 2.6, 2.8 + 2.6 + 2.3 + 2.8 + 3.0 + 2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1 + 2.3 + 2.2) |
| Sentence of 5 words | 3 | 15 | 45 | f16 | Python + FlashInfer | 2528 (2475, 2528, 2458) | 17.8 | 92% | 11.4 GB | 110.2 (2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 1.6 + 2.4 + 2.1 + 2.0 + 3.2 + 2.5 + 1.8 + 2.7 + 3.1, 2.6 + 2.3 + 2.9 + 2.7 + 2.9 + 2.3 + 2.7 + 2.8 + 3.0 + 2.6 + 2.3 + 2.5 + 2.6 + 2.6 + 2.6, 2.4 + 3.2 + 3.2 + 2.0 + 1.9 + 2.1 + 2.7 + 2.2 + 2.3 + 2.1 + 2.7 + 2.0 + 1.7 + 2.7 + 2.6) |
| Sentence of 5 words | 3 | 16 | 48 | f16 | Python + FlashInfer | 2696 (2591, 2696, 2608) | 17.8 | 94% | 11.7 GB | 117.6 (2.4 + 2.1 + 2.1 + 2.5 + 2.2 + 2.4 + 1.9 + 2.3 + 2.0 + 2.0 + 2.9 + 2.2 + 2.1 + 3.0 + 2.6 + 2.6, 2.1 + 3.1 + 2.7 + 2.9 + 2.4 + 2.7 + 2.8 + 3.0 + 2.6 + 2.2 + 2.5 + 2.1 + 2.8 + 2.6 + 2.4 + 3.2, 2.8 + 2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 1.5 + 2.7 + 2.2 + 2.0 + 3.2 + 2.6 + 1.6 + 2.7 + 3.0) |
| Sentence of 5 words | 3 | 17 | 51 | f16 | Python + FlashInfer | 2812 (2795, 2812, 2780) | 18.1 | 94% | 11.9 GB | 124.0 (2.2 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1 + 2.4 + 2.2 + 2.0 + 3.2 + 2.2 + 1.9 + 2.7 + 2.5 + 2.3 + 2.3, 2.8 + 2.2 + 2.9 + 2.3 + 2.7 + 2.8 + 2.6 + 2.6 + 2.3 + 2.2 + 2.5 + 2.7 + 2.6 + 2.3 + 3.2 + 3.2 + 2.2, 2.1 + 2.1 + 2.2 + 1.9 + 2.4 + 2.1 + 2.5 + 2.2 + 1.7 + 3.2 + 2.6 + 2.1 + 3.1 + 2.7 + 2.5 + 2.3 + 2.8) |
| Sentence of 5 words | 3 | 18 | 54 | f16 | Python + FlashInfer | 3048 (2975, 3048, 2992) | 17.7 | 95% | 12.2 GB | 131.8 (2.4 + 2.1 + 2.1 + 2.5 + 2.2 + 2.1 + 2.1 + 2.3 + 2.2 + 2.0 + 2.9 + 2.5 + 1.8 + 2.7 + 2.6 + 2.6 + 2.3 + 2.6, 2.2 + 2.7 + 2.4 + 2.5 + 2.8 + 2.5 + 2.6 + 2.8 + 2.2 + 2.2 + 2.8 + 2.6 + 2.4 + 3.2 + 3.0 + 2.2 + 2.1 + 2.1, 2.6 + 1.9 + 2.4 + 1.9 + 2.2 + 2.2 + 2.0 + 3.2 + 2.6 + 2.0 + 3.1 + 2.6 + 2.6 + 2.3 + 2.8 + 2.7 + 2.9 + 2.4) |
| Sentence of 5 words | 3 | 19 | 57 | f16 | Python + FlashInfer | 3168 (3138, 3168, 3111) | 18.0 | 93% | 12.5 GB | 141.7 (2.4 + 2.0 + 2.1 + 2.7 + 2.2 + 2.0 + 2.1 + 2.7 + 2.0 + 2.0 + 2.9 + 2.6 + 1.8 + 3.0 + 2.6 + 2.6 + 2.3 + 2.5 + 2.7, 2.9 + 2.4 + 2.6 + 2.5 + 3.0 + 2.5 + 2.8 + 2.5 + 2.6 + 2.8 + 2.6 + 2.4 + 2.9 + 3.2 + 2.4 + 2.1 + 2.0 + 2.7 + 2.2, 2.4 + 2.1 + 2.7 + 2.2 + 1.6 + 2.8 + 2.5 + 2.1 + 3.1 + 2.7 + 2.6 + 2.3 + 2.7 + 2.7 + 2.9 + 2.3 + 2.6 + 2.6 + 2.6) |
| Sentence of 5 words | 3 | 20 | 60 | f16 | Python + FlashInfer | 3333 (3287, 3333, 3277) | 18.0 | 93% | 12.9 GB | 148.9 (2.4 + 2.1 + 1.8 + 2.5 + 2.2 + 2.4 + 2.1 + 2.7 + 2.2 + 2.0 + 3.2 + 2.3 + 1.8 + 2.7 + 2.8 + 2.4 + 2.3 + 3.1 + 2.4 + 2.9, 2.4 + 2.7 + 2.8 + 2.5 + 2.6 + 2.8 + 2.5 + 2.5 + 2.5 + 2.6 + 2.4 + 3.2 + 3.0 + 2.4 + 2.1 + 2.1 + 2.7 + 2.0 + 2.1 + 2.1, 2.6 + 2.1 + 2.0 + 3.2 + 2.6 + 1.8 + 3.1 + 2.6 + 2.6 + 2.3 + 3.1 + 2.7 + 2.9 + 2.4 + 2.5 + 2.8 + 2.5 + 2.1 + 2.3 + 2.5) |
| Sentence of 5 words | 4 | 1 | 4 | f16 | Python + FlashInfer | 585 (555, 578, 585, 579) | 6.8 | 91% | 13.3 GB | 9.3 (2.4, 2.1, 2.1, 2.7) |
| Sentence of 5 words | 4 | 2 | 8 | f16 | Python + FlashInfer | 764 (764, 570, 622, 627) | 10.5 | 85% | 13.3 GB | 17.6 (2.4 + 2.1, 2.1 + 2.3, 2.2 + 2.4, 1.8 + 2.3) |
| Sentence of 5 words | 4 | 3 | 12 | f16 | Python + FlashInfer | 775 (728, 768, 748, 775) | 15.5 | 87% | 13.3 GB | 27.5 (2.2 + 2.1 + 2.1, 2.4 + 2.0 + 2.4, 2.1 + 2.7 + 2.2, 1.7 + 3.2 + 2.5) |
| Sentence of 5 words | 4 | 4 | 16 | f16 | Python + FlashInfer | 988 (959, 958, 977, 988) | 16.2 | 94% | 13.3 GB | 37.5 (2.4 + 2.1 + 2.0 + 2.7, 2.2 + 2.4 + 1.7 + 2.7, 2.2 + 2.0 + 2.9 + 2.6, 1.5 + 2.7 + 2.9 + 2.4) |
| Sentence of 5 words | 4 | 5 | 20 | f16 | Python + FlashInfer | 1148 (1109, 1106, 1148, 1130) | 17.4 | 93% | 13.3 GB | 47.9 (2.4 + 2.1 + 2.1 + 2.7 + 2.0, 2.4 + 2.1 + 2.2 + 2.2 + 1.6, 3.1 + 2.2 + 2.0 + 3.1 + 2.6, 2.6 + 2.3 + 2.7 + 2.7 + 2.9) |
| Sentence of 5 words | 4 | 6 | 24 | f16 | Python + FlashInfer | 1332 (1267, 1330, 1332, 1298) | 18.0 | 95% | 13.3 GB | 58.4 (2.4 + 2.1 + 2.1 + 2.3 + 2.2 + 2.4, 1.7 + 2.7 + 2.2 + 2.0 + 2.9 + 2.6, 2.1 + 3.1 + 2.5 + 2.6 + 2.3 + 2.6, 2.5 + 2.9 + 2.4 + 2.7 + 2.6 + 2.5) |
| Sentence of 5 words | 4 | 7 | 28 | f16 | Python + FlashInfer | 1559 (1552, 1513, 1552, 1559) | 18.0 | 95% | 13.3 GB | 67.5 (2.4 + 2.1 + 1.9 + 2.7 + 2.2 + 2.4 + 2.1, 2.3 + 2.2 + 1.6 + 3.2 + 2.2 + 2.1 + 2.7, 2.6 + 2.6 + 1.8 + 3.1 + 2.2 + 2.9 + 2.4, 2.7 + 2.8 + 2.6 + 2.6 + 2.1 + 2.5 + 2.6) |
| Sentence of 5 words | 4 | 8 | 32 | f16 | Python + FlashInfer | 1809 (1718, 1809, 1768, 1786) | 17.7 | 96% | 13.3 GB | 80.3 (2.4 + 2.1 + 2.1 + 2.7 + 2.0 + 2.4 + 2.1 + 2.4, 2.2 + 2.0 + 3.2 + 2.5 + 2.1 + 3.1 + 2.8 + 2.6, 2.3 + 2.3 + 2.7 + 2.9 + 2.4 + 2.7 + 2.8 + 2.9, 2.1 + 2.8 + 2.5 + 2.6 + 2.6 + 2.6 + 2.4 + 3.1) |
| Sentence of 5 words | 4 | 9 | 36 | f16 | Python + FlashInfer | 2000 (1811, 1988, 1959, 2000) | 18.0 | 96% | 13.6 GB | 89.1 (2.1 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1 + 2.7 + 2.2, 1.7 + 2.9 + 2.5 + 2.1 + 3.0 + 2.6 + 2.6 + 2.3 + 2.6, 2.7 + 2.8 + 2.4 + 2.7 + 2.6 + 2.5 + 2.6 + 2.8 + 2.5, 2.5 + 2.8 + 2.6 + 2.4 + 2.6 + 3.0 + 2.4 + 2.1 + 2.1) |
| Sentence of 5 words | 5 | 1 | 5 | f16 | Python + FlashInfer | 855 (837, 767, 855, 619, 695) | 5.8 | 92% | 16.6 GB | 11.4 (2.4, 2.1, 2.1, 2.7, 2.0) |
| Sentence of 5 words | 5 | 2 | 10 | f16 | Python + FlashInfer | 730 (710, 716, 706, 730, 709) | 13.7 | 75% | 16.6 GB | 21.8 (2.4 + 2.1, 2.1 + 2.3, 2.2 + 2.4, 1.8 + 2.3, 2.2 + 2.0) |
| Sentence of 5 words | 5 | 3 | 15 | f16 | Python + FlashInfer | 976 (888, 941, 914, 959, 976) | 15.4 | 94% | 16.6 GB | 35.1 (2.2 + 2.1 + 2.1, 2.4 + 2.0 + 2.4, 2.1 + 2.7 + 2.2, 1.7 + 3.2 + 2.5, 2.0 + 3.1 + 2.5) |
| Sentence of 5 words | 5 | 4 | 20 | f16 | Python + FlashInfer | 1256 (1183, 1230, 1201, 1256, 1224) | 15.9 | 94% | 16.6 GB | 48.4 (2.4 + 2.1 + 2.0 + 2.7, 2.2 + 2.4 + 1.7 + 2.7, 2.2 + 2.0 + 2.9 + 2.6, 1.5 + 2.7 + 2.9 + 2.4, 2.3 + 2.9 + 2.7 + 2.9) |
| Sentence of 5 words | 5 | 5 | 25 | f16 | Python + FlashInfer | 1417 (1370, 1376, 1417, 1415, 1402) | 17.6 | 97% | 16.6 GB | 60.9 (2.4 + 2.1 + 2.1 + 2.7 + 2.0, 2.4 + 2.1 + 2.2 + 2.2 + 1.6, 3.1 + 2.2 + 2.0 + 3.1 + 2.6, 2.6 + 2.3 + 2.7 + 2.7 + 2.9, 2.4 + 2.5 + 2.5 + 3.0 + 2.6) |
| Sentence of 5 words | 5 | 6 | 30 | f16 | Python + FlashInfer | 1666 (1590, 1575, 1653, 1666, 1581) | 18.0 | 96% | 16.6 GB | 73.9 (2.4 + 2.1 + 2.1 + 2.3 + 2.2 + 2.4, 1.7 + 2.7 + 2.2 + 2.0 + 2.9 + 2.6, 2.1 + 3.1 + 2.5 + 2.6 + 2.3 + 2.6, 2.5 + 2.9 + 2.4 + 2.7 + 2.6 + 2.5, 2.5 + 2.6 + 2.5 + 2.6 + 2.8 + 2.6) |
| Sentence of 5 words | 5 | 7 | 35 | f16 | Python + FlashInfer | 1952 (1781, 1772, 1941, 1859, 1952) | 17.9 | 97% | 16.6 GB | 85.3 (2.4 + 2.1 + 1.9 + 2.7 + 2.2 + 2.4 + 2.1, 2.3 + 2.2 + 1.6 + 3.2 + 2.2 + 2.1 + 2.7, 2.6 + 2.6 + 1.8 + 3.1 + 2.2 + 2.9 + 2.4, 2.7 + 2.8 + 2.6 + 2.6 + 2.1 + 2.5 + 2.6, 2.8 + 2.6 + 2.3 + 2.8 + 3.1 + 2.1 + 2.1) |
| Sentence of 5 words | 5 | 8 | 40 | f16 | Python + FlashInfer | 2194 (2140, 2169, 2194, 2153, 2115) | 18.2 | 97% | 16.6 GB | 98.3 (2.4 + 2.1 + 2.1 + 2.7 + 2.0 + 2.4 + 2.1 + 2.4, 2.2 + 2.0 + 3.2 + 2.5 + 2.1 + 3.1 + 2.8 + 2.6, 2.3 + 2.3 + 2.7 + 2.9 + 2.4 + 2.7 + 2.8 + 2.9, 2.1 + 2.8 + 2.5 + 2.6 + 2.6 + 2.6 + 2.4 + 3.1, 2.8 + 1.9 + 2.1 + 1.8 + 2.7 + 2.2 + 2.4 + 2.1) |
| Sentence of 5 words | 5 | 9 | 45 | f16 | Python + FlashInfer | 2471 (2282, 2471, 2424, 2417, 2407) | 18.2 | 96% | 16.9 GB | 109.7 (2.1 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1 + 2.7 + 2.2, 1.7 + 2.9 + 2.5 + 2.1 + 3.0 + 2.6 + 2.6 + 2.3 + 2.6, 2.7 + 2.8 + 2.4 + 2.7 + 2.6 + 2.5 + 2.6 + 2.8 + 2.5, 2.5 + 2.8 + 2.6 + 2.4 + 2.6 + 3.0 + 2.4 + 2.1 + 2.1, 2.7 + 2.2 + 2.4 + 1.8 + 2.5 + 2.2 + 1.5 + 2.8 + 2.6) |
| Sentence of 5 words | 5 | 10 | 50 | f16 | Python + FlashInfer | 2701 (2602, 2701, 2666, 2655, 2659) | 18.5 | 96% | 17.2 GB | 121.0 (2.4 + 2.0 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1 + 2.4 + 2.2 + 2.0, 2.7 + 2.6 + 1.7 + 2.7 + 2.6 + 2.3 + 2.3 + 2.6 + 2.4 + 2.6, 2.4 + 2.7 + 2.5 + 2.7 + 2.6 + 2.2 + 2.5 + 2.6 + 2.6 + 2.6, 2.4 + 3.1 + 3.2 + 2.4 + 2.1 + 2.1 + 2.4 + 2.2 + 2.1 + 1.7, 2.3 + 2.2 + 2.0 + 2.8 + 2.6 + 2.1 + 3.0 + 3.1 + 2.6 + 2.3) |
| Sentence of 5 words | 5 | 11 | 55 | f16 | Python + FlashInfer | 2960 (2912, 2951, 2960, 2884, 2942) | 18.6 | 97% | 17.5 GB | 133.4 (2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.2 + 1.7 + 2.4 + 2.2 + 1.9 + 2.9, 2.6 + 2.1 + 2.8 + 3.0 + 2.3 + 1.9 + 2.6 + 2.3 + 2.9 + 2.4 + 2.7, 2.8 + 3.0 + 2.2 + 2.1 + 2.5 + 2.1 + 2.7 + 2.6 + 2.4 + 3.2 + 3.2, 2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.2 + 1.7 + 2.4 + 2.2 + 1.9 + 2.9, 2.6 + 2.1 + 2.8 + 3.0 + 2.3 + 1.9 + 2.6 + 2.3 + 2.9 + 2.4 + 2.7) |
| Sentence of 5 words | 5 | 12 | 60 | f16 | Python + FlashInfer | 3239 (3111, 3239, 3220, 3204, 3239) | 18.5 | 98% | 17.8 GB | 148.0 (2.4 + 2.1 + 2.1 + 2.6 + 2.2 + 2.4 + 2.1 + 2.7 + 2.2 + 2.0 + 3.2 + 2.5, 2.0 + 2.8 + 2.5 + 2.6 + 2.3 + 2.4 + 2.7 + 2.9 + 2.4 + 2.7 + 2.8 + 3.0, 2.6 + 2.2 + 2.5 + 2.5 + 2.8 + 2.6 + 2.4 + 3.2 + 2.8 + 2.1 + 2.1 + 2.0, 2.7 + 2.2 + 2.3 + 2.1 + 2.2 + 2.0 + 1.6 + 2.8 + 2.6 + 2.1 + 3.1 + 2.5, 2.6 + 2.3 + 2.5 + 2.7 + 2.9 + 2.4 + 2.4 + 2.8 + 2.5 + 2.6 + 2.2 + 2.5) |
| Sentence of 5 words | 5 | 13 | 65 | f16 | Python + FlashInfer | 3492 (3348, 3492, 3408, 3473, 3454) | 18.6 | 98% | 18.2 GB | 160.6 (2.4 + 2.1 + 2.1 + 2.5 + 2.2 + 2.4 + 2.1 + 2.7 + 2.2 + 2.0 + 2.9 + 2.1 + 1.8, 3.1 + 2.6 + 2.4 + 2.3 + 3.1 + 2.7 + 2.9 + 2.4 + 2.5 + 2.8 + 2.7 + 2.6 + 2.2, 2.5 + 2.6 + 2.8 + 2.6 + 2.4 + 2.9 + 3.0 + 2.4 + 2.1 + 2.1 + 2.7 + 2.0 + 2.4, 1.7 + 2.7 + 2.0 + 1.5 + 2.8 + 2.6 + 1.8 + 3.1 + 2.7 + 2.6 + 1.9 + 2.5 + 2.3, 2.9 + 2.4 + 2.7 + 2.8 + 2.3 + 2.6 + 2.7 + 2.5 + 2.5 + 2.6 + 2.6 + 2.4 + 3.2) |
| Sentence of 5 words | 5 | 14 | 70 | f16 | Python + FlashInfer | 3762 (3643, 3709, 3658, 3711, 3762) | 18.6 | 97% | 18.6 GB | 171.3 (2.1 + 2.1 + 2.1 + 2.4 + 2.2 + 2.1 + 2.1 + 2.3 + 2.2 + 1.6 + 3.2 + 2.5 + 1.8 + 2.7, 2.6 + 2.6 + 2.2 + 2.9 + 2.7 + 2.7 + 2.4 + 2.7 + 2.8 + 3.0 + 2.3 + 2.6 + 2.5 + 2.6, 2.8 + 2.6 + 2.3 + 2.8 + 3.0 + 2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1 + 2.3 + 2.2, 1.6 + 2.9 + 2.6 + 1.9 + 2.8 + 2.8 + 2.6 + 1.9 + 2.7 + 2.3 + 2.9 + 2.4 + 2.5 + 2.4, 2.5 + 2.6 + 2.2 + 2.5 + 2.4 + 2.8 + 2.6 + 2.4 + 3.2 + 3.2 + 2.4 + 2.1 + 1.8 + 2.5) |
| Sentence of 5 words | 5 | 15 | 75 | f16 | Python + FlashInfer | 3983 (3884, 3983, 3884, 3962, 3897) | 18.8 | 94% | 19.0 GB | 183.9 (2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 1.6 + 2.4 + 2.1 + 2.0 + 3.2 + 2.5 + 1.8 + 2.7 + 3.1, 2.6 + 2.3 + 2.9 + 2.7 + 2.9 + 2.3 + 2.7 + 2.8 + 3.0 + 2.6 + 2.3 + 2.5 + 2.6 + 2.6 + 2.6, 2.4 + 3.2 + 3.2 + 2.0 + 1.9 + 2.1 + 2.7 + 2.2 + 2.3 + 2.1 + 2.7 + 2.0 + 1.7 + 2.7 + 2.6, 2.1 + 3.1 + 2.9 + 2.6 + 2.3 + 2.5 + 2.4 + 2.9 + 2.1 + 2.7 + 2.8 + 2.5 + 2.6 + 2.3 + 2.5, 2.4 + 2.7 + 2.6 + 2.2 + 3.0 + 3.0 + 2.2 + 2.1 + 2.1 + 2.7 + 1.9 + 2.4 + 1.6 + 2.3 + 2.2) |
| Sentence of 5 words | 5 | 16 | 80 | f16 | Python + FlashInfer | 4175 (4094, 4175, 4133, 4163, 4092) | 19.2 | 97% | 19.4 GB | 196.4 (2.4 + 2.1 + 2.1 + 2.5 + 2.2 + 2.4 + 1.9 + 2.3 + 2.0 + 2.0 + 2.9 + 2.2 + 2.1 + 3.0 + 2.6 + 2.6, 2.1 + 3.1 + 2.7 + 2.9 + 2.4 + 2.7 + 2.8 + 3.0 + 2.6 + 2.2 + 2.5 + 2.1 + 2.8 + 2.6 + 2.4 + 3.2, 2.8 + 2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 1.5 + 2.7 + 2.2 + 2.0 + 3.2 + 2.6 + 1.6 + 2.7 + 3.0, 2.6 + 2.3 + 2.4 + 2.6 + 2.7 + 2.4 + 2.6 + 2.3 + 2.9 + 2.6 + 2.3 + 2.5 + 2.3 + 2.8 + 2.6 + 2.3, 2.9 + 3.2 + 2.4 + 2.1 + 2.1 + 2.3 + 1.9 + 2.4 + 2.1 + 2.7 + 2.2 + 1.7 + 2.8 + 2.6 + 2.1 + 3.1) |
| Sentence of 5 words | 5 | 17 | 85 | f16 | Python + FlashInfer | 4421 (4353, 4421, 4355, 4418, 4371) | 19.2 | 95% | 19.9 GB | 208.3 (2.2 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1 + 2.4 + 2.2 + 2.0 + 3.2 + 2.2 + 1.9 + 2.7 + 2.5 + 2.3 + 2.3, 2.8 + 2.2 + 2.9 + 2.3 + 2.7 + 2.8 + 2.6 + 2.6 + 2.3 + 2.2 + 2.5 + 2.7 + 2.6 + 2.3 + 3.2 + 3.2 + 2.2, 2.1 + 2.1 + 2.2 + 1.9 + 2.4 + 2.1 + 2.5 + 2.2 + 1.7 + 3.2 + 2.6 + 2.1 + 3.1 + 2.7 + 2.5 + 2.3 + 2.8, 2.5 + 2.9 + 2.4 + 2.7 + 2.8 + 2.5 + 2.5 + 2.8 + 2.5 + 2.2 + 2.8 + 2.6 + 2.4 + 3.2 + 3.2 + 2.4 + 2.1, 1.8 + 2.7 + 2.0 + 2.4 + 1.5 + 2.3 + 2.2 + 1.6 + 3.1 + 2.6 + 2.1 + 2.8 + 3.1 + 2.6 + 2.3 + 2.4 + 2.3) |
| Sentence of 5 words | 5 | 18 | 90 | f16 | Python + FlashInfer | 4789 (4709, 4743, 4744, 4789, 4750) | 18.8 | 98% | 20.4 GB | 221.6 (2.4 + 2.1 + 2.1 + 2.5 + 2.2 + 2.1 + 2.1 + 2.3 + 2.2 + 2.0 + 2.9 + 2.5 + 1.8 + 2.7 + 2.6 + 2.6 + 2.3 + 2.6, 2.2 + 2.7 + 2.4 + 2.5 + 2.8 + 2.5 + 2.6 + 2.8 + 2.2 + 2.2 + 2.8 + 2.6 + 2.4 + 3.2 + 3.0 + 2.2 + 2.1 + 2.1, 2.6 + 1.9 + 2.4 + 1.9 + 2.2 + 2.2 + 2.0 + 3.2 + 2.6 + 2.0 + 3.1 + 2.6 + 2.6 + 2.3 + 2.8 + 2.7 + 2.9 + 2.4, 2.7 + 2.6 + 2.4 + 2.3 + 2.3 + 2.5 + 2.6 + 2.8 + 2.6 + 2.4 + 3.2 + 3.2 + 2.4 + 1.9 + 2.1 + 2.5 + 2.2 + 2.1, 2.1 + 2.7 + 2.2 + 1.5 + 2.7 + 2.6 + 2.1 + 3.0 + 3.0 + 2.6 + 2.3 + 2.5 + 2.7 + 2.8 + 2.2 + 2.7 + 2.8 + 2.6) |
| Sentence of 5 words | 5 | 19 | 95 | f16 | Python + FlashInfer | 5059 (4898, 4957, 4965, 4895, 5059) | 18.8 | 98% | 20.9 GB | 235.9 (2.4 + 2.0 + 2.1 + 2.7 + 2.2 + 2.0 + 2.1 + 2.7 + 2.0 + 2.0 + 2.9 + 2.6 + 1.8 + 3.0 + 2.6 + 2.6 + 2.3 + 2.5 + 2.7, 2.9 + 2.4 + 2.6 + 2.5 + 3.0 + 2.5 + 2.8 + 2.5 + 2.6 + 2.8 + 2.6 + 2.4 + 2.9 + 3.2 + 2.4 + 2.1 + 2.0 + 2.7 + 2.2, 2.4 + 2.1 + 2.7 + 2.2 + 1.6 + 2.8 + 2.5 + 2.1 + 3.1 + 2.7 + 2.6 + 2.3 + 2.7 + 2.7 + 2.9 + 2.3 + 2.6 + 2.6 + 2.6, 2.6 + 2.2 + 2.5 + 2.5 + 2.7 + 2.6 + 2.4 + 2.9 + 3.1 + 2.4 + 2.1 + 2.0 + 2.5 + 1.9 + 2.1 + 2.1 + 2.7 + 2.2 + 1.6, 2.7 + 2.6 + 2.0 + 3.1 + 2.5 + 2.6 + 2.3 + 2.6 + 2.5 + 2.8 + 2.4 + 2.7 + 2.8 + 2.6 + 2.3 + 2.8 + 2.5 + 2.4 + 2.8) |
| Sentence of 5 words | 5 | 20 | 100 | f16 | Python + FlashInfer | 5258 (5247, 5061, 5258, 5047, 5142) | 19.0 | 98% | 21.4 GB | 248.1 (2.4 + 2.1 + 1.8 + 2.5 + 2.2 + 2.4 + 2.1 + 2.7 + 2.2 + 2.0 + 3.2 + 2.3 + 1.8 + 2.7 + 2.8 + 2.4 + 2.3 + 3.1 + 2.4 + 2.9, 2.4 + 2.7 + 2.8 + 2.5 + 2.6 + 2.8 + 2.5 + 2.5 + 2.5 + 2.6 + 2.4 + 3.2 + 3.0 + 2.4 + 2.1 + 2.1 + 2.7 + 2.0 + 2.1 + 2.1, 2.6 + 2.1 + 2.0 + 3.2 + 2.6 + 1.8 + 3.1 + 2.6 + 2.6 + 2.3 + 3.1 + 2.7 + 2.9 + 2.4 + 2.5 + 2.8 + 2.5 + 2.1 + 2.3 + 2.5, 2.5 + 2.8 + 2.4 + 2.4 + 2.9 + 3.0 + 2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1 + 2.3 + 1.9 + 2.0 + 2.9 + 2.1 + 1.6 + 2.9, 2.6 + 2.6 + 2.3 + 2.5 + 2.7 + 2.8 + 2.2 + 2.4 + 2.8 + 3.0 + 2.3 + 2.8 + 2.5 + 2.6 + 2.8 + 2.4 + 2.4 + 3.1 + 3.2 + 2.0) |
| Sentence of 5 words | 6 | 1 | 6 | f16 | Python + FlashInfer | 860 (671, 635, 659, 860, 768, 670) | 7.0 | 87% | 20.0 GB | 13.6 (2.4, 2.1, 2.1, 2.7, 2.0, 2.2) |
| Sentence of 5 words | 6 | 2 | 12 | f16 | Python + FlashInfer | 878 (848, 853, 854, 855, 878, 876) | 13.7 | 92% | 20.0 GB | 27.1 (2.4 + 2.1, 2.1 + 2.3, 2.2 + 2.4, 1.8 + 2.3, 2.2 + 2.0, 2.7 + 2.6) |
| Sentence of 5 words | 6 | 3 | 18 | f16 | Python + FlashInfer | 1177 (1083, 1137, 1105, 1160, 1177, 1169) | 15.3 | 95% | 20.0 GB | 42.2 (2.2 + 2.1 + 2.1, 2.4 + 2.0 + 2.4, 2.1 + 2.7 + 2.2, 1.7 + 3.2 + 2.5, 2.0 + 3.1 + 2.5, 2.3 + 2.3 + 2.5) |
| Sentence of 5 words | 6 | 4 | 24 | f16 | Python + FlashInfer | 1464 (1422, 1426, 1444, 1460, 1447, 1464) | 16.4 | 96% | 20.0 GB | 58.5 (2.4 + 2.1 + 2.0 + 2.7, 2.2 + 2.4 + 1.7 + 2.7, 2.2 + 2.0 + 2.9 + 2.6, 1.5 + 2.7 + 2.9 + 2.4, 2.3 + 2.9 + 2.7 + 2.9, 2.4 + 2.6 + 2.8 + 2.3) |
| Sentence of 5 words | 6 | 5 | 30 | f16 | Python + FlashInfer | 1667 (1621, 1619, 1659, 1656, 1667, 1638) | 18.0 | 96% | 20.0 GB | 73.6 (2.4 + 2.1 + 2.1 + 2.7 + 2.0, 2.4 + 2.1 + 2.2 + 2.2 + 1.6, 3.1 + 2.2 + 2.0 + 3.1 + 2.6, 2.6 + 2.3 + 2.7 + 2.7 + 2.9, 2.4 + 2.5 + 2.5 + 3.0 + 2.6, 2.3 + 2.5 + 2.6 + 2.8 + 2.6) |
| Sentence of 5 words | 6 | 6 | 36 | f16 | Python + FlashInfer | 1949 (1809, 1949, 1927, 1918, 1920, 1892) | 18.5 | 96% | 20.0 GB | 89.2 (2.4 + 2.1 + 2.1 + 2.3 + 2.2 + 2.4, 1.7 + 2.7 + 2.2 + 2.0 + 2.9 + 2.6, 2.1 + 3.1 + 2.5 + 2.6 + 2.3 + 2.6, 2.5 + 2.9 + 2.4 + 2.7 + 2.6 + 2.5, 2.5 + 2.6 + 2.5 + 2.6 + 2.8 + 2.6, 2.3 + 3.2 + 3.2 + 2.4 + 2.1 + 2.0) |
| Sentence of 5 words | 6 | 7 | 42 | f16 | Python + FlashInfer | 2249 (2160, 2227, 2237, 2249, 2240, 2206) | 18.7 | 97% | 20.0 GB | 100.5 (2.4 + 2.1 + 1.9 + 2.7 + 2.2 + 2.4 + 2.1, 2.3 + 2.2 + 1.6 + 3.2 + 2.2 + 2.1 + 2.7, 2.6 + 2.6 + 1.8 + 3.1 + 2.2 + 2.9 + 2.4, 2.7 + 2.8 + 2.6 + 2.6 + 2.1 + 2.5 + 2.6, 2.8 + 2.6 + 2.3 + 2.8 + 3.1 + 2.1 + 2.1, 1.8 + 2.3 + 2.1 + 2.3 + 2.1 + 2.4 + 2.2) |
| Sentence of 5 words | 6 | 8 | 48 | f16 | Python + FlashInfer | 2576 (2537, 2560, 2574, 2574, 2533, 2576) | 18.6 | 96% | 20.0 GB | 117.6 (2.4 + 2.1 + 2.1 + 2.7 + 2.0 + 2.4 + 2.1 + 2.4, 2.2 + 2.0 + 3.2 + 2.5 + 2.1 + 3.1 + 2.8 + 2.6, 2.3 + 2.3 + 2.7 + 2.9 + 2.4 + 2.7 + 2.8 + 2.9, 2.1 + 2.8 + 2.5 + 2.6 + 2.6 + 2.6 + 2.4 + 3.1, 2.8 + 1.9 + 2.1 + 1.8 + 2.7 + 2.2 + 2.4 + 2.1, 2.6 + 2.0 + 2.0 + 2.8 + 2.6 + 2.1 + 2.7 + 2.6) |
| Sentence of 5 words | 6 | 9 | 54 | f16 | Python + FlashInfer | 2898 (2733, 2898, 2898, 2878, 2882, 2897) | 18.6 | 97% | 20.3 GB | 133.0 (2.1 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1 + 2.7 + 2.2, 1.7 + 2.9 + 2.5 + 2.1 + 3.0 + 2.6 + 2.6 + 2.3 + 2.6, 2.7 + 2.8 + 2.4 + 2.7 + 2.6 + 2.5 + 2.6 + 2.8 + 2.5, 2.5 + 2.8 + 2.6 + 2.4 + 2.6 + 3.0 + 2.4 + 2.1 + 2.1, 2.7 + 2.2 + 2.4 + 1.8 + 2.5 + 2.2 + 1.5 + 2.8 + 2.6, 2.1 + 3.1 + 2.6 + 2.5 + 2.3 + 3.0 + 2.7 + 2.9 + 2.1) |
| Sentence of 5 words | 6 | 10 | 60 | f16 | Python + FlashInfer | 3251 (3080, 3251, 3165, 3125, 3155, 3224) | 18.5 | 97% | 20.7 GB | 145.7 (2.4 + 2.0 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1 + 2.4 + 2.2 + 2.0, 2.7 + 2.6 + 1.7 + 2.7 + 2.6 + 2.3 + 2.3 + 2.6 + 2.4 + 2.6, 2.4 + 2.7 + 2.5 + 2.7 + 2.6 + 2.2 + 2.5 + 2.6 + 2.6 + 2.6, 2.4 + 3.1 + 3.2 + 2.4 + 2.1 + 2.1 + 2.4 + 2.2 + 2.1 + 1.7, 2.3 + 2.2 + 2.0 + 2.8 + 2.6 + 2.1 + 3.0 + 3.1 + 2.6 + 2.3, 2.7 + 2.2 + 2.9 + 2.4 + 2.4 + 2.5 + 3.0 + 2.1 + 2.1 + 2.3) |
| Sentence of 5 words | 1 | 1 | 1 | bf16 | Python + FlashInfer | failed: bf16 not supported: upstream `apply_flashinfer` plans attention as f16 only | – | – | – | – |
| Sentence of 5 words | 1 | 1 | 1 | f16 | Python + FlashInfer + CUDA graphs | 184 | 5.4 | 71% | 3.5 GB | 2.4 (2.4) |
| Sentence of 5 words | 1 | 2 | 2 | f16 | Python + FlashInfer + CUDA graphs | 311 | 6.4 | 69% | 3.6 GB | 4.5 (2.4 + 2.1) |
| Sentence of 5 words | 1 | 3 | 3 | f16 | Python + FlashInfer + CUDA graphs | 413 | 7.3 | 80% | 3.7 GB | 6.6 (2.4 + 2.1 + 2.1) |
| Sentence of 5 words | 1 | 4 | 4 | f16 | Python + FlashInfer + CUDA graphs | 559 | 7.2 | 82% | 3.8 GB | 9.1 (2.4 + 2.1 + 1.8 + 2.7) |
| Sentence of 5 words | 1 | 5 | 5 | f16 | Python + FlashInfer + CUDA graphs | 681 | 7.3 | 87% | 4.0 GB | 10.9 (2.1 + 2.1 + 2.0 + 2.5 + 2.2) |
| Sentence of 5 words | 1 | 6 | 6 | f16 | Python + FlashInfer + CUDA graphs | 808 | 7.4 | 87% | 4.2 GB | 13.8 (2.4 + 1.9 + 2.1 + 2.7 + 2.2 + 2.4) |
| Sentence of 5 words | 1 | 7 | 7 | f16 | Python + FlashInfer + CUDA graphs | 944 | 7.4 | 87% | 4.4 GB | 15.0 (2.4 + 2.1 + 1.8 + 2.5 + 2.2 + 2.1 + 1.8) |
| Sentence of 5 words | 1 | 8 | 8 | f16 | Python + FlashInfer + CUDA graphs | 1064 | 7.5 | 85% | 4.7 GB | 18.1 (2.4 + 2.1 + 1.8 + 2.7 + 2.0 + 2.4 + 2.1 + 2.5) |
| Sentence of 5 words | 1 | 9 | 9 | f16 | Python + FlashInfer + CUDA graphs | 1188 | 7.6 | 87% | 5.0 GB | 20.7 (2.4 + 2.1 + 2.1 + 2.7 + 2.0 + 2.4 + 2.1 + 2.6 + 2.2) |
| Sentence of 5 words | 1 | 10 | 10 | f16 | Python + FlashInfer + CUDA graphs | 1323 | 7.6 | 87% | 5.3 GB | 21.4 (2.4 + 2.1 + 1.8 + 2.5 + 2.2 + 2.4 + 1.7 + 2.2 + 2.2 + 1.9) |
| Sentence of 5 words | 1 | 11 | 11 | f16 | Python + FlashInfer + CUDA graphs | 1619 | 6.8 | 89% | 5.7 GB | 25.2 (2.4 + 2.0 + 2.1 + 2.7 + 2.2 + 2.4 + 1.8 + 2.3 + 2.0 + 2.0 + 3.2) |
| Sentence of 5 words | 1 | 12 | 12 | f16 | Python + FlashInfer + CUDA graphs | 1751 | 6.9 | 89% | 6.1 GB | 27.5 (2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1 + 2.7 + 2.2 + 1.5 + 2.8 + 2.3) |
| Sentence of 5 words | 2 | 1 | 2 | f16 | Python + FlashInfer + CUDA graphs | 336 (336, 332) | 6.0 | 91% | 6.9 GB | 4.6 (2.4, 2.1) |
| Sentence of 5 words | 2 | 2 | 4 | f16 | Python + FlashInfer + CUDA graphs | 574 (574, 564) | 7.0 | 94% | 7.1 GB | 8.9 (2.4 + 2.1, 2.1 + 2.3) |
| Sentence of 5 words | 2 | 3 | 6 | f16 | Python + FlashInfer + CUDA graphs | 783 (783, 781) | 7.7 | 95% | 7.4 GB | 13.8 (2.4 + 2.1 + 2.1, 2.7 + 2.2 + 2.3) |
| Sentence of 5 words | 2 | 4 | 8 | f16 | Python + FlashInfer + CUDA graphs | 1121 (1113, 1121) | 7.1 | 95% | 7.6 GB | 17.8 (2.4 + 2.1 + 1.8 + 2.7, 2.2 + 2.2 + 1.7 + 2.7) |
| Sentence of 5 words | 2 | 5 | 10 | f16 | Python + FlashInfer + CUDA graphs | 1347 (1339, 1347) | 7.4 | 95% | 8.0 GB | 21.8 (2.1 + 2.1 + 2.0 + 2.5 + 2.2, 2.4 + 1.6 + 2.7 + 2.2 + 1.9) |
| Sentence of 5 words | 2 | 6 | 12 | f16 | Python + FlashInfer + CUDA graphs | 1694 (1600, 1694) | 7.1 | 94% | 8.5 GB | 27.7 (2.4 + 1.9 + 2.1 + 2.7 + 2.2 + 2.4, 1.6 + 2.3 + 2.2 + 2.0 + 3.2 + 2.6) |
| Sentence of 5 words | 3 | 1 | 3 | f16 | Python + FlashInfer + CUDA graphs | 505 (498, 505, 492) | 5.9 | 92% | 10.3 GB | 6.6 (2.4, 2.1, 2.0) |
| Sentence of 5 words | 3 | 2 | 6 | f16 | Python + FlashInfer + CUDA graphs | 839 (818, 839, 816) | 7.1 | 93% | 10.7 GB | 13.5 (2.4 + 2.1, 2.1 + 2.3, 2.2 + 2.4) |
| Sentence of 5 words | 3 | 3 | 9 | f16 | Python + FlashInfer + CUDA graphs | 1178 (1178, 1145, 1142) | 7.6 | 94% | 11.0 GB | 20.2 (2.4 + 2.1 + 2.1, 2.7 + 2.2 + 2.3, 1.6 + 2.7 + 2.0) |
| Sentence of 5 words | 3 | 4 | 12 | f16 | Python + FlashInfer + CUDA graphs | 1620 (1600, 1614, 1620) | 7.4 | 96% | 11.5 GB | 27.1 (2.4 + 2.1 + 1.8 + 2.7, 2.2 + 2.2 + 1.7 + 2.7, 2.2 + 1.8 + 2.8 + 2.6) |
| Sentence of 5 words | 3 | 5 | 15 | f16 | Python + FlashInfer + CUDA graphs | 2051 (1992, 1982, 2051) | 7.3 | 96% | 12.1 GB | 34.8 (2.1 + 2.1 + 2.0 + 2.5 + 2.2, 2.4 + 1.6 + 2.7 + 2.2 + 1.9, 2.7 + 2.1 + 2.1 + 3.1 + 2.9) |
| Sentence of 5 words | 3 | 6 | 18 | f16 | Python + FlashInfer + CUDA graphs | 2458 (2408, 2458, 2319) | 7.3 | 96% | 12.7 GB | 43.0 (2.4 + 1.9 + 2.1 + 2.7 + 2.2 + 2.4, 1.6 + 2.3 + 2.2 + 2.0 + 3.2 + 2.6, 1.9 + 3.1 + 2.6 + 2.6 + 2.0 + 3.1) |
| Sentence of 5 words | 4 | 1 | 4 | f16 | Python + FlashInfer + CUDA graphs | 633 (633, 632, 615, 630) | 6.3 | 91% | 13.8 GB | 8.8 (2.4, 2.1, 2.0, 2.3) |
| Sentence of 5 words | 4 | 2 | 8 | f16 | Python + FlashInfer + CUDA graphs | 1075 (1075, 1055, 1072, 1055) | 7.4 | 96% | 14.2 GB | 17.6 (2.4 + 2.1, 2.1 + 2.3, 2.2 + 2.4, 1.8 + 2.3) |
| Sentence of 5 words | 4 | 3 | 12 | f16 | Python + FlashInfer + CUDA graphs | 1566 (1512, 1547, 1535, 1566) | 7.7 | 94% | 14.7 GB | 27.7 (2.4 + 2.1 + 2.1, 2.7 + 2.2 + 2.3, 1.6 + 2.7 + 2.0, 2.0 + 2.9 + 2.6) |
| Sentence of 5 words | 4 | 4 | 16 | f16 | Python + FlashInfer + CUDA graphs | 2139 (2117, 2084, 2139, 2087) | 7.5 | 96% | 15.3 GB | 36.9 (2.4 + 2.1 + 1.8 + 2.7, 2.2 + 2.2 + 1.7 + 2.7, 2.2 + 1.8 + 2.8 + 2.6, 1.6 + 3.1 + 2.5 + 2.6) |
| Sentence of 5 words | 4 | 5 | 20 | f16 | Python + FlashInfer + CUDA graphs | 2675 (2609, 2610, 2675, 2549) | 7.5 | 97% | 16.1 GB | 47.6 (2.1 + 2.1 + 2.0 + 2.5 + 2.2, 2.4 + 1.6 + 2.7 + 2.2 + 1.9, 2.7 + 2.1 + 2.1 + 3.1 + 2.9, 2.6 + 2.3 + 2.6 + 2.4 + 2.9) |
| Sentence of 5 words | 4 | 6 | 24 | f16 | Python + FlashInfer + CUDA graphs | 3194 (3059, 3194, 3104, 3131) | 7.5 | 97% | 16.9 GB | 58.9 (2.4 + 1.9 + 2.1 + 2.7 + 2.2 + 2.4, 1.6 + 2.3 + 2.2 + 2.0 + 3.2 + 2.6, 1.9 + 3.1 + 2.6 + 2.6 + 2.0 + 3.1, 2.5 + 2.9 + 2.4 + 2.5 + 2.8 + 2.9) |
| Sentence of 5 words | 1 | 1 | 1 | bf16 | Python + FlashInfer + CUDA graphs | 198 | 5.0 | 67% | 3.5 GB | 2.4 (2.4) |
| Sentence of 5 words | 1 | 2 | 2 | bf16 | Python + FlashInfer + CUDA graphs | 315 | 6.4 | 82% | 3.6 GB | 4.6 (2.4 + 2.1) |
| Sentence of 5 words | 1 | 3 | 3 | bf16 | Python + FlashInfer + CUDA graphs | 451 | 6.6 | 75% | 3.7 GB | 6.6 (2.4 + 2.1 + 2.1) |
| Sentence of 5 words | 1 | 4 | 4 | bf16 | Python + FlashInfer + CUDA graphs | 580 | 6.9 | 82% | 3.8 GB | 8.7 (2.4 + 1.7 + 1.8 + 2.7) |
| Sentence of 5 words | 1 | 5 | 5 | bf16 | Python + FlashInfer + CUDA graphs | 707 | 7.1 | 83% | 4.0 GB | 11.2 (2.4 + 2.1 + 2.0 + 2.5 + 2.2) |
| Sentence of 5 words | 1 | 6 | 6 | bf16 | Python + FlashInfer + CUDA graphs | 882 | 6.8 | 79% | 4.2 GB | 13.6 (2.4 + 2.1 + 2.1 + 2.4 + 2.1 + 2.4) |
| Sentence of 5 words | 1 | 7 | 7 | bf16 | Python + FlashInfer + CUDA graphs | 947 | 7.4 | 83% | 4.4 GB | 15.5 (2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.1 + 1.8) |
| Sentence of 5 words | 1 | 8 | 8 | bf16 | Python + FlashInfer + CUDA graphs | 1071 | 7.5 | 97% | 4.7 GB | 18.1 (2.4 + 2.1 + 2.1 + 2.7 + 2.0 + 2.4 + 2.1 + 2.2) |
| Sentence of 5 words | 1 | 9 | 9 | bf16 | Python + FlashInfer + CUDA graphs | 1304 | 6.9 | 81% | 5.0 GB | 20.2 (2.4 + 1.9 + 2.1 + 2.6 + 2.0 + 2.4 + 2.1 + 2.6 + 2.2) |
| Sentence of 5 words | 1 | 10 | 10 | bf16 | Python + FlashInfer + CUDA graphs | 1318 | 7.6 | 87% | 5.3 GB | 21.2 (2.4 + 1.9 + 2.1 + 2.4 + 2.2 + 2.4 + 1.9 + 2.2 + 2.2 + 1.5) |
| Sentence of 5 words | 1 | 11 | 11 | bf16 | Python + FlashInfer + CUDA graphs | 1618 | 6.8 | 89% | 5.7 GB | 24.8 (2.4 + 2.1 + 1.8 + 2.3 + 2.2 + 2.4 + 2.1 + 2.3 + 2.1 + 2.0 + 3.1) |
| Sentence of 5 words | 1 | 12 | 12 | bf16 | Python + FlashInfer + CUDA graphs | 1748 | 6.9 | 88% | 6.1 GB | 28.2 (2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 1.8 + 2.7 + 2.2 + 2.0 + 3.0 + 2.6) |
| Sentence of 5 words | 1 | 13 | 13 | bf16 | Python + FlashInfer + CUDA graphs | 1903 | 6.8 | 86% | 6.6 GB | 29.7 (2.4 + 2.1 + 2.0 + 2.7 + 2.2 + 2.4 + 2.1 + 2.4 + 2.1 + 2.0 + 2.8 + 2.5 + 1.8) |
| Sentence of 5 words | 2 | 1 | 2 | bf16 | Python + FlashInfer + CUDA graphs | 331 (329, 331) | 6.0 | 91% | 6.9 GB | 4.6 (2.4, 2.1) |
| Sentence of 5 words | 2 | 2 | 4 | bf16 | Python + FlashInfer + CUDA graphs | 573 (569, 573) | 7.0 | 100% | 7.1 GB | 9.3 (2.4 + 2.1, 2.0 + 2.7) |
| Sentence of 5 words | 2 | 3 | 6 | bf16 | Python + FlashInfer + CUDA graphs | 781 (763, 781) | 7.7 | 94% | 7.4 GB | 13.5 (2.4 + 2.1 + 2.1, 2.7 + 2.2 + 2.0) |
| Sentence of 5 words | 2 | 4 | 8 | bf16 | Python + FlashInfer + CUDA graphs | 1097 (1058, 1097) | 7.3 | 94% | 7.6 GB | 17.2 (2.4 + 1.7 + 1.8 + 2.7, 2.2 + 2.2 + 1.9 + 2.2) |
| Sentence of 5 words | 2 | 5 | 10 | bf16 | Python + FlashInfer + CUDA graphs | 1337 (1336, 1337) | 7.5 | 97% | 8.0 GB | 22.1 (2.4 + 2.1 + 2.0 + 2.5 + 2.2, 2.4 + 1.6 + 2.7 + 2.2 + 1.9) |
| Sentence of 5 words | 2 | 6 | 12 | bf16 | Python + FlashInfer + CUDA graphs | 1689 (1588, 1689) | 7.1 | 94% | 8.5 GB | 28.1 (2.4 + 2.1 + 2.1 + 2.4 + 2.1 + 2.4, 1.7 + 2.7 + 2.2 + 2.0 + 3.2 + 2.6) |
| Sentence of 5 words | 3 | 1 | 3 | bf16 | Python + FlashInfer + CUDA graphs | 494 (492, 494, 480) | 6.1 | 92% | 10.3 GB | 6.3 (2.4, 2.1, 1.8) |
| Sentence of 5 words | 3 | 2 | 6 | bf16 | Python + FlashInfer + CUDA graphs | 825 (816, 795, 825) | 7.3 | 95% | 10.7 GB | 13.9 (2.4 + 2.1, 2.0 + 2.7, 2.2 + 2.4) |
| Sentence of 5 words | 3 | 3 | 9 | bf16 | Python + FlashInfer + CUDA graphs | 1144 (1109, 1130, 1144) | 7.9 | 96% | 11.0 GB | 20.5 (2.4 + 2.1 + 2.1, 2.7 + 2.2 + 2.0, 2.1 + 2.7 + 2.2) |
| Sentence of 5 words | 3 | 4 | 12 | bf16 | Python + FlashInfer + CUDA graphs | 1632 (1605, 1632, 1624) | 7.4 | 96% | 11.5 GB | 26.7 (2.4 + 1.7 + 1.8 + 2.7, 2.2 + 2.2 + 1.9 + 2.2, 2.2 + 2.0 + 2.8 + 2.6) |
| Sentence of 5 words | 3 | 5 | 15 | bf16 | Python + FlashInfer + CUDA graphs | 2058 (2018, 1952, 2058) | 7.3 | 95% | 12.1 GB | 35.2 (2.4 + 2.1 + 2.0 + 2.5 + 2.2, 2.4 + 1.6 + 2.7 + 2.2 + 1.9, 2.7 + 2.5 + 2.1 + 3.1 + 2.7) |
| Sentence of 5 words | 3 | 6 | 18 | bf16 | Python + FlashInfer + CUDA graphs | 2421 (2352, 2421, 2374) | 7.4 | 97% | 12.7 GB | 43.8 (2.4 + 2.1 + 2.1 + 2.4 + 2.1 + 2.4, 1.7 + 2.7 + 2.2 + 2.0 + 3.2 + 2.6, 2.1 + 2.7 + 3.1 + 2.3 + 2.3 + 3.1) |
| Sentence of 5 words | 1 | 1 | 1 | f16 | C++ (`perf_mode=flash_attention`) | 106 | 9.4 | 81% | 2.3 GB | 2.4 (2.4) |
| Sentence of 5 words | 1 | 2 | 2 | f16 | C++ (`perf_mode=flash_attention`) | 182 | 11.0 | 88% | 2.3 GB | 4.6 (2.4 + 2.1) |
| Sentence of 5 words | 1 | 3 | 3 | f16 | C++ (`perf_mode=flash_attention`) | 232 | 12.9 | 89% | 2.3 GB | 6.6 (2.4 + 2.1 + 2.1) |
| Sentence of 5 words | 1 | 4 | 4 | f16 | C++ (`perf_mode=flash_attention`) | 319 | 12.5 | 90% | 2.3 GB | 9.4 (2.4 + 2.1 + 2.1 + 2.7) |
| Sentence of 5 words | 1 | 5 | 5 | f16 | C++ (`perf_mode=flash_attention`) | 401 | 12.5 | 87% | 2.3 GB | 11.5 (2.4 + 2.1 + 2.1 + 2.7 + 2.2) |
| Sentence of 5 words | 1 | 6 | 6 | f16 | C++ (`perf_mode=flash_attention`) | 468 | 12.8 | 88% | 2.3 GB | 14.0 (2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4) |
| Sentence of 5 words | 2 | 1 | 2 | f16 | C++ (`perf_mode=flash_attention`) | 210 (210, 204) | 9.5 | 94% | 4.5 GB | 4.6 (2.4, 2.1) |
| Sentence of 5 words | 2 | 2 | 4 | f16 | C++ (`perf_mode=flash_attention`) | 368 (363, 368) | 10.9 | 94% | 4.5 GB | 9.4 (2.4 + 2.1, 2.1 + 2.7) |
| Sentence of 5 words | 2 | 3 | 6 | f16 | C++ (`perf_mode=flash_attention`) | 493 (489, 493) | 12.2 | 96% | 4.6 GB | 14.0 (2.4 + 2.1 + 2.1, 2.7 + 2.2 + 2.4) |
| Sentence of 5 words | 2 | 4 | 8 | f16 | C++ (`perf_mode=flash_attention`) | 669 (669, 665) | 12.0 | 96% | 4.6 GB | 18.8 (2.4 + 2.1 + 2.1 + 2.7, 2.2 + 2.4 + 2.1 + 2.7) |
| Sentence of 5 words | 2 | 5 | 10 | f16 | C++ (`perf_mode=flash_attention`) | 806 (803, 806) | 12.4 | 96% | 4.6 GB | 22.4 (2.4 + 2.1 + 2.1 + 2.7 + 2.2, 2.4 + 2.1 + 2.7 + 2.2 + 1.5) |
| Sentence of 5 words | 2 | 6 | 12 | f16 | C++ (`perf_mode=flash_attention`) | 994 (984, 994) | 12.1 | 96% | 4.7 GB | 27.7 (2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4, 2.1 + 2.7 + 2.2 + 1.5 + 2.7 + 2.6) |
| Sentence of 5 words | 2 | 7 | 14 | f16 | C++ (`perf_mode=flash_attention`) | 1195 (1191, 1195) | 11.7 | 96% | 4.7 GB | 32.9 (2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1, 2.7 + 2.2 + 1.5 + 2.7 + 2.6 + 2.1 + 3.1) |
| Sentence of 5 words | 2 | 8 | 16 | f16 | C++ (`perf_mode=flash_attention`) | 1391 (1385, 1391) | 11.5 | 96% | 4.8 GB | 39.1 (2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1 + 2.7, 2.2 + 1.5 + 3.2 + 2.6 + 2.1 + 3.1 + 3.1 + 2.6) |
| Sentence of 5 words | 1 | 1 | 1 | bf16 | C++ (`perf_mode=flash_attention`) | 108 | 9.2 | 85% | 2.3 GB | 2.4 (2.4) |
| Sentence of 5 words | 1 | 2 | 2 | bf16 | C++ (`perf_mode=flash_attention`) | 186 | 10.8 | 89% | 2.3 GB | 4.2 (2.1 + 2.1) |
| Sentence of 5 words | 1 | 3 | 3 | bf16 | C++ (`perf_mode=flash_attention`) | 238 | 12.6 | 93% | 2.3 GB | 6.6 (2.4 + 2.1 + 2.1) |
| Sentence of 5 words | 1 | 4 | 4 | bf16 | C++ (`perf_mode=flash_attention`) | 348 | 11.5 | 90% | 2.3 GB | 9.4 (2.4 + 2.1 + 2.1 + 2.7) |
| Sentence of 5 words | 1 | 5 | 5 | bf16 | C++ (`perf_mode=flash_attention`) | 427 | 11.7 | 87% | 2.4 GB | 11.5 (2.4 + 2.1 + 2.1 + 2.7 + 2.2) |
| Sentence of 5 words | 1 | 6 | 6 | bf16 | C++ (`perf_mode=flash_attention`) | 505 | 11.9 | 91% | 2.4 GB | 14.0 (2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4) |
| Sentence of 5 words | 2 | 1 | 2 | bf16 | C++ (`perf_mode=flash_attention`) | 219 (214, 219) | 9.1 | 96% | 4.6 GB | 4.6 (2.4, 2.1) |
| Sentence of 5 words | 2 | 2 | 4 | bf16 | C++ (`perf_mode=flash_attention`) | 377 (373, 377) | 10.6 | 95% | 4.6 GB | 9.1 (2.1 + 2.1, 2.1 + 2.7) |
| Sentence of 5 words | 2 | 3 | 6 | bf16 | C++ (`perf_mode=flash_attention`) | 520 (516, 520) | 11.5 | 96% | 4.6 GB | 14.0 (2.4 + 2.1 + 2.1, 2.7 + 2.2 + 2.4) |
| Sentence of 5 words | 2 | 4 | 8 | bf16 | C++ (`perf_mode=flash_attention`) | 736 (736, 731) | 10.9 | 96% | 4.7 GB | 18.8 (2.4 + 2.1 + 2.1 + 2.7, 2.2 + 2.4 + 2.1 + 2.7) |
| Sentence of 5 words | 2 | 5 | 10 | bf16 | C++ (`perf_mode=flash_attention`) | 882 (878, 882) | 11.3 | 95% | 4.7 GB | 22.4 (2.4 + 2.1 + 2.1 + 2.7 + 2.2, 2.4 + 2.1 + 2.7 + 2.2 + 1.4) |
| Sentence of 5 words | 2 | 6 | 12 | bf16 | C++ (`perf_mode=flash_attention`) | 1097 (1097, 1095) | 10.9 | 96% | 4.7 GB | 28.2 (2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4, 2.1 + 2.7 + 2.2 + 1.4 + 3.2 + 2.6) |
| Sentence of 5 words | 1 | 1 | 1 | f16 | C++ (`perf_mode=off`) | 125 | 8.0 | 88% | 2.2 GB | 2.4 (2.4) |
| Sentence of 5 words | 1 | 2 | 2 | f16 | C++ (`perf_mode=off`) | 224 | 8.9 | 85% | 2.2 GB | 4.6 (2.4 + 2.1) |
| Sentence of 5 words | 1 | 3 | 3 | f16 | C++ (`perf_mode=off`) | 297 | 10.1 | 92% | 2.3 GB | 6.6 (2.4 + 2.1 + 2.1) |
| Sentence of 5 words | 1 | 4 | 4 | f16 | C++ (`perf_mode=off`) | 425 | 9.4 | 88% | 2.3 GB | 9.4 (2.4 + 2.1 + 2.1 + 2.7) |
| Sentence of 5 words | 1 | 5 | 5 | f16 | C++ (`perf_mode=off`) | 512 | 9.8 | 92% | 2.3 GB | 11.5 (2.4 + 2.1 + 2.1 + 2.7 + 2.2) |
| Sentence of 5 words | 1 | 6 | 6 | f16 | C++ (`perf_mode=off`) | 602 | 10.0 | 91% | 2.3 GB | 14.0 (2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4) |
| Sentence of 5 words | 2 | 1 | 2 | f16 | C++ (`perf_mode=off`) | 254 (254, 252) | 7.9 | 93% | 4.4 GB | 4.6 (2.4, 2.1) |
| Sentence of 5 words | 2 | 2 | 4 | f16 | C++ (`perf_mode=off`) | 462 (462, 459) | 8.7 | 94% | 4.5 GB | 9.4 (2.4 + 2.1, 2.1 + 2.7) |
| Sentence of 5 words | 2 | 3 | 6 | f16 | C++ (`perf_mode=off`) | 642 (638, 642) | 9.3 | 96% | 4.5 GB | 14.0 (2.4 + 2.1 + 2.1, 2.7 + 2.2 + 2.4) |
| Sentence of 5 words | 2 | 4 | 8 | f16 | C++ (`perf_mode=off`) | 857 (857, 853) | 9.3 | 97% | 4.6 GB | 18.8 (2.4 + 2.1 + 2.1 + 2.7, 2.2 + 2.4 + 2.1 + 2.7) |
| Sentence of 5 words | 2 | 5 | 10 | f16 | C++ (`perf_mode=off`) | 1049 (1049, 1038) | 9.5 | 97% | 4.6 GB | 23.0 (2.4 + 2.1 + 2.1 + 2.7 + 2.2, 2.4 + 2.1 + 2.7 + 2.2 + 2.0) |
| Sentence of 5 words | 2 | 6 | 12 | f16 | C++ (`perf_mode=off`) | 1288 (1288, 1285) | 9.3 | 95% | 4.7 GB | 28.1 (2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4, 2.1 + 2.7 + 2.2 + 1.4 + 3.1 + 2.6) |
| Sentence of 5 words | 2 | 7 | 14 | f16 | C++ (`perf_mode=off`) | 1537 (1530, 1537) | 9.1 | 96% | 4.7 GB | 33.2 (2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1, 2.7 + 2.2 + 1.4 + 3.1 + 2.6 + 2.1 + 3.1) |
| Sentence of 5 words | 2 | 8 | 16 | f16 | C++ (`perf_mode=off`) | 1783 (1777, 1783) | 9.0 | 96% | 4.8 GB | 38.5 (2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1 + 2.7, 2.2 + 1.4 + 2.7 + 2.6 + 2.0 + 3.1 + 3.1 + 2.6) |
| Sentence of 5 words | 1 | 1 | 1 | bf16 | C++ (`perf_mode=off`) | 132 | 7.6 | 79% | 2.2 GB | 2.4 (2.4) |
| Sentence of 5 words | 1 | 2 | 2 | bf16 | C++ (`perf_mode=off`) | 226 | 8.9 | 88% | 2.3 GB | 4.6 (2.4 + 2.1) |
| Sentence of 5 words | 1 | 3 | 3 | bf16 | C++ (`perf_mode=off`) | 319 | 9.4 | 89% | 2.3 GB | 6.6 (2.4 + 2.1 + 2.1) |
| Sentence of 5 words | 1 | 4 | 4 | bf16 | C++ (`perf_mode=off`) | 449 | 8.9 | 91% | 2.3 GB | 9.4 (2.4 + 2.1 + 2.1 + 2.7) |
| Sentence of 5 words | 1 | 5 | 5 | bf16 | C++ (`perf_mode=off`) | 545 | 9.2 | 91% | 2.3 GB | 11.5 (2.4 + 2.1 + 2.1 + 2.7 + 2.2) |
| Sentence of 5 words | 1 | 6 | 6 | bf16 | C++ (`perf_mode=off`) | 649 | 9.2 | 91% | 2.4 GB | 14.0 (2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4) |
| Sentence of 5 words | 2 | 1 | 2 | bf16 | C++ (`perf_mode=off`) | 265 (265, 265) | 7.6 | 95% | 4.5 GB | 4.6 (2.4, 2.1) |
| Sentence of 5 words | 2 | 2 | 4 | bf16 | C++ (`perf_mode=off`) | 469 (469, 466) | 8.5 | 94% | 4.5 GB | 9.4 (2.4 + 2.1, 2.1 + 2.7) |
| Sentence of 5 words | 2 | 3 | 6 | bf16 | C++ (`perf_mode=off`) | 669 (669, 669) | 9.0 | 95% | 4.6 GB | 14.0 (2.4 + 2.1 + 2.1, 2.7 + 2.2 + 2.4) |
| Sentence of 5 words | 2 | 4 | 8 | bf16 | C++ (`perf_mode=off`) | 927 (927, 923) | 8.6 | 96% | 4.6 GB | 18.8 (2.4 + 2.1 + 2.1 + 2.7, 2.2 + 2.4 + 2.1 + 2.7) |
| Sentence of 5 words | 2 | 5 | 10 | bf16 | C++ (`perf_mode=off`) | 1109 (1109, 1106) | 9.0 | 96% | 4.7 GB | 22.4 (2.4 + 2.1 + 2.1 + 2.7 + 2.2, 2.4 + 2.1 + 2.7 + 2.2 + 1.4) |
| Sentence of 5 words | 2 | 6 | 12 | bf16 | C++ (`perf_mode=off`) | 1375 (1357, 1375) | 8.7 | 97% | 4.7 GB | 28.2 (2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4, 2.1 + 2.7 + 2.2 + 1.4 + 3.2 + 2.6) |
| Sentence of 5 words | 2 | 7 | 14 | bf16 | C++ (`perf_mode=off`) | 1640 (1633, 1640) | 8.5 | 96% | 4.8 GB | 33.0 (2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1, 2.7 + 2.2 + 1.4 + 3.2 + 2.6 + 2.1 + 2.7) |
| Sentence of 5 words | 2 | 8 | 16 | bf16 | C++ (`perf_mode=off`) | 1876 (1869, 1876) | 8.5 | 96% | 4.8 GB | 38.5 (2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1 + 2.7, 2.2 + 1.4 + 3.2 + 2.6 + 2.1 + 3.1 + 2.6 + 2.6) |
| Sentence of 5 words | 1 | 1 | 1 | f16 | Python (PyTorch) | 973 | 1.0 | 21% | 2.6 GB | 2.4 (2.4) |
| Sentence of 5 words | 1 | 2 | 2 | f16 | Python (PyTorch) | 1035 | 1.9 | 30% | 2.6 GB | 4.4 (2.4 + 1.9) |
| Sentence of 5 words | 1 | 3 | 3 | f16 | Python (PyTorch) | 1032 | 2.9 | 38% | 2.8 GB | 6.3 (2.2 + 2.1 + 2.1) |
| Sentence of 5 words | 1 | 4 | 4 | f16 | Python (PyTorch) | 1070 | 3.7 | 43% | 2.9 GB | 8.9 (2.4 + 2.0 + 1.8 + 2.7) |
| Sentence of 5 words | 1 | 5 | 5 | f16 | Python (PyTorch) | 1092 | 4.6 | 52% | 3.1 GB | 11.4 (2.4 + 2.0 + 2.1 + 2.7 + 2.2) |
| Sentence of 5 words | 1 | 6 | 6 | f16 | Python (PyTorch) | 1118 | 5.4 | 62% | 3.3 GB | 13.0 (2.2 + 1.9 + 2.1 + 2.7 + 2.1 + 2.0) |
| Sentence of 5 words | 1 | 7 | 7 | f16 | Python (PyTorch) | 1154 | 6.1 | 70% | 3.6 GB | 15.2 (2.4 + 2.1 + 1.8 + 2.7 + 2.2 + 2.0 + 2.1) |
| Sentence of 5 words | 1 | 8 | 8 | f16 | Python (PyTorch) | 1178 | 6.8 | 79% | 3.8 GB | 18.2 (2.4 + 1.9 + 1.8 + 2.7 + 2.2 + 2.4 + 2.1 + 2.7) |
| Sentence of 5 words | 1 | 9 | 9 | f16 | Python (PyTorch) | 1234 | 7.3 | 85% | 4.2 GB | 20.3 (2.4 + 2.1 + 1.7 + 2.7 + 2.0 + 2.4 + 2.0 + 2.7 + 2.2) |
| Sentence of 5 words | 1 | 10 | 10 | f16 | Python (PyTorch) | 1368 | 7.3 | 88% | 4.5 GB | 22.5 (2.4 + 2.1 + 2.1 + 2.7 + 2.2 + 2.4 + 1.7 + 2.7 + 2.2 + 2.0) |
| Sentence of 5 words | 1 | 11 | 11 | f16 | Python (PyTorch) | 1643 | 6.7 | 88% | 5.0 GB | 23.6 (2.1 + 1.9 + 2.1 + 2.1 + 2.2 + 2.4 + 1.7 + 2.7 + 1.9 + 1.6 + 2.8) |
| Sentence of 5 words | 1 | 12 | 12 | f16 | Python (PyTorch) | 1832 | 6.6 | 88% | 5.4 GB | 26.6 (2.4 + 2.1 + 2.1 + 2.7 + 2.1 + 2.0 + 1.7 + 2.7 + 2.2 + 1.7 + 2.8 + 2.2) |
| Sentence of 5 words | 1 | 13 | 13 | f16 | Python (PyTorch) | 2013 | 6.5 | 89% | 5.9 GB | 29.7 (2.4 + 2.0 + 2.1 + 2.7 + 2.0 + 2.4 + 2.1 + 2.7 + 2.2 + 1.6 + 2.8 + 2.6 + 2.1) |
| Sentence of 5 words | 2 | 1 | 2 | f16 | Python (PyTorch) | 1093 (1052, 1093) | 1.8 | 94% | 5.2 GB | 4.6 (2.4, 2.1) |
| Sentence of 5 words | 2 | 2 | 4 | f16 | Python (PyTorch) | 1070 (1070, 1017) | 3.7 | 93% | 5.3 GB | 8.2 (2.4 + 1.9, 1.8 + 2.1) |
| Sentence of 5 words | 2 | 3 | 6 | f16 | Python (PyTorch) | 1052 (1052, 1047) | 5.7 | 93% | 5.5 GB | 13.5 (2.2 + 2.1 + 2.1, 2.7 + 2.0 + 2.4) |
| Sentence of 5 words | 2 | 4 | 8 | f16 | Python (PyTorch) | 1105 (1105, 1058) | 7.2 | 94% | 5.8 GB | 18.1 (2.4 + 2.0 + 1.8 + 2.7, 2.2 + 2.4 + 1.9 + 2.7) |
| Sentence of 5 words | 2 | 5 | 10 | f16 | Python (PyTorch) | 1327 (1327, 1312) | 7.5 | 95% | 6.2 GB | 22.2 (2.4 + 2.0 + 2.1 + 2.7 + 2.2, 2.4 + 1.9 + 2.3 + 2.2 + 2.0) |
| Sentence of 5 words | 2 | 6 | 12 | f16 | Python (PyTorch) | 1658 (1605, 1658) | 7.2 | 95% | 6.6 GB | 25.8 (2.2 + 1.9 + 2.1 + 2.7 + 2.1 + 2.0, 1.6 + 2.3 + 2.2 + 1.6 + 2.9 + 2.2) |
| Sentence of 5 words | 2 | 7 | 14 | f16 | Python (PyTorch) | 1960 (1908, 1960) | 7.1 | 95% | 7.2 GB | 31.0 (2.4 + 2.1 + 1.8 + 2.7 + 2.2 + 2.0 + 2.1, 2.2 + 1.9 + 1.6 + 2.8 + 2.6 + 1.9 + 2.8) |
| Sentence of 5 words | 2 | 8 | 16 | f16 | Python (PyTorch) | 2234 (2157, 2234) | 7.2 | 94% | 7.8 GB | 38.5 (2.4 + 1.9 + 1.8 + 2.7 + 2.2 + 2.4 + 2.1 + 2.7, 2.2 + 2.0 + 3.2 + 2.6 + 2.0 + 3.1 + 2.6 + 2.6) |
| Sentence of 5 words | 3 | 1 | 3 | f16 | Python (PyTorch) | 1297 (1297, 1196, 1176) | 2.3 | 92% | 7.8 GB | 6.6 (2.4, 2.1, 2.1) |
| Sentence of 5 words | 3 | 2 | 6 | f16 | Python (PyTorch) | 1636 (1636, 1559, 1356) | 3.7 | 89% | 7.9 GB | 12.6 (2.4 + 1.9, 1.8 + 2.1, 2.2 + 2.2) |
| Sentence of 5 words | 3 | 3 | 9 | f16 | Python (PyTorch) | 1312 (1299, 1312, 1269) | 6.9 | 94% | 8.3 GB | 19.9 (2.2 + 2.1 + 2.1, 2.7 + 2.0 + 2.4, 1.6 + 2.6 + 2.2) |
| Sentence of 5 words | 3 | 4 | 12 | f16 | Python (PyTorch) | 1635 (1593, 1598, 1635) | 7.3 | 96% | 8.7 GB | 28.0 (2.4 + 2.0 + 1.8 + 2.7, 2.2 + 2.4 + 1.9 + 2.7, 2.1 + 2.0 + 3.2 + 2.6) |
| Sentence of 5 words | 3 | 5 | 15 | f16 | Python (PyTorch) | 2019 (1965, 1958, 2019) | 7.4 | 94% | 9.3 GB | 34.9 (2.4 + 2.0 + 2.1 + 2.7 + 2.2, 2.4 + 1.9 + 2.3 + 2.2 + 2.0, 2.7 + 2.5 + 1.7 + 2.7 + 3.1) |
| Sentence of 5 words | 3 | 6 | 18 | f16 | Python (PyTorch) | 2485 (2452, 2468, 2485) | 7.2 | 97% | 10.0 GB | 41.1 (2.2 + 1.9 + 2.1 + 2.7 + 2.1 + 2.0, 1.6 + 2.3 + 2.2 + 1.6 + 2.9 + 2.2, 2.0 + 2.9 + 3.1 + 2.6 + 2.3 + 2.4) |
| Sentence of 5 words | 3 | 7 | 21 | f16 | Python (PyTorch) | 2933 (2728, 2933, 2918) | 7.2 | 97% | 10.8 GB | 48.7 (2.4 + 2.1 + 1.8 + 2.7 + 2.2 + 2.0 + 2.1, 2.2 + 1.9 + 1.6 + 2.8 + 2.6 + 1.9 + 2.8, 2.6 + 2.6 + 2.2 + 2.7 + 2.3 + 2.9 + 2.4) |
| Sentence of 5 words | 3 | 8 | 24 | f16 | Python (PyTorch) | 3319 (3224, 3319, 3278) | 7.2 | 98% | 11.7 GB | 59.5 (2.4 + 1.9 + 1.8 + 2.7 + 2.2 + 2.4 + 2.1 + 2.7, 2.2 + 2.0 + 3.2 + 2.6 + 2.0 + 3.1 + 2.6 + 2.6, 2.3 + 2.5 + 2.5 + 2.9 + 2.4 + 2.7 + 2.8 + 2.9) |
| Sentence of 5 words | 4 | 1 | 4 | f16 | Python (PyTorch) | 1382 (1281, 1382, 1365, 1244) | 2.9 | 84% | 10.4 GB | 9.3 (2.4, 2.1, 2.1, 2.7) |
| Sentence of 5 words | 4 | 2 | 8 | f16 | Python (PyTorch) | 1244 (1241, 1238, 1218, 1244) | 6.4 | 96% | 10.5 GB | 16.9 (2.4 + 1.9, 1.8 + 2.1, 2.2 + 2.2, 1.6 + 2.7) |
| Sentence of 5 words | 4 | 3 | 12 | f16 | Python (PyTorch) | 1668 (1646, 1654, 1654, 1668) | 7.2 | 87% | 11.1 GB | 27.4 (2.2 + 2.1 + 2.1, 2.7 + 2.0 + 2.4, 1.6 + 2.6 + 2.2, 1.9 + 3.0 + 2.6) |
| Sentence of 5 words | 4 | 4 | 16 | f16 | Python (PyTorch) | 2154 (2101, 2141, 2136, 2154) | 7.4 | 96% | 11.7 GB | 38.0 (2.4 + 2.0 + 1.8 + 2.7, 2.2 + 2.4 + 1.9 + 2.7, 2.1 + 2.0 + 3.2 + 2.6, 1.6 + 2.9 + 2.9 + 2.6) |
| Sentence of 5 words | 4 | 5 | 20 | f16 | Python (PyTorch) | 2668 (2620, 2502, 2638, 2668) | 7.5 | 96% | 12.5 GB | 47.7 (2.4 + 2.0 + 2.1 + 2.7 + 2.2, 2.4 + 1.9 + 2.3 + 2.2 + 2.0, 2.7 + 2.5 + 1.7 + 2.7 + 3.1, 2.4 + 2.1 + 2.8 + 2.7 + 2.9) |
| Sentence of 5 words | 4 | 6 | 24 | f16 | Python (PyTorch) | 3274 (3058, 3274, 3225, 3165) | 7.3 | 98% | 13.4 GB | 56.9 (2.2 + 1.9 + 2.1 + 2.7 + 2.1 + 2.0, 1.6 + 2.3 + 2.2 + 1.6 + 2.9 + 2.2, 2.0 + 2.9 + 3.1 + 2.6 + 2.3 + 2.4, 2.7 + 2.9 + 2.4 + 2.7 + 2.2 + 3.0) |
| Sentence of 5 words | 4 | 7 | 28 | f16 | Python (PyTorch) | 3865 (3638, 3847, 3865, 3738) | 7.2 | 98% | 14.5 GB | 65.6 (2.4 + 2.1 + 1.8 + 2.7 + 2.2 + 2.0 + 2.1, 2.2 + 1.9 + 1.6 + 2.8 + 2.6 + 1.9 + 2.8, 2.6 + 2.6 + 2.2 + 2.7 + 2.3 + 2.9 + 2.4, 2.7 + 2.4 + 2.6 + 2.6 + 2.1 + 2.4 + 2.1) |
| Sentence of 5 words | 4 | 8 | 32 | f16 | Python (PyTorch) | 4364 (4171, 4293, 4364, 4347) | 7.3 | 98% | 15.7 GB | 79.0 (2.4 + 1.9 + 1.8 + 2.7 + 2.2 + 2.4 + 2.1 + 2.7, 2.2 + 2.0 + 3.2 + 2.6 + 2.0 + 3.1 + 2.6 + 2.6, 2.3 + 2.5 + 2.5 + 2.9 + 2.4 + 2.7 + 2.8 + 2.9, 2.2 + 2.1 + 2.5 + 2.2 + 2.8 + 2.6 + 2.4 + 2.8) |
| Sentence of 5 words | 1 | 1 | 1 | bf16 | Python (PyTorch) | 944 | 1.1 | 21% | 2.6 GB | 2.2 (2.2) |
| Sentence of 5 words | 1 | 2 | 2 | bf16 | Python (PyTorch) | 983 | 2.0 | 31% | 2.6 GB | 4.4 (2.4 + 1.9) |
| Sentence of 5 words | 1 | 3 | 3 | bf16 | Python (PyTorch) | 986 | 3.0 | 38% | 2.8 GB | 6.2 (2.2 + 1.9 + 2.1) |
| Sentence of 5 words | 1 | 4 | 4 | bf16 | Python (PyTorch) | 1031 | 3.9 | 45% | 2.9 GB | 8.8 (2.4 + 1.9 + 1.7 + 2.7) |
| Sentence of 5 words | 1 | 5 | 5 | bf16 | Python (PyTorch) | 1040 | 4.8 | 54% | 3.1 GB | 11.3 (2.4 + 1.9 + 2.1 + 2.7 + 2.2) |
| Sentence of 5 words | 1 | 6 | 6 | bf16 | Python (PyTorch) | 1086 | 5.5 | 63% | 3.3 GB | 13.5 (2.2 + 1.9 + 2.0 + 2.7 + 2.2 + 2.4) |
| Sentence of 5 words | 1 | 7 | 7 | bf16 | Python (PyTorch) | 1118 | 6.3 | 72% | 3.6 GB | 15.1 (2.4 + 1.9 + 1.7 + 2.7 + 2.2 + 2.4 + 1.7) |
| Sentence of 5 words | 1 | 8 | 8 | bf16 | Python (PyTorch) | 1109 | 7.2 | 83% | 3.8 GB | 18.1 (2.1 + 1.9 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1 + 2.7) |
| Sentence of 5 words | 1 | 9 | 9 | bf16 | Python (PyTorch) | 1225 | 7.3 | 87% | 4.2 GB | 20.3 (2.4 + 1.9 + 1.8 + 2.7 + 2.2 + 2.4 + 2.1 + 2.7 + 2.1) |
| Sentence of 5 words | 1 | 10 | 10 | bf16 | Python (PyTorch) | 1369 | 7.3 | 87% | 4.5 GB | 21.5 (2.1 + 2.1 + 2.1 + 2.7 + 2.2 + 2.1 + 1.7 + 2.7 + 2.2 + 1.6) |
| Sentence of 5 words | 1 | 11 | 11 | bf16 | Python (PyTorch) | 1637 | 6.7 | 88% | 5.0 GB | 24.0 (2.4 + 1.9 + 2.0 + 2.1 + 2.2 + 2.4 + 2.1 + 2.4 + 1.9 + 1.6 + 3.0) |
| Sentence of 5 words | 1 | 12 | 12 | bf16 | Python (PyTorch) | 1864 | 6.4 | 89% | 5.4 GB | 27.3 (2.4 + 2.1 + 2.1 + 2.7 + 2.0 + 2.4 + 2.1 + 2.7 + 2.2 + 1.6 + 2.8 + 2.2) |
| Sentence of 5 words | 2 | 1 | 2 | bf16 | Python (PyTorch) | 1417 (1120, 1417) | 1.4 | 89% | 5.2 GB | 4.3 (2.2, 2.1) |
| Sentence of 5 words | 2 | 2 | 4 | bf16 | Python (PyTorch) | 1311 (1165, 1311) | 3.1 | 88% | 5.3 GB | 9.2 (2.4 + 1.9, 2.1 + 2.7) |
| Sentence of 5 words | 2 | 3 | 6 | bf16 | Python (PyTorch) | 1239 (1134, 1239) | 4.8 | 90% | 5.5 GB | 13.0 (2.2 + 1.9 + 2.1, 2.4 + 2.0 + 2.4) |
| Sentence of 5 words | 2 | 4 | 8 | bf16 | Python (PyTorch) | 1173 (1173, 1171) | 6.8 | 91% | 5.8 GB | 17.5 (2.4 + 1.9 + 1.7 + 2.7, 2.2 + 2.4 + 1.8 + 2.4) |
| Sentence of 5 words | 2 | 5 | 10 | bf16 | Python (PyTorch) | 1319 (1316, 1319) | 7.6 | 95% | 6.2 GB | 22.1 (2.4 + 1.9 + 2.1 + 2.7 + 2.2, 2.4 + 2.1 + 2.3 + 2.2 + 1.9) |
| Sentence of 5 words | 2 | 6 | 12 | bf16 | Python (PyTorch) | 1637 (1574, 1637) | 7.3 | 95% | 6.6 GB | 27.0 (2.2 + 1.9 + 2.0 + 2.7 + 2.2 + 2.4, 1.6 + 2.6 + 2.2 + 1.6 + 3.2 + 2.3) |
| Sentence of 5 words | 2 | 7 | 14 | bf16 | Python (PyTorch) | 1926 (1871, 1926) | 7.3 | 94% | 7.2 GB | 31.1 (2.4 + 1.9 + 1.7 + 2.7 + 2.2 + 2.4 + 1.7, 2.2 + 1.9 + 1.6 + 2.8 + 2.6 + 1.8 + 3.1) |
| Sentence of 5 words | 2 | 8 | 16 | bf16 | Python (PyTorch) | 2219 (2145, 2219) | 7.2 | 95% | 7.8 GB | 38.6 (2.1 + 1.9 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1 + 2.7, 2.1 + 2.0 + 3.2 + 2.6 + 2.0 + 2.9 + 3.1 + 2.6) |
| Sentence of 5 words | 3 | 1 | 3 | bf16 | Python (PyTorch) | 1247 (1247, 1160, 1084) | 2.4 | 94% | 7.8 GB | 6.4 (2.2, 2.1, 2.1) |
| Sentence of 5 words | 3 | 2 | 6 | bf16 | Python (PyTorch) | 1605 (1220, 1605, 1396) | 3.7 | 83% | 7.9 GB | 13.5 (2.4 + 1.9, 2.1 + 2.7, 2.2 + 2.2) |
| Sentence of 5 words | 3 | 3 | 9 | bf16 | Python (PyTorch) | 1457 (1457, 1346, 1223) | 6.2 | 80% | 8.3 GB | 19.0 (2.2 + 1.9 + 2.1, 2.4 + 2.0 + 2.4, 1.6 + 2.3 + 2.2) |
| Sentence of 5 words | 3 | 4 | 12 | bf16 | Python (PyTorch) | 1638 (1600, 1580, 1638) | 7.3 | 97% | 8.7 GB | 27.2 (2.4 + 1.9 + 1.7 + 2.7, 2.2 + 2.4 + 1.8 + 2.4, 2.2 + 1.7 + 3.2 + 2.6) |
| Sentence of 5 words | 3 | 5 | 15 | bf16 | Python (PyTorch) | 2003 (1990, 1909, 2003) | 7.5 | 95% | 9.3 GB | 34.4 (2.4 + 1.9 + 2.1 + 2.7 + 2.2, 2.4 + 2.1 + 2.3 + 2.2 + 1.9, 2.7 + 2.2 + 1.7 + 2.7 + 2.9) |
| Sentence of 5 words | 3 | 6 | 18 | bf16 | Python (PyTorch) | 2463 (2330, 2463, 2431) | 7.3 | 95% | 10.0 GB | 42.7 (2.2 + 1.9 + 2.0 + 2.7 + 2.2 + 2.4, 1.6 + 2.6 + 2.2 + 1.6 + 3.2 + 2.3, 2.0 + 3.1 + 3.1 + 2.6 + 2.3 + 2.6) |
| Sentence of 5 words | 3 | 7 | 21 | bf16 | Python (PyTorch) | 2863 (2824, 2805, 2863) | 7.3 | 96% | 10.8 GB | 49.3 (2.4 + 1.9 + 1.7 + 2.7 + 2.2 + 2.4 + 1.7, 2.2 + 1.9 + 1.6 + 2.8 + 2.6 + 1.8 + 3.1, 2.6 + 2.6 + 2.3 + 3.1 + 2.7 + 2.9 + 2.0) |
| Sentence of 5 words | 3 | 8 | 24 | bf16 | Python (PyTorch) | 3332 (3236, 3332, 3220) | 7.2 | 98% | 11.7 GB | 58.7 (2.1 + 1.9 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1 + 2.7, 2.1 + 2.0 + 3.2 + 2.6 + 2.0 + 2.9 + 3.1 + 2.6, 2.3 + 2.4 + 2.1 + 2.9 + 2.4 + 2.4 + 2.5 + 2.9) |
| Sentence of 5 words | 4 | 1 | 4 | bf16 | Python (PyTorch) | 1392 (1142, 1392, 1144, 1220) | 2.9 | 88% | 10.4 GB | 8.7 (2.2, 2.1, 2.1, 2.3) |
| Sentence of 5 words | 4 | 2 | 8 | bf16 | Python (PyTorch) | 1499 (1182, 1499, 1373, 1199) | 5.3 | 90% | 10.5 GB | 17.9 (2.4 + 1.9, 2.1 + 2.7, 2.2 + 2.2, 1.7 + 2.7) |
| Sentence of 5 words | 4 | 3 | 12 | bf16 | Python (PyTorch) | 1663 (1649, 1522, 1542, 1663) | 7.2 | 97% | 11.1 GB | 25.7 (2.2 + 1.9 + 2.1, 2.4 + 2.0 + 2.4, 1.6 + 2.3 + 2.2, 1.6 + 3.0 + 2.1) |
| Sentence of 5 words | 4 | 4 | 16 | bf16 | Python (PyTorch) | 2156 (2099, 2074, 2141, 2156) | 7.4 | 96% | 11.7 GB | 37.2 (2.4 + 1.9 + 1.7 + 2.7, 2.2 + 2.4 + 1.8 + 2.4, 2.2 + 1.7 + 3.2 + 2.6, 2.1 + 2.9 + 2.5 + 2.6) |
| Sentence of 5 words | 4 | 5 | 20 | bf16 | Python (PyTorch) | 2661 (2544, 2545, 2650, 2661) | 7.5 | 97% | 12.5 GB | 47.4 (2.4 + 1.9 + 2.1 + 2.7 + 2.2, 2.4 + 2.1 + 2.3 + 2.2 + 1.9, 2.7 + 2.2 + 1.7 + 2.7 + 2.9, 2.6 + 2.1 + 2.8 + 2.7 + 2.9) |
| Sentence of 5 words | 4 | 6 | 24 | bf16 | Python (PyTorch) | 3200 (3101, 3175, 3200, 3174) | 7.5 | 98% | 13.4 GB | 58.5 (2.2 + 1.9 + 2.0 + 2.7 + 2.2 + 2.4, 1.6 + 2.6 + 2.2 + 1.6 + 3.2 + 2.3, 2.0 + 3.1 + 3.1 + 2.6 + 2.3 + 2.6, 2.7 + 2.8 + 2.4 + 2.6 + 2.2 + 3.0) |
| Sentence of 5 words | 4 | 7 | 28 | bf16 | Python (PyTorch) | 3821 (3611, 3821, 3767, 3738) | 7.3 | 97% | 14.5 GB | 66.3 (2.4 + 1.9 + 1.7 + 2.7 + 2.2 + 2.4 + 1.7, 2.2 + 1.9 + 1.6 + 2.8 + 2.6 + 1.8 + 3.1, 2.6 + 2.6 + 2.3 + 3.1 + 2.7 + 2.9 + 2.0, 2.7 + 2.4 + 2.6 + 2.6 + 2.2 + 2.5 + 2.1) |
| Sentence of 5 words | 4 | 8 | 32 | bf16 | Python (PyTorch) | 4336 (4223, 4336, 4296, 4287) | 7.4 | 99% | 15.7 GB | 78.8 (2.1 + 1.9 + 2.1 + 2.7 + 2.2 + 2.4 + 2.1 + 2.7, 2.1 + 2.0 + 3.2 + 2.6 + 2.0 + 2.9 + 3.1 + 2.6, 2.3 + 2.4 + 2.1 + 2.9 + 2.4 + 2.4 + 2.5 + 2.9, 2.6 + 2.2 + 2.5 + 2.6 + 2.8 + 2.6 + 2.4 + 2.6) |
| Sentence of 10 words | 1 | 1 | 1 | f16 | Python + FlashInfer | 713 | 1.4 | 13% | 3.3 GB | 3.4 (3.4) |
| Sentence of 10 words | 1 | 2 | 2 | f16 | Python + FlashInfer | 729 | 2.7 | 21% | 3.3 GB | 7.8 (3.9 + 3.9) |
| Sentence of 10 words | 1 | 3 | 3 | f16 | Python + FlashInfer | 542 | 5.5 | 39% | 3.3 GB | 11.6 (3.3 + 4.1 + 4.2) |
| Sentence of 10 words | 1 | 4 | 4 | f16 | Python + FlashInfer | 643 | 6.2 | 38% | 3.3 GB | 15.2 (3.5 + 3.8 + 4.2 + 3.7) |
| Sentence of 10 words | 1 | 5 | 5 | f16 | Python + FlashInfer | 599 | 8.3 | 52% | 3.3 GB | 19.4 (3.3 + 4.2 + 4.2 + 3.7 + 4.0) |
| Sentence of 10 words | 1 | 6 | 6 | f16 | Python + FlashInfer | 589 | 10.2 | 60% | 3.4 GB | 23.0 (3.3 + 4.0 + 4.2 + 3.7 + 4.3 + 3.6) |
| Sentence of 10 words | 1 | 7 | 7 | f16 | Python + FlashInfer | 623 | 11.2 | 69% | 3.4 GB | 26.4 (3.3 + 3.9 + 4.2 + 3.7 + 4.3 + 3.1 + 3.8) |
| Sentence of 10 words | 1 | 8 | 8 | f16 | Python + FlashInfer | 649 | 12.3 | 75% | 3.5 GB | 30.8 (3.6 + 4.2 + 4.2 + 3.7 + 4.3 + 3.6 + 3.8 + 3.5) |
| Sentence of 10 words | 1 | 9 | 9 | f16 | Python + FlashInfer | 734 | 12.3 | 92% | 3.6 GB | 34.7 (3.7 + 3.6 + 4.2 + 3.7 + 4.0 + 3.1 + 3.8 + 3.5 + 5.2) |
| Sentence of 10 words | 1 | 10 | 10 | f16 | Python + FlashInfer | 816 | 12.3 | 68% | 3.7 GB | 39.0 (3.9 + 3.6 + 4.2 + 3.4 + 4.2 + 3.4 + 3.8 + 3.7 + 4.6 + 4.3) |
| Sentence of 10 words | 1 | 11 | 11 | f16 | Python + FlashInfer | 896 | 12.3 | 74% | 3.7 GB | 43.1 (3.4 + 4.2 + 4.2 + 3.7 + 4.3 + 3.6 + 3.6 + 3.5 + 4.5 + 4.0 + 4.2) |
| Sentence of 10 words | 2 | 1 | 2 | f16 | Python + FlashInfer | 527 (527, 517) | 3.8 | 92% | 6.7 GB | 7.0 (3.4, 3.6) |
| Sentence of 10 words | 2 | 2 | 4 | f16 | Python + FlashInfer | 586 (530, 586) | 6.8 | 87% | 6.7 GB | 15.7 (3.9 + 3.9, 4.2 + 3.7) |
| Sentence of 10 words | 2 | 3 | 6 | f16 | Python + FlashInfer | 928 (711, 928) | 6.5 | 89% | 6.7 GB | 23.1 (3.3 + 4.1 + 4.2, 3.7 + 4.2 + 3.6) |
| Sentence of 10 words | 2 | 4 | 8 | f16 | Python + FlashInfer | 814 (653, 814) | 9.8 | 77% | 6.7 GB | 29.9 (3.5 + 3.8 + 4.2 + 3.7, 4.3 + 3.6 + 3.5 + 3.4) |
| Sentence of 10 words | 2 | 5 | 10 | f16 | Python + FlashInfer | 800 (793, 800) | 12.5 | 85% | 6.7 GB | 39.2 (3.3 + 4.2 + 4.2 + 3.7 + 4.0, 3.3 + 3.7 + 3.7 + 4.9 + 4.3) |
| Sentence of 10 words | 2 | 6 | 12 | f16 | Python + FlashInfer | 996 (948, 996) | 12.0 | 93% | 6.7 GB | 46.7 (3.3 + 4.0 + 4.2 + 3.7 + 4.3 + 3.6, 3.8 + 3.1 + 4.7 + 4.3 + 3.9 + 3.9) |
| Sentence of 10 words | 2 | 7 | 14 | f16 | Python + FlashInfer | 1156 (1151, 1156) | 12.1 | 84% | 6.9 GB | 56.6 (3.3 + 3.9 + 4.2 + 3.7 + 4.3 + 3.1 + 3.8, 3.5 + 4.8 + 4.1 + 3.9 + 4.3 + 5.8 + 3.9) |
| Sentence of 10 words | 2 | 8 | 16 | f16 | Python + FlashInfer | 1263 (1187, 1263) | 12.7 | 87% | 7.0 GB | 65.0 (3.6 + 4.2 + 4.2 + 3.7 + 4.3 + 3.6 + 3.8 + 3.5, 4.7 + 4.3 + 3.5 + 3.9 + 5.8 + 3.4 + 4.5 + 4.2) |
| Sentence of 10 words | 2 | 9 | 18 | f16 | Python + FlashInfer | 1417 (1398, 1417) | 12.7 | 89% | 7.2 GB | 73.1 (3.7 + 3.6 + 4.2 + 3.7 + 4.0 + 3.1 + 3.8 + 3.5 + 5.2, 4.0 + 4.2 + 4.0 + 5.3 + 3.4 + 4.5 + 4.2 + 4.0 + 4.8) |
| Sentence of 10 words | 2 | 10 | 20 | f16 | Python + FlashInfer | 1655 (1528, 1655) | 12.1 | 85% | 7.5 GB | 83.3 (3.9 + 3.6 + 4.2 + 3.4 + 4.2 + 3.4 + 3.8 + 3.7 + 4.6 + 4.3, 3.7 + 4.1 + 5.3 + 3.5 + 4.8 + 4.2 + 4.6 + 4.7 + 4.4 + 5.0) |
| Sentence of 10 words | 2 | 11 | 22 | f16 | Python + FlashInfer | 1779 (1695, 1779) | 12.4 | 86% | 7.8 GB | 93.1 (3.4 + 4.2 + 4.2 + 3.7 + 4.3 + 3.6 + 3.6 + 3.5 + 4.5 + 4.0 + 4.2, 4.3 + 5.8 + 3.9 + 4.8 + 4.1 + 4.7 + 5.2 + 3.9 + 4.6 + 4.4 + 4.3) |
| Sentence of 10 words | 2 | 12 | 24 | f16 | Python + FlashInfer | 1954 (1864, 1954) | 12.3 | 87% | 8.0 GB | 102.4 (3.6 + 4.2 + 4.2 + 3.7 + 4.3 + 3.3 + 3.6 + 3.7 + 5.2 + 4.3 + 4.2 + 4.2, 5.8 + 3.9 + 4.5 + 4.2 + 4.2 + 4.5 + 4.1 + 5.0 + 4.0 + 4.0 + 4.4 + 5.4) |
| Sentence of 10 words | 3 | 1 | 3 | f16 | Python + FlashInfer | 683 (557, 683, 561) | 4.4 | 90% | 12.9 GB | 11.1 (3.4, 3.6, 4.1) |
| Sentence of 10 words | 3 | 2 | 6 | f16 | Python + FlashInfer | 723 (609, 723, 592) | 8.3 | 81% | 12.9 GB | 23.1 (3.9 + 3.9, 4.2 + 3.7, 4.1 + 3.3) |
| Sentence of 10 words | 3 | 3 | 9 | f16 | Python + FlashInfer | 856 (712, 751, 856) | 10.5 | 91% | 12.9 GB | 35.0 (3.3 + 4.1 + 4.2, 3.7 + 4.2 + 3.6, 3.5 + 3.7 + 4.7) |
| Sentence of 10 words | 3 | 4 | 12 | f16 | Python + FlashInfer | 921 (873, 921, 921) | 13.0 | 76% | 12.9 GB | 46.7 (3.5 + 3.8 + 4.2 + 3.7, 4.3 + 3.6 + 3.5 + 3.4, 4.7 + 4.1 + 3.6 + 4.3) |
| Sentence of 10 words | 3 | 5 | 15 | f16 | Python + FlashInfer | 1170 (1128, 1106, 1170) | 12.8 | 91% | 12.9 GB | 60.9 (3.3 + 4.2 + 4.2 + 3.7 + 4.0, 3.3 + 3.7 + 3.7 + 4.9 + 4.3, 3.6 + 4.3 + 5.3 + 3.7 + 4.8) |
| Sentence of 10 words | 3 | 6 | 18 | f16 | Python + FlashInfer | 1396 (1321, 1382, 1396) | 12.9 | 90% | 12.9 GB | 74.2 (3.3 + 4.0 + 4.2 + 3.7 + 4.3 + 3.6, 3.8 + 3.1 + 4.7 + 4.3 + 3.9 + 3.9, 5.7 + 3.9 + 4.8 + 4.2 + 4.2 + 4.8) |
| Sentence of 10 words | 3 | 7 | 21 | f16 | Python + FlashInfer | 1656 (1550, 1610, 1656) | 12.7 | 92% | 12.9 GB | 87.5 (3.3 + 3.9 + 4.2 + 3.7 + 4.3 + 3.1 + 3.8, 3.5 + 4.8 + 4.1 + 3.9 + 4.3 + 5.8 + 3.9, 4.5 + 4.2 + 4.0 + 4.8 + 4.4 + 5.0 + 3.9) |
| Sentence of 10 words | 4 | 1 | 4 | f16 | Python + FlashInfer | 610 (577, 610, 527, 573) | 6.6 | 90% | 13.6 GB | 14.5 (3.4, 3.6, 4.1, 3.5) |
| Sentence of 10 words | 4 | 2 | 8 | f16 | Python + FlashInfer | 705 (705, 694, 690, 675) | 11.3 | 93% | 13.6 GB | 30.0 (3.9 + 3.9, 4.2 + 3.7, 4.1 + 3.3, 3.6 + 3.4) |
| Sentence of 10 words | 4 | 3 | 12 | f16 | Python + FlashInfer | 990 (954, 988, 982, 990) | 12.1 | 93% | 13.6 GB | 47.0 (3.3 + 4.1 + 4.2, 3.7 + 4.2 + 3.6, 3.5 + 3.7 + 4.7, 4.3 + 4.2 + 3.5) |
| Sentence of 10 words | 4 | 4 | 16 | f16 | Python + FlashInfer | 1196 (1144, 1110, 1152, 1196) | 13.4 | 95% | 13.6 GB | 64.2 (3.5 + 3.8 + 4.2 + 3.7, 4.3 + 3.6 + 3.5 + 3.4, 4.7 + 4.1 + 3.6 + 4.3, 5.4 + 3.6 + 4.4 + 4.1) |
| Sentence of 10 words | 4 | 5 | 20 | f16 | Python + FlashInfer | 1522 (1418, 1450, 1522, 1493) | 13.1 | 87% | 13.6 GB | 82.8 (3.3 + 4.2 + 4.2 + 3.7 + 4.0, 3.3 + 3.7 + 3.7 + 4.9 + 4.3, 3.6 + 4.3 + 5.3 + 3.7 + 4.8, 4.2 + 3.8 + 5.2 + 3.9 + 4.9) |
| Sentence of 10 words | 4 | 6 | 24 | f16 | Python + FlashInfer | 1831 (1726, 1831, 1781, 1808) | 13.1 | 90% | 13.6 GB | 100.9 (3.3 + 4.0 + 4.2 + 3.7 + 4.3 + 3.6, 3.8 + 3.1 + 4.7 + 4.3 + 3.9 + 3.9, 5.7 + 3.9 + 4.8 + 4.2 + 4.2 + 4.8, 4.0 + 4.7 + 4.0 + 4.1 + 4.1 + 5.8) |
| Sentence of 10 words | 4 | 7 | 28 | f16 | Python + FlashInfer | 2177 (2025, 2109, 2135, 2177) | 12.9 | 91% | 13.9 GB | 120.1 (3.3 + 3.9 + 4.2 + 3.7 + 4.3 + 3.1 + 3.8, 3.5 + 4.8 + 4.1 + 3.9 + 4.3 + 5.8 + 3.9, 4.5 + 4.2 + 4.0 + 4.8 + 4.4 + 5.0 + 3.9, 4.1 + 4.6 + 5.5 + 5.0 + 4.7 + 3.5 + 5.1) |
| Sentence of 10 words | 5 | 1 | 5 | f16 | Python + FlashInfer | 634 (614, 601, 581, 634, 586) | 7.9 | 89% | 21.4 GB | 18.8 (3.4, 3.6, 4.1, 3.5, 4.2) |
| Sentence of 10 words | 5 | 2 | 10 | f16 | Python + FlashInfer | 854 (852, 854, 850, 832, 850) | 11.7 | 95% | 21.4 GB | 39.1 (3.9 + 3.9, 4.2 + 3.7, 4.1 + 3.3, 3.6 + 3.4, 4.8 + 4.3) |
| Sentence of 10 words | 5 | 3 | 15 | f16 | Python + FlashInfer | 1216 (1213, 1170, 1200, 1216, 1206) | 12.3 | 94% | 21.4 GB | 60.7 (3.3 + 4.1 + 4.2, 3.7 + 4.2 + 3.6, 3.5 + 3.7 + 4.7, 4.3 + 4.2 + 3.5, 5.4 + 3.9 + 4.4) |
| Sentence of 10 words | 5 | 4 | 20 | f16 | Python + FlashInfer | 1460 (1406, 1408, 1460, 1451, 1440) | 13.7 | 95% | 21.4 GB | 82.6 (3.5 + 3.8 + 4.2 + 3.7, 4.3 + 3.6 + 3.5 + 3.4, 4.7 + 4.1 + 3.6 + 4.3, 5.4 + 3.6 + 4.4 + 4.1, 4.1 + 4.9 + 4.4 + 5.0) |
| Sentence of 10 words | 5 | 5 | 25 | f16 | Python + FlashInfer | 1904 (1767, 1794, 1861, 1893, 1904) | 13.1 | 95% | 21.4 GB | 106.0 (3.3 + 4.2 + 4.2 + 3.7 + 4.0, 3.3 + 3.7 + 3.7 + 4.9 + 4.3, 3.6 + 4.3 + 5.3 + 3.7 + 4.8, 4.2 + 3.8 + 5.2 + 3.9 + 4.9, 4.3 + 4.1 + 4.3 + 5.5 + 4.9) |
| Sentence of 10 words | 5 | 6 | 30 | f16 | Python + FlashInfer | 2286 (2100, 2286, 2221, 2219, 2190) | 13.1 | 95% | 21.4 GB | 127.8 (3.3 + 4.0 + 4.2 + 3.7 + 4.3 + 3.6, 3.8 + 3.1 + 4.7 + 4.3 + 3.9 + 3.9, 5.7 + 3.9 + 4.8 + 4.2 + 4.2 + 4.8, 4.0 + 4.7 + 4.0 + 4.1 + 4.1 + 5.8, 4.9 + 4.7 + 3.8 + 5.2 + 4.4 + 3.9) |
| Sentence of 10 words | 5 | 7 | 35 | f16 | Python + FlashInfer | 2625 (2472, 2565, 2625, 2602, 2506) | 13.3 | 95% | 21.4 GB | 147.8 (3.3 + 3.9 + 4.2 + 3.7 + 4.3 + 3.1 + 3.8, 3.5 + 4.8 + 4.1 + 3.9 + 4.3 + 5.8 + 3.9, 4.5 + 4.2 + 4.0 + 4.8 + 4.4 + 5.0 + 3.9, 4.1 + 4.6 + 5.5 + 5.0 + 4.7 + 3.5 + 5.1, 4.4 + 3.6 + 4.2 + 3.8 + 3.7 + 4.1 + 3.9) |
| Sentence of 10 words | 6 | 1 | 6 | f16 | Python + FlashInfer | 694 (626, 625, 636, 596, 694, 636) | 8.6 | 92% | 20.7 GB | 22.1 (3.4, 3.6, 4.1, 3.5, 4.2, 3.3) |
| Sentence of 10 words | 6 | 2 | 12 | f16 | Python + FlashInfer | 1031 (1019, 1015, 1031, 999, 1029, 1024) | 11.6 | 93% | 20.7 GB | 47.0 (3.9 + 3.9, 4.2 + 3.7, 4.1 + 3.3, 3.6 + 3.4, 4.8 + 4.3, 3.6 + 4.3) |
| Sentence of 10 words | 6 | 3 | 18 | f16 | Python + FlashInfer | 1465 (1445, 1392, 1454, 1454, 1434, 1465) | 12.3 | 95% | 20.7 GB | 73.6 (3.3 + 4.1 + 4.2, 3.7 + 4.2 + 3.6, 3.5 + 3.7 + 4.7, 4.3 + 4.2 + 3.5, 5.4 + 3.9 + 4.4, 3.8 + 3.8 + 5.2) |
| Sentence of 10 words | 6 | 4 | 24 | f16 | Python + FlashInfer | 1748 (1686, 1679, 1748, 1723, 1744, 1743) | 13.7 | 95% | 20.7 GB | 100.4 (3.5 + 3.8 + 4.2 + 3.7, 4.3 + 3.6 + 3.5 + 3.4, 4.7 + 4.1 + 3.6 + 4.3, 5.4 + 3.6 + 4.4 + 4.1, 4.1 + 4.9 + 4.4 + 5.0, 4.0 + 4.3 + 4.3 + 5.2) |
| Sentence of 10 words | 6 | 5 | 30 | f16 | Python + FlashInfer | 2235 (2097, 2126, 2187, 2200, 2235, 2202) | 13.4 | 96% | 20.7 GB | 127.4 (3.3 + 4.2 + 4.2 + 3.7 + 4.0, 3.3 + 3.7 + 3.7 + 4.9 + 4.3, 3.6 + 4.3 + 5.3 + 3.7 + 4.8, 4.2 + 3.8 + 5.2 + 3.9 + 4.9, 4.3 + 4.1 + 4.3 + 5.5 + 4.9, 4.4 + 3.2 + 5.6 + 4.3 + 3.9) |
| Sentence of 10 words | 6 | 6 | 36 | f16 | Python + FlashInfer | 2652 (2528, 2568, 2632, 2647, 2652, 2562) | 13.6 | 96% | 20.7 GB | 152.2 (3.3 + 4.0 + 4.2 + 3.7 + 4.3 + 3.6, 3.8 + 3.1 + 4.7 + 4.3 + 3.9 + 3.9, 5.7 + 3.9 + 4.8 + 4.2 + 4.2 + 4.8, 4.0 + 4.7 + 4.0 + 4.1 + 4.1 + 5.8, 4.9 + 4.7 + 3.8 + 5.2 + 4.4 + 3.9, 4.1 + 4.0 + 3.7 + 4.2 + 3.9 + 4.7) |
| Sentence of 10 words | 6 | 7 | 42 | f16 | Python + FlashInfer | 3147 (2977, 3090, 3098, 3147, 3013, 3111) | 13.3 | 96% | 20.8 GB | 180.0 (3.3 + 3.9 + 4.2 + 3.7 + 4.3 + 3.1 + 3.8, 3.5 + 4.8 + 4.1 + 3.9 + 4.3 + 5.8 + 3.9, 4.5 + 4.2 + 4.0 + 4.8 + 4.4 + 5.0 + 3.9, 4.1 + 4.6 + 5.5 + 5.0 + 4.7 + 3.5 + 5.1, 4.4 + 3.6 + 4.2 + 3.8 + 3.7 + 4.1 + 3.9, 4.7 + 4.4 + 5.9 + 4.2 + 3.4 + 5.4 + 4.3) |
| Sentence of 10 words | 1 | 1 | 1 | bf16 | Python + FlashInfer | failed: bf16 not supported: upstream `apply_flashinfer` plans attention as f16 only | – | – | – | – |
| Sentence of 10 words | 1 | 1 | 1 | f16 | Python + FlashInfer + CUDA graphs | 213 | 4.7 | 66% | 6.1 GB | 3.6 (3.6) |
| Sentence of 10 words | 1 | 2 | 2 | f16 | Python + FlashInfer + CUDA graphs | 338 | 5.9 | 68% | 6.2 GB | 7.0 (3.4 + 3.6) |
| Sentence of 10 words | 1 | 3 | 3 | f16 | Python + FlashInfer + CUDA graphs | 468 | 6.4 | 76% | 6.3 GB | 11.7 (3.4 + 4.1 + 4.2) |
| Sentence of 10 words | 1 | 4 | 4 | f16 | Python + FlashInfer + CUDA graphs | 605 | 6.6 | 85% | 6.4 GB | 15.4 (3.4 + 4.1 + 4.2 + 3.7) |
| Sentence of 10 words | 1 | 5 | 5 | f16 | Python + FlashInfer + CUDA graphs | 776 | 6.4 | 86% | 6.7 GB | 20.0 (3.9 + 3.9 + 4.2 + 3.7 + 4.3) |
| Sentence of 10 words | 1 | 6 | 6 | f16 | Python + FlashInfer + CUDA graphs | 916 | 6.6 | 86% | 6.9 GB | 23.2 (3.6 + 3.8 + 4.2 + 3.7 + 4.3 + 3.6) |
| Sentence of 10 words | 1 | 7 | 7 | f16 | Python + FlashInfer + CUDA graphs | 1056 | 6.6 | 85% | 7.1 GB | 26.8 (3.7 + 3.6 + 4.2 + 3.7 + 4.3 + 3.6 + 3.8) |
| Sentence of 10 words | 1 | 8 | 8 | f16 | Python + FlashInfer + CUDA graphs | 1193 | 6.7 | 87% | 7.4 GB | 30.3 (3.6 + 3.9 + 4.2 + 3.7 + 4.2 + 3.2 + 3.8 + 3.7) |
| Sentence of 10 words | 1 | 9 | 9 | f16 | Python + FlashInfer + CUDA graphs | 1592 | 5.7 | 88% | 7.9 GB | 35.5 (3.8 + 3.6 + 4.2 + 3.5 + 4.3 + 3.6 + 3.6 + 3.6 + 5.2) |
| Sentence of 10 words | 1 | 10 | 10 | f16 | Python + FlashInfer + CUDA graphs | 1750 | 5.7 | 87% | 8.3 GB | 39.2 (3.9 + 3.7 + 4.2 + 3.5 + 4.3 + 3.4 + 3.8 + 3.4 + 4.8 + 4.3) |
| Sentence of 10 words | 1 | 11 | 11 | f16 | Python + FlashInfer + CUDA graphs | 1930 | 5.7 | 85% | 8.8 GB | 42.7 (3.9 + 3.7 + 4.2 + 3.7 + 3.9 + 3.6 + 3.8 + 3.7 + 4.8 + 4.0 + 3.5) |
| Sentence of 10 words | 2 | 1 | 2 | f16 | Python + FlashInfer + CUDA graphs | 384 (384, 375) | 5.2 | 91% | 8.5 GB | 7.3 (3.6, 3.7) |
| Sentence of 10 words | 2 | 2 | 4 | f16 | Python + FlashInfer + CUDA graphs | 610 (609, 610) | 6.6 | 93% | 8.8 GB | 14.8 (3.4 + 3.6, 4.2 + 3.7) |
| Sentence of 10 words | 2 | 3 | 6 | f16 | Python + FlashInfer + CUDA graphs | 903 (903, 877) | 6.6 | 91% | 9.0 GB | 22.1 (3.4 + 4.1 + 4.2, 3.3 + 3.8 + 3.4) |
| Sentence of 10 words | 2 | 4 | 8 | f16 | Python + FlashInfer + CUDA graphs | 1195 (1195, 1163) | 6.7 | 92% | 9.2 GB | 30.5 (3.4 + 4.1 + 4.2 + 3.7, 4.1 + 3.5 + 3.8 + 3.7) |
| Sentence of 10 words | 2 | 5 | 10 | f16 | Python + FlashInfer + CUDA graphs | 1653 (1574, 1653) | 6.0 | 94% | 9.7 GB | 39.9 (3.9 + 3.9 + 4.2 + 3.7 + 4.3, 3.3 + 3.8 + 3.7 + 4.8 + 4.3) |
| Sentence of 10 words | 2 | 6 | 12 | f16 | Python + FlashInfer + CUDA graphs | 1996 (1819, 1996) | 6.0 | 92% | 10.2 GB | 48.0 (3.6 + 3.8 + 4.2 + 3.7 + 4.3 + 3.6, 3.7 + 3.5 + 5.2 + 4.3 + 3.8 + 4.2) |
| Sentence of 10 words | 2 | 7 | 14 | f16 | Python + FlashInfer + CUDA graphs | 2301 (2183, 2301) | 6.1 | 94% | 10.8 GB | 56.8 (3.7 + 3.6 + 4.2 + 3.7 + 4.3 + 3.6 + 3.8, 3.7 + 4.9 + 4.3 + 4.1 + 3.8 + 5.8 + 3.4) |
| Sentence of 10 words | 3 | 1 | 3 | f16 | Python + FlashInfer + CUDA graphs | 530 (530, 516, 517) | 5.7 | 93% | 12.9 GB | 11.4 (3.6, 3.7, 4.1) |
| Sentence of 10 words | 3 | 2 | 6 | f16 | Python + FlashInfer + CUDA graphs | 888 (876, 888, 865) | 6.8 | 94% | 13.2 GB | 22.8 (3.4 + 3.6, 4.2 + 3.7, 4.3 + 3.6) |
| Sentence of 10 words | 3 | 3 | 9 | f16 | Python + FlashInfer + CUDA graphs | 1423 (1324, 1326, 1423) | 6.3 | 94% | 13.6 GB | 34.5 (3.4 + 4.1 + 4.2, 3.3 + 3.8 + 3.4, 3.8 + 3.4 + 5.2) |
| Sentence of 10 words | 3 | 4 | 12 | f16 | Python + FlashInfer + CUDA graphs | 1870 (1682, 1752, 1870) | 6.4 | 95% | 14.0 GB | 47.3 (3.4 + 4.1 + 4.2 + 3.7, 4.1 + 3.5 + 3.8 + 3.7, 4.7 + 4.3 + 3.6 + 4.2) |
| Sentence of 10 words | 3 | 5 | 15 | f16 | Python + FlashInfer + CUDA graphs | 2524 (2197, 2524, 2488) | 5.9 | 96% | 14.6 GB | 61.0 (3.9 + 3.9 + 4.2 + 3.7 + 4.3, 3.3 + 3.8 + 3.7 + 4.8 + 4.3, 3.7 + 3.9 + 5.8 + 3.4 + 4.3) |
| Sentence of 10 words | 4 | 1 | 4 | f16 | Python + FlashInfer + CUDA graphs | 685 (685, 679, 676, 676) | 5.8 | 93% | 17.1 GB | 15.0 (3.6, 3.7, 4.1, 3.6) |
| Sentence of 10 words | 4 | 2 | 8 | f16 | Python + FlashInfer + CUDA graphs | 1177 (1157, 1159, 1160, 1177) | 6.8 | 94% | 17.6 GB | 30.0 (3.4 + 3.6, 4.2 + 3.7, 4.3 + 3.6, 3.6 + 3.6) |
| Sentence of 10 words | 4 | 3 | 12 | f16 | Python + FlashInfer + CUDA graphs | 1835 (1764, 1739, 1835, 1694) | 6.5 | 95% | 18.0 GB | 46.3 (3.4 + 4.1 + 4.2, 3.3 + 3.8 + 3.4, 3.8 + 3.4 + 5.2, 4.0 + 3.8 + 4.0) |
| Sentence of 10 words | 4 | 4 | 16 | f16 | Python + FlashInfer + CUDA graphs | 2552 (2318, 2257, 2550, 2552) | 6.3 | 97% | 18.6 GB | 65.6 (3.4 + 4.1 + 4.2 + 3.7, 4.1 + 3.5 + 3.8 + 3.7, 4.7 + 4.3 + 3.6 + 4.2, 5.4 + 3.9 + 4.8 + 4.2) |
| Sentence of 10 words | 4 | 5 | 20 | f16 | Python + FlashInfer + CUDA graphs | 3329 (2981, 3280, 3329, 3308) | 6.0 | 97% | 19.4 GB | 82.4 (3.9 + 3.9 + 4.2 + 3.7 + 4.3, 3.3 + 3.8 + 3.7 + 4.8 + 4.3, 3.7 + 3.9 + 5.8 + 3.4 + 4.3, 3.8 + 4.3 + 4.4 + 4.0 + 4.9) |
| Sentence of 10 words | 1 | 1 | 1 | bf16 | Python + FlashInfer + CUDA graphs | 210 | 4.8 | 66% | 6.5 GB | 3.4 (3.4) |
| Sentence of 10 words | 1 | 2 | 2 | bf16 | Python + FlashInfer + CUDA graphs | 335 | 6.0 | 73% | 6.6 GB | 7.0 (3.4 + 3.6) |
| Sentence of 10 words | 1 | 3 | 3 | bf16 | Python + FlashInfer + CUDA graphs | 467 | 6.4 | 82% | 6.7 GB | 11.8 (3.8 + 3.9 + 4.2) |
| Sentence of 10 words | 1 | 4 | 4 | bf16 | Python + FlashInfer + CUDA graphs | 608 | 6.6 | 84% | 6.9 GB | 15.7 (3.7 + 4.2 + 4.2 + 3.7) |
| Sentence of 10 words | 1 | 5 | 5 | bf16 | Python + FlashInfer + CUDA graphs | 765 | 6.5 | 93% | 7.1 GB | 20.2 (3.9 + 4.2 + 4.2 + 3.7 + 4.3) |
| Sentence of 10 words | 1 | 6 | 6 | bf16 | Python + FlashInfer + CUDA graphs | 914 | 6.6 | 86% | 7.4 GB | 22.5 (3.6 + 3.8 + 4.2 + 3.7 + 4.0 + 3.2) |
| Sentence of 10 words | 1 | 7 | 7 | bf16 | Python + FlashInfer + CUDA graphs | 1065 | 6.6 | 86% | 7.6 GB | 26.4 (3.3 + 3.6 + 4.2 + 3.7 + 4.3 + 3.6 + 3.7) |
| Sentence of 10 words | 1 | 8 | 8 | bf16 | Python + FlashInfer + CUDA graphs | 1195 | 6.7 | 85% | 7.8 GB | 30.6 (3.6 + 3.9 + 4.2 + 3.7 + 4.2 + 3.6 + 3.8 + 3.6) |
| Sentence of 10 words | 1 | 9 | 9 | bf16 | Python + FlashInfer + CUDA graphs | 1592 | 5.7 | 86% | 8.4 GB | 35.0 (3.8 + 3.6 + 4.2 + 3.7 + 4.0 + 3.6 + 3.5 + 3.7 + 4.8) |
| Sentence of 10 words | 1 | 10 | 10 | bf16 | Python + FlashInfer + CUDA graphs | 1745 | 5.7 | 88% | 8.8 GB | 39.6 (3.7 + 3.7 + 4.2 + 3.5 + 4.3 + 3.4 + 3.8 + 3.4 + 5.2 + 4.3) |
| Sentence of 10 words | 1 | 11 | 11 | bf16 | Python + FlashInfer + CUDA graphs | 1963 | 5.6 | 88% | 9.2 GB | 42.3 (3.6 + 3.6 + 4.2 + 3.7 + 4.2 + 3.4 + 3.6 + 3.7 + 4.8 + 4.0 + 3.5) |
| Sentence of 10 words | 2 | 1 | 2 | bf16 | Python + FlashInfer + CUDA graphs | 359 (354, 359) | 5.6 | 89% | 8.5 GB | 7.4 (3.4, 4.0) |
| Sentence of 10 words | 2 | 2 | 4 | bf16 | Python + FlashInfer + CUDA graphs | 630 (630, 628) | 6.4 | 96% | 8.8 GB | 14.7 (3.4 + 3.6, 4.2 + 3.6) |
| Sentence of 10 words | 2 | 3 | 6 | bf16 | Python + FlashInfer + CUDA graphs | 907 (885, 907) | 6.6 | 94% | 9.0 GB | 22.5 (3.8 + 3.9 + 4.2, 3.3 + 3.8 + 3.6) |
| Sentence of 10 words | 2 | 4 | 8 | bf16 | Python + FlashInfer + CUDA graphs | 1182 (1180, 1182) | 6.8 | 94% | 9.2 GB | 30.7 (3.7 + 4.2 + 4.2 + 3.7, 4.2 + 3.4 + 3.8 + 3.7) |
| Sentence of 10 words | 2 | 5 | 10 | bf16 | Python + FlashInfer + CUDA graphs | 1650 (1500, 1650) | 6.1 | 92% | 9.7 GB | 39.9 (3.9 + 4.2 + 4.2 + 3.7 + 4.3, 3.3 + 3.8 + 3.5 + 4.8 + 4.3) |
| Sentence of 10 words | 2 | 6 | 12 | bf16 | Python + FlashInfer + CUDA graphs | 1971 (1809, 1971) | 6.1 | 94% | 10.2 GB | 47.5 (3.6 + 3.8 + 4.2 + 3.7 + 4.0 + 3.2, 3.7 + 3.5 + 5.2 + 4.3 + 3.9 + 4.3) |
| Sentence of 10 words | 2 | 7 | 14 | bf16 | Python + FlashInfer + CUDA graphs | 2294 (2113, 2294) | 6.1 | 94% | 10.8 GB | 56.6 (3.3 + 3.6 + 4.2 + 3.7 + 4.3 + 3.6 + 3.7, 3.5 + 5.2 + 4.3 + 4.1 + 3.8 + 5.8 + 3.4) |
| Sentence of 10 words | 3 | 1 | 3 | bf16 | Python + FlashInfer + CUDA graphs | 509 (480, 509, 502) | 5.9 | 92% | 12.9 GB | 11.6 (3.4, 4.0, 4.2) |
| Sentence of 10 words | 3 | 2 | 6 | bf16 | Python + FlashInfer + CUDA graphs | 913 (889, 913, 862) | 6.6 | 93% | 13.2 GB | 22.5 (3.4 + 3.6, 4.2 + 3.6, 4.2 + 3.6) |
| Sentence of 10 words | 3 | 3 | 9 | bf16 | Python + FlashInfer + CUDA graphs | 1431 (1341, 1351, 1431) | 6.3 | 94% | 13.6 GB | 34.9 (3.8 + 3.9 + 4.2, 3.3 + 3.8 + 3.6, 3.8 + 3.7 + 5.0) |
| Sentence of 10 words | 3 | 4 | 12 | bf16 | Python + FlashInfer + CUDA graphs | 1891 (1763, 1754, 1891) | 6.3 | 94% | 14.0 GB | 47.0 (3.7 + 4.2 + 4.2 + 3.7, 4.2 + 3.4 + 3.8 + 3.7, 4.6 + 4.1 + 3.7 + 3.8) |
| Sentence of 10 words | 3 | 5 | 15 | bf16 | Python + FlashInfer + CUDA graphs | 2528 (2119, 2528, 2451) | 5.9 | 93% | 14.6 GB | 61.0 (3.9 + 4.2 + 4.2 + 3.7 + 4.3, 3.3 + 3.8 + 3.5 + 4.8 + 4.3, 3.7 + 4.3 + 5.5 + 3.4 + 4.2) |
| Sentence of 10 words | 1 | 1 | 1 | f16 | C++ (`perf_mode=flash_attention`) | 112 | 9.0 | 89% | 2.3 GB | 3.9 (3.9) |
| Sentence of 10 words | 1 | 2 | 2 | f16 | C++ (`perf_mode=flash_attention`) | 220 | 9.1 | 89% | 2.3 GB | 8.0 (3.9 + 4.2) |
| Sentence of 10 words | 1 | 3 | 3 | f16 | C++ (`perf_mode=flash_attention`) | 326 | 9.2 | 86% | 2.3 GB | 12.2 (3.9 + 4.2 + 4.2) |
| Sentence of 10 words | 1 | 4 | 4 | f16 | C++ (`perf_mode=flash_attention`) | 395 | 10.1 | 85% | 2.3 GB | 15.2 (3.9 + 3.5 + 4.2 + 3.7) |
| Sentence of 10 words | 1 | 5 | 5 | f16 | C++ (`perf_mode=flash_attention`) | 513 | 9.7 | 90% | 2.3 GB | 19.6 (3.9 + 3.5 + 4.2 + 3.7 + 4.3) |
| Sentence of 10 words | 1 | 6 | 6 | f16 | C++ (`perf_mode=flash_attention`) | 705 | 8.5 | 90% | 2.4 GB | 23.2 (3.9 + 3.5 + 4.2 + 3.7 + 4.3 + 3.6) |
| Sentence of 10 words | 1 | 7 | 7 | f16 | C++ (`perf_mode=flash_attention`) | 757 | 9.3 | 90% | 2.4 GB | 26.9 (3.9 + 3.5 + 4.2 + 3.7 + 4.3 + 3.6 + 3.8) |
| Sentence of 10 words | 2 | 1 | 2 | f16 | C++ (`perf_mode=flash_attention`) | 233 (233, 228) | 8.6 | 92% | 4.6 GB | 8.0 (3.9, 4.2) |
| Sentence of 10 words | 2 | 2 | 4 | f16 | C++ (`perf_mode=flash_attention`) | 442 (442, 434) | 9.1 | 93% | 4.6 GB | 15.9 (3.9 + 4.2, 4.2 + 3.7) |
| Sentence of 10 words | 2 | 3 | 6 | f16 | C++ (`perf_mode=flash_attention`) | 644 (644, 639) | 9.3 | 95% | 4.7 GB | 23.8 (3.9 + 4.2 + 4.2, 3.7 + 4.3 + 3.6) |
| Sentence of 10 words | 2 | 4 | 8 | f16 | C++ (`perf_mode=flash_attention`) | 789 (785, 789) | 10.1 | 95% | 4.7 GB | 30.6 (3.9 + 3.5 + 4.2 + 3.7, 4.3 + 3.6 + 3.8 + 3.7) |
| Sentence of 10 words | 2 | 5 | 10 | f16 | C++ (`perf_mode=flash_attention`) | 1067 (1062, 1067) | 9.4 | 95% | 4.7 GB | 40.3 (3.9 + 3.5 + 4.2 + 3.7 + 4.3, 3.6 + 3.8 + 3.7 + 5.2 + 4.3) |
| Sentence of 10 words | 2 | 6 | 12 | f16 | C++ (`perf_mode=flash_attention`) | 1338 (1321, 1338) | 9.0 | 96% | 4.8 GB | 48.1 (3.9 + 3.5 + 4.2 + 3.7 + 4.3 + 3.6, 3.8 + 3.7 + 5.2 + 4.1 + 3.7 + 4.3) |
| Sentence of 10 words | 2 | 7 | 14 | f16 | C++ (`perf_mode=flash_attention`) | 1649 (1639, 1649) | 8.5 | 96% | 4.8 GB | 57.7 (3.9 + 3.5 + 4.2 + 3.7 + 4.3 + 3.6 + 3.8, 3.7 + 5.2 + 4.3 + 3.6 + 4.3 + 5.8 + 3.9) |
| Sentence of 10 words | 1 | 1 | 1 | bf16 | C++ (`perf_mode=flash_attention`) | 114 | 8.7 | 89% | 2.3 GB | 3.9 (3.9) |
| Sentence of 10 words | 1 | 2 | 2 | bf16 | C++ (`perf_mode=flash_attention`) | 221 | 9.0 | 87% | 2.3 GB | 8.0 (3.9 + 4.2) |
| Sentence of 10 words | 1 | 3 | 3 | bf16 | C++ (`perf_mode=flash_attention`) | 356 | 8.4 | 83% | 2.3 GB | 12.2 (3.9 + 4.2 + 4.2) |
| Sentence of 10 words | 1 | 4 | 4 | bf16 | C++ (`perf_mode=flash_attention`) | 446 | 9.0 | 86% | 2.4 GB | 15.9 (3.9 + 4.2 + 4.2 + 3.7) |
| Sentence of 10 words | 1 | 5 | 5 | bf16 | C++ (`perf_mode=flash_attention`) | 566 | 8.8 | 91% | 2.4 GB | 20.2 (3.9 + 4.2 + 4.2 + 3.7 + 4.3) |
| Sentence of 10 words | 2 | 1 | 2 | bf16 | C++ (`perf_mode=flash_attention`) | 242 (242, 237) | 8.3 | 94% | 4.6 GB | 8.0 (3.9, 4.2) |
| Sentence of 10 words | 2 | 2 | 4 | bf16 | C++ (`perf_mode=flash_attention`) | 444 (439, 444) | 9.0 | 95% | 4.7 GB | 15.9 (3.9 + 4.2, 4.2 + 3.7) |
| Sentence of 10 words | 2 | 3 | 6 | bf16 | C++ (`perf_mode=flash_attention`) | 700 (700, 694) | 8.6 | 96% | 4.7 GB | 23.8 (3.9 + 4.2 + 4.2, 3.7 + 4.3 + 3.6) |
| Sentence of 10 words | 2 | 4 | 8 | bf16 | C++ (`perf_mode=flash_attention`) | 914 (910, 914) | 8.8 | 97% | 4.7 GB | 31.3 (3.9 + 4.2 + 4.2 + 3.7, 4.3 + 3.6 + 3.8 + 3.7) |
| Sentence of 10 words | 2 | 5 | 10 | bf16 | C++ (`perf_mode=flash_attention`) | 1189 (1189, 1185) | 8.4 | 97% | 4.8 GB | 40.8 (3.9 + 4.2 + 4.2 + 3.7 + 4.3, 3.6 + 3.8 + 3.7 + 5.2 + 4.3) |
| Sentence of 10 words | 1 | 1 | 1 | f16 | C++ (`perf_mode=off`) | 144 | 6.9 | 81% | 2.3 GB | 3.9 (3.9) |
| Sentence of 10 words | 1 | 2 | 2 | f16 | C++ (`perf_mode=off`) | 276 | 7.3 | 88% | 2.3 GB | 7.4 (3.9 + 3.5) |
| Sentence of 10 words | 1 | 3 | 3 | f16 | C++ (`perf_mode=off`) | 391 | 7.7 | 91% | 2.3 GB | 11.5 (3.9 + 3.5 + 4.2) |
| Sentence of 10 words | 1 | 4 | 4 | f16 | C++ (`perf_mode=off`) | 495 | 8.1 | 91% | 2.3 GB | 15.2 (3.9 + 3.5 + 4.2 + 3.7) |
| Sentence of 10 words | 1 | 5 | 5 | f16 | C++ (`perf_mode=off`) | 650 | 7.7 | 91% | 2.3 GB | 20.2 (3.9 + 4.2 + 4.2 + 3.7 + 4.3) |
| Sentence of 10 words | 1 | 6 | 6 | f16 | C++ (`perf_mode=off`) | 798 | 7.5 | 90% | 2.4 GB | 23.8 (3.9 + 4.2 + 4.2 + 3.7 + 4.3 + 3.6) |
| Sentence of 10 words | 1 | 7 | 7 | f16 | C++ (`perf_mode=off`) | 952 | 7.3 | 91% | 2.4 GB | 27.6 (3.9 + 4.2 + 4.2 + 3.7 + 4.3 + 3.6 + 3.8) |
| Sentence of 10 words | 2 | 1 | 2 | f16 | C++ (`perf_mode=off`) | 292 (292, 292) | 6.8 | 96% | 4.5 GB | 7.4 (3.9, 3.5) |
| Sentence of 10 words | 2 | 2 | 4 | f16 | C++ (`perf_mode=off`) | 547 (547, 545) | 7.3 | 96% | 4.6 GB | 15.3 (3.9 + 3.5, 4.2 + 3.7) |
| Sentence of 10 words | 2 | 3 | 6 | f16 | C++ (`perf_mode=off`) | 806 (804, 806) | 7.4 | 96% | 4.6 GB | 23.2 (3.9 + 3.5 + 4.2, 3.7 + 4.3 + 3.6) |
| Sentence of 10 words | 2 | 4 | 8 | f16 | C++ (`perf_mode=off`) | 1032 (1032, 1019) | 7.8 | 95% | 4.7 GB | 30.6 (3.9 + 3.5 + 4.2 + 3.7, 4.3 + 3.6 + 3.8 + 3.7) |
| Sentence of 10 words | 2 | 5 | 10 | f16 | C++ (`perf_mode=off`) | 1374 (1374, 1369) | 7.3 | 96% | 4.7 GB | 40.9 (3.9 + 4.2 + 4.2 + 3.7 + 4.3, 3.6 + 3.8 + 3.7 + 5.2 + 4.3) |
| Sentence of 10 words | 2 | 6 | 12 | f16 | C++ (`perf_mode=off`) | 1702 (1694, 1702) | 7.1 | 95% | 4.8 GB | 48.7 (3.9 + 4.2 + 4.2 + 3.7 + 4.3 + 3.6, 3.8 + 3.7 + 5.2 + 4.3 + 3.6 + 4.3) |
| Sentence of 10 words | 2 | 7 | 14 | f16 | C++ (`perf_mode=off`) | 2076 (2065, 2076) | 6.7 | 96% | 4.8 GB | 58.4 (3.9 + 4.2 + 4.2 + 3.7 + 4.3 + 3.6 + 3.8, 3.7 + 5.2 + 4.3 + 3.6 + 4.3 + 5.8 + 3.9) |
| Sentence of 10 words | 1 | 1 | 1 | bf16 | C++ (`perf_mode=off`) | 144 | 6.9 | 88% | 2.3 GB | 3.9 (3.9) |
| Sentence of 10 words | 1 | 2 | 2 | bf16 | C++ (`perf_mode=off`) | 283 | 7.1 | 87% | 2.3 GB | 8.0 (3.9 + 4.2) |
| Sentence of 10 words | 1 | 3 | 3 | bf16 | C++ (`perf_mode=off`) | 424 | 7.1 | 90% | 2.3 GB | 12.2 (3.9 + 4.2 + 4.2) |
| Sentence of 10 words | 1 | 4 | 4 | bf16 | C++ (`perf_mode=off`) | 551 | 7.3 | 91% | 2.3 GB | 15.9 (3.9 + 4.2 + 4.2 + 3.7) |
| Sentence of 10 words | 1 | 5 | 5 | bf16 | C++ (`perf_mode=off`) | 705 | 7.1 | 91% | 2.4 GB | 19.9 (3.6 + 4.2 + 4.2 + 3.7 + 4.3) |
| Sentence of 10 words | 1 | 6 | 6 | bf16 | C++ (`perf_mode=off`) | 857 | 7.0 | 90% | 2.4 GB | 23.5 (3.6 + 4.2 + 4.2 + 3.7 + 4.3 + 3.6) |
| Sentence of 10 words | 1 | 7 | 7 | bf16 | C++ (`perf_mode=off`) | 998 | 7.0 | 91% | 2.4 GB | 27.3 (3.6 + 4.2 + 4.2 + 3.7 + 4.3 + 3.6 + 3.8) |
| Sentence of 10 words | 2 | 1 | 2 | bf16 | C++ (`perf_mode=off`) | 298 (298, 293) | 6.7 | 94% | 4.6 GB | 8.0 (3.9, 4.2) |
| Sentence of 10 words | 2 | 2 | 4 | bf16 | C++ (`perf_mode=off`) | 557 (557, 552) | 7.2 | 96% | 4.6 GB | 15.9 (3.9 + 4.2, 4.2 + 3.7) |
| Sentence of 10 words | 2 | 3 | 6 | bf16 | C++ (`perf_mode=off`) | 877 (877, 867) | 6.8 | 96% | 4.7 GB | 23.8 (3.9 + 4.2 + 4.2, 3.7 + 4.3 + 3.6) |
| Sentence of 10 words | 2 | 4 | 8 | bf16 | C++ (`perf_mode=off`) | 1151 (1148, 1151) | 6.9 | 96% | 4.7 GB | 31.3 (3.9 + 4.2 + 4.2 + 3.7, 4.3 + 3.6 + 3.8 + 3.7) |
| Sentence of 10 words | 2 | 5 | 10 | bf16 | C++ (`perf_mode=off`) | 1484 (1477, 1484) | 6.7 | 96% | 4.8 GB | 40.6 (3.6 + 4.2 + 4.2 + 3.7 + 4.3, 3.6 + 3.8 + 3.7 + 5.2 + 4.3) |
| Sentence of 10 words | 1 | 1 | 1 | f16 | Python (PyTorch) | 985 | 1.0 | 23% | 5.9 GB | 3.4 (3.4) |
| Sentence of 10 words | 1 | 2 | 2 | f16 | Python (PyTorch) | 1016 | 2.0 | 33% | 5.9 GB | 6.9 (3.3 + 3.6) |
| Sentence of 10 words | 1 | 3 | 3 | f16 | Python (PyTorch) | 1056 | 2.8 | 40% | 5.9 GB | 11.2 (3.4 + 3.6 + 4.2) |
| Sentence of 10 words | 1 | 4 | 4 | f16 | Python (PyTorch) | 1067 | 3.8 | 54% | 5.9 GB | 14.8 (3.4 + 3.6 + 4.2 + 3.7) |
| Sentence of 10 words | 1 | 5 | 5 | f16 | Python (PyTorch) | 1115 | 4.5 | 64% | 5.9 GB | 19.0 (3.4 + 3.6 + 4.2 + 3.7 + 4.1) |
| Sentence of 10 words | 1 | 6 | 6 | f16 | Python (PyTorch) | 1161 | 5.2 | 74% | 5.9 GB | 22.6 (3.9 + 3.6 + 4.2 + 3.7 + 4.1 + 3.1) |
| Sentence of 10 words | 1 | 7 | 7 | f16 | Python (PyTorch) | 1168 | 6.0 | 87% | 5.9 GB | 26.7 (3.4 + 3.7 + 4.2 + 3.7 + 4.3 + 3.6 + 3.8) |
| Sentence of 10 words | 1 | 8 | 8 | f16 | Python (PyTorch) | 1359 | 5.9 | 88% | 5.9 GB | 30.2 (3.7 + 3.6 + 4.2 + 3.4 + 4.3 + 3.6 + 3.8 + 3.7) |
| Sentence of 10 words | 1 | 9 | 9 | f16 | Python (PyTorch) | 1725 | 5.2 | 88% | 5.9 GB | 34.1 (3.4 + 3.6 + 4.2 + 3.5 + 4.1 + 3.0 + 3.7 + 3.5 + 5.2) |
| Sentence of 10 words | 1 | 10 | 10 | f16 | Python (PyTorch) | 1952 | 5.1 | 88% | 5.9 GB | 39.9 (3.9 + 3.6 + 4.2 + 3.7 + 4.3 + 3.2 + 3.8 + 3.7 + 5.2 + 4.3) |
| Sentence of 10 words | 2 | 1 | 2 | f16 | Python (PyTorch) | 1036 (1036, 1007) | 1.9 | 96% | 7.8 GB | 7.0 (3.4, 3.7) |
| Sentence of 10 words | 2 | 2 | 4 | f16 | Python (PyTorch) | 1073 (1073, 1028) | 3.7 | 93% | 7.8 GB | 14.4 (3.3 + 3.6, 4.1 + 3.3) |
| Sentence of 10 words | 2 | 3 | 6 | f16 | Python (PyTorch) | 1104 (1104, 1069) | 5.4 | 92% | 7.8 GB | 22.8 (3.4 + 3.6 + 4.2, 3.7 + 4.3 + 3.5) |
| Sentence of 10 words | 2 | 4 | 8 | f16 | Python (PyTorch) | 1368 (1368, 1284) | 5.8 | 90% | 7.8 GB | 30.2 (3.4 + 3.6 + 4.2 + 3.7, 4.3 + 3.5 + 3.8 + 3.7) |
| Sentence of 10 words | 2 | 5 | 10 | f16 | Python (PyTorch) | 1754 (1678, 1754) | 5.7 | 95% | 7.8 GB | 38.4 (3.4 + 3.6 + 4.2 + 3.7 + 4.1, 3.3 + 3.8 + 3.4 + 4.7 + 4.3) |
| Sentence of 10 words | 2 | 6 | 12 | f16 | Python (PyTorch) | 2136 (2067, 2136) | 5.6 | 94% | 7.8 GB | 47.0 (3.9 + 3.6 + 4.2 + 3.7 + 4.1 + 3.1, 3.8 + 3.5 + 4.8 + 4.3 + 4.2 + 3.9) |
| Sentence of 10 words | 2 | 7 | 14 | f16 | Python (PyTorch) | 2581 (2380, 2581) | 5.4 | 94% | 8.4 GB | 55.9 (3.4 + 3.7 + 4.2 + 3.7 + 4.3 + 3.6 + 3.8, 3.6 + 4.7 + 4.0 + 4.2 + 4.0 + 5.3 + 3.5) |
| Sentence of 10 words | 3 | 1 | 3 | f16 | Python (PyTorch) | 1518 (1264, 1496, 1518) | 2.0 | 90% | 11.7 GB | 11.1 (3.4, 3.7, 4.1) |
| Sentence of 10 words | 3 | 2 | 6 | f16 | Python (PyTorch) | 1304 (1304, 1207, 1191) | 4.6 | 89% | 11.7 GB | 22.0 (3.3 + 3.6, 4.1 + 3.3, 4.2 + 3.5) |
| Sentence of 10 words | 3 | 3 | 9 | f16 | Python (PyTorch) | 1635 (1518, 1543, 1635) | 5.5 | 85% | 11.7 GB | 35.3 (3.4 + 3.6 + 4.2, 3.7 + 4.3 + 3.5, 3.8 + 3.5 + 5.2) |
| Sentence of 10 words | 3 | 4 | 12 | f16 | Python (PyTorch) | 2017 (1940, 1936, 2017) | 5.9 | 97% | 11.7 GB | 47.0 (3.4 + 3.6 + 4.2 + 3.7, 4.3 + 3.5 + 3.8 + 3.7, 5.1 + 4.3 + 3.6 + 3.8) |
| Sentence of 10 words | 3 | 5 | 15 | f16 | Python (PyTorch) | 2667 (2489, 2629, 2667) | 5.6 | 97% | 11.7 GB | 60.3 (3.4 + 3.6 + 4.2 + 3.7 + 4.1, 3.3 + 3.8 + 3.4 + 4.7 + 4.3, 3.7 + 3.9 + 5.6 + 3.9 + 4.8) |
| Sentence of 10 words | 3 | 6 | 18 | f16 | Python (PyTorch) | 3282 (2959, 3223, 3282) | 5.5 | 96% | 12.0 GB | 73.7 (3.9 + 3.6 + 4.2 + 3.7 + 4.1 + 3.1, 3.8 + 3.5 + 4.8 + 4.3 + 4.2 + 3.9, 5.8 + 3.8 + 4.3 + 4.2 + 4.0 + 4.5) |
| Sentence of 10 words | 3 | 7 | 21 | f16 | Python (PyTorch) | 3851 (3519, 3851, 3806) | 5.5 | 95% | 13.0 GB | 87.1 (3.4 + 3.7 + 4.2 + 3.7 + 4.3 + 3.6 + 3.8, 3.6 + 4.7 + 4.0 + 4.2 + 4.0 + 5.3 + 3.5, 4.8 + 4.0 + 4.0 + 4.9 + 4.4 + 4.6 + 4.4) |
| Sentence of 10 words | 4 | 1 | 4 | f16 | Python (PyTorch) | 1325 (1184, 1325, 1243, 1175) | 3.0 | 84% | 15.7 GB | 14.6 (3.4, 3.7, 4.1, 3.5) |
| Sentence of 10 words | 4 | 2 | 8 | f16 | Python (PyTorch) | 1398 (1385, 1382, 1398, 1375) | 5.7 | 95% | 15.7 GB | 29.1 (3.3 + 3.6, 4.1 + 3.3, 4.2 + 3.5, 3.6 + 3.5) |
| Sentence of 10 words | 4 | 3 | 12 | f16 | Python (PyTorch) | 2095 (1994, 2005, 2095, 2007) | 5.7 | 96% | 15.7 GB | 47.5 (3.4 + 3.6 + 4.2, 3.7 + 4.3 + 3.5, 3.8 + 3.5 + 5.2, 4.1 + 4.0 + 4.2) |
| Sentence of 10 words | 4 | 4 | 16 | f16 | Python (PyTorch) | 2732 (2483, 2601, 2702, 2732) | 5.9 | 97% | 15.7 GB | 64.7 (3.4 + 3.6 + 4.2 + 3.7, 4.3 + 3.5 + 3.8 + 3.7, 5.1 + 4.3 + 3.6 + 3.8, 5.8 + 3.5 + 4.3 + 4.1) |
| Sentence of 10 words | 4 | 5 | 20 | f16 | Python (PyTorch) | 3531 (3235, 3448, 3531, 3468) | 5.7 | 97% | 15.7 GB | 81.9 (3.4 + 3.6 + 4.2 + 3.7 + 4.1, 3.3 + 3.8 + 3.4 + 4.7 + 4.3, 3.7 + 3.9 + 5.6 + 3.9 + 4.8, 3.8 + 4.2 + 4.7 + 4.0 + 5.0) |
| Sentence of 10 words | 4 | 6 | 24 | f16 | Python (PyTorch) | 4381 (3936, 4261, 4381, 4371) | 5.5 | 97% | 16.3 GB | 100.9 (3.9 + 3.6 + 4.2 + 3.7 + 4.1 + 3.1, 3.8 + 3.5 + 4.8 + 4.3 + 4.2 + 3.9, 5.8 + 3.8 + 4.3 + 4.2 + 4.0 + 4.5, 4.4 + 4.8 + 3.9 + 4.3 + 4.6 + 5.2) |
| Sentence of 10 words | 4 | 7 | 28 | f16 | Python (PyTorch) | 5164 (4648, 5164, 5049, 5106) | 5.4 | 98% | 17.7 GB | 119.9 (3.4 + 3.7 + 4.2 + 3.7 + 4.3 + 3.6 + 3.8, 3.6 + 4.7 + 4.0 + 4.2 + 4.0 + 5.3 + 3.5, 4.8 + 4.0 + 4.0 + 4.9 + 4.4 + 4.6 + 4.4, 4.3 + 4.1 + 5.2 + 5.1 + 4.7 + 3.8 + 5.6) |
| Sentence of 10 words | 1 | 1 | 1 | bf16 | Python (PyTorch) | 960 | 1.0 | 22% | 5.4 GB | 3.3 (3.3) |
| Sentence of 10 words | 1 | 2 | 2 | bf16 | Python (PyTorch) | 1087 | 1.8 | 32% | 5.4 GB | 6.9 (3.3 + 3.6) |
| Sentence of 10 words | 1 | 3 | 3 | bf16 | Python (PyTorch) | 991 | 3.0 | 44% | 5.4 GB | 11.1 (3.4 + 3.6 + 4.2) |
| Sentence of 10 words | 1 | 4 | 4 | bf16 | Python (PyTorch) | 1001 | 4.0 | 54% | 5.4 GB | 15.0 (3.6 + 3.6 + 4.2 + 3.7) |
| Sentence of 10 words | 1 | 5 | 5 | bf16 | Python (PyTorch) | 1130 | 4.4 | 66% | 5.4 GB | 19.1 (3.7 + 3.6 + 4.2 + 3.5 + 4.1) |
| Sentence of 10 words | 1 | 6 | 6 | bf16 | Python (PyTorch) | 1100 | 5.5 | 77% | 5.4 GB | 23.2 (3.9 + 3.6 + 4.2 + 3.7 + 4.3 + 3.6) |
| Sentence of 10 words | 1 | 7 | 7 | bf16 | Python (PyTorch) | 1151 | 6.1 | 88% | 5.4 GB | 26.7 (3.8 + 3.6 + 4.2 + 3.5 + 4.3 + 3.6 + 3.8) |
| Sentence of 10 words | 1 | 8 | 8 | bf16 | Python (PyTorch) | 1360 | 5.9 | 87% | 5.4 GB | 30.5 (3.7 + 3.6 + 4.0 + 3.7 + 4.3 + 3.6 + 3.8 + 3.7) |
| Sentence of 10 words | 1 | 9 | 9 | bf16 | Python (PyTorch) | 1780 | 5.1 | 82% | 5.4 GB | 34.8 (3.8 + 3.6 + 4.2 + 3.4 + 4.3 + 3.0 + 3.6 + 3.7 + 5.2) |
| Sentence of 10 words | 1 | 10 | 10 | bf16 | Python (PyTorch) | 1984 | 5.0 | 88% | 5.9 GB | 39.5 (3.6 + 3.6 + 4.2 + 3.7 + 4.2 + 3.2 + 3.8 + 3.7 + 5.2 + 4.3) |
| Sentence of 10 words | 2 | 1 | 2 | bf16 | Python (PyTorch) | 1202 (1118, 1202) | 1.7 | 94% | 7.8 GB | 7.0 (3.3, 3.7) |
| Sentence of 10 words | 2 | 2 | 4 | bf16 | Python (PyTorch) | 1010 (1010, 1005) | 4.0 | 94% | 7.8 GB | 14.8 (3.3 + 3.6, 4.2 + 3.7) |
| Sentence of 10 words | 2 | 3 | 6 | bf16 | Python (PyTorch) | 1050 (1028, 1050) | 5.7 | 96% | 7.8 GB | 22.5 (3.4 + 3.6 + 4.2, 3.7 + 4.1 + 3.6) |
| Sentence of 10 words | 2 | 4 | 8 | bf16 | Python (PyTorch) | 1302 (1302, 1301) | 6.1 | 94% | 7.8 GB | 30.2 (3.6 + 3.6 + 4.2 + 3.7, 4.3 + 3.6 + 3.6 + 3.7) |
| Sentence of 10 words | 2 | 5 | 10 | bf16 | Python (PyTorch) | 1724 (1655, 1724) | 5.8 | 94% | 7.8 GB | 38.4 (3.7 + 3.6 + 4.2 + 3.5 + 4.1, 3.3 + 3.8 + 3.3 + 4.6 + 4.3) |
| Sentence of 10 words | 2 | 6 | 12 | bf16 | Python (PyTorch) | 2108 (2024, 2108) | 5.7 | 95% | 7.8 GB | 47.5 (3.9 + 3.6 + 4.2 + 3.7 + 4.3 + 3.6, 3.8 + 3.7 + 4.8 + 3.9 + 4.2 + 3.9) |
| Sentence of 10 words | 2 | 7 | 14 | bf16 | Python (PyTorch) | 2580 (2390, 2580) | 5.4 | 94% | 8.4 GB | 56.1 (3.8 + 3.6 + 4.2 + 3.5 + 4.3 + 3.6 + 3.8, 3.6 + 4.8 + 3.9 + 3.9 + 4.0 + 5.3 + 3.8) |
| Sentence of 10 words | 3 | 1 | 3 | bf16 | Python (PyTorch) | 1553 (1552, 1497, 1553) | 1.9 | 92% | 11.7 GB | 11.1 (3.3, 3.7, 4.1) |
| Sentence of 10 words | 3 | 2 | 6 | bf16 | Python (PyTorch) | 1934 (1934, 1292, 1474) | 3.1 | 76% | 11.7 GB | 22.3 (3.3 + 3.6, 4.2 + 3.7, 4.3 + 3.2) |
| Sentence of 10 words | 3 | 3 | 9 | bf16 | Python (PyTorch) | 1679 (1633, 1632, 1679) | 5.4 | 93% | 11.7 GB | 35.0 (3.4 + 3.6 + 4.2, 3.7 + 4.1 + 3.6, 3.8 + 3.5 + 5.2) |
| Sentence of 10 words | 3 | 4 | 12 | bf16 | Python (PyTorch) | 2013 (1919, 1927, 2013) | 6.0 | 95% | 11.7 GB | 46.6 (3.6 + 3.6 + 4.2 + 3.7, 4.3 + 3.6 + 3.6 + 3.7, 4.7 + 4.3 + 3.5 + 3.8) |
| Sentence of 10 words | 3 | 5 | 15 | bf16 | Python (PyTorch) | 2672 (2478, 2535, 2672) | 5.6 | 96% | 11.7 GB | 60.5 (3.7 + 3.6 + 4.2 + 3.5 + 4.1, 3.3 + 3.8 + 3.3 + 4.6 + 4.3, 3.7 + 3.9 + 5.8 + 3.9 + 4.8) |
| Sentence of 10 words | 3 | 6 | 18 | bf16 | Python (PyTorch) | 3224 (2965, 3120, 3224) | 5.6 | 96% | 12.0 GB | 74.0 (3.9 + 3.6 + 4.2 + 3.7 + 4.3 + 3.6, 3.8 + 3.7 + 4.8 + 3.9 + 4.2 + 3.9, 5.8 + 3.8 + 4.3 + 3.9 + 4.2 + 4.5) |
| Sentence of 10 words | 3 | 7 | 21 | bf16 | Python (PyTorch) | 3844 (3397, 3844, 3750) | 5.5 | 96% | 13.0 GB | 87.9 (3.8 + 3.6 + 4.2 + 3.5 + 4.3 + 3.6 + 3.8, 3.6 + 4.8 + 3.9 + 3.9 + 4.0 + 5.3 + 3.8, 4.8 + 4.2 + 4.1 + 5.2 + 4.4 + 4.8 + 4.2) |
| Sentence of 10 words | 4 | 1 | 4 | bf16 | Python (PyTorch) | 1386 (1321, 1253, 1263, 1386) | 2.9 | 88% | 15.7 GB | 14.6 (3.3, 3.7, 4.1, 3.4) |
| Sentence of 10 words | 4 | 2 | 8 | bf16 | Python (PyTorch) | 1546 (1257, 1546, 1525, 1465) | 5.2 | 94% | 15.7 GB | 29.5 (3.3 + 3.6, 4.2 + 3.7, 4.3 + 3.2, 3.8 + 3.5) |
| Sentence of 10 words | 4 | 3 | 12 | bf16 | Python (PyTorch) | 2054 (1920, 1999, 2054, 2005) | 5.8 | 97% | 15.7 GB | 47.3 (3.4 + 3.6 + 4.2, 3.7 + 4.1 + 3.6, 3.8 + 3.5 + 5.2, 4.3 + 4.1 + 3.9) |
| Sentence of 10 words | 4 | 4 | 16 | bf16 | Python (PyTorch) | 2722 (2501, 2521, 2722, 2716) | 5.9 | 96% | 15.7 GB | 64.1 (3.6 + 3.6 + 4.2 + 3.7, 4.3 + 3.6 + 3.6 + 3.7, 4.7 + 4.3 + 3.5 + 3.8, 5.8 + 3.5 + 4.3 + 3.9) |
| Sentence of 10 words | 4 | 5 | 20 | bf16 | Python (PyTorch) | 3498 (3308, 3395, 3498, 3459) | 5.7 | 96% | 15.7 GB | 82.2 (3.7 + 3.6 + 4.2 + 3.5 + 4.1, 3.3 + 3.8 + 3.3 + 4.6 + 4.3, 3.7 + 3.9 + 5.8 + 3.9 + 4.8, 4.0 + 4.2 + 4.6 + 4.0 + 4.8) |
| Sentence of 10 words | 4 | 6 | 24 | bf16 | Python (PyTorch) | 4345 (3831, 4241, 4314, 4345) | 5.5 | 96% | 16.3 GB | 101.1 (3.9 + 3.6 + 4.2 + 3.7 + 4.3 + 3.6, 3.8 + 3.7 + 4.8 + 3.9 + 4.2 + 3.9, 5.8 + 3.8 + 4.3 + 3.9 + 4.2 + 4.5, 4.3 + 5.0 + 4.1 + 3.9 + 4.6 + 5.2) |
| Sentence of 10 words | 4 | 7 | 28 | bf16 | Python (PyTorch) | 5172 (4490, 5172, 4956, 5168) | 5.4 | 97% | 17.7 GB | 120.3 (3.8 + 3.6 + 4.2 + 3.5 + 4.3 + 3.6 + 3.8, 3.6 + 4.8 + 3.9 + 3.9 + 4.0 + 5.3 + 3.8, 4.8 + 4.2 + 4.1 + 5.2 + 4.4 + 4.8 + 4.2, 4.3 + 4.0 + 5.2 + 5.1 + 4.5 + 3.8 + 5.6) |
| Sentence of 15 words | 1 | 1 | 1 | f16 | Python + FlashInfer | 499 | 2.0 | 25% | 3.7 GB | 6.6 (6.6) |
| Sentence of 15 words | 1 | 2 | 2 | f16 | Python + FlashInfer | 516 | 3.9 | 38% | 3.7 GB | 13.4 (6.4 + 7.0) |
| Sentence of 15 words | 1 | 3 | 3 | f16 | Python + FlashInfer | 552 | 5.4 | 50% | 3.7 GB | 19.4 (6.3 + 6.6 + 6.5) |
| Sentence of 15 words | 1 | 4 | 4 | f16 | Python + FlashInfer | 568 | 7.0 | 62% | 3.7 GB | 26.7 (6.5 + 6.8 + 6.4 + 6.9) |
| Sentence of 15 words | 1 | 5 | 5 | f16 | Python + FlashInfer | 739 | 6.8 | 61% | 3.7 GB | 34.5 (6.5 + 7.1 + 6.5 + 6.9 + 7.5) |
| Sentence of 15 words | 1 | 6 | 6 | f16 | Python + FlashInfer | 776 | 7.7 | 69% | 3.7 GB | 40.6 (6.4 + 6.6 + 6.8 + 6.8 + 7.5 + 6.5) |
| Sentence of 15 words | 1 | 7 | 7 | f16 | Python + FlashInfer | 855 | 8.2 | 76% | 3.9 GB | 48.1 (6.3 + 7.1 + 6.6 + 6.8 + 7.4 + 6.4 + 7.5) |
| Sentence of 15 words | 1 | 8 | 8 | f16 | Python + FlashInfer | 934 | 8.6 | 78% | 4.1 GB | 56.1 (6.3 + 7.1 + 6.8 + 7.0 + 7.5 + 6.4 + 7.3 + 7.7) |
| Sentence of 15 words | 1 | 9 | 9 | f16 | Python + FlashInfer | 1058 | 8.5 | 71% | 4.4 GB | 64.6 (6.4 + 6.6 + 6.5 + 7.1 + 7.4 + 6.7 + 7.5 + 7.9 + 8.5) |
| Sentence of 15 words | 1 | 10 | 10 | f16 | Python + FlashInfer | 1192 | 8.4 | 78% | 4.0 GB | 71.4 (6.8 + 6.6 + 6.8 + 6.9 + 7.5 + 6.7 + 7.2 + 8.1 + 8.4 + 6.5) |
| Sentence of 15 words | 1 | 11 | 11 | f16 | Python + FlashInfer | 1292 | 8.5 | 73% | 4.3 GB | 77.8 (6.4 + 6.6 + 6.4 + 6.6 + 7.6 + 6.7 + 7.5 + 8.1 + 8.8 + 6.9 + 6.1) |
| Sentence of 15 words | 2 | 1 | 2 | f16 | Python + FlashInfer | 622 (559, 622) | 3.2 | 87% | 6.9 GB | 13.1 (6.6, 6.5) |
| Sentence of 15 words | 2 | 2 | 4 | f16 | Python + FlashInfer | 799 (679, 799) | 5.0 | 75% | 6.9 GB | 26.9 (6.4 + 7.0, 6.5 + 7.1) |
| Sentence of 15 words | 2 | 3 | 6 | f16 | Python + FlashInfer | 724 (724, 722) | 8.3 | 84% | 6.9 GB | 40.4 (6.3 + 6.6 + 6.5, 7.1 + 7.9 + 6.0) |
| Sentence of 15 words | 2 | 4 | 8 | f16 | Python + FlashInfer | 915 (910, 915) | 8.7 | 83% | 6.9 GB | 55.7 (6.5 + 6.8 + 6.4 + 6.9, 7.6 + 6.1 + 7.3 + 8.1) |
| Sentence of 15 words | 2 | 5 | 10 | f16 | Python + FlashInfer | 1177 (1132, 1177) | 8.5 | 89% | 7.2 GB | 71.3 (6.5 + 7.1 + 6.5 + 6.9 + 7.5, 6.2 + 7.3 + 8.1 + 8.2 + 7.0) |
| Sentence of 15 words | 2 | 6 | 12 | f16 | Python + FlashInfer | 1378 (1360, 1378) | 8.7 | 87% | 7.6 GB | 83.9 (6.4 + 6.6 + 6.8 + 6.8 + 7.5 + 6.5, 7.2 + 7.7 + 8.2 + 6.7 + 5.6 + 8.0) |
| Sentence of 15 words | 2 | 7 | 14 | f16 | Python + FlashInfer | 1617 (1594, 1617) | 8.7 | 85% | 8.0 GB | 97.9 (6.3 + 7.1 + 6.6 + 6.8 + 7.4 + 6.4 + 7.5, 7.6 + 8.8 + 6.6 + 5.7 + 8.4 + 5.6 + 7.1) |
| Sentence of 15 words | 3 | 1 | 3 | f16 | Python + FlashInfer | 595 (555, 595, 563) | 5.0 | 81% | 12.9 GB | 19.6 (6.6, 6.5, 6.5) |
| Sentence of 15 words | 3 | 2 | 6 | f16 | Python + FlashInfer | 732 (724, 713, 732) | 8.2 | 100% | 12.9 GB | 40.6 (6.4 + 7.0, 6.5 + 7.1, 7.5 + 6.2) |
| Sentence of 15 words | 3 | 3 | 9 | f16 | Python + FlashInfer | 1028 (986, 977, 1028) | 8.8 | 90% | 12.9 GB | 64.1 (6.3 + 6.6 + 6.5, 7.1 + 7.9 + 6.0, 7.1 + 7.7 + 8.8) |
| Sentence of 15 words | 3 | 4 | 12 | f16 | Python + FlashInfer | 1351 (1292, 1319, 1351) | 8.9 | 89% | 12.9 GB | 84.7 (6.5 + 6.8 + 6.4 + 6.9, 7.6 + 6.1 + 7.3 + 8.1, 8.4 + 7.1 + 5.7 + 7.9) |
| Sentence of 15 words | 3 | 5 | 15 | f16 | Python + FlashInfer | 1620 (1581, 1620, 1608) | 9.3 | 94% | 12.9 GB | 103.3 (6.5 + 7.1 + 6.5 + 6.9 + 7.5, 6.2 + 7.3 + 8.1 + 8.2 + 7.0, 5.7 + 7.8 + 5.6 + 6.7 + 6.1) |
| Sentence of 15 words | 3 | 6 | 18 | f16 | Python + FlashInfer | 1952 (1952, 1940, 1886) | 9.2 | 91% | 12.9 GB | 123.8 (6.4 + 6.6 + 6.8 + 6.8 + 7.5 + 6.5, 7.2 + 7.7 + 8.2 + 6.7 + 5.6 + 8.0, 5.5 + 7.1 + 6.2 + 6.7 + 7.1 + 7.2) |
| Sentence of 15 words | 3 | 7 | 21 | f16 | Python + FlashInfer | 2248 (2248, 2214, 2210) | 9.3 | 91% | 12.9 GB | 144.2 (6.3 + 7.1 + 6.6 + 6.8 + 7.4 + 6.4 + 7.5, 7.6 + 8.8 + 6.6 + 5.7 + 8.4 + 5.6 + 7.1, 6.0 + 6.9 + 7.0 + 7.2 + 6.5 + 6.3 + 6.4) |
| Sentence of 15 words | 3 | 8 | 24 | f16 | Python + FlashInfer | 2551 (2551, 2486, 2512) | 9.4 | 92% | 12.9 GB | 166.7 (6.3 + 7.1 + 6.8 + 7.0 + 7.5 + 6.4 + 7.3 + 7.7, 8.8 + 6.8 + 5.8 + 8.4 + 5.8 + 6.9 + 6.0 + 6.7, 7.1 + 7.0 + 7.4 + 6.6 + 6.6 + 7.8 + 6.5 + 6.5) |
| Sentence of 15 words | 3 | 9 | 27 | f16 | Python + FlashInfer | 2832 (2832, 2765, 2793) | 9.5 | 91% | 12.9 GB | 186.8 (6.4 + 6.6 + 6.5 + 7.1 + 7.4 + 6.7 + 7.5 + 7.9 + 8.5, 7.1 + 6.0 + 8.1 + 5.8 + 6.8 + 6.0 + 7.0 + 7.4 + 6.6, 6.9 + 6.2 + 6.3 + 8.1 + 6.7 + 7.0 + 6.7 + 6.4 + 7.2) |
| Sentence of 15 words | 3 | 10 | 30 | f16 | Python + FlashInfer | 3120 (3120, 3066, 3103) | 9.6 | 90% | 13.1 GB | 203.6 (6.8 + 6.6 + 6.8 + 6.9 + 7.5 + 6.7 + 7.2 + 8.1 + 8.4 + 6.5, 5.8 + 7.8 + 5.7 + 7.0 + 5.9 + 6.3 + 6.9 + 6.5 + 6.7 + 6.6, 6.5 + 7.6 + 7.0 + 6.6 + 6.7 + 6.3 + 7.4 + 6.7 + 6.0 + 6.1) |
| Sentence of 15 words | 3 | 11 | 33 | f16 | Python + FlashInfer | 3416 (3416, 3365, 3270) | 9.7 | 91% | 13.8 GB | 224.5 (6.4 + 6.6 + 6.4 + 6.6 + 7.6 + 6.7 + 7.5 + 8.1 + 8.8 + 6.9 + 6.1, 7.9 + 5.6 + 7.1 + 6.1 + 6.6 + 7.4 + 6.9 + 7.1 + 6.4 + 6.4 + 7.9, 6.5 + 6.7 + 6.7 + 6.6 + 7.2 + 6.7 + 6.0 + 6.1 + 6.3 + 5.9 + 6.6) |
| Sentence of 15 words | 3 | 12 | 36 | f16 | Python + FlashInfer | 3651 (3651, 3640, 3549) | 9.9 | 93% | 14.8 GB | 244.3 (6.4 + 6.7 + 6.8 + 7.0 + 7.5 + 6.7 + 7.5 + 8.1 + 8.6 + 7.1 + 5.8 + 8.4, 5.8 + 7.1 + 5.7 + 6.6 + 7.4 + 7.2 + 7.0 + 6.5 + 6.5 + 7.9 + 6.8 + 6.5, 6.6 + 6.6 + 7.5 + 6.7 + 5.8 + 6.6 + 6.2 + 5.9 + 6.6 + 5.8 + 6.7 + 6.1) |
| Sentence of 15 words | 3 | 13 | 39 | f16 | Python + FlashInfer | 3959 (3959, 3899, 3915) | 9.9 | 93% | 15.8 GB | 265.9 (6.5 + 7.1 + 6.5 + 7.0 + 8.0 + 6.7 + 7.5 + 8.1 + 8.4 + 7.1 + 6.1 + 7.8 + 5.8, 6.9 + 6.3 + 6.7 + 7.4 + 6.9 + 7.5 + 6.6 + 6.6 + 8.1 + 7.0 + 6.9 + 6.6 + 6.6, 7.0 + 6.3 + 6.0 + 6.2 + 6.1 + 5.5 + 6.3 + 6.2 + 7.4 + 6.4 + 7.1 + 5.8 + 7.3) |
| Sentence of 15 words | 3 | 14 | 42 | f16 | Python + FlashInfer | 4319 (4319, 4204, 4122) | 9.7 | 93% | 16.8 GB | 282.9 (6.4 + 6.6 + 6.4 + 6.9 + 7.7 + 6.3 + 7.1 + 8.1 + 8.4 + 7.1 + 5.7 + 8.4 + 5.8 + 7.1, 5.8 + 7.0 + 7.4 + 7.0 + 6.5 + 6.2 + 6.4 + 7.9 + 6.5 + 7.0 + 6.7 + 6.5 + 7.0 + 6.7, 6.0 + 6.2 + 6.3 + 5.5 + 6.6 + 6.1 + 7.0 + 6.1 + 6.6 + 5.7 + 7.3 + 6.5 + 7.6 + 7.0) |
| Sentence of 15 words | 3 | 15 | 45 | f16 | Python + FlashInfer | 4586 (4559, 4586, 4527) | 9.8 | 93% | 17.9 GB | 308.1 (6.4 + 6.9 + 6.8 + 7.0 + 7.4 + 6.7 + 7.5 + 8.1 + 8.4 + 7.1 + 5.7 + 8.1 + 5.8 + 7.0 + 6.1, 7.0 + 7.4 + 7.1 + 7.1 + 6.1 + 6.6 + 8.1 + 6.9 + 6.6 + 6.7 + 6.6 + 7.4 + 6.7 + 6.0 + 6.2, 6.3 + 5.6 + 6.2 + 6.4 + 7.4 + 6.2 + 6.8 + 6.2 + 6.9 + 6.5 + 8.1 + 7.1 + 7.0 + 7.7 + 6.5) |
| Sentence of 15 words | 4 | 1 | 4 | f16 | Python + FlashInfer | 720 (717, 720, 599, 596) | 5.6 | 90% | 13.9 GB | 26.5 (6.6, 6.5, 6.5, 6.9) |
| Sentence of 15 words | 4 | 2 | 8 | f16 | Python + FlashInfer | 1089 (879, 1089, 946, 921) | 7.3 | 92% | 13.9 GB | 56.0 (6.4 + 7.0, 6.5 + 7.1, 7.5 + 6.2, 7.3 + 8.1) |
| Sentence of 15 words | 4 | 3 | 12 | f16 | Python + FlashInfer | 1309 (1268, 1237, 1309, 1230) | 9.2 | 92% | 13.9 GB | 84.8 (6.3 + 6.6 + 6.5, 7.1 + 7.9 + 6.0, 7.1 + 7.7 + 8.8, 6.7 + 6.2 + 8.0) |
| Sentence of 15 words | 4 | 4 | 16 | f16 | Python + FlashInfer | 1692 (1650, 1690, 1692, 1630) | 9.5 | 93% | 13.9 GB | 110.2 (6.5 + 6.8 + 6.4 + 6.9, 7.6 + 6.1 + 7.3 + 8.1, 8.4 + 7.1 + 5.7 + 7.9, 5.6 + 7.1 + 5.8 + 7.0) |
| Sentence of 15 words | 4 | 5 | 20 | f16 | Python + FlashInfer | 2127 (2089, 2127, 2068, 2089) | 9.4 | 94% | 14.1 GB | 137.6 (6.5 + 7.1 + 6.5 + 6.9 + 7.5, 6.2 + 7.3 + 8.1 + 8.2 + 7.0, 5.7 + 7.8 + 5.6 + 6.7 + 6.1, 6.7 + 7.4 + 7.2 + 6.9 + 6.1) |
| Sentence of 15 words | 4 | 6 | 24 | f16 | Python + FlashInfer | 2470 (2442, 2470, 2431, 2455) | 9.7 | 95% | 14.9 GB | 165.0 (6.4 + 6.6 + 6.8 + 6.8 + 7.5 + 6.5, 7.2 + 7.7 + 8.2 + 6.7 + 5.6 + 8.0, 5.5 + 7.1 + 6.2 + 6.7 + 7.1 + 7.2, 6.6 + 6.2 + 6.6 + 7.9 + 7.0 + 7.0) |
| Sentence of 15 words | 4 | 7 | 28 | f16 | Python + FlashInfer | 2890 (2837, 2871, 2823, 2890) | 9.7 | 94% | 15.8 GB | 193.5 (6.3 + 7.1 + 6.6 + 6.8 + 7.4 + 6.4 + 7.5, 7.6 + 8.8 + 6.6 + 5.7 + 8.4 + 5.6 + 7.1, 6.0 + 6.9 + 7.0 + 7.2 + 6.5 + 6.3 + 6.4, 7.8 + 7.0 + 7.0 + 6.6 + 6.3 + 7.9 + 6.7) |
| Sentence of 15 words | 4 | 8 | 32 | f16 | Python + FlashInfer | 3428 (3428, 3189, 3187, 3119) | 9.3 | 93% | 16.8 GB | 217.1 (6.3 + 7.1 + 6.8 + 7.0 + 7.5 + 6.4 + 7.3 + 7.7, 8.8 + 6.8 + 5.8 + 8.4 + 5.8 + 6.9 + 6.0 + 6.7, 7.1 + 7.0 + 7.4 + 6.6 + 6.6 + 7.8 + 6.5 + 6.5, 6.5 + 6.6 + 7.3 + 6.2 + 6.0 + 6.1 + 6.2 + 5.6) |
| Sentence of 15 words | 4 | 9 | 36 | f16 | Python + FlashInfer | 3589 (3585, 3564, 3589, 3397) | 10.0 | 95% | 17.9 GB | 243.1 (6.4 + 6.6 + 6.5 + 7.1 + 7.4 + 6.7 + 7.5 + 7.9 + 8.5, 7.1 + 6.0 + 8.1 + 5.8 + 6.8 + 6.0 + 7.0 + 7.4 + 6.6, 6.9 + 6.2 + 6.3 + 8.1 + 6.7 + 7.0 + 6.7 + 6.4 + 7.2, 6.4 + 5.9 + 6.1 + 6.2 + 5.9 + 6.1 + 6.4 + 7.0 + 6.4) |
| Sentence of 15 words | 4 | 10 | 40 | f16 | Python + FlashInfer | 3993 (3993, 3943, 3922, 3796) | 10.0 | 95% | 19.1 GB | 268.1 (6.8 + 6.6 + 6.8 + 6.9 + 7.5 + 6.7 + 7.2 + 8.1 + 8.4 + 6.5, 5.8 + 7.8 + 5.7 + 7.0 + 5.9 + 6.3 + 6.9 + 6.5 + 6.7 + 6.6, 6.5 + 7.6 + 7.0 + 6.6 + 6.7 + 6.3 + 7.4 + 6.7 + 6.0 + 6.1, 6.2 + 5.9 + 6.2 + 6.4 + 7.0 + 6.4 + 6.8 + 5.9 + 7.1 + 6.7) |
| Sentence of 15 words | 4 | 11 | 44 | f16 | Python + FlashInfer | 4396 (4392, 4396, 4207, 4316) | 10.0 | 96% | 20.3 GB | 299.0 (6.4 + 6.6 + 6.4 + 6.6 + 7.6 + 6.7 + 7.5 + 8.1 + 8.8 + 6.9 + 6.1, 7.9 + 5.6 + 7.1 + 6.1 + 6.6 + 7.4 + 6.9 + 7.1 + 6.4 + 6.4 + 7.9, 6.5 + 6.7 + 6.7 + 6.6 + 7.2 + 6.7 + 6.0 + 6.1 + 6.3 + 5.9 + 6.6, 5.9 + 6.5 + 6.4 + 7.1 + 5.8 + 7.2 + 6.6 + 7.5 + 7.0 + 6.7 + 7.6) |
| Sentence of 15 words | 4 | 12 | 48 | f16 | Python + FlashInfer | 4765 (4765, 4713, 4761, 4733) | 10.1 | 96% | 21.6 GB | 325.7 (6.4 + 6.7 + 6.8 + 7.0 + 7.5 + 6.7 + 7.5 + 8.1 + 8.6 + 7.1 + 5.8 + 8.4, 5.8 + 7.1 + 5.7 + 6.6 + 7.4 + 7.2 + 7.0 + 6.5 + 6.5 + 7.9 + 6.8 + 6.5, 6.6 + 6.6 + 7.5 + 6.7 + 5.8 + 6.6 + 6.2 + 5.9 + 6.6 + 5.8 + 6.7 + 6.1, 6.7 + 5.9 + 7.3 + 6.4 + 7.6 + 6.8 + 7.0 + 7.3 + 6.5 + 7.6 + 5.9 + 6.4) |
| Sentence of 15 words | 4 | 13 | 52 | f16 | Python + FlashInfer | 5227 (5227, 5198, 5070, 5172) | 9.9 | 95% | 23.1 GB | 359.2 (6.5 + 7.1 + 6.5 + 7.0 + 8.0 + 6.7 + 7.5 + 8.1 + 8.4 + 7.1 + 6.1 + 7.8 + 5.8, 6.9 + 6.3 + 6.7 + 7.4 + 6.9 + 7.5 + 6.6 + 6.6 + 8.1 + 7.0 + 6.9 + 6.6 + 6.6, 7.0 + 6.3 + 6.0 + 6.2 + 6.1 + 5.5 + 6.3 + 6.2 + 7.4 + 6.4 + 7.1 + 5.8 + 7.3, 6.5 + 8.1 + 6.9 + 7.3 + 8.3 + 6.5 + 8.2 + 6.0 + 6.1 + 7.1 + 7.5 + 8.0 + 6.8) |
| Sentence of 15 words | 4 | 14 | 56 | f16 | Python + FlashInfer | 5612 (5543, 5541, 5412, 5612) | 10.0 | 96% | 14.9 GB | 381.8 (6.4 + 6.6 + 6.4 + 6.9 + 7.7 + 6.3 + 7.1 + 8.1 + 8.4 + 7.1 + 5.7 + 8.4 + 5.8 + 7.1, 5.8 + 7.0 + 7.4 + 7.0 + 6.5 + 6.2 + 6.4 + 7.9 + 6.5 + 7.0 + 6.7 + 6.5 + 7.0 + 6.7, 6.0 + 6.2 + 6.3 + 5.5 + 6.6 + 6.1 + 7.0 + 6.1 + 6.6 + 5.7 + 7.3 + 6.5 + 7.6 + 7.0, 7.0 + 7.7 + 6.7 + 8.2 + 5.9 + 6.3 + 7.3 + 7.8 + 7.8 + 6.7 + 6.1 + 6.4 + 7.5 + 7.4) |
| Sentence of 15 words | 4 | 15 | 60 | f16 | Python + FlashInfer | 6048 (6033, 6048, 5916, 5961) | 9.9 | 96% | 16.6 GB | 413.2 (6.4 + 6.9 + 6.8 + 7.0 + 7.4 + 6.7 + 7.5 + 8.1 + 8.4 + 7.1 + 5.7 + 8.1 + 5.8 + 7.0 + 6.1, 7.0 + 7.4 + 7.1 + 7.1 + 6.1 + 6.6 + 8.1 + 6.9 + 6.6 + 6.7 + 6.6 + 7.4 + 6.7 + 6.0 + 6.2, 6.3 + 5.6 + 6.2 + 6.4 + 7.4 + 6.2 + 6.8 + 6.2 + 6.9 + 6.5 + 8.1 + 7.1 + 7.0 + 7.7 + 6.5, 8.2 + 5.9 + 6.1 + 7.1 + 8.0 + 7.4 + 7.1 + 6.8 + 6.7 + 6.9 + 7.1 + 7.1 + 6.9 + 7.0 + 6.8) |
| Sentence of 15 words | 5 | 1 | 5 | f16 | Python + FlashInfer | 729 (698, 699, 694, 708, 729) | 6.9 | 87% | 21.4 GB | 33.6 (6.6, 6.5, 6.5, 6.9, 7.1) |
| Sentence of 15 words | 5 | 2 | 10 | f16 | Python + FlashInfer | 1216 (1159, 1166, 1176, 1216, 1196) | 8.2 | 92% | 21.4 GB | 71.3 (6.4 + 7.0, 6.5 + 7.1, 7.5 + 6.2, 7.3 + 8.1, 8.5 + 6.8) |
| Sentence of 15 words | 5 | 3 | 15 | f16 | Python + FlashInfer | 1575 (1515, 1510, 1575, 1503, 1483) | 9.5 | 93% | 21.4 GB | 103.6 (6.3 + 6.6 + 6.5, 7.1 + 7.9 + 6.0, 7.1 + 7.7 + 8.8, 6.7 + 6.2 + 8.0, 5.7 + 6.9 + 6.3) |
| Sentence of 15 words | 5 | 4 | 20 | f16 | Python + FlashInfer | 2118 (2028, 2080, 2118, 1983, 2064) | 9.4 | 95% | 21.4 GB | 137.4 (6.5 + 6.8 + 6.4 + 6.9, 7.6 + 6.1 + 7.3 + 8.1, 8.4 + 7.1 + 5.7 + 7.9, 5.6 + 7.1 + 5.8 + 7.0, 7.1 + 7.0 + 6.8 + 6.3) |
| Sentence of 15 words | 5 | 5 | 25 | f16 | Python + FlashInfer | 2585 (2585, 2580, 2497, 2530, 2516) | 9.7 | 97% | 21.4 GB | 172.8 (6.5 + 7.1 + 6.5 + 6.9 + 7.5, 6.2 + 7.3 + 8.1 + 8.2 + 7.0, 5.7 + 7.8 + 5.6 + 6.7 + 6.1, 6.7 + 7.4 + 7.2 + 6.9 + 6.1, 6.5 + 8.1 + 7.0 + 7.0 + 6.7) |
| Sentence of 15 words | 5 | 6 | 30 | f16 | Python + FlashInfer | 3051 (3009, 3051, 2996, 3007, 2995) | 9.8 | 96% | 21.4 GB | 204.0 (6.4 + 6.6 + 6.8 + 6.8 + 7.5 + 6.5, 7.2 + 7.7 + 8.2 + 6.7 + 5.6 + 8.0, 5.5 + 7.1 + 6.2 + 6.7 + 7.1 + 7.2, 6.6 + 6.2 + 6.6 + 7.9 + 7.0 + 7.0, 6.6 + 6.5 + 7.4 + 6.7 + 5.7 + 6.1) |
| Sentence of 15 words | 5 | 7 | 35 | f16 | Python + FlashInfer | 3525 (3509, 3525, 3480, 3487, 3343) | 9.9 | 97% | 21.4 GB | 236.5 (6.3 + 7.1 + 6.6 + 6.8 + 7.4 + 6.4 + 7.5, 7.6 + 8.8 + 6.6 + 5.7 + 8.4 + 5.6 + 7.1, 6.0 + 6.9 + 7.0 + 7.2 + 6.5 + 6.3 + 6.4, 7.8 + 7.0 + 7.0 + 6.6 + 6.3 + 7.9 + 6.7, 5.7 + 6.2 + 6.3 + 5.5 + 6.0 + 6.3 + 7.0) |
| Sentence of 15 words | 5 | 8 | 40 | f16 | Python + FlashInfer | 3962 (3962, 3948, 3945, 3877, 3857) | 10.1 | 96% | 21.4 GB | 268.7 (6.3 + 7.1 + 6.8 + 7.0 + 7.5 + 6.4 + 7.3 + 7.7, 8.8 + 6.8 + 5.8 + 8.4 + 5.8 + 6.9 + 6.0 + 6.7, 7.1 + 7.0 + 7.4 + 6.6 + 6.6 + 7.8 + 6.5 + 6.5, 6.5 + 6.6 + 7.3 + 6.2 + 6.0 + 6.1 + 6.2 + 5.6, 6.1 + 6.0 + 6.8 + 5.8 + 7.1 + 6.2 + 7.1 + 6.5) |
| Sentence of 15 words | 5 | 9 | 45 | f16 | Python + FlashInfer | 4461 (4461, 4363, 4435, 4226, 4423) | 10.1 | 96% | 21.4 GB | 305.9 (6.4 + 6.6 + 6.5 + 7.1 + 7.4 + 6.7 + 7.5 + 7.9 + 8.5, 7.1 + 6.0 + 8.1 + 5.8 + 6.8 + 6.0 + 7.0 + 7.4 + 6.6, 6.9 + 6.2 + 6.3 + 8.1 + 6.7 + 7.0 + 6.7 + 6.4 + 7.2, 6.4 + 5.9 + 6.1 + 6.2 + 5.9 + 6.1 + 6.4 + 7.0 + 6.4, 6.6 + 6.2 + 7.3 + 7.0 + 7.6 + 7.4 + 7.0 + 7.3 + 6.5) |
| Sentence of 15 words | 5 | 10 | 50 | f16 | Python + FlashInfer | 4965 (4943, 4875, 4965, 4699, 4964) | 10.1 | 96% | 21.8 GB | 340.4 (6.8 + 6.6 + 6.8 + 6.9 + 7.5 + 6.7 + 7.2 + 8.1 + 8.4 + 6.5, 5.8 + 7.8 + 5.7 + 7.0 + 5.9 + 6.3 + 6.9 + 6.5 + 6.7 + 6.6, 6.5 + 7.6 + 7.0 + 6.6 + 6.7 + 6.3 + 7.4 + 6.7 + 6.0 + 6.1, 6.2 + 5.9 + 6.2 + 6.4 + 7.0 + 6.4 + 6.8 + 5.9 + 7.1 + 6.7, 7.6 + 7.3 + 7.1 + 7.8 + 6.7 + 8.2 + 6.1 + 6.0 + 7.6 + 8.0) |
| Sentence of 15 words | 5 | 11 | 55 | f16 | Python + FlashInfer | 5392 (5372, 5392, 5145, 5383, 5386) | 10.2 | 97% | 23.0 GB | 377.4 (6.4 + 6.6 + 6.4 + 6.6 + 7.6 + 6.7 + 7.5 + 8.1 + 8.8 + 6.9 + 6.1, 7.9 + 5.6 + 7.1 + 6.1 + 6.6 + 7.4 + 6.9 + 7.1 + 6.4 + 6.4 + 7.9, 6.5 + 6.7 + 6.7 + 6.6 + 7.2 + 6.7 + 6.0 + 6.1 + 6.3 + 5.9 + 6.6, 5.9 + 6.5 + 6.4 + 7.1 + 5.8 + 7.2 + 6.6 + 7.5 + 7.0 + 6.7 + 7.6, 6.5 + 8.2 + 6.0 + 6.4 + 7.6 + 7.7 + 8.2 + 7.1 + 6.8 + 6.4 + 7.5) |
| Sentence of 15 words | 5 | 12 | 60 | f16 | Python + FlashInfer | 5955 (5955, 5860, 5667, 5920, 5874) | 10.1 | 97% | 18.8 GB | 409.4 (6.4 + 6.7 + 6.8 + 7.0 + 7.5 + 6.7 + 7.5 + 8.1 + 8.6 + 7.1 + 5.8 + 8.4, 5.8 + 7.1 + 5.7 + 6.6 + 7.4 + 7.2 + 7.0 + 6.5 + 6.5 + 7.9 + 6.8 + 6.5, 6.6 + 6.6 + 7.5 + 6.7 + 5.8 + 6.6 + 6.2 + 5.9 + 6.6 + 5.8 + 6.7 + 6.1, 6.7 + 5.9 + 7.3 + 6.4 + 7.6 + 6.8 + 7.0 + 7.3 + 6.5 + 7.6 + 5.9 + 6.4, 7.0 + 7.9 + 7.7 + 6.3 + 6.8 + 6.7 + 7.0 + 6.9 + 6.9 + 6.9 + 7.1 + 6.5) |
| Sentence of 15 words | 5 | 13 | 65 | f16 | Python + FlashInfer | 6429 (6376, 6429, 6151, 6420, 6318) | 10.1 | 96% | 20.7 GB | 445.9 (6.5 + 7.1 + 6.5 + 7.0 + 8.0 + 6.7 + 7.5 + 8.1 + 8.4 + 7.1 + 6.1 + 7.8 + 5.8, 6.9 + 6.3 + 6.7 + 7.4 + 6.9 + 7.5 + 6.6 + 6.6 + 8.1 + 7.0 + 6.9 + 6.6 + 6.6, 7.0 + 6.3 + 6.0 + 6.2 + 6.1 + 5.5 + 6.3 + 6.2 + 7.4 + 6.4 + 7.1 + 5.8 + 7.3, 6.5 + 8.1 + 6.9 + 7.3 + 8.3 + 6.5 + 8.2 + 6.0 + 6.1 + 7.1 + 7.5 + 8.0 + 6.8, 6.5 + 6.3 + 7.5 + 6.6 + 7.2 + 7.1 + 6.9 + 6.6 + 7.1 + 5.8 + 5.7 + 6.7 + 6.6) |
| Sentence of 15 words | 5 | 14 | 70 | f16 | Python + FlashInfer | 6898 (6898, 6836, 6704, 6896, 6829) | 10.1 | 97% | 22.3 GB | 476.6 (6.4 + 6.6 + 6.4 + 6.9 + 7.7 + 6.3 + 7.1 + 8.1 + 8.4 + 7.1 + 5.7 + 8.4 + 5.8 + 7.1, 5.8 + 7.0 + 7.4 + 7.0 + 6.5 + 6.2 + 6.4 + 7.9 + 6.5 + 7.0 + 6.7 + 6.5 + 7.0 + 6.7, 6.0 + 6.2 + 6.3 + 5.5 + 6.6 + 6.1 + 7.0 + 6.1 + 6.6 + 5.7 + 7.3 + 6.5 + 7.6 + 7.0, 7.0 + 7.7 + 6.7 + 8.2 + 5.9 + 6.3 + 7.3 + 7.8 + 7.8 + 6.7 + 6.1 + 6.4 + 7.5 + 7.4, 7.0 + 6.9 + 6.8 + 6.5 + 7.3 + 6.0 + 6.0 + 7.4 + 6.6 + 7.3 + 7.5 + 6.6 + 6.2 + 6.7) |
| Sentence of 15 words | 6 | 1 | 6 | f16 | Python + FlashInfer | 888 (834, 728, 781, 879, 794, 888) | 6.8 | 95% | 20.8 GB | 39.9 (6.6, 6.5, 6.5, 6.9, 7.1, 6.2) |
| Sentence of 15 words | 6 | 2 | 12 | f16 | Python + FlashInfer | 1452 (1380, 1387, 1406, 1452, 1430, 1398) | 8.3 | 99% | 20.8 GB | 85.4 (6.4 + 7.0, 6.5 + 7.1, 7.5 + 6.2, 7.3 + 8.1, 8.5 + 6.8, 6.1 + 8.0) |
| Sentence of 15 words | 6 | 3 | 18 | f16 | Python + FlashInfer | 1880 (1811, 1808, 1880, 1787, 1759, 1811) | 9.6 | 94% | 20.8 GB | 124.0 (6.3 + 6.6 + 6.5, 7.1 + 7.9 + 6.0, 7.1 + 7.7 + 8.8, 6.7 + 6.2 + 8.0, 5.7 + 6.9 + 6.3, 6.6 + 7.0 + 6.7) |
| Sentence of 15 words | 6 | 4 | 24 | f16 | Python + FlashInfer | 2491 (2416, 2488, 2491, 2369, 2463, 2467) | 9.6 | 96% | 20.8 GB | 165.6 (6.5 + 6.8 + 6.4 + 6.9, 7.6 + 6.1 + 7.3 + 8.1, 8.4 + 7.1 + 5.7 + 7.9, 5.6 + 7.1 + 5.8 + 7.0, 7.1 + 7.0 + 6.8 + 6.3, 6.5 + 8.1 + 6.6 + 7.0) |
| Sentence of 15 words | 6 | 5 | 30 | f16 | Python + FlashInfer | 3084 (3067, 3084, 2979, 3030, 3010, 3026) | 9.7 | 95% | 21.0 GB | 204.4 (6.5 + 7.1 + 6.5 + 6.9 + 7.5, 6.2 + 7.3 + 8.1 + 8.2 + 7.0, 5.7 + 7.8 + 5.6 + 6.7 + 6.1, 6.7 + 7.4 + 7.2 + 6.9 + 6.1, 6.5 + 8.1 + 7.0 + 7.0 + 6.7, 6.3 + 7.0 + 6.3 + 6.0 + 6.0) |
| Sentence of 15 words | 6 | 6 | 36 | f16 | Python + FlashInfer | 3603 (3603, 3575, 3524, 3549, 3547, 3529) | 10.0 | 96% | 22.1 GB | 241.6 (6.4 + 6.6 + 6.8 + 6.8 + 7.5 + 6.5, 7.2 + 7.7 + 8.2 + 6.7 + 5.6 + 8.0, 5.5 + 7.1 + 6.2 + 6.7 + 7.1 + 7.2, 6.6 + 6.2 + 6.6 + 7.9 + 7.0 + 7.0, 6.6 + 6.5 + 7.4 + 6.7 + 5.7 + 6.1, 6.0 + 5.9 + 6.5 + 6.0 + 7.0 + 6.2) |
| Sentence of 15 words | 6 | 7 | 42 | f16 | Python + FlashInfer | 4185 (4172, 4185, 4110, 4160, 4002, 4146) | 10.0 | 97% | 23.4 GB | 283.6 (6.3 + 7.1 + 6.6 + 6.8 + 7.4 + 6.4 + 7.5, 7.6 + 8.8 + 6.6 + 5.7 + 8.4 + 5.6 + 7.1, 6.0 + 6.9 + 7.0 + 7.2 + 6.5 + 6.3 + 6.4, 7.8 + 7.0 + 7.0 + 6.6 + 6.3 + 7.9 + 6.7, 5.7 + 6.2 + 6.3 + 5.5 + 6.0 + 6.3 + 7.0, 6.2 + 6.8 + 6.0 + 6.8 + 6.3 + 8.1 + 6.9) |
| Sentence of 15 words | 6 | 8 | 48 | f16 | Python + FlashInfer | 4744 (4690, 4698, 4721, 4563, 4608, 4744) | 10.1 | 97% | 23.3 GB | 324.4 (6.3 + 7.1 + 6.8 + 7.0 + 7.5 + 6.4 + 7.3 + 7.7, 8.8 + 6.8 + 5.8 + 8.4 + 5.8 + 6.9 + 6.0 + 6.7, 7.1 + 7.0 + 7.4 + 6.6 + 6.6 + 7.8 + 6.5 + 6.5, 6.5 + 6.6 + 7.3 + 6.2 + 6.0 + 6.1 + 6.2 + 5.6, 6.1 + 6.0 + 6.8 + 5.8 + 7.1 + 6.2 + 7.1 + 6.5, 7.6 + 6.9 + 7.1 + 7.5 + 6.6 + 8.2 + 5.9 + 6.0) |
| Sentence of 15 words | 6 | 9 | 54 | f16 | Python + FlashInfer | 5293 (5293, 5213, 5270, 5039, 5274, 5254) | 10.2 | 100% | 23.4 GB | 369.2 (6.4 + 6.6 + 6.5 + 7.1 + 7.4 + 6.7 + 7.5 + 7.9 + 8.5, 7.1 + 6.0 + 8.1 + 5.8 + 6.8 + 6.0 + 7.0 + 7.4 + 6.6, 6.9 + 6.2 + 6.3 + 8.1 + 6.7 + 7.0 + 6.7 + 6.4 + 7.2, 6.4 + 5.9 + 6.1 + 6.2 + 5.9 + 6.1 + 6.4 + 7.0 + 6.4, 6.6 + 6.2 + 7.3 + 7.0 + 7.6 + 7.4 + 7.0 + 7.3 + 6.5, 7.8 + 5.9 + 6.3 + 7.0 + 8.0 + 8.2 + 7.1 + 6.6 + 6.4) |
| Sentence of 15 words | 6 | 10 | 60 | f16 | Python + FlashInfer | 5928 (5898, 5786, 5791, 5616, 5928, 5857) | 10.1 | 97% | 23.2 GB | 410.2 (6.8 + 6.6 + 6.8 + 6.9 + 7.5 + 6.7 + 7.2 + 8.1 + 8.4 + 6.5, 5.8 + 7.8 + 5.7 + 7.0 + 5.9 + 6.3 + 6.9 + 6.5 + 6.7 + 6.6, 6.5 + 7.6 + 7.0 + 6.6 + 6.7 + 6.3 + 7.4 + 6.7 + 6.0 + 6.1, 6.2 + 5.9 + 6.2 + 6.4 + 7.0 + 6.4 + 6.8 + 5.9 + 7.1 + 6.7, 7.6 + 7.3 + 7.1 + 7.8 + 6.7 + 8.2 + 6.1 + 6.0 + 7.6 + 8.0, 7.9 + 6.7 + 6.8 + 6.3 + 7.5 + 7.2 + 7.2 + 6.9 + 6.9 + 6.3) |
| Sentence of 15 words | 6 | 11 | 66 | f16 | Python + FlashInfer | 6461 (6461, 6363, 6220, 6428, 6389, 6332) | 10.2 | 98% | 22.2 GB | 452.6 (6.4 + 6.6 + 6.4 + 6.6 + 7.6 + 6.7 + 7.5 + 8.1 + 8.8 + 6.9 + 6.1, 7.9 + 5.6 + 7.1 + 6.1 + 6.6 + 7.4 + 6.9 + 7.1 + 6.4 + 6.4 + 7.9, 6.5 + 6.7 + 6.7 + 6.6 + 7.2 + 6.7 + 6.0 + 6.1 + 6.3 + 5.9 + 6.6, 5.9 + 6.5 + 6.4 + 7.1 + 5.8 + 7.2 + 6.6 + 7.5 + 7.0 + 6.7 + 7.6, 6.5 + 8.2 + 6.0 + 6.4 + 7.6 + 7.7 + 8.2 + 7.1 + 6.8 + 6.4 + 7.5, 7.1 + 6.9 + 7.1 + 7.1 + 6.3 + 7.4 + 6.1 + 6.0 + 7.3 + 6.9 + 7.1) |
| Sentence of 15 words | 6 | 12 | 72 | f16 | Python + FlashInfer | 7066 (7066, 6933, 6818, 6989, 7035, 6948) | 10.2 | 97% | 21.8 GB | 490.1 (6.4 + 6.7 + 6.8 + 7.0 + 7.5 + 6.7 + 7.5 + 8.1 + 8.6 + 7.1 + 5.8 + 8.4, 5.8 + 7.1 + 5.7 + 6.6 + 7.4 + 7.2 + 7.0 + 6.5 + 6.5 + 7.9 + 6.8 + 6.5, 6.6 + 6.6 + 7.5 + 6.7 + 5.8 + 6.6 + 6.2 + 5.9 + 6.6 + 5.8 + 6.7 + 6.1, 6.7 + 5.9 + 7.3 + 6.4 + 7.6 + 6.8 + 7.0 + 7.3 + 6.5 + 7.6 + 5.9 + 6.4, 7.0 + 7.9 + 7.7 + 6.3 + 6.8 + 6.7 + 7.0 + 6.9 + 6.9 + 6.9 + 7.1 + 6.5, 6.9 + 6.0 + 5.9 + 7.1 + 7.1 + 7.1 + 7.5 + 6.6 + 6.2 + 6.3 + 7.8 + 6.3) |
| Sentence of 15 words | 6 | 13 | 78 | f16 | Python + FlashInfer | 7665 (7646, 7578, 7369, 7665, 7665, 7626) | 10.2 | 98% | 21.7 GB | 534.9 (6.5 + 7.1 + 6.5 + 7.0 + 8.0 + 6.7 + 7.5 + 8.1 + 8.4 + 7.1 + 6.1 + 7.8 + 5.8, 6.9 + 6.3 + 6.7 + 7.4 + 6.9 + 7.5 + 6.6 + 6.6 + 8.1 + 7.0 + 6.9 + 6.6 + 6.6, 7.0 + 6.3 + 6.0 + 6.2 + 6.1 + 5.5 + 6.3 + 6.2 + 7.4 + 6.4 + 7.1 + 5.8 + 7.3, 6.5 + 8.1 + 6.9 + 7.3 + 8.3 + 6.5 + 8.2 + 6.0 + 6.1 + 7.1 + 7.5 + 8.0 + 6.8, 6.5 + 6.3 + 7.5 + 6.6 + 7.2 + 7.1 + 6.9 + 6.6 + 7.1 + 5.8 + 5.7 + 6.7 + 6.6, 6.8 + 7.5 + 6.2 + 6.6 + 6.3 + 7.8 + 6.7 + 6.2 + 6.7 + 7.6 + 7.4 + 6.2 + 7.2) |
| Sentence of 15 words | 6 | 14 | 84 | f16 | Python + FlashInfer | 8192 (8130, 8192, 8085, 8172, 8125, 8120) | 10.3 | 98% | 21.9 GB | 574.7 (6.4 + 6.6 + 6.4 + 6.9 + 7.7 + 6.3 + 7.1 + 8.1 + 8.4 + 7.1 + 5.7 + 8.4 + 5.8 + 7.1, 5.8 + 7.0 + 7.4 + 7.0 + 6.5 + 6.2 + 6.4 + 7.9 + 6.5 + 7.0 + 6.7 + 6.5 + 7.0 + 6.7, 6.0 + 6.2 + 6.3 + 5.5 + 6.6 + 6.1 + 7.0 + 6.1 + 6.6 + 5.7 + 7.3 + 6.5 + 7.6 + 7.0, 7.0 + 7.7 + 6.7 + 8.2 + 5.9 + 6.3 + 7.3 + 7.8 + 7.8 + 6.7 + 6.1 + 6.4 + 7.5 + 7.4, 7.0 + 6.9 + 6.8 + 6.5 + 7.3 + 6.0 + 6.0 + 7.4 + 6.6 + 7.3 + 7.5 + 6.6 + 6.2 + 6.7, 7.9 + 6.4 + 6.6 + 7.0 + 7.7 + 7.2 + 5.9 + 6.8 + 6.7 + 8.0 + 6.6 + 7.1 + 7.6 + 6.7) |
| Sentence of 15 words | 6 | 15 | 90 | f16 | Python + FlashInfer | 8845 (8790, 8741, 8757, 8845, 8794, 8839) | 10.2 | 98% | 21.7 GB | 618.9 (6.4 + 6.9 + 6.8 + 7.0 + 7.4 + 6.7 + 7.5 + 8.1 + 8.4 + 7.1 + 5.7 + 8.1 + 5.8 + 7.0 + 6.1, 7.0 + 7.4 + 7.1 + 7.1 + 6.1 + 6.6 + 8.1 + 6.9 + 6.6 + 6.7 + 6.6 + 7.4 + 6.7 + 6.0 + 6.2, 6.3 + 5.6 + 6.2 + 6.4 + 7.4 + 6.2 + 6.8 + 6.2 + 6.9 + 6.5 + 8.1 + 7.1 + 7.0 + 7.7 + 6.5, 8.2 + 5.9 + 6.1 + 7.1 + 8.0 + 7.4 + 7.1 + 6.8 + 6.7 + 6.9 + 7.1 + 7.1 + 6.9 + 7.0 + 6.8, 6.9 + 6.4 + 6.0 + 6.8 + 6.8 + 7.0 + 7.5 + 6.1 + 6.2 + 6.3 + 7.8 + 6.7 + 5.9 + 7.0 + 7.6, 7.4 + 6.2 + 7.2 + 6.7 + 7.8 + 6.4 + 6.7 + 7.6 + 6.7 + 7.8 + 6.8 + 7.0 + 6.3 + 7.9 + 6.1) |
| Sentence of 15 words | 6 | 16 | 96 | f16 | Python + FlashInfer | 9442 (9442, 9243, 9300, 9426, 9442, 9425) | 10.2 | 98% | 22.3 GB | 659.9 (6.5 + 6.8 + 6.8 + 7.1 + 7.3 + 6.7 + 7.5 + 8.1 + 8.4 + 7.0 + 5.7 + 8.3 + 5.8 + 7.1 + 6.1 + 6.1, 6.9 + 6.9 + 7.4 + 6.3 + 6.6 + 8.1 + 6.5 + 6.4 + 6.7 + 6.6 + 7.4 + 6.2 + 6.0 + 6.7 + 6.3 + 5.9, 6.1 + 6.0 + 7.4 + 5.8 + 7.1 + 6.1 + 7.3 + 6.8 + 8.0 + 7.4 + 7.0 + 7.5 + 6.7 + 7.9 + 6.0 + 6.1, 7.6 + 8.0 + 7.7 + 6.7 + 6.0 + 6.2 + 7.0 + 6.8 + 7.2 + 6.9 + 7.1 + 6.8 + 7.0 + 6.0 + 6.0 + 6.8, 6.8 + 7.4 + 7.3 + 6.2 + 6.5 + 6.3 + 7.9 + 6.4 + 6.2 + 6.7 + 8.0 + 7.7 + 6.2 + 6.9 + 6.8 + 8.0, 6.2 + 6.3 + 7.7 + 6.7 + 8.1 + 6.4 + 6.9 + 6.4 + 8.1 + 6.6 + 7.1 + 6.4 + 7.6 + 7.0 + 6.9 + 6.7) |
| Sentence of 15 words | 6 | 17 | 102 | f16 | Python + FlashInfer | failed: instance 4 exited (rc 1): CUDA out of memory | – | – | – | – |
| Sentence of 15 words | 1 | 1 | 1 | bf16 | Python + FlashInfer | failed: bf16 not supported: upstream `apply_flashinfer` plans attention as f16 only | – | – | – | – |
| Sentence of 15 words | 1 | 1 | 1 | f16 | Python + FlashInfer + CUDA graphs | 239 | 4.2 | 56% | 8.7 GB | 6.2 (6.2) |
| Sentence of 15 words | 1 | 2 | 2 | f16 | Python + FlashInfer + CUDA graphs | 388 | 5.2 | 78% | 8.8 GB | 13.5 (6.4 + 7.0) |
| Sentence of 15 words | 1 | 3 | 3 | f16 | Python + FlashInfer + CUDA graphs | 574 | 5.2 | 84% | 8.9 GB | 19.9 (6.5 + 6.6 + 6.8) |
| Sentence of 15 words | 1 | 4 | 4 | f16 | Python + FlashInfer + CUDA graphs | 760 | 5.3 | 84% | 9.1 GB | 27.2 (6.6 + 6.8 + 6.8 + 7.0) |
| Sentence of 15 words | 1 | 5 | 5 | f16 | Python + FlashInfer + CUDA graphs | 925 | 5.4 | 85% | 9.2 GB | 34.4 (6.6 + 6.6 + 6.8 + 7.1 + 7.3) |
| Sentence of 15 words | 1 | 6 | 6 | f16 | Python + FlashInfer + CUDA graphs | 1107 | 5.4 | 84% | 9.5 GB | 40.5 (6.4 + 6.7 + 6.5 + 7.1 + 7.4 + 6.4) |
| Sentence of 15 words | 1 | 7 | 7 | f16 | Python + FlashInfer + CUDA graphs | 1311 | 5.3 | 88% | 9.9 GB | 48.5 (6.4 + 6.6 + 6.8 + 7.1 + 7.4 + 6.7 + 7.5) |
| Sentence of 15 words | 1 | 8 | 8 | f16 | Python + FlashInfer + CUDA graphs | 1479 | 5.4 | 85% | 10.2 GB | 56.1 (6.5 + 6.5 + 6.5 + 7.1 + 7.4 + 6.7 + 7.3 + 8.1) |
| Sentence of 15 words | 1 | 9 | 9 | f16 | Python + FlashInfer + CUDA graphs | 1970 | 4.6 | 85% | 10.9 GB | 65.1 (6.8 + 6.7 + 6.8 + 6.9 + 7.7 + 6.7 + 7.1 + 8.0 + 8.4) |
| Sentence of 15 words | 2 | 1 | 2 | f16 | Python + FlashInfer + CUDA graphs | 415 (410, 415) | 4.8 | 87% | 10.9 GB | 12.8 (6.2, 6.5) |
| Sentence of 15 words | 2 | 2 | 4 | f16 | Python + FlashInfer + CUDA graphs | 753 (753, 732) | 5.3 | 91% | 11.2 GB | 27.4 (6.4 + 7.0, 6.8 + 7.1) |
| Sentence of 15 words | 2 | 3 | 6 | f16 | Python + FlashInfer + CUDA graphs | 1127 (1090, 1127) | 5.3 | 95% | 11.4 GB | 40.7 (6.5 + 6.6 + 6.8, 6.8 + 7.7 + 6.3) |
| Sentence of 15 words | 2 | 4 | 8 | f16 | Python + FlashInfer + CUDA graphs | 1499 (1482, 1499) | 5.3 | 93% | 11.7 GB | 56.5 (6.6 + 6.8 + 6.8 + 7.0, 7.4 + 6.7 + 7.2 + 8.1) |
| Sentence of 15 words | 2 | 5 | 10 | f16 | Python + FlashInfer + CUDA graphs | 2023 (1803, 2023) | 4.9 | 91% | 12.2 GB | 71.9 (6.6 + 6.6 + 6.8 + 7.1 + 7.3, 6.4 + 7.5 + 7.7 + 8.8 + 7.1) |
| Sentence of 15 words | 2 | 6 | 12 | f16 | Python + FlashInfer + CUDA graphs | 2402 (2189, 2402) | 5.0 | 93% | 12.9 GB | 84.6 (6.4 + 6.7 + 6.5 + 7.1 + 7.4 + 6.4, 7.2 + 8.1 + 8.3 + 6.5 + 6.2 + 7.8) |
| Sentence of 15 words | 2 | 7 | 14 | f16 | Python + FlashInfer + CUDA graphs | 2811 (2613, 2811) | 5.0 | 93% | 13.7 GB | 96.9 (6.4 + 6.6 + 6.8 + 7.1 + 7.4 + 6.7 + 7.5, 7.6 + 8.1 + 6.6 + 5.8 + 8.1 + 5.4 + 6.7) |
| Sentence of 15 words | 3 | 1 | 3 | f16 | Python + FlashInfer + CUDA graphs | 625 (577, 567, 625) | 4.8 | 92% | 15.0 GB | 19.0 (6.2, 6.5, 6.2) |
| Sentence of 15 words | 3 | 2 | 6 | f16 | Python + FlashInfer + CUDA graphs | 1071 (1068, 1071, 1039) | 5.6 | 95% | 15.5 GB | 41.7 (6.4 + 7.0, 6.8 + 7.1, 7.6 + 6.7) |
| Sentence of 15 words | 3 | 3 | 9 | f16 | Python + FlashInfer + CUDA graphs | 1752 (1599, 1630, 1752) | 5.1 | 94% | 15.7 GB | 64.2 (6.5 + 6.6 + 6.8, 6.8 + 7.7 + 6.3, 7.2 + 8.1 + 8.3) |
| Sentence of 15 words | 3 | 4 | 12 | f16 | Python + FlashInfer + CUDA graphs | 2315 (2198, 2187, 2315) | 5.2 | 95% | 16.3 GB | 85.5 (6.6 + 6.8 + 6.8 + 7.0, 7.4 + 6.7 + 7.2 + 8.1, 8.6 + 6.7 + 5.7 + 7.9) |
| Sentence of 15 words | 3 | 5 | 15 | f16 | Python + FlashInfer + CUDA graphs | 3048 (2690, 3043, 3048) | 4.9 | 95% | 17.1 GB | 104.6 (6.6 + 6.6 + 6.8 + 7.1 + 7.3, 6.4 + 7.5 + 7.7 + 8.8 + 7.1, 5.8 + 7.7 + 5.8 + 7.1 + 6.3) |
| Sentence of 15 words | 4 | 1 | 4 | f16 | Python + FlashInfer + CUDA graphs | 772 (772, 769, 769, 769) | 5.2 | 91% | 19.9 GB | 25.9 (6.2, 6.5, 6.2, 6.9) |
| Sentence of 15 words | 4 | 2 | 8 | f16 | Python + FlashInfer + CUDA graphs | 1430 (1382, 1406, 1430, 1404) | 5.6 | 94% | 20.5 GB | 56.8 (6.4 + 7.0, 6.8 + 7.1, 7.6 + 6.7, 7.0 + 8.1) |
| Sentence of 15 words | 4 | 3 | 12 | f16 | Python + FlashInfer + CUDA graphs | 2372 (2181, 2170, 2372, 2369) | 5.1 | 96% | 20.8 GB | 85.1 (6.5 + 6.6 + 6.8, 6.8 + 7.7 + 6.3, 7.2 + 8.1 + 8.3, 6.7 + 6.2 + 8.0) |
| Sentence of 15 words | 4 | 4 | 16 | f16 | Python + FlashInfer + CUDA graphs | 2997 (2861, 2807, 2997, 2851) | 5.3 | 96% | 21.4 GB | 110.3 (6.6 + 6.8 + 6.8 + 7.0, 7.4 + 6.7 + 7.2 + 8.1, 8.6 + 6.7 + 5.7 + 7.9, 5.5 + 7.0 + 6.0 + 6.3) |
| Sentence of 15 words | 4 | 5 | 20 | f16 | Python + FlashInfer + CUDA graphs | 3884 (3579, 3884, 3871, 3529) | 5.1 | 96% | 22.4 GB | 139.6 (6.6 + 6.6 + 6.8 + 7.1 + 7.3, 6.4 + 7.5 + 7.7 + 8.8 + 7.1, 5.8 + 7.7 + 5.8 + 7.1 + 6.3, 6.8 + 7.4 + 7.2 + 7.5 + 6.2) |
| Sentence of 15 words | 1 | 1 | 1 | bf16 | Python + FlashInfer + CUDA graphs | 238 | 4.2 | 68% | 9.1 GB | 6.5 (6.5) |
| Sentence of 15 words | 1 | 2 | 2 | bf16 | Python + FlashInfer + CUDA graphs | 389 | 5.1 | 76% | 9.2 GB | 13.1 (6.5 + 6.6) |
| Sentence of 15 words | 1 | 3 | 3 | bf16 | Python + FlashInfer + CUDA graphs | 570 | 5.3 | 84% | 9.3 GB | 20.2 (6.4 + 6.9 + 6.8) |
| Sentence of 15 words | 1 | 4 | 4 | bf16 | Python + FlashInfer + CUDA graphs | 753 | 5.3 | 81% | 9.5 GB | 27.5 (6.6 + 7.1 + 6.8 + 7.0) |
| Sentence of 15 words | 1 | 5 | 5 | bf16 | Python + FlashInfer + CUDA graphs | 932 | 5.4 | 81% | 9.7 GB | 34.5 (6.3 + 7.1 + 6.6 + 7.1 + 7.3) |
| Sentence of 15 words | 1 | 6 | 6 | bf16 | Python + FlashInfer + CUDA graphs | 1104 | 5.4 | 83% | 10.0 GB | 41.6 (6.8 + 6.9 + 6.8 + 7.1 + 7.4 + 6.4) |
| Sentence of 15 words | 1 | 7 | 7 | bf16 | Python + FlashInfer + CUDA graphs | 1281 | 5.5 | 87% | 10.3 GB | 47.9 (6.5 + 6.6 + 6.6 + 7.1 + 7.4 + 6.4 + 7.4) |
| Sentence of 15 words | 1 | 8 | 8 | bf16 | Python + FlashInfer + CUDA graphs | 1471 | 5.4 | 84% | 10.7 GB | 56.0 (6.6 + 6.6 + 6.5 + 7.1 + 7.4 + 6.7 + 7.3 + 7.8) |
| Sentence of 15 words | 1 | 9 | 9 | bf16 | Python + FlashInfer + CUDA graphs | 1956 | 4.6 | 88% | 11.3 GB | 65.1 (6.8 + 7.1 + 6.8 + 6.9 + 7.7 + 6.4 + 7.0 + 8.1 + 8.3) |
| Sentence of 15 words | 1 | 10 | 10 | bf16 | Python + FlashInfer + CUDA graphs | 2188 | 4.6 | 82% | 11.8 GB | 70.1 (6.5 + 6.7 + 6.5 + 7.1 + 7.6 + 6.3 + 7.1 + 7.5 + 8.0 + 6.8) |
| Sentence of 15 words | 2 | 1 | 2 | bf16 | Python + FlashInfer + CUDA graphs | 407 (394, 407) | 4.9 | 87% | 10.9 GB | 13.1 (6.5, 6.6) |
| Sentence of 15 words | 2 | 2 | 4 | bf16 | Python + FlashInfer + CUDA graphs | 760 (734, 760) | 5.3 | 90% | 11.2 GB | 27.0 (6.5 + 6.6, 6.7 + 7.1) |
| Sentence of 15 words | 2 | 3 | 6 | bf16 | Python + FlashInfer + CUDA graphs | 1137 (1137, 1108) | 5.3 | 92% | 11.4 GB | 41.6 (6.4 + 6.9 + 6.8, 6.8 + 8.0 + 6.7) |
| Sentence of 15 words | 2 | 4 | 8 | bf16 | Python + FlashInfer + CUDA graphs | 1480 (1480, 1452) | 5.4 | 93% | 11.7 GB | 56.6 (6.6 + 7.1 + 6.8 + 7.0, 7.6 + 6.3 + 7.2 + 8.1) |
| Sentence of 15 words | 2 | 5 | 10 | bf16 | Python + FlashInfer + CUDA graphs | 2031 (1842, 2031) | 4.9 | 92% | 12.2 GB | 72.2 (6.3 + 7.1 + 6.6 + 7.1 + 7.3, 6.4 + 7.5 + 7.9 + 8.8 + 7.1) |
| Sentence of 15 words | 2 | 6 | 12 | bf16 | Python + FlashInfer + CUDA graphs | 2418 (2224, 2418) | 5.0 | 92% | 12.9 GB | 85.4 (6.8 + 6.9 + 6.8 + 7.1 + 7.4 + 6.4, 7.2 + 7.7 + 8.8 + 6.5 + 5.9 + 7.8) |
| Sentence of 15 words | 2 | 7 | 14 | bf16 | Python + FlashInfer + CUDA graphs | 2804 (2556, 2804) | 5.0 | 92% | 13.6 GB | 96.2 (6.5 + 6.6 + 6.6 + 7.1 + 7.4 + 6.4 + 7.4, 7.6 + 8.1 + 6.5 + 5.8 + 8.1 + 5.4 + 6.7) |
| Sentence of 15 words | 3 | 1 | 3 | bf16 | Python + FlashInfer + CUDA graphs | 578 (554, 574, 578) | 5.2 | 92% | 15.0 GB | 19.6 (6.5, 6.6, 6.5) |
| Sentence of 15 words | 3 | 2 | 6 | bf16 | Python + FlashInfer + CUDA graphs | 1073 (1025, 1059, 1073) | 5.6 | 93% | 15.5 GB | 41.2 (6.5 + 6.6, 6.7 + 7.1, 7.6 + 6.7) |
| Sentence of 15 words | 3 | 3 | 9 | bf16 | Python + FlashInfer + CUDA graphs | 1761 (1652, 1644, 1761) | 5.1 | 96% | 15.7 GB | 65.2 (6.4 + 6.9 + 6.8, 6.8 + 8.0 + 6.7, 7.2 + 8.1 + 8.3) |
| Sentence of 15 words | 3 | 4 | 12 | bf16 | Python + FlashInfer + CUDA graphs | 2324 (2156, 2210, 2324) | 5.2 | 94% | 16.3 GB | 86.4 (6.6 + 7.1 + 6.8 + 7.0, 7.6 + 6.3 + 7.2 + 8.1, 8.6 + 7.1 + 6.1 + 8.0) |
| Sentence of 15 words | 3 | 5 | 15 | bf16 | Python + FlashInfer + CUDA graphs | 3136 (2679, 3054, 3136) | 4.8 | 96% | 17.1 GB | 105.1 (6.3 + 7.1 + 6.6 + 7.1 + 7.3, 6.4 + 7.5 + 7.9 + 8.8 + 7.1, 5.8 + 7.9 + 5.8 + 7.1 + 6.3) |
| Sentence of 15 words | 1 | 1 | 1 | f16 | C++ (`perf_mode=flash_attention`) | 143 | 7.0 | 76% | 2.3 GB | 6.8 (6.8) |
| Sentence of 15 words | 1 | 2 | 2 | f16 | C++ (`perf_mode=flash_attention`) | 289 | 6.9 | 83% | 2.4 GB | 13.2 (6.8 + 6.4) |
| Sentence of 15 words | 1 | 3 | 3 | f16 | C++ (`perf_mode=flash_attention`) | 405 | 7.4 | 88% | 2.4 GB | 20.7 (6.8 + 7.0 + 6.8) |
| Sentence of 15 words | 1 | 4 | 4 | f16 | C++ (`perf_mode=flash_attention`) | 572 | 7.0 | 96% | 2.4 GB | 27.2 (6.8 + 6.4 + 6.8 + 7.1) |
| Sentence of 15 words | 1 | 5 | 5 | f16 | C++ (`perf_mode=flash_attention`) | 778 | 6.4 | 90% | 2.5 GB | 35.7 (6.8 + 7.0 + 6.8 + 7.1 + 7.9) |
| Sentence of 15 words | 1 | 6 | 6 | f16 | C++ (`perf_mode=flash_attention`) | 948 | 6.3 | 91% | 2.5 GB | 41.9 (6.8 + 7.0 + 6.8 + 7.1 + 7.9 + 6.2) |
| Sentence of 15 words | 2 | 1 | 2 | f16 | C++ (`perf_mode=flash_attention`) | 290 (289, 290) | 6.9 | 91% | 4.7 GB | 13.9 (6.8, 7.1) |
| Sentence of 15 words | 2 | 2 | 4 | f16 | C++ (`perf_mode=flash_attention`) | 578 (570, 578) | 6.9 | 95% | 4.7 GB | 27.2 (6.8 + 6.4, 6.8 + 7.1) |
| Sentence of 15 words | 2 | 3 | 6 | f16 | C++ (`perf_mode=flash_attention`) | 844 (844, 838) | 7.1 | 95% | 4.8 GB | 41.9 (6.8 + 7.0 + 6.8, 7.1 + 7.9 + 6.2) |
| Sentence of 15 words | 2 | 4 | 8 | f16 | C++ (`perf_mode=flash_attention`) | 1218 (1202, 1218) | 6.6 | 95% | 4.9 GB | 57.3 (6.8 + 6.4 + 6.8 + 7.1, 7.9 + 6.6 + 7.5 + 8.1) |
| Sentence of 15 words | 2 | 5 | 10 | f16 | C++ (`perf_mode=flash_attention`) | 1634 (1623, 1634) | 6.1 | 95% | 4.9 GB | 73.3 (6.8 + 7.0 + 6.8 + 7.1 + 7.9, 6.2 + 7.5 + 8.1 + 8.7 + 7.1) |
| Sentence of 15 words | 2 | 6 | 12 | f16 | C++ (`perf_mode=flash_attention`) | 2035 (2021, 2035) | 5.9 | 95% | 5.0 GB | 87.6 (6.8 + 7.0 + 6.8 + 7.1 + 7.9 + 6.2, 7.5 + 8.1 + 8.7 + 7.1 + 6.2 + 8.1) |
| Sentence of 15 words | 1 | 1 | 1 | bf16 | C++ (`perf_mode=flash_attention`) | 157 | 6.4 | 88% | 2.3 GB | 6.8 (6.8) |
| Sentence of 15 words | 1 | 2 | 2 | bf16 | C++ (`perf_mode=flash_attention`) | 310 | 6.4 | 82% | 2.4 GB | 13.5 (6.8 + 6.7) |
| Sentence of 15 words | 1 | 3 | 3 | bf16 | C++ (`perf_mode=flash_attention`) | 460 | 6.5 | 89% | 2.4 GB | 20.3 (6.8 + 6.7 + 6.8) |
| Sentence of 15 words | 1 | 4 | 4 | bf16 | C++ (`perf_mode=flash_attention`) | 620 | 6.5 | 91% | 2.4 GB | 27.2 (6.8 + 6.7 + 6.6 + 7.1) |
| Sentence of 15 words | 1 | 5 | 5 | bf16 | C++ (`perf_mode=flash_attention`) | 823 | 6.1 | 90% | 2.5 GB | 35.4 (6.8 + 6.7 + 6.8 + 7.1 + 8.0) |
| Sentence of 15 words | 1 | 6 | 6 | bf16 | C++ (`perf_mode=flash_attention`) | 976 | 6.1 | 91% | 2.5 GB | 41.6 (6.8 + 6.7 + 6.8 + 7.1 + 8.0 + 6.2) |
| Sentence of 15 words | 2 | 1 | 2 | bf16 | C++ (`perf_mode=flash_attention`) | 321 (321, 319) | 6.2 | 95% | 4.7 GB | 13.5 (6.8, 6.7) |
| Sentence of 15 words | 2 | 2 | 4 | bf16 | C++ (`perf_mode=flash_attention`) | 620 (616, 620) | 6.4 | 94% | 4.8 GB | 27.2 (6.8 + 6.7, 6.6 + 7.1) |
| Sentence of 15 words | 2 | 3 | 6 | bf16 | C++ (`perf_mode=flash_attention`) | 947 (929, 947) | 6.3 | 96% | 4.8 GB | 41.6 (6.8 + 6.7 + 6.8, 7.1 + 8.0 + 6.2) |
| Sentence of 15 words | 2 | 4 | 8 | bf16 | C++ (`perf_mode=flash_attention`) | 1330 (1308, 1330) | 6.0 | 95% | 4.9 GB | 57.0 (6.8 + 6.7 + 6.6 + 7.1, 8.0 + 6.2 + 7.5 + 8.1) |
| Sentence of 15 words | 2 | 5 | 10 | bf16 | C++ (`perf_mode=flash_attention`) | 1734 (1723, 1734) | 5.8 | 95% | 5.0 GB | 73.1 (6.8 + 6.7 + 6.8 + 7.1 + 8.0, 6.2 + 7.5 + 8.1 + 8.8 + 7.1) |
| Sentence of 15 words | 1 | 1 | 1 | f16 | C++ (`perf_mode=off`) | 184 | 5.4 | 89% | 2.3 GB | 6.8 (6.8) |
| Sentence of 15 words | 1 | 2 | 2 | f16 | C++ (`perf_mode=off`) | 364 | 5.5 | 91% | 2.4 GB | 13.9 (6.8 + 7.0) |
| Sentence of 15 words | 1 | 3 | 3 | f16 | C++ (`perf_mode=off`) | 520 | 5.8 | 88% | 2.4 GB | 20.2 (6.8 + 6.6 + 6.8) |
| Sentence of 15 words | 1 | 4 | 4 | f16 | C++ (`perf_mode=off`) | 730 | 5.5 | 91% | 2.4 GB | 27.8 (6.8 + 7.0 + 6.8 + 7.1) |
| Sentence of 15 words | 1 | 5 | 5 | f16 | C++ (`perf_mode=off`) | 964 | 5.2 | 91% | 2.5 GB | 35.2 (6.8 + 6.6 + 6.8 + 7.1 + 7.9) |
| Sentence of 15 words | 1 | 6 | 6 | f16 | C++ (`perf_mode=off`) | 1185 | 5.1 | 91% | 2.5 GB | 41.9 (6.8 + 6.6 + 6.8 + 7.1 + 7.9 + 6.7) |
| Sentence of 15 words | 2 | 1 | 2 | f16 | C++ (`perf_mode=off`) | 374 (374, 374) | 5.3 | 96% | 4.6 GB | 13.8 (6.8, 7.1) |
| Sentence of 15 words | 2 | 2 | 4 | f16 | C++ (`perf_mode=off`) | 743 (743, 735) | 5.4 | 97% | 4.7 GB | 27.8 (6.8 + 7.0, 6.8 + 7.1) |
| Sentence of 15 words | 2 | 3 | 6 | f16 | C++ (`perf_mode=off`) | 1107 (1089, 1107) | 5.4 | 96% | 4.8 GB | 41.9 (6.8 + 6.6 + 6.8, 7.1 + 7.9 + 6.7) |
| Sentence of 15 words | 2 | 4 | 8 | f16 | C++ (`perf_mode=off`) | 1566 (1554, 1566) | 5.1 | 95% | 4.9 GB | 58.0 (6.8 + 7.0 + 6.8 + 7.1, 7.9 + 6.7 + 7.5 + 8.1) |
| Sentence of 15 words | 2 | 5 | 10 | f16 | C++ (`perf_mode=off`) | 2126 (2126, 2110) | 4.7 | 95% | 4.9 GB | 73.4 (6.8 + 6.6 + 6.8 + 7.1 + 7.9, 6.7 + 7.5 + 8.1 + 8.8 + 7.1) |
| Sentence of 15 words | 2 | 6 | 12 | f16 | C++ (`perf_mode=off`) | 2637 (2623, 2637) | 4.6 | 96% | 5.0 GB | 87.5 (6.8 + 6.6 + 6.8 + 7.1 + 7.9 + 6.7, 7.5 + 8.1 + 8.8 + 7.1 + 6.2 + 7.9) |
| Sentence of 15 words | 1 | 1 | 1 | bf16 | C++ (`perf_mode=off`) | 197 | 5.1 | 90% | 2.3 GB | 6.8 (6.8) |
| Sentence of 15 words | 1 | 2 | 2 | bf16 | C++ (`perf_mode=off`) | 395 | 5.1 | 88% | 2.4 GB | 13.5 (6.8 + 6.7) |
| Sentence of 15 words | 1 | 3 | 3 | bf16 | C++ (`perf_mode=off`) | 573 | 5.2 | 90% | 2.4 GB | 20.3 (6.8 + 6.7 + 6.8) |
| Sentence of 15 words | 1 | 4 | 4 | bf16 | C++ (`perf_mode=off`) | 774 | 5.2 | 91% | 2.4 GB | 27.8 (6.8 + 7.0 + 6.8 + 7.1) |
| Sentence of 15 words | 1 | 5 | 5 | bf16 | C++ (`perf_mode=off`) | 1016 | 4.9 | 91% | 2.5 GB | 35.4 (6.8 + 6.7 + 6.8 + 7.1 + 8.0) |
| Sentence of 15 words | 1 | 6 | 6 | bf16 | C++ (`perf_mode=off`) | 1215 | 4.9 | 91% | 2.5 GB | 42.0 (6.8 + 7.1 + 6.8 + 7.1 + 8.0 + 6.2) |
| Sentence of 15 words | 2 | 1 | 2 | bf16 | C++ (`perf_mode=off`) | 416 (415, 416) | 4.8 | 95% | 4.7 GB | 13.5 (6.8, 6.7) |
| Sentence of 15 words | 2 | 2 | 4 | bf16 | C++ (`perf_mode=off`) | 789 (789, 786) | 5.1 | 95% | 4.8 GB | 27.4 (6.8 + 6.7, 6.8 + 7.1) |
| Sentence of 15 words | 2 | 3 | 6 | bf16 | C++ (`perf_mode=off`) | 1204 (1186, 1204) | 5.0 | 98% | 4.8 GB | 41.6 (6.8 + 6.7 + 6.8, 7.1 + 8.0 + 6.2) |
| Sentence of 15 words | 2 | 4 | 8 | bf16 | C++ (`perf_mode=off`) | 1672 (1659, 1672) | 4.8 | 96% | 4.9 GB | 57.2 (6.8 + 7.0 + 6.8 + 7.1, 8.0 + 6.2 + 7.5 + 7.7) |
| Sentence of 15 words | 2 | 5 | 10 | bf16 | C++ (`perf_mode=off`) | 2196 (2173, 2196) | 4.6 | 96% | 5.0 GB | 72.5 (6.8 + 6.7 + 6.8 + 7.1 + 8.0, 6.3 + 7.5 + 7.7 + 8.5 + 7.1) |
| Sentence of 15 words | 1 | 1 | 1 | f16 | Python (PyTorch) | 975 | 1.0 | 25% | 5.9 GB | 6.6 (6.6) |
| Sentence of 15 words | 1 | 2 | 2 | f16 | Python (PyTorch) | 1027 | 1.9 | 38% | 5.9 GB | 13.2 (6.6 + 6.6) |
| Sentence of 15 words | 1 | 3 | 3 | f16 | Python (PyTorch) | 1083 | 2.8 | 52% | 5.9 GB | 19.8 (6.8 + 6.7 + 6.3) |
| Sentence of 15 words | 1 | 4 | 4 | f16 | Python (PyTorch) | 1192 | 3.4 | 63% | 5.9 GB | 26.7 (6.6 + 6.7 + 6.3 + 7.1) |
| Sentence of 15 words | 1 | 5 | 5 | f16 | Python (PyTorch) | 1161 | 4.3 | 87% | 5.9 GB | 34.6 (6.6 + 6.7 + 6.4 + 7.1 + 7.9) |
| Sentence of 15 words | 1 | 6 | 6 | f16 | Python (PyTorch) | 1410 | 4.3 | 87% | 5.9 GB | 40.8 (6.8 + 6.7 + 6.4 + 6.8 + 7.8 + 6.3) |
| Sentence of 15 words | 1 | 7 | 7 | f16 | Python (PyTorch) | 1682 | 4.2 | 87% | 5.9 GB | 48.5 (6.8 + 6.7 + 6.4 + 6.6 + 7.8 + 6.7 + 7.5) |
| Sentence of 15 words | 1 | 8 | 8 | f16 | Python (PyTorch) | 1990 | 4.0 | 89% | 5.9 GB | 56.1 (6.6 + 6.7 + 6.4 + 7.1 + 7.5 + 6.7 + 7.2 + 8.1) |
| Sentence of 15 words | 2 | 1 | 2 | f16 | Python (PyTorch) | 1460 (1460, 1118) | 1.4 | 87% | 8.4 GB | 13.1 (6.6, 6.5) |
| Sentence of 15 words | 2 | 2 | 4 | f16 | Python (PyTorch) | 1060 (1060, 1017) | 3.8 | 93% | 8.4 GB | 27.1 (6.6 + 6.6, 6.8 + 7.1) |
| Sentence of 15 words | 2 | 3 | 6 | f16 | Python (PyTorch) | 1368 (1312, 1368) | 4.4 | 94% | 8.4 GB | 41.2 (6.8 + 6.7 + 6.3, 7.1 + 7.6 + 6.7) |
| Sentence of 15 words | 2 | 4 | 8 | f16 | Python (PyTorch) | 1810 (1789, 1810) | 4.4 | 94% | 8.4 GB | 54.9 (6.6 + 6.7 + 6.3 + 7.1, 7.2 + 6.3 + 6.9 + 7.8) |
| Sentence of 15 words | 2 | 5 | 10 | f16 | Python (PyTorch) | 2498 (2374, 2498) | 4.0 | 95% | 8.4 GB | 70.6 (6.6 + 6.7 + 6.4 + 7.1 + 7.9, 6.2 + 7.0 + 7.7 + 8.4 + 6.7) |
| Sentence of 15 words | 2 | 6 | 12 | f16 | Python (PyTorch) | 3051 (2876, 3051) | 3.9 | 93% | 9.2 GB | 84.2 (6.8 + 6.7 + 6.4 + 6.8 + 7.8 + 6.3, 7.5 + 7.7 + 8.0 + 6.7 + 5.8 + 7.7) |
| Sentence of 15 words | 2 | 7 | 14 | f16 | Python (PyTorch) | 3659 (3452, 3659) | 3.8 | 95% | 10.1 GB | 98.9 (6.8 + 6.7 + 6.4 + 6.6 + 7.8 + 6.7 + 7.5, 7.6 + 8.8 + 7.1 + 6.2 + 7.8 + 5.8 + 7.1) |
| Sentence of 15 words | 3 | 1 | 3 | f16 | Python (PyTorch) | 1328 (1328, 1232, 1228) | 2.3 | 76% | 13.0 GB | 19.6 (6.6, 6.5, 6.5) |
| Sentence of 15 words | 3 | 2 | 6 | f16 | Python (PyTorch) | 1361 (1324, 1341, 1361) | 4.4 | 89% | 13.0 GB | 40.9 (6.6 + 6.6, 6.8 + 7.1, 7.5 + 6.3) |
| Sentence of 15 words | 3 | 3 | 9 | f16 | Python (PyTorch) | 2084 (1971, 2011, 2084) | 4.3 | 95% | 13.0 GB | 65.2 (6.8 + 6.7 + 6.3, 7.1 + 7.6 + 6.7, 7.5 + 7.7 + 8.8) |
| Sentence of 15 words | 3 | 4 | 12 | f16 | Python (PyTorch) | 2786 (2650, 2715, 2786) | 4.3 | 98% | 13.0 GB | 84.8 (6.6 + 6.7 + 6.3 + 7.1, 7.2 + 6.3 + 6.9 + 7.8, 8.6 + 7.1 + 6.2 + 8.1) |
| Sentence of 15 words | 3 | 5 | 15 | f16 | Python (PyTorch) | 3723 (3491, 3723, 3703) | 4.0 | 96% | 13.0 GB | 103.2 (6.6 + 6.7 + 6.4 + 7.1 + 7.9, 6.2 + 7.0 + 7.7 + 8.4 + 6.7, 5.7 + 8.3 + 5.8 + 6.7 + 6.2) |
| Sentence of 15 words | 4 | 1 | 4 | f16 | Python (PyTorch) | 1318 (1318, 1164, 1131, 1175) | 3.0 | 78% | 17.7 GB | 26.5 (6.6, 6.5, 6.5, 6.9) |
| Sentence of 15 words | 4 | 2 | 8 | f16 | Python (PyTorch) | 1933 (1709, 1729, 1745, 1933) | 4.1 | 96% | 17.7 GB | 56.1 (6.6 + 6.6, 6.8 + 7.1, 7.5 + 6.3, 7.5 + 7.6) |
| Sentence of 15 words | 4 | 3 | 12 | f16 | Python (PyTorch) | 2793 (2586, 2663, 2793, 2743) | 4.3 | 96% | 17.7 GB | 86.1 (6.8 + 6.7 + 6.3, 7.1 + 7.6 + 6.7, 7.5 + 7.7 + 8.8, 7.1 + 6.0 + 7.8) |
| Sentence of 15 words | 4 | 4 | 16 | f16 | Python (PyTorch) | 3646 (3480, 3562, 3646, 3462) | 4.4 | 95% | 17.7 GB | 110.3 (6.6 + 6.7 + 6.3 + 7.1, 7.2 + 6.3 + 6.9 + 7.8, 8.6 + 7.1 + 6.2 + 8.1, 5.8 + 6.6 + 6.2 + 7.0) |
| Sentence of 15 words | 4 | 5 | 20 | f16 | Python (PyTorch) | 4767 (4544, 4756, 4767, 4498) | 4.2 | 97% | 17.7 GB | 137.9 (6.6 + 6.7 + 6.4 + 7.1 + 7.9, 6.2 + 7.0 + 7.7 + 8.4 + 6.7, 5.7 + 8.3 + 5.8 + 6.7 + 6.2, 7.0 + 7.4 + 7.2 + 6.5 + 6.6) |
| Sentence of 15 words | 4 | 6 | 24 | f16 | Python (PyTorch) | 5860 (5507, 5860, 5494, 5786) | 4.1 | 97% | 19.2 GB | 164.0 (6.8 + 6.7 + 6.4 + 6.8 + 7.8 + 6.3, 7.5 + 7.7 + 8.0 + 6.7 + 5.8 + 7.7, 5.8 + 6.6 + 6.2 + 6.5 + 7.0 + 6.6, 7.2 + 6.3 + 6.5 + 7.7 + 7.0 + 6.4) |
| Sentence of 15 words | 4 | 7 | 28 | f16 | Python (PyTorch) | 7055 (6690, 7055, 6633, 6974) | 4.0 | 97% | 20.9 GB | 196.0 (6.8 + 6.7 + 6.4 + 6.6 + 7.8 + 6.7 + 7.5, 7.6 + 8.8 + 7.1 + 6.2 + 7.8 + 5.8 + 7.1, 6.3 + 6.9 + 7.4 + 7.2 + 7.3 + 5.9 + 6.6, 7.9 + 7.0 + 7.0 + 6.6 + 6.6 + 7.7 + 6.7) |
| Sentence of 15 words | 1 | 1 | 1 | bf16 | Python (PyTorch) | 1199 | 0.8 | 21% | 5.9 GB | 6.4 (6.4) |
| Sentence of 15 words | 1 | 2 | 2 | bf16 | Python (PyTorch) | 971 | 2.1 | 40% | 5.9 GB | 13.0 (6.4 + 6.7) |
| Sentence of 15 words | 1 | 3 | 3 | bf16 | Python (PyTorch) | 1021 | 2.9 | 56% | 5.9 GB | 19.3 (6.4 + 6.6 + 6.3) |
| Sentence of 15 words | 1 | 4 | 4 | bf16 | Python (PyTorch) | 1064 | 3.8 | 70% | 5.9 GB | 27.0 (6.5 + 7.0 + 6.4 + 7.1) |
| Sentence of 15 words | 1 | 5 | 5 | bf16 | Python (PyTorch) | 1147 | 4.4 | 88% | 5.9 GB | 35.0 (6.8 + 6.7 + 6.7 + 7.1 + 7.7) |
| Sentence of 15 words | 1 | 6 | 6 | bf16 | Python (PyTorch) | 1422 | 4.2 | 85% | 5.9 GB | 40.2 (6.6 + 6.7 + 6.4 + 6.7 + 7.5 + 6.3) |
| Sentence of 15 words | 1 | 7 | 7 | bf16 | Python (PyTorch) | 1679 | 4.2 | 87% | 5.9 GB | 48.5 (6.8 + 7.0 + 6.4 + 6.7 + 7.7 + 6.7 + 7.2) |
| Sentence of 15 words | 1 | 8 | 8 | bf16 | Python (PyTorch) | 1995 | 4.0 | 89% | 6.4 GB | 56.2 (6.8 + 6.6 + 6.4 + 7.1 + 7.8 + 6.3 + 7.3 + 7.8) |
| Sentence of 15 words | 2 | 1 | 2 | bf16 | Python (PyTorch) | 1120 (1120, 1100) | 1.8 | 85% | 8.4 GB | 12.9 (6.4, 6.5) |
| Sentence of 15 words | 2 | 2 | 4 | bf16 | Python (PyTorch) | 1636 (1352, 1636) | 2.4 | 78% | 8.4 GB | 26.5 (6.4 + 6.7, 6.5 + 7.0) |
| Sentence of 15 words | 2 | 3 | 6 | bf16 | Python (PyTorch) | 1348 (1291, 1348) | 4.5 | 85% | 8.4 GB | 40.8 (6.4 + 6.6 + 6.3, 7.1 + 7.6 + 6.7) |
| Sentence of 15 words | 2 | 4 | 8 | bf16 | Python (PyTorch) | 1786 (1749, 1786) | 4.5 | 95% | 8.4 GB | 55.4 (6.5 + 7.0 + 6.4 + 7.1, 7.2 + 6.3 + 6.8 + 8.1) |
| Sentence of 15 words | 2 | 5 | 10 | bf16 | Python (PyTorch) | 2471 (2324, 2471) | 4.0 | 93% | 8.4 GB | 71.4 (6.8 + 6.7 + 6.7 + 7.1 + 7.7, 6.2 + 7.0 + 8.1 + 8.4 + 6.7) |
| Sentence of 15 words | 2 | 6 | 12 | bf16 | Python (PyTorch) | 3046 (2895, 3046) | 3.9 | 94% | 9.2 GB | 84.0 (6.6 + 6.7 + 6.4 + 6.7 + 7.5 + 6.3, 7.5 + 7.7 + 8.0 + 6.7 + 6.2 + 7.8) |
| Sentence of 15 words | 2 | 7 | 14 | bf16 | Python (PyTorch) | 3645 (3458, 3645) | 3.8 | 94% | 10.1 GB | 97.8 (6.8 + 7.0 + 6.4 + 6.7 + 7.7 + 6.7 + 7.2, 7.6 + 8.4 + 7.1 + 5.7 + 7.8 + 5.8 + 7.0) |
| Sentence of 15 words | 3 | 1 | 3 | bf16 | Python (PyTorch) | 1441 (1205, 1441, 1200) | 2.1 | 83% | 13.0 GB | 19.7 (6.4, 6.5, 6.8) |
| Sentence of 15 words | 3 | 2 | 6 | bf16 | Python (PyTorch) | 1363 (1337, 1363, 1355) | 4.4 | 94% | 13.0 GB | 40.6 (6.4 + 6.7, 6.5 + 7.0, 7.8 + 6.3) |
| Sentence of 15 words | 3 | 3 | 9 | bf16 | Python (PyTorch) | 2071 (1936, 1999, 2071) | 4.3 | 95% | 13.0 GB | 63.9 (6.4 + 6.6 + 6.3, 7.1 + 7.6 + 6.7, 7.5 + 8.0 + 7.7) |
| Sentence of 15 words | 3 | 4 | 12 | bf16 | Python (PyTorch) | 2777 (2645, 2628, 2777) | 4.3 | 100% | 13.0 GB | 85.2 (6.5 + 7.0 + 6.4 + 7.1, 7.2 + 6.3 + 6.8 + 8.1, 8.8 + 6.7 + 6.2 + 8.1) |
| Sentence of 15 words | 3 | 5 | 15 | bf16 | Python (PyTorch) | 3674 (3470, 3674, 3585) | 4.1 | 96% | 13.0 GB | 103.8 (6.8 + 6.7 + 6.7 + 7.1 + 7.7, 6.2 + 7.0 + 8.1 + 8.4 + 6.7, 5.7 + 8.0 + 5.8 + 6.7 + 6.3) |
| Sentence of 15 words | 4 | 1 | 4 | bf16 | Python (PyTorch) | 1239 (1208, 1151, 1239, 1233) | 3.2 | 92% | 17.7 GB | 26.7 (6.4, 6.5, 6.8, 6.9) |
| Sentence of 15 words | 4 | 2 | 8 | bf16 | Python (PyTorch) | 1772 (1732, 1730, 1761, 1772) | 4.5 | 96% | 17.7 GB | 55.4 (6.4 + 6.7, 6.5 + 7.0, 7.8 + 6.3, 7.2 + 7.7) |
| Sentence of 15 words | 4 | 3 | 12 | bf16 | Python (PyTorch) | 2787 (2538, 2607, 2787, 2745) | 4.3 | 94% | 17.7 GB | 84.7 (6.4 + 6.6 + 6.3, 7.1 + 7.6 + 6.7, 7.5 + 8.0 + 7.7, 7.1 + 6.0 + 7.8) |
| Sentence of 15 words | 4 | 4 | 16 | bf16 | Python (PyTorch) | 3599 (3384, 3523, 3599, 3466) | 4.4 | 98% | 17.7 GB | 110.5 (6.5 + 7.0 + 6.4 + 7.1, 7.2 + 6.3 + 6.8 + 8.1, 8.8 + 6.7 + 6.2 + 8.1, 5.5 + 7.0 + 5.8 + 7.0) |
| Sentence of 15 words | 4 | 5 | 20 | bf16 | Python (PyTorch) | 4756 (4512, 4756, 4690, 4497) | 4.2 | 97% | 17.7 GB | 137.9 (6.8 + 6.7 + 6.7 + 7.1 + 7.7, 6.2 + 7.0 + 8.1 + 8.4 + 6.7, 5.7 + 8.0 + 5.8 + 6.7 + 6.3, 7.0 + 7.3 + 6.8 + 6.8 + 6.1) |
| Sentence of 20 words | 1 | 1 | 1 | f16 | Python + FlashInfer | 539 | 1.9 | 30% | 4.4 GB | 8.4 (8.4) |
| Sentence of 20 words | 1 | 2 | 2 | f16 | Python + FlashInfer | 540 | 3.7 | 41% | 4.4 GB | 16.9 (7.8 + 9.1) |
| Sentence of 20 words | 1 | 3 | 3 | f16 | Python + FlashInfer | 569 | 5.3 | 59% | 4.4 GB | 26.3 (8.3 + 8.6 + 9.4) |
| Sentence of 20 words | 1 | 4 | 4 | f16 | Python + FlashInfer | 659 | 6.1 | 64% | 4.4 GB | 35.2 (8.8 + 8.8 + 9.1 + 8.4) |
| Sentence of 20 words | 1 | 5 | 5 | f16 | Python + FlashInfer | 705 | 7.1 | 71% | 4.4 GB | 43.7 (8.5 + 9.1 + 9.4 + 8.4 + 8.2) |
| Sentence of 20 words | 1 | 6 | 6 | f16 | Python + FlashInfer | 847 | 7.1 | 72% | 4.4 GB | 51.5 (8.7 + 9.1 + 9.4 + 8.4 + 7.9 + 8.0) |
| Sentence of 20 words | 1 | 7 | 7 | f16 | Python + FlashInfer | 955 | 7.3 | 78% | 4.4 GB | 62.0 (8.7 + 9.1 + 9.2 + 8.7 + 8.2 + 8.0 + 10.0) |
| Sentence of 20 words | 1 | 8 | 8 | f16 | Python + FlashInfer | 1089 | 7.3 | 77% | 4.3 GB | 70.5 (8.3 + 9.1 + 9.4 + 8.4 + 8.2 + 7.8 + 10.0 + 9.2) |
| Sentence of 20 words | 1 | 9 | 9 | f16 | Python + FlashInfer | 1222 | 7.4 | 87% | 4.5 GB | 78.7 (8.5 + 8.8 + 9.2 + 8.6 + 7.9 + 7.6 + 10.0 + 9.2 + 8.8) |
| Sentence of 20 words | 1 | 10 | 10 | f16 | Python + FlashInfer | 1347 | 7.4 | 76% | 4.8 GB | 88.4 (8.8 + 9.1 + 9.4 + 8.7 + 8.2 + 8.0 + 10.0 + 9.0 + 8.8 + 8.3) |
| Sentence of 20 words | 1 | 11 | 11 | f16 | Python + FlashInfer | 1514 | 7.3 | 73% | 4.0 GB | 96.8 (8.5 + 9.1 + 9.4 + 8.4 + 8.2 + 8.0 + 10.0 + 8.9 + 8.6 + 8.2 + 9.4) |
| Sentence of 20 words | 1 | 12 | 12 | f16 | Python + FlashInfer | 1737 | 6.9 | 68% | 4.4 GB | 105.9 (8.5 + 8.6 + 9.4 + 8.5 + 8.1 + 8.0 + 10.0 + 9.0 + 8.6 + 8.5 + 9.3 + 9.5) |
| Sentence of 20 words | 1 | 13 | 13 | f16 | Python + FlashInfer | 1836 | 7.1 | 72% | 4.7 GB | 115.1 (8.5 + 9.1 + 9.2 + 8.5 + 8.1 + 8.0 + 9.9 + 9.2 + 8.6 + 8.2 + 9.5 + 9.6 + 8.6) |
| Sentence of 20 words | 2 | 1 | 2 | f16 | Python + FlashInfer | 633 (592, 633) | 3.2 | 92% | 7.6 GB | 17.3 (8.4, 8.9) |
| Sentence of 20 words | 2 | 2 | 4 | f16 | Python + FlashInfer | 679 (642, 679) | 5.9 | 81% | 7.6 GB | 34.3 (7.8 + 9.1, 9.2 + 8.1) |
| Sentence of 20 words | 2 | 3 | 6 | f16 | Python + FlashInfer | 961 (961, 851) | 6.2 | 89% | 7.6 GB | 50.7 (8.3 + 8.6 + 9.4, 8.4 + 8.2 + 7.7) |
| Sentence of 20 words | 2 | 4 | 8 | f16 | Python + FlashInfer | 1073 (1047, 1073) | 7.5 | 89% | 7.6 GB | 70.2 (8.8 + 8.8 + 9.1 + 8.4, 8.2 + 7.6 + 10.0 + 9.1) |
| Sentence of 20 words | 2 | 5 | 10 | f16 | Python + FlashInfer | 1335 (1323, 1335) | 7.5 | 89% | 7.7 GB | 86.9 (8.5 + 9.1 + 9.4 + 8.4 + 8.2, 7.9 + 10.0 + 8.8 + 8.4 + 8.1) |
| Sentence of 20 words | 2 | 6 | 12 | f16 | Python + FlashInfer | 1663 (1607, 1663) | 7.2 | 86% | 8.2 GB | 106.3 (8.7 + 9.1 + 9.4 + 8.4 + 7.9 + 8.0, 9.8 + 8.8 + 8.6 + 8.3 + 9.2 + 10.0) |
| Sentence of 20 words | 2 | 7 | 14 | f16 | Python + FlashInfer | 1841 (1841, 1836) | 7.6 | 88% | 8.4 GB | 124.5 (8.7 + 9.1 + 9.2 + 8.7 + 8.2 + 8.0 + 10.0, 8.9 + 8.3 + 8.5 + 9.2 + 9.5 + 8.9 + 9.1) |
| Sentence of 20 words | 2 | 8 | 16 | f16 | Python + FlashInfer | 2085 (2037, 2085) | 7.7 | 91% | 8.9 GB | 140.0 (8.3 + 9.1 + 9.4 + 8.4 + 8.2 + 7.8 + 10.0 + 9.2, 8.5 + 8.6 + 9.3 + 8.9 + 8.5 + 9.1 + 8.7 + 7.9) |
| Sentence of 20 words | 2 | 9 | 18 | f16 | Python + FlashInfer | 2361 (2317, 2361) | 7.6 | 87% | 9.3 GB | 157.7 (8.5 + 8.8 + 9.2 + 8.6 + 7.9 + 7.6 + 10.0 + 9.2 + 8.8, 8.5 + 9.5 + 8.9 + 8.5 + 9.1 + 9.2 + 8.2 + 7.8 + 9.3) |
| Sentence of 20 words | 2 | 10 | 20 | f16 | Python + FlashInfer | 2573 (2573, 2554) | 7.8 | 88% | 7.9 GB | 177.5 (8.8 + 9.1 + 9.4 + 8.7 + 8.2 + 8.0 + 10.0 + 9.0 + 8.8 + 8.3, 9.1 + 9.7 + 8.9 + 9.1 + 9.2 + 8.2 + 8.0 + 8.9 + 9.2 + 8.8) |
| Sentence of 20 words | 2 | 11 | 22 | f16 | Python + FlashInfer | 2912 (2860, 2912) | 7.6 | 88% | 8.6 GB | 193.1 (8.5 + 9.1 + 9.4 + 8.4 + 8.2 + 8.0 + 10.0 + 8.9 + 8.6 + 8.2 + 9.4, 9.5 + 8.5 + 9.4 + 8.6 + 8.2 + 8.0 + 9.5 + 9.1 + 8.5 + 8.4 + 8.6) |
| Sentence of 20 words | 2 | 12 | 24 | f16 | Python + FlashInfer | 3172 (3168, 3172) | 7.6 | 84% | 9.3 GB | 212.3 (8.5 + 8.6 + 9.4 + 8.5 + 8.1 + 8.0 + 10.0 + 9.0 + 8.6 + 8.5 + 9.3 + 9.5, 8.4 + 9.4 + 8.9 + 7.9 + 7.9 + 9.7 + 9.7 + 8.3 + 8.7 + 9.1 + 9.3 + 9.1) |
| Sentence of 20 words | 2 | 13 | 26 | f16 | Python + FlashInfer | 3378 (3378, 3376) | 7.7 | 84% | 10.0 GB | 230.2 (8.5 + 9.1 + 9.2 + 8.5 + 8.1 + 8.0 + 9.9 + 9.2 + 8.6 + 8.2 + 9.5 + 9.6 + 8.6, 8.9 + 8.3 + 7.7 + 7.9 + 9.8 + 9.4 + 8.8 + 8.5 + 9.1 + 9.5 + 9.5 + 9.5 + 8.1) |
| Sentence of 20 words | 3 | 1 | 3 | f16 | Python + FlashInfer | 804 (804, 753, 630) | 3.7 | 92% | 17.9 GB | 26.5 (8.4, 8.9, 9.2) |
| Sentence of 20 words | 3 | 2 | 6 | f16 | Python + FlashInfer | 800 (779, 800, 776) | 7.5 | 87% | 17.9 GB | 49.9 (7.8 + 9.1, 9.2 + 8.1, 7.7 + 7.9) |
| Sentence of 20 words | 3 | 3 | 9 | f16 | Python + FlashInfer | 1277 (1277, 1183, 1185) | 7.0 | 89% | 17.9 GB | 78.0 (8.3 + 8.6 + 9.4, 8.4 + 8.2 + 7.7, 9.8 + 8.7 + 8.8) |
| Sentence of 20 words | 3 | 4 | 12 | f16 | Python + FlashInfer | 1533 (1512, 1488, 1533) | 7.8 | 92% | 17.9 GB | 105.8 (8.8 + 8.8 + 9.1 + 8.4, 8.2 + 7.6 + 10.0 + 9.1, 8.5 + 8.3 + 9.1 + 9.7) |
| Sentence of 20 words | 3 | 5 | 15 | f16 | Python + FlashInfer | 1931 (1836, 1931, 1918) | 7.8 | 92% | 17.9 GB | 131.7 (8.5 + 9.1 + 9.4 + 8.4 + 8.2, 7.9 + 10.0 + 8.8 + 8.4 + 8.1, 9.3 + 9.6 + 8.7 + 8.9 + 8.3) |
| Sentence of 20 words | 3 | 6 | 18 | f16 | Python + FlashInfer | 2294 (2233, 2294, 2251) | 7.8 | 91% | 17.9 GB | 158.0 (8.7 + 9.1 + 9.4 + 8.4 + 7.9 + 8.0, 9.8 + 8.8 + 8.6 + 8.3 + 9.2 + 10.0, 8.7 + 9.4 + 8.7 + 7.8 + 7.8 + 9.3) |
| Sentence of 20 words | 3 | 7 | 21 | f16 | Python + FlashInfer | 2598 (2567, 2592, 2598) | 8.1 | 93% | 17.9 GB | 185.3 (8.7 + 9.1 + 9.2 + 8.7 + 8.2 + 8.0 + 10.0, 8.9 + 8.3 + 8.5 + 9.2 + 9.5 + 8.9 + 9.1, 8.7 + 7.7 + 7.9 + 9.8 + 9.7 + 8.6 + 8.6) |
| Sentence of 20 words | 3 | 8 | 24 | f16 | Python + FlashInfer | 3053 (2956, 3053, 3029) | 7.9 | 89% | 17.9 GB | 212.1 (8.3 + 9.1 + 9.4 + 8.4 + 8.2 + 7.8 + 10.0 + 9.2, 8.5 + 8.6 + 9.3 + 8.9 + 8.5 + 9.1 + 8.7 + 7.9, 7.8 + 9.8 + 9.7 + 8.8 + 8.6 + 8.7 + 9.5 + 9.1) |
| Sentence of 20 words | 3 | 9 | 27 | f16 | Python + FlashInfer | 3401 (3328, 3401, 3313) | 7.9 | 92% | 17.9 GB | 237.4 (8.5 + 8.8 + 9.2 + 8.6 + 7.9 + 7.6 + 10.0 + 9.2 + 8.8, 8.5 + 9.5 + 8.9 + 8.5 + 9.1 + 9.2 + 8.2 + 7.8 + 9.3, 9.3 + 8.6 + 8.7 + 8.7 + 9.3 + 9.5 + 9.3 + 8.0 + 8.3) |
| Sentence of 20 words | 3 | 10 | 30 | f16 | Python + FlashInfer | 3796 (3664, 3796, 3659) | 7.9 | 89% | 17.9 GB | 263.4 (8.8 + 9.1 + 9.4 + 8.7 + 8.2 + 8.0 + 10.0 + 9.0 + 8.8 + 8.3, 9.1 + 9.7 + 8.9 + 9.1 + 9.2 + 8.2 + 8.0 + 8.9 + 9.2 + 8.8, 8.3 + 8.6 + 9.5 + 9.2 + 9.3 + 8.4 + 8.6 + 7.9 + 8.0 + 8.0) |
| Sentence of 20 words | 4 | 1 | 4 | f16 | Python + FlashInfer | 665 (665, 662, 644, 638) | 6.0 | 90% | 16.6 GB | 35.2 (8.4, 8.9, 9.2, 8.7) |
| Sentence of 20 words | 4 | 2 | 8 | f16 | Python + FlashInfer | 1026 (1014, 1022, 1017, 1026) | 7.8 | 92% | 16.6 GB | 68.6 (7.8 + 9.1, 9.2 + 8.1, 7.7 + 7.9, 9.8 + 8.8) |
| Sentence of 20 words | 4 | 3 | 12 | f16 | Python + FlashInfer | 1501 (1484, 1475, 1501, 1430) | 8.0 | 94% | 16.6 GB | 105.5 (8.3 + 8.6 + 9.4, 8.4 + 8.2 + 7.7, 9.8 + 8.7 + 8.8, 8.3 + 9.5 + 9.7) |
| Sentence of 20 words | 4 | 4 | 16 | f16 | Python + FlashInfer | 2021 (1927, 1903, 2021, 1930) | 7.9 | 93% | 16.6 GB | 141.1 (8.8 + 8.8 + 9.1 + 8.4, 8.2 + 7.6 + 10.0 + 9.1, 8.5 + 8.3 + 9.1 + 9.7, 8.9 + 9.4 + 9.2 + 7.8) |
| Sentence of 20 words | 4 | 5 | 20 | f16 | Python + FlashInfer | 2524 (2437, 2450, 2524, 2468) | 7.9 | 94% | 16.6 GB | 174.7 (8.5 + 9.1 + 9.4 + 8.4 + 8.2, 7.9 + 10.0 + 8.8 + 8.4 + 8.1, 9.3 + 9.6 + 8.7 + 8.9 + 8.3, 7.7 + 7.8 + 9.8 + 9.3 + 8.3) |
| Sentence of 20 words | 4 | 6 | 24 | f16 | Python + FlashInfer | 3010 (2942, 2969, 2880, 3010) | 8.0 | 93% | 16.6 GB | 212.2 (8.7 + 9.1 + 9.4 + 8.4 + 7.9 + 8.0, 9.8 + 8.8 + 8.6 + 8.3 + 9.2 + 10.0, 8.7 + 9.4 + 8.7 + 7.8 + 7.8 + 9.3, 9.6 + 8.8 + 8.4 + 8.8 + 9.0 + 9.5) |
| Sentence of 20 words | 5 | 1 | 5 | f16 | Python + FlashInfer | 813 (763, 777, 813, 773, 752) | 6.2 | 92% | 22.3 GB | 43.0 (8.4, 8.9, 9.2, 8.7, 7.8) |
| Sentence of 20 words | 5 | 2 | 10 | f16 | Python + FlashInfer | 1287 (1251, 1255, 1228, 1287, 1255) | 7.8 | 93% | 22.3 GB | 85.4 (7.8 + 9.1, 9.2 + 8.1, 7.7 + 7.9, 9.8 + 8.8, 8.6 + 8.2) |
| Sentence of 20 words | 5 | 3 | 15 | f16 | Python + FlashInfer | 1887 (1868, 1835, 1883, 1872, 1887) | 7.9 | 99% | 22.3 GB | 132.6 (8.3 + 8.6 + 9.4, 8.4 + 8.2 + 7.7, 9.8 + 8.7 + 8.8, 8.3 + 9.5 + 9.7, 8.5 + 9.4 + 9.2) |
| Sentence of 20 words | 5 | 4 | 20 | f16 | Python + FlashInfer | 2491 (2374, 2366, 2449, 2385, 2491) | 8.0 | 94% | 22.3 GB | 175.9 (8.8 + 8.8 + 9.1 + 8.4, 8.2 + 7.6 + 10.0 + 9.1, 8.5 + 8.3 + 9.1 + 9.7, 8.9 + 9.4 + 9.2 + 7.8, 7.8 + 9.5 + 9.3 + 8.2) |
| Sentence of 20 words | 5 | 5 | 25 | f16 | Python + FlashInfer | 3052 (3032, 3033, 3050, 3037, 3052) | 8.2 | 97% | 22.3 GB | 219.8 (8.5 + 9.1 + 9.4 + 8.4 + 8.2, 7.9 + 10.0 + 8.8 + 8.4 + 8.1, 9.3 + 9.6 + 8.7 + 8.9 + 8.3, 7.7 + 7.8 + 9.8 + 9.3 + 8.3, 8.3 + 8.4 + 9.5 + 9.5 + 9.3) |
| Sentence of 20 words | 5 | 6 | 30 | f16 | Python + FlashInfer | 3644 (3583, 3644, 3581, 3625, 3522) | 8.2 | 95% | 22.3 GB | 262.2 (8.7 + 9.1 + 9.4 + 8.4 + 7.9 + 8.0, 9.8 + 8.8 + 8.6 + 8.3 + 9.2 + 10.0, 8.7 + 9.4 + 8.7 + 7.8 + 7.8 + 9.3, 9.6 + 8.8 + 8.4 + 8.8 + 9.0 + 9.5, 9.3 + 8.4 + 8.3 + 7.9 + 7.8 + 8.3) |
| Sentence of 20 words | 5 | 7 | 35 | f16 | Python + FlashInfer | 4210 (4150, 4204, 4146, 4162, 4210) | 8.3 | 96% | 22.3 GB | 311.5 (8.7 + 9.1 + 9.2 + 8.7 + 8.2 + 8.0 + 10.0, 8.9 + 8.3 + 8.5 + 9.2 + 9.5 + 8.9 + 9.1, 8.7 + 7.7 + 7.9 + 9.8 + 9.7 + 8.6 + 8.6, 8.7 + 9.5 + 9.5 + 9.6 + 8.4 + 8.6 + 7.7, 7.9 + 8.3 + 9.6 + 8.7 + 10.2 + 9.3 + 10.2) |
| Sentence of 20 words | 5 | 8 | 40 | f16 | Python + FlashInfer | 4830 (4739, 4798, 4813, 4720, 4830) | 8.3 | 96% | 22.3 GB | 353.7 (8.3 + 9.1 + 9.4 + 8.4 + 8.2 + 7.8 + 10.0 + 9.2, 8.5 + 8.6 + 9.3 + 8.9 + 8.5 + 9.1 + 8.7 + 7.9, 7.8 + 9.8 + 9.7 + 8.8 + 8.6 + 8.7 + 9.5 + 9.1, 9.2 + 8.1 + 8.3 + 7.5 + 8.0 + 8.2 + 9.1 + 8.4, 10.1 + 9.4 + 10.4 + 9.0 + 8.0 + 10.9 + 9.5 + 7.7) |
| Sentence of 20 words | 5 | 9 | 45 | f16 | Python + FlashInfer | 5398 (5365, 5367, 5338, 5398, 5358) | 8.3 | 97% | 22.3 GB | 399.1 (8.5 + 8.8 + 9.2 + 8.6 + 7.9 + 7.6 + 10.0 + 9.2 + 8.8, 8.5 + 9.5 + 8.9 + 8.5 + 9.1 + 9.2 + 8.2 + 7.8 + 9.3, 9.3 + 8.6 + 8.7 + 8.7 + 9.3 + 9.5 + 9.3 + 8.0 + 8.3, 7.5 + 7.7 + 8.0 + 9.6 + 8.4 + 10.2 + 9.4 + 10.6 + 8.6, 7.7 + 11.2 + 9.5 + 7.9 + 8.8 + 8.8 + 8.7 + 9.9 + 8.9) |
| Sentence of 20 words | 5 | 10 | 50 | f16 | Python + FlashInfer | 6036 (5898, 5991, 5847, 6036, 5949) | 8.3 | 96% | 22.3 GB | 444.3 (8.8 + 9.1 + 9.4 + 8.7 + 8.2 + 8.0 + 10.0 + 9.0 + 8.8 + 8.3, 9.1 + 9.7 + 8.9 + 9.1 + 9.2 + 8.2 + 8.0 + 8.9 + 9.2 + 8.8, 8.3 + 8.6 + 9.5 + 9.2 + 9.3 + 8.4 + 8.6 + 7.9 + 8.0 + 8.0, 8.8 + 8.7 + 10.0 + 9.6 + 10.3 + 8.4 + 7.6 + 11.3 + 9.2 + 7.9, 8.8 + 8.8 + 8.7 + 9.5 + 9.3 + 9.1 + 8.6 + 8.5 + 9.6 + 8.0) |
| Sentence of 20 words | 5 | 11 | 55 | f16 | Python + FlashInfer | 6627 (6524, 6589, 6547, 6627, 6625) | 8.3 | 97% | 23.5 GB | 489.6 (8.5 + 9.1 + 9.4 + 8.4 + 8.2 + 8.0 + 10.0 + 8.9 + 8.6 + 8.2 + 9.4, 9.5 + 8.5 + 9.4 + 8.6 + 8.2 + 8.0 + 9.5 + 9.1 + 8.5 + 8.4 + 8.6, 9.3 + 9.1 + 9.3 + 8.4 + 8.2 + 7.9 + 7.8 + 8.0 + 9.0 + 8.7 + 10.2, 9.3 + 10.6 + 8.6 + 8.0 + 11.3 + 9.2 + 7.7 + 9.1 + 8.4 + 8.7 + 9.9, 9.3 + 9.3 + 9.0 + 8.9 + 9.2 + 8.4 + 8.7 + 8.7 + 10.2 + 9.0 + 8.9) |
| Sentence of 20 words | 5 | 12 | 60 | f16 | Python + FlashInfer | 7192 (7153, 7114, 7049, 7129, 7192) | 8.3 | 97% | 19.8 GB | 531.1 (8.5 + 8.6 + 9.4 + 8.5 + 8.1 + 8.0 + 10.0 + 9.0 + 8.6 + 8.5 + 9.3 + 9.5, 8.4 + 9.4 + 8.9 + 7.9 + 7.9 + 9.7 + 9.7 + 8.3 + 8.7 + 9.1 + 9.3 + 9.1, 9.3 + 8.1 + 8.4 + 7.4 + 8.0 + 7.9 + 8.9 + 8.7 + 10.2 + 9.6 + 10.2 + 8.9, 7.7 + 10.9 + 9.5 + 7.9 + 8.9 + 8.8 + 8.7 + 9.4 + 9.1 + 9.0 + 8.8 + 8.3, 9.2 + 8.3 + 8.2 + 8.7 + 9.8 + 9.2 + 9.2 + 8.0 + 8.0 + 10.0 + 8.5 + 8.9) |
| Sentence of 20 words | 5 | 13 | 65 | f16 | Python + FlashInfer | 7742 (7672, 7732, 7742, 7647, 7654) | 8.4 | 97% | 21.6 GB | 572.7 (8.5 + 9.1 + 9.2 + 8.5 + 8.1 + 8.0 + 9.9 + 9.2 + 8.6 + 8.2 + 9.5 + 9.6 + 8.6, 8.9 + 8.3 + 7.7 + 7.9 + 9.8 + 9.4 + 8.8 + 8.5 + 9.1 + 9.5 + 9.5 + 9.5 + 8.1, 8.1 + 7.9 + 7.7 + 8.1 + 9.2 + 8.4 + 9.9 + 9.0 + 10.6 + 8.6 + 8.0 + 10.9 + 9.0, 7.7 + 8.9 + 8.7 + 8.7 + 9.7 + 9.2 + 9.1 + 8.7 + 8.9 + 9.5 + 7.8 + 8.5 + 8.7, 10.2 + 9.2 + 8.7 + 8.0 + 7.9 + 10.0 + 8.5 + 8.7 + 8.9 + 7.6 + 8.6 + 8.3 + 8.6) |
| Sentence of 20 words | 5 | 14 | 70 | f16 | Python + FlashInfer | 8434 (8340, 8360, 8434, 8397, 8309) | 8.3 | 97% | 23.5 GB | 619.7 (8.4 + 8.8 + 9.4 + 8.7 + 7.9 + 8.0 + 9.9 + 9.0 + 8.8 + 8.3 + 9.2 + 10.0 + 8.6 + 9.2, 8.7 + 7.6 + 7.8 + 9.8 + 9.5 + 8.6 + 8.6 + 9.1 + 9.1 + 9.5 + 9.3 + 8.0 + 8.6 + 7.7, 7.6 + 7.7 + 9.2 + 8.7 + 10.2 + 9.6 + 10.4 + 9.0 + 7.7 + 10.8 + 9.2 + 7.9 + 8.7 + 8.8, 8.7 + 9.5 + 9.2 + 9.4 + 9.0 + 8.9 + 9.6 + 7.9 + 8.7 + 8.7 + 9.8 + 9.2 + 9.1 + 8.0, 7.7 + 10.0 + 8.6 + 9.1 + 8.5 + 7.6 + 8.7 + 8.3 + 8.4 + 9.0 + 8.7 + 9.0 + 9.1 + 9.8) |
| Sentence of 20 words | 5 | 15 | 75 | f16 | Python + FlashInfer | 9051 (8980, 8844, 9051, 8907, 8998) | 8.3 | 97% | 20.7 GB | 665.0 (8.6 + 8.8 + 9.2 + 8.6 + 8.1 + 8.0 + 10.0 + 9.2 + 8.8 + 8.4 + 9.2 + 10.0 + 8.5 + 9.0 + 9.2, 7.9 + 8.2 + 9.5 + 9.3 + 8.3 + 8.5 + 9.0 + 9.1 + 9.5 + 9.6 + 8.2 + 8.5 + 7.6 + 8.0 + 8.3, 9.2 + 8.7 + 10.2 + 9.3 + 10.4 + 8.5 + 7.6 + 10.8 + 9.4 + 7.9 + 9.1 + 8.7 + 8.4 + 9.4 + 9.4, 9.0 + 9.0 + 8.9 + 9.2 + 8.2 + 8.3 + 9.0 + 10.2 + 9.2 + 8.7 + 7.7 + 7.9 + 9.8 + 8.5 + 8.6, 8.4 + 7.6 + 8.7 + 8.3 + 8.8 + 9.8 + 8.5 + 9.2 + 8.4 + 10.1 + 8.1 + 7.5 + 10.4 + 9.4 + 9.4) |
| Sentence of 20 words | 5 | 16 | 80 | f16 | Python + FlashInfer | 9701 (9509, 9470, 9701, 9453, 9579) | 8.2 | 97% | 22.9 GB | 708.6 (8.0 + 9.1 + 9.2 + 8.4 + 8.2 + 8.0 + 10.0 + 9.0 + 8.8 + 8.2 + 9.5 + 10.0 + 8.6 + 8.9 + 8.5 + 7.9, 7.8 + 9.8 + 9.3 + 8.3 + 8.2 + 9.1 + 9.5 + 9.5 + 9.4 + 8.2 + 8.6 + 7.5 + 8.0 + 8.3 + 9.5 + 8.4, 9.9 + 9.6 + 10.6 + 8.8 + 8.0 + 10.8 + 9.5 + 7.7 + 9.1 + 8.8 + 8.3 + 9.6 + 9.4 + 8.9 + 8.8 + 8.8, 9.3 + 8.1 + 8.7 + 9.0 + 9.8 + 9.2 + 8.8 + 7.7 + 8.0 + 9.7 + 8.6 + 9.1 + 8.7 + 7.6 + 8.7 + 8.3, 8.6 + 9.7 + 8.6 + 9.2 + 9.1 + 10.1 + 8.3 + 7.5 + 10.4 + 9.8 + 9.7 + 8.2 + 8.1 + 8.4 + 9.0 + 8.1) |
| Sentence of 20 words | 6 | 1 | 6 | f16 | Python + FlashInfer | 942 (931, 935, 942, 923, 921, 913) | 6.4 | 92% | 20.0 GB | 50.5 (8.4, 8.9, 9.2, 8.7, 7.8, 7.5) |
| Sentence of 20 words | 6 | 2 | 12 | f16 | Python + FlashInfer | 1549 (1479, 1491, 1478, 1547, 1475, 1549) | 7.7 | 100% | 20.0 GB | 104.3 (7.8 + 9.1, 9.2 + 8.1, 7.7 + 7.9, 9.8 + 8.8, 8.6 + 8.2, 9.1 + 9.8) |
| Sentence of 20 words | 6 | 3 | 18 | f16 | Python + FlashInfer | 2220 (2220, 2118, 2219, 2213, 2176, 2167) | 8.1 | 94% | 20.4 GB | 158.4 (8.3 + 8.6 + 9.4, 8.4 + 8.2 + 7.7, 9.8 + 8.7 + 8.8, 8.3 + 9.5 + 9.7, 8.5 + 9.4 + 9.2, 7.9 + 8.2 + 9.8) |
| Sentence of 20 words | 6 | 4 | 24 | f16 | Python + FlashInfer | 2954 (2828, 2854, 2954, 2854, 2934, 2937) | 8.1 | 96% | 21.1 GB | 212.2 (8.8 + 8.8 + 9.1 + 8.4, 8.2 + 7.6 + 10.0 + 9.1, 8.5 + 8.3 + 9.1 + 9.7, 8.9 + 9.4 + 9.2 + 7.8, 7.8 + 9.5 + 9.3 + 8.2, 8.6 + 8.7 + 9.5 + 9.5) |
| Sentence of 20 words | 6 | 5 | 30 | f16 | Python + FlashInfer | 3633 (3611, 3593, 3633, 3627, 3630, 3532) | 8.3 | 97% | 22.2 GB | 259.6 (8.5 + 9.1 + 9.4 + 8.4 + 8.2, 7.9 + 10.0 + 8.8 + 8.4 + 8.1, 9.3 + 9.6 + 8.7 + 8.9 + 8.3, 7.7 + 7.8 + 9.8 + 9.3 + 8.3, 8.3 + 8.4 + 9.5 + 9.5 + 9.3, 8.2 + 8.2 + 7.4 + 7.7 + 8.3) |
| Sentence of 20 words | 6 | 6 | 36 | f16 | Python + FlashInfer | 4374 (4216, 4374, 4272, 4315, 4192, 4371) | 8.2 | 96% | 23.4 GB | 318.4 (8.7 + 9.1 + 9.4 + 8.4 + 7.9 + 8.0, 9.8 + 8.8 + 8.6 + 8.3 + 9.2 + 10.0, 8.7 + 9.4 + 8.7 + 7.8 + 7.8 + 9.3, 9.6 + 8.8 + 8.4 + 8.8 + 9.0 + 9.5, 9.3 + 8.4 + 8.3 + 7.9 + 7.8 + 8.3, 9.0 + 8.4 + 9.9 + 9.3 + 10.6 + 9.0) |
| Sentence of 20 words | 6 | 7 | 42 | f16 | Python + FlashInfer | 5065 (4996, 5020, 4967, 4941, 5065, 5001) | 8.3 | 97% | 20.5 GB | 372.5 (8.7 + 9.1 + 9.2 + 8.7 + 8.2 + 8.0 + 10.0, 8.9 + 8.3 + 8.5 + 9.2 + 9.5 + 8.9 + 9.1, 8.7 + 7.7 + 7.9 + 9.8 + 9.7 + 8.6 + 8.6, 8.7 + 9.5 + 9.5 + 9.6 + 8.4 + 8.6 + 7.7, 7.9 + 8.3 + 9.6 + 8.7 + 10.2 + 9.3 + 10.2, 8.5 + 8.0 + 10.8 + 9.1 + 7.6 + 8.7 + 8.2) |
| Sentence of 20 words | 6 | 8 | 48 | f16 | Python + FlashInfer | 5786 (5599, 5748, 5715, 5569, 5786, 5739) | 8.3 | 97% | 22.5 GB | 424.8 (8.3 + 9.1 + 9.4 + 8.4 + 8.2 + 7.8 + 10.0 + 9.2, 8.5 + 8.6 + 9.3 + 8.9 + 8.5 + 9.1 + 8.7 + 7.9, 7.8 + 9.8 + 9.7 + 8.8 + 8.6 + 8.7 + 9.5 + 9.1, 9.2 + 8.1 + 8.3 + 7.5 + 8.0 + 8.2 + 9.1 + 8.4, 10.1 + 9.4 + 10.4 + 9.0 + 8.0 + 10.9 + 9.5 + 7.7, 8.8 + 8.8 + 8.3 + 9.3 + 9.4 + 9.4 + 8.5 + 8.6) |
| Sentence of 20 words | 6 | 9 | 54 | f16 | Python + FlashInfer | 6430 (6350, 6416, 6416, 6403, 6430, 6427) | 8.4 | 98% | 21.6 GB | 478.9 (8.5 + 8.8 + 9.2 + 8.6 + 7.9 + 7.6 + 10.0 + 9.2 + 8.8, 8.5 + 9.5 + 8.9 + 8.5 + 9.1 + 9.2 + 8.2 + 7.8 + 9.3, 9.3 + 8.6 + 8.7 + 8.7 + 9.3 + 9.5 + 9.3 + 8.0 + 8.3, 7.5 + 7.7 + 8.0 + 9.6 + 8.4 + 10.2 + 9.4 + 10.6 + 8.6, 7.7 + 11.2 + 9.5 + 7.9 + 8.8 + 8.8 + 8.7 + 9.9 + 8.9, 9.0 + 9.0 + 8.9 + 9.2 + 7.8 + 8.7 + 8.7 + 9.9 + 8.7) |
| Sentence of 20 words | 6 | 10 | 60 | f16 | Python + FlashInfer | 7123 (7008, 7122, 6974, 7123, 7106, 7042) | 8.4 | 98% | 23.3 GB | 532.4 (8.8 + 9.1 + 9.4 + 8.7 + 8.2 + 8.0 + 10.0 + 9.0 + 8.8 + 8.3, 9.1 + 9.7 + 8.9 + 9.1 + 9.2 + 8.2 + 8.0 + 8.9 + 9.2 + 8.8, 8.3 + 8.6 + 9.5 + 9.2 + 9.3 + 8.4 + 8.6 + 7.9 + 8.0 + 8.0, 8.8 + 8.7 + 10.0 + 9.6 + 10.3 + 8.4 + 7.6 + 11.3 + 9.2 + 7.9, 8.8 + 8.8 + 8.7 + 9.5 + 9.3 + 9.1 + 8.6 + 8.5 + 9.6 + 8.0, 8.2 + 9.0 + 9.7 + 9.2 + 8.7 + 7.9 + 8.2 + 9.7 + 8.7 + 8.7) |
| Sentence of 20 words | 6 | 11 | 66 | f16 | Python + FlashInfer | 7858 (7803, 7825, 7789, 7844, 7858, 7725) | 8.4 | 98% | 22.9 GB | 584.3 (8.5 + 9.1 + 9.4 + 8.4 + 8.2 + 8.0 + 10.0 + 8.9 + 8.6 + 8.2 + 9.4, 9.5 + 8.5 + 9.4 + 8.6 + 8.2 + 8.0 + 9.5 + 9.1 + 8.5 + 8.4 + 8.6, 9.3 + 9.1 + 9.3 + 8.4 + 8.2 + 7.9 + 7.8 + 8.0 + 9.0 + 8.7 + 10.2, 9.3 + 10.6 + 8.6 + 8.0 + 11.3 + 9.2 + 7.7 + 9.1 + 8.4 + 8.7 + 9.9, 9.3 + 9.3 + 9.0 + 8.9 + 9.2 + 8.4 + 8.7 + 8.7 + 10.2 + 9.0 + 8.9, 7.7 + 8.2 + 10.0 + 8.5 + 9.1 + 8.9 + 7.6 + 8.7 + 8.0 + 8.5 + 9.5) |
| Sentence of 20 words | 6 | 12 | 72 | f16 | Python + FlashInfer | 8492 (8415, 8492, 8437, 8491, 8415, 8389) | 8.5 | 98% | 22.1 GB | 634.2 (8.5 + 8.6 + 9.4 + 8.5 + 8.1 + 8.0 + 10.0 + 9.0 + 8.6 + 8.5 + 9.3 + 9.5, 8.4 + 9.4 + 8.9 + 7.9 + 7.9 + 9.7 + 9.7 + 8.3 + 8.7 + 9.1 + 9.3 + 9.1, 9.3 + 8.1 + 8.4 + 7.4 + 8.0 + 7.9 + 8.9 + 8.7 + 10.2 + 9.6 + 10.2 + 8.9, 7.7 + 10.9 + 9.5 + 7.9 + 8.9 + 8.8 + 8.7 + 9.4 + 9.1 + 9.0 + 8.8 + 8.3, 9.2 + 8.3 + 8.2 + 8.7 + 9.8 + 9.2 + 9.2 + 8.0 + 8.0 + 10.0 + 8.5 + 8.9, 8.4 + 7.6 + 8.7 + 8.1 + 8.8 + 9.4 + 8.7 + 9.2 + 8.8 + 9.8 + 8.1 + 7.6) |
| Sentence of 20 words | 6 | 13 | 78 | f16 | Python + FlashInfer | failed: instance 1 exited (rc 1): CUDA out of memory | – | – | – | – |
| Sentence of 20 words | 1 | 1 | 1 | bf16 | Python + FlashInfer | failed: bf16 not supported: upstream `apply_flashinfer` plans attention as f16 only | – | – | – | – |
| Sentence of 20 words | 1 | 1 | 1 | f16 | Python + FlashInfer + CUDA graphs | 277 | 3.6 | 75% | 10.7 GB | 8.4 (8.4) |
| Sentence of 20 words | 1 | 2 | 2 | f16 | Python + FlashInfer + CUDA graphs | 471 | 4.2 | 83% | 10.8 GB | 17.5 (8.6 + 8.9) |
| Sentence of 20 words | 1 | 3 | 3 | f16 | Python + FlashInfer + CUDA graphs | 693 | 4.3 | 85% | 10.8 GB | 26.8 (8.3 + 9.1 + 9.3) |
| Sentence of 20 words | 1 | 4 | 4 | f16 | Python + FlashInfer + CUDA graphs | 924 | 4.3 | 80% | 11.0 GB | 35.2 (8.4 + 8.7 + 9.4 + 8.7) |
| Sentence of 20 words | 1 | 5 | 5 | f16 | Python + FlashInfer + CUDA graphs | 1114 | 4.5 | 86% | 11.3 GB | 43.1 (8.3 + 8.8 + 9.3 + 8.7 + 7.9) |
| Sentence of 20 words | 1 | 6 | 6 | f16 | Python + FlashInfer + CUDA graphs | 1357 | 4.4 | 86% | 11.7 GB | 50.9 (8.4 + 8.9 + 9.3 + 8.4 + 7.9 + 7.9) |
| Sentence of 20 words | 1 | 7 | 7 | f16 | Python + FlashInfer + CUDA graphs | 1627 | 4.3 | 81% | 12.1 GB | 60.5 (8.4 + 8.9 + 9.4 + 8.2 + 8.2 + 7.6 + 9.8) |
| Sentence of 20 words | 1 | 8 | 8 | f16 | Python + FlashInfer + CUDA graphs | 1776 | 4.5 | 86% | 12.6 GB | 70.1 (8.5 + 8.8 + 9.3 + 8.6 + 8.2 + 7.5 + 9.9 + 9.2) |
| Sentence of 20 words | 1 | 9 | 9 | f16 | Python + FlashInfer + CUDA graphs | 1998 | 4.5 | 83% | 13.1 GB | 78.0 (8.5 + 8.9 + 9.4 + 8.4 + 7.9 + 7.6 + 9.7 + 9.2 + 8.4) |
| Sentence of 20 words | 1 | 10 | 10 | f16 | Python + FlashInfer + CUDA graphs | 2222 | 4.5 | 86% | 13.2 GB | 86.6 (8.4 + 9.1 + 9.3 + 8.3 + 8.0 + 8.0 + 9.9 + 9.0 + 8.4 + 8.1) |
| Sentence of 20 words | 1 | 11 | 11 | f16 | Python + FlashInfer + CUDA graphs | 2449 | 4.5 | 86% | 13.7 GB | 96.7 (8.8 + 9.1 + 9.3 + 8.4 + 8.2 + 7.8 + 9.8 + 9.2 + 8.4 + 8.3 + 9.2) |
| Sentence of 20 words | 1 | 12 | 12 | f16 | Python + FlashInfer + CUDA graphs | 2694 | 4.5 | 87% | 14.3 GB | 105.8 (8.4 + 9.1 + 9.4 + 8.7 + 8.0 + 7.9 + 9.8 + 9.2 + 8.4 + 8.4 + 9.0 + 9.5) |
| Sentence of 20 words | 2 | 1 | 2 | f16 | Python + FlashInfer + CUDA graphs | 519 (519, 505) | 3.9 | 91% | 13.7 GB | 17.3 (8.4, 8.9) |
| Sentence of 20 words | 2 | 2 | 4 | f16 | Python + FlashInfer + CUDA graphs | 915 (906, 915) | 4.4 | 92% | 13.9 GB | 35.0 (8.6 + 8.9, 9.2 + 8.3) |
| Sentence of 20 words | 2 | 3 | 6 | f16 | Python + FlashInfer + CUDA graphs | 1345 (1345, 1339) | 4.5 | 92% | 14.0 GB | 51.0 (8.3 + 9.1 + 9.3, 8.4 + 8.2 + 7.6) |
| Sentence of 20 words | 2 | 4 | 8 | f16 | Python + FlashInfer + CUDA graphs | 1813 (1766, 1813) | 4.4 | 93% | 14.5 GB | 69.8 (8.4 + 8.7 + 9.4 + 8.7, 8.0 + 7.9 + 10.0 + 8.8) |
| Sentence of 20 words | 2 | 5 | 10 | f16 | Python + FlashInfer + CUDA graphs | 2198 (2198, 2178) | 4.6 | 94% | 15.1 GB | 87.2 (8.3 + 8.8 + 9.3 + 8.7 + 7.9, 7.8 + 10.0 + 9.2 + 8.8 + 8.3) |
| Sentence of 20 words | 2 | 6 | 12 | f16 | Python + FlashInfer + CUDA graphs | 2651 (2644, 2651) | 4.5 | 95% | 15.8 GB | 105.6 (8.4 + 8.9 + 9.3 + 8.4 + 7.9 + 7.9, 9.7 + 9.0 + 8.6 + 8.3 + 9.2 + 10.0) |
| Sentence of 20 words | 2 | 7 | 14 | f16 | Python + FlashInfer + CUDA graphs | 3102 (3102, 3101) | 4.5 | 95% | 16.4 GB | 123.2 (8.4 + 8.9 + 9.4 + 8.2 + 8.2 + 7.6 + 9.8, 8.9 + 8.6 + 8.1 + 9.1 + 10.0 + 8.7 + 9.2) |
| Sentence of 20 words | 2 | 8 | 16 | f16 | Python + FlashInfer + CUDA graphs | 3550 (3550, 3541) | 4.5 | 94% | 16.8 GB | 140.5 (8.5 + 8.8 + 9.3 + 8.6 + 8.2 + 7.5 + 9.9 + 9.2, 8.5 + 8.6 + 9.1 + 10.0 + 8.9 + 9.0 + 8.6 + 7.7) |
| Sentence of 20 words | 3 | 1 | 3 | f16 | Python + FlashInfer + CUDA graphs | 710 (710, 709, 687) | 4.2 | 94% | 17.5 GB | 26.5 (8.4, 8.9, 9.2) |
| Sentence of 20 words | 3 | 2 | 6 | f16 | Python + FlashInfer + CUDA graphs | 1319 (1319, 1314, 1292) | 4.5 | 94% | 17.8 GB | 51.0 (8.6 + 8.9, 9.2 + 8.3, 8.0 + 8.0) |
| Sentence of 20 words | 3 | 3 | 9 | f16 | Python + FlashInfer + CUDA graphs | 1987 (1953, 1987, 1953) | 4.5 | 96% | 18.0 GB | 78.7 (8.3 + 9.1 + 9.3, 8.4 + 8.2 + 7.6, 9.8 + 9.0 + 8.8) |
| Sentence of 20 words | 3 | 4 | 12 | f16 | Python + FlashInfer + CUDA graphs | 2627 (2608, 2590, 2627) | 4.6 | 96% | 18.7 GB | 105.7 (8.4 + 8.7 + 9.4 + 8.7, 8.0 + 7.9 + 10.0 + 8.8, 8.6 + 8.6 + 9.1 + 9.6) |
| Sentence of 20 words | 3 | 5 | 15 | f16 | Python + FlashInfer + CUDA graphs | 3226 (3226, 3224, 3176) | 4.7 | 96% | 19.5 GB | 133.2 (8.3 + 8.8 + 9.3 + 8.7 + 7.9, 7.8 + 10.0 + 9.2 + 8.8 + 8.3, 9.3 + 9.7 + 8.4 + 9.4 + 9.2) |
| Sentence of 20 words | 3 | 6 | 18 | f16 | Python + FlashInfer + CUDA graphs | 3886 (3865, 3863, 3886) | 4.6 | 95% | 20.1 GB | 158.0 (8.4 + 8.9 + 9.3 + 8.4 + 7.9 + 7.9, 9.7 + 9.0 + 8.6 + 8.3 + 9.2 + 10.0, 8.8 + 9.4 + 8.6 + 7.8 + 7.9 + 9.8) |
| Sentence of 20 words | 3 | 7 | 21 | f16 | Python + FlashInfer + CUDA graphs | 4541 (4524, 4541, 4527) | 4.6 | 96% | 21.3 GB | 183.9 (8.4 + 8.9 + 9.4 + 8.2 + 8.2 + 7.6 + 9.8, 8.9 + 8.6 + 8.1 + 9.1 + 10.0 + 8.7 + 9.2, 8.7 + 7.9 + 8.2 + 9.5 + 9.7 + 8.3 + 8.5) |
| Sentence of 20 words | 3 | 8 | 24 | f16 | Python + FlashInfer + CUDA graphs | 5246 (5171, 5157, 5246) | 4.6 | 96% | 22.7 GB | 211.8 (8.5 + 8.8 + 9.3 + 8.6 + 8.2 + 7.5 + 9.9 + 9.2, 8.5 + 8.6 + 9.1 + 10.0 + 8.9 + 9.0 + 8.6 + 7.7, 7.9 + 9.3 + 9.2 + 8.8 + 8.6 + 8.6 + 9.3 + 9.5) |
| Sentence of 20 words | 4 | 1 | 4 | f16 | Python + FlashInfer + CUDA graphs | 945 (935, 944, 927, 945) | 4.2 | 93% | 22.8 GB | 35.1 (8.4, 8.9, 9.2, 8.6) |
| Sentence of 20 words | 4 | 2 | 8 | f16 | Python + FlashInfer + CUDA graphs | 1734 (1724, 1734, 1726, 1730) | 4.6 | 96% | 23.1 GB | 70.1 (8.6 + 8.9, 9.2 + 8.3, 8.0 + 8.0, 9.9 + 9.2) |
| Sentence of 20 words | 4 | 3 | 12 | f16 | Python + FlashInfer + CUDA graphs | 2620 (2595, 2583, 2620, 2593) | 4.6 | 95% | 23.2 GB | 106.1 (8.3 + 9.1 + 9.3, 8.4 + 8.2 + 7.6, 9.8 + 9.0 + 8.8, 8.2 + 9.5 + 9.6) |
| Sentence of 20 words | 4 | 4 | 16 | f16 | Python + FlashInfer + CUDA graphs | failed: instance 0 exited (rc 1): CUDA out of memory | – | – | – | – |
| Sentence of 20 words | 1 | 1 | 1 | bf16 | Python + FlashInfer + CUDA graphs | 265 | 3.8 | 66% | 11.5 GB | 8.3 (8.3) |
| Sentence of 20 words | 1 | 2 | 2 | bf16 | Python + FlashInfer + CUDA graphs | 490 | 4.1 | 83% | 11.6 GB | 17.4 (8.5 + 8.9) |
| Sentence of 20 words | 1 | 3 | 3 | bf16 | Python + FlashInfer + CUDA graphs | 689 | 4.4 | 85% | 11.6 GB | 26.7 (8.5 + 8.9 + 9.3) |
| Sentence of 20 words | 1 | 4 | 4 | bf16 | Python + FlashInfer + CUDA graphs | 916 | 4.4 | 86% | 11.9 GB | 35.1 (8.5 + 8.7 + 9.4 + 8.5) |
| Sentence of 20 words | 1 | 5 | 5 | bf16 | Python + FlashInfer + CUDA graphs | 1136 | 4.4 | 86% | 12.2 GB | 43.4 (8.3 + 8.8 + 9.3 + 8.7 + 8.2) |
| Sentence of 20 words | 1 | 6 | 6 | bf16 | Python + FlashInfer + CUDA graphs | 1328 | 4.5 | 86% | 12.6 GB | 51.3 (8.8 + 8.9 + 9.4 + 8.7 + 7.6 + 7.9) |
| Sentence of 20 words | 1 | 7 | 7 | bf16 | Python + FlashInfer + CUDA graphs | 1562 | 4.5 | 84% | 13.0 GB | 61.0 (8.5 + 8.9 + 9.4 + 8.4 + 8.2 + 7.7 + 9.9) |
| Sentence of 20 words | 1 | 8 | 8 | bf16 | Python + FlashInfer + CUDA graphs | 1772 | 4.5 | 85% | 13.4 GB | 70.4 (8.5 + 9.1 + 9.4 + 8.7 + 8.2 + 7.4 + 9.9 + 9.2) |
| Sentence of 20 words | 1 | 9 | 9 | bf16 | Python + FlashInfer + CUDA graphs | 1988 | 4.5 | 87% | 13.9 GB | 77.9 (8.3 + 8.9 + 9.4 + 8.2 + 8.2 + 7.4 + 9.8 + 9.2 + 8.4) |
| Sentence of 20 words | 1 | 10 | 10 | bf16 | Python + FlashInfer + CUDA graphs | 2218 | 4.5 | 86% | 14.3 GB | 87.0 (8.4 + 9.1 + 9.4 + 8.4 + 8.0 + 7.6 + 9.8 + 9.2 + 8.7 + 8.3) |
| Sentence of 20 words | 1 | 11 | 11 | bf16 | Python + FlashInfer + CUDA graphs | 2457 | 4.5 | 85% | 14.1 GB | 96.0 (8.8 + 9.1 + 9.4 + 8.4 + 8.0 + 7.4 + 9.8 + 9.1 + 8.4 + 8.3 + 9.2) |
| Sentence of 20 words | 1 | 12 | 12 | bf16 | Python + FlashInfer + CUDA graphs | 2773 | 4.3 | 84% | 14.8 GB | 105.4 (8.4 + 9.1 + 9.3 + 8.7 + 8.0 + 7.6 + 9.9 + 9.2 + 8.4 + 8.3 + 8.9 + 9.5) |
| Sentence of 20 words | 2 | 1 | 2 | bf16 | Python + FlashInfer + CUDA graphs | 499 (499, 497) | 4.0 | 90% | 13.7 GB | 17.2 (8.3, 8.9) |
| Sentence of 20 words | 2 | 2 | 4 | bf16 | Python + FlashInfer + CUDA graphs | 901 (901, 900) | 4.4 | 93% | 13.9 GB | 35.1 (8.5 + 8.9, 9.4 + 8.3) |
| Sentence of 20 words | 2 | 3 | 6 | bf16 | Python + FlashInfer + CUDA graphs | 1345 (1345, 1343) | 4.5 | 94% | 14.0 GB | 51.0 (8.5 + 8.9 + 9.3, 8.4 + 7.9 + 8.0) |
| Sentence of 20 words | 2 | 4 | 8 | bf16 | Python + FlashInfer + CUDA graphs | 1801 (1801, 1772) | 4.4 | 94% | 14.5 GB | 69.8 (8.5 + 8.7 + 9.4 + 8.5, 8.0 + 7.9 + 10.0 + 8.8) |
| Sentence of 20 words | 2 | 5 | 10 | bf16 | Python + FlashInfer + CUDA graphs | 2196 (2190, 2196) | 4.6 | 92% | 15.1 GB | 87.8 (8.3 + 8.8 + 9.3 + 8.7 + 8.2, 7.7 + 10.0 + 9.2 + 8.8 + 8.6) |
| Sentence of 20 words | 2 | 6 | 12 | bf16 | Python + FlashInfer + CUDA graphs | 2647 (2638, 2647) | 4.5 | 94% | 15.8 GB | 106.6 (8.8 + 8.9 + 9.4 + 8.7 + 7.6 + 7.9, 9.6 + 9.0 + 8.6 + 8.6 + 9.5 + 10.0) |
| Sentence of 20 words | 2 | 7 | 14 | bf16 | Python + FlashInfer + CUDA graphs | 3133 (3059, 3133) | 4.5 | 93% | 16.4 GB | 123.7 (8.5 + 8.9 + 9.4 + 8.4 + 8.2 + 7.7 + 9.9, 8.9 + 8.6 + 8.2 + 9.1 + 9.7 + 8.9 + 9.4) |
| Sentence of 20 words | 2 | 8 | 16 | bf16 | Python + FlashInfer + CUDA graphs | 3618 (3580, 3618) | 4.4 | 91% | 16.8 GB | 141.3 (8.5 + 9.1 + 9.4 + 8.7 + 8.2 + 7.4 + 9.9 + 9.2, 8.7 + 8.2 + 9.1 + 10.0 + 8.9 + 9.1 + 8.6 + 8.2) |
| Sentence of 20 words | 3 | 1 | 3 | bf16 | Python + FlashInfer + CUDA graphs | 735 (730, 712, 735) | 4.1 | 91% | 17.5 GB | 26.4 (8.3, 8.9, 9.2) |
| Sentence of 20 words | 3 | 2 | 6 | bf16 | Python + FlashInfer + CUDA graphs | 1326 (1326, 1295, 1316) | 4.5 | 94% | 17.8 GB | 51.2 (8.5 + 8.9, 9.4 + 8.3, 8.1 + 8.0) |
| Sentence of 20 words | 3 | 3 | 9 | bf16 | Python + FlashInfer + CUDA graphs | 1979 (1963, 1979, 1966) | 4.5 | 95% | 18.0 GB | 78.7 (8.5 + 8.9 + 9.3, 8.4 + 7.9 + 8.0, 9.8 + 9.0 + 8.8) |
| Sentence of 20 words | 3 | 4 | 12 | bf16 | Python + FlashInfer + CUDA graphs | 2605 (2605, 2578, 2594) | 4.6 | 95% | 18.7 GB | 105.8 (8.5 + 8.7 + 9.4 + 8.5, 8.0 + 7.9 + 10.0 + 8.8, 8.6 + 8.6 + 9.5 + 9.4) |
| Sentence of 20 words | 3 | 5 | 15 | bf16 | Python + FlashInfer + CUDA graphs | 3274 (3177, 3274, 3242) | 4.6 | 96% | 19.5 GB | 134.2 (8.3 + 8.8 + 9.3 + 8.7 + 8.2, 7.7 + 10.0 + 9.2 + 8.8 + 8.6, 9.3 + 10.0 + 8.5 + 9.4 + 9.2) |
| Sentence of 20 words | 3 | 6 | 18 | bf16 | Python + FlashInfer + CUDA graphs | 3865 (3862, 3865, 3851) | 4.7 | 96% | 20.1 GB | 158.6 (8.8 + 8.9 + 9.4 + 8.7 + 7.6 + 7.9, 9.6 + 9.0 + 8.6 + 8.6 + 9.5 + 10.0, 8.4 + 9.4 + 8.6 + 7.9 + 7.9 + 9.8) |
| Sentence of 20 words | 3 | 7 | 21 | bf16 | Python + FlashInfer + CUDA graphs | 4567 (4515, 4567, 4551) | 4.6 | 96% | 21.3 GB | 184.9 (8.5 + 8.9 + 9.4 + 8.4 + 8.2 + 7.7 + 9.9, 8.9 + 8.6 + 8.2 + 9.1 + 9.7 + 8.9 + 9.4, 8.7 + 7.8 + 8.2 + 9.7 + 9.7 + 8.7 + 8.5) |
| Sentence of 20 words | 3 | 8 | 24 | bf16 | Python + FlashInfer + CUDA graphs | 5211 (5153, 5211, 5145) | 4.6 | 95% | 22.7 GB | 211.9 (8.5 + 9.1 + 9.4 + 8.7 + 8.2 + 7.4 + 9.9 + 9.2, 8.7 + 8.2 + 9.1 + 10.0 + 8.9 + 9.1 + 8.6 + 8.2, 7.8 + 9.3 + 9.2 + 8.8 + 8.7 + 8.6 + 9.0 + 9.1) |
| Sentence of 20 words | 3 | 9 | 27 | bf16 | Python + FlashInfer + CUDA graphs | failed: instance 2 exited (rc 1): CUDA out of memory | – | – | – | – |
| Sentence of 20 words | 1 | 1 | 1 | f16 | C++ (`perf_mode=flash_attention`) | 182 | 5.5 | 87% | 2.4 GB | 8.8 (8.8) |
| Sentence of 20 words | 1 | 2 | 2 | f16 | C++ (`perf_mode=flash_attention`) | 345 | 5.8 | 80% | 2.4 GB | 18.0 (8.8 + 9.1) |
| Sentence of 20 words | 1 | 3 | 3 | f16 | C++ (`perf_mode=flash_attention`) | 535 | 5.6 | 89% | 2.5 GB | 27.4 (8.8 + 9.1 + 9.4) |
| Sentence of 20 words | 1 | 4 | 4 | f16 | C++ (`perf_mode=flash_attention`) | 707 | 5.7 | 90% | 2.5 GB | 36.0 (8.8 + 9.1 + 9.4 + 8.7) |
| Sentence of 20 words | 1 | 5 | 5 | f16 | C++ (`perf_mode=flash_attention`) | 934 | 5.4 | 91% | 2.5 GB | 44.3 (8.8 + 9.1 + 9.4 + 8.7 + 8.2) |
| Sentence of 20 words | 2 | 1 | 2 | f16 | C++ (`perf_mode=flash_attention`) | 377 (375, 377) | 5.3 | 93% | 4.8 GB | 18.0 (8.8, 9.1) |
| Sentence of 20 words | 2 | 2 | 4 | f16 | C++ (`perf_mode=flash_attention`) | 700 (700, 664) | 5.7 | 94% | 4.8 GB | 36.0 (8.8 + 9.1, 9.4 + 8.7) |
| Sentence of 20 words | 2 | 3 | 6 | f16 | C++ (`perf_mode=flash_attention`) | 1053 (1053, 1029) | 5.7 | 95% | 4.9 GB | 52.3 (8.8 + 9.1 + 9.4, 8.7 + 8.2 + 8.0) |
| Sentence of 20 words | 2 | 4 | 8 | f16 | C++ (`perf_mode=flash_attention`) | 1463 (1453, 1463) | 5.5 | 96% | 5.0 GB | 71.5 (8.8 + 9.1 + 9.4 + 8.7, 8.2 + 8.0 + 10.0 + 9.2) |
| Sentence of 20 words | 2 | 5 | 10 | f16 | C++ (`perf_mode=flash_attention`) | 1936 (1936, 1928) | 5.2 | 96% | 5.1 GB | 88.8 (8.8 + 9.1 + 9.4 + 8.7 + 8.2, 8.0 + 10.0 + 9.2 + 8.8 + 8.4) |
| Sentence of 20 words | 1 | 1 | 1 | bf16 | C++ (`perf_mode=flash_attention`) | 189 | 5.3 | 84% | 2.4 GB | 8.8 (8.8) |
| Sentence of 20 words | 1 | 2 | 2 | bf16 | C++ (`perf_mode=flash_attention`) | 395 | 5.1 | 86% | 2.4 GB | 18.0 (8.8 + 9.1) |
| Sentence of 20 words | 1 | 3 | 3 | bf16 | C++ (`perf_mode=flash_attention`) | 590 | 5.1 | 90% | 2.5 GB | 27.4 (8.8 + 9.1 + 9.4) |
| Sentence of 20 words | 1 | 4 | 4 | bf16 | C++ (`perf_mode=flash_attention`) | 750 | 5.3 | 90% | 2.5 GB | 36.0 (8.8 + 9.1 + 9.4 + 8.7) |
| Sentence of 20 words | 1 | 5 | 5 | bf16 | C++ (`perf_mode=flash_attention`) | 962 | 5.2 | 91% | 2.5 GB | 44.0 (8.6 + 9.1 + 9.4 + 8.7 + 8.2) |
| Sentence of 20 words | 1 | 6 | 6 | bf16 | C++ (`perf_mode=flash_attention`) | 1181 | 5.1 | 87% | 2.6 GB | 52.0 (8.5 + 9.1 + 9.4 + 8.7 + 8.2 + 8.0) |
| Sentence of 20 words | 1 | 7 | 7 | bf16 | C++ (`perf_mode=flash_attention`) | 1394 | 5.0 | 89% | 2.6 GB | 62.3 (8.8 + 9.1 + 9.4 + 8.7 + 8.2 + 8.0 + 10.0) |
| Sentence of 20 words | 2 | 1 | 2 | bf16 | C++ (`perf_mode=flash_attention`) | 390 (387, 390) | 5.1 | 93% | 4.8 GB | 17.9 (8.8, 9.1) |
| Sentence of 20 words | 2 | 2 | 4 | bf16 | C++ (`perf_mode=flash_attention`) | 776 (766, 776) | 5.2 | 95% | 4.9 GB | 36.0 (8.8 + 9.1, 9.4 + 8.7) |
| Sentence of 20 words | 2 | 3 | 6 | bf16 | C++ (`perf_mode=flash_attention`) | 1163 (1163, 1139) | 5.2 | 96% | 4.9 GB | 52.3 (8.8 + 9.1 + 9.4, 8.7 + 8.2 + 8.0) |
| Sentence of 20 words | 2 | 4 | 8 | bf16 | C++ (`perf_mode=flash_attention`) | 1549 (1540, 1549) | 5.2 | 96% | 5.0 GB | 71.5 (8.8 + 9.1 + 9.4 + 8.7, 8.2 + 8.0 + 10.0 + 9.2) |
| Sentence of 20 words | 2 | 5 | 10 | bf16 | C++ (`perf_mode=flash_attention`) | 1998 (1998, 1991) | 5.0 | 96% | 5.1 GB | 88.7 (8.6 + 9.1 + 9.4 + 8.7 + 8.2, 8.0 + 10.0 + 9.2 + 8.8 + 8.6) |
| Sentence of 20 words | 2 | 6 | 12 | bf16 | C++ (`perf_mode=flash_attention`) | 2469 (2438, 2469) | 4.9 | 95% | 5.1 GB | 107.8 (8.5 + 9.1 + 9.4 + 8.7 + 8.2 + 8.0, 10.0 + 9.2 + 8.8 + 8.6 + 9.2 + 10.0) |
| Sentence of 20 words | 2 | 7 | 14 | bf16 | C++ (`perf_mode=flash_attention`) | 2928 (2914, 2928) | 4.8 | 95% | 5.2 GB | 126.4 (8.8 + 9.1 + 9.4 + 8.7 + 8.2 + 8.0 + 10.0, 9.2 + 8.8 + 8.6 + 9.2 + 10.0 + 8.9 + 9.4) |
| Sentence of 20 words | 1 | 1 | 1 | f16 | C++ (`perf_mode=off`) | 260 | 3.8 | 87% | 2.4 GB | 8.8 (8.8) |
| Sentence of 20 words | 1 | 2 | 2 | f16 | C++ (`perf_mode=off`) | 492 | 4.1 | 92% | 2.4 GB | 18.0 (8.8 + 9.1) |
| Sentence of 20 words | 1 | 3 | 3 | f16 | C++ (`perf_mode=off`) | 779 | 3.9 | 92% | 2.5 GB | 27.4 (8.8 + 9.1 + 9.4) |
| Sentence of 20 words | 1 | 4 | 4 | f16 | C++ (`perf_mode=off`) | 1038 | 3.9 | 92% | 2.5 GB | 36.0 (8.8 + 9.1 + 9.4 + 8.7) |
| Sentence of 20 words | 1 | 5 | 5 | f16 | C++ (`perf_mode=off`) | 1340 | 3.7 | 92% | 2.6 GB | 44.3 (8.8 + 9.1 + 9.4 + 8.7 + 8.2) |
| Sentence of 20 words | 2 | 1 | 2 | f16 | C++ (`perf_mode=off`) | 545 (543, 545) | 3.7 | 97% | 4.8 GB | 18.0 (8.8, 9.1) |
| Sentence of 20 words | 2 | 2 | 4 | f16 | C++ (`perf_mode=off`) | 1024 (1024, 1014) | 3.9 | 97% | 4.9 GB | 36.0 (8.8 + 9.1, 9.4 + 8.7) |
| Sentence of 20 words | 2 | 3 | 6 | f16 | C++ (`perf_mode=off`) | 1535 (1535, 1531) | 3.9 | 96% | 4.9 GB | 52.3 (8.8 + 9.1 + 9.4, 8.7 + 8.2 + 8.0) |
| Sentence of 20 words | 2 | 4 | 8 | f16 | C++ (`perf_mode=off`) | 2145 (2145, 2134) | 3.7 | 96% | 5.0 GB | 71.5 (8.8 + 9.1 + 9.4 + 8.7, 8.2 + 8.0 + 10.0 + 9.2) |
| Sentence of 20 words | 2 | 5 | 10 | f16 | C++ (`perf_mode=off`) | 2776 (2776, 2768) | 3.6 | 97% | 5.1 GB | 88.8 (8.8 + 9.1 + 9.4 + 8.7 + 8.2, 8.0 + 10.0 + 9.2 + 8.8 + 8.5) |
| Sentence of 20 words | 2 | 6 | 12 | f16 | C++ (`perf_mode=off`) | 3496 (3479, 3496) | 3.4 | 96% | 5.2 GB | 108.3 (8.8 + 9.1 + 9.4 + 8.7 + 8.2 + 8.0, 10.0 + 9.2 + 8.8 + 8.5 + 9.5 + 10.0) |
| Sentence of 20 words | 1 | 1 | 1 | bf16 | C++ (`perf_mode=off`) | 261 | 3.8 | 87% | 2.4 GB | 8.8 (8.8) |
| Sentence of 20 words | 1 | 2 | 2 | bf16 | C++ (`perf_mode=off`) | 540 | 3.7 | 92% | 2.4 GB | 18.0 (8.8 + 9.1) |
| Sentence of 20 words | 1 | 3 | 3 | bf16 | C++ (`perf_mode=off`) | 829 | 3.6 | 97% | 2.5 GB | 27.4 (8.8 + 9.1 + 9.4) |
| Sentence of 20 words | 1 | 4 | 4 | bf16 | C++ (`perf_mode=off`) | 1078 | 3.7 | 92% | 2.5 GB | 36.0 (8.8 + 9.1 + 9.4 + 8.7) |
| Sentence of 20 words | 2 | 1 | 2 | bf16 | C++ (`perf_mode=off`) | 560 (544, 560) | 3.6 | 96% | 4.8 GB | 18.0 (8.8, 9.1) |
| Sentence of 20 words | 2 | 2 | 4 | bf16 | C++ (`perf_mode=off`) | 1111 (1111, 1101) | 3.6 | 96% | 4.9 GB | 36.0 (8.8 + 9.1, 9.4 + 8.7) |
| Sentence of 20 words | 2 | 3 | 6 | bf16 | C++ (`perf_mode=off`) | 1636 (1636, 1622) | 3.7 | 96% | 4.9 GB | 52.3 (8.8 + 9.1 + 9.4, 8.7 + 8.2 + 8.0) |
| Sentence of 20 words | 2 | 4 | 8 | bf16 | C++ (`perf_mode=off`) | 2234 (2234, 2214) | 3.6 | 96% | 5.0 GB | 71.6 (8.8 + 9.1 + 9.4 + 8.7, 8.2 + 8.0 + 10.0 + 9.2) |
| Sentence of 20 words | 2 | 5 | 10 | bf16 | C++ (`perf_mode=off`) | 2833 (2833, 2830) | 3.5 | 97% | 5.1 GB | 88.7 (8.8 + 9.1 + 9.4 + 8.7 + 8.0, 8.0 + 10.0 + 9.2 + 8.8 + 8.6) |
| Sentence of 20 words | 2 | 6 | 12 | bf16 | C++ (`perf_mode=off`) | 3511 (3510, 3511) | 3.4 | 96% | 5.2 GB | 107.4 (8.8 + 9.1 + 9.4 + 8.7 + 8.2 + 8.0, 10.0 + 9.2 + 8.8 + 8.2 + 9.2 + 9.6) |
| Sentence of 20 words | 1 | 1 | 1 | f16 | Python (PyTorch) | 1152 | 0.9 | 25% | 5.9 GB | 8.2 (8.2) |
| Sentence of 20 words | 1 | 2 | 2 | f16 | Python (PyTorch) | 1052 | 1.9 | 45% | 5.9 GB | 17.5 (8.6 + 8.9) |
| Sentence of 20 words | 1 | 3 | 3 | f16 | Python (PyTorch) | 1103 | 2.7 | 62% | 5.9 GB | 26.5 (8.5 + 8.6 + 9.4) |
| Sentence of 20 words | 1 | 4 | 4 | f16 | Python (PyTorch) | 1145 | 3.5 | 81% | 5.9 GB | 35.0 (8.1 + 8.8 + 9.4 + 8.7) |
| Sentence of 20 words | 1 | 5 | 5 | f16 | Python (PyTorch) | 1365 | 3.7 | 87% | 5.9 GB | 43.1 (8.5 + 8.8 + 9.4 + 8.2 + 8.2) |
| Sentence of 20 words | 1 | 6 | 6 | f16 | Python (PyTorch) | 1675 | 3.6 | 87% | 5.9 GB | 50.2 (8.7 + 8.8 + 9.1 + 8.2 + 7.8 + 7.6) |
| Sentence of 20 words | 1 | 7 | 7 | f16 | Python (PyTorch) | 2059 | 3.4 | 89% | 5.9 GB | 61.9 (8.5 + 9.1 + 9.4 + 8.7 + 8.2 + 8.0 + 10.0) |
| Sentence of 20 words | 1 | 8 | 8 | f16 | Python (PyTorch) | 2431 | 3.3 | 89% | 6.5 GB | 70.6 (8.5 + 8.9 + 9.4 + 8.2 + 8.2 + 8.0 + 10.0 + 9.2) |
| Sentence of 20 words | 2 | 1 | 2 | f16 | Python (PyTorch) | 1120 (1095, 1120) | 1.8 | 87% | 10.1 GB | 17.1 (8.2, 8.9) |
| Sentence of 20 words | 2 | 2 | 4 | f16 | Python (PyTorch) | 1114 (1101, 1114) | 3.6 | 92% | 10.1 GB | 35.5 (8.6 + 8.9, 9.4 + 8.7) |
| Sentence of 20 words | 2 | 3 | 6 | f16 | Python (PyTorch) | 1564 (1564, 1543) | 3.8 | 94% | 10.1 GB | 51.1 (8.5 + 8.6 + 9.4, 8.7 + 7.9 + 8.0) |
| Sentence of 20 words | 2 | 4 | 8 | f16 | Python (PyTorch) | 2200 (2179, 2200) | 3.6 | 96% | 10.1 GB | 70.3 (8.1 + 8.8 + 9.4 + 8.7, 8.2 + 7.7 + 10.0 + 9.2) |
| Sentence of 20 words | 2 | 5 | 10 | f16 | Python (PyTorch) | 2836 (2772, 2836) | 3.5 | 97% | 10.1 GB | 87.1 (8.5 + 8.8 + 9.4 + 8.2 + 8.2, 8.0 + 9.8 + 8.8 + 8.8 + 8.6) |
| Sentence of 20 words | 2 | 6 | 12 | f16 | Python (PyTorch) | 3502 (3422, 3502) | 3.4 | 94% | 10.1 GB | 105.7 (8.7 + 8.8 + 9.1 + 8.2 + 7.8 + 7.6, 10.0 + 9.2 + 8.8 + 8.2 + 9.5 + 9.8) |
| Sentence of 20 words | 3 | 1 | 3 | f16 | Python (PyTorch) | 1575 (1575, 1254, 1567) | 1.9 | 95% | 13.0 GB | 26.4 (8.2, 8.9, 9.2) |
| Sentence of 20 words | 3 | 2 | 6 | f16 | Python (PyTorch) | 1615 (1615, 1614, 1578) | 3.7 | 95% | 13.0 GB | 51.1 (8.6 + 8.9, 9.4 + 8.7, 7.8 + 7.8) |
| Sentence of 20 words | 3 | 3 | 9 | f16 | Python (PyTorch) | 2416 (2291, 2266, 2416) | 3.7 | 94% | 13.0 GB | 78.7 (8.5 + 8.6 + 9.4, 8.7 + 7.9 + 8.0, 10.0 + 8.7 + 8.8) |
| Sentence of 20 words | 3 | 4 | 12 | f16 | Python (PyTorch) | 3288 (3215, 3236, 3288) | 3.6 | 96% | 13.0 GB | 107.1 (8.1 + 8.8 + 9.4 + 8.7, 8.2 + 7.7 + 10.0 + 9.2, 8.8 + 8.5 + 9.4 + 10.0) |
| Sentence of 20 words | 3 | 5 | 15 | f16 | Python (PyTorch) | 4246 (4053, 4246, 4229) | 3.5 | 96% | 13.7 GB | 133.8 (8.5 + 8.8 + 9.4 + 8.2 + 8.2, 8.0 + 9.8 + 8.8 + 8.8 + 8.6, 9.3 + 9.9 + 8.9 + 9.4 + 9.2) |
| Sentence of 20 words | 3 | 6 | 18 | f16 | Python (PyTorch) | 5156 (5110, 5156, 5133) | 3.5 | 96% | 15.0 GB | 158.2 (8.7 + 8.8 + 9.1 + 8.2 + 7.8 + 7.6, 10.0 + 9.2 + 8.8 + 8.2 + 9.5 + 9.8, 8.5 + 9.2 + 8.9 + 8.2 + 8.2 + 9.4) |
| Sentence of 20 words | 4 | 1 | 4 | f16 | Python (PyTorch) | 1276 (1243, 1276, 1260, 1232) | 3.1 | 91% | 20.9 GB | 34.8 (8.2, 8.9, 9.2, 8.4) |
| Sentence of 20 words | 4 | 2 | 8 | f16 | Python (PyTorch) | 2144 (2114, 2124, 2051, 2144) | 3.7 | 96% | 20.9 GB | 69.8 (8.6 + 8.9, 9.4 + 8.7, 7.8 + 7.8, 10.0 + 8.7) |
| Sentence of 20 words | 4 | 3 | 12 | f16 | Python (PyTorch) | 3192 (3064, 2977, 3192, 3185) | 3.8 | 96% | 20.9 GB | 106.2 (8.5 + 8.6 + 9.4, 8.7 + 7.9 + 8.0, 10.0 + 8.7 + 8.8, 8.3 + 9.2 + 10.0) |
| Sentence of 20 words | 4 | 4 | 16 | f16 | Python (PyTorch) | 4307 (4276, 4245, 4307, 4274) | 3.7 | 97% | 20.9 GB | 142.2 (8.1 + 8.8 + 9.4 + 8.7, 8.2 + 7.7 + 10.0 + 9.2, 8.8 + 8.5 + 9.4 + 10.0, 8.9 + 9.4 + 8.5 + 8.2) |
| Sentence of 20 words | 4 | 5 | 20 | f16 | Python (PyTorch) | 5600 (5369, 5547, 5600, 5472) | 3.6 | 97% | 20.9 GB | 177.7 (8.5 + 8.8 + 9.4 + 8.2 + 8.2, 8.0 + 9.8 + 8.8 + 8.8 + 8.6, 9.3 + 9.9 + 8.9 + 9.4 + 9.2, 8.0 + 7.9 + 9.8 + 9.6 + 8.8) |
| Sentence of 20 words | 4 | 6 | 24 | f16 | Python (PyTorch) | 6776 (6722, 6776, 6729, 6707) | 3.5 | 98% | 21.3 GB | 212.4 (8.7 + 8.8 + 9.1 + 8.2 + 7.8 + 7.6, 10.0 + 9.2 + 8.8 + 8.2 + 9.5 + 9.8, 8.5 + 9.2 + 8.9 + 8.2 + 8.2 + 9.4, 9.4 + 8.8 + 8.5 + 9.1 + 9.3 + 9.0) |
| Sentence of 20 words | 1 | 1 | 1 | bf16 | Python (PyTorch) | 952 | 1.1 | 32% | 6.4 GB | 8.4 (8.4) |
| Sentence of 20 words | 1 | 2 | 2 | bf16 | Python (PyTorch) | 1000 | 2.0 | 48% | 6.4 GB | 17.5 (8.6 + 8.9) |
| Sentence of 20 words | 1 | 3 | 3 | bf16 | Python (PyTorch) | 1048 | 2.9 | 66% | 6.4 GB | 26.7 (8.5 + 8.8 + 9.4) |
| Sentence of 20 words | 1 | 4 | 4 | bf16 | Python (PyTorch) | 1085 | 3.7 | 85% | 6.4 GB | 34.8 (8.2 + 8.5 + 9.4 + 8.7) |
| Sentence of 20 words | 1 | 5 | 5 | bf16 | Python (PyTorch) | 1351 | 3.7 | 88% | 6.4 GB | 42.6 (8.5 + 8.8 + 9.4 + 8.2 + 7.8) |
| Sentence of 20 words | 1 | 6 | 6 | bf16 | Python (PyTorch) | 1653 | 3.6 | 89% | 6.4 GB | 51.0 (8.6 + 8.9 + 9.4 + 8.2 + 8.2 + 7.6) |
| Sentence of 20 words | 1 | 7 | 7 | bf16 | Python (PyTorch) | 2051 | 3.4 | 89% | 6.9 GB | 60.6 (8.8 + 9.1 + 9.4 + 8.2 + 7.8 + 7.5 + 9.8) |
| Sentence of 20 words | 1 | 8 | 8 | bf16 | Python (PyTorch) | 2422 | 3.3 | 90% | 7.5 GB | 70.4 (8.8 + 8.9 + 9.4 + 8.1 + 8.2 + 8.0 + 9.8 + 9.1) |
| Sentence of 20 words | 2 | 1 | 2 | bf16 | Python (PyTorch) | 1127 (1127, 1047) | 1.8 | 93% | 10.1 GB | 17.3 (8.4, 8.9) |
| Sentence of 20 words | 2 | 2 | 4 | bf16 | Python (PyTorch) | 1124 (1124, 1079) | 3.6 | 99% | 10.1 GB | 35.3 (8.6 + 8.9, 9.4 + 8.4) |
| Sentence of 20 words | 2 | 3 | 6 | bf16 | Python (PyTorch) | 1564 (1564, 1519) | 3.8 | 95% | 10.1 GB | 51.2 (8.5 + 8.8 + 9.4, 8.7 + 7.9 + 8.0) |
| Sentence of 20 words | 2 | 4 | 8 | bf16 | Python (PyTorch) | 2228 (2091, 2228) | 3.6 | 95% | 10.1 GB | 70.0 (8.2 + 8.5 + 9.4 + 8.7, 8.2 + 8.0 + 9.8 + 9.1) |
| Sentence of 20 words | 2 | 5 | 10 | bf16 | Python (PyTorch) | 2820 (2750, 2820) | 3.5 | 94% | 10.1 GB | 86.0 (8.5 + 8.8 + 9.4 + 8.2 + 7.8, 7.7 + 9.9 + 8.8 + 8.5 + 8.6) |
| Sentence of 20 words | 2 | 6 | 12 | bf16 | Python (PyTorch) | 3483 (3402, 3483) | 3.4 | 95% | 10.1 GB | 106.4 (8.6 + 8.9 + 9.4 + 8.2 + 8.2 + 7.6, 10.0 + 9.1 + 8.8 + 8.3 + 9.5 + 9.7) |
| Sentence of 20 words | 3 | 1 | 3 | bf16 | Python (PyTorch) | 1322 (1190, 1322, 1313) | 2.3 | 78% | 13.0 GB | 26.5 (8.4, 8.9, 9.2) |
| Sentence of 20 words | 3 | 2 | 6 | bf16 | Python (PyTorch) | 1588 (1585, 1563, 1588) | 3.8 | 96% | 13.0 GB | 51.1 (8.6 + 8.9, 9.4 + 8.4, 7.8 + 8.0) |
| Sentence of 20 words | 3 | 3 | 9 | bf16 | Python (PyTorch) | 2346 (2330, 2288, 2346) | 3.8 | 95% | 13.0 GB | 78.6 (8.5 + 8.8 + 9.4, 8.7 + 7.9 + 8.0, 9.8 + 8.8 + 8.8) |
| Sentence of 20 words | 3 | 4 | 12 | bf16 | Python (PyTorch) | 3248 (3145, 3193, 3248) | 3.7 | 96% | 13.0 GB | 106.3 (8.2 + 8.5 + 9.4 + 8.7, 8.2 + 8.0 + 9.8 + 9.1, 8.6 + 8.3 + 9.5 + 10.0) |
| Sentence of 20 words | 3 | 5 | 15 | bf16 | Python (PyTorch) | 4257 (4089, 4169, 4257) | 3.5 | 96% | 13.7 GB | 132.1 (8.5 + 8.8 + 9.4 + 8.2 + 7.8, 7.7 + 9.9 + 8.8 + 8.5 + 8.6, 9.2 + 9.4 + 8.9 + 9.4 + 9.2) |
| Sentence of 20 words | 3 | 6 | 18 | bf16 | Python (PyTorch) | 5126 (5041, 5119, 5126) | 3.5 | 99% | 15.0 GB | 158.2 (8.6 + 8.9 + 9.4 + 8.2 + 8.2 + 7.6, 10.0 + 9.1 + 8.8 + 8.3 + 9.5 + 9.7, 8.5 + 9.2 + 8.5 + 8.1 + 8.2 + 9.4) |
| Sentence of 20 words | 4 | 1 | 4 | bf16 | Python (PyTorch) | 1299 (1299, 1179, 1256, 1297) | 3.1 | 87% | 17.7 GB | 35.0 (8.4, 8.9, 9.2, 8.4) |
| Sentence of 20 words | 4 | 2 | 8 | bf16 | Python (PyTorch) | 2141 (2117, 2098, 2078, 2141) | 3.7 | 96% | 17.7 GB | 69.9 (8.6 + 8.9, 9.4 + 8.4, 7.8 + 8.0, 10.0 + 8.8) |
| Sentence of 20 words | 4 | 3 | 12 | bf16 | Python (PyTorch) | 3151 (3042, 2975, 3151, 3117) | 3.8 | 97% | 17.7 GB | 106.5 (8.5 + 8.8 + 9.4, 8.7 + 7.9 + 8.0, 9.8 + 8.8 + 8.8, 8.6 + 9.2 + 10.0) |
| Sentence of 20 words | 4 | 4 | 16 | bf16 | Python (PyTorch) | 4254 (4124, 4254, 4230, 4215) | 3.8 | 97% | 17.7 GB | 141.3 (8.2 + 8.5 + 9.4 + 8.7, 8.2 + 8.0 + 9.8 + 9.1, 8.6 + 8.3 + 9.5 + 10.0, 8.9 + 9.4 + 8.5 + 8.1) |
| Sentence of 20 words | 4 | 5 | 20 | bf16 | Python (PyTorch) | 5575 (5351, 5525, 5575, 5475) | 3.6 | 97% | 18.4 GB | 176.0 (8.5 + 8.8 + 9.4 + 8.2 + 7.8, 7.7 + 9.9 + 8.8 + 8.5 + 8.6, 9.2 + 9.4 + 8.9 + 9.4 + 9.2, 8.0 + 7.9 + 9.8 + 9.4 + 8.8) |
| Sentence of 20 words | 4 | 6 | 24 | bf16 | Python (PyTorch) | 6767 (6591, 6767, 6678, 6735) | 3.5 | 98% | 20.0 GB | 212.9 (8.6 + 8.9 + 9.4 + 8.2 + 8.2 + 7.6, 10.0 + 9.1 + 8.8 + 8.3 + 9.5 + 9.7, 8.5 + 9.2 + 8.5 + 8.1 + 8.2 + 9.4, 9.4 + 8.8 + 8.7 + 9.1 + 9.5 + 9.1) |
| Sentence of 25 words | 1 | 1 | 1 | f16 | Python + FlashInfer | 691 | 1.4 | 27% | 4.4 GB | 9.5 (9.5) |
| Sentence of 25 words | 1 | 2 | 2 | f16 | Python + FlashInfer | 559 | 3.6 | 43% | 4.4 GB | 19.6 (9.5 + 10.2) |
| Sentence of 25 words | 1 | 3 | 3 | f16 | Python + FlashInfer | 575 | 5.2 | 67% | 4.4 GB | 30.7 (9.5 + 10.2 + 10.9) |
| Sentence of 25 words | 1 | 4 | 4 | f16 | Python + FlashInfer | 739 | 5.4 | 66% | 4.4 GB | 42.1 (9.4 + 10.2 + 10.9 + 11.5) |
| Sentence of 25 words | 1 | 5 | 5 | f16 | Python + FlashInfer | 829 | 6.0 | 80% | 4.4 GB | 54.3 (9.7 + 10.3 + 11.2 + 11.5 + 11.6) |
| Sentence of 25 words | 1 | 6 | 6 | f16 | Python + FlashInfer | 976 | 6.1 | 77% | 4.4 GB | 64.3 (9.4 + 10.2 + 11.2 + 11.5 + 11.0 + 11.0) |
| Sentence of 25 words | 1 | 7 | 7 | f16 | Python + FlashInfer | 1143 | 6.1 | 78% | 4.7 GB | 75.7 (9.6 + 10.2 + 11.1 + 11.2 + 11.6 + 11.1 + 11.0) |
| Sentence of 25 words | 1 | 8 | 8 | f16 | Python + FlashInfer | 1339 | 6.0 | 79% | 4.7 GB | 87.8 (9.5 + 10.4 + 11.2 + 11.5 + 11.0 + 10.9 + 11.2 + 12.1) |
| Sentence of 25 words | 1 | 9 | 9 | f16 | Python + FlashInfer | 1472 | 6.1 | 76% | 4.7 GB | 97.7 (9.4 + 10.3 + 11.2 + 11.5 + 11.5 + 11.0 + 11.3 + 12.1 + 9.4) |
| Sentence of 25 words | 2 | 1 | 2 | f16 | Python + FlashInfer | 624 (624, 613) | 3.2 | 87% | 8.2 GB | 19.5 (9.5, 10.0) |
| Sentence of 25 words | 2 | 2 | 4 | f16 | Python + FlashInfer | 763 (616, 763) | 5.2 | 82% | 8.2 GB | 42.3 (9.5 + 10.2, 11.2 + 11.5) |
| Sentence of 25 words | 2 | 3 | 6 | f16 | Python + FlashInfer | 1009 (954, 1009) | 5.9 | 88% | 10.0 GB | 64.3 (9.5 + 10.2 + 10.9, 11.3 + 11.2 + 11.1) |
| Sentence of 25 words | 2 | 4 | 8 | f16 | Python + FlashInfer | 1299 (1253, 1299) | 6.2 | 96% | 10.0 GB | 87.4 (9.4 + 10.2 + 10.9 + 11.5, 11.2 + 10.7 + 11.1 + 12.3) |
| Sentence of 25 words | 2 | 5 | 10 | f16 | Python + FlashInfer | 1586 (1585, 1586) | 6.3 | 90% | 10.0 GB | 109.3 (9.7 + 10.3 + 11.2 + 11.5 + 11.6, 10.9 + 11.3 + 12.1 + 9.4 + 11.2) |
| Sentence of 25 words | 2 | 6 | 12 | f16 | Python + FlashInfer | 1901 (1901, 1897) | 6.3 | 86% | 10.0 GB | 129.7 (9.4 + 10.2 + 11.2 + 11.5 + 11.0 + 11.0, 11.0 + 12.0 + 9.4 + 11.2 + 11.6 + 10.1) |
| Sentence of 25 words | 2 | 7 | 14 | f16 | Python + FlashInfer | 2356 (2178, 2356) | 5.9 | 87% | 10.0 GB | 152.8 (9.6 + 10.2 + 11.1 + 11.2 + 11.6 + 11.1 + 11.0, 12.2 + 9.6 + 10.8 + 11.9 + 10.1 + 10.5 + 11.9) |
| Sentence of 25 words | 2 | 8 | 16 | f16 | Python + FlashInfer | 2707 (2707, 2468) | 5.9 | 85% | 10.0 GB | 177.8 (9.5 + 10.4 + 11.2 + 11.5 + 11.0 + 10.9 + 11.2 + 12.1, 9.6 + 10.8 + 12.0 + 10.4 + 10.4 + 11.7 + 12.2 + 13.0) |
| Sentence of 25 words | 2 | 9 | 18 | f16 | Python + FlashInfer | 2802 (2772, 2802) | 6.4 | 88% | 10.0 GB | 199.8 (9.4 + 10.3 + 11.2 + 11.5 + 11.5 + 11.0 + 11.3 + 12.1 + 9.4, 10.9 + 12.0 + 10.1 + 10.6 + 12.0 + 12.2 + 13.0 + 11.6 + 9.7) |
| Sentence of 25 words | 2 | 10 | 20 | f16 | Python + FlashInfer | 3120 (3114, 3120) | 6.4 | 88% | 10.0 GB | 221.9 (9.5 + 10.3 + 11.2 + 11.5 + 11.6 + 10.9 + 11.0 + 12.3 + 9.5 + 10.9, 11.7 + 10.4 + 10.6 + 11.8 + 11.8 + 12.7 + 12.0 + 10.2 + 11.4 + 10.4) |
| Sentence of 25 words | 2 | 11 | 22 | f16 | Python + FlashInfer | 3430 (3430, 3417) | 6.4 | 89% | 10.7 GB | 245.2 (9.4 + 10.2 + 11.2 + 11.5 + 11.6 + 10.8 + 11.3 + 12.1 + 9.5 + 11.2 + 11.7, 10.2 + 10.6 + 12.0 + 12.2 + 12.7 + 11.8 + 9.9 + 11.4 + 10.6 + 10.8 + 12.1) |
| Sentence of 25 words | 2 | 12 | 24 | f16 | Python + FlashInfer | 3800 (3749, 3800) | 6.3 | 88% | 11.5 GB | 266.7 (9.6 + 10.2 + 11.2 + 11.5 + 11.4 + 10.9 + 11.0 + 12.1 + 9.6 + 11.2 + 11.8 + 10.1, 10.4 + 12.0 + 11.9 + 13.0 + 12.0 + 10.0 + 11.4 + 10.7 + 11.4 + 12.5 + 10.3 + 10.7) |
| Sentence of 25 words | 3 | 1 | 3 | f16 | Python + FlashInfer | 746 (637, 623, 746) | 4.0 | 78% | 17.9 GB | 30.5 (9.5, 10.0, 11.0) |
| Sentence of 25 words | 3 | 2 | 6 | f16 | Python + FlashInfer | 946 (922, 942, 946) | 6.3 | 91% | 17.9 GB | 64.4 (9.5 + 10.2, 11.2 + 11.5, 11.3 + 10.8) |
| Sentence of 25 words | 3 | 3 | 9 | f16 | Python + FlashInfer | 1413 (1380, 1413, 1392) | 6.4 | 90% | 17.9 GB | 97.0 (9.5 + 10.2 + 10.9, 11.3 + 11.2 + 11.1, 11.0 + 12.3 + 9.4) |
| Sentence of 25 words | 3 | 4 | 12 | f16 | Python + FlashInfer | 1845 (1845, 1838, 1833) | 6.5 | 93% | 17.9 GB | 130.6 (9.4 + 10.2 + 10.9 + 11.5, 11.2 + 10.7 + 11.1 + 12.3, 9.6 + 11.2 + 12.0 + 10.4) |
| Sentence of 25 words | 3 | 5 | 15 | f16 | Python + FlashInfer | 2287 (2287, 2252, 2278) | 6.6 | 92% | 17.9 GB | 165.2 (9.7 + 10.3 + 11.2 + 11.5 + 11.6, 10.9 + 11.3 + 12.1 + 9.4 + 11.2, 11.8 + 10.2 + 10.4 + 11.7 + 11.8) |
| Sentence of 25 words | 3 | 6 | 18 | f16 | Python + FlashInfer | 2709 (2691, 2664, 2709) | 6.6 | 91% | 17.9 GB | 198.4 (9.4 + 10.2 + 11.2 + 11.5 + 11.0 + 11.0, 11.0 + 12.0 + 9.4 + 11.2 + 11.6 + 10.1, 10.4 + 12.0 + 12.0 + 12.8 + 11.6 + 9.9) |
| Sentence of 25 words | 3 | 7 | 21 | f16 | Python + FlashInfer | 3216 (3178, 3131, 3216) | 6.5 | 93% | 17.9 GB | 232.4 (9.6 + 10.2 + 11.1 + 11.2 + 11.6 + 11.1 + 11.0, 12.2 + 9.6 + 10.8 + 11.9 + 10.1 + 10.5 + 11.9, 11.9 + 12.4 + 12.0 + 10.1 + 11.4 + 10.7 + 11.0) |
| Sentence of 25 words | 3 | 8 | 24 | f16 | Python + FlashInfer | 3603 (3559, 3553, 3603) | 6.7 | 92% | 17.9 GB | 266.4 (9.5 + 10.4 + 11.2 + 11.5 + 11.0 + 10.9 + 11.2 + 12.1, 9.6 + 10.8 + 12.0 + 10.4 + 10.4 + 11.7 + 12.2 + 13.0, 11.7 + 10.2 + 11.4 + 10.5 + 11.4 + 12.2 + 10.3 + 11.0) |
| Sentence of 25 words | 3 | 9 | 27 | f16 | Python + FlashInfer | 3998 (3968, 3998, 3963) | 6.8 | 94% | 17.9 GB | 297.3 (9.4 + 10.3 + 11.2 + 11.5 + 11.5 + 11.0 + 11.3 + 12.1 + 9.4, 10.9 + 12.0 + 10.1 + 10.6 + 12.0 + 12.2 + 13.0 + 11.6 + 9.7, 11.2 + 10.7 + 11.2 + 12.5 + 10.1 + 10.7 + 10.8 + 10.4 + 9.9) |
| Sentence of 25 words | 3 | 10 | 30 | f16 | Python + FlashInfer | 4645 (4645, 4439, 4401) | 6.5 | 91% | 18.9 GB | 328.6 (9.5 + 10.3 + 11.2 + 11.5 + 11.6 + 10.9 + 11.0 + 12.3 + 9.5 + 10.9, 11.7 + 10.4 + 10.6 + 11.8 + 11.8 + 12.7 + 12.0 + 10.2 + 11.4 + 10.4, 11.2 + 12.3 + 10.1 + 10.4 + 11.2 + 10.6 + 10.0 + 11.2 + 9.9 + 9.7) |
| Sentence of 25 words | 3 | 11 | 33 | f16 | Python + FlashInfer | 4925 (4906, 4925, 4861) | 6.7 | 93% | 20.1 GB | 361.9 (9.4 + 10.2 + 11.2 + 11.5 + 11.6 + 10.8 + 11.3 + 12.1 + 9.5 + 11.2 + 11.7, 10.2 + 10.6 + 12.0 + 12.2 + 12.7 + 11.8 + 9.9 + 11.4 + 10.6 + 10.8 + 12.1, 10.0 + 11.0 + 11.0 + 10.4 + 10.0 + 11.0 + 10.3 + 9.7 + 11.6 + 10.1 + 11.7) |
| Sentence of 25 words | 3 | 12 | 36 | f16 | Python + FlashInfer | 5332 (5296, 5332, 5314) | 6.8 | 93% | 21.3 GB | 394.5 (9.6 + 10.2 + 11.2 + 11.5 + 11.4 + 10.9 + 11.0 + 12.1 + 9.6 + 11.2 + 11.8 + 10.1, 10.4 + 12.0 + 11.9 + 13.0 + 12.0 + 10.0 + 11.4 + 10.7 + 11.4 + 12.5 + 10.3 + 10.7, 10.9 + 10.6 + 10.0 + 10.9 + 10.5 + 10.0 + 11.4 + 10.0 + 11.4 + 12.1 + 10.6 + 9.3) |
| Sentence of 25 words | 4 | 1 | 4 | f16 | Python + FlashInfer | 728 (687, 715, 728, 727) | 5.5 | 90% | 16.6 GB | 41.8 (9.5, 10.0, 11.0, 11.3) |
| Sentence of 25 words | 4 | 2 | 8 | f16 | Python + FlashInfer | 1240 (1170, 1213, 1234, 1240) | 6.4 | 92% | 16.6 GB | 87.7 (9.5 + 10.2, 11.2 + 11.5, 11.3 + 10.8, 11.1 + 12.2) |
| Sentence of 25 words | 4 | 3 | 12 | f16 | Python + FlashInfer | 1846 (1782, 1846, 1825, 1844) | 6.5 | 99% | 16.6 GB | 130.0 (9.5 + 10.2 + 10.9, 11.3 + 11.2 + 11.1, 11.0 + 12.3 + 9.4, 10.8 + 11.8 + 10.4) |
| Sentence of 25 words | 4 | 4 | 16 | f16 | Python + FlashInfer | 2487 (2354, 2487, 2333, 2465) | 6.4 | 94% | 16.6 GB | 178.1 (9.4 + 10.2 + 10.9 + 11.5, 11.2 + 10.7 + 11.1 + 12.3, 9.6 + 11.2 + 12.0 + 10.4, 10.4 + 11.9 + 12.2 + 13.0) |
| Sentence of 25 words | 4 | 5 | 20 | f16 | Python + FlashInfer | 2998 (2918, 2954, 2967, 2998) | 6.7 | 95% | 16.6 GB | 221.8 (9.7 + 10.3 + 11.2 + 11.5 + 11.6, 10.9 + 11.3 + 12.1 + 9.4 + 11.2, 11.8 + 10.2 + 10.4 + 11.7 + 11.8, 12.6 + 12.0 + 10.1 + 11.4 + 10.4) |
| Sentence of 25 words | 4 | 6 | 24 | f16 | Python + FlashInfer | 3536 (3473, 3507, 3533, 3536) | 6.8 | 95% | 16.6 GB | 265.3 (9.4 + 10.2 + 11.2 + 11.5 + 11.0 + 11.0, 11.0 + 12.0 + 9.4 + 11.2 + 11.6 + 10.1, 10.4 + 12.0 + 12.0 + 12.8 + 11.6 + 9.9, 11.2 + 10.7 + 11.0 + 12.4 + 10.5 + 11.0) |
| Sentence of 25 words | 4 | 7 | 28 | f16 | Python + FlashInfer | 4237 (4029, 4220, 4090, 4237) | 6.6 | 95% | 16.6 GB | 307.8 (9.6 + 10.2 + 11.1 + 11.2 + 11.6 + 11.1 + 11.0, 12.2 + 9.6 + 10.8 + 11.9 + 10.1 + 10.5 + 11.9, 11.9 + 12.4 + 12.0 + 10.1 + 11.4 + 10.7 + 11.0, 12.4 + 10.5 + 10.7 + 10.6 + 10.6 + 9.9 + 10.8) |
| Sentence of 25 words | 4 | 8 | 32 | f16 | Python + FlashInfer | 4676 (4653, 4676, 4648, 4531) | 6.8 | 93% | 16.6 GB | 350.1 (9.5 + 10.4 + 11.2 + 11.5 + 11.0 + 10.9 + 11.2 + 12.1, 9.6 + 10.8 + 12.0 + 10.4 + 10.4 + 11.7 + 12.2 + 13.0, 11.7 + 10.2 + 11.4 + 10.5 + 11.4 + 12.2 + 10.3 + 11.0, 10.7 + 10.4 + 9.6 + 11.1 + 10.4 + 10.0 + 11.6 + 9.9) |
| Sentence of 25 words | 4 | 9 | 36 | f16 | Python + FlashInfer | 5179 (5145, 5179, 5171, 5094) | 7.0 | 96% | 16.6 GB | 394.7 (9.4 + 10.3 + 11.2 + 11.5 + 11.5 + 11.0 + 11.3 + 12.1 + 9.4, 10.9 + 12.0 + 10.1 + 10.6 + 12.0 + 12.2 + 13.0 + 11.6 + 9.7, 11.2 + 10.7 + 11.2 + 12.5 + 10.1 + 10.7 + 10.8 + 10.4 + 9.9, 11.0 + 10.9 + 10.0 + 11.4 + 10.2 + 11.7 + 12.4 + 10.5 + 9.2) |
| Sentence of 25 words | 4 | 10 | 40 | f16 | Python + FlashInfer | 5762 (5724, 5762, 5731, 5744) | 6.9 | 96% | 17.9 GB | 437.9 (9.5 + 10.3 + 11.2 + 11.5 + 11.6 + 10.9 + 11.0 + 12.3 + 9.5 + 10.9, 11.7 + 10.4 + 10.6 + 11.8 + 11.8 + 12.7 + 12.0 + 10.2 + 11.4 + 10.4, 11.2 + 12.3 + 10.1 + 10.4 + 11.2 + 10.6 + 10.0 + 11.2 + 9.9 + 9.7, 11.6 + 10.2 + 11.7 + 12.5 + 10.6 + 9.2 + 11.0 + 11.2 + 11.2 + 10.3) |
| Sentence of 25 words | 4 | 11 | 44 | f16 | Python + FlashInfer | 6407 (6407, 6400, 6237, 6349) | 6.9 | 97% | 19.4 GB | 483.1 (9.4 + 10.2 + 11.2 + 11.5 + 11.6 + 10.8 + 11.3 + 12.1 + 9.5 + 11.2 + 11.7, 10.2 + 10.6 + 12.0 + 12.2 + 12.7 + 11.8 + 9.9 + 11.4 + 10.6 + 10.8 + 12.1, 10.0 + 11.0 + 11.0 + 10.4 + 10.0 + 11.0 + 10.3 + 9.7 + 11.6 + 10.1 + 11.7, 12.0 + 10.6 + 9.5 + 11.4 + 11.5 + 11.3 + 10.5 + 12.2 + 10.7 + 11.7 + 9.9) |
| Sentence of 25 words | 4 | 12 | 48 | f16 | Python + FlashInfer | 7053 (6940, 7009, 6872, 7053) | 6.8 | 96% | 21.0 GB | 529.5 (9.6 + 10.2 + 11.2 + 11.5 + 11.4 + 10.9 + 11.0 + 12.1 + 9.6 + 11.2 + 11.8 + 10.1, 10.4 + 12.0 + 11.9 + 13.0 + 12.0 + 10.0 + 11.4 + 10.7 + 11.4 + 12.5 + 10.3 + 10.7, 10.9 + 10.6 + 10.0 + 10.9 + 10.5 + 10.0 + 11.4 + 10.0 + 11.4 + 12.1 + 10.6 + 9.3, 11.2 + 11.3 + 11.6 + 10.5 + 12.2 + 10.4 + 11.9 + 10.1 + 11.4 + 11.6 + 12.3 + 10.5) |
| Sentence of 25 words | 5 | 1 | 5 | f16 | Python + FlashInfer | 903 (859, 896, 900, 903, 896) | 5.5 | 81% | 22.9 GB | 53.0 (9.5, 10.0, 11.0, 11.3, 11.2) |
| Sentence of 25 words | 5 | 2 | 10 | f16 | Python + FlashInfer | 1511 (1471, 1496, 1497, 1511, 1469) | 6.6 | 94% | 22.9 GB | 108.4 (9.5 + 10.2, 11.2 + 11.5, 11.3 + 10.8, 11.1 + 12.2, 9.5 + 11.2) |
| Sentence of 25 words | 5 | 3 | 15 | f16 | Python + FlashInfer | 2277 (2211, 2267, 2233, 2234, 2277) | 6.6 | 94% | 22.9 GB | 164.4 (9.5 + 10.2 + 10.9, 11.3 + 11.2 + 11.1, 11.0 + 12.3 + 9.4, 10.8 + 11.8 + 10.4, 10.4 + 12.0 + 12.0) |
| Sentence of 25 words | 5 | 4 | 20 | f16 | Python + FlashInfer | 3073 (2908, 3037, 2914, 3073, 2962) | 6.5 | 95% | 22.9 GB | 222.0 (9.4 + 10.2 + 10.9 + 11.5, 11.2 + 10.7 + 11.1 + 12.3, 9.6 + 11.2 + 12.0 + 10.4, 10.4 + 11.9 + 12.2 + 13.0, 11.6 + 10.2 + 11.4 + 10.7) |
| Sentence of 25 words | 5 | 5 | 25 | f16 | Python + FlashInfer | 3718 (3594, 3677, 3677, 3660, 3718) | 6.7 | 95% | 22.9 GB | 277.7 (9.7 + 10.3 + 11.2 + 11.5 + 11.6, 10.9 + 11.3 + 12.1 + 9.4 + 11.2, 11.8 + 10.2 + 10.4 + 11.7 + 11.8, 12.6 + 12.0 + 10.1 + 11.4 + 10.4, 11.4 + 12.2 + 10.5 + 11.0 + 10.9) |
| Sentence of 25 words | 5 | 6 | 30 | f16 | Python + FlashInfer | 4335 (4255, 4282, 4335, 4329, 4232) | 6.9 | 96% | 22.9 GB | 328.4 (9.4 + 10.2 + 11.2 + 11.5 + 11.0 + 11.0, 11.0 + 12.0 + 9.4 + 11.2 + 11.6 + 10.1, 10.4 + 12.0 + 12.0 + 12.8 + 11.6 + 9.9, 11.2 + 10.7 + 11.0 + 12.4 + 10.5 + 11.0, 10.9 + 10.4 + 10.0 + 10.9 + 10.9 + 10.0) |
| Sentence of 25 words | 5 | 7 | 35 | f16 | Python + FlashInfer | 5076 (4994, 5009, 5076, 5025, 5030) | 6.9 | 97% | 22.9 GB | 382.9 (9.6 + 10.2 + 11.1 + 11.2 + 11.6 + 11.1 + 11.0, 12.2 + 9.6 + 10.8 + 11.9 + 10.1 + 10.5 + 11.9, 11.9 + 12.4 + 12.0 + 10.1 + 11.4 + 10.7 + 11.0, 12.4 + 10.5 + 10.7 + 10.6 + 10.6 + 9.9 + 10.8, 9.8 + 9.5 + 11.2 + 9.9 + 11.8 + 12.1 + 10.7) |
| Sentence of 25 words | 5 | 8 | 40 | f16 | Python + FlashInfer | 5738 (5681, 5738, 5675, 5520, 5732) | 7.0 | 97% | 22.9 GB | 438.1 (9.5 + 10.4 + 11.2 + 11.5 + 11.0 + 10.9 + 11.2 + 12.1, 9.6 + 10.8 + 12.0 + 10.4 + 10.4 + 11.7 + 12.2 + 13.0, 11.7 + 10.2 + 11.4 + 10.5 + 11.4 + 12.2 + 10.3 + 11.0, 10.7 + 10.4 + 9.6 + 11.1 + 10.4 + 10.0 + 11.6 + 9.9, 11.8 + 12.3 + 10.7 + 9.2 + 10.9 + 11.2 + 11.6 + 10.4) |
| Sentence of 25 words | 5 | 9 | 45 | f16 | Python + FlashInfer | 6428 (6379, 6428, 6287, 6249, 6406) | 7.0 | 97% | 22.9 GB | 495.3 (9.4 + 10.3 + 11.2 + 11.5 + 11.5 + 11.0 + 11.3 + 12.1 + 9.4, 10.9 + 12.0 + 10.1 + 10.6 + 12.0 + 12.2 + 13.0 + 11.6 + 9.7, 11.2 + 10.7 + 11.2 + 12.5 + 10.1 + 10.7 + 10.8 + 10.4 + 9.9, 11.0 + 10.9 + 10.0 + 11.4 + 10.2 + 11.7 + 12.4 + 10.5 + 9.2, 11.3 + 11.3 + 11.6 + 10.5 + 12.3 + 10.4 + 11.5 + 10.4 + 11.3) |
| Sentence of 25 words | 5 | 10 | 50 | f16 | Python + FlashInfer | 7166 (7090, 7163, 7044, 7115, 7166) | 7.0 | 98% | 22.9 GB | 549.0 (9.5 + 10.3 + 11.2 + 11.5 + 11.6 + 10.9 + 11.0 + 12.3 + 9.5 + 10.9, 11.7 + 10.4 + 10.6 + 11.8 + 11.8 + 12.7 + 12.0 + 10.2 + 11.4 + 10.4, 11.2 + 12.3 + 10.1 + 10.4 + 11.2 + 10.6 + 10.0 + 11.2 + 9.9 + 9.7, 11.6 + 10.2 + 11.7 + 12.5 + 10.6 + 9.2 + 11.0 + 11.2 + 11.2 + 10.3, 12.0 + 10.3 + 11.5 + 10.4 + 11.2 + 11.4 + 12.2 + 10.2 + 12.2 + 9.6) |
| Sentence of 25 words | 5 | 11 | 55 | f16 | Python + FlashInfer | 7872 (7836, 7872, 7705, 7864, 7719) | 7.0 | 96% | 22.9 GB | 601.8 (9.4 + 10.2 + 11.2 + 11.5 + 11.6 + 10.8 + 11.3 + 12.1 + 9.5 + 11.2 + 11.7, 10.2 + 10.6 + 12.0 + 12.2 + 12.7 + 11.8 + 9.9 + 11.4 + 10.6 + 10.8 + 12.1, 10.0 + 11.0 + 11.0 + 10.4 + 10.0 + 11.0 + 10.3 + 9.7 + 11.6 + 10.1 + 11.7, 12.0 + 10.6 + 9.5 + 11.4 + 11.5 + 11.3 + 10.5 + 12.2 + 10.7 + 11.7 + 9.9, 11.1 + 11.1 + 12.6 + 10.3 + 12.4 + 9.3 + 10.4 + 11.7 + 9.4 + 10.6 + 9.7) |
| Sentence of 25 words | 5 | 12 | 60 | f16 | Python + FlashInfer | 8649 (8540, 8649, 8455, 8627, 8527) | 6.9 | 97% | 22.9 GB | 658.7 (9.6 + 10.2 + 11.2 + 11.5 + 11.4 + 10.9 + 11.0 + 12.1 + 9.6 + 11.2 + 11.8 + 10.1, 10.4 + 12.0 + 11.9 + 13.0 + 12.0 + 10.0 + 11.4 + 10.7 + 11.4 + 12.5 + 10.3 + 10.7, 10.9 + 10.6 + 10.0 + 10.9 + 10.5 + 10.0 + 11.4 + 10.0 + 11.4 + 12.1 + 10.6 + 9.3, 11.2 + 11.3 + 11.6 + 10.5 + 12.2 + 10.4 + 11.9 + 10.1 + 11.4 + 11.6 + 12.3 + 10.5, 12.0 + 9.8 + 10.4 + 11.6 + 9.3 + 10.4 + 9.4 + 11.1 + 10.5 + 12.7 + 10.3 + 11.6) |
| Sentence of 25 words | 6 | 1 | 6 | f16 | Python + FlashInfer | 1065 (1015, 1056, 1016, 1058, 1065, 1059) | 5.6 | 93% | 20.3 GB | 64.1 (9.5, 10.0, 11.0, 11.3, 11.2, 11.1) |
| Sentence of 25 words | 6 | 2 | 12 | f16 | Python + FlashInfer | 1794 (1744, 1794, 1790, 1757, 1749, 1767) | 6.7 | 95% | 20.7 GB | 130.9 (9.5 + 10.2, 11.2 + 11.5, 11.3 + 10.8, 11.1 + 12.2, 9.5 + 11.2, 12.0 + 10.4) |
| Sentence of 25 words | 6 | 3 | 18 | f16 | Python + FlashInfer | 2768 (2643, 2725, 2677, 2713, 2730, 2768) | 6.5 | 94% | 21.1 GB | 198.5 (9.5 + 10.2 + 10.9, 11.3 + 11.2 + 11.1, 11.0 + 12.3 + 9.4, 10.8 + 11.8 + 10.4, 10.4 + 12.0 + 12.0, 12.6 + 11.6 + 9.8) |
| Sentence of 25 words | 6 | 4 | 24 | f16 | Python + FlashInfer | 3628 (3480, 3628, 3464, 3626, 3533, 3551) | 6.6 | 96% | 22.1 GB | 266.1 (9.4 + 10.2 + 10.9 + 11.5, 11.2 + 10.7 + 11.1 + 12.3, 9.6 + 11.2 + 12.0 + 10.4, 10.4 + 11.9 + 12.2 + 13.0, 11.6 + 10.2 + 11.4 + 10.7, 11.0 + 12.0 + 10.5 + 10.7) |
| Sentence of 25 words | 6 | 5 | 30 | f16 | Python + FlashInfer | 4374 (4311, 4311, 4374, 4367, 4344, 4250) | 6.9 | 97% | 23.3 GB | 329.0 (9.7 + 10.3 + 11.2 + 11.5 + 11.6, 10.9 + 11.3 + 12.1 + 9.4 + 11.2, 11.8 + 10.2 + 10.4 + 11.7 + 11.8, 12.6 + 12.0 + 10.1 + 11.4 + 10.4, 11.4 + 12.2 + 10.5 + 11.0 + 10.9, 10.2 + 9.7 + 11.4 + 10.4 + 9.7) |
| Sentence of 25 words | 6 | 6 | 36 | f16 | Python + FlashInfer | 5140 (5050, 5078, 5140, 5136, 4982, 5088) | 7.0 | 97% | 21.5 GB | 394.5 (9.4 + 10.2 + 11.2 + 11.5 + 11.0 + 11.0, 11.0 + 12.0 + 9.4 + 11.2 + 11.6 + 10.1, 10.4 + 12.0 + 12.0 + 12.8 + 11.6 + 9.9, 11.2 + 10.7 + 11.0 + 12.4 + 10.5 + 11.0, 10.9 + 10.4 + 10.0 + 10.9 + 10.9 + 10.0, 11.6 + 10.2 + 11.6 + 12.4 + 10.7 + 9.5) |
| Sentence of 25 words | 6 | 7 | 42 | f16 | Python + FlashInfer | 6082 (6061, 6010, 6082, 5959, 6000, 5980) | 6.9 | 98% | 23.3 GB | 459.8 (9.6 + 10.2 + 11.1 + 11.2 + 11.6 + 11.1 + 11.0, 12.2 + 9.6 + 10.8 + 11.9 + 10.1 + 10.5 + 11.9, 11.9 + 12.4 + 12.0 + 10.1 + 11.4 + 10.7 + 11.0, 12.4 + 10.5 + 10.7 + 10.6 + 10.6 + 9.9 + 10.8, 9.8 + 9.5 + 11.2 + 9.9 + 11.8 + 12.1 + 10.7, 9.4 + 11.4 + 11.1 + 11.6 + 10.5 + 12.1 + 10.7) |
| Sentence of 25 words | 6 | 8 | 48 | f16 | Python + FlashInfer | 6840 (6829, 6807, 6819, 6689, 6840, 6834) | 7.0 | 97% | 22.5 GB | 529.2 (9.5 + 10.4 + 11.2 + 11.5 + 11.0 + 10.9 + 11.2 + 12.1, 9.6 + 10.8 + 12.0 + 10.4 + 10.4 + 11.7 + 12.2 + 13.0, 11.7 + 10.2 + 11.4 + 10.5 + 11.4 + 12.2 + 10.3 + 11.0, 10.7 + 10.4 + 9.6 + 11.1 + 10.4 + 10.0 + 11.6 + 9.9, 11.8 + 12.3 + 10.7 + 9.2 + 10.9 + 11.2 + 11.6 + 10.4, 12.6 + 10.2 + 11.9 + 10.4 + 11.4 + 11.6 + 12.6 + 10.3) |
| Sentence of 25 words | 6 | 9 | 54 | f16 | Python + FlashInfer | 7702 (7532, 7702, 7603, 7539, 7661, 7553) | 7.0 | 98% | 21.5 GB | 592.3 (9.4 + 10.3 + 11.2 + 11.5 + 11.5 + 11.0 + 11.3 + 12.1 + 9.4, 10.9 + 12.0 + 10.1 + 10.6 + 12.0 + 12.2 + 13.0 + 11.6 + 9.7, 11.2 + 10.7 + 11.2 + 12.5 + 10.1 + 10.7 + 10.8 + 10.4 + 9.9, 11.0 + 10.9 + 10.0 + 11.4 + 10.2 + 11.7 + 12.4 + 10.5 + 9.2, 11.3 + 11.3 + 11.6 + 10.5 + 12.3 + 10.4 + 11.5 + 10.4 + 11.3, 11.4 + 12.1 + 10.5 + 12.4 + 9.3 + 10.1 + 11.7 + 9.4 + 10.3) |
| Sentence of 25 words | 6 | 10 | 60 | f16 | Python + FlashInfer | 8576 (8459, 8576, 8434, 8472, 8554, 8448) | 7.0 | 98% | 22.9 GB | 657.3 (9.5 + 10.3 + 11.2 + 11.5 + 11.6 + 10.9 + 11.0 + 12.3 + 9.5 + 10.9, 11.7 + 10.4 + 10.6 + 11.8 + 11.8 + 12.7 + 12.0 + 10.2 + 11.4 + 10.4, 11.2 + 12.3 + 10.1 + 10.4 + 11.2 + 10.6 + 10.0 + 11.2 + 9.9 + 9.7, 11.6 + 10.2 + 11.7 + 12.5 + 10.6 + 9.2 + 11.0 + 11.2 + 11.2 + 10.3, 12.0 + 10.3 + 11.5 + 10.4 + 11.2 + 11.4 + 12.2 + 10.2 + 12.2 + 9.6, 10.3 + 11.4 + 9.4 + 10.6 + 9.7 + 10.5 + 10.7 + 13.3 + 10.7 + 11.7) |
| Sentence of 25 words | 6 | 11 | 66 | f16 | Python + FlashInfer | failed: instance 4 exited (rc 1): CUDA out of memory | – | – | – | – |
| Sentence of 25 words | 1 | 1 | 1 | bf16 | Python + FlashInfer | failed: bf16 not supported: upstream `apply_flashinfer` plans attention as f16 only | – | – | – | – |
| Sentence of 25 words | 1 | 1 | 1 | f16 | Python + FlashInfer + CUDA graphs | 277 | 3.6 | 80% | 14.3 GB | 9.4 (9.4) |
| Sentence of 25 words | 1 | 2 | 2 | f16 | Python + FlashInfer + CUDA graphs | 480 | 4.2 | 81% | 14.3 GB | 19.4 (9.3 + 10.0) |
| Sentence of 25 words | 1 | 3 | 3 | f16 | Python + FlashInfer + CUDA graphs | 703 | 4.3 | 85% | 14.3 GB | 30.2 (9.4 + 9.8 + 11.0) |
| Sentence of 25 words | 1 | 4 | 4 | f16 | Python + FlashInfer + CUDA graphs | 931 | 4.3 | 85% | 14.3 GB | 42.4 (9.6 + 10.1 + 11.2 + 11.4) |
| Sentence of 25 words | 1 | 5 | 5 | f16 | Python + FlashInfer + CUDA graphs | 1151 | 4.3 | 92% | 14.3 GB | 53.4 (9.6 + 10.2 + 11.2 + 11.5 + 10.8) |
| Sentence of 25 words | 1 | 6 | 6 | f16 | Python + FlashInfer + CUDA graphs | 1368 | 4.4 | 85% | 14.3 GB | 64.1 (9.4 + 10.2 + 11.1 + 11.4 + 10.9 + 11.1) |
| Sentence of 25 words | 1 | 7 | 7 | f16 | Python + FlashInfer + CUDA graphs | 1602 | 4.4 | 86% | 14.3 GB | 75.1 (9.4 + 10.1 + 11.0 + 11.4 + 11.0 + 11.0 + 11.2) |
| Sentence of 25 words | 1 | 8 | 8 | f16 | Python + FlashInfer + CUDA graphs | 2091 | 3.8 | 89% | 14.7 GB | 87.4 (9.4 + 10.2 + 11.2 + 11.3 + 11.2 + 11.1 + 10.9 + 12.1) |
| Sentence of 25 words | 1 | 9 | 9 | f16 | Python + FlashInfer + CUDA graphs | 2349 | 3.8 | 89% | 15.3 GB | 96.5 (9.4 + 10.2 + 10.8 + 11.5 + 11.1 + 11.1 + 11.0 + 12.0 + 9.4) |
| Sentence of 25 words | 2 | 1 | 2 | f16 | Python + FlashInfer + CUDA graphs | 497 (490, 497) | 4.0 | 90% | 16.8 GB | 19.6 (9.4, 10.2) |
| Sentence of 25 words | 2 | 2 | 4 | f16 | Python + FlashInfer + CUDA graphs | 916 (912, 916) | 4.4 | 92% | 16.8 GB | 41.6 (9.3 + 10.0, 10.9 + 11.4) |
| Sentence of 25 words | 2 | 3 | 6 | f16 | Python + FlashInfer + CUDA graphs | 1375 (1375, 1374) | 4.4 | 94% | 16.8 GB | 64.2 (9.4 + 9.8 + 11.0, 11.3 + 11.5 + 11.1) |
| Sentence of 25 words | 2 | 4 | 8 | f16 | Python + FlashInfer + CUDA graphs | 1948 (1860, 1948) | 4.1 | 93% | 16.9 GB | 88.0 (9.6 + 10.1 + 11.2 + 11.4, 11.2 + 11.0 + 11.2 + 12.3) |
| Sentence of 25 words | 2 | 5 | 10 | f16 | Python + FlashInfer + CUDA graphs | 2429 (2273, 2429) | 4.1 | 92% | 17.2 GB | 108.3 (9.6 + 10.2 + 11.2 + 11.5 + 10.8, 11.1 + 10.9 + 12.2 + 9.4 + 11.2) |
| Sentence of 25 words | 3 | 1 | 3 | f16 | Python + FlashInfer + CUDA graphs | 722 (690, 717, 722) | 4.2 | 92% | 22.7 GB | 30.6 (9.4, 10.2, 11.0) |
| Sentence of 25 words | 3 | 2 | 6 | f16 | Python + FlashInfer + CUDA graphs | 1343 (1343, 1309, 1339) | 4.5 | 94% | 22.7 GB | 63.9 (9.3 + 10.0, 10.9 + 11.4, 11.2 + 11.1) |
| Sentence of 25 words | 3 | 3 | 9 | f16 | Python + FlashInfer + CUDA graphs | 2089 (2013, 2033, 2089) | 4.3 | 94% | 22.6 GB | 96.9 (9.4 + 9.8 + 11.0, 11.3 + 11.5 + 11.1, 11.2 + 12.2 + 9.3) |
| Sentence of 25 words | 3 | 4 | 12 | f16 | Python + FlashInfer + CUDA graphs | 2812 (2644, 2812, 2630) | 4.3 | 94% | 22.6 GB | 130.4 (9.6 + 10.1 + 11.2 + 11.4, 11.2 + 11.0 + 11.2 + 12.3, 9.6 + 11.2 + 11.4 + 10.2) |
| Sentence of 25 words | 3 | 5 | 15 | f16 | Python + FlashInfer + CUDA graphs | 3504 (3336, 3504, 3333) | 4.3 | 93% | 23.2 GB | 164.6 (9.6 + 10.2 + 11.2 + 11.5 + 10.8, 11.1 + 10.9 + 12.2 + 9.4 + 11.2, 11.7 + 10.2 + 10.6 + 11.6 + 12.2) |
| Sentence of 25 words | 4 | 1 | 4 | f16 | Python + FlashInfer + CUDA graphs | 947 (947, 941, 931, 939) | 4.2 | 93% | 14.4 GB | 41.9 (9.4, 10.2, 11.0, 11.3) |
| Sentence of 25 words | 4 | 2 | 8 | f16 | Python + FlashInfer + CUDA graphs | 1834 (1768, 1774, 1734, 1834) | 4.4 | 99% | 14.8 GB | 87.3 (9.3 + 10.0, 10.9 + 11.4, 11.2 + 11.1, 11.2 + 12.2) |
| Sentence of 25 words | 4 | 3 | 12 | f16 | Python + FlashInfer + CUDA graphs | 2731 (2641, 2650, 2731, 2683) | 4.4 | 96% | 15.4 GB | 130.1 (9.4 + 9.8 + 11.0, 11.3 + 11.5 + 11.1, 11.2 + 12.2 + 9.3, 10.8 + 12.0 + 10.4) |
| Sentence of 25 words | 4 | 4 | 16 | f16 | Python + FlashInfer + CUDA graphs | 3769 (3577, 3766, 3556, 3769) | 4.2 | 96% | 16.4 GB | 177.4 (9.6 + 10.1 + 11.2 + 11.4, 11.2 + 11.0 + 11.2 + 12.3, 9.6 + 11.2 + 11.4 + 10.2, 10.6 + 11.8 + 11.6 + 13.0) |
| Sentence of 25 words | 4 | 5 | 20 | f16 | Python + FlashInfer + CUDA graphs | 4734 (4352, 4734, 4371, 4726) | 4.2 | 96% | 17.8 GB | 220.9 (9.6 + 10.2 + 11.2 + 11.5 + 10.8, 11.1 + 10.9 + 12.2 + 9.4 + 11.2, 11.7 + 10.2 + 10.6 + 11.6 + 12.2, 12.6 + 11.5 + 10.2 + 11.4 + 10.7) |
| Sentence of 25 words | 4 | 6 | 24 | f16 | Python + FlashInfer + CUDA graphs | 5825 (5278, 5783, 5792, 5825) | 4.1 | 97% | 19.6 GB | 265.9 (9.4 + 10.2 + 11.1 + 11.4 + 10.9 + 11.1, 11.1 + 12.3 + 9.5 + 11.2 + 12.0 + 10.3, 10.6 + 12.0 + 11.7 + 12.7 + 12.0 + 10.2, 11.2 + 10.4 + 11.2 + 12.5 + 10.5 + 10.3) |
| Sentence of 25 words | 1 | 1 | 1 | bf16 | Python + FlashInfer + CUDA graphs | 276 | 3.6 | 80% | 14.8 GB | 9.4 (9.4) |
| Sentence of 25 words | 1 | 2 | 2 | bf16 | Python + FlashInfer + CUDA graphs | 490 | 4.1 | 78% | 14.8 GB | 19.4 (9.5 + 9.9) |
| Sentence of 25 words | 1 | 3 | 3 | bf16 | Python + FlashInfer + CUDA graphs | 720 | 4.2 | 82% | 14.8 GB | 30.5 (9.4 + 9.9 + 11.1) |
| Sentence of 25 words | 1 | 4 | 4 | bf16 | Python + FlashInfer + CUDA graphs | 944 | 4.2 | 83% | 14.8 GB | 42.5 (9.7 + 10.1 + 11.2 + 11.4) |
| Sentence of 25 words | 1 | 5 | 5 | bf16 | Python + FlashInfer + CUDA graphs | 1167 | 4.3 | 83% | 14.8 GB | 54.0 (9.6 + 10.2 + 11.1 + 11.5 + 11.6) |
| Sentence of 25 words | 1 | 6 | 6 | bf16 | Python + FlashInfer + CUDA graphs | 1394 | 4.3 | 88% | 14.8 GB | 63.9 (9.4 + 9.7 + 11.2 + 11.5 + 11.1 + 11.0) |
| Sentence of 25 words | 1 | 7 | 7 | bf16 | Python + FlashInfer + CUDA graphs | 1635 | 4.3 | 82% | 14.8 GB | 75.8 (9.4 + 10.3 + 11.2 + 11.5 + 10.9 + 11.0 + 11.3) |
| Sentence of 25 words | 1 | 8 | 8 | bf16 | Python + FlashInfer + CUDA graphs | 2117 | 3.8 | 83% | 15.1 GB | 87.0 (9.4 + 10.2 + 10.7 + 11.5 + 11.1 + 11.0 + 10.9 + 12.2) |
| Sentence of 25 words | 1 | 9 | 9 | bf16 | Python + FlashInfer + CUDA graphs | 2391 | 3.8 | 84% | 15.7 GB | 96.5 (9.4 + 10.2 + 10.8 + 11.5 + 11.2 + 11.1 + 10.9 + 11.9 + 9.4) |
| Sentence of 25 words | 2 | 1 | 2 | bf16 | Python + FlashInfer + CUDA graphs | 516 (495, 516) | 3.9 | 90% | 16.8 GB | 19.6 (9.4, 10.2) |
| Sentence of 25 words | 2 | 2 | 4 | bf16 | Python + FlashInfer + CUDA graphs | 931 (931, 927) | 4.3 | 91% | 16.8 GB | 41.7 (9.5 + 9.9, 10.9 + 11.5) |
| Sentence of 25 words | 2 | 3 | 6 | bf16 | Python + FlashInfer + CUDA graphs | 1367 (1336, 1367) | 4.4 | 91% | 16.8 GB | 64.5 (9.4 + 9.9 + 11.1, 11.3 + 11.6 + 11.1) |
| Sentence of 25 words | 2 | 4 | 8 | bf16 | Python + FlashInfer + CUDA graphs | 1957 (1852, 1957) | 4.1 | 93% | 16.9 GB | 88.0 (9.7 + 10.1 + 11.2 + 11.4, 11.4 + 11.1 + 11.0 + 12.0) |
| Sentence of 25 words | 2 | 5 | 10 | bf16 | Python + FlashInfer + CUDA graphs | 2425 (2255, 2425) | 4.1 | 92% | 17.2 GB | 108.4 (9.6 + 10.2 + 11.1 + 11.5 + 11.6, 10.9 + 10.9 + 12.2 + 9.4 + 10.9) |
| Sentence of 25 words | 2 | 6 | 12 | bf16 | Python + FlashInfer + CUDA graphs | 2906 (2724, 2906) | 4.1 | 93% | 17.7 GB | 130.0 (9.4 + 9.7 + 11.2 + 11.5 + 11.1 + 11.0, 11.0 + 12.2 + 9.4 + 11.2 + 12.0 + 10.2) |
| Sentence of 25 words | 3 | 1 | 3 | bf16 | Python + FlashInfer + CUDA graphs | 723 (721, 723, 708) | 4.2 | 92% | 10.8 GB | 30.6 (9.4, 10.2, 11.0) |
| Sentence of 25 words | 3 | 2 | 6 | bf16 | Python + FlashInfer + CUDA graphs | 1346 (1346, 1337, 1311) | 4.5 | 94% | 11.1 GB | 64.1 (9.5 + 9.9, 10.9 + 11.5, 11.2 + 11.1) |
| Sentence of 25 words | 3 | 3 | 9 | bf16 | Python + FlashInfer + CUDA graphs | 2085 (2005, 2027, 2085) | 4.3 | 96% | 11.5 GB | 97.7 (9.4 + 9.9 + 11.1, 11.3 + 11.6 + 11.1, 11.3 + 12.3 + 9.6) |
| Sentence of 25 words | 3 | 4 | 12 | bf16 | Python + FlashInfer + CUDA graphs | 2822 (2627, 2822, 2681) | 4.3 | 94% | 12.2 GB | 130.1 (9.7 + 10.1 + 11.2 + 11.4, 11.4 + 11.1 + 11.0 + 12.0, 9.5 + 10.9 + 11.6 + 10.2) |
| Sentence of 25 words | 3 | 5 | 15 | bf16 | Python + FlashInfer + CUDA graphs | 3496 (3242, 3496, 3311) | 4.3 | 94% | 13.2 GB | 164.7 (9.6 + 10.2 + 11.1 + 11.5 + 11.6, 10.9 + 10.9 + 12.2 + 9.4 + 10.9, 11.7 + 10.2 + 10.6 + 12.0 + 11.9) |
| Sentence of 25 words | 1 | 1 | 1 | f16 | C++ (`perf_mode=flash_attention`) | 194 | 5.2 | 89% | 2.4 GB | 9.7 (9.7) |
| Sentence of 25 words | 1 | 2 | 2 | f16 | C++ (`perf_mode=flash_attention`) | 381 | 5.2 | 88% | 2.5 GB | 20.4 (9.7 + 10.7) |
| Sentence of 25 words | 1 | 3 | 3 | f16 | C++ (`perf_mode=flash_attention`) | 623 | 4.8 | 94% | 2.5 GB | 31.6 (9.7 + 10.7 + 11.2) |
| Sentence of 25 words | 1 | 4 | 4 | f16 | C++ (`perf_mode=flash_attention`) | 888 | 4.5 | 91% | 2.5 GB | 43.1 (9.7 + 10.7 + 11.2 + 11.5) |
| Sentence of 25 words | 1 | 5 | 5 | f16 | C++ (`perf_mode=flash_attention`) | 1183 | 4.2 | 89% | 2.6 GB | 54.7 (9.7 + 10.7 + 11.2 + 11.5 + 11.6) |
| Sentence of 25 words | 2 | 1 | 2 | f16 | C++ (`perf_mode=flash_attention`) | 408 (405, 408) | 4.9 | 95% | 4.8 GB | 20.4 (9.7, 10.7) |
| Sentence of 25 words | 2 | 2 | 4 | f16 | C++ (`perf_mode=flash_attention`) | 812 (802, 812) | 4.9 | 95% | 4.9 GB | 43.1 (9.7 + 10.7, 11.2 + 11.5) |
| Sentence of 25 words | 2 | 3 | 6 | f16 | C++ (`perf_mode=flash_attention`) | 1328 (1311, 1328) | 4.5 | 95% | 5.0 GB | 65.8 (9.7 + 10.7 + 11.2, 11.5 + 11.6 + 11.1) |
| Sentence of 25 words | 2 | 4 | 8 | f16 | C++ (`perf_mode=flash_attention`) | 1906 (1897, 1906) | 4.2 | 95% | 5.1 GB | 89.2 (9.7 + 10.7 + 11.2 + 11.5, 11.6 + 10.9 + 11.3 + 12.3) |
| Sentence of 25 words | 2 | 5 | 10 | f16 | C++ (`perf_mode=flash_attention`) | 2452 (2438, 2452) | 4.1 | 96% | 5.2 GB | 110.0 (9.7 + 10.7 + 11.2 + 11.5 + 11.6, 10.9 + 11.3 + 12.3 + 9.6 + 11.2) |
| Sentence of 25 words | 1 | 1 | 1 | bf16 | C++ (`perf_mode=flash_attention`) | 206 | 4.9 | 86% | 2.4 GB | 9.7 (9.7) |
| Sentence of 25 words | 1 | 2 | 2 | bf16 | C++ (`perf_mode=flash_attention`) | 425 | 4.7 | 84% | 2.5 GB | 20.3 (9.7 + 10.7) |
| Sentence of 25 words | 1 | 3 | 3 | bf16 | C++ (`perf_mode=flash_attention`) | 728 | 4.1 | 89% | 2.5 GB | 31.6 (9.7 + 10.7 + 11.2) |
| Sentence of 25 words | 1 | 4 | 4 | bf16 | C++ (`perf_mode=flash_attention`) | 922 | 4.3 | 91% | 2.6 GB | 43.0 (9.7 + 10.7 + 11.2 + 11.5) |
| Sentence of 25 words | 2 | 1 | 2 | bf16 | C++ (`perf_mode=flash_attention`) | 420 (420, 414) | 4.8 | 93% | 4.8 GB | 20.4 (9.7, 10.7) |
| Sentence of 25 words | 2 | 2 | 4 | bf16 | C++ (`perf_mode=flash_attention`) | 903 (872, 903) | 4.4 | 96% | 5.0 GB | 43.0 (9.7 + 10.7, 11.2 + 11.5) |
| Sentence of 25 words | 2 | 3 | 6 | bf16 | C++ (`perf_mode=flash_attention`) | 1434 (1417, 1434) | 4.2 | 96% | 5.0 GB | 65.6 (9.7 + 10.7 + 11.2, 11.5 + 11.6 + 10.9) |
| Sentence of 25 words | 2 | 4 | 8 | bf16 | C++ (`perf_mode=flash_attention`) | 1970 (1956, 1970) | 4.1 | 96% | 5.1 GB | 88.6 (9.7 + 10.7 + 11.2 + 11.5, 11.6 + 10.9 + 11.0 + 12.0) |
| Sentence of 25 words | 1 | 1 | 1 | f16 | C++ (`perf_mode=off`) | 284 | 3.5 | 90% | 2.4 GB | 9.7 (9.7) |
| Sentence of 25 words | 1 | 2 | 2 | f16 | C++ (`perf_mode=off`) | 576 | 3.5 | 91% | 2.5 GB | 20.4 (9.7 + 10.7) |
| Sentence of 25 words | 1 | 3 | 3 | f16 | C++ (`perf_mode=off`) | 917 | 3.3 | 92% | 2.5 GB | 31.6 (9.7 + 10.7 + 11.2) |
| Sentence of 25 words | 1 | 4 | 4 | f16 | C++ (`perf_mode=off`) | 1282 | 3.1 | 92% | 2.6 GB | 43.1 (9.7 + 10.7 + 11.2 + 11.5) |
| Sentence of 25 words | 2 | 1 | 2 | f16 | C++ (`perf_mode=off`) | 614 (609, 614) | 3.3 | 96% | 4.8 GB | 20.4 (9.7, 10.7) |
| Sentence of 25 words | 2 | 2 | 4 | f16 | C++ (`perf_mode=off`) | 1242 (1232, 1242) | 3.2 | 96% | 5.0 GB | 43.0 (9.7 + 10.7, 11.2 + 11.4) |
| Sentence of 25 words | 2 | 3 | 6 | f16 | C++ (`perf_mode=off`) | 1991 (1975, 1991) | 3.0 | 96% | 5.0 GB | 65.6 (9.7 + 10.7 + 11.2, 11.5 + 11.6 + 10.9) |
| Sentence of 25 words | 2 | 4 | 8 | f16 | C++ (`perf_mode=off`) | 2773 (2754, 2773) | 2.9 | 96% | 5.1 GB | 89.2 (9.7 + 10.7 + 11.2 + 11.5, 11.6 + 11.0 + 11.3 + 12.3) |
| Sentence of 25 words | 1 | 1 | 1 | bf16 | C++ (`perf_mode=off`) | 294 | 3.4 | 94% | 2.4 GB | 9.7 (9.7) |
| Sentence of 25 words | 1 | 2 | 2 | bf16 | C++ (`perf_mode=off`) | 602 | 3.3 | 91% | 2.5 GB | 20.4 (9.7 + 10.7) |
| Sentence of 25 words | 1 | 3 | 3 | bf16 | C++ (`perf_mode=off`) | 964 | 3.1 | 92% | 2.5 GB | 31.6 (9.7 + 10.7 + 11.2) |
| Sentence of 25 words | 1 | 4 | 4 | bf16 | C++ (`perf_mode=off`) | 1313 | 3.0 | 92% | 2.6 GB | 43.0 (9.7 + 10.7 + 11.2 + 11.5) |
| Sentence of 25 words | 2 | 1 | 2 | bf16 | C++ (`perf_mode=off`) | 631 (625, 631) | 3.2 | 96% | 4.8 GB | 20.3 (9.7, 10.7) |
| Sentence of 25 words | 2 | 2 | 4 | bf16 | C++ (`perf_mode=off`) | 1337 (1326, 1337) | 3.0 | 96% | 5.0 GB | 43.1 (9.7 + 10.7, 11.2 + 11.5) |
| Sentence of 25 words | 2 | 3 | 6 | bf16 | C++ (`perf_mode=off`) | 2087 (2069, 2087) | 2.9 | 97% | 5.0 GB | 65.6 (9.7 + 10.7 + 11.2, 11.5 + 11.6 + 10.9) |
| Sentence of 25 words | 2 | 4 | 8 | bf16 | C++ (`perf_mode=off`) | 2826 (2807, 2826) | 2.8 | 96% | 5.2 GB | 89.1 (9.7 + 10.7 + 11.2 + 11.5, 11.6 + 10.9 + 11.3 + 12.3) |
| Sentence of 25 words | 1 | 1 | 1 | f16 | Python (PyTorch) | 1001 | 1.0 | 32% | 6.5 GB | 9.5 (9.5) |
| Sentence of 25 words | 1 | 2 | 2 | f16 | Python (PyTorch) | 1050 | 1.9 | 49% | 6.5 GB | 19.8 (9.5 + 10.3) |
| Sentence of 25 words | 1 | 3 | 3 | f16 | Python (PyTorch) | 1100 | 2.7 | 75% | 6.5 GB | 31.0 (9.7 + 10.4 + 10.9) |
| Sentence of 25 words | 1 | 4 | 4 | f16 | Python (PyTorch) | 1289 | 3.1 | 89% | 6.5 GB | 42.2 (9.4 + 10.4 + 10.9 + 11.5) |
| Sentence of 25 words | 1 | 5 | 5 | f16 | Python (PyTorch) | 1663 | 3.0 | 88% | 6.5 GB | 53.6 (9.4 + 10.4 + 11.2 + 11.5 + 11.1) |
| Sentence of 25 words | 1 | 6 | 6 | f16 | Python (PyTorch) | 2031 | 3.0 | 90% | 6.5 GB | 64.5 (9.7 + 10.2 + 10.9 + 11.5 + 11.3 + 10.9) |
| Sentence of 25 words | 1 | 7 | 7 | f16 | Python (PyTorch) | 2395 | 2.9 | 89% | 6.5 GB | 76.1 (9.7 + 10.3 + 10.9 + 11.5 + 11.6 + 11.1 + 11.0) |
| Sentence of 25 words | 2 | 1 | 2 | f16 | Python (PyTorch) | 1110 (1036, 1110) | 1.8 | 93% | 10.1 GB | 19.5 (9.5, 10.0) |
| Sentence of 25 words | 2 | 2 | 4 | f16 | Python (PyTorch) | 1263 (1211, 1263) | 3.2 | 96% | 10.1 GB | 42.5 (9.5 + 10.3, 11.2 + 11.5) |
| Sentence of 25 words | 2 | 3 | 6 | f16 | Python (PyTorch) | 1900 (1886, 1900) | 3.2 | 95% | 10.1 GB | 65.0 (9.7 + 10.4 + 10.9, 11.3 + 11.6 + 11.1) |
| Sentence of 25 words | 2 | 4 | 8 | f16 | Python (PyTorch) | 2671 (2632, 2671) | 3.0 | 95% | 10.1 GB | 88.0 (9.4 + 10.4 + 10.9 + 11.5, 11.4 + 10.9 + 11.3 + 12.2) |
| Sentence of 25 words | 2 | 5 | 10 | f16 | Python (PyTorch) | 3476 (3404, 3476) | 2.9 | 95% | 10.1 GB | 108.5 (9.4 + 10.4 + 11.2 + 11.5 + 11.1, 11.0 + 11.0 + 12.3 + 9.4 + 11.2) |
| Sentence of 25 words | 3 | 1 | 3 | f16 | Python (PyTorch) | 1518 (1389, 1430, 1518) | 2.0 | 95% | 15.0 GB | 30.5 (9.5, 10.0, 11.0) |
| Sentence of 25 words | 3 | 2 | 6 | f16 | Python (PyTorch) | 1886 (1816, 1886, 1874) | 3.2 | 96% | 15.0 GB | 65.2 (9.5 + 10.3, 11.2 + 11.5, 11.6 + 11.1) |
| Sentence of 25 words | 3 | 3 | 9 | f16 | Python (PyTorch) | 2850 (2822, 2850, 2837) | 3.2 | 98% | 15.0 GB | 97.8 (9.7 + 10.4 + 10.9, 11.3 + 11.6 + 11.1, 11.3 + 12.1 + 9.4) |
| Sentence of 25 words | 3 | 4 | 12 | f16 | Python (PyTorch) | 3917 (3917, 3901, 3902) | 3.1 | 97% | 15.0 GB | 130.9 (9.4 + 10.4 + 10.9 + 11.5, 11.4 + 10.9 + 11.3 + 12.2, 9.5 + 10.9 + 12.0 + 10.4) |
| Sentence of 25 words | 3 | 5 | 15 | f16 | Python (PyTorch) | 5179 (5036, 5179, 5086) | 2.9 | 97% | 15.0 GB | 165.1 (9.4 + 10.4 + 11.2 + 11.5 + 11.1, 11.0 + 11.0 + 12.3 + 9.4 + 11.2, 12.0 + 10.4 + 10.6 + 11.8 + 11.8) |
| Sentence of 25 words | 4 | 1 | 4 | f16 | Python (PyTorch) | 1400 (1331, 1371, 1400, 1395) | 2.9 | 87% | 21.3 GB | 41.8 (9.5, 10.0, 11.0, 11.3) |
| Sentence of 25 words | 4 | 2 | 8 | f16 | Python (PyTorch) | 2509 (2353, 2494, 2492, 2509) | 3.2 | 96% | 21.3 GB | 88.6 (9.5 + 10.3, 11.2 + 11.5, 11.6 + 11.1, 11.3 + 12.1) |
| Sentence of 25 words | 4 | 3 | 12 | f16 | Python (PyTorch) | 3736 (3692, 3717, 3736, 3706) | 3.2 | 97% | 21.3 GB | 131.2 (9.7 + 10.4 + 10.9, 11.3 + 11.6 + 11.1, 11.3 + 12.1 + 9.4, 11.2 + 11.9 + 10.4) |
| Sentence of 25 words | 4 | 4 | 16 | f16 | Python (PyTorch) | 5273 (5148, 5215, 5182, 5273) | 3.0 | 96% | 21.3 GB | 178.3 (9.4 + 10.4 + 10.9 + 11.5, 11.4 + 10.9 + 11.3 + 12.2, 9.5 + 10.9 + 12.0 + 10.4, 10.6 + 11.6 + 12.2 + 13.0) |
| Sentence of 25 words | 4 | 5 | 20 | f16 | Python (PyTorch) | 6872 (6639, 6813, 6839, 6872) | 2.9 | 98% | 21.3 GB | 220.8 (9.4 + 10.4 + 11.2 + 11.5 + 11.1, 11.0 + 11.0 + 12.3 + 9.4 + 11.2, 12.0 + 10.4 + 10.6 + 11.8 + 11.8, 12.2 + 11.9 + 9.9 + 11.4 + 10.3) |
| Sentence of 25 words | 4 | 6 | 24 | f16 | Python (PyTorch) | 8530 (8240, 8390, 8530, 8375) | 2.8 | 98% | 23.3 GB | 266.1 (9.7 + 10.2 + 10.9 + 11.5 + 11.3 + 10.9, 11.0 + 12.1 + 9.4 + 10.9 + 12.0 + 10.4, 10.6 + 11.7 + 12.2 + 12.8 + 12.0 + 10.2, 11.4 + 10.5 + 11.2 + 12.2 + 10.1 + 10.8) |
| Sentence of 25 words | 1 | 1 | 1 | bf16 | Python (PyTorch) | 966 | 1.0 | 34% | 7.5 GB | 9.4 (9.4) |
| Sentence of 25 words | 1 | 2 | 2 | bf16 | Python (PyTorch) | 1008 | 2.0 | 51% | 7.5 GB | 20.0 (9.7 + 10.3) |
| Sentence of 25 words | 1 | 3 | 3 | bf16 | Python (PyTorch) | 1039 | 2.9 | 80% | 7.5 GB | 31.1 (9.4 + 10.4 + 11.2) |
| Sentence of 25 words | 1 | 4 | 4 | bf16 | Python (PyTorch) | 1277 | 3.1 | 88% | 7.5 GB | 42.2 (9.7 + 10.5 + 10.9 + 11.2) |
| Sentence of 25 words | 1 | 5 | 5 | bf16 | Python (PyTorch) | 1637 | 3.1 | 87% | 7.5 GB | 53.6 (9.5 + 10.4 + 11.2 + 11.4 + 11.1) |
| Sentence of 25 words | 1 | 6 | 6 | bf16 | Python (PyTorch) | 2021 | 3.0 | 88% | 7.5 GB | 64.0 (9.5 + 10.2 + 10.8 + 11.5 + 11.2 + 10.9) |
| Sentence of 25 words | 1 | 7 | 7 | bf16 | Python (PyTorch) | 2387 | 2.9 | 89% | 7.5 GB | 75.7 (9.7 + 10.2 + 11.2 + 11.5 + 10.9 + 11.1 + 11.0) |
| Sentence of 25 words | 2 | 1 | 2 | bf16 | Python (PyTorch) | 1112 (1028, 1112) | 1.8 | 88% | 10.1 GB | 19.8 (9.4, 10.4) |
| Sentence of 25 words | 2 | 2 | 4 | bf16 | Python (PyTorch) | 1247 (1187, 1247) | 3.2 | 93% | 10.1 GB | 42.7 (9.7 + 10.3, 11.2 + 11.5) |
| Sentence of 25 words | 2 | 3 | 6 | bf16 | Python (PyTorch) | 1895 (1895, 1883) | 3.2 | 94% | 10.1 GB | 65.3 (9.4 + 10.4 + 11.2, 11.5 + 11.6 + 11.1) |
| Sentence of 25 words | 2 | 4 | 8 | bf16 | Python (PyTorch) | 2653 (2560, 2653) | 3.0 | 94% | 10.1 GB | 88.1 (9.7 + 10.5 + 10.9 + 11.2, 11.5 + 10.9 + 11.3 + 12.2) |
| Sentence of 25 words | 2 | 5 | 10 | bf16 | Python (PyTorch) | 3411 (3327, 3411) | 2.9 | 95% | 10.1 GB | 108.9 (9.5 + 10.4 + 11.2 + 11.4 + 11.1, 11.1 + 11.1 + 12.3 + 9.6 + 11.2) |
| Sentence of 25 words | 3 | 1 | 3 | bf16 | Python (PyTorch) | 1461 (1269, 1461, 1312) | 2.1 | 93% | 15.0 GB | 30.9 (9.4, 10.4, 11.1) |
| Sentence of 25 words | 3 | 2 | 6 | bf16 | Python (PyTorch) | 1878 (1771, 1878, 1875) | 3.2 | 94% | 15.0 GB | 65.3 (9.7 + 10.3, 11.2 + 11.5, 11.5 + 11.1) |
| Sentence of 25 words | 3 | 3 | 9 | bf16 | Python (PyTorch) | 2831 (2831, 2827, 2794) | 3.2 | 96% | 15.0 GB | 98.2 (9.4 + 10.4 + 11.2, 11.5 + 11.6 + 11.1, 11.3 + 12.3 + 9.4) |
| Sentence of 25 words | 3 | 4 | 12 | bf16 | Python (PyTorch) | 3898 (3803, 3898, 3842) | 3.1 | 97% | 15.0 GB | 131.3 (9.7 + 10.5 + 10.9 + 11.2, 11.5 + 10.9 + 11.3 + 12.2, 9.5 + 11.2 + 12.0 + 10.4) |
| Sentence of 25 words | 3 | 5 | 15 | bf16 | Python (PyTorch) | 5069 (4996, 5069, 5059) | 3.0 | 97% | 15.0 GB | 165.4 (9.5 + 10.4 + 11.2 + 11.4 + 11.1, 11.1 + 11.1 + 12.3 + 9.6 + 11.2, 12.0 + 10.4 + 10.6 + 11.6 + 11.8) |
| Sentence of 25 words | 4 | 1 | 4 | bf16 | Python (PyTorch) | 1470 (1301, 1316, 1470, 1339) | 2.7 | 94% | 20.0 GB | 42.2 (9.4, 10.4, 11.1, 11.3) |
| Sentence of 25 words | 4 | 2 | 8 | bf16 | Python (PyTorch) | 2496 (2341, 2496, 2469, 2479) | 3.2 | 96% | 20.0 GB | 88.7 (9.7 + 10.3, 11.2 + 11.5, 11.5 + 11.1, 11.3 + 12.1) |
| Sentence of 25 words | 4 | 3 | 12 | bf16 | Python (PyTorch) | 3738 (3635, 3721, 3718, 3738) | 3.2 | 98% | 20.0 GB | 131.9 (9.4 + 10.4 + 11.2, 11.5 + 11.6 + 11.1, 11.3 + 12.3 + 9.4, 11.2 + 12.0 + 10.4) |
| Sentence of 25 words | 4 | 4 | 16 | bf16 | Python (PyTorch) | 5251 (5051, 5192, 5116, 5251) | 3.0 | 96% | 20.0 GB | 179.0 (9.7 + 10.5 + 10.9 + 11.2, 11.5 + 10.9 + 11.3 + 12.2, 9.5 + 11.2 + 12.0 + 10.4, 10.6 + 11.9 + 12.2 + 13.0) |
| Sentence of 25 words | 4 | 5 | 20 | bf16 | Python (PyTorch) | 6814 (6611, 6728, 6749, 6814) | 2.9 | 97% | 20.5 GB | 221.6 (9.5 + 10.4 + 11.2 + 11.4 + 11.1, 11.1 + 11.1 + 12.3 + 9.6 + 11.2, 12.0 + 10.4 + 10.6 + 11.6 + 11.8, 12.2 + 12.0 + 10.2 + 11.1 + 10.7) |
| Sentence of 25 words | 4 | 6 | 24 | bf16 | Python (PyTorch) | 8457 (8180, 8307, 8457, 8364) | 2.8 | 98% | 22.5 GB | 265.3 (9.5 + 10.2 + 10.8 + 11.5 + 11.2 + 10.9, 11.0 + 12.1 + 9.4 + 11.2 + 11.8 + 10.2, 10.6 + 11.7 + 12.2 + 12.8 + 12.0 + 9.9, 11.4 + 10.5 + 11.4 + 12.2 + 10.0 + 10.9) |
| Sentence of 30 words | 1 | 1 | 1 | f16 | Python + FlashInfer | 513 | 1.9 | 37% | 4.7 GB | 14.3 (14.3) |
| Sentence of 30 words | 1 | 2 | 2 | f16 | Python + FlashInfer | 573 | 3.5 | 57% | 4.7 GB | 28.1 (14.5 + 13.6) |
| Sentence of 30 words | 1 | 3 | 3 | f16 | Python + FlashInfer | 622 | 4.8 | 79% | 4.7 GB | 40.7 (14.5 + 13.4 + 12.8) |
| Sentence of 30 words | 1 | 4 | 4 | f16 | Python + FlashInfer | 808 | 5.0 | 78% | 4.7 GB | 53.1 (14.3 + 13.4 + 12.8 + 12.6) |
| Sentence of 30 words | 1 | 5 | 5 | f16 | Python + FlashInfer | 978 | 5.1 | 80% | 4.7 GB | 67.0 (14.3 + 13.6 + 12.8 + 13.0 + 13.3) |
| Sentence of 30 words | 1 | 6 | 6 | f16 | Python + FlashInfer | 1202 | 5.0 | 77% | 4.7 GB | 80.1 (14.3 + 13.4 + 12.8 + 13.0 + 13.4 + 13.3) |
| Sentence of 30 words | 1 | 7 | 7 | f16 | Python + FlashInfer | 1371 | 5.1 | 76% | 4.7 GB | 93.1 (14.4 + 13.3 + 12.8 + 12.9 + 13.4 + 13.3 + 13.0) |
| Sentence of 30 words | 1 | 8 | 8 | f16 | Python + FlashInfer | 1572 | 5.1 | 76% | 4.7 GB | 106.5 (14.3 + 13.3 + 13.1 + 13.0 + 13.4 + 13.5 + 13.0 + 12.9) |
| Sentence of 30 words | 2 | 1 | 2 | f16 | Python + FlashInfer | 683 (683, 602) | 2.9 | 85% | 11.5 GB | 27.8 (14.3, 13.5) |
| Sentence of 30 words | 2 | 2 | 4 | f16 | Python + FlashInfer | 953 (953, 750) | 4.2 | 85% | 11.5 GB | 54.0 (14.5 + 13.6, 13.1 + 12.8) |
| Sentence of 30 words | 2 | 3 | 6 | f16 | Python + FlashInfer | 1180 (1180, 1171) | 5.1 | 87% | 11.5 GB | 79.8 (14.5 + 13.4 + 12.8, 13.0 + 13.2 + 12.9) |
| Sentence of 30 words | 2 | 4 | 8 | f16 | Python + FlashInfer | 1575 (1575, 1519) | 5.1 | 88% | 11.5 GB | 105.5 (14.3 + 13.4 + 12.8 + 12.6, 12.9 + 13.5 + 12.9 + 13.1) |
| Sentence of 30 words | 2 | 5 | 10 | f16 | Python + FlashInfer | 1891 (1885, 1891) | 5.3 | 89% | 11.5 GB | 133.4 (14.3 + 13.6 + 12.8 + 13.0 + 13.3, 13.1 + 13.1 + 12.9 + 13.8 + 13.5) |
| Sentence of 30 words | 2 | 6 | 12 | f16 | Python + FlashInfer | 2214 (2214, 2201) | 5.4 | 90% | 11.5 GB | 158.8 (14.3 + 13.4 + 12.8 + 13.0 + 13.4 + 13.3, 12.9 + 12.8 + 13.6 + 13.5 + 13.2 + 12.8) |
| Sentence of 30 words | 2 | 7 | 14 | f16 | Python + FlashInfer | 2627 (2572, 2627) | 5.3 | 88% | 11.5 GB | 186.7 (14.4 + 13.3 + 12.8 + 12.9 + 13.4 + 13.3 + 13.0, 12.8 + 14.0 + 13.0 + 13.0 + 12.8 + 13.0 + 15.0) |
| Sentence of 30 words | 2 | 8 | 16 | f16 | Python + FlashInfer | 2984 (2977, 2984) | 5.4 | 90% | 11.5 GB | 213.4 (14.3 + 13.3 + 13.1 + 13.0 + 13.4 + 13.5 + 13.0 + 12.9, 13.6 + 13.5 + 13.2 + 13.1 + 12.7 + 14.8 + 13.8 + 12.3) |
| Sentence of 30 words | 2 | 9 | 18 | f16 | Python + FlashInfer | 3368 (3281, 3368) | 5.3 | 90% | 11.5 GB | 239.4 (14.3 + 13.3 + 13.0 + 12.6 + 13.3 + 13.2 + 12.9 + 13.1 + 13.7, 13.2 + 13.2 + 13.0 + 13.0 + 14.9 + 13.3 + 12.6 + 13.6 + 13.2) |
| Sentence of 30 words | 3 | 1 | 3 | f16 | Python + FlashInfer | 886 (630, 886, 575) | 3.4 | 75% | 21.3 GB | 40.7 (14.3, 13.5, 12.9) |
| Sentence of 30 words | 3 | 2 | 6 | f16 | Python + FlashInfer | 1149 (1149, 1091, 1114) | 5.2 | 92% | 21.3 GB | 80.5 (14.5 + 13.6, 13.1 + 12.8, 13.0 + 13.5) |
| Sentence of 30 words | 3 | 3 | 9 | f16 | Python + FlashInfer | 1674 (1673, 1674, 1626) | 5.4 | 90% | 10.7 GB | 119.3 (14.5 + 13.4 + 12.8, 13.0 + 13.2 + 12.9, 13.0 + 12.8 + 13.7) |
| Sentence of 30 words | 3 | 4 | 12 | f16 | Python + FlashInfer | 2207 (2207, 2188, 2141) | 5.4 | 94% | 11.2 GB | 158.5 (14.3 + 13.4 + 12.8 + 12.6, 12.9 + 13.5 + 12.9 + 13.1, 13.7 + 13.2 + 13.1 + 13.1) |
| Sentence of 30 words | 3 | 5 | 15 | f16 | Python + FlashInfer | 2665 (2638, 2635, 2665) | 5.6 | 94% | 11.9 GB | 200.5 (14.3 + 13.6 + 12.8 + 13.0 + 13.3, 13.1 + 13.1 + 12.9 + 13.8 + 13.5, 12.9 + 12.9 + 13.0 + 14.8 + 13.4) |
| Sentence of 30 words | 3 | 6 | 18 | f16 | Python + FlashInfer | 3208 (3193, 3161, 3208) | 5.6 | 91% | 12.6 GB | 239.6 (14.3 + 13.4 + 12.8 + 13.0 + 13.4 + 13.3, 12.9 + 12.8 + 13.6 + 13.5 + 13.2 + 12.8, 12.9 + 14.9 + 13.5 + 12.6 + 13.8 + 13.1) |
| Sentence of 30 words | 3 | 7 | 21 | f16 | Python + FlashInfer | 3674 (3617, 3674, 3641) | 5.7 | 94% | 13.5 GB | 278.8 (14.4 + 13.3 + 12.8 + 12.9 + 13.4 + 13.3 + 13.0, 12.8 + 14.0 + 13.0 + 13.0 + 12.8 + 13.0 + 15.0, 13.5 + 12.6 + 14.0 + 12.8 + 11.8 + 14.8 + 12.6) |
| Sentence of 30 words | 3 | 8 | 24 | f16 | Python + FlashInfer | 4334 (4334, 4240, 4278) | 5.5 | 92% | 14.5 GB | 323.9 (14.3 + 13.3 + 13.1 + 13.0 + 13.4 + 13.5 + 13.0 + 12.9, 13.6 + 13.5 + 13.2 + 13.1 + 12.7 + 14.8 + 13.8 + 12.3, 13.8 + 13.1 + 11.8 + 15.1 + 13.0 + 13.0 + 15.7 + 15.0) |
| Sentence of 30 words | 3 | 9 | 27 | f16 | Python + FlashInfer | 4826 (4799, 4813, 4826) | 5.6 | 93% | 15.6 GB | 365.1 (14.3 + 13.3 + 13.0 + 12.6 + 13.3 + 13.2 + 12.9 + 13.1 + 13.7, 13.2 + 13.2 + 13.0 + 13.0 + 14.9 + 13.3 + 12.6 + 13.6 + 13.2, 11.5 + 14.8 + 13.0 + 13.3 + 15.7 + 15.1 + 14.4 + 12.7 + 15.2) |
| Sentence of 30 words | 3 | 10 | 30 | f16 | Python + FlashInfer | 5408 (5345, 5341, 5408) | 5.5 | 94% | 16.8 GB | 408.0 (14.4 + 13.6 + 12.8 + 12.8 + 13.4 + 13.3 + 12.8 + 13.1 + 13.9 + 13.2, 12.9 + 13.0 + 13.0 + 15.2 + 13.8 + 12.3 + 14.1 + 13.4 + 11.8 + 14.7, 12.7 + 13.2 + 15.3 + 14.9 + 14.8 + 12.8 + 15.3 + 14.2 + 14.1 + 13.1) |
| Sentence of 30 words | 4 | 1 | 4 | f16 | Python + FlashInfer | 855 (855, 841, 821, 786) | 4.7 | 90% | 21.0 GB | 53.6 (14.3, 13.5, 12.9, 13.0) |
| Sentence of 30 words | 4 | 2 | 8 | f16 | Python + FlashInfer | 1475 (1475, 1411, 1443, 1441) | 5.4 | 94% | 21.0 GB | 106.2 (14.5 + 13.6, 13.1 + 12.8, 13.0 + 13.5, 12.8 + 12.9) |
| Sentence of 30 words | 4 | 3 | 12 | f16 | Python + FlashInfer | 2194 (2179, 2194, 2158, 2136) | 5.5 | 95% | 21.0 GB | 158.8 (14.5 + 13.4 + 12.8, 13.0 + 13.2 + 12.9, 13.0 + 12.8 + 13.7, 13.2 + 13.2 + 13.1) |
| Sentence of 30 words | 4 | 4 | 16 | f16 | Python + FlashInfer | 2851 (2851, 2815, 2814, 2834) | 5.6 | 96% | 21.0 GB | 212.2 (14.3 + 13.4 + 12.8 + 12.6, 12.9 + 13.5 + 12.9 + 13.1, 13.7 + 13.2 + 13.1 + 13.1, 12.6 + 14.9 + 13.5 + 12.6) |
| Sentence of 30 words | 4 | 5 | 20 | f16 | Python + FlashInfer | 3513 (3513, 3507, 3510, 3496) | 5.7 | 96% | 21.0 GB | 266.3 (14.3 + 13.6 + 12.8 + 13.0 + 13.3, 13.1 + 13.1 + 12.9 + 13.8 + 13.5, 12.9 + 12.9 + 13.0 + 14.8 + 13.4, 12.2 + 13.9 + 12.8 + 11.8 + 15.1) |
| Sentence of 30 words | 4 | 6 | 24 | f16 | Python + FlashInfer | 4237 (4198, 4128, 4151, 4237) | 5.7 | 96% | 21.0 GB | 322.8 (14.3 + 13.4 + 12.8 + 13.0 + 13.4 + 13.3, 12.9 + 12.8 + 13.6 + 13.5 + 13.2 + 12.8, 12.9 + 14.9 + 13.5 + 12.6 + 13.8 + 13.1, 11.5 + 15.1 + 12.6 + 13.4 + 15.4 + 15.1) |
| Sentence of 30 words | 4 | 7 | 28 | f16 | Python + FlashInfer | 4917 (4817, 4870, 4797, 4917) | 5.7 | 96% | 21.0 GB | 379.0 (14.4 + 13.3 + 12.8 + 12.9 + 13.4 + 13.3 + 13.0, 12.8 + 14.0 + 13.0 + 13.0 + 12.8 + 13.0 + 15.0, 13.5 + 12.6 + 14.0 + 12.8 + 11.8 + 14.8 + 12.6, 12.9 + 15.4 + 15.1 + 14.8 + 12.8 + 15.0 + 14.2) |
| Sentence of 30 words | 4 | 8 | 32 | f16 | Python + FlashInfer | 5610 (5491, 5470, 5596, 5610) | 5.7 | 97% | 21.0 GB | 434.4 (14.3 + 13.3 + 13.1 + 13.0 + 13.4 + 13.5 + 13.0 + 12.9, 13.6 + 13.5 + 13.2 + 13.1 + 12.7 + 14.8 + 13.8 + 12.3, 13.8 + 13.1 + 11.8 + 15.1 + 13.0 + 13.0 + 15.7 + 15.0, 14.8 + 12.8 + 15.3 + 14.2 + 14.2 + 13.3 + 12.8 + 13.0) |
| Sentence of 30 words | 4 | 9 | 36 | f16 | Python + FlashInfer | 6352 (6262, 6196, 6352, 6291) | 5.7 | 95% | 21.0 GB | 485.4 (14.3 + 13.3 + 13.0 + 12.6 + 13.3 + 13.2 + 12.9 + 13.1 + 13.7, 13.2 + 13.2 + 13.0 + 13.0 + 14.9 + 13.3 + 12.6 + 13.6 + 13.2, 11.5 + 14.8 + 13.0 + 13.3 + 15.7 + 15.1 + 14.4 + 12.7 + 15.2, 14.2 + 14.4 + 13.1 + 12.8 + 12.8 + 13.6 + 12.2 + 13.8 + 13.4) |
| Sentence of 30 words | 4 | 10 | 40 | f16 | Python + FlashInfer | 7024 (6945, 6879, 7024, 6986) | 5.7 | 96% | 22.3 GB | 544.3 (14.4 + 13.6 + 12.8 + 12.8 + 13.4 + 13.3 + 12.8 + 13.1 + 13.9 + 13.2, 12.9 + 13.0 + 13.0 + 15.2 + 13.8 + 12.3 + 14.1 + 13.4 + 11.8 + 14.7, 12.7 + 13.2 + 15.3 + 14.9 + 14.8 + 12.8 + 15.3 + 14.2 + 14.1 + 13.1, 12.8 + 12.9 + 13.6 + 12.1 + 13.8 + 13.4 + 12.6 + 15.6 + 14.2 + 15.2) |
| Sentence of 30 words | 4 | 11 | 44 | f16 | Python + FlashInfer | 7867 (7631, 7581, 7867, 7811) | 5.6 | 94% | 20.4 GB | 594.4 (14.5 + 13.3 + 12.7 + 13.0 + 13.2 + 13.3 + 13.1 + 12.9 + 14.0 + 13.4 + 12.9, 12.9 + 13.0 + 14.9 + 13.8 + 12.4 + 13.9 + 12.3 + 11.8 + 14.5 + 12.8 + 12.9, 15.4 + 14.7 + 14.4 + 12.4 + 15.2 + 14.2 + 14.3 + 13.0 + 12.8 + 13.2 + 13.6, 11.8 + 13.8 + 13.2 + 13.1 + 15.7 + 14.4 + 14.9 + 13.0 + 13.8 + 13.4 + 12.6) |
| Sentence of 30 words | 5 | 1 | 5 | f16 | Python + FlashInfer | 1069 (1069, 1047, 1035, 1000, 1049) | 4.7 | 92% | 22.9 GB | 66.6 (14.3, 13.5, 12.9, 13.0, 13.0) |
| Sentence of 30 words | 5 | 2 | 10 | f16 | Python + FlashInfer | 1818 (1818, 1764, 1775, 1782, 1797) | 5.5 | 94% | 22.9 GB | 133.4 (14.5 + 13.6, 13.1 + 12.8, 13.0 + 13.5, 12.8 + 12.9, 13.7 + 13.5) |
| Sentence of 30 words | 5 | 3 | 15 | f16 | Python + FlashInfer | 2691 (2686, 2647, 2653, 2621, 2691) | 5.6 | 94% | 22.9 GB | 200.3 (14.5 + 13.4 + 12.8, 13.0 + 13.2 + 12.9, 13.0 + 12.8 + 13.7, 13.2 + 13.2 + 13.1, 12.9 + 15.2 + 13.5) |
| Sentence of 30 words | 5 | 4 | 20 | f16 | Python + FlashInfer | 3564 (3548, 3517, 3474, 3515, 3564) | 5.6 | 97% | 22.9 GB | 265.3 (14.3 + 13.4 + 12.8 + 12.6, 12.9 + 13.5 + 12.9 + 13.1, 13.7 + 13.2 + 13.1 + 13.1, 12.6 + 14.9 + 13.5 + 12.6, 13.8 + 12.6 + 11.8 + 14.9) |
| Sentence of 30 words | 5 | 5 | 25 | f16 | Python + FlashInfer | 4350 (4312, 4269, 4287, 4281, 4350) | 5.7 | 96% | 22.9 GB | 337.4 (14.3 + 13.6 + 12.8 + 13.0 + 13.3, 13.1 + 13.1 + 12.9 + 13.8 + 13.5, 12.9 + 12.9 + 13.0 + 14.8 + 13.4, 12.2 + 13.9 + 12.8 + 11.8 + 15.1, 12.9 + 13.1 + 15.4 + 15.0 + 14.8) |
| Sentence of 30 words | 5 | 6 | 30 | f16 | Python + FlashInfer | 5210 (5159, 5167, 5174, 5210, 5132) | 5.8 | 97% | 22.9 GB | 406.2 (14.3 + 13.4 + 12.8 + 13.0 + 13.4 + 13.3, 12.9 + 12.8 + 13.6 + 13.5 + 13.2 + 12.8, 12.9 + 14.9 + 13.5 + 12.6 + 13.8 + 13.1, 11.5 + 15.1 + 12.6 + 13.4 + 15.4 + 15.1, 14.5 + 12.8 + 14.8 + 14.2 + 14.2 + 12.9) |
| Sentence of 30 words | 5 | 7 | 35 | f16 | Python + FlashInfer | 6061 (5891, 5957, 5873, 6061, 5875) | 5.8 | 97% | 22.9 GB | 471.5 (14.4 + 13.3 + 12.8 + 12.9 + 13.4 + 13.3 + 13.0, 12.8 + 14.0 + 13.0 + 13.0 + 12.8 + 13.0 + 15.0, 13.5 + 12.6 + 14.0 + 12.8 + 11.8 + 14.8 + 12.6, 12.9 + 15.4 + 15.1 + 14.8 + 12.8 + 15.0 + 14.2, 14.0 + 13.3 + 12.8 + 13.2 + 13.6 + 12.0 + 13.4) |
| Sentence of 30 words | 5 | 8 | 40 | f16 | Python + FlashInfer | 6969 (6885, 6902, 6946, 6965, 6969) | 5.7 | 98% | 22.9 GB | 545.6 (14.3 + 13.3 + 13.1 + 13.0 + 13.4 + 13.5 + 13.0 + 12.9, 13.6 + 13.5 + 13.2 + 13.1 + 12.7 + 14.8 + 13.8 + 12.3, 13.8 + 13.1 + 11.8 + 15.1 + 13.0 + 13.0 + 15.7 + 15.0, 14.8 + 12.8 + 15.3 + 14.2 + 14.2 + 13.3 + 12.8 + 13.0, 13.4 + 12.3 + 13.8 + 13.4 + 13.1 + 15.7 + 14.4 + 15.1) |
| Sentence of 30 words | 5 | 9 | 45 | f16 | Python + FlashInfer | 7805 (7674, 7699, 7786, 7639, 7805) | 5.8 | 98% | 22.9 GB | 607.8 (14.3 + 13.3 + 13.0 + 12.6 + 13.3 + 13.2 + 12.9 + 13.1 + 13.7, 13.2 + 13.2 + 13.0 + 13.0 + 14.9 + 13.3 + 12.6 + 13.6 + 13.2, 11.5 + 14.8 + 13.0 + 13.3 + 15.7 + 15.1 + 14.4 + 12.7 + 15.2, 14.2 + 14.4 + 13.1 + 12.8 + 12.8 + 13.6 + 12.2 + 13.8 + 13.4, 13.0 + 15.6 + 14.4 + 15.1 + 13.0 + 13.3 + 13.6 + 12.4 + 12.1) |
| Sentence of 30 words | 5 | 10 | 50 | f16 | Python + FlashInfer | 8651 (8597, 8576, 8651, 8624, 8407) | 5.8 | 97% | 22.9 GB | 671.6 (14.4 + 13.6 + 12.8 + 12.8 + 13.4 + 13.3 + 12.8 + 13.1 + 13.9 + 13.2, 12.9 + 13.0 + 13.0 + 15.2 + 13.8 + 12.3 + 14.1 + 13.4 + 11.8 + 14.7, 12.7 + 13.2 + 15.3 + 14.9 + 14.8 + 12.8 + 15.3 + 14.2 + 14.1 + 13.1, 12.8 + 12.9 + 13.6 + 12.1 + 13.8 + 13.4 + 12.6 + 15.6 + 14.2 + 15.2, 12.6 + 13.8 + 13.6 + 12.3 + 12.1 + 14.4 + 12.6 + 12.6 + 11.2 + 12.2) |
| Sentence of 30 words | 5 | 11 | 55 | f16 | Python + FlashInfer | 9572 (9464, 9447, 9570, 9572, 9402) | 5.7 | 97% | 19.9 GB | 740.5 (14.5 + 13.3 + 12.7 + 13.0 + 13.2 + 13.3 + 13.1 + 12.9 + 14.0 + 13.4 + 12.9, 12.9 + 13.0 + 14.9 + 13.8 + 12.4 + 13.9 + 12.3 + 11.8 + 14.5 + 12.8 + 12.9, 15.4 + 14.7 + 14.4 + 12.4 + 15.2 + 14.2 + 14.3 + 13.0 + 12.8 + 13.2 + 13.6, 11.8 + 13.8 + 13.2 + 13.1 + 15.7 + 14.4 + 14.9 + 13.0 + 13.8 + 13.4 + 12.6, 11.9 + 14.3 + 12.9 + 12.7 + 11.1 + 12.3 + 13.1 + 14.9 + 14.0 + 12.8 + 16.0) |
| Sentence of 30 words | 5 | 12 | 60 | f16 | Python + FlashInfer | 10327 (10225, 10327, 10281, 10320, 10192) | 5.8 | 98% | 23.2 GB | 808.3 (14.4 + 13.3 + 12.8 + 13.0 + 13.3 + 13.5 + 13.0 + 13.1 + 14.1 + 13.4 + 12.9 + 13.1, 12.9 + 15.2 + 13.5 + 12.6 + 14.1 + 13.3 + 11.8 + 14.8 + 13.0 + 13.3 + 15.7 + 15.1, 14.5 + 12.3 + 14.9 + 14.0 + 14.2 + 12.8 + 12.5 + 13.1 + 13.4 + 12.4 + 13.8 + 13.4, 12.8 + 15.7 + 14.4 + 15.0 + 13.0 + 13.8 + 13.9 + 12.2 + 12.1 + 14.4 + 13.2 + 12.9, 11.1 + 12.3 + 13.4 + 15.4 + 14.3 + 12.3 + 15.8 + 12.5 + 12.7 + 12.8 + 12.4 + 13.3) |
| Sentence of 30 words | 5 | 13 | 65 | f16 | Python + FlashInfer | 11282 (11164, 11282, 11270, 11124, 11256) | 5.8 | 97% | 21.5 GB | 877.7 (14.3 + 13.2 + 13.1 + 12.7 + 13.4 + 13.3 + 13.1 + 13.1 + 13.9 + 13.2 + 13.2 + 13.1 + 13.0, 14.8 + 13.8 + 12.4 + 14.1 + 13.4 + 11.6 + 15.1 + 12.7 + 13.4 + 15.4 + 15.0 + 14.8 + 12.4, 15.1 + 13.8 + 14.0 + 13.1 + 12.8 + 13.1 + 13.6 + 12.0 + 13.5 + 13.2 + 13.1 + 15.8 + 14.2, 14.8 + 13.3 + 13.7 + 13.9 + 12.6 + 11.8 + 14.4 + 13.2 + 13.0 + 11.2 + 12.3 + 13.5 + 15.3, 14.1 + 12.8 + 16.0 + 12.5 + 12.7 + 12.8 + 12.7 + 13.7 + 12.3 + 14.5 + 14.2 + 13.8 + 13.9) |
| Sentence of 30 words | 5 | 14 | 70 | f16 | Python + FlashInfer | 12272 (12024, 12272, 12146, 12127, 12040) | 5.7 | 98% | 20.0 GB | 943.6 (14.6 + 13.6 + 12.8 + 13.0 + 13.4 + 13.3 + 13.1 + 13.1 + 13.9 + 13.2 + 13.2 + 12.9 + 13.0 + 15.2, 13.5 + 12.3 + 14.1 + 13.1 + 11.7 + 14.7 + 13.0 + 13.2 + 15.7 + 15.1 + 14.4 + 12.2 + 15.3 + 13.9, 14.1 + 13.3 + 12.8 + 12.7 + 13.6 + 12.0 + 13.4 + 13.0 + 12.9 + 15.6 + 14.0 + 15.2 + 13.0 + 13.8, 13.7 + 12.4 + 11.5 + 14.2 + 12.8 + 12.9 + 11.2 + 12.3 + 13.2 + 15.4 + 14.4 + 12.5 + 16.0 + 12.5, 12.7 + 12.4 + 12.5 + 13.4 + 12.6 + 14.4 + 14.3 + 14.0 + 13.9 + 13.6 + 12.5 + 14.8 + 14.3 + 13.1) |
| Sentence of 30 words | 5 | 15 | 75 | f16 | Python + FlashInfer | 13108 (12978, 12999, 12992, 13017, 13108) | 5.7 | 98% | 21.9 GB | 1008.2 (14.1 + 13.6 + 12.8 + 12.8 + 13.0 + 13.3 + 12.9 + 13.1 + 13.7 + 13.2 + 12.8 + 12.8 + 12.7 + 15.2 + 13.8, 12.3 + 13.9 + 13.4 + 11.4 + 14.8 + 13.0 + 13.3 + 15.7 + 14.9 + 14.8 + 12.2 + 14.8 + 13.9 + 14.2 + 12.9, 12.8 + 13.2 + 13.4 + 12.0 + 13.6 + 13.4 + 12.9 + 15.7 + 14.4 + 14.6 + 12.7 + 13.8 + 13.9 + 12.4 + 12.1, 14.2 + 13.2 + 12.7 + 11.2 + 12.3 + 13.5 + 15.2 + 14.1 + 12.8 + 16.0 + 12.3 + 13.0 + 12.8 + 12.5 + 13.2, 12.1 + 14.6 + 14.4 + 13.4 + 13.9 + 13.9 + 12.3 + 14.8 + 14.0 + 13.5 + 12.2 + 12.6 + 14.7 + 13.3 + 13.1) |
| Sentence of 30 words | 6 | 1 | 6 | f16 | Python + FlashInfer | 1256 (1256, 1251, 1246, 1164, 1246, 1232) | 4.8 | 92% | 20.9 GB | 80.0 (14.3, 13.5, 12.9, 13.0, 13.0, 13.4) |
| Sentence of 30 words | 6 | 2 | 12 | f16 | Python + FlashInfer | 2187 (2187, 2077, 2137, 2112, 2186, 2119) | 5.5 | 95% | 21.0 GB | 159.1 (14.5 + 13.6, 13.1 + 12.8, 13.0 + 13.5, 12.8 + 12.9, 13.7 + 13.5, 12.9 + 12.9) |
| Sentence of 30 words | 6 | 3 | 18 | f16 | Python + FlashInfer | 3183 (3164, 3141, 3172, 3134, 3183, 3146) | 5.7 | 97% | 21.5 GB | 239.7 (14.5 + 13.4 + 12.8, 13.0 + 13.2 + 12.9, 13.0 + 12.8 + 13.7, 13.2 + 13.2 + 13.1, 12.9 + 15.2 + 13.5, 12.3 + 13.9 + 13.1) |
| Sentence of 30 words | 6 | 4 | 24 | f16 | Python + FlashInfer | 4269 (4217, 4183, 4183, 4210, 4240, 4269) | 5.6 | 100% | 22.7 GB | 321.5 (14.3 + 13.4 + 12.8 + 12.6, 12.9 + 13.5 + 12.9 + 13.1, 13.7 + 13.2 + 13.1 + 13.1, 12.6 + 14.9 + 13.5 + 12.6, 13.8 + 12.6 + 11.8 + 14.9, 12.7 + 13.1 + 15.4 + 15.1) |
| Sentence of 30 words | 6 | 5 | 30 | f16 | Python + FlashInfer | 5247 (5148, 5155, 5147, 5113, 5247, 5177) | 5.7 | 97% | 20.7 GB | 407.2 (14.3 + 13.6 + 12.8 + 13.0 + 13.3, 13.1 + 13.1 + 12.9 + 13.8 + 13.5, 12.9 + 12.9 + 13.0 + 14.8 + 13.4, 12.2 + 13.9 + 12.8 + 11.8 + 15.1, 12.9 + 13.1 + 15.4 + 15.0 + 14.8, 12.5 + 15.3 + 14.2 + 14.4 + 13.3) |
| Sentence of 30 words | 6 | 6 | 36 | f16 | Python + FlashInfer | 6251 (6146, 6131, 6143, 6251, 6209, 6078) | 5.8 | 97% | 22.8 GB | 484.2 (14.3 + 13.4 + 12.8 + 13.0 + 13.4 + 13.3, 12.9 + 12.8 + 13.6 + 13.5 + 13.2 + 12.8, 12.9 + 14.9 + 13.5 + 12.6 + 13.8 + 13.1, 11.5 + 15.1 + 12.6 + 13.4 + 15.4 + 15.1, 14.5 + 12.8 + 14.8 + 14.2 + 14.2 + 12.9, 12.8 + 13.0 + 13.1 + 12.2 + 13.5 + 13.4) |
| Sentence of 30 words | 6 | 7 | 42 | f16 | Python + FlashInfer | 7288 (7071, 7163, 7047, 7278, 7158, 7288) | 5.8 | 97% | 23.0 GB | 568.5 (14.4 + 13.3 + 12.8 + 12.9 + 13.4 + 13.3 + 13.0, 12.8 + 14.0 + 13.0 + 13.0 + 12.8 + 13.0 + 15.0, 13.5 + 12.6 + 14.0 + 12.8 + 11.8 + 14.8 + 12.6, 12.9 + 15.4 + 15.1 + 14.8 + 12.8 + 15.0 + 14.2, 14.0 + 13.3 + 12.8 + 13.2 + 13.6 + 12.0 + 13.4, 13.2 + 13.1 + 15.6 + 14.3 + 14.5 + 12.9 + 13.3) |
| Sentence of 30 words | 6 | 8 | 48 | f16 | Python + FlashInfer | 8279 (8149, 8169, 8270, 8221, 8279, 8112) | 5.8 | 99% | 22.3 GB | 651.3 (14.3 + 13.3 + 13.1 + 13.0 + 13.4 + 13.5 + 13.0 + 12.9, 13.6 + 13.5 + 13.2 + 13.1 + 12.7 + 14.8 + 13.8 + 12.3, 13.8 + 13.1 + 11.8 + 15.1 + 13.0 + 13.0 + 15.7 + 15.0, 14.8 + 12.8 + 15.3 + 14.2 + 14.2 + 13.3 + 12.8 + 13.0, 13.4 + 12.3 + 13.8 + 13.4 + 13.1 + 15.7 + 14.4 + 15.1, 13.0 + 13.8 + 13.9 + 12.6 + 12.1 + 14.4 + 12.9 + 12.9) |
| Sentence of 30 words | 6 | 9 | 54 | f16 | Python + FlashInfer | 9239 (9145, 9123, 9239, 9166, 9216, 9090) | 5.8 | 98% | 22.5 GB | 727.1 (14.3 + 13.3 + 13.0 + 12.6 + 13.3 + 13.2 + 12.9 + 13.1 + 13.7, 13.2 + 13.2 + 13.0 + 13.0 + 14.9 + 13.3 + 12.6 + 13.6 + 13.2, 11.5 + 14.8 + 13.0 + 13.3 + 15.7 + 15.1 + 14.4 + 12.7 + 15.2, 14.2 + 14.4 + 13.1 + 12.8 + 12.8 + 13.6 + 12.2 + 13.8 + 13.4, 13.0 + 15.6 + 14.4 + 15.1 + 13.0 + 13.3 + 13.6 + 12.4 + 12.1, 14.3 + 13.1 + 12.7 + 11.0 + 12.3 + 13.5 + 15.4 + 14.3 + 12.7) |
| Sentence of 30 words | 6 | 10 | 60 | f16 | Python + FlashInfer | failed: instance 1 exited (rc 1): CUDA out of memory | – | – | – | – |
| Sentence of 30 words | 1 | 1 | 1 | bf16 | Python + FlashInfer | failed: bf16 not supported: upstream `apply_flashinfer` plans attention as f16 only | – | – | – | – |
| Sentence of 30 words | 1 | 1 | 1 | f16 | Python + FlashInfer + CUDA graphs | 303 | 3.3 | 66% | 14.9 GB | 14.5 (14.5) |
| Sentence of 30 words | 1 | 2 | 2 | f16 | Python + FlashInfer + CUDA graphs | 563 | 3.6 | 85% | 15.0 GB | 28.1 (14.5 + 13.6) |
| Sentence of 30 words | 1 | 3 | 3 | f16 | Python + FlashInfer + CUDA graphs | 819 | 3.7 | 85% | 15.1 GB | 40.2 (14.4 + 13.2 + 12.6) |
| Sentence of 30 words | 1 | 4 | 4 | f16 | Python + FlashInfer + CUDA graphs | 1078 | 3.7 | 91% | 15.4 GB | 53.0 (14.4 + 13.3 + 12.4 + 13.0) |
| Sentence of 30 words | 1 | 5 | 5 | f16 | Python + FlashInfer + CUDA graphs | 1347 | 3.7 | 84% | 15.7 GB | 66.3 (14.5 + 13.3 + 12.6 + 13.0 + 13.0) |
| Sentence of 30 words | 1 | 6 | 6 | f16 | Python + FlashInfer + CUDA graphs | 1602 | 3.7 | 84% | 16.1 GB | 79.4 (14.1 + 13.2 + 12.6 + 13.0 + 13.0 + 13.5) |
| Sentence of 30 words | 1 | 7 | 7 | f16 | Python + FlashInfer + CUDA graphs | 1909 | 3.7 | 85% | 16.6 GB | 92.6 (14.4 + 13.4 + 12.6 + 12.7 + 13.2 + 13.3 + 13.1) |
| Sentence of 30 words | 1 | 8 | 8 | f16 | Python + FlashInfer + CUDA graphs | 2149 | 3.7 | 82% | 17.2 GB | 105.9 (14.4 + 13.2 + 13.1 + 13.0 + 13.2 + 13.3 + 12.9 + 12.8) |
| Sentence of 30 words | 1 | 9 | 9 | f16 | Python + FlashInfer + CUDA graphs | 2417 | 3.7 | 84% | 17.6 GB | 119.9 (14.5 + 13.3 + 13.1 + 12.7 + 13.4 + 13.5 + 12.8 + 12.8 + 13.9) |
| Sentence of 30 words | 2 | 1 | 2 | f16 | Python + FlashInfer + CUDA graphs | 569 (569, 556) | 3.5 | 91% | 17.2 GB | 27.9 (14.5, 13.5) |
| Sentence of 30 words | 2 | 2 | 4 | f16 | Python + FlashInfer + CUDA graphs | 1075 (1075, 1067) | 3.7 | 93% | 17.3 GB | 53.6 (14.5 + 13.6, 12.7 + 12.8) |
| Sentence of 30 words | 2 | 3 | 6 | f16 | Python + FlashInfer + CUDA graphs | 1595 (1595, 1593) | 3.8 | 93% | 17.7 GB | 79.8 (14.4 + 13.2 + 12.6, 13.0 + 13.4 + 13.2) |
| Sentence of 30 words | 2 | 4 | 8 | f16 | Python + FlashInfer + CUDA graphs | 2117 (2117, 2112) | 3.8 | 92% | 18.2 GB | 105.3 (14.4 + 13.3 + 12.4 + 13.0, 13.0 + 13.4 + 12.8 + 13.1) |
| Sentence of 30 words | 2 | 5 | 10 | f16 | Python + FlashInfer + CUDA graphs | 2645 (2645, 2641) | 3.8 | 92% | 18.8 GB | 132.1 (14.5 + 13.3 + 12.6 + 13.0 + 13.0, 13.3 + 12.8 + 13.1 + 13.5 + 13.2) |
| Sentence of 30 words | 2 | 6 | 12 | f16 | Python + FlashInfer + CUDA graphs | 3156 (3156, 3146) | 3.8 | 93% | 19.3 GB | 158.0 (14.1 + 13.2 + 12.6 + 13.0 + 13.0 + 13.5, 12.8 + 13.0 + 13.5 + 13.2 + 13.2 + 12.8) |
| Sentence of 30 words | 2 | 7 | 14 | f16 | Python + FlashInfer + CUDA graphs | 3680 (3680, 3665) | 3.8 | 94% | 20.2 GB | 186.2 (14.4 + 13.4 + 12.6 + 12.7 + 13.2 + 13.3 + 13.1, 12.9 + 14.1 + 13.5 + 12.9 + 12.8 + 12.8 + 14.6) |
| Sentence of 30 words | 2 | 8 | 16 | f16 | Python + FlashInfer + CUDA graphs | 4297 (4296, 4297) | 3.7 | 92% | 21.2 GB | 212.2 (14.4 + 13.2 + 13.1 + 13.0 + 13.2 + 13.3 + 12.9 + 12.8, 13.9 + 13.2 + 12.9 + 13.1 + 13.0 + 14.6 + 13.4 + 12.2) |
| Sentence of 30 words | 2 | 9 | 18 | f16 | Python + FlashInfer + CUDA graphs | 4832 (4832, 4828) | 3.7 | 93% | 22.5 GB | 239.5 (14.5 + 13.3 + 13.1 + 12.7 + 13.4 + 13.5 + 12.8 + 12.8 + 13.9, 13.2 + 12.9 + 12.7 + 13.0 + 14.5 + 13.8 + 12.5 + 13.9 + 13.2) |
| Sentence of 30 words | 2 | 10 | 20 | f16 | Python + FlashInfer + CUDA graphs | 5367 (5367, 5359) | 3.7 | 94% | 23.4 GB | 266.5 (14.5 + 13.3 + 13.1 + 12.7 + 13.1 + 13.2 + 13.1 + 13.1 + 13.9 + 13.5, 12.9 + 13.1 + 12.9 + 14.9 + 13.4 + 12.4 + 13.8 + 13.3 + 11.8 + 14.6) |
| Sentence of 30 words | 3 | 1 | 3 | f16 | Python + FlashInfer + CUDA graphs | 831 (803, 812, 831) | 3.6 | 92% | 23.1 GB | 41.0 (14.5, 13.5, 13.1) |
| Sentence of 30 words | 3 | 2 | 6 | f16 | Python + FlashInfer + CUDA graphs | 1576 (1570, 1576, 1549) | 3.8 | 93% | 23.4 GB | 80.0 (14.5 + 13.6, 12.7 + 12.8, 13.0 + 13.4) |
| Sentence of 30 words | 3 | 3 | 9 | f16 | Python + FlashInfer + CUDA graphs | 2323 (2301, 2323, 2290) | 3.9 | 95% | 23.5 GB | 119.7 (14.4 + 13.2 + 12.6, 13.0 + 13.4 + 13.2, 13.0 + 13.1 + 13.9) |
| Sentence of 30 words | 3 | 4 | 12 | f16 | Python + FlashInfer + CUDA graphs | failed: instance 0 exited (rc 1): CUDA out of memory | – | – | – | – |
| Sentence of 30 words | 4 | 1 | 4 | f16 | Python + FlashInfer + CUDA graphs | 1082 (1082, 1072, 1051, 1064) | 3.7 | 92% | 19.3 GB | 53.8 (14.5, 13.5, 13.1, 12.8) |
| Sentence of 30 words | 4 | 2 | 8 | f16 | Python + FlashInfer + CUDA graphs | 2079 (2079, 2069, 2070, 2079) | 3.8 | 98% | 19.7 GB | 105.9 (14.5 + 13.6, 12.7 + 12.8, 13.0 + 13.4, 12.8 + 13.1) |
| Sentence of 30 words | 4 | 3 | 12 | f16 | Python + FlashInfer + CUDA graphs | 3050 (2990, 3040, 3050, 3002) | 3.9 | 96% | 20.4 GB | 159.1 (14.4 + 13.2 + 12.6, 13.0 + 13.4 + 13.2, 13.0 + 13.1 + 13.9, 13.2 + 13.2 + 12.9) |
| Sentence of 30 words | 4 | 4 | 16 | f16 | Python + FlashInfer + CUDA graphs | 4052 (4038, 4021, 4006, 4052) | 3.9 | 97% | 21.3 GB | 212.0 (14.4 + 13.3 + 12.4 + 13.0, 13.0 + 13.4 + 12.8 + 13.1, 14.1 + 13.2 + 13.0 + 13.0, 13.0 + 15.2 + 13.1 + 12.2) |
| Sentence of 30 words | 4 | 5 | 20 | f16 | Python + FlashInfer + CUDA graphs | 5088 (5088, 5057, 5079, 5042) | 3.9 | 96% | 22.5 GB | 265.4 (14.5 + 13.3 + 12.6 + 13.0 + 13.0, 13.3 + 12.8 + 13.1 + 13.5 + 13.2, 12.9 + 13.0 + 13.0 + 15.2 + 13.5, 12.4 + 13.4 + 13.0 + 11.8 + 15.1) |
| Sentence of 30 words | 4 | 6 | 24 | f16 | Python + FlashInfer + CUDA graphs | 6130 (6130, 6099, 6107, 6065) | 3.9 | 97% | 22.6 GB | 321.3 (14.1 + 13.2 + 12.6 + 13.0 + 13.0 + 13.5, 12.8 + 13.0 + 13.5 + 13.2 + 13.2 + 12.8, 12.3 + 14.9 + 13.2 + 12.6 + 14.0 + 13.1, 11.4 + 15.1 + 12.7 + 13.4 + 15.7 + 14.8) |
| Sentence of 30 words | 4 | 7 | 28 | f16 | Python + FlashInfer + CUDA graphs | failed: instance 2 exited (rc 1): CUDA out of memory | – | – | – | – |
| Sentence of 30 words | 1 | 1 | 1 | bf16 | Python + FlashInfer + CUDA graphs | 316 | 3.2 | 61% | 15.4 GB | 14.2 (14.2) |
| Sentence of 30 words | 1 | 2 | 2 | bf16 | Python + FlashInfer + CUDA graphs | 584 | 3.4 | 83% | 15.4 GB | 28.3 (14.6 + 13.6) |
| Sentence of 30 words | 1 | 3 | 3 | bf16 | Python + FlashInfer + CUDA graphs | 829 | 3.6 | 88% | 15.5 GB | 40.5 (14.6 + 13.3 + 12.6) |
| Sentence of 30 words | 1 | 4 | 4 | bf16 | Python + FlashInfer + CUDA graphs | 1099 | 3.6 | 83% | 15.8 GB | 53.1 (14.4 + 13.2 + 12.6 + 13.0) |
| Sentence of 30 words | 1 | 5 | 5 | bf16 | Python + FlashInfer + CUDA graphs | 1357 | 3.7 | 84% | 16.1 GB | 67.2 (14.6 + 13.6 + 12.6 + 13.0 + 13.4) |
| Sentence of 30 words | 1 | 6 | 6 | bf16 | Python + FlashInfer + CUDA graphs | 1625 | 3.7 | 84% | 16.6 GB | 80.5 (14.6 + 13.6 + 12.6 + 12.8 + 13.4 + 13.5) |
| Sentence of 30 words | 1 | 7 | 7 | bf16 | Python + FlashInfer + CUDA graphs | 1880 | 3.7 | 83% | 17.0 GB | 93.1 (14.3 + 13.3 + 13.1 + 13.0 + 13.1 + 13.3 + 13.0) |
| Sentence of 30 words | 1 | 8 | 8 | bf16 | Python + FlashInfer + CUDA graphs | 2169 | 3.7 | 82% | 17.6 GB | 105.6 (14.4 + 13.3 + 12.5 + 13.0 + 13.4 + 13.2 + 12.8 + 13.1) |
| Sentence of 30 words | 1 | 9 | 9 | bf16 | Python + FlashInfer + CUDA graphs | 2433 | 3.7 | 85% | 18.0 GB | 119.9 (14.2 + 13.3 + 13.1 + 13.0 + 13.3 + 13.5 + 12.8 + 12.8 + 14.0) |
| Sentence of 30 words | 1 | 10 | 10 | bf16 | Python + FlashInfer + CUDA graphs | 2697 | 3.7 | 85% | 17.8 GB | 133.8 (14.4 + 13.4 + 13.1 + 12.9 + 13.4 + 13.3 + 13.1 + 13.1 + 13.9 + 13.2) |
| Sentence of 30 words | 2 | 1 | 2 | bf16 | Python + FlashInfer + CUDA graphs | 616 (581, 616) | 3.2 | 95% | 17.5 GB | 27.7 (14.2, 13.5) |
| Sentence of 30 words | 2 | 2 | 4 | bf16 | Python + FlashInfer + CUDA graphs | 1060 (1060, 1035) | 3.8 | 91% | 17.6 GB | 53.9 (14.6 + 13.6, 12.6 + 13.0) |
| Sentence of 30 words | 2 | 3 | 6 | bf16 | Python + FlashInfer + CUDA graphs | 1574 (1574, 1543) | 3.8 | 93% | 18.0 GB | 79.8 (14.6 + 13.3 + 12.6, 12.9 + 13.1 + 13.3) |
| Sentence of 30 words | 2 | 4 | 8 | bf16 | Python + FlashInfer + CUDA graphs | 2123 (2123, 2120) | 3.8 | 93% | 18.6 GB | 105.5 (14.4 + 13.2 + 12.6 + 13.0, 13.0 + 13.4 + 13.1 + 12.8) |
| Sentence of 30 words | 2 | 5 | 10 | bf16 | Python + FlashInfer + CUDA graphs | 2657 (2646, 2657) | 3.8 | 92% | 19.2 GB | 133.3 (14.6 + 13.6 + 12.6 + 13.0 + 13.4, 13.3 + 12.8 + 13.1 + 13.5 + 13.5) |
| Sentence of 30 words | 2 | 6 | 12 | bf16 | Python + FlashInfer + CUDA graphs | 3179 (3179, 3178) | 3.8 | 93% | 19.9 GB | 158.7 (14.6 + 13.6 + 12.6 + 12.8 + 13.4 + 13.5, 12.8 + 13.0 + 13.6 + 13.2 + 12.9 + 12.7) |
| Sentence of 30 words | 3 | 1 | 3 | bf16 | Python + FlashInfer + CUDA graphs | 844 (827, 844, 838) | 3.6 | 93% | 13.5 GB | 40.5 (14.2, 13.5, 12.8) |
| Sentence of 30 words | 3 | 2 | 6 | bf16 | Python + FlashInfer + CUDA graphs | 1551 (1551, 1512, 1547) | 3.9 | 94% | 13.8 GB | 80.1 (14.6 + 13.6, 12.6 + 13.0, 12.7 + 13.5) |
| Sentence of 30 words | 3 | 3 | 9 | bf16 | Python + FlashInfer + CUDA graphs | 2312 (2312, 2289, 2295) | 3.9 | 98% | 14.4 GB | 119.7 (14.6 + 13.3 + 12.6, 12.9 + 13.1 + 13.3, 13.1 + 12.9 + 13.9) |
| Sentence of 30 words | 3 | 4 | 12 | bf16 | Python + FlashInfer + CUDA graphs | 3072 (3072, 3056, 3059) | 3.9 | 95% | 15.0 GB | 158.7 (14.4 + 13.2 + 12.6 + 13.0, 13.0 + 13.4 + 13.1 + 12.8, 13.8 + 13.5 + 12.9 + 13.1) |
| Sentence of 30 words | 3 | 5 | 15 | bf16 | Python + FlashInfer + CUDA graphs | 3868 (3861, 3868, 3822) | 3.9 | 95% | 16.0 GB | 200.4 (14.6 + 13.6 + 12.6 + 13.0 + 13.4, 13.3 + 12.8 + 13.1 + 13.5 + 13.5, 12.9 + 13.0 + 13.0 + 14.7 + 13.5) |
| Sentence of 30 words | 3 | 6 | 18 | bf16 | Python + FlashInfer + CUDA graphs | 4609 (4595, 4609, 4595) | 3.9 | 96% | 16.8 GB | 239.1 (14.6 + 13.6 + 12.6 + 12.8 + 13.4 + 13.5, 12.8 + 13.0 + 13.6 + 13.2 + 12.9 + 12.7, 12.3 + 15.2 + 13.2 + 12.4 + 14.1 + 13.2) |
| Sentence of 30 words | 3 | 7 | 21 | bf16 | Python + FlashInfer + CUDA graphs | 5374 (5369, 5291, 5374) | 3.9 | 96% | 18.2 GB | 279.3 (14.3 + 13.3 + 13.1 + 13.0 + 13.1 + 13.3 + 13.0, 12.9 + 14.0 + 13.5 + 12.9 + 13.1 + 12.8 + 14.8, 13.5 + 12.6 + 14.0 + 13.1 + 11.6 + 15.1 + 12.4) |
| Sentence of 30 words | 3 | 8 | 24 | bf16 | Python + FlashInfer + CUDA graphs | 6282 (6203, 6282, 6200) | 3.8 | 95% | 19.9 GB | 322.0 (14.4 + 13.3 + 12.5 + 13.0 + 13.4 + 13.2 + 12.8 + 13.1, 13.7 + 13.5 + 13.2 + 13.1 + 13.0 + 14.7 + 13.3 + 12.1, 13.8 + 13.1 + 11.7 + 15.0 + 12.7 + 13.2 + 15.3 + 14.9) |
| Sentence of 30 words | 3 | 9 | 27 | bf16 | Python + FlashInfer + CUDA graphs | 7094 (6982, 7094, 6999) | 3.8 | 95% | 21.8 GB | 365.3 (14.2 + 13.3 + 13.1 + 13.0 + 13.3 + 13.5 + 12.8 + 12.8 + 14.0, 13.2 + 13.0 + 12.8 + 13.0 + 14.9 + 13.8 + 12.0 + 14.0 + 13.4, 11.5 + 14.9 + 12.7 + 13.4 + 15.6 + 15.0 + 14.5 + 12.4 + 15.3) |
| Sentence of 30 words | 3 | 10 | 30 | bf16 | Python + FlashInfer + CUDA graphs | 7919 (7870, 7899, 7919) | 3.8 | 94% | 23.5 GB | 405.9 (14.4 + 13.4 + 13.1 + 12.9 + 13.4 + 13.3 + 13.1 + 13.1 + 13.9 + 13.2, 12.9 + 13.1 + 13.0 + 14.9 + 13.4 + 12.4 + 13.7 + 13.0 + 11.8 + 14.6, 12.7 + 13.4 + 15.3 + 15.1 + 14.4 + 12.4 + 14.9 + 13.9 + 14.2 + 13.0) |
| Sentence of 30 words | 1 | 1 | 1 | f16 | C++ (`perf_mode=flash_attention`) | 270 | 3.7 | 89% | 2.5 GB | 14.6 (14.6) |
| Sentence of 30 words | 1 | 2 | 2 | f16 | C++ (`perf_mode=flash_attention`) | 517 | 3.9 | 91% | 2.5 GB | 28.3 (14.6 + 13.6) |
| Sentence of 30 words | 1 | 3 | 3 | f16 | C++ (`perf_mode=flash_attention`) | 805 | 3.7 | 90% | 2.6 GB | 41.4 (14.6 + 13.6 + 13.1) |
| Sentence of 30 words | 1 | 4 | 4 | f16 | C++ (`perf_mode=flash_attention`) | 1117 | 3.6 | 91% | 2.6 GB | 54.3 (14.6 + 13.6 + 13.1 + 13.0) |
| Sentence of 30 words | 1 | 5 | 5 | f16 | C++ (`perf_mode=flash_attention`) | 1423 | 3.5 | 91% | 2.7 GB | 67.8 (14.6 + 13.6 + 13.1 + 13.0 + 13.4) |
| Sentence of 30 words | 2 | 1 | 2 | f16 | C++ (`perf_mode=flash_attention`) | 528 (510, 528) | 3.8 | 93% | 5.0 GB | 28.3 (14.6, 13.6) |
| Sentence of 30 words | 2 | 2 | 4 | f16 | C++ (`perf_mode=flash_attention`) | 1030 (1030, 1007) | 3.9 | 95% | 5.1 GB | 54.3 (14.6 + 13.6, 13.1 + 13.0) |
| Sentence of 30 words | 2 | 3 | 6 | f16 | C++ (`perf_mode=flash_attention`) | 1624 (1624, 1605) | 3.7 | 96% | 5.2 GB | 81.3 (14.6 + 13.6 + 13.1, 13.0 + 13.4 + 13.5) |
| Sentence of 30 words | 2 | 4 | 8 | f16 | C++ (`perf_mode=flash_attention`) | 2290 (2290, 2273) | 3.5 | 96% | 5.2 GB | 107.4 (14.6 + 13.6 + 13.1 + 13.0, 13.4 + 13.5 + 13.1 + 13.0) |
| Sentence of 30 words | 2 | 5 | 10 | f16 | C++ (`perf_mode=flash_attention`) | 2925 (2925, 2908) | 3.4 | 96% | 5.4 GB | 134.9 (14.6 + 13.6 + 13.1 + 13.0 + 13.4, 13.5 + 13.1 + 12.9 + 14.1 + 13.5) |
| Sentence of 30 words | 1 | 1 | 1 | bf16 | C++ (`perf_mode=flash_attention`) | 296 | 3.4 | 84% | 2.5 GB | 14.6 (14.6) |
| Sentence of 30 words | 1 | 2 | 2 | bf16 | C++ (`perf_mode=flash_attention`) | 590 | 3.4 | 91% | 2.6 GB | 28.3 (14.6 + 13.6) |
| Sentence of 30 words | 1 | 3 | 3 | bf16 | C++ (`perf_mode=flash_attention`) | 853 | 3.5 | 91% | 2.6 GB | 41.3 (14.6 + 13.6 + 13.0) |
| Sentence of 30 words | 1 | 4 | 4 | bf16 | C++ (`perf_mode=flash_attention`) | 1126 | 3.6 | 92% | 2.6 GB | 54.3 (14.6 + 13.6 + 13.0 + 13.0) |
| Sentence of 30 words | 1 | 5 | 5 | bf16 | C++ (`perf_mode=flash_attention`) | 1426 | 3.5 | 91% | 2.7 GB | 67.4 (14.6 + 13.6 + 13.1 + 13.0 + 13.1) |
| Sentence of 30 words | 1 | 6 | 6 | bf16 | C++ (`perf_mode=flash_attention`) | 1719 | 3.5 | 90% | 2.7 GB | 80.9 (14.6 + 13.6 + 13.1 + 13.0 + 13.1 + 13.5) |
| Sentence of 30 words | 1 | 7 | 7 | bf16 | C++ (`perf_mode=flash_attention`) | 2011 | 3.5 | 97% | 2.8 GB | 94.3 (14.6 + 13.6 + 13.1 + 13.0 + 13.4 + 13.5 + 13.0) |
| Sentence of 30 words | 2 | 1 | 2 | bf16 | C++ (`perf_mode=flash_attention`) | 564 (564, 528) | 3.5 | 95% | 5.0 GB | 28.3 (14.6, 13.6) |
| Sentence of 30 words | 2 | 2 | 4 | bf16 | C++ (`perf_mode=flash_attention`) | 1144 (1144, 1133) | 3.5 | 96% | 5.1 GB | 53.8 (14.6 + 13.6, 12.8 + 12.7) |
| Sentence of 30 words | 2 | 3 | 6 | bf16 | C++ (`perf_mode=flash_attention`) | 1730 (1730, 1713) | 3.5 | 97% | 5.2 GB | 80.9 (14.6 + 13.6 + 13.0, 12.9 + 13.1 + 13.5) |
| Sentence of 30 words | 2 | 4 | 8 | bf16 | C++ (`perf_mode=flash_attention`) | 2300 (2300, 2267) | 3.5 | 96% | 5.3 GB | 106.9 (14.6 + 13.6 + 13.0 + 13.0, 13.1 + 13.5 + 13.1 + 12.9) |
| Sentence of 30 words | 1 | 1 | 1 | f16 | C++ (`perf_mode=off`) | 393 | 2.5 | 92% | 2.5 GB | 14.6 (14.6) |
| Sentence of 30 words | 1 | 2 | 2 | f16 | C++ (`perf_mode=off`) | 777 | 2.6 | 93% | 2.6 GB | 28.3 (14.6 + 13.6) |
| Sentence of 30 words | 1 | 3 | 3 | f16 | C++ (`perf_mode=off`) | 1174 | 2.6 | 93% | 2.6 GB | 41.4 (14.6 + 13.6 + 13.1) |
| Sentence of 30 words | 1 | 4 | 4 | f16 | C++ (`perf_mode=off`) | 1603 | 2.5 | 92% | 2.7 GB | 54.3 (14.6 + 13.6 + 13.1 + 13.0) |
| Sentence of 30 words | 1 | 5 | 5 | f16 | C++ (`perf_mode=off`) | 2029 | 2.5 | 93% | 2.7 GB | 67.8 (14.6 + 13.6 + 13.1 + 13.0 + 13.4) |
| Sentence of 30 words | 2 | 1 | 2 | f16 | C++ (`perf_mode=off`) | 797 (797, 792) | 2.5 | 96% | 5.0 GB | 28.2 (14.6, 13.6) |
| Sentence of 30 words | 2 | 2 | 4 | f16 | C++ (`perf_mode=off`) | 1562 (1562, 1542) | 2.6 | 96% | 5.1 GB | 54.3 (14.6 + 13.6, 13.1 + 13.0) |
| Sentence of 30 words | 2 | 3 | 6 | f16 | C++ (`perf_mode=off`) | 2405 (2405, 2388) | 2.5 | 96% | 5.2 GB | 81.3 (14.6 + 13.6 + 13.1, 13.0 + 13.4 + 13.5) |
| Sentence of 30 words | 2 | 4 | 8 | f16 | C++ (`perf_mode=off`) | 3318 (3318, 3300) | 2.4 | 96% | 5.3 GB | 107.5 (14.6 + 13.6 + 13.1 + 13.0, 13.4 + 13.5 + 13.1 + 13.1) |
| Sentence of 30 words | 2 | 5 | 10 | f16 | C++ (`perf_mode=off`) | 4240 (4240, 4224) | 2.4 | 96% | 5.4 GB | 135.0 (14.6 + 13.6 + 13.1 + 13.0 + 13.4, 13.5 + 13.1 + 13.1 + 14.1 + 13.5) |
| Sentence of 30 words | 1 | 1 | 1 | bf16 | C++ (`perf_mode=off`) | 416 | 2.4 | 93% | 2.5 GB | 14.6 (14.6) |
| Sentence of 30 words | 1 | 2 | 2 | bf16 | C++ (`perf_mode=off`) | 831 | 2.4 | 94% | 2.6 GB | 27.9 (14.6 + 13.3) |
| Sentence of 30 words | 1 | 3 | 3 | bf16 | C++ (`perf_mode=off`) | 1223 | 2.5 | 93% | 2.6 GB | 41.4 (14.6 + 13.6 + 13.1) |
| Sentence of 30 words | 1 | 4 | 4 | bf16 | C++ (`perf_mode=off`) | 1612 | 2.5 | 92% | 2.7 GB | 54.3 (14.6 + 13.6 + 13.1 + 13.0) |
| Sentence of 30 words | 1 | 5 | 5 | bf16 | C++ (`perf_mode=off`) | 2036 | 2.5 | 93% | 2.7 GB | 67.0 (14.3 + 13.6 + 13.0 + 13.0 + 13.1) |
| Sentence of 30 words | 1 | 6 | 6 | bf16 | C++ (`perf_mode=off`) | 2450 | 2.4 | 92% | 2.8 GB | 80.5 (14.3 + 13.6 + 13.0 + 13.0 + 13.1 + 13.5) |
| Sentence of 30 words | 1 | 7 | 7 | bf16 | C++ (`perf_mode=off`) | 2871 | 2.4 | 92% | 2.8 GB | 93.8 (14.6 + 13.6 + 13.0 + 13.0 + 13.1 + 13.5 + 13.0) |
| Sentence of 30 words | 2 | 1 | 2 | bf16 | C++ (`perf_mode=off`) | 833 (833, 828) | 2.4 | 96% | 5.0 GB | 28.3 (14.6, 13.6) |
| Sentence of 30 words | 2 | 2 | 4 | bf16 | C++ (`perf_mode=off`) | 1665 (1665, 1645) | 2.4 | 96% | 5.1 GB | 53.8 (14.6 + 13.3, 13.1 + 12.8) |
| Sentence of 30 words | 2 | 3 | 6 | bf16 | C++ (`perf_mode=off`) | 2514 (2514, 2497) | 2.4 | 96% | 5.2 GB | 80.7 (14.6 + 13.6 + 13.1, 12.8 + 13.1 + 13.5) |
| Sentence of 30 words | 2 | 4 | 8 | bf16 | C++ (`perf_mode=off`) | 3339 (3339, 3322) | 2.4 | 96% | 5.3 GB | 106.9 (14.6 + 13.6 + 13.1 + 13.0, 13.1 + 13.5 + 13.1 + 12.9) |
| Sentence of 30 words | 2 | 5 | 10 | bf16 | C++ (`perf_mode=off`) | 4254 (4254, 4237) | 2.4 | 97% | 5.4 GB | 134.1 (14.3 + 13.6 + 13.0 + 13.0 + 13.1, 13.5 + 13.1 + 13.1 + 13.8 + 13.5) |
| Sentence of 30 words | 1 | 1 | 1 | f16 | Python (PyTorch) | 1018 | 1.0 | 39% | 6.5 GB | 14.4 (14.4) |
| Sentence of 30 words | 1 | 2 | 2 | f16 | Python (PyTorch) | 1095 | 1.8 | 63% | 6.5 GB | 28.0 (14.6 + 13.3) |
| Sentence of 30 words | 1 | 3 | 3 | f16 | Python (PyTorch) | 1178 | 2.5 | 89% | 6.5 GB | 40.8 (14.6 + 13.6 + 12.6) |
| Sentence of 30 words | 1 | 4 | 4 | f16 | Python (PyTorch) | 1627 | 2.5 | 88% | 6.5 GB | 53.9 (14.6 + 13.2 + 13.1 + 13.0) |
| Sentence of 30 words | 1 | 5 | 5 | f16 | Python (PyTorch) | 2087 | 2.4 | 94% | 6.5 GB | 66.8 (14.6 + 13.2 + 12.6 + 13.0 + 13.4) |
| Sentence of 30 words | 1 | 6 | 6 | f16 | Python (PyTorch) | 2543 | 2.4 | 90% | 6.5 GB | 80.5 (14.6 + 13.3 + 13.1 + 13.0 + 13.2 + 13.3) |
| Sentence of 30 words | 2 | 1 | 2 | f16 | Python (PyTorch) | 1052 (1052, 1020) | 1.9 | 94% | 10.1 GB | 28.0 (14.4, 13.6) |
| Sentence of 30 words | 2 | 2 | 4 | f16 | Python (PyTorch) | 1516 (1516, 1461) | 2.6 | 99% | 10.1 GB | 54.0 (14.6 + 13.3, 13.1 + 12.9) |
| Sentence of 30 words | 2 | 3 | 6 | f16 | Python (PyTorch) | 2288 (2288, 2197) | 2.6 | 95% | 10.1 GB | 80.7 (14.6 + 13.6 + 12.6, 13.0 + 13.4 + 13.5) |
| Sentence of 30 words | 2 | 4 | 8 | f16 | Python (PyTorch) | 3148 (3148, 3018) | 2.5 | 95% | 10.1 GB | 106.4 (14.6 + 13.2 + 13.1 + 13.0, 13.3 + 13.5 + 12.8 + 12.8) |
| Sentence of 30 words | 2 | 5 | 10 | f16 | Python (PyTorch) | 4227 (4227, 4111) | 2.4 | 94% | 11.0 GB | 133.7 (14.6 + 13.2 + 12.6 + 13.0 + 13.4, 13.5 + 13.1 + 12.8 + 14.1 + 13.5) |
| Sentence of 30 words | 3 | 1 | 3 | f16 | Python (PyTorch) | 1466 (1268, 1250, 1466) | 2.0 | 77% | 15.0 GB | 40.8 (14.4, 13.6, 12.9) |
| Sentence of 30 words | 3 | 2 | 6 | f16 | Python (PyTorch) | 2196 (2196, 2106, 2144) | 2.7 | 96% | 15.0 GB | 79.9 (14.6 + 13.3, 13.1 + 12.9, 13.1 + 12.8) |
| Sentence of 30 words | 3 | 3 | 9 | f16 | Python (PyTorch) | 3347 (3347, 3222, 3313) | 2.7 | 96% | 15.0 GB | 120.6 (14.6 + 13.6 + 12.6, 13.0 + 13.4 + 13.5, 13.1 + 12.9 + 13.8) |
| Sentence of 30 words | 3 | 4 | 12 | f16 | Python (PyTorch) | 4607 (4607, 4454, 4543) | 2.6 | 96% | 15.0 GB | 160.1 (14.6 + 13.2 + 13.1 + 13.0, 13.3 + 13.5 + 12.8 + 12.8, 14.1 + 13.4 + 13.2 + 13.1) |
| Sentence of 30 words | 3 | 5 | 15 | f16 | Python (PyTorch) | 6257 (6228, 6086, 6257) | 2.4 | 97% | 16.4 GB | 201.3 (14.6 + 13.2 + 12.6 + 13.0 + 13.4, 13.5 + 13.1 + 12.8 + 14.1 + 13.5, 12.9 + 12.8 + 13.0 + 15.2 + 13.8) |
| Sentence of 30 words | 4 | 1 | 4 | f16 | Python (PyTorch) | 1609 (1609, 1556, 1526, 1527) | 2.5 | 95% | 23.3 GB | 53.6 (14.4, 13.6, 12.9, 12.8) |
| Sentence of 30 words | 4 | 2 | 8 | f16 | Python (PyTorch) | 2864 (2864, 2794, 2814, 2752) | 2.8 | 97% | 23.3 GB | 105.8 (14.6 + 13.3, 13.1 + 12.9, 13.1 + 12.8, 13.1 + 12.8) |
| Sentence of 30 words | 4 | 3 | 12 | f16 | Python (PyTorch) | 4421 (4421, 4240, 4336, 4272) | 2.7 | 96% | 23.3 GB | 159.7 (14.6 + 13.6 + 12.6, 13.0 + 13.4 + 13.5, 13.1 + 12.9 + 13.8, 13.2 + 12.8 + 13.1) |
| Sentence of 30 words | 4 | 4 | 16 | f16 | Python (PyTorch) | 6199 (6172, 5904, 6023, 6199) | 2.6 | 99% | 23.3 GB | 214.1 (14.6 + 13.2 + 13.1 + 13.0, 13.3 + 13.5 + 12.8 + 12.8, 14.1 + 13.4 + 13.2 + 13.1, 12.9 + 14.9 + 13.8 + 12.3) |
| Sentence of 30 words | 4 | 5 | 20 | f16 | Python (PyTorch) | 8281 (8272, 8090, 8262, 8281) | 2.4 | 98% | 23.3 GB | 267.1 (14.6 + 13.2 + 12.6 + 13.0 + 13.4, 13.5 + 13.1 + 12.8 + 14.1 + 13.5, 12.9 + 12.8 + 13.0 + 15.2 + 13.8, 12.6 + 13.9 + 12.8 + 11.3 + 15.1) |
| Sentence of 30 words | 1 | 1 | 1 | bf16 | Python (PyTorch) | 1006 | 1.0 | 39% | 7.5 GB | 14.4 (14.4) |
| Sentence of 30 words | 1 | 2 | 2 | bf16 | Python (PyTorch) | 1038 | 1.9 | 67% | 7.5 GB | 27.9 (14.6 + 13.3) |
| Sentence of 30 words | 1 | 3 | 3 | bf16 | Python (PyTorch) | 1215 | 2.5 | 86% | 7.5 GB | 41.1 (14.4 + 13.6 + 13.1) |
| Sentence of 30 words | 1 | 4 | 4 | bf16 | Python (PyTorch) | 1628 | 2.5 | 87% | 7.5 GB | 53.8 (14.6 + 13.2 + 13.1 + 13.0) |
| Sentence of 30 words | 1 | 5 | 5 | bf16 | Python (PyTorch) | 2070 | 2.4 | 90% | 7.5 GB | 67.3 (14.6 + 13.4 + 13.1 + 13.0 + 13.2) |
| Sentence of 30 words | 1 | 6 | 6 | bf16 | Python (PyTorch) | 2520 | 2.4 | 90% | 7.5 GB | 80.7 (14.6 + 13.2 + 13.1 + 13.0 + 13.4 + 13.3) |
| Sentence of 30 words | 2 | 1 | 2 | bf16 | Python (PyTorch) | 1026 (1016, 1026) | 1.9 | 93% | 10.1 GB | 28.0 (14.4, 13.6) |
| Sentence of 30 words | 2 | 2 | 4 | bf16 | Python (PyTorch) | 1510 (1510, 1452) | 2.6 | 94% | 10.1 GB | 53.7 (14.6 + 13.3, 12.9 + 12.8) |
| Sentence of 30 words | 2 | 3 | 6 | bf16 | Python (PyTorch) | 2264 (2264, 2173) | 2.7 | 95% | 10.1 GB | 81.0 (14.4 + 13.6 + 13.1, 13.0 + 13.4 + 13.5) |
| Sentence of 30 words | 2 | 4 | 8 | bf16 | Python (PyTorch) | 3129 (3129, 3023) | 2.6 | 94% | 10.1 GB | 106.3 (14.6 + 13.2 + 13.1 + 13.0, 13.3 + 13.2 + 12.8 + 13.1) |
| Sentence of 30 words | 2 | 5 | 10 | bf16 | Python (PyTorch) | 4160 (4160, 4082) | 2.4 | 96% | 11.0 GB | 133.3 (14.6 + 13.4 + 13.1 + 13.0 + 13.2, 12.9 + 13.1 + 12.8 + 13.7 + 13.5) |
| Sentence of 30 words | 2 | 6 | 12 | bf16 | Python (PyTorch) | 5094 (5094, 5009) | 2.4 | 95% | 12.1 GB | 160.1 (14.6 + 13.2 + 13.1 + 13.0 + 13.4 + 13.3, 13.1 + 13.0 + 14.1 + 13.3 + 12.9 + 13.1) |
| Sentence of 30 words | 3 | 1 | 3 | bf16 | Python (PyTorch) | 1669 (1353, 1669, 1577) | 1.8 | 86% | 15.0 GB | 40.8 (14.4, 13.6, 12.8) |
| Sentence of 30 words | 3 | 2 | 6 | bf16 | Python (PyTorch) | 2184 (2184, 2102, 2117) | 2.7 | 96% | 15.0 GB | 79.4 (14.6 + 13.3, 12.9 + 12.8, 12.9 + 12.8) |
| Sentence of 30 words | 3 | 3 | 9 | bf16 | Python (PyTorch) | 3357 (3357, 3183, 3323) | 2.7 | 98% | 15.0 GB | 120.9 (14.4 + 13.6 + 13.1, 13.0 + 13.4 + 13.5, 13.1 + 12.8 + 13.9) |
| Sentence of 30 words | 3 | 4 | 12 | bf16 | Python (PyTorch) | 4612 (4612, 4457, 4544) | 2.6 | 96% | 15.0 GB | 159.8 (14.6 + 13.2 + 13.1 + 13.0, 13.3 + 13.2 + 12.8 + 13.1, 13.9 + 13.5 + 13.0 + 13.1) |
| Sentence of 30 words | 3 | 5 | 15 | bf16 | Python (PyTorch) | 6268 (6245, 6010, 6268) | 2.4 | 96% | 16.4 GB | 200.9 (14.6 + 13.4 + 13.1 + 13.0 + 13.2, 12.9 + 13.1 + 12.8 + 13.7 + 13.5, 13.2 + 12.9 + 13.0 + 14.9 + 13.8) |
| Sentence of 30 words | 4 | 1 | 4 | bf16 | Python (PyTorch) | 1576 (1576, 1541, 1500, 1518) | 2.5 | 95% | 22.5 GB | 53.7 (14.4, 13.6, 12.8, 13.0) |
| Sentence of 30 words | 4 | 2 | 8 | bf16 | Python (PyTorch) | 2867 (2867, 2730, 2755, 2784) | 2.8 | 96% | 22.5 GB | 105.3 (14.6 + 13.3, 12.9 + 12.8, 12.9 + 12.8, 13.1 + 12.8) |
| Sentence of 30 words | 4 | 3 | 12 | bf16 | Python (PyTorch) | 4354 (4354, 4222, 4281, 4201) | 2.8 | 97% | 22.5 GB | 160.2 (14.4 + 13.6 + 13.1, 13.0 + 13.4 + 13.5, 13.1 + 12.8 + 13.9, 13.2 + 13.1 + 13.1) |
| Sentence of 30 words | 4 | 4 | 16 | bf16 | Python (PyTorch) | 6198 (6198, 5831, 5981, 6184) | 2.6 | 97% | 22.5 GB | 213.8 (14.6 + 13.2 + 13.1 + 13.0, 13.3 + 13.2 + 12.8 + 13.1, 13.9 + 13.5 + 13.0 + 13.1, 13.0 + 14.9 + 13.8 + 12.3) |
| Sentence of 30 words | 4 | 5 | 20 | bf16 | Python (PyTorch) | 8289 (8174, 8033, 8289, 8219) | 2.4 | 97% | 22.5 GB | 267.2 (14.6 + 13.4 + 13.1 + 13.0 + 13.2, 12.9 + 13.1 + 12.8 + 13.7 + 13.5, 13.2 + 12.9 + 13.0 + 14.9 + 13.8, 12.6 + 13.8 + 12.9 + 11.8 + 15.1) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 1 | 1 | f16 | Python + FlashInfer | 466 | 2.1 | 19% | 4.7 GB | 2.4 (2.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 2 | 2 | f16 | Python + FlashInfer | 515 | 3.9 | 31% | 4.7 GB | 8.9 (2.4 + 6.5) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 3 | 3 | f16 | Python + FlashInfer | 530 | 5.7 | 47% | 4.7 GB | 19.0 (2.4 + 6.8 + 9.7) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 4 | 4 | f16 | Python + FlashInfer | 600 | 6.7 | 72% | 4.7 GB | 33.6 (2.4 + 6.8 + 9.7 + 14.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 5 | 5 | f16 | Python + FlashInfer | 641 | 7.8 | 78% | 4.7 GB | 36.7 (2.4 + 6.5 + 9.4 + 14.6 + 3.8) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 6 | 6 | f16 | Python + FlashInfer | 781 | 7.7 | 76% | 4.7 GB | 46.2 (2.4 + 6.8 + 9.5 + 14.6 + 3.9 + 8.8) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 7 | 7 | f16 | Python + FlashInfer | 825 | 8.5 | 81% | 4.7 GB | 47.5 (2.4 + 6.6 + 9.4 + 14.6 + 3.7 + 8.8 + 1.9) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 8 | 8 | f16 | Python + FlashInfer | 1008 | 7.9 | 74% | 4.7 GB | 61.2 (2.2 + 6.8 + 9.7 + 14.4 + 3.6 + 8.8 + 2.1 + 13.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 9 | 9 | f16 | Python + FlashInfer | 1059 | 8.5 | 76% | 4.7 GB | 63.6 (2.4 + 6.8 + 9.4 + 14.6 + 3.8 + 8.8 + 2.1 + 13.5 + 2.1) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 10 | 10 | f16 | Python + FlashInfer | 1207 | 8.3 | 74% | 4.7 GB | 70.5 (2.4 + 6.8 + 9.4 + 14.6 + 3.7 + 8.8 + 2.1 + 13.3 + 2.1 + 7.1) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 11 | 11 | f16 | Python + FlashInfer | 1366 | 8.1 | 75% | 4.7 GB | 79.8 (2.4 + 6.5 + 9.7 + 14.2 + 3.9 + 8.6 + 2.0 + 13.3 + 2.1 + 7.1 + 10.1) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 12 | 12 | f16 | Python + FlashInfer | 1544 | 7.8 | 74% | 4.7 GB | 93.1 (2.4 + 6.5 + 9.4 + 14.6 + 3.9 + 8.8 + 2.1 + 13.2 + 2.1 + 6.8 + 10.2 + 13.1) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 1 | 2 | f16 | Python + FlashInfer | 711 (711, 619) | 2.8 | 90% | 11.5 GB | 9.0 (2.4, 6.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 2 | 4 | f16 | Python + FlashInfer | 663 (663, 646) | 6.0 | 87% | 11.5 GB | 32.5 (2.4 + 6.5, 9.5 + 14.2) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 3 | 6 | f16 | Python + FlashInfer | 769 (707, 769) | 7.8 | 84% | 11.5 GB | 45.1 (2.4 + 6.8 + 9.7, 14.3 + 3.5 + 8.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 4 | 8 | f16 | Python + FlashInfer | 1000 (1000, 961) | 8.0 | 89% | 11.5 GB | 61.3 (2.4 + 6.8 + 9.7 + 14.6, 3.6 + 8.4 + 2.1 + 13.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 5 | 10 | f16 | Python + FlashInfer | 1151 (1143, 1151) | 8.7 | 87% | 11.5 GB | 70.1 (2.4 + 6.5 + 9.4 + 14.6 + 3.8, 8.8 + 2.1 + 13.3 + 2.1 + 7.1) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 6 | 12 | f16 | Python + FlashInfer | 1533 (1488, 1533) | 7.8 | 87% | 11.5 GB | 94.1 (2.4 + 6.8 + 9.5 + 14.6 + 3.9 + 8.8, 2.1 + 13.4 + 2.1 + 7.1 + 10.2 + 13.1) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 7 | 14 | f16 | Python + FlashInfer | 1695 (1637, 1695) | 8.3 | 87% | 11.5 GB | 105.6 (2.4 + 6.6 + 9.4 + 14.6 + 3.7 + 8.8 + 1.9, 13.2 + 2.1 + 6.8 + 10.2 + 12.8 + 4.2 + 8.8) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 8 | 16 | f16 | Python + FlashInfer | 1909 (1909, 1876) | 8.4 | 86% | 11.5 GB | 121.7 (2.2 + 6.8 + 9.7 + 14.4 + 3.6 + 8.8 + 2.1 + 13.6, 2.1 + 6.8 + 10.7 + 12.8 + 3.9 + 9.1 + 2.7 + 12.5) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 1 | 3 | f16 | Python + FlashInfer | 688 (688, 605, 562) | 4.4 | 90% | 16.8 GB | 18.5 (2.4, 6.6, 9.5) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 2 | 6 | f16 | Python + FlashInfer | 885 (637, 722, 885) | 6.8 | 90% | 16.8 GB | 44.3 (2.4 + 6.5, 9.5 + 14.2, 3.4 + 8.3) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 3 | 9 | f16 | Python + FlashInfer | 1037 (958, 1037, 929) | 8.7 | 89% | 16.9 GB | 62.7 (2.4 + 6.8 + 9.7, 14.3 + 3.5 + 8.4, 2.1 + 13.4 + 2.0) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 4 | 12 | f16 | Python + FlashInfer | 1435 (1431, 1351, 1435) | 8.4 | 93% | 16.9 GB | 93.2 (2.4 + 6.8 + 9.7 + 14.6, 3.6 + 8.4 + 2.1 + 13.6, 1.8 + 6.8 + 10.3 + 13.1) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 5 | 15 | f16 | Python + FlashInfer | 1753 (1702, 1648, 1753) | 8.6 | 93% | 16.9 GB | 108.6 (2.4 + 6.5 + 9.4 + 14.6 + 3.8, 8.8 + 2.1 + 13.3 + 2.1 + 7.1, 10.2 + 13.1 + 3.6 + 9.1 + 2.5) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 6 | 18 | f16 | Python + FlashInfer | 2061 (2006, 2061, 1830) | 8.7 | 91% | 16.9 GB | 131.2 (2.4 + 6.8 + 9.5 + 14.6 + 3.9 + 8.8, 2.1 + 13.4 + 2.1 + 7.1 + 10.2 + 13.1, 3.7 + 9.1 + 2.7 + 12.8 + 2.2 + 6.7) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 7 | 21 | f16 | Python + FlashInfer | 2438 (2343, 2438, 2357) | 8.6 | 92% | 16.9 GB | 158.3 (2.4 + 6.6 + 9.4 + 14.6 + 3.7 + 8.8 + 1.9, 13.2 + 2.1 + 6.8 + 10.2 + 12.8 + 4.2 + 8.8, 2.6 + 13.0 + 2.1 + 6.5 + 11.0 + 13.4 + 4.0) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 8 | 24 | f16 | Python + FlashInfer | 2704 (2704, 2673, 2690) | 8.9 | 93% | 16.9 GB | 183.8 (2.2 + 6.8 + 9.7 + 14.4 + 3.6 + 8.8 + 2.1 + 13.6, 2.1 + 6.8 + 10.7 + 12.8 + 3.9 + 9.1 + 2.7 + 12.5, 2.2 + 6.8 + 10.9 + 13.4 + 3.9 + 9.2 + 2.4 + 13.2) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 9 | 27 | f16 | Python + FlashInfer | 3081 (2899, 2960, 3081) | 8.8 | 92% | 16.9 GB | 204.6 (2.4 + 6.8 + 9.4 + 14.6 + 3.8 + 8.8 + 2.1 + 13.5 + 2.1, 6.6 + 10.2 + 13.1 + 4.2 + 8.6 + 2.7 + 12.7 + 2.2 + 6.8, 11.0 + 13.4 + 3.9 + 9.2 + 2.4 + 13.5 + 1.8 + 7.1 + 11.5) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 10 | 30 | f16 | Python + FlashInfer | 3422 (3316, 3422, 3323) | 8.8 | 91% | 16.9 GB | 230.5 (2.4 + 6.8 + 9.4 + 14.6 + 3.7 + 8.8 + 2.1 + 13.3 + 2.1 + 7.1, 10.5 + 13.1 + 3.8 + 8.8 + 2.7 + 13.0 + 2.2 + 6.8 + 11.1 + 13.3, 4.0 + 9.4 + 2.4 + 12.8 + 2.0 + 7.1 + 11.3 + 13.1 + 3.7 + 8.7) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 11 | 33 | f16 | Python + FlashInfer | 3664 (3567, 3664, 3579) | 9.0 | 93% | 16.9 GB | 245.8 (2.4 + 6.5 + 9.7 + 14.2 + 3.9 + 8.6 + 2.0 + 13.3 + 2.1 + 7.1 + 10.1, 12.8 + 4.0 + 8.8 + 2.7 + 12.8 + 2.2 + 6.8 + 11.2 + 13.0 + 4.1 + 9.0, 2.4 + 13.3 + 1.7 + 6.6 + 11.5 + 13.0 + 3.5 + 8.4 + 2.7 + 13.1 + 2.2) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 12 | 36 | f16 | Python + FlashInfer | 4123 (4037, 3993, 4123) | 8.7 | 93% | 16.9 GB | 279.1 (2.4 + 6.5 + 9.4 + 14.6 + 3.9 + 8.8 + 2.1 + 13.2 + 2.1 + 6.8 + 10.2 + 13.1, 3.7 + 8.8 + 2.7 + 13.0 + 2.2 + 6.8 + 11.2 + 13.3 + 4.2 + 9.2 + 2.0 + 13.3, 1.7 + 6.7 + 11.5 + 13.1 + 3.7 + 8.6 + 2.7 + 13.1 + 2.2 + 7.8 + 10.9 + 13.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 13 | 39 | f16 | Python + FlashInfer | 4354 (4313, 4250, 4354) | 9.0 | 92% | 16.9 GB | 293.3 (2.4 + 6.8 + 9.7 + 14.3 + 3.9 + 8.6 + 2.0 + 13.2 + 2.0 + 6.7 + 10.1 + 13.1 + 4.2, 8.8 + 2.7 + 12.8 + 2.2 + 6.8 + 11.2 + 13.2 + 4.2 + 9.1 + 2.1 + 13.5 + 1.7 + 6.9, 11.3 + 13.1 + 3.4 + 8.7 + 2.7 + 12.9 + 2.2 + 7.6 + 10.8 + 14.1 + 4.3 + 7.9 + 2.0) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 14 | 42 | f16 | Python + FlashInfer | 4709 (4644, 4709, 4515) | 8.9 | 93% | 16.9 GB | 314.9 (2.4 + 6.5 + 9.4 + 14.6 + 3.5 + 8.7 + 2.0 + 13.6 + 2.1 + 6.7 + 10.1 + 13.1 + 3.6 + 9.1, 2.1 + 12.7 + 2.1 + 6.8 + 11.0 + 13.4 + 4.1 + 9.4 + 2.1 + 13.5 + 2.1 + 6.9 + 11.2 + 12.8, 3.5 + 8.1 + 2.4 + 12.9 + 2.2 + 7.2 + 11.3 + 14.1 + 4.3 + 8.2 + 2.0 + 13.5 + 2.8 + 6.7) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 1 | 4 | f16 | Python + FlashInfer | 878 (647, 629, 878, 686) | 4.6 | 86% | 20.4 GB | 32.8 (2.4, 6.6, 9.5, 14.3) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 2 | 8 | f16 | Python + FlashInfer | 1002 (725, 1002, 846, 957) | 8.0 | 90% | 20.4 GB | 59.8 (2.4 + 6.5, 9.5 + 14.2, 3.4 + 8.3, 2.1 + 13.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 3 | 12 | f16 | Python + FlashInfer | 1433 (1223, 1406, 1221, 1433) | 8.4 | 90% | 20.4 GB | 92.9 (2.4 + 6.8 + 9.7, 14.3 + 3.5 + 8.4, 2.1 + 13.4 + 2.0, 6.8 + 10.3 + 13.1) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 4 | 16 | f16 | Python + FlashInfer | 1857 (1857, 1758, 1834, 1748) | 8.6 | 92% | 20.4 GB | 121.0 (2.4 + 6.8 + 9.7 + 14.6, 3.6 + 8.4 + 2.1 + 13.6, 1.8 + 6.8 + 10.3 + 13.1, 3.6 + 8.8 + 2.7 + 12.7) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 5 | 20 | f16 | Python + FlashInfer | 2349 (2213, 2115, 2293, 2349) | 8.5 | 99% | 20.4 GB | 154.9 (2.4 + 6.5 + 9.4 + 14.6 + 3.8, 8.8 + 2.1 + 13.3 + 2.1 + 7.1, 10.2 + 13.1 + 3.6 + 9.1 + 2.5, 12.8 + 2.2 + 6.8 + 11.2 + 13.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 6 | 24 | f16 | Python + FlashInfer | 2808 (2745, 2707, 2354, 2808) | 8.5 | 93% | 20.4 GB | 184.8 (2.4 + 6.8 + 9.5 + 14.6 + 3.9 + 8.8, 2.1 + 13.4 + 2.1 + 7.1 + 10.2 + 13.1, 3.7 + 9.1 + 2.7 + 12.8 + 2.2 + 6.7, 11.0 + 13.3 + 4.2 + 9.2 + 2.4 + 13.5) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 7 | 28 | f16 | Python + FlashInfer | 3184 (2967, 3184, 3093, 3158) | 8.8 | 94% | 20.4 GB | 216.8 (2.4 + 6.6 + 9.4 + 14.6 + 3.7 + 8.8 + 1.9, 13.2 + 2.1 + 6.8 + 10.2 + 12.8 + 4.2 + 8.8, 2.6 + 13.0 + 2.1 + 6.5 + 11.0 + 13.4 + 4.0, 9.2 + 2.4 + 13.5 + 1.8 + 7.1 + 11.5 + 12.9) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 8 | 32 | f16 | Python + FlashInfer | 3563 (3520, 3474, 3535, 3563) | 9.0 | 94% | 20.4 GB | 244.5 (2.2 + 6.8 + 9.7 + 14.4 + 3.6 + 8.8 + 2.1 + 13.6, 2.1 + 6.8 + 10.7 + 12.8 + 3.9 + 9.1 + 2.7 + 12.5, 2.2 + 6.8 + 10.9 + 13.4 + 3.9 + 9.2 + 2.4 + 13.2, 1.6 + 7.1 + 11.5 + 12.9 + 3.7 + 8.2 + 2.6 + 13.1) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 9 | 36 | f16 | Python + FlashInfer | 4076 (3838, 3877, 4017, 4076) | 8.8 | 94% | 20.4 GB | 280.4 (2.4 + 6.8 + 9.4 + 14.6 + 3.8 + 8.8 + 2.1 + 13.5 + 2.1, 6.6 + 10.2 + 13.1 + 4.2 + 8.6 + 2.7 + 12.7 + 2.2 + 6.8, 11.0 + 13.4 + 3.9 + 9.2 + 2.4 + 13.5 + 1.8 + 7.1 + 11.5, 12.8 + 3.3 + 8.7 + 2.7 + 12.8 + 2.2 + 8.0 + 11.3 + 14.0) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 10 | 40 | f16 | Python + FlashInfer | 4438 (4319, 4438, 4320, 4382) | 9.0 | 95% | 20.4 GB | 307.5 (2.4 + 6.8 + 9.4 + 14.6 + 3.7 + 8.8 + 2.1 + 13.3 + 2.1 + 7.1, 10.5 + 13.1 + 3.8 + 8.8 + 2.7 + 13.0 + 2.2 + 6.8 + 11.1 + 13.3, 4.0 + 9.4 + 2.4 + 12.8 + 2.0 + 7.1 + 11.3 + 13.1 + 3.7 + 8.7, 2.2 + 13.0 + 1.9 + 8.0 + 10.9 + 13.7 + 4.3 + 7.8 + 1.7 + 13.5) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 11 | 44 | f16 | Python + FlashInfer | 4962 (4717, 4875, 4697, 4962) | 8.9 | 92% | 20.4 GB | 340.3 (2.4 + 6.5 + 9.7 + 14.2 + 3.9 + 8.6 + 2.0 + 13.3 + 2.1 + 7.1 + 10.1, 12.8 + 4.0 + 8.8 + 2.7 + 12.8 + 2.2 + 6.8 + 11.2 + 13.0 + 4.1 + 9.0, 2.4 + 13.3 + 1.7 + 6.6 + 11.5 + 13.0 + 3.5 + 8.4 + 2.7 + 13.1 + 2.2, 7.5 + 11.6 + 13.9 + 4.2 + 8.2 + 1.7 + 13.5 + 2.9 + 6.7 + 11.1 + 13.2) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 12 | 48 | f16 | Python + FlashInfer | 5405 (5405, 5119, 5279, 5145) | 8.9 | 95% | 20.4 GB | 366.3 (2.4 + 6.5 + 9.4 + 14.6 + 3.9 + 8.8 + 2.1 + 13.2 + 2.1 + 6.8 + 10.2 + 13.1, 3.7 + 8.8 + 2.7 + 13.0 + 2.2 + 6.8 + 11.2 + 13.3 + 4.2 + 9.2 + 2.0 + 13.3, 1.7 + 6.7 + 11.5 + 13.1 + 3.7 + 8.6 + 2.7 + 13.1 + 2.2 + 7.8 + 10.9 + 13.6, 4.3 + 8.0 + 2.0 + 13.4 + 2.8 + 6.7 + 10.8 + 12.9 + 3.6 + 7.8 + 2.1 + 12.8) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 13 | 52 | f16 | Python + FlashInfer | 5757 (5541, 5629, 5737, 5757) | 9.0 | 96% | 20.4 GB | 400.8 (2.4 + 6.8 + 9.7 + 14.3 + 3.9 + 8.6 + 2.0 + 13.2 + 2.0 + 6.7 + 10.1 + 13.1 + 4.2, 8.8 + 2.7 + 12.8 + 2.2 + 6.8 + 11.2 + 13.2 + 4.2 + 9.1 + 2.1 + 13.5 + 1.7 + 6.9, 11.3 + 13.1 + 3.4 + 8.7 + 2.7 + 12.9 + 2.2 + 7.6 + 10.8 + 14.1 + 4.3 + 7.9 + 2.0, 13.2 + 2.9 + 6.7 + 11.1 + 12.9 + 3.6 + 8.0 + 2.6 + 13.1 + 1.7 + 7.5 + 11.3 + 13.0) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 14 | 56 | f16 | Python + FlashInfer | 6236 (6074, 6207, 5882, 6236) | 9.0 | 96% | 20.4 GB | 428.8 (2.4 + 6.5 + 9.4 + 14.6 + 3.5 + 8.7 + 2.0 + 13.6 + 2.1 + 6.7 + 10.1 + 13.1 + 3.6 + 9.1, 2.1 + 12.7 + 2.1 + 6.8 + 11.0 + 13.4 + 4.1 + 9.4 + 2.1 + 13.5 + 2.1 + 6.9 + 11.2 + 12.8, 3.5 + 8.1 + 2.4 + 12.9 + 2.2 + 7.2 + 11.3 + 14.1 + 4.3 + 8.2 + 2.0 + 13.5 + 2.8 + 6.7, 11.0 + 12.9 + 3.6 + 7.6 + 2.0 + 13.1 + 2.1 + 7.0 + 11.0 + 13.0 + 3.4 + 9.9 + 2.6 + 14.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 15 | 60 | f16 | Python + FlashInfer | 6784 (6433, 6709, 6600, 6784) | 8.8 | 96% | 20.4 GB | 468.9 (2.4 + 6.8 + 9.4 + 14.2 + 3.6 + 8.8 + 1.9 + 13.5 + 1.7 + 7.0 + 10.5 + 13.0 + 3.8 + 9.1 + 2.4, 12.8 + 2.2 + 6.4 + 11.1 + 13.2 + 4.2 + 9.4 + 2.4 + 13.4 + 2.1 + 7.1 + 11.5 + 12.7 + 3.7 + 8.1, 2.7 + 13.1 + 2.2 + 8.0 + 11.1 + 14.1 + 4.3 + 7.8 + 1.6 + 13.5 + 3.0 + 6.7 + 11.1 + 13.2 + 3.6, 7.7 + 2.6 + 13.1 + 2.1 + 7.3 + 11.3 + 13.0 + 3.8 + 9.8 + 2.9 + 14.8 + 3.0 + 7.6 + 12.3 + 13.5) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 16 | 64 | f16 | Python + FlashInfer | 7146 (7021, 7063, 7034, 7146) | 9.0 | 96% | 20.4 GB | 495.7 (2.4 + 6.5 + 9.7 + 14.6 + 3.9 + 8.7 + 1.6 + 13.3 + 2.1 + 7.0 + 10.4 + 13.1 + 3.9 + 8.8 + 2.7 + 12.8, 2.2 + 6.4 + 11.0 + 13.2 + 4.1 + 9.4 + 2.4 + 13.2 + 2.1 + 7.0 + 11.5 + 13.0 + 3.7 + 8.3 + 2.7 + 12.9, 2.2 + 7.4 + 11.4 + 14.1 + 4.3 + 7.6 + 1.5 + 13.5 + 3.2 + 6.4 + 11.1 + 13.1 + 3.2 + 7.7 + 2.1 + 13.0, 2.1 + 7.1 + 11.3 + 13.0 + 3.8 + 10.0 + 3.1 + 14.4 + 2.7 + 8.1 + 12.3 + 13.6 + 3.7 + 9.0 + 2.6 + 12.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 5 | 1 | 5 | f16 | Python + FlashInfer | 757 (634, 752, 683, 757, 646) | 6.6 | 88% | 21.9 GB | 36.2 (2.4, 6.6, 9.5, 14.3, 3.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 5 | 2 | 10 | f16 | Python + FlashInfer | 1170 (908, 1170, 1017, 1109, 900) | 8.6 | 92% | 21.9 GB | 68.9 (2.4 + 6.5, 9.5 + 14.2, 3.4 + 8.3, 2.1 + 13.4, 2.0 + 7.1) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 5 | 3 | 15 | f16 | Python + FlashInfer | 1687 (1442, 1674, 1471, 1687, 1349) | 8.9 | 92% | 21.9 GB | 108.0 (2.4 + 6.8 + 9.7, 14.3 + 3.5 + 8.4, 2.1 + 13.4 + 2.0, 6.8 + 10.3 + 13.1, 3.7 + 8.8 + 2.7) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 5 | 4 | 20 | f16 | Python + FlashInfer | 2308 (2300, 2178, 2303, 2189, 2308) | 8.7 | 94% | 21.9 GB | 154.3 (2.4 + 6.8 + 9.7 + 14.6, 3.6 + 8.4 + 2.1 + 13.6, 1.8 + 6.8 + 10.3 + 13.1, 3.6 + 8.8 + 2.7 + 12.7, 2.2 + 6.8 + 11.2 + 13.2) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 5 | 5 | 25 | f16 | Python + FlashInfer | 2789 (2663, 2567, 2696, 2789, 2406) | 9.0 | 95% | 21.9 GB | 185.6 (2.4 + 6.5 + 9.4 + 14.6 + 3.8, 8.8 + 2.1 + 13.3 + 2.1 + 7.1, 10.2 + 13.1 + 3.6 + 9.1 + 2.5, 12.8 + 2.2 + 6.8 + 11.2 + 13.4, 4.2 + 9.2 + 1.9 + 13.3 + 2.1) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 5 | 6 | 30 | f16 | Python + FlashInfer | 3392 (3301, 3374, 2836, 3392, 3303) | 8.8 | 94% | 21.9 GB | 230.7 (2.4 + 6.8 + 9.5 + 14.6 + 3.9 + 8.8, 2.1 + 13.4 + 2.1 + 7.1 + 10.2 + 13.1, 3.7 + 9.1 + 2.7 + 12.8 + 2.2 + 6.7, 11.0 + 13.3 + 4.2 + 9.2 + 2.4 + 13.5, 2.1 + 7.1 + 11.5 + 12.9 + 3.7 + 8.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 5 | 7 | 35 | f16 | Python + FlashInfer | 3863 (3614, 3863, 3768, 3827, 3635) | 9.1 | 94% | 21.9 GB | 265.6 (2.4 + 6.6 + 9.4 + 14.6 + 3.7 + 8.8 + 1.9, 13.2 + 2.1 + 6.8 + 10.2 + 12.8 + 4.2 + 8.8, 2.6 + 13.0 + 2.1 + 6.5 + 11.0 + 13.4 + 4.0, 9.2 + 2.4 + 13.5 + 1.8 + 7.1 + 11.5 + 12.9, 3.7 + 8.4 + 2.7 + 12.9 + 2.2 + 8.0 + 10.9) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 5 | 8 | 40 | f16 | Python + FlashInfer | 4383 (4351, 4327, 4367, 4379, 4383) | 9.1 | 97% | 21.9 GB | 306.7 (2.2 + 6.8 + 9.7 + 14.4 + 3.6 + 8.8 + 2.1 + 13.6, 2.1 + 6.8 + 10.7 + 12.8 + 3.9 + 9.1 + 2.7 + 12.5, 2.2 + 6.8 + 10.9 + 13.4 + 3.9 + 9.2 + 2.4 + 13.2, 1.6 + 7.1 + 11.5 + 12.9 + 3.7 + 8.2 + 2.6 + 13.1, 2.2 + 7.4 + 11.3 + 13.8 + 4.3 + 8.2 + 1.7 + 13.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 5 | 9 | 45 | f16 | Python + FlashInfer | 4984 (4694, 4747, 4928, 4984, 4707) | 9.0 | 95% | 21.9 GB | 344.5 (2.4 + 6.8 + 9.4 + 14.6 + 3.8 + 8.8 + 2.1 + 13.5 + 2.1, 6.6 + 10.2 + 13.1 + 4.2 + 8.6 + 2.7 + 12.7 + 2.2 + 6.8, 11.0 + 13.4 + 3.9 + 9.2 + 2.4 + 13.5 + 1.8 + 7.1 + 11.5, 12.8 + 3.3 + 8.7 + 2.7 + 12.8 + 2.2 + 8.0 + 11.3 + 14.0, 4.0 + 8.0 + 2.0 + 13.3 + 2.8 + 6.4 + 11.1 + 12.9 + 3.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 5 | 10 | 50 | f16 | Python + FlashInfer | 5434 (5117, 5434, 5318, 5321, 5086) | 9.2 | 95% | 21.9 GB | 376.6 (2.4 + 6.8 + 9.4 + 14.6 + 3.7 + 8.8 + 2.1 + 13.3 + 2.1 + 7.1, 10.5 + 13.1 + 3.8 + 8.8 + 2.7 + 13.0 + 2.2 + 6.8 + 11.1 + 13.3, 4.0 + 9.4 + 2.4 + 12.8 + 2.0 + 7.1 + 11.3 + 13.1 + 3.7 + 8.7, 2.2 + 13.0 + 1.9 + 8.0 + 10.9 + 13.7 + 4.3 + 7.8 + 1.7 + 13.5, 2.7 + 6.6 + 11.1 + 12.9 + 3.2 + 8.0 + 2.3 + 13.1 + 2.1 + 7.2) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 5 | 11 | 55 | f16 | Python + FlashInfer | 6012 (5769, 5893, 5710, 6012, 5541) | 9.1 | 94% | 21.9 GB | 416.2 (2.4 + 6.5 + 9.7 + 14.2 + 3.9 + 8.6 + 2.0 + 13.3 + 2.1 + 7.1 + 10.1, 12.8 + 4.0 + 8.8 + 2.7 + 12.8 + 2.2 + 6.8 + 11.2 + 13.0 + 4.1 + 9.0, 2.4 + 13.3 + 1.7 + 6.6 + 11.5 + 13.0 + 3.5 + 8.4 + 2.7 + 13.1 + 2.2, 7.5 + 11.6 + 13.9 + 4.2 + 8.2 + 1.7 + 13.5 + 2.9 + 6.7 + 11.1 + 13.2, 3.6 + 7.7 + 2.2 + 13.1 + 2.1 + 7.2 + 11.1 + 13.0 + 3.5 + 9.8 + 2.5) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 5 | 12 | 60 | f16 | Python + FlashInfer | 6670 (6513, 6338, 6560, 6385, 6670) | 9.0 | 97% | 21.9 GB | 466.6 (2.4 + 6.5 + 9.4 + 14.6 + 3.9 + 8.8 + 2.1 + 13.2 + 2.1 + 6.8 + 10.2 + 13.1, 3.7 + 8.8 + 2.7 + 13.0 + 2.2 + 6.8 + 11.2 + 13.3 + 4.2 + 9.2 + 2.0 + 13.3, 1.7 + 6.7 + 11.5 + 13.1 + 3.7 + 8.6 + 2.7 + 13.1 + 2.2 + 7.8 + 10.9 + 13.6, 4.3 + 8.0 + 2.0 + 13.4 + 2.8 + 6.7 + 10.8 + 12.9 + 3.6 + 7.8 + 2.1 + 12.8, 2.0 + 7.0 + 10.8 + 13.0 + 3.8 + 10.0 + 3.1 + 14.6 + 2.4 + 8.1 + 12.0 + 13.7) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 5 | 13 | 65 | f16 | Python + FlashInfer | 7077 (6924, 6832, 7039, 7077, 6913) | 9.2 | 96% | 21.9 GB | 499.1 (2.4 + 6.8 + 9.7 + 14.3 + 3.9 + 8.6 + 2.0 + 13.2 + 2.0 + 6.7 + 10.1 + 13.1 + 4.2, 8.8 + 2.7 + 12.8 + 2.2 + 6.8 + 11.2 + 13.2 + 4.2 + 9.1 + 2.1 + 13.5 + 1.7 + 6.9, 11.3 + 13.1 + 3.4 + 8.7 + 2.7 + 12.9 + 2.2 + 7.6 + 10.8 + 14.1 + 4.3 + 7.9 + 2.0, 13.2 + 2.9 + 6.7 + 11.1 + 12.9 + 3.6 + 8.0 + 2.6 + 13.1 + 1.7 + 7.5 + 11.3 + 13.0, 3.8 + 10.0 + 3.1 + 15.2 + 3.1 + 7.9 + 12.3 + 13.6 + 3.3 + 8.8 + 2.3 + 12.6 + 2.3) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 6 | 1 | 6 | f16 | Python + FlashInfer | 924 (665, 779, 838, 924, 664, 812) | 6.5 | 87% | 20.1 GB | 44.6 (2.4, 6.6, 9.5, 14.3, 3.4, 8.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 6 | 2 | 12 | f16 | Python + FlashInfer | 1504 (1089, 1476, 1244, 1383, 1085, 1504) | 8.0 | 92% | 20.5 GB | 92.2 (2.4 + 6.5, 9.5 + 14.2, 3.4 + 8.3, 2.1 + 13.4, 2.0 + 7.1, 10.2 + 13.0) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 6 | 3 | 18 | f16 | Python + FlashInfer | 2011 (1761, 1973, 1760, 2011, 1618, 1813) | 9.0 | 93% | 20.6 GB | 129.3 (2.4 + 6.8 + 9.7, 14.3 + 3.5 + 8.4, 2.1 + 13.4 + 2.0, 6.8 + 10.3 + 13.1, 3.7 + 8.8 + 2.7, 12.7 + 2.2 + 6.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 6 | 4 | 24 | f16 | Python + FlashInfer | 2724 (2702, 2610, 2715, 2572, 2724, 2617) | 8.8 | 94% | 21.2 GB | 183.3 (2.4 + 6.8 + 9.7 + 14.6, 3.6 + 8.4 + 2.1 + 13.6, 1.8 + 6.8 + 10.3 + 13.1, 3.6 + 8.8 + 2.7 + 12.7, 2.2 + 6.8 + 11.2 + 13.2, 4.1 + 9.4 + 2.4 + 13.1) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 6 | 5 | 30 | f16 | Python + FlashInfer | 3385 (3232, 3109, 3241, 3385, 2930, 3365) | 8.9 | 95% | 21.6 GB | 228.8 (2.4 + 6.5 + 9.4 + 14.6 + 3.8, 8.8 + 2.1 + 13.3 + 2.1 + 7.1, 10.2 + 13.1 + 3.6 + 9.1 + 2.5, 12.8 + 2.2 + 6.8 + 11.2 + 13.4, 4.2 + 9.2 + 1.9 + 13.3 + 2.1, 7.0 + 11.3 + 12.9 + 3.7 + 8.2) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 6 | 6 | 36 | f16 | Python + FlashInfer | 4165 (4001, 4020, 3497, 4165, 3954, 4074) | 8.6 | 95% | 22.5 GB | 282.0 (2.4 + 6.8 + 9.5 + 14.6 + 3.9 + 8.8, 2.1 + 13.4 + 2.1 + 7.1 + 10.2 + 13.1, 3.7 + 9.1 + 2.7 + 12.8 + 2.2 + 6.7, 11.0 + 13.3 + 4.2 + 9.2 + 2.4 + 13.5, 2.1 + 7.1 + 11.5 + 12.9 + 3.7 + 8.6, 2.5 + 13.0 + 2.2 + 8.0 + 11.6 + 14.1) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 1 | 1 | bf16 | Python + FlashInfer | failed: bf16 not supported: upstream `apply_flashinfer` plans attention as f16 only | – | – | – | – |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 1 | 1 | f16 | Python + FlashInfer + CUDA graphs | 201 | 5.0 | 68% | 17.6 GB | 2.4 (2.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 2 | 2 | f16 | Python + FlashInfer + CUDA graphs | 377 | 5.3 | 83% | 17.6 GB | 9.2 (2.4 + 6.8) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 3 | 3 | f16 | Python + FlashInfer + CUDA graphs | 671 | 4.5 | 85% | 17.6 GB | 18.7 (2.4 + 6.5 + 9.7) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 4 | 4 | f16 | Python + FlashInfer + CUDA graphs | 1023 | 3.9 | 86% | 17.6 GB | 33.2 (2.4 + 6.4 + 9.7 + 14.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 5 | 5 | f16 | Python + FlashInfer + CUDA graphs | 1258 | 4.0 | 89% | 17.6 GB | 36.6 (2.4 + 6.3 + 9.7 + 14.6 + 3.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 1 | 2 | f16 | Python + FlashInfer + CUDA graphs | 376 (350, 376) | 5.3 | 84% | 23.4 GB | 8.7 (2.4, 6.2) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 2 | 4 | f16 | Python + FlashInfer + CUDA graphs | 908 (746, 908) | 4.4 | 91% | 23.4 GB | 33.5 (2.4 + 6.8, 9.6 + 14.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 3 | 6 | f16 | Python + FlashInfer + CUDA graphs | 1453 (1381, 1453) | 4.1 | 93% | 23.4 GB | 45.8 (2.4 + 6.5 + 9.7, 14.4 + 3.9 + 8.8) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 4 | 8 | f16 | Python + FlashInfer + CUDA graphs | 2043 (2037, 2043) | 3.9 | 93% | 23.4 GB | 60.9 (2.4 + 6.4 + 9.7 + 14.6, 3.8 + 8.4 + 1.8 + 13.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 1 | 3 | f16 | Python + FlashInfer + CUDA graphs | 611 (510, 595, 611) | 4.9 | 91% | 10.6 GB | 18.1 (2.4, 6.2, 9.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 2 | 6 | f16 | Python + FlashInfer + CUDA graphs | 1321 (1114, 1321, 1245) | 4.5 | 95% | 11.2 GB | 45.9 (2.4 + 6.8, 9.6 + 14.6, 3.9 + 8.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 3 | 9 | f16 | Python + FlashInfer + CUDA graphs | 2163 (2001, 2163, 2150) | 4.2 | 95% | 11.6 GB | 62.8 (2.4 + 6.5 + 9.7, 14.4 + 3.9 + 8.8, 2.1 + 13.2 + 1.7) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 4 | 12 | f16 | Python + FlashInfer + CUDA graphs | 3003 (3003, 2934, 2987) | 4.0 | 96% | 12.5 GB | 92.3 (2.4 + 6.4 + 9.7 + 14.6, 3.8 + 8.4 + 1.8 + 13.6, 1.8 + 6.8 + 10.0 + 12.8) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 1 | 4 | f16 | Python + FlashInfer + CUDA graphs | 883 (630, 749, 838, 883) | 4.5 | 90% | 14.3 GB | 32.5 (2.4, 6.2, 9.4, 14.5) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 2 | 8 | f16 | Python + FlashInfer + CUDA graphs | 1822 (1407, 1822, 1671, 1787) | 4.4 | 95% | 15.0 GB | 61.7 (2.4 + 6.8, 9.6 + 14.6, 3.9 + 8.6, 2.1 + 13.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 3 | 12 | f16 | Python + FlashInfer + CUDA graphs | 2887 (2603, 2887, 2875, 2882) | 4.2 | 96% | 15.4 GB | 92.1 (2.4 + 6.5 + 9.7, 14.4 + 3.9 + 8.8, 2.1 + 13.2 + 1.7, 6.5 + 10.2 + 12.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 4 | 16 | f16 | Python + FlashInfer + CUDA graphs | 3933 (3933, 3904, 3926, 3911) | 4.1 | 98% | 16.6 GB | 120.1 (2.4 + 6.4 + 9.7 + 14.6, 3.8 + 8.4 + 1.8 + 13.6, 1.8 + 6.8 + 10.0 + 12.8, 3.7 + 9.1 + 2.0 + 13.0) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 1 | 1 | bf16 | Python + FlashInfer + CUDA graphs | 185 | 5.4 | 75% | 17.8 GB | 2.4 (2.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 2 | 2 | bf16 | Python + FlashInfer + CUDA graphs | 381 | 5.3 | 83% | 17.8 GB | 8.8 (2.4 + 6.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 3 | 3 | bf16 | Python + FlashInfer + CUDA graphs | 675 | 4.4 | 88% | 17.8 GB | 18.7 (2.2 + 6.8 + 9.7) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 4 | 4 | bf16 | Python + FlashInfer + CUDA graphs | 1026 | 3.9 | 87% | 17.8 GB | 33.4 (2.4 + 6.8 + 9.5 + 14.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 1 | 2 | bf16 | Python + FlashInfer + CUDA graphs | 375 (323, 375) | 5.3 | 84% | 19.9 GB | 8.9 (2.4, 6.5) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 2 | 4 | bf16 | Python + FlashInfer + CUDA graphs | 891 (739, 891) | 4.5 | 91% | 19.9 GB | 33.1 (2.4 + 6.4, 9.7 + 14.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 3 | 6 | bf16 | Python + FlashInfer + CUDA graphs | 1439 (1350, 1439) | 4.2 | 93% | 19.9 GB | 46.0 (2.2 + 6.8 + 9.7, 14.6 + 3.9 + 8.8) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 4 | 8 | bf16 | Python + FlashInfer + CUDA graphs | 2035 (2035, 2034) | 3.9 | 95% | 19.9 GB | 61.2 (2.4 + 6.8 + 9.5 + 14.6, 3.8 + 8.8 + 1.9 + 13.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 1 | 3 | bf16 | Python + FlashInfer + CUDA graphs | 612 (498, 572, 612) | 4.9 | 90% | 22.4 GB | 18.4 (2.4, 6.5, 9.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 2 | 6 | bf16 | Python + FlashInfer + CUDA graphs | 1329 (1089, 1329, 1248) | 4.5 | 91% | 22.9 GB | 45.1 (2.4 + 6.4, 9.7 + 14.6, 3.4 + 8.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 3 | 9 | bf16 | Python + FlashInfer + CUDA graphs | 2199 (1982, 2199, 2147) | 4.1 | 94% | 23.1 GB | 63.1 (2.2 + 6.8 + 9.7, 14.6 + 3.9 + 8.8, 2.1 + 13.3 + 1.7) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 4 | 12 | bf16 | Python + FlashInfer + CUDA graphs | 3033 (3033, 3005, 2987) | 4.0 | 97% | 23.2 GB | 94.0 (2.4 + 6.8 + 9.5 + 14.6, 3.8 + 8.8 + 1.9 + 13.4, 2.1 + 7.1 + 10.6 + 13.1) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 1 | 1 | f16 | C++ (`perf_mode=flash_attention`) | 101 | 9.9 | 88% | 2.5 GB | 2.4 (2.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 2 | 2 | f16 | C++ (`perf_mode=flash_attention`) | 232 | 8.6 | 86% | 2.5 GB | 9.2 (2.4 + 6.8) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 3 | 3 | f16 | C++ (`perf_mode=flash_attention`) | 400 | 7.5 | 88% | 2.5 GB | 18.9 (2.4 + 6.8 + 9.7) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 4 | 4 | f16 | C++ (`perf_mode=flash_attention`) | 744 | 5.4 | 85% | 2.6 GB | 33.6 (2.4 + 6.8 + 9.7 + 14.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 1 | 2 | f16 | C++ (`perf_mode=flash_attention`) | 252 (239, 252) | 7.9 | 93% | 4.9 GB | 9.2 (2.4, 6.8) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 2 | 4 | f16 | C++ (`perf_mode=flash_attention`) | 692 (602, 692) | 5.8 | 94% | 5.1 GB | 33.6 (2.4 + 6.8, 9.7 + 14.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 3 | 6 | f16 | C++ (`perf_mode=flash_attention`) | 991 (941, 991) | 6.1 | 95% | 5.1 GB | 46.3 (2.4 + 6.8 + 9.7, 14.6 + 3.9 + 8.8) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 4 | 8 | f16 | C++ (`perf_mode=flash_attention`) | 1393 (1393, 1368) | 5.7 | 96% | 5.3 GB | 62.1 (2.4 + 6.8 + 9.7 + 14.6, 3.9 + 8.8 + 2.1 + 13.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 1 | 1 | bf16 | C++ (`perf_mode=flash_attention`) | 104 | 9.6 | 90% | 2.5 GB | 2.4 (2.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 2 | 2 | bf16 | C++ (`perf_mode=flash_attention`) | 244 | 8.2 | 90% | 2.5 GB | 9.2 (2.4 + 6.8) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 3 | 3 | bf16 | C++ (`perf_mode=flash_attention`) | 457 | 6.6 | 88% | 2.6 GB | 18.9 (2.4 + 6.8 + 9.7) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 4 | 4 | bf16 | C++ (`perf_mode=flash_attention`) | 763 | 5.2 | 89% | 2.7 GB | 33.5 (2.4 + 6.8 + 9.7 + 14.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 1 | 2 | bf16 | C++ (`perf_mode=flash_attention`) | 268 (257, 268) | 7.5 | 94% | 4.9 GB | 9.2 (2.4, 6.8) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 2 | 4 | bf16 | C++ (`perf_mode=flash_attention`) | 757 (625, 757) | 5.3 | 94% | 5.1 GB | 33.5 (2.4 + 6.8, 9.7 + 14.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 3 | 6 | bf16 | C++ (`perf_mode=flash_attention`) | 1096 (1069, 1096) | 5.5 | 95% | 5.1 GB | 46.3 (2.4 + 6.8 + 9.7, 14.6 + 3.9 + 8.8) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 4 | 8 | bf16 | C++ (`perf_mode=flash_attention`) | 1485 (1476, 1485) | 5.4 | 95% | 5.3 GB | 61.7 (2.4 + 6.8 + 9.7 + 14.6, 3.6 + 8.8 + 2.1 + 13.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 1 | 1 | f16 | C++ (`perf_mode=off`) | 126 | 7.9 | 87% | 2.5 GB | 2.4 (2.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 2 | 2 | f16 | C++ (`perf_mode=off`) | 296 | 6.8 | 90% | 2.5 GB | 9.3 (2.4 + 6.8) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 3 | 3 | f16 | C++ (`perf_mode=off`) | 556 | 5.4 | 92% | 2.5 GB | 18.9 (2.4 + 6.8 + 9.7) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 4 | 4 | f16 | C++ (`perf_mode=off`) | 998 | 4.0 | 92% | 2.6 GB | 33.6 (2.4 + 6.8 + 9.7 + 14.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 1 | 2 | f16 | C++ (`perf_mode=off`) | 317 (304, 317) | 6.3 | 95% | 4.9 GB | 9.2 (2.4, 6.8) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 2 | 4 | f16 | C++ (`perf_mode=off`) | 977 (674, 977) | 4.1 | 95% | 5.1 GB | 33.6 (2.4 + 6.8, 9.7 + 14.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 3 | 6 | f16 | C++ (`perf_mode=off`) | 1403 (1388, 1403) | 4.3 | 95% | 5.1 GB | 46.2 (2.4 + 6.8 + 9.7, 14.6 + 3.9 + 8.8) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 4 | 8 | f16 | C++ (`perf_mode=off`) | 1945 (1945, 1900) | 4.1 | 97% | 5.3 GB | 62.1 (2.4 + 6.8 + 9.7 + 14.6, 3.9 + 8.8 + 2.1 + 13.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 1 | 1 | bf16 | C++ (`perf_mode=off`) | 126 | 7.9 | 90% | 2.5 GB | 2.4 (2.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 2 | 2 | bf16 | C++ (`perf_mode=off`) | 306 | 6.5 | 92% | 2.5 GB | 9.2 (2.4 + 6.8) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 3 | 3 | bf16 | C++ (`perf_mode=off`) | 612 | 4.9 | 92% | 2.6 GB | 18.9 (2.4 + 6.8 + 9.7) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 4 | 4 | bf16 | C++ (`perf_mode=off`) | 1062 | 3.8 | 92% | 2.6 GB | 33.5 (2.4 + 6.8 + 9.7 + 14.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 1 | 2 | bf16 | C++ (`perf_mode=off`) | 339 (328, 339) | 5.9 | 96% | 4.9 GB | 9.2 (2.4, 6.8) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 2 | 4 | bf16 | C++ (`perf_mode=off`) | 1050 (716, 1050) | 3.8 | 95% | 5.1 GB | 33.5 (2.4 + 6.8, 9.7 + 14.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 3 | 6 | bf16 | C++ (`perf_mode=off`) | 1513 (1486, 1513) | 4.0 | 96% | 5.1 GB | 46.3 (2.4 + 6.8 + 9.7, 14.6 + 3.9 + 8.8) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 4 | 8 | bf16 | C++ (`perf_mode=off`) | 2054 (2054, 2008) | 3.9 | 97% | 5.3 GB | 61.7 (2.4 + 6.8 + 9.7 + 14.6, 3.6 + 8.8 + 2.1 + 13.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 1 | 1 | f16 | Python (PyTorch) | 968 | 1.0 | 21% | 6.5 GB | 2.4 (2.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 2 | 2 | f16 | Python (PyTorch) | 1004 | 2.0 | 39% | 6.5 GB | 9.0 (2.4 + 6.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 3 | 3 | f16 | Python (PyTorch) | 1058 | 2.8 | 69% | 6.5 GB | 18.4 (2.2 + 6.6 + 9.7) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 4 | 4 | f16 | Python (PyTorch) | 1558 | 2.6 | 91% | 6.5 GB | 33.3 (2.4 + 6.5 + 9.7 + 14.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 5 | 5 | f16 | Python (PyTorch) | 1992 | 2.5 | 92% | 6.5 GB | 37.0 (2.4 + 6.6 + 9.7 + 14.6 + 3.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 6 | 6 | f16 | Python (PyTorch) | 2445 | 2.5 | 92% | 6.5 GB | 44.8 (2.4 + 6.4 + 9.5 + 14.3 + 3.9 + 8.3) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 1 | 2 | f16 | Python (PyTorch) | 1013 (1013, 994) | 2.0 | 94% | 11.0 GB | 9.0 (2.4, 6.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 2 | 4 | f16 | Python (PyTorch) | 1245 (1181, 1245) | 3.2 | 94% | 11.0 GB | 32.9 (2.4 + 6.6, 9.5 + 14.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 3 | 6 | f16 | Python (PyTorch) | 1990 (1694, 1990) | 3.0 | 100% | 11.0 GB | 45.5 (2.2 + 6.6 + 9.7, 14.6 + 3.6 + 8.8) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 4 | 8 | f16 | Python (PyTorch) | 3085 (3085, 2953) | 2.6 | 96% | 11.0 GB | 61.4 (2.4 + 6.5 + 9.7 + 14.6, 3.8 + 8.8 + 2.1 + 13.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 5 | 10 | f16 | Python (PyTorch) | 4016 (4016, 3876) | 2.5 | 96% | 11.0 GB | 70.0 (2.4 + 6.6 + 9.7 + 14.6 + 3.6, 8.5 + 2.0 + 13.4 + 2.1 + 7.1) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 1 | 3 | f16 | Python (PyTorch) | 1505 (1434, 1406, 1505) | 2.0 | 89% | 16.4 GB | 18.5 (2.4, 6.6, 9.5) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 2 | 6 | f16 | Python (PyTorch) | 1751 (1400, 1751, 1704) | 3.4 | 89% | 16.4 GB | 44.9 (2.4 + 6.6, 9.5 + 14.4, 3.3 + 8.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 3 | 9 | f16 | Python (PyTorch) | 3085 (2537, 3085, 3029) | 2.9 | 96% | 16.4 GB | 63.1 (2.2 + 6.6 + 9.7, 14.6 + 3.6 + 8.8, 2.0 + 13.5 + 2.1) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 4 | 12 | f16 | Python (PyTorch) | 4472 (4472, 4345, 4297) | 2.7 | 97% | 16.4 GB | 93.0 (2.4 + 6.5 + 9.7 + 14.6, 3.8 + 8.8 + 2.1 + 13.4, 2.1 + 6.7 + 10.0 + 12.8) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 5 | 15 | f16 | Python (PyTorch) | 5830 (5830, 5613, 5409) | 2.6 | 97% | 16.4 GB | 109.2 (2.4 + 6.6 + 9.7 + 14.6 + 3.6, 8.5 + 2.0 + 13.4 + 2.1 + 7.1, 10.6 + 12.7 + 4.2 + 9.1 + 2.7) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 1 | 4 | f16 | Python (PyTorch) | 1428 (1215, 1428, 1389, 1199) | 2.8 | 95% | 23.3 GB | 32.8 (2.4, 6.6, 9.5, 14.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 2 | 8 | f16 | Python (PyTorch) | 2455 (1776, 2455, 2026, 2339) | 3.3 | 95% | 23.3 GB | 60.6 (2.4 + 6.6, 9.5 + 14.4, 3.3 + 8.6, 2.1 + 13.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 3 | 12 | f16 | Python (PyTorch) | 4056 (3272, 4056, 3992, 3898) | 3.0 | 97% | 23.3 GB | 93.2 (2.2 + 6.6 + 9.7, 14.6 + 3.6 + 8.8, 2.0 + 13.5 + 2.1, 6.6 + 10.4 + 13.1) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 4 | 16 | f16 | Python (PyTorch) | 5825 (5825, 5658, 5606, 5457) | 2.7 | 96% | 23.3 GB | 120.7 (2.4 + 6.5 + 9.7 + 14.6, 3.8 + 8.8 + 2.1 + 13.4, 2.1 + 6.7 + 10.0 + 12.8, 3.6 + 8.6 + 2.5 + 13.0) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 5 | 20 | f16 | Python (PyTorch) | 7605 (7605, 7428, 7267, 7491) | 2.6 | 98% | 23.3 GB | 154.9 (2.4 + 6.6 + 9.7 + 14.6 + 3.6, 8.5 + 2.0 + 13.4 + 2.1 + 7.1, 10.6 + 12.7 + 4.2 + 9.1 + 2.7, 13.0 + 1.9 + 6.4 + 11.0 + 13.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 1 | 1 | bf16 | Python (PyTorch) | 955 | 1.0 | 21% | 7.5 GB | 2.2 (2.2) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 2 | 2 | bf16 | Python (PyTorch) | 973 | 2.1 | 40% | 7.5 GB | 8.8 (2.1 + 6.7) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 3 | 3 | bf16 | Python (PyTorch) | 1029 | 2.9 | 69% | 7.5 GB | 18.3 (2.4 + 6.5 + 9.5) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 4 | 4 | bf16 | Python (PyTorch) | 1544 | 2.6 | 94% | 7.5 GB | 33.6 (2.4 + 6.8 + 9.7 + 14.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 5 | 5 | bf16 | Python (PyTorch) | 2000 | 2.5 | 91% | 7.5 GB | 37.4 (2.4 + 6.7 + 9.7 + 14.6 + 3.9) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 1 | 6 | 6 | bf16 | Python (PyTorch) | 2427 | 2.5 | 93% | 7.5 GB | 44.9 (2.4 + 6.4 + 9.7 + 14.3 + 3.9 + 8.3) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 1 | 2 | bf16 | Python (PyTorch) | 1148 (1148, 1029) | 1.7 | 72% | 12.1 GB | 8.6 (2.2, 6.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 2 | 4 | bf16 | Python (PyTorch) | 1290 (1290, 1218) | 3.1 | 92% | 12.1 GB | 32.8 (2.1 + 6.7, 9.5 + 14.5) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 3 | 6 | bf16 | Python (PyTorch) | 1979 (1656, 1979) | 3.0 | 95% | 12.1 GB | 45.3 (2.4 + 6.5 + 9.5, 14.4 + 3.8 + 8.8) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 4 | 8 | bf16 | Python (PyTorch) | 3124 (3124, 3005) | 2.6 | 98% | 12.1 GB | 61.8 (2.4 + 6.8 + 9.7 + 14.6, 3.6 + 8.8 + 2.1 + 13.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 2 | 5 | 10 | bf16 | Python (PyTorch) | 4002 (4002, 3798) | 2.5 | 96% | 12.1 GB | 70.3 (2.4 + 6.7 + 9.7 + 14.6 + 3.9, 8.5 + 2.0 + 13.6 + 1.8 + 7.1) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 1 | 3 | bf16 | Python (PyTorch) | 1244 (1206, 1244, 1210) | 2.4 | 88% | 16.4 GB | 18.0 (2.2, 6.4, 9.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 2 | 6 | bf16 | Python (PyTorch) | 1770 (1502, 1770, 1607) | 3.4 | 88% | 16.4 GB | 45.0 (2.1 + 6.7, 9.5 + 14.5, 3.4 + 8.8) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 3 | 9 | bf16 | Python (PyTorch) | 3020 (2432, 3020, 2946) | 3.0 | 100% | 16.4 GB | 63.1 (2.4 + 6.5 + 9.5, 14.4 + 3.8 + 8.8, 2.1 + 13.6 + 2.1) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 4 | 12 | bf16 | Python (PyTorch) | 4468 (4468, 4319, 4312) | 2.7 | 96% | 16.4 GB | 94.0 (2.4 + 6.8 + 9.7 + 14.6, 3.6 + 8.8 + 2.1 + 13.6, 2.1 + 7.1 + 10.0 + 13.1) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 3 | 5 | 15 | bf16 | Python (PyTorch) | 5772 (5772, 5646, 5478) | 2.6 | 97% | 16.4 GB | 109.1 (2.4 + 6.7 + 9.7 + 14.6 + 3.9, 8.5 + 2.0 + 13.6 + 1.8 + 7.1, 10.7 + 12.6 + 3.9 + 8.8 + 2.7) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 1 | 4 | bf16 | Python (PyTorch) | 1586 (1513, 1231, 1251, 1586) | 2.5 | 95% | 22.5 GB | 32.4 (2.2, 6.4, 9.4, 14.4) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 2 | 8 | bf16 | Python (PyTorch) | 2410 (1756, 2410, 1963, 2344) | 3.3 | 96% | 22.5 GB | 60.7 (2.1 + 6.7, 9.5 + 14.5, 3.4 + 8.8, 2.1 + 13.6) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 3 | 12 | bf16 | Python (PyTorch) | 4017 (3233, 4017, 3905, 3885) | 3.0 | 97% | 22.5 GB | 93.2 (2.4 + 6.5 + 9.5, 14.4 + 3.8 + 8.8, 2.1 + 13.6 + 2.1, 6.6 + 10.4 + 13.1) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 4 | 16 | bf16 | Python (PyTorch) | 5795 (5795, 5665, 5608, 5347) | 2.8 | 98% | 22.5 GB | 121.5 (2.4 + 6.8 + 9.7 + 14.6, 3.6 + 8.8 + 2.1 + 13.6, 2.1 + 7.1 + 10.0 + 13.1, 3.5 + 8.5 + 2.5 + 13.0) |
| Mixed: 5, 15, 25, 30, 10, 20, 5, 30 words, repeating | 4 | 5 | 20 | bf16 | Python (PyTorch) | 7608 (7608, 7402, 7201, 7401) | 2.6 | 98% | 22.5 GB | 154.7 (2.4 + 6.7 + 9.7 + 14.6 + 3.9, 8.5 + 2.0 + 13.6 + 1.8 + 7.1, 10.7 + 12.6 + 3.9 + 8.8 + 2.7, 13.0 + 1.9 + 6.4 + 11.0 + 13.4) |
