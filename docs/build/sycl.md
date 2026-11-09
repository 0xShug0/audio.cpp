# SYCL (Intel GPUs)

SYCL is disabled by default. It runs ggml's SYCL backend through the Intel oneAPI DPC++ compiler and Level Zero.

## Requirements

- Intel oneAPI Base Toolkit with the DPC++ compiler (`icx`/`icpx`), oneMKL, and oneDNN.
- An Intel GPU with a working Level Zero driver.

## Build

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build/sycl -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DENGINE_ENABLE_SYCL=ON
cmake --build build/sycl --target audiocpp_cli -j 8
```

oneDNN is used for SYCL matrix multiplication when ggml finds it (`GGML_SYCL_DNN`, on by default). `ENGINE_ENABLE_SYCL` also turns on `GGML_SYCL_F16`, which runs those matmuls in F16. Pass `-DGGML_SYCL_F16=OFF` to keep them in F32.

## Run

Load the oneAPI environment in the shell that runs the binary, then pass `--backend sycl`. Use `--list-devices` and `--device <index>` to pick a GPU. To hide other SYCL devices, set `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` (or `level_zero:<index>`).

## Runtime controls

These environment variables tune the SYCL backend.

| Variable | Default | Effect |
|---|---|---|
| `GGML_SYCL_DISABLE_GRAPH` | `0` | `1` runs every graph eagerly. Otherwise repeated graphs are recorded and replayed as SYCL command graphs. |
| `GGML_SYCL_GRAPH_EVICTION_TIMEOUT` | `10` | Seconds a recorded graph may stay unused before it is released. |
| `GGML_SYCL_FA_ONEDNN` | `1` | `0` disables oneDNN's fused attention. Long F16 flash attention then uses the TILE kernel. |
| `GGML_SYCL_FA_ONEDNN_MAX_KV` | `0` | KV length above which fused attention is skipped. `0` means no limit. |
| `GGML_SYCL_WEIGHT_F16_CACHE` | `0` | `1` keeps the F16 expansion of quantized weights used by oneDNN matmuls, up to a third of device memory and only on devices with at least 20 GiB. It trades device memory, about twice the size of those weights, for not converting them on every graph run. |
| `GGML_SYCL_DISABLE_DNN` | `0` | `1` disables oneDNN entirely. |

## Checked models

| Model | Status |
|---|---|
| YuE2 | Full song generation (AR, NAR, VAE). See [docs/models/yue2.md](../models/yue2.md). |

Other models may use ggml operators that the SYCL backend does not implement yet.

## Known issues

- With the oneAPI compiler, `kaldi_fbank_sparse_test` and `mel_spectrogram_frontend_test` fail on CPU value checks. They fail the same way on an unmodified tree built with `icx`, independent of `ENGINE_ENABLE_SYCL`.

## Validated models

Which models have been checked on SYCL, their measurements, and the ones that do not work yet are in the [SYCL validation report](../reports/sycl_validation.md).
