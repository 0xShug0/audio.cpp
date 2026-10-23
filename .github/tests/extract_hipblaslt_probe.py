from pathlib import Path
import sys

source, destination = map(Path, sys.argv[1:3])
common = (source / 'external/ggml/src/ggml-cuda/common.cuh').read_text()
cu = (source / 'external/ggml/src/ggml-cuda/ggml-cuda.cu').read_text()
start = cu.index('static hipDataType ggml_hipblaslt_convert_type(')
stop = cu.index('#endif // defined(GGML_USE_HIP) && defined(GGML_HIP_USE_HIPBLASLT)', start)
helper = cu[start:stop]
# The observer records overlap and delegates to the real library, without sleeps
# or altered results. Otherwise the helper and cache bodies are extracted verbatim.
assert helper.count('HIPBLASLT_CHECK(hipblasLtMatmul(') == 1
helper = helper.replace('HIPBLASLT_CHECK(hipblasLtMatmul(', 'HIPBLASLT_CHECK(observed_matmul(')
cache = ''
if '    using hipblaslt_algo_cache_key =' in common:
    start = common.index('    using hipblaslt_algo_cache_key =')
    stop = common.index('    hipblasLtHandle_t hipblaslt_handle(int device)', start)
    cache = common[start:stop]
destination.mkdir(parents=True, exist_ok=True)
(destination / 'hipblaslt_gemm.inc').write_text(helper)
(destination / 'hipblaslt_cache.inc').write_text(cache)
print(f'Extracted real helper ({len(helper)} bytes), cache ({len(cache)} bytes)')
