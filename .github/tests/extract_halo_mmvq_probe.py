"""Freeze the actual stock MMVQ kernel for independent Halo measurements."""
import hashlib
from pathlib import Path

prefix = Path('external/ggml/src/ggml-cuda/mmvq.cu').read_text().split('// Dedicated MoE multi-token kernel.')[0]
assert hashlib.sha256(prefix.encode()).hexdigest() == 'f20485b273230a5fe1911baaaa1e82f8a58a800040cf7e9e2ef51c3978c58699'
start = prefix.index('typedef float')
includes = prefix[:start].replace('#include "mmvq.cuh"', '#include "external/ggml/src/ggml-cuda/mmvq.cuh"')
Path('original_mmvq.cuh').write_text(includes+'\nnamespace original {\n'+prefix[start:]+'\n}\n')
