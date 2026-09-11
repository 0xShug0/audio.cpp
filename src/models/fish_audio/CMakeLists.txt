# Scoped 3.21 minimum: enable_language(HIP) requires CMake 3.21 while the
# top-level minimum stays at 3.20 (same pattern as ggml-hip).
cmake_minimum_required(VERSION 3.21)

# ROCm discovery (mirrors external/ggml/src/ggml-hip/CMakeLists.txt).
if (NOT EXISTS $ENV{ROCM_PATH})
    if (NOT EXISTS /opt/rocm)
        set(ROCM_PATH /usr)
    else()
        set(ROCM_PATH /opt/rocm)
    endif()
else()
    set(ROCM_PATH $ENV{ROCM_PATH})
endif()
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} "${ROCM_PATH}/lib64/cmake")

# Seed HIP architectures before enable_language, same precedence as ggml-hip.
if (NOT CMAKE_HIP_ARCHITECTURES)
    if (GPU_TARGETS)
        set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})
    elseif (AMDGPU_TARGETS)
        set(CMAKE_HIP_ARCHITECTURES ${AMDGPU_TARGETS})
    endif()
endif()

enable_language(HIP)
find_package(hip REQUIRED)

# CMake maps .cu to CUDA by default; mark it HIP explicitly like ggml-hip does.
set_source_files_properties(hip_fast_sampler.cu PROPERTIES LANGUAGE HIP)
add_library(engine_fish_audio_hip_sampler STATIC hip_fast_sampler.cu)
set_target_properties(engine_fish_audio_hip_sampler PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON)
# The AMDGPU backend can emit illegal DPP instructions for GFX10+ at -O0
# (https://github.com/ROCm/ROCm/issues/5826), same workaround as ggml-hip.
target_compile_options(engine_fish_audio_hip_sampler PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O2>)
# PUBLIC: this is a static library, so consumers of engine_runtime must link
# libamdhip64 for the kernel's HIP runtime calls.
target_link_libraries(engine_fish_audio_hip_sampler PUBLIC hip::host)
