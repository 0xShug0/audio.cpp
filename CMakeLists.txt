cmake_minimum_required(VERSION 3.20)

project(AudioCpp LANGUAGES C CXX)

set(AUDIOCPP_VERSION "dev" CACHE STRING "audio.cpp version string")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(AUDIOCPP_DEPLOYMENT_BUILD
    "Compile model specs into the runtime for self-contained deployments"
    OFF)

# Model specs remain editable runtime data in normal builds. The converter always
# carries the source catalog so the conversion binary is portable. Deployment builds
# additionally compile the catalog into engine_runtime as a fallback for CLI/server.
#
# Each spec becomes one C++ string, and MSVC caps a single string literal at 16380
# single-byte characters (error C2026) before adjacent literals are concatenated --
# qwen3_tts.json is 16.5 KB on its own. So split each spec into adjacent literals,
# cutting at newline boundaries so no chunk splits a UTF-8 sequence and every chunk
# stays plain JSON text.
function(audiocpp_model_spec_literal out json)
    set(AUDIOCPP_SPEC_LIMIT 16000)
    set(AUDIOCPP_SPEC_REST "${json}")
    set(AUDIOCPP_SPEC_LITERALS "")
    while(TRUE)
        string(LENGTH "${AUDIOCPP_SPEC_REST}" AUDIOCPP_SPEC_LEN)
        if(AUDIOCPP_SPEC_LEN LESS_EQUAL AUDIOCPP_SPEC_LIMIT)
            string(APPEND AUDIOCPP_SPEC_LITERALS "R\"AUDIOCPP_SPEC(${AUDIOCPP_SPEC_REST})AUDIOCPP_SPEC\"")
            break()
        endif()
        string(SUBSTRING "${AUDIOCPP_SPEC_REST}" 0 ${AUDIOCPP_SPEC_LIMIT} AUDIOCPP_SPEC_HEAD)
        string(FIND "${AUDIOCPP_SPEC_HEAD}" "\n" AUDIOCPP_SPEC_CUT REVERSE)
        if(AUDIOCPP_SPEC_CUT LESS 1)
            # A spec with no newline in the first 16 KB: cut at the limit.
            set(AUDIOCPP_SPEC_CUT ${AUDIOCPP_SPEC_LIMIT})
        endif()
        string(SUBSTRING "${AUDIOCPP_SPEC_REST}" 0 ${AUDIOCPP_SPEC_CUT} AUDIOCPP_SPEC_CHUNK)
        string(SUBSTRING "${AUDIOCPP_SPEC_REST}" ${AUDIOCPP_SPEC_CUT} -1 AUDIOCPP_SPEC_REST)
        string(APPEND AUDIOCPP_SPEC_LITERALS "R\"AUDIOCPP_SPEC(${AUDIOCPP_SPEC_CHUNK})AUDIOCPP_SPEC\" ")
    endwhile()
    set(${out} "${AUDIOCPP_SPEC_LITERALS}" PARENT_SCOPE)
endfunction()

file(GLOB AUDIOCPP_MODEL_SPEC_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/model_specs/*.json")
set(AUDIOCPP_BUILTIN_MODEL_SPEC_ENTRIES "")
set(AUDIOCPP_CONVERTER_MODEL_SPEC_ENTRIES "")
set(AUDIOCPP_NATIVE_PACKAGE_SPEC_ENTRIES "")
foreach(AUDIOCPP_MODEL_SPEC_FILE IN LISTS AUDIOCPP_MODEL_SPEC_FILES)
    get_filename_component(AUDIOCPP_MODEL_SPEC_FAMILY "${AUDIOCPP_MODEL_SPEC_FILE}" NAME_WE)
    file(READ "${AUDIOCPP_MODEL_SPEC_FILE}" AUDIOCPP_MODEL_SPEC_JSON)
    audiocpp_model_spec_literal(AUDIOCPP_MODEL_SPEC_LITERAL "${AUDIOCPP_MODEL_SPEC_JSON}")
    string(APPEND AUDIOCPP_CONVERTER_MODEL_SPEC_ENTRIES
        "    {\"${AUDIOCPP_MODEL_SPEC_FAMILY}\", ${AUDIOCPP_MODEL_SPEC_LITERAL}},\n")
    string(APPEND AUDIOCPP_NATIVE_PACKAGE_SPEC_ENTRIES
        "    {\"${AUDIOCPP_MODEL_SPEC_FAMILY}\", ${AUDIOCPP_MODEL_SPEC_LITERAL}},\n")
    if (AUDIOCPP_DEPLOYMENT_BUILD)
        string(APPEND AUDIOCPP_BUILTIN_MODEL_SPEC_ENTRIES
            "    {\"${AUDIOCPP_MODEL_SPEC_FAMILY}\", ${AUDIOCPP_MODEL_SPEC_LITERAL}},\n")
    endif()
endforeach()
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/model_specs.inc"
    CONTENT "${AUDIOCPP_BUILTIN_MODEL_SPEC_ENTRIES}")
file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/gguf_converter_model_specs.inc"
    CONTENT "${AUDIOCPP_CONVERTER_MODEL_SPEC_ENTRIES}")
file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/native_package_specs.inc"
    CONTENT "${AUDIOCPP_NATIVE_PACKAGE_SPEC_ENTRIES}")

if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

if (NOT MSVC)
    set(AUDIOCPP_DEBUG_OPT_FLAGS "-O3 -g")
    set(CMAKE_C_FLAGS_DEBUG "${AUDIOCPP_DEBUG_OPT_FLAGS}" CACHE STRING "Optimized debug C flags" FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "${AUDIOCPP_DEBUG_OPT_FLAGS}" CACHE STRING "Optimized debug C++ flags" FORCE)
endif()

set(ENGINE_DEFAULT_ENABLE_CUDA OFF)
set(ENGINE_DEFAULT_ENABLE_HIP OFF)
set(ENGINE_DEFAULT_ENABLE_VULKAN OFF)
set(ENGINE_DEFAULT_ENABLE_WEBGPU OFF)
set(ENGINE_DEFAULT_ENABLE_SYCL OFF)
set(ENGINE_DEFAULT_ENABLE_METAL OFF)
set(ENGINE_DEFAULT_ENABLE_LLAMAFILE ON)
set(ENGINE_DEFAULT_ENABLE_CUDA_GRAPHS ON)
set(ENGINE_DEFAULT_ENABLE_NATIVE_CPU ON)
if (APPLE)
    set(ENGINE_DEFAULT_ENABLE_METAL ON)
endif()
if (DEFINED GGML_CUDA)
    set(ENGINE_DEFAULT_ENABLE_CUDA ${GGML_CUDA})
endif()
if (DEFINED GGML_HIP)
    set(ENGINE_DEFAULT_ENABLE_HIP ${GGML_HIP})
endif()
if (DEFINED GGML_VULKAN)
    set(ENGINE_DEFAULT_ENABLE_VULKAN ${GGML_VULKAN})
endif()
if (DEFINED GGML_WEBGPU)
    set(ENGINE_DEFAULT_ENABLE_WEBGPU ${GGML_WEBGPU})
endif()
if (DEFINED GGML_SYCL)
    set(ENGINE_DEFAULT_ENABLE_SYCL ${GGML_SYCL})
endif()
if (DEFINED GGML_METAL)
    set(ENGINE_DEFAULT_ENABLE_METAL ${GGML_METAL})
endif()
if (DEFINED GGML_LLAMAFILE)
    set(ENGINE_DEFAULT_ENABLE_LLAMAFILE ${GGML_LLAMAFILE})
endif()
if (DEFINED GGML_CUDA_GRAPHS)
    set(ENGINE_DEFAULT_ENABLE_CUDA_GRAPHS ${GGML_CUDA_GRAPHS})
endif()

option(ENGINE_ENABLE_CUDA "Build ggml with CUDA backend support" ${ENGINE_DEFAULT_ENABLE_CUDA})
option(ENGINE_ENABLE_HIP "Build ggml with HIP/ROCm backend support" ${ENGINE_DEFAULT_ENABLE_HIP})
option(ENGINE_ENABLE_VULKAN "Build ggml with Vulkan backend support" ${ENGINE_DEFAULT_ENABLE_VULKAN})
option(ENGINE_ENABLE_WEBGPU "Build ggml with WebGPU backend support" ${ENGINE_DEFAULT_ENABLE_WEBGPU})
option(ENGINE_ENABLE_SYCL "Build ggml with SYCL backend support (Intel oneAPI)" ${ENGINE_DEFAULT_ENABLE_SYCL})
option(ENGINE_ENABLE_METAL "Build ggml with Metal backend support" ${ENGINE_DEFAULT_ENABLE_METAL})
option(ENGINE_ENABLE_LLAMAFILE "Build ggml with llamafile SGEMM support" ${ENGINE_DEFAULT_ENABLE_LLAMAFILE})
option(ENGINE_ENABLE_CUDA_GRAPHS "Enable ggml CUDA graphs support" ${ENGINE_DEFAULT_ENABLE_CUDA_GRAPHS})
option(ENGINE_HIP_STRIX_HALO_OPTIMIZATIONS
    "Enable opt-in HIP optimizations validated only for Strix Halo (gfx1151)"
    OFF)
# Rockchip NPU (RK3588 & co): a model-local compute path, not a ggml backend —
# the Qwen3 TTS speech decoder runs a pre-compiled .rknn step graph through
# librknnrt. The engine links the board runtime; the models themselves are
# converted offline on x86 with rknn-toolkit2.
option(ENGINE_ENABLE_RKNN "Build the Rockchip NPU (librknnrt) decoder path" OFF)
option(ENGINE_ENABLE_NEON "Build opt-in NEON library and native model paths" OFF)
set(ENGINE_RKNN_INCLUDE_DIR "" CACHE PATH "Directory containing rknn_api.h (empty = use the system include path)")
set(ENGINE_RKNN_LIBRARY "rknnrt" CACHE STRING "librknnrt to link (path or library name)")
option(ENGINE_ENABLE_NATIVE_CPU "Build ggml CPU kernels with native host ISA flags" ${ENGINE_DEFAULT_ENABLE_NATIVE_CPU})
option(ENGINE_ENABLE_OPENMP "Build host code with OpenMP support" ON)
option(AUDIOCPP_STATIC_ESPEAK "Build and statically link GPL-3.0-or-later eSpeak-ng; keep data separate" OFF)
option(AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER
    "Build native model-manager tools and server download/install support"
    OFF)
include("${CMAKE_CURRENT_SOURCE_DIR}/app/server/server_frontends.cmake")
option(AUDIOCPP_USE_SYSTEM_OPENSSL
    "Use system OpenSSL for native model management instead of bundled BoringSSL"
    OFF)
if (AUDIOCPP_USE_SYSTEM_OPENSSL AND NOT AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER)
    message(FATAL_ERROR
        "AUDIOCPP_USE_SYSTEM_OPENSSL requires AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON")
endif()
option(ENGINE_ENABLE_CPU_ALL_VARIANTS
    "Build CPU backends as dynamic libraries with per-ISA variants (for Docker/portable builds)"
    OFF)
option(ENGINE_BUILD_EXAMPLES "Build example binaries" OFF)
option(ENGINE_BUILD_TESTS "Build framework unit tests" OFF)
option(ENGINE_BUILD_EXTENDED_TESTS "Build extended non-model tests and probes" OFF)
option(ENGINE_BUILD_MODEL_TESTS "Build model-specific tests and probes" OFF)
option(ENGINE_BUILD_WARMBENCH "Build warmbench helper binaries" OFF)
option(AUDIOCPP_BUILD_C_API "Build the C ABI (include/audiocpp.h) as a shared library" OFF)
option(AUDIOCPP_BUILD_C_API_STATIC "Also build the C ABI as a static library (libaudiocpp.a) for embedding" OFF)
set(AUDIOCPP_C_API_MODEL_ROOT "" CACHE PATH
    "Models root for the C API model test. Unset leaves the test registered but skipping.")
set(AUDIOCPP_RKNN_TEST_ENGINE "" CACHE PATH
    "A .rknn engine compiled for the build host's SoC, for the RKNN sidecar test. Unset leaves the test registered but skipping.")
set(AUDIOCPP_GGML_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/external/ggml"
    CACHE PATH "ggml source tree to build with AudioCPP")

set(AUDIOCPP_GIT_SHA "unknown")
set(AUDIOCPP_GIT_DATE "unknown")
find_package(Git QUIET)
if (GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE AUDIOCPP_GIT_SHA
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} show -s --format=%cs HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE AUDIOCPP_GIT_DATE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
endif()

set(AUDIOCPP_BUILD_BACKENDS "cpu")
if (ENGINE_ENABLE_CUDA)
    string(APPEND AUDIOCPP_BUILD_BACKENDS ",cuda")
endif()
if (ENGINE_ENABLE_HIP)
    string(APPEND AUDIOCPP_BUILD_BACKENDS ",hip")
endif()
if (ENGINE_ENABLE_VULKAN)
    string(APPEND AUDIOCPP_BUILD_BACKENDS ",vulkan")
endif()
if (ENGINE_ENABLE_WEBGPU)
    string(APPEND AUDIOCPP_BUILD_BACKENDS ",webgpu")
endif()
if (ENGINE_ENABLE_SYCL)
    string(APPEND AUDIOCPP_BUILD_BACKENDS ",sycl")
endif()
if (ENGINE_ENABLE_METAL)
    string(APPEND AUDIOCPP_BUILD_BACKENDS ",metal")
endif()
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/app/common/build_info_config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/build_info_config.h"
    @ONLY
)

# HIP compiles ggml's CUDA backend sources as HIP code, so the two options drive the
# same ggml backend and cannot coexist: enabling both starts CUDA backend setup and
# HIP backend setup in the same configure, which fails in confusing ways downstream.
if (ENGINE_ENABLE_CUDA AND ENGINE_ENABLE_HIP)
    message(FATAL_ERROR
        "ENGINE_ENABLE_CUDA and ENGINE_ENABLE_HIP are mutually exclusive; "
        "configure with exactly one of them (e.g. -DENGINE_ENABLE_CUDA=OFF -DENGINE_ENABLE_HIP=ON)")
endif()

set(AUDIOCPP_HIP_STRIX_HALO_OPTIMIZATIONS_ACTIVE OFF)
if (ENGINE_HIP_STRIX_HALO_OPTIMIZATIONS)
    if (NOT ENGINE_ENABLE_HIP)
        message(FATAL_ERROR
            "ENGINE_HIP_STRIX_HALO_OPTIMIZATIONS=ON requires ENGINE_ENABLE_HIP=ON")
    endif()

    # Match ggml HIP target precedence: an explicit CMake HIP architecture wins,
    # followed by GPU_TARGETS and then the legacy AMDGPU_TARGETS spelling.
    if (CMAKE_HIP_ARCHITECTURES)
        set(AUDIOCPP_EFFECTIVE_HIP_TARGETS ${CMAKE_HIP_ARCHITECTURES})
        set(AUDIOCPP_EFFECTIVE_HIP_TARGET_SOURCE "CMAKE_HIP_ARCHITECTURES")
    elseif (GPU_TARGETS)
        set(AUDIOCPP_EFFECTIVE_HIP_TARGETS ${GPU_TARGETS})
        set(AUDIOCPP_EFFECTIVE_HIP_TARGET_SOURCE "GPU_TARGETS")
    elseif (AMDGPU_TARGETS)
        set(AUDIOCPP_EFFECTIVE_HIP_TARGETS ${AMDGPU_TARGETS})
        set(AUDIOCPP_EFFECTIVE_HIP_TARGET_SOURCE "AMDGPU_TARGETS")
    else()
        message(FATAL_ERROR
            "ENGINE_HIP_STRIX_HALO_OPTIMIZATIONS=ON requires exactly one explicit HIP target, gfx1151; "
            "set -DGPU_TARGETS=gfx1151 (no project-wide gfx1151 default is provided)")
    endif()

    list(LENGTH AUDIOCPP_EFFECTIVE_HIP_TARGETS AUDIOCPP_EFFECTIVE_HIP_TARGET_COUNT)
    if (NOT AUDIOCPP_EFFECTIVE_HIP_TARGET_COUNT EQUAL 1 OR
        NOT "${AUDIOCPP_EFFECTIVE_HIP_TARGETS}" STREQUAL "gfx1151")
        message(FATAL_ERROR
            "ENGINE_HIP_STRIX_HALO_OPTIMIZATIONS=ON requires the effective HIP target list to be exactly "
            "gfx1151; ${AUDIOCPP_EFFECTIVE_HIP_TARGET_SOURCE}='${AUDIOCPP_EFFECTIVE_HIP_TARGETS}'")
    endif()

    # hipBLASLt regressed decoder compute by roughly 3 seconds on gfx1151.
    # Keep the validated retained-graph path on rocBLAS; the generic HIP build
    # remains unchanged because this option is both target- and backend-gated.
    set(GGML_HIP_HIPBLASLT OFF CACHE BOOL
        "Use hipBLASLt instead of hipBLAS (rocBLAS) for HIP GEMM" FORCE)
    set(AUDIOCPP_HIP_STRIX_HALO_OPTIMIZATIONS_ACTIVE ON)
    message(STATUS "Strix Halo HIP optimizations enabled for gfx1151 with rocBLAS")
endif()

if (ENGINE_ENABLE_CUDA AND NOT ENGINE_ENABLE_HIP)
    # Require at least CUDA 12.0. Older toolkits configure fine but die at the
    # first .cu file. Run this before enable_language(CUDA) because the version
    # is needed for the default arch list below.
    find_package(CUDAToolkit 12.0 REQUIRED)

    # Define default CUDA architectures for every build path.
    # We stay close to llama.cpp/ggml-cuda here, unless there is a reason
    # to deviate. See llama.cpp's ggml-cuda CMakeLists.txt for additional
    # comments.
    # Must be set before enable_language(CUDA), as that call otherwise seeds
    # CMAKE_CUDA_ARCHITECTURES from nvcc's default arch (sm_75 for CUDA 13,
    # sm_52 for CUDA 12), silently making every build without an explicit
    # list single-arch (the seed ignores the local GPU).
    # This can be overridden with -DCMAKE_CUDA_ARCHITECTURES=... or CUDAARCHS;
    # set to "native" to build for the local GPU only (needs CMake >= 3.24).

    # "native" fallback for CMake < 3.24
    if ((CMAKE_CUDA_ARCHITECTURES STREQUAL "native" OR "$ENV{CUDAARCHS}" STREQUAL "native")
        AND CMAKE_VERSION VERSION_LESS "3.24")
        message(STATUS "CMAKE_CUDA_ARCHITECTURES=native requires CMake >= 3.24; "
                       "falling back to the portable default architecture list")
        set(CMAKE_CUDA_ARCHITECTURES "")
    endif()

    # Define default archs
    if ("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "" AND "$ENV{CUDAARCHS}" STREQUAL "")
        # Remembered so the summary below can tell the difference between "the
        # portable list is what you asked for" and "the portable list is what you
        # got by not asking".
        set(_AUDIOCPP_CUDA_ARCH_DEFAULTED TRUE)
        # XX-virtual = PTX (JIT, forward-compatible); XX-real = SASS (native).
        if (CUDAToolkit_VERSION VERSION_LESS "13")
            # 50/61/70 (Maxwell/Pascal/Volta) unsupported by CUDA 13 toolkits
            list(APPEND CMAKE_CUDA_ARCHITECTURES 50-virtual 61-virtual 70-virtual)
        endif()
        list(APPEND CMAKE_CUDA_ARCHITECTURES 75-real 80-virtual 86-real)
        if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.8")
            list(APPEND CMAKE_CUDA_ARCHITECTURES 89-real 90-virtual)
        endif()
        if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
            # 12Xa, not 12X: Blackwell FP4 tensor cores are arch-specific.
            list(APPEND CMAKE_CUDA_ARCHITECTURES 120a-real)
        endif()
        if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.9")
            list(APPEND CMAKE_CUDA_ARCHITECTURES 121a-real)
        endif()
    endif()

    enable_language(CUDA)

    # Upgrade plain 12X (user value or resolved "native") to 12Xa (Blackwell).
    foreach (_ARCHS_VAR IN ITEMS CMAKE_CUDA_ARCHITECTURES CMAKE_CUDA_ARCHITECTURES_NATIVE)
        set(_FIXED_ARCHS "")
        foreach (_ARCH IN LISTS ${_ARCHS_VAR})
            if (_ARCH MATCHES "^12[0-9](-real|-virtual)?$")
                string(REGEX REPLACE "^(12[0-9])((-real|-virtual)?)$" "\\1a\\2" _FIXED_ARCH "${_ARCH}")
                message(STATUS "Replacing ${_ARCH} in ${_ARCHS_VAR} with ${_FIXED_ARCH}")
                list(APPEND _FIXED_ARCHS "${_FIXED_ARCH}")
            else()
                list(APPEND _FIXED_ARCHS "${_ARCH}")
            endif()
        endforeach()
        set(${_ARCHS_VAR} "${_FIXED_ARCHS}")
    endforeach()

    # Resolve "native" for the log; left as-is when no GPU was detected
    # (nvcc then warns and uses its default arch).
    if (CMAKE_CUDA_ARCHITECTURES STREQUAL "native" AND CMAKE_CUDA_ARCHITECTURES_NATIVE MATCHES "^[0-9]+(a|f)?(-real|-virtual)?(;[0-9]+(a|f)?(-real|-virtual)?|;)*$")
        set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
    endif()
    message(STATUS "Using CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} CMAKE_CUDA_ARCHITECTURES_NATIVE=${CMAKE_CUDA_ARCHITECTURES_NATIVE}")

    # The line above prints both halves of this and interprets neither: how many
    # architectures are about to be compiled, and which one this host can run.
    # The option is documented (README, docs/build/linux.md), but configure is
    # where the choice is actually being made, so say it here too.
    if (_AUDIOCPP_CUDA_ARCH_DEFAULTED AND
        CMAKE_CUDA_ARCHITECTURES_NATIVE MATCHES "^[0-9]")
        list(LENGTH CMAKE_CUDA_ARCHITECTURES _AUDIOCPP_ARCH_COUNT)
        message(STATUS "  portable default: ${_AUDIOCPP_ARCH_COUNT} architectures, so every .cu file is compiled ${_AUDIOCPP_ARCH_COUNT} times")
        message(STATUS "  this host is ${CMAKE_CUDA_ARCHITECTURES_NATIVE}; for a local build roughly 3x faster:")
        message(STATUS "      -DCMAKE_CUDA_ARCHITECTURES=native    (the result then runs on this GPU only)")
        message(STATUS "      scripts/build_linux.sh: --cuda-arch native")
    endif()

    if (NOT MSVC)
        set(CMAKE_CUDA_FLAGS_DEBUG "${AUDIOCPP_DEBUG_OPT_FLAGS}" CACHE STRING "Optimized debug CUDA flags" FORCE)
    endif()
endif()

# ⚠ Outside the CUDA block on purpose: ccache is not a CUDA concern, and a
# CPU-only, Vulkan or HIP build benefits from it just as much. GGML_CCACHE is
# forced OFF above, so ggml's own probe never runs and nothing caches anything
# unless the caller wires a launcher up by hand.
find_program(AUDIOCPP_CCACHE_PROGRAM ccache)
if (AUDIOCPP_CCACHE_PROGRAM AND
    NOT CMAKE_C_COMPILER_LAUNCHER AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    message(STATUS "ccache is installed (${AUDIOCPP_CCACHE_PROGRAM}) and NOT in use. Rebuilds are much faster with:")
    message(STATUS "    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    message(STATUS "    scripts/build_linux.sh: --ccache")
endif()

if (ENGINE_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

if (AUDIOCPP_BUILD_C_API)
    # engine_runtime and ggml are static/object libraries here; linking them into
    # a shared libaudiocpp requires position-independent code throughout.
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if (ENGINE_ENABLE_CPU_ALL_VARIANTS)
    message(STATUS "CPU all variants mode: enabling shared libraries + backends as dynamic libs")
    set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries" FORCE)
    set(GGML_BACKEND_DL ON CACHE BOOL "Build backends as dynamic libraries" FORCE)
    set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "Build all CPU backend variants" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "Build ggml CPU kernels with native host ISA flags" FORCE)
    # Ensure libraries in the same directory as the binary are found at runtime
    set(CMAKE_INSTALL_RPATH "\$ORIGIN")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
else()
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
    set(GGML_BACKEND_DL OFF CACHE BOOL "Build backends as dynamic libraries" FORCE)
    set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "Build all CPU backend variants" FORCE)
    set(GGML_NATIVE ${ENGINE_ENABLE_NATIVE_CPU} CACHE BOOL "Build ggml CPU kernels with native host ISA flags" FORCE)
endif()

set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "Do not build upstream examples" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "Do not build upstream tests" FORCE)
set(GGML_ALL_WARNINGS OFF CACHE BOOL "Keep upstream warnings quiet in this sample" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "Do not probe for ccache in this local setup" FORCE)
set(GGML_CUDA ${ENGINE_ENABLE_CUDA} CACHE BOOL "Build ggml with CUDA backend support" FORCE)
set(GGML_HIP ${ENGINE_ENABLE_HIP} CACHE BOOL "Build ggml with HIP backend support" FORCE)
set(GGML_VULKAN ${ENGINE_ENABLE_VULKAN} CACHE BOOL "Build ggml with Vulkan backend support" FORCE)
set(GGML_WEBGPU ${ENGINE_ENABLE_WEBGPU} CACHE BOOL "Build ggml with WebGPU backend support" FORCE)
set(GGML_SYCL ${ENGINE_ENABLE_SYCL} CACHE BOOL "Build ggml with SYCL backend support" FORCE)
if (ENGINE_ENABLE_SYCL)
    set(GGML_SYCL_F16 ON CACHE BOOL "ggml: run SYCL oneDNN matmuls in F16")
endif()
set(GGML_METAL ${ENGINE_ENABLE_METAL} CACHE BOOL "Build ggml with Metal backend support" FORCE)
set(GGML_LLAMAFILE ${ENGINE_ENABLE_LLAMAFILE} CACHE BOOL "Build ggml with llamafile SGEMM support" FORCE)
set(GGML_CUDA_GRAPHS ${ENGINE_ENABLE_CUDA_GRAPHS} CACHE BOOL "Enable ggml CUDA graphs support" FORCE)
# HIP graphs are controlled by a separate upstream option that defaults to ON;
# keep it in sync so ENGINE_ENABLE_CUDA_GRAPHS=OFF also disables graphs on HIP builds
set(GGML_HIP_GRAPHS ${ENGINE_ENABLE_CUDA_GRAPHS} CACHE BOOL "Enable ggml HIP graphs support" FORCE)

set(SPM_ENABLE_SHARED OFF CACHE BOOL "Build sentencepiece shared libs" FORCE)
set(SPM_BUILD_TEST OFF CACHE BOOL "Do not build sentencepiece tests" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "Do not probe for tcmalloc" FORCE)

if (NOT EXISTS "${AUDIOCPP_GGML_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "AUDIOCPP_GGML_SOURCE_DIR does not point to a ggml source tree: ${AUDIOCPP_GGML_SOURCE_DIR}")
endif()
message(STATUS "Using ggml source: ${AUDIOCPP_GGML_SOURCE_DIR}")
add_subdirectory("${AUDIOCPP_GGML_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/ggml")
add_subdirectory(external/sentencepiece EXCLUDE_FROM_ALL)

add_compile_options(
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wextra>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wpedantic>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-pedantic-errors>
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->
)

set(AUDIOCPP_MODEL_SET "full" CACHE STRING "Model composite to build: full, core, or custom")
set_property(CACHE AUDIOCPP_MODEL_SET PROPERTY STRINGS full core custom)
set(AUDIOCPP_MODELS "" CACHE STRING "Comma or semicolon separated model targets for AUDIOCPP_MODEL_SET=custom")
set(AUDIOCPP_LIBYAML_COMPILE_DEFINITIONS
    _POSIX_C_SOURCE=200809L
    YAML_DECLARE_STATIC=1
    YAML_VERSION_MAJOR=0
    YAML_VERSION_MINOR=2
    YAML_VERSION_PATCH=5
    YAML_VERSION_STRING=\"0.2.5\"
)

function(audiocpp_configure_runtime_object target_name)
    target_include_directories(${target_name} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/generated
        ${CMAKE_CURRENT_SOURCE_DIR}/external/sentencepiece/src
        ${CMAKE_CURRENT_SOURCE_DIR}/external/llama_tokenizer
        ${CMAKE_CURRENT_SOURCE_DIR}/external/cJSON
        ${CMAKE_CURRENT_SOURCE_DIR}/external/libyaml/include
    )
    target_compile_definitions(${target_name} PRIVATE ${AUDIOCPP_LIBYAML_COMPILE_DEFINITIONS})
    if (AUDIOCPP_HIP_STRIX_HALO_OPTIMIZATIONS_ACTIVE)
        target_compile_definitions(${target_name} PRIVATE ENGINE_HIP_STRIX_HALO_OPTIMIZATIONS=1)
    endif()
    if (ENGINE_ENABLE_RKNN)
        target_compile_definitions(${target_name} PRIVATE ENGINE_HAVE_RKNN=1)
        if (ENGINE_RKNN_INCLUDE_DIR)
            target_include_directories(${target_name} PRIVATE ${ENGINE_RKNN_INCLUDE_DIR})
        endif()
    endif()
    target_link_libraries(${target_name} PRIVATE ggml sentencepiece)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
        if (MSVC)
            target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/openmp:experimental>)
        endif()
    endif()
endfunction()

function(audiocpp_add_model target_name)
    cmake_parse_arguments(ARG "" "" "SOURCES;INCLUDES;LOADERS;ALIASES;DEPENDS;OBJECT_LIBRARIES" ${ARGN})
    add_library(engine_model_${target_name} OBJECT ${ARG_SOURCES})
    set_target_properties(engine_model_${target_name} PROPERTIES EXCLUDE_FROM_ALL TRUE)
    audiocpp_configure_runtime_object(engine_model_${target_name})
    if (ARG_OBJECT_LIBRARIES)
        target_link_libraries(engine_model_${target_name} PRIVATE ${ARG_OBJECT_LIBRARIES})
    endif()
    if (ENGINE_ENABLE_NEON)
        target_compile_definitions(engine_model_${target_name} PRIVATE ENGINE_HAVE_NEON=1)
    endif()
    set(AUDIOCPP_MODEL_TARGETS ${AUDIOCPP_MODEL_TARGETS} ${target_name} PARENT_SCOPE)
    set(AUDIOCPP_MODEL_${target_name}_INCLUDES "${ARG_INCLUDES}" PARENT_SCOPE)
    set(AUDIOCPP_MODEL_${target_name}_LOADERS "${ARG_LOADERS}" PARENT_SCOPE)
    set(AUDIOCPP_MODEL_${target_name}_DEPENDS "${ARG_DEPENDS}" PARENT_SCOPE)
    set(AUDIOCPP_MODEL_${target_name}_OBJECT_LIBRARIES "${ARG_OBJECT_LIBRARIES}" PARENT_SCOPE)
    set(AUDIOCPP_MODEL_ALIAS_${target_name} ${target_name} PARENT_SCOPE)
    foreach(alias IN LISTS ARG_ALIASES)
        set(AUDIOCPP_MODEL_ALIAS_${alias} ${target_name} PARENT_SCOPE)
    endforeach()
endfunction()

if (ENGINE_ENABLE_NEON)
    add_library(engine_neon OBJECT
        src/framework/neon/tensor.cpp
        src/framework/neon/operators.cpp
        src/framework/neon/tensor_operators.cpp
        src/framework/neon/execution.cpp
        src/framework/neon/worker_team.cpp
        src/framework/neon/blocks.cpp
        src/framework/neon/hifigan.cpp
        src/framework/neon/hift.cpp
        src/framework/neon/vocoder_operators.cpp
        src/framework/neon/kernels.cpp
    )
    target_include_directories(engine_neon PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_features(engine_neon PUBLIC cxx_std_17)
    target_link_libraries(engine_neon PUBLIC Threads::Threads)
    target_compile_definitions(engine_neon PUBLIC ENGINE_HAVE_NEON=1)
endif()

add_library(engine_core OBJECT
    src/framework/assets/resource_bundle.cpp
    src/framework/model_spec/package.cpp
    src/framework/model_spec/options.cpp
    src/framework/model_spec/schema.cpp
    src/framework/assets/tensor_source.cpp
    src/framework/assets/gguf_metadata.cpp
    src/framework/assets/weight_norm_folding.cpp
    src/framework/assets/weight_adapter.cpp
    src/framework/assets/weight_adapter_source.cpp
    src/framework/assets/torch_bin.cpp
    src/framework/core/module.cpp
    src/framework/core/backend.cpp
    src/framework/core/attention_fallback.cpp
    src/framework/core/deferred_tensor_writer.cpp
    src/framework/core/graph_workspace.cpp
    src/framework/core/prepared_graph.cpp
    src/framework/core/execution_context.cpp
    src/framework/moe_engine/moe_engine.cpp
    src/framework/moe_engine/moe_prepared_graph.cpp
    src/framework/moe_engine/expert_math.cpp
    src/framework/moe_engine/expert_storage.cpp
    src/framework/moe_engine/expert_execution.cpp
    src/framework/moe_engine/moe_policy.cpp
    src/framework/moe_engine/placement_policy.cpp
    src/framework/moe_engine/cache_policy.cpp
    src/framework/runtime/moe_options.cpp
    src/framework/core/host_memory.cpp
    src/framework/core/load_progress.cpp
    src/framework/runtime/model.cpp
    src/framework/model_spec/metadata.cpp
    src/framework/runtime/session.cpp
    src/framework/runtime/task_vocabulary.cpp
    src/framework/runtime/partial_text.cpp
    src/framework/chat/history.cpp
    src/framework/chat/reply.cpp
    src/framework/chat/session.cpp
    src/framework/runtime/artifacts.cpp
    src/framework/runtime/cache.cpp
    src/framework/runtime/graph_executor.cpp
    src/framework/runtime/graph_optimizer.cpp
    src/framework/runtime/host_ops.cpp
    src/framework/decoders/ctc_host_decoder.cpp
    src/framework/runtime/kv_cache.cpp
    src/framework/runtime/bounded_static_kv_decode.cpp
    src/framework/runtime/greedy_causal_decoder.cpp
    src/framework/runtime/options.cpp
    src/framework/runtime/session_base.cpp
    src/framework/runtime/workspace.cpp
    src/framework/midi/midi_file.cpp
    src/framework/io/filesystem.cpp
    # Vision/video modules, runtimes and media helpers.
    src/framework/vision/media/image.cpp
    src/framework/vision/media/processing.cpp
    src/framework/vision/media/video.cpp
    src/framework/vision/modules/face_modules.cpp
    src/framework/vision/modules/wan_video_vae_modules.cpp
    src/framework/vision/runtimes/cav_mae_st_conditioner_runtime.cpp
    src/framework/vision/runtimes/liveportrait_runtime.cpp
    src/framework/vision/runtimes/ltx_video_vae_runtime.cpp
    src/framework/vision/runtimes/open_clip_conditioner_runtime.cpp
    src/framework/vision/runtimes/synchformer_conditioner_runtime.cpp
    src/framework/text/espeak_phonemizer.cpp
    src/framework/text/espeak_data.cpp
    src/framework/io/config.cpp
    src/framework/io/json.cpp
    src/framework/io/text.cpp
    src/framework/text/chinese_normalization.cpp
    src/framework/text/chinese_variant.cpp
    src/framework/text/chunking.cpp
    src/framework/text/subtitle.cpp
    src/framework/text/moss_tts_normalization.cpp
    src/framework/text/text_normalization.cpp
    src/framework/text/unicode_normalization.cpp
    src/framework/io/yaml.cpp
    src/framework/io/binary.cpp
    src/framework/io/safetensors.cpp
    src/framework/debug/trace.cpp
    src/framework/debug/profiler.cpp
    src/framework/rknn/engine.cpp
    src/framework/rknn/engine_resources.cpp
    src/framework/decoders/moss_tts_delay/backbone.cpp
    src/framework/decoders/moss_tts_delay/config.cpp
    src/framework/decoders/moss_tts_delay/delay_decoder.cpp
    src/framework/decoders/moss_tts_delay/heads.cpp
    src/framework/decoders/moss_tts_delay/prompt.cpp
    src/framework/decoders/tdt_decoder_runner.cpp
    src/framework/decoders/tdt_decoder_greedy_customized.cpp
    src/framework/decoders/tdt_decoder_greedy_duration_loop.cpp
    src/framework/decoders/tdt_decoder_greedy_separate_heads.cpp
    src/framework/modules/linear_module.cpp
    src/framework/modules/packed_linear_weights.cpp
    src/framework/modules/convnext_modules.cpp
    src/framework/modules/vocoders/vocos_vocoder.cpp
    src/framework/modules/primitive_modules.cpp
    src/framework/modules/activation_modules.cpp
    src/framework/modules/norm_modules.cpp
    src/framework/modules/lookup_modules.cpp
    src/framework/modules/multi_codebook_embedding.cpp
    src/framework/modules/structural_modules.cpp
    src/framework/modules/asr_helpers.cpp
    src/framework/modules/conv_modules.cpp
    src/framework/modules/zipformer_modules.cpp
    src/framework/modules/speech_encoders/whisper_embedding.cpp
    src/framework/modules/speech_decoders/ming_flow_runtime.cpp
    src/framework/modules/speech_encoders/whisper_frontend.cpp
    src/framework/modules/speaker_encoders/campplus_encoder.cpp
    src/framework/modules/speech_encoders/s3_tokenizer.cpp
    src/framework/codecs/nemo_nano_codec_runtime.cpp
    src/framework/modules/pitch_extractors/rmvpe_pitch_extractor.cpp
    src/framework/modules/vocoders/bigvgan_vocoder.cpp
    src/framework/modules/vocoders/hifigan_vocoder.cpp
    src/framework/modules/vocoders/hift_vocoder.cpp
    src/framework/modules/speech_encoders/wav2vec2_encoder.cpp
    src/framework/modules/speech_encoders/wav2vec2_bert_encoder.cpp
    src/framework/modules/speech_encoders/wavlm_encoder.cpp
    src/framework/modules/speech_encoders/sanm.cpp
    src/framework/modules/optimizations/fast_conv_modules.cpp
    src/framework/modules/optimizations/fast_projection_modules.cpp
    src/framework/modules/recurrent_modules.cpp
    src/framework/modules/streaming_conv_modules.cpp
    src/framework/modules/attention/positional_encoding.cpp
    src/framework/modules/gated_delta_net_module.cpp
    src/framework/modules/mamba_modules.cpp
    src/framework/modules/feed_forward_modules.cpp
    src/framework/modules/attention/self_attention.cpp
    src/framework/modules/attention/longformer_attention.cpp
    src/framework/modules/attention/longformer_attention_gpu.cpp
    src/framework/modules/attention/common_relative_attention.cpp
    src/framework/modules/attention/relative_attention.cpp
    src/framework/modules/attention/scaled_dot_product_attention.cpp
    src/framework/modules/attention/grouped_query_attention.cpp
    src/framework/modules/attention/projected_grouped_self_attention.cpp
    src/framework/modules/attention/streaming_frame_attention_cache.cpp
    src/framework/modules/transformers/decoder.cpp
    src/framework/modules/transformers/causal_decoder.cpp
    src/framework/modules/transformers/causal_decoder_runtime.cpp
    src/framework/modules/transformers/dit.cpp
    src/framework/modules/transformers/dit_runtime.cpp
    src/framework/modules/speech_encoders/lfm3_audio_encoder_runtime.cpp
    src/framework/modules/speech_encoders/rope_transformer_encoder_runtime.cpp
    src/framework/modules/transformers/lfm_decoder_runtime.cpp
    src/framework/modules/transformers/qwen35_decoder_runtime.cpp
    src/framework/modules/text_encoders/qwen3_vl_encoder_runtime.cpp
    src/framework/modules/speech_encoders/qwen3_audio_encoder_runtime.cpp
    src/framework/modules/attention/cross_attention.cpp
    src/framework/modules/transformers/transformer_blocks.cpp
    src/framework/modules/conditioning_modules.cpp
    src/framework/modules/conformer_modules.cpp
    src/framework/modules/ebranchformer_modules.cpp
    src/framework/sampling/device_denoiser_runtime.cpp
    src/framework/sampling/flow_sampler_runtime.cpp
    src/framework/modules/predictor_modules.cpp
    src/framework/modules/positional_modules.cpp
    src/framework/modules/text_encoders/t5_base_encoder.cpp
    src/framework/modules/text_encoders/t5_gemma_encoder.cpp
    src/framework/modules/transformers/gemma_decoder.cpp
    src/framework/modules/optimizations/fast_kv_modules.cpp
    src/framework/tokenizers/hf_tokenizer_json.cpp
    src/framework/tokenizers/llama_bpe.cpp
    src/framework/tokenizers/qwen_bpe_bundle.cpp
    external/llama_tokenizer/bpe-core.cpp
    external/llama_tokenizer/unicode.cpp
    external/llama_tokenizer/unicode-data.cpp
    src/framework/tokenizers/sentencepiece.cpp
    src/framework/audio/fft.cpp
    src/framework/audio/kaldi_fbank.cpp
    src/framework/audio/activity.cpp
    src/framework/audio/probability_segmenter.cpp
    src/framework/audio/activity_graph.cpp
    src/framework/audio/chunking.cpp
    src/framework/audio/conversion.cpp
    src/framework/audio/utilities/deepfilternet2.cpp
    src/framework/audio/dsp.cpp
    src/framework/audio/nemo_mel_frontend.cpp
    src/framework/audio/mel_spectrogram_frontend.cpp
    src/framework/audio/reference_audio_frontend.cpp
    src/framework/audio/utilities/flashsr.cpp
    src/framework/audio/utilities/gtcrn.cpp
    src/framework/audio/istft_graph.cpp
    src/framework/audio/mixing.cpp
    src/framework/audio/resampling.cpp
    src/framework/audio/output.cpp
    src/framework/audio/utilities/rnnoise.cpp
    src/framework/audio/utilities/utility_api.cpp
    src/framework/audio/waveform_ops.cpp
    src/framework/audio/wav_reader.cpp
    src/framework/audio/wav_writer.cpp
    src/framework/audio/utilities/zipenhancer.cpp
    src/framework/runtime/flow_kv_cache.cpp
    src/framework/modules/speaker_encoders/ecapa_tdnn_runtime.cpp
    src/framework/modules/speaker_encoders/titanet_runtime.cpp
    src/framework/codecs/fsq_audio_codec_runtime.cpp
    src/framework/codecs/heartcodec_runtime.cpp
    src/framework/codecs/fish_dac_codec_runtime.cpp
    src/framework/codecs/mel_latent_vae44k_runtime.cpp
    src/framework/codecs/mimi_codec_runtime.cpp
    src/framework/codecs/mimi_latent_decoder_runtime.cpp
    src/framework/codecs/moss_audio_tokenizer_codec_runtime.cpp
    src/framework/codecs/neural_audio.cpp
    src/framework/codecs/oobleck_audio_vae_runtime.cpp
    src/framework/codecs/redae_codec_runtime.cpp
    src/framework/codecs/qwen3_audio_tokenizer_encoder.cpp
    src/framework/codecs/qwen3_audio_tokenizer_decoder.cpp
    src/framework/codecs/snac_decoder_runtime.cpp
    src/framework/codecs/ming_audio_vae_runtime.cpp
    src/framework/codecs/s3gen_runtime.cpp
    src/framework/conditioners/clap_audio_conditioner_runtime.cpp
    src/framework/conditioners/musicgen_style_conditioner_runtime.cpp
    src/framework/conditioners/musiccoca_runtime.cpp
    src/models/controlfoley/flow_denoiser.cpp
    src/framework/sampling/greedy_decode.cpp
    src/framework/sampling/diffusion_math.cpp
    src/framework/sampling/hf_sampler.cpp
    src/framework/sampling/noise.cpp
    src/framework/sampling/torch_random.cpp
)
audiocpp_configure_runtime_object(engine_core)

# These two small built-in model paths are kept outside the selectable composite list for now.
add_library(engine_model_silero_vad OBJECT
    src/models/silero_vad/assets.cpp
    src/models/silero_vad/runtime.cpp
    src/models/silero_vad/session.cpp
)
set_target_properties(engine_model_silero_vad PROPERTIES EXCLUDE_FROM_ALL TRUE)
audiocpp_configure_runtime_object(engine_model_silero_vad)

add_library(engine_model_marblenet_vad OBJECT
    src/models/marblenet_vad/assets.cpp
    src/models/marblenet_vad/runtime.cpp
    src/models/marblenet_vad/session.cpp
)
set_target_properties(engine_model_marblenet_vad PROPERTIES EXCLUDE_FROM_ALL TRUE)
audiocpp_configure_runtime_object(engine_model_marblenet_vad)

audiocpp_add_model(moss_transcribe_diarize
    SOURCES
        src/models/moss_transcribe_diarize/runtime.cpp
        src/models/moss_transcribe_diarize/session.cpp
    INCLUDES
        engine/models/moss_transcribe_diarize/session.h
    LOADERS
        engine::models::moss_transcribe_diarize::make_moss_transcribe_diarize_loader
)

audiocpp_add_model(universr
    SOURCES
        src/models/universr/assets.cpp
        src/models/universr/network.cpp
        src/models/universr/runtime.cpp
        src/models/universr/frontend.cpp
        src/models/universr/session.cpp
    INCLUDES
        engine/models/universr/session.h
    LOADERS
        engine::models::universr::make_universr_loader
)

audiocpp_add_model(sam_audio
    SOURCES
        src/models/sam_audio/frontend.cpp
        src/models/sam_audio/vision.cpp
        src/models/sam_audio/codec.cpp
        src/models/sam_audio/conditioner.cpp
        src/models/sam_audio/dit.cpp
        src/models/sam_audio/session.cpp
    INCLUDES
        engine/models/sam_audio/session.h
    LOADERS
        engine::models::sam_audio::make_sam_audio_loader
)

audiocpp_add_model(mossformer2
    SOURCES
        src/models/mossformer2/assets.cpp
        src/models/mossformer2/routing.cpp
        src/models/mossformer2/runtime.cpp
        src/models/mossformer2/restoration.cpp
        src/models/mossformer2/vocoder.cpp
        src/models/mossformer2/session.cpp
    INCLUDES
        engine/models/mossformer2/session.h
    LOADERS
        engine::models::mossformer2::make_mossformer2_loader
)

audiocpp_add_model(tf_gridnet
    SOURCES
        src/models/tf_gridnet/assets.cpp
        src/models/tf_gridnet/runtime.cpp
        src/models/tf_gridnet/session.cpp
    INCLUDES
        engine/models/tf_gridnet/session.h
    LOADERS
        engine::models::tf_gridnet::make_tf_gridnet_loader
)

audiocpp_add_model(apollo
    SOURCES
        src/models/apollo/assets.cpp
        src/models/apollo/runtime.cpp
        src/models/apollo/session.cpp
    INCLUDES
        engine/models/apollo/session.h
    LOADERS
        engine::models::apollo::make_apollo_loader
)

audiocpp_add_model(sidon
    SOURCES
        src/models/sidon/runtime.cpp
        src/models/sidon/frontend.cpp
        src/models/sidon/session.cpp
    INCLUDES
        engine/models/sidon/session.h
    LOADERS
        engine::models::sidon::make_sidon_loader
)

audiocpp_add_model(tone_color_vc
    SOURCES
        src/models/tone_color_vc/runtime.cpp
        src/models/tone_color_vc/session.cpp
    INCLUDES
        engine/models/tone_color_vc/session.h
    LOADERS
        engine::models::tone_color_vc::make_tone_color_vc_loader
)

audiocpp_add_model(firered_vad
    SOURCES
        src/models/firered_vad/runtime.cpp
        src/models/firered_vad/session.cpp
    INCLUDES
        engine/models/firered_vad/session.h
    LOADERS
        engine::models::firered_vad::make_firered_vad_loader
)

audiocpp_add_model(pulsevad
    SOURCES
        src/models/pulsevad/runtime.cpp
        src/models/pulsevad/session.cpp
    INCLUDES
        engine/models/pulsevad/session.h
    LOADERS
        engine::models::pulsevad::make_pulsevad_loader
)

# Earshot VAD: plain C++ port with bundled weights (assets/framework/models/earshot_vad).
audiocpp_add_model(earshot_vad
    SOURCES
        src/community_models/earshot_vad/runtime.cpp
        src/community_models/earshot_vad/session.cpp
    INCLUDES
        engine/community_models/earshot_vad/session.h
    LOADERS
        engine::community_models::earshot_vad::make_earshot_vad_loader
    ALIASES
        earshot
)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Keep multiply-add rounding identical to the Rust reference.
    set_source_files_properties(src/community_models/earshot_vad/runtime.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()

audiocpp_add_model(smart_turn
    SOURCES
        src/models/smart_turn/runtime.cpp
        src/models/smart_turn/session.cpp
    INCLUDES
        engine/models/smart_turn/session.h
    LOADERS
        engine::models::smart_turn::make_smart_turn_loader
)

audiocpp_add_model(ced
    SOURCES
        src/models/ced/assets.cpp
        src/models/ced/frontend.cpp
        src/models/ced/runtime.cpp
        src/models/ced/session.cpp
    INCLUDES
        engine/models/ced/session.h
    LOADERS
        engine::models::ced::make_ced_loader
)

audiocpp_add_model(lfm_d1_omni
    SOURCES
        src/models/lfm_d1_omni/assets.cpp
        src/models/lfm_d1_omni/audio.cpp
        src/models/lfm_d1_omni/questions.cpp
        src/models/lfm_d1_omni/session.cpp
    INCLUDES
        engine/models/lfm_d1_omni/session.h
    LOADERS
        engine::models::lfm_d1_omni::make_lfm_d1_omni_loader
)

audiocpp_add_model(ast_audioset
    SOURCES
        src/models/ast_audioset/assets.cpp
        src/models/ast_audioset/frontend.cpp
        src/models/ast_audioset/runtime.cpp
        src/models/ast_audioset/session.cpp
    INCLUDES
        engine/models/ast_audioset/session.h
    LOADERS
        engine::models::ast_audioset::make_ast_audioset_loader
)

audiocpp_add_model(roformer
    SOURCES
        src/models/roformer/assets.cpp
        src/models/roformer/runtime.cpp
        src/models/roformer/session.cpp
    INCLUDES
        engine/models/roformer/session.h
    LOADERS
        engine::models::roformer::make_mel_band_roformer_loader
        engine::models::roformer::make_bs_roformer_loader
    ALIASES
        mel_band_roformer
        bs_roformer
)

audiocpp_add_model(demucs
    SOURCES
        src/models/demucs/assets.cpp
        src/models/demucs/frontend.cpp
        src/models/demucs/pipeline.cpp
        src/models/demucs/postprocess.cpp
        src/models/demucs/session.cpp
    INCLUDES
        engine/models/demucs/session.h
    LOADERS
        engine::models::demucs::make_htdemucs_loader
        engine::models::demucs::make_htdemucs_6stems_loader
    ALIASES
        htdemucs
        htdemucs_6stems
        htdemucs_6s
)

audiocpp_add_model(kokoro_tts
    SOURCES
        src/models/kokoro_tts/assets.cpp
        src/models/kokoro_tts/g2p_multilingual.cpp
        src/models/kokoro_tts/frontend.cpp
        src/models/kokoro_tts/predictor.cpp
        src/models/kokoro_tts/decoder.cpp
        src/models/kokoro_tts/rknn_generator.cpp
        src/models/kokoro_tts/plbert.cpp
        src/models/kokoro_tts/session.cpp
    INCLUDES engine/models/kokoro_tts/session.h
    LOADERS engine::models::kokoro_tts::make_kokoro_tts_loader
)
if(MSVC)
    target_compile_options(engine_model_kokoro_tts PRIVATE /utf-8)
endif()
if(ENGINE_ENABLE_NEON)
    target_sources(engine_model_kokoro_tts PRIVATE src/models/kokoro_tts/neon_generator.cpp)
endif()
if(AUDIOCPP_STATIC_ESPEAK)
    target_compile_definitions(engine_model_kokoro_tts PRIVATE AUDIOCPP_STATIC_ESPEAK=1)
endif()

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    # ovz_engine.cpp (omnivoice rknn_zeroop) uses ARMv8.2-A fp16 NEON
    # intrinsics; the board's default -march lacks the +fp16 extension. Scoped
    # to this source: RK3562/RK3561-class boards have no FP16 SIMD at all, so
    # the flag must not leak into unrelated model sources.
    set_source_files_properties(
        src/models/omnivoice/rknn_zeroop/ovz_engine.cpp
        PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+fp16"
    )
endif()

audiocpp_add_model(omnivoice
    SOURCES
        src/models/omnivoice/assets.cpp
        src/models/omnivoice/tokenizer_text.cpp
        src/models/omnivoice/audio_tokenizer.cpp
        src/models/omnivoice/prompt_builder.cpp
        src/models/omnivoice/generator.cpp
        src/models/omnivoice/rknn_generator.cpp
        src/models/omnivoice/rknn_zeroop_generator.cpp
        src/models/omnivoice/rknn_zeroop/ovz_engine.cpp
        src/models/omnivoice/postprocess.cpp
        src/models/omnivoice/session.cpp
        src/models/omnivoice/loader.cpp
    INCLUDES
        engine/models/omnivoice/loader.h
    LOADERS
        engine::models::omnivoice::make_omnivoice_loader
    OBJECT_LIBRARIES
        utf8proc_vendor
)

audiocpp_add_model(reuse
    SOURCES
        src/community_models/reuse/frontend.cpp
        src/community_models/reuse/runtime.cpp
        src/community_models/reuse/session.cpp
    INCLUDES
        engine/community_models/reuse/session.h
    LOADERS
        engine::models::reuse::make_reuse_loader
)

audiocpp_add_model(realtime_reuse
    SOURCES
        src/community_models/realtime_reuse/assets.cpp
        src/community_models/realtime_reuse/runtime.cpp
        src/community_models/realtime_reuse/frontend.cpp
        src/community_models/realtime_reuse/session.cpp
    INCLUDES
        engine/community_models/realtime_reuse/session.h
    LOADERS
        engine::models::realtime_reuse::make_realtime_reuse_loader
)

audiocpp_add_model(voice_isolator
    SOURCES
        src/community_models/voice_isolator/assets.cpp
        src/community_models/voice_isolator/attention_window.cpp
        src/community_models/voice_isolator/config.cpp
        src/community_models/voice_isolator/network.cpp
        src/community_models/voice_isolator/runtime.cpp
        src/community_models/voice_isolator/session.cpp
        src/community_models/voice_isolator/spectral.cpp
    INCLUDES
        engine/community_models/voice_isolator/session.h
    LOADERS
        engine::models::voice_isolator::make_voice_isolator_loader
)

audiocpp_add_model(glm_tts
    SOURCES
        src/community_models/glm_tts/assets.cpp
        src/community_models/glm_tts/flow.cpp
        src/community_models/glm_tts/frontend.cpp
        src/community_models/glm_tts/llama.cpp
        src/community_models/glm_tts/session.cpp
        src/community_models/glm_tts/prompt.cpp
        src/community_models/glm_tts/speech_tokenizer.cpp
        src/community_models/glm_tts/tokenizer_text.cpp
    INCLUDES
        engine/community_models/glm_tts/session.h
    LOADERS
        engine::models::glm_tts::make_glm_tts_loader
    DEPENDS
        outetts
)

audiocpp_add_model(kroko_asr
    SOURCES
        src/community_models/kroko_asr/assets.cpp
        src/community_models/kroko_asr/decoder.cpp
        src/community_models/kroko_asr/encoder.cpp
        src/community_models/kroko_asr/frontend.cpp
        src/community_models/kroko_asr/session.cpp
        src/community_models/kroko_asr/tokenizer.cpp
        src/community_models/kroko_asr/zipformer.cpp
    INCLUDES
        engine/community_models/kroko_asr/session.h
    LOADERS
        engine::models::kroko_asr::make_kroko_asr_loader
)

audiocpp_add_model(open_wake_word
    SOURCES
        src/models/open_wake_word/frontend.cpp
        src/models/open_wake_word/weights.cpp
        src/models/open_wake_word/runtime.cpp
        src/models/open_wake_word/session.cpp
    INCLUDES
        engine/models/open_wake_word/session.h
    LOADERS
        engine::models::open_wake_word::make_open_wake_word_loader
)
if (ENGINE_ENABLE_NEON)
    target_sources(engine_model_open_wake_word PRIVATE src/models/open_wake_word/neon_embedding.cpp)
endif()

audiocpp_add_model(micro_wake_word
    SOURCES
        src/models/micro_wake_word/frontend.cpp
        src/models/micro_wake_word/runtime.cpp
        src/models/micro_wake_word/session.cpp
    INCLUDES
        engine/models/micro_wake_word/session.h
    LOADERS
        engine::models::micro_wake_word::make_micro_wake_word_loader
)
if(ENGINE_ENABLE_NEON)
    target_sources(engine_model_micro_wake_word PRIVATE src/models/micro_wake_word/neon_network.cpp)
endif()

audiocpp_add_model(sherpa_kws
    SOURCES
        src/models/sherpa_kws/assets.cpp
        src/models/sherpa_kws/encoder.cpp
        src/models/sherpa_kws/frontend.cpp
        src/models/sherpa_kws/keyword_decoder.cpp
        src/models/sherpa_kws/session.cpp
        src/models/sherpa_kws/transducer.cpp
        src/models/sherpa_kws/zipformer.cpp
    INCLUDES
        engine/models/sherpa_kws/session.h
    LOADERS
        engine::models::sherpa_kws::make_sherpa_kws_loader
)

audiocpp_add_model(outetts
    SOURCES
        src/community_models/outetts/assets.cpp
        src/community_models/outetts/dac.cpp
        src/community_models/outetts/llama.cpp
        src/community_models/outetts/session.cpp
        src/community_models/outetts/tokenizer.cpp
    INCLUDES
        engine/community_models/outetts/session.h
    LOADERS
        engine::models::outetts::make_outetts_loader
    DEPENDS
        qwen3_asr
        qwen3_forced_aligner
)

audiocpp_add_model(voxtral_tts
    SOURCES
        src/models/voxtral_tts/assets.cpp
        src/models/voxtral_tts/ar.cpp
        src/models/voxtral_tts/acoustic.cpp
        src/models/voxtral_tts/codec.cpp
        src/models/voxtral_tts/session.cpp
    INCLUDES
        engine/models/voxtral_tts/session.h
    LOADERS
        engine::models::voxtral_tts::make_voxtral_tts_loader
)

audiocpp_add_model(kugelaudio
    SOURCES
        src/models/kugelaudio/ar.cpp
        src/models/kugelaudio/assets.cpp
        src/models/kugelaudio/codec.cpp
        src/models/kugelaudio/diffusion.cpp
        src/models/kugelaudio/generator.cpp
        src/models/kugelaudio/reference_encoder.cpp
        src/models/kugelaudio/scheduler.cpp
        src/models/kugelaudio/session.cpp
    INCLUDES
        engine/models/kugelaudio/session.h
    LOADERS
        engine::models::kugelaudio::make_kugelaudio_loader
)

audiocpp_add_model(maya1
    SOURCES
        src/models/maya1/assets.cpp
        src/models/maya1/generator.cpp
        src/models/maya1/session.cpp
        src/models/maya1/tokenizer.cpp
    INCLUDES
        engine/models/maya1/session.h
    LOADERS
        engine::models::maya1::make_maya1_loader
)

audiocpp_add_model(parakeet_tdt
    SOURCES
        src/community_models/parakeet_tdt/routing.cpp
        src/community_models/parakeet_tdt/predictor.cpp
        src/community_models/parakeet_tdt/assets.cpp
        src/community_models/parakeet_tdt/bark3.cpp
        src/community_models/parakeet_tdt/session.cpp
        src/community_models/parakeet_tdt/encoder.cpp
        src/community_models/parakeet_tdt/decoder.cpp
        src/community_models/parakeet_tdt/kda_predictor.cpp
        src/community_models/parakeet_tdt/hotwords.cpp
        src/community_models/parakeet_tdt/phonon_streaming.cpp
        src/community_models/parakeet_tdt/frontend.cpp
        src/community_models/parakeet_tdt/weights.cpp
    INCLUDES
        engine/community_models/parakeet_tdt/session.h
    LOADERS
        engine::community_models::parakeet_tdt::make_parakeet_tdt_loader
        engine::community_models::parakeet_tdt::make_phonon2_asr_loader
        engine::community_models::parakeet_tdt::make_underdog_bark3_loader
        engine::community_models::parakeet_tdt::make_haas_asr_loader
    ALIASES
        phonon2_asr
        underdog_bark3
        haas_asr
)

audiocpp_add_model(personaplex
    SOURCES
        src/models/personaplex/assets.cpp
        src/models/personaplex/depformer.cpp
        src/models/personaplex/lm_runtime.cpp
        src/models/personaplex/request.cpp
        src/models/personaplex/session.cpp
    INCLUDES
        engine/models/personaplex/session.h
    LOADERS
        engine::models::personaplex::make_personaplex_loader
)

set(AUDIOCPP_SANOTTS_NEON_SOURCES src/community_models/sanotts/neon_decoder.cpp src/community_models/sanotts/neon_istft.cpp)
if (ENGINE_ENABLE_NEON)
    set(AUDIOCPP_SANOTTS_NEON_SOURCES
        src/community_models/sanotts/neon_library_decoder.cpp
        src/community_models/sanotts/neon_istft.cpp
        src/community_models/sanotts/neon_weights.cpp)
endif()
audiocpp_add_model(sanotts
    SOURCES
        src/community_models/sanotts/assets.cpp
        src/community_models/sanotts/frontend.cpp
        ${AUDIOCPP_SANOTTS_NEON_SOURCES}
        src/community_models/sanotts/piper_runtime.cpp
        src/community_models/sanotts/runtime.cpp
        src/community_models/sanotts/session.cpp
        src/community_models/sanotts/word_timestamps.cpp
    INCLUDES
        engine/community_models/sanotts/session.h
    LOADERS
        engine::models::sanotts::make_sanotts_loader
)

audiocpp_add_model(inflect_v2
    SOURCES
        src/community_models/inflect_v2/assets.cpp
        src/community_models/inflect_v2/frontend.cpp
        src/community_models/inflect_v2/runtime.cpp
        src/community_models/inflect_v2/rknn_decoder.cpp
        src/community_models/inflect_v2/session.cpp
    INCLUDES
        engine/community_models/inflect_v2/session.h
    LOADERS
        engine::models::inflect_v2::make_inflect_v2_loader
)

audiocpp_add_model(tiny_voca
    SOURCES
        src/community_models/tiny_voca/assets.cpp
        src/community_models/tiny_voca/frontend.cpp
        src/community_models/tiny_voca/runtime.cpp
        src/community_models/tiny_voca/session.cpp
        src/community_models/tiny_voca/source_filter.cpp
        src/community_models/tiny_voca/dizione/align.cpp
        src/community_models/tiny_voca/dizione/convert.cpp
        src/community_models/tiny_voca/dizione/data.cpp
        src/community_models/tiny_voca/dizione/engine.cpp
        src/community_models/tiny_voca/dizione/g2p.cpp
        src/community_models/tiny_voca/dizione/homographs.cpp
        src/community_models/tiny_voca/dizione/json.cpp
        src/community_models/tiny_voca/dizione/morphology.cpp
        src/community_models/tiny_voca/dizione/normalize.cpp
        src/community_models/tiny_voca/dizione/normalize_figures.cpp
        src/community_models/tiny_voca/dizione/normalize_numbers.cpp
        src/community_models/tiny_voca/dizione/pyre.cpp
        src/community_models/tiny_voca/dizione/sentence.cpp
        src/community_models/tiny_voca/dizione/symbols.cpp
        src/community_models/tiny_voca/dizione/tables.cpp
        src/community_models/tiny_voca/dizione/text.cpp
        src/community_models/tiny_voca/dizione/unicode.cpp
    INCLUDES
        engine/community_models/tiny_voca/session.h
    LOADERS
        engine::models::tiny_voca::make_tiny_voca_loader
)
if (MSVC)
    # The phonemizer's sources hold phonetic symbols and currency signs as UTF-8.
    target_compile_options(engine_model_tiny_voca PRIVATE /utf-8)
endif()

audiocpp_add_model(kale_tts
    SOURCES
        src/community_models/kale_tts/assets.cpp
        src/community_models/kale_tts/ettin_tokenizer.cpp
        src/community_models/kale_tts/frontend.cpp
        src/community_models/kale_tts/runtime.cpp
        src/community_models/kale_tts/session.cpp
        src/community_models/kale_tts/source_filter.cpp
    INCLUDES
        engine/community_models/kale_tts/session.h
    LOADERS
        engine::models::kale_tts::make_kale_tts_loader
    OBJECT_LIBRARIES utf8proc_vendor
)

audiocpp_add_model(piper_tts
    SOURCES
        src/community_models/piper_tts/assets.cpp
        src/community_models/piper_tts/frontend.cpp
        src/community_models/piper_tts/vits_runtime.cpp
        src/community_models/piper_tts/rknn_decoder.cpp
        src/community_models/piper_tts/rknn_duration_encoder.cpp
        src/community_models/piper_tts/session.cpp
    INCLUDES
        engine/community_models/piper_tts/session.h
    LOADERS
        engine::models::piper_tts::make_piper_tts_loader
)

audiocpp_add_model(kitten_tts2
    SOURCES
        src/community_models/kitten_tts2/session.cpp
        src/community_models/kitten_tts2/decoder.cpp
        src/community_models/kitten_tts2/speaker.cpp
    INCLUDES engine/community_models/kitten_tts2/session.h
    LOADERS engine::community_models::kitten_tts2::make_kitten_tts2_loader
    DEPENDS chatterbox)

if (ENGINE_ENABLE_NEON)
    target_sources(engine_model_kitten_tts2 PRIVATE src/community_models/kitten_tts2/neon_vocoder.cpp)
endif()

audiocpp_add_model(kitten_tts
    SOURCES
        src/community_models/kitten_tts/assets.cpp
        src/community_models/kitten_tts/decoder.cpp
        src/community_models/kitten_tts/frontend.cpp
        src/community_models/kitten_tts/plbert.cpp
        src/community_models/kitten_tts/predictor.cpp
        src/community_models/kitten_tts/session.cpp
    INCLUDES
        engine/community_models/kitten_tts/session.h
    LOADERS
        engine::models::kitten_tts::make_kitten_tts_loader
)
if(MSVC)
    target_compile_options(engine_model_kitten_tts PRIVATE /utf-8)
endif()

audiocpp_add_model(audio_flamingo
    SOURCES
        src/models/audio_flamingo/assets.cpp
        src/models/audio_flamingo/audio_encoder.cpp
        src/models/audio_flamingo/decoder.cpp
        src/models/audio_flamingo/frontend.cpp
        src/models/audio_flamingo/projector.cpp
        src/models/audio_flamingo/session.cpp
        src/models/audio_flamingo/tokenizer_text.cpp
    INCLUDES
        engine/models/audio_flamingo/session.h
    LOADERS
        engine::models::audio_flamingo::make_audio_flamingo_loader
)

audiocpp_add_model(nemotron_transcribe
    SOURCES
        src/community_models/nemotron_transcribe/assets.cpp
        src/community_models/nemotron_transcribe/frontend.cpp
        src/community_models/nemotron_transcribe/encoder.cpp
        src/community_models/nemotron_transcribe/decoder.cpp
        src/community_models/nemotron_transcribe/repetition_guard.cpp
        src/community_models/nemotron_transcribe/tokenizer.cpp
        src/community_models/nemotron_transcribe/ctc.cpp
        src/community_models/nemotron_transcribe/alignment.cpp
        src/community_models/nemotron_transcribe/session.cpp
    INCLUDES engine/community_models/nemotron_transcribe/session.h
    LOADERS engine::models::nemotron_transcribe::make_nemotron_transcribe_loader
    OBJECT_LIBRARIES utf8proc_vendor
)
# The aligner reproduces NumPy/PyTorch operation ordering. Fused operations in
# that source are explicit std::fma calls.
if(MSVC)
    set_source_files_properties(src/community_models/nemotron_transcribe/alignment.cpp src/framework/decoders/ctc_host_decoder.cpp PROPERTIES COMPILE_OPTIONS "/fp:strict")
else()
    set_source_files_properties(src/community_models/nemotron_transcribe/alignment.cpp src/framework/decoders/ctc_host_decoder.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()

audiocpp_add_model(samsone
    SOURCES
        src/models/samsone/assets.cpp
        src/models/samsone/runtime.cpp
        src/models/samsone/session.cpp
    INCLUDES
        engine/models/samsone/session.h
    LOADERS
        engine::models::samsone::make_samsone_loader
)

audiocpp_add_model(f5_tts
    SOURCES
        src/community_models/f5_tts/session.cpp
        src/community_models/f5_tts/runtime.cpp
        src/community_models/f5_tts/weights.cpp
        src/community_models/f5_tts/dit_modules.cpp
        src/community_models/f5_tts/synthesize.cpp
    INCLUDES
        engine/community_models/f5_tts/session.h
        engine/community_models/f5_tts/runtime.h
    LOADERS
        engine::models::f5_tts::make_f5_tts_loader
    ALIASES
        habibi
        habibi_tts
)

audiocpp_add_model(zipvoice
    SOURCES
        src/community_models/zipvoice/session.cpp
        src/community_models/zipvoice/synthesize.cpp
        src/community_models/zipvoice/emilia_tokenizer.cpp
        src/community_models/zipvoice/jieba_segmenter.cpp
        src/community_models/zipvoice/weights.cpp
        src/community_models/zipvoice/zipformer.cpp
    INCLUDES
        engine/community_models/zipvoice/session.h
        engine/community_models/zipvoice/synthesize.h
    LOADERS
        engine::models::zipvoice::make_zipvoice_loader
)

audiocpp_add_model(minimax_h3
    SOURCES
        src/community_models/minimax_h3/assets.cpp
        src/community_models/minimax_h3/audio_vae_decoder.cpp
        src/community_models/minimax_h3/dit_acceleration.cpp
        src/community_models/minimax_h3/dit_denoiser.cpp
        src/community_models/minimax_h3/generation_plan.cpp
        src/community_models/minimax_h3/pipeline.cpp
        src/community_models/minimax_h3/sampler.cpp
        src/community_models/minimax_h3/session.cpp
        src/community_models/minimax_h3/video_vae_decoder.cpp
    INCLUDES
        engine/community_models/minimax_h3/session.h
    LOADERS
        engine::models::minimax_h3::make_minimax_h3_loader
)

audiocpp_add_model(latentsync
    SOURCES
        src/community_models/latentsync/whisper.cpp
        src/community_models/latentsync/unet.cpp
        src/community_models/latentsync/sd_vae.cpp
        src/community_models/latentsync/face.cpp
        src/community_models/latentsync/sampling.cpp
        src/community_models/latentsync/pipeline.cpp
        src/community_models/latentsync/video.cpp
        src/community_models/latentsync/session.cpp
    INCLUDES
        engine/community_models/latentsync/session.h
    LOADERS
        engine::models::latentsync::make_latentsync_loader
)
if (GGML_CUDA_CUDNN_CONV2D)
    target_compile_definitions(engine_model_latentsync PRIVATE GGML_CUDA_CUDNN_CONV2D=1)
endif()

audiocpp_add_model(soulx_flashhead
    SOURCES
        src/models/soulx_flashhead/color.cpp
        src/models/soulx_flashhead/audio.cpp
        src/models/soulx_flashhead/assets.cpp
        src/models/soulx_flashhead/dit.cpp
        src/models/soulx_flashhead/vae.cpp
        src/models/soulx_flashhead/portrait.cpp
        src/models/soulx_flashhead/pipeline.cpp
        src/models/soulx_flashhead/session.cpp
    INCLUDES
        engine/models/soulx_flashhead/session.h
    LOADERS
        engine::models::soulx_flashhead::make_soulx_flashhead_loader
)

if (TARGET engine_model_soulx_flashhead)
    if (GGML_CUDA_CUDNN_CONV3D AND ENGINE_ENABLE_CUDA)
        target_compile_definitions(engine_model_soulx_flashhead PRIVATE GGML_CUDA_CUDNN_CONV3D=1)
    endif()
    if (GGML_CUDA_CUDNN_CONV2D)
        target_compile_definitions(engine_model_soulx_flashhead PRIVATE GGML_CUDA_CUDNN_CONV2D=1)
    endif()
    add_executable(soulx_flashhead_components EXCLUDE_FROM_ALL tests/soulx_flashhead/components.cpp)
    target_link_libraries(soulx_flashhead_components PRIVATE engine_runtime ggml)
    add_executable(soulx_flashhead_session_test EXCLUDE_FROM_ALL tests/soulx_flashhead/session_test.cpp)
    target_link_libraries(soulx_flashhead_session_test PRIVATE engine_runtime ggml)
endif()

audiocpp_add_model(liveavatar
    SOURCES
        src/community_models/liveavatar/assets.cpp
        src/community_models/liveavatar/pipeline.cpp
        src/community_models/liveavatar/pipeline_state.cpp
        src/community_models/liveavatar/pipeline_text_encoder.cpp
        src/community_models/liveavatar/pipeline_vae.cpp
        src/community_models/liveavatar/pipeline_denoiser.cpp
        src/community_models/liveavatar/pipeline_liveavatar.cpp
        src/community_models/liveavatar/pipeline_offline.cpp
        src/community_models/liveavatar/wan_s2v_audio_conditioner.cpp
        src/community_models/liveavatar/session.cpp
    INCLUDES
        engine/community_models/liveavatar/session.h
    LOADERS
        engine::community_models::liveavatar::make_liveavatar_loader
)

audiocpp_add_model(minimax_music3
    SOURCES
        src/community_models/minimax_music3/assets.cpp
        src/community_models/minimax_music3/prompt.cpp
        src/community_models/minimax_music3/global_lm.cpp
        src/community_models/minimax_music3/depth_decoder.cpp
        src/community_models/minimax_music3/ar_runtime.cpp
        src/community_models/minimax_music3/condition_encoder.cpp
        src/community_models/minimax_music3/flow_transformer.cpp
        src/community_models/minimax_music3/flow_sampler.cpp
        src/community_models/minimax_music3/vocoder.cpp
        src/community_models/minimax_music3/pipeline.cpp
        src/community_models/minimax_music3/session.cpp
    INCLUDES
        engine/community_models/minimax_music3/session.h
    LOADERS
        engine::models::minimax_music3::make_minimax_music3_loader
)

audiocpp_add_model(neutts
    SOURCES
        src/models/neutts/ar.cpp
        src/models/neutts/assets.cpp
        src/models/neutts/backbone.cpp
        src/models/neutts/codec.cpp
        src/models/neutts/prompt.cpp
        src/models/neutts/session.cpp
    INCLUDES
        engine/models/neutts/ar.h
        engine/models/neutts/assets.h
        engine/models/neutts/backbone.h
        engine/models/neutts/codec.h
        engine/models/neutts/prompt.h
        engine/models/neutts/session.h
    LOADERS
        engine::models::neutts::make_neutts_loader
)

audiocpp_add_model(moss_soundeffect
    SOURCES
        src/models/moss_soundeffect/assets.cpp
        src/models/moss_soundeffect/dit.cpp
        src/models/moss_soundeffect/vae.cpp
        src/models/moss_soundeffect/text_encoder.cpp
        src/models/moss_soundeffect/session.cpp
    INCLUDES
        engine/models/moss_soundeffect/session.h
    LOADERS
        engine::models::moss_soundeffect::make_moss_soundeffect_loader
    OBJECT_LIBRARIES
        utf8proc_vendor
)

audiocpp_add_model(echo_tts
    SOURCES
        src/community_models/echo_tts/session.cpp
        src/community_models/echo_tts/tokenizer.cpp
        src/community_models/echo_tts/latent_post.cpp
        src/community_models/echo_tts/sampler.cpp
        src/community_models/echo_tts/dit.cpp
    INCLUDES
        engine/community_models/echo_tts/session.h
        engine/community_models/echo_tts/config.h
        engine/community_models/echo_tts/tokenizer.h
        engine/community_models/echo_tts/latent_post.h
        engine/community_models/echo_tts/sampler.h
        engine/community_models/echo_tts/dit.h
    LOADERS
        engine::models::echo_tts::make_echo_tts_loader
)

if (MSVC)
    set_source_files_properties(
        src/community_models/inflect_v2/frontend.cpp
        PROPERTIES COMPILE_OPTIONS /utf-8
    )
endif()

audiocpp_add_model(pocket_tts
    SOURCES
        src/models/pocket_tts/assets.cpp
        src/models/pocket_tts/backend_weights.cpp
        src/models/pocket_tts/acoustic_model.cpp
        src/models/pocket_tts/flow_lm.cpp
        src/models/pocket_tts/mimi_encoder.cpp
        src/models/pocket_tts/text_conditioner.cpp
        src/models/pocket_tts/voice_state_assets.cpp
        src/models/pocket_tts/voice_conditioner.cpp
        src/models/pocket_tts/loader.cpp
        src/models/pocket_tts/session.cpp
        src/models/pocket_tts/gradium_phonon.cpp
    INCLUDES
        engine/models/pocket_tts/loader.h
    LOADERS
        engine::models::pocket_tts::make_pocket_tts_loader
        engine::models::pocket_tts::make_gradium_phonon_loader
    ALIASES
        gradium_phonon
)

audiocpp_add_model(pocket_tts_small
    SOURCES
        src/models/pocket_tts_small/assets.cpp
        src/models/pocket_tts_small/conditioning.cpp
        src/models/pocket_tts_small/flow_lm.cpp
        src/models/pocket_tts_small/mimi_encoder.cpp
        src/models/pocket_tts_small/session.cpp
    INCLUDES
        engine/models/pocket_tts_small/session.h
    LOADERS
        engine::models::pocket_tts_small::make_pocket_tts_small_loader
)

if (ENGINE_ENABLE_NEON)
    target_sources(engine_model_pocket_tts_small PRIVATE src/models/pocket_tts_small/native_flow.cpp)
endif()

audiocpp_add_model(miocodec
    SOURCES
        src/models/miocodec/assets.cpp
        src/models/miocodec/weights.cpp
        src/models/miocodec/audio_pipeline.cpp
        src/models/miocodec/graph_ops.cpp
        src/models/miocodec/graphs.cpp
        src/models/miocodec/session.cpp
        src/models/miocodec/loader.cpp
    INCLUDES
        engine/models/miocodec/loader.h
    LOADERS
        engine::models::miocodec::make_miocodec_loader
)

audiocpp_add_model(miotts
    SOURCES
        src/models/miotts/assets.cpp
        src/models/miotts/tokenizer.cpp
        src/models/miotts/causal_lm.cpp
        src/models/miotts/session.cpp
        src/models/miotts/loader.cpp
    INCLUDES
        engine/models/miotts/loader.h
    LOADERS
        engine::models::miotts::make_miotts_loader
    DEPENDS
        miocodec
        qwen3_asr
)

audiocpp_add_model(moss
    SOURCES
        src/models/moss/moss_tts_local/assets.cpp
        src/models/moss/moss_tts_local/backbone.cpp
        src/models/moss/moss_tts_nano/assets.cpp
        src/models/moss/moss_tts_nano/generator.cpp
        src/models/moss/moss_tts_nano/global_transformer.cpp
        src/models/moss/moss_tts_nano/local_frame_decoder.cpp
        src/models/moss/moss_tts_nano/loader.cpp
        src/models/moss/moss_tts_nano/prompt_builder.cpp
        src/models/moss/moss_tts_nano/session.cpp
        src/models/moss/moss_tts_nano/tokenizer_text.cpp
        src/models/moss/moss_tts_local/depth_transformer.cpp
        src/models/moss/moss_tts_local/generator.cpp
        src/models/moss/moss_tts_local/loader.cpp
        src/models/moss/moss_tts_local/tokenizer_text.cpp
        src/models/moss/moss_tts_local/session.cpp
    INCLUDES
        engine/models/moss/moss_tts_local/loader.h
        engine/models/moss/moss_tts_nano/loader.h
    LOADERS
        engine::models::moss_tts_local::make_moss_tts_local_loader
        engine::models::moss_tts_nano::make_moss_tts_nano_loader
    ALIASES
        moss_tts_local
        moss_tts_nano
)

audiocpp_add_model(moss_ttsd
    SOURCES
        src/community_models/moss_ttsd/assets.cpp
        src/community_models/moss_ttsd/prompt.cpp
        src/community_models/moss_ttsd/session.cpp
        src/community_models/moss_ttsd/loader.cpp
    INCLUDES
        engine/community_models/moss_ttsd/loader.h
    LOADERS
        engine::models::moss_ttsd::make_moss_ttsd_loader
)

audiocpp_add_model(moss_tts_v15
    SOURCES
        src/community_models/moss_tts_v15/assets.cpp
        src/community_models/moss_tts_v15/prompt.cpp
        src/community_models/moss_tts_v15/session.cpp
        src/community_models/moss_tts_v15/loader.cpp
    INCLUDES
        engine/community_models/moss_tts_v15/loader.h
    LOADERS
        engine::models::moss_tts_v15::make_moss_tts_v15_loader
)

audiocpp_add_model(moss_voicegen
    SOURCES
        src/community_models/moss_voicegen/assets.cpp
        src/community_models/moss_voicegen/loader.cpp
        src/community_models/moss_voicegen/session.cpp
        src/community_models/moss_voicegen/tokenizer_text.cpp
    INCLUDES
        engine/community_models/moss_voicegen/loader.h
    LOADERS
        engine::models::moss_voicegen::make_moss_voicegen_loader
    DEPENDS
        moss
)

audiocpp_add_model(dettato
    SOURCES
        src/community_models/dettato/assets.cpp src/community_models/dettato/frontend.cpp
        src/community_models/dettato/transcriber.cpp src/community_models/dettato/note_repair.cpp
        src/community_models/dettato/grouper.cpp src/community_models/dettato/drums.cpp
        src/community_models/dettato/gru.cpp src/community_models/dettato/pipeline.cpp src/community_models/dettato/session.cpp
    INCLUDES engine/community_models/dettato/session.h
    LOADERS engine::community_models::dettato::make_dettato_loader
)

audiocpp_add_model(rosvot
    SOURCES src/models/rosvot/runtime.cpp src/models/rosvot/session.cpp
    INCLUDES engine/models/rosvot/session.h
    LOADERS engine::models::rosvot::make_rosvot_loader
)

audiocpp_add_model(muscriptor
    SOURCES
        src/models/muscriptor/assets.cpp
        src/models/muscriptor/frontend.cpp
        src/models/muscriptor/tokenizer.cpp
        src/models/muscriptor/decoder.cpp
        src/models/muscriptor/session.cpp
    INCLUDES
        engine/models/muscriptor/session.h
    LOADERS
        engine::models::muscriptor::make_muscriptor_loader
)

audiocpp_add_model(voxcpm2
    SOURCES
        src/models/voxcpm2/assets.cpp
        src/models/voxcpm2/audiovae.cpp
        src/models/voxcpm2/generator.cpp
        src/models/voxcpm2/loader.cpp
        src/models/voxcpm2/minicpm.cpp
        src/models/voxcpm2/session.cpp
        src/models/voxcpm2/tokenizer_text.cpp
    INCLUDES
        engine/models/voxcpm2/loader.h
    LOADERS
        engine::models::voxcpm2::make_voxcpm2_loader
)

audiocpp_add_model(voxcpm1
    SOURCES
        src/community_models/voxcpm1/assets.cpp
        src/community_models/voxcpm1/audiovae.cpp
        src/community_models/voxcpm1/generator.cpp
        src/community_models/voxcpm1/minicpm.cpp
        src/community_models/voxcpm1/session.cpp
        src/community_models/voxcpm1/tokenizer_text.cpp
    INCLUDES
        engine/community_models/voxcpm1/session.h
    LOADERS
        engine::community_models::voxcpm1::make_voxcpm1_loader
)

audiocpp_add_model(vibevoice
    SOURCES
        src/models/vibevoice/assets.cpp
        src/models/vibevoice/connector.cpp
        src/models/vibevoice/decoder.cpp
        src/models/vibevoice/diffusion_head.cpp
        src/models/vibevoice/diffusion_sampler.cpp
        src/models/vibevoice/generator.cpp
        src/models/vibevoice/loader.cpp
        src/models/vibevoice/lora.cpp
        src/models/vibevoice/scheduler.cpp
        src/models/vibevoice/session.cpp
        src/models/vibevoice/tokenizer_audio.cpp
        src/models/vibevoice/tokenizer_text.cpp
    INCLUDES
        engine/models/vibevoice/loader.h
    LOADERS
        engine::models::vibevoice::make_vibevoice_loader
)

audiocpp_add_model(vibevoice_asr
    SOURCES
        src/models/vibevoice_asr/assets.cpp
        src/models/vibevoice_asr/connector.cpp
        src/models/vibevoice_asr/frontend.cpp
        src/models/vibevoice_asr/loader.cpp
        src/models/vibevoice_asr/postprocess.cpp
        src/models/vibevoice_asr/session.cpp
        src/models/vibevoice_asr/speech_encoder.cpp
        src/models/vibevoice_asr/speech_tokenizer.cpp
        src/models/vibevoice_asr/text_decoder.cpp
        src/models/vibevoice_asr/tokenizer_text.cpp
    INCLUDES
        engine/models/vibevoice_asr/loader.h
    LOADERS
        engine::models::vibevoice_asr::make_vibevoice_asr_loader
        engine::models::vibevoice_asr::make_vibevoice_asr_streaming_loader
)

audiocpp_add_model(voxtral_realtime
    SOURCES
        src/models/voxtral_realtime/assets.cpp
        src/models/voxtral_realtime/tokenizer_text.cpp
        src/models/voxtral_realtime/frontend.cpp
        src/models/voxtral_realtime/audio_encoder.cpp
        src/models/voxtral_realtime/text_decoder.cpp
        src/models/voxtral_realtime/session.cpp
        src/models/voxtral_realtime/loader.cpp
    INCLUDES
        engine/models/voxtral_realtime/loader.h
    LOADERS
        engine::models::voxtral_realtime::make_voxtral_realtime_loader
)

audiocpp_add_model(fish_audio
    SOURCES
        src/models/fish_audio/ar.cpp
        src/models/fish_audio/assets.cpp
        src/models/fish_audio/generator.cpp
        src/models/fish_audio/loader.cpp
        src/models/fish_audio/prompt_builder.cpp
        src/models/fish_audio/session.cpp
        src/models/fish_audio/tokenizer_text.cpp
    INCLUDES
        engine/models/fish_audio/loader.h
    LOADERS
        engine::models::fish_audio::make_fish_audio_loader
)

add_library(utf8proc_vendor OBJECT external/utf8proc/utf8proc.c)
set_target_properties(utf8proc_vendor PROPERTIES EXCLUDE_FROM_ALL TRUE POSITION_INDEPENDENT_CODE ON)
target_include_directories(utf8proc_vendor PUBLIC external/utf8proc)
target_compile_definitions(utf8proc_vendor PUBLIC UTF8PROC_STATIC)

audiocpp_add_model(ming3_audio
    SOURCES
        src/models/ming3_audio/assets.cpp
        src/models/ming3_audio/audio_encoder.cpp
        src/models/ming3_audio/backbone.cpp
        src/models/ming3_audio/frontend.cpp
        src/models/ming3_audio/session.cpp
        src/models/ming3_audio/talker.cpp
        src/models/ming3_audio/talker_frontend.cpp
    INCLUDES
        engine/models/ming3_audio/session.h
    LOADERS
        engine::models::ming3_audio::make_ming3_audio_loader
)

audiocpp_add_model(ming3_tts
    SOURCES
        src/models/ming3_tts/assets.cpp
        src/models/ming3_tts/generator.cpp
        src/models/ming3_tts/prompt.cpp
        src/models/ming3_tts/session.cpp
    INCLUDES
        engine/models/ming3_tts/session.h
    LOADERS
        engine::models::ming3_tts::make_ming3_tts_loader
)

audiocpp_add_model(plapre
    SOURCES
        src/models/plapre/assets.cpp
        src/models/plapre/frontend.cpp
        src/models/plapre/generator.cpp
        src/models/plapre/kanade.cpp
        src/models/plapre/prompt.cpp
        src/models/plapre/session.cpp
    INCLUDES
        engine/models/plapre/session.h
    LOADERS
        engine::models::plapre::make_plapre_loader
    OBJECT_LIBRARIES
        utf8proc_vendor
)

audiocpp_add_model(fish_drama3
    SOURCES
        src/community_models/fish_drama3/assets.cpp
        src/community_models/fish_drama3/backbone.cpp
        src/community_models/fish_drama3/generator.cpp
        src/community_models/fish_drama3/loader.cpp
        src/community_models/fish_drama3/prompt.cpp
        src/community_models/fish_drama3/reference_cache.cpp
        src/community_models/fish_drama3/session.cpp
        src/community_models/fish_drama3/tokenizer_text.cpp
        src/community_models/fish_drama3/tower.cpp
    INCLUDES
        engine/community_models/fish_drama3/loader.h
    LOADERS
        engine::models::fish_drama3::make_fish_drama3_loader
    OBJECT_LIBRARIES
        utf8proc_vendor
)

audiocpp_add_model(zonos3
    SOURCES
        src/community_models/zonos3/assets.cpp
        src/community_models/zonos3/backbone.cpp
        src/community_models/zonos3/codec_decoder.cpp
        src/community_models/zonos3/generator.cpp
        src/community_models/zonos3/loader.cpp
        src/community_models/zonos3/npy.cpp
        src/community_models/zonos3/prompt.cpp
        src/community_models/zonos3/reference_encoder.cpp
        src/community_models/zonos3/resample.cpp
        src/community_models/zonos3/sampler.cpp
        src/community_models/zonos3/session.cpp
        src/community_models/zonos3/speaker_encoder.cpp
        src/community_models/zonos3/ssl_teacher.cpp
        src/community_models/zonos3/textnorm.cpp
        src/community_models/zonos3/tokenizer.cpp
    INCLUDES
        engine/community_models/zonos3/loader.h
    LOADERS
        engine::models::zonos3::make_zonos3_loader
)

audiocpp_add_model(zonos2
    SOURCES
        src/community_models/zonos2/assets.cpp
        src/community_models/zonos2/backbone.cpp
        src/community_models/zonos2/codec_decoder.cpp
        src/community_models/zonos2/generator.cpp
        src/community_models/zonos2/loader.cpp
        src/community_models/zonos2/npy.cpp
        src/community_models/zonos2/prompt.cpp
        src/community_models/zonos2/sampler.cpp
        src/community_models/zonos2/session.cpp
        src/community_models/zonos2/speaker_encoder.cpp
    INCLUDES
        engine/community_models/zonos2/loader.h
    LOADERS
        engine::models::zonos2::make_zonos2_loader
)

audiocpp_add_model(audio8_tts
    SOURCES
        src/community_models/audio8_tts/ar.cpp
        src/community_models/audio8_tts/assets.cpp
        src/community_models/audio8_tts/codec.cpp
        src/community_models/audio8_tts/generator.cpp
        src/community_models/audio8_tts/prompt_builder.cpp
        src/community_models/audio8_tts/session.cpp
        src/community_models/audio8_tts/tokenizer_text.cpp
    INCLUDES
        engine/community_models/audio8_tts/session.h
    LOADERS
        engine::models::audio8_tts::make_audio8_tts_loader
)

audiocpp_add_model(magpie_tts
    SOURCES
        src/models/magpie_tts/assets.cpp
        src/models/magpie_tts/request.cpp
        src/models/magpie_tts/runtime.cpp
        src/models/magpie_tts/session.cpp
        src/models/magpie_tts/tokenizer_text.cpp
    INCLUDES
        engine/models/magpie_tts/request.h
        engine/models/magpie_tts/runtime.h
        engine/models/magpie_tts/session.h
        engine/models/magpie_tts/tokenizer_text.h
        engine/models/magpie_tts/types.h
    LOADERS
        engine::models::magpie_tts::make_magpie_tts_loader
)

audiocpp_add_model(confucius4_tts
    SOURCES
        src/models/confucius4_tts/assets.cpp
        src/models/confucius4_tts/audio_features.cpp
        src/models/confucius4_tts/request.cpp
        src/models/confucius4_tts/s2a.cpp
        src/models/confucius4_tts/semantic_encoder.cpp
        src/models/confucius4_tts/session.cpp
        src/models/confucius4_tts/style_encoder.cpp
        src/models/confucius4_tts/t2s.cpp
        src/models/confucius4_tts/tokenizer_text.cpp
        src/models/confucius4_tts/vocoder.cpp
    INCLUDES
        engine/models/confucius4_tts/session.h
    LOADERS
        engine::models::confucius4_tts::make_confucius4_tts_loader
)

audiocpp_add_model(dots_tts
    SOURCES
        src/models/dots_tts/assets.cpp
        src/models/dots_tts/audio_features.cpp
        src/models/dots_tts/audio_vae.cpp
        src/models/dots_tts/flow.cpp
        src/models/dots_tts/flow_meanflow.cpp
        src/models/dots_tts/flow_soar.cpp
        src/models/dots_tts/latent.cpp
        src/models/dots_tts/llm.cpp
        src/models/dots_tts/patch_encoder.cpp
        src/models/dots_tts/request.cpp
        src/models/dots_tts/session.cpp
        src/models/dots_tts/tokenizer.cpp
    INCLUDES
        engine/models/dots_tts/session.h
    LOADERS
        engine::models::dots_tts::make_dots_tts_loader
)

audiocpp_add_model(dramabox
    SOURCES
        src/models/dramabox/assets.cpp
        src/models/dramabox/audio_vae.cpp
        src/models/dramabox/dit.cpp
        src/models/dramabox/gemma_tokenizer.cpp
        src/models/dramabox/gemma3_encoder.cpp
        src/models/dramabox/latent_state.cpp
        src/models/dramabox/prompt_connector.cpp
        src/models/dramabox/session.cpp
        src/models/dramabox/vocoder.cpp
    INCLUDES
        engine/models/dramabox/session.h
    LOADERS
        engine::models::dramabox::make_dramabox_loader
)

audiocpp_add_model(fun_asr_nano
    SOURCES
        src/models/fun_asr_nano/assets.cpp
        src/models/fun_asr_nano/frontend.cpp
        src/models/fun_asr_nano/encoder.cpp
        src/models/fun_asr_nano/adaptor.cpp
        src/models/fun_asr_nano/tokenizer_text.cpp
        src/models/fun_asr_nano/prompt.cpp
        src/models/fun_asr_nano/decoder.cpp
        src/models/fun_asr_nano/session.cpp
        src/models/fun_asr_nano/loader.cpp
    INCLUDES
        engine/models/fun_asr_nano/loader.h
    LOADERS
        engine::models::fun_asr_nano::make_fun_asr_nano_loader
)

audiocpp_add_model(sense_asr
    SOURCES
        src/community_models/sense_asr/assets.cpp
        src/community_models/sense_asr/frontend.cpp
        src/community_models/sense_asr/encoder.cpp
        src/community_models/sense_asr/session.cpp
    INCLUDES
        engine/community_models/sense_asr/session.h
    LOADERS
        engine::community_models::sense_asr::make_sense_asr_loader
)

audiocpp_add_model(heartmula
    SOURCES
        src/models/heartmula/assets.cpp
        src/models/heartmula/generator.cpp
        src/models/heartmula/mula.cpp
        src/models/heartmula/session.cpp
        src/models/heartmula/tokenizer_text.cpp
    INCLUDES
        engine/models/heartmula/session.h
    LOADERS
        engine::models::heartmula::make_heartmula_loader
)

audiocpp_add_model(mulacover
    SOURCES
        src/models/mulacover/ar.cpp
        src/models/mulacover/assets.cpp
        src/models/mulacover/chord.cpp
        src/models/mulacover/conditioning.cpp
        src/models/mulacover/frontend.cpp
        src/models/mulacover/prompt.cpp
        src/models/mulacover/session.cpp
        src/models/mulacover/style_encoder.cpp
        src/models/mulacover/symbolic.cpp
        src/models/mulacover/transcriber.cpp
    INCLUDES
        engine/models/mulacover/session.h
    LOADERS
        engine::models::mulacover::make_mulacover_loader
)

audiocpp_add_model(higgs_audio_stt
    SOURCES
        src/models/higgs_audio_stt/assets.cpp
        src/models/higgs_audio_stt/tokenizer_text.cpp
        src/models/higgs_audio_stt/frontend_whisper.cpp
        src/models/higgs_audio_stt/audio_encoder.cpp
        src/models/higgs_audio_stt/text_decoder.cpp
        src/models/higgs_audio_stt/prompt_asr.cpp
        src/models/higgs_audio_stt/postprocess.cpp
        src/models/higgs_audio_stt/session.cpp
    INCLUDES
        engine/models/higgs_audio_stt/session.h
    LOADERS
        engine::models::higgs_audio_stt::make_higgs_audio_stt_loader
)

audiocpp_add_model(higgs_audio_tts
    SOURCES
        src/models/higgs_audio_tts/ar.cpp
        src/models/higgs_audio_tts/assets.cpp
        src/models/higgs_audio_tts/codec.cpp
        src/models/higgs_audio_tts/codebooks.cpp
        src/models/higgs_audio_tts/generator.cpp
        src/models/higgs_audio_tts/loader.cpp
        src/models/higgs_audio_tts/sampler.cpp
        src/models/higgs_audio_tts/session.cpp
        src/models/higgs_audio_tts/tokenizer_text.cpp
    INCLUDES
        engine/models/higgs_audio_tts/loader.h
    LOADERS
        engine::models::higgs_audio_tts::make_higgs_audio_tts_loader
)

audiocpp_add_model(hviske_asr
    SOURCES
        src/models/hviske_asr/assets.cpp
        src/models/hviske_asr/frontend.cpp
        src/models/hviske_asr/encoder.cpp
        src/models/hviske_asr/decoder.cpp
        src/models/hviske_asr/session.cpp
        src/models/hviske_asr/weights.cpp
    INCLUDES
        engine/models/hviske_asr/session.h
    LOADERS
        engine::models::hviske_asr::make_hviske_asr_loader
)

audiocpp_add_model(hviske_asr_v6
    SOURCES
        src/models/hviske_asr_v6/assets.cpp
        src/models/hviske_asr_v6/frontend.cpp
        src/models/hviske_asr_v6/encoder.cpp
        src/models/hviske_asr_v6/decoder.cpp
        src/models/hviske_asr_v6/session.cpp
    INCLUDES
        engine/models/hviske_asr_v6/session.h
    LOADERS
        engine::models::hviske_asr_v6::make_hviske_asr_v6_loader
)

audiocpp_add_model(irodori_tts
    SOURCES
        src/models/irodori_tts/assets.cpp
        src/models/irodori_tts/codec.cpp
        src/models/irodori_tts/condition_encoder.cpp
        src/models/irodori_tts/duration.cpp
        src/models/irodori_tts/mf_dit.cpp
        src/models/irodori_tts/rf_dit.cpp
        src/models/irodori_tts/session.cpp
        src/models/irodori_tts/tokenizer_text.cpp
    INCLUDES
        engine/models/irodori_tts/session.h
    LOADERS
        engine::models::irodori_tts::make_irodori_tts_loader
    OBJECT_LIBRARIES
        utf8proc_vendor
)

audiocpp_add_model(index_tts2
    SOURCES
        src/models/index_tts2/assets.cpp
        src/models/index_tts2/audio_features.cpp
        src/models/index_tts2/gpt.cpp
        src/models/index_tts2/loader.cpp
        src/models/index_tts2/qwen_emotion.cpp
        src/models/index_tts2/request.cpp
        src/models/index_tts2/s2mel.cpp
        src/models/index_tts2/semantic_codec.cpp
        src/models/index_tts2/semantic_encoder.cpp
        src/models/index_tts2/session.cpp
        src/models/index_tts2/style_encoder.cpp
        src/models/index_tts2/tokenizer_text.cpp
        src/models/index_tts2/vocoder.cpp
    INCLUDES
        engine/models/index_tts2/loader.h
    LOADERS
        engine::models::index_tts2::make_index_tts2_loader
)

audiocpp_add_model(nemotron_asr
    SOURCES
        src/models/nemotron_asr/assets.cpp
        src/models/nemotron_asr/frontend.cpp
        src/models/nemotron_asr/encoder.cpp
        src/models/nemotron_asr/decoder.cpp
        src/models/nemotron_asr/session.cpp
        src/models/nemotron_asr/speaker_tagging.cpp
        src/models/nemotron_asr/masked_asr.cpp
        src/models/nemotron_asr/loader.cpp
        src/models/nemotron_asr/weights.cpp
    INCLUDES
        engine/models/nemotron_asr/loader.h
    LOADERS
        engine::models::nemotron_asr::make_nemotron_asr_loader
)

audiocpp_add_model(gigaam_asr
    SOURCES
        src/models/gigaam_asr/assets.cpp
        src/models/gigaam_asr/conformer.cpp
        src/models/gigaam_asr/frontend.cpp
        src/models/gigaam_asr/weights.cpp
        src/models/gigaam_asr/runtime.cpp
        src/models/gigaam_asr/session.cpp
    INCLUDES
        engine/models/gigaam_asr/model.h
    LOADERS
        engine::models::gigaam_asr::make_gigaam_asr_loader
)

audiocpp_add_model(crisperwhisper
    SOURCES
        src/models/crisperwhisper/assets.cpp
        src/models/crisperwhisper/alignment.cpp
        src/models/crisperwhisper/transcript_alignment.cpp
        src/models/crisperwhisper/whisper_runtime.cpp
        src/models/crisperwhisper/session.cpp
    INCLUDES
        engine/models/crisperwhisper/model.h
    LOADERS
        engine::models::crisperwhisper::make_crisperwhisper_loader
)

audiocpp_add_model(owsm
    SOURCES
        src/models/owsm/assets.cpp
        src/models/owsm/runtime.cpp
        src/models/owsm/session.cpp
    INCLUDES
        engine/models/owsm/model.h
    LOADERS
        engine::models::owsm::make_owsm_loader
)

audiocpp_add_model(owsm_ctc
    SOURCES
        src/models/owsm_ctc/assets.cpp
        src/models/owsm_ctc/runtime.cpp
        src/models/owsm_ctc/session.cpp
    INCLUDES
        engine/models/owsm_ctc/model.h
    LOADERS
        engine::models::owsm_ctc::make_owsm_ctc_loader
)

audiocpp_add_model(cohere_asr
    SOURCES
        src/models/cohere_asr/assets.cpp
        src/models/cohere_asr/weights.cpp
        src/models/cohere_asr/runtime.cpp
        src/models/cohere_asr/session.cpp
    INCLUDES
        engine/models/cohere_asr/model.h
    LOADERS
        engine::models::cohere_asr::make_cohere_asr_loader
)

audiocpp_add_model(canary_asr
    SOURCES
        src/models/canary_asr/assets.cpp
        src/models/canary_asr/weights.cpp
        src/models/canary_asr/runtime.cpp
        src/models/canary_asr/session.cpp
    INCLUDES
        engine/models/canary_asr/model.h
    LOADERS
        engine::models::canary_asr::make_canary_asr_loader
)

audiocpp_add_model(moonshine_asr
    SOURCES
        src/models/moonshine_asr/assets.cpp
        src/models/moonshine_asr/runtime.cpp
        src/models/moonshine_asr/session.cpp
        src/models/moonshine_asr/weights.cpp
    INCLUDES
        engine/models/moonshine_asr/session.h
        engine/models/moonshine_asr/runtime.h
    LOADERS
        engine::models::moonshine_asr::make_moonshine_asr_loader
)
if (ENGINE_ENABLE_NEON)
    target_sources(engine_model_moonshine_asr PRIVATE
        src/models/moonshine_asr/neon_decoder.cpp
        src/models/moonshine_asr/neon_weights.cpp)
endif()

audiocpp_add_model(qwen3_tts
    SOURCES
        src/models/qwen3_tts/assets.cpp
        src/models/qwen3_tts/tokenizer_text.cpp
        src/models/qwen3_tts/rknn_speech_decoder.cpp
        src/models/qwen3_tts/speaker_encoder.cpp
        src/models/qwen3_tts/talker.cpp
        src/models/qwen3_tts/prompt_tts_voice_clone.cpp
        src/models/qwen3_tts/prompt_tts_voice_design.cpp
        src/models/qwen3_tts/prompt_tts_custom_voice.cpp
        src/models/qwen3_tts/session.cpp
        src/models/qwen3_tts/loader.cpp
    INCLUDES
        engine/models/qwen3_tts/loader.h
    LOADERS
        engine::models::qwen3_tts::make_qwen3_tts_loader
)

audiocpp_add_model(vieneu_v3_turbo
    SOURCES
        src/community_models/vieneu_v3_turbo/assets.cpp
        src/community_models/vieneu_v3_turbo/tokenizer_text.cpp
        src/community_models/vieneu_v3_turbo/tokenizer_speech_encoder.cpp
        src/community_models/vieneu_v3_turbo/tokenizer_speech_decoder.cpp
        src/community_models/vieneu_v3_turbo/speaker_encoder.cpp
        src/community_models/vieneu_v3_turbo/talker.cpp
        src/community_models/vieneu_v3_turbo/prompt_tts_voice_clone.cpp
        src/community_models/vieneu_v3_turbo/session.cpp
        src/community_models/vieneu_v3_turbo/loader.cpp
        src/community_models/vieneu_v3_turbo/frame_cap.cpp
        src/community_models/vieneu_v3_turbo/orchestration.cpp
        src/community_models/vieneu_v3_turbo/text_frontend.cpp
    INCLUDES
        engine/community_models/vieneu_v3_turbo/loader.h
    LOADERS
        engine::models::vieneu_v3_turbo::make_vieneu_v3_turbo_loader
)
if(MSVC)
    target_compile_options(engine_model_vieneu_v3_turbo PRIVATE /utf-8)
endif()

audiocpp_add_model(qwen3_asr
    SOURCES
        src/models/qwen3_asr/assets.cpp
        src/models/qwen3_asr/tokenizer_text.cpp
        src/models/qwen3_asr/frontend_whisper.cpp
        src/models/qwen3_asr/thinker.cpp
        src/models/qwen3_asr/prompt_asr.cpp
        src/models/qwen3_asr/postprocess.cpp
        src/models/qwen3_asr/session.cpp
        src/models/qwen3_asr/loader.cpp
    INCLUDES
        engine/models/qwen3_asr/loader.h
    LOADERS
        engine::models::qwen3_asr::make_qwen3_asr_loader
    DEPENDS
        qwen3_forced_aligner
    OBJECT_LIBRARIES
        utf8proc_vendor
)

audiocpp_add_model(confucius4_r2t2
    SOURCES
        src/community_models/confucius4_r2t2/assets.cpp
        src/community_models/confucius4_r2t2/text_postprocess.cpp
        src/community_models/confucius4_r2t2/tokenizer_text.cpp
        src/community_models/confucius4_r2t2/frontend_whisper.cpp
        src/community_models/confucius4_r2t2/audio_encoder.cpp
        src/community_models/confucius4_r2t2/thinker.cpp
        src/community_models/confucius4_r2t2/prompt_asr.cpp
        src/community_models/confucius4_r2t2/session.cpp
    INCLUDES
        engine/community_models/confucius4_r2t2/session.h
    LOADERS
        engine::community_models::confucius4_r2t2::make_confucius4_r2t2_loader
)

audiocpp_add_model(qwen3_forced_aligner
    SOURCES
        src/models/qwen3_forced_aligner/nagisa_segmenter.cpp
        src/models/qwen3_forced_aligner/processor.cpp
        src/models/qwen3_forced_aligner/session.cpp
        src/models/qwen3_forced_aligner/loader.cpp
    INCLUDES
        engine/models/qwen3_forced_aligner/loader.h
    LOADERS
        engine::models::qwen3_forced_aligner::make_qwen3_forced_aligner_loader
    DEPENDS
        qwen3_asr
    OBJECT_LIBRARIES
        utf8proc_vendor
)

audiocpp_add_model(mms_forced_aligner
    SOURCES
        src/community_models/mms_forced_aligner/assets.cpp
        src/community_models/mms_forced_aligner/ctc_alignment.cpp
        src/community_models/mms_forced_aligner/emissions.cpp
        src/community_models/mms_forced_aligner/session.cpp
        src/community_models/mms_forced_aligner/text_processor.cpp
    INCLUDES
        engine/community_models/mms_forced_aligner/session.h
    LOADERS
        engine::community_models::mms_forced_aligner::make_mms_forced_aligner_loader
)

audiocpp_add_model(sortformer_diar
    SOURCES
        src/models/sortformer_diar/assets.cpp
        src/models/sortformer_diar/frontend.cpp
        src/models/sortformer_diar/graph.cpp
        src/models/sortformer_diar/modules.cpp
        src/models/sortformer_diar/postprocess.cpp
        src/models/sortformer_diar/session.cpp
    INCLUDES
        engine/models/sortformer_diar/session.h
    LOADERS
        engine::models::sortformer_diar::make_sortformer_diar_loader
)

audiocpp_add_model(sortformer_diar_v2
    SOURCES
        src/community_models/sortformer_diar/assets.cpp
        src/community_models/sortformer_diar/frontend.cpp
        src/community_models/sortformer_diar/weights.cpp
        src/community_models/sortformer_diar/aosc_state.cpp
        src/community_models/sortformer_diar/stream_schedule.cpp
        src/community_models/sortformer_diar/encoder.cpp
        src/community_models/sortformer_diar/session.cpp
    INCLUDES
        engine/community_models/sortformer_diar/aosc_state.h
        engine/community_models/sortformer_diar/stream_schedule.h
        engine/community_models/sortformer_diar/encoder.h
        engine/community_models/sortformer_diar/session.h
        engine/community_models/sortformer_diar/frontend.h
        engine/community_models/sortformer_diar/weights.h
        engine/community_models/sortformer_diar/assets.h
    LOADERS
        engine::community_models::sortformer_diar::make_sortformer_diar_v2_loader
)

audiocpp_add_model(nemotron_3_diar
    SOURCES
        src/models/nemotron_3_diar/assets.cpp
        src/models/nemotron_3_diar/frontend.cpp
        src/models/nemotron_3_diar/streaming.cpp
        src/models/nemotron_3_diar/encoder.cpp
        src/models/nemotron_3_diar/session.cpp
    INCLUDES
        engine/models/nemotron_3_diar/assets.h
        engine/models/nemotron_3_diar/frontend.h
        engine/models/nemotron_3_diar/streaming.h
        engine/models/nemotron_3_diar/encoder.h
        engine/models/nemotron_3_diar/session.h
    LOADERS
        engine::models::nemotron_3_diar::make_nemotron_3_diar_loader
)

audiocpp_add_model(stable_audio
    SOURCES
        src/models/stable_audio/assets.cpp
        src/models/stable_audio/conditioner.cpp
        src/models/stable_audio/conditioner_runtime.cpp
        src/models/stable_audio/foundation/conditioner.cpp
        src/models/stable_audio/foundation/oobleck_autoencoder.cpp
        src/models/stable_audio/foundation/rf_dit.cpp
        src/models/stable_audio/foundation/session.cpp
        src/models/stable_audio/loader.cpp
        src/models/stable_audio/request.cpp
        src/models/stable_audio/rf_dit.cpp
        src/models/stable_audio/same_autoencoder.cpp
        src/models/stable_audio/sampler.cpp
        src/models/stable_audio/session.cpp
        src/models/stable_audio/t5gemma.cpp
    INCLUDES
        engine/models/stable_audio/loader.h
    LOADERS
        engine::models::stable_audio::make_stable_audio_loader
)

audiocpp_add_model(yue2
    SOURCES
        src/models/yue2/ar_runtime.cpp
        src/models/yue2/assets.cpp
        src/models/yue2/lora.cpp
        src/models/yue2/nar_runtime.cpp
        src/models/yue2/pipeline.cpp
        src/models/yue2/request.cpp
        src/models/yue2/session.cpp
        src/models/yue2/tokenizer_text.cpp
        src/models/yue2/types.cpp
    INCLUDES
        engine/models/yue2/session.h
    LOADERS
        engine::models::yue2::make_yue2_loader
)

audiocpp_add_model(sheetsage2
    SOURCES
        src/models/sheetsage/audio_frontend.cpp
        src/models/sheetsage/processing.cpp
        src/models/sheetsage/runtime.cpp
        src/models/sheetsage/session.cpp
    INCLUDES
        engine/models/sheetsage/audio_frontend.h
        engine/models/sheetsage/processing.h
        engine/models/sheetsage/types.h
        engine/models/sheetsage/runtime.h
        engine/models/sheetsage/session.h
    LOADERS
        engine::models::sheetsage::make_sheetsage2_loader
)

audiocpp_add_model(auk
    SOURCES
        src/community_models/auk/audio_conditioning.cpp
        src/community_models/auk/conditioning.cpp
        src/community_models/auk/flow.cpp
        src/community_models/auk/vae.cpp
        src/community_models/auk/session.cpp
    INCLUDES
        engine/community_models/auk/session.h
    LOADERS
        engine::models::auk::make_auk_loader
)
if(TARGET engine_model_auk)
    target_link_libraries(engine_model_auk PRIVATE yaml_vendor)
endif()

audiocpp_add_model(supertonic
    SOURCES
        src/models/supertonic/assets.cpp
        src/models/supertonic/loader.cpp
        src/models/supertonic/runtime.cpp
        src/models/supertonic/session.cpp
        src/models/supertonic/tokenizer_text.cpp
    INCLUDES
        engine/models/supertonic/loader.h
    LOADERS
        engine::models::supertonic::make_supertonic_loader
)

audiocpp_add_model(citrinet_asr
    SOURCES
        src/models/citrinet_asr/assets.cpp
        src/models/citrinet_asr/runtime.cpp
        src/models/citrinet_asr/session.cpp
    INCLUDES
        engine/models/citrinet_asr/session.h
    LOADERS
        engine::models::citrinet_asr::make_citrinet_asr_loader
)

# granite_speech5 must precede granite5asr: granite5asr's can_load also
# matches model_type "granite_speech", so registration order decides native
# auto-detect between the TurboCTC checkpoint and this conditional-generation
# checkpoint (this loader's triple check rejects TurboCTC snapshots).
audiocpp_add_model(granite_speech5
    SOURCES
        src/community_models/granite_speech5/assets.cpp
        src/community_models/granite_speech5/frontend.cpp
        src/community_models/granite_speech5/encoder.cpp
        src/community_models/granite_speech5/projector.cpp
        src/community_models/granite_speech5/decoder.cpp
        src/community_models/granite_speech5/prompt.cpp
        src/community_models/granite_speech5/session.cpp
    INCLUDES
        engine/community_models/granite_speech5/session.h
    LOADERS
        engine::community_models::granite_speech5::make_granite_speech5_loader
)

audiocpp_add_model(granite5asr
    SOURCES
        src/community_models/granite5asr/assets.cpp
        src/community_models/granite5asr/frontend.cpp
        src/community_models/granite5asr/encoder.cpp
        src/community_models/granite5asr/session.cpp
    INCLUDES
        engine/community_models/granite5asr/session.h
    LOADERS
        engine::community_models::granite5asr::make_granite5asr_loader
    ALIASES
        granite_speech5_asr
        granite_speech
        granite_speech5_ctc
)

audiocpp_add_model(niagara_asr
    SOURCES
        src/models/niagara_asr/assets.cpp
        src/models/niagara_asr/encoder.cpp
        src/models/niagara_asr/frontend.cpp
        src/models/niagara_asr/runtime.cpp
        src/models/niagara_asr/session.cpp
        src/models/niagara_asr/weights.cpp
    INCLUDES
        engine/models/niagara_asr/assets.h
        engine/models/niagara_asr/encoder.h
        engine/models/niagara_asr/frontend.h
        engine/models/niagara_asr/runtime.h
        engine/models/niagara_asr/session.h
        engine/models/niagara_asr/weights.h
    LOADERS
        engine::models::niagara_asr::make_niagara_asr_loader
)

audiocpp_add_model(edda_asr
    SOURCES
        src/community_models/edda_asr/assets.cpp
        src/community_models/edda_asr/weights.cpp
        src/community_models/edda_asr/runtime.cpp
        src/community_models/edda_asr/session.cpp
    INCLUDES
        engine/community_models/edda_asr/model.h
    LOADERS
        engine::community_models::edda_asr::make_edda_asr_loader
)

audiocpp_add_model(whistle_asr
    SOURCES
        src/community_models/whistle_asr/assets.cpp
        src/community_models/whistle_asr/decoder.cpp
        src/community_models/whistle_asr/encoder.cpp
        src/community_models/whistle_asr/frontend.cpp
        src/community_models/whistle_asr/graph.cpp
        src/community_models/whistle_asr/runtime.cpp
        src/community_models/whistle_asr/session.cpp
    INCLUDES
        engine/community_models/whistle_asr/session.h
    LOADERS
        engine::community_models::whistle_asr::make_whistle_asr_loader
)

audiocpp_add_model(audio8_asr
    SOURCES
        src/community_models/audio8_asr/assets.cpp
        src/community_models/audio8_asr/projector.cpp
        src/community_models/audio8_asr/thinker.cpp
        src/community_models/audio8_asr/session.cpp
    INCLUDES
        engine/community_models/audio8_asr/session.h
    LOADERS
        engine::community_models::audio8_asr::make_audio8_asr_loader
    DEPENDS
        qwen3_asr
)

audiocpp_add_model(vevo2
    SOURCES
        src/models/vevo2/ar.cpp
        src/models/vevo2/assets.cpp
        src/models/vevo2/components.cpp
        src/models/vevo2/fm.cpp
        src/models/vevo2/prompt_builder.cpp
        src/models/vevo2/session.cpp
        src/models/vevo2/tokenizer_ar.cpp
        src/models/vevo2/vocoder.cpp
        src/models/vevo2/loader.cpp
    INCLUDES
        engine/models/vevo2/loader.h
    LOADERS
        engine::models::vevo2::make_vevo2_loader
)

audiocpp_add_model(seed_vc
    SOURCES
        src/models/seed_vc/assets.cpp
        src/models/seed_vc/audio_features.cpp
        src/models/seed_vc/length_regulator.cpp
        src/models/seed_vc/astral_quantizer.cpp
        src/models/seed_vc/content_features.cpp
        src/models/seed_vc/rmvpe.cpp
        src/models/seed_vc/whisper_content.cpp
        src/models/seed_vc/v1_cfm.cpp
        src/models/seed_vc/v2_cfm.cpp
        src/models/seed_vc/session.cpp
    INCLUDES
        engine/models/seed_vc/session.h
    LOADERS
        engine::models::seed_vc::make_seed_vc_loader
)

audiocpp_add_model(meanvc2
    SOURCES
        src/models/meanvc2/assets.cpp
        src/models/meanvc2/asr_encoder.cpp
        src/models/meanvc2/audio_pipeline.cpp
        src/models/meanvc2/session.cpp
        src/models/meanvc2/speaker_encoder.cpp
        src/models/meanvc2/flow_sampler_runtime.cpp
        src/models/meanvc2/vocoder.cpp
    INCLUDES
        engine/models/meanvc2/assets.h
        engine/models/meanvc2/asr_encoder.h
        engine/models/meanvc2/audio_pipeline.h
        engine/models/meanvc2/session.h
        engine/models/meanvc2/speaker_encoder.h
        engine/models/meanvc2/flow_sampler_runtime.h
        engine/models/meanvc2/vocoder.h
    LOADERS
        engine::models::meanvc2::make_meanvc2_loader
)

audiocpp_add_model(rvc
    SOURCES
        src/models/rvc/assets.cpp
        src/models/rvc/hubert.cpp
        src/models/rvc/native_pipeline.cpp
        src/models/rvc/rmvpe.cpp
        src/models/rvc/retrieval_index.cpp
        src/models/rvc/session.cpp
        src/models/rvc/synthesizer.cpp
        src/models/rvc/voice_model.cpp
    INCLUDES
        engine/models/rvc/session.h
    LOADERS
        engine::models::rvc::make_rvc_loader
)

audiocpp_add_model(chatterbox
    SOURCES
        src/models/chatterbox/assets.cpp
        src/models/chatterbox/campplus_encoder_impl.cpp
        src/models/chatterbox/audio_features.cpp
        src/models/chatterbox/audio_processing.cpp
        src/models/chatterbox/component_weights.cpp
        src/models/chatterbox/conditionals.cpp
        src/models/chatterbox/loader.cpp
        src/models/chatterbox/s3_tokenizer_impl.cpp
        src/models/chatterbox/session.cpp
        src/models/chatterbox/t3_component.cpp
        src/models/chatterbox/t3_weights.cpp
        src/models/chatterbox/text_tokenizer.cpp
        src/models/chatterbox/tts.cpp
        src/models/chatterbox/vc.cpp
        src/models/chatterbox/voice_encoder_impl.cpp
    INCLUDES
        engine/models/chatterbox/loader.h
    LOADERS
        engine::models::chatterbox::make_chatterbox_loader
)

audiocpp_add_model(chatterbox_turbo
    SOURCES
        src/community_models/chatterbox_turbo/t3_turbo_weights.cpp
        src/community_models/chatterbox_turbo/t3_turbo_component.cpp
        src/community_models/chatterbox_turbo/s3gen_turbo.cpp
        src/community_models/chatterbox_turbo/text_tokenizer_turbo.cpp
        src/community_models/chatterbox_turbo/assets.cpp
        src/community_models/chatterbox_turbo/loader.cpp
        src/community_models/chatterbox_turbo/session.cpp
        src/community_models/chatterbox_turbo/tts.cpp
    INCLUDES
        engine/community_models/chatterbox_turbo/loader.h
    LOADERS
        engine::community_models::chatterbox_turbo::make_chatterbox_turbo_loader
)

audiocpp_add_model(ace_step
    SOURCES
        src/models/ace_step/assets.cpp
        src/models/ace_step/condition_encoder.cpp
        src/models/ace_step/cover_tokenizer.cpp
        src/models/ace_step/detokenizer.cpp
        src/models/ace_step/diffusion.cpp
        src/models/ace_step/dit_weights_runtime.cpp
        src/models/ace_step/loader.cpp
        src/models/ace_step/planner.cpp
        src/models/ace_step/pre_dit.cpp
        src/models/ace_step/prompt_builder.cpp
        src/models/ace_step/repaint.cpp
        src/models/ace_step/request_parser.cpp
        src/models/ace_step/session.cpp
        src/models/ace_step/task_route.cpp
        src/models/ace_step/text_encoder.cpp
        src/models/ace_step/tokenizer_text.cpp
        src/models/ace_step/vae_decoder.cpp
        src/models/ace_step/vae_encoder.cpp
    INCLUDES
        engine/models/ace_step/loader.h
    LOADERS
        engine::models::ace_step::make_ace_step_loader
)
audiocpp_add_model(sopro_tts
    SOURCES
        src/community_models/sopro_tts/acoustic.cpp
        src/community_models/sopro_tts/assets.cpp
        src/community_models/sopro_tts/reference.cpp
        src/community_models/sopro_tts/semantic_encoder.cpp
        src/community_models/sopro_tts/semantic_lm.cpp
        src/community_models/sopro_tts/session.cpp
        src/community_models/sopro_tts/speaker_encoder.cpp
        src/community_models/sopro_tts/streaming.cpp
        src/community_models/sopro_tts/text_tokenizer.cpp
        src/community_models/sopro_tts/vocoder.cpp
    INCLUDES
        engine/community_models/sopro_tts/session.h
    LOADERS
        engine::community_models::sopro_tts::make_sopro_tts_loader
)

audiocpp_add_model(soulx_singer
    SOURCES
        src/models/soulx_singer/runtime.cpp
        src/models/soulx_singer/score.cpp
        src/models/soulx_singer/session.cpp
        src/models/soulx_singer/routing.cpp
    INCLUDES engine/models/soulx_singer/session.h
    LOADERS engine::models::soulx_singer::make_soulx_singer_loader
)

audiocpp_add_model(soprano_tts
    SOURCES
        src/community_models/soprano_tts/assets.cpp
        src/community_models/soprano_tts/generator.cpp
        src/community_models/soprano_tts/session.cpp
        src/community_models/soprano_tts/text_normalizer.cpp
        src/community_models/soprano_tts/tokenizer_text.cpp
        src/community_models/soprano_tts/vocoder.cpp
    INCLUDES
        engine/community_models/soprano_tts/session.h
    LOADERS
        engine::community_models::soprano_tts::make_soprano_tts_loader
)

audiocpp_add_model(voxmel_tts
    SOURCES
        src/models/voxmel_tts/assets.cpp
        src/models/voxmel_tts/ar.cpp
        src/models/voxmel_tts/runtime.cpp
        src/models/voxmel_tts/session.cpp
    INCLUDES
        engine/models/voxmel_tts/session.h
    LOADERS
        engine::models::voxmel_tts::make_voxmel_tts_loader
)

audiocpp_add_model(m5_tts
    SOURCES
        src/models/m5_tts/ar.cpp
        src/models/m5_tts/assets.cpp
        src/models/m5_tts/session.cpp
    INCLUDES
        engine/models/m5_tts/session.h
    LOADERS
        engine::models::m5_tts::make_m5_tts_loader
)

audiocpp_add_model(mira_tts
    SOURCES
        src/community_models/mira_tts/assets.cpp
        src/community_models/mira_tts/decoder.cpp
        src/community_models/mira_tts/generator.cpp
        src/community_models/mira_tts/processor.cpp
        src/community_models/mira_tts/prompt.cpp
        src/community_models/mira_tts/session.cpp
        src/community_models/mira_tts/speaker_encoder.cpp
    INCLUDES
        engine/community_models/mira_tts/session.h
    LOADERS
        engine::community_models::mira_tts::make_mira_tts_loader
)


audiocpp_add_model(magenta_realtime_2
    SOURCES
        src/models/magenta_realtime_2/assets.cpp
        src/models/magenta_realtime_2/depthformer.cpp
        src/models/magenta_realtime_2/session.cpp
        src/models/magenta_realtime_2/spectrostream.cpp
    INCLUDES
        engine/models/magenta_realtime_2/session.h
    LOADERS
        engine::models::magenta_realtime_2::make_magenta_realtime_2_loader
)

audiocpp_add_model(midashenglm_gen
    SOURCES
        src/models/midashenglm_gen/assets.cpp
        src/models/midashenglm_gen/ar_runtime.cpp
        src/models/midashenglm_gen/audio_tokenizer.cpp
        src/models/midashenglm_gen/flow_runtime.cpp
        src/models/midashenglm_gen/prompt_encoder.cpp
        src/models/midashenglm_gen/session.cpp
        src/models/midashenglm_gen/tokenizer_text.cpp
    INCLUDES
        engine/models/midashenglm_gen/assets.h
        engine/models/midashenglm_gen/ar_runtime.h
        engine/models/midashenglm_gen/audio_tokenizer.h
        engine/models/midashenglm_gen/flow_runtime.h
        engine/models/midashenglm_gen/prompt_encoder.h
        engine/models/midashenglm_gen/session.h
        engine/models/midashenglm_gen/tokenizer_text.h
    LOADERS
        engine::models::midashenglm_gen::make_midashenglm_gen_loader
)

audiocpp_add_model(audiosr
    SOURCES
        src/models/audiosr/assets.cpp
        src/models/audiosr/autoencoder.cpp
        src/models/audiosr/ddim.cpp
        src/models/audiosr/frontend.cpp
        src/models/audiosr/hifigan.cpp
        src/models/audiosr/pipeline.cpp
        src/models/audiosr/session.cpp
        src/models/audiosr/unet.cpp
    INCLUDES
        engine/models/audiosr/assets.h
        engine/models/audiosr/autoencoder.h
        engine/models/audiosr/ddim.h
        engine/models/audiosr/frontend.h
        engine/models/audiosr/hifigan.h
        engine/models/audiosr/pipeline.h
        engine/models/audiosr/session.h
        engine/models/audiosr/unet.h
    LOADERS
        engine::models::audiosr::make_audiosr_loader
)

audiocpp_add_model(builtin_audio_utils
    SOURCES
        src/models/builtin_audio_utils/session.cpp
    INCLUDES
        engine/models/builtin_audio_utils/session.h
    LOADERS
        engine::models::builtin_audio_utils::make_builtin_audio_utils_loader
)

audiocpp_add_model(controlfoley
    SOURCES
        src/models/controlfoley/assets.cpp
        src/models/controlfoley/conditioning.cpp
        src/models/controlfoley/pipeline.cpp
        src/models/controlfoley/session.cpp
        src/models/controlfoley/video_processing.cpp
    INCLUDES
        engine/models/controlfoley/assets.h
        engine/models/controlfoley/conditioning.h
        engine/models/controlfoley/pipeline.h
        engine/models/controlfoley/session.h
        engine/models/controlfoley/video_processing.h
    LOADERS
        engine::models::controlfoley::make_controlfoley_loader
)

audiocpp_add_model(fireredtts3
    SOURCES
        src/models/fireredtts3/assets.cpp
        src/models/fireredtts3/ar.cpp
        src/models/fireredtts3/flow.cpp
        src/models/fireredtts3/pipeline.cpp
        src/models/fireredtts3/redae.cpp
        src/models/fireredtts3/session.cpp
        src/models/fireredtts3/tokenizer_text.cpp
    INCLUDES
        engine/models/fireredtts3/assets.h
        engine/models/fireredtts3/ar.h
        engine/models/fireredtts3/flow.h
        engine/models/fireredtts3/pipeline.h
        engine/models/fireredtts3/redae.h
        engine/models/fireredtts3/session.h
        engine/models/fireredtts3/tokenizer_text.h
    LOADERS
        engine::models::fireredtts3::make_fireredtts3_loader
)

audiocpp_add_model(firered_audio
    SOURCES
        src/models/firered_audio/assets.cpp
        src/models/firered_audio/audio_encoder.cpp
        src/models/firered_audio/flow.cpp
        src/models/firered_audio/patch_encoder.cpp
        src/models/firered_audio/pipeline.cpp
        src/models/firered_audio/redae.cpp
        src/models/firered_audio/session.cpp
        src/models/firered_audio/tokenizer.cpp
        src/models/firered_audio/understanding.cpp
    INCLUDES
        engine/models/firered_audio/assets.h
        engine/models/firered_audio/audio_encoder.h
        engine/models/firered_audio/flow.h
        engine/models/firered_audio/patch_encoder.h
        engine/models/firered_audio/pipeline.h
        engine/models/firered_audio/redae.h
        engine/models/firered_audio/session.h
        engine/models/firered_audio/tokenizer.h
        engine/models/firered_audio/understanding.h
    LOADERS
        engine::models::firered_audio::make_firered_audio_loader
)

audiocpp_add_model(index_echo
    SOURCES
        src/models/index_echo/assets.cpp
        src/models/index_echo/connector.cpp
        src/models/index_echo/mapper.cpp
        src/models/index_echo/session.cpp
        src/models/index_echo/text_translation.cpp
    INCLUDES
        engine/models/index_echo/assets.h
        engine/models/index_echo/connector.h
        engine/models/index_echo/mapper.h
        engine/models/index_echo/session.h
        engine/models/index_echo/text_translation.h
    LOADERS
        engine::models::index_echo::make_index_echo_loader
    DEPENDS
        qwen3_asr
)

audiocpp_add_model(cosyvoice3
    SOURCES
        src/models/cosyvoice3/ar.cpp
        src/models/cosyvoice3/assets.cpp
        src/models/cosyvoice3/flow.cpp
        src/models/cosyvoice3/frontend.cpp
        src/models/cosyvoice3/hift.cpp
        src/models/cosyvoice3/rknn_flow.cpp
        src/models/cosyvoice3/session.cpp
        src/models/cosyvoice3/tokenizer_text.cpp
    INCLUDES
        engine/models/cosyvoice3/ar.h
        engine/models/cosyvoice3/assets.h
        engine/models/cosyvoice3/flow.h
        engine/models/cosyvoice3/frontend.h
        engine/models/cosyvoice3/hift.h
        engine/models/cosyvoice3/session.h
        engine/models/cosyvoice3/tokenizer_text.h
    LOADERS
        engine::models::cosyvoice3::make_cosyvoice3_loader
)

audiocpp_add_model(breeze_tts
    SOURCES
        src/models/breeze_tts/assets.cpp
        src/models/breeze_tts/generator.cpp
        src/models/breeze_tts/session.cpp
        src/models/breeze_tts/speech_decoder.cpp
        src/models/breeze_tts/speech_encoder.cpp
        src/models/breeze_tts/text_encoder.cpp
        src/models/breeze_tts/tokenizer_text.cpp
    INCLUDES
        engine/models/breeze_tts/assets.h
        engine/models/breeze_tts/generator.h
        engine/models/breeze_tts/session.h
        engine/models/breeze_tts/speech_decoder.h
        engine/models/breeze_tts/speech_encoder.h
        engine/models/breeze_tts/text_encoder.h
        engine/models/breeze_tts/tokenizer_text.h
    LOADERS
        engine::models::breeze_tts::make_breeze_tts_loader
)

audiocpp_add_model(vibeasr
    SOURCES
        src/community_models/vibeasr/assets.cpp
        src/community_models/vibeasr/vae_encoder.cpp
        src/community_models/vibeasr/lm_decoder.cpp
        src/community_models/vibeasr/session.cpp
    INCLUDES
        engine/community_models/vibeasr/session.h
    LOADERS
        engine::community_models::vibeasr::make_vibeasr_loader
)

audiocpp_add_model(lfm2_audio
    SOURCES
        src/community_models/lfm2_audio/asr_inputs.cpp
        src/community_models/lfm2_audio/assets.cpp
        src/community_models/lfm2_audio/chat.cpp
        src/community_models/lfm2_audio/tokenizer.cpp
        src/community_models/lfm2_audio/audio_encoder.cpp
        src/community_models/lfm2_audio/backbone.cpp
        src/community_models/lfm2_audio/lfm2_blocks.cpp
        src/community_models/lfm2_audio/depthformer.cpp
        src/community_models/lfm2_audio/detokenizer.cpp
        src/community_models/lfm2_audio/tts.cpp
        src/community_models/lfm2_audio/interleaved.cpp
        src/community_models/lfm2_audio/session.cpp
        src/community_models/lfm2_audio/weight_stores.cpp
    INCLUDES
        engine/community_models/lfm2_audio/session.h
    LOADERS
        engine::community_models::lfm2_audio::make_lfm2_audio_loader
)

audiocpp_add_model(lfm3_audio
    SOURCES
        src/community_models/lfm3_audio/assets.cpp
        src/community_models/lfm3_audio/weights.cpp
        src/community_models/lfm3_audio/prompt.cpp
        src/community_models/lfm3_audio/depth.cpp
        src/community_models/lfm3_audio/codec.cpp
        src/community_models/lfm3_audio/generator.cpp
        src/community_models/lfm3_audio/frontend.cpp
        src/community_models/lfm3_audio/session.cpp
    INCLUDES
        engine/community_models/lfm3_audio/session.h
    LOADERS
        engine::community_models::lfm3_audio::make_lfm3_audio_loader
    DEPENDS
        lfm2_audio
)

audiocpp_add_model(xtts_v2
    SOURCES
        src/community_models/xtts_v2/assets.cpp
        src/community_models/xtts_v2/frontend.cpp
        src/community_models/xtts_v2/gpt.cpp
        src/community_models/xtts_v2/loader.cpp
        src/community_models/xtts_v2/session.cpp
        src/community_models/xtts_v2/tokenizer_text.cpp
        src/community_models/xtts_v2/vocoder.cpp
    INCLUDES
        engine/community_models/xtts_v2/loader.h
    LOADERS
        engine::models::xtts_v2::make_xtts_v2_loader
)

set(AUDIOCPP_ENABLED_MODELS "")
if (AUDIOCPP_MODEL_SET STREQUAL "full")
    set(AUDIOCPP_ENABLED_MODELS ${AUDIOCPP_MODEL_TARGETS})
elseif (AUDIOCPP_MODEL_SET STREQUAL "core")
    set(AUDIOCPP_ENABLED_MODELS "")
elseif (AUDIOCPP_MODEL_SET STREQUAL "custom")
    string(REPLACE "," ";" AUDIOCPP_REQUESTED_MODELS "${AUDIOCPP_MODELS}")
    foreach(requested_model IN LISTS AUDIOCPP_REQUESTED_MODELS)
        string(STRIP "${requested_model}" requested_model)
        if (requested_model STREQUAL "")
            continue()
        endif()
        if (DEFINED AUDIOCPP_MODEL_ALIAS_${requested_model})
            list(APPEND AUDIOCPP_ENABLED_MODELS ${AUDIOCPP_MODEL_ALIAS_${requested_model}})
        else()
            message(FATAL_ERROR "Unknown AUDIOCPP_MODELS entry: ${requested_model}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES AUDIOCPP_ENABLED_MODELS)
else()
    message(FATAL_ERROR "AUDIOCPP_MODEL_SET must be full, core, or custom")
endif()

set(AUDIOCPP_REGISTRY_INCLUDE_CONTENT "")
set(AUDIOCPP_REGISTRY_LOADER_CONTENT "")
foreach(model_name IN LISTS AUDIOCPP_ENABLED_MODELS)
    foreach(include_name IN LISTS AUDIOCPP_MODEL_${model_name}_INCLUDES)
        string(APPEND AUDIOCPP_REGISTRY_INCLUDE_CONTENT "#include \"${include_name}\"\n")
    endforeach()
    foreach(loader_name IN LISTS AUDIOCPP_MODEL_${model_name}_LOADERS)
        string(APPEND AUDIOCPP_REGISTRY_LOADER_CONTENT "        ${loader_name}(),\n")
    endforeach()
endforeach()

set(AUDIOCPP_LINKED_MODELS ${AUDIOCPP_ENABLED_MODELS})
set(AUDIOCPP_DEPENDENCY_SCAN_INDEX 0)
list(LENGTH AUDIOCPP_LINKED_MODELS AUDIOCPP_LINKED_MODEL_COUNT)
while(AUDIOCPP_DEPENDENCY_SCAN_INDEX LESS AUDIOCPP_LINKED_MODEL_COUNT)
    list(GET AUDIOCPP_LINKED_MODELS ${AUDIOCPP_DEPENDENCY_SCAN_INDEX} model_name)
    foreach(dependency IN LISTS AUDIOCPP_MODEL_${model_name}_DEPENDS)
        if (NOT dependency IN_LIST AUDIOCPP_MODEL_TARGETS)
            message(FATAL_ERROR "Unknown composite dependency for ${model_name}: ${dependency}")
        endif()
        if (NOT dependency IN_LIST AUDIOCPP_LINKED_MODELS)
            list(APPEND AUDIOCPP_LINKED_MODELS ${dependency})
        endif()
    endforeach()
    math(EXPR AUDIOCPP_DEPENDENCY_SCAN_INDEX "${AUDIOCPP_DEPENDENCY_SCAN_INDEX} + 1")
    list(LENGTH AUDIOCPP_LINKED_MODELS AUDIOCPP_LINKED_MODEL_COUNT)
endwhile()
file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/model_registry_includes.inc"
    CONTENT "${AUDIOCPP_REGISTRY_INCLUDE_CONTENT}")
file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/model_registry_loaders.inc"
    CONTENT "${AUDIOCPP_REGISTRY_LOADER_CONTENT}")

set(AUDIOCPP_RUNTIME_OBJECTS
    $<TARGET_OBJECTS:engine_core>
    $<TARGET_OBJECTS:engine_model_silero_vad>
    $<TARGET_OBJECTS:engine_model_marblenet_vad>
)
if (ENGINE_ENABLE_NEON)
    list(APPEND AUDIOCPP_RUNTIME_OBJECTS $<TARGET_OBJECTS:engine_neon>)
endif()
set(AUDIOCPP_MODEL_OBJECT_LIBRARIES "")
foreach(model_name IN LISTS AUDIOCPP_LINKED_MODELS)
    list(APPEND AUDIOCPP_RUNTIME_OBJECTS $<TARGET_OBJECTS:engine_model_${model_name}>)
    list(APPEND AUDIOCPP_MODEL_OBJECT_LIBRARIES ${AUDIOCPP_MODEL_${model_name}_OBJECT_LIBRARIES})
endforeach()
list(REMOVE_DUPLICATES AUDIOCPP_MODEL_OBJECT_LIBRARIES)
foreach(object_library IN LISTS AUDIOCPP_MODEL_OBJECT_LIBRARIES)
    list(APPEND AUDIOCPP_RUNTIME_OBJECTS $<TARGET_OBJECTS:${object_library}>)
endforeach()

message(STATUS "audio.cpp model composite: ${AUDIOCPP_MODEL_SET} selected [${AUDIOCPP_ENABLED_MODELS}] linked [${AUDIOCPP_LINKED_MODELS}]")
add_library(engine_runtime STATIC
    src/framework/runtime/registry.cpp
    ${AUDIOCPP_RUNTIME_OBJECTS}
)
if (ENGINE_ENABLE_NEON)
    target_compile_definitions(engine_runtime PUBLIC ENGINE_HAVE_NEON=1)
endif()

add_library(cjson_vendor STATIC
    external/cJSON/cJSON.c
)

# cJSON annotates every CJSON_PUBLIC function with __declspec(dllexport) on
# Windows unless told otherwise (external/cJSON/cJSON.h:59-66 defaults
# CJSON_EXPORT_SYMBOLS on). Those annotations live in the object files, so any
# DLL that absorbs this static library re-exports all 78 of them -- dllexport is
# additive, it says "also export this", never "export only this". Measured: a
# shared build exported 147 symbols where 69 were intended.
#
# No DLL is built from this tree by default, so this is latent rather than
# broken today; AUDIOCPP_BUILD_C_API is the first target to expose it.
# CJSON_HIDE_SYMBOLS is cJSON's own mechanism for exactly this case and selects
# a plain declaration with no storage class. PUBLIC rather than PRIVATE so every
# translation unit that includes cJSON.h agrees on the linkage.
target_compile_definitions(cjson_vendor PUBLIC CJSON_HIDE_SYMBOLS)

target_include_directories(cjson_vendor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/external/cJSON
)

# Native package management is deliberately opt-in. Normal CLI and server
# builds do not configure an HTTP/TLS dependency. When enabled, package specs
# remain embedded so the standalone manager and managed WebUI do not require a
# source checkout or an external model_specs directory.
if (AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER)
    add_subdirectory(external/cpp-httplib)

    # Hugging Face Xet xorb chunks are LZ4-frame compressed.
    add_library(lz4_vendor STATIC
        external/lz4/lz4.c
        external/lz4/lz4frame.c
        external/lz4/lz4hc.c
        external/lz4/xxhash.c
    )
    target_include_directories(lz4_vendor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/external/lz4)
    set_property(TARGET lz4_vendor PROPERTY COMPILE_WARNING_AS_ERROR OFF)
    if (MSVC)
        target_compile_options(lz4_vendor PRIVATE /w)
    else()
        target_compile_options(lz4_vendor PRIVATE -w)
    endif()

    add_library(audiocpp_package_manager STATIC
        src/framework/package_manager/manager.cpp
        src/framework/package_manager/http_util.cpp
        src/framework/package_manager/xet.cpp
        src/framework/io/json.cpp
        src/framework/io/filesystem.cpp
    )
    target_include_directories(audiocpp_package_manager PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_include_directories(audiocpp_package_manager PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/generated"
    )
    target_link_libraries(audiocpp_package_manager PRIVATE
        cjson_vendor
        audiocpp_cpp_httplib
        lz4_vendor
    )
endif()

add_library(yaml_vendor STATIC
    external/libyaml/src/api.c
    external/libyaml/src/reader.c
    external/libyaml/src/scanner.c
    external/libyaml/src/parser.c
    external/libyaml/src/loader.c
)

target_include_directories(yaml_vendor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/external/libyaml/include
)

target_compile_definitions(yaml_vendor PUBLIC ${AUDIOCPP_LIBYAML_COMPILE_DEFINITIONS})

target_include_directories(engine_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(engine_runtime PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")

target_include_directories(engine_runtime PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/external/sentencepiece/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external/llama_tokenizer
)

target_link_libraries(engine_runtime PUBLIC ggml)
find_package(Threads REQUIRED)
target_link_libraries(engine_runtime PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
target_link_libraries(engine_runtime PRIVATE sentencepiece cjson_vendor yaml_vendor)

option(AUDIOCPP_ZONOS3_TEXTNORM "Link Zyphra's zyphra-textnorm (Rust/cargo) for ZONOS3 text normalization" OFF)
set(AUDIOCPP_ZONOS3_TEXTNORM_DIR "" CACHE PATH "zyphra-textnorm checkout (Cargo.toml root), e.g. ../ZONOS3/third_party/zyphra-textnorm")
if (AUDIOCPP_ZONOS3_TEXTNORM AND zonos3 IN_LIST AUDIOCPP_LINKED_MODELS)
    if (NOT EXISTS "${AUDIOCPP_ZONOS3_TEXTNORM_DIR}/Cargo.toml")
        message(FATAL_ERROR "AUDIOCPP_ZONOS3_TEXTNORM needs AUDIOCPP_ZONOS3_TEXTNORM_DIR pointing at a zyphra-textnorm checkout (https://github.com/Zyphra/zyphra-textnorm, tag v0.1.0)")
    endif()
    find_program(AUDIOCPP_CARGO cargo HINTS $ENV{CARGO_HOME}/bin $ENV{HOME}/.cargo/bin /opt/homebrew/opt/rustup/bin REQUIRED)
    # The hinted directory may not be on PATH, where cargo looks for rustc.
    find_program(AUDIOCPP_RUSTC rustc HINTS $ENV{CARGO_HOME}/bin $ENV{HOME}/.cargo/bin /opt/homebrew/opt/rustup/bin REQUIRED)
    # One cargo profile for every CMake config: the grammars are slow unoptimized and debug info is not needed.
    set(AUDIOCPP_ZONOS3_TEXTNORM_TARGET_DIR ${CMAKE_BINARY_DIR}/zyphra_textnorm)
    if (WIN32)
        set(AUDIOCPP_ZONOS3_TEXTNORM_LIB ${AUDIOCPP_ZONOS3_TEXTNORM_TARGET_DIR}/release/zyphra_textnorm.lib)
    else()
        set(AUDIOCPP_ZONOS3_TEXTNORM_LIB ${AUDIOCPP_ZONOS3_TEXTNORM_TARGET_DIR}/release/libzyphra_textnorm.a)
    endif()
    file(GLOB_RECURSE AUDIOCPP_ZONOS3_TEXTNORM_SOURCES CONFIGURE_DEPENDS
        ${AUDIOCPP_ZONOS3_TEXTNORM_DIR}/src/*.rs ${AUDIOCPP_ZONOS3_TEXTNORM_DIR}/data/*)
    add_custom_command(
        OUTPUT ${AUDIOCPP_ZONOS3_TEXTNORM_LIB}
        COMMAND ${CMAKE_COMMAND} -E env RUSTC=${AUDIOCPP_RUSTC}
            ${AUDIOCPP_CARGO} rustc --release --locked --lib -p zyphra-textnorm --features capi --crate-type staticlib
            --manifest-path ${AUDIOCPP_ZONOS3_TEXTNORM_DIR}/Cargo.toml --target-dir ${AUDIOCPP_ZONOS3_TEXTNORM_TARGET_DIR}
        DEPENDS ${AUDIOCPP_ZONOS3_TEXTNORM_SOURCES} ${AUDIOCPP_ZONOS3_TEXTNORM_DIR}/Cargo.toml
            ${AUDIOCPP_ZONOS3_TEXTNORM_DIR}/Cargo.lock
        COMMENT "cargo build zyphra-textnorm"
        USES_TERMINAL
        VERBATIM)
    add_custom_target(audiocpp_zyphra_textnorm_build DEPENDS ${AUDIOCPP_ZONOS3_TEXTNORM_LIB})
    add_library(audiocpp_zyphra_textnorm STATIC IMPORTED GLOBAL)
    set_target_properties(audiocpp_zyphra_textnorm PROPERTIES
        IMPORTED_LOCATION ${AUDIOCPP_ZONOS3_TEXTNORM_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${AUDIOCPP_ZONOS3_TEXTNORM_DIR}/include)
    add_dependencies(audiocpp_zyphra_textnorm audiocpp_zyphra_textnorm_build)
    if (WIN32)
        # Rust std's system libraries on MSVC.
        set_property(TARGET audiocpp_zyphra_textnorm APPEND PROPERTY INTERFACE_LINK_LIBRARIES ws2_32 userenv ntdll bcrypt)
    else()
        set_property(TARGET audiocpp_zyphra_textnorm APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${CMAKE_DL_LIBS} Threads::Threads)
    endif()
    target_compile_definitions(engine_model_zonos3 PRIVATE AUDIOCPP_ZONOS3_HAVE_TEXTNORM=1)
    target_include_directories(engine_model_zonos3 PRIVATE "${AUDIOCPP_ZONOS3_TEXTNORM_DIR}/include")
    add_dependencies(engine_model_zonos3 audiocpp_zyphra_textnorm_build)
    target_link_libraries(engine_runtime PRIVATE audiocpp_zyphra_textnorm)
endif()
if(AUDIOCPP_STATIC_ESPEAK)
    include(cmake/espeak_static.cmake)
endif()
if (AUDIOCPP_HIP_STRIX_HALO_OPTIMIZATIONS_ACTIVE)
    target_compile_definitions(engine_runtime PRIVATE ENGINE_HIP_STRIX_HALO_OPTIMIZATIONS=1)
endif()
if (ENGINE_ENABLE_RKNN)
    target_compile_definitions(engine_runtime PRIVATE ENGINE_HAVE_RKNN=1)
    if (ENGINE_RKNN_INCLUDE_DIR)
        target_include_directories(engine_runtime PUBLIC ${ENGINE_RKNN_INCLUDE_DIR})
    endif()
    target_link_libraries(engine_runtime PUBLIC ${ENGINE_RKNN_LIBRARY})
endif()
if (ENGINE_ENABLE_HIP)
    # ggml-hip publicly defines GGML_USE_CUDA for consumers, so engine code
    # needs this marker to tell a real CUDA build apart from a HIP build.
    target_compile_definitions(engine_core PRIVATE ENGINE_GGML_HIP_BACKEND=1)
endif()

# Fish Fast-AR emits a 4096-entry logits tensor for each acoustic codebook.
# On Linux HIP, a small device-side exact top-k helper keeps only the compact
# sampling candidates crossing back to the host. Built as a proper HIP target
# so CMake tracks its HIP runtime dependency; other backends stay unchanged.
if (ENGINE_ENABLE_HIP AND UNIX AND NOT APPLE)
    add_subdirectory(src/models/fish_audio)
    target_link_libraries(engine_runtime PRIVATE engine_fish_audio_hip_sampler)
    target_compile_definitions(engine_model_fish_audio PRIVATE ENGINE_HAS_HIP_FISH_FAST_SAMPLER=1)
endif()

if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(engine_runtime PRIVATE OpenMP::OpenMP_CXX)
    if (MSVC)
        # MSVC's default /openmp implements only OpenMP 2.0 and rejects the
        # '#pragma omp simd' directives in longformer_attention.cpp (error C7660).
        # /openmp:experimental enables the OpenMP 4.0 SIMD support; it overrides the
        # /openmp added by OpenMP::OpenMP_CXX above (harmless D9025 override notice).
        # Guard on CXX: nvcc does not forward a bare /openmp:experimental to the host
        # compiler and instead treats it as a second input file, which is fatal.
        target_compile_options(engine_runtime PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/openmp:experimental>)
    endif()
endif()

if (MSVC)
    target_compile_options(engine_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    target_compile_options(engine_runtime PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()

if (ENGINE_ENABLE_CUDA AND NOT ENGINE_ENABLE_HIP)
    target_sources(engine_runtime PRIVATE
        src/framework/modules/pitch_extractors/rmvpe_cuda_runtime.cu
        src/framework/audio/istft_cuda_runtime.cu
        src/framework/sampling/torch_random_cuda_runtime.cu
    )
    set_source_files_properties(
        src/framework/modules/pitch_extractors/rmvpe_cuda_runtime.cu
        src/framework/audio/istft_cuda_runtime.cu
        src/framework/sampling/torch_random_cuda_runtime.cu
        PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    set_source_files_properties(src/framework/modules/pitch_extractors/rmvpe_cuda_runtime.cu
        PROPERTIES COMPILE_OPTIONS "--use_fast_math")
    target_compile_definitions(engine_core PRIVATE ENGINE_HAS_CUDA_RMVPE=1 ENGINE_HAS_CUDA_ISTFT=1 ENGINE_HAS_CUDA_TORCH_RANDOM=1)
    target_compile_definitions(engine_runtime PUBLIC ENGINE_HAS_CUDA_RMVPE=1 ENGINE_HAS_CUDA_ISTFT=1 ENGINE_HAS_CUDA_TORCH_RANDOM=1)
    target_link_libraries(engine_runtime PRIVATE CUDA::cudart CUDA::cufft)
endif()

# Windows decodes narrow strings - argv, file paths, console output - with the
# process code page, and the default is a legacy one: a path holding "Thiền"
# then names a file that is not there, and converting such a path back to a
# narrow string throws. This manifest asks the process for UTF-8 instead
# (Windows 10 1903 and later), which is what every string in this codebase
# already is.
function(audiocpp_use_utf8_codepage target)
    if (MSVC)
        target_sources(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/audiocpp_utf8.manifest")
    endif()
endfunction()

file(GLOB AUDIOCPP_RECIPE_PLUGIN_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/app/recipe/plugins/*.cpp")
set(AUDIOCPP_RECIPE_SOURCES
    app/recipe/streaming_tempo.cpp
    app/recipe/recipe.cpp
    ${AUDIOCPP_RECIPE_PLUGIN_SOURCES})

add_executable(audiocpp_cli
    app/common/base64.cpp
    app/common/build_info.cpp
    app/cli/main.cpp
    app/cli/args.cpp
    app/cli/batch.cpp
    app/cli/request.cpp
    app/streaming/pcm_source.cpp
    app/streaming/streaming.cpp
    ${AUDIOCPP_RECIPE_SOURCES}
    app/recipe/cli_recipe.cpp
    app/workflow/execution.cpp
    app/workflow/file_sink.cpp
    app/workflow/pipeline.cpp
    app/workflow/workflow.cpp
)

target_link_libraries(audiocpp_cli PRIVATE engine_runtime ggml)
if (EMSCRIPTEN AND ENGINE_ENABLE_WEBGPU)
    target_link_options(audiocpp_cli PRIVATE
        "-sMODULARIZE=1" "-sEXPORT_NAME=createAudioCpp"
        "-sINVOKE_RUN=0" "-sEXIT_RUNTIME=0"
        "-sALLOW_MEMORY_GROWTH=1"
        "-sSTACK_SIZE=8388608" "-sFORCE_FILESYSTEM=1"
        "-sINCOMING_MODULE_JS_API=preRun,print,printErr,locateFile,mainScriptUrlOrBlob,instantiateWasm,wasmMemory"
        "-sEXPORTED_RUNTIME_METHODS=callMain,FS,ENV" "-sENVIRONMENT=web,worker")
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        target_link_options(audiocpp_cli PRIVATE "-sMAXIMUM_MEMORY=17179869184"
            "-sEXPORTED_RUNTIME_METHODS=callMain,FS,ENV,WORKERFS" "-lworkerfs.js")
    else()
        target_link_options(audiocpp_cli PRIVATE "-sMAXIMUM_MEMORY=4294967296")
    endif()
    configure_file(examples/webgpu/index.html "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/index.html" COPYONLY)
    configure_file(examples/webgpu/worker.js "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/worker.js" COPYONLY)
endif()
audiocpp_use_utf8_codepage(audiocpp_cli)
if(AUDIOCPP_STATIC_ESPEAK)
    audiocpp_stage_espeak(audiocpp_cli)
endif()
target_include_directories(audiocpp_cli PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
# MinGW selects the wmain() entry point only when linked with -municode;
# without it the CLI fails to link (undefined reference / ld error 5).
# MSVC detects wmain natively, so this is MinGW-only. Link option only:
# -municode also defines UNICODE at compile time, so passing it link-only
# keeps the source compiled with no macro side effects.
if (MINGW)
    target_link_options(audiocpp_cli PRIVATE -municode)
endif()
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(audiocpp_cli PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------
# C ABI (opt-in)
#
# A thin facade over engine::runtime for embedding audio.cpp in-process from
# languages that can call C. Purely additive: nothing above this point changes
# when AUDIOCPP_BUILD_C_API is OFF, which is the default.
# ---------------------------------------------------------------------------
if (AUDIOCPP_BUILD_C_API)
    add_library(audiocpp SHARED src/capi/audiocpp.cpp)
    target_link_libraries(audiocpp PRIVATE engine_runtime ggml)
    target_include_directories(audiocpp PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    target_compile_definitions(audiocpp
        PRIVATE
            AUDIOCPP_BUILD_DLL
            AUDIOCPP_VERSION_STRING="${AUDIOCPP_VERSION}"
        # Anything linking this gets __declspec(dllimport) on Windows without
        # having to know to ask for it. No effect elsewhere.
        INTERFACE
            AUDIOCPP_USE_DLL
    )
    set_target_properties(audiocpp PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        VERSION 0.1.0
        SOVERSION 0
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
    if (APPLE)
        set_target_properties(audiocpp PROPERTIES
            INSTALL_RPATH "@loader_path" BUILD_WITH_INSTALL_RPATH ON)
    elseif (UNIX)
        set_target_properties(audiocpp PROPERTIES
            INSTALL_RPATH "$ORIGIN" BUILD_WITH_INSTALL_RPATH ON)
    endif()
    # The visibility presets above only reach this target's own sources. The
    # static libraries linked in (ggml, cJSON, sentencepiece, ...) are built
    # without -fvisibility=hidden, so without an explicit export list they are
    # re-exported from libaudiocpp -- measured at 2767 dynamic symbols where the
    # header declares 69 -- and can collide with whatever the host application
    # has already loaded. Restrict the surface to audiocpp_*.
    #
    # Windows has no equivalent of these allowlists: __declspec(dllexport) is
    # additive, so a DLL's surface is whatever its own annotations plus its
    # dependencies' annotations add up to. Exclusivity there comes from making
    # sure no dependency annotates its symbols, which is why cjson_vendor is
    # built with CJSON_HIDE_SYMBOLS above.
    if (APPLE)
        target_link_options(audiocpp PRIVATE
            "-Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/src/capi/audiocpp.symbols")
        set_property(TARGET audiocpp APPEND PROPERTY
            LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/capi/audiocpp.symbols")
    elseif (NOT WIN32)
        target_link_options(audiocpp PRIVATE
            "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/capi/audiocpp.map"
            "-Wl,--exclude-libs,ALL")
        set_property(TARGET audiocpp APPEND PROPERTY
            LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/capi/audiocpp.map")
    endif()
    if (AUDIOCPP_BUILD_C_API_STATIC)
        # Static twin of the C ABI for embedding: the same facade over
        # engine_runtime, archived as libaudiocpp.a alongside libaudiocpp.so.
        # No AUDIOCPP_BUILD_DLL/AUDIOCPP_USE_DLL annotations so consumers do
        # not inherit dllimport semantics, and no version script: static
        # archives have no dynamic symbol surface to restrict.
        add_library(audiocpp_static STATIC src/capi/audiocpp.cpp)
        target_link_libraries(audiocpp_static PRIVATE engine_runtime ggml)
        target_include_directories(audiocpp_static PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        )
        target_compile_definitions(audiocpp_static PRIVATE
            AUDIOCPP_VERSION_STRING="${AUDIOCPP_VERSION}"
        )
        set_target_properties(audiocpp_static PROPERTIES
            ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
            POSITION_INDEPENDENT_CODE ON
        )
        if (WIN32)
            # The shared target's MSVC import library also lands in bin/ as
            # audiocpp.lib; give the static archive its own name so the two
            # never collide.
            set_target_properties(audiocpp_static PROPERTIES OUTPUT_NAME audiocpp_static)
        else()
            set_target_properties(audiocpp_static PROPERTIES OUTPUT_NAME audiocpp)
        endif()
        if (ENGINE_ENABLE_OPENMP)
            target_link_libraries(audiocpp_static PRIVATE OpenMP::OpenMP_CXX)
        endif()
        if(AUDIOCPP_STATIC_ESPEAK)
            audiocpp_stage_espeak(audiocpp_static)
        endif()
    endif()
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(audiocpp PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(AUDIOCPP_STATIC_ESPEAK)
        audiocpp_stage_espeak(audiocpp)
    endif()

endif()

set(AUDIOCPP_UI_DIST "${CMAKE_CURRENT_SOURCE_DIR}/webui/native/dist/index.html")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${AUDIOCPP_UI_DIST}")
if (EXISTS "${AUDIOCPP_UI_DIST}")
    file(READ "${AUDIOCPP_UI_DIST}" AUDIOCPP_UI_HEX HEX)
else()
    string(HEX "<!doctype html><title>audio.cpp</title><p>Embedded WebUI assets are unavailable.</p>" AUDIOCPP_UI_HEX)
endif()
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," AUDIOCPP_UI_BYTES "${AUDIOCPP_UI_HEX}")
string(REPLACE "," ",\n" AUDIOCPP_UI_BYTES "${AUDIOCPP_UI_BYTES}")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/app/server/audiocpp_ui_asset.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/audiocpp_ui_asset.h"
    @ONLY
)

set(AUDIOCPP_DEMO_VOICE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/webui/native/demo_voices")
macro(audiocpp_embed_demo_voice_file INPUT_FILE OUTPUT_VARIABLE)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${INPUT_FILE}")
    file(READ "${INPUT_FILE}" _audiocpp_demo_voice_hex HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," ${OUTPUT_VARIABLE} "${_audiocpp_demo_voice_hex}")
    string(REPLACE "," ",\n" ${OUTPUT_VARIABLE} "${${OUTPUT_VARIABLE}}")
endmacro()
audiocpp_embed_demo_voice_file(
    "${AUDIOCPP_DEMO_VOICE_DIR}/demo_1_man.wav"
    AUDIOCPP_DEMO_1_MAN_BYTES)
audiocpp_embed_demo_voice_file(
    "${AUDIOCPP_DEMO_VOICE_DIR}/demo_2_man.wav"
    AUDIOCPP_DEMO_2_MAN_BYTES)
audiocpp_embed_demo_voice_file(
    "${AUDIOCPP_DEMO_VOICE_DIR}/demo_3_woman.wav"
    AUDIOCPP_DEMO_3_WOMAN_BYTES)
audiocpp_embed_demo_voice_file(
    "${AUDIOCPP_DEMO_VOICE_DIR}/demo_4_woman.wav"
    AUDIOCPP_DEMO_4_WOMAN_BYTES)
audiocpp_embed_demo_voice_file(
    "${AUDIOCPP_DEMO_VOICE_DIR}/prompt_text"
    AUDIOCPP_DEMO_PROMPT_TEXT_BYTES)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/app/server/audiocpp_demo_voice_assets.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/audiocpp_demo_voice_assets.h"
    @ONLY
)

add_executable(audiocpp_server
    app/common/base64.cpp
    app/common/build_info.cpp
    app/server/main.cpp
    app/server/config.cpp
    app/server/frontend.cpp
    app/server/http.cpp
    app/server/parallel_http.cpp
    app/server/model_memory.cpp
    app/server/multipart.cpp
    app/server/server_protocol.cpp
    app/server/server_ui.cpp
    app/server/runtime.cpp
    app/server/parallel_runtime.cpp
    app/server/ui_assets.cpp
    app/cli/args.cpp
    app/cli/request.cpp
    app/streaming/pcm_source.cpp
    app/streaming/streaming.cpp
    ${AUDIOCPP_RECIPE_SOURCES}
)

target_link_libraries(audiocpp_server PRIVATE engine_runtime ggml)
audiocpp_use_utf8_codepage(audiocpp_server)
if(AUDIOCPP_STATIC_ESPEAK)
    audiocpp_stage_espeak(audiocpp_server)
endif()
audiocpp_configure_server_frontends(audiocpp_server)
if (AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER)
    target_sources(audiocpp_server PRIVATE app/server/model_installer.cpp)
    target_link_libraries(audiocpp_server PRIVATE audiocpp_package_manager)
    target_compile_definitions(audiocpp_server PRIVATE AUDIOCPP_HAS_NATIVE_MODEL_MANAGER=1)
endif()
target_include_directories(audiocpp_server PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
if (WIN32)
    target_link_libraries(audiocpp_server PRIVATE ws2_32)
endif()
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(audiocpp_server PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(audiocpp_gguf
    app/gguf/main.cpp
)
target_link_libraries(audiocpp_gguf PRIVATE engine_runtime ggml)
audiocpp_use_utf8_codepage(audiocpp_gguf)
target_include_directories(audiocpp_gguf PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(audiocpp_gguf PRIVATE OpenMP::OpenMP_CXX)
endif()

if (AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER)
    add_executable(audiocpp_model_manager
        app/model_manager/main.cpp
    )
    target_link_libraries(audiocpp_model_manager PRIVATE audiocpp_package_manager)
endif()

if ("kitten_tts2" IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(audiocpp_kitten_tts2_prepare_voices EXCLUDE_FROM_ALL
        tools/gguf/kitten_tts2/kitten_tts2_voices_prepare.cpp)
    audiocpp_use_utf8_codepage(audiocpp_kitten_tts2_prepare_voices)
    target_link_libraries(audiocpp_kitten_tts2_prepare_voices PRIVATE engine_runtime)
endif()

if (ENGINE_BUILD_WARMBENCH)
    function(add_engine_warmbench target_name source_file)
        add_executable(${target_name}
            ${source_file}
        )
        target_link_libraries(${target_name} PRIVATE engine_runtime ggml)
        if (ENGINE_ENABLE_OPENMP)
            target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endfunction()

    add_engine_warmbench(campplus_shared_default_probe tests/glm_tts/campplus_shared_default_probe.cpp)
    add_engine_warmbench(chatterbox_warm_bench tests/chatterbox/chatterbox_warm_bench.cpp)
    if(kokoro_tts IN_LIST AUDIOCPP_LINKED_MODELS)
        add_engine_warmbench(kokoro_tts_warm_bench tests/kokoro_tts/kokoro_tts_warm_bench.cpp)
    endif()
    add_engine_warmbench(citrinet_asr_warm_bench tests/citrinet_asr/citrinet_asr_warm_bench.cpp)
    add_engine_warmbench(confucius4_tts_warm_bench tests/confucius4_tts/confucius4_tts_warm_bench.cpp)
    add_engine_warmbench(controlfoley_warm_bench tests/controlfoley/controlfoley_warm_bench.cpp)
    add_engine_warmbench(fun_asr_nano_warm_bench tests/fun_asr_nano/fun_asr_nano_warm_bench.cpp)
    add_engine_warmbench(granite5asr_warm_bench tests/granite5asr/granite5asr_warm_bench.cpp)
    target_compile_definitions(granite5asr_warm_bench PRIVATE
        ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_engine_warmbench(higgs_audio_stt_warm_bench tests/higgs_audio_stt/higgs_audio_stt_warm_bench.cpp)
    add_engine_warmbench(higgs_audio_tts_warm_bench tests/higgs_audio_tts/higgs_audio_tts_warm_bench.cpp)
    add_engine_warmbench(hviske_asr_warm_bench tests/hviske_asr/hviske_asr_warm_bench.cpp)
    add_engine_warmbench(index_tts2_warm_bench tests/index_tts2/index_tts2_warm_bench.cpp)
    add_engine_warmbench(irodori_tts_warm_bench tests/irodori_tts/irodori_tts_warm_bench.cpp)
    add_engine_warmbench(marblenet_vad_warm_bench tests/marblenet_vad/marblenet_vad_warm_bench.cpp)
    add_engine_warmbench(miocodec_warm_bench tests/miocodec/miocodec_warm_bench.cpp)
    if (voice_isolator IN_LIST AUDIOCPP_LINKED_MODELS)
        add_engine_warmbench(voice_isolator_warm_bench tests/voice_isolator/voice_isolator_warm_bench.cpp)
        target_sources(voice_isolator_warm_bench PRIVATE app/cli/args.cpp)
        target_include_directories(voice_isolator_warm_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    endif()
    add_engine_warmbench(mira_tts_warm_bench tests/mira_tts/mira_tts_warm_bench.cpp)
    add_engine_warmbench(muscriptor_warm_bench tests/muscriptor/muscriptor_warm_bench.cpp)
    add_engine_warmbench(moss_tts_nano_warm_bench tests/moss_tts_nano/moss_tts_nano_warm_bench.cpp)
    add_engine_warmbench(moss_tts_local_warm_bench tests/moss_tts_local/moss_tts_local_warm_bench.cpp)
    add_engine_warmbench(nemotron_asr_warm_bench tests/nemotron_asr/nemotron_asr_warm_bench.cpp)
    add_engine_warmbench(neutts_warm_bench tests/neutts/neutts_warm_bench.cpp)
    add_engine_warmbench(omnivoice_warm_bench tests/omnivoice/omnivoice_warm_bench.cpp)
    add_engine_warmbench(omnivoice_batch_test tests/omnivoice/omnivoice_batch_test.cpp)
    add_engine_warmbench(outetts_warm_bench tests/outetts/outetts_warm_bench.cpp)
    add_engine_warmbench(parakeet_warm_bench tests/parakeet_tdt/parakeet_warm_bench.cpp)
    add_engine_warmbench(glm_tts_tokenizer_probe tests/glm_tts/glm_tts_tokenizer_probe.cpp)
    add_engine_warmbench(glm_tts_assets_probe tests/glm_tts/glm_tts_assets_probe.cpp)
    add_engine_warmbench(glm_tts_prompt_probe tests/glm_tts/glm_tts_prompt_probe.cpp)
    add_engine_warmbench(glm_tts_speech_tokenizer_probe tests/glm_tts/glm_tts_speech_tokenizer_probe.cpp)
    add_engine_warmbench(glm_tts_flow_probe tests/glm_tts/glm_tts_flow_probe.cpp)
    add_engine_warmbench(glm_tts_flow_conditioned_probe tests/glm_tts/glm_tts_flow_conditioned_probe.cpp)
    add_engine_warmbench(glm_tts_frontend_probe tests/glm_tts/glm_tts_frontend_probe.cpp)
    add_engine_warmbench(glm_tts_campplus_probe tests/glm_tts/glm_tts_campplus_probe.cpp)
    add_engine_warmbench(glm_tts_llama_probe tests/glm_tts/glm_tts_llama_probe.cpp)
    add_engine_warmbench(glm_tts_warm_bench tests/glm_tts/glm_tts_warm_bench.cpp)
    add_engine_warmbench(kroko_asr_encoder_probe tests/kroko_asr/kroko_asr_encoder_probe.cpp)
    add_engine_warmbench(kroko_asr_frontend_probe tests/kroko_asr/kroko_asr_frontend_probe.cpp)
    add_engine_warmbench(kroko_asr_streaming_probe tests/kroko_asr/kroko_asr_streaming_probe.cpp)
    add_engine_warmbench(kroko_asr_zipformer_probe tests/kroko_asr/kroko_asr_zipformer_probe.cpp)
    add_engine_warmbench(pocket_tts_warm_bench tests/pocket_tts/pocket_tts_warm_bench.cpp)
    add_engine_warmbench(qwen3_asr_warm_bench tests/qwen3_asr/qwen3_asr_warm_bench.cpp)
    add_engine_warmbench(qwen3_forced_aligner_warm_bench tests/qwen3_forced_aligner/qwen3_forced_aligner_warm_bench.cpp)
    add_engine_warmbench(mms_forced_aligner_warm_bench tests/mms_forced_aligner/mms_forced_aligner_warm_bench.cpp)
    add_engine_warmbench(qwen3_tts_warm_bench tests/qwen3_tts/qwen3_tts_warm_bench.cpp)
    add_engine_warmbench(seed_vc_warm_bench tests/seed_vc/seed_vc_warm_bench.cpp)
    add_engine_warmbench(silero_vad_warm_bench tests/silero_vad/silero_vad_warm_bench.cpp)
    add_engine_warmbench(soprano_warm_bench tests/soprano_tts/soprano_warm_bench.cpp)
    target_compile_definitions(soprano_warm_bench PRIVATE
        ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_engine_warmbench(sopro_probe tests/sopro_tts/sopro_probe.cpp)
    add_engine_warmbench(sortformer_diar_warm_bench tests/sortformer_diar/sortformer_diar_warm_bench.cpp)
    add_engine_warmbench(supertonic_warm_bench tests/supertonic/supertonic_warm_bench.cpp)
    add_engine_warmbench(vevo2_warm_bench tests/vevo2/vevo2_warm_bench.cpp)
    add_engine_warmbench(vibevoice_asr_warm_bench tests/vibevoice_asr/vibevoice_asr_warm_bench.cpp)
    add_engine_warmbench(voxtral_realtime_warm_bench tests/voxtral_realtime/voxtral_realtime_warm_bench.cpp)
    add_engine_warmbench(voxcpm2_warm_bench tests/voxcpm2/voxcpm2_warm_bench.cpp)
endif()

# Shared test setup; declarations and category selection stay separate.
if (ENGINE_BUILD_TESTS OR ENGINE_BUILD_EXTENDED_TESTS OR ENGINE_BUILD_MODEL_TESTS)
    enable_testing()
    find_package(Threads REQUIRED)
    if (AUDIOCPP_BUILD_C_API)
        find_package(Python3 COMPONENTS Interpreter QUIET)
    endif()
    set(ENGINE_UNITTEST_ASSET_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests/assets")

    # Target-only probes and executables with multiple/conditional CTest entries.
    function(add_engine_test_target target source)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE engine_runtime ggml)
    endfunction()

    # Standard test: create the executable and register its command together.
    function(add_engine_test target source)
        add_engine_test_target(${target} ${source})
        target_include_directories(${target} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests")
        add_test(NAME ${target} COMMAND ${target} ${ARGN})
    endfunction()

    # Shared link support, not a test executable or registration.
    if (granite_speech5 IN_LIST AUDIOCPP_LINKED_MODELS)
        add_library(granite_speech5_test_impl STATIC EXCLUDE_FROM_ALL
            $<TARGET_OBJECTS:engine_model_granite_speech5>)
        target_link_libraries(granite_speech5_test_impl PUBLIC engine_runtime ggml)
    endif()
endif()

if (ENGINE_BUILD_TESTS)
    include(cmake/tests/unit.cmake)
endif()
if (ENGINE_BUILD_EXTENDED_TESTS)
    include(cmake/tests/extended.cmake)
endif()
if (ENGINE_BUILD_MODEL_TESTS)
    include(cmake/tests/models.cmake)
endif()

if (ENGINE_BUILD_EXAMPLES)
    add_executable(ggml_simple_inference
        examples/ggml_simple_inference.cpp
    )

    target_link_libraries(ggml_simple_inference PRIVATE ggml)

    add_executable(model_spec_demo
        examples/model_spec_demo/model_spec_demo.cpp
    )
    target_link_libraries(model_spec_demo PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(model_spec_demo PRIVATE OpenMP::OpenMP_CXX)
    endif()

    add_executable(model_spec_download_demo
        examples/model_spec_demo/model_spec_download_demo.cpp
    )
    target_link_libraries(model_spec_download_demo PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(model_spec_download_demo PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
