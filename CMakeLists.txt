cmake_minimum_required(VERSION 3.20)

project(AudioCpp LANGUAGES C CXX)

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
file(GLOB AUDIOCPP_MODEL_SPEC_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/model_specs/*.json")
set(AUDIOCPP_BUILTIN_MODEL_SPEC_ENTRIES "")
set(AUDIOCPP_CONVERTER_MODEL_SPEC_ENTRIES "")
set(AUDIOCPP_NATIVE_PACKAGE_SPEC_ENTRIES "")
foreach(AUDIOCPP_MODEL_SPEC_FILE IN LISTS AUDIOCPP_MODEL_SPEC_FILES)
    get_filename_component(AUDIOCPP_MODEL_SPEC_FAMILY "${AUDIOCPP_MODEL_SPEC_FILE}" NAME_WE)
    file(READ "${AUDIOCPP_MODEL_SPEC_FILE}" AUDIOCPP_MODEL_SPEC_JSON)
    string(APPEND AUDIOCPP_CONVERTER_MODEL_SPEC_ENTRIES
        "    {\"${AUDIOCPP_MODEL_SPEC_FAMILY}\", R\"AUDIOCPP_SPEC(${AUDIOCPP_MODEL_SPEC_JSON})AUDIOCPP_SPEC\"},\n")
    string(APPEND AUDIOCPP_NATIVE_PACKAGE_SPEC_ENTRIES
        "    {\"${AUDIOCPP_MODEL_SPEC_FAMILY}\", R\"AUDIOCPP_SPEC(${AUDIOCPP_MODEL_SPEC_JSON})AUDIOCPP_SPEC\"},\n")
    if (AUDIOCPP_DEPLOYMENT_BUILD)
        string(APPEND AUDIOCPP_BUILTIN_MODEL_SPEC_ENTRIES
            "    {\"${AUDIOCPP_MODEL_SPEC_FAMILY}\", R\"AUDIOCPP_SPEC(${AUDIOCPP_MODEL_SPEC_JSON})AUDIOCPP_SPEC\"},\n")
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
option(ENGINE_ENABLE_METAL "Build ggml with Metal backend support" ${ENGINE_DEFAULT_ENABLE_METAL})
option(ENGINE_ENABLE_LLAMAFILE "Build ggml with llamafile SGEMM support" ${ENGINE_DEFAULT_ENABLE_LLAMAFILE})
option(ENGINE_ENABLE_CUDA_GRAPHS "Enable ggml CUDA graphs support" ${ENGINE_DEFAULT_ENABLE_CUDA_GRAPHS})
option(ENGINE_HIP_STRIX_HALO_OPTIMIZATIONS
    "Enable opt-in HIP optimizations validated only for Strix Halo (gfx1151)"
    OFF)
option(ENGINE_ENABLE_NATIVE_CPU "Build ggml CPU kernels with native host ISA flags" ${ENGINE_DEFAULT_ENABLE_NATIVE_CPU})
option(ENGINE_ENABLE_OPENMP "Build host code with OpenMP support" ON)
option(AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER
    "Build native model-manager tools and server download/install support"
    OFF)
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
option(ENGINE_BUILD_WARMBENCH "Build warmbench helper binaries" OFF)
set(AUDIOCPP_GGML_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/external/ggml"
    CACHE PATH "ggml source tree to build with AudioCPP")

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
    enable_language(CUDA)
    # 12.0 is a floor, not a target: any 12.x or 13.x works (the Dockerfile pins
    # 12.9.0, and docs/build/windows.md ships a CUDA 13 package). The minimum is
    # enforced because an older toolkit configures happily and only fails at the
    # first .cu file, which reads as a code bug rather than a toolchain one: nvcc
    # < 11.6 cannot parse libstdc++ >= 11.3 headers ("parameter packs not expanded
    # with '...'" in <functional>), and distros still ship such pairings (Ubuntu
    # 22.04's nvidia-cuda-toolkit is 11.5). Fail here, with a pointer, instead.
    # The upper bound is the host compiler the chosen toolkit accepts, not CUDA
    # itself (e.g. CUDA 12.9 supports GCC <= 14), so it is left to the toolkit.
    find_package(CUDAToolkit 12.0 REQUIRED)
    if (NOT MSVC)
        set(CMAKE_CUDA_FLAGS_DEBUG "${AUDIOCPP_DEBUG_OPT_FLAGS}" CACHE STRING "Optimized debug CUDA flags" FORCE)
    endif()
endif()
if (ENGINE_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
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
    target_link_libraries(${target_name} PRIVATE ggml sentencepiece)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
        if (MSVC)
            target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/openmp:experimental>)
        endif()
    endif()
endfunction()

function(audiocpp_add_model target_name)
    cmake_parse_arguments(ARG "" "" "SOURCES;INCLUDES;LOADERS;ALIASES;DEPENDS" ${ARGN})
    add_library(engine_model_${target_name} OBJECT ${ARG_SOURCES})
    set_target_properties(engine_model_${target_name} PROPERTIES EXCLUDE_FROM_ALL TRUE)
    audiocpp_configure_runtime_object(engine_model_${target_name})
    set(AUDIOCPP_MODEL_TARGETS ${AUDIOCPP_MODEL_TARGETS} ${target_name} PARENT_SCOPE)
    set(AUDIOCPP_MODEL_${target_name}_INCLUDES "${ARG_INCLUDES}" PARENT_SCOPE)
    set(AUDIOCPP_MODEL_${target_name}_LOADERS "${ARG_LOADERS}" PARENT_SCOPE)
    set(AUDIOCPP_MODEL_${target_name}_DEPENDS "${ARG_DEPENDS}" PARENT_SCOPE)
    set(AUDIOCPP_MODEL_ALIAS_${target_name} ${target_name} PARENT_SCOPE)
    foreach(alias IN LISTS ARG_ALIASES)
        set(AUDIOCPP_MODEL_ALIAS_${alias} ${target_name} PARENT_SCOPE)
    endforeach()
endfunction()

add_library(engine_core OBJECT
    src/framework/assets/resource_bundle.cpp
    src/framework/model_spec/package.cpp
    src/framework/model_spec/options.cpp
    src/framework/model_spec/schema.cpp
    src/framework/assets/tensor_source.cpp
    src/framework/assets/weight_norm_folding.cpp
    src/framework/assets/torch_bin.cpp
    src/framework/core/module.cpp
    src/framework/core/backend.cpp
    src/framework/core/deferred_tensor_writer.cpp
    src/framework/core/execution_context.cpp
    src/framework/core/host_memory.cpp
    src/framework/runtime/model.cpp
    src/framework/model_spec/metadata.cpp
    src/framework/runtime/session.cpp
    src/framework/runtime/artifacts.cpp
    src/framework/runtime/cache.cpp
    src/framework/runtime/graph_executor.cpp
    src/framework/runtime/graph_optimizer.cpp
    src/framework/runtime/host_ops.cpp
    src/framework/runtime/kv_cache.cpp
    src/framework/runtime/bounded_static_kv_decode.cpp
    src/framework/runtime/options.cpp
    src/framework/runtime/session_base.cpp
    src/framework/runtime/workspace.cpp
    src/framework/midi/midi_file.cpp
    src/framework/io/filesystem.cpp
    src/framework/io/config.cpp
    src/framework/io/json.cpp
    src/framework/io/text.cpp
    src/framework/text/chinese_normalization.cpp
    src/framework/text/chunking.cpp
    src/framework/text/subtitle.cpp
    src/framework/text/text_normalization.cpp
    src/framework/text/unicode_normalization.cpp
    src/framework/io/yaml.cpp
    src/framework/io/binary.cpp
    src/framework/io/safetensors.cpp
    src/framework/debug/trace.cpp
    src/framework/debug/profiler.cpp
    src/framework/decoders/tdt_decoder_runner.cpp
    src/framework/decoders/tdt_decoder_greedy_customized.cpp
    src/framework/decoders/tdt_decoder_greedy_duration_loop.cpp
    src/framework/decoders/tdt_decoder_greedy_separate_heads.cpp
    src/framework/modules/linear_module.cpp
    src/framework/modules/packed_linear_weights.cpp
    src/framework/modules/primitive_modules.cpp
    src/framework/modules/activation_modules.cpp
    src/framework/modules/norm_modules.cpp
    src/framework/modules/lookup_modules.cpp
    src/framework/modules/structural_modules.cpp
    src/framework/modules/asr_helpers.cpp
    src/framework/modules/conv_modules.cpp
    src/framework/modules/zipformer_modules.cpp
    src/framework/modules/speech_encoders/whisper_embedding.cpp
    src/framework/modules/speech_encoders/whisper_frontend.cpp
    src/framework/modules/speech_encoders/campplus_encoder.cpp
    src/framework/modules/codecs/nemo_nano_codec.cpp
    src/framework/modules/pitch_extractors/rmvpe_pitch_extractor.cpp
    src/framework/modules/vocoders/bigvgan_vocoder.cpp
    src/framework/modules/vocoders/hift_vocoder.cpp
    src/framework/modules/speech_encoders/hubert_encoder.cpp
    src/framework/modules/speech_encoders/wav2vec2_bert_encoder.cpp
    src/framework/modules/speech_encoders/wavlm_encoder.cpp
    src/framework/modules/speech_encoders/sanm.cpp
    src/framework/modules/optimizations/fast_conv_modules.cpp
    src/framework/modules/optimizations/fast_projection_modules.cpp
    src/framework/modules/recurrent_modules.cpp
    src/framework/modules/streaming_conv_modules.cpp
    src/framework/modules/attention/positional_encoding.cpp
    src/framework/modules/attention/feed_forward.cpp
    src/framework/modules/attention/self_attention.cpp
    src/framework/modules/attention/longformer_attention.cpp
    src/framework/modules/attention/longformer_attention_gpu.cpp
    src/framework/modules/attention/common_relative_attention.cpp
    src/framework/modules/attention/relative_attention.cpp
    src/framework/modules/attention/scaled_dot_product_attention.cpp
    src/framework/modules/attention/grouped_query_attention.cpp
    src/framework/modules/attention/projected_grouped_self_attention.cpp
    src/framework/modules/attention/streaming_frame_attention_cache.cpp
    src/framework/modules/transformers/qwen_decoder.cpp
    src/framework/modules/transformers/qwen_causal_decoder.cpp
    src/framework/modules/transformers/qwen_causal_decode_runtime.cpp
    src/framework/modules/transformers/qwen3_vl_encoder_runtime.cpp
    src/framework/modules/attention/cross_attention.cpp
    src/framework/modules/attention/transformer_blocks.cpp
    src/framework/modules/conditioning_modules.cpp
    src/framework/modules/conformer_modules.cpp
    src/framework/modules/flow_sampler_runtime.cpp
    src/framework/modules/predictor_modules.cpp
    src/framework/modules/positional_modules.cpp
    src/framework/modules/text_encoders/t5_base_encoder.cpp
    src/framework/modules/text_encoders/t5_gemma_encoder.cpp
    src/framework/modules/transformers/gemma_decoder.cpp
    src/framework/modules/optimizations/fast_kv_modules.cpp
    src/framework/tokenizers/hf_tokenizer_json.cpp
    src/framework/tokenizers/llama_bpe.cpp
    external/llama_tokenizer/bpe-core.cpp
    external/llama_tokenizer/unicode.cpp
    external/llama_tokenizer/unicode-data.cpp
    src/framework/tokenizers/sentencepiece.cpp
    src/framework/audio/fft.cpp
    src/framework/audio/kaldi_fbank.cpp
    src/framework/audio/activity.cpp
    src/framework/audio/chunking.cpp
    src/framework/audio/conversion.cpp
    src/framework/audio/deepfilternet2.cpp
    src/framework/audio/dsp.cpp
    src/framework/audio/flashsr.cpp
    src/framework/audio/istft_graph.cpp
    src/framework/audio/mixing.cpp
    src/framework/audio/resampling.cpp
    src/framework/audio/output.cpp
    src/framework/audio/rnnoise.cpp
    src/framework/audio/utility_api.cpp
    src/framework/audio/waveform_ops.cpp
    src/framework/audio/wav_reader.cpp
    src/framework/audio/wav_writer.cpp
    src/framework/audio/zipenhancer.cpp
    src/framework/runtime/flow_kv_cache.cpp
    src/framework/modules/speaker_encoders/ecapa_tdnn_runtime.cpp
    src/framework/modules/speaker_encoders/titanet_runtime.cpp
    src/framework/codecs/fsq_audio_codec_runtime.cpp
    src/framework/codecs/mimi_codec_runtime.cpp
    src/framework/codecs/neural_audio.cpp
    src/framework/sampling/decode_modules.cpp
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
    ALIASES
        htdemucs
)

audiocpp_add_model(omnivoice
    SOURCES
        src/models/omnivoice/assets.cpp
        src/models/omnivoice/tokenizer_text.cpp
        src/models/omnivoice/audio_tokenizer.cpp
        src/models/omnivoice/prompt_builder.cpp
        src/models/omnivoice/generator.cpp
        src/models/omnivoice/postprocess.cpp
        src/models/omnivoice/session.cpp
        src/models/omnivoice/loader.cpp
    INCLUDES
        engine/models/omnivoice/loader.h
    LOADERS
        engine::models::omnivoice::make_omnivoice_loader
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

audiocpp_add_model(parakeet_tdt
    SOURCES
        src/community_models/parakeet_tdt/assets.cpp
        src/community_models/parakeet_tdt/session.cpp
        src/community_models/parakeet_tdt/encoder.cpp
        src/community_models/parakeet_tdt/decoder.cpp
        src/community_models/parakeet_tdt/frontend.cpp
        src/community_models/parakeet_tdt/weights.cpp
    INCLUDES
        engine/community_models/parakeet_tdt/session.h
    LOADERS
        engine::community_models::parakeet_tdt::make_parakeet_tdt_loader
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

audiocpp_add_model(inflect_v2
    SOURCES
        src/community_models/inflect_v2/assets.cpp
        src/community_models/inflect_v2/frontend.cpp
        src/community_models/inflect_v2/runtime.cpp
        src/community_models/inflect_v2/session.cpp
    INCLUDES
        engine/community_models/inflect_v2/session.h
    LOADERS
        engine::models::inflect_v2::make_inflect_v2_loader
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
        src/models/pocket_tts/audio_decoder.cpp
        src/models/pocket_tts/flow_lm.cpp
        src/models/pocket_tts/mimi_decoder.cpp
        src/models/pocket_tts/mimi_encoder.cpp
        src/models/pocket_tts/text_conditioner.cpp
        src/models/pocket_tts/voice_state_assets.cpp
        src/models/pocket_tts/voice_conditioner.cpp
        src/models/pocket_tts/loader.cpp
        src/models/pocket_tts/session.cpp
    INCLUDES
        engine/models/pocket_tts/loader.h
    LOADERS
        engine::models::pocket_tts::make_pocket_tts_loader
)

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
        src/models/moss/shared/audio_tokenizer_decoder.cpp
        src/models/moss/shared/audio_tokenizer_encoder.cpp
        src/models/moss/shared/audio_tokenizer_config.cpp
        src/models/moss/shared/audio_tokenizer_quantizer.cpp
        src/models/moss/shared/sampling.cpp
        src/models/moss/shared/token_rows.cpp
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
        src/models/fish_audio/codec.cpp
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
        src/models/heartmula/codec.cpp
        src/models/heartmula/generator.cpp
        src/models/heartmula/mula.cpp
        src/models/heartmula/session.cpp
        src/models/heartmula/tokenizer_text.cpp
    INCLUDES
        engine/models/heartmula/session.h
    LOADERS
        engine::models::heartmula::make_heartmula_loader
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

audiocpp_add_model(irodori_tts
    SOURCES
        src/models/irodori_tts/assets.cpp
        src/models/irodori_tts/codec.cpp
        src/models/irodori_tts/condition_encoder.cpp
        src/models/irodori_tts/duration.cpp
        src/models/irodori_tts/rf_dit.cpp
        src/models/irodori_tts/session.cpp
        src/models/irodori_tts/tokenizer_text.cpp
    INCLUDES
        engine/models/irodori_tts/session.h
    LOADERS
        engine::models::irodori_tts::make_irodori_tts_loader
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
        src/models/nemotron_asr/loader.cpp
        src/models/nemotron_asr/weights.cpp
    INCLUDES
        engine/models/nemotron_asr/loader.h
    LOADERS
        engine::models::nemotron_asr::make_nemotron_asr_loader
)

audiocpp_add_model(qwen3_tts
    SOURCES
        src/models/qwen3_tts/assets.cpp
        src/models/qwen3_tts/tokenizer_text.cpp
        src/models/qwen3_tts/tokenizer_speech_encoder.cpp
        src/models/qwen3_tts/tokenizer_speech_decoder.cpp
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

audiocpp_add_model(vietneu_tts
    SOURCES
        src/community_models/vietneu_tts/assets.cpp
        src/community_models/vietneu_tts/tokenizer_text.cpp
        src/community_models/vietneu_tts/tokenizer_speech_encoder.cpp
        src/community_models/vietneu_tts/tokenizer_speech_decoder.cpp
        src/community_models/vietneu_tts/speaker_encoder.cpp
        src/community_models/vietneu_tts/talker.cpp
        src/community_models/vietneu_tts/prompt_tts_voice_clone.cpp
        src/community_models/vietneu_tts/session.cpp
        src/community_models/vietneu_tts/loader.cpp
    INCLUDES
        engine/community_models/vietneu_tts/loader.h
    LOADERS
        engine::models::vietneu_tts::make_vietneu_tts_loader
    DEPENDS
        moss
)

audiocpp_add_model(qwen3_asr
    SOURCES
        src/models/qwen3_asr/assets.cpp
        src/models/qwen3_asr/tokenizer_text.cpp
        src/models/qwen3_asr/frontend_whisper.cpp
        src/models/qwen3_asr/audio_encoder.cpp
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
)

audiocpp_add_model(qwen3_forced_aligner
    SOURCES
        src/models/qwen3_forced_aligner/processor.cpp
        src/models/qwen3_forced_aligner/session.cpp
        src/models/qwen3_forced_aligner/loader.cpp
    INCLUDES
        engine/models/qwen3_forced_aligner/loader.h
    LOADERS
        engine::models::qwen3_forced_aligner::make_qwen3_forced_aligner_loader
    DEPENDS
        qwen3_asr
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
        src/models/chatterbox/hift_vocoder_impl.cpp
        src/models/chatterbox/loader.cpp
        src/models/chatterbox/s3_tokenizer_impl.cpp
        src/models/chatterbox/s3gen_flow.cpp
        src/models/chatterbox/s3gen_inference.cpp
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
foreach(model_name IN LISTS AUDIOCPP_LINKED_MODELS)
    list(APPEND AUDIOCPP_RUNTIME_OBJECTS $<TARGET_OBJECTS:engine_model_${model_name}>)
endforeach()

message(STATUS "audio.cpp model composite: ${AUDIOCPP_MODEL_SET} selected [${AUDIOCPP_ENABLED_MODELS}] linked [${AUDIOCPP_LINKED_MODELS}]")
add_library(engine_runtime STATIC
    src/framework/runtime/registry.cpp
    ${AUDIOCPP_RUNTIME_OBJECTS}
)

add_library(cjson_vendor STATIC
    external/cJSON/cJSON.c
)

target_include_directories(cjson_vendor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/external/cJSON
)

# Native package management is deliberately opt-in. Normal CLI and server
# builds do not configure an HTTP/TLS dependency. When enabled, package specs
# remain embedded so the standalone manager and managed WebUI do not require a
# source checkout or an external model_specs directory.
if (AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER)
    add_subdirectory(external/cpp-httplib)
    add_library(audiocpp_package_manager STATIC
        src/framework/package_manager/manager.cpp
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
target_link_libraries(engine_runtime PRIVATE sentencepiece cjson_vendor yaml_vendor)
if (AUDIOCPP_HIP_STRIX_HALO_OPTIMIZATIONS_ACTIVE)
    target_compile_definitions(engine_runtime PRIVATE ENGINE_HIP_STRIX_HALO_OPTIMIZATIONS=1)
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
    target_compile_options(engine_runtime PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()

if (ENGINE_ENABLE_CUDA AND NOT ENGINE_ENABLE_HIP)
    target_sources(engine_runtime PRIVATE
        src/framework/audio/istft_cuda_runtime.cu
        src/framework/sampling/torch_random_cuda_runtime.cu
    )
    set_source_files_properties(
        src/framework/audio/istft_cuda_runtime.cu
        src/framework/sampling/torch_random_cuda_runtime.cu
        PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    if (CMAKE_CUDA_ARCHITECTURES)
        set_target_properties(engine_runtime PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    else()
        set_target_properties(engine_runtime PROPERTIES CUDA_ARCHITECTURES native)
    endif()
    target_compile_definitions(engine_core PRIVATE ENGINE_HAS_CUDA_ISTFT=1 ENGINE_HAS_CUDA_TORCH_RANDOM=1)
    target_compile_definitions(engine_runtime PUBLIC ENGINE_HAS_CUDA_ISTFT=1 ENGINE_HAS_CUDA_TORCH_RANDOM=1)
    target_link_libraries(engine_runtime PRIVATE CUDA::cudart CUDA::cufft)
endif()

add_executable(audiocpp_cli
    app/cli/main.cpp
    app/cli/args.cpp
    app/cli/batch.cpp
    app/cli/request.cpp
    app/streaming/pcm_source.cpp
    app/streaming/streaming.cpp
    app/workflow/execution.cpp
    app/workflow/file_sink.cpp
    app/workflow/pipeline.cpp
    app/workflow/workflow.cpp
)

target_link_libraries(audiocpp_cli PRIVATE engine_runtime ggml)
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
    app/server/main.cpp
    app/server/base64.cpp
    app/server/config.cpp
    app/server/http.cpp
    app/server/multipart.cpp
    app/server/runtime.cpp
    app/server/ui_assets.cpp
    app/cli/args.cpp
    app/cli/request.cpp
    app/streaming/pcm_source.cpp
    app/streaming/streaming.cpp
)

target_link_libraries(audiocpp_server PRIVATE engine_runtime ggml)
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

add_executable(model_perf
    tests/perf/model_perf.cpp
    app/cli/args.cpp
    app/cli/request.cpp
)
target_link_libraries(model_perf PRIVATE engine_runtime ggml)
target_include_directories(model_perf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(model_perf PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(miocodec_wavlm_parity
    tests/miocodec/miocodec_wavlm_parity.cpp
)

target_link_libraries(miocodec_wavlm_parity PRIVATE engine_runtime ggml)

add_executable(torch_bin_parity
    tests/vibevoice/torch_bin_parity.cpp
)

target_link_libraries(torch_bin_parity PRIVATE engine_runtime ggml)
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(torch_bin_parity PRIVATE OpenMP::OpenMP_CXX)
endif()

# Only build this parity check when the vibevoice model is linked, since it
# directly references model-internal symbols (load_vibevoice_assets etc.).
if (vibevoice IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(vibevoice_finetune_overlay_check
        tests/vibevoice/finetune_overlay_check.cpp
    )
    target_link_libraries(vibevoice_finetune_overlay_check PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(vibevoice_finetune_overlay_check PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

if (ENGINE_BUILD_TESTS)
    add_executable(moss_tts_local_smoke
        tests/moss_tts_local/moss_tts_local_smoke.cpp
    )

    target_link_libraries(moss_tts_local_smoke PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(moss_tts_local_smoke PRIVATE OpenMP::OpenMP_CXX)
    endif()

    add_executable(codec_dequant_parity
        tests/moss_tts_local/codec_dequant_parity.cpp
    )

    target_link_libraries(codec_dequant_parity PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(codec_dequant_parity PRIVATE OpenMP::OpenMP_CXX)
    endif()

    add_executable(codec_decode_parity
        tests/moss_tts_local/codec_decode_parity.cpp
    )

    target_link_libraries(codec_decode_parity PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(codec_decode_parity PRIVATE OpenMP::OpenMP_CXX)
    endif()

    add_executable(codec_encode_parity
        tests/moss_tts_local/codec_encode_parity.cpp
    )

    target_link_libraries(codec_encode_parity PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(codec_encode_parity PRIVATE OpenMP::OpenMP_CXX)
    endif()

    add_executable(personaplex_ring_kv_cache_parity
        tests/personaplex/personaplex_ring_kv_cache_parity.cpp
    )

    target_link_libraries(personaplex_ring_kv_cache_parity PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(personaplex_ring_kv_cache_parity PRIVATE OpenMP::OpenMP_CXX)
    endif()
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
    add_engine_warmbench(citrinet_asr_warm_bench tests/citrinet_asr/citrinet_asr_warm_bench.cpp)
    add_engine_warmbench(confucius4_tts_warm_bench tests/confucius4_tts/confucius4_tts_warm_bench.cpp)
    add_engine_warmbench(dramabox_warm_bench tools/dramabox/dramabox_warm_bench.cpp)
    add_engine_warmbench(fun_asr_nano_warm_bench tests/fun_asr_nano/fun_asr_nano_warm_bench.cpp)
    add_engine_warmbench(higgs_audio_stt_warm_bench tests/higgs_audio_stt/higgs_audio_stt_warm_bench.cpp)
    add_engine_warmbench(higgs_audio_tts_warm_bench tests/higgs_audio_tts/higgs_audio_tts_warm_bench.cpp)
    add_engine_warmbench(hviske_asr_warm_bench tests/hviske_asr/hviske_asr_warm_bench.cpp)
    add_engine_warmbench(index_tts2_warm_bench tests/index_tts2/index_tts2_warm_bench.cpp)
    add_engine_warmbench(irodori_tts_warm_bench tests/irodori_tts/irodori_tts_warm_bench.cpp)
    add_engine_warmbench(marblenet_vad_warm_bench tests/marblenet_vad/marblenet_vad_warm_bench.cpp)
    add_engine_warmbench(miocodec_warm_bench tests/miocodec/miocodec_warm_bench.cpp)
    add_engine_warmbench(muscriptor_warm_bench tests/muscriptor/muscriptor_warm_bench.cpp)
    add_engine_warmbench(moss_tts_nano_warm_bench tests/moss_tts_nano/moss_tts_nano_warm_bench.cpp)
    add_engine_warmbench(moss_tts_local_warm_bench tests/moss_tts_local/moss_tts_local_warm_bench.cpp)
    add_engine_warmbench(nemotron_asr_warm_bench tests/nemotron_asr/nemotron_asr_warm_bench.cpp)
    add_engine_warmbench(neutts_warm_bench tests/neutts/neutts_warm_bench.cpp)
    add_engine_warmbench(omnivoice_warm_bench tests/omnivoice/omnivoice_warm_bench.cpp)
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
    add_engine_warmbench(qwen3_tts_warm_bench tests/qwen3_tts/qwen3_tts_warm_bench.cpp)
    add_engine_warmbench(seed_vc_warm_bench tests/seed_vc/seed_vc_warm_bench.cpp)
    add_engine_warmbench(silero_vad_warm_bench tests/silero_vad/silero_vad_warm_bench.cpp)
    add_engine_warmbench(sortformer_diar_warm_bench tests/sortformer_diar/sortformer_diar_warm_bench.cpp)
    add_engine_warmbench(supertonic_warm_bench tests/supertonic/supertonic_warm_bench.cpp)
    add_engine_warmbench(vevo2_warm_bench tests/vevo2/vevo2_warm_bench.cpp)
    add_engine_warmbench(vibevoice_asr_warm_bench tests/vibevoice_asr/vibevoice_asr_warm_bench.cpp)
    add_engine_warmbench(voxtral_realtime_warm_bench tests/voxtral_realtime/voxtral_realtime_warm_bench.cpp)
    add_engine_warmbench(voxcpm2_warm_bench tests/voxcpm2/voxcpm2_warm_bench.cpp)
endif()

if (ENGINE_BUILD_TESTS)
    enable_testing()

    set(ENGINE_UNITTEST_ASSET_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests/assets")

    function(add_engine_unittest target source)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE engine_runtime ggml)
    endfunction()

    add_executable(sentencepiece_tokenizer1_test
        tests/unittests/test_sentencepiece_tokenizer1.cpp
    )

    target_link_libraries(sentencepiece_tokenizer1_test PRIVATE engine_runtime ggml)
    target_compile_definitions(sentencepiece_tokenizer1_test PRIVATE
        ENGINE_TEST_ASSET_ROOT="${ENGINE_UNITTEST_ASSET_ROOT}"
    )

    add_test(
        NAME sentencepiece_tokenizer1_test
        COMMAND sentencepiece_tokenizer1_test
    )

    add_engine_unittest(citrinet_tokenizer_decode_test tests/unittests/test_citrinet_tokenizer_decode.cpp)
    target_compile_definitions(citrinet_tokenizer_decode_test PRIVATE
        ENGINE_TEST_ASSET_ROOT="${ENGINE_UNITTEST_ASSET_ROOT}"
    )

    add_test(
        NAME citrinet_tokenizer_decode_test
        COMMAND citrinet_tokenizer_decode_test
    )

    add_engine_unittest(audio_dsp_test tests/unittests/test_audio_dsp.cpp)

    add_test(
        NAME audio_dsp_test
        COMMAND audio_dsp_test
    )

    add_engine_unittest(dots_tts_vocoder_parity tests/dots_tts/dots_tts_vocoder_parity.cpp)

    add_engine_unittest(midi_file_test tests/unittests/test_midi_file.cpp)

    add_test(
        NAME midi_file_test
        COMMAND midi_file_test
    )

    add_engine_unittest(wav_reader_test tests/unittests/test_wav_reader.cpp)

    add_test(
        NAME wav_reader_test
        COMMAND wav_reader_test
    )

    add_engine_unittest(wav_reader_chunk_bounds_test tests/unittests/test_wav_reader_chunk_bounds.cpp)

    add_test(
        NAME wav_reader_chunk_bounds_test
        COMMAND wav_reader_chunk_bounds_test
    )

    add_engine_unittest(hf_tokenizer_vocab_bounds_test tests/unittests/test_hf_tokenizer_vocab_bounds.cpp)

    add_test(
        NAME hf_tokenizer_vocab_bounds_test
        COMMAND hf_tokenizer_vocab_bounds_test
    )

    add_engine_unittest(safetensors_offsets_test tests/unittests/test_safetensors_offsets.cpp)

    add_test(
        NAME safetensors_offsets_test
        COMMAND safetensors_offsets_test
    )

    add_engine_unittest(audio_chunking_test tests/unittests/test_audio_chunking.cpp)

    add_test(
        NAME audio_chunking_test
        COMMAND audio_chunking_test
    )

    add_engine_unittest(inflect_v2_frontend_test tests/unittests/test_inflect_v2_frontend.cpp)
    target_include_directories(inflect_v2_frontend_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)

    add_test(
        NAME inflect_v2_frontend_test
        COMMAND inflect_v2_frontend_test
    )

    add_engine_unittest(voxtral_realtime_stream_chunking_test tests/unittests/test_voxtral_realtime_stream_chunking.cpp)
    target_include_directories(voxtral_realtime_stream_chunking_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)

    add_test(
        NAME voxtral_realtime_stream_chunking_test
        COMMAND voxtral_realtime_stream_chunking_test
    )

    add_engine_unittest(chinese_normalization_test tests/unittests/test_chinese_normalization.cpp)

    add_test(
        NAME chinese_normalization_test
        COMMAND chinese_normalization_test
    )

    add_engine_unittest(index_tts_tn_corpus_test tests/unittests/test_index_tts_tn_corpus.cpp)
    target_compile_definitions(index_tts_tn_corpus_test PRIVATE
        ENGINE_TEST_ASSET_ROOT="${ENGINE_UNITTEST_ASSET_ROOT}"
    )

    add_test(
        NAME index_tts_tn_corpus_test
        COMMAND index_tts_tn_corpus_test
    )

    add_engine_unittest(unicode_normalization_test tests/unittests/test_unicode_normalization.cpp)

    add_test(
        NAME unicode_normalization_test
        COMMAND unicode_normalization_test
    )

    add_engine_unittest(text_normalization_test tests/unittests/test_text_normalization.cpp)

    add_test(
        NAME text_normalization_test
        COMMAND text_normalization_test
    )

    add_engine_unittest(outetts_generation_budget_test tests/unittests/test_outetts_generation_budget.cpp)
    target_include_directories(outetts_generation_budget_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)

    add_test(
        NAME outetts_generation_budget_test
        COMMAND outetts_generation_budget_test
    )


    add_engine_unittest(subtitle_formatter_test tests/unittests/test_subtitle_formatter.cpp)

    add_test(
        NAME subtitle_formatter_test
        COMMAND subtitle_formatter_test
    )

    add_engine_unittest(rnnoise_utility_test tests/unittests/test_rnnoise_utility.cpp)
    target_compile_definitions(rnnoise_utility_test PRIVATE
        ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
        ENGINE_TEST_ASSET_ROOT="${ENGINE_UNITTEST_ASSET_ROOT}"
    )

    add_test(
        NAME rnnoise_utility_test
        COMMAND rnnoise_utility_test
    )

    add_engine_unittest(flashsr_utility_test tests/unittests/test_flashsr_utility.cpp)
    target_compile_definitions(flashsr_utility_test PRIVATE
        ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
        ENGINE_TEST_ASSET_ROOT="${ENGINE_UNITTEST_ASSET_ROOT}"
    )

    add_test(
        NAME flashsr_utility_test
        COMMAND flashsr_utility_test
    )

    add_engine_unittest(zipenhancer_utility_test tests/unittests/test_zipenhancer_utility.cpp)
    target_compile_definitions(zipenhancer_utility_test PRIVATE
        ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
        ENGINE_TEST_ASSET_ROOT="${ENGINE_UNITTEST_ASSET_ROOT}"
    )

    add_test(
        NAME zipenhancer_utility_test
        COMMAND zipenhancer_utility_test
    )

    add_engine_unittest(audio_utility_api_test tests/unittests/test_audio_utility_api.cpp)
    target_compile_definitions(audio_utility_api_test PRIVATE
        ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    )

    add_test(
        NAME audio_utility_api_test
        COMMAND audio_utility_api_test
    )

    add_engine_unittest(torch_random_test tests/unittests/test_torch_random.cpp)

    add_test(
        NAME torch_random_test
        COMMAND torch_random_test
    )

    add_engine_unittest(backend_device_resolution_test tests/unittests/test_backend_device_resolution.cpp)
    target_include_directories(backend_device_resolution_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)

    add_test(
        NAME backend_device_resolution_test
        COMMAND backend_device_resolution_test
    )

    add_executable(streaming_audio_input_test
        tests/unittests/test_streaming_audio_input.cpp
        app/streaming/pcm_source.cpp
        app/streaming/streaming.cpp
    )
    target_link_libraries(streaming_audio_input_test PRIVATE engine_runtime ggml)
    target_include_directories(streaming_audio_input_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)

    add_test(
        NAME streaming_audio_input_test
        COMMAND streaming_audio_input_test
    )

    add_executable(http_live_body_test
        tests/unittests/test_http_live_body.cpp
        app/server/http.cpp
    )
    target_link_libraries(http_live_body_test PRIVATE engine_runtime ggml)
    if(WIN32)
        target_link_libraries(http_live_body_test PRIVATE ws2_32)
    endif()
    target_include_directories(http_live_body_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)

    add_test(
        NAME http_live_body_test
        COMMAND http_live_body_test
    )
    # This test talks to a real loopback socket, so a framing regression could
    # block forever rather than fail. Bound it so CI reports a failure instead.
    set_tests_properties(http_live_body_test PROPERTIES TIMEOUT 120)

    add_engine_unittest(partial_text_render_test tests/unittests/test_partial_text_render.cpp)

    add_test(
        NAME partial_text_render_test
        COMMAND partial_text_render_test
    )

    add_engine_unittest(hf_sampler_test tests/unittests/test_hf_sampler.cpp)

    add_test(
        NAME hf_sampler_test
        COMMAND hf_sampler_test
    )

    add_engine_unittest(diffusion_math_test tests/unittests/test_diffusion_math.cpp)

    add_test(
        NAME diffusion_math_test
        COMMAND diffusion_math_test
    )

    add_engine_unittest(encoder_module_test tests/unittests/test_encoder_modules.cpp)

    add_test(
        NAME encoder_module_test
        COMMAND encoder_module_test
    )

    add_engine_unittest(
        qwen_decoder_packed_projection_test
        tests/unittests/test_qwen_decoder_packed_projections.cpp
    )

    add_test(
        NAME qwen_decoder_packed_projection_test
        COMMAND qwen_decoder_packed_projection_test
    )

    add_engine_unittest(conv_transpose_fast_path_test tests/unittests/test_conv_transpose_fast_path.cpp)

    add_test(
        NAME conv_transpose_fast_path_test
        COMMAND conv_transpose_fast_path_test
    )

    add_engine_unittest(supertonic_vector_convnext_exp_test tests/unittests/test_supertonic_vector_convnext_exp.cpp)

    add_test(
        NAME supertonic_vector_convnext_exp_test
        COMMAND supertonic_vector_convnext_exp_test
    )

    add_engine_unittest(depthwise_conv1d_lowering_test tests/unittests/test_depthwise_conv1d_lowering.cpp)

    add_test(
        NAME depthwise_conv1d_lowering_test
        COMMAND depthwise_conv1d_lowering_test
    )

    add_engine_unittest(conv_lowering_matrix_test tests/unittests/test_conv_lowering_matrix.cpp)
    if (ENGINE_ENABLE_HIP)
        target_compile_definitions(conv_lowering_matrix_test PRIVATE ENGINE_TEST_ENABLE_HIP=1)
    endif()

    add_test(
        NAME conv_lowering_matrix_test
        COMMAND conv_lowering_matrix_test
    )

    add_engine_unittest(gguf_tensor_source_test tests/unittests/test_gguf_tensor_source.cpp)
    target_include_directories(gguf_tensor_source_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)

    add_test(
        NAME gguf_tensor_source_test
        COMMAND gguf_tensor_source_test
    )

    add_engine_unittest(model_spec_system_test tests/unittests/test_model_spec_system.cpp)
    target_include_directories(model_spec_system_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)

    add_test(
        NAME model_spec_system_test
        COMMAND model_spec_system_test
    )

    add_engine_unittest(qwen3_forced_aligner_processor_test tests/unittests/test_qwen3_forced_aligner_processor.cpp)
    target_include_directories(qwen3_forced_aligner_processor_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)

    add_test(
        NAME qwen3_forced_aligner_processor_test
        COMMAND qwen3_forced_aligner_processor_test
    )

    add_engine_unittest(fun_asr_nano_assets_test tests/unittests/test_fun_asr_nano_assets.cpp)
    target_sources(fun_asr_nano_assets_test PRIVATE src/models/fun_asr_nano/assets.cpp)
    target_include_directories(fun_asr_nano_assets_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)

    add_test(
        NAME fun_asr_nano_assets_test
        COMMAND fun_asr_nano_assets_test
    )

    add_engine_unittest(fun_asr_nano_frontend_probe tests/fun_asr_nano/fun_asr_nano_frontend_probe.cpp)

    add_test(
        NAME fun_asr_nano_frontend_probe
        COMMAND fun_asr_nano_frontend_probe
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/fun_asr_nano/frontend_reference.json
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/fun_asr_nano/frontend_reference.bin
            ${CMAKE_CURRENT_SOURCE_DIR}/assets/resources/sample_16k.wav
    )

    add_engine_unittest(fun_asr_nano_sanm_probe tests/fun_asr_nano/sanm_probe.cpp)

    add_test(
        NAME fun_asr_nano_sanm_probe
        COMMAND fun_asr_nano_sanm_probe
            --backend cpu
            --reference ${CMAKE_CURRENT_SOURCE_DIR}/tests/fun_asr_nano/sanm_reference.json
            --data ${CMAKE_CURRENT_SOURCE_DIR}/tests/fun_asr_nano/sanm_reference.bin
    )

    add_engine_unittest(fun_asr_nano_encoder_probe tests/fun_asr_nano/fun_asr_nano_encoder_probe.cpp)
    target_sources(
        fun_asr_nano_encoder_probe
        PRIVATE
            src/models/fun_asr_nano/assets.cpp
            src/models/fun_asr_nano/frontend.cpp
            src/models/fun_asr_nano/encoder.cpp
    )
    set(
        AUDIOCPP_FUN_ASR_NANO_TEST_MODEL
        ""
        CACHE PATH
        "Optional local Fun-ASR-Nano model directory for full runtime parity tests"
    )
    set(
        AUDIOCPP_FUN_ASR_NANO_TEST_GGUF
        ""
        CACHE FILEPATH
        "Optional standalone Fun-ASR-Nano GGUF for full CLI transcript tests"
    )
    if (AUDIOCPP_FUN_ASR_NANO_TEST_MODEL)
        add_test(
            NAME fun_asr_nano_encoder_probe_cpu
            COMMAND fun_asr_nano_encoder_probe
                --backend cpu
                --model ${AUDIOCPP_FUN_ASR_NANO_TEST_MODEL}
                --reference ${CMAKE_CURRENT_SOURCE_DIR}/tests/fun_asr_nano/encoder_reference.json
                --data ${CMAKE_CURRENT_SOURCE_DIR}/tests/fun_asr_nano/encoder_reference.bin
        )
        if (GGML_CUDA)
            add_test(
                NAME fun_asr_nano_encoder_probe_cuda
                COMMAND fun_asr_nano_encoder_probe
                    --backend cuda
                    --model ${AUDIOCPP_FUN_ASR_NANO_TEST_MODEL}
                    --reference ${CMAKE_CURRENT_SOURCE_DIR}/tests/fun_asr_nano/encoder_reference.json
                    --data ${CMAKE_CURRENT_SOURCE_DIR}/tests/fun_asr_nano/encoder_reference.bin
            )
        endif()

        add_engine_unittest(fun_asr_nano_session_probe tests/fun_asr_nano/fun_asr_nano_session_probe.cpp)
        add_test(
            NAME fun_asr_nano_session_probe_cpu
            COMMAND fun_asr_nano_session_probe
                --backend cpu
                --model ${AUDIOCPP_FUN_ASR_NANO_TEST_MODEL}
                --audio ${CMAKE_CURRENT_SOURCE_DIR}/assets/resources/sample_16k.wav
        )
        if (GGML_CUDA)
            add_test(
                NAME fun_asr_nano_session_probe_cuda
                COMMAND fun_asr_nano_session_probe
                    --backend cuda
                    --model ${AUDIOCPP_FUN_ASR_NANO_TEST_MODEL}
                    --audio ${CMAKE_CURRENT_SOURCE_DIR}/assets/resources/sample_16k.wav
            )
        endif()
    endif()
    if (AUDIOCPP_FUN_ASR_NANO_TEST_GGUF)
        add_test(
            NAME fun_asr_nano_gguf_cli_cpu
            COMMAND audiocpp_cli
                --task asr
                --family fun_asr_nano
                --model ${AUDIOCPP_FUN_ASR_NANO_TEST_GGUF}
                --backend cpu
                --audio ${CMAKE_CURRENT_SOURCE_DIR}/assets/resources/sample_16k.wav
        )
        set_tests_properties(
            fun_asr_nano_gguf_cli_cpu
            PROPERTIES PASS_REGULAR_EXPRESSION
                "text_output=Some call me nature, others call me mother nature"
        )
        if (GGML_CUDA)
            add_test(
                NAME fun_asr_nano_gguf_cli_cuda
                COMMAND audiocpp_cli
                    --task asr
                    --family fun_asr_nano
                    --model ${AUDIOCPP_FUN_ASR_NANO_TEST_GGUF}
                    --backend cuda
                    --audio ${CMAKE_CURRENT_SOURCE_DIR}/assets/resources/sample_16k.wav
            )
            set_tests_properties(
                fun_asr_nano_gguf_cli_cuda
                PROPERTIES PASS_REGULAR_EXPRESSION
                    "text_output=Some call me nature, others call me mother nature"
            )
            add_test(
                NAME fun_asr_nano_gguf_cli_cuda_shared_q8
                COMMAND audiocpp_cli
                    --task asr
                    --family fun_asr_nano
                    --model ${AUDIOCPP_FUN_ASR_NANO_TEST_GGUF}
                    --backend cuda
                    --session-option fun_asr_nano.weight_type=q8_0
                    --audio ${CMAKE_CURRENT_SOURCE_DIR}/assets/resources/sample_16k.wav
            )
            set_tests_properties(
                fun_asr_nano_gguf_cli_cuda_shared_q8
                PROPERTIES PASS_REGULAR_EXPRESSION
                    "text_output=Some call me nature, others call me mother nature"
            )
            add_test(
                NAME fun_asr_nano_gguf_cli_best_shared_q8
                COMMAND audiocpp_cli
                    --task asr
                    --family fun_asr_nano
                    --model ${AUDIOCPP_FUN_ASR_NANO_TEST_GGUF}
                    --backend best
                    --session-option fun_asr_nano.weight_type=q8_0
                    --audio ${CMAKE_CURRENT_SOURCE_DIR}/assets/resources/sample_16k.wav
            )
            set_tests_properties(
                fun_asr_nano_gguf_cli_best_shared_q8
                PROPERTIES PASS_REGULAR_EXPRESSION
                    "text_output=Some call me nature, others call me mother nature"
            )
        endif()
    endif()

    add_engine_unittest(fun_asr_nano_adaptor_probe tests/fun_asr_nano/fun_asr_nano_adaptor_probe.cpp)
    target_sources(
        fun_asr_nano_adaptor_probe
        PRIVATE
            src/models/fun_asr_nano/assets.cpp
            src/models/fun_asr_nano/adaptor.cpp
    )
    if (AUDIOCPP_FUN_ASR_NANO_TEST_MODEL)
        add_test(
            NAME fun_asr_nano_adaptor_probe_cpu
            COMMAND fun_asr_nano_adaptor_probe
                --backend cpu
                --model ${AUDIOCPP_FUN_ASR_NANO_TEST_MODEL}
                --reference ${CMAKE_CURRENT_SOURCE_DIR}/tests/fun_asr_nano/adaptor_reference.json
                --data ${CMAKE_CURRENT_SOURCE_DIR}/tests/fun_asr_nano/adaptor_reference.bin
        )
        if (GGML_CUDA)
            add_test(
                NAME fun_asr_nano_adaptor_probe_cuda
                COMMAND fun_asr_nano_adaptor_probe
                    --backend cuda
                    --model ${AUDIOCPP_FUN_ASR_NANO_TEST_MODEL}
                    --reference ${CMAKE_CURRENT_SOURCE_DIR}/tests/fun_asr_nano/adaptor_reference.json
                    --data ${CMAKE_CURRENT_SOURCE_DIR}/tests/fun_asr_nano/adaptor_reference.bin
            )
        endif()
    endif()

    add_engine_unittest(fun_asr_nano_decoder_probe tests/fun_asr_nano/fun_asr_nano_decoder_probe.cpp)
    target_sources(
        fun_asr_nano_decoder_probe
        PRIVATE
            src/models/fun_asr_nano/assets.cpp
            src/models/fun_asr_nano/tokenizer_text.cpp
            src/models/fun_asr_nano/prompt.cpp
            src/models/fun_asr_nano/decoder.cpp
    )
    if (AUDIOCPP_FUN_ASR_NANO_TEST_MODEL)
        add_test(
            NAME fun_asr_nano_decoder_probe_cpu
            COMMAND fun_asr_nano_decoder_probe
                --backend cpu
                --model ${AUDIOCPP_FUN_ASR_NANO_TEST_MODEL}
                --reference ${CMAKE_CURRENT_SOURCE_DIR}/tests/fun_asr_nano/decoder_reference.json
                --data ${CMAKE_CURRENT_SOURCE_DIR}/tests/fun_asr_nano/decoder_reference.bin
        )
        if (GGML_CUDA)
            add_test(
                NAME fun_asr_nano_decoder_probe_cuda
                COMMAND fun_asr_nano_decoder_probe
                    --backend cuda
                    --model ${AUDIOCPP_FUN_ASR_NANO_TEST_MODEL}
                    --reference ${CMAKE_CURRENT_SOURCE_DIR}/tests/fun_asr_nano/decoder_reference.json
                    --data ${CMAKE_CURRENT_SOURCE_DIR}/tests/fun_asr_nano/decoder_reference.bin
            )
        endif()
    endif()

    add_engine_unittest(tdt_decoder_duration_loop_test tests/unittests/test_tdt_decoder_duration_loop.cpp)
    target_include_directories(tdt_decoder_duration_loop_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)

    add_test(
        NAME tdt_decoder_duration_loop_test
        COMMAND tdt_decoder_duration_loop_test
    )

    add_engine_unittest(scaled_dot_product_attention_test tests/unittests/test_scaled_dot_product_attention.cpp)

    add_test(
        NAME scaled_dot_product_attention_test
        COMMAND scaled_dot_product_attention_test
    )

    add_engine_unittest(asr_standalone_gguf_test tests/unittests/test_asr_standalone_gguf.cpp)
    target_include_directories(asr_standalone_gguf_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    target_compile_definitions(asr_standalone_gguf_test PRIVATE
        ENGINE_TEST_ASSET_ROOT="${ENGINE_UNITTEST_ASSET_ROOT}"
    )

    add_test(
        NAME asr_standalone_gguf_test
        COMMAND asr_standalone_gguf_test
    )

    add_executable(cli_request_options_test
        tests/unittests/test_cli_request_options.cpp
        app/cli/args.cpp
        app/cli/request.cpp
    )
    target_link_libraries(cli_request_options_test PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(cli_request_options_test PRIVATE OpenMP::OpenMP_CXX)
    endif()

    add_test(
        NAME cli_request_options_test
        COMMAND cli_request_options_test
    )

    add_executable(server_multipart_test
        tests/unittests/test_server_multipart.cpp
        app/server/multipart.cpp
    )
    target_include_directories(server_multipart_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/app/server)

    add_test(
        NAME server_multipart_test
        COMMAND server_multipart_test
    )

    add_executable(server_base64_test
        tests/unittests/test_server_base64.cpp
        app/server/base64.cpp
    )
    target_include_directories(server_base64_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/app/server)

    add_test(
        NAME server_base64_test
        COMMAND server_base64_test
    )

    add_executable(server_busy_guard_test
        tests/unittests/test_server_busy_guard.cpp
    )
    target_include_directories(server_busy_guard_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/app/server)
    find_package(Threads REQUIRED)
    target_link_libraries(server_busy_guard_test PRIVATE Threads::Threads)

    add_test(
        NAME server_busy_guard_test
        COMMAND server_busy_guard_test
    )
    # The suite deliberately drives threads against a held lock; bound it so a
    # regression that reintroduces unbounded queuing fails instead of hanging CI.
    set_tests_properties(server_busy_guard_test PROPERTIES TIMEOUT 60)

    add_executable(server_config_test
        tests/unittests/test_server_config.cpp
        app/server/config.cpp
        app/cli/args.cpp
        app/cli/request.cpp
    )
    target_include_directories(server_config_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/app/server)
    target_link_libraries(server_config_test PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(server_config_test PRIVATE OpenMP::OpenMP_CXX)
    endif()

    add_test(
        NAME server_config_test
        COMMAND server_config_test
    )

    add_engine_unittest(parakeet_golden_transcription_test tests/parakeet_tdt/test_golden_transcription.cpp)
    target_compile_definitions(parakeet_golden_transcription_test PRIVATE
        ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    )

    add_test(
        NAME parakeet_golden_transcription_test
        COMMAND parakeet_golden_transcription_test
    )
    # Requires the real (multi-GB) Parakeet-TDT weights, which aren't
    # downloaded as part of a normal checkout/CI build; the test itself
    # detects this and exits with kExitSkip (125) rather than failing.
    set_tests_properties(parakeet_golden_transcription_test PROPERTIES
        SKIP_RETURN_CODE 125
        TIMEOUT 120
    )

    add_engine_unittest(parakeet_streaming_transcription_test tests/parakeet_tdt/test_streaming_transcription.cpp)
    target_compile_definitions(parakeet_streaming_transcription_test PRIVATE
        ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    )

    add_test(
        NAME parakeet_streaming_transcription_test
        COMMAND parakeet_streaming_transcription_test
    )
    set_tests_properties(parakeet_streaming_transcription_test PROPERTIES
        SKIP_RETURN_CODE 125
        TIMEOUT 180
    )

    # Numerical parity dump tool (not registered as an add_test — it's a
    # manual/pre-release check, not a CI gate). Run it against a NeMo
    # reference dump via compare_parity.py; see tests/parakeet_tdt/parity/README.md.
    add_executable(parakeet_parity_dump
        tests/parakeet_tdt/parity/dump_cpp_reference.cpp
    )
    target_include_directories(parakeet_parity_dump PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/parakeet_tdt/parity)
    target_link_libraries(parakeet_parity_dump PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(parakeet_parity_dump PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if (AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER)
        add_executable(server_model_installer_test
            tests/unittests/test_server_model_installer.cpp
            app/server/model_installer.cpp
        )
        target_include_directories(server_model_installer_test PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/app/server
            ${CMAKE_CURRENT_BINARY_DIR}/generated)
        target_link_libraries(server_model_installer_test PRIVATE
            Threads::Threads engine_runtime audiocpp_package_manager)
        target_compile_definitions(server_model_installer_test PRIVATE
            AUDIOCPP_NATIVE_MANAGER_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/native_model_manager_server.py")

        add_test(
            NAME server_model_installer_test
            COMMAND server_model_installer_test
        )
    endif()
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
