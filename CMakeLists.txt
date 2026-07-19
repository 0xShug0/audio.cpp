cmake_minimum_required(VERSION 3.20)

project(AudioCpp LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(AUDIOCPP_DEPLOYMENT_BUILD
    "Compile model package specs into the runtime for self-contained deployments"
    OFF)

# Package layouts remain editable runtime data in normal builds. The converter always
# carries the source catalog so the conversion binary is portable. Deployment builds
# additionally compile the catalog into engine_runtime as a fallback for CLI/server.
file(GLOB AUDIOCPP_MODEL_SPEC_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/model_specs/*.json")
set(AUDIOCPP_BUILTIN_MODEL_SPEC_ENTRIES "")
set(AUDIOCPP_CONVERTER_MODEL_SPEC_ENTRIES "")
foreach(AUDIOCPP_MODEL_SPEC_FILE IN LISTS AUDIOCPP_MODEL_SPEC_FILES)
    get_filename_component(AUDIOCPP_MODEL_SPEC_FAMILY "${AUDIOCPP_MODEL_SPEC_FILE}" NAME_WE)
    file(READ "${AUDIOCPP_MODEL_SPEC_FILE}" AUDIOCPP_MODEL_SPEC_JSON)
    string(APPEND AUDIOCPP_CONVERTER_MODEL_SPEC_ENTRIES
        "    {\"${AUDIOCPP_MODEL_SPEC_FAMILY}\", R\"AUDIOCPP_SPEC(${AUDIOCPP_MODEL_SPEC_JSON})AUDIOCPP_SPEC\"},\n")
    if (AUDIOCPP_DEPLOYMENT_BUILD)
        string(APPEND AUDIOCPP_BUILTIN_MODEL_SPEC_ENTRIES
            "    {\"${AUDIOCPP_MODEL_SPEC_FAMILY}\", R\"AUDIOCPP_SPEC(${AUDIOCPP_MODEL_SPEC_JSON})AUDIOCPP_SPEC\"},\n")
    endif()
endforeach()
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/model_package_specs.inc"
    CONTENT "${AUDIOCPP_BUILTIN_MODEL_SPEC_ENTRIES}")
file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/gguf_converter_model_specs.inc"
    CONTENT "${AUDIOCPP_CONVERTER_MODEL_SPEC_ENTRIES}")

if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

if (NOT MSVC)
    set(AUDIOCPP_DEBUG_OPT_FLAGS "-O3 -g")
    set(CMAKE_C_FLAGS_DEBUG "${AUDIOCPP_DEBUG_OPT_FLAGS}" CACHE STRING "Optimized debug C flags" FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "${AUDIOCPP_DEBUG_OPT_FLAGS}" CACHE STRING "Optimized debug C++ flags" FORCE)
endif()

set(ENGINE_DEFAULT_ENABLE_CUDA OFF)
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
option(ENGINE_ENABLE_VULKAN "Build ggml with Vulkan backend support" ${ENGINE_DEFAULT_ENABLE_VULKAN})
option(ENGINE_ENABLE_METAL "Build ggml with Metal backend support" ${ENGINE_DEFAULT_ENABLE_METAL})
option(ENGINE_ENABLE_LLAMAFILE "Build ggml with llamafile SGEMM support" ${ENGINE_DEFAULT_ENABLE_LLAMAFILE})
option(ENGINE_ENABLE_CUDA_GRAPHS "Enable ggml CUDA graphs support" ${ENGINE_DEFAULT_ENABLE_CUDA_GRAPHS})
option(ENGINE_ENABLE_NATIVE_CPU "Build ggml CPU kernels with native host ISA flags" ${ENGINE_DEFAULT_ENABLE_NATIVE_CPU})
option(ENGINE_ENABLE_OPENMP "Build host code with OpenMP support" ON)
option(ENGINE_BUILD_EXAMPLES "Build example binaries" OFF)
option(ENGINE_BUILD_TESTS "Build framework unit tests" OFF)
option(ENGINE_BUILD_WARMBENCH "Build warmbench helper binaries" OFF)
set(AUDIOCPP_GGML_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/external/ggml"
    CACHE PATH "ggml source tree to build with AudioCPP")

if (ENGINE_ENABLE_CUDA)
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

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "Do not build upstream examples" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "Do not build upstream tests" FORCE)
set(GGML_ALL_WARNINGS OFF CACHE BOOL "Keep upstream warnings quiet in this sample" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "Do not probe for ccache in this local setup" FORCE)
set(GGML_CUDA ${ENGINE_ENABLE_CUDA} CACHE BOOL "Build ggml with CUDA backend support" FORCE)
set(GGML_VULKAN ${ENGINE_ENABLE_VULKAN} CACHE BOOL "Build ggml with Vulkan backend support" FORCE)
set(GGML_METAL ${ENGINE_ENABLE_METAL} CACHE BOOL "Build ggml with Metal backend support" FORCE)
set(GGML_LLAMAFILE ${ENGINE_ENABLE_LLAMAFILE} CACHE BOOL "Build ggml with llamafile SGEMM support" FORCE)
set(GGML_CUDA_GRAPHS ${ENGINE_ENABLE_CUDA_GRAPHS} CACHE BOOL "Enable ggml CUDA graphs support" FORCE)
set(GGML_NATIVE ${ENGINE_ENABLE_NATIVE_CPU} CACHE BOOL "Build ggml CPU kernels with native host ISA flags" FORCE)

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

add_library(engine_runtime STATIC
    src/framework/assets/resource_bundle.cpp
    src/framework/assets/model_package.cpp
    src/framework/assets/tensor_source.cpp
    src/framework/assets/torch_bin.cpp
    src/framework/core/module.cpp
    src/framework/core/backend.cpp
    src/framework/core/deferred_tensor_writer.cpp
    src/framework/core/execution_context.cpp
    src/framework/core/host_memory.cpp
    src/framework/runtime/registry.cpp
    src/framework/runtime/model.cpp
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
    src/framework/io/filesystem.cpp
    src/framework/io/config.cpp
    src/framework/io/json.cpp
    src/framework/io/text.cpp
    src/framework/text/chunking.cpp
    src/framework/text/subtitle.cpp
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
    src/framework/modules/primitive_modules.cpp
    src/framework/modules/activation_modules.cpp
    src/framework/modules/norm_modules.cpp
    src/framework/modules/lookup_modules.cpp
    src/framework/modules/structural_modules.cpp
    src/framework/modules/asr_helpers.cpp
    src/framework/modules/conv_modules.cpp
    src/framework/modules/whisper_embedding.cpp
    src/framework/modules/campplus_encoder.cpp
    src/framework/modules/bigvgan_vocoder.cpp
    src/framework/modules/hift_vocoder.cpp
    src/framework/modules/hubert_encoder.cpp
    src/framework/modules/wavlm_encoder.cpp
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
    src/framework/modules/attention/streaming_frame_attention_cache.cpp
    src/framework/modules/attention/qwen_decoder.cpp
    src/framework/modules/attention/qwen_causal_decoder.cpp
    src/framework/modules/attention/cross_attention.cpp
    src/framework/modules/attention/transformer_blocks.cpp
    src/framework/modules/conditioning_modules.cpp
    src/framework/modules/conformer_modules.cpp
    src/framework/modules/predictor_modules.cpp
    src/framework/modules/positional_modules.cpp
    src/framework/modules/text_encoders/t5_base_encoder.cpp
    src/framework/modules/text_encoders/t5_gemma_encoder.cpp
    src/framework/modules/optimizations/fast_kv_modules.cpp
    src/framework/tokenizers/hf_tokenizer_json.cpp
    src/framework/tokenizers/llama_bpe.cpp
    external/llama_tokenizer/bpe-core.cpp
    external/llama_tokenizer/unicode.cpp
    external/llama_tokenizer/unicode-data.cpp
    src/framework/tokenizers/sentencepiece.cpp
    src/framework/audio/fft.cpp
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
    src/framework/modules/ecapa_tdnn_runtime.cpp
    src/framework/modules/titanet_runtime.cpp
    src/framework/codecs/neural_audio.cpp
    src/framework/sampling/decode_modules.cpp
    src/framework/sampling/diffusion_math.cpp
    src/framework/sampling/noise.cpp
    src/framework/sampling/torch_random.cpp
    src/models/roformer/assets.cpp
    src/models/roformer/loader.cpp
    src/models/roformer/runtime.cpp
    src/models/roformer/session.cpp
    src/models/demucs/assets.cpp
    src/models/demucs/frontend.cpp
    src/models/demucs/pipeline.cpp
    src/models/demucs/postprocess.cpp
    src/models/demucs/loader.cpp
    src/models/demucs/session.cpp
    src/models/omnivoice/assets.cpp
    src/models/omnivoice/tokenizer_text.cpp
    src/models/omnivoice/audio_tokenizer.cpp
    src/models/omnivoice/prompt_builder.cpp
    src/models/omnivoice/generator.cpp
    src/models/omnivoice/postprocess.cpp
    src/models/omnivoice/session.cpp
    src/models/omnivoice/loader.cpp
    src/models/pocket_tts/assets.cpp
    src/models/pocket_tts/acoustic_model.cpp
    src/models/pocket_tts/audio_decoder.cpp
    src/models/pocket_tts/flow_lm.cpp
    src/models/pocket_tts/mimi_decoder.cpp
    src/models/pocket_tts/mimi_encoder.cpp
    src/models/pocket_tts/text_conditioner.cpp
    src/models/pocket_tts/voice_conditioner.cpp
    src/models/pocket_tts/loader.cpp
    src/models/pocket_tts/session.cpp
    src/models/miocodec/assets.cpp
    src/models/miocodec/weights.cpp
    src/models/miocodec/audio_pipeline.cpp
    src/models/miocodec/graph_ops.cpp
    src/models/miocodec/graphs.cpp
    src/models/miocodec/session.cpp
    src/models/miocodec/loader.cpp
    src/models/miotts/assets.cpp
    src/models/miotts/tokenizer.cpp
    src/models/miotts/causal_lm.cpp
    src/models/miotts/session.cpp
    src/models/miotts/loader.cpp
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
    src/models/voxcpm2/assets.cpp
    src/models/voxcpm2/audiovae.cpp
    src/models/voxcpm2/generator.cpp
    src/models/voxcpm2/loader.cpp
    src/models/voxcpm2/minicpm.cpp
    src/models/voxcpm2/session.cpp
    src/models/voxcpm2/tokenizer_text.cpp
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
    src/models/heartmula/assets.cpp
    src/models/heartmula/codec.cpp
    src/models/heartmula/generator.cpp
    src/models/heartmula/loader.cpp
    src/models/heartmula/mula.cpp
    src/models/heartmula/session.cpp
    src/models/heartmula/tokenizer_text.cpp
    src/models/hviske_asr/assets.cpp
    src/models/hviske_asr/frontend.cpp
    src/models/hviske_asr/encoder.cpp
    src/models/hviske_asr/decoder.cpp
    src/models/hviske_asr/session.cpp
    src/models/hviske_asr/loader.cpp
    src/models/hviske_asr/weights.cpp
    src/models/higgs_audio_stt/assets.cpp
    src/models/higgs_audio_stt/tokenizer_text.cpp
    src/models/higgs_audio_stt/frontend_whisper.cpp
    src/models/higgs_audio_stt/audio_encoder.cpp
    src/models/higgs_audio_stt/text_decoder.cpp
    src/models/higgs_audio_stt/prompt_asr.cpp
    src/models/higgs_audio_stt/postprocess.cpp
    src/models/higgs_audio_stt/session.cpp
    src/models/higgs_audio_stt/loader.cpp
    src/models/higgs_tts/ar.cpp
    src/models/higgs_tts/assets.cpp
    src/models/higgs_tts/codec.cpp
    src/models/higgs_tts/codebooks.cpp
    src/models/higgs_tts/generator.cpp
    src/models/higgs_tts/loader.cpp
    src/models/higgs_tts/sampler.cpp
    src/models/higgs_tts/session.cpp
    src/models/higgs_tts/tokenizer_text.cpp
    src/models/irodori_tts/assets.cpp
    src/models/irodori_tts/codec.cpp
    src/models/irodori_tts/condition_encoder.cpp
    src/models/irodori_tts/duration.cpp
    src/models/irodori_tts/loader.cpp
    src/models/irodori_tts/rf_dit.cpp
    src/models/irodori_tts/session.cpp
    src/models/irodori_tts/tokenizer_text.cpp
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
    src/models/nemotron_asr/assets.cpp
    src/models/nemotron_asr/frontend.cpp
    src/models/nemotron_asr/encoder.cpp
    src/models/nemotron_asr/decoder.cpp
    src/models/nemotron_asr/session.cpp
    src/models/nemotron_asr/loader.cpp
    src/models/nemotron_asr/weights.cpp
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
    src/models/qwen3_asr/assets.cpp
    src/models/qwen3_asr/tokenizer_text.cpp
    src/models/qwen3_asr/frontend_whisper.cpp
    src/models/qwen3_asr/audio_encoder.cpp
    src/models/qwen3_asr/thinker.cpp
    src/models/qwen3_asr/prompt_asr.cpp
    src/models/qwen3_asr/postprocess.cpp
    src/models/qwen3_asr/session.cpp
    src/models/qwen3_asr/loader.cpp
    src/models/vevo2/ar.cpp
    src/models/vevo2/assets.cpp
    src/models/vevo2/components.cpp
    src/models/vevo2/fm.cpp
    src/models/vevo2/prompt_builder.cpp
    src/models/vevo2/session.cpp
    src/models/vevo2/tokenizer_ar.cpp
    src/models/vevo2/vocoder.cpp
    src/models/vevo2/loader.cpp
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
    src/models/seed_vc/loader.cpp
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
    src/models/qwen3_forced_aligner/processor.cpp
    src/models/qwen3_forced_aligner/session.cpp
    src/models/qwen3_forced_aligner/loader.cpp
    src/models/ace_step/assets.cpp
    src/models/ace_step/audio_preprocess.cpp
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
    src/models/supertonic/assets.cpp
    src/models/supertonic/loader.cpp
    src/models/supertonic/runtime.cpp
    src/models/supertonic/session.cpp
    src/models/supertonic/tokenizer_text.cpp
    src/models/sortformer_diar/assets.cpp
    src/models/sortformer_diar/frontend.cpp
    src/models/sortformer_diar/graph.cpp
    src/models/sortformer_diar/modules.cpp
    src/models/sortformer_diar/postprocess.cpp
    src/models/sortformer_diar/loader.cpp
    src/models/sortformer_diar/session.cpp
    src/models/silero_vad/assets.cpp
    src/models/silero_vad/runtime.cpp
    src/models/silero_vad/session.cpp
    src/models/citrinet_asr/assets.cpp
    src/models/citrinet_asr/runtime.cpp
    src/models/citrinet_asr/session.cpp
    src/models/marblenet_vad/assets.cpp
    src/models/marblenet_vad/runtime.cpp
    src/models/marblenet_vad/session.cpp
)

add_library(cjson_vendor STATIC
    external/cJSON/cJSON.c
)

target_include_directories(cjson_vendor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/external/cJSON
)

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

target_compile_definitions(yaml_vendor PUBLIC
    _POSIX_C_SOURCE=200809L
    YAML_DECLARE_STATIC=1
    YAML_VERSION_MAJOR=0
    YAML_VERSION_MINOR=2
    YAML_VERSION_PATCH=5
    YAML_VERSION_STRING=\"0.2.5\"
)

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
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(engine_runtime PRIVATE OpenMP::OpenMP_CXX)
endif()

if (ENGINE_ENABLE_CUDA)
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
    target_compile_definitions(engine_runtime PUBLIC ENGINE_HAS_CUDA_ISTFT=1 ENGINE_HAS_CUDA_TORCH_RANDOM=1)
    target_link_libraries(engine_runtime PRIVATE CUDA::cudart CUDA::cufft)
endif()

add_executable(audiocpp_cli
    app/cli/main.cpp
    app/cli/args.cpp
    app/cli/batch.cpp
    app/cli/request.cpp
    app/streaming/streaming.cpp
    app/workflow/execution.cpp
    app/workflow/file_sink.cpp
    app/workflow/pipeline.cpp
    app/workflow/workflow.cpp
)

target_link_libraries(audiocpp_cli PRIVATE engine_runtime ggml)
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(audiocpp_cli PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(audiocpp_server
    app/server/main.cpp
    app/server/config.cpp
    app/server/http.cpp
    app/server/multipart.cpp
    app/server/runtime.cpp
    app/cli/args.cpp
    app/cli/request.cpp
    app/streaming/streaming.cpp
)

target_link_libraries(audiocpp_server PRIVATE engine_runtime ggml)
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

add_executable(vibevoice_finetune_overlay_check
    tests/vibevoice/finetune_overlay_check.cpp
)

target_link_libraries(vibevoice_finetune_overlay_check PRIVATE engine_runtime ggml)
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(vibevoice_finetune_overlay_check PRIVATE OpenMP::OpenMP_CXX)
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

    add_engine_warmbench(chatterbox_warm_bench tests/chatterbox/chatterbox_warm_bench.cpp)
    add_engine_warmbench(citrinet_asr_warm_bench tests/citrinet_asr/citrinet_asr_warm_bench.cpp)
    add_engine_warmbench(higgs_audio_stt_warm_bench tests/higgs_audio_stt/higgs_audio_stt_warm_bench.cpp)
    add_engine_warmbench(higgs_tts_warm_bench tests/higgs_tts/higgs_tts_warm_bench.cpp)
    add_engine_warmbench(hviske_asr_warm_bench tests/hviske_asr/hviske_asr_warm_bench.cpp)
    add_engine_warmbench(index_tts2_warm_bench tests/index_tts2/index_tts2_warm_bench.cpp)
    add_engine_warmbench(irodori_tts_warm_bench tests/irodori_tts/irodori_tts_warm_bench.cpp)
    add_engine_warmbench(marblenet_vad_warm_bench tests/marblenet_vad/marblenet_vad_warm_bench.cpp)
    add_engine_warmbench(miocodec_warm_bench tests/miocodec/miocodec_warm_bench.cpp)
    add_engine_warmbench(moss_tts_nano_warm_bench tests/moss_tts_nano/moss_tts_nano_warm_bench.cpp)
    add_engine_warmbench(moss_tts_local_warm_bench tests/moss_tts_local/moss_tts_local_warm_bench.cpp)
    add_engine_warmbench(nemotron_asr_warm_bench tests/nemotron_asr/nemotron_asr_warm_bench.cpp)
    add_engine_warmbench(omnivoice_warm_bench tests/omnivoice/omnivoice_warm_bench.cpp)
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

    add_engine_unittest(wav_reader_test tests/unittests/test_wav_reader.cpp)

    add_test(
        NAME wav_reader_test
        COMMAND wav_reader_test
    )

    add_engine_unittest(audio_chunking_test tests/unittests/test_audio_chunking.cpp)

    add_test(
        NAME audio_chunking_test
        COMMAND audio_chunking_test
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

    add_executable(server_multipart_test
        tests/unittests/test_server_multipart.cpp
        app/server/multipart.cpp
    )
    target_include_directories(server_multipart_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/app/server)

    add_test(
        NAME server_multipart_test
        COMMAND server_multipart_test
    )

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
endif()

if (ENGINE_BUILD_EXAMPLES)
    add_executable(ggml_simple_inference
        examples/ggml_simple_inference.cpp
    )

    target_link_libraries(ggml_simple_inference PRIVATE ggml)
endif()
