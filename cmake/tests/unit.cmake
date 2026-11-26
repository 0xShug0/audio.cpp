# Self-contained unit/component tests and existing application contract checks.
# Model names identify component ownership, not general model-test sections.
# No pretrained checkpoints, full-model inference, or model benchmarks.
# Put those tests in models.cmake under ENGINE_BUILD_MODEL_TESTS.
# Included only when ENGINE_BUILD_TESTS is enabled.

# asr_graph_capacity
add_engine_test(asr_graph_capacity_test tests/unittests/test_asr_graph_capacity.cpp)

# attention_fallback
add_engine_test(attention_fallback_test tests/unittests/test_attention_fallback.cpp)

# Falcon KV-cache append layout (Audio8 TTS)
if (audio8_tts IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(audio8_tts_falcon_kv_cache_test tests/unittests/test_audio8_tts_falcon_kv_cache.cpp)
    target_include_directories(audio8_tts_falcon_kv_cache_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/community_models/audio8_tts)
endif()

# audio_chunking
add_engine_test(audio_chunking_test tests/unittests/test_audio_chunking.cpp)

# audio_dsp
add_engine_test(audio_dsp_test tests/unittests/test_audio_dsp.cpp)

# backend_device_resolution
add_engine_test(backend_device_resolution_test tests/unittests/test_backend_device_resolution.cpp)

# C ABI export contract
if (AUDIOCPP_BUILD_C_API)
    if (Python3_Interpreter_FOUND)
        set(AUDIOCPP_EXPORT_TOOL_ARGS "")
        if (MSVC)
            get_filename_component(AUDIOCPP_MSVC_BIN "${CMAKE_CXX_COMPILER}" DIRECTORY)
            find_program(AUDIOCPP_DUMPBIN dumpbin HINTS "${AUDIOCPP_MSVC_BIN}" NO_DEFAULT_PATH)
            if (AUDIOCPP_DUMPBIN)
                list(APPEND AUDIOCPP_EXPORT_TOOL_ARGS --symbol-tool "${AUDIOCPP_DUMPBIN}")
            endif()
            list(APPEND AUDIOCPP_EXPORT_TOOL_ARGS --require-tool)
        endif()
        add_test(NAME audiocpp_c_api_exports
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/capi/export_surface.py"
            --build-dir "${CMAKE_BINARY_DIR}"
            --header "${CMAKE_CURRENT_SOURCE_DIR}/include/audiocpp.h"
            ${AUDIOCPP_EXPORT_TOOL_ARGS})
        set_tests_properties(audiocpp_c_api_exports PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()

# chat_session
add_engine_test(chat_session_test tests/unittests/test_chat_session.cpp --log)

# Text tokenization (Chatterbox)
if (chatterbox IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(chatterbox_text_tokenizer_test tests/unittests/test_chatterbox_text_tokenizer.cpp)
endif()

# chinese_normalization
add_engine_test(chinese_normalization_test tests/unittests/test_chinese_normalization.cpp)

# chinese_variant
add_engine_test(chinese_variant_test tests/unittests/test_chinese_variant.cpp)

# citrinet_tokenizer_decode
add_engine_test(citrinet_tokenizer_decode_test tests/unittests/test_citrinet_tokenizer_decode.cpp)
target_compile_definitions(citrinet_tokenizer_decode_test PRIVATE
    ENGINE_TEST_ASSET_ROOT="${ENGINE_UNITTEST_ASSET_ROOT}"
)

# conv1d_pertap_fast_path
add_engine_test(conv1d_pertap_fast_path_test tests/unittests/test_conv1d_pertap_fast_path.cpp)

# convnext_modules
add_engine_test(convnext_modules_test
    tests/unittests/test_convnext_modules.cpp
    --log ${CMAKE_CURRENT_BINARY_DIR}/convnext_modules_test.log)

# cpu_extra_buffers
add_engine_test(cpu_extra_buffers_test tests/unittests/test_cpu_extra_buffers.cpp)

# Transcript-to-word timestamp alignment (CrisperWhisper)
if (crisperwhisper IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(crisperwhisper_transcript_alignment_test
        tests/unittests/test_crisperwhisper_transcript_alignment.cpp)
endif()

# ctc_host_decoder
add_engine_test(ctc_host_decoder_test tests/unittests/test_ctc_host_decoder.cpp)

# Synthetic CUDA operator and random-generator checks
if (ENGINE_ENABLE_CUDA)
    add_engine_test(cuda_bias_fusion_test tests/unittests/test_cuda_bias_fusion.cpp)
    set_tests_properties(cuda_bias_fusion_test PROPERTIES SKIP_RETURN_CODE 77)
    if (NOT GGML_BACKEND_DL)
        add_engine_test(cuda_head_convert_test tests/unittests/test_cuda_head_convert.cpp)
        target_link_libraries(cuda_head_convert_test PRIVATE CUDA::cudart_static)
        set_tests_properties(cuda_head_convert_test PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    add_engine_test(cuda_rope_rows_test tests/unittests/test_cuda_rope_rows.cpp)
    set_tests_properties(cuda_rope_rows_test PROPERTIES SKIP_RETURN_CODE 77)
    add_engine_test(cuda_round_bf16_fusion_test tests/unittests/test_cuda_round_bf16_fusion.cpp)
    set_tests_properties(cuda_round_bf16_fusion_test PROPERTIES SKIP_RETURN_CODE 77)
    add_engine_test(torch_random_cuda_multi_device_test tests/unittests/test_torch_random_cuda_multi_device.cu)
    target_include_directories(torch_random_cuda_multi_device_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(torch_random_cuda_multi_device_test PRIVATE CUDA::cudart)
    set_tests_properties(torch_random_cuda_multi_device_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

# Synthetic GRU recurrence checks (Dettato)
if (dettato IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(dettato_gru_test tests/dettato/gru_test.cpp --log)
endif()

# device_denoiser_runtime
add_engine_test(device_denoiser_runtime_test
    tests/unittests/test_device_denoiser_runtime.cpp
    --log ${CMAKE_CURRENT_BINARY_DIR}/device_denoiser_runtime_test.log)

# diffusion_math
add_engine_test(diffusion_math_test tests/unittests/test_diffusion_math.cpp)

# dit_runtime
add_engine_test(dit_runtime_test
    tests/unittests/test_dit_runtime.cpp
    --log ${CMAKE_CURRENT_BINARY_DIR}/dit_runtime_test.log)

# encoder_module
add_engine_test(encoder_module_test tests/unittests/test_encoder_modules.cpp)

# espeak
add_engine_test(espeak_data_test tests/unittests/test_espeak_data.cpp)
target_link_libraries(espeak_data_test PRIVATE Threads::Threads)
add_engine_test(espeak_phonemizer_test
    tests/unittests/test_espeak_phonemizer.cpp
    $<TARGET_FILE:espeak_test_library> $<TARGET_FILE:espeak_test_missing_symbol>)
target_link_libraries(espeak_phonemizer_test PRIVATE Threads::Threads)
add_dependencies(espeak_phonemizer_test espeak_test_library espeak_test_missing_symbol)
add_library(espeak_test_library SHARED tests/unittests/espeak_test_library.cpp)
target_link_libraries(espeak_test_library PRIVATE Threads::Threads)
add_library(espeak_test_missing_symbol SHARED tests/unittests/espeak_test_library.cpp)
target_link_libraries(espeak_test_missing_symbol PRIVATE Threads::Threads)
target_compile_definitions(espeak_test_missing_symbol PRIVATE ESPEAK_TEST_MISSING_SYMBOL=1)

# Speech-probability segmentation rules (FireRed VAD)
if (firered_vad IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(firered_vad_rules_test tests/firered_vad/test_rules.cpp)
    target_link_libraries(firered_vad_rules_test PRIVATE engine_runtime ggml)
    add_test(NAME firered_vad_rules_test COMMAND firered_vad_rules_test)
endif()

# Expert quantization, reference caching and sampling (Drama3)
if (fish_drama3 IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(fish_drama3_expert_quantization_test
        tests/fish_drama3/expert_quantization_test.cpp)
    add_engine_test(fish_drama3_reference_cache_test tests/fish_drama3/reference_cache_test.cpp)
    if (CMAKE_DL_LIBS)
        target_link_libraries(fish_drama3_reference_cache_test PRIVATE ${CMAKE_DL_LIBS})
    endif()
    add_engine_test(fish_drama3_sampler_test tests/fish_drama3/sampler_test.cpp --log)
endif()

# gated_delta_net_module
add_engine_test(gated_delta_net_module_test tests/unittests/test_gated_delta_net_module.cpp cpu --log)

# gguf_tensor_source
add_engine_test(gguf_tensor_source_test tests/unittests/test_gguf_tensor_source.cpp)

# CTC timestamp conversion (Granite5ASR)
if (granite5asr IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(granite5asr_ctc_timestamps_test tests/granite5asr/test_ctc_timestamps.cpp)
endif()

# hf_sampler
add_engine_test(hf_sampler_test tests/unittests/test_hf_sampler.cpp)

# hf_tokenizer_sequence_decode
add_engine_test(hf_tokenizer_sequence_decode_test tests/unittests/test_hf_tokenizer_sequence_decode.cpp)

# hf_tokenizer_vocab_bounds
add_engine_test(hf_tokenizer_vocab_bounds_test tests/unittests/test_hf_tokenizer_vocab_bounds.cpp)

# Request validation (IndexTTS2)
if (index_tts2 IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(index_tts2_request_test tests/index_tts2/request_test.cpp)
endif()

# kaldi_fbank_sparse
add_engine_test(kaldi_fbank_sparse_test tests/unittests/test_kaldi_fbank_sparse.cpp)

# kokoro_phoneme_rules
if (kokoro_tts IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(kokoro_phoneme_rules_test tests/unittests/test_kokoro_phoneme_rules.cpp --log)
endif()

# kokoro_voice_blending
if (kokoro_tts IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(kokoro_voice_blending_test tests/unittests/test_kokoro_voice_blending.cpp)
endif()

# kokoro_word_timings
add_engine_test(kokoro_word_timings_test tests/unittests/test_kokoro_word_timings.cpp)
target_link_libraries(kokoro_word_timings_test PRIVATE engine_model_kokoro_tts)

# Reference-audio preprocessing (KugelAudio)
if (kugelaudio IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(kugelaudio_reference_audio_test tests/kugelaudio/test_reference_audio.cpp)
endif()

# Synthetic package loading, frontend, tokenizer and TTS utilities (LFM2)
if (lfm2_audio IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(lfm2_audio_asr_inputs_test tests/lfm2_audio/lfm2_audio_asr_inputs_test.cpp)
    target_compile_definitions(lfm2_audio_asr_inputs_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    add_engine_test(lfm2_audio_assets_test tests/lfm2_audio/lfm2_audio_assets_test.cpp)
    target_compile_definitions(lfm2_audio_assets_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    add_engine_test(lfm2_audio_features_test tests/lfm2_audio/lfm2_audio_features_test.cpp)
    target_compile_definitions(lfm2_audio_features_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    add_engine_test(lfm2_audio_tokenizer_test tests/lfm2_audio/lfm2_audio_tokenizer_test.cpp)
    target_compile_definitions(lfm2_audio_tokenizer_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    add_engine_test(lfm2_audio_tts_test tests/lfm2_audio/lfm2_audio_tts_test.cpp)
    target_compile_definitions(lfm2_audio_tts_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# Synthetic package loading, frontend, prompts, chat artifacts and sampling (LFM3)
if (lfm3_audio IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(lfm3_audio_assets_test tests/lfm3_audio/lfm3_audio_assets_test.cpp)
    target_compile_definitions(lfm3_audio_assets_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(lfm3_audio_assets_test PROPERTIES SKIP_RETURN_CODE 77)
    add_engine_test(lfm3_audio_chat_test tests/lfm3_audio/lfm3_audio_chat_test.cpp)
    target_compile_definitions(lfm3_audio_chat_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(lfm3_audio_chat_test PROPERTIES SKIP_RETURN_CODE 77)
    add_engine_test(lfm3_audio_features_test tests/lfm3_audio/lfm3_audio_features_test.cpp)
    target_compile_definitions(lfm3_audio_features_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(lfm3_audio_features_test PROPERTIES SKIP_RETURN_CODE 77)
    add_engine_test(lfm3_audio_prompt_test tests/lfm3_audio/lfm3_audio_prompt_test.cpp)
    target_compile_definitions(lfm3_audio_prompt_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(lfm3_audio_prompt_test PROPERTIES SKIP_RETURN_CODE 77)
    add_engine_test(lfm3_audio_sampling_test tests/lfm3_audio/lfm3_audio_sampling_test.cpp)
    target_compile_definitions(lfm3_audio_sampling_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(lfm3_audio_sampling_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

# llama_bpe_pretokenizer_hook
add_engine_test(llama_bpe_pretokenizer_hook_test tests/unittests/test_llama_bpe_pretokenizer_hook.cpp)

# lstm_sequence
add_engine_test(lstm_sequence_test tests/unittests/test_lstm_sequence.cpp --log)

# mel_spectrogram_frontend
add_engine_test(mel_spectrogram_frontend_test tests/unittests/test_mel_spectrogram_frontend.cpp)

# midi_file
add_engine_test(midi_file_test tests/unittests/test_midi_file.cpp)

# Waveform trimming (Ming3 Audio)
if (ming3_audio IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(ming3_audio_waveform_trim_test
        tests/ming3_audio/test_ming3_audio_waveform_trim.cpp)
    target_include_directories(ming3_audio_waveform_trim_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/models/ming3_audio)
    target_link_libraries(ming3_audio_waveform_trim_test PRIVATE engine_runtime ggml)
    add_test(NAME ming3_audio_waveform_trim_test COMMAND ming3_audio_waveform_trim_test)
endif()

# Framework model-family routing contracts
add_engine_test(model_routing_test
    tests/unittests/test_model_routing.cpp
    --log ${CMAKE_CURRENT_BINARY_DIR}/model_routing_test.log)

# Model-spec schema and contract validation
add_engine_test(model_spec_system_test tests/unittests/test_model_spec_system.cpp)
target_compile_definitions(model_spec_system_test
    PRIVATE AUDIOCPP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# moe_engine
add_engine_test(moe_engine_test tests/unittests/test_moe_engine.cpp --log)
target_sources(moe_engine_test PRIVATE tests/unittests/test_allocation_failure.cpp)

# moe_engine_composition
add_engine_test(moe_engine_composition_test tests/unittests/test_moe_engine_composition.cpp --log)

# moe_engine_math
add_engine_test(moe_engine_math_test tests/unittests/test_moe_engine_math.cpp --log)

# moe_policy
add_engine_test(moe_policy_test tests/unittests/test_moe_policy.cpp --log)

# moe_prepared_graph
add_engine_test(moe_prepared_graph_test tests/unittests/test_moe_prepared_graph.cpp --log)
target_include_directories(moe_prepared_graph_test PRIVATE external/ggml/src)

# moonshine_neon_decoder
if (ENGINE_ENABLE_NEON AND moonshine_asr IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(moonshine_neon_decoder_test tests/unittests/test_moonshine_neon_decoder.cpp
        src/models/moonshine_asr/neon_decoder.cpp)
    target_include_directories(moonshine_neon_decoder_test PRIVATE include tests/unittests)
    target_link_libraries(moonshine_neon_decoder_test PRIVATE engine_neon)
    add_test(NAME moonshine_neon_decoder_test COMMAND moonshine_neon_decoder_test)
endif()

# Text chunking (MOSS-TTS-Nano)
if (moss IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(moss_tts_nano_text_chunking_test
        tests/moss_tts_nano/test_moss_tts_nano_text_chunking.cpp)
    target_include_directories(moss_tts_nano_text_chunking_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/models/moss/moss_tts_nano)
    target_link_libraries(moss_tts_nano_text_chunking_test PRIVATE engine_runtime ggml)
    add_test(NAME moss_tts_nano_text_chunking_test COMMAND moss_tts_nano_text_chunking_test)
endif()

# Text-normalization reference vectors (MOSS-TTS v1.5)
add_executable(moss_tts_v15_text_normalization_test
    tests/moss_tts_v15/text_normalization_test.cpp
)
target_link_libraries(moss_tts_v15_text_normalization_test PRIVATE engine_runtime ggml)
add_test(NAME moss_tts_v15_text_normalization
    COMMAND moss_tts_v15_text_normalization_test --vectors
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/moss_tts_v15/reference/text_normalization_vectors.json")
add_test(NAME moss_tts_v15_text_normalization_regressions
    COMMAND moss_tts_v15_text_normalization_test --vectors
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/moss_tts_v15/reference/text_normalization_regressions.json")

# Speaker-option handling (MOSS-TTSD)
add_engine_test(moss_ttsd_speakers_test tests/unittests/test_moss_ttsd_speakers.cpp)
target_link_libraries(moss_ttsd_speakers_test PRIVATE engine_model_moss_ttsd)

# Streaming frame scheduling and frontend state (Nemotron Diarization)
if (nemotron_3_diar IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(nemotron_3_diar_streaming_test tests/nemotron_3_diar/streaming_schedule_test.cpp)
endif()

# Speaker tagging and word-event assembly (Nemotron ASR)
if (nemotron_asr IN_LIST AUDIOCPP_LINKED_MODELS AND nemotron_3_diar IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(nemotron_asr_speaker_tagging_test tests/nemotron_asr/speaker_tagging_test.cpp)
endif()

# Alignment safety and request policies (Nemotron Transcribe)
if (nemotron_transcribe IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(nemotron_transcribe_alignment_safety_test
        tests/nemotron_transcribe/alignment_safety_test.cpp)
    add_engine_test(nemotron_transcribe_policy_test tests/nemotron_transcribe/policy_test.cpp)
endif()

# NEON tensor, block and vocoder component checks
if (ENGINE_ENABLE_NEON)
    add_engine_test(neon_blocks_test tests/unittests/test_neon_blocks.cpp)
    target_include_directories(neon_blocks_test PRIVATE src/framework/neon)
    add_executable(neon_library_test tests/unittests/test_neon_library.cpp)
    target_include_directories(neon_library_test PRIVATE tests/unittests)
    target_link_libraries(neon_library_test PRIVATE engine_neon)
    add_test(NAME neon_library_test COMMAND neon_library_test)
    add_executable(neon_tensor_ops_test tests/unittests/test_neon_tensor_ops.cpp)
    target_link_libraries(neon_tensor_ops_test PRIVATE engine_neon)
    add_test(NAME neon_tensor_ops_test COMMAND neon_tensor_ops_test)
    add_executable(neon_vocoder_test tests/unittests/test_neon_vocoder.cpp)
    target_link_libraries(neon_vocoder_test PRIVATE engine_neon)
    add_test(NAME neon_vocoder_test COMMAND neon_vocoder_test)
    set_tests_properties(neon_vocoder_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

# Synthetic streaming-session contracts, variant configuration and hotwords (Parakeet TDT)
if (parakeet_tdt IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(parakeet_bark3_session_test tests/parakeet_tdt/test_bark3_session.cpp)
    add_engine_test(parakeet_variant_config_test tests/parakeet_tdt/test_variant_config.cpp)
    target_compile_definitions(parakeet_variant_config_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
endif()
add_engine_test(phonon_hotwords_test tests/parakeet_tdt/test_hotwords.cpp)
target_sources(phonon_hotwords_test PRIVATE src/community_models/parakeet_tdt/hotwords.cpp)
target_include_directories(phonon_hotwords_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/llama_tokenizer)

# partial_text
add_engine_test(partial_text_test tests/unittests/test_partial_text.cpp)

# partial_text_render
add_engine_test(partial_text_render_test tests/unittests/test_partial_text_render.cpp)

# prepared_graph
add_engine_test(prepared_graph_test
    tests/unittests/test_prepared_graph.cpp
    --log ${CMAKE_CURRENT_BINARY_DIR}/prepared_graph_test.log)
target_sources(prepared_graph_test PRIVATE tests/unittests/test_allocation_failure.cpp)
target_include_directories(prepared_graph_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/ggml/src)

# quantize_chunking
add_engine_test(quantize_chunking_test tests/unittests/test_quantize_chunking.cpp)

# recipe
add_engine_test(recipe_test tests/unittests/test_recipe.cpp)
target_sources(recipe_test PRIVATE ${AUDIOCPP_RECIPE_SOURCES})
target_include_directories(recipe_test PRIVATE app/recipe)

# rknn_engine
add_engine_test(rknn_engine_test tests/unittests/test_rknn_engine.cpp)
if (ENGINE_ENABLE_RKNN)
    target_compile_definitions(rknn_engine_test PRIVATE ENGINE_HAVE_RKNN=1)
    if (ENGINE_RKNN_INCLUDE_DIR)
        target_include_directories(rknn_engine_test PRIVATE ${ENGINE_RKNN_INCLUDE_DIR})
    endif()
endif()

# rope_transformer_encoder
add_engine_test(rope_transformer_encoder_test tests/unittests/test_rope_transformer_encoder.cpp)

# safetensors_metadata_depth
add_engine_test(safetensors_metadata_depth_test tests/unittests/test_safetensors_metadata_depth.cpp)

# safetensors_offsets
add_engine_test(safetensors_offsets_test tests/unittests/test_safetensors_offsets.cpp)

# sense_asr
if (sense_asr IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(sense_asr_word_timestamps_test tests/sense_asr/test_word_timestamps.cpp)
endif()

# sentencepiece_tokenizer1
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

# Server selection, configuration and lifecycle contracts
set(AUDIOCPP_SERVER_TEST_GENERATED_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/generated/server_runtime_tests")
file(MAKE_DIRECTORY "${AUDIOCPP_SERVER_TEST_GENERATED_DIR}")
file(WRITE "${AUDIOCPP_SERVER_TEST_GENERATED_DIR}/server_frontend_module_declarations.inc" "")
file(WRITE "${AUDIOCPP_SERVER_TEST_GENERATED_DIR}/server_frontend_module_registrations.inc" "")
get_target_property(server_selection_sources audiocpp_server SOURCES)
list(FILTER server_selection_sources EXCLUDE REGEX "app/server/(main|model_installer|parallel_runtime)\\.cpp$")
list(FILTER server_selection_sources EXCLUDE REGEX "audiocpp_server_frontends")
get_target_property(server_lifecycle_sources audiocpp_server SOURCES)
list(FILTER server_lifecycle_sources EXCLUDE REGEX "app/server/(main|model_installer|runtime)\\.cpp$")
list(FILTER server_lifecycle_sources EXCLUDE REGEX "audiocpp_server_frontends")
add_executable(server_model_slots_test tests/unittests/test_server_model_slots.cpp)
target_include_directories(server_model_slots_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/app/server ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(server_model_slots_test PRIVATE Threads::Threads)
add_test(NAME server_model_slots_test COMMAND server_model_slots_test)
set_tests_properties(server_model_slots_test PROPERTIES TIMEOUT 60)
add_executable(server_parallel_lifecycle_test
    tests/unittests/test_server_parallel_lifecycle.cpp ${server_lifecycle_sources})
target_include_directories(server_parallel_lifecycle_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/app/server
    "${AUDIOCPP_SERVER_TEST_GENERATED_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(server_parallel_lifecycle_test PRIVATE engine_runtime ggml Threads::Threads)
if (WIN32)
    target_link_libraries(server_parallel_lifecycle_test PRIVATE ws2_32)
endif()
add_test(NAME server_parallel_lifecycle_test COMMAND server_parallel_lifecycle_test)
set_tests_properties(server_parallel_lifecycle_test PROPERTIES TIMEOUT 90)
add_executable(server_runtime_selection_test
    tests/unittests/test_server_runtime_selection.cpp ${server_selection_sources})
target_include_directories(server_runtime_selection_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/app/server
    "${AUDIOCPP_SERVER_TEST_GENERATED_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(server_runtime_selection_test PRIVATE engine_runtime ggml Threads::Threads)
if (WIN32)
    target_link_libraries(server_runtime_selection_test PRIVATE ws2_32)
endif()
add_test(NAME server_runtime_selection_test COMMAND server_runtime_selection_test)
set_tests_properties(server_runtime_selection_test PROPERTIES TIMEOUT 60)
add_executable(server_speech_option_exceptions_test
    tests/unittests/test_server_speech_option_exceptions.cpp)
target_include_directories(server_speech_option_exceptions_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/app/server)
add_test(NAME server_speech_option_exceptions_test COMMAND server_speech_option_exceptions_test)
add_executable(server_voice_preset_test
    tests/unittests/test_server_voice_preset.cpp ${server_selection_sources})
target_include_directories(server_voice_preset_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/app/server
    "${AUDIOCPP_SERVER_TEST_GENERATED_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(server_voice_preset_test PRIVATE engine_runtime ggml Threads::Threads)
if (WIN32)
    target_link_libraries(server_voice_preset_test PRIVATE ws2_32)
endif()
add_test(NAME server_voice_preset_test COMMAND server_voice_preset_test)
add_executable(session_pool_test tests/unittests/test_session_pool.cpp)
target_include_directories(session_pool_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/app/server)
target_link_libraries(session_pool_test PRIVATE engine_runtime ggml-base Threads::Threads)
add_test(NAME session_pool_test COMMAND session_pool_test)
set_tests_properties(session_pool_test PROPERTIES TIMEOUT 60)

# shared_weight_cache
add_engine_test(shared_weight_cache_test tests/unittests/test_shared_weight_cache.cpp)

# Audio preprocessing and musical-notation utilities (SheetSage)
if (sheetsage2 IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(sheetsage_audio_frontend_test tests/unittests/test_sheetsage_audio_frontend.cpp --log)
    add_engine_test(sheetsage_processing_test tests/unittests/test_sheetsage_processing.cpp --log)
endif()

# snake_beta
add_engine_test(snake_beta_test tests/unittests/test_snake_beta.cpp --log)

# Streaming speaker state and scheduling (Sortformer)
if (sortformer_diar_v2 IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(sortformer_v2_aosc_test tests/sortformer_diar/sortformer_v2_aosc_test.cpp)
    target_sources(sortformer_v2_aosc_test PRIVATE src/community_models/sortformer_diar/aosc_state.cpp)
    add_engine_test(sortformer_v2_schedule_test
        tests/sortformer_diar/sortformer_v2_schedule_test.cpp)
    target_sources(sortformer_v2_schedule_test PRIVATE
        src/community_models/sortformer_diar/frontend.cpp
        src/community_models/sortformer_diar/stream_schedule.cpp)
endif()

# streaming_audio
add_engine_test(streaming_audio_test tests/unittests/test_streaming_audio.cpp)

# subtitle_formatter
add_engine_test(subtitle_formatter_test tests/unittests/test_subtitle_formatter.cpp)

# text_chunking
add_engine_test(text_chunking_test tests/unittests/test_text_chunking.cpp)

# text_normalization
add_engine_test(text_normalization_test tests/unittests/test_text_normalization.cpp)

# Text rules, source-filter math and graph-cache behavior (Tiny Voca)
if (tiny_voca IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(tiny_voca_unit_test tests/tiny_voca/unit_test.cpp)
    target_include_directories(tiny_voca_unit_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/community_models/tiny_voca)
    if (MSVC)
        target_compile_options(tiny_voca_unit_test PRIVATE /utf-8)
    endif()
endif()

# torch_bin_storage_offset
add_engine_test(torch_bin_storage_offset_test tests/unittests/test_torch_bin_storage_offset.cpp)

# torch_random
add_engine_test(torch_random_test tests/unittests/test_torch_random.cpp)

# tq2_1
add_engine_test(tq2_1_test tests/unittests/test_tq2_1.cpp)
target_include_directories(tq2_1_test PRIVATE external/ggml/src)

# transformer_kv_ring
add_engine_test(transformer_kv_ring_test tests/unittests/test_transformer_kv_ring.cpp)

# transpose_module
add_engine_test(transpose_module_test tests/unittests/test_transpose_module.cpp)

# unicode_normalization
if (supertonic IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(unicode_normalization_test tests/unittests/test_unicode_normalization.cpp)
    target_sources(unicode_normalization_test PRIVATE
        src/models/supertonic/tokenizer_text.cpp)
endif()

# vad_loader
add_engine_test(vad_loader_test tests/unittests/test_vad_loader.cpp)
target_compile_definitions(vad_loader_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

# video_processing
add_engine_test(video_processing_test tests/unittests/test_video_processing.cpp --log)
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(video_processing_test PRIVATE OpenMP::OpenMP_CXX)
endif()

# vulkan_shared_weight_cache
if (ENGINE_ENABLE_VULKAN)
    add_engine_test(vulkan_shared_weight_cache_test tests/unittests/test_vulkan_shared_weight_cache.cpp)
    set_tests_properties(vulkan_shared_weight_cache_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

# wav_reader
add_engine_test(wav_reader_test tests/unittests/test_wav_reader.cpp)

# wav_reader_chunk_bounds
add_engine_test(wav_reader_chunk_bounds_test tests/unittests/test_wav_reader_chunk_bounds.cpp)

# wav_writer_formats
add_engine_test(wav_writer_formats_test tests/unittests/test_wav_writer_formats.cpp)

# weight_adapter
add_engine_test(weight_adapter_test tests/unittests/test_weight_adapter.cpp --log)

# weight_adapter_source
add_engine_test(weight_adapter_source_test tests/unittests/test_weight_adapter_source.cpp --log)

# Audio frontend and token decoding (Whistle)
if (whistle_asr IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(whistle_frontend_test
        tests/whistle_asr/test_frontend.cpp
        src/community_models/whistle_asr/frontend.cpp)
    target_link_libraries(whistle_frontend_test PRIVATE engine_runtime ggml)
    add_test(NAME whistle_frontend_test COMMAND whistle_frontend_test)
    add_executable(whistle_tokenizer_test
        tests/whistle_asr/test_tokenizer.cpp
        src/community_models/whistle_asr/assets.cpp)
    target_link_libraries(whistle_tokenizer_test PRIVATE engine_runtime ggml)
    add_test(NAME whistle_tokenizer_test COMMAND whistle_tokenizer_test)
endif()

# Attention layout, NAR conditioning/windowing and request validation (YuE2)
if (yue2 IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(yue2_attention_layout_test tests/yue2/attention_layout_test.cpp)
    add_engine_test(yue2_nar_conditioning_test tests/yue2/nar_conditioning_test.cpp)
    add_engine_test(yue2_nar_windowing_test tests/yue2/nar_windowing_test.cpp)
    add_engine_test(yue2_request_test tests/yue2/request_test.cpp)
endif()

# zipformer_modules
add_engine_test(zipformer_modules_test tests/unittests/test_zipformer_modules.cpp)

# Text normalization, tokenization and reference-file parsing (Zonos3)
if (zonos3 IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(zonos3_text_test tests/zonos3/text_test.cpp)
endif()
