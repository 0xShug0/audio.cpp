# Self-contained unit and framework/application contract tests.
# Included only when ENGINE_BUILD_TESTS is enabled.

# asr_graph_capacity
add_engine_unittest(asr_graph_capacity_test tests/unittests/test_asr_graph_capacity.cpp)
add_test(NAME asr_graph_capacity_test COMMAND asr_graph_capacity_test)

# attention_fallback
add_engine_unittest(attention_fallback_test tests/unittests/test_attention_fallback.cpp)
target_include_directories(attention_fallback_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
add_test(NAME attention_fallback_test COMMAND attention_fallback_test)

# audio8_tts
if (audio8_tts IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_unittest(audio8_tts_falcon_kv_cache_test tests/unittests/test_audio8_tts_falcon_kv_cache.cpp)
    target_include_directories(audio8_tts_falcon_kv_cache_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/community_models/audio8_tts)
    add_test(NAME audio8_tts_falcon_kv_cache_test COMMAND audio8_tts_falcon_kv_cache_test)
endif()

# audio_chunking
add_engine_unittest(audio_chunking_test tests/unittests/test_audio_chunking.cpp)
add_test(NAME audio_chunking_test COMMAND audio_chunking_test)

# audio_dsp
add_engine_unittest(audio_dsp_test tests/unittests/test_audio_dsp.cpp)
add_test(NAME audio_dsp_test COMMAND audio_dsp_test)

# backend_device_resolution
add_engine_unittest(backend_device_resolution_test tests/unittests/test_backend_device_resolution.cpp)
target_include_directories(backend_device_resolution_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
add_test(NAME backend_device_resolution_test COMMAND backend_device_resolution_test)

# c_api
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
add_engine_unittest(chat_session_test tests/unittests/test_chat_session.cpp)
add_test(NAME chat_session_test COMMAND chat_session_test --log)

# chatterbox
if (chatterbox IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_unittest(chatterbox_text_tokenizer_test tests/unittests/test_chatterbox_text_tokenizer.cpp)
    add_test(NAME chatterbox_text_tokenizer_test COMMAND chatterbox_text_tokenizer_test)
endif()

# chinese_normalization
add_engine_unittest(chinese_normalization_test tests/unittests/test_chinese_normalization.cpp)
add_test(NAME chinese_normalization_test COMMAND chinese_normalization_test)

# chinese_variant
add_engine_unittest(chinese_variant_test tests/unittests/test_chinese_variant.cpp)
add_test(NAME chinese_variant_test COMMAND chinese_variant_test)

# citrinet_tokenizer_decode
add_engine_unittest(citrinet_tokenizer_decode_test tests/unittests/test_citrinet_tokenizer_decode.cpp)
target_compile_definitions(citrinet_tokenizer_decode_test PRIVATE
    ENGINE_TEST_ASSET_ROOT="${ENGINE_UNITTEST_ASSET_ROOT}"
)
add_test(
    NAME citrinet_tokenizer_decode_test
    COMMAND citrinet_tokenizer_decode_test
)

# conv1d_pertap_fast_path
add_engine_unittest(conv1d_pertap_fast_path_test tests/unittests/test_conv1d_pertap_fast_path.cpp)
add_test(NAME conv1d_pertap_fast_path_test COMMAND conv1d_pertap_fast_path_test)

# convnext_modules
add_engine_unittest(convnext_modules_test tests/unittests/test_convnext_modules.cpp)
add_test(NAME convnext_modules_test COMMAND convnext_modules_test
    --log ${CMAKE_CURRENT_BINARY_DIR}/convnext_modules_test.log)

# cpu_extra_buffers
add_engine_unittest(cpu_extra_buffers_test tests/unittests/test_cpu_extra_buffers.cpp)
add_test(NAME cpu_extra_buffers_test COMMAND cpu_extra_buffers_test)

# crisperwhisper
if (crisperwhisper IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_unittest(crisperwhisper_transcript_alignment_test
        tests/unittests/test_crisperwhisper_transcript_alignment.cpp)
    add_test(NAME crisperwhisper_transcript_alignment_test
        COMMAND crisperwhisper_transcript_alignment_test)
endif()

# ctc_host_decoder
add_engine_unittest(ctc_host_decoder_test tests/unittests/test_ctc_host_decoder.cpp)
add_test(NAME ctc_host_decoder_test COMMAND ctc_host_decoder_test)

# cuda
if (ENGINE_ENABLE_CUDA)
    add_engine_unittest(cuda_bias_fusion_test tests/unittests/test_cuda_bias_fusion.cpp)
    add_test(NAME cuda_bias_fusion_test COMMAND cuda_bias_fusion_test)
    set_tests_properties(cuda_bias_fusion_test PROPERTIES SKIP_RETURN_CODE 77)
    if (NOT GGML_BACKEND_DL)
        add_engine_unittest(cuda_head_convert_test tests/unittests/test_cuda_head_convert.cpp)
        target_link_libraries(cuda_head_convert_test PRIVATE CUDA::cudart_static)
        add_test(NAME cuda_head_convert_test COMMAND cuda_head_convert_test)
        set_tests_properties(cuda_head_convert_test PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    add_engine_unittest(cuda_rope_rows_test tests/unittests/test_cuda_rope_rows.cpp)
    add_test(NAME cuda_rope_rows_test COMMAND cuda_rope_rows_test)
    set_tests_properties(cuda_rope_rows_test PROPERTIES SKIP_RETURN_CODE 77)
    add_engine_unittest(cuda_round_bf16_fusion_test tests/unittests/test_cuda_round_bf16_fusion.cpp)
    add_test(NAME cuda_round_bf16_fusion_test COMMAND cuda_round_bf16_fusion_test)
    set_tests_properties(cuda_round_bf16_fusion_test PROPERTIES SKIP_RETURN_CODE 77)
    add_engine_unittest(torch_random_cuda_multi_device_test
        tests/unittests/test_torch_random_cuda_multi_device.cu)
    target_include_directories(torch_random_cuda_multi_device_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(torch_random_cuda_multi_device_test PRIVATE CUDA::cudart)
    add_test(NAME torch_random_cuda_multi_device_test COMMAND torch_random_cuda_multi_device_test)
    set_tests_properties(torch_random_cuda_multi_device_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

# dettato
if (dettato IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_unittest(dettato_gru_test tests/community_models/dettato/gru_test.cpp)
    add_test(NAME dettato_gru_test COMMAND dettato_gru_test --log)
endif()

# device_denoiser_runtime
add_engine_unittest(device_denoiser_runtime_test tests/unittests/test_device_denoiser_runtime.cpp)
add_test(NAME device_denoiser_runtime_test COMMAND device_denoiser_runtime_test
    --log ${CMAKE_CURRENT_BINARY_DIR}/device_denoiser_runtime_test.log)

# diffusion_math
add_engine_unittest(diffusion_math_test tests/unittests/test_diffusion_math.cpp)
add_test(NAME diffusion_math_test COMMAND diffusion_math_test)

# dit_runtime
add_engine_unittest(dit_runtime_test tests/unittests/test_dit_runtime.cpp)
add_test(NAME dit_runtime_test COMMAND dit_runtime_test
    --log ${CMAKE_CURRENT_BINARY_DIR}/dit_runtime_test.log)

# encoder_module
add_engine_unittest(encoder_module_test tests/unittests/test_encoder_modules.cpp)
add_test(NAME encoder_module_test COMMAND encoder_module_test)

# espeak
add_engine_unittest(espeak_data_test tests/unittests/test_espeak_data.cpp)
target_link_libraries(espeak_data_test PRIVATE Threads::Threads)
add_test(NAME espeak_data_test COMMAND espeak_data_test)
add_engine_unittest(espeak_phonemizer_test tests/unittests/test_espeak_phonemizer.cpp)
target_link_libraries(espeak_phonemizer_test PRIVATE Threads::Threads)
add_dependencies(espeak_phonemizer_test espeak_test_library espeak_test_missing_symbol)
add_test(NAME espeak_phonemizer_test COMMAND espeak_phonemizer_test
    $<TARGET_FILE:espeak_test_library> $<TARGET_FILE:espeak_test_missing_symbol>)
add_library(espeak_test_library SHARED tests/unittests/espeak_test_library.cpp)
target_link_libraries(espeak_test_library PRIVATE Threads::Threads)
add_library(espeak_test_missing_symbol SHARED tests/unittests/espeak_test_library.cpp)
target_link_libraries(espeak_test_missing_symbol PRIVATE Threads::Threads)
target_compile_definitions(espeak_test_missing_symbol PRIVATE ESPEAK_TEST_MISSING_SYMBOL=1)

# firered_vad
if (firered_vad IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(firered_vad_rules_test tests/firered_vad/test_rules.cpp)
    target_link_libraries(firered_vad_rules_test PRIVATE engine_runtime ggml)
    add_test(NAME firered_vad_rules_test COMMAND firered_vad_rules_test)
endif()

# fish_drama3
if (fish_drama3 IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_unittest(fish_drama3_expert_quantization_test tests/community_models/fish_drama3/expert_quantization_test.cpp)
    add_test(NAME fish_drama3_expert_quantization_test COMMAND fish_drama3_expert_quantization_test)
    add_engine_unittest(fish_drama3_reference_cache_test tests/community_models/fish_drama3/reference_cache_test.cpp)
    add_test(NAME fish_drama3_reference_cache_test COMMAND fish_drama3_reference_cache_test)
    add_engine_unittest(fish_drama3_sampler_test tests/community_models/fish_drama3/sampler_test.cpp)
    add_test(NAME fish_drama3_sampler_test COMMAND fish_drama3_sampler_test --log)
endif()

# gated_delta_net_module
add_engine_unittest(gated_delta_net_module_test tests/unittests/test_gated_delta_net_module.cpp)
add_test(NAME gated_delta_net_module_test COMMAND gated_delta_net_module_test cpu --log)

# gguf_tensor_source
add_engine_unittest(gguf_tensor_source_test tests/unittests/test_gguf_tensor_source.cpp)
target_include_directories(gguf_tensor_source_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
add_test(NAME gguf_tensor_source_test COMMAND gguf_tensor_source_test)

# granite5asr
if (granite5asr IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_unittest(granite5asr_ctc_timestamps_test tests/granite5asr/test_ctc_timestamps.cpp)
    target_include_directories(granite5asr_ctc_timestamps_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    add_test(NAME granite5asr_ctc_timestamps_test COMMAND granite5asr_ctc_timestamps_test)
endif()

# hf_sampler
add_engine_unittest(hf_sampler_test tests/unittests/test_hf_sampler.cpp)
add_test(NAME hf_sampler_test COMMAND hf_sampler_test)

# hf_tokenizer_sequence_decode
add_engine_unittest(hf_tokenizer_sequence_decode_test tests/unittests/test_hf_tokenizer_sequence_decode.cpp)
add_test(NAME hf_tokenizer_sequence_decode_test COMMAND hf_tokenizer_sequence_decode_test)

# hf_tokenizer_vocab_bounds
add_engine_unittest(hf_tokenizer_vocab_bounds_test tests/unittests/test_hf_tokenizer_vocab_bounds.cpp)
add_test(NAME hf_tokenizer_vocab_bounds_test COMMAND hf_tokenizer_vocab_bounds_test)

# index_tts2
if (index_tts2 IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_unittest(index_tts2_request_test tests/index_tts2/request_test.cpp)
    add_test(NAME index_tts2_request_test COMMAND index_tts2_request_test)
endif()

# kaldi_fbank_sparse
add_engine_unittest(kaldi_fbank_sparse_test tests/unittests/test_kaldi_fbank_sparse.cpp)
add_test(NAME kaldi_fbank_sparse_test COMMAND kaldi_fbank_sparse_test)

# kokoro_phoneme_rules
if (kokoro_tts IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_unittest(kokoro_phoneme_rules_test tests/unittests/test_kokoro_phoneme_rules.cpp)
    add_test(NAME kokoro_phoneme_rules_test COMMAND kokoro_phoneme_rules_test --log)
endif()

# kokoro_voice_blending
if (kokoro_tts IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_unittest(kokoro_voice_blending_test tests/unittests/test_kokoro_voice_blending.cpp)
    add_test(NAME kokoro_voice_blending_test COMMAND kokoro_voice_blending_test)
endif()

# kokoro_word_timings
add_engine_unittest(kokoro_word_timings_test tests/unittests/test_kokoro_word_timings.cpp)
target_include_directories(kokoro_word_timings_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
target_link_libraries(kokoro_word_timings_test PRIVATE engine_model_kokoro_tts)
add_test(NAME kokoro_word_timings_test COMMAND kokoro_word_timings_test)

# kugelaudio
if (kugelaudio IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_unittest(kugelaudio_reference_audio_test tests/kugelaudio/test_reference_audio.cpp)
    target_include_directories(kugelaudio_reference_audio_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    add_test(NAME kugelaudio_reference_audio_test COMMAND kugelaudio_reference_audio_test)
endif()

# lfm2_audio
if (lfm2_audio IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_unittest(lfm2_audio_asr_inputs_test tests/community_models/lfm2_audio/lfm2_audio_asr_inputs_test.cpp)
    target_include_directories(lfm2_audio_asr_inputs_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    target_compile_definitions(lfm2_audio_asr_inputs_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME lfm2_audio_asr_inputs_test COMMAND lfm2_audio_asr_inputs_test)
    add_engine_unittest(lfm2_audio_assets_test tests/community_models/lfm2_audio/lfm2_audio_assets_test.cpp)
    target_include_directories(lfm2_audio_assets_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    target_compile_definitions(lfm2_audio_assets_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME lfm2_audio_assets_test COMMAND lfm2_audio_assets_test)
    add_engine_unittest(lfm2_audio_features_test tests/community_models/lfm2_audio/lfm2_audio_features_test.cpp)
    target_include_directories(lfm2_audio_features_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    target_compile_definitions(lfm2_audio_features_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME lfm2_audio_features_test COMMAND lfm2_audio_features_test)
    add_engine_unittest(lfm2_audio_tokenizer_test tests/community_models/lfm2_audio/lfm2_audio_tokenizer_test.cpp)
    target_include_directories(lfm2_audio_tokenizer_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    target_compile_definitions(lfm2_audio_tokenizer_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME lfm2_audio_tokenizer_test COMMAND lfm2_audio_tokenizer_test)
    add_engine_unittest(lfm2_audio_tts_test tests/community_models/lfm2_audio/lfm2_audio_tts_test.cpp)
    target_include_directories(lfm2_audio_tts_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    target_compile_definitions(lfm2_audio_tts_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME lfm2_audio_tts_test COMMAND lfm2_audio_tts_test)
endif()

# lfm3_audio
if (lfm3_audio IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_unittest(lfm3_audio_assets_test tests/community_models/lfm3_audio/lfm3_audio_assets_test.cpp)
    target_include_directories(lfm3_audio_assets_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    target_compile_definitions(lfm3_audio_assets_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME lfm3_audio_assets_test COMMAND lfm3_audio_assets_test)
    set_tests_properties(lfm3_audio_assets_test PROPERTIES SKIP_RETURN_CODE 77)
    add_engine_unittest(lfm3_audio_chat_test tests/community_models/lfm3_audio/lfm3_audio_chat_test.cpp)
    target_include_directories(lfm3_audio_chat_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    target_compile_definitions(lfm3_audio_chat_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME lfm3_audio_chat_test COMMAND lfm3_audio_chat_test)
    set_tests_properties(lfm3_audio_chat_test PROPERTIES SKIP_RETURN_CODE 77)
    add_engine_unittest(lfm3_audio_features_test tests/community_models/lfm3_audio/lfm3_audio_features_test.cpp)
    target_include_directories(lfm3_audio_features_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    target_compile_definitions(lfm3_audio_features_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME lfm3_audio_features_test COMMAND lfm3_audio_features_test)
    set_tests_properties(lfm3_audio_features_test PROPERTIES SKIP_RETURN_CODE 77)
    add_engine_unittest(lfm3_audio_prompt_test tests/community_models/lfm3_audio/lfm3_audio_prompt_test.cpp)
    target_include_directories(lfm3_audio_prompt_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    target_compile_definitions(lfm3_audio_prompt_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME lfm3_audio_prompt_test COMMAND lfm3_audio_prompt_test)
    set_tests_properties(lfm3_audio_prompt_test PROPERTIES SKIP_RETURN_CODE 77)
    add_engine_unittest(lfm3_audio_sampling_test tests/community_models/lfm3_audio/lfm3_audio_sampling_test.cpp)
    target_include_directories(lfm3_audio_sampling_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    target_compile_definitions(lfm3_audio_sampling_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME lfm3_audio_sampling_test COMMAND lfm3_audio_sampling_test)
    set_tests_properties(lfm3_audio_sampling_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

# llama_bpe_pretokenizer_hook
add_engine_unittest(llama_bpe_pretokenizer_hook_test tests/unittests/test_llama_bpe_pretokenizer_hook.cpp)
add_test(NAME llama_bpe_pretokenizer_hook_test COMMAND llama_bpe_pretokenizer_hook_test)

# lstm_sequence
add_engine_unittest(lstm_sequence_test tests/unittests/test_lstm_sequence.cpp)
add_test(NAME lstm_sequence_test COMMAND lstm_sequence_test --log)

# mel_spectrogram_frontend
add_engine_unittest(mel_spectrogram_frontend_test
    tests/unittests/test_mel_spectrogram_frontend.cpp)
add_test(NAME mel_spectrogram_frontend_test COMMAND mel_spectrogram_frontend_test)

# midi_file
add_engine_unittest(midi_file_test tests/unittests/test_midi_file.cpp)
add_test(NAME midi_file_test COMMAND midi_file_test)

# ming3_audio
if (ming3_audio IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(ming3_audio_waveform_trim_test
        tests/ming3_audio/test_ming3_audio_waveform_trim.cpp)
    target_include_directories(ming3_audio_waveform_trim_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/models/ming3_audio)
    target_link_libraries(ming3_audio_waveform_trim_test PRIVATE engine_runtime ggml)
    add_test(NAME ming3_audio_waveform_trim_test COMMAND ming3_audio_waveform_trim_test)
endif()

# model_routing
add_engine_unittest(model_routing_test tests/unittests/test_model_routing.cpp)
add_test(NAME model_routing_test COMMAND model_routing_test
    --log ${CMAKE_CURRENT_BINARY_DIR}/model_routing_test.log)

# model_spec_system
add_engine_unittest(model_spec_system_test tests/unittests/test_model_spec_system.cpp)
target_compile_definitions(model_spec_system_test
    PRIVATE AUDIOCPP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(model_spec_system_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
add_test(NAME model_spec_system_test COMMAND model_spec_system_test)

# moe_engine
add_engine_unittest(moe_engine_test tests/unittests/test_moe_engine.cpp)
target_sources(moe_engine_test PRIVATE tests/unittests/test_allocation_failure.cpp)
add_test(NAME moe_engine_test COMMAND moe_engine_test --log)

# moe_engine_composition
add_engine_unittest(moe_engine_composition_test tests/unittests/test_moe_engine_composition.cpp)
add_test(NAME moe_engine_composition_test COMMAND moe_engine_composition_test --log)

# moe_engine_math
add_engine_unittest(moe_engine_math_test tests/unittests/test_moe_engine_math.cpp)
add_test(NAME moe_engine_math_test COMMAND moe_engine_math_test --log)

# moe_policy
add_engine_unittest(moe_policy_test tests/unittests/test_moe_policy.cpp)
add_test(NAME moe_policy_test COMMAND moe_policy_test --log)

# moe_prepared_graph
add_engine_unittest(moe_prepared_graph_test tests/unittests/test_moe_prepared_graph.cpp)
target_include_directories(moe_prepared_graph_test PRIVATE external/ggml/src)
add_test(NAME moe_prepared_graph_test COMMAND moe_prepared_graph_test --log)

# moonshine_neon_decoder
if (ENGINE_ENABLE_NEON AND moonshine_asr IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(moonshine_neon_decoder_test tests/unittests/test_moonshine_neon_decoder.cpp
        src/models/moonshine_asr/neon_decoder.cpp)
    target_include_directories(moonshine_neon_decoder_test PRIVATE include tests/unittests)
    target_link_libraries(moonshine_neon_decoder_test PRIVATE engine_neon)
    add_test(NAME moonshine_neon_decoder_test COMMAND moonshine_neon_decoder_test)
endif()

# moss_tts_nano
if (moss IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(moss_tts_nano_text_chunking_test
        tests/moss_tts_nano/test_moss_tts_nano_text_chunking.cpp)
    target_include_directories(moss_tts_nano_text_chunking_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/models/moss/moss_tts_nano)
    target_link_libraries(moss_tts_nano_text_chunking_test PRIVATE engine_runtime ggml)
    add_test(NAME moss_tts_nano_text_chunking_test COMMAND moss_tts_nano_text_chunking_test)
endif()

# moss_tts_v15
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

# moss_ttsd
add_engine_unittest(moss_ttsd_speakers_test tests/unittests/test_moss_ttsd_speakers.cpp)
target_link_libraries(moss_ttsd_speakers_test PRIVATE engine_model_moss_ttsd)
add_test(NAME moss_ttsd_speakers_test COMMAND moss_ttsd_speakers_test)

# nemotron_3_diar
if (nemotron_3_diar IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_unittest(nemotron_3_diar_streaming_test tests/nemotron_3_diar/streaming_schedule_test.cpp)
    add_test(NAME nemotron_3_diar_streaming_test COMMAND nemotron_3_diar_streaming_test)
endif()

# nemotron_asr
if (nemotron_asr IN_LIST AUDIOCPP_LINKED_MODELS AND nemotron_3_diar IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_unittest(nemotron_asr_speaker_tagging_test tests/nemotron_asr/speaker_tagging_test.cpp)
    add_test(NAME nemotron_asr_speaker_tagging_test COMMAND nemotron_asr_speaker_tagging_test)
endif()

# nemotron_transcribe
if (nemotron_transcribe IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_unittest(nemotron_transcribe_alignment_safety_test tests/community_models/nemotron_transcribe/alignment_safety_test.cpp)
    add_test(NAME nemotron_transcribe_alignment_safety_test COMMAND nemotron_transcribe_alignment_safety_test)
    add_engine_unittest(nemotron_transcribe_policy_test tests/community_models/nemotron_transcribe/policy_test.cpp)
    add_test(NAME nemotron_transcribe_policy_test COMMAND nemotron_transcribe_policy_test)
endif()

# neon
if (ENGINE_ENABLE_NEON)
    add_engine_unittest(neon_blocks_test tests/unittests/test_neon_blocks.cpp)
    target_include_directories(neon_blocks_test PRIVATE src/framework/neon)
    add_test(NAME neon_blocks_test COMMAND neon_blocks_test)
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

# parakeet_tdt
if (parakeet_tdt IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_unittest(parakeet_bark3_session_test tests/parakeet_tdt/test_bark3_session.cpp)
    target_include_directories(parakeet_bark3_session_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    add_test(NAME parakeet_bark3_session_test COMMAND parakeet_bark3_session_test)
    add_engine_unittest(parakeet_variant_config_test tests/parakeet_tdt/test_variant_config.cpp)
    target_include_directories(parakeet_variant_config_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    target_compile_definitions(parakeet_variant_config_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME parakeet_variant_config_test COMMAND parakeet_variant_config_test)
endif()
add_engine_unittest(phonon_hotwords_test tests/parakeet_tdt/test_hotwords.cpp)
target_sources(phonon_hotwords_test PRIVATE src/community_models/parakeet_tdt/hotwords.cpp)
target_include_directories(phonon_hotwords_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/llama_tokenizer)
add_test(NAME phonon_hotwords_test COMMAND phonon_hotwords_test)

# partial_text
add_engine_unittest(partial_text_test tests/unittests/test_partial_text.cpp)
add_test(NAME partial_text_test COMMAND partial_text_test)

# partial_text_render
add_engine_unittest(partial_text_render_test tests/unittests/test_partial_text_render.cpp)
add_test(NAME partial_text_render_test COMMAND partial_text_render_test)

# prepared_graph
add_engine_unittest(prepared_graph_test tests/unittests/test_prepared_graph.cpp)
target_sources(prepared_graph_test PRIVATE tests/unittests/test_allocation_failure.cpp)
target_include_directories(prepared_graph_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/ggml/src)
add_test(NAME prepared_graph_test COMMAND prepared_graph_test
    --log ${CMAKE_CURRENT_BINARY_DIR}/prepared_graph_test.log)

# quantize_chunking
add_engine_unittest(quantize_chunking_test tests/unittests/test_quantize_chunking.cpp)
add_test(NAME quantize_chunking_test COMMAND quantize_chunking_test)

# recipe
add_engine_unittest(recipe_test tests/unittests/test_recipe.cpp)
target_sources(recipe_test PRIVATE ${AUDIOCPP_RECIPE_SOURCES})
target_include_directories(recipe_test PRIVATE app/recipe)
add_test(NAME recipe_test COMMAND recipe_test)

# rknn_engine
add_engine_unittest(rknn_engine_test tests/unittests/test_rknn_engine.cpp)
add_test(NAME rknn_engine_test COMMAND rknn_engine_test)
if (ENGINE_ENABLE_RKNN)
    target_compile_definitions(rknn_engine_test PRIVATE ENGINE_HAVE_RKNN=1)
    if (ENGINE_RKNN_INCLUDE_DIR)
        target_include_directories(rknn_engine_test PRIVATE ${ENGINE_RKNN_INCLUDE_DIR})
    endif()
endif()

# rope_transformer_encoder
add_engine_unittest(rope_transformer_encoder_test tests/unittests/test_rope_transformer_encoder.cpp)
add_test(NAME rope_transformer_encoder_test COMMAND rope_transformer_encoder_test)

# safetensors_metadata_depth
add_engine_unittest(safetensors_metadata_depth_test tests/unittests/test_safetensors_metadata_depth.cpp)
add_test(NAME safetensors_metadata_depth_test COMMAND safetensors_metadata_depth_test)

# safetensors_offsets
add_engine_unittest(safetensors_offsets_test tests/unittests/test_safetensors_offsets.cpp)
add_test(NAME safetensors_offsets_test COMMAND safetensors_offsets_test)

# sense_asr
if (sense_asr IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_unittest(sense_asr_word_timestamps_test tests/sense_asr/test_word_timestamps.cpp)
    target_include_directories(sense_asr_word_timestamps_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    add_test(NAME sense_asr_word_timestamps_test COMMAND sense_asr_word_timestamps_test)
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

# server
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
add_engine_unittest(shared_weight_cache_test tests/unittests/test_shared_weight_cache.cpp)
add_test(NAME shared_weight_cache_test COMMAND shared_weight_cache_test)

# sheetsage
if (sheetsage2 IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_unittest(sheetsage_audio_frontend_test tests/unittests/test_sheetsage_audio_frontend.cpp)
    add_test(NAME sheetsage_audio_frontend_test COMMAND sheetsage_audio_frontend_test --log)
    add_engine_unittest(sheetsage_processing_test tests/unittests/test_sheetsage_processing.cpp)
    add_test(NAME sheetsage_processing_test COMMAND sheetsage_processing_test --log)
endif()

# snake_beta
add_engine_unittest(snake_beta_test tests/unittests/test_snake_beta.cpp)
add_test(NAME snake_beta_test COMMAND snake_beta_test --log)

# sortformer_diar
if (sortformer_diar_v2 IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_unittest(sortformer_v2_aosc_test
        tests/community_models/sortformer_diar/sortformer_v2_aosc_test.cpp)
    target_sources(sortformer_v2_aosc_test PRIVATE src/community_models/sortformer_diar/aosc_state.cpp)
    add_test(NAME sortformer_v2_aosc_test COMMAND sortformer_v2_aosc_test)
    add_engine_unittest(sortformer_v2_schedule_test
        tests/community_models/sortformer_diar/sortformer_v2_schedule_test.cpp)
    target_sources(sortformer_v2_schedule_test PRIVATE
        src/community_models/sortformer_diar/frontend.cpp
        src/community_models/sortformer_diar/stream_schedule.cpp)
    add_test(NAME sortformer_v2_schedule_test COMMAND sortformer_v2_schedule_test)
endif()

# streaming_audio
add_engine_unittest(streaming_audio_test tests/unittests/test_streaming_audio.cpp)
add_test(NAME streaming_audio_test COMMAND streaming_audio_test)

# subtitle_formatter
add_engine_unittest(subtitle_formatter_test tests/unittests/test_subtitle_formatter.cpp)
add_test(NAME subtitle_formatter_test COMMAND subtitle_formatter_test)

# text_chunking
add_engine_unittest(text_chunking_test tests/unittests/test_text_chunking.cpp)
add_test(NAME text_chunking_test COMMAND text_chunking_test)

# text_normalization
add_engine_unittest(text_normalization_test tests/unittests/test_text_normalization.cpp)
add_test(NAME text_normalization_test COMMAND text_normalization_test)

# tiny_voca
if (tiny_voca IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_unittest(tiny_voca_unit_test tests/community_models/tiny_voca/unit_test.cpp)
    target_include_directories(tiny_voca_unit_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/community_models/tiny_voca)
    if (MSVC)
        target_compile_options(tiny_voca_unit_test PRIVATE /utf-8)
    endif()
    add_test(NAME tiny_voca_unit_test COMMAND tiny_voca_unit_test)
endif()

# torch_bin_storage_offset
add_engine_unittest(torch_bin_storage_offset_test tests/unittests/test_torch_bin_storage_offset.cpp)
add_test(NAME torch_bin_storage_offset_test COMMAND torch_bin_storage_offset_test)

# torch_random
add_engine_unittest(torch_random_test tests/unittests/test_torch_random.cpp)
add_test(NAME torch_random_test COMMAND torch_random_test)

# tq2_1
add_engine_unittest(tq2_1_test tests/unittests/test_tq2_1.cpp)
target_include_directories(tq2_1_test PRIVATE external/ggml/src)
add_test(NAME tq2_1_test COMMAND tq2_1_test)

# transformer_kv_ring
add_engine_unittest(transformer_kv_ring_test tests/unittests/test_transformer_kv_ring.cpp)
add_test(NAME transformer_kv_ring_test COMMAND transformer_kv_ring_test)

# transpose_module
add_engine_unittest(transpose_module_test tests/unittests/test_transpose_module.cpp)
target_include_directories(transpose_module_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
add_test(NAME transpose_module_test COMMAND transpose_module_test)

# unicode_normalization
if (supertonic IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_unittest(unicode_normalization_test tests/unittests/test_unicode_normalization.cpp)
    target_sources(unicode_normalization_test PRIVATE
        src/models/supertonic/tokenizer_text.cpp)
    add_test(NAME unicode_normalization_test COMMAND unicode_normalization_test)
endif()

# vad_loader
add_engine_unittest(vad_loader_test tests/unittests/test_vad_loader.cpp)
target_compile_definitions(vad_loader_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
add_test(NAME vad_loader_test COMMAND vad_loader_test)

# video_processing
add_engine_unittest(video_processing_test tests/unittests/test_video_processing.cpp)
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(video_processing_test PRIVATE OpenMP::OpenMP_CXX)
endif()
add_test(NAME video_processing_test COMMAND video_processing_test --log)

# vulkan_shared_weight_cache
if (ENGINE_ENABLE_VULKAN)
    add_engine_unittest(vulkan_shared_weight_cache_test tests/unittests/test_vulkan_shared_weight_cache.cpp)
    add_test(NAME vulkan_shared_weight_cache_test COMMAND vulkan_shared_weight_cache_test)
    set_tests_properties(vulkan_shared_weight_cache_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

# wav_reader
add_engine_unittest(wav_reader_test tests/unittests/test_wav_reader.cpp)
add_test(NAME wav_reader_test COMMAND wav_reader_test)

# wav_reader_chunk_bounds
add_engine_unittest(wav_reader_chunk_bounds_test tests/unittests/test_wav_reader_chunk_bounds.cpp)
add_test(NAME wav_reader_chunk_bounds_test COMMAND wav_reader_chunk_bounds_test)

# wav_writer_formats
add_engine_unittest(wav_writer_formats_test tests/unittests/test_wav_writer_formats.cpp)
add_test(NAME wav_writer_formats_test COMMAND wav_writer_formats_test)

# weight_adapter
add_engine_unittest(weight_adapter_test tests/unittests/test_weight_adapter.cpp)
add_test(NAME weight_adapter_test COMMAND weight_adapter_test --log)

# weight_adapter_source
add_engine_unittest(weight_adapter_source_test tests/unittests/test_weight_adapter_source.cpp)
add_test(NAME weight_adapter_source_test COMMAND weight_adapter_source_test --log)

# whistle_asr
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

# yue2
if (yue2 IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_unittest(yue2_attention_layout_test tests/yue2/attention_layout_test.cpp)
    add_test(NAME yue2_attention_layout_test COMMAND yue2_attention_layout_test)
    add_engine_unittest(yue2_nar_conditioning_test tests/yue2/nar_conditioning_test.cpp)
    add_test(NAME yue2_nar_conditioning_test COMMAND yue2_nar_conditioning_test)
    add_engine_unittest(yue2_nar_windowing_test tests/yue2/nar_windowing_test.cpp)
    add_test(NAME yue2_nar_windowing_test COMMAND yue2_nar_windowing_test)
    add_engine_unittest(yue2_request_test tests/yue2/request_test.cpp)
    add_test(NAME yue2_request_test COMMAND yue2_request_test)
endif()

# zipformer_modules
add_engine_unittest(zipformer_modules_test tests/unittests/test_zipformer_modules.cpp)
add_test(NAME zipformer_modules_test COMMAND zipformer_modules_test)

# zonos3
if (zonos3 IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_unittest(zonos3_text_test tests/community_models/zonos3/text_test.cpp)
    add_test(NAME zonos3_text_test COMMAND zonos3_text_test)
endif()
