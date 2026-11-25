# Checkpoint-dependent model tests and manual model parity probes.
# Included only when ENGINE_BUILD_MODEL_TESTS is enabled.

# ace_step
if (ace_step IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(ace_step_reference_audio_test tests/ace_step/reference_audio.cpp --log)
    add_engine_test_target(ace_step_timbre_cache_probe tests/ace_step/timbre_cache_probe.cpp)
endif()

# asr_standalone_gguf
if (hviske_asr IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(asr_standalone_gguf_test tests/unittests/test_asr_standalone_gguf.cpp)
    target_sources(asr_standalone_gguf_test PRIVATE src/models/hviske_asr/assets.cpp)
    target_compile_definitions(asr_standalone_gguf_test PRIVATE
        ENGINE_TEST_ASSET_ROOT="${ENGINE_UNITTEST_ASSET_ROOT}"
    )
endif()

# audio8_asr
if (audio8_asr IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(test_audio8_asr_golden_transcription
        tests/audio8_asr/test_audio8_asr_golden_transcription.cpp
    )
    target_compile_definitions(test_audio8_asr_golden_transcription PRIVATE
        ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    target_link_libraries(test_audio8_asr_golden_transcription PRIVATE engine_runtime ggml)
    target_include_directories(test_audio8_asr_golden_transcription PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(test_audio8_asr_golden_transcription PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_executable(test_audio8_asr_units
        tests/audio8_asr/test_audio8_asr_units.cpp
    )
    target_link_libraries(test_audio8_asr_units PRIVATE engine_runtime ggml)
    target_include_directories(test_audio8_asr_units PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(test_audio8_asr_units PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_test(
        NAME test_audio8_asr_units
        COMMAND test_audio8_asr_units
    )
endif()

# auk
if (auk IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test_target(auk_conditioning_test tests/auk/conditioning_test.cpp)
    target_sources(auk_conditioning_test PRIVATE
        src/community_models/auk/audio_conditioning.cpp
        src/community_models/auk/conditioning.cpp)
    add_engine_test_target(auk_flow_test tests/auk/flow_test.cpp)
    target_sources(auk_flow_test PRIVATE
        src/community_models/auk/audio_conditioning.cpp
        src/community_models/auk/conditioning.cpp
        src/community_models/auk/flow.cpp
        src/community_models/auk/session.cpp
        src/community_models/auk/vae.cpp)
    target_link_libraries(auk_flow_test PRIVATE yaml_vendor)
    add_engine_test_target(auk_vae_test tests/auk/vae_test.cpp)
    target_sources(auk_vae_test PRIVATE src/community_models/auk/vae.cpp)
endif()

# c_api
if (AUDIOCPP_BUILD_C_API)
    if (GGML_BACKEND_DL)
        set(AUDIOCPP_C_API_LAYOUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/capi-library-layout")
        set(AUDIOCPP_C_API_LAYOUT_ENV
            "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:audiocpp>:$ENV{LD_LIBRARY_PATH}")
        if (WIN32)
            set(AUDIOCPP_C_API_LAYOUT_ENV "PATH=$<TARGET_FILE_DIR:audiocpp>;$ENV{PATH}")
        endif()
        if ((APPLE) AND NOT ((WIN32)))
            set(AUDIOCPP_C_API_LAYOUT_ENV
                "DYLD_LIBRARY_PATH=$<TARGET_FILE_DIR:audiocpp>:$ENV{DYLD_LIBRARY_PATH}")
        endif()
    endif()
    include(ProcessorCount)
    ProcessorCount(AUDIOCPP_C_API_TEST_THREADS)
    if (AUDIOCPP_C_API_TEST_THREADS EQUAL 0)
        set(AUDIOCPP_C_API_TEST_THREADS 4)
    endif()
    if (GGML_BACKEND_DL)
        add_test(NAME audiocpp_c_api_library_layout
            COMMAND ${CMAKE_COMMAND} -E env
            "${AUDIOCPP_C_API_LAYOUT_ENV}"
            "${AUDIOCPP_C_API_LAYOUT_DIR}/$<TARGET_FILE_NAME:audiocpp_c_api_path_test>"
            "${CMAKE_CURRENT_SOURCE_DIR}/assets/framework/models/silero_vad"
            "${CMAKE_CURRENT_SOURCE_DIR}/assets/resources/sample_16k.wav")
        set_tests_properties(audiocpp_c_api_library_layout PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
    if (Python3_Interpreter_FOUND)
        add_test(NAME audiocpp_c_api_parity
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/capi/parity.py"
            --build-dir "${CMAKE_BINARY_DIR}"
            --models-root "${AUDIOCPP_C_API_MODEL_ROOT}"
            --audio "${CMAKE_CURRENT_SOURCE_DIR}/assets/resources/sample_16k.wav"
            --threads "${AUDIOCPP_C_API_TEST_THREADS}")
        set_tests_properties(audiocpp_c_api_parity PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    add_executable(audiocpp_c_api_model_test tests/capi/model_test.c)
    target_link_libraries(audiocpp_c_api_model_test PRIVATE audiocpp)
    set_target_properties(audiocpp_c_api_model_test PROPERTIES C_STANDARD 99)
    add_test(NAME audiocpp_c_api_model
        COMMAND audiocpp_c_api_model_test
        "${AUDIOCPP_C_API_MODEL_ROOT}"
        "${CMAKE_CURRENT_SOURCE_DIR}/assets/resources/sample_16k.wav"
        "cpu"
        ""
        "${AUDIOCPP_C_API_TEST_THREADS}")
    set_tests_properties(audiocpp_c_api_model PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(audiocpp_c_api_path_test tests/capi/path_test.c)
    target_link_libraries(audiocpp_c_api_path_test PRIVATE audiocpp)
    set_target_properties(audiocpp_c_api_path_test PROPERTIES C_STANDARD 99)
    add_test(NAME audiocpp_c_api_path
        COMMAND audiocpp_c_api_path_test
        "${CMAKE_CURRENT_SOURCE_DIR}/assets/framework/models/silero_vad"
        "${CMAKE_CURRENT_SOURCE_DIR}/assets/resources/sample_16k.wav")
    if (GGML_BACKEND_DL)
        add_custom_command(TARGET audiocpp_c_api_path_test POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${AUDIOCPP_C_API_LAYOUT_DIR}"
            COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:audiocpp_c_api_path_test>"
            "${AUDIOCPP_C_API_LAYOUT_DIR}/$<TARGET_FILE_NAME:audiocpp_c_api_path_test>")
    endif()
endif()

# causal_decoder_chunked_prefill
add_engine_test(causal_decoder_chunked_prefill_test tests/unittests/test_causal_decoder_chunked_prefill.cpp)

# confucius4_r2t2
if (confucius4_r2t2 IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(test_confucius4_r2t2_graph_reuse tests/confucius4_r2t2/test_graph_reuse.cpp)
    target_compile_definitions(test_confucius4_r2t2_graph_reuse PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(test_confucius4_r2t2_graph_reuse PRIVATE engine_runtime ggml)
    add_executable(test_confucius4_r2t2_transcription
        tests/confucius4_r2t2/test_confucius4_r2t2_transcription.cpp
    )
    target_compile_definitions(test_confucius4_r2t2_transcription PRIVATE
        ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    target_link_libraries(test_confucius4_r2t2_transcription PRIVATE engine_runtime ggml)
    target_include_directories(test_confucius4_r2t2_transcription PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(test_confucius4_r2t2_transcription PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# decoder_packed_projection
add_engine_test(decoder_packed_projection_test tests/unittests/test_decoder_packed_projections.cpp)

# dettato
if (dettato IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test_target(dettato_components tests/dettato/components.cpp)
endif()

# dots_tts
if (dots_tts IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test_target(dots_tts_flow_parity tests/dots_tts/dots_tts_flow_parity.cpp)
    add_engine_test_target(dots_tts_vocoder_parity tests/dots_tts/dots_tts_vocoder_parity.cpp)
endif()

# earshot_vad
if (earshot_vad IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(earshot_vad_probe tests/earshot_vad/probe.cpp)
    target_link_libraries(earshot_vad_probe PRIVATE engine_runtime ggml)
    add_engine_test(earshot_vad_test tests/unittests/test_earshot_vad.cpp)
    target_compile_definitions(earshot_vad_test PRIVATE
        ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
        ENGINE_TEST_ASSET_ROOT="${ENGINE_UNITTEST_ASSET_ROOT}"
    )
endif()

# echo_tts
if (echo_tts IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test_target(echo_tts_dit_parity tests/echo_tts/echo_tts_dit_parity.cpp)
    add_engine_test(echo_tts_host_units tests/echo_tts/echo_tts_host_units.cpp)
    target_sources(echo_tts_host_units PRIVATE
        src/community_models/echo_tts/latent_post.cpp
        src/community_models/echo_tts/tokenizer.cpp)
endif()

# espeak
if (sanotts IN_LIST AUDIOCPP_LINKED_MODELS AND inflect_v2 IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test_target(espeak_frontend_probe tests/unittests/espeak_frontend_probe.cpp)
    if (AUDIOCPP_STATIC_ESPEAK)
        audiocpp_stage_espeak(espeak_frontend_probe)
        add_test(NAME espeak_static_frontend_test COMMAND espeak_frontend_probe - - concurrent)
    endif()
    if (MSVC)
        target_compile_options(espeak_frontend_probe PRIVATE /utf-8)
    endif()
endif()

# f5_tts
if (f5_tts IN_LIST AUDIOCPP_LINKED_MODELS)
    target_compile_definitions(engine_model_f5_tts PRIVATE F5_MEL_TEST=1)
    add_executable(f5_cfg_parity tests/f5_tts/f5_cfg_parity_main.cpp)
    target_compile_definitions(f5_cfg_parity PRIVATE F5_MEL_TEST=1)
    target_link_libraries(f5_cfg_parity PRIVATE engine_runtime ggml)
    target_include_directories(f5_cfg_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(f5_cfg_parity PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_executable(f5_e2e tests/f5_tts/f5_e2e_main.cpp)
    target_compile_definitions(f5_e2e PRIVATE F5_MEL_TEST=1)
    target_link_libraries(f5_e2e PRIVATE engine_runtime ggml)
    target_include_directories(f5_e2e PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(f5_e2e PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_executable(f5_parity tests/f5_tts/f5_parity_main.cpp)
    target_compile_definitions(f5_parity PRIVATE F5_MEL_TEST=1)
    target_link_libraries(f5_parity PRIVATE engine_runtime ggml)
    target_include_directories(f5_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(f5_parity PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_executable(f5_tokenizer tests/f5_tts/f5_tokenizer_main.cpp)
    target_compile_definitions(f5_tokenizer PRIVATE F5_MEL_TEST=1)
    target_link_libraries(f5_tokenizer PRIVATE engine_runtime ggml)
    target_include_directories(f5_tokenizer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(f5_tokenizer PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# firered_vad
if (firered_vad IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(firered_vad_probe tests/firered_vad/probe.cpp)
    target_link_libraries(firered_vad_probe PRIVATE engine_runtime ggml)
endif()

# fish_drama3
if (fish_drama3 IN_LIST AUDIOCPP_LINKED_MODELS)
    set(AUDIOCPP_DRAMA3_TOKENIZER_DIR "" CACHE PATH "Optional Drama 3 tokenizer sidecars for reference tests")
    add_engine_test(fish_drama3_backbone_allocation_test
        tests/fish_drama3/backbone_allocation_test.cpp)
    target_include_directories(fish_drama3_backbone_allocation_test PRIVATE external/ggml/src)
    add_engine_test(fish_drama3_codec_source_test tests/fish_drama3/codec_source_test.cpp)
    add_engine_test_target(fish_drama3_tokenizer_test tests/fish_drama3/tokenizer_test.cpp)
    if (AUDIOCPP_DRAMA3_TOKENIZER_DIR)
        add_test(NAME fish_drama3_tokenizer_test COMMAND fish_drama3_tokenizer_test
            "${AUDIOCPP_DRAMA3_TOKENIZER_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/fish_drama3/tokenizer_vectors.json")
    endif()
endif()

# flashsr_utility
add_engine_test(flashsr_utility_test tests/unittests/test_flashsr_utility.cpp)
target_compile_definitions(flashsr_utility_test PRIVATE
    ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    ENGINE_TEST_ASSET_ROOT="${ENGINE_UNITTEST_ASSET_ROOT}"
)

# fun_asr_nano
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
add_engine_test_target(fun_asr_nano_adaptor_probe tests/fun_asr_nano/fun_asr_nano_adaptor_probe.cpp)
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
add_engine_test(fun_asr_nano_assets_test tests/unittests/test_fun_asr_nano_assets.cpp)
target_sources(fun_asr_nano_assets_test PRIVATE src/models/fun_asr_nano/assets.cpp)
add_engine_test_target(fun_asr_nano_decoder_probe tests/fun_asr_nano/fun_asr_nano_decoder_probe.cpp)
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
add_engine_test_target(fun_asr_nano_encoder_probe tests/fun_asr_nano/fun_asr_nano_encoder_probe.cpp)
target_sources(
    fun_asr_nano_encoder_probe
    PRIVATE
    src/models/fun_asr_nano/assets.cpp
    src/models/fun_asr_nano/frontend.cpp
    src/models/fun_asr_nano/encoder.cpp
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
endif()
add_engine_test(fun_asr_nano_frontend_probe
    tests/fun_asr_nano/fun_asr_nano_frontend_probe.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/fun_asr_nano/frontend_reference.json ${CMAKE_CURRENT_SOURCE_DIR}/tests/fun_asr_nano/frontend_reference.bin ${CMAKE_CURRENT_SOURCE_DIR}/assets/resources/sample_16k.wav)
add_engine_test(fun_asr_nano_sanm_probe
    tests/fun_asr_nano/sanm_probe.cpp
    --backend cpu --reference ${CMAKE_CURRENT_SOURCE_DIR}/tests/fun_asr_nano/sanm_reference.json --data ${CMAKE_CURRENT_SOURCE_DIR}/tests/fun_asr_nano/sanm_reference.bin)
if (AUDIOCPP_FUN_ASR_NANO_TEST_MODEL)
    add_engine_test_target(fun_asr_nano_session_probe tests/fun_asr_nano/fun_asr_nano_session_probe.cpp)
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

# granite5asr
add_executable(test_granite5asr_golden_transcription
    tests/granite5asr/test_granite5asr_golden_transcription.cpp
)
target_compile_definitions(test_granite5asr_golden_transcription PRIVATE
    ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(test_granite5asr_golden_transcription PRIVATE engine_runtime ggml)
target_include_directories(test_granite5asr_golden_transcription PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(test_granite5asr_golden_transcription PRIVATE OpenMP::OpenMP_CXX)
endif()

# granite_speech5
if (granite_speech5 IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(granite_speech5_b1_units_test
        tests/granite_speech5/test_granite_speech5_b1_units.cpp)
    target_link_libraries(granite_speech5_b1_units_test PRIVATE granite_speech5_test_impl)
    target_include_directories(granite_speech5_b1_units_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    target_compile_definitions(granite_speech5_b1_units_test PRIVATE AUDIOCPP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME granite_speech5_b1_units_test COMMAND granite_speech5_b1_units_test)
    add_executable(granite_speech5_b2_units_test
        tests/granite_speech5/test_granite_speech5_b2_units.cpp)
    target_link_libraries(granite_speech5_b2_units_test PRIVATE granite_speech5_test_impl)
    target_include_directories(granite_speech5_b2_units_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    target_compile_definitions(granite_speech5_b2_units_test PRIVATE AUDIOCPP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME granite_speech5_b2_units_test COMMAND granite_speech5_b2_units_test)
    add_executable(granite_speech5_c1_units_test
        tests/granite_speech5/test_granite_speech5_c1_units.cpp)
    target_link_libraries(granite_speech5_c1_units_test PRIVATE granite_speech5_test_impl)
    target_include_directories(granite_speech5_c1_units_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    target_compile_definitions(granite_speech5_c1_units_test PRIVATE AUDIOCPP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME granite_speech5_c1_units_test COMMAND granite_speech5_c1_units_test)
    add_executable(granite_speech5_c2_units_test
        tests/granite_speech5/test_granite_speech5_c2_units.cpp)
    target_link_libraries(granite_speech5_c2_units_test PRIVATE granite_speech5_test_impl)
    target_include_directories(granite_speech5_c2_units_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    target_compile_definitions(granite_speech5_c2_units_test PRIVATE AUDIOCPP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME granite_speech5_c2_units_test COMMAND granite_speech5_c2_units_test)
    add_executable(granite_speech5_d_units_test
        tests/granite_speech5/test_granite_speech5_d_units.cpp)
    target_link_libraries(granite_speech5_d_units_test PRIVATE granite_speech5_test_impl)
    target_include_directories(granite_speech5_d_units_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    target_compile_definitions(granite_speech5_d_units_test PRIVATE AUDIOCPP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME granite_speech5_d_units_test COMMAND granite_speech5_d_units_test)
    add_executable(granite_speech5_f_units_test
        tests/granite_speech5/test_granite_speech5_f_units.cpp)
    target_link_libraries(granite_speech5_f_units_test PRIVATE granite_speech5_test_impl)
    target_include_directories(granite_speech5_f_units_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    target_compile_definitions(granite_speech5_f_units_test PRIVATE AUDIOCPP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME granite_speech5_f_units_test COMMAND granite_speech5_f_units_test)
    add_executable(granite_speech5_parity_probe
        tests/granite_speech5/granite_speech5_parity_probe.cpp)
    target_link_libraries(granite_speech5_parity_probe PRIVATE granite_speech5_test_impl)
endif()

# gtcrn_utility
add_engine_test(gtcrn_utility_test tests/unittests/test_gtcrn_utility.cpp)
target_compile_definitions(gtcrn_utility_test PRIVATE
    ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    ENGINE_TEST_ASSET_ROOT="${ENGINE_UNITTEST_ASSET_ROOT}"
)

# heartmula
if (heartmula IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(heartmula_codec_lifecycle tests/heartmula/codec_lifecycle.cpp)
    target_link_libraries(heartmula_codec_lifecycle PRIVATE engine_runtime ggml)
endif()

# hifigan_vocoder_variant_parity
add_executable(hifigan_vocoder_variant_parity
    tests/unittests/test_hifigan_vocoder_variants.cpp
)
target_link_libraries(hifigan_vocoder_variant_parity PRIVATE engine_runtime ggml)
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(hifigan_vocoder_variant_parity PRIVATE OpenMP::OpenMP_CXX)
endif()

# higgs_audio_tts
if (higgs_audio_tts IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(higgs_audio_tts_sampler_test tests/higgs_audio_tts/test_sampler.cpp --log)
endif()

# index_tts2
if (index_tts2 IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test_target(index_tts2_text_controls_probe tests/index_tts2/text_controls_probe.cpp)
endif()

# index_tts_tn_corpus
add_engine_test(index_tts_tn_corpus_test tests/unittests/test_index_tts_tn_corpus.cpp)
target_compile_definitions(index_tts_tn_corpus_test PRIVATE
    ENGINE_TEST_ASSET_ROOT="${ENGINE_UNITTEST_ASSET_ROOT}"
)

# inflect_v2
if (inflect_v2 IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(inflect_v2_frontend_test tests/unittests/test_inflect_v2_frontend.cpp)
endif()

# irodori_tts
if (irodori_tts IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(irodori_schedule_test tests/irodori_tts/test_schedule.cpp)
    target_link_libraries(irodori_schedule_test PRIVATE engine_runtime ggml)
    add_test(NAME irodori_schedule_test COMMAND irodori_schedule_test --log)
endif()

# kitten_tts
if ("kitten_tts" IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(kitten_tts_mini_parity_test tests/kitten_tts/test_tts2_mini_parity.cpp)
    target_link_libraries(kitten_tts_mini_parity_test PRIVATE engine_runtime)
    audiocpp_use_utf8_codepage(kitten_tts_mini_parity_test)
endif()

# kitten_tts2
if ("kitten_tts2" IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(kitten_tts2_components_probe tests/kitten_tts2/components_probe.cpp)
    audiocpp_use_utf8_codepage(kitten_tts2_components_probe)
    target_link_libraries(kitten_tts2_components_probe PRIVATE engine_runtime)
    add_executable(kitten_tts2_session_probe tests/kitten_tts2/session_probe.cpp)
    audiocpp_use_utf8_codepage(kitten_tts2_session_probe)
    target_link_libraries(kitten_tts2_session_probe PRIVATE engine_runtime)
endif()

# kokoro_tts
if (kokoro_tts IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(kokoro_cpu_kernel_test tests/kokoro_tts/kokoro_cpu_kernel_test.cpp)
    add_engine_test_target(kokoro_g2p_probe tests/kokoro_tts/kokoro_g2p_probe.cpp)
    if (AUDIOCPP_STATIC_ESPEAK)
        audiocpp_stage_espeak(kokoro_g2p_probe)
    endif()
endif()

# kugelaudio
if (kugelaudio IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test_target(kugelaudio_cloning_session_probe tests/kugelaudio/cloning_session_probe.cpp)
    add_engine_test_target(kugelaudio_reference_encoder_probe tests/kugelaudio/reference_encoder_probe.cpp)
endif()

# latentsync
if (latentsync IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(latentsync_component_parity tests/latentsync/component_parity.cpp)
    target_link_libraries(latentsync_component_parity PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(latentsync_component_parity PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_executable(latentsync_graph_ops_test tests/latentsync/graph_ops.cpp)
    target_link_libraries(latentsync_graph_ops_test PRIVATE engine_runtime ggml)
    add_test(NAME latentsync_graph_ops_cpu COMMAND latentsync_graph_ops_test cpu)
    if (GGML_METAL)
        add_test(NAME latentsync_graph_ops_metal COMMAND latentsync_graph_ops_test metal)
    endif()
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(latentsync_graph_ops_test PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# lfm2_audio
if (lfm2_audio IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(lfm2_audio_backbone_test tests/lfm2_audio/lfm2_audio_backbone_test.cpp)
    target_compile_definitions(lfm2_audio_backbone_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    add_engine_test(lfm2_audio_chat_test tests/lfm2_audio/lfm2_audio_chat_test.cpp)
    target_compile_definitions(lfm2_audio_chat_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    if (ENGINE_ENABLE_CUDA)
        add_engine_test(lfm2_audio_cuda_test tests/lfm2_audio/lfm2_audio_cuda_test.cpp)
        set_tests_properties(lfm2_audio_cuda_test PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    add_engine_test(lfm2_audio_encoder_test tests/lfm2_audio/lfm2_audio_encoder_test.cpp)
    target_compile_definitions(lfm2_audio_encoder_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME lfm2_audio_encoder_cpu_repack_test
        COMMAND lfm2_audio_encoder_test --cpu-repack-only)
    add_engine_test(lfm2_audio_session_test tests/lfm2_audio/lfm2_audio_session_test.cpp)
    target_compile_definitions(lfm2_audio_session_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    add_executable(test_lfm2_audio_asr tests/lfm2_audio/test_lfm2_audio_asr.cpp)
    target_compile_definitions(test_lfm2_audio_asr PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(test_lfm2_audio_asr PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(test_lfm2_audio_asr PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_test(NAME test_lfm2_audio_asr COMMAND test_lfm2_audio_asr)
    set_tests_properties(test_lfm2_audio_asr PROPERTIES
        SKIP_RETURN_CODE 125
        TIMEOUT 600
    )
    add_executable(test_lfm2_audio_s2s tests/lfm2_audio/test_lfm2_audio_s2s.cpp)
    target_compile_definitions(test_lfm2_audio_s2s PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(test_lfm2_audio_s2s PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(test_lfm2_audio_s2s PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_test(NAME test_lfm2_audio_s2s COMMAND test_lfm2_audio_s2s)
    set_tests_properties(test_lfm2_audio_s2s PROPERTIES
        SKIP_RETURN_CODE 125
        TIMEOUT 900
    )
    add_executable(test_lfm2_audio_tts tests/lfm2_audio/test_lfm2_audio_tts.cpp)
    target_compile_definitions(test_lfm2_audio_tts PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(test_lfm2_audio_tts PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(test_lfm2_audio_tts PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_test(NAME test_lfm2_audio_tts COMMAND test_lfm2_audio_tts)
    set_tests_properties(test_lfm2_audio_tts PROPERTIES
        SKIP_RETURN_CODE 125
        TIMEOUT 900
    )
endif()

# lfm3_audio
if (lfm3_audio IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(lfm3_audio_backbone_test tests/lfm3_audio/lfm3_audio_backbone_test.cpp)
    target_compile_definitions(lfm3_audio_backbone_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(lfm3_audio_backbone_test PROPERTIES SKIP_RETURN_CODE 77)
    add_engine_test(lfm3_audio_blocks_test tests/lfm3_audio/lfm3_audio_blocks_test.cpp)
    target_compile_definitions(lfm3_audio_blocks_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(lfm3_audio_blocks_test PROPERTIES SKIP_RETURN_CODE 77)
    add_engine_test(lfm3_audio_codec_test tests/lfm3_audio/lfm3_audio_codec_test.cpp)
    target_compile_definitions(lfm3_audio_codec_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(lfm3_audio_codec_test PROPERTIES SKIP_RETURN_CODE 77)
    add_engine_test(lfm3_audio_depth_test tests/lfm3_audio/lfm3_audio_depth_test.cpp)
    target_compile_definitions(lfm3_audio_depth_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(lfm3_audio_depth_test PROPERTIES SKIP_RETURN_CODE 77)
    add_engine_test(lfm3_audio_encoder_test tests/lfm3_audio/lfm3_audio_encoder_test.cpp)
    target_compile_definitions(lfm3_audio_encoder_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(lfm3_audio_encoder_test PROPERTIES SKIP_RETURN_CODE 77)
    add_engine_test(lfm3_audio_generator_test tests/lfm3_audio/lfm3_audio_generator_test.cpp)
    target_compile_definitions(lfm3_audio_generator_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(lfm3_audio_generator_test PROPERTIES SKIP_RETURN_CODE 77)
    add_engine_test(lfm3_audio_session_test tests/lfm3_audio/lfm3_audio_session_test.cpp)
    target_compile_definitions(lfm3_audio_session_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(lfm3_audio_session_test PROPERTIES SKIP_RETURN_CODE 77)
    add_engine_test(lfm3_audio_tokenizer_test tests/lfm3_audio/lfm3_audio_tokenizer_test.cpp)
    target_compile_definitions(lfm3_audio_tokenizer_test PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(lfm3_audio_tokenizer_test PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(test_lfm3_audio_asr tests/lfm3_audio/test_lfm3_audio_asr.cpp)
    target_compile_definitions(test_lfm3_audio_asr PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(test_lfm3_audio_asr PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(test_lfm3_audio_asr PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_test(NAME test_lfm3_audio_asr COMMAND test_lfm3_audio_asr)
    set_tests_properties(test_lfm3_audio_asr PROPERTIES
        SKIP_RETURN_CODE 125
        TIMEOUT 900
    )
    add_executable(test_lfm3_audio_parity tests/lfm3_audio/test_lfm3_audio_parity.cpp)
    target_compile_definitions(test_lfm3_audio_parity PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(test_lfm3_audio_parity PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(test_lfm3_audio_parity PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_test(NAME test_lfm3_audio_parity COMMAND test_lfm3_audio_parity --skip-long)
    set_tests_properties(test_lfm3_audio_parity PROPERTIES
        SKIP_RETURN_CODE 125
        TIMEOUT 900
    )
    add_executable(test_lfm3_audio_s2s tests/lfm3_audio/test_lfm3_audio_s2s.cpp)
    target_compile_definitions(test_lfm3_audio_s2s PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(test_lfm3_audio_s2s PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(test_lfm3_audio_s2s PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_test(NAME test_lfm3_audio_s2s COMMAND test_lfm3_audio_s2s)
    set_tests_properties(test_lfm3_audio_s2s PROPERTIES
        SKIP_RETURN_CODE 125
        TIMEOUT 900
    )
    add_executable(test_lfm3_audio_tts tests/lfm3_audio/test_lfm3_audio_tts.cpp)
    target_compile_definitions(test_lfm3_audio_tts PRIVATE ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(test_lfm3_audio_tts PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(test_lfm3_audio_tts PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_test(NAME test_lfm3_audio_tts COMMAND test_lfm3_audio_tts)
    set_tests_properties(test_lfm3_audio_tts PROPERTIES
        SKIP_RETURN_CODE 125
        TIMEOUT 900
    )
endif()

# lfm_decoder
add_engine_test(lfm_decoder_test
    tests/unittests/test_lfm_decoder.cpp
    --backend cpu --log "${CMAKE_CURRENT_BINARY_DIR}/lfm_decoder_test.log")

# m5_tts
if (m5_tts IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(m5_tts_components tests/m5_tts/components.cpp app/cli/args.cpp)
    target_link_libraries(m5_tts_components PRIVATE engine_runtime ggml)
    target_include_directories(m5_tts_components PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# magenta_realtime_2
if (magenta_realtime_2 IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(magenta_codec_parity tests/magenta_realtime_2/codec_parity.cpp)
    target_link_libraries(magenta_codec_parity PRIVATE engine_runtime ggml)
    add_executable(magenta_greedy_parity tests/magenta_realtime_2/greedy_parity.cpp)
    target_link_libraries(magenta_greedy_parity PRIVATE engine_runtime ggml)
    add_executable(magenta_musiccoca_parity tests/magenta_realtime_2/musiccoca_parity.cpp)
    target_link_libraries(magenta_musiccoca_parity PRIVATE engine_runtime ggml)
    add_executable(magenta_validation tests/magenta_realtime_2/validation.cpp)
    target_link_libraries(magenta_validation PRIVATE engine_runtime ggml)
endif()

# mimi_latent_decoder_lifecycle
if (pocket_tts_small IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(mimi_latent_decoder_lifecycle tests/codecs/mimi_latent_decoder_lifecycle.cpp app/cli/args.cpp)
    target_link_libraries(mimi_latent_decoder_lifecycle PRIVATE engine_runtime ggml)
    target_include_directories(mimi_latent_decoder_lifecycle PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# ming3_audio
if ("ming3_audio" IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(ming3_audio_talker_parity EXCLUDE_FROM_ALL tests/ming3_audio/talker_parity.cpp)
    target_link_libraries(ming3_audio_talker_parity PRIVATE engine_runtime ggml)
    add_executable(ming3_audio_thinker_parity EXCLUDE_FROM_ALL tests/ming3_audio/thinker_parity.cpp)
    target_link_libraries(ming3_audio_thinker_parity PRIVATE engine_runtime ggml)
endif()

# ming3_tts
if ("ming3_tts" IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(ming3_tts_component_parity EXCLUDE_FROM_ALL tests/ming3_tts/component_parity.cpp)
    target_link_libraries(ming3_tts_component_parity PRIVATE engine_runtime ggml)
endif()

# minimax_music3
add_engine_test(minimax_music3_graph_release_policy_test
    tests/unittests/test_minimax_music3_graph_release_policy.cpp)
if (minimax_music3 IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(minimax_music3_lm_head_test tests/unittests/test_minimax_music3_lm_head.cpp)
    target_sources(minimax_music3_lm_head_test PRIVATE src/community_models/minimax_music3/global_lm.cpp)
    add_engine_test(minimax_music3_pipeline_buffers_test tests/unittests/test_minimax_music3_pipeline_buffers.cpp)
endif()

# miocodec
add_executable(miocodec_wavlm_parity EXCLUDE_FROM_ALL
    tests/miocodec/miocodec_wavlm_parity.cpp
)
target_link_libraries(miocodec_wavlm_parity PRIVATE engine_runtime ggml)

# mms_forced_aligner
add_engine_test(mms_ctc_alignment_test tests/unittests/test_mms_ctc_alignment.cpp)
target_sources(mms_ctc_alignment_test PRIVATE src/community_models/mms_forced_aligner/ctc_alignment.cpp)
add_engine_test(mms_emissions_test tests/unittests/test_mms_emissions.cpp)
target_sources(mms_emissions_test PRIVATE src/community_models/mms_forced_aligner/emissions.cpp)
add_engine_test(mms_forced_aligner_assets_test tests/unittests/test_mms_forced_aligner_assets.cpp)
target_sources(mms_forced_aligner_assets_test PRIVATE src/community_models/mms_forced_aligner/assets.cpp)
if (mms_forced_aligner IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(mms_forced_aligner_session_test tests/unittests/test_mms_forced_aligner_session.cpp)
    target_sources(mms_forced_aligner_session_test PRIVATE
        src/community_models/mms_forced_aligner/assets.cpp
        src/community_models/mms_forced_aligner/ctc_alignment.cpp
        src/community_models/mms_forced_aligner/emissions.cpp
        src/community_models/mms_forced_aligner/session.cpp
        src/community_models/mms_forced_aligner/text_processor.cpp)
    target_include_directories(mms_forced_aligner_session_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests
        ${CMAKE_CURRENT_SOURCE_DIR}/external/llama_tokenizer)
endif()
add_engine_test(mms_text_processor_test tests/unittests/test_mms_text_processor.cpp)
target_sources(mms_text_processor_test PRIVATE src/community_models/mms_forced_aligner/text_processor.cpp)
target_include_directories(mms_text_processor_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests
    ${CMAKE_CURRENT_SOURCE_DIR}/external/llama_tokenizer)
add_engine_test(mms_wav2vec2_positional_conv_test tests/unittests/test_mms_wav2vec2_positional_conv.cpp)

# moss
add_executable(moss_codec_decode_length_sweep
    tests/moss_tts_delay/codec_decode_length_sweep.cpp
)
target_link_libraries(moss_codec_decode_length_sweep PRIVATE engine_runtime ggml)
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(moss_codec_decode_length_sweep PRIVATE OpenMP::OpenMP_CXX)
endif()

# moss_soundeffect
if (moss_soundeffect IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(moss_soundeffect_parity tests/moss_soundeffect/validate_reference.cpp)
    target_link_libraries(moss_soundeffect_parity PRIVATE engine_runtime ggml)
endif()

# moss_tts_delay
add_executable(moss_tts_delay_backbone_parity
    tests/moss_tts_delay/backbone_parity.cpp
)
target_link_libraries(moss_tts_delay_backbone_parity PRIVATE engine_runtime ggml)
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(moss_tts_delay_backbone_parity PRIVATE OpenMP::OpenMP_CXX)
endif()

# moss_tts_local
if (moss IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test_target(moss_tts_local_codec_decode_parity tests/moss_tts_local/codec_decode_parity.cpp)
    add_engine_test_target(moss_tts_local_codec_encode_parity tests/moss_tts_local/codec_encode_parity.cpp)
    add_executable(moss_tts_local_smoke
        tests/moss_tts_local/moss_tts_local_smoke.cpp
    )
    target_link_libraries(moss_tts_local_smoke PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(moss_tts_local_smoke PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# moss_tts_nano
if (moss IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(moss_tts_nano_kv_cache_parity
        tests/moss_tts_nano/kv_cache_parity.cpp
    )
    target_link_libraries(moss_tts_nano_kv_cache_parity PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(moss_tts_nano_kv_cache_parity PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_executable(moss_tts_nano_reference_cache_test
        tests/moss_tts_nano/reference_cache_test.cpp
    )
    target_link_libraries(moss_tts_nano_reference_cache_test PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(moss_tts_nano_reference_cache_test PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_test(NAME moss_tts_nano_reference_cache_test COMMAND moss_tts_nano_reference_cache_test)
endif()

# moss_tts_v15
add_executable(moss_tts_v15_prompt_parity
    tests/moss_tts_v15/prompt_parity.cpp
    src/community_models/moss_tts_v15/prompt.cpp
)
target_link_libraries(moss_tts_v15_prompt_parity PRIVATE engine_runtime ggml)
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(moss_tts_v15_prompt_parity PRIVATE OpenMP::OpenMP_CXX)
endif()

# moss_ttsd
add_executable(moss_ttsd_prompt_parity
    tests/moss_ttsd/prompt_parity.cpp
    src/community_models/moss_ttsd/prompt.cpp
)
target_link_libraries(moss_ttsd_prompt_parity PRIVATE engine_runtime ggml)
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(moss_ttsd_prompt_parity PRIVATE OpenMP::OpenMP_CXX)
endif()

# moss_voicegen
if (moss_voicegen IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(moss_voicegen_backbone_parity
        tests/moss_voicegen/backbone_parity.cpp
    )
    target_link_libraries(moss_voicegen_backbone_parity PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(moss_voicegen_backbone_parity PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
add_executable(moss_voicegen_codec_encode_parity
    tests/moss_voicegen/codec_encode_parity.cpp
)
target_link_libraries(moss_voicegen_codec_encode_parity PRIVATE engine_runtime ggml)
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(moss_voicegen_codec_encode_parity PRIVATE OpenMP::OpenMP_CXX)
endif()
if (moss_voicegen IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(moss_voicegen_codec_parity
        tests/moss_voicegen/codec_decode_parity.cpp
    )
    target_link_libraries(moss_voicegen_codec_parity PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(moss_voicegen_codec_parity PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_executable(moss_voicegen_generation_parity
        tests/moss_voicegen/generation_parity.cpp
    )
    target_link_libraries(moss_voicegen_generation_parity PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(moss_voicegen_generation_parity PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_executable(moss_voicegen_prompt_parity
        tests/moss_voicegen/prompt_parity.cpp
    )
    target_link_libraries(moss_voicegen_prompt_parity PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(moss_voicegen_prompt_parity PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_executable(moss_voicegen_smoke
        tests/moss_voicegen/voicegen_smoke.cpp
    )
    target_link_libraries(moss_voicegen_smoke PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(moss_voicegen_smoke PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# mossformer2
if (mossformer2 IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(mossformer2_validation tests/mossformer2/validation.cpp)
    target_link_libraries(mossformer2_validation PRIVATE engine_runtime ggml)
endif()

# mulacover
if (mulacover IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(mulacover_codec tests/mulacover/codec.cpp)
    target_link_libraries(mulacover_codec PRIVATE engine_runtime ggml)
    add_executable(mulacover_components tests/mulacover/components.cpp)
    target_link_libraries(mulacover_components PRIVATE engine_runtime ggml)
    add_executable(mulacover_waveform tests/mulacover/waveform.cpp)
    target_link_libraries(mulacover_waveform PRIVATE engine_runtime ggml)
endif()

# nemo_mel_migration_probe
if (cohere_asr IN_LIST AUDIOCPP_LINKED_MODELS AND canary_asr IN_LIST AUDIOCPP_LINKED_MODELS AND citrinet_asr IN_LIST AUDIOCPP_LINKED_MODELS AND parakeet_tdt IN_LIST AUDIOCPP_LINKED_MODELS AND sortformer_diar IN_LIST AUDIOCPP_LINKED_MODELS AND sortformer_diar_v2 IN_LIST AUDIOCPP_LINKED_MODELS AND hviske_asr IN_LIST AUDIOCPP_LINKED_MODELS AND nemotron_asr IN_LIST AUDIOCPP_LINKED_MODELS AND granite5asr IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(nemo_mel_migration_probe tests/nemo_mel_migration/probe.cpp)
    target_link_libraries(nemo_mel_migration_probe PRIVATE engine_runtime ggml)
endif()

# nemotron_asr
if (ENGINE_ENABLE_WEBGPU AND nemotron_asr IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test_target(nemotron_asr_webgpu_rnnt_test tests/nemotron_asr/webgpu_rnnt_test.cpp)
    target_include_directories(nemotron_asr_webgpu_rnnt_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
endif()

# nemotron_transcribe
if (nemotron_transcribe IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(nemotron_transcribe_decoder_padding_test
        tests/nemotron_transcribe/decoder_padding_test.cpp)
    add_engine_test(nemotron_transcribe_decoder_safety_test
        tests/nemotron_transcribe/decoder_safety_test.cpp)
    target_include_directories(nemotron_transcribe_decoder_safety_test PRIVATE external/ggml/src)
    add_engine_test(nemotron_transcribe_encoder_bounds_test
        tests/nemotron_transcribe/encoder_bounds_test.cpp)
    target_include_directories(nemotron_transcribe_encoder_bounds_test PRIVATE external/ggml/src)
endif()

# omnivoice
if (omnivoice IN_LIST AUDIOCPP_LINKED_MODELS)
    set(AUDIOCPP_OMNIVOICE_TEST_MODEL "" CACHE PATH "Local OmniVoice model directory for integration tests")
    add_engine_test_target(omnivoice_batch_parity_test tests/omnivoice/batch_parity_test.cpp)
    if (AUDIOCPP_OMNIVOICE_TEST_MODEL)
        add_test(NAME omnivoice_batch_parity COMMAND omnivoice_batch_parity_test
            --model "${AUDIOCPP_OMNIVOICE_TEST_MODEL}"
            --clone-audio "${CMAKE_CURRENT_SOURCE_DIR}/assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0001.wav"
            --reference-text-file "${CMAKE_CURRENT_SOURCE_DIR}/assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0001.txt")
    endif()
endif()

# open_wake_word
if (open_wake_word IN_LIST AUDIOCPP_LINKED_MODELS)
    set(AUDIOCPP_OPEN_WAKE_WORD_TRACE "" CACHE FILEPATH "Official OpenWakeWord parity trace GGUF")
    add_engine_test_target(open_wake_word_frontend_test tests/open_wake_word/test_frontend.cpp)
    if (AUDIOCPP_OPEN_WAKE_WORD_TRACE)
        add_test(NAME open_wake_word_frontend_parity COMMAND open_wake_word_frontend_test
            "${AUDIOCPP_OPEN_WAKE_WORD_TRACE}")
    endif()
    add_engine_test_target(open_wake_word_runtime_test tests/open_wake_word/test_runtime_parity.cpp)
    if (AUDIOCPP_OPEN_WAKE_WORD_TRACE)
        add_test(NAME open_wake_word_runtime_parity COMMAND open_wake_word_runtime_test
            "${AUDIOCPP_OPEN_WAKE_WORD_TRACE}" 4)
    endif()
endif()

# outetts
if (outetts IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(outetts_generation_budget_test tests/unittests/test_outetts_generation_budget.cpp)
    target_sources(outetts_generation_budget_test PRIVATE src/community_models/outetts/tokenizer.cpp)
endif()

# parakeet_tdt
add_engine_test(parakeet_golden_transcription_test tests/parakeet_tdt/test_golden_transcription.cpp)
target_compile_definitions(parakeet_golden_transcription_test PRIVATE
    ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)
set_tests_properties(parakeet_golden_transcription_test PROPERTIES
    SKIP_RETURN_CODE 125
    TIMEOUT 120
)
if (parakeet_tdt IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(parakeet_parity_dump
        tests/parakeet_tdt/parity/dump_cpp_reference.cpp
    )
    target_include_directories(parakeet_parity_dump PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/parakeet_tdt/parity)
    target_link_libraries(parakeet_parity_dump PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(parakeet_parity_dump PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
add_engine_test(parakeet_streaming_transcription_test tests/parakeet_tdt/test_streaming_transcription.cpp)
target_compile_definitions(parakeet_streaming_transcription_test PRIVATE
    ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)
set_tests_properties(parakeet_streaming_transcription_test PROPERTIES
    SKIP_RETURN_CODE 125
    TIMEOUT 180
)
if (parakeet_tdt IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test_target(phonon_features_probe tests/parakeet_tdt/phonon_features_probe.cpp)
endif()

# personaplex
add_executable(personaplex_ring_kv_cache_parity
    tests/personaplex/personaplex_ring_kv_cache_parity.cpp
)
target_link_libraries(personaplex_ring_kv_cache_parity PRIVATE engine_runtime ggml)
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(personaplex_ring_kv_cache_parity PRIVATE OpenMP::OpenMP_CXX)
endif()

# plapre
if ("plapre" IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(plapre_codec_parity EXCLUDE_FROM_ALL tests/plapre/codec_parity.cpp)
    target_link_libraries(plapre_codec_parity PRIVATE engine_runtime ggml)
endif()

# pocket_tts
if (pocket_tts IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(pocket_tts_mimi_lifecycle tests/pocket_tts/mimi_lifecycle.cpp app/cli/args.cpp)
    target_link_libraries(pocket_tts_mimi_lifecycle PRIVATE engine_runtime ggml)
    target_include_directories(pocket_tts_mimi_lifecycle PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# pocket_tts_small
if (pocket_tts_small IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(pocket_tts_small_batch tests/pocket_tts_small/batch.cpp app/cli/args.cpp app/streaming/streaming.cpp)
    target_link_libraries(pocket_tts_small_batch PRIVATE engine_runtime ggml)
    target_include_directories(pocket_tts_small_batch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_executable(pocket_tts_small_reference tests/pocket_tts_small/reference.cpp app/cli/args.cpp)
    target_link_libraries(pocket_tts_small_reference PRIVATE engine_runtime ggml)
    target_include_directories(pocket_tts_small_reference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# qwen3_forced_aligner
if (qwen3_forced_aligner IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(qwen3_forced_aligner_processor_test tests/unittests/test_qwen3_forced_aligner_processor.cpp)
    target_sources(qwen3_forced_aligner_processor_test PRIVATE
        src/models/qwen3_asr/postprocess.cpp
        src/models/qwen3_asr/tokenizer_text.cpp
        src/models/qwen3_forced_aligner/nagisa_segmenter.cpp
        src/models/qwen3_forced_aligner/processor.cpp)
    target_link_libraries(qwen3_forced_aligner_processor_test PRIVATE utf8proc_vendor)
endif()

# qwen3_tts
if (qwen3_tts IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test_target(qwen3_tts_decoder_streaming_test tests/qwen3_tts/test_speech_decoder_streaming.cpp)
endif()

# realtime_reuse
if (realtime_reuse IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(test_realtime_reuse_reference tests/realtime_reuse/test_reference.cpp app/cli/args.cpp)
    target_link_libraries(test_realtime_reuse_reference PRIVATE engine_runtime ggml)
    target_include_directories(test_realtime_reuse_reference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_executable(test_realtime_reuse_streaming tests/realtime_reuse/test_streaming.cpp app/cli/args.cpp)
    target_link_libraries(test_realtime_reuse_streaming PRIVATE engine_runtime ggml)
    target_include_directories(test_realtime_reuse_streaming PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# reuse
if (reuse IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(test_reuse_scan tests/reuse/test_scan.cpp app/cli/args.cpp)
    target_link_libraries(test_reuse_scan PRIVATE engine_runtime ggml)
    target_include_directories(test_reuse_scan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# rknn_sidecar
if (ENGINE_ENABLE_RKNN)
    add_executable(rknn_sidecar_test tests/unittests/test_rknn_sidecar.cpp)
    target_compile_definitions(rknn_sidecar_test PRIVATE ENGINE_HAVE_RKNN=1)
    foreach(model IN LISTS AUDIOCPP_LINKED_MODELS)
        string(TOUPPER "${model}" model_upper)
        target_compile_definitions(rknn_sidecar_test PRIVATE
            "AUDIOCPP_TEST_HAS_MODEL_${model_upper}=1")
    endforeach()
    if (ENGINE_RKNN_INCLUDE_DIR)
        target_include_directories(rknn_sidecar_test PRIVATE ${ENGINE_RKNN_INCLUDE_DIR})
    endif()
    target_link_libraries(rknn_sidecar_test PRIVATE engine_runtime ggml)
    add_test(NAME rknn_sidecar_test COMMAND rknn_sidecar_test "${AUDIOCPP_RKNN_TEST_ENGINE}")
    set_tests_properties(rknn_sidecar_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

# rnnoise_utility
add_engine_test(rnnoise_utility_test tests/unittests/test_rnnoise_utility.cpp)
target_compile_definitions(rnnoise_utility_test PRIVATE
    ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    ENGINE_TEST_ASSET_ROOT="${ENGINE_UNITTEST_ASSET_ROOT}"
)

# rosvot
if (rosvot IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test_target(rosvot_components tests/rosvot/components.cpp)
endif()

# sanotts
if (sanotts IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(sanotts_neon_istft_test tests/unittests/test_sanotts_neon_istft.cpp)
    add_engine_test(sanotts_frontend_test tests/unittests/test_sanotts_frontend.cpp)
    target_sources(sanotts_frontend_test PRIVATE
        src/community_models/sanotts/frontend.cpp
        ${AUDIOCPP_SANOTTS_NEON_SOURCES}
        src/community_models/sanotts/runtime.cpp)
    add_engine_test(sanotts_neon_decoder_test tests/unittests/test_sanotts_neon_decoder.cpp)
    target_include_directories(sanotts_neon_decoder_test PRIVATE tests/unittests src/community_models/sanotts)
    add_engine_test(sanotts_word_timestamps_test tests/unittests/test_sanotts_word_timestamps.cpp)
    target_sources(sanotts_word_timestamps_test PRIVATE src/community_models/sanotts/word_timestamps.cpp)
    target_include_directories(sanotts_word_timestamps_test PRIVATE tests/unittests src/community_models/sanotts)
endif()

# shared_mel_migration_probe
if (index_tts2 IN_LIST AUDIOCPP_LINKED_MODELS AND confucius4_tts IN_LIST AUDIOCPP_LINKED_MODELS AND seed_vc IN_LIST AUDIOCPP_LINKED_MODELS AND glm_tts IN_LIST AUDIOCPP_LINKED_MODELS AND niagara_asr IN_LIST AUDIOCPP_LINKED_MODELS AND qwen3_tts IN_LIST AUDIOCPP_LINKED_MODELS AND vieneu_v3_turbo IN_LIST AUDIOCPP_LINKED_MODELS AND dramabox IN_LIST AUDIOCPP_LINKED_MODELS AND mira_tts IN_LIST AUDIOCPP_LINKED_MODELS AND chatterbox IN_LIST AUDIOCPP_LINKED_MODELS AND dots_tts IN_LIST AUDIOCPP_LINKED_MODELS AND kroko_asr IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(shared_mel_migration_probe tests/shared_mel_migration/probe.cpp)
    target_link_libraries(shared_mel_migration_probe PRIVATE engine_runtime ggml)
    target_include_directories(shared_mel_migration_probe PRIVATE src/models/chatterbox)
endif()

# sheetsage
add_executable(sheetsage2_decoder_parity_probe
    src/models/sheetsage/runtime.cpp
    tests/yue2/sheetsage2_decoder_parity_probe.cpp
)
target_link_libraries(sheetsage2_decoder_parity_probe PRIVATE engine_runtime ggml)
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(sheetsage2_decoder_parity_probe PRIVATE OpenMP::OpenMP_CXX)
endif()

# soprano_text_cleaning
add_engine_test(soprano_text_cleaning_test tests/unittests/test_soprano_text_cleaning.cpp)
target_sources(soprano_text_cleaning_test PRIVATE
    src/community_models/soprano_tts/text_normalizer.cpp
    src/community_models/soprano_tts/tokenizer_text.cpp)

# sopro_tts
if (sopro_tts IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(test_sopro_tts_audio_ops
        tests/sopro_tts/test_sopro_tts_audio_ops.cpp
    )
    target_link_libraries(test_sopro_tts_audio_ops PRIVATE engine_runtime ggml)
    target_include_directories(test_sopro_tts_audio_ops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(test_sopro_tts_audio_ops PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_test(
        NAME test_sopro_tts_audio_ops
        COMMAND test_sopro_tts_audio_ops
    )
endif()

# soulx_singer
if (soulx_singer IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(soulx_singer_components tests/soulx_singer/components.cpp app/cli/args.cpp)
    target_link_libraries(soulx_singer_components PRIVATE engine_runtime ggml)
    target_include_directories(soulx_singer_components PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# streaming_asr_session
add_engine_test(streaming_asr_session_test tests/unittests/test_streaming_asr_session.cpp)

# supertonic
add_engine_test(supertonic_vector_convnext_exp_test tests/unittests/test_supertonic_vector_convnext_exp.cpp)

# tdt_decoder_duration_loop
add_engine_test(tdt_decoder_duration_loop_test tests/unittests/test_tdt_decoder_duration_loop.cpp)

# tiny_voca
if (tiny_voca IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(tiny_voca_dizione_probe tests/tiny_voca/dizione_probe.cpp)
    target_include_directories(tiny_voca_dizione_probe PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/community_models/tiny_voca/dizione)
    target_link_libraries(tiny_voca_dizione_probe PRIVATE engine_runtime ggml)
    add_executable(tiny_voca_synth_probe tests/tiny_voca/synth_probe.cpp)
    target_link_libraries(tiny_voca_synth_probe PRIVATE engine_runtime ggml)
endif()

# vibeasr
if (vibeasr IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(test_vibeasr_asr
        tests/vibeasr/test_vibeasr_asr.cpp
    )
    target_compile_definitions(test_vibeasr_asr PRIVATE
        ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    target_link_libraries(test_vibeasr_asr PRIVATE engine_runtime ggml)
    target_include_directories(test_vibeasr_asr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(test_vibeasr_asr PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_test(
        NAME test_vibeasr_asr
        COMMAND test_vibeasr_asr --threads 8
    )
    set_tests_properties(test_vibeasr_asr PROPERTIES
        SKIP_RETURN_CODE 125
        TIMEOUT 600
    )
    add_executable(test_vibeasr_vae_encoder
        tests/vibeasr/test_vibeasr_vae_encoder.cpp
    )
    target_compile_definitions(test_vibeasr_vae_encoder PRIVATE
        ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    target_link_libraries(test_vibeasr_vae_encoder PRIVATE engine_runtime ggml)
    target_include_directories(test_vibeasr_vae_encoder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(test_vibeasr_vae_encoder PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_test(
        NAME test_vibeasr_vae_encoder
        COMMAND test_vibeasr_vae_encoder
        --model ${CMAKE_CURRENT_SOURCE_DIR}/models/vibeasr/vibeasr-vae-encoder-i8_s.gguf
        --audio ${CMAKE_CURRENT_SOURCE_DIR}/assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav
    )
    set_tests_properties(test_vibeasr_vae_encoder PROPERTIES
        SKIP_RETURN_CODE 125
        TIMEOUT 300
    )
endif()

# vibevoice
if (vibevoice IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(vibevoice_finetune_overlay_check
        tests/vibevoice/finetune_overlay_check.cpp
    )
    target_link_libraries(vibevoice_finetune_overlay_check PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(vibevoice_finetune_overlay_check PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# voice_isolator
if (voice_isolator IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(test_voice_isolator_reference tests/voice_isolator/test_reference.cpp app/cli/args.cpp)
    target_link_libraries(test_voice_isolator_reference PRIVATE engine_runtime ggml)
    target_include_directories(test_voice_isolator_reference PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
    add_engine_test(voice_isolator_domain_test tests/voice_isolator/test_domain.cpp)
endif()

# voxmel_tts
if (voxmel_tts IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(voxmel_tts_components tests/voxmel_tts/components.cpp app/cli/args.cpp)
    target_link_libraries(voxmel_tts_components PRIVATE engine_runtime ggml)
    target_include_directories(voxmel_tts_components PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# voxtral_realtime
if (voxtral_realtime IN_LIST AUDIOCPP_LINKED_MODELS)
    add_engine_test(voxtral_realtime_stream_chunking_test
        tests/unittests/test_voxtral_realtime_stream_chunking.cpp)
    target_sources(voxtral_realtime_stream_chunking_test PRIVATE src/models/voxtral_realtime/frontend.cpp)
endif()

# whistle_asr
if (whistle_asr IN_LIST AUDIOCPP_LINKED_MODELS)
    set(AUDIOCPP_WHISTLE_TEST_MODEL "" CACHE FILEPATH "Locally converted Whistle GGUF for integration tests")
    add_executable(whistle_assets_test
        tests/whistle_asr/test_assets.cpp
        src/community_models/whistle_asr/assets.cpp
        src/community_models/whistle_asr/decoder.cpp
        src/community_models/whistle_asr/encoder.cpp
        src/community_models/whistle_asr/frontend.cpp
        src/community_models/whistle_asr/graph.cpp
        src/community_models/whistle_asr/runtime.cpp)
    target_link_libraries(whistle_assets_test PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(whistle_assets_test PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if (AUDIOCPP_WHISTLE_TEST_MODEL)
        add_test(NAME whistle_asr_integration COMMAND whistle_assets_test
            "${AUDIOCPP_WHISTLE_TEST_MODEL}"
            "${CMAKE_CURRENT_SOURCE_DIR}/assets/resources/sample_16k.wav"
            "Some call me Nature.")
        add_test(NAME whistle_asr_full_integration COMMAND whistle_assets_test
            "${AUDIOCPP_WHISTLE_TEST_MODEL}"
            "${CMAKE_CURRENT_SOURCE_DIR}/assets/resources/sample_16k.wav"
            "Some call me Nature. Others call me Mother Nature. I've been here for over 4.5 billion years, twenty two thousand five hundred times longer than you."
            --full)
    endif()
endif()

# yue2
if ("yue2" IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(yue2_nar_parity_probe
        tests/yue2/yue2_nar_parity_probe.cpp
    )
    target_link_libraries(yue2_nar_parity_probe PRIVATE engine_runtime ggml)
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(yue2_nar_parity_probe PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
add_executable(yue2_vae_parity_probe
    tests/yue2/yue2_vae_parity_probe.cpp
)
target_link_libraries(yue2_vae_parity_probe PRIVATE engine_runtime ggml)
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(yue2_vae_parity_probe PRIVATE OpenMP::OpenMP_CXX)
endif()

# zipenhancer_utility
add_engine_test(zipenhancer_utility_test tests/unittests/test_zipenhancer_utility.cpp)
target_compile_definitions(zipenhancer_utility_test PRIVATE
    ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    ENGINE_TEST_ASSET_ROOT="${ENGINE_UNITTEST_ASSET_ROOT}"
)

# zipvoice
if (zipvoice IN_LIST AUDIOCPP_LINKED_MODELS)
    add_executable(zipvoice_parity tests/zipvoice/zipvoice_parity_main.cpp)
    target_link_libraries(zipvoice_parity PRIVATE engine_runtime ggml)
    target_include_directories(zipvoice_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if (ENGINE_ENABLE_OPENMP)
        target_link_libraries(zipvoice_parity PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_executable(zipvoice_zh_tokens tests/zipvoice/zipvoice_zh_tokens_main.cpp)
    target_link_libraries(zipvoice_zh_tokens PRIVATE engine_runtime)
endif()

# zonos2
if (zonos2 IN_LIST AUDIOCPP_LINKED_MODELS)
    set(AUDIOCPP_ZONOS2_MODEL_DIR "" CACHE PATH "ZONOS2 GGUF directory for the zonos2 parity tests")
    set(AUDIOCPP_ZONOS2_PARITY_DIR "" CACHE PATH "ZONOS2 oracle directory (tests/zonos2/make_oracle.sh)")
    add_engine_test_target(zonos2_parity tests/zonos2/parity.cpp)
    if (AUDIOCPP_ZONOS2_MODEL_DIR AND AUDIOCPP_ZONOS2_PARITY_DIR)
        add_test(NAME zonos2_parity_backbone COMMAND zonos2_parity backbone
            "${AUDIOCPP_ZONOS2_MODEL_DIR}/zonos2-q8_0.gguf" "${AUDIOCPP_ZONOS2_PARITY_DIR}")
        add_test(NAME zonos2_parity_backbone_speaker COMMAND zonos2_parity backbone_speaker
            "${AUDIOCPP_ZONOS2_MODEL_DIR}/zonos2-q8_0.gguf" "${AUDIOCPP_ZONOS2_PARITY_DIR}")
        add_test(NAME zonos2_parity_backbone_disk COMMAND zonos2_parity backbone
            "${AUDIOCPP_ZONOS2_MODEL_DIR}/zonos2-q8_0.gguf" "${AUDIOCPP_ZONOS2_PARITY_DIR}" --disk-experts)
        add_test(NAME zonos2_parity_decoder COMMAND zonos2_parity decoder
            "${AUDIOCPP_ZONOS2_MODEL_DIR}/dac.gguf" "${AUDIOCPP_ZONOS2_PARITY_DIR}")
        add_test(NAME zonos2_parity_speaker COMMAND zonos2_parity speaker
            "${AUDIOCPP_ZONOS2_MODEL_DIR}/spk-encoder.gguf" "${AUDIOCPP_ZONOS2_PARITY_DIR}")
    endif()
    add_engine_test_target(zonos2_prompt_test tests/zonos2/prompt_test.cpp)
    if (AUDIOCPP_ZONOS2_PARITY_DIR)
        add_test(NAME zonos2_prompt_test COMMAND zonos2_prompt_test "${AUDIOCPP_ZONOS2_PARITY_DIR}")
    endif()
    if (NOT AUDIOCPP_ZONOS2_PARITY_DIR)
        add_test(NAME zonos2_prompt_test COMMAND zonos2_prompt_test)
    endif()
    if (AUDIOCPP_ZONOS2_MODEL_DIR AND AUDIOCPP_ZONOS2_PARITY_DIR)
        set_tests_properties(zonos2_parity_backbone zonos2_parity_backbone_speaker zonos2_parity_backbone_disk
            PROPERTIES
            PASS_REGULAR_EXPRESSION "PASS"
            TIMEOUT 1800
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
        set_tests_properties(zonos2_parity_decoder zonos2_parity_speaker PROPERTIES
            PASS_REGULAR_EXPRESSION "PASS"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
endif()

# zonos3
if (zonos3 IN_LIST AUDIOCPP_LINKED_MODELS)
    set(AUDIOCPP_ZONOS3_BACKBONE "" CACHE FILEPATH "backbone-ptq.gguf for ZONOS3 tokenizer vectors")
    set(AUDIOCPP_ZONOS3_MODEL_DIR "" CACHE PATH "ZONOS3-Mini GGUF directory for the zonos3 parity tests")
    set(AUDIOCPP_ZONOS3_PARITY_DIR "" CACHE PATH "ZONOS3 oracle directory (tests/zonos3/make_oracle.sh)")
    add_engine_test_target(zonos3_parity tests/zonos3/parity.cpp)
    if (AUDIOCPP_ZONOS3_MODEL_DIR AND AUDIOCPP_ZONOS3_PARITY_DIR)
        add_test(NAME zonos3_parity_backbone COMMAND zonos3_parity backbone
            "${AUDIOCPP_ZONOS3_MODEL_DIR}/backbone-ptq.gguf" "${AUDIOCPP_ZONOS3_PARITY_DIR}")
        add_test(NAME zonos3_parity_backbone_disk COMMAND zonos3_parity backbone
            "${AUDIOCPP_ZONOS3_MODEL_DIR}/backbone-ptq.gguf" "${AUDIOCPP_ZONOS3_PARITY_DIR}" --disk-experts)
        add_test(NAME zonos3_parity_speaker COMMAND zonos3_parity speaker
            "${AUDIOCPP_ZONOS3_MODEL_DIR}/spk-encoder.gguf" "${AUDIOCPP_ZONOS3_PARITY_DIR}/voice.wav" "${AUDIOCPP_ZONOS3_PARITY_DIR}")
        add_test(NAME zonos3_parity_decoder COMMAND zonos3_parity decoder
            "${AUDIOCPP_ZONOS3_MODEL_DIR}/zac-keep2.gguf" "${AUDIOCPP_ZONOS3_PARITY_DIR}")
        add_test(NAME zonos3_parity_reference COMMAND zonos3_parity reference
            "${AUDIOCPP_ZONOS3_MODEL_DIR}" "${AUDIOCPP_ZONOS3_PARITY_DIR}/voice.wav" "${AUDIOCPP_ZONOS3_PARITY_DIR}")
        set_tests_properties(zonos3_parity_reference PROPERTIES
            PASS_REGULAR_EXPRESSION "PASS"
            TIMEOUT 3600
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
    add_engine_test_target(zonos3_tokenizer_test tests/zonos3/tokenizer_test.cpp)
    if (AUDIOCPP_ZONOS3_BACKBONE)
        add_test(NAME zonos3_tokenizer_test COMMAND zonos3_tokenizer_test
            "${AUDIOCPP_ZONOS3_BACKBONE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/zonos3/tokenizer_vectors.json")
    endif()
    if (AUDIOCPP_ZONOS3_MODEL_DIR AND AUDIOCPP_ZONOS3_PARITY_DIR)
        set_tests_properties(zonos3_parity_backbone zonos3_parity_backbone_disk zonos3_parity_speaker
            zonos3_parity_decoder PROPERTIES
            PASS_REGULAR_EXPRESSION "PASS"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
endif()
