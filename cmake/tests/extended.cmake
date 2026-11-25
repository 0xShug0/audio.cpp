# Optional backend and extended integration tests.
# Included only when ENGINE_BUILD_EXTENDED_TESTS is enabled.

# audio_utility_api
add_engine_unittest(audio_utility_api_test tests/unittests/test_audio_utility_api.cpp)
target_compile_definitions(audio_utility_api_test PRIVATE
    ENGINE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)
add_test(NAME audio_utility_api_test COMMAND audio_utility_api_test)

# base64
add_executable(base64_test
    tests/unittests/test_base64.cpp
    app/common/base64.cpp
)
add_test(NAME base64_test COMMAND base64_test)

# cli_request_options
add_executable(cli_request_options_test
    tests/unittests/test_cli_request_options.cpp
    app/cli/args.cpp
    app/cli/request.cpp
    app/common/base64.cpp
)
target_link_libraries(cli_request_options_test PRIVATE engine_runtime ggml)
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(cli_request_options_test PRIVATE OpenMP::OpenMP_CXX)
endif()
add_test(NAME cli_request_options_test COMMAND cli_request_options_test)

# conv2d_modules
add_engine_unittest(conv2d_modules_test tests/unittests/test_conv2d_modules.cpp)
add_test(NAME conv2d_modules_test COMMAND conv2d_modules_test)

# conv_lowering_matrix
add_engine_unittest(conv_lowering_matrix_test tests/unittests/test_conv_lowering_matrix.cpp)
if (ENGINE_ENABLE_HIP)
    target_compile_definitions(conv_lowering_matrix_test PRIVATE ENGINE_TEST_ENABLE_HIP=1)
endif()
add_test(NAME conv_lowering_matrix_test COMMAND conv_lowering_matrix_test)

# conv_transpose_fast_path
add_engine_unittest(conv_transpose_fast_path_test tests/unittests/test_conv_transpose_fast_path.cpp)
add_test(NAME conv_transpose_fast_path_test COMMAND conv_transpose_fast_path_test)

# cuda
if (ENGINE_ENABLE_CUDA)
    add_executable(cuda_conv_transpose_1d_test tests/unittests/test_cuda_conv_transpose_1d.cu)
    set_target_properties(cuda_conv_transpose_1d_test PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_compile_options(cuda_conv_transpose_1d_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
    if (GGML_STATIC)
        set_target_properties(cuda_conv_transpose_1d_test PROPERTIES CUDA_RUNTIME_LIBRARY Static)
        target_link_libraries(cuda_conv_transpose_1d_test PRIVATE ggml CUDA::cudart_static)
    endif()
    if (NOT GGML_STATIC)
        set_target_properties(cuda_conv_transpose_1d_test PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
        target_link_libraries(cuda_conv_transpose_1d_test PRIVATE ggml CUDA::cudart)
    endif()
    add_test(NAME cuda_conv_transpose_1d_test COMMAND cuda_conv_transpose_1d_test)
    set_tests_properties(cuda_conv_transpose_1d_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
endif()
if (ENGINE_ENABLE_CUDA AND GGML_CUDA_CUDNN_CONV3D)
    add_engine_unittest(cuda_cudnn_conv3d_test tests/unittests/test_cuda_cudnn_conv3d.cpp)
    add_test(NAME cuda_cudnn_conv3d_test COMMAND cuda_cudnn_conv3d_test
        --log "${CMAKE_CURRENT_BINARY_DIR}/cuda_cudnn_conv3d_test.log")
    set_tests_properties(cuda_cudnn_conv3d_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

# depthwise_conv1d_lowering
add_engine_unittest(depthwise_conv1d_lowering_test tests/unittests/test_depthwise_conv1d_lowering.cpp)
add_test(NAME depthwise_conv1d_lowering_test COMMAND depthwise_conv1d_lowering_test)

# devices
add_test(NAME audiocpp_cli_list_devices COMMAND audiocpp_cli --list-devices)
add_test(NAME audiocpp_server_list_devices COMMAND audiocpp_server --list-devices)
set_tests_properties(
    audiocpp_cli_list_devices
    audiocpp_server_list_devices
    PROPERTIES PASS_REGULAR_EXPRESSION "available_devices=[0-9]+"
)

# file_sink_json
add_engine_unittest(file_sink_json_test tests/unittests/test_file_sink_json.cpp)
target_sources(file_sink_json_test PRIVATE app/workflow/file_sink.cpp)
add_test(NAME file_sink_json_test COMMAND file_sink_json_test)

# i2_s_mul_mat
add_engine_unittest(i2_s_mul_mat_test tests/unittests/test_i2_s_mul_mat.cpp)
add_test(NAME i2_s_mul_mat_test COMMAND i2_s_mul_mat_test)

# i8_s_fused_ops
add_engine_unittest(i8_s_fused_ops_test tests/unittests/test_i8_s_fused_ops.cpp)
add_test(NAME i8_s_fused_ops_test COMMAND i8_s_fused_ops_test)

# metal
if (ENGINE_ENABLE_METAL)
    add_executable(metal_conv_implicit_test tests/unittests/test_metal_conv_implicit.cpp)
    target_link_libraries(metal_conv_implicit_test PRIVATE engine_runtime ggml-metal ggml-cpu)
    add_test(NAME metal_conv_implicit_test COMMAND metal_conv_implicit_test)
    set_tests_properties(metal_conv_implicit_test PROPERTIES LABELS "extended;metal" SKIP_RETURN_CODE 77)
    add_executable(metal_f32_matmul_test tests/unittests/test_metal_f32_matmul.cpp)
    target_link_libraries(metal_f32_matmul_test PRIVATE ggml ggml-metal)
    add_test(NAME metal_f32_matmul_test COMMAND metal_f32_matmul_test)
    set_tests_properties(metal_f32_matmul_test PROPERTIES
        SKIP_RETURN_CODE 77 LABELS "extended;metal")
    add_executable(metal_im2col_3d_test tests/unittests/test_metal_im2col_3d.cpp)
    target_link_libraries(metal_im2col_3d_test PRIVATE ggml ggml-metal)
    add_test(NAME metal_im2col_3d_test COMMAND metal_im2col_3d_test)
    set_tests_properties(metal_im2col_3d_test PROPERTIES
        SKIP_RETURN_CODE 77 LABELS "extended;metal")
    add_executable(metal_im2col_test tests/unittests/test_metal_im2col.cpp)
    target_link_libraries(metal_im2col_test PRIVATE ggml ggml-metal)
    target_compile_definitions(metal_im2col_test PRIVATE
        METAL_IM2COL_GOLDEN_PATH="${ENGINE_UNITTEST_ASSET_ROOT}/metal_im2col/original_metal_sha256.txt")
    add_test(NAME metal_im2col_test COMMAND metal_im2col_test)
    set_tests_properties(metal_im2col_test PROPERTIES
        SKIP_RETURN_CODE 77 LABELS "extended;metal")
    add_executable(metal_im2col_tiled_test tests/unittests/test_metal_im2col_tiled.cpp)
    target_link_libraries(metal_im2col_tiled_test PRIVATE ggml ggml-metal)
    add_test(NAME metal_im2col_tiled_test COMMAND metal_im2col_tiled_test)
    set_tests_properties(metal_im2col_tiled_test PROPERTIES
        SKIP_RETURN_CODE 77 LABELS "extended;metal")
    add_executable(metal_ssm_mamba1_test tests/unittests/test_ssm_mamba1.cpp app/cli/args.cpp)
    target_link_libraries(metal_ssm_mamba1_test PRIVATE engine_runtime ggml)
    target_include_directories(metal_ssm_mamba1_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME metal_ssm_mamba1_test COMMAND metal_ssm_mamba1_test --backend metal --log
        --log-file "${CMAKE_CURRENT_BINARY_DIR}/ssm-mamba1-metal.log")
    set_tests_properties(metal_ssm_mamba1_test PROPERTIES LABELS "extended;metal" SKIP_RETURN_CODE 77)
    add_executable(metal_ssm_scan_test tests/unittests/test_metal_ssm_scan.cpp)
    target_link_libraries(metal_ssm_scan_test PRIVATE ggml ggml-metal)
    add_test(NAME metal_ssm_scan_test COMMAND metal_ssm_scan_test)
    set_tests_properties(metal_ssm_scan_test PROPERTIES
        SKIP_RETURN_CODE 77 LABELS "extended;metal")
endif()

# package_manager
if (AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER)
    add_executable(package_manager_http_test tests/unittests/test_package_manager_http.cpp)
    target_link_libraries(package_manager_http_test PRIVATE
        Threads::Threads audiocpp_package_manager audiocpp_cpp_httplib)
    add_test(NAME package_manager_http_test COMMAND package_manager_http_test)
    add_executable(package_manager_modelscope_test
        tests/unittests/test_package_manager_modelscope.cpp
    )
    target_link_libraries(package_manager_modelscope_test PRIVATE
        Threads::Threads audiocpp_package_manager)
    target_compile_definitions(package_manager_modelscope_test PRIVATE
        AUDIOCPP_NATIVE_MANAGER_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/native_model_manager_server.py")
    add_test(NAME package_manager_modelscope_test COMMAND package_manager_modelscope_test)
    add_executable(package_manager_xet_test tests/unittests/test_package_manager_xet.cpp)
    target_include_directories(package_manager_xet_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/framework/package_manager)
    target_link_libraries(package_manager_xet_test PRIVATE
        Threads::Threads audiocpp_package_manager audiocpp_cpp_httplib lz4_vendor)
    add_test(NAME package_manager_xet_test COMMAND package_manager_xet_test)
endif()

# request_seed_dispatch
if (AUDIOCPP_BUILD_C_API)
    add_engine_unittest(request_seed_dispatch_test tests/unittests/test_request_seed_dispatch.cpp)
    target_sources(request_seed_dispatch_test PRIVATE
        app/workflow/execution.cpp app/workflow/workflow.cpp app/workflow/file_sink.cpp
        app/recipe/recipe.cpp app/recipe/cli_recipe.cpp
        app/streaming/streaming.cpp app/streaming/pcm_source.cpp
        app/cli/args.cpp app/cli/request.cpp app/common/base64.cpp)
    add_test(NAME request_seed_dispatch_test COMMAND request_seed_dispatch_test)
endif()

# scaled_dot_product_attention
add_engine_unittest(scaled_dot_product_attention_test tests/unittests/test_scaled_dot_product_attention.cpp)
add_test(NAME scaled_dot_product_attention_test COMMAND scaled_dot_product_attention_test)

# server
add_executable(http_live_body_test
    tests/unittests/test_http_live_body.cpp
    app/server/http.cpp
)
target_link_libraries(http_live_body_test PRIVATE engine_runtime ggml)
if (WIN32)
    target_link_libraries(http_live_body_test PRIVATE ws2_32)
endif()
target_include_directories(http_live_body_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
add_test(NAME http_live_body_test COMMAND http_live_body_test)
set_tests_properties(http_live_body_test PROPERTIES TIMEOUT 120)
if (UNIX)
    add_test(NAME http_live_body_unix_test COMMAND http_live_body_test --unix)
endif()
add_executable(parallel_http_live_body_test
    tests/unittests/test_http_live_body.cpp app/server/http.cpp app/server/parallel_http.cpp)
target_compile_definitions(parallel_http_live_body_test PRIVATE AUDIOCPP_TEST_PARALLEL_HTTP=1)
target_include_directories(parallel_http_live_body_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
target_link_libraries(parallel_http_live_body_test PRIVATE engine_runtime ggml Threads::Threads)
add_test(NAME parallel_http_live_body_test COMMAND parallel_http_live_body_test)
set_tests_properties(parallel_http_live_body_test PROPERTIES TIMEOUT 120)
if (UNIX)
    add_test(NAME parallel_http_live_body_unix_test COMMAND parallel_http_live_body_test --unix)
endif()
if (WIN32)
    target_link_libraries(parallel_http_live_body_test PRIVATE ws2_32)
endif()
add_executable(parallel_http_ownership_test
    tests/unittests/test_parallel_http_ownership.cpp app/server/http.cpp app/server/parallel_http.cpp)
target_include_directories(parallel_http_ownership_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/app/server)
target_link_libraries(parallel_http_ownership_test PRIVATE engine_runtime ggml Threads::Threads)
add_test(NAME parallel_http_ownership_test COMMAND parallel_http_ownership_test)
set_tests_properties(parallel_http_ownership_test PROPERTIES TIMEOUT 30)
if (WIN32)
    target_link_libraries(parallel_http_ownership_test PRIVATE ws2_32)
endif()
add_executable(server_busy_guard_test
    tests/unittests/test_server_busy_guard.cpp
)
target_include_directories(server_busy_guard_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/app/server)
target_link_libraries(server_busy_guard_test PRIVATE Threads::Threads)
add_test(NAME server_busy_guard_test COMMAND server_busy_guard_test)
set_tests_properties(server_busy_guard_test PROPERTIES TIMEOUT 60)
add_executable(server_config_test
    tests/unittests/test_server_config.cpp
    app/server/config.cpp
    app/server/model_memory.cpp
    app/cli/args.cpp
    app/cli/request.cpp
    app/common/base64.cpp
)
target_include_directories(server_config_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/app/server)
target_link_libraries(server_config_test PRIVATE engine_runtime ggml)
if (ENGINE_ENABLE_OPENMP)
    target_link_libraries(server_config_test PRIVATE OpenMP::OpenMP_CXX)
endif()
add_test(NAME server_config_test COMMAND server_config_test)
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
    add_test(NAME server_model_installer_test COMMAND server_model_installer_test)
endif()
add_executable(server_multipart_test
    tests/unittests/test_server_multipart.cpp
    app/server/multipart.cpp
)
target_include_directories(server_multipart_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/app/server)
add_test(NAME server_multipart_test COMMAND server_multipart_test)
set_tests_properties(http_live_body_test parallel_http_live_body_test PROPERTIES RESOURCE_LOCK http_live_body_port)
if (UNIX)
    set_tests_properties(http_live_body_unix_test parallel_http_live_body_unix_test
        PROPERTIES TIMEOUT 120)
endif()

# stft_parity_dump
add_executable(stft_parity_dump tests/audio/stft_parity_dump.cpp)
target_link_libraries(stft_parity_dump PRIVATE engine_runtime ggml)

# streaming_audio_input
add_executable(streaming_audio_input_test
    tests/unittests/test_streaming_audio_input.cpp
    app/streaming/pcm_source.cpp
    app/streaming/streaming.cpp
)
target_link_libraries(streaming_audio_input_test PRIVATE engine_runtime ggml)
target_include_directories(streaming_audio_input_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests)
add_test(NAME streaming_audio_input_test COMMAND streaming_audio_input_test)

# test-backend-ops
if (ENGINE_ENABLE_WEBGPU OR ENGINE_ENABLE_CUDA)
    add_executable(test-backend-ops external/ggml/tests/test-backend-ops.cpp)
    target_link_libraries(test-backend-ops PRIVATE ggml Threads::Threads)
    if (NOT ENGINE_ENABLE_WEBGPU)
        set_target_properties(test-backend-ops PROPERTIES EXCLUDE_FROM_ALL TRUE)
    endif()
endif()

# vulkan_ssm_mamba1
if (ENGINE_ENABLE_VULKAN)
    add_executable(vulkan_ssm_mamba1_test tests/unittests/test_ssm_mamba1.cpp app/cli/args.cpp)
    target_link_libraries(vulkan_ssm_mamba1_test PRIVATE engine_runtime ggml)
    target_include_directories(vulkan_ssm_mamba1_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME vulkan_ssm_mamba1_test COMMAND vulkan_ssm_mamba1_test --backend vulkan --log
        --log-file "${CMAKE_CURRENT_BINARY_DIR}/ssm-mamba1-vulkan.log")
    set_tests_properties(vulkan_ssm_mamba1_test PROPERTIES LABELS "extended;vulkan" SKIP_RETURN_CODE 77)
endif()
