#include "engine/framework/sampling/torch_random.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/debug/profiler.h"

#include <cstring>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void require_close(float actual, float expected, float tolerance, const char * label) {
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            std::string(label) + " mismatch: expected " + std::to_string(expected) +
            ", got " + std::to_string(actual));
    }
}

void require_vector_close(
    const std::vector<float> & actual,
    const std::vector<float> & expected,
    float tolerance,
    const char * label) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(std::string(label) + " size mismatch");
    }
    for (size_t index = 0; index < actual.size(); ++index) {
        require_close(actual[index], expected[index], tolerance, label);
    }
}

void test_float32_matches_torch_cuda() {
    const std::vector<float> expected_seed_1234 = {
        -1.61649048F, 0.56845516F, -0.510224819F, -0.911339402F,
        -1.15551639F, -0.22615087F, -1.2891326F, 1.065382F,
        -0.7166605F, -0.533334374F, 0.207757875F, -0.979843318F,
        0.744693458F, -0.239484429F, 0.27371496F, 0.092008315F,
    };
    const auto actual = engine::sampling::generate_torch_cuda_randn(
        expected_seed_1234.size(),
        1234,
        engine::sampling::TorchRandnPrecision::Float32);
    require_vector_close(actual, expected_seed_1234, 2.0e-6F, "float32 torch_cuda_randn");

    const auto extended = engine::sampling::generate_torch_cuda_randn(
        300,
        1234,
        engine::sampling::TorchRandnPrecision::Float32);
    const std::vector<std::pair<size_t, float>> expected_positions = {
        {16, -0.803610802F},
        {17, -1.6377064F},
        {31, -0.300402969F},
        {32, -0.251393795F},
        {63, -0.924830973F},
        {64, 0.799798131F},
        {127, -0.167921364F},
        {128, -0.322486848F},
        {255, -1.76037288F},
        {256, 0.534284174F},
        {299, -0.986868858F},
    };
    for (const auto & [index, expected] : expected_positions) {
        require_close(extended[index], expected, 2.0e-6F, "float32 torch_cuda_randn high index");
    }
}

void test_bfloat16_matches_torch_cuda() {
    const std::vector<float> expected_seed_5678 = {
        -0.65625F, -0.5625F, 0.384765625F, -0.151367188F,
        0.7421875F, 0.244140625F, 0.609375F, 0.00321960449F,
        -0.46875F, 0.306640625F, -0.225585938F, -0.9140625F,
        0.7734375F, 0.120117188F, 0.52734375F, -0.5078125F,
    };
    const auto actual = engine::sampling::generate_torch_cuda_randn(
        expected_seed_5678.size(),
        5678,
        engine::sampling::TorchRandnPrecision::BFloat16);
    require_vector_close(actual, expected_seed_5678, 0.0F, "bfloat16 torch_cuda_randn");

    const auto extended = engine::sampling::generate_torch_cuda_randn(
        300,
        1234,
        engine::sampling::TorchRandnPrecision::BFloat16);
    const std::vector<std::pair<size_t, float>> expected_positions = {
        {16, -0.8046875F},
        {17, -1.640625F},
        {31, -0.30078125F},
        {32, -0.251953125F},
        {63, -0.92578125F},
        {64, 0.80078125F},
        {127, -0.16796875F},
        {128, -0.322265625F},
        {255, -1.7578125F},
        {256, 0.53515625F},
        {299, -0.98828125F},
    };
    for (const auto & [index, expected] : expected_positions) {
        require_close(extended[index], expected, 0.0F, "bfloat16 torch_cuda_randn high index");
    }
}

void test_fill_matches_vector_api() {
    std::vector<float> filled(257, 0.0F);
    engine::sampling::fill_torch_cuda_randn(
        filled.data(),
        filled.size(),
        1,
        engine::sampling::TorchRandnPrecision::Float32);
    const auto generated = engine::sampling::generate_torch_cuda_randn(
        filled.size(),
        1,
        engine::sampling::TorchRandnPrecision::Float32);
    require_vector_close(filled, generated, 0.0F, "fill/vector torch_cuda_randn");
}

void test_start_index_matches_full_generation_slice() {
    const auto full = engine::sampling::generate_torch_cuda_randn(
        300,
        1234,
        engine::sampling::TorchRandnPrecision::Float32);
    const auto sliced = engine::sampling::generate_torch_cuda_randn(
        32,
        1234,
        engine::sampling::TorchRandnPrecision::Float32,
        128);
    for (size_t index = 0; index < sliced.size(); ++index) {
        require_close(sliced[index], full[128 + index], 0.0F, "torch_cuda_randn start index");
    }
}

void test_uniform_matches_torch_cuda() {
    const std::vector<float> expected_seed_1234 = {
        0.127207950F, 0.816736281F, 0.543999255F, 0.660079062F,
        0.272055715F, 0.973670244F, 0.390334547F, 0.339447320F,
        0.545121968F, 0.731194019F, 0.386437327F, 0.595885396F,
        0.757832348F, 0.212593451F, 0.719796896F, 0.984495401F,
    };
    const auto actual = engine::sampling::generate_torch_cuda_uniform(expected_seed_1234.size(), 1234);
    require_vector_close(actual, expected_seed_1234, 1.0e-7F, "float32 torch_cuda_uniform");

    const auto extended = engine::sampling::generate_torch_cuda_uniform(300, 1234);
    const std::vector<std::pair<size_t, float>> expected_positions = {
        {16, 0.551820338F},
        {17, 0.0980522931F},
        {31, 0.388187110F},
        {32, 0.797672272F},
        {63, 0.546383321F},
        {64, 0.694423795F},
        {127, 0.768725276F},
        {128, 0.798199594F},
        {255, 0.210613519F},
        {256, 0.294832915F},
        {299, 0.588956773F},
    };
    for (const auto & [index, expected] : expected_positions) {
        require_close(extended[index], expected, 1.0e-7F, "float32 torch_cuda_uniform high index");
    }

    const auto full = engine::sampling::generate_torch_cuda_uniform(300, 5678);
    const auto sliced = engine::sampling::generate_torch_cuda_uniform(32, 5678, 128);
    for (size_t index = 0; index < sliced.size(); ++index) {
        require_close(sliced[index], full[128 + index], 0.0F, "torch_cuda_uniform start index");
    }
}

void test_tensor_iterator_exponential_matches_torch_cuda() {
    constexpr uint64_t seed = 1234;
    constexpr uint64_t vocab_size = 168565;
    constexpr uint64_t call_index = 104;
    constexpr int64_t multiprocessor_count = 128;
    constexpr int64_t max_threads_per_multiprocessor = 1536;
    const std::vector<std::pair<uint64_t, float>> expected_positions = {
        {167785, 0.301640809F},
        {168248, 0.333908796F},
        {156088, 0.690724790F},
        {162701, 0.975778222F},
        {160228, 3.30194235F},
        {168039, 0.803459823F},
        {163896, 0.841065466F},
        {167841, 1.16925406F},
        {161373, 0.112876698F},
    };
    for (const auto & [index, expected] : expected_positions) {
        const float actual = engine::sampling::torch_cuda_tensor_iterator_exponential_element(
            seed,
            vocab_size,
            index,
            call_index,
            multiprocessor_count,
            max_threads_per_multiprocessor);
        require_close(actual, expected, 2.0e-6F, "torch_cuda_tensor_iterator_exponential");
    }
}

void test_empty_null_output_is_allowed() {
    engine::sampling::fill_torch_cuda_randn(
        nullptr,
        0,
        0,
        engine::sampling::TorchRandnPrecision::Float32);
    engine::sampling::fill_torch_cuda_uniform(nullptr, 0, 0);
}

void test_backend_tensor_fill(engine::core::BackendType backend) {
    using namespace engine;
    core::BackendConfig config;
    config.type = backend;
    config.threads = 8;
    core::ExecutionContext execution(config);
    const auto policy = sampling::resolve_torch_cuda_sampling_policy(backend, config.device,
        "test.rng", "Tensor RNG test");
    for (int64_t count : {1,17,1025,131073}) {
        core::PreparedGraph bank(execution, {"test.rng", 64 * 1024, 8, 64 * 1024});
        auto root = core::make_tensor(bank.persistent_build_context(), GGML_TYPE_F32,
            core::TensorShape::from_dims({count + 4}));
        auto view = core::wrap_tensor(ggml_view_1d(bank.persistent_build_context().ggml,
            root.tensor, count, 4 * sizeof(float)), core::TensorShape::from_dims({count}), GGML_TYPE_F32);
        bank.allocate_persistent();
        for (const auto & target : {root, view}) {
            for (auto precision : {sampling::TorchRandnPrecision::Float32, sampling::TorchRandnPrecision::BFloat16}) {
                for (uint64_t offset : {uint64_t(0),uint64_t(4),uint64_t(8)}) {
                    const auto expected = sampling::generate_torch_cuda_tensor_iterator_randn(
                        static_cast<size_t>(ggml_nelements(target.tensor)), 42, offset, policy, precision);
                    sampling::fill_torch_cuda_tensor_iterator_randn(execution.backend(), target,
                        43, offset, policy, precision);
                    sampling::fill_torch_cuda_tensor_iterator_randn(execution.backend(), target,
                        42, offset, policy, precision);
                    ggml_backend_synchronize(execution.backend());
                    const auto actual = core::read_tensor_f32(target.tensor);
                    if (actual.size() != expected.size() ||
                        std::memcmp(actual.data(), expected.data(), expected.size() * sizeof(float)) != 0)
                        throw std::runtime_error("Backend tensor RNG differs from the existing host API");
                }
            }
        }
    }
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        auto backend = engine::core::BackendType::Cpu;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            if (++i >= argc) throw std::runtime_error("Missing test argument");
            const std::string value = argv[i];
            if (flag == "--log") engine::debug::configure_logging({true, value});
            else if (flag == "--backend") {
                if (value == "cuda") backend = engine::core::BackendType::Cuda;
                else if (value == "vulkan") backend = engine::core::BackendType::Vulkan;
                else if (value != "cpu") throw std::runtime_error("Unknown backend");
            } else throw std::runtime_error("Unknown test argument");
        }
        const auto started = std::chrono::steady_clock::now();
        test_float32_matches_torch_cuda();
        test_bfloat16_matches_torch_cuda();
        test_fill_matches_vector_api();
        test_start_index_matches_full_generation_slice();
        test_uniform_matches_torch_cuda();
        test_tensor_iterator_exponential_matches_torch_cuda();
        test_empty_null_output_is_allowed();
        test_backend_tensor_fill(backend);
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "torch_random_test passed\n";
    return 0;
}
