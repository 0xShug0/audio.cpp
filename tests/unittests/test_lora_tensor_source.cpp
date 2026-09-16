#include "engine/framework/assets/lora_tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/safetensors.h"
#include "test_assert.h"

#include <cstring>
#include <filesystem>
#include <iostream>

namespace {
using namespace engine;
using test::require;

template<class T>
std::vector<unsigned char> bytes(const std::vector<T> & values) {
    std::vector<unsigned char> out(values.size() * sizeof(T));
    std::memcpy(out.data(), values.data(), out.size());
    return out;
}

template<class F>
void rejects(F fn) {
    bool threw = false;
    try { fn(); } catch (const std::runtime_error &) { threw = true; }
    require(threw, "invalid overlay was accepted");
}

void check_upload(const assets::TensorSource & source, const std::string & name,
                  const std::vector<int64_t> & shape, assets::TensorStorageType type) {
    auto backend = core::init_backend({core::BackendType::Cpu, 0, 1});
    auto ctx = ggml_init({ggml_tensor_overhead() * 2, nullptr, true});
    auto tensor = ggml_new_tensor_2d(ctx, assets::ggml_type_for_tensor_storage(type), shape[1], shape[0]);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "test backend allocation failed");
    source.set_backend_tensor(tensor, name, type, shape);
    std::vector<std::byte> actual(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
    const auto expected = source.require_tensor(name, type, shape);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    require(actual == expected.bytes, "backend upload differs from raw export conversion");
}

void run(const std::filesystem::path & root) {
    std::vector<float> weights(64), a(64), b = {1.0F, 1.0F, -0.75F, 0.25F};
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = static_cast<float>(i) / 9.0F;
        a[i] = static_cast<float>(static_cast<int>(i % 11) - 5) / 7.0F;
    }
    // Rank-wise accumulation must not silently become a summed delta + base.
    weights[0] = 100000000.0F;
    a[0] = a[32] = 4.0F;
    io::write_safetensors_file(root / "base.safetensors", {
        {"weight", "F32", {2, 32}, bytes(weights)},
        {"untouched", "F32", {2, 32}, bytes(weights)},
        {"step", "I64", {1}, bytes(std::vector<int64_t>{42})}
    });
    io::write_safetensors_file(root / "adapter.safetensors", {
        {"a", "F32", {2, 32}, bytes(a)}, {"b", "F32", {2, 2}, bytes(b)}
    });
    const auto base = assets::open_tensor_source(root / "base.safetensors");
    const auto adapter = assets::open_tensor_source(root / "adapter.safetensors");
    require(assets::make_lora_tensor_source(base, {}).get() == base.get(), "empty overlay must be identity");
    auto delta = assets::load_lora_tensor_delta(*base, *adapter, "weight", "a", "b", 1.0F);
    auto overlay = assets::make_lora_tensor_source(base, {{"weight", delta}});
    auto expected = weights;
    for (int o = 0; o < 2; ++o) {
        for (int k = 0; k < 2; ++k) {
            const float scaled_b = delta.scale * b[o * 2 + k];
            for (int i = 0; i < 32; ++i) expected[o * 32 + i] += scaled_b * a[k * 32 + i];
        }
    }
    require(expected[0] == weights[0], "rounding fixture must retain original rank-wise result");
    require(overlay->require_f32("weight") == expected, "merge arithmetic changed");
    require(overlay->require_f32("weight") == expected, "repeated read applies adapter twice");
    require(base->require_f32("weight") == weights, "overlay changed base storage");
    require(overlay->require_tensor_data("untouched").bytes == base->require_tensor_data("untouched").bytes,
            "unadapted tensor changed");
    require(overlay->require_metadata("weight").dtype == base->require_metadata("weight").dtype,
            "storage metadata changed");
    require(overlay->source_path() == base->source_path(), "base path changed");
    require(overlay->tensors().size() == base->tensors().size(), "tensor inventory changed");
    require(overlay->require_i64_scalar("step") == 42, "scalar passthrough failed");
    require(!overlay->optional_f32("missing"), "missing optional tensor should remain absent");
    require(overlay->optional_f32("weight").value() == expected, "optional merge differs");
    rejects([&] { (void)overlay->require_f32("weight", {32, 2}); });
    rejects([&] { assets::load_lora_tensor_delta(*base, *adapter, "weight", "a", "a", 1.0F); });
    rejects([&] { assets::load_lora_tensor_delta(*base, *adapter, "weight", "a", "missing", 1.0F); });
    auto bad = delta;
    bad.a.pop_back();
    rejects([&] { assets::make_lora_tensor_source(base, {{"weight", bad}}); });

    const std::vector<float> replacement(64, 0.75F);
    auto overridden = assets::make_lora_tensor_source(base, {{"weight", delta}},
        {{"weight", {{2, 32}, replacement}}});
    require(overridden->require_f32("weight") == replacement, "full override must win over delta");
    rejects([&] { (void)overridden->require_f32("weight", {32, 2}); });
    rejects([&] { assets::make_lora_tensor_source(base, {}, {{"weight", {{2, 32}, {1.0F}}}}); });
    rejects([&] { assets::make_lora_tensor_source(base, {}, {{"missing", {{2, 32}, replacement}}}); });
    for (auto type : {assets::TensorStorageType::F32, assets::TensorStorageType::F16,
                      assets::TensorStorageType::BF16, assets::TensorStorageType::Q8_0,
                      assets::TensorStorageType::Q4_0}) {
        check_upload(*overlay, "weight", {2, 32}, type);
        check_upload(*overlay, "untouched", {2, 32}, type);
        check_upload(*overridden, "weight", {2, 32}, type);
    }
    overlay->release_storage();
    require(overlay->require_f32("weight") == expected, "release/reopen changed merge");
}
}  // namespace

int main(int argc, char ** argv) {
    const auto root = std::filesystem::temp_directory_path() / "audiocpp_lora_tensor_source_test";
    try {
        debug::configure_logging({argc > 1 && std::string(argv[1]) == "--log", std::nullopt});
        std::filesystem::create_directories(root);
        run(root);
        std::filesystem::remove_all(root);
        std::cout << "lora_tensor_source_test passed\n";
    } catch (const std::exception & e) {
        std::filesystem::remove_all(root);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
