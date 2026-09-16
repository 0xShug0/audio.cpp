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
                  const std::vector<int64_t> & shape, assets::TensorStorageType type,
                  const assets::TensorSource * reference = nullptr) {
    auto backend = core::init_backend({core::BackendType::Cpu, 0, 1});
    auto ctx = ggml_init({ggml_tensor_overhead() * 2, nullptr, true});
    auto tensor = ggml_new_tensor_2d(ctx, assets::ggml_type_for_tensor_storage(type), shape[1], shape[0]);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "test backend allocation failed");
    source.set_backend_tensor(tensor, name, type, shape);
    std::vector<std::byte> actual(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
    const auto expected = (reference ? *reference : source).require_tensor(name, type, shape);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    require(actual == expected.bytes, "backend upload differs from raw export conversion");
}

class CountingSource final : public assets::TensorSource {
public:
    explicit CountingSource(std::shared_ptr<const assets::TensorSource> source) : source_(std::move(source)) {}
    const std::filesystem::path & source_path() const noexcept override { return source_->source_path(); }
    bool has_tensor(std::string_view name) const noexcept override { return source_->has_tensor(name); }
    assets::TensorMetadata require_metadata(std::string_view name) const override {
        return source_->require_metadata(name);
    }
    std::vector<assets::TensorMetadata> tensors() const override { return source_->tensors(); }
    void release_storage() const override { source_->release_storage(); }
    assets::RawTensorData require_tensor_data(std::string_view name) const override {
        return source_->require_tensor_data(name);
    }
    std::vector<float> require_f32(std::string_view name,
        const std::optional<std::vector<int64_t>> & shape) const override {
        ++reads;
        return source_->require_f32(name, shape);
    }
    std::optional<std::vector<float>> optional_f32(std::string_view name,
        const std::optional<std::vector<int64_t>> & shape) const override {
        return source_->optional_f32(name, shape);
    }
    int64_t require_i64_scalar(std::string_view name) const override { return source_->require_i64_scalar(name); }
    mutable size_t reads = 0;
private:
    std::shared_ptr<const assets::TensorSource> source_;
};

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
    auto rounded_delta = delta;
    rounded_delta.merge_mode = assets::LoraMergeMode::RoundedBF16Delta;
    auto rounded = assets::make_lora_tensor_source(base, {{"weight", rounded_delta}});
    auto rounded_expected = weights;
    for (int o = 0; o < 2; ++o) {
        for (int i = 0; i < 32; ++i) {
            float product = 0.0F;
            for (int k = 0; k < 2; ++k) product += b[o * 2 + k] * a[k * 32 + i];
            const float update = ggml_bf16_to_fp32(ggml_fp32_to_bf16(product));
            rounded_expected[o * 32 + i] = ggml_bf16_to_fp32(
                ggml_fp32_to_bf16(weights[o * 32 + i] + update));
        }
    }
    require(rounded_expected != expected, "merge modes need distinct rounding fixtures");
    require(rounded->require_f32("weight") == rounded_expected, "BF16 delta merge differs");
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
        check_upload(*rounded, "weight", {2, 32}, type);
        check_upload(*overlay, "untouched", {2, 32}, type);
        check_upload(*overridden, "weight", {2, 32}, type);
    }
    overlay->release_storage();
    require(overlay->require_f32("weight") == expected, "release/reopen changed merge");

    for (auto mode : {assets::LoraMergeMode::AccumulateF32, assets::LoraMergeMode::RoundedBF16Delta}) {
        auto counted = std::make_shared<CountingSource>(base);
        auto cached_delta = delta;
        cached_delta.merge_mode = mode;
        auto reference = assets::make_lora_tensor_source(base, {{"weight", cached_delta}});
        auto cached = assets::make_lora_tensor_source(counted, {{"weight", cached_delta}}, {}, "cached", true);
        size_t expected_reads = 0;
        for (auto type : {assets::TensorStorageType::F32, assets::TensorStorageType::F16,
                          assets::TensorStorageType::BF16, assets::TensorStorageType::Q8_0,
                          assets::TensorStorageType::Q4_0, assets::TensorStorageType::Q8_0,
                          assets::TensorStorageType::Q4_0}) {
            check_upload(*cached, "weight", {2, 32}, type, reference.get());
            require(counted->reads == ++expected_reads, "dtype replacement did not merge from original base");
            cached->release_storage();
            for (int repeat = 0; repeat < 3; ++repeat) {
                check_upload(*cached, "weight", {2, 32}, type, reference.get());
                require(counted->reads == expected_reads, "cached upload re-read/re-merged weights");
            }
        }
        auto uncached = assets::make_lora_tensor_source(counted, {{"weight", cached_delta}});
        for (int repeat = 0; repeat < 2; ++repeat) {
            check_upload(*uncached, "weight", {2, 32}, assets::TensorStorageType::Q8_0, reference.get());
            require(counted->reads == ++expected_reads, "default overlay unexpectedly caches weights");
        }
    }

    // Exercise retained bytes in the parallel F16/BF16 conversion paths too.
    std::vector<float> large_weights(1024 * 1024);
    for (size_t i = 0; i < large_weights.size(); ++i) large_weights[i] = float(i % 997) / 997.0F;
    io::write_safetensors_file(root / "large.safetensors", {
        {"weight", "F32", {1024, 1024}, bytes(large_weights)}
    });
    auto large_base = assets::open_tensor_source(root / "large.safetensors");
    assets::LoraTensorDelta large_delta;
    large_delta.in = large_delta.out = 1024;
    large_delta.r = 1;
    large_delta.a.assign(1024, 0.123F);
    large_delta.b.assign(1024, -0.456F);
    auto large_cached = assets::make_lora_tensor_source(large_base, {{"weight", large_delta}}, {}, "large", true);
    auto large_reference = assets::make_lora_tensor_source(large_base, {{"weight", large_delta}});
    for (auto type : {assets::TensorStorageType::F16, assets::TensorStorageType::BF16}) {
        check_upload(*large_reference, "weight", {1024, 1024}, type);
        check_upload(*large_cached, "weight", {1024, 1024}, type, large_reference.get());
        check_upload(*large_cached, "weight", {1024, 1024}, type, large_reference.get());
    }
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
