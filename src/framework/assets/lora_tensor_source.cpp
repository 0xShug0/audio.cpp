#include "engine/framework/assets/lora_tensor_source.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"

#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace engine::assets {
namespace {
#ifdef _OPENMP
constexpr int64_t kParallelLoraMergeWorkItems = 1ll << 20;
#endif

struct LoraMergeResult {
    std::vector<float> values;
    double base_read_ms = 0.0;
    double compute_ms = 0.0;
};

LoraMergeResult merged_f32_values(
    const TensorSource & base,
    const std::string & name,
    const LoraTensorDelta & delta) {
    const auto base_read_started = std::chrono::steady_clock::now();
    auto values = base.require_f32(name, std::optional<std::vector<int64_t>>({delta.out, delta.in}));
    const double base_read_ms = engine::debug::elapsed_ms(base_read_started);
    // values[o, i] += scale * sum_k B[o, k] * A[k, i]
    const auto compute_started = std::chrono::steady_clock::now();
    if (delta.merge_mode == LoraMergeMode::RoundedBF16Delta) {
        std::vector<float> product(values.size(), 0.0F);
        #ifdef _OPENMP
        const int64_t merge_work_items = delta.out * delta.in * delta.r;
        #pragma omp parallel for if(merge_work_items >= kParallelLoraMergeWorkItems) schedule(static)
        #endif
        for (int64_t o = 0; o < delta.out; ++o) {
            const int64_t row = o * delta.in;
            for (int64_t k = 0; k < delta.r; ++k) {
                const float b = delta.b[static_cast<size_t>(o * delta.r + k)];
                const float * a = delta.a.data() + static_cast<size_t>(k * delta.in);
                for (int64_t i = 0; i < delta.in; ++i) product[row + i] += b * a[i];
            }
            for (int64_t i = 0; i < delta.in; ++i) {
                const float update = ggml_bf16_to_fp32(ggml_fp32_to_bf16(delta.scale * product[row + i]));
                values[row + i] = ggml_bf16_to_fp32(ggml_fp32_to_bf16(values[row + i] + update));
            }
        }
        return {std::move(values), base_read_ms, engine::debug::elapsed_ms(compute_started)};
    }
    #ifdef _OPENMP
    const int64_t merge_work_items = delta.out * delta.in * delta.r;
    #pragma omp parallel for if(merge_work_items >= kParallelLoraMergeWorkItems) schedule(static)
    #endif
    for (int64_t o = 0; o < delta.out; ++o) {
        const int64_t row = o * delta.in;
        for (int64_t k = 0; k < delta.r; ++k) {
            const float b = delta.scale * delta.b[static_cast<size_t>(o * delta.r + k)];
            if (b == 0.0F) {
                continue;
            }
            const float * a_row = delta.a.data() + static_cast<size_t>(k * delta.in);
            for (int64_t i = 0; i < delta.in; ++i) {
                values[static_cast<size_t>(row + i)] += b * a_row[i];
            }
        }
    }
    return LoraMergeResult{std::move(values), base_read_ms, engine::debug::elapsed_ms(compute_started)};
}

// A weight source that overlays a fine-tune adapter onto a base source: full overrides replace a
// base tensor outright, LoRA deltas add B*A to it, everything else falls through to the base.
class LoraTensorSource final : public TensorSource {
public:
    LoraTensorSource(
        std::shared_ptr<const TensorSource> base,
        std::unordered_map<std::string, LoraTensorDelta> deltas,
        std::unordered_map<std::string, TensorOverride> overrides,
        std::string log_prefix,
        bool cache_backend_weights)
        : base_(std::move(base)),
          deltas_(std::move(deltas)),
          overrides_(std::move(overrides)), log_prefix_(std::move(log_prefix)),
          cache_backend_weights_(cache_backend_weights) {}

    const std::filesystem::path & source_path() const noexcept override {
        return base_->source_path();
    }

    bool has_tensor(std::string_view name) const noexcept override {
        return base_->has_tensor(name);
    }

    TensorMetadata require_metadata(std::string_view name) const override {
        return base_->require_metadata(name);
    }

    std::vector<TensorMetadata> tensors() const override {
        return base_->tensors();
    }

    void release_storage() const override {
        base_->release_storage();
    }

    RawTensorData require_tensor_data(std::string_view name) const override {
        const std::string key(name);
        if (const auto override_it = overrides_.find(key); override_it != overrides_.end()) {
            const auto started = std::chrono::steady_clock::now();
            auto raw = raw_from_f32(key, override_it->second.shape, override_it->second.values);
            record_override_export(engine::debug::elapsed_ms(started), override_it->second.values.size());
            return raw;
        }
        const auto delta = deltas_.find(key);
        if (delta == deltas_.end()) {
            return base_->require_tensor_data(name);
        }
        auto merged = merged_f32_values(*base_, key, delta->second);
        record_decoder_merge(merged.base_read_ms, merged.compute_ms, merged.values.size());
        return raw_from_f32(key, {delta->second.out, delta->second.in}, merged.values);
    }

    void set_backend_tensor(
        ggml_tensor * tensor,
        std::string_view name,
        TensorStorageType storage_type,
        const std::vector<int64_t> & expected_shape) const override {
        const std::string key(name);
        const auto shape = shape_from_expected(key, expected_shape);
        const ggml_type type = ggml_type_for_tensor_storage(storage_type);
        if (const auto override_it = overrides_.find(key); override_it != overrides_.end()) {
            if (override_it->second.shape != expected_shape) {
                throw std::runtime_error("tensor shape mismatch for " + key);
            }
            const auto started = std::chrono::steady_clock::now();
            set_backend_tensor_from_f32_parallel(tensor, key, override_it->second.values, shape, type);
            record_override_upload(engine::debug::elapsed_ms(started), override_it->second.values.size());
            return;
        }
        const auto delta = deltas_.find(key);
        if (delta == deltas_.end()) {
            base_->set_backend_tensor(tensor, name, storage_type, expected_shape);
            return;
        }
        if (expected_shape != std::vector<int64_t>({delta->second.out, delta->second.in})) {
            throw std::runtime_error("tensor shape mismatch for " + key);
        }
        std::unique_lock<std::mutex> lock(upload_cache_mutex_, std::defer_lock);
        if (cache_backend_weights_) {
            lock.lock();
            auto cached = upload_cache_.find(key);
            if (cached != upload_cache_.end() && cached->second.type != type) {
                // Replace, never accumulate precision variants or requantize a lossy copy.
                upload_cache_.erase(cached);
                cached = upload_cache_.end();
            }
            if (cached != upload_cache_.end()) {
                const auto & data = cached->second;
                if (tensor->type != data.type || ggml_nbytes(tensor) != data.bytes.size()) {
                    throw std::runtime_error("backend tensor storage mismatch for " + key);
                }
                ggml_backend_tensor_set(tensor, data.bytes.data(), 0, data.bytes.size());
                return;
            }
        }
        auto merged = merged_f32_values(*base_, key, delta->second);
        record_decoder_merge(merged.base_read_ms, merged.compute_ms, merged.values.size());
        const auto upload_started = std::chrono::steady_clock::now();
        TensorData cached{shape, type, {}};
        set_backend_tensor_from_f32_parallel(
            tensor, key, merged.values, shape, type, cache_backend_weights_ ? &cached.bytes : nullptr);
        if (cache_backend_weights_) upload_cache_.emplace(key, std::move(cached));
        record_decoder_upload(engine::debug::elapsed_ms(upload_started), merged.values.size());
    }

    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        if (expected_shape) {
            require_tensor_shape(*base_, name, *expected_shape);
        }
        const std::string key(name);
        if (const auto override_it = overrides_.find(key); override_it != overrides_.end()) {
            const auto started = std::chrono::steady_clock::now();
            auto values = override_it->second.values;
            record_override_export(engine::debug::elapsed_ms(started), values.size());
            return values;
        }
        const auto delta = deltas_.find(key);
        if (delta == deltas_.end()) {
            return base_->require_f32(name, expected_shape);
        }
        auto merged = merged_f32_values(*base_, key, delta->second);
        record_decoder_merge(merged.base_read_ms, merged.compute_ms, merged.values.size());
        return std::move(merged.values);
    }

    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        if (!has_tensor(name)) {
            return std::nullopt;
        }
        return require_f32(name, expected_shape);
    }

    int64_t require_i64_scalar(std::string_view name) const override {
        return base_->require_i64_scalar(name);
    }

private:
    static RawTensorData raw_from_f32(
        const std::string & name,
        const std::vector<int64_t> & shape,
        const std::vector<float> & values) {
        RawTensorData raw;
        raw.metadata = TensorMetadata{name, "F32", shape};
        raw.bytes.resize(values.size() * sizeof(float));
        std::memcpy(raw.bytes.data(), values.data(), raw.bytes.size());
        return raw;
    }

    static engine::core::TensorShape shape_from_expected(
        const std::string & name,
        const std::vector<int64_t> & expected_shape) {
        switch (expected_shape.size()) {
            case 1:
                return engine::core::TensorShape::from_dims({expected_shape[0]});
            case 2:
                return engine::core::TensorShape::from_dims({expected_shape[0], expected_shape[1]});
            case 3:
                return engine::core::TensorShape::from_dims({expected_shape[0], expected_shape[1], expected_shape[2]});
            case 4:
                return engine::core::TensorShape::from_dims(
                    {expected_shape[0], expected_shape[1], expected_shape[2], expected_shape[3]});
            default:
                throw std::runtime_error("tensor rank must be between 1 and 4: " + name);
        }
    }

    void record_decoder_merge(double base_read_ms, double compute_ms, size_t values) const {
        decoder_base_read_ms_ += base_read_ms;
        decoder_merge_compute_ms_ += compute_ms;
        decoder_merge_values_ += static_cast<uint64_t>(values);
        ++decoder_merge_tensors_;
        if (!decoder_merge_logged_ && decoder_merge_tensors_ == deltas_.size()) {
            engine::debug::timing_log_scalar(log_prefix_ + ".decoder_base_read_ms", decoder_base_read_ms_);
            engine::debug::timing_log_scalar(log_prefix_ + ".decoder_merge_compute_ms", decoder_merge_compute_ms_);
            engine::debug::timing_log_scalar(
                log_prefix_ + ".decoder_merge_ms",
                decoder_base_read_ms_ + decoder_merge_compute_ms_);
            engine::debug::timing_log_scalar(log_prefix_ + ".decoder_merge_tensors", decoder_merge_tensors_);
            engine::debug::timing_log_scalar(log_prefix_ + ".decoder_merge_values", decoder_merge_values_);
            decoder_merge_logged_ = true;
        }
    }

    void record_decoder_upload(double ms, size_t values) const {
        decoder_upload_ms_ += ms;
        decoder_upload_values_ += static_cast<uint64_t>(values);
        ++decoder_upload_tensors_;
        if (!decoder_upload_logged_ && decoder_upload_tensors_ == deltas_.size()) {
            engine::debug::timing_log_scalar(log_prefix_ + ".decoder_upload_ms", decoder_upload_ms_);
            engine::debug::timing_log_scalar(log_prefix_ + ".decoder_upload_tensors", decoder_upload_tensors_);
            engine::debug::timing_log_scalar(log_prefix_ + ".decoder_upload_values", decoder_upload_values_);
            decoder_upload_logged_ = true;
        }
    }

    void record_override_export(double ms, size_t values) const {
        override_export_ms_ += ms;
        override_export_values_ += static_cast<uint64_t>(values);
        ++override_export_tensors_;
        maybe_log_override_summary();
    }

    void record_override_upload(double ms, size_t values) const {
        override_upload_ms_ += ms;
        override_upload_values_ += static_cast<uint64_t>(values);
        ++override_upload_tensors_;
        maybe_log_override_summary();
    }

    void maybe_log_override_summary() const {
        if (!override_summary_logged_ && override_upload_tensors_ + override_export_tensors_ == overrides_.size()) {
            engine::debug::timing_log_scalar(log_prefix_ + ".override_export_ms", override_export_ms_);
            engine::debug::timing_log_scalar(log_prefix_ + ".override_export_tensors", override_export_tensors_);
            engine::debug::timing_log_scalar(log_prefix_ + ".override_export_values", override_export_values_);
            engine::debug::timing_log_scalar(log_prefix_ + ".override_upload_ms", override_upload_ms_);
            engine::debug::timing_log_scalar(log_prefix_ + ".override_upload_tensors", override_upload_tensors_);
            engine::debug::timing_log_scalar(log_prefix_ + ".override_upload_values", override_upload_values_);
            override_summary_logged_ = true;
        }
    }

    std::shared_ptr<const TensorSource> base_;
    std::unordered_map<std::string, LoraTensorDelta> deltas_;
    std::unordered_map<std::string, TensorOverride> overrides_;
    std::string log_prefix_;
    bool cache_backend_weights_;
    mutable std::mutex upload_cache_mutex_;
    mutable std::unordered_map<std::string, TensorData> upload_cache_;
    mutable double decoder_base_read_ms_ = 0.0;
    mutable double decoder_merge_compute_ms_ = 0.0;
    mutable uint64_t decoder_merge_values_ = 0;
    mutable size_t decoder_merge_tensors_ = 0;
    mutable bool decoder_merge_logged_ = false;
    mutable double decoder_upload_ms_ = 0.0;
    mutable uint64_t decoder_upload_values_ = 0;
    mutable size_t decoder_upload_tensors_ = 0;
    mutable bool decoder_upload_logged_ = false;
    mutable double override_export_ms_ = 0.0;
    mutable uint64_t override_export_values_ = 0;
    mutable size_t override_export_tensors_ = 0;
    mutable double override_upload_ms_ = 0.0;
    mutable uint64_t override_upload_values_ = 0;
    mutable size_t override_upload_tensors_ = 0;
    mutable bool override_summary_logged_ = false;
};

}  // namespace

LoraTensorDelta load_lora_tensor_delta(
    const TensorSource & base, const TensorSource & adapter,
    const std::string & base_name, const std::string & a_name,
    const std::string & b_name, float scale) {
    const auto a_meta = adapter.require_metadata(a_name);
    const auto b_meta = adapter.require_metadata(b_name);
    if (a_meta.shape.size() != 2 || b_meta.shape.size() != 2) {
        throw std::runtime_error("LoRA A/B tensors must be rank-2: " + base_name);
    }
    LoraTensorDelta delta;
    delta.r = a_meta.shape[0];
    delta.in = a_meta.shape[1];
    delta.out = b_meta.shape[0];
    delta.scale = scale;
    if (b_meta.shape[1] != delta.r) {
        throw std::runtime_error("LoRA A/B rank mismatch for " + base_name);
    }
    if (base.require_metadata(base_name).shape != std::vector<int64_t>({delta.out, delta.in})) {
        throw std::runtime_error(
            "LoRA shape mismatch for " + base_name + " (adapter is trained for a different model size)");
    }
    delta.a = adapter.require_f32(a_name, std::optional<std::vector<int64_t>>({delta.r, delta.in}));
    delta.b = adapter.require_f32(b_name, std::optional<std::vector<int64_t>>({delta.out, delta.r}));
    return delta;
}

std::shared_ptr<const TensorSource> make_lora_tensor_source(
    std::shared_ptr<const TensorSource> base,
    std::unordered_map<std::string, LoraTensorDelta> deltas,
    std::unordered_map<std::string, TensorOverride> overrides,
    std::string log_prefix,
    bool cache_backend_weights) {
    if (!base) {
        throw std::runtime_error("LoRA overlay requires a base tensor source");
    }
    if (deltas.empty() && overrides.empty()) {
        return base;
    }
    for (const auto & [name, delta] : deltas) {
        if (delta.r <= 0 || delta.in <= 0 || delta.out <= 0 ||
            delta.a.size() % delta.r != 0 || delta.a.size() / delta.r != static_cast<size_t>(delta.in) ||
            delta.b.size() % delta.r != 0 || delta.b.size() / delta.r != static_cast<size_t>(delta.out)) {
            throw std::runtime_error("LoRA A/B dimensions or values mismatch for " + name);
        }
        require_tensor_shape(*base, name, {delta.out, delta.in});
    }
    for (const auto & [name, tensor] : overrides) {
        require_tensor_shape(*base, name, tensor.shape);
        size_t count = 1;
        for (const auto dim : tensor.shape) {
            if (dim <= 0 || count > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
                throw std::runtime_error("invalid override shape for " + name);
            }
            count *= static_cast<size_t>(dim);
        }
        if (count != tensor.values.size()) {
            throw std::runtime_error("override value count mismatch for " + name);
        }
    }
    return std::make_shared<LoraTensorSource>(
        std::move(base), std::move(deltas), std::move(overrides), std::move(log_prefix), cache_backend_weights);
}

}  // namespace engine::assets
