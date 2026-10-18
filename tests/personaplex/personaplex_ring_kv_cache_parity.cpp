#include "engine/framework/core/backend.h"
#include "engine/framework/modules/structural_modules.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace core = engine::core;
namespace modules = engine::modules;

struct BackendDeleter {
    void operator()(ggml_backend * backend) const noexcept {
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

core::TensorValue slice_bhtd_time(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t start,
    int64_t length) {
    return modules::SliceModule({2, start, length}).build(ctx, input);
}

core::TensorValue set_ring_cache_bhtd_rows(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    const core::TensorValue & row,
    const core::TensorValue & row_indexes) {
    require(cache.shape.rank == 4 && row.shape.rank == 4, "ring cache row update expects rank-4 tensors");
    require(cache.shape.dims[0] == 1 && row.shape.dims[0] == 1, "ring cache row update expects batch size 1");
    require(cache.shape.dims[1] == row.shape.dims[1], "ring cache row update head count mismatch");
    require(row.shape.dims[2] == 1, "ring cache row update expects one time step");
    require(cache.shape.dims[3] == row.shape.dims[3], "ring cache row update head dim mismatch");
    require(row_indexes.shape.rank == 1 && row_indexes.shape.dims[0] == cache.shape.dims[1],
            "ring cache row update expects one index per head");
    const int64_t heads = cache.shape.dims[1];
    const int64_t steps = cache.shape.dims[2];
    const int64_t head_dim = cache.shape.dims[3];
    auto flat_cache = core::reshape_tensor(ctx, cache, core::TensorShape::from_dims({heads * steps, head_dim}));
    auto flat_row = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, row),
        core::TensorShape::from_dims({heads, head_dim}));
    auto * updated_raw = ggml_set_rows(ctx.ggml, flat_cache.tensor, flat_row.tensor, row_indexes.tensor);
    updated_raw->src[2] = cache.tensor;
    auto updated = core::wrap_tensor(updated_raw, flat_cache.shape, cache.type);
    return core::reshape_tensor(ctx, updated, cache.shape);
}

struct RingOracle {
    int64_t heads = 0;
    int64_t head_dim = 0;
    int64_t capacity = 0;
    int64_t end_offset = 0;
    std::vector<float> key_cache;
    std::vector<float> value_cache;

    RingOracle(int64_t input_heads, int64_t input_head_dim, int64_t input_capacity)
        : heads(input_heads),
          head_dim(input_head_dim),
          capacity(input_capacity),
          key_cache(static_cast<size_t>(input_capacity * input_heads * input_head_dim), 0.0F),
          value_cache(static_cast<size_t>(input_capacity * input_heads * input_head_dim), 0.0F) {}

    std::vector<int32_t> slots(int64_t frames) const {
        std::vector<int32_t> out(static_cast<size_t>(frames));
        for (int64_t frame = 0; frame < frames; ++frame) {
            out[static_cast<size_t>(frame)] = static_cast<int32_t>((end_offset + frame) % capacity);
        }
        return out;
    }

    void complete(const std::vector<float> & key_chunk, const std::vector<float> & value_chunk, int64_t frames) {
        require(static_cast<int64_t>(key_chunk.size()) == frames * heads * head_dim, "oracle key chunk size mismatch");
        require(static_cast<int64_t>(value_chunk.size()) == frames * heads * head_dim, "oracle value chunk size mismatch");
        for (int64_t frame = 0; frame < frames; ++frame) {
            const int64_t slot = (end_offset + frame) % capacity;
            for (int64_t head = 0; head < heads; ++head) {
                const size_t src = static_cast<size_t>((head * frames + frame) * head_dim);
                const size_t dst = static_cast<size_t>((head * capacity + slot) * head_dim);
                std::copy(
                    key_chunk.begin() + static_cast<std::ptrdiff_t>(src),
                    key_chunk.begin() + static_cast<std::ptrdiff_t>(src + head_dim),
                    key_cache.begin() + static_cast<std::ptrdiff_t>(dst));
                std::copy(
                    value_chunk.begin() + static_cast<std::ptrdiff_t>(src),
                    value_chunk.begin() + static_cast<std::ptrdiff_t>(src + head_dim),
                    value_cache.begin() + static_cast<std::ptrdiff_t>(dst));
            }
        }
        end_offset += frames;
    }

    std::vector<int64_t> positions() const {
        std::vector<int64_t> out(static_cast<size_t>(capacity), -1);
        const int64_t end_index = end_offset % capacity;
        for (int64_t index = 0; index < capacity; ++index) {
            const bool invalid = index >= end_offset;
            const int64_t delta = index - end_index;
            const int64_t pos = delta < 0 ? end_offset + delta : end_offset + delta - capacity;
            out[static_cast<size_t>(index)] = invalid ? -1 : pos;
        }
        return out;
    }

    std::vector<float> attention_mask(int64_t query_start, int64_t query_frames) const {
        const auto pos = positions();
        std::vector<float> out(
            static_cast<size_t>(query_frames * capacity),
            -std::numeric_limits<float>::infinity());
        for (int64_t query = 0; query < query_frames; ++query) {
            const int64_t pos_q = query_start + query;
            for (int64_t slot = 0; slot < capacity; ++slot) {
                const int64_t pos_k = pos[static_cast<size_t>(slot)];
                const int64_t delta = pos_q - pos_k;
                if (pos_k >= 0 && delta >= 0 && delta < capacity) {
                    out[static_cast<size_t>(query * capacity + slot)] = 0.0F;
                }
            }
        }
        return out;
    }
};

std::vector<float> make_chunk(int64_t start, int64_t frames, int64_t heads, int64_t head_dim, float scale) {
    std::vector<float> out(static_cast<size_t>(frames * heads * head_dim));
    for (int64_t frame = 0; frame < frames; ++frame) {
        const int64_t absolute_frame = start + frame;
        for (int64_t head = 0; head < heads; ++head) {
            for (int64_t dim = 0; dim < head_dim; ++dim) {
                const float phase = static_cast<float>(
                    absolute_frame * 0.013 + head * 0.071 + dim * 0.003);
                out[static_cast<size_t>((head * frames + frame) * head_dim + dim)] =
                    std::sin(phase) * scale + std::cos(phase * 0.37F) * (scale * 0.25F);
            }
        }
    }
    return out;
}

std::vector<float> bhtd_to_bthd(const std::vector<float> & input, int64_t frames, int64_t heads, int64_t head_dim) {
    std::vector<float> output(input.size(), 0.0F);
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int64_t head = 0; head < heads; ++head) {
            const size_t src = static_cast<size_t>((head * frames + frame) * head_dim);
            const size_t dst = static_cast<size_t>((frame * heads + head) * head_dim);
            std::copy(
                input.begin() + static_cast<std::ptrdiff_t>(src),
                input.begin() + static_cast<std::ptrdiff_t>(src + head_dim),
                output.begin() + static_cast<std::ptrdiff_t>(dst));
        }
    }
    return output;
}

void assert_close(
    const std::vector<float> & actual,
    const std::vector<float> & expected,
    const std::string & label,
    float tolerance = 1.0e-6F) {
    require(actual.size() == expected.size(), label + " size mismatch");
    float max_diff = 0.0F;
    size_t max_index = 0;
    for (size_t index = 0; index < actual.size(); ++index) {
        const float diff = std::fabs(actual[index] - expected[index]);
        if (diff > max_diff) {
            max_diff = diff;
            max_index = index;
        }
    }
    if (max_diff > tolerance) {
        throw std::runtime_error(
            label + " mismatch max_abs_diff=" + std::to_string(max_diff) +
            " index=" + std::to_string(max_index) +
            " actual=" + std::to_string(actual[max_index]) +
            " expected=" + std::to_string(expected[max_index]));
    }
}

void assert_equal(
    const std::vector<int64_t> & actual,
    const std::vector<int64_t> & expected,
    const std::string & label) {
    require(actual.size() == expected.size(), label + " size mismatch");
    for (size_t index = 0; index < actual.size(); ++index) {
        if (actual[index] != expected[index]) {
            throw std::runtime_error(
                label + " mismatch index=" + std::to_string(index) +
                " actual=" + std::to_string(actual[index]) +
                " expected=" + std::to_string(expected[index]));
        }
    }
}

class LocalRingKVCachePrimitive {
public:
    LocalRingKVCachePrimitive(
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        int64_t heads,
        int64_t head_dim,
        int64_t capacity,
        int64_t chunk_frames)
        : backend_(backend),
          heads_(heads),
          head_dim_(head_dim),
          capacity_(capacity),
          chunk_frames_(chunk_frames),
          key_cache_host_(static_cast<size_t>(capacity * heads * head_dim), 0.0F),
          value_cache_host_(static_cast<size_t>(capacity * heads * head_dim), 0.0F) {
        ctx_.reset(ggml_init({32ull * 1024ull * 1024ull, nullptr, true}));
        require(ctx_ != nullptr, "ring kv cache test ctx init failed");
        core::ModuleBuildContext build{ctx_.get(), "personaplex.ring_kv_cache_test", backend_type};
        key_cache_ = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({1, heads, capacity, head_dim}));
        value_cache_ = core::make_tensor(build, GGML_TYPE_F32, key_cache_.shape);
        key_chunk_bthd_ = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({1, chunk_frames, heads, head_dim}));
        value_chunk_bthd_ = core::make_tensor(build, GGML_TYPE_F32, key_chunk_bthd_.shape);
        key_chunk_ = core::ensure_backend_addressable_layout(
            build,
            modules::TransposeModule({{0, 2, 1, 3}, 4}).build(build, key_chunk_bthd_));
        value_chunk_ = core::ensure_backend_addressable_layout(
            build,
            modules::TransposeModule({{0, 2, 1, 3}, 4}).build(build, value_chunk_bthd_));
        for (int64_t frame = 0; frame < chunk_frames; ++frame) {
            slot_inputs_.push_back(core::make_tensor(build, GGML_TYPE_I32, core::TensorShape::from_dims({heads})));
        }

        auto updated_key = key_cache_;
        auto updated_value = value_cache_;
        for (int64_t frame = 0; frame < chunk_frames; ++frame) {
            auto key_row = slice_bhtd_time(build, key_chunk_, frame, 1);
            auto value_row = slice_bhtd_time(build, value_chunk_, frame, 1);
            updated_key = set_ring_cache_bhtd_rows(build, updated_key, key_row, slot_inputs_[static_cast<size_t>(frame)]);
            updated_value = set_ring_cache_bhtd_rows(build, updated_value, value_row, slot_inputs_[static_cast<size_t>(frame)]);
        }
        updated_key_ = updated_key;
        updated_value_ = updated_value;

        core::set_backend_threads(backend_, threads);
        graph_ = ggml_new_graph_custom(ctx_.get(), 32768, false);
        ggml_build_forward_expand(graph_, updated_key_.tensor);
        ggml_build_forward_expand(graph_, updated_value_.tensor);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        require(gallocr_ != nullptr && ggml_gallocr_reserve(gallocr_, graph_) && ggml_gallocr_alloc_graph(gallocr_, graph_),
                "ring kv cache test graph allocation failed");
        core::write_tensor_f32(key_cache_, key_cache_host_);
        core::write_tensor_f32(value_cache_, value_cache_host_);
    }

    ~LocalRingKVCachePrimitive() {
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    void reset() {
        end_offset_ = 0;
        std::fill(key_cache_host_.begin(), key_cache_host_.end(), 0.0F);
        std::fill(value_cache_host_.begin(), value_cache_host_.end(), 0.0F);
        core::write_tensor_f32(key_cache_, key_cache_host_);
        core::write_tensor_f32(value_cache_, value_cache_host_);
    }

    void complete(const std::vector<float> & key_chunk, const std::vector<float> & value_chunk) {
        require(static_cast<int64_t>(key_chunk.size()) == chunk_frames_ * heads_ * head_dim_, "key chunk size mismatch");
        require(static_cast<int64_t>(value_chunk.size()) == chunk_frames_ * heads_ * head_dim_, "value chunk size mismatch");
        core::write_tensor_f32(key_chunk_bthd_, bhtd_to_bthd(key_chunk, chunk_frames_, heads_, head_dim_));
        core::write_tensor_f32(value_chunk_bthd_, bhtd_to_bthd(value_chunk, chunk_frames_, heads_, head_dim_));
        for (int64_t frame = 0; frame < chunk_frames_; ++frame) {
            const auto indexes = row_indexes(frame);
            core::write_tensor_i32(slot_inputs_[static_cast<size_t>(frame)], indexes);
        }
        require(core::compute_backend_graph(backend_, graph_) == GGML_STATUS_SUCCESS, "ring kv cache graph failed");
        key_cache_host_ = core::read_tensor_f32(updated_key_.tensor);
        value_cache_host_ = core::read_tensor_f32(updated_value_.tensor);
        ggml_backend_tensor_copy(updated_key_.tensor, key_cache_.tensor);
        ggml_backend_tensor_copy(updated_value_.tensor, value_cache_.tensor);
        end_offset_ += chunk_frames_;
    }

    std::vector<int64_t> positions() const {
        std::vector<int64_t> out(static_cast<size_t>(capacity_), -1);
        const int64_t end_index = end_offset_ % capacity_;
        for (int64_t index = 0; index < capacity_; ++index) {
            const bool invalid = index >= end_offset_;
            const int64_t delta = index - end_index;
            const int64_t pos = delta < 0 ? end_offset_ + delta : end_offset_ + delta - capacity_;
            out[static_cast<size_t>(index)] = invalid ? -1 : pos;
        }
        return out;
    }

    std::vector<float> attention_mask(int64_t query_start) const {
        const auto pos = positions();
        std::vector<float> out(
            static_cast<size_t>(chunk_frames_ * capacity_),
            -std::numeric_limits<float>::infinity());
        for (int64_t query = 0; query < chunk_frames_; ++query) {
            const int64_t pos_q = query_start + query;
            for (int64_t slot = 0; slot < capacity_; ++slot) {
                const int64_t pos_k = pos[static_cast<size_t>(slot)];
                const int64_t delta = pos_q - pos_k;
                if (pos_k >= 0 && delta >= 0 && delta < capacity_) {
                    out[static_cast<size_t>(query * capacity_ + slot)] = 0.0F;
                }
            }
        }
        return out;
    }

    const std::vector<float> & key_cache() const noexcept {
        return key_cache_host_;
    }

    const std::vector<float> & value_cache() const noexcept {
        return value_cache_host_;
    }

private:
    std::vector<int32_t> row_indexes(int64_t frame) const {
        std::vector<int32_t> out(static_cast<size_t>(heads_));
        const int64_t slot = (end_offset_ + frame) % capacity_;
        for (int64_t head = 0; head < heads_; ++head) {
            out[static_cast<size_t>(head)] = static_cast<int32_t>(head * capacity_ + slot);
        }
        return out;
    }

    ggml_backend_t backend_ = nullptr;
    int64_t heads_ = 0;
    int64_t head_dim_ = 0;
    int64_t capacity_ = 0;
    int64_t chunk_frames_ = 0;
    int64_t end_offset_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    core::TensorValue key_cache_;
    core::TensorValue value_cache_;
    core::TensorValue key_chunk_bthd_;
    core::TensorValue value_chunk_bthd_;
    core::TensorValue key_chunk_;
    core::TensorValue value_chunk_;
    std::vector<core::TensorValue> slot_inputs_;
    core::TensorValue updated_key_;
    core::TensorValue updated_value_;
    std::vector<float> key_cache_host_;
    std::vector<float> value_cache_host_;
};

void run_case(LocalRingKVCachePrimitive & primitive, RingOracle & oracle, int64_t absolute_start, int64_t chunk_frames) {
    const auto key = make_chunk(absolute_start, chunk_frames, oracle.heads, oracle.head_dim, 0.73F);
    const auto value = make_chunk(absolute_start, chunk_frames, oracle.heads, oracle.head_dim, 0.41F);
    const int64_t query_start = oracle.end_offset;
    primitive.complete(key, value);
    oracle.complete(key, value, chunk_frames);
    assert_close(primitive.key_cache(), oracle.key_cache, "ring key cache");
    assert_close(primitive.value_cache(), oracle.value_cache, "ring value cache");
    assert_equal(primitive.positions(), oracle.positions(), "ring positions");
    assert_close(primitive.attention_mask(query_start), oracle.attention_mask(query_start, chunk_frames), "ring attention mask");
}

}  // namespace

int main() {
    try {
        constexpr int64_t kHeads = 8;
        constexpr int64_t kHeadDim = 64;
        constexpr int64_t kCapacity = 250;
        constexpr int64_t kChunkFrames = 2;
        constexpr int kThreads = 8;

        core::BackendConfig backend_config;
        backend_config.type = core::BackendType::Cuda;
        backend_config.device = 0;
        backend_config.threads = kThreads;
        std::unique_ptr<ggml_backend, BackendDeleter> backend(core::init_backend(backend_config));
        require(backend != nullptr, "failed to initialize CUDA backend");

        LocalRingKVCachePrimitive primitive(
            backend.get(),
            core::BackendType::Cuda,
            kThreads,
            kHeads,
            kHeadDim,
            kCapacity,
            kChunkFrames);
        RingOracle oracle(kHeads, kHeadDim, kCapacity);

        run_case(primitive, oracle, 0, kChunkFrames);
        for (int64_t step = 1; step < 124; ++step) {
            run_case(primitive, oracle, step * kChunkFrames, kChunkFrames);
        }
        run_case(primitive, oracle, 248, kChunkFrames);
        run_case(primitive, oracle, 250, kChunkFrames);
        run_case(primitive, oracle, 252, kChunkFrames);

        primitive.reset();
        oracle = RingOracle(kHeads, kHeadDim, kCapacity);
        run_case(primitive, oracle, 1000, kChunkFrames);

        std::cout << "personaplex_ring_kv_cache_parity passed"
                  << " heads=" << kHeads
                  << " head_dim=" << kHeadDim
                  << " capacity=" << kCapacity
                  << " chunk_frames=" << kChunkFrames << "\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "personaplex_ring_kv_cache_parity failed: " << ex.what() << "\n";
        return 1;
    }
}
