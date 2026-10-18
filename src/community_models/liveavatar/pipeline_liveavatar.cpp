#include "pipeline_internal.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace engine::community_models::liveavatar {
namespace {

constexpr size_t kLiveAvatarDenoiserGraphContextBytes = 4096ull * 1024ull * 1024ull;
constexpr int64_t kLiveAvatarStreamChunkFrames = 12;

class LiveAvatarWeightStreamingGraph {
public:
    LiveAvatarWeightStreamingGraph(
        engine::core::ExecutionContext & execution,
        ggml_cgraph * graph)
        : execution_(execution) {
        auto * cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_device == nullptr) {
            throw std::runtime_error("LiveAvatar denoiser weight streaming requires the CPU backend");
        }
        cpu_backend_ = ggml_backend_dev_init(cpu_device, nullptr);
        if (cpu_backend_ == nullptr) {
            throw std::runtime_error("LiveAvatar denoiser weight streaming could not initialize the CPU backend");
        }
        engine::core::set_backend_threads(cpu_backend_, std::max(1, execution_.config().threads));
        auto * host_buffer_type = ggml_backend_dev_host_buffer_type(
            ggml_backend_get_device(execution_.backend()));
        if (host_buffer_type == nullptr) {
            throw std::runtime_error("LiveAvatar denoiser weight streaming requires accelerator host buffers");
        }
        ggml_backend_t backends[] = {execution_.backend(), cpu_backend_};
        ggml_backend_buffer_type_t buffer_types[] = {
            ggml_backend_get_default_buffer_type(execution_.backend()),
            host_buffer_type,
        };
        scheduler_ = ggml_backend_sched_new(
            backends,
            buffer_types,
            2,
            ggml_graph_size(graph),
            false,
            false);
        if (scheduler_ == nullptr) {
            throw std::runtime_error("LiveAvatar denoiser weight streaming scheduler initialization failed");
        }
        auto assign_execution_backend = [&]() {
            std::unordered_set<ggml_tensor *> visited;
            auto assign_tensor = [&](auto && self, ggml_tensor * tensor) -> void {
                if (tensor == nullptr || !visited.insert(tensor).second) {
                    return;
                }
                self(self, tensor->view_src);
                for (auto * source : tensor->src) {
                    self(self, source);
                }
                const ggml_tensor * storage = tensor;
                while (storage->view_src != nullptr) {
                    storage = storage->view_src;
                }
                const bool host_resident =
                    storage->buffer != nullptr && ggml_backend_buffer_is_host(storage->buffer);
                ggml_backend_sched_set_tensor_backend(
                    scheduler_,
                    tensor,
                    host_resident ? cpu_backend_ : execution_.backend());
            };
            for (int index = 0; index < ggml_graph_n_nodes(graph); ++index) {
                assign_tensor(assign_tensor, ggml_graph_node(graph, index));
            }
        };
        assign_execution_backend();
        if (!ggml_backend_sched_alloc_graph(scheduler_, graph)) {
            throw std::runtime_error("LiveAvatar denoiser weight streaming graph allocation failed");
        }
    }

    ~LiveAvatarWeightStreamingGraph() {
        if (scheduler_ != nullptr) {
            ggml_backend_sched_free(scheduler_);
        }
        if (cpu_backend_ != nullptr) {
            ggml_backend_free(cpu_backend_);
        }
    }

    ggml_status compute(ggml_cgraph * graph) const {
        const ggml_status status = ggml_backend_sched_graph_compute(scheduler_, graph);
        ggml_backend_sched_synchronize(scheduler_);
        return status;
    }

private:
    engine::core::ExecutionContext & execution_;
    ggml_backend_t cpu_backend_ = nullptr;
    ggml_backend_sched_t scheduler_ = nullptr;
};

int64_t liveavatar_motion_token_count(int64_t latent_height, int64_t latent_width) {
    if (latent_height <= 0 || latent_width <= 0) {
        throw std::runtime_error("LiveAvatar LiveAvatar motion token shape is invalid");
    }
    const int64_t post_h = latent_height / 2;
    const int64_t post_w = latent_width / 2;
    const int64_t two_x_h = latent_height / 4;
    const int64_t two_x_w = latent_width / 4;
    const int64_t four_x_h = latent_height / 8;
    const int64_t four_x_w = latent_width / 8;
    return post_h * post_w + two_x_h * two_x_w + 4 * four_x_h * four_x_w;
}

engine::core::TensorValue build_liveavatar_motion_tokens(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & motion_latents,
    const LiveAvatarDenoiserFramePackerWeights & weights,
    const LiveAvatarConfig & config) {
    if (motion_latents.shape.rank != 4 ||
        motion_latents.shape.dims[0] != config.latent_channels ||
        motion_latents.shape.dims[1] != config.latent_motion_frames) {
        throw std::runtime_error("LiveAvatar LiveAvatar motion latent shape is invalid");
    }
    auto clean_4x = engine::modules::SliceModule({1, 0, 16}).build(ctx, motion_latents);
    auto clean_2x = engine::modules::SliceModule({1, 16, 2}).build(ctx, motion_latents);
    auto clean_post = engine::modules::SliceModule({1, 18, 1}).build(ctx, motion_latents);
    auto post = conv3d_tokens(ctx, clean_post, weights.proj, config.latent_channels, config.hidden_size, 1, 2, 2, 1, 2, 2);
    auto two_x = conv3d_tokens(ctx, clean_2x, weights.proj_2x, config.latent_channels, config.hidden_size, 2, 4, 4, 2, 4, 4);
    auto four_x = conv3d_tokens(ctx, clean_4x, weights.proj_4x, config.latent_channels, config.hidden_size, 4, 8, 8, 4, 8, 8);
    return engine::modules::ConcatModule({1, true}).build(
        ctx, engine::modules::ConcatModule({1, true}).build(ctx, post, two_x), four_x);
}

engine::core::TensorValue build_liveavatar_target_mask(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & embedding,
    const engine::core::TensorValue & target_like) {
    const int64_t hidden = target_like.shape.last_dim();
    auto target = engine::modules::SliceModule({0, 0, 1}).build(ctx, embedding);
    engine::core::TensorShape target_broadcast_shape;
    target_broadcast_shape.rank = target_like.shape.rank;
    target_broadcast_shape.dims.fill(1);
    target_broadcast_shape.dims[target_broadcast_shape.rank - 1] = hidden;
    target = engine::core::reshape_tensor(ctx, target, target_broadcast_shape);
    return engine::modules::RepeatModule({target_like.shape}).build(ctx, target);
}

engine::core::TensorValue build_liveavatar_condition_mask(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & embedding,
    const engine::core::TensorValue & ref_like,
    const engine::core::TensorValue & motion_like) {
    const int64_t hidden = ref_like.shape.last_dim();
    auto ref = engine::modules::SliceModule({0, 1, 1}).build(ctx, embedding);
    engine::core::TensorShape ref_broadcast_shape;
    ref_broadcast_shape.rank = ref_like.shape.rank;
    ref_broadcast_shape.dims.fill(1);
    ref_broadcast_shape.dims[ref_broadcast_shape.rank - 1] = hidden;
    ref = engine::core::reshape_tensor(ctx, ref, ref_broadcast_shape);
    ref = engine::modules::RepeatModule({ref_like.shape}).build(ctx, ref);
    auto motion = engine::modules::SliceModule({0, 2, 1}).build(ctx, embedding);
    engine::core::TensorShape motion_broadcast_shape;
    motion_broadcast_shape.rank = motion_like.shape.rank;
    motion_broadcast_shape.dims.fill(1);
    motion_broadcast_shape.dims[motion_broadcast_shape.rank - 1] = hidden;
    motion = engine::core::reshape_tensor(ctx, motion, motion_broadcast_shape);
    motion = engine::modules::RepeatModule({motion_like.shape}).build(ctx, motion);
    return engine::modules::ConcatModule({1, true}).build(ctx, ref, motion);
}

std::vector<LiveAvatarRopeRange> liveavatar_condition_rope_ranges(
    int64_t latent_height,
    int64_t latent_width,
    int64_t reference_rollout_frames = 0) {
    const int64_t grid_h = latent_height / 2;
    const int64_t grid_w = latent_width / 2;
    return {
        {30 + reference_rollout_frames, 0, 0, 31 + reference_rollout_frames, grid_h, grid_w, 1, grid_h, grid_w},
        {-1, 0, 0, 0, grid_h, grid_w, 1, grid_h, grid_w},
        {-3, 0, 0, -2, latent_height / 4, latent_width / 4, 2, grid_h, grid_w},
        {-19, 0, 0, -15, latent_height / 8, latent_width / 8, 16, grid_h, grid_w},
    };
}

std::vector<LiveAvatarRopeRange> liveavatar_target_rope_ranges(
    int64_t target_frames,
    int64_t latent_height,
    int64_t latent_width,
    int64_t target_start_frame) {
    const int64_t grid_h = latent_height / 2;
    const int64_t grid_w = latent_width / 2;
    return {
        {target_start_frame, 0, 0, target_start_frame + target_frames, grid_h, grid_w, target_frames, grid_h, grid_w},
    };
}

int64_t liveavatar_condition_rollout_frames(std::mt19937_64 & rng, int64_t target_start_frame) {
    std::uniform_int_distribution<int64_t> dist(4, 30);
    const int64_t relative_dist = dist(rng);
    const int64_t start_idx = 30 - relative_dist;
    return std::max<int64_t>(0, target_start_frame - start_idx);
}

std::vector<float> make_liveavatar_euler_sigmas(int64_t steps, float shift) {
    if (steps <= 0 || shift <= 0.0F) {
        throw std::runtime_error("LiveAvatar LiveAvatar flow schedule requires positive steps and shift");
    }

    constexpr float kSigmaMax = 1.0F;
    constexpr float kSigmaMin = 0.002994012087583542F;
    std::vector<float> sigmas;
    sigmas.reserve(static_cast<size_t>(steps + 1));
    for (int64_t i = 0; i < steps; ++i) {
        const float ratio = steps == 1 ? 0.0F : static_cast<float>(i) / static_cast<float>(steps - 1);
        const float sigma = kSigmaMax + (kSigmaMin - kSigmaMax) * ratio;
        sigmas.push_back(shift * sigma / (1.0F + (shift - 1.0F) * sigma));
    }
    sigmas.push_back(0.0F);
    return sigmas;
}

std::vector<float> prepend_audio_motion_frames(
    const WanS2VAudioBuckets & audio,
    int64_t motion_frames) {
    if (audio.batch != 1 || audio.layers <= 0 || audio.dims <= 0 || audio.frames <= 0 || motion_frames < 0 ||
        static_cast<int64_t>(audio.values.size()) != audio.batch * audio.layers * audio.dims * audio.frames) {
        throw std::runtime_error("LiveAvatar denoiser audio bucket shape mismatch");
    }
    const int64_t total_frames = motion_frames + audio.frames;
    std::vector<float> out(static_cast<size_t>(audio.layers * audio.dims * total_frames), 0.0F);
    for (int64_t layer = 0; layer < audio.layers; ++layer) {
        for (int64_t dim = 0; dim < audio.dims; ++dim) {
            const float first = audio.values[static_cast<size_t>((layer * audio.dims + dim) * audio.frames)];
            for (int64_t t = 0; t < motion_frames; ++t) {
                out[static_cast<size_t>((layer * audio.dims + dim) * total_frames + t)] = first;
            }
            for (int64_t t = 0; t < audio.frames; ++t) {
                out[static_cast<size_t>((layer * audio.dims + dim) * total_frames + motion_frames + t)] =
                    audio.values[static_cast<size_t>((layer * audio.dims + dim) * audio.frames + t)];
            }
        }
    }
    return out;
}

engine::core::TensorValue view_liveavatar_kv_cache(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & cache,
    int64_t start,
    int64_t tokens,
    int64_t heads,
    int64_t head_dim,
    const char * label) {
    if (start < 0 || tokens <= 0 || start + tokens > cache.shape.dims[1]) {
        throw std::runtime_error(std::string(label) + " cache view range is invalid");
    }
    return engine::core::wrap_tensor(
        ggml_view_4d(
            ctx.ggml,
            cache.tensor,
            head_dim,
            heads,
            tokens,
            1,
            cache.tensor->nb[1],
            cache.tensor->nb[2],
            cache.tensor->nb[3],
            static_cast<size_t>(start) * cache.tensor->nb[2]),
        engine::core::TensorShape::from_dims({1, tokens, heads, head_dim}),
        cache.type);
}

engine::core::TensorValue write_liveavatar_kv_cache(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & src,
    const engine::core::TensorValue & cache,
    int64_t start,
    const char * label) {
    const int64_t tokens = src.shape.dims[1];
    const int64_t heads = src.shape.dims[2];
    const int64_t head_dim = src.shape.dims[3];
    auto dst = view_liveavatar_kv_cache(ctx, cache, start, tokens, heads, head_dim, label);
    return engine::core::wrap_tensor(
        ggml_cpy(ctx.ggml, engine::core::ensure_backend_addressable_layout(ctx, src).tensor, dst.tensor),
        dst.shape,
        dst.type);
}

engine::core::TensorValue materialize_liveavatar_kv_f16(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & src) {
    if (src.type == GGML_TYPE_F16) {
        return engine::core::ensure_backend_addressable_layout(ctx, src);
    }
    auto dst = engine::core::make_tensor(ctx, GGML_TYPE_F16, src.shape);
    return engine::core::wrap_tensor(
        ggml_cpy(ctx.ggml, engine::core::ensure_backend_addressable_layout(ctx, src).tensor, dst.tensor),
        src.shape,
        GGML_TYPE_F16);
}

class LiveAvatarKVCache {
public:
    LiveAvatarKVCache(
        engine::core::ExecutionContext & execution,
        LiveAvatarConfig config,
        int64_t max_target_tokens,
        int64_t condition_tokens,
        LiveAvatarKVCache * condition_owner = nullptr,
        bool host_resident = false)
        : execution_(execution),
          config_(std::move(config)),
          max_target_tokens_(max_target_tokens),
          condition_tokens_(condition_tokens),
          condition_owner_(condition_owner),
          host_resident_(host_resident) {
        build();
    }

    ~LiveAvatarKVCache() {
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    LiveAvatarKVCache(const LiveAvatarKVCache &) = delete;
    LiveAvatarKVCache & operator=(const LiveAvatarKVCache &) = delete;

    bool has_target_cache() const noexcept { return max_target_tokens_ > 0; }
    const engine::core::TensorValue & target_key(size_t layer) const {
        if (!has_target_cache()) {
            throw std::runtime_error("LiveAvatar LiveAvatar target KV cache is disabled");
        }
        return target_keys_.at(layer);
    }
    const engine::core::TensorValue & target_value(size_t layer) const {
        if (!has_target_cache()) {
            throw std::runtime_error("LiveAvatar LiveAvatar target KV cache is disabled");
        }
        return target_values_.at(layer);
    }
    const engine::core::TensorValue & target_key_storage() const {
        if (!has_target_cache()) {
            throw std::runtime_error("LiveAvatar LiveAvatar target KV cache is disabled");
        }
        return target_key_storage_;
    }
    const engine::core::TensorValue & target_value_storage() const {
        if (!has_target_cache()) {
            throw std::runtime_error("LiveAvatar LiveAvatar target KV cache is disabled");
        }
        return target_value_storage_;
    }
    const engine::core::TensorValue & condition_key(size_t layer) const {
        return condition_owner_ != nullptr ? condition_owner_->condition_key(layer) : condition_keys_.at(layer);
    }
    const engine::core::TensorValue & condition_value(size_t layer) const {
        return condition_owner_ != nullptr ? condition_owner_->condition_value(layer) : condition_values_.at(layer);
    }
    int64_t max_target_tokens() const noexcept { return max_target_tokens_; }
    int64_t condition_tokens() const noexcept { return condition_tokens_; }

    void copy_condition_from(const LiveAvatarKVCache & source) {
        if (condition_owner_ != nullptr || source.condition_tokens() != condition_tokens_ ||
            source.config_.num_layers != config_.num_layers) {
            throw std::runtime_error("LiveAvatar condition KV cache copy shape mismatch");
        }
        std::vector<std::byte> staging;
        for (int64_t layer = 0; layer < config_.num_layers; ++layer) {
            const auto copy_tensor = [&](const engine::core::TensorValue & src, const engine::core::TensorValue & dst) {
                const size_t bytes = ggml_nbytes(src.tensor);
                if (bytes != ggml_nbytes(dst.tensor)) {
                    throw std::runtime_error("LiveAvatar condition KV cache tensor size mismatch");
                }
                staging.resize(bytes);
                ggml_backend_tensor_get(src.tensor, staging.data(), 0, bytes);
                ggml_backend_tensor_set(dst.tensor, staging.data(), 0, bytes);
            };
            copy_tensor(source.condition_key(static_cast<size_t>(layer)), condition_key(static_cast<size_t>(layer)));
            copy_tensor(source.condition_value(static_cast<size_t>(layer)), condition_value(static_cast<size_t>(layer)));
        }
    }

private:
    void build() {
        if (max_target_tokens_ < 0 || condition_tokens_ <= 0) {
            throw std::runtime_error("LiveAvatar LiveAvatar KV cache shape is invalid");
        }
        const int64_t hidden = config_.hidden_size;
        const int64_t heads = config_.num_heads;
        const int64_t head_dim = hidden / heads;
        if (hidden % heads != 0) {
            throw std::runtime_error("LiveAvatar LiveAvatar KV cache head shape is invalid");
        }
        ggml_init_params params{
            ggml_tensor_overhead() * static_cast<size_t>(config_.num_layers * 4 + 10),
            nullptr,
            true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar LiveAvatar KV cache context initialization failed");
        }
        engine::core::ModuleBuildContext ctx{ctx_.get(), "liveavatar.kv_cache", execution_.backend_type()};
        auto make_cache = [&](int64_t tokens) {
            return engine::core::make_tensor(
                ctx,
                GGML_TYPE_F16,
                engine::core::TensorShape::from_dims({1, tokens, heads, head_dim}));
        };
        auto make_target_view = [&](const engine::core::TensorValue & storage, int64_t layer) {
            return engine::core::wrap_tensor(
                ggml_view_4d(
                    ctx.ggml,
                    storage.tensor,
                    head_dim,
                    heads,
                    max_target_tokens_,
                    1,
                    storage.tensor->nb[1],
                    storage.tensor->nb[2],
                    storage.tensor->nb[3],
                    static_cast<size_t>(layer) * storage.tensor->nb[3]),
                engine::core::TensorShape::from_dims({1, max_target_tokens_, heads, head_dim}),
                GGML_TYPE_F16);
        };
        if (has_target_cache()) {
            target_key_storage_ = engine::core::make_tensor(
                ctx,
                GGML_TYPE_F16,
                engine::core::TensorShape::from_dims({config_.num_layers, max_target_tokens_, heads, head_dim}));
            target_value_storage_ = engine::core::make_tensor(
                ctx,
                GGML_TYPE_F16,
                engine::core::TensorShape::from_dims({config_.num_layers, max_target_tokens_, heads, head_dim}));
            target_keys_.reserve(static_cast<size_t>(config_.num_layers));
            target_values_.reserve(static_cast<size_t>(config_.num_layers));
        }
        if (condition_owner_ == nullptr) {
            condition_keys_.reserve(static_cast<size_t>(config_.num_layers));
            condition_values_.reserve(static_cast<size_t>(config_.num_layers));
        }
        for (int64_t layer = 0; layer < config_.num_layers; ++layer) {
            if (has_target_cache()) {
                target_keys_.push_back(make_target_view(target_key_storage_, layer));
                target_values_.push_back(make_target_view(target_value_storage_, layer));
            }
            if (condition_owner_ == nullptr) {
                condition_keys_.push_back(make_cache(condition_tokens_));
                condition_values_.push_back(make_cache(condition_tokens_));
            }
        }
        if (has_target_cache() || condition_owner_ == nullptr) {
            if (host_resident_) {
                auto * host_buffer_type = ggml_backend_dev_host_buffer_type(
                    ggml_backend_get_device(execution_.backend()));
                if (host_buffer_type == nullptr) {
                    throw std::runtime_error("LiveAvatar host KV cache requires accelerator host buffers");
                }
                buffer_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_.get(), host_buffer_type);
            } else {
                buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), execution_.backend());
            }
            if (buffer_ == nullptr) {
                throw std::runtime_error("LiveAvatar LiveAvatar KV cache backend buffer allocation failed");
            }
        }
    }

    engine::core::ExecutionContext & execution_;
    LiveAvatarConfig config_;
    int64_t max_target_tokens_ = 0;
    int64_t condition_tokens_ = 0;
    LiveAvatarKVCache * condition_owner_ = nullptr;
    bool host_resident_ = false;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_backend_buffer_t buffer_ = nullptr;
    engine::core::TensorValue target_key_storage_;
    engine::core::TensorValue target_value_storage_;
    std::vector<engine::core::TensorValue> target_keys_;
    std::vector<engine::core::TensorValue> target_values_;
    std::vector<engine::core::TensorValue> condition_keys_;
    std::vector<engine::core::TensorValue> condition_values_;
};

struct LiveAvatarHostTargetCache {
    int64_t valid_tokens = 0;
    std::vector<ggml_fp16_t> keys;
    std::vector<ggml_fp16_t> values;
};

void import_liveavatar_target_cache(
    const LiveAvatarHostTargetCache & state,
    const LiveAvatarKVCache & cache) {
    const size_t bytes = ggml_nbytes(cache.target_key_storage().tensor);
    if (state.valid_tokens <= 0) {
        return;
    }
    if (state.keys.size() * sizeof(ggml_fp16_t) != bytes ||
        state.values.size() * sizeof(ggml_fp16_t) != bytes) {
        throw std::runtime_error("LiveAvatar LiveAvatar host target cache size mismatch");
    }
    ggml_backend_tensor_set(cache.target_key_storage().tensor, state.keys.data(), 0, bytes);
    ggml_backend_tensor_set(cache.target_value_storage().tensor, state.values.data(), 0, bytes);
}

void export_liveavatar_target_cache(
    LiveAvatarHostTargetCache & state,
    const LiveAvatarKVCache & cache,
    int64_t valid_tokens) {
    if (valid_tokens <= 0) {
        throw std::runtime_error("LiveAvatar LiveAvatar target cache export length is invalid");
    }
    if (valid_tokens > cache.max_target_tokens()) {
        throw std::runtime_error("LiveAvatar LiveAvatar target cache export length exceeds cache capacity");
    }
    const size_t bytes = ggml_nbytes(cache.target_key_storage().tensor);
    const size_t elems = bytes / sizeof(ggml_fp16_t);
    state.valid_tokens = valid_tokens;
    state.keys.resize(elems);
    state.values.resize(elems);
    ggml_backend_tensor_get(cache.target_key_storage().tensor, state.keys.data(), 0, bytes);
    ggml_backend_tensor_get(cache.target_value_storage().tensor, state.values.data(), 0, bytes);
}

class LiveAvatarSinkGraph {
public:
    LiveAvatarSinkGraph(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserWeights & weights,
        LiveAvatarConfig config,
        engine::core::TensorShape latent_shape,
        LiveAvatarKVCache & cache,
        bool use_sage_attention,
        int64_t layer_batch)
        : execution_(execution),
          weights_(&weights),
          config_(std::move(config)),
          latent_shape_(std::move(latent_shape)),
          cache_(&cache),
          use_sage_attention_(use_sage_attention),
          layer_batch_(layer_batch) {
        build();
    }

    ~LiveAvatarSinkGraph() {
        if (execution_.backend() != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_.backend_type(), execution_.backend(), graph_, true);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (const_buffer_ != nullptr) {
            ggml_backend_buffer_free(const_buffer_);
        }
        if (state_buffer_ != nullptr) {
            ggml_backend_buffer_free(state_buffer_);
        }
    }

    LiveAvatarSinkGraph(const LiveAvatarSinkGraph &) = delete;
    LiveAvatarSinkGraph & operator=(const LiveAvatarSinkGraph &) = delete;

    void run(
        const std::vector<float> & ref_latents,
        const std::vector<float> & motion_latents,
        const std::vector<float> & projected_text_context) const {
        const auto ref_shape = engine::core::TensorShape::from_dims({
            config_.latent_channels,
            1,
            latent_shape_.dims[2],
            latent_shape_.dims[3],
        });
        const auto motion_shape = engine::core::TensorShape::from_dims({
            config_.latent_channels,
            config_.latent_motion_frames,
            latent_shape_.dims[2],
            latent_shape_.dims[3],
        });
        const auto context_shape = engine::core::TensorShape::from_dims({1, config_.text_len, config_.hidden_size});
        if (static_cast<int64_t>(ref_latents.size()) != ref_shape.num_elements() ||
            static_cast<int64_t>(motion_latents.size()) != motion_shape.num_elements() ||
            static_cast<int64_t>(projected_text_context.size()) != context_shape.num_elements()) {
            throw std::runtime_error("LiveAvatar LiveAvatar sink input payload size mismatch");
        }
        const auto write_start = Clock::now();
        engine::core::write_tensor_f32(ref_latents_, ref_latents);
        engine::core::write_tensor_f32(motion_latents_, motion_latents);
        engine::core::write_tensor_f32(context_, projected_text_context);
        engine::core::write_tensor_f32(timestep_, make_timestep_features(0.0F));
        engine::debug::timing_log_scalar("liveavatar.sink_input_write_ms", engine::debug::elapsed_ms(write_start));
        engine::core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_graph(execution_, graph_, plan_, "liveavatar.sink");
        engine::debug::timing_log_scalar("liveavatar.sink_compute_ms", engine::debug::elapsed_ms(compute_start));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LiveAvatar LiveAvatar sink graph compute failed");
        }
        if (weights_->block_weights_host_resident) {
            run_streaming_layers();
        }
    }

private:
    engine::core::TensorValue build_sink_self_attention(
        engine::core::ModuleBuildContext & ctx,
        const engine::core::TensorValue & input,
        const engine::core::TensorValue & rope_cos,
        const engine::core::TensorValue & rope_sin,
        const LiveAvatarDenoiserAttentionWeights & weights,
        int64_t layer) const {
        const int64_t hidden = config_.hidden_size;
        const int64_t heads = config_.num_heads;
        const int64_t head_dim = hidden / heads;
        auto q = linear_native(ctx, input, weights.q, hidden, hidden);
        auto k = linear_native(ctx, input, weights.k, hidden, hidden);
        auto v = linear_native(ctx, input, weights.v, hidden, hidden);
        q = engine::modules::RMSNormModule({hidden, 1.0e-6F, true, false})
                .build(ctx, q, {weights.norm_q, std::nullopt});
        k = engine::modules::RMSNormModule({hidden, 1.0e-6F, true, false})
                .build(ctx, k, {weights.norm_k, std::nullopt});
        q = engine::core::reshape_tensor(
            ctx,
            engine::core::ensure_backend_addressable_layout(ctx, q),
            engine::core::TensorShape::from_dims({1, input.shape.dims[1], heads, head_dim}));
        k = engine::core::reshape_tensor(
            ctx,
            engine::core::ensure_backend_addressable_layout(ctx, k),
            engine::core::TensorShape::from_dims({1, input.shape.dims[1], heads, head_dim}));
        v = engine::core::reshape_tensor(
            ctx,
            engine::core::ensure_backend_addressable_layout(ctx, v),
            engine::core::TensorShape::from_dims({1, input.shape.dims[1], heads, head_dim}));
        auto cached_k = write_liveavatar_kv_cache(ctx, k, cache_->condition_key(static_cast<size_t>(layer)), 0, "condition key");
        auto cached_v = write_liveavatar_kv_cache(ctx, v, cache_->condition_value(static_cast<size_t>(layer)), 0, "condition value");
        auto q_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, 4})
                           .build(ctx, apply_wan_rope(ctx, q, rope_cos, rope_sin, head_dim));
        auto k_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, 4})
                           .build(ctx, apply_wan_rope(ctx, cached_k, rope_cos, rope_sin, head_dim));
        auto v_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, cached_v);
        auto attended = scaled_attention(ctx, execution_.backend(), use_sage_attention_, q_heads, k_heads, v_heads, head_dim);
        return linear_native(ctx, attended, weights.o, hidden, hidden);
    }

    engine::core::TensorValue build_sink_block(
        engine::core::ModuleBuildContext & ctx,
        const engine::core::TensorValue & input,
        const engine::core::TensorValue & context,
        const engine::core::TensorValue & zero_e0,
        const LiveAvatarDenoiserBlockWeights & weights,
        int64_t layer) const {
        const int64_t hidden = config_.hidden_size;
        auto block_zero = engine::modules::AddModule().build(
            ctx,
            zero_e0,
            engine::modules::RepeatModule({zero_e0.shape}).build(ctx, weights.modulation));
        auto norm1 = engine::modules::LayerNormModule({hidden, 1.0e-6F, false, false}).build(ctx, input, {});
        auto self_shift = engine::modules::SliceModule({1, 0, 1}).build(ctx, block_zero);
        auto self_scale = engine::modules::SliceModule({1, 1, 1}).build(ctx, block_zero);
        auto self_input = apply_shift_scale(ctx, norm1, self_shift, self_scale);
        auto self_out = build_sink_self_attention(ctx, self_input, rope_cos_, rope_sin_, weights.self_attention, layer);
        auto self_gate = engine::modules::SliceModule({1, 2, 1}).build(ctx, block_zero);
        self_out = engine::modules::MulModule().build(
            ctx,
            self_out,
            engine::modules::RepeatModule({self_out.shape}).build(ctx, self_gate));
        auto x = engine::modules::AddModule().build(ctx, input, self_out);
        auto norm3 = engine::modules::LayerNormModule({hidden, 1.0e-6F, true, true}).build(ctx, x, weights.norm3);
        x = engine::modules::AddModule().build(ctx, x, build_cross_attention(ctx, execution_.backend(), use_sage_attention_, norm3, context, weights.cross_attention, config_));
        auto norm2 = engine::modules::LayerNormModule({hidden, 1.0e-6F, false, false}).build(ctx, x, {});
        auto ffn_shift = engine::modules::SliceModule({1, 3, 1}).build(ctx, block_zero);
        auto ffn_scale = engine::modules::SliceModule({1, 4, 1}).build(ctx, block_zero);
        auto ffn_input = apply_shift_scale(ctx, norm2, ffn_shift, ffn_scale);
        auto ffn = linear_native(ctx, ffn_input, weights.ffn_in, hidden, config_.ffn_dim);
        ffn = engine::modules::GeluModule({engine::modules::GeluApproximation::Tanh}).build(ctx, ffn);
        ffn = linear_native(ctx, ffn, weights.ffn_out, config_.ffn_dim, hidden);
        auto ffn_gate = engine::modules::SliceModule({1, 5, 1}).build(ctx, block_zero);
        ffn = engine::modules::MulModule().build(
            ctx,
            ffn,
            engine::modules::RepeatModule({ffn.shape}).build(ctx, ffn_gate));
        return engine::modules::AddModule().build(ctx, x, ffn);
    }

    void build() {
        if (latent_shape_.rank != 4 ||
            latent_shape_.dims[0] != config_.latent_channels ||
            latent_shape_.dims[1] != 3 ||
            latent_shape_.dims[2] <= 0 ||
            latent_shape_.dims[3] <= 0) {
            throw std::runtime_error("LiveAvatar LiveAvatar sink shape is invalid");
        }
        const int64_t hidden = config_.hidden_size;
        const int64_t head_dim = hidden / config_.num_heads;
        const int64_t cond_tokens =
            (latent_shape_.dims[2] / 2) * (latent_shape_.dims[3] / 2) +
            liveavatar_motion_token_count(latent_shape_.dims[2], latent_shape_.dims[3]);
        if (cache_->condition_tokens() != cond_tokens) {
            throw std::runtime_error("LiveAvatar LiveAvatar sink cache condition length mismatch");
        }
        if (weights_->block_weights_host_resident && layer_batch_ <= 0) {
            throw std::runtime_error("LiveAvatar denoiser weight streaming layer batch must be positive");
        }
        rope_cos_values_ = make_liveavatar_rope_table(
            config_.num_heads,
            head_dim,
            liveavatar_condition_rope_ranges(latent_shape_.dims[2], latent_shape_.dims[3]),
            false);
        rope_sin_values_ = make_liveavatar_rope_table(
            config_.num_heads,
            head_dim,
            liveavatar_condition_rope_ranges(latent_shape_.dims[2], latent_shape_.dims[3]),
            true);

        ggml_init_params const_params{ggml_tensor_overhead() * 4, nullptr, true};
        const_ctx_.reset(ggml_init(const_params));
        if (const_ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar LiveAvatar sink constant context initialization failed");
        }
        engine::core::ModuleBuildContext const_ctx{const_ctx_.get(), "liveavatar.sink.const", execution_.backend_type()};
        rope_cos_ = engine::core::make_tensor(
            const_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, cond_tokens, config_.num_heads, head_dim / 2}));
        rope_sin_ = engine::core::make_tensor(
            const_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, cond_tokens, config_.num_heads, head_dim / 2}));
        const_buffer_ = ggml_backend_alloc_ctx_tensors(const_ctx_.get(), execution_.backend());
        if (const_buffer_ == nullptr) {
            throw std::runtime_error("LiveAvatar LiveAvatar sink constant buffer allocation failed");
        }
        engine::core::write_tensor_f32(rope_cos_, rope_cos_values_);
        engine::core::write_tensor_f32(rope_sin_, rope_sin_values_);

        ggml_init_params params{kLiveAvatarDenoiserGraphContextBytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar LiveAvatar sink graph context initialization failed");
        }
        engine::core::ModuleBuildContext build_ctx{ctx_.get(), "liveavatar.sink", execution_.backend_type()};
        ref_latents_ = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({config_.latent_channels, 1, latent_shape_.dims[2], latent_shape_.dims[3]}));
        motion_latents_ = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({
                config_.latent_channels,
                config_.latent_motion_frames,
                latent_shape_.dims[2],
                latent_shape_.dims[3],
            }));
        context_ = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, config_.text_len, hidden}));
        timestep_ = engine::core::make_tensor(build_ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({2, 256}));
        ggml_set_input(ref_latents_.tensor);
        ggml_set_input(motion_latents_.tensor);
        ggml_set_input(context_.tensor);
        ggml_set_input(timestep_.tensor);

        auto time_emb = linear_native(build_ctx, timestep_, weights_->time_embedding_0, 256, hidden);
        time_emb = engine::modules::SiluModule{}.build(build_ctx, time_emb);
        time_emb = linear_native(build_ctx, time_emb, weights_->time_embedding_2, hidden, hidden);
        auto time_proj = engine::modules::SiluModule{}.build(build_ctx, time_emb);
        time_proj = linear_native(build_ctx, time_proj, weights_->time_projection, hidden, 6 * hidden);
        time_proj = engine::core::reshape_tensor(build_ctx, time_proj, engine::core::TensorShape::from_dims({2, 6, hidden}));
        const auto zero_e0 = engine::modules::SliceModule({0, 1, 1}).build(build_ctx, time_proj);

        auto ref = conv3d_tokens(build_ctx, ref_latents_, weights_->patch_embedding, config_.latent_channels, hidden, 1, 2, 2, 1, 2, 2);
        auto motion = build_liveavatar_motion_tokens(build_ctx, motion_latents_, weights_->frame_packer, config_);
        auto x = engine::modules::ConcatModule({1, true}).build(build_ctx, ref, motion);
        x = engine::modules::AddModule().build(build_ctx, x, build_liveavatar_condition_mask(build_ctx, weights_->trainable_cond_mask, ref, motion));
        if (weights_->block_weights_host_resident) {
            build_streaming_state(cond_tokens, hidden);
            ggml_build_forward_expand(
                graph_ = ggml_new_graph_custom(ctx_.get(), 1048576, false),
                ggml_cpy(build_ctx.ggml, engine::core::ensure_backend_addressable_layout(build_ctx, x).tensor, sink_hidden_[0].tensor));
            ggml_build_forward_expand(graph_, ggml_cpy(build_ctx.ggml, context_.tensor, sink_context_.tensor));
            ggml_build_forward_expand(
                graph_,
                ggml_cpy(build_ctx.ggml, engine::core::ensure_backend_addressable_layout(build_ctx, zero_e0).tensor, sink_zero_e0_.tensor));
        } else {
            for (int64_t layer = 0; layer < config_.num_layers; ++layer) {
                x = build_sink_block(
                    build_ctx,
                    x,
                    context_,
                    zero_e0,
                    weights_->blocks.at(static_cast<size_t>(layer)),
                    layer);
            }
            output_ = engine::core::ensure_backend_addressable_layout(build_ctx, x).tensor;
            ggml_set_output(output_);
            graph_ = ggml_new_graph_custom(ctx_.get(), 1048576, false);
            ggml_build_forward_expand(graph_, output_);
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("LiveAvatar LiveAvatar sink backend buffer allocation failed");
        }
        engine::core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    void build_streaming_state(int64_t condition_tokens, int64_t hidden) {
        ggml_init_params params{ggml_tensor_overhead() * 4, nullptr, true};
        state_ctx_.reset(ggml_init(params));
        if (state_ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar sink streaming state context initialization failed");
        }
        engine::core::ModuleBuildContext ctx{state_ctx_.get(), "liveavatar.sink.streaming_state", execution_.backend_type()};
        const auto hidden_shape = engine::core::TensorShape::from_dims({1, condition_tokens, hidden});
        sink_hidden_[0] = engine::core::make_tensor(ctx, GGML_TYPE_F32, hidden_shape);
        sink_hidden_[1] = engine::core::make_tensor(ctx, GGML_TYPE_F32, hidden_shape);
        sink_context_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, config_.text_len, hidden}));
        sink_zero_e0_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, 6, hidden}));
        state_buffer_ = ggml_backend_alloc_ctx_tensors(state_ctx_.get(), execution_.backend());
        if (state_buffer_ == nullptr) {
            throw std::runtime_error("LiveAvatar sink streaming state allocation failed");
        }
    }

    void run_streaming_layers() const {
        int64_t src = 0;
        int64_t dst = 1;
        for (int64_t layer_start = 0; layer_start < config_.num_layers; layer_start += layer_batch_) {
            const int64_t layer_end = std::min<int64_t>(layer_start + layer_batch_, config_.num_layers);
            ggml_init_params params{kLiveAvatarDenoiserGraphContextBytes, nullptr, true};
            std::unique_ptr<ggml_context, GgmlContextDeleter> ctx(ggml_init(params));
            if (ctx == nullptr) {
                throw std::runtime_error("LiveAvatar sink streaming graph context initialization failed");
            }
            engine::core::ModuleBuildContext build_ctx{ctx.get(), "liveavatar.sink.streaming", execution_.backend_type()};
            auto x = sink_hidden_[static_cast<size_t>(src)];
            for (int64_t layer = layer_start; layer < layer_end; ++layer) {
                x = build_sink_block(
                    build_ctx,
                    x,
                    sink_context_,
                    sink_zero_e0_,
                    weights_->blocks.at(static_cast<size_t>(layer)),
                    layer);
            }
            auto * copy = ggml_cpy(
                build_ctx.ggml,
                engine::core::ensure_backend_addressable_layout(build_ctx, x).tensor,
                sink_hidden_[static_cast<size_t>(dst)].tensor);
            auto * graph = ggml_new_graph_custom(ctx.get(), 1048576, false);
            ggml_build_forward_expand(graph, copy);
            LiveAvatarWeightStreamingGraph streaming_graph(execution_, graph);
            if (streaming_graph.compute(graph) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("LiveAvatar sink streaming layer graph compute failed");
            }
            std::swap(src, dst);
        }
    }

    engine::core::ExecutionContext & execution_;
    const LiveAvatarDenoiserWeights * weights_ = nullptr;
    LiveAvatarConfig config_;
    engine::core::TensorShape latent_shape_;
    LiveAvatarKVCache * cache_ = nullptr;
    bool use_sage_attention_ = false;
    int64_t layer_batch_ = 1;
    std::unique_ptr<ggml_context, GgmlContextDeleter> const_ctx_;
    ggml_backend_buffer_t const_buffer_ = nullptr;
    std::unique_ptr<ggml_context, GgmlContextDeleter> state_ctx_;
    ggml_backend_buffer_t state_buffer_ = nullptr;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    engine::core::TensorValue ref_latents_;
    engine::core::TensorValue motion_latents_;
    engine::core::TensorValue context_;
    engine::core::TensorValue timestep_;
    engine::core::TensorValue rope_cos_;
    engine::core::TensorValue rope_sin_;
    std::array<engine::core::TensorValue, 2> sink_hidden_;
    engine::core::TensorValue sink_context_;
    engine::core::TensorValue sink_zero_e0_;
    std::vector<float> rope_cos_values_;
    std::vector<float> rope_sin_values_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    mutable engine::core::HostGraphPlan plan_;
};

class LiveAvatarBlockConditionCache {
public:
    LiveAvatarBlockConditionCache(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserWeights & weights,
        LiveAvatarConfig config,
        int64_t target_frames,
        bool use_sage_attention,
        const LiveAvatarBlockConditionCache * shared_text_cache = nullptr)
        : execution_(execution),
          weights_(&weights),
          config_(std::move(config)),
          target_frames_(target_frames),
          use_sage_attention_(use_sage_attention),
          shared_text_cache_(shared_text_cache) {
        const auto build_start = Clock::now();
        build();
        engine::debug::timing_log_scalar("liveavatar.block_condition_cache_build_ms", engine::debug::elapsed_ms(build_start));
    }

    ~LiveAvatarBlockConditionCache() {
        release_populate_graph();
        if (cache_buffer_ != nullptr) {
            ggml_backend_buffer_free(cache_buffer_);
        }
    }

    LiveAvatarBlockConditionCache(const LiveAvatarBlockConditionCache &) = delete;
    LiveAvatarBlockConditionCache & operator=(const LiveAvatarBlockConditionCache &) = delete;

    void populate(const LiveAvatarDenoiserPreparedCondition & condition) {
        const size_t text_values = static_cast<size_t>(config_.text_len * config_.hidden_size);
        const size_t audio_local_values =
            static_cast<size_t>(target_frames_ * (config_.audio_tokens + 1) * config_.hidden_size);
        const size_t audio_global_values = static_cast<size_t>(target_frames_ * config_.hidden_size);
        if (condition.projected_text_context.size() != text_values ||
            condition.encoded_audio_local.size() != audio_local_values ||
            condition.encoded_audio_global.size() != audio_global_values) {
            throw std::runtime_error("LiveAvatar LiveAvatar block condition cache input payload size mismatch");
        }
        const auto write_start = Clock::now();
        if (shared_text_cache_ == nullptr) {
            engine::core::write_tensor_f32(context_, condition.projected_text_context);
        }
        engine::core::write_tensor_f32(audio_local_, condition.encoded_audio_local);
        engine::core::write_tensor_f32(audio_global_, condition.encoded_audio_global);
        engine::debug::timing_log_scalar("liveavatar.block_condition_cache_input_write_ms", engine::debug::elapsed_ms(write_start));
        engine::core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const auto compute_start = Clock::now();
        const ggml_status status = streaming_graph_ != nullptr
            ? streaming_graph_->compute(graph_)
            : engine::core::compute_graph(execution_, graph_, plan_, "liveavatar.block_condition_cache");
        engine::debug::timing_log_scalar("liveavatar.block_condition_cache_compute_ms", engine::debug::elapsed_ms(compute_start));
        ggml_backend_synchronize(execution_.backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LiveAvatar LiveAvatar block condition cache graph compute failed");
        }
        release_populate_graph();
    }

    LiveAvatarAttentionKV text_kv(int64_t layer) const {
        if (shared_text_cache_ != nullptr) {
            return shared_text_cache_->text_kv(layer);
        }
        const auto index = static_cast<size_t>(layer);
        return {text_keys_.at(index), text_values_.at(index)};
    }

    LiveAvatarAttentionKV audio_kv(int64_t injector) const {
        const auto index = static_cast<size_t>(injector);
        return {audio_keys_.at(index), audio_values_.at(index)};
    }

    const engine::core::TensorValue & audio_shift(int64_t injector) const {
        return audio_shifts_.at(static_cast<size_t>(injector));
    }

    const engine::core::TensorValue & audio_scale(int64_t injector) const {
        return audio_scales_.at(static_cast<size_t>(injector));
    }

private:
    static engine::core::TensorValue contiguous(
        engine::core::ModuleBuildContext & ctx,
        const engine::core::TensorValue & value) {
        return engine::core::ensure_backend_addressable_layout(ctx, value);
    }

    void release_populate_graph() {
        streaming_graph_.reset();
        if (execution_.backend() != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_.backend_type(), execution_.backend(), graph_, true);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        graph_ctx_.reset();
        graph_ = nullptr;
    }

    void build() {
        if (target_frames_ <= 0) {
            throw std::runtime_error("LiveAvatar LiveAvatar block condition cache shape is invalid");
        }
        const int64_t hidden = config_.hidden_size;
        const int64_t heads = config_.num_heads;
        const int64_t head_dim = hidden / heads;
        const size_t audio_injectors = weights_->audio_injectors.size();
        const ggml_type kv_cache_type =
            use_sage_attention_ &&
                    execution_.backend_type() == engine::core::BackendType::Cuda &&
                    (head_dim == 64 || head_dim == 128)
                ? GGML_TYPE_F16
                : GGML_TYPE_F32;

        ggml_init_params cache_params{
            ggml_tensor_overhead() * (weights_->blocks.size() * 2 + audio_injectors * 4 + 16),
            nullptr,
            true};
        cache_ctx_.reset(ggml_init(cache_params));
        if (cache_ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar LiveAvatar block condition cache context initialization failed");
        }
        engine::core::ModuleBuildContext cache_ctx{cache_ctx_.get(), "liveavatar.block_condition_cache.data", execution_.backend_type()};
        auto make_text_cache = [&]() {
            return engine::core::make_tensor(
                cache_ctx,
                kv_cache_type,
                engine::core::TensorShape::from_dims({1, heads, config_.text_len, head_dim}));
        };
        if (shared_text_cache_ == nullptr) {
            text_keys_.reserve(weights_->blocks.size());
            text_values_.reserve(weights_->blocks.size());
            for (size_t layer = 0; layer < weights_->blocks.size(); ++layer) {
                text_keys_.push_back(make_text_cache());
                text_values_.push_back(make_text_cache());
            }
        }
        auto make_audio_cache = [&]() {
            return engine::core::make_tensor(
                cache_ctx,
                kv_cache_type,
                engine::core::TensorShape::from_dims({target_frames_, heads, config_.audio_tokens + 1, head_dim}));
        };
        auto make_audio_modulation = [&]() {
            return engine::core::make_tensor(
                cache_ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({target_frames_, 1, hidden}));
        };
        audio_keys_.reserve(audio_injectors);
        audio_values_.reserve(audio_injectors);
        audio_shifts_.reserve(audio_injectors);
        audio_scales_.reserve(audio_injectors);
        for (size_t injector = 0; injector < audio_injectors; ++injector) {
            audio_keys_.push_back(make_audio_cache());
            audio_values_.push_back(make_audio_cache());
            audio_shifts_.push_back(make_audio_modulation());
            audio_scales_.push_back(make_audio_modulation());
        }
        cache_buffer_ = ggml_backend_alloc_ctx_tensors(cache_ctx_.get(), execution_.backend());
        if (cache_buffer_ == nullptr) {
            throw std::runtime_error("LiveAvatar LiveAvatar block condition cache backend buffer allocation failed");
        }

        ggml_init_params graph_params{kLiveAvatarDenoiserGraphContextBytes, nullptr, true};
        graph_ctx_.reset(ggml_init(graph_params));
        if (graph_ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar LiveAvatar block condition cache graph context initialization failed");
        }
        engine::core::ModuleBuildContext graph_ctx{graph_ctx_.get(), "liveavatar.block_condition_cache", execution_.backend_type()};
        context_ = engine::core::make_tensor(
            graph_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, config_.text_len, hidden}));
        audio_local_ = engine::core::make_tensor(
            graph_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, target_frames_, config_.audio_tokens + 1, hidden}));
        audio_global_ = engine::core::make_tensor(
            graph_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, target_frames_, 1, hidden}));
        ggml_set_input(context_.tensor);
        ggml_set_input(audio_local_.tensor);
        ggml_set_input(audio_global_.tensor);

        graph_ = ggml_new_graph_custom(graph_ctx_.get(), 1048576, false);
        if (shared_text_cache_ == nullptr) {
            for (size_t layer = 0; layer < weights_->blocks.size(); ++layer) {
                const auto kv = build_cross_attention_kv(
                    graph_ctx,
                    context_,
                    weights_->blocks[layer].cross_attention,
                    config_);
                ggml_build_forward_expand(graph_, ggml_cpy(graph_ctx.ggml, contiguous(graph_ctx, kv.key).tensor, text_keys_[layer].tensor));
                ggml_build_forward_expand(graph_, ggml_cpy(graph_ctx.ggml, contiguous(graph_ctx, kv.value).tensor, text_values_[layer].tensor));
            }
        }

        auto audio_local = engine::core::reshape_tensor(
            graph_ctx,
            audio_local_,
            engine::core::TensorShape::from_dims({target_frames_, config_.audio_tokens + 1, hidden}));
        auto temb = engine::core::reshape_tensor(
            graph_ctx,
            engine::modules::SliceModule({2, 0, 1}).build(graph_ctx, audio_global_),
            engine::core::TensorShape::from_dims({target_frames_, hidden}));
        for (size_t injector = 0; injector < audio_injectors; ++injector) {
            const auto & weights = weights_->audio_injectors[injector];
            auto scale_shift = linear_native(
                graph_ctx,
                engine::modules::SiluModule{}.build(graph_ctx, temb),
                weights.adain_linear,
                hidden,
                2 * hidden);
            auto shift = engine::core::reshape_tensor(
                graph_ctx,
                contiguous(graph_ctx, engine::modules::SliceModule({1, 0, hidden}).build(graph_ctx, scale_shift)),
                engine::core::TensorShape::from_dims({target_frames_, 1, hidden}));
            auto scale = engine::core::reshape_tensor(
                graph_ctx,
                contiguous(graph_ctx, engine::modules::SliceModule({1, hidden, hidden}).build(graph_ctx, scale_shift)),
                engine::core::TensorShape::from_dims({target_frames_, 1, hidden}));
            const auto kv = build_cross_attention_kv(graph_ctx, audio_local, weights.attention, config_);
            ggml_build_forward_expand(graph_, ggml_cpy(graph_ctx.ggml, contiguous(graph_ctx, kv.key).tensor, audio_keys_[injector].tensor));
            ggml_build_forward_expand(graph_, ggml_cpy(graph_ctx.ggml, contiguous(graph_ctx, kv.value).tensor, audio_values_[injector].tensor));
            ggml_build_forward_expand(graph_, ggml_cpy(graph_ctx.ggml, contiguous(graph_ctx, shift).tensor, audio_shifts_[injector].tensor));
            ggml_build_forward_expand(graph_, ggml_cpy(graph_ctx.ggml, contiguous(graph_ctx, scale).tensor, audio_scales_[injector].tensor));
        }
        if (weights_->block_weights_host_resident) {
            streaming_graph_ = std::make_unique<LiveAvatarWeightStreamingGraph>(execution_, graph_);
        } else {
            gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
            if (gallocr_ == nullptr ||
                !ggml_gallocr_reserve(gallocr_, graph_) ||
                !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
                throw std::runtime_error("LiveAvatar LiveAvatar block condition cache graph allocation failed");
            }
            engine::core::prepare_host_graph_plan(execution_, graph_, plan_);
        }
    }

    engine::core::ExecutionContext & execution_;
    const LiveAvatarDenoiserWeights * weights_ = nullptr;
    LiveAvatarConfig config_;
    int64_t target_frames_ = 0;
    bool use_sage_attention_ = false;
    const LiveAvatarBlockConditionCache * shared_text_cache_ = nullptr;
    std::unique_ptr<ggml_context, GgmlContextDeleter> cache_ctx_;
    ggml_backend_buffer_t cache_buffer_ = nullptr;
    std::unique_ptr<ggml_context, GgmlContextDeleter> graph_ctx_;
    engine::core::TensorValue context_;
    engine::core::TensorValue audio_local_;
    engine::core::TensorValue audio_global_;
    std::vector<engine::core::TensorValue> text_keys_;
    std::vector<engine::core::TensorValue> text_values_;
    std::vector<engine::core::TensorValue> audio_keys_;
    std::vector<engine::core::TensorValue> audio_values_;
    std::vector<engine::core::TensorValue> audio_shifts_;
    std::vector<engine::core::TensorValue> audio_scales_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    std::unique_ptr<LiveAvatarWeightStreamingGraph> streaming_graph_;
    mutable engine::core::HostGraphPlan plan_;
};

class LiveAvatarBlockTargetCache {
public:
    LiveAvatarBlockTargetCache(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserWeights & weights,
        LiveAvatarConfig config,
        engine::core::TensorShape latent_shape)
        : execution_(execution),
          weights_(&weights),
          config_(std::move(config)),
          latent_shape_(std::move(latent_shape)) {
        const auto build_start = Clock::now();
        build();
        engine::debug::timing_log_scalar("liveavatar.block_target_cache_build_ms", engine::debug::elapsed_ms(build_start));
    }

    ~LiveAvatarBlockTargetCache() {
        release_populate_graph();
        if (cache_buffer_ != nullptr) {
            ggml_backend_buffer_free(cache_buffer_);
        }
    }

    LiveAvatarBlockTargetCache(const LiveAvatarBlockTargetCache &) = delete;
    LiveAvatarBlockTargetCache & operator=(const LiveAvatarBlockTargetCache &) = delete;

    void populate(const std::vector<float> & cond_latents) {
        if (static_cast<int64_t>(cond_latents.size()) != latent_shape_.num_elements()) {
            throw std::runtime_error("LiveAvatar LiveAvatar block target cache input payload size mismatch");
        }
        const auto write_start = Clock::now();
        engine::core::write_tensor_f32(cond_latents_, cond_latents);
        engine::debug::timing_log_scalar("liveavatar.block_target_cache_input_write_ms", engine::debug::elapsed_ms(write_start));
        engine::core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const auto compute_start = Clock::now();
        const ggml_status status =
            engine::core::compute_graph(execution_, graph_, plan_, "liveavatar.block_target_cache");
        engine::debug::timing_log_scalar("liveavatar.block_target_cache_compute_ms", engine::debug::elapsed_ms(compute_start));
        ggml_backend_synchronize(execution_.backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LiveAvatar LiveAvatar block target cache graph compute failed");
        }
        release_populate_graph();
    }

    const engine::core::TensorValue & target_condition() const noexcept { return target_condition_; }

private:
    void release_populate_graph() {
        if (execution_.backend() != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_.backend_type(), execution_.backend(), graph_, true);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        graph_ctx_.reset();
        graph_ = nullptr;
    }

    void build() {
        if (latent_shape_.rank != 4 ||
            latent_shape_.dims[0] != config_.latent_channels ||
            latent_shape_.dims[1] != 3 ||
            latent_shape_.dims[2] <= 0 ||
            latent_shape_.dims[3] <= 0) {
            throw std::runtime_error("LiveAvatar LiveAvatar block target cache shape is invalid");
        }
        const int64_t hidden = config_.hidden_size;
        const int64_t target_tokens = latent_shape_.dims[1] * (latent_shape_.dims[2] / 2) * (latent_shape_.dims[3] / 2);

        ggml_init_params cache_params{ggml_tensor_overhead() * 4, nullptr, true};
        cache_ctx_.reset(ggml_init(cache_params));
        if (cache_ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar LiveAvatar block target cache context initialization failed");
        }
        engine::core::ModuleBuildContext cache_ctx{cache_ctx_.get(), "liveavatar.block_target_cache.data", execution_.backend_type()};
        target_condition_ = engine::core::make_tensor(
            cache_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, target_tokens, hidden}));
        cache_buffer_ = ggml_backend_alloc_ctx_tensors(cache_ctx_.get(), execution_.backend());
        if (cache_buffer_ == nullptr) {
            throw std::runtime_error("LiveAvatar LiveAvatar block target cache backend buffer allocation failed");
        }

        ggml_init_params graph_params{kLiveAvatarDenoiserGraphContextBytes, nullptr, true};
        graph_ctx_.reset(ggml_init(graph_params));
        if (graph_ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar LiveAvatar block target cache graph context initialization failed");
        }
        engine::core::ModuleBuildContext graph_ctx{graph_ctx_.get(), "liveavatar.block_target_cache", execution_.backend_type()};
        cond_latents_ = engine::core::make_tensor(graph_ctx, GGML_TYPE_F32, latent_shape_);
        ggml_set_input(cond_latents_.tensor);

        auto cond = conv3d_tokens(graph_ctx, cond_latents_, weights_->cond_encoder, config_.latent_channels, hidden, 1, 2, 2, 1, 2, 2);
        auto copied = ggml_cpy(
            graph_ctx.ggml,
            engine::core::ensure_backend_addressable_layout(graph_ctx, cond).tensor,
            target_condition_.tensor);
        graph_ = ggml_new_graph_custom(graph_ctx_.get(), 1048576, false);
        ggml_build_forward_expand(graph_, copied);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("LiveAvatar LiveAvatar block target cache graph allocation failed");
        }
        engine::core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    engine::core::ExecutionContext & execution_;
    const LiveAvatarDenoiserWeights * weights_ = nullptr;
    LiveAvatarConfig config_;
    engine::core::TensorShape latent_shape_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> cache_ctx_;
    ggml_backend_buffer_t cache_buffer_ = nullptr;
    engine::core::TensorValue target_condition_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> graph_ctx_;
    engine::core::TensorValue cond_latents_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    mutable engine::core::HostGraphPlan plan_;
};

class LiveAvatarConditionRopeVariant {
public:
    LiveAvatarConditionRopeVariant(
        engine::core::ExecutionContext & execution,
        const LiveAvatarConfig & config,
        const engine::core::TensorShape & latent_shape,
        int64_t condition_rollout_frames)
        : execution_(execution) {
        const int64_t heads = config.num_heads;
        const int64_t head_dim = config.hidden_size / heads;
        const int64_t cond_tokens =
            (latent_shape.dims[2] / 2) * (latent_shape.dims[3] / 2) +
            liveavatar_motion_token_count(latent_shape.dims[2], latent_shape.dims[3]);
        ggml_init_params params{ggml_tensor_overhead() * 4, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar LiveAvatar condition RoPE cache context initialization failed");
        }
        engine::core::ModuleBuildContext ctx{ctx_.get(), "liveavatar.block.condition_rope_cache", execution_.backend_type()};
        cos_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, cond_tokens, heads, head_dim / 2}));
        sin_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, cond_tokens, heads, head_dim / 2}));
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), execution_.backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("LiveAvatar LiveAvatar condition RoPE cache backend buffer allocation failed");
        }
        engine::core::write_tensor_f32(
            cos_,
            make_liveavatar_rope_table(heads, head_dim, liveavatar_condition_rope_ranges(latent_shape.dims[2], latent_shape.dims[3], condition_rollout_frames), false));
        engine::core::write_tensor_f32(
            sin_,
            make_liveavatar_rope_table(heads, head_dim, liveavatar_condition_rope_ranges(latent_shape.dims[2], latent_shape.dims[3], condition_rollout_frames), true));
    }

    ~LiveAvatarConditionRopeVariant() {
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    LiveAvatarConditionRopeVariant(const LiveAvatarConditionRopeVariant &) = delete;
    LiveAvatarConditionRopeVariant & operator=(const LiveAvatarConditionRopeVariant &) = delete;

    const engine::core::TensorValue & cos() const noexcept { return cos_; }
    const engine::core::TensorValue & sin() const noexcept { return sin_; }

private:
    engine::core::ExecutionContext & execution_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_backend_buffer_t buffer_ = nullptr;
    engine::core::TensorValue cos_;
    engine::core::TensorValue sin_;
};

class LiveAvatarBlockConstants {
public:
    LiveAvatarBlockConstants(
        engine::core::ExecutionContext & execution,
        const LiveAvatarConfig & config,
        const engine::core::TensorShape & latent_shape,
        int64_t target_start_frame,
        int64_t condition_rollout_frames)
        : execution_(execution),
          config_(config),
          latent_shape_(latent_shape),
          condition_rollout_frames_(condition_rollout_frames) {
        if (latent_shape_.rank != 4 ||
            latent_shape_.dims[0] != config_.latent_channels ||
            latent_shape_.dims[1] != 3 ||
            latent_shape_.dims[2] <= 0 ||
            latent_shape_.dims[3] <= 0) {
            throw std::runtime_error("LiveAvatar LiveAvatar block constant shape is invalid");
        }
        const int64_t heads = config_.num_heads;
        const int64_t head_dim = config_.hidden_size / heads;
        const int64_t target_tokens = latent_shape_.dims[1] * (latent_shape_.dims[2] / 2) * (latent_shape_.dims[3] / 2);
        const int64_t cond_tokens =
            (latent_shape_.dims[2] / 2) * (latent_shape_.dims[3] / 2) +
            liveavatar_motion_token_count(latent_shape_.dims[2], latent_shape_.dims[3]);
        const auto target_ranges =
            liveavatar_target_rope_ranges(latent_shape_.dims[1], latent_shape_.dims[2], latent_shape_.dims[3], target_start_frame);
        target_rope_cos_values_ = make_liveavatar_rope_table(heads, head_dim, target_ranges, false);
        target_rope_sin_values_ = make_liveavatar_rope_table(heads, head_dim, target_ranges, true);
        condition_rope_cos_values_ = make_liveavatar_rope_table(
            heads,
            head_dim,
            liveavatar_condition_rope_ranges(latent_shape_.dims[2], latent_shape_.dims[3], condition_rollout_frames_),
            false);
        condition_rope_sin_values_ = make_liveavatar_rope_table(
            heads,
            head_dim,
            liveavatar_condition_rope_ranges(latent_shape_.dims[2], latent_shape_.dims[3], condition_rollout_frames_),
            true);

        ggml_init_params params{ggml_tensor_overhead() * 8, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar LiveAvatar block constant context initialization failed");
        }
        engine::core::ModuleBuildContext ctx{ctx_.get(), "liveavatar.block.const", execution_.backend_type()};
        target_rope_cos_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, target_tokens, heads, head_dim / 2}));
        target_rope_sin_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, target_tokens, heads, head_dim / 2}));
        condition_rope_cos_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, cond_tokens, heads, head_dim / 2}));
        condition_rope_sin_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, cond_tokens, heads, head_dim / 2}));
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), execution_.backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("LiveAvatar LiveAvatar block constant buffer allocation failed");
        }
        engine::core::write_tensor_f32(target_rope_cos_, target_rope_cos_values_);
        engine::core::write_tensor_f32(target_rope_sin_, target_rope_sin_values_);
        engine::core::write_tensor_f32(condition_rope_cos_, condition_rope_cos_values_);
        engine::core::write_tensor_f32(condition_rope_sin_, condition_rope_sin_values_);
    }

    ~LiveAvatarBlockConstants() {
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    LiveAvatarBlockConstants(const LiveAvatarBlockConstants &) = delete;
    LiveAvatarBlockConstants & operator=(const LiveAvatarBlockConstants &) = delete;

    void update_condition_rope(int64_t condition_rollout_frames) {
        if (condition_rollout_frames_ == condition_rollout_frames) {
            return;
        }
        auto found = condition_rope_variants_.find(condition_rollout_frames);
        if (found == condition_rope_variants_.end()) {
            found = condition_rope_variants_
                        .emplace(
                            condition_rollout_frames,
                            std::make_unique<LiveAvatarConditionRopeVariant>(
                                execution_,
                                config_,
                                latent_shape_,
                                condition_rollout_frames))
                        .first;
        }
        ggml_backend_tensor_copy(found->second->cos().tensor, condition_rope_cos_.tensor);
        ggml_backend_tensor_copy(found->second->sin().tensor, condition_rope_sin_.tensor);
        condition_rollout_frames_ = condition_rollout_frames;
    }

    const engine::core::TensorValue & target_rope_cos() const noexcept { return target_rope_cos_; }
    const engine::core::TensorValue & target_rope_sin() const noexcept { return target_rope_sin_; }
    const engine::core::TensorValue & condition_rope_cos() const noexcept { return condition_rope_cos_; }
    const engine::core::TensorValue & condition_rope_sin() const noexcept { return condition_rope_sin_; }

private:
    engine::core::ExecutionContext & execution_;
    LiveAvatarConfig config_;
    engine::core::TensorShape latent_shape_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_backend_buffer_t buffer_ = nullptr;
    engine::core::TensorValue target_rope_cos_;
    engine::core::TensorValue target_rope_sin_;
    engine::core::TensorValue condition_rope_cos_;
    engine::core::TensorValue condition_rope_sin_;
    std::vector<float> target_rope_cos_values_;
    std::vector<float> target_rope_sin_values_;
    std::vector<float> condition_rope_cos_values_;
    std::vector<float> condition_rope_sin_values_;
    int64_t condition_rollout_frames_ = 0;
    std::unordered_map<int64_t, std::unique_ptr<LiveAvatarConditionRopeVariant>> condition_rope_variants_;
};

class LiveAvatarBlockGraph {
public:
    LiveAvatarBlockGraph(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserWeights & weights,
        LiveAvatarConfig config,
        engine::core::TensorShape latent_shape,
        LiveAvatarKVCache & cache,
        const LiveAvatarBlockConditionCache & condition_cache,
        const LiveAvatarBlockTargetCache & target_cache,
        int64_t target_start_frame,
        int64_t active_target_start_token,
        int64_t target_cache_write_token,
        int64_t active_target_tokens,
        bool use_bounded_target_cache,
        bool use_sage_attention,
        bool memory_saver,
        ggml_gallocr_t shared_gallocr = nullptr)
        : execution_(execution),
          weights_(&weights),
          config_(std::move(config)),
          latent_shape_(std::move(latent_shape)),
          cache_(&cache),
          condition_cache_(&condition_cache),
          target_cache_(&target_cache),
          target_start_frame_(target_start_frame),
          active_target_start_token_(active_target_start_token),
          target_cache_write_token_(target_cache_write_token),
          active_target_tokens_(active_target_tokens),
          use_bounded_target_cache_(use_bounded_target_cache),
          use_sage_attention_(use_sage_attention),
          memory_saver_(memory_saver),
          gallocr_(shared_gallocr),
          own_gallocr_(shared_gallocr == nullptr) {
        build();
    }

    ~LiveAvatarBlockGraph() {
        if (execution_.backend() != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_.backend_type(), execution_.backend(), graph_, true);
        }
        if (own_gallocr_ && gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (const_buffer_ != nullptr) {
            ggml_backend_buffer_free(const_buffer_);
        }
    }

    LiveAvatarBlockGraph(const LiveAvatarBlockGraph &) = delete;
    LiveAvatarBlockGraph & operator=(const LiveAvatarBlockGraph &) = delete;

    std::vector<float> run(
        const LiveAvatarDenoiserRunInput & input,
        int64_t condition_rollout_frames) {
        const auto context_shape = engine::core::TensorShape::from_dims({1, config_.text_len, config_.hidden_size});
        const auto audio_local_shape =
            engine::core::TensorShape::from_dims({1, latent_shape_.dims[1], config_.audio_tokens + 1, config_.hidden_size});
        const auto audio_global_shape =
            engine::core::TensorShape::from_dims({1, latent_shape_.dims[1], 1, config_.hidden_size});
        if (input.latent == nullptr ||
            input.projected_text_context == nullptr ||
            input.encoded_audio_local == nullptr ||
            input.encoded_audio_global == nullptr ||
            static_cast<int64_t>(input.latent->size()) != latent_shape_.num_elements() ||
            static_cast<int64_t>(input.projected_text_context->size()) != context_shape.num_elements() ||
            static_cast<int64_t>(input.encoded_audio_local->size()) != audio_local_shape.num_elements() ||
            static_cast<int64_t>(input.encoded_audio_global->size()) != audio_global_shape.num_elements()) {
            throw std::runtime_error("LiveAvatar LiveAvatar block input payload size mismatch");
        }
        if (!own_gallocr_ && !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("LiveAvatar LiveAvatar block backend buffer allocation failed");
        }
        const auto write_start = Clock::now();
        engine::core::write_tensor_f32(latent_, *input.latent);
        engine::core::write_tensor_f32(timestep_, make_timestep_features(input.timestep));
        update_condition_rope(condition_rollout_frames);
        engine::debug::timing_log_scalar("liveavatar.block_input_write_ms", engine::debug::elapsed_ms(write_start));
        engine::core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_graph(execution_, graph_, plan_, "liveavatar.block");
        engine::debug::timing_log_scalar("liveavatar.block_compute_ms", engine::debug::elapsed_ms(compute_start));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LiveAvatar LiveAvatar block graph compute failed");
        }
        const auto read_start = Clock::now();
        auto out = unpatchify_head_tokens(
            engine::core::read_tensor_f32(output_),
            latent_shape_.dims[1],
            latent_shape_.dims[2],
            latent_shape_.dims[3]);
        engine::debug::timing_log_scalar("liveavatar.block_output_read_ms", engine::debug::elapsed_ms(read_start));
        return out;
    }

private:
    void update_condition_rope(int64_t condition_rollout_frames) {
        if (condition_rollout_frames_ == condition_rollout_frames) {
            return;
        }
        if (memory_saver_) {
            const int64_t heads = config_.num_heads;
            const int64_t head_dim = config_.hidden_size / heads;
            const auto ranges = liveavatar_condition_rope_ranges(
                latent_shape_.dims[2],
                latent_shape_.dims[3],
                condition_rollout_frames);
            engine::core::write_tensor_f32(
                condition_rope_cos_,
                make_liveavatar_rope_table(heads, head_dim, ranges, false));
            engine::core::write_tensor_f32(
                condition_rope_sin_,
                make_liveavatar_rope_table(heads, head_dim, ranges, true));
            condition_rollout_frames_ = condition_rollout_frames;
            return;
        }
        auto found = condition_rope_variants_.find(condition_rollout_frames);
        if (found == condition_rope_variants_.end()) {
            found = condition_rope_variants_
                        .emplace(
                            condition_rollout_frames,
                            std::make_unique<LiveAvatarConditionRopeVariant>(
                                execution_,
                                config_,
                                latent_shape_,
                                condition_rollout_frames))
                        .first;
        }
        ggml_backend_tensor_copy(found->second->cos().tensor, condition_rope_cos_.tensor);
        ggml_backend_tensor_copy(found->second->sin().tensor, condition_rope_sin_.tensor);
        condition_rollout_frames_ = condition_rollout_frames;
    }

    engine::core::TensorValue build_stream_self_attention(
        engine::core::ModuleBuildContext & ctx,
        const engine::core::TensorValue & input,
        const LiveAvatarDenoiserAttentionWeights & weights,
        int64_t layer) {
        const int64_t hidden = config_.hidden_size;
        const int64_t heads = config_.num_heads;
        const int64_t head_dim = hidden / heads;
        const int64_t target_tokens = input.shape.dims[1];
        auto q = linear_native(ctx, input, weights.q, hidden, hidden);
        auto k = linear_native(ctx, input, weights.k, hidden, hidden);
        auto v = linear_native(ctx, input, weights.v, hidden, hidden);
        q = engine::modules::RMSNormModule({hidden, 1.0e-6F, true, false})
                .build(ctx, q, {weights.norm_q, std::nullopt});
        k = engine::modules::RMSNormModule({hidden, 1.0e-6F, true, false})
                .build(ctx, k, {weights.norm_k, std::nullopt});
        q = engine::core::reshape_tensor(
            ctx,
            engine::core::ensure_backend_addressable_layout(ctx, q),
            engine::core::TensorShape::from_dims({1, target_tokens, heads, head_dim}));
        k = engine::core::reshape_tensor(
            ctx,
            engine::core::ensure_backend_addressable_layout(ctx, k),
            engine::core::TensorShape::from_dims({1, target_tokens, heads, head_dim}));
        v = engine::core::reshape_tensor(
            ctx,
            engine::core::ensure_backend_addressable_layout(ctx, v),
            engine::core::TensorShape::from_dims({1, target_tokens, heads, head_dim}));
        auto q_roped = apply_wan_rope(ctx, q, target_rope_cos_, target_rope_sin_, head_dim);
        auto k_roped = apply_wan_rope(ctx, k, target_rope_cos_, target_rope_sin_, head_dim);
        engine::core::TensorValue active_target_k;
        engine::core::TensorValue active_target_v;
        if (cache_->has_target_cache()) {
            auto cached_k = write_liveavatar_kv_cache(ctx, k_roped, cache_->target_key(static_cast<size_t>(layer)), target_cache_write_token_, "target key");
            auto cached_v = write_liveavatar_kv_cache(ctx, v, cache_->target_value(static_cast<size_t>(layer)), target_cache_write_token_, "target value");
            if (!use_bounded_target_cache_) {
                active_target_k = view_liveavatar_kv_cache(
                    ctx,
                    cache_->target_key(static_cast<size_t>(layer)),
                    0,
                    active_target_tokens_,
                    heads,
                    head_dim,
                    "active target key");
                active_target_v = view_liveavatar_kv_cache(
                    ctx,
                    cache_->target_value(static_cast<size_t>(layer)),
                    0,
                    active_target_tokens_,
                    heads,
                    head_dim,
                    "active target value");
                if (target_cache_write_token_ + target_tokens == active_target_tokens_) {
                    active_target_k = target_cache_write_token_ == 0
                        ? cached_k
                        : engine::modules::ConcatModule({1, true}).build(
                              ctx,
                              view_liveavatar_kv_cache(ctx, cache_->target_key(static_cast<size_t>(layer)), 0, target_cache_write_token_, heads, head_dim, "past target key"),
                              cached_k);
                    active_target_v = target_cache_write_token_ == 0
                        ? cached_v
                        : engine::modules::ConcatModule({1, true}).build(
                              ctx,
                              view_liveavatar_kv_cache(ctx, cache_->target_value(static_cast<size_t>(layer)), 0, target_cache_write_token_, heads, head_dim, "past target value"),
                              cached_v);
                }
            } else {
                auto build_active_window = [&](const engine::core::TensorValue & storage,
                                               const engine::core::TensorValue & cached,
                                               const char * label) {
                    std::vector<engine::core::TensorValue> parts;
                    parts.reserve(3);
                    int64_t remaining = active_target_tokens_;
                    int64_t offset = active_target_start_token_;
                    while (remaining > 0) {
                        int64_t part_tokens = std::min<int64_t>(remaining, cache_->max_target_tokens() - offset);
                        const int64_t write_end_token = target_cache_write_token_ + target_tokens;
                        if (offset < target_cache_write_token_ && offset + part_tokens > target_cache_write_token_) {
                            part_tokens = target_cache_write_token_ - offset;
                        } else if (offset < write_end_token && offset + part_tokens > write_end_token) {
                            part_tokens = write_end_token - offset;
                        }
                        const bool is_current =
                            offset == target_cache_write_token_ && part_tokens == target_tokens;
                        parts.push_back(
                            is_current
                                ? cached
                                : view_liveavatar_kv_cache(ctx, storage, offset, part_tokens, heads, head_dim, label));
                        remaining -= part_tokens;
                        offset = (offset + part_tokens) % cache_->max_target_tokens();
                    }
                    auto out = parts.front();
                    for (size_t i = 1; i < parts.size(); ++i) {
                        out = engine::modules::ConcatModule({1, true}).build(ctx, out, parts[i]);
                    }
                    return out;
                };
                active_target_k = build_active_window(
                    cache_->target_key(static_cast<size_t>(layer)),
                    cached_k,
                    "active target key");
                active_target_v = build_active_window(
                    cache_->target_value(static_cast<size_t>(layer)),
                    cached_v,
                    "active target value");
            }
        } else {
            if (active_target_start_token_ != 0 || target_cache_write_token_ != 0 || active_target_tokens_ != target_tokens) {
                throw std::runtime_error("LiveAvatar LiveAvatar disabled target KV cache requires a current-block attention window");
            }
            active_target_k = materialize_liveavatar_kv_f16(ctx, k_roped);
            active_target_v = materialize_liveavatar_kv_f16(ctx, v);
        }
        auto cond_k = view_liveavatar_kv_cache(
            ctx,
            cache_->condition_key(static_cast<size_t>(layer)),
            0,
            cache_->condition_tokens(),
            heads,
            head_dim,
            "condition key");
        auto cond_v = view_liveavatar_kv_cache(
            ctx,
            cache_->condition_value(static_cast<size_t>(layer)),
            0,
            cache_->condition_tokens(),
            heads,
            head_dim,
            "condition value");
        cond_k = apply_wan_rope(ctx, cond_k, condition_rope_cos_, condition_rope_sin_, head_dim);
        auto all_k = engine::modules::ConcatModule({1, true}).build(ctx, active_target_k, cond_k);
        auto all_v = engine::modules::ConcatModule({1, true}).build(ctx, active_target_v, cond_v);
        auto q_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q_roped);
        auto k_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, all_k);
        auto v_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, all_v);
        auto attended = scaled_attention(ctx, execution_.backend(), use_sage_attention_, q_heads, k_heads, v_heads, head_dim);
        return linear_native(ctx, attended, weights.o, hidden, hidden);
    }

    engine::core::TensorValue build_target_block(
        engine::core::ModuleBuildContext & ctx,
        const engine::core::TensorValue & input,
        const engine::core::TensorValue & actual_e0,
        const engine::core::TensorValue & zero_e0,
        const LiveAvatarDenoiserBlockWeights & weights,
        int64_t layer) {
        const int64_t hidden = config_.hidden_size;
        auto block_actual = engine::modules::AddModule().build(
            ctx,
            actual_e0,
            engine::modules::RepeatModule({actual_e0.shape}).build(ctx, weights.modulation));
        auto block_zero = engine::modules::AddModule().build(
            ctx,
            zero_e0,
            engine::modules::RepeatModule({zero_e0.shape}).build(ctx, weights.modulation));
        auto norm1 = engine::modules::LayerNormModule({hidden, 1.0e-6F, false, false}).build(ctx, input, {});
        auto self_input = segmented_adaln_input(ctx, norm1, block_actual, block_zero, 0, 1, input.shape.dims[1]);
        auto self_out = build_stream_self_attention(ctx, self_input, weights.self_attention, layer);
        self_out = split_modulated_tokens(ctx, self_out, block_actual, block_zero, 2, input.shape.dims[1], false);
        auto x = engine::modules::AddModule().build(ctx, input, self_out);
        auto norm3 = engine::modules::LayerNormModule({hidden, 1.0e-6F, true, true}).build(ctx, x, weights.norm3);
        x = engine::modules::AddModule().build(
            ctx,
            x,
            build_cross_attention_with_cached_kv(
                ctx,
                execution_.backend(),
                use_sage_attention_,
                norm3,
                condition_cache_->text_kv(layer),
                weights.cross_attention,
                config_));
        auto norm2 = engine::modules::LayerNormModule({hidden, 1.0e-6F, false, false}).build(ctx, x, {});
        auto ffn_input = segmented_adaln_input(ctx, norm2, block_actual, block_zero, 3, 4, input.shape.dims[1]);
        auto ffn = linear_native(ctx, ffn_input, weights.ffn_in, hidden, config_.ffn_dim);
        ffn = engine::modules::GeluModule({engine::modules::GeluApproximation::Tanh}).build(ctx, ffn);
        ffn = linear_native(ctx, ffn, weights.ffn_out, config_.ffn_dim, hidden);
        ffn = split_modulated_tokens(ctx, ffn, block_actual, block_zero, 5, input.shape.dims[1], false);
        return engine::modules::AddModule().build(ctx, x, ffn);
    }

    void build() {
        if (latent_shape_.rank != 4 ||
            latent_shape_.dims[0] != config_.latent_channels ||
            latent_shape_.dims[1] != 3 ||
            latent_shape_.dims[2] <= 0 ||
            latent_shape_.dims[3] <= 0) {
            throw std::runtime_error("LiveAvatar LiveAvatar block shape is invalid");
        }
        const int64_t hidden = config_.hidden_size;
        const int64_t heads = config_.num_heads;
        const int64_t head_dim = hidden / heads;
        const int64_t target_tokens = latent_shape_.dims[1] * (latent_shape_.dims[2] / 2) * (latent_shape_.dims[3] / 2);
        const int64_t cond_tokens =
            (latent_shape_.dims[2] / 2) * (latent_shape_.dims[3] / 2) +
            liveavatar_motion_token_count(latent_shape_.dims[2], latent_shape_.dims[3]);
        target_rope_cos_values_ = make_liveavatar_rope_table(
            heads,
            head_dim,
            liveavatar_target_rope_ranges(latent_shape_.dims[1], latent_shape_.dims[2], latent_shape_.dims[3], target_start_frame_),
            false);
        target_rope_sin_values_ = make_liveavatar_rope_table(
            heads,
            head_dim,
            liveavatar_target_rope_ranges(latent_shape_.dims[1], latent_shape_.dims[2], latent_shape_.dims[3], target_start_frame_),
            true);
        condition_rope_cos_values_ = make_liveavatar_rope_table(
            heads,
            head_dim,
            liveavatar_condition_rope_ranges(latent_shape_.dims[2], latent_shape_.dims[3], condition_rollout_frames_),
            false);
        condition_rope_sin_values_ = make_liveavatar_rope_table(
            heads,
            head_dim,
            liveavatar_condition_rope_ranges(latent_shape_.dims[2], latent_shape_.dims[3], 0),
            true);
        condition_rollout_frames_ = 0;

        ggml_init_params const_params{ggml_tensor_overhead() * 8, nullptr, true};
        const_ctx_.reset(ggml_init(const_params));
        if (const_ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar LiveAvatar block constant context initialization failed");
        }
        engine::core::ModuleBuildContext const_ctx{const_ctx_.get(), "liveavatar.block.const", execution_.backend_type()};
        target_rope_cos_ = engine::core::make_tensor(
            const_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, target_tokens, heads, head_dim / 2}));
        target_rope_sin_ = engine::core::make_tensor(
            const_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, target_tokens, heads, head_dim / 2}));
        condition_rope_cos_ = engine::core::make_tensor(
            const_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, cond_tokens, heads, head_dim / 2}));
        condition_rope_sin_ = engine::core::make_tensor(
            const_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, cond_tokens, heads, head_dim / 2}));
        const_buffer_ = ggml_backend_alloc_ctx_tensors(const_ctx_.get(), execution_.backend());
        if (const_buffer_ == nullptr) {
            throw std::runtime_error("LiveAvatar LiveAvatar block constant buffer allocation failed");
        }
        engine::core::write_tensor_f32(target_rope_cos_, target_rope_cos_values_);
        engine::core::write_tensor_f32(target_rope_sin_, target_rope_sin_values_);
        engine::core::write_tensor_f32(condition_rope_cos_, condition_rope_cos_values_);
        engine::core::write_tensor_f32(condition_rope_sin_, condition_rope_sin_values_);

        ggml_init_params params{kLiveAvatarDenoiserGraphContextBytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar LiveAvatar block graph context initialization failed");
        }
        engine::core::ModuleBuildContext build_ctx{ctx_.get(), "liveavatar.block", execution_.backend_type()};
        latent_ = engine::core::make_tensor(build_ctx, GGML_TYPE_F32, latent_shape_);
        timestep_ = engine::core::make_tensor(build_ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({2, 256}));
        ggml_set_input(latent_.tensor);
        ggml_set_input(timestep_.tensor);

        auto time_emb = linear_native(build_ctx, timestep_, weights_->time_embedding_0, 256, hidden);
        time_emb = engine::modules::SiluModule{}.build(build_ctx, time_emb);
        time_emb = linear_native(build_ctx, time_emb, weights_->time_embedding_2, hidden, hidden);
        auto time_proj = engine::modules::SiluModule{}.build(build_ctx, time_emb);
        time_proj = linear_native(build_ctx, time_proj, weights_->time_projection, hidden, 6 * hidden);
        time_proj = engine::core::reshape_tensor(build_ctx, time_proj, engine::core::TensorShape::from_dims({2, 6, hidden}));
        const auto actual_e0 = engine::modules::SliceModule({0, 0, 1}).build(build_ctx, time_proj);
        const auto zero_e0 = engine::modules::SliceModule({0, 1, 1}).build(build_ctx, time_proj);
        const auto actual_e = engine::core::reshape_tensor(
            build_ctx,
            engine::core::ensure_backend_addressable_layout(
                build_ctx,
                engine::modules::SliceModule({0, 0, 1}).build(build_ctx, time_emb)),
            engine::core::TensorShape::from_dims({1, 1, hidden}));

        auto x = conv3d_tokens(build_ctx, latent_, weights_->patch_embedding, config_.latent_channels, hidden, 1, 2, 2, 1, 2, 2);
        x = engine::modules::AddModule().build(build_ctx, x, target_cache_->target_condition());
        x = engine::modules::AddModule().build(build_ctx, x, build_liveavatar_target_mask(build_ctx, weights_->trainable_cond_mask, x));
        constexpr std::array<int64_t, 12> kAudioInjectLayers{0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39};
        int64_t audio_injector_index = 0;
        for (int64_t layer = 0; layer < config_.num_layers; ++layer) {
            x = build_target_block(
                build_ctx,
                x,
                actual_e0,
                zero_e0,
                weights_->blocks.at(static_cast<size_t>(layer)),
                layer);
            if (audio_injector_index < static_cast<int64_t>(kAudioInjectLayers.size()) &&
                layer == kAudioInjectLayers[static_cast<size_t>(audio_injector_index)]) {
                x = build_audio_injection_with_cached_condition(
                    build_ctx,
                    execution_.backend(),
                    use_sage_attention_,
                    x,
                    condition_cache_->audio_kv(audio_injector_index),
                    condition_cache_->audio_shift(audio_injector_index),
                    condition_cache_->audio_scale(audio_injector_index),
                    weights_->audio_injectors.at(static_cast<size_t>(audio_injector_index)),
                    config_,
                    target_tokens,
                    latent_shape_.dims[1]);
                ++audio_injector_index;
            }
        }
        x = engine::modules::LayerNormModule({hidden, 1.0e-6F, false, false}).build(build_ctx, x, {});
        auto head_e = engine::modules::AddModule().build(
            build_ctx,
            weights_->head.modulation,
            engine::modules::RepeatModule({weights_->head.modulation.shape}).build(build_ctx, actual_e));
        const auto head_shift = engine::modules::SliceModule({1, 0, 1}).build(build_ctx, head_e);
        const auto head_scale = engine::modules::SliceModule({1, 1, 1}).build(build_ctx, head_e);
        x = apply_shift_scale(build_ctx, x, head_shift, head_scale);
        output_ = linear_native(build_ctx, x, weights_->head.projection, hidden, config_.latent_channels * 4).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 1048576, false);
        ggml_build_forward_expand(graph_, output_);
        if (gallocr_ == nullptr) {
            gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        }
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            (own_gallocr_ && !ggml_gallocr_alloc_graph(gallocr_, graph_))) {
            throw std::runtime_error("LiveAvatar LiveAvatar block backend buffer allocation failed");
        }
        engine::core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    engine::core::ExecutionContext & execution_;
    const LiveAvatarDenoiserWeights * weights_ = nullptr;
    LiveAvatarConfig config_;
    engine::core::TensorShape latent_shape_;
    LiveAvatarKVCache * cache_ = nullptr;
    const LiveAvatarBlockConditionCache * condition_cache_ = nullptr;
    const LiveAvatarBlockTargetCache * target_cache_ = nullptr;
    int64_t target_start_frame_ = 0;
    int64_t active_target_start_token_ = 0;
    int64_t target_cache_write_token_ = 0;
    int64_t active_target_tokens_ = 0;
    int64_t condition_rollout_frames_ = 0;
    bool use_bounded_target_cache_ = false;
    bool use_sage_attention_ = false;
    bool memory_saver_ = false;
    std::unique_ptr<ggml_context, GgmlContextDeleter> const_ctx_;
    ggml_backend_buffer_t const_buffer_ = nullptr;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    engine::core::TensorValue latent_;
    engine::core::TensorValue timestep_;
    engine::core::TensorValue target_rope_cos_;
    engine::core::TensorValue target_rope_sin_;
    engine::core::TensorValue condition_rope_cos_;
    engine::core::TensorValue condition_rope_sin_;
    std::vector<float> target_rope_cos_values_;
    std::vector<float> target_rope_sin_values_;
    std::vector<float> condition_rope_cos_values_;
    std::vector<float> condition_rope_sin_values_;
    std::unordered_map<int64_t, std::unique_ptr<LiveAvatarConditionRopeVariant>> condition_rope_variants_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    bool own_gallocr_ = true;
    mutable engine::core::HostGraphPlan plan_;
};

class LiveAvatarBlockState {
public:
    LiveAvatarBlockState(
        engine::core::ExecutionContext & execution,
        const LiveAvatarConfig & config,
        const engine::core::TensorShape & latent_shape)
        : execution_(execution) {
        if (latent_shape.rank != 4 || latent_shape.dims[0] != config.latent_channels ||
            latent_shape.dims[1] != 3 || latent_shape.dims[2] <= 0 || latent_shape.dims[3] <= 0) {
            throw std::runtime_error("LiveAvatar LiveAvatar layerwise state shape is invalid");
        }
        const int64_t hidden = config.hidden_size;
        const int64_t target_tokens = latent_shape.dims[1] * (latent_shape.dims[2] / 2) * (latent_shape.dims[3] / 2);
        ggml_init_params params{ggml_tensor_overhead() * 8, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar LiveAvatar layerwise state context initialization failed");
        }
        engine::core::ModuleBuildContext ctx{ctx_.get(), "liveavatar.layerwise.state", execution_.backend_type()};
        hidden_[0] = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, target_tokens, hidden}));
        hidden_[1] = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, target_tokens, hidden}));
        actual_e0_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, 6, hidden}));
        zero_e0_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, 6, hidden}));
        actual_e_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, 1, hidden}));
        output_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, target_tokens, config.latent_channels * 4}));
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), execution_.backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("LiveAvatar LiveAvatar layerwise state backend buffer allocation failed");
        }
    }

    ~LiveAvatarBlockState() {
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    LiveAvatarBlockState(const LiveAvatarBlockState &) = delete;
    LiveAvatarBlockState & operator=(const LiveAvatarBlockState &) = delete;

    const engine::core::TensorValue & hidden(int index) const { return hidden_.at(static_cast<size_t>(index)); }
    const engine::core::TensorValue & actual_e0() const noexcept { return actual_e0_; }
    const engine::core::TensorValue & zero_e0() const noexcept { return zero_e0_; }
    const engine::core::TensorValue & actual_e() const noexcept { return actual_e_; }
    const engine::core::TensorValue & output() const noexcept { return output_; }

private:
    engine::core::ExecutionContext & execution_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::array<engine::core::TensorValue, 2> hidden_;
    engine::core::TensorValue actual_e0_;
    engine::core::TensorValue zero_e0_;
    engine::core::TensorValue actual_e_;
    engine::core::TensorValue output_;
};

class LiveAvatarBlockPreludeGraph {
public:
    LiveAvatarBlockPreludeGraph(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserWeights & weights,
        LiveAvatarConfig config,
        engine::core::TensorShape latent_shape,
        const LiveAvatarBlockTargetCache & target_cache,
        LiveAvatarBlockState & state)
        : execution_(execution),
          weights_(&weights),
          config_(std::move(config)),
          latent_shape_(std::move(latent_shape)),
          target_cache_(&target_cache),
          state_(&state) {
        build();
    }

    ~LiveAvatarBlockPreludeGraph() {
        if (execution_.backend() != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_.backend_type(), execution_.backend(), graph_, true);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    LiveAvatarBlockPreludeGraph(const LiveAvatarBlockPreludeGraph &) = delete;
    LiveAvatarBlockPreludeGraph & operator=(const LiveAvatarBlockPreludeGraph &) = delete;

    void run(const LiveAvatarDenoiserRunInput & input) const {
        if (input.latent == nullptr || static_cast<int64_t>(input.latent->size()) != latent_shape_.num_elements()) {
            throw std::runtime_error("LiveAvatar LiveAvatar layerwise prelude input payload size mismatch");
        }
        const auto write_start = Clock::now();
        engine::core::write_tensor_f32(latent_, *input.latent);
        engine::core::write_tensor_f32(timestep_, make_timestep_features(input.timestep));
        engine::debug::timing_log_scalar("liveavatar.layerwise_prelude_input_write_ms", engine::debug::elapsed_ms(write_start));
        engine::core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const auto compute_start = Clock::now();
        const ggml_status status =
            engine::core::compute_graph(execution_, graph_, plan_, "liveavatar.layerwise.prelude");
        engine::debug::timing_log_scalar("liveavatar.layerwise_prelude_compute_ms", engine::debug::elapsed_ms(compute_start));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LiveAvatar LiveAvatar layerwise prelude graph compute failed");
        }
    }

private:
    void build() {
        const int64_t hidden = config_.hidden_size;
        ggml_init_params params{kLiveAvatarDenoiserGraphContextBytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar LiveAvatar layerwise prelude graph context initialization failed");
        }
        engine::core::ModuleBuildContext ctx{ctx_.get(), "liveavatar.layerwise.prelude", execution_.backend_type()};
        latent_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, latent_shape_);
        timestep_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({2, 256}));
        ggml_set_input(latent_.tensor);
        ggml_set_input(timestep_.tensor);

        auto time_emb = linear_native(ctx, timestep_, weights_->time_embedding_0, 256, hidden);
        time_emb = engine::modules::SiluModule{}.build(ctx, time_emb);
        time_emb = linear_native(ctx, time_emb, weights_->time_embedding_2, hidden, hidden);
        auto time_proj = engine::modules::SiluModule{}.build(ctx, time_emb);
        time_proj = linear_native(ctx, time_proj, weights_->time_projection, hidden, 6 * hidden);
        time_proj = engine::core::reshape_tensor(ctx, time_proj, engine::core::TensorShape::from_dims({2, 6, hidden}));
        const auto actual_e0 = engine::modules::SliceModule({0, 0, 1}).build(ctx, time_proj);
        const auto zero_e0 = engine::modules::SliceModule({0, 1, 1}).build(ctx, time_proj);
        const auto actual_e = engine::core::reshape_tensor(
            ctx,
            engine::core::ensure_backend_addressable_layout(
                ctx,
                engine::modules::SliceModule({0, 0, 1}).build(ctx, time_emb)),
            engine::core::TensorShape::from_dims({1, 1, hidden}));

        auto x = conv3d_tokens(ctx, latent_, weights_->patch_embedding, config_.latent_channels, hidden, 1, 2, 2, 1, 2, 2);
        x = engine::modules::AddModule().build(ctx, x, target_cache_->target_condition());
        x = engine::modules::AddModule().build(ctx, x, build_liveavatar_target_mask(ctx, weights_->trainable_cond_mask, x));

        auto copy_hidden = ggml_cpy(ctx.ggml, engine::core::ensure_backend_addressable_layout(ctx, x).tensor, state_->hidden(0).tensor);
        auto copy_actual_e0 = ggml_cpy(ctx.ggml, engine::core::ensure_backend_addressable_layout(ctx, actual_e0).tensor, state_->actual_e0().tensor);
        auto copy_zero_e0 = ggml_cpy(ctx.ggml, engine::core::ensure_backend_addressable_layout(ctx, zero_e0).tensor, state_->zero_e0().tensor);
        auto copy_actual_e = ggml_cpy(ctx.ggml, actual_e.tensor, state_->actual_e().tensor);

        graph_ = ggml_new_graph_custom(ctx_.get(), 1048576, false);
        ggml_build_forward_expand(graph_, copy_hidden);
        ggml_build_forward_expand(graph_, copy_actual_e0);
        ggml_build_forward_expand(graph_, copy_zero_e0);
        ggml_build_forward_expand(graph_, copy_actual_e);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("LiveAvatar LiveAvatar layerwise prelude backend buffer allocation failed");
        }
        engine::core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    engine::core::ExecutionContext & execution_;
    const LiveAvatarDenoiserWeights * weights_ = nullptr;
    LiveAvatarConfig config_;
    engine::core::TensorShape latent_shape_;
    const LiveAvatarBlockTargetCache * target_cache_ = nullptr;
    LiveAvatarBlockState * state_ = nullptr;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    engine::core::TensorValue latent_;
    engine::core::TensorValue timestep_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    mutable engine::core::HostGraphPlan plan_;
};

class LiveAvatarBlockLayerGraph {
public:
    LiveAvatarBlockLayerGraph(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserWeights & weights,
        LiveAvatarConfig config,
        engine::core::TensorShape latent_shape,
        LiveAvatarKVCache & cache,
        const LiveAvatarBlockConditionCache & condition_cache,
        const LiveAvatarBlockConstants & constants,
        LiveAvatarBlockState & state,
        int64_t layer_start,
        int64_t layer_end,
        int64_t src_slot,
        int64_t dst_slot,
        int64_t active_target_start_token,
        int64_t target_cache_write_token,
        int64_t active_target_tokens,
        bool use_bounded_target_cache,
        ggml_gallocr_t shared_gallocr,
        bool use_sage_attention)
        : execution_(execution),
          weights_(&weights),
          config_(std::move(config)),
          latent_shape_(std::move(latent_shape)),
          cache_(&cache),
          condition_cache_(&condition_cache),
          constants_(&constants),
          state_(&state),
          layer_start_(layer_start),
          layer_end_(layer_end),
          src_slot_(src_slot),
          dst_slot_(dst_slot),
          active_target_start_token_(active_target_start_token),
          target_cache_write_token_(target_cache_write_token),
          active_target_tokens_(active_target_tokens),
          use_bounded_target_cache_(use_bounded_target_cache),
          use_sage_attention_(use_sage_attention),
          gallocr_(shared_gallocr),
          own_gallocr_(shared_gallocr == nullptr) {
        build();
    }

    ~LiveAvatarBlockLayerGraph() {
        if (execution_.backend() != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_.backend_type(), execution_.backend(), graph_, true);
        }
        if (own_gallocr_ && gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    LiveAvatarBlockLayerGraph(const LiveAvatarBlockLayerGraph &) = delete;
    LiveAvatarBlockLayerGraph & operator=(const LiveAvatarBlockLayerGraph &) = delete;

    void run() const {
        engine::core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        if (streaming_graph_ == nullptr && !own_gallocr_ && !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("LiveAvatar LiveAvatar layerwise block backend buffer allocation failed");
        }
        const auto compute_start = Clock::now();
        const ggml_status status = streaming_graph_ != nullptr
            ? streaming_graph_->compute(graph_)
            : engine::core::compute_graph(execution_, graph_, plan_, "liveavatar.layerwise.block");
        engine::debug::timing_log_scalar("liveavatar.layerwise_block_compute_ms", engine::debug::elapsed_ms(compute_start));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LiveAvatar LiveAvatar layerwise block graph compute failed");
        }
    }

private:
    engine::core::TensorValue build_stream_self_attention(
        engine::core::ModuleBuildContext & ctx,
        const engine::core::TensorValue & input,
        const LiveAvatarDenoiserAttentionWeights & weights,
        int64_t layer) {
        const int64_t hidden = config_.hidden_size;
        const int64_t heads = config_.num_heads;
        const int64_t head_dim = hidden / heads;
        const int64_t target_tokens = input.shape.dims[1];
        auto q = linear_native(ctx, input, weights.q, hidden, hidden);
        auto k = linear_native(ctx, input, weights.k, hidden, hidden);
        auto v = linear_native(ctx, input, weights.v, hidden, hidden);
        q = engine::modules::RMSNormModule({hidden, 1.0e-6F, true, false})
                .build(ctx, q, {weights.norm_q, std::nullopt});
        k = engine::modules::RMSNormModule({hidden, 1.0e-6F, true, false})
                .build(ctx, k, {weights.norm_k, std::nullopt});
        q = engine::core::reshape_tensor(
            ctx,
            engine::core::ensure_backend_addressable_layout(ctx, q),
            engine::core::TensorShape::from_dims({1, target_tokens, heads, head_dim}));
        k = engine::core::reshape_tensor(
            ctx,
            engine::core::ensure_backend_addressable_layout(ctx, k),
            engine::core::TensorShape::from_dims({1, target_tokens, heads, head_dim}));
        v = engine::core::reshape_tensor(
            ctx,
            engine::core::ensure_backend_addressable_layout(ctx, v),
            engine::core::TensorShape::from_dims({1, target_tokens, heads, head_dim}));
        auto q_roped = apply_wan_rope(ctx, q, constants_->target_rope_cos(), constants_->target_rope_sin(), head_dim);
        auto k_roped = apply_wan_rope(ctx, k, constants_->target_rope_cos(), constants_->target_rope_sin(), head_dim);
        engine::core::TensorValue active_target_k;
        engine::core::TensorValue active_target_v;
        if (cache_->has_target_cache()) {
            auto cached_k = write_liveavatar_kv_cache(ctx, k_roped, cache_->target_key(static_cast<size_t>(layer)), target_cache_write_token_, "target key");
            auto cached_v = write_liveavatar_kv_cache(ctx, v, cache_->target_value(static_cast<size_t>(layer)), target_cache_write_token_, "target value");
            if (!use_bounded_target_cache_) {
                active_target_k = view_liveavatar_kv_cache(
                    ctx,
                    cache_->target_key(static_cast<size_t>(layer)),
                    0,
                    active_target_tokens_,
                    heads,
                    head_dim,
                    "active target key");
                active_target_v = view_liveavatar_kv_cache(
                    ctx,
                    cache_->target_value(static_cast<size_t>(layer)),
                    0,
                    active_target_tokens_,
                    heads,
                    head_dim,
                    "active target value");
                if (target_cache_write_token_ + target_tokens == active_target_tokens_) {
                    active_target_k = target_cache_write_token_ == 0
                        ? cached_k
                        : engine::modules::ConcatModule({1, true}).build(
                              ctx,
                              view_liveavatar_kv_cache(ctx, cache_->target_key(static_cast<size_t>(layer)), 0, target_cache_write_token_, heads, head_dim, "past target key"),
                              cached_k);
                    active_target_v = target_cache_write_token_ == 0
                        ? cached_v
                        : engine::modules::ConcatModule({1, true}).build(
                              ctx,
                              view_liveavatar_kv_cache(ctx, cache_->target_value(static_cast<size_t>(layer)), 0, target_cache_write_token_, heads, head_dim, "past target value"),
                              cached_v);
                }
            } else {
                auto build_active_window = [&](const engine::core::TensorValue & storage,
                                               const engine::core::TensorValue & cached,
                                               const char * label) {
                    std::vector<engine::core::TensorValue> parts;
                    parts.reserve(3);
                    int64_t remaining = active_target_tokens_;
                    int64_t offset = active_target_start_token_;
                    while (remaining > 0) {
                        int64_t part_tokens = std::min<int64_t>(remaining, cache_->max_target_tokens() - offset);
                        const int64_t write_end_token = target_cache_write_token_ + target_tokens;
                        if (offset < target_cache_write_token_ && offset + part_tokens > target_cache_write_token_) {
                            part_tokens = target_cache_write_token_ - offset;
                        } else if (offset < write_end_token && offset + part_tokens > write_end_token) {
                            part_tokens = write_end_token - offset;
                        }
                        const bool is_current = offset == target_cache_write_token_ && part_tokens == target_tokens;
                        parts.push_back(
                            is_current
                                ? cached
                                : view_liveavatar_kv_cache(ctx, storage, offset, part_tokens, heads, head_dim, label));
                        remaining -= part_tokens;
                        offset = (offset + part_tokens) % cache_->max_target_tokens();
                    }
                    auto out = parts.front();
                    for (size_t i = 1; i < parts.size(); ++i) {
                        out = engine::modules::ConcatModule({1, true}).build(ctx, out, parts[i]);
                    }
                    return out;
                };
                active_target_k = build_active_window(cache_->target_key(static_cast<size_t>(layer)), cached_k, "active target key");
                active_target_v = build_active_window(cache_->target_value(static_cast<size_t>(layer)), cached_v, "active target value");
            }
        } else {
            if (active_target_start_token_ != 0 || target_cache_write_token_ != 0 || active_target_tokens_ != target_tokens) {
                throw std::runtime_error("LiveAvatar LiveAvatar disabled target KV cache requires a current-block attention window");
            }
            active_target_k = materialize_liveavatar_kv_f16(ctx, k_roped);
            active_target_v = materialize_liveavatar_kv_f16(ctx, v);
        }
        auto cond_k = view_liveavatar_kv_cache(
            ctx,
            cache_->condition_key(static_cast<size_t>(layer)),
            0,
            cache_->condition_tokens(),
            heads,
            head_dim,
            "condition key");
        auto cond_v = view_liveavatar_kv_cache(
            ctx,
            cache_->condition_value(static_cast<size_t>(layer)),
            0,
            cache_->condition_tokens(),
            heads,
            head_dim,
            "condition value");
        cond_k = apply_wan_rope(ctx, cond_k, constants_->condition_rope_cos(), constants_->condition_rope_sin(), head_dim);
        auto all_k = engine::modules::ConcatModule({1, true}).build(ctx, active_target_k, cond_k);
        auto all_v = engine::modules::ConcatModule({1, true}).build(ctx, active_target_v, cond_v);
        auto q_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q_roped);
        auto k_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, all_k);
        auto v_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, all_v);
        auto attended = scaled_attention(ctx, execution_.backend(), use_sage_attention_, q_heads, k_heads, v_heads, head_dim);
        return linear_native(ctx, attended, weights.o, hidden, hidden);
    }

    engine::core::TensorValue build_target_block(
        engine::core::ModuleBuildContext & ctx,
        const engine::core::TensorValue & input,
        const LiveAvatarDenoiserBlockWeights & weights,
        int64_t layer) {
        const int64_t hidden = config_.hidden_size;
        auto block_actual = engine::modules::AddModule().build(
            ctx,
            state_->actual_e0(),
            engine::modules::RepeatModule({state_->actual_e0().shape}).build(ctx, weights.modulation));
        auto block_zero = engine::modules::AddModule().build(
            ctx,
            state_->zero_e0(),
            engine::modules::RepeatModule({state_->zero_e0().shape}).build(ctx, weights.modulation));
        auto norm1 = engine::modules::LayerNormModule({hidden, 1.0e-6F, false, false}).build(ctx, input, {});
        auto self_input = segmented_adaln_input(ctx, norm1, block_actual, block_zero, 0, 1, input.shape.dims[1]);
        auto self_out = build_stream_self_attention(ctx, self_input, weights.self_attention, layer);
        self_out = split_modulated_tokens(ctx, self_out, block_actual, block_zero, 2, input.shape.dims[1], false);
        auto x = engine::modules::AddModule().build(ctx, input, self_out);
        auto norm3 = engine::modules::LayerNormModule({hidden, 1.0e-6F, true, true}).build(ctx, x, weights.norm3);
        x = engine::modules::AddModule().build(
            ctx,
            x,
            build_cross_attention_with_cached_kv(
                ctx,
                execution_.backend(),
                use_sage_attention_,
                norm3,
                condition_cache_->text_kv(layer),
                weights.cross_attention,
                config_));
        auto norm2 = engine::modules::LayerNormModule({hidden, 1.0e-6F, false, false}).build(ctx, x, {});
        auto ffn_input = segmented_adaln_input(ctx, norm2, block_actual, block_zero, 3, 4, input.shape.dims[1]);
        auto ffn = linear_native(ctx, ffn_input, weights.ffn_in, hidden, config_.ffn_dim);
        ffn = engine::modules::GeluModule({engine::modules::GeluApproximation::Tanh}).build(ctx, ffn);
        ffn = linear_native(ctx, ffn, weights.ffn_out, config_.ffn_dim, hidden);
        ffn = split_modulated_tokens(ctx, ffn, block_actual, block_zero, 5, input.shape.dims[1], false);
        return engine::modules::AddModule().build(ctx, x, ffn);
    }

    void build() {
        if (layer_start_ < 0 || layer_end_ <= layer_start_ || layer_end_ > config_.num_layers ||
            src_slot_ < 0 || src_slot_ > 1 || dst_slot_ < 0 || dst_slot_ > 1 || src_slot_ == dst_slot_) {
            throw std::runtime_error("LiveAvatar LiveAvatar layerwise block index is invalid");
        }
        const int64_t target_tokens = latent_shape_.dims[1] * (latent_shape_.dims[2] / 2) * (latent_shape_.dims[3] / 2);
        ggml_init_params params{kLiveAvatarDenoiserGraphContextBytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar LiveAvatar layerwise block graph context initialization failed");
        }
        engine::core::ModuleBuildContext ctx{ctx_.get(), "liveavatar.layerwise.block", execution_.backend_type()};
        auto x = state_->hidden(static_cast<int>(src_slot_));
        constexpr std::array<int64_t, 12> kAudioInjectLayers{0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39};
        int64_t audio_injector_index = static_cast<int64_t>(
            std::count_if(kAudioInjectLayers.begin(), kAudioInjectLayers.end(), [&](int64_t layer) { return layer < layer_start_; }));
        for (int64_t layer = layer_start_; layer < layer_end_; ++layer) {
            x = build_target_block(ctx, x, weights_->blocks.at(static_cast<size_t>(layer)), layer);
            if (audio_injector_index < static_cast<int64_t>(kAudioInjectLayers.size()) &&
                layer == kAudioInjectLayers[static_cast<size_t>(audio_injector_index)]) {
                x = build_audio_injection_with_cached_condition(
                    ctx,
                    execution_.backend(),
                    use_sage_attention_,
                    x,
                    condition_cache_->audio_kv(audio_injector_index),
                    condition_cache_->audio_shift(audio_injector_index),
                    condition_cache_->audio_scale(audio_injector_index),
                    weights_->audio_injectors.at(static_cast<size_t>(audio_injector_index)),
                    config_,
                    target_tokens,
                    latent_shape_.dims[1]);
                ++audio_injector_index;
            }
        }
        auto copy_hidden = ggml_cpy(
            ctx.ggml,
            engine::core::ensure_backend_addressable_layout(ctx, x).tensor,
            state_->hidden(static_cast<int>(dst_slot_)).tensor);
        graph_ = ggml_new_graph_custom(ctx_.get(), 1048576, false);
        ggml_build_forward_expand(graph_, copy_hidden);
        if (weights_->block_weights_host_resident) {
            streaming_graph_ = std::make_unique<LiveAvatarWeightStreamingGraph>(execution_, graph_);
        } else {
            if (gallocr_ == nullptr) {
                gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
            }
            if (gallocr_ == nullptr ||
                !ggml_gallocr_reserve(gallocr_, graph_) ||
                !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
                throw std::runtime_error("LiveAvatar LiveAvatar layerwise block backend buffer allocation failed");
            }
            engine::core::prepare_host_graph_plan(execution_, graph_, plan_);
        }
    }

    engine::core::ExecutionContext & execution_;
    const LiveAvatarDenoiserWeights * weights_ = nullptr;
    LiveAvatarConfig config_;
    engine::core::TensorShape latent_shape_;
    LiveAvatarKVCache * cache_ = nullptr;
    const LiveAvatarBlockConditionCache * condition_cache_ = nullptr;
    const LiveAvatarBlockConstants * constants_ = nullptr;
    LiveAvatarBlockState * state_ = nullptr;
    int64_t layer_start_ = 0;
    int64_t layer_end_ = 0;
    int64_t src_slot_ = 0;
    int64_t dst_slot_ = 1;
    int64_t active_target_start_token_ = 0;
    int64_t target_cache_write_token_ = 0;
    int64_t active_target_tokens_ = 0;
    bool use_bounded_target_cache_ = false;
    bool use_sage_attention_ = false;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    bool own_gallocr_ = true;
    std::unique_ptr<LiveAvatarWeightStreamingGraph> streaming_graph_;
    mutable engine::core::HostGraphPlan plan_;
};

class LiveAvatarBlockFinalGraph {
public:
    LiveAvatarBlockFinalGraph(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserWeights & weights,
        LiveAvatarConfig config,
        engine::core::TensorShape latent_shape,
        LiveAvatarBlockState & state,
        int64_t src_slot)
        : execution_(execution),
          weights_(&weights),
          config_(std::move(config)),
          latent_shape_(std::move(latent_shape)),
          state_(&state),
          src_slot_(src_slot) {
        build();
    }

    ~LiveAvatarBlockFinalGraph() {
        if (execution_.backend() != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_.backend_type(), execution_.backend(), graph_, true);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    LiveAvatarBlockFinalGraph(const LiveAvatarBlockFinalGraph &) = delete;
    LiveAvatarBlockFinalGraph & operator=(const LiveAvatarBlockFinalGraph &) = delete;

    std::vector<float> run() const {
        engine::core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const auto compute_start = Clock::now();
        const ggml_status status =
            engine::core::compute_graph(execution_, graph_, plan_, "liveavatar.layerwise.final");
        engine::debug::timing_log_scalar("liveavatar.layerwise_final_compute_ms", engine::debug::elapsed_ms(compute_start));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LiveAvatar LiveAvatar layerwise final graph compute failed");
        }
        const auto read_start = Clock::now();
        auto out = unpatchify_head_tokens(
            engine::core::read_tensor_f32(state_->output().tensor),
            latent_shape_.dims[1],
            latent_shape_.dims[2],
            latent_shape_.dims[3]);
        engine::debug::timing_log_scalar("liveavatar.layerwise_final_output_read_ms", engine::debug::elapsed_ms(read_start));
        return out;
    }

private:
    void build() {
        const int64_t hidden = config_.hidden_size;
        ggml_init_params params{kLiveAvatarDenoiserGraphContextBytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar LiveAvatar layerwise final graph context initialization failed");
        }
        engine::core::ModuleBuildContext ctx{ctx_.get(), "liveavatar.layerwise.final", execution_.backend_type()};
        auto x = state_->hidden(static_cast<int>(src_slot_));
        x = engine::modules::LayerNormModule({hidden, 1.0e-6F, false, false}).build(ctx, x, {});
        auto head_e = engine::modules::AddModule().build(
            ctx,
            weights_->head.modulation,
            engine::modules::RepeatModule({weights_->head.modulation.shape}).build(ctx, state_->actual_e()));
        const auto head_shift = engine::modules::SliceModule({1, 0, 1}).build(ctx, head_e);
        const auto head_scale = engine::modules::SliceModule({1, 1, 1}).build(ctx, head_e);
        x = apply_shift_scale(ctx, x, head_shift, head_scale);
        x = linear_native(ctx, x, weights_->head.projection, hidden, config_.latent_channels * 4);
        auto copied = ggml_cpy(ctx.ggml, engine::core::ensure_backend_addressable_layout(ctx, x).tensor, state_->output().tensor);
        graph_ = ggml_new_graph_custom(ctx_.get(), 1048576, false);
        ggml_build_forward_expand(graph_, copied);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("LiveAvatar LiveAvatar layerwise final backend buffer allocation failed");
        }
        engine::core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    engine::core::ExecutionContext & execution_;
    const LiveAvatarDenoiserWeights * weights_ = nullptr;
    LiveAvatarConfig config_;
    engine::core::TensorShape latent_shape_;
    LiveAvatarBlockState * state_ = nullptr;
    int64_t src_slot_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    mutable engine::core::HostGraphPlan plan_;
};

class LiveAvatarLayerwiseBlockRunner {
public:
    LiveAvatarLayerwiseBlockRunner(
        engine::core::ExecutionContext & execution,
        const LiveAvatarDenoiserWeights & weights,
        LiveAvatarConfig config,
        engine::core::TensorShape latent_shape,
        LiveAvatarKVCache & cache,
        const LiveAvatarBlockConditionCache & condition_cache,
        const LiveAvatarBlockTargetCache & target_cache,
        int64_t target_start_frame,
        int64_t active_target_start_token,
        int64_t target_cache_write_token,
        int64_t active_target_tokens,
        int64_t layer_batch,
        int64_t initial_condition_rollout_frames,
        bool use_bounded_target_cache,
        bool use_sage_attention)
        : execution_(execution),
          weights_(&weights),
          config_(std::move(config)),
          latent_shape_(std::move(latent_shape)),
          cache_(&cache),
          condition_cache_(&condition_cache),
          target_cache_(&target_cache),
          constants_(std::make_unique<LiveAvatarBlockConstants>(
              execution_, config_, latent_shape_, target_start_frame, initial_condition_rollout_frames)),
          state_(std::make_unique<LiveAvatarBlockState>(execution_, config_, latent_shape_)),
          prelude_(std::make_unique<LiveAvatarBlockPreludeGraph>(execution_, *weights_, config_, latent_shape_, *target_cache_, *state_)),
          active_target_start_token_(active_target_start_token),
          target_cache_write_token_(target_cache_write_token),
          active_target_tokens_(active_target_tokens),
          layer_batch_(layer_batch),
          use_bounded_target_cache_(use_bounded_target_cache),
          use_sage_attention_(use_sage_attention) {
        if (layer_batch_ <= 0) {
            throw std::runtime_error("LiveAvatar LiveAvatar layerwise batch must be positive");
        }
        if (!weights_->block_weights_host_resident) {
            layer_gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
            if (layer_gallocr_ == nullptr) {
                throw std::runtime_error("LiveAvatar LiveAvatar layerwise workspace allocation failed");
            }
        }
        build_layer_graphs();
    }

    ~LiveAvatarLayerwiseBlockRunner() {
        layer_graphs_.clear();
        final_graph_.reset();
        if (layer_gallocr_ != nullptr) {
            ggml_gallocr_free(layer_gallocr_);
        }
    }

    std::vector<float> run(const LiveAvatarDenoiserRunInput & input, int64_t condition_rollout_frames) {
        const auto total_start = Clock::now();
        constants_->update_condition_rope(condition_rollout_frames);
        prelude_->run(input);
        if (weights_->block_weights_host_resident) {
            int64_t src = 0;
            int64_t dst = 1;
            for (int64_t layer = 0; layer < config_.num_layers; layer += layer_batch_) {
                const int64_t layer_end = std::min<int64_t>(layer + layer_batch_, config_.num_layers);
                LiveAvatarBlockLayerGraph layer_graph(
                    execution_,
                    *weights_,
                    config_,
                    latent_shape_,
                    *cache_,
                    *condition_cache_,
                    *constants_,
                    *state_,
                    layer,
                    layer_end,
                    src,
                    dst,
                    active_target_start_token_,
                    target_cache_write_token_,
                    active_target_tokens_,
                    use_bounded_target_cache_,
                    nullptr,
                    use_sage_attention_);
                layer_graph.run();
                std::swap(src, dst);
            }
        } else {
            for (const auto & layer_graph : layer_graphs_) {
                layer_graph->run();
            }
        }
        auto out = final_graph_->run();
        engine::debug::timing_log_scalar("liveavatar.layerwise_total_ms", engine::debug::elapsed_ms(total_start));
        return out;
    }

private:
    void build_layer_graphs() {
        int64_t src = 0;
        int64_t dst = 1;
        for (int64_t layer = 0; layer < config_.num_layers; layer += layer_batch_) {
            const int64_t layer_end = std::min<int64_t>(layer + layer_batch_, config_.num_layers);
            if (!weights_->block_weights_host_resident) {
                layer_graphs_.push_back(std::make_unique<LiveAvatarBlockLayerGraph>(
                    execution_,
                    *weights_,
                    config_,
                    latent_shape_,
                    *cache_,
                    *condition_cache_,
                    *constants_,
                    *state_,
                    layer,
                    layer_end,
                    src,
                    dst,
                    active_target_start_token_,
                    target_cache_write_token_,
                    active_target_tokens_,
                    use_bounded_target_cache_,
                    layer_gallocr_,
                    use_sage_attention_));
            }
            std::swap(src, dst);
        }
        final_graph_ = std::make_unique<LiveAvatarBlockFinalGraph>(
            execution_,
            *weights_,
            config_,
            latent_shape_,
            *state_,
            src);
    }

    engine::core::ExecutionContext & execution_;
    const LiveAvatarDenoiserWeights * weights_ = nullptr;
    LiveAvatarConfig config_;
    engine::core::TensorShape latent_shape_;
    LiveAvatarKVCache * cache_ = nullptr;
    const LiveAvatarBlockConditionCache * condition_cache_ = nullptr;
    const LiveAvatarBlockTargetCache * target_cache_ = nullptr;
    std::unique_ptr<LiveAvatarBlockConstants> constants_;
    std::unique_ptr<LiveAvatarBlockState> state_;
    std::unique_ptr<LiveAvatarBlockPreludeGraph> prelude_;
    int64_t active_target_start_token_ = 0;
    int64_t target_cache_write_token_ = 0;
    int64_t active_target_tokens_ = 0;
    int64_t layer_batch_ = 1;
    bool use_bounded_target_cache_ = false;
    ggml_gallocr_t layer_gallocr_ = nullptr;
    std::vector<std::unique_ptr<LiveAvatarBlockLayerGraph>> layer_graphs_;
    std::unique_ptr<LiveAvatarBlockFinalGraph> final_graph_;
    bool use_sage_attention_ = false;
};

}  // namespace

LiveAvatarVideoResult generate_liveavatar_blockwise(
    LiveAvatarPipelineState & state,
    engine::core::ExecutionContext & execution,
    const std::shared_ptr<const LiveAvatarAssets> & assets,
    LiveAvatarGenerateShared & shared,
    const LiveAvatarVideoChunkCallback & chunk_callback) {
    const auto & request = shared.request;
    const auto total_start = shared.total_start;
    const auto & text_context = shared.text_context;
    const auto & reference_image_input = shared.reference_image_input;
    auto & ref_latents = shared.ref_latents;
    const auto & audio_buckets = shared.audio_buckets;
    auto * impl_ = &state;
    const auto * assets_ = assets.get();
    auto & execution_ = execution;

    constexpr int64_t kLiveAvatarBlockLatentFrames = 3;
    const int64_t live_latent_frames =
        (request.frames_per_clip + 3 + assets_->config.motion_frames) / 4 - assets_->config.latent_motion_frames;
    const int64_t live_blocks = live_latent_frames / kLiveAvatarBlockLatentFrames;
    if (live_blocks <= 0) {
        throw std::runtime_error("LiveAvatar num_frames is too small for blockwise generation");
    }
    const int64_t latent_height = request.height / 8;
    const int64_t latent_width = request.width / 8;
    const int64_t block_audio_frames = kLiveAvatarBlockLatentFrames * 4;
    const auto live_sigmas = make_liveavatar_euler_sigmas(request.steps, request.shift);

    const auto & ref_frame_values = reference_image_input;

    auto ref_video_state = repeat_video_frame(ref_frame_values, 3, request.height, request.width, 5);
    engine::core::TensorShape live_ref_latents_shape;
    const auto live_ref_vae_start = Clock::now();
    auto live_ref_latents = impl_->vae_encode_cached(
        execution_,
        ref_video_state,
        3,
        5,
        request.height,
        request.width,
        request.vae_encoder_chunk_size,
        request.vae_cache_f16,
        &live_ref_latents_shape);
    apply_wan21_latent_process_in(live_ref_latents, live_ref_latents_shape);
    if (live_ref_latents_shape.dims[1] < 2) {
        throw std::runtime_error("LiveAvatar LiveAvatar reference VAE did not produce the expected temporal latent");
    }
    ref_latents = slice_video_time(
        live_ref_latents,
        live_ref_latents_shape.dims[0],
        live_ref_latents_shape.dims[1],
        live_ref_latents_shape.dims[2],
        live_ref_latents_shape.dims[3],
        1,
        1);
    engine::debug::timing_log_scalar("liveavatar.vae_encode_ref_ms", engine::debug::elapsed_ms(live_ref_vae_start));

    auto motion_video_state = repeat_video_frame(
        ref_frame_values,
        3,
        request.height,
        request.width,
        5 * assets_->config.motion_frames);
    engine::core::TensorShape motion_latents_shape;
    const auto motion_vae_start = Clock::now();
    auto motion_latents = impl_->vae_encode_cached(
        execution_,
        motion_video_state,
        3,
        5 * assets_->config.motion_frames,
        request.height,
        request.width,
        request.vae_encoder_chunk_size,
        request.vae_cache_f16,
        &motion_latents_shape);
    auto latent_process_start = Clock::now();
    apply_wan21_latent_process_in(motion_latents, motion_latents_shape);
    engine::debug::timing_log_scalar(
        "liveavatar.vae_encode_motion_latent_process_ms",
        engine::debug::elapsed_ms(latent_process_start));
    if (motion_latents_shape.dims[1] < assets_->config.latent_motion_frames) {
        throw std::runtime_error("LiveAvatar LiveAvatar motion VAE did not produce enough temporal latents");
    }
    auto motion_condition_latents = slice_video_time(
        motion_latents,
        motion_latents_shape.dims[0],
        motion_latents_shape.dims[1],
        motion_latents_shape.dims[2],
        motion_latents_shape.dims[3],
        motion_latents_shape.dims[1] - assets_->config.latent_motion_frames,
        assets_->config.latent_motion_frames);
    engine::debug::timing_log_scalar("liveavatar.vae_encode_motion_ms", engine::debug::elapsed_ms(motion_vae_start));
    impl_->release_vae_weights(execution_);

    const engine::core::TensorShape block_shape = engine::core::TensorShape::from_dims({
        assets_->config.latent_channels,
        kLiveAvatarBlockLatentFrames,
        latent_height,
        latent_width,
    });
    const int64_t block_target_tokens =
        kLiveAvatarBlockLatentFrames * (latent_height / 2) * (latent_width / 2);
    const int64_t condition_tokens =
        (latent_height / 2) * (latent_width / 2) +
        liveavatar_motion_token_count(latent_height, latent_width);
    const bool use_bounded_target_cache = request.target_cache_blocks > 0;
    const int64_t target_cache_blocks = use_bounded_target_cache
        ? std::min<int64_t>(request.target_cache_blocks, live_blocks)
        : live_blocks;
    const int64_t target_cache_tokens = target_cache_blocks * block_target_tokens;
    const bool carry_target_cache_across_clips = target_cache_blocks > 1;
    engine::debug::trace_log_scalar("liveavatar.target_cache_blocks", target_cache_blocks);
    engine::debug::trace_log_scalar("liveavatar.target_cache_tokens", target_cache_tokens);
    const std::vector<float> block_cond_latents(static_cast<size_t>(block_shape.num_elements()), 0.0F);
    const engine::core::TensorShape clip_shape = engine::core::TensorShape::from_dims({
        assets_->config.latent_channels,
        live_latent_frames,
        latent_height,
        latent_width,
    });
    std::vector<float> clip_latents;
    std::vector<float> clip_output(static_cast<size_t>(clip_shape.num_elements()), 0.0F);
    const int64_t active_clips = std::min<int64_t>(
        request.max_clips,
        std::max<int64_t>(1, audio_buckets.repeats));
    double denoiser_graph_run_ms = 0.0;
    const auto denoise_start = Clock::now();
    double initial_condition_ms = 0.0;

    const auto initial_condition_start = Clock::now();
    const auto first_segment_audio = slice_audio_bucket_frames(
        audio_buckets.values,
        audio_buckets.layers,
        audio_buckets.dims,
        audio_buckets.frames,
        0,
        block_audio_frames);
    WanS2VAudioBuckets first_segment_bucket;
    first_segment_bucket.values = first_segment_audio;
    first_segment_bucket.layers = audio_buckets.layers;
    first_segment_bucket.dims = audio_buckets.dims;
    first_segment_bucket.frames = block_audio_frames;
    const auto first_block_audio_input = prepend_audio_motion_frames(first_segment_bucket, assets_->config.motion_frames);
    const auto first_condition = impl_->prepare_denoiser_condition(
        execution_,
        LiveAvatarDenoiserConditionRunInput{&text_context, &first_block_audio_input},
        kLiveAvatarBlockLatentFrames,
        assets_->config.motion_frames + block_audio_frames,
        assets_->config.latent_motion_frames);
    impl_->release_denoiser_condition_graph();
    initial_condition_ms += engine::debug::elapsed_ms(initial_condition_start);

    std::vector<LiveAvatarHostTargetCache> liveavatar_host_caches(static_cast<size_t>(request.steps));
    std::mt19937_64 liveavatar_rope_rng(request.seed);
    std::vector<float> generated_video;
    int64_t generated_frames = 0;
    auto motion_latents_pp = motion_condition_latents;
    auto videos_last_frames = std::move(motion_video_state);
    double vae_decode_ms = 0.0;
    double block_graph_build_ms = 0.0;
    double target_cache_import_ms = 0.0;
    double target_cache_export_ms = 0.0;
    double block_setup_ms = 0.0;
    double latent_update_ms = 0.0;
    double clip_pack_ms = 0.0;
    double denoiser_release_ms = 0.0;
    double decode_prep_ms = 0.0;
    double post_decode_pack_ms = 0.0;
    double motion_state_update_ms = 0.0;
    double vae_release_ms = 0.0;
    double block_target_cache_ms = 0.0;
    double stream_first_chunk_ms = -1.0;
    int64_t stream_chunk_count = 0;
    int64_t stream_output_frames = 0;
    std::vector<float> stream_pending_video;
    int64_t stream_pending_frames = 0;

    auto flush_stream_chunks = [&](bool flush_tail) {
        if (!chunk_callback) {
            return;
        }
        while (stream_pending_frames >= kLiveAvatarStreamChunkFrames ||
               (flush_tail && stream_pending_frames > 0)) {
            const int64_t frames = std::min<int64_t>(kLiveAvatarStreamChunkFrames, stream_pending_frames);
            if (stream_chunk_count == 0) {
                stream_first_chunk_ms = engine::debug::elapsed_ms(total_start);
            }
            LiveAvatarVideoResult chunk;
            chunk.width = request.width;
            chunk.height = request.height;
            chunk.frames = frames;
            chunk.fps = assets_->config.fps;
            chunk.rgb24 = video_to_rgb24(
                stream_pending_video,
                3,
                stream_pending_frames,
                request.height,
                request.width,
                0,
                frames);
            chunk_callback(std::move(chunk));
            ++stream_chunk_count;
            stream_output_frames += frames;
            if (frames == stream_pending_frames) {
                stream_pending_video.clear();
                stream_pending_frames = 0;
            } else {
                stream_pending_video = slice_video_time(
                    stream_pending_video,
                    3,
                    stream_pending_frames,
                    request.height,
                    request.width,
                    frames,
                    stream_pending_frames - frames);
                stream_pending_frames -= frames;
            }
        }
    };

    const auto block_target_cache_start = Clock::now();
    auto block_target_cache = std::make_unique<LiveAvatarBlockTargetCache>(
        execution_,
        impl_->ensure_denoiser_weights(execution_),
        assets_->config,
        block_shape);
    block_target_cache->populate(block_cond_latents);
    block_target_cache_ms += engine::debug::elapsed_ms(block_target_cache_start);

    for (int64_t clip = 0; clip < active_clips; ++clip) {
        auto & denoiser_weights = impl_->ensure_denoiser_weights(execution_);
        const auto sink_start = Clock::now();
        auto liveavatar_condition_cache = std::make_unique<LiveAvatarKVCache>(
            execution_,
            assets_->config,
            1,
            condition_tokens);
        {
            LiveAvatarSinkGraph sink_graph(
                execution_,
                denoiser_weights,
                assets_->config,
                block_shape,
                *liveavatar_condition_cache,
                request.sage_attention,
                request.denoiser_layerwise_batch);
            sink_graph.run(ref_latents, motion_condition_latents, first_condition.projected_text_context);
        }
        if (denoiser_weights.block_weights_host_resident) {
            auto host_condition_cache = std::make_unique<LiveAvatarKVCache>(
                execution_,
                assets_->config,
                1,
                condition_tokens,
                nullptr,
                true);
            host_condition_cache->copy_condition_from(*liveavatar_condition_cache);
            liveavatar_condition_cache = std::move(host_condition_cache);
            engine::core::trim_backend_pools(execution_.backend());
        }
        engine::debug::timing_log_scalar("liveavatar.sink_total_ms", engine::debug::elapsed_ms(sink_start));
        const bool use_target_kv_cache = target_cache_blocks > 1;
        auto liveavatar_work_cache = std::make_unique<LiveAvatarKVCache>(
            execution_,
            assets_->config,
            use_target_kv_cache ? target_cache_tokens : 0,
            condition_tokens,
            liveavatar_condition_cache.get());

        clip_latents = make_initial_noise(clip_shape, request.seed + static_cast<uint64_t>(clip));
        std::fill(clip_output.begin(), clip_output.end(), 0.0F);
        std::vector<std::vector<float>> block_latents_list;
        std::vector<LiveAvatarDenoiserPreparedCondition> block_conditions;
        std::vector<std::unique_ptr<LiveAvatarBlockConditionCache>> block_condition_caches;
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> shared_block_gallocr(
            request.memory_saver && !request.denoiser_layerwise
                ? ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()))
                : nullptr,
            &ggml_gallocr_free);
        if (request.memory_saver && !request.denoiser_layerwise && shared_block_gallocr == nullptr) {
            throw std::runtime_error("LiveAvatar shared block allocator initialization failed");
        }
        std::vector<std::unique_ptr<LiveAvatarBlockGraph>> block_graphs;
        std::vector<std::vector<int64_t>> block_rollout_frames;
        block_latents_list.reserve(static_cast<size_t>(live_blocks));
        block_conditions.reserve(static_cast<size_t>(live_blocks));
        block_condition_caches.reserve(static_cast<size_t>(live_blocks));
        block_graphs.reserve(static_cast<size_t>(live_blocks));
        block_rollout_frames.reserve(static_cast<size_t>(live_blocks));
        const auto block_setup_start = Clock::now();
        for (int64_t block = 0; block < live_blocks; ++block) {
            const auto segment_audio = slice_audio_bucket_frames(
                audio_buckets.values,
                audio_buckets.layers,
                audio_buckets.dims,
                audio_buckets.frames,
                clip * request.frames_per_clip + block * block_audio_frames,
                block_audio_frames);
            WanS2VAudioBuckets segment_bucket;
            segment_bucket.values = segment_audio;
            segment_bucket.layers = audio_buckets.layers;
            segment_bucket.dims = audio_buckets.dims;
            segment_bucket.frames = block_audio_frames;
            const auto block_audio_input = prepend_audio_motion_frames(segment_bucket, assets_->config.motion_frames);
            const int64_t denoiser_audio_frames = assets_->config.motion_frames + block_audio_frames;
            const int64_t latent_audio_start_frame = assets_->config.latent_motion_frames;

            const auto condition_start = Clock::now();
            auto cond_condition = impl_->prepare_denoiser_condition(
                execution_,
                LiveAvatarDenoiserConditionRunInput{&text_context, &block_audio_input},
                kLiveAvatarBlockLatentFrames,
                denoiser_audio_frames,
                latent_audio_start_frame);
            impl_->release_denoiser_condition_graph();
            engine::debug::timing_log_scalar("liveavatar.denoiser_condition_ms", engine::debug::elapsed_ms(condition_start));
            block_conditions.push_back(std::move(cond_condition));
            const auto * shared_text_cache =
                request.memory_saver && !block_condition_caches.empty()
                    ? block_condition_caches.front().get()
                    : nullptr;
            auto condition_cache = std::make_unique<LiveAvatarBlockConditionCache>(
                execution_,
                denoiser_weights,
                assets_->config,
                kLiveAvatarBlockLatentFrames,
                request.sage_attention,
                shared_text_cache);
            condition_cache->populate(block_conditions.back());
            block_condition_caches.push_back(std::move(condition_cache));

            block_latents_list.push_back(slice_video_time(
                clip_latents,
                clip_shape.dims[0],
                clip_shape.dims[1],
                clip_shape.dims[2],
                clip_shape.dims[3],
                block * kLiveAvatarBlockLatentFrames,
                kLiveAvatarBlockLatentFrames));
            const int64_t absolute_start_frame = (clip * live_blocks + block) * kLiveAvatarBlockLatentFrames;
            const int64_t absolute_start_token = (clip * live_blocks + block) * block_target_tokens;
            const int64_t absolute_end_token = absolute_start_token + block_target_tokens;
            int64_t active_target_start_token = 0;
            int64_t write_start_token = 0;
            int64_t active_target_tokens = block_target_tokens;
            if (use_target_kv_cache) {
                write_start_token =
                    absolute_start_token % liveavatar_work_cache->max_target_tokens();
                if (!use_bounded_target_cache) {
                    active_target_tokens = absolute_start_token >= liveavatar_work_cache->max_target_tokens()
                        ? liveavatar_work_cache->max_target_tokens()
                        : write_start_token + block_target_tokens;
                } else {
                    const int64_t active_start_absolute_token =
                        std::max<int64_t>(0, absolute_end_token - liveavatar_work_cache->max_target_tokens());
                    active_target_start_token =
                        active_start_absolute_token % liveavatar_work_cache->max_target_tokens();
                    active_target_tokens = absolute_end_token - active_start_absolute_token;
                }
            }
            std::vector<int64_t> rollouts;
            rollouts.reserve(static_cast<size_t>(request.steps));
            for (int64_t step = 0; step < request.steps; ++step) {
                rollouts.push_back(liveavatar_condition_rollout_frames(liveavatar_rope_rng, absolute_start_frame));
            }
            const auto block_graph_build_start = Clock::now();
            if (!request.denoiser_layerwise) {
                block_graphs.push_back(std::make_unique<LiveAvatarBlockGraph>(
                    execution_,
                    denoiser_weights,
                    assets_->config,
                    block_shape,
                    *liveavatar_work_cache,
                    *block_condition_caches.back(),
                    *block_target_cache,
                    absolute_start_frame,
                    active_target_start_token,
                    write_start_token,
                    active_target_tokens,
                    use_bounded_target_cache,
                    request.sage_attention,
                    request.memory_saver,
                    shared_block_gallocr.get()));
            }
            block_graph_build_ms += engine::debug::elapsed_ms(block_graph_build_start);
            block_rollout_frames.push_back(std::move(rollouts));
        }
        block_setup_ms += engine::debug::elapsed_ms(block_setup_start);

        const int64_t final_absolute_token = (clip * live_blocks + live_blocks) * block_target_tokens;
        const int64_t final_active_target_tokens =
            final_absolute_token >= liveavatar_work_cache->max_target_tokens()
                ? liveavatar_work_cache->max_target_tokens()
                : final_absolute_token;
        if (request.denoiser_layerwise && !use_target_kv_cache) {
            for (int64_t block = 0; block < live_blocks; ++block) {
                const auto & cond_condition = block_conditions.at(static_cast<size_t>(block));
                auto & block_latents = block_latents_list.at(static_cast<size_t>(block));
                auto layerwise_runner = LiveAvatarLayerwiseBlockRunner(
                    execution_,
                    denoiser_weights,
                    assets_->config,
                    block_shape,
                    *liveavatar_work_cache,
                    *block_condition_caches.at(static_cast<size_t>(block)),
                    *block_target_cache,
                    (clip * live_blocks + block) * kLiveAvatarBlockLatentFrames,
                    0,
                    0,
                    block_target_tokens,
                    request.denoiser_layerwise_batch,
                    block_rollout_frames.at(static_cast<size_t>(block)).front(),
                    use_bounded_target_cache,
                    request.sage_attention);
                for (int64_t step = 0; step < request.steps; ++step) {
                    const float sigma = live_sigmas[static_cast<size_t>(step)];
                    const float sigma_next = live_sigmas[static_cast<size_t>(step + 1)];
                    LiveAvatarDenoiserRunInput cond_input;
                    cond_input.latent = &block_latents;
                    cond_input.ref_latents = &ref_latents;
                    cond_input.cond_latents = &block_cond_latents;
                    cond_input.motion_latents = &motion_condition_latents;
                    cond_input.projected_text_context = &cond_condition.projected_text_context;
                    cond_input.encoded_audio_local = &cond_condition.encoded_audio_local;
                    cond_input.encoded_audio_global = &cond_condition.encoded_audio_global;
                    cond_input.timestep = sigma * 1000.0F;
                    const auto run_start = Clock::now();
                    auto model_output = layerwise_runner.run(
                        cond_input,
                        block_rollout_frames.at(static_cast<size_t>(block)).at(static_cast<size_t>(step)));
                    denoiser_graph_run_ms += engine::debug::elapsed_ms(run_start);
                    const auto update_start = Clock::now();
                    flow_euler_step_in_place(block_latents, model_output, sigma, sigma_next);
                    latent_update_ms += engine::debug::elapsed_ms(update_start);
                }
            }
        } else {
            for (int64_t step = 0; step < request.steps; ++step) {
                const float sigma = live_sigmas[static_cast<size_t>(step)];
                const float sigma_next = live_sigmas[static_cast<size_t>(step + 1)];
                const auto import_start = Clock::now();
                if (carry_target_cache_across_clips) {
                    import_liveavatar_target_cache(
                        liveavatar_host_caches.at(static_cast<size_t>(step)),
                        *liveavatar_work_cache);
                }
                target_cache_import_ms += engine::debug::elapsed_ms(import_start);
                for (int64_t block = 0; block < live_blocks; ++block) {
                    const auto & cond_condition = block_conditions.at(static_cast<size_t>(block));
                    auto & block_latents = block_latents_list.at(static_cast<size_t>(block));
                    LiveAvatarDenoiserRunInput cond_input;
                    cond_input.latent = &block_latents;
                    cond_input.ref_latents = &ref_latents;
                    cond_input.cond_latents = &block_cond_latents;
                    cond_input.motion_latents = &motion_condition_latents;
                    cond_input.projected_text_context = &cond_condition.projected_text_context;
                    cond_input.encoded_audio_local = &cond_condition.encoded_audio_local;
                    cond_input.encoded_audio_global = &cond_condition.encoded_audio_global;
                    cond_input.timestep = sigma * 1000.0F;
                    const auto run_start = Clock::now();
                    std::vector<float> model_output;
                    if (request.denoiser_layerwise) {
                        const int64_t absolute_start_frame = (clip * live_blocks + block) * kLiveAvatarBlockLatentFrames;
                        const int64_t absolute_start_token = (clip * live_blocks + block) * block_target_tokens;
                        const int64_t absolute_end_token = absolute_start_token + block_target_tokens;
                        const int64_t write_start_token =
                            absolute_start_token % liveavatar_work_cache->max_target_tokens();
                        int64_t active_target_start_token = 0;
                        int64_t active_target_tokens = absolute_start_token >= liveavatar_work_cache->max_target_tokens()
                            ? liveavatar_work_cache->max_target_tokens()
                            : write_start_token + block_target_tokens;
                        if (use_bounded_target_cache) {
                            const int64_t active_start_absolute_token =
                                std::max<int64_t>(0, absolute_end_token - liveavatar_work_cache->max_target_tokens());
                            active_target_start_token =
                                active_start_absolute_token % liveavatar_work_cache->max_target_tokens();
                            active_target_tokens = absolute_end_token - active_start_absolute_token;
                        }
                        auto layerwise_runner = LiveAvatarLayerwiseBlockRunner(
                            execution_,
                            denoiser_weights,
                            assets_->config,
                            block_shape,
                            *liveavatar_work_cache,
                            *block_condition_caches.at(static_cast<size_t>(block)),
                            *block_target_cache,
                            absolute_start_frame,
                            active_target_start_token,
                            write_start_token,
                            active_target_tokens,
                            request.denoiser_layerwise_batch,
                            block_rollout_frames.at(static_cast<size_t>(block)).at(static_cast<size_t>(step)),
                            use_bounded_target_cache,
                            request.sage_attention);
                        model_output = layerwise_runner.run(
                            cond_input,
                            block_rollout_frames.at(static_cast<size_t>(block)).at(static_cast<size_t>(step)));
                    } else {
                        model_output = block_graphs.at(static_cast<size_t>(block))->run(
                            cond_input,
                            block_rollout_frames.at(static_cast<size_t>(block)).at(static_cast<size_t>(step)));
                    }
                    denoiser_graph_run_ms += engine::debug::elapsed_ms(run_start);
                    const auto update_start = Clock::now();
                    flow_euler_step_in_place(block_latents, model_output, sigma, sigma_next);
                    latent_update_ms += engine::debug::elapsed_ms(update_start);
                }
                if (clip + 1 < active_clips) {
                    const auto export_start = Clock::now();
                    if (carry_target_cache_across_clips) {
                        export_liveavatar_target_cache(
                            liveavatar_host_caches.at(static_cast<size_t>(step)),
                            *liveavatar_work_cache,
                            final_active_target_tokens);
                    }
                    target_cache_export_ms += engine::debug::elapsed_ms(export_start);
                }
            }
        }

        const auto clip_pack_start = Clock::now();
        for (int64_t block = 0; block < live_blocks; ++block) {
            copy_video_time(
                clip_output,
                assets_->config.latent_channels,
                live_latent_frames,
                latent_height,
                latent_width,
                block * kLiveAvatarBlockLatentFrames,
                block_latents_list.at(static_cast<size_t>(block)),
                kLiveAvatarBlockLatentFrames);
        }
        clip_pack_ms += engine::debug::elapsed_ms(clip_pack_start);
        if (clip + 1 < active_clips) {
            ref_latents = slice_video_time(
                clip_output,
                assets_->config.latent_channels,
                live_latent_frames,
                latent_height,
                latent_width,
                0,
                1);
        }

        const auto denoiser_release_start = Clock::now();
        block_graphs.clear();
        shared_block_gallocr.reset();
        liveavatar_work_cache.reset();
        liveavatar_condition_cache.reset();
        impl_->release_denoiser(execution_);
        engine::core::trim_backend_pools(execution_.backend());
        denoiser_release_ms += engine::debug::elapsed_ms(denoiser_release_start);

        const auto decode_prep_start = Clock::now();
        auto decode_latents = concat_video_time(
            motion_latents_pp,
            assets_->config.latent_channels,
            assets_->config.latent_motion_frames,
            latent_height,
            latent_width,
            clip_output,
            live_latent_frames);
        auto decode_shape = engine::core::TensorShape::from_dims({
            assets_->config.latent_channels,
            assets_->config.latent_motion_frames + live_latent_frames,
            latent_height,
            latent_width,
        });
        apply_wan21_latent_process_out(decode_latents, decode_shape);
        decode_prep_ms += engine::debug::elapsed_ms(decode_prep_start);
        engine::core::TensorShape decoded_shape;
        const auto decode_start = Clock::now();
        auto decoded = impl_->vae_decode(
            execution_,
            decode_latents,
            decode_shape.dims[0],
            decode_shape.dims[1],
            decode_shape.dims[2],
            decode_shape.dims[3],
            request.vae_decoder_tile_size,
            &decoded_shape);
        vae_decode_ms += engine::debug::elapsed_ms(decode_start);
        const auto post_decode_start = Clock::now();
        int64_t image_frames = std::min<int64_t>(request.frames_per_clip, decoded_shape.dims[1]);
        int64_t image_start_frame = decoded_shape.dims[1] - image_frames;
        if (clip == 0 && image_frames > 3) {
            image_start_frame += 3;
            image_frames -= 3;
        }
        auto image = slice_video_time(
            decoded,
            decoded_shape.dims[0],
            decoded_shape.dims[1],
            decoded_shape.dims[2],
            decoded_shape.dims[3],
            image_start_frame,
            image_frames);
        if (chunk_callback && image_frames > 0) {
            if (stream_pending_video.empty()) {
                stream_pending_video = image;
                stream_pending_frames = image_frames;
            } else {
                append_video_time(
                    stream_pending_video,
                    3,
                    stream_pending_frames,
                    image_frames,
                    request.height,
                    request.width,
                    image);
                stream_pending_frames += image_frames;
            }
            flush_stream_chunks(false);
        }
        if (generated_video.empty()) {
            generated_video = image;
            generated_frames = image_frames;
        } else {
            append_video_time(
                generated_video,
                3,
                generated_frames,
                image_frames,
                request.height,
                request.width,
                image);
            generated_frames += image_frames;
        }
        post_decode_pack_ms += engine::debug::elapsed_ms(post_decode_start);
        if (clip + 1 < active_clips) {
            const auto motion_state_start = Clock::now();
            const int64_t overlap_frames = std::min<int64_t>(assets_->config.motion_frames, image_frames);
            videos_last_frames = concat_video_time(
                slice_video_time(
                    videos_last_frames,
                    3,
                    5 * assets_->config.motion_frames,
                    request.height,
                    request.width,
                    overlap_frames,
                    5 * assets_->config.motion_frames - overlap_frames),
                3,
                5 * assets_->config.motion_frames - overlap_frames,
                request.height,
                request.width,
                slice_video_time(image, 3, image_frames, request.height, request.width, image_frames - overlap_frames, overlap_frames),
                overlap_frames);
            motion_state_update_ms += engine::debug::elapsed_ms(motion_state_start);
            engine::core::TensorShape motion_pp_shape;
            const auto motion_pp_start = Clock::now();
            motion_latents_pp = impl_->vae_encode_cached(
                execution_,
                videos_last_frames,
                3,
                5 * assets_->config.motion_frames,
                request.height,
                request.width,
                request.vae_encoder_chunk_size,
                request.vae_cache_f16,
                &motion_pp_shape);
            latent_process_start = Clock::now();
            apply_wan21_latent_process_in(motion_latents_pp, motion_pp_shape);
            if (motion_pp_shape.dims[1] < assets_->config.latent_motion_frames) {
                throw std::runtime_error("LiveAvatar LiveAvatar refreshed motion VAE did not produce enough temporal latents");
            }
            motion_latents_pp = slice_video_time(
                motion_latents_pp,
                motion_pp_shape.dims[0],
                motion_pp_shape.dims[1],
                motion_pp_shape.dims[2],
                motion_pp_shape.dims[3],
                motion_pp_shape.dims[1] - assets_->config.latent_motion_frames,
                assets_->config.latent_motion_frames);
            motion_condition_latents = motion_latents_pp;
            engine::debug::timing_log_scalar(
                "liveavatar.motion_refresh_latent_process_ms",
                engine::debug::elapsed_ms(latent_process_start));
            engine::debug::timing_log_scalar("liveavatar.motion_refresh_vae_encode_ms", engine::debug::elapsed_ms(motion_pp_start));
        }
        const auto vae_release_start = Clock::now();
        impl_->release_vae_weights(execution_);
        engine::core::trim_backend_pools(execution_.backend());
        vae_release_ms += engine::debug::elapsed_ms(vae_release_start);
    }
    liveavatar_host_caches.clear();
    flush_stream_chunks(true);
    engine::debug::timing_log_scalar("liveavatar.vae_decode_ms", vae_decode_ms);
    engine::debug::timing_log_scalar("liveavatar.block_graph_build_ms", block_graph_build_ms);
    engine::debug::timing_log_scalar("liveavatar.block_target_cache_ms", block_target_cache_ms);
    engine::debug::timing_log_scalar("liveavatar.target_cache_import_ms", target_cache_import_ms);
    engine::debug::timing_log_scalar("liveavatar.target_cache_export_ms", target_cache_export_ms);
    engine::debug::timing_log_scalar("liveavatar.initial_condition_ms", initial_condition_ms);
    engine::debug::timing_log_scalar("liveavatar.block_setup_ms", block_setup_ms);
    engine::debug::timing_log_scalar("liveavatar.latent_update_ms", latent_update_ms);
    engine::debug::timing_log_scalar("liveavatar.clip_pack_ms", clip_pack_ms);
    engine::debug::timing_log_scalar("liveavatar.denoiser_release_ms", denoiser_release_ms);
    engine::debug::timing_log_scalar("liveavatar.decode_prep_ms", decode_prep_ms);
    engine::debug::timing_log_scalar("liveavatar.post_decode_pack_ms", post_decode_pack_ms);
    engine::debug::timing_log_scalar("liveavatar.motion_state_update_ms", motion_state_update_ms);
    engine::debug::timing_log_scalar("liveavatar.vae_release_ms", vae_release_ms);
    if (chunk_callback) {
        engine::debug::timing_log_scalar("liveavatar.stream_frames_per_chunk", kLiveAvatarStreamChunkFrames);
        engine::debug::timing_log_scalar("liveavatar.stream_chunk_count", stream_chunk_count);
        engine::debug::timing_log_scalar("liveavatar.stream_output_frames", stream_output_frames);
        engine::debug::timing_log_scalar("liveavatar.stream_first_chunk_ms", stream_first_chunk_ms);
    }

    engine::debug::timing_log_scalar("liveavatar.denoiser_graph_run_ms", denoiser_graph_run_ms);
    engine::debug::timing_log_scalar("liveavatar.denoise_ms", engine::debug::elapsed_ms(denoise_start));
    LiveAvatarVideoResult result;
    result.width = request.width;
    result.height = request.height;
    result.frames = generated_frames;
    result.fps = assets_->config.fps;
    result.rgb24 = video_to_rgb24(
        generated_video,
        3,
        generated_frames,
        request.height,
        request.width,
        0,
        generated_frames);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(total_start));
    return result;
}

}  // namespace engine::community_models::liveavatar
