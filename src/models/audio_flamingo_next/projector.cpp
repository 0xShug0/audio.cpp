#include "engine/models/audio_flamingo_next/projector.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::audio_flamingo_next {

namespace assets = engine::assets;
namespace modules = engine::modules;

using Clock = std::chrono::steady_clock;

constexpr size_t kProjectorWeightContextBytes = 4ull * 1024ull * 1024ull;
constexpr float kPi = 3.14159265358979323846F;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct AFNextAudioProjectorWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::LinearWeights linear_1;
    modules::LinearWeights linear_2;
};

std::shared_ptr<const AFNextAudioProjectorWeights> load_projector_weights(
    const AFNextAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType storage_type) {
    const auto & source = *assets.model_weights;
    const auto & audio_config = assets.config.audio_encoder;
    const auto & text_config = assets.config.text_decoder;
    auto weights = std::make_shared<AFNextAudioProjectorWeights>();
    auto store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "audio_flamingo_next.audio_projector.weights",
        kProjectorWeightContextBytes);
    weights->store = store;
    weights->linear_1 = modules::binding::linear_from_source(
        *store,
        source,
        "multi_modal_projector.linear_1",
        storage_type,
        text_config.hidden_size,
        audio_config.hidden_size,
        assets.config.projector_bias);
    weights->linear_2 = modules::binding::linear_from_source(
        *store,
        source,
        "multi_modal_projector.linear_2",
        storage_type,
        text_config.hidden_size,
        text_config.hidden_size,
        assets.config.projector_bias);
    store->upload();
    return weights;
}

int64_t rotary_axis_dim(const AFNextConfig & config) {
    const auto dim = static_cast<int64_t>(static_cast<double>(config.audio_encoder.hidden_size) * config.rote.partial_rotary_factor);
    if (dim <= 0 || dim % 2 != 0 || dim * 2 > config.audio_encoder.hidden_size) {
        throw std::runtime_error("Audio Flamingo Next projector rotary dimension is invalid");
    }
    return dim;
}

std::vector<int64_t> audio_sample_lengths(const AFNextPrompt & prompt, int64_t audio_token_id) {
    std::vector<int64_t> lengths;
    bool in_span = false;
    int64_t current = 0;
    for (const int32_t token : prompt.input_ids) {
        if (token == audio_token_id) {
            in_span = true;
            ++current;
        } else if (in_span) {
            lengths.push_back(current);
            in_span = false;
            current = 0;
        }
    }
    if (in_span) {
        lengths.push_back(current);
    }
    if (lengths.empty()) {
        throw std::runtime_error("Audio Flamingo Next projector prompt has no audio token span");
    }
    return lengths;
}

std::vector<float> build_audio_timestamps(
    const AFNextAudioFeatures & features,
    const AFNextPrompt & prompt,
    const AFNextConfig & config,
    int64_t max_post_length) {
    const auto sample_lengths = audio_sample_lengths(prompt, config.text_decoder.audio_token_id);
    const int64_t n_audio_tokens = std::accumulate(sample_lengths.begin(), sample_lengths.end(), int64_t{0});
    const int64_t n_audio_features = std::accumulate(features.post_lengths.begin(), features.post_lengths.end(), int64_t{0});
    if (n_audio_tokens != n_audio_features) {
        throw std::runtime_error("Audio Flamingo Next projector audio token count does not match feature count");
    }
    if (static_cast<int64_t>(features.post_lengths.size()) != features.batch) {
        throw std::runtime_error("Audio Flamingo Next projector post length count does not match encoded batch");
    }

    std::vector<int64_t> cumsum_samples(sample_lengths.size());
    std::partial_sum(sample_lengths.begin(), sample_lengths.end(), cumsum_samples.begin());

    std::vector<int64_t> cumsum_post(features.post_lengths.size(), 0);
    for (size_t i = 1; i < features.post_lengths.size(); ++i) {
        cumsum_post[i] = cumsum_post[i - 1] + features.post_lengths[i - 1];
    }

    std::vector<int64_t> sample_indices(features.post_lengths.size());
    for (size_t i = 0; i < cumsum_post.size(); ++i) {
        sample_indices[i] = std::upper_bound(cumsum_samples.begin(), cumsum_samples.end(), cumsum_post[i])
            - cumsum_samples.begin();
        if (sample_indices[i] < 0 || sample_indices[i] >= static_cast<int64_t>(sample_lengths.size())) {
            throw std::runtime_error("Audio Flamingo Next projector sample index is out of range");
        }
    }

    std::vector<int64_t> sample_start_rows(sample_lengths.size(), 0);
    for (size_t i = 0; i < sample_lengths.size(); ++i) {
        sample_start_rows[i] = std::lower_bound(sample_indices.begin(), sample_indices.end(), static_cast<int64_t>(i))
            - sample_indices.begin();
    }

    const float audio_embed_frame_step = config.rote.audio_frame_step * 4.0F;
    std::vector<float> timestamps(static_cast<size_t>(features.batch * max_post_length));
    for (int64_t b = 0; b < features.batch; ++b) {
        const int64_t sample_index = sample_indices[static_cast<size_t>(b)];
        const int64_t window_index = b - sample_start_rows[static_cast<size_t>(sample_index)];
        const float window_base = static_cast<float>(window_index * max_post_length) * audio_embed_frame_step;
        for (int64_t t = 0; t < max_post_length; ++t) {
            timestamps[static_cast<size_t>(b * max_post_length + t)] = window_base + static_cast<float>(t) * audio_embed_frame_step;
        }
    }
    return timestamps;
}

void build_rotary_time_embedding(
    const AFNextConfig & config,
    const std::vector<float> & timestamps,
    int64_t batch,
    int64_t seq_len,
    int64_t rot_dim,
    std::vector<float> & cos_values,
    std::vector<float> & sin_values) {
    const int64_t axis_dim = rot_dim / 2;
    const int64_t inv_count = axis_dim / 2;
    std::vector<float> inv_freq(static_cast<size_t>(inv_count));
    for (int64_t i = 0; i < inv_count; ++i) {
        inv_freq[static_cast<size_t>(i)] = std::pow(config.rote.rope_theta, -static_cast<float>(2 * i) / static_cast<float>(axis_dim));
    }

    cos_values.resize(static_cast<size_t>(batch * seq_len * rot_dim));
    sin_values.resize(static_cast<size_t>(batch * seq_len * rot_dim));
    const float window_duration = config.rote.audio_frame_step * 4.0F * static_cast<float>(seq_len);
    for (int64_t b = 0; b < batch; ++b) {
        const float window_start = timestamps[static_cast<size_t>(b * seq_len)];
        const float window_position = std::round(window_start / window_duration) / static_cast<float>(config.rote.max_position_embeddings);
        for (int64_t t = 0; t < seq_len; ++t) {
            const float timestamp = timestamps[static_cast<size_t>(b * seq_len + t)];
            const float position = static_cast<float>(t) / static_cast<float>(config.rote.max_position_embeddings) * (2.0F * kPi);
            const float angle = -timestamp * 2.0F * kPi;
            for (int64_t i = 0; i < inv_count; ++i) {
                const float window_freq = window_position * inv_freq[static_cast<size_t>(i)];
                const float time_freq = position * inv_freq[static_cast<size_t>(i)];
                const int64_t base = (b * seq_len + t) * rot_dim + i * 2;
                const float window_value = window_freq * angle;
                const float time_value = time_freq * angle;
                cos_values[static_cast<size_t>(base)] = std::cos(window_value);
                sin_values[static_cast<size_t>(base)] = std::sin(window_value);
                cos_values[static_cast<size_t>(base + 1)] = cos_values[static_cast<size_t>(base)];
                sin_values[static_cast<size_t>(base + 1)] = sin_values[static_cast<size_t>(base)];
                const int64_t time_base = base + axis_dim;
                cos_values[static_cast<size_t>(time_base)] = std::cos(time_value);
                sin_values[static_cast<size_t>(time_base)] = std::sin(time_value);
                cos_values[static_cast<size_t>(time_base + 1)] = cos_values[static_cast<size_t>(time_base)];
                sin_values[static_cast<size_t>(time_base + 1)] = sin_values[static_cast<size_t>(time_base)];
            }
        }
    }
}

core::TensorValue apply_rotary_time_embedding(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & hidden,
    const core::TensorValue & cos_values,
    const core::TensorValue & sin_values,
    int64_t rot_dim) {
    auto rotated = modules::SliceModule({2, 0, rot_dim}).build(ctx, hidden);
    auto passthrough = modules::SliceModule({2, rot_dim, hidden.shape.last_dim() - rot_dim}).build(ctx, hidden);
    const auto pairs = core::TensorShape::from_dims({hidden.shape.dims[0], hidden.shape.dims[1], rot_dim / 2, 2});
    auto paired = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, rotated), pairs);
    auto cos_pairs = core::reshape_tensor(ctx, cos_values, pairs);
    auto sin_pairs = core::reshape_tensor(ctx, sin_values, pairs);
    const auto even = modules::SliceModule({3, 0, 1}).build(ctx, paired);
    const auto odd = modules::SliceModule({3, 1, 1}).build(ctx, paired);
    const auto cos = modules::SliceModule({3, 0, 1}).build(ctx, cos_pairs);
    const auto sin = modules::SliceModule({3, 0, 1}).build(ctx, sin_pairs);
    core::TensorValue result;
    if (ctx.backend_type == core::BackendType::Cpu) {
        // The fused rotary-pair op has no CPU kernel.
        result = modules::SplitRoPEModule({2}).build(ctx, paired, cos, sin);
    } else {
        result = modules::RopeInterleavedPairsModule{}.build(ctx, even, odd, cos, sin);
    }
    rotated = core::reshape_tensor(ctx, result, rotated.shape);
    return modules::ConcatModule({2}).build(ctx, rotated, passthrough);
}

class AFNextAudioProjectorGraph {
public:
    AFNextAudioProjectorGraph(
        std::shared_ptr<const AFNextAssets> assets,
        std::shared_ptr<const AFNextAudioProjectorWeights> weights,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        int64_t batch,
        int64_t tokens)
        : assets_(std::move(assets)),
          weights_(std::move(weights)),
          backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          compute_threads_(std::max(1, execution.config().threads)),
          batch_(batch),
          tokens_(tokens) {
        if (assets_ == nullptr || weights_ == nullptr) {
            throw std::runtime_error("Audio Flamingo Next projector graph requires assets and weights");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("Audio Flamingo Next projector backend is not initialized");
        }
        const auto build_start = Clock::now();
        const auto & config = assets_->config;
        if (batch_ <= 0 || tokens_ <= 0) {
            throw std::runtime_error("Audio Flamingo Next projector graph shape is invalid");
        }
        rot_dim_ = rotary_axis_dim(config) * 2;
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Audio Flamingo Next projector graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "audio_flamingo_next.audio_projector", backend_type_};
        auto input = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({batch_, tokens_, config.audio_encoder.hidden_size}));
        input_ = input.tensor;
        auto cos_values = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_, tokens_, rot_dim_}));
        auto sin_values = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_, tokens_, rot_dim_}));
        cos_ = cos_values.tensor;
        sin_ = sin_values.tensor;
        // These tables are reused until their timestamps change.
        ggml_set_input(cos_);
        ggml_set_output(cos_);
        ggml_set_input(sin_);
        ggml_set_output(sin_);

        auto x = apply_rotary_time_embedding(ctx, input, cos_values, sin_values, rot_dim_);
        x = modules::LinearModule({config.audio_encoder.hidden_size, config.text_decoder.hidden_size, config.projector_bias})
                .build(ctx, x, {weights_->linear_1.weight, weights_->linear_1.bias});
        x = modules::GeluModule().build(ctx, x);
        x = modules::LinearModule({config.text_decoder.hidden_size, config.text_decoder.hidden_size, config.projector_bias})
                .build(ctx, x, {weights_->linear_2.weight, weights_->linear_2.bias});
        output_ = x.tensor;
        output_dim_ = config.text_decoder.hidden_size;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 32768, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate Audio Flamingo Next projector graph");
        }
        debug::timing_log_scalar("audio_flamingo_next.audio_projector.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("audio_flamingo_next.audio_projector.batch", batch_);
        debug::trace_log_scalar("audio_flamingo_next.audio_projector.tokens", tokens_);
        debug::trace_log_scalar("audio_flamingo_next.audio_projector.rot_dim", rot_dim_);
    }

    ~AFNextAudioProjectorGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(const AFNextAudioProjectorWeights & weights, int64_t batch, int64_t tokens, ggml_backend_t backend, int threads) const {
        return weights_.get() == &weights && batch_ == batch && tokens_ == tokens && backend_ == backend && compute_threads_ == std::max(1, threads);
    }

    AFNextAudioProjectorOutput run(
        const AFNextAudioEncoderOutput & encoded,
        const AFNextAudioFeatures & features,
        const AFNextPrompt & prompt) {
        const auto & config = assets_->config;
        if (encoded.batch != batch_ || encoded.tokens != tokens_ || encoded.hidden_size != config.audio_encoder.hidden_size) {
            throw std::runtime_error("Audio Flamingo Next projector encoded shape mismatch");
        }
        if (static_cast<int64_t>(encoded.values.size()) != batch_ * tokens_ * config.audio_encoder.hidden_size) {
            throw std::runtime_error("Audio Flamingo Next projector encoded value count mismatch");
        }
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(input_, encoded.values.data(), 0, encoded.values.size() * sizeof(float));
        const auto timestamps = build_audio_timestamps(features, prompt, config, tokens_);
        if (cached_timestamps_ != timestamps) {
            std::vector<float> cos_values;
            std::vector<float> sin_values;
            build_rotary_time_embedding(config, timestamps, batch_, tokens_, rot_dim_, cos_values, sin_values);
            ggml_backend_tensor_set(cos_, cos_values.data(), 0, cos_values.size() * sizeof(float));
            ggml_backend_tensor_set(sin_, sin_values.data(), 0, sin_values.size() * sizeof(float));
            cached_timestamps_ = timestamps;
        }
        debug::timing_log_scalar("audio_flamingo_next.audio_projector.input_upload_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));

        core::set_backend_threads(backend_, compute_threads_);
        timing_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        debug::timing_log_scalar("audio_flamingo_next.audio_projector.graph.compute_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Audio Flamingo Next projector graph compute failed");
        }

        std::vector<float> full_output(static_cast<size_t>(batch_ * tokens_ * output_dim_));
        timing_start = Clock::now();
        ggml_backend_tensor_get(output_, full_output.data(), 0, full_output.size() * sizeof(float));
        AFNextAudioProjectorOutput out;
        out.hidden_size = output_dim_;
        out.tokens = std::accumulate(features.post_lengths.begin(), features.post_lengths.end(), int64_t{0});
        out.values.resize(static_cast<size_t>(out.tokens * out.hidden_size));
        size_t dst = 0;
        for (int64_t b = 0; b < batch_; ++b) {
            const int64_t valid = features.post_lengths[static_cast<size_t>(b)];
            if (valid < 0 || valid > tokens_) {
                throw std::runtime_error("Audio Flamingo Next projector post length is out of range");
            }
            const size_t src = static_cast<size_t>(b * tokens_ * output_dim_);
            const size_t count = static_cast<size_t>(valid * output_dim_);
            std::copy_n(full_output.data() + src, count, out.values.data() + dst);
            dst += count;
        }
        debug::timing_log_scalar("audio_flamingo_next.audio_projector.output_read_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        return out;
    }

private:
    std::shared_ptr<const AFNextAssets> assets_;
    std::shared_ptr<const AFNextAudioProjectorWeights> weights_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    std::vector<float> cached_timestamps_;
    int compute_threads_ = 1;
    int64_t batch_ = 0;
    int64_t tokens_ = 0;
    int64_t rot_dim_ = 0;
    int64_t output_dim_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * cos_ = nullptr;
    ggml_tensor * sin_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

AFNextAudioProjectorRuntime::AFNextAudioProjectorRuntime(
    std::shared_ptr<const AFNextAssets> assets,
    core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    assets::TensorStorageType weight_storage_type)
    : assets_(std::move(assets)),
      execution_(&execution),
      graph_arena_bytes_(graph_arena_bytes) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Audio Flamingo Next projector requires assets");
    }
    if (graph_arena_bytes_ == 0) {
        throw std::runtime_error("Audio Flamingo Next projector graph arena must be non-zero");
    }
    weights_ = load_projector_weights(*assets_, execution.backend(), execution.backend_type(), weight_storage_type);
}

AFNextAudioProjectorRuntime::~AFNextAudioProjectorRuntime() = default;

AFNextAudioProjectorOutput AFNextAudioProjectorRuntime::project(
    const AFNextAudioEncoderOutput & encoded,
    const AFNextAudioFeatures & features,
    const AFNextPrompt & prompt) {
    if (execution_ == nullptr) {
        throw std::runtime_error("Audio Flamingo Next projector execution context is null");
    }
    const int threads = std::max(1, execution_->config().threads);
    if (graph_ == nullptr || !graph_->matches(*weights_, encoded.batch, encoded.tokens, execution_->backend(), threads)) {
        graph_ = std::make_unique<AFNextAudioProjectorGraph>(
            assets_,
            weights_,
            *execution_,
            graph_arena_bytes_,
            encoded.batch,
            encoded.tokens);
    } else {
        debug::timing_log_scalar("audio_flamingo_next.audio_projector.graph.build_ms", 0.0);
        debug::trace_log_scalar("audio_flamingo_next.audio_projector.batch", encoded.batch);
        debug::trace_log_scalar("audio_flamingo_next.audio_projector.tokens", encoded.tokens);
        debug::trace_log_scalar("audio_flamingo_next.audio_projector.rot_dim", rotary_axis_dim(assets_->config) * 2);
    }
    return graph_->run(encoded, features, prompt);
}

}  // namespace engine::models::audio_flamingo_next
