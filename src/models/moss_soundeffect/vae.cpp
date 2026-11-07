#include "engine/models/moss_soundeffect/vae.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace engine::models::moss_soundeffect {
namespace {
namespace binding = modules::binding;
using core::TensorShape;
using core::TensorValue;
using assets::TensorStorageType;

// Signed floor division is needed when receptive fields cross the true left edge.
int64_t floor_div(int64_t value, int64_t divisor) {
    return value / divisor - (value % divisor < 0);
}

struct Conv {
    modules::Conv1dConfig config;
    modules::Conv1dWeights weights;
};
struct Residual {
    std::array<modules::Snake1dWeights, 2> snakes;
    std::array<Conv, 2> convolutions;
};
struct Stage {
    modules::Snake1dWeights snake;
    modules::ConvTranspose1dConfig config;
    modules::ConvTranspose1dWeights weights;
    std::array<Residual, 3> residuals;
};

struct TileGraph {
    core::ExecutionContext & execution;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
    core::HostGraphPlan plan;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;

    explicit TileGraph(core::ExecutionContext & execution) : execution(execution) {
        context.reset(ggml_init({16 * 1024 * 1024, nullptr, true}));
        if (!context) throw std::runtime_error("failed to initialize MOSS DAC tile context");
    }
    ~TileGraph() {
        plan.reset();
        if (graph) core::release_backend_graph_resources(execution.backend(), graph, true);
    }
};
} // namespace

struct VaeRuntime::Impl {
    core::ExecutionContext & execution;
    std::shared_ptr<const Assets> assets;
    core::BackendWeightStore weights;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> weight_views{nullptr, ggml_free};
    Conv post, initial, final;
    std::vector<Stage> stages;
    modules::Snake1dWeights final_snake;
    int64_t frames = 0, hop = 0;
    // Fixed cores partition the trained canvas, independently of requested duration.
    static constexpr int64_t core_frames = 128;
    using GraphKey = std::tuple<int64_t, int64_t, int64_t>; // input length, crop offset, core samples
    std::map<GraphKey, std::unique_ptr<TileGraph>> graphs;
    struct Tile {
        int64_t first, last, output_first, output_count;
        TileGraph * graph;
    };
    std::vector<Tile> tiles;
    std::vector<std::vector<float>> inputs;

    Impl(core::ExecutionContext & execution, std::shared_ptr<const Assets> assets)
        : execution(execution), assets(std::move(assets)),
          weights(execution.backend(), execution.backend_type(), "moss_soundeffect.vae.weights", 4 * 1024 * 1024) {
        const auto started = std::chrono::steady_clock::now();
        const auto & c = this->assets->vae;
        const auto & source = *this->assets->vae_weights;
        frames = int64_t(this->assets->sample_rate) * this->assets->max_duration_sec / c.hop_length;
        hop = 1;
        for (const auto rate : c.rates) hop *= rate;
        if (hop != c.hop_length || frames <= 0)
            throw std::runtime_error("MOSS DAC rates must match the latent hop length");
        weight_views.reset(ggml_init({1024 * 1024, nullptr, true}));
        if (!weight_views) throw std::runtime_error("failed to initialize MOSS DAC weight views");
        const auto load_snake = [&](const std::string & name, int64_t channels) {
            auto alpha = source.require_f32(name);
            auto inverse = alpha;
            for (auto & v : inverse) v = 1.0F / (v + 1e-9F);
            return modules::Snake1dWeights{
                weights.make_f32(TensorShape::from_dims({channels}), std::move(alpha)),
                weights.make_f32(TensorShape::from_dims({channels}), std::move(inverse))};
        };
        post.config = {c.latent_dim, c.latent_dim, 1};
        post.weights = binding::conv1d_from_source(weights, source, "post_quant_conv", TensorStorageType::F32,
            c.latent_dim, c.latent_dim, 1, true);
        initial.config = {c.latent_dim, c.decoder_dim, 7, 1, 3};
        initial.weights = binding::conv1d_from_source(weights, source, "decoder.model.0", TensorStorageType::F32,
            c.decoder_dim, c.latent_dim, 7, true);
        stages.resize(c.rates.size());
        core::ModuleBuildContext layout{weight_views.get(), "moss_soundeffect.vae.weights", execution.backend_type()};
        for (size_t index = 0; index < stages.size(); ++index) {
            auto & stage = stages[index];
            const int64_t in = c.decoder_dim >> index, out = in / 2;
            const auto prefix = "decoder.model." + std::to_string(index + 1) + ".block.";
            stage.snake = load_snake(prefix + "0.alpha", in);
            const int rate = static_cast<int>(c.rates[index]);
            stage.config = {in, out, 2 * rate, rate, 0};
            if (modules::is_conv_transpose1d_col2im_fast_path_eligible(layout, stage.config)) {
                // Store only the shared GEMM layout. The logical native weight is
                // a metadata view of it, used for the module's shape validation.
                const auto raw = source.require_f32(prefix + "1.weight", {in, out, 2 * rate});
                std::vector<float> packed(raw.size());
                for (int64_t i = 0; i < in; ++i)
                    for (int64_t o = 0; o < out; ++o)
                        for (int64_t k = 0; k < 2 * rate; ++k)
                            packed[(o * (2 * rate) + k) * in + i] = raw[(i * out + o) * (2 * rate) + k];
                const auto matrix = weights.make_f32(TensorShape::from_dims({out * (2 * rate), in}), std::move(packed));
                stage.weights.col2im_matrix = matrix;
                auto * native = ggml_permute(weight_views.get(),
                    ggml_reshape_3d(weight_views.get(), matrix.tensor, in, 2 * rate, out), 2, 0, 1, 3);
                stage.weights.weight = core::wrap_tensor(native, TensorShape::from_dims({in, out, 2 * rate}), GGML_TYPE_F32);
                stage.weights.bias = weights.load_f32_tensor(source, prefix + "1.bias", {out});
            } else {
                stage.weights = binding::conv_transpose1d_from_source(weights, source, prefix + "1",
                    TensorStorageType::F32, in, out, 2 * rate, true);
            }
            int dilation = 1;
            for (size_t block = 0; block < stage.residuals.size(); ++block, dilation *= 3) {
                auto & residual = stage.residuals[block];
                const auto rp = prefix + std::to_string(block + 2) + ".block.";
                for (int conv = 0; conv < 2; ++conv) {
                    residual.snakes[conv] = load_snake(rp + std::to_string(2 * conv) + ".alpha", out);
                    auto & layer = residual.convolutions[conv];
                    const int kernel = conv == 0 ? 7 : 1;
                    layer.config = {out, out, kernel, 1, conv == 0 ? 3 * dilation : 0,
                        conv == 0 ? dilation : 1};
                    layer.weights = binding::conv1d_from_source(weights, source, rp + std::to_string(2 * conv + 1),
                        TensorStorageType::F32, out, out, kernel, true);
                }
            }
        }
        const int64_t channels = c.decoder_dim >> c.rates.size();
        final_snake = load_snake("decoder.model." + std::to_string(c.rates.size() + 1) + ".alpha", channels);
        final.config = {channels, 1, 7, 1, 3};
        final.weights = binding::conv1d_from_source(weights, source,
            "decoder.model." + std::to_string(c.rates.size() + 2), TensorStorageType::F32, 1, channels, 7, true);
        weights.upload();
        for (int64_t begin = 0; begin < frames; begin += core_frames) {
            const auto end = std::min(begin + core_frames, frames);
            const auto interval = input_interval(begin * hop, end * hop);
            const auto first = std::max<int64_t>(0, interval.first);
            const auto last = std::min(frames, interval.second);
            const GraphKey key{last - first, (begin - first) * hop, (end - begin) * hop};
            auto found = graphs.find(key);
            if (found == graphs.end()) found = graphs.emplace(key, build_graph(key)).first;
            tiles.push_back({first, last, begin * hop, (end - begin) * hop, found->second.get()});
        }
        inputs.resize(tiles.size());
        for (size_t i = 0; i < tiles.size(); ++i)
            inputs[i].resize(static_cast<size_t>((tiles[i].last - tiles[i].first) * c.latent_dim));
        debug::timing_log_scalar("moss_soundeffect.vae.build_ms", debug::elapsed_ms(started));
    }

    // Half-open output interval -> conservative exact dependency hull at the input.
    // Do not clip intermediate intervals: synthetic tile edges must stay outside
    // every dependency path, including biased layers. Clip only at the true canvas.
    std::pair<int64_t, int64_t> input_interval(int64_t first, int64_t last) const {
        const auto undo_conv = [&](const modules::Conv1dConfig & c) {
            first = first * c.stride - c.padding;
            last = (last - 1) * c.stride - c.padding + c.dilation * (c.kernel_size - 1) + 1;
        };
        undo_conv(final.config);
        for (auto stage = stages.rbegin(); stage != stages.rend(); ++stage) {
            for (auto residual = stage->residuals.rbegin(); residual != stage->residuals.rend(); ++residual) {
                const auto identity = std::make_pair(first, last);
                undo_conv(residual->convolutions[1].config);
                undo_conv(residual->convolutions[0].config);
                first = std::min(first, identity.first);
                last = std::max(last, identity.second);
            }
            const auto & c = stage->config;
            const int64_t crop = (c.stride + 1) / 2;
            first = -floor_div(-(first + crop + c.padding - c.dilation * (c.kernel_size - 1)), c.stride);
            last = floor_div(last - 1 + crop + c.padding, c.stride) + 1;
        }
        undo_conv(initial.config);
        undo_conv(post.config);
        return {first, last};
    }

    std::unique_ptr<TileGraph> build_graph(const GraphKey & key) {
        const auto [input_frames, offset, samples] = key;
        auto g = std::make_unique<TileGraph>(execution);
        core::ModuleBuildContext build{g->context.get(), "moss_soundeffect.vae", execution.backend_type()};
        auto x = core::make_tensor(build, GGML_TYPE_F32,
            TensorShape::from_dims({1, assets->vae.latent_dim, input_frames}));
        g->input = x.tensor;
        ggml_set_input(g->input);
        x = modules::Conv1dModule(post.config).build(build, x, post.weights);
        x = modules::Conv1dModule(initial.config).build(build, x, initial.weights);
        for (const auto & stage : stages) {
            x = modules::Snake1dModule({stage.config.in_channels}).build(build, x, stage.snake);
            const auto upsampled = x.shape.dims[2] * stage.config.stride;
            x = modules::ConvTranspose1dModule(stage.config).build(build, x, stage.weights);
            x = modules::SliceModule({2, (stage.config.stride + 1) / 2, upsampled}).build(build, x);
            for (const auto & residual : stage.residuals) {
                const auto identity = x;
                for (size_t conv = 0; conv < residual.convolutions.size(); ++conv) {
                    const auto & layer = residual.convolutions[conv];
                    x = modules::Snake1dModule({layer.config.in_channels}).build(build, x, residual.snakes[conv]);
                    x = modules::Conv1dModule(layer.config).build(build, x, layer.weights);
                }
                x = modules::AddModule().build(build, identity, x);
            }
        }
        x = modules::Snake1dModule({final.config.in_channels}).build(build, x, final_snake);
        x = modules::Conv1dModule(final.config).build(build, x, final.weights);
        x = modules::SliceModule({2, offset, samples}).build(build, x);
        g->output = modules::TanhModule().build(build, x).tensor;
        ggml_set_output(g->output);
        g->graph = ggml_new_graph_custom(g->context.get(), 32768, false);
        ggml_build_forward_expand(g->graph, g->output);
        core::validate_backend_graph_supported(execution.backend(), g->graph, "MOSS DAC tile");
        g->allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!g->allocator || !ggml_gallocr_alloc_graph(g->allocator.get(), g->graph))
            throw std::runtime_error("failed to allocate MOSS DAC tile graph");
        core::prepare_host_graph_plan(execution, g->graph, g->plan);
        debug::timing_log_scalar("moss_soundeffect.vae.buffer_mb",
            ggml_gallocr_get_buffer_size(g->allocator.get(), 0) / (1024.0 * 1024.0));
        return g;
    }
};

VaeRuntime::VaeRuntime(core::ExecutionContext & execution, std::shared_ptr<const Assets> assets)
    : impl_(std::make_unique<Impl>(execution, std::move(assets))) {}
VaeRuntime::~VaeRuntime() = default;

runtime::AudioBuffer VaeRuntime::decode(const std::vector<float> & latents) {
    return decode(latents, static_cast<size_t>(impl_->frames * impl_->hop));
}

runtime::AudioBuffer VaeRuntime::decode(const std::vector<float> & latents, size_t output_samples) {
    const auto started = std::chrono::steady_clock::now();
    auto & p = *impl_;
    const auto channels = p.assets->vae.latent_dim;
    if (latents.size() != size_t(p.frames * channels))
        throw std::runtime_error("MOSS DAC latent size mismatch");
    if (output_samples > static_cast<size_t>(p.frames * p.hop))
        throw std::runtime_error("MOSS DAC requested output exceeds the trained canvas");
    runtime::AudioBuffer audio;
    audio.sample_rate = p.assets->sample_rate;
    audio.samples.resize(output_samples);
    try {
        for (size_t index = 0; index < p.tiles.size(); ++index) {
            const auto & tile = p.tiles[index];
            if (static_cast<size_t>(tile.output_first) >= output_samples) break;
            const auto length = tile.last - tile.first;
            auto & input = p.inputs[index];
            for (int64_t channel = 0; channel < channels; ++channel)
                std::copy_n(latents.data() + channel * p.frames + tile.first, length, input.data() + channel * length);
            auto & g = *tile.graph;
            ggml_backend_tensor_set_async(p.execution.backend(), g.input, input.data(), 0, input.size() * sizeof(float));
            if (g.plan.active()) {
                if (core::compute_graph(p.execution, g.graph, g.plan, "MOSS DAC tile") != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("MOSS DAC tile graph execution failed");
            } else if (ggml_backend_graph_compute_async(p.execution.backend(), g.graph) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("MOSS DAC tile graph execution failed");
            }
            const auto count = std::min(static_cast<size_t>(tile.output_count), output_samples - static_cast<size_t>(tile.output_first));
            ggml_backend_tensor_get_async(p.execution.backend(), g.output,
                audio.samples.data() + tile.output_first, 0, count * sizeof(float));
        }
        ggml_backend_synchronize(p.execution.backend());
    } catch (...) {
        // Complete queued readbacks before the destination audio is destroyed.
        ggml_backend_synchronize(p.execution.backend());
        throw;
    }
    debug::timing_log_scalar("moss_soundeffect.vae.decode_ms", debug::elapsed_ms(started));
    return audio;
}

} // namespace engine::models::moss_soundeffect
