#include "engine/community_models/minimax_music3/vocoder.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/primitive_modules.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::minimax_music3 {
namespace {

namespace assets = engine::assets;
namespace core = engine::core;
namespace modules = engine::modules;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct ConvWeights {
    core::TensorValue weight;
    core::TensorValue bias;
};

struct ResidualUnitWeights {
    core::TensorValue snake1_alpha;
    ConvWeights conv1;
    core::TensorValue snake2_alpha;
    ConvWeights conv2;
};

struct UpsampleBlockWeights {
    core::TensorValue snake1_alpha;
    ConvWeights conv_t1;
    ResidualUnitWeights res_units[3];
};

struct VocoderWeights {
    core::BackendWeightStore store;
    ConvWeights dec_in_proj;
    ConvWeights conv_in;
    std::vector<UpsampleBlockWeights> blocks;
    core::TensorValue snake_out_alpha;
    ConvWeights conv_out;

    VocoderWeights(
        core::ExecutionContext & execution,
        const assets::TensorSource & source,
        const MiniMaxMusic3Config & config,
        size_t weight_context_bytes)
        : store(execution.backend(), execution.backend_type(), "minimax_music3.vocoder", weight_context_bytes) {
        const int64_t half_latent = config.vocoder_latent_channels / 2;
        dec_in_proj = load_conv(source, "dec_in_proj", {config.vocoder_input_dim, half_latent, 1});
        conv_in = load_conv(source, "conv_in", {config.vocoder_hidden_dim, config.vocoder_input_dim, 7});
        const size_t num_blocks = config.vocoder_strides.size();
        blocks.reserve(num_blocks);
        for (size_t block = 0; block < num_blocks; ++block) {
            const int64_t input_dim = config.vocoder_hidden_dim >> block;
            const int64_t output_dim = config.vocoder_hidden_dim >> (block + 1);
            const int64_t stride = config.vocoder_strides[block];
            const std::string prefix = "blocks." + std::to_string(block) + ".";
            UpsampleBlockWeights out;
            out.snake1_alpha = load_alpha(source, prefix + "snake1.alpha", input_dim);
            out.conv_t1 = load_conv(source, prefix + "conv_t1", {input_dim, output_dim, 2 * stride});
            for (int unit = 0; unit < 3; ++unit) {
                const std::string unit_prefix = prefix + "res_unit" + std::to_string(unit + 1) + ".";
                auto & res = out.res_units[unit];
                res.snake1_alpha = load_alpha(source, unit_prefix + "snake1.alpha", output_dim);
                res.conv1 = load_conv(source, unit_prefix + "conv1", {output_dim, output_dim, 7});
                res.snake2_alpha = load_alpha(source, unit_prefix + "snake2.alpha", output_dim);
                res.conv2 = load_conv(source, unit_prefix + "conv2", {output_dim, output_dim, 1});
            }
            blocks.push_back(std::move(out));
        }
        const int64_t final_dim = config.vocoder_hidden_dim >> num_blocks;
        snake_out_alpha = load_alpha(source, "snake_out.alpha", final_dim);
        conv_out = load_conv(source, "conv_out", {1, final_dim, 7});
        store.upload();
        source.release_storage();
    }

private:
    ConvWeights load_conv(
        const assets::TensorSource & source,
        const std::string & name,
        std::initializer_list<int64_t> weight_shape) {
        ConvWeights out;
        out.weight = store.load_tensor(source, name + ".weight", assets::TensorStorageType::Native, weight_shape);
        const int64_t bias_size = *weight_shape.begin();
        // ConvTranspose1d weights are [in, out, k]; their bias size is the out dim.
        const auto bias_meta = source.require_metadata(name + ".bias");
        out.bias = store.load_tensor(
            source,
            name + ".bias",
            assets::TensorStorageType::F32,
            {bias_meta.shape.at(0)});
        (void)bias_size;
        return out;
    }

    core::TensorValue load_alpha(
        const assets::TensorSource & source,
        const std::string & name,
        int64_t channels) {
        return store.load_tensor(source, name, assets::TensorStorageType::F32, {1, channels, 1});
    }
};

class VocoderDecodeGraph {
public:
    VocoderDecodeGraph(
        core::ExecutionContext & execution,
        VocoderWeights & weights,
        const MiniMaxMusic3Config & config,
        int64_t length)
        : execution_(execution) {
        ctx_.reset(ggml_init({512 * 1024 * 1024, nullptr, true}));
        input_ctx_.reset(ggml_init({4 * 1024 * 1024, nullptr, true}));
        if (ctx_ == nullptr || input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MiniMax-Music3 vocoder graph context");
        }
        const int64_t half_latent = config.vocoder_latent_channels / 2;

        core::ModuleBuildContext input_ctx{input_ctx_.get(), "minimax_music3.vocoder.inputs", execution_.backend_type()};
        // The stereo fold: latent rows [128, L] are two stacked [64, L] channel streams,
        // which is exactly a [2, 64, L] batch in row-major order.
        latent_ = core::make_tensor(
            input_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({2, half_latent, length}));
        ggml_set_input(latent_.tensor);

        core::ModuleBuildContext build{ctx_.get(), "minimax_music3.vocoder.decode", execution_.backend_type()};
        auto x = modules::Conv1dModule({half_latent, config.vocoder_input_dim, 1, 1, 0, 1, true})
                     .build(build, latent_, {weights.dec_in_proj.weight, weights.dec_in_proj.bias});
        x = modules::Conv1dModule({config.vocoder_input_dim, config.vocoder_hidden_dim, 7, 1, 3, 1, true})
                .build(build, x, {weights.conv_in.weight, weights.conv_in.bias});
        for (size_t block = 0; block < weights.blocks.size(); ++block) {
            const int64_t input_dim = config.vocoder_hidden_dim >> block;
            const int64_t output_dim = config.vocoder_hidden_dim >> (block + 1);
            const int64_t stride = config.vocoder_strides[block];
            const auto & w = weights.blocks[block];
            x = snake(build, x, w.snake1_alpha, input_dim);
            x = modules::ConvTranspose1dModule({
                    input_dim,
                    output_dim,
                    2 * stride,
                    static_cast<int>(stride),
                    static_cast<int>((stride + 1) / 2),
                    1,
                    true})
                    .build(build, x, {w.conv_t1.weight, w.conv_t1.bias});
            for (int unit = 0; unit < 3; ++unit) {
                const auto & res = w.res_units[unit];
                const int dilation = unit == 0 ? 1 : (unit == 1 ? 3 : 9);
                auto y = snake(build, x, res.snake1_alpha, output_dim);
                y = modules::Conv1dModule({output_dim, output_dim, 7, 1, (7 - 1) * dilation / 2, dilation, true})
                        .build(build, y, {res.conv1.weight, res.conv1.bias});
                y = snake(build, y, res.snake2_alpha, output_dim);
                y = modules::Conv1dModule({output_dim, output_dim, 1, 1, 0, 1, true})
                        .build(build, y, {res.conv2.weight, res.conv2.bias});
                x = core::wrap_tensor(
                    ggml_add(build.ggml, x.tensor, y.tensor),
                    x.shape,
                    GGML_TYPE_F32);
            }
        }
        const int64_t final_dim = config.vocoder_hidden_dim >> weights.blocks.size();
        x = snake(build, x, weights.snake_out_alpha, final_dim);
        x = modules::Conv1dModule({final_dim, 1, 7, 1, 3, 1, true})
                .build(build, x, {weights.conv_out.weight, weights.conv_out.bias});
        auto waveform = core::wrap_tensor(ggml_tanh(build.ggml, x.tensor), x.shape, GGML_TYPE_F32);
        output_ = core::ensure_backend_addressable_layout(build, waveform).tensor;

        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_set_output(output_);
        ggml_build_forward_expand(graph_, output_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate MiniMax-Music3 vocoder inputs");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate MiniMax-Music3 vocoder graph");
        }
        core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    ~VocoderDecodeGraph() {
        plan_.reset();
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
        }
    }

    // Returns the two decoded channel streams as one row-major [2, samples] vector.
    std::vector<float> run(const std::vector<float> & latents) {
        core::write_tensor_f32(latent_, latents);
        const ggml_status status = core::compute_graph(execution_, graph_, plan_, "minimax_music3.vocoder");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MiniMax-Music3 vocoder graph compute failed");
        }
        return core::read_tensor_f32(output_);
    }

private:
    static core::TensorValue snake(
        core::ModuleBuildContext & build,
        const core::TensorValue & input,
        const core::TensorValue & alpha,
        int64_t channels) {
        const auto alpha_flat = core::reshape_tensor(build, alpha, core::TensorShape::from_dims({channels}));
        return modules::Snake1dModule({channels}).build(build, input, {alpha_flat});
    }

    core::ExecutionContext & execution_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    core::HostGraphPlan plan_;
    core::TensorValue latent_;
    ggml_tensor * output_ = nullptr;
};

}  // namespace

struct MiniMaxMusic3VocoderRuntime::Impl {
    core::ExecutionContext & execution;
    MiniMaxMusic3Config config;
    VocoderWeights weights;

    Impl(
        core::ExecutionContext & execution_context,
        std::shared_ptr<const assets::TensorSource> source,
        const MiniMaxMusic3Config & cfg,
        size_t weight_context_bytes)
        : execution(execution_context),
          config(cfg),
          weights(execution_context, *source, cfg, weight_context_bytes) {}
};

MiniMaxMusic3VocoderRuntime::MiniMaxMusic3VocoderRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const assets::TensorSource> source,
    const MiniMaxMusic3Config & config,
    size_t weight_context_bytes) {
    if (source == nullptr) {
        throw std::runtime_error("MiniMax-Music3 vocoder tensor source is missing");
    }
    impl_ = std::make_unique<Impl>(execution, std::move(source), config, weight_context_bytes);
}

MiniMaxMusic3VocoderRuntime::~MiniMaxMusic3VocoderRuntime() = default;

std::vector<float> MiniMaxMusic3VocoderRuntime::decode(const std::vector<float> & latents, int64_t length) {
    const auto & config = impl_->config;
    if (length <= 0 ||
        latents.size() != static_cast<size_t>(config.vocoder_latent_channels * length)) {
        throw std::runtime_error("MiniMax-Music3 vocoder latent size mismatch");
    }
    VocoderDecodeGraph graph(impl_->execution, impl_->weights, config, length);
    auto planar = graph.run(latents);
    if (planar.size() % 2 != 0) {
        throw std::runtime_error("MiniMax-Music3 vocoder output size is not stereo");
    }
    const int64_t samples = static_cast<int64_t>(planar.size() / 2);
    for (float & value : planar) {
        value = std::clamp(value, -1.0F, 1.0F);
    }
    return engine::audio::interleave_planar_channels(planar, 2, samples);
}

}  // namespace engine::models::minimax_music3
