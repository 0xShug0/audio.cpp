#include "engine/models/sidon/runtime.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>

#include <array>
#include <algorithm>
#include <map>
#include <stdexcept>

namespace engine::models::sidon {
namespace {

constexpr size_t kContextBytes = 4 * 1024 * 1024;

struct ContextDeleter {
    void operator()(ggml_context * context) const { ggml_free(context); }
};

struct AllocatorDeleter {
    void operator()(ggml_gallocr_t allocator) const { ggml_gallocr_free(allocator); }
};

struct SidonSnakeVocoderWeights {
    core::BackendWeightStore store;
    std::map<std::string, core::TensorValue> tensors;

    SidonSnakeVocoderWeights(const assets::TensorSource & source, core::ExecutionContext & execution)
        : store(execution.backend(), execution.backend_type(), "sidon.decoder", kContextBytes) {
        for (const auto & tensor : source.tensors()) {
            if (tensor.name.rfind("decoder.", 0) == 0)
                tensors.emplace(tensor.name, store.load_tensor(source, tensor.name,
                    assets::TensorStorageType::Native, tensor.shape));
        }
        store.upload();
    }
};

class SidonSnakeVocoderGraph {
public:
    SidonSnakeVocoderGraph(core::ExecutionContext & execution, const SidonSnakeVocoderWeights & weights, int64_t frames)
        : execution_(execution), frames_(frames), context_(ggml_init({kContextBytes, nullptr, true})) {
        if (!context_) throw std::runtime_error("Sidon decoder context allocation failed");
        core::ModuleBuildContext ctx{context_.get(), "sidon.decoder", execution.backend_type()};
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1024, frames}));
        ggml_set_input(input_.tensor);
        const auto conv = [&](core::TensorValue x, const std::string & name, int dilation) {
            const auto & w = weights.tensors.at(name + ".weight");
            const auto & dims = w.shape.dims;
            if (dims[2] == 1) {
                auto time_major = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
                auto matrix = core::reshape_tensor(ctx, w, core::TensorShape::from_dims({dims[0], dims[1]}));
                auto projected = modules::LinearModule({dims[1], dims[0], true})
                    .build(ctx, time_major, {matrix, weights.tensors.at(name + ".bias")});
                return modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, projected);
            }
            return modules::Conv1dModule({dims[1], dims[0], dims[2], 1,
                int(dims[2] / 2) * dilation, dilation, true})
                .build(ctx, x, {w, weights.tensors.at(name + ".bias")});
        };
        const auto snake = [&](core::TensorValue x, const std::string & name) {
            return modules::Snake1dModule({x.shape.dims[1]}).build(ctx, x,
                {weights.tensors.at(name + ".alpha"), weights.tensors.at(name + ".inv_alpha")});
        };
        auto x = conv(input_, "decoder.input", 1);
        constexpr std::array<int, 5> strides{8, 5, 4, 3, 2};
        constexpr std::array<int, 3> dilations{1, 3, 9};
        for (size_t stage = 0; stage < strides.size(); ++stage) {
            const auto prefix = "decoder.stages." + std::to_string(stage) + ".";
            x = snake(x, prefix + "activation");
            const auto & w = weights.tensors.at(prefix + "upsample.weight");
            const auto & dims = w.shape.dims;
            const int padding = (strides[stage] + 1) / 2;
            modules::ConvTranspose1dConfig upsample{dims[0], dims[1], dims[2], strides[stage], padding, 1, true};
            const bool crop_padding = !modules::is_conv_transpose1d_col2im_fast_path_eligible(ctx, upsample);
            if (crop_padding) upsample.padding = 0;
            x = modules::ConvTranspose1dModule(upsample).build(ctx, x, {w, weights.tensors.at(prefix + "upsample.bias")});
            if (crop_padding)
                x = modules::SliceModule({2, padding, x.shape.dims[2] - 2 * padding}).build(ctx, x);
            for (size_t unit = 0; unit < dilations.size(); ++unit) {
                const auto block = prefix + "residuals." + std::to_string(unit) + ".";
                auto residual = conv(snake(x, block + "activation1"), block + "conv1", dilations[unit]);
                residual = conv(snake(residual, block + "activation2"), block + "conv2", 1);
                x = modules::AddModule().build(ctx, x, residual);
            }
        }
        output_ = modules::TanhModule().build(ctx, conv(snake(x, "decoder.output_activation"), "decoder.output", 1));
        ggml_set_output(output_.tensor);
        graph_ = ggml_new_graph_custom(context_.get(), 8192, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        core::validate_backend_graph_supported(execution_.backend(), graph_, "sidon.decoder");
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (!ggml_gallocr_alloc_graph(allocator_.get(), graph_))
            throw std::runtime_error("Sidon decoder graph allocation failed");
        core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    ~SidonSnakeVocoderGraph() { core::release_backend_graph_resources(execution_.backend(), graph_, true); }
    int64_t frames() const { return frames_; }

    std::vector<float> run(const std::vector<float> & hidden) {
        if (hidden.size() != static_cast<size_t>(frames_ * 1024))
            throw std::runtime_error("Sidon decoder hidden shape mismatch");
        auto started = std::chrono::steady_clock::now();
        std::vector<float> channel_first(hidden.size());
        for (int64_t t = 0; t < frames_; ++t)
            for (int64_t c = 0; c < 1024; ++c)
                channel_first[c * frames_ + t] = hidden[t * 1024 + c];
        debug::timing_log_scalar("sidon.decoder.input_prepare_ms", debug::elapsed_ms(started));
        started = std::chrono::steady_clock::now();
        core::write_tensor_f32(input_, channel_first);
        debug::timing_log_scalar("sidon.decoder.input_upload_ms", debug::elapsed_ms(started));
        started = std::chrono::steady_clock::now();
        if (core::compute_graph(execution_, graph_, plan_, "sidon.decoder") != GGML_STATUS_SUCCESS)
            throw std::runtime_error("Sidon decoder compute failed");
        ggml_backend_synchronize(execution_.backend());
        debug::timing_log_scalar("sidon.decoder.graph.compute_ms", debug::elapsed_ms(started));
        started = std::chrono::steady_clock::now();
        auto output = core::read_tensor_f32(output_.tensor);
        debug::timing_log_scalar("sidon.decoder.output_read_ms", debug::elapsed_ms(started));
        return output;
    }

private:
    core::ExecutionContext & execution_;
    int64_t frames_;
    std::unique_ptr<ggml_context, ContextDeleter> context_;
    std::unique_ptr<ggml_gallocr, AllocatorDeleter> allocator_;
    core::HostGraphPlan plan_;
    ggml_cgraph * graph_ = nullptr;
    core::TensorValue input_, output_;
};

}  // namespace

struct SidonRuntime::State {
    core::ExecutionContext & execution;
    SidonSnakeVocoderWeights decoder_weights;
    modules::Wav2Vec2BertEncoderComponent wav2vec2bert_encoder;
    std::unique_ptr<SidonSnakeVocoderGraph> decoder;

    State(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & context)
        : execution(context), decoder_weights(*source, context),
          wav2vec2bert_encoder(modules::Wav2Vec2BertEncoderComponent::load_from_tensor_source(
              source, nullptr, context, kContextBytes, kContextBytes, [] {
                  modules::Wav2Vec2BertEncoderConfig config;
                  config.num_hidden_layers = 8;
                  config.output_hidden_layer = 8;
                  config.apply_semantic_normalization = false;
                  config.mask_padded_frames = false;
                  config.project_relative_keys_first = true;
                  config.pointwise_conv_as_linear = true;
                  return config;
              }())) {}
};

SidonRuntime::SidonRuntime(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution)
    : state_(std::make_unique<State>(std::move(source), execution)) {}
SidonRuntime::~SidonRuntime() = default;

modules::Wav2Vec2BertEncoderOutput SidonRuntime::encode(const modules::Wav2Vec2BertEncoderInput & input) {
    state_->wav2vec2bert_encoder.prepare(input.frames);
    return state_->wav2vec2bert_encoder.encode(input);
}

std::vector<float> SidonRuntime::decode(const std::vector<float> & hidden, int64_t frames) {
    if (frames > 2048) {
        // The decoder has finite support (at most 10 latent frames on either side).
        // Bound its intermediate transposes without changing the encoder's context.
        constexpr int64_t tile_frames = 1024;
        constexpr int64_t halo = 16;
        constexpr int64_t samples_per_frame = 960;
        std::vector<float> output;
        output.reserve(static_cast<size_t>(frames * samples_per_frame));
        for (int64_t begin = 0; begin < frames; begin += tile_frames) {
            const int64_t end = std::min(begin + tile_frames, frames);
            const int64_t left = std::max(int64_t(0), begin - halo);
            const int64_t right = std::min(frames, end + halo);
            std::vector<float> tile(hidden.begin() + left * 1024, hidden.begin() + right * 1024);
            auto decoded = decode(tile, right - left);
            const auto first = decoded.begin() + (begin - left) * samples_per_frame;
            const auto last = end == frames ? decoded.end() : first + (end - begin) * samples_per_frame;
            output.insert(output.end(), first, last);
        }
        return output;
    }
    if (!state_->decoder || state_->decoder->frames() != frames) {
        state_->decoder.reset();
        state_->decoder = std::make_unique<SidonSnakeVocoderGraph>(state_->execution, state_->decoder_weights, frames);
    }
    return state_->decoder->run(hidden);
}

}  // namespace engine::models::sidon
