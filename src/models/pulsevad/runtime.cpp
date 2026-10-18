#include "engine/models/pulsevad/runtime.h"
#include "engine/framework/runtime/graph_optimizer.h"

#include "engine/framework/audio/dsp.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>

#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace engine::models::pulsevad {

class PulseVADRuntime::Graph {
public:
    Graph(const assets::TensorSource & source, core::ExecutionContext & execution,
          assets::TensorStorageType storage)
        : execution_(execution), store_(execution.backend(), execution.backend_type(),
                                       "pulsevad.weights", 128 * 1024) {
        ctx_.reset(ggml_init({2 * 1024 * 1024, nullptr, true}));
        if (!ctx_) {
            throw std::runtime_error("PulseVAD graph context allocation failed");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "pulsevad", execution.backend_type()};
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 64, 21}));
        ggml_set_input(input_.tensor);
        auto x = input_;
        const auto conv = [&](core::TensorValue input, const std::string & name,
                              bool depthwise, int dilation, bool relu) {
            const auto shape = source.require_metadata(name + ".weight").shape;
            if (shape.size() != 3) {
                throw std::runtime_error("PulseVAD convolution weight must have rank 3: " + name);
            }
            const bool bias = !depthwise;
            modules::Conv1dWeights weights;
            weights.weight = store_.load_tensor(source, name + ".weight", storage, shape);
            if (bias) {
                weights.bias = store_.load_f32_tensor(source, name + ".bias", {shape[0]});
            }
            if (depthwise) {
                input = modules::DepthwiseConv1dModule({shape[0], shape[2], 1,
                    static_cast<int>((shape[2] - 1) * dilation / 2), dilation, false})
                    .build(ctx, input, {weights.weight, weights.bias});
            } else {
                if (shape[2] != 1) {
                    throw std::runtime_error("PulseVAD pointwise kernel must be 1: " + name);
                }
                auto weight = core::reshape_tensor(ctx, weights.weight,
                    core::TensorShape::from_dims({shape[0], shape[1]}));
                input = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
                if (execution.backend_type() == core::BackendType::Cuda) {
                    // Preserve frames as independent GEMV batches: the CUDA
                    // unbatched SGEMM path enables TF32 even with PREC_F32.
                    input = core::ensure_backend_addressable_layout(ctx, input);
                    input = core::reshape_tensor(ctx, input,
                        core::TensorShape::from_dims({21, 1, shape[1]}));
                    input = core::wrap_tensor(ggml_mul_mat(ctx.ggml, weight.tensor, input.tensor),
                        core::TensorShape::from_dims({21, 1, shape[0]}), GGML_TYPE_F32);
                    const auto bias = core::wrap_tensor(weights.bias->tensor,
                        core::TensorShape::from_dims({1, 1, shape[0]}));
                    input = modules::AddModule().build(ctx, input,
                        modules::RepeatModule({input.shape}).build(ctx, bias));
                    input = core::reshape_tensor(ctx, input, core::TensorShape::from_dims({1, 21, shape[0]}));
                } else {
                    input = modules::LinearModule({shape[1], shape[0], true, GGML_PREC_F32})
                        .build(ctx, input, {weight, weights.bias});
                }
                input = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
            }
            return relu ? modules::ReluModule{}.build(ctx, input) : input;
        };
        x = conv(x, "adapter", false, 1, true);
        x = conv(x, "conv0_dw", true, 1, false);
        x = conv(x, "conv0_pw", false, 1, true);
        x = conv(x, "block1", false, 1, true);
        x = conv(x, "block2", false, 1, true);
        auto main = conv(x, "subA_dw", true, 1, false);
        main = conv(main, "subA_pw", false, 1, true);
        main = conv(main, "subC_dw", true, 1, false);
        main = conv(main, "subC_pw", false, 1, false);
        auto skip = conv(x, "skip", false, 1, false);
        x = modules::ReluModule{}.build(ctx, modules::ResidualAddModule{}.build(ctx, main, skip));
        x = conv(x, "conv4_dw", true, 2, false);
        x = conv(x, "conv4_pw", false, 1, true);
        x = conv(x, "conv5", false, 1, true);
        const int64_t channels = x.shape.dims[1];
        x = modules::ReduceMeanModule({2}).build(ctx, x);
        x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, channels}));
        modules::LinearWeights classifier;
        classifier.weight = store_.load_tensor(source, "classifier.weight", storage, {2, channels});
        classifier.bias = store_.load_f32_tensor(source, "classifier.bias", {2});
        output_ = modules::LinearModule({channels, 2, true, GGML_PREC_F32}).build(ctx, x, classifier);
        ggml_set_output(output_.tensor);
        store_.upload();
        graph_ = ggml_new_graph_custom(ctx_.get(), 512, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        // Fold explicit module broadcasts without changing other graph lowerings.
        runtime::GraphOptimizationOptions optimization;
        optimization.fold_commutative_lhs_repeats = false;
        optimization.fold_two_sided_broadcast_repeats = false;
        optimization.fold_unary_broadcast_repeats = false;
        optimization.fold_identity_materializations = false;
        optimization.elide_noop_nodes = false;
        optimization.elide_metadata_only_ops = false;
        runtime::optimize_graph(*graph_, optimization);
        core::validate_backend_graph_supported(execution.backend(), graph_, "pulsevad");
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!allocator_ || !ggml_gallocr_alloc_graph(allocator_.get(), graph_)) {
            throw std::runtime_error("PulseVAD graph allocation failed");
        }
        core::prepare_host_graph_plan(execution, graph_, plan_);
    }

    ~Graph() {
        core::release_backend_graph_resources(execution_.backend(), graph_, true);
    }

    std::vector<float> run(const std::vector<float> & features) {
        if (features.size() != 64 * 21) {
            throw std::runtime_error("PulseVAD features must have shape [1, 64, 21]");
        }
        core::write_tensor_f32(input_, features);
        if (core::compute_graph(execution_, graph_, plan_, "pulsevad") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("PulseVAD graph compute failed");
        }
        return core::read_tensor_f32(output_.tensor);
    }

private:
    struct ContextDeleter {
        void operator()(ggml_context * value) const { ggml_free(value); }
    };
    struct AllocatorDeleter {
        void operator()(ggml_gallocr_t value) const { ggml_gallocr_free(value); }
    };
    core::ExecutionContext & execution_;
    core::BackendWeightStore store_;
    std::unique_ptr<ggml_context, ContextDeleter> ctx_;
    std::unique_ptr<ggml_gallocr, AllocatorDeleter> allocator_;
    core::HostGraphPlan plan_;
    core::TensorValue input_;
    core::TensorValue output_;
    ggml_cgraph * graph_ = nullptr;
};

PulseVADRuntime::PulseVADRuntime(std::shared_ptr<const assets::TensorSource> source,
                               core::ExecutionContext & context,
                               assets::TensorStorageType storage_type)
    : graph_(std::make_unique<Graph>(*source, context, storage_type)),
      window_(source->require_f32("frontend.window", {400})),
      filterbank_(audio::MelFilterbank{}.prepare_sparse(
          {source->require_f32("frontend.mel_filterbank", {64, 257}), {64, 257}})) {}

PulseVADRuntime::~PulseVADRuntime() = default;

std::vector<float> PulseVADRuntime::infer_features(const std::vector<float> & features) {
    return graph_->run(features);
}

std::vector<float> PulseVADRuntime::extract_features(const std::vector<float> & window) const {
    if (window.size() != 3200) {
        throw std::runtime_error("PulseVAD requires a 3200-sample window at 16000 Hz");
    }
    std::vector<float> normalized(window.size());
    normalized[0] = window[0];
    for (size_t i = 1; i < window.size(); ++i) {
        normalized[i] = window[i] - 0.97f * window[i - 1];
    }
    const float mean = std::accumulate(normalized.begin(), normalized.end(), 0.0f) / normalized.size();
    float variance = 0.0f;
    for (float value : normalized) {
        variance += (value - mean) * (value - mean);
    }
    const float stddev = std::sqrt(variance / normalized.size());
    for (float & value : normalized) {
        value = (value - mean) / (stddev + 1e-5f);
    }
    const audio::STFTConfig config{512, 160, 400, true, audio::STFTPadMode::Reflect};
    auto magnitude = audio::STFT{}.compute_magnitude(normalized, window_, 1, 3200, config, 1);
    auto mel = audio::MelFilterbank{}.compute_custom_sparse_from_magnitude(
        magnitude.values, 1, 257, 21, 21, filterbank_);
    for (int bin = 0; bin < 64; ++bin) {
        double sum = 0.0;
        std::array<double, 21> values{};
        for (int frame = 0; frame < 21; ++frame) {
            values[frame] = std::log(static_cast<double>(mel.values[bin * 21 + frame]) + 1e-5);
            sum += values[frame];
        }
        const double bin_mean = sum / 21;
        double squared = 0.0;
        for (double value : values) {
            squared += (value - bin_mean) * (value - bin_mean);
        }
        const double scale = std::sqrt(squared / 21) + 1e-5;
        for (int frame = 0; frame < 21; ++frame) {
            mel.values[bin * 21 + frame] = static_cast<float>((values[frame] - bin_mean) / scale);
        }
    }
    return mel.values;
}

}  // namespace engine::models::pulsevad
