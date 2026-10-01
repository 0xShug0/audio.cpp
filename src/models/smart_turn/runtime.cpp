#include "engine/models/smart_turn/runtime.h"

#include "engine/framework/audio/dsp.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/speech_encoders/whisper_frontend.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace engine::models::smart_turn {
namespace {

constexpr int64_t kSamples = 8 * 16000;
constexpr int64_t kFrames = 400;
constexpr int64_t kChannels = 384;
constexpr size_t kContextBytes = 4ull * 1024ull * 1024ull;

struct GgmlContextDeleter {
    void operator()(ggml_context * context) const { ggml_free(context); }
};

struct AllocatorDeleter {
    void operator()(ggml_gallocr_t allocator) const { ggml_gallocr_free(allocator); }
};

std::vector<float> normalize_window(const std::vector<float> & audio) {
    if (audio.empty()) {
        throw std::runtime_error("Smart Turn requires non-empty audio");
    }
    std::vector<float> window(static_cast<size_t>(kSamples), 0.0f);
    const size_t count = std::min(audio.size(), window.size());
    std::copy(audio.end() - static_cast<std::ptrdiff_t>(count), audio.end(),
              window.end() - static_cast<std::ptrdiff_t>(count));
    double sum = 0.0;
    for (float value : window) sum += value;
    const float mean = static_cast<float>(sum / kSamples);
    double squared = 0.0;
    for (float value : window) squared += double(value - mean) * double(value - mean);
    const float inverse_std = 1.0f / std::sqrt(static_cast<float>(squared / kSamples) + 1.0e-7f);
    for (float & value : window) value = (value - mean) * inverse_std;
    return window;
}

}  // namespace

class SmartTurnWhisperTinyRuntime::Impl {
public:
    Impl(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution)
        : execution_(execution),
          store_(execution.backend(), execution.backend_type(), "smart_turn.head", kContextBytes),
          mel_({}) {
        if (!source) throw std::runtime_error("Smart Turn requires model weights");
        const auto linear = [&](const std::string & name, int64_t out, int64_t in) {
            return modules::binding::linear_from_source(store_, *source, name,
                assets::TensorStorageType::Native, out, in, true);
        };
        pool_first_ = linear("head.pool.0", 256, kChannels);
        pool_second_ = linear("head.pool.2", 1, 256);
        classifier_first_ = linear("head.classifier.0", 256, kChannels);
        classifier_norm_ = modules::binding::norm_from_source(store_, *source, "head.classifier.1", 256);
        classifier_middle_ = linear("head.classifier.4", 64, 256);
        classifier_last_ = linear("head.classifier.6", 1, 64);
        store_.upload();

        modules::WhisperFrontendComponentConfig config;
        config.name = "smart_turn.whisper_tiny";
        config.weight_context_bytes = kContextBytes;
        config.graph_context_bytes = kContextBytes;
        whisper_ = modules::WhisperFrontendComponent::load_openai_layout(
            std::move(source), execution.config(), {80, kFrames, kChannels, 6, 4, 1.0e-5f}, config);
        build_graph();
    }

    ~Impl() {
        core::release_backend_graph_resources(execution_.backend(), graph_, true);
    }

    float run(const std::vector<float> & audio) {
        const auto frontend_start = std::chrono::steady_clock::now();
        auto window = normalize_window(audio);
        auto mel = mel_.compute(window);
        if (mel.frames != kFrames * 2 || mel.mel_bins != 80) {
            throw std::runtime_error("Smart Turn frontend returned unexpected mel shape");
        }
        debug::timing_log_scalar("smart_turn.frontend.ms", debug::elapsed_ms(frontend_start));
        const auto encoder_start = std::chrono::steady_clock::now();
        auto hidden = whisper_.encode_log_mel(mel.values);
        debug::timing_log_scalar("smart_turn.encoder.ms", debug::elapsed_ms(encoder_start));
        const auto classifier_start = std::chrono::steady_clock::now();
        core::write_tensor_f32(input_, hidden);
        if (core::compute_backend_graph(execution_.backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Smart Turn classifier graph execution failed");
        }
        const float probability = core::read_tensor_f32(output_).at(0);
        debug::timing_log_scalar("smart_turn.classifier.ms", debug::elapsed_ms(classifier_start));
        return probability;
    }

private:
    void build_graph() {
        context_.reset(ggml_init({kContextBytes, nullptr, true}));
        if (!context_) throw std::runtime_error("Smart Turn classifier graph context allocation failed");
        core::ModuleBuildContext ctx{context_.get(), "smart_turn.head", execution_.backend_type()};
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, kFrames, kChannels}));
        ggml_set_input(input_.tensor);

        auto scores = modules::LinearModule({kChannels, 256, true}).build(ctx, input_, pool_first_);
        scores = modules::TanhModule().build(ctx, scores);
        scores = modules::LinearModule({256, 1, true}).build(ctx, scores, pool_second_);
        scores = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, scores);
        scores = modules::SoftmaxModule().build(ctx, scores);
        scores = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, scores);
        scores = core::wrap_tensor(ggml_cont(ctx.ggml, scores.tensor), scores.shape, scores.type);
        scores = modules::RepeatModule({input_.shape}).build(ctx, scores);
        auto pooled = modules::ReduceSumModule({1}).build(ctx, modules::MulModule().build(ctx, input_, scores));

        auto x = modules::LinearModule({kChannels, 256, true}).build(ctx, pooled, classifier_first_);
        x = modules::LayerNormModule({256, 1.0e-5f, true, true}).build(ctx, x, classifier_norm_);
        x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
        x = modules::LinearModule({256, 64, true}).build(ctx, x, classifier_middle_);
        x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
        x = modules::LinearModule({64, 1, true}).build(ctx, x, classifier_last_);
        output_ = modules::SigmoidModule().build(ctx, x).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(context_.get(), 1024, false);
        ggml_build_forward_expand(graph_, output_);
        core::validate_backend_graph_supported(execution_.backend(), graph_, "smart_turn.head");
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (!ggml_gallocr_alloc_graph(allocator_.get(), graph_)) {
            throw std::runtime_error("Smart Turn classifier graph allocation failed");
        }
    }

    core::ExecutionContext & execution_;
    core::BackendWeightStore store_;
    audio::WhisperLogMelExtractor mel_;
    modules::WhisperFrontendComponent whisper_;
    modules::LinearWeights pool_first_, pool_second_;
    modules::LinearWeights classifier_first_, classifier_middle_, classifier_last_;
    modules::NormWeights classifier_norm_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> context_;
    std::unique_ptr<ggml_gallocr, AllocatorDeleter> allocator_;
    core::TensorValue input_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
};

SmartTurnWhisperTinyRuntime::SmartTurnWhisperTinyRuntime(
    std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution)
    : impl_(std::make_unique<Impl>(std::move(source), execution)) {}

SmartTurnWhisperTinyRuntime::~SmartTurnWhisperTinyRuntime() = default;

float SmartTurnWhisperTinyRuntime::completion_probability(const std::vector<float> & audio_16k) {
    return impl_->run(audio_16k);
}

}  // namespace engine::models::smart_turn
