#include "engine/models/ced/session.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/models/ced/assets.h"
#include "engine/models/ced/frontend.h"
#include "engine/models/ced/runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace engine::models::ced {
namespace {

class CedSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    CedSession(const runtime::TaskSpec & task, const runtime::SessionOptions & options,
               std::shared_ptr<const CedAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)), contract_(std::move(contract)),
          frontend_(assets_->config, *assets_->resources.open_tensor_source("weights")),
          encoder_(assets_->resources.open_tensor_source("weights"), assets_->config, execution_context(),
                   runtime::parse_tensor_storage_option(options.options, "ced.weight_type", "ced.weight_type",
                       assets::TensorStorageType::Native,
                       {assets::TensorStorageType::Native, assets::TensorStorageType::F32,
                        assets::TensorStorageType::F16, assets::TensorStorageType::BF16,
                        assets::TensorStorageType::Q8_0, assets::TensorStorageType::Q4_0, assets::TensorStorageType::Q4_K}),
                   core::resolve_flash_attention(execution_context().backend(), assets_->config.hidden / assets_->config.heads,
                       core::parse_attention_preference(runtime::find_option(options.options, {"ced.attention"}).value_or("auto"), "ced.attention"))) {
        if (task.task != runtime::VoiceTaskKind::AudioClassification || task.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("CED supports offline audio classification");
        }
        runtime::validate_spec_backed_session_options(options, *contract_, "ced", "CED");
    }

    std::string family() const override { return "ced"; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }

    void prepare(const runtime::SessionPreparationRequest & request) override {
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "CED");
        if (!request.audio) throw std::runtime_error("CED preparation requires audio");
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("CED run");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "CED");
        if (!request.audio_input) throw std::runtime_error("CED requires audio_input");
        const int top_k = runtime::parse_int_option(request.options, {"top_k"}).value_or(10);
        if (top_k < 1 || static_cast<size_t>(top_k) > assets_->config.labels.size()) {
            throw std::runtime_error("CED top_k is outside the label count");
        }
        const auto started = std::chrono::steady_clock::now();
        const auto & audio = *request.audio_input;
        auto mono = audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
            audio.samples, audio.sample_rate, audio.channels, assets_->config.sample_rate,
            audio::torchaudio_sinc_hann_float32_options(), audio::MonoMixAccumulation::Float32);
        const auto & config = assets_->config;
        const int64_t minimum_samples = (config.center ? 0 : config.fft_size) +
            (config.patch_size - 1) * config.hop_size;
        if (static_cast<int64_t>(mono.size()) < minimum_samples) {
            throw std::runtime_error("CED requires at least " + std::to_string(minimum_samples) +
                " samples at " + std::to_string(config.sample_rate) + " Hz for one spectrogram patch");
        }
        debug::timing_log_scalar("ced.frontend.resample_ms", debug::elapsed_ms(started));
        const auto features = frontend_.extract(mono, execution_context().config().threads);
        debug::timing_log_scalar("ced.frontend_ms", debug::elapsed_ms(started));
        const auto logits = encoder_.classify(features.values, features.shape.at(2));
        std::vector<size_t> indices(logits.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
            [&](size_t a, size_t b) { return logits[a] > logits[b]; });
        io::json::Value::Array scores;
        for (int i = 0; i < top_k; ++i) {
            const auto index = indices[static_cast<size_t>(i)];
            scores.push_back(io::json::Value::make_object({
                {"index", io::json::Value::make_number(static_cast<double>(index))},
                {"label", io::json::Value::make_string(assets_->config.labels[index])},
                {"score", io::json::Value::make_number(1.0f / (1.0f + std::exp(-logits[index])))},
                {"logit", io::json::Value::make_number(logits[index])},
            }));
        }
        runtime::TaskResult result;
        result.custom_schema_output = runtime::CustomSchemaOutput{"ced.classification.v1",
            io::json::Value::make_object({{"scores", io::json::Value::make_array(std::move(scores))}})};
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const CedAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    CedLogMelFrontend frontend_;
    CedViTRuntime encoder_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_ced_loader() {
    runtime::SpecBackedVoiceModelConfig<CedAssets> config;
    config.family = "ced";
    config.load_assets = load_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                               std::shared_ptr<const CedAssets> assets,
                               std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<CedSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::ced
