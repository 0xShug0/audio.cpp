#include "engine/models/ast_audioset/session.h"

#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/models/ast_audioset/assets.h"
#include "engine/models/ast_audioset/frontend.h"
#include "engine/models/ast_audioset/runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace engine::models::ast_audioset {
namespace {

constexpr const char * kFamily = "ast_audioset";

class Session final : public runtime::RuntimeSessionBase,
                      public runtime::IOfflineVoiceTaskSession {
public:
    Session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options,
        std::shared_ptr<const Assets> assets,
        std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)), contract_(std::move(contract)),
          runtime_(
              assets_->resources.open_tensor_source("weights"),
              assets_->config,
              execution_context(),
              runtime::parse_tensor_storage_option(
                  options.options,
                  "ast_audioset.weight_type",
                  "ast_audioset.weight_type",
                  assets::TensorStorageType::Native,
                  {assets::TensorStorageType::Native, assets::TensorStorageType::F32,
                   assets::TensorStorageType::F16, assets::TensorStorageType::BF16,
                   assets::TensorStorageType::Q8_0, assets::TensorStorageType::Q4_0,
                   assets::TensorStorageType::Q4_K}),
              core::resolve_flash_attention(
                  execution_context().backend(),
                  assets_->config.hidden_size / assets_->config.heads,
                  core::parse_attention_preference(
                      runtime::find_option(options.options, {"ast_audioset.attention"}).value_or("auto"),
                      "ast_audioset.attention"))) {
        if (task_.task != runtime::VoiceTaskKind::AudioClassification || task_.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("AST AudioSet supports offline audio classification");
        }
        runtime::validate_spec_backed_session_options(options, *contract_, kFamily, "AST AudioSet");
    }

    std::string family() const override { return kFamily; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }

    void prepare(const runtime::SessionPreparationRequest & request) override {
        runtime::validate_spec_backed_request_options(
            request.options, request.option_arrays, *contract_, "AST AudioSet");
        if (!request.audio.has_value()) {
            throw std::runtime_error("AST AudioSet preparation requires audio");
        }
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("AST AudioSet run");
        runtime::validate_spec_backed_request_options(
            request.options, request.option_arrays, *contract_, "AST AudioSet");
        if (!request.audio_input.has_value()) {
            throw std::runtime_error("AST AudioSet requires audio_input");
        }
        const auto started = std::chrono::steady_clock::now();
        const size_t values_per_item = static_cast<size_t>(assets_->config.max_length * assets_->config.num_mel_bins);
        auto features = extract_features(
            *request.audio_input,
            assets_->config);
        if (features.size() != values_per_item) {
            throw std::runtime_error("AST AudioSet frontend output size mismatch");
        }
        const int top_k = runtime::parse_int_option(request.options, {"top_k"}).value_or(10);
        if (top_k <= 0 || top_k > static_cast<int>(assets_->config.labels.size())) {
            throw std::runtime_error("AST AudioSet top_k must be between 1 and 527");
        }
        const auto logits = runtime_.classify(features);
        const size_t classes = assets_->config.labels.size();
        if (logits.size() != classes) {
            throw std::runtime_error("AST AudioSet classifier output size mismatch");
        }
        std::vector<int> indices(classes);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(
            indices.begin(), indices.begin() + top_k, indices.end(),
            [&](int left, int right) { return logits[left] > logits[right]; });
        io::json::Value::Array scores;
        scores.reserve(static_cast<size_t>(top_k));
        for (int rank = 0; rank < top_k; ++rank) {
            const int index = indices[static_cast<size_t>(rank)];
            const float logit = logits[static_cast<size_t>(index)];
            scores.push_back(io::json::Value::make_object({
                {"index", io::json::Value::make_number(index)},
                {"label", io::json::Value::make_string(
                              assets_->config.labels[static_cast<size_t>(index)])},
                {"score", io::json::Value::make_number(
                              1.0f / (1.0f + std::exp(-logit)))},
                {"logit", io::json::Value::make_number(logit)},
            }));
        }
        runtime::TaskResult result;
        result.custom_schema_output = runtime::CustomSchemaOutput{
            "ast_audioset.classification.v1",
            io::json::Value::make_object({
                {"scores", io::json::Value::make_array(std::move(scores))},
            }),
        };
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const Assets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    Runtime runtime_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_ast_audioset_loader() {
    runtime::SpecBackedVoiceModelConfig<Assets> config;
    config.family = kFamily;
    config.load_assets = load_assets;
    config.create_session = [](
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options,
        std::shared_ptr<const Assets> assets,
        std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<Session>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::ast_audioset
