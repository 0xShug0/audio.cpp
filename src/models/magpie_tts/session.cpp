#include "engine/models/magpie_tts/session.h"

#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/debug/trace.h"
#include "engine/models/magpie_tts/request.h"

#include <stdexcept>
#include <utility>

namespace engine::models::magpie_tts {
namespace {

constexpr const char * kFamily = "magpie_tts";

std::shared_ptr<const MagpieTTSAssets> require_assets(std::shared_ptr<const MagpieTTSAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("MagpieTTS session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("MagpieTTS session requires a model contract");
    }
    return contract;
}

std::unique_ptr<runtime::IVoiceTaskSession> create_magpie_tts_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const MagpieTTSAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    return std::make_unique<MagpieTTSSession>(
        task,
        options,
        std::move(assets),
        std::move(contract));
}

}  // namespace

MagpieTTSSession::MagpieTTSSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const MagpieTTSAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))),
      tokenizer_(assets_->resources.model_root()) {
    runtime::validate_spec_backed_session_options(options, *contract_, kFamily, "MagpieTTS");
    graph_arena_bytes_ = runtime::parse_size_mb_option(
        options.options,
        {"magpie_tts.graph_arena_mb"},
        graph_arena_bytes_);
    weight_context_bytes_ = runtime::parse_size_mb_option(
        options.options,
        {"magpie_tts.weight_context_mb"},
        weight_context_bytes_);
    using T = engine::assets::TensorStorageType;
    matmul_weight_storage_type_ = runtime::parse_tensor_storage_option(
        options.options,
        "magpie_tts.weight_type",
        matmul_weight_storage_type_,
        {T::Native, T::F32, T::F16, T::BF16, T::Q8_0});
    conv_weight_storage_type_ = runtime::parse_tensor_storage_option(
        options.options,
        "magpie_tts.conv_weight_type",
        conv_weight_storage_type_,
        {T::Native, T::F32, T::F16});
    if (task_.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("MagpieTTS supports the Tts task");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("MagpieTTS currently supports offline sessions");
    }
    runtime_ = std::make_unique<MagpieTTSRuntime>(
        assets_,
        execution_context(),
        MagpieTTSRuntimeOptions{
            graph_arena_bytes_,
            weight_context_bytes_,
            matmul_weight_storage_type_,
            conv_weight_storage_type_,
        });
}

MagpieTTSSession::~MagpieTTSSession() = default;

std::string MagpieTTSSession::family() const {
    return kFamily;
}

runtime::VoiceTaskKind MagpieTTSSession::task_kind() const {
    return task_.task;
}

runtime::RunMode MagpieTTSSession::run_mode() const {
    return task_.mode;
}

void MagpieTTSSession::prepare(const runtime::SessionPreparationRequest & request) {
    runtime::validate_spec_backed_request_options(request.options, *contract_, "MagpieTTS");
    prepared_defaults_ = make_magpie_tts_prepare_defaults(*assets_, request);
    if (prepared_defaults_.has_value() && !prepared_defaults_->text.empty()) {
        (void) tokenizer_.tokenize(prepared_defaults_->text, prepared_defaults_->generation);
    }
    mark_prepared();
}

runtime::TaskResult MagpieTTSSession::run(const runtime::TaskRequest & request) {
    runtime::validate_spec_backed_request_options(request.options, *contract_, "MagpieTTS");
    require_prepared("MagpieTTS run");
    const auto parsed = make_magpie_tts_request(*assets_, request, prepared_defaults_);
    const auto tokenized = tokenizer_.tokenize(parsed.text, parsed.generation);
    engine::debug::trace_log_scalar("magpie_tts.tokenizer.name", tokenized.tokenizer_name);
    runtime::TaskResult result;
    result.audio_output = runtime_->synthesize(tokenized, parsed.generation);
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_magpie_tts_loader() {
    runtime::SpecBackedVoiceModelConfig<MagpieTTSAssets> config;
    config.family = kFamily;
    config.load_assets = load_magpie_tts_assets;
    config.create_session = create_magpie_tts_session;
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::magpie_tts
