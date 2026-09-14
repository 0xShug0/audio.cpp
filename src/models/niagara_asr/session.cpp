#include "engine/models/niagara_asr/session.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/text.h"
#include "engine/framework/runtime/host_ops.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::models::niagara_asr {
namespace {

using Clock = std::chrono::steady_clock;
constexpr size_t kMiB = 1024ULL * 1024ULL;
constexpr size_t kDefaultGraphArenaBytes = 1024ULL * kMiB;
constexpr size_t kDefaultWeightContextBytes = 256ULL * kMiB;

const engine::model_spec::ModelContract & require_contract(
    const std::shared_ptr<const engine::model_spec::ModelContract> & contract) {
    if (contract == nullptr) {
        throw std::runtime_error("Niagara ASR session requires a model contract");
    }
    return *contract;
}

runtime::SessionOptions validate_session_setup(
    const runtime::TaskSpec & task,
    runtime::SessionOptions options,
    const engine::model_spec::ModelContract & contract) {
    if (task.task != runtime::VoiceTaskKind::Asr || task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Niagara ASR only supports offline ASR");
    }
    if (options.backend.type != core::BackendType::Cpu) {
        throw std::runtime_error("Niagara ASR CPU variants require --backend cpu");
    }
    for (const char * key : {"weight_type", "graph_arena_mb", "weight_context_mb"}) {
        if (options.options.find(key) != options.options.end()) {
            throw std::runtime_error(
                "Niagara ASR session option " + std::string(key) +
                " must use the normalized key niagara_asr." + key);
        }
    }
    runtime::validate_spec_backed_session_options(options, contract, "niagara_asr", "Niagara ASR");
    return options;
}

std::unordered_map<std::string, std::string> validate_request_options(
    std::unordered_map<std::string, std::string> options,
    const engine::model_spec::ModelContract & contract) {
    runtime::validate_spec_backed_request_options(options, contract, "Niagara ASR");
    const auto it = options.find("audio_chunk_mode");
    if (it != options.end() && it->second != "none") {
        throw std::runtime_error("Niagara ASR supports only audio_chunk_mode=none");
    }
    return options;
}

}  // namespace

NiagaraAsrSession::NiagaraAsrSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const NiagaraAsrAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(validate_session_setup(task, std::move(options), require_contract(contract))),
      task_(std::move(task)),
      assets_(std::move(assets)),
      contract_(std::move(contract)),
      frontend_(assets_),
      runtime_(
          assets_,
          execution_context(),
          runtime::parse_tensor_storage_option(
              RuntimeSessionBase::options().options,
              "niagara_asr.weight_type",
              engine::assets::TensorStorageType::Native,
              {engine::assets::TensorStorageType::Native,
               engine::assets::TensorStorageType::F32,
               engine::assets::TensorStorageType::F16,
               engine::assets::TensorStorageType::BF16,
               engine::assets::TensorStorageType::Q8_0,
               engine::assets::TensorStorageType::Q4_0,
               engine::assets::TensorStorageType::Q4_1,
               engine::assets::TensorStorageType::Q5_0,
               engine::assets::TensorStorageType::Q5_1,
               engine::assets::TensorStorageType::Q2_K,
               engine::assets::TensorStorageType::Q3_K,
               engine::assets::TensorStorageType::Q4_K,
               engine::assets::TensorStorageType::Q5_K,
               engine::assets::TensorStorageType::Q6_K}),
          runtime::parse_size_mb_option(
              RuntimeSessionBase::options().options,
              {"niagara_asr.graph_arena_mb"},
              kDefaultGraphArenaBytes),
          runtime::parse_size_mb_option(
              RuntimeSessionBase::options().options,
              {"niagara_asr.weight_context_mb"},
              kDefaultWeightContextBytes)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Niagara ASR session requires assets");
    }
}

NiagaraAsrSession::~NiagaraAsrSession() = default;

std::string NiagaraAsrSession::family() const {
    return "niagara_asr";
}

runtime::VoiceTaskKind NiagaraAsrSession::task_kind() const {
    return task_.task;
}

runtime::RunMode NiagaraAsrSession::run_mode() const {
    return task_.mode;
}

void NiagaraAsrSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void) request;
    mark_prepared();
}

std::string NiagaraAsrSession::decode(const NiagaraInferenceResult & inference) const {
    const auto decoded = runtime::CTCDecoder().compute(
        inference.logits,
        1,
        inference.frames,
        inference.vocab_size,
        static_cast<int32_t>(assets_->config.blank_id));
    std::vector<int32_t> ids;
    ids.reserve(static_cast<size_t>(inference.frames));
    for (int32_t id : decoded.values) {
        if (id != static_cast<int32_t>(assets_->config.blank_id) && id > 0) {
            ids.push_back(id);
        }
    }
    return engine::io::trim_ascii_whitespace(
        tokenizers::decode_sentencepiece(assets_->tokenizer_pieces, ids));
}

runtime::TaskResult NiagaraAsrSession::run(const runtime::TaskRequest & request) {
    require_prepared("Niagara ASR run()");
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("Niagara ASR run() requires audio_input");
    }
    (void) validate_request_options(request.options, require_contract(contract_));
    const auto wall_start = Clock::now();
    const auto features = frontend_.extract(*request.audio_input);
    const auto inference = runtime_.infer(features);
    runtime::TaskResult result;
    result.text_output = runtime::Transcript{decode(inference), "en"};
    engine::debug::timing_log_scalar("niagara.frontend_frames", static_cast<double>(features.frames));
    engine::debug::timing_log_scalar("niagara.model_ms", inference.elapsed_ms);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_niagara_asr_loader() {
    runtime::SpecBackedVoiceModelConfig<NiagaraAsrAssets> config;
    config.family = "niagara_asr";
    config.load_assets = [](const std::filesystem::path & model_path) {
        return load_niagara_asr_assets(model_path);
    };
    config.create_session =
        [](const runtime::TaskSpec & task,
           const runtime::SessionOptions & options,
           std::shared_ptr<const NiagaraAsrAssets> assets,
           std::shared_ptr<const engine::model_spec::ModelContract> contract) {
            return std::make_unique<NiagaraAsrSession>(
                task, options, std::move(assets), std::move(contract));
        };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::niagara_asr
