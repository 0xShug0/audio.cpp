#include "engine/models/bark_tts/session.h"

#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/models/bark_tts/generator.h"

#include <cmath>
#include <stdexcept>

namespace engine::models::bark_tts {
namespace {
constexpr const char * kFamily = "bark_tts";

engine::assets::TensorStorageType storage(const engine::runtime::SessionOptions & options, const char * name) {
    return engine::runtime::parse_tensor_storage_option(options.options, name,
        engine::assets::TensorStorageType::Native, {engine::assets::TensorStorageType::Native,
        engine::assets::TensorStorageType::F32, engine::assets::TensorStorageType::F16,
        engine::assets::TensorStorageType::BF16, engine::assets::TensorStorageType::Q8_0});
}

std::unique_ptr<engine::runtime::IVoiceTaskSession> create(const engine::runtime::TaskSpec & task,
    const engine::runtime::SessionOptions & options, std::shared_ptr<const BarkAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    return std::make_unique<BarkSession>(task, options, std::move(assets), std::move(contract));
}
}  // namespace

BarkSession::BarkSession(engine::runtime::TaskSpec task, engine::runtime::SessionOptions options,
                         std::shared_ptr<const BarkAssets> assets,
                         std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options), task_(task), options_(std::move(options)), assets_(std::move(assets)),
      contract_(std::move(contract)) {
    if (!assets_ || !contract_) throw std::runtime_error("Bark session requires assets and model contract");
    engine::runtime::validate_spec_backed_session_options(options_, *contract_, kFamily, "Bark");
    if (task_.task != engine::runtime::VoiceTaskKind::Tts || task_.mode != engine::runtime::RunMode::Offline)
        throw std::runtime_error("Bark supports offline TTS");
    execution_ = std::make_unique<engine::core::ExecutionContext>(options_.backend);
    generator_ = std::make_unique<BarkGenerator>(assets_, *execution_, storage(options_, "bark_tts.weight_type"),
                                                  storage(options_, "bark_tts.codec_weight_type"));
}
BarkSession::~BarkSession() = default;
std::string BarkSession::family() const { return kFamily; }
engine::runtime::VoiceTaskKind BarkSession::task_kind() const { return task_.task; }
engine::runtime::RunMode BarkSession::run_mode() const { return task_.mode; }
void BarkSession::prepare(const engine::runtime::SessionPreparationRequest & request) {
    engine::runtime::validate_spec_backed_request_options(request.options, *contract_, "Bark"); mark_prepared();
}
engine::runtime::TaskResult BarkSession::run(const engine::runtime::TaskRequest & request) {
    require_prepared("Bark run");
    engine::runtime::validate_spec_backed_request_options(request.options, *contract_, "Bark");
    if (!request.text_input.has_value() || request.text_input->text.empty()) throw std::runtime_error("Bark requires text_input");
    BarkGenerationOptions generation;
    generation.voice_id = engine::runtime::find_option(request.options, {"history_prompt"}).value_or(generation.voice_id);
    generation.temperature = engine::runtime::parse_float_option(request.options, {"temperature"}).value_or(generation.temperature);
    generation.top_k = engine::runtime::parse_int_option(request.options, {"top_k"}).value_or(generation.top_k);
    generation.max_tokens = engine::runtime::parse_i64_option(request.options, {"max_tokens"}).value_or(generation.max_tokens);
    generation.seed = engine::runtime::parse_u64_option(request.options, {"seed"}).value_or(generation.seed);
    if (!(generation.temperature > 0.0F) || !std::isfinite(generation.temperature) || generation.top_k <= 0 || generation.max_tokens <= 0)
        throw std::runtime_error("invalid Bark generation options");
    engine::runtime::TaskResult result;
    result.audio_output = engine::runtime::AudioBuffer{24000, 1, generator_->synthesize(request.text_input->text, generation)};
    return result;
}

std::shared_ptr<engine::runtime::IVoiceModelLoader> make_bark_tts_loader() {
    engine::runtime::SpecBackedVoiceModelConfig<BarkAssets> config;
    config.family = kFamily; config.load_assets = load_bark_assets; config.create_session = create;
    return engine::runtime::make_spec_backed_voice_loader(std::move(config));
}
}  // namespace engine::models::bark_tts
