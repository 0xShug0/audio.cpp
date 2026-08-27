#include "engine/models/midashenglm_gen/session.h"

#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::midashenglm_gen {
namespace {

constexpr const char * kFamily = "midashenglm_gen";
constexpr size_t kWeightContextBytes = 768ull * 1024ull * 1024ull;
constexpr size_t kSmallGraphContextBytes = 128ull * 1024ull * 1024ull;
constexpr size_t kLargeGraphContextBytes = 512ull * 1024ull * 1024ull;

std::shared_ptr<const MiDashengLmGenAssets> require_assets(
    std::shared_ptr<const MiDashengLmGenAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("MiDashengLM-Gen session requires assets");
    }
    return assets;
}

engine::assets::TensorStorageType weight_type(const engine::runtime::SessionOptions & options) {
    return engine::runtime::parse_tensor_storage_option(
        options.options,
        "midashenglm_gen.weight_type",
        engine::assets::TensorStorageType::Native,
        {
            engine::assets::TensorStorageType::Native,
            engine::assets::TensorStorageType::F32,
            engine::assets::TensorStorageType::F16,
            engine::assets::TensorStorageType::BF16,
            engine::assets::TensorStorageType::Q8_0,
        });
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("MiDashengLM-Gen session requires a model contract");
    }
    return contract;
}

int64_t positive_request_i64(
    const engine::runtime::TaskRequest & request,
    std::initializer_list<std::string_view> keys,
    int64_t fallback,
    const char * label) {
    const auto value = engine::runtime::parse_i64_option(request.options, keys).value_or(fallback);
    if (value <= 0) {
        throw std::runtime_error(std::string("MiDashengLM-Gen ") + label + " must be positive");
    }
    return value;
}

float finite_request_float(
    const engine::runtime::TaskRequest & request,
    std::initializer_list<std::string_view> keys,
    float fallback,
    const char * label) {
    const auto value = engine::runtime::parse_finite_float_option(request.options, keys).value_or(fallback);
    if (!std::isfinite(value)) {
        throw std::runtime_error(std::string("MiDashengLM-Gen ") + label + " must be finite");
    }
    return value;
}

int64_t frame_budget_from_duration(float duration_sec, const MiDashengLmGenConfig & config) {
    const double frames_per_second =
        static_cast<double>(config.sample_rate) / static_cast<double>(2 * config.istft_hop);
    return std::max<int64_t>(1, static_cast<int64_t>(std::ceil(static_cast<double>(duration_sec) * frames_per_second)));
}

std::unique_ptr<engine::runtime::IVoiceTaskSession> create_midashenglm_gen_session(
    const engine::runtime::TaskSpec & task,
    const engine::runtime::SessionOptions & options,
    std::shared_ptr<const MiDashengLmGenAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    return std::make_unique<MiDashengLmGenSession>(
        task,
        options,
        std::move(assets),
        std::move(contract));
}

}  // namespace

MiDashengLmGenSession::MiDashengLmGenSession(
    engine::runtime::TaskSpec task,
    engine::runtime::SessionOptions options,
    std::shared_ptr<const MiDashengLmGenAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : engine::runtime::RuntimeSessionBase(options),
      task_(task),
      options_(std::move(options)),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))) {
    engine::runtime::validate_spec_backed_session_options(options_, *contract_, kFamily, "MiDashengLM-Gen");
    if (task_.task != engine::runtime::VoiceTaskKind::AudioGeneration ||
        task_.mode != engine::runtime::RunMode::Offline) {
        throw std::runtime_error("MiDashengLM-Gen supports only offline gen");
    }
    execution_ = std::make_unique<engine::core::ExecutionContext>(options_.backend);
    const auto storage_type = weight_type(options_);
    tokenizer_ = std::make_unique<MiDashengLmGenTextTokenizer>(assets_);
    prompt_encoder_ = std::make_unique<MiDashengLmGenPromptEncoderRuntime>(
        assets_,
        *execution_,
        kSmallGraphContextBytes,
        kSmallGraphContextBytes,
        storage_type);
    flow_ = std::make_unique<MiDashengLmGenFlowRuntime>(
        assets_,
        *execution_,
        kLargeGraphContextBytes,
        kWeightContextBytes,
        storage_type);
    ar_ = std::make_unique<MiDashengLmGenARRuntime>(
        assets_,
        *execution_,
        *flow_,
        kLargeGraphContextBytes,
        kLargeGraphContextBytes,
        kSmallGraphContextBytes,
        kWeightContextBytes,
        storage_type);
    audio_tokenizer_ = std::make_unique<MiDashengLmGenAudioTokenizerRuntime>(
        assets_,
        *execution_,
        kLargeGraphContextBytes,
        kWeightContextBytes,
        storage_type,
        storage_type);
}

MiDashengLmGenSession::~MiDashengLmGenSession() = default;

std::string MiDashengLmGenSession::family() const {
    return kFamily;
}

engine::runtime::VoiceTaskKind MiDashengLmGenSession::task_kind() const {
    return task_.task;
}

engine::runtime::RunMode MiDashengLmGenSession::run_mode() const {
    return task_.mode;
}

void MiDashengLmGenSession::prepare(const engine::runtime::SessionPreparationRequest & request) {
    engine::runtime::validate_spec_backed_request_options(request.options, *contract_, "MiDashengLM-Gen");
    mark_prepared();
}

MiDashengLmGenGenerationOptions MiDashengLmGenSession::generation_options(
    const engine::runtime::TaskRequest & request) const {
    MiDashengLmGenGenerationOptions out;
    const float default_duration_sec =
        static_cast<float>(assets_->config.sequence_length * 2 * assets_->config.istft_hop) /
        static_cast<float>(assets_->config.sample_rate);
    const float duration_sec = engine::runtime::parse_positive_finite_float_option(
        request.options,
        {"duration_sec"}).value_or(default_duration_sec);
    out.seq_len = frame_budget_from_duration(duration_sec, assets_->config);
    out.eval_cfg = finite_request_float(request, {"guidance_scale"}, 2.0F, "guidance_scale");
    out.stop_threshold = finite_request_float(request, {"stop_threshold"}, 0.5F, "stop_threshold");
    out.min_stop_step = positive_request_i64(request, {"min_stop_step"}, 5, "min_stop_step");
    out.seed = engine::runtime::parse_u32_option(request.options, {"seed"}).value_or(0);
    return out;
}

engine::runtime::TaskResult MiDashengLmGenSession::run(const engine::runtime::TaskRequest & request) {
    const auto wall_start = std::chrono::steady_clock::now();
    require_prepared("MiDashengLM-Gen run");
    engine::runtime::validate_spec_backed_request_options(request.options, *contract_, "MiDashengLM-Gen");
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("MiDashengLM-Gen requires --text input");
    }
    const auto options = generation_options(request);
    const auto prompt = tokenizer_->encode_batch({request.text_input->text});
    prompt_encoder_->prepare(prompt.batch, prompt.tokens);
    auto prompt_state = prompt_encoder_->encode(prompt);
    const auto ar = ar_->generate(prompt_state, options);
    audio_tokenizer_->prepare_decode(ar.batch, ar.frames);
    auto decoded = audio_tokenizer_->decode({ar.latents, ar.batch, ar.frames, ar.dims});
    if (decoded.audio.empty()) {
        throw std::runtime_error("MiDashengLM-Gen audio tokenizer returned no audio");
    }
    auto audio = std::move(decoded.audio.front());
    const int64_t num_iter = ar.frames / assets_->config.patch_size;
    int64_t stop_step = num_iter;
    for (int64_t step = 0; step < num_iter; ++step) {
        if (ar.stop_probs[static_cast<size_t>(step)] > options.stop_threshold) {
            stop_step = std::max<int64_t>(step, options.min_stop_step);
            break;
        }
    }
    const size_t target_samples = static_cast<size_t>(
        std::min<int64_t>(
            static_cast<int64_t>(audio.samples.size()),
            std::max<int64_t>(1, stop_step) * assets_->config.block_size));
    audio.samples.resize(target_samples);
    engine::runtime::TaskResult result;
    result.audio_output = std::move(audio);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

std::shared_ptr<engine::runtime::IVoiceModelLoader> make_midashenglm_gen_loader() {
    engine::runtime::SpecBackedVoiceModelConfig<MiDashengLmGenAssets> config;
    config.family = kFamily;
    config.load_assets = load_midashenglm_gen_assets;
    config.create_session = create_midashenglm_gen_session;
    return engine::runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::midashenglm_gen
