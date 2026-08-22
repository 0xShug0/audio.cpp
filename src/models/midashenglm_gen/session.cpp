#include "engine/models/midashenglm_gen/session.h"

#include "engine/framework/io/filesystem.h"
#include "engine/framework/runtime/options.h"

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

engine::runtime::ModelMetadata make_metadata(const MiDashengLmGenAssets &) {
    engine::runtime::ModelMetadata out;
    out.family = kFamily;
    out.variant = "default";
    out.description = "MiDashengLM-Gen audio generation from local safetensors.";
    return out;
}

engine::runtime::CapabilitySet make_capabilities(const MiDashengLmGenAssets &) {
    engine::runtime::CapabilitySet out;
    out.supported_tasks = {
        {engine::runtime::VoiceTaskKind::AudioGeneration, {engine::runtime::RunMode::Offline}},
    };
    return out;
}

engine::runtime::ModelCliInterface cli() {
    engine::runtime::ModelCliInterface out;
    out.request_options = {
        {"duration_sec", "seconds", "Target audio duration budget."},
        {"guidance_scale", "scale", "Flow classifier-free guidance scale."},
        {"stop_threshold", "prob", "Stop probability threshold."},
        {"min_stop_step", "steps", "Minimum AR patch count before stop truncation."},
        {"seed", "n", "Generation seed."},
    };
    out.session_options = {
        {"midashenglm_gen.weight_type", "native|f32|f16|bf16|q8_0", "Shared weight storage type."},
    };
    return out;
}

class MiDashengLmGenLoadedModel final : public engine::runtime::ILoadedVoiceModel {
public:
    explicit MiDashengLmGenLoadedModel(std::shared_ptr<const MiDashengLmGenAssets> assets)
        : assets_(require_assets(std::move(assets))),
          metadata_(make_metadata(*assets_)),
          capabilities_(make_capabilities(*assets_)) {}

    const engine::runtime::ModelMetadata & metadata() const noexcept override {
        return metadata_;
    }

    const engine::runtime::CapabilitySet & capabilities() const noexcept override {
        return capabilities_;
    }

    std::unique_ptr<engine::runtime::IVoiceTaskSession> create_task_session(
        const engine::runtime::TaskSpec & task,
        const engine::runtime::SessionOptions & options) const override {
        return std::make_unique<MiDashengLmGenSession>(task, options, assets_);
    }

private:
    std::shared_ptr<const MiDashengLmGenAssets> assets_;
    engine::runtime::ModelMetadata metadata_;
    engine::runtime::CapabilitySet capabilities_;
};

class MiDashengLmGenLoader final : public engine::runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return kFamily;
    }

    engine::runtime::CapabilitySet advertised_capabilities() const override {
        engine::runtime::CapabilitySet out;
        out.supported_tasks = {
            {engine::runtime::VoiceTaskKind::AudioGeneration, {engine::runtime::RunMode::Offline}},
        };
        return out;
    }

    bool can_load(const engine::runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        return engine::io::is_existing_file(request.model_path / "config.json") &&
            engine::io::is_existing_file(request.model_path / "model.safetensors.index.json");
    }

    engine::runtime::ModelInspection inspect(const engine::runtime::ModelLoadRequest & request) const override {
        const auto assets = load_midashenglm_gen_assets(request.model_path);
        engine::runtime::ModelInspection out;
        out.model_root = assets->model_root;
        out.metadata = make_metadata(*assets);
        out.capabilities = make_capabilities(*assets);
        out.cli = cli();
        return out;
    }

    std::unique_ptr<engine::runtime::ILoadedVoiceModel> load(
        const engine::runtime::ModelLoadRequest & request) const override {
        return std::make_unique<MiDashengLmGenLoadedModel>(load_midashenglm_gen_assets(request.model_path));
    }
};

int64_t frame_budget_from_duration(float duration_sec, const MiDashengLmGenConfig & config) {
    const double frames_per_second =
        static_cast<double>(config.sample_rate) / static_cast<double>(2 * config.istft_hop);
    return std::max<int64_t>(1, static_cast<int64_t>(std::ceil(static_cast<double>(duration_sec) * frames_per_second)));
}

}  // namespace

MiDashengLmGenSession::MiDashengLmGenSession(
    engine::runtime::TaskSpec task,
    engine::runtime::SessionOptions options,
    std::shared_ptr<const MiDashengLmGenAssets> assets)
    : engine::runtime::RuntimeSessionBase(options),
      task_(task),
      options_(std::move(options)),
      assets_(require_assets(std::move(assets))) {
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
    (void)request;
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
    require_prepared("MiDashengLM-Gen run");
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
    return result;
}

std::shared_ptr<engine::runtime::IVoiceModelLoader> make_midashenglm_gen_loader() {
    return std::make_shared<MiDashengLmGenLoader>();
}

}  // namespace engine::models::midashenglm_gen
