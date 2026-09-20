#include "engine/community_models/auk/session.h"

#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/debug/profiler.h"
#include <yaml.h>

#include <cmath>
#include <algorithm>
#include <limits>
#include <string_view>
#include <utility>

namespace engine::models::auk {
namespace {

bool validate_auk_config(const std::filesystem::path & path) {
    const auto yaml = io::read_text_file(path);
    yaml_parser_t parser;
    if (!yaml_parser_initialize(&parser)) throw std::runtime_error("AuK YAML parser initialization failed");
    yaml_parser_set_input_string(&parser, reinterpret_cast<const unsigned char *>(yaml.data()), yaml.size());
    auto document = std::unique_ptr<yaml_document_t, void (*)(yaml_document_t *)>(
        new yaml_document_t{}, [](yaml_document_t * value) { yaml_document_delete(value); delete value; });
    if (!yaml_parser_load(&parser, document.get())) {
        const std::string error = parser.problem ? parser.problem : "unknown YAML parse error";
        yaml_parser_delete(&parser);
        throw std::runtime_error("AuK config: " + error);
    }
    yaml_parser_delete(&parser);
    // Query required scalar paths directly; unrelated nested lists remain intact.
    io::yaml::FlattenedDocument config;
    for (const std::string key : {"model.name", "model.backbone", "model.arch.dim", "model.arch.heads",
             "model.arch.ff_mult", "model.arch.text_hidden_dim", "model.arch.num_layers",
             "model.arch.num_single_layers", "model.vae.latent_dim", "model.vae.downsample_rate",
             "model.vae.target_sample_rate"}) {
        auto * node = yaml_document_get_root_node(document.get());
        size_t start = 0;
        while (start < key.size()) {
            const auto end = key.find('.', start);
            const auto part = key.substr(start, end == std::string::npos ? end : end - start);
            if (!node || node->type != YAML_MAPPING_NODE) throw std::runtime_error("AuK config missing mapping for " + key);
            yaml_node_t * child = nullptr;
            for (auto * pair = node->data.mapping.pairs.start; pair != node->data.mapping.pairs.top; ++pair) {
                const auto * name = yaml_document_get_node(document.get(), pair->key);
                if (name && name->type == YAML_SCALAR_NODE && part == std::string(
                        reinterpret_cast<const char *>(name->data.scalar.value), name->data.scalar.length)) {
                    if (child) throw std::runtime_error("duplicate AuK config key: " + key);
                    child = yaml_document_get_node(document.get(), pair->value);
                }
            }
            node = child;
            if (end == std::string::npos) break;
            start = end + 1;
        }
        if (!node || node->type != YAML_SCALAR_NODE) throw std::runtime_error("AuK config requires scalar " + key);
        config.scalars.emplace(key, std::string(reinterpret_cast<const char *>(node->data.scalar.value), node->data.scalar.length));
    }
    const auto model_name = io::yaml::require_string(config, "model.name");
    if ((model_name != "AuK" && model_name != "AuK-Flash") ||
        io::yaml::require_string(config, "model.backbone") != "Flux2Edit") {
        throw std::runtime_error("AuK loader requires a supported Flux2Edit checkpoint");
    }
    for (const auto & item : std::initializer_list<std::pair<const char *, int>>{
             {"model.arch.dim", 1536}, {"model.arch.heads", 24}, {"model.arch.ff_mult", 2},
             {"model.arch.text_hidden_dim", 2048}, {"model.arch.num_layers", 10},
             {"model.arch.num_single_layers", 20}, {"model.vae.latent_dim", 64},
             {"model.vae.downsample_rate", 480}, {"model.vae.target_sample_rate", 24000}}) {
        if (io::yaml::require_int(config, item.first) != item.second) {
            throw std::runtime_error(std::string("unsupported AuK configuration: ") + item.first);
        }
    }
    return model_name == "AuK-Flash";
}

std::shared_ptr<const AukAssets> load_auk_assets(const std::filesystem::path & path) {
    auto assets = std::make_shared<AukAssets>();
    assets->model_root = assets::prepare_model_directory(path).model_root;
    for (const char * name : {"config/auk-base.yaml", "config/auk-flash.yaml"}) {
        if (!io::is_existing_file(assets->model_root / name)) {
            throw std::runtime_error("missing AuK config: " + (assets->model_root / name).string());
        }
    }
    tokenizers::LlamaBpeTokenizerSpec tokenizer;
    tokenizer.tokenizer_json_path = assets->model_root / "tokenizer/tokenizer.json";
    tokenizer.tokenizer_config_path = assets->model_root / "tokenizer/tokenizer_config.json";
    tokenizer.pre_type = tokenizers::LlamaBpePreTokenizer::Qwen2;
    assets->tokenizer = tokenizers::load_llama_bpe_tokenizer(tokenizer);
    return assets;
}

std::filesystem::path resolve_component_gguf_path(
    const AukAssets & assets, std::string_view option_name, const std::string & value) {
    const std::filesystem::path relative(value);
    if (value.empty() || relative.is_absolute()) {
        throw std::runtime_error(std::string(option_name) + " must be a nonempty path relative to the AuK model root");
    }
    const auto path = assets.model_root / relative;
    if (!io::is_existing_file(path) || path.extension() != ".gguf") {
        throw std::runtime_error(std::string(option_name) + " must name an existing GGUF file: " + path.string());
    }
    return path;
}

std::shared_ptr<const AukAssets> select_component_assets(
    std::shared_ptr<const AukAssets> base, const runtime::SessionOptions & options) {
    auto selected = std::make_shared<AukAssets>(*base);
    const auto variant = runtime::find_option(options.options, {"auk.variant"}).value_or("base");
    if (variant != "base" && variant != "flash") {
        throw std::runtime_error("auk.variant must be base or flash");
    }
    selected->flash = variant == "flash";
    const auto config = selected->model_root / ("config/auk-" + variant + ".yaml");
    if (validate_auk_config(config) != selected->flash) {
        throw std::runtime_error("AuK component variant does not match its configuration");
    }
    const auto model_name = runtime::find_option(options.options, {"auk.model_gguf"})
        .value_or("auk-" + variant + "-f32.gguf");
    const auto qwen_name = runtime::find_option(options.options, {"auk.qwen_gguf"})
        .value_or("qwen2.5-omni-3b-bf16.gguf");
    const auto vae_name = runtime::find_option(options.options, {"auk.vae_gguf"})
        .value_or("auk-vae-f32.gguf");
    selected->model = assets::open_tensor_source(
        resolve_component_gguf_path(*selected, "auk.model_gguf", model_name));
    selected->qwen = assets::open_tensor_source(
        resolve_component_gguf_path(*selected, "auk.qwen_gguf", qwen_name));
    selected->vae = assets::open_tensor_source(
        resolve_component_gguf_path(*selected, "auk.vae_gguf", vae_name));
    assets::require_tensor_shape(*selected->model, "transformer.txt_proj.weight", {1536, 2048});
    assets::require_tensor_shape(*selected->qwen, "thinker.model.embed_tokens.weight", {151936, 2048});
    assets::require_tensor_shape(*selected->vae, "global_mean", {64});
    return selected;
}

}  // namespace

AukSession::AukSession(runtime::TaskSpec task, runtime::SessionOptions options,
                       std::shared_ptr<const AukAssets> assets,
                       std::shared_ptr<const model_spec::ModelContract> contract)
    : RuntimeSessionBase(options), task_(task), assets_(select_component_assets(std::move(assets), options)),
      contract_(std::move(contract)) {
    runtime::validate_spec_backed_session_options(options, *contract_, "auk", "AuK");
    if ((task.task != runtime::VoiceTaskKind::Tts && task.task != runtime::VoiceTaskKind::AudioGeneration) ||
        task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("AuK currently implements offline TTS and audio editing");
    }
    if (!assets_ || !assets_->model || !assets_->qwen || !assets_->vae || !assets_->tokenizer) {
        throw std::runtime_error("AuK requires model, Qwen, VAE, and tokenizer assets");
    }
    if (options.backend.type != core::BackendType::Cuda) {
        throw std::runtime_error("AuK native session currently requires CUDA");
    }
    if (const auto value = runtime::find_option(options.options, {"auk.mem_saver"})) {
        mem_saver_ = runtime::parse_bool_option(*value, "auk.mem_saver");
    }
}

AukSession::~AukSession() = default;
std::string AukSession::family() const { return "auk"; }
runtime::VoiceTaskKind AukSession::task_kind() const { return task_.task; }
runtime::RunMode AukSession::run_mode() const { return task_.mode; }

void AukSession::prepare(const runtime::SessionPreparationRequest & request) {
    runtime::validate_spec_backed_request_options(request.options, *contract_, "AuK");
    mark_prepared();
}

runtime::TaskResult AukSession::run(const runtime::TaskRequest & request) {
    require_prepared("AuK run");
    runtime::validate_spec_backed_request_options(request.options, *contract_, "AuK");
    const auto session_started = std::chrono::steady_clock::now();
    if (!request.text_input || request.text_input->text.empty()) {
        throw std::runtime_error("AuK requires a text instruction");
    }
    if (task_.task == runtime::VoiceTaskKind::AudioGeneration && !request.audio_input) {
        throw std::runtime_error("AuK audio editing requires source audio");
    }
    if (!request.input_artifacts.empty()) {
        throw std::runtime_error("AuK cached artifacts are not supported");
    }
    const runtime::AudioBuffer * reference_input = request.audio_input ? &*request.audio_input : nullptr;
    if (request.voice) {
        if (task_.task != runtime::VoiceTaskKind::Tts) {
            throw std::runtime_error("AuK audio editing takes its source in audio_input, not a voice reference");
        }
        if (request.voice->style || !request.voice->speaker || request.voice->speaker->cached_voice_id ||
            !request.voice->speaker->audio) {
            throw std::runtime_error("AuK voice conditioning requires speaker audio; cached voices and style fields are unsupported");
        }
        if (reference_input) throw std::runtime_error("AuK accepts only one reference: audio_input or speaker audio");
        reference_input = &*request.voice->speaker->audio;
    }
    const auto duration = runtime::find_option(request.options, {"duration_sec"});
    if (!duration && !reference_input) throw std::runtime_error("AuK no-reference synthesis requires an explicit duration");
    // Float option parsing can turn 0.1 seconds into six frames instead of five.
    size_t consumed = 0;
    const double seconds = duration ? std::stod(*duration, &consumed) : 0.0;
    if (duration && (consumed != duration->size() || !std::isfinite(seconds) || seconds <= 0.0)) {
        throw std::runtime_error("AuK duration must be a positive finite number");
    }
    const double frame_count = std::ceil(seconds * 24000.0 / 480.0);
    if (frame_count >= double(std::numeric_limits<int64_t>::max() / 480)) {
        throw std::runtime_error("AuK requested duration exceeds supported sample count");
    }
    auto frames = static_cast<int64_t>(frame_count);
    const int steps = assets_->flash ? 4 :
        runtime::parse_int_option(request.options, {"num_inference_steps"}).value_or(32);
    const float guidance = assets_->flash ? 0.0F :
        runtime::parse_finite_float_option(request.options, {"guidance_scale"}).value_or(2.0F);
    const float sway = runtime::parse_finite_float_option(request.options, {"sway_sampling_coef"}).value_or(-1.0F);
    if (steps <= 0) throw std::runtime_error("AuK num_inference_steps must be positive");
    const auto requested_seed = runtime::parse_i64_option(request.options, {"seed"}).value_or(-1);
    if (requested_seed < -1) throw std::runtime_error("AuK seed must be nonnegative or -1 for random");
    const uint64_t seed = requested_seed == -1 ? runtime::random_u64_seed() : static_cast<uint64_t>(requested_seed);
    std::string instruction = request.text_input->text;
    // Without a voice description, text carries the complete upstream instruction.
    if (const auto description = runtime::find_option(request.options, {"instruct"})) {
        if (task_.task != runtime::VoiceTaskKind::Tts) {
            throw std::runtime_error("AuK editing takes its instruction in text; instruct is for TTS");
        }
        instruction = "Generate speech based on the following description: \"" + *description +
            "\". The content to speak is: \"" + request.text_input->text + "\".";
    }
    auto & execution = execution_context();
    const auto policy = sampling::resolve_torch_cuda_sampling_policy(core::BackendType::Cuda,
        options().backend.device, "auk", "AuK");
    std::vector<float> audio_embeddings, reference_latents;
    int64_t reference_frames = 0, valid_reference_frames = 0;
    auto stage_started = std::chrono::steady_clock::now();
    if (reference_input) {
        const auto & audio = *reference_input;
        const auto features = extract_audio_features(audio.samples, audio.sample_rate, audio.channels, options().backend.threads);
        debug::timing_log_scalar("auk.stage.audio_frontend_ms", debug::elapsed_ms(stage_started));
        stage_started = std::chrono::steady_clock::now();
        if (!audio_conditioning_) {
            audio_conditioning_ = std::make_unique<AudioConditioningRuntime>(execution, *assets_->qwen, features.frames);
        } else {
            audio_conditioning_->prepare(features.frames);
        }
        audio_embeddings = audio_conditioning_->encode(features.values);
        debug::timing_log_scalar("auk.stage.audio_encoder_ms", debug::elapsed_ms(stage_started));
        if (mem_saver_) audio_conditioning_.reset();
        stage_started = std::chrono::steady_clock::now();
        audio::TorchaudioSincHannResampleOptions resampling;
        resampling.accumulation = audio::TorchaudioSincHannAccumulation::Float32;
        const auto reference_audio = audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
            audio.samples, audio.sample_rate, audio.channels, 24000, resampling);
        debug::timing_log_scalar("auk.stage.reference_resample_ms", debug::elapsed_ms(stage_started));
        stage_started = std::chrono::steady_clock::now();
        if (!reference_encoder_) {
            reference_encoder_ = std::make_unique<VaeEncoderRuntime>(execution, *assets_->vae, reference_audio.size());
        } else {
            reference_encoder_->prepare(reference_audio.size());
        }
        reference_frames = reference_encoder_->frames();
        valid_reference_frames = std::min<int64_t>(reference_audio.size() / 480, reference_frames);
        // Encoder and target noise use separate seed resets, as in the seeded Python baseline.
        const auto encoder_noise = sampling::generate_torch_cuda_tensor_iterator_randn(reference_frames * 64, seed, 0, policy);
        reference_latents = reference_encoder_->encode(reference_audio, encoder_noise);
        debug::timing_log_scalar("auk.stage.vae_encoder_ms", debug::elapsed_ms(stage_started));
        if (!duration) frames = std::max<int64_t>(1, valid_reference_frames);
    }
    stage_started = std::chrono::steady_clock::now();
    // CFMEdit clamps total duration before subtracting the valid reference prefix.
    frames = std::max<int64_t>(1, std::min<int64_t>(frames, 65536 - valid_reference_frames));
    const int64_t audio_tokens = audio_embeddings.size() / 2048;
    const auto input = prepare_conditioning(*assets_->tokenizer, instruction, audio_tokens);
    const auto attention = core::parse_attention_preference(
        runtime::find_option(options().options, {"auk.attention"}).value_or("auto"), "auk.attention");
    const bool use_flash_attention = core::resolve_flash_attention(execution.backend(), 64, attention);
    debug::trace_log_scalar("auk.attention.allow_flash", use_flash_attention);
    if (!conditioning_) {
        conditioning_ = std::make_unique<ConditioningRuntime>(execution, *assets_->qwen, *assets_->model,
            input.token_ids.size(), false, false, audio_tokens);
    } else {
        conditioning_->prepare(input.token_ids.size(), audio_tokens);
    }
    const auto prepare_generation = [&] {
        if (!flow_) {
            flow_ = std::make_unique<FlowRuntime>(execution, *assets_->model, frames, input.token_ids.size(), guidance >= 1e-5F,
                use_flash_attention, false, reference_frames, valid_reference_frames, steps);
        } else {
            flow_->prepare(frames, input.token_ids.size(), guidance >= 1e-5F, reference_frames, valid_reference_frames, steps);
        }
        if (!decoder_) {
            decoder_ = std::make_unique<VaeDecoderRuntime>(execution, *assets_->vae, frames);
        } else {
            decoder_->prepare(frames);
        }
    };
    if (!mem_saver_) prepare_generation();
    debug::timing_log_scalar("auk.stage.prepare_ms", debug::elapsed_ms(stage_started));
    stage_started = std::chrono::steady_clock::now();
    const auto conditioning = conditioning_->encode(input, audio_embeddings);
    debug::timing_log_scalar("auk.stage.text_encoder_ms", debug::elapsed_ms(stage_started));
    if (mem_saver_) {
        stage_started = std::chrono::steady_clock::now();
        conditioning_.reset();
        prepare_generation();
        debug::timing_log_scalar("auk.stage.prepare_ms", debug::elapsed_ms(stage_started));
    }
    stage_started = std::chrono::steady_clock::now();
    const auto noise = sampling::generate_torch_cuda_tensor_iterator_randn(frames * 64, seed, 0, policy);
    debug::timing_log_scalar("auk.stage.target_noise_ms", debug::elapsed_ms(stage_started));
    stage_started = std::chrono::steady_clock::now();
    const auto latents = flow_->sample(noise, conditioning, steps, sway, guidance, reference_latents, assets_->flash);
    debug::timing_log_scalar("auk.stage.flow_ms", debug::elapsed_ms(stage_started));
    stage_started = std::chrono::steady_clock::now();
    runtime::TaskResult result;
    result.audio_output = runtime::AudioBuffer{24000, 1, decoder_->decode(latents)};
    debug::timing_log_scalar("auk.stage.vae_decoder_ms", debug::elapsed_ms(stage_started));
    if (mem_saver_) {
        flow_.reset();
        decoder_.reset();
    }
    debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(session_started));
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_auk_loader() {
    class LoadedModel final : public runtime::ILoadedVoiceModel {
    public:
        explicit LoadedModel(std::shared_ptr<const AukAssets> assets)
            : assets_(std::move(assets)), contract_(runtime::require_model_contract("auk")) {}

        const runtime::ModelMetadata & metadata() const noexcept override { return contract_->metadata; }
        const runtime::CapabilitySet & capabilities() const noexcept override { return contract_->capabilities; }
        std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
            const runtime::TaskSpec & task, const runtime::SessionOptions & options) const override {
            return std::make_unique<AukSession>(task, options, assets_, contract_);
        }

    private:
        std::shared_ptr<const AukAssets> assets_;
        std::shared_ptr<const model_spec::ModelContract> contract_;
    };

    class Loader final : public runtime::IVoiceModelLoader {
    public:
        std::string family() const override { return "auk"; }
        bool can_load(const runtime::ModelLoadRequest & request) const override {
            if (request.family_hint && *request.family_hint != "auk") return false;
            try {
                (void) load_auk_assets(request.model_path);
                return true;
            } catch (const std::exception &) {
                return false;
            }
        }
        runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
            const auto assets = load_auk_assets(request.model_path);
            const auto contract = runtime::require_model_contract("auk");
            runtime::ModelInspection inspection;
            inspection.model_root = assets->model_root;
            inspection.metadata = contract->metadata;
            inspection.capabilities = contract->capabilities;
            inspection.cli = contract->cli;
            inspection.discovered_configs = runtime::discover_named_assets(
                inspection.model_root, inspection.metadata.config_candidates);
            inspection.discovered_weights = runtime::discover_named_assets(
                inspection.model_root, inspection.metadata.weight_candidates);
            return inspection;
        }
        std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
            return std::make_unique<LoadedModel>(load_auk_assets(request.model_path));
        }
        runtime::CapabilitySet advertised_capabilities() const override {
            return runtime::require_model_contract("auk")->capabilities;
        }
    };
    return std::make_shared<Loader>();
}

}  // namespace engine::models::auk
