#include "engine/community_models/glm_tts/session.h"

#include "engine/community_models/glm_tts/frontend.h"
#include "engine/community_models/glm_tts/prompt.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/binary.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/sampling/torch_random.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace engine::models::glm_tts {
namespace {

using Clock = std::chrono::steady_clock;
constexpr std::string_view kFamily = "glm_tts";
constexpr size_t kDefaultReferenceCacheSlots = 1;

std::shared_ptr<const GlmTTSAssets> require_assets(
    std::shared_ptr<const GlmTTSAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("GLM-TTS session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error(
            "GLM-TTS session requires a model contract");
    }
    return contract;
}

void validate_session_option_keys(
    const runtime::SessionOptions & options,
    const engine::model_spec::ModelContract & contract) {
    const std::string family_prefix = std::string(kFamily) + ".";
    for (const auto & [key, _] : options.options) {
        if (key.rfind(family_prefix, 0) == 0 &&
            contract.session_option_keys.find(key) ==
                contract.session_option_keys.end()) {
            throw std::runtime_error(
                "unknown GLM-TTS session option: " + key);
        }
    }
}

const runtime::AudioBuffer * reference_audio(
    const runtime::TaskRequest & request) {
    if (request.voice.has_value() &&
        request.voice->speaker.has_value() &&
        request.voice->speaker->audio.has_value()) {
        return &*request.voice->speaker->audio;
    }
    return request.audio_input.has_value()
        ? &*request.audio_input
        : nullptr;
}

assets::TensorStorageType requested_weight_type(
    const runtime::SessionOptions & options) {
    const auto value = runtime::find_option(
        options.options,
        {"glm_tts.weight_type", "weight_type"});
    return value.has_value()
        ? assets::parse_tensor_storage_type(*value)
        : assets::TensorStorageType::Native;
}

bool requested_mem_saver(
    const runtime::SessionOptions & options) {
    const auto value = runtime::find_option(
        options.options, {"glm_tts.mem_saver", "mem_saver"});
    return value.has_value()
        ? runtime::parse_bool_option(*value, "glm_tts.mem_saver")
        : false;
}

bool requested_aggressive_mem_saver(
    const runtime::SessionOptions & options) {
    const auto value = runtime::find_option(
        options.options,
        {"glm_tts.aggressive_mem_saver", "aggressive_mem_saver"});
    return value.has_value()
        ? runtime::parse_bool_option(
            *value, "glm_tts.aggressive_mem_saver")
        : false;
}

size_t requested_reference_cache_slots(
    const runtime::SessionOptions & options) {
    const int64_t slots = runtime::parse_i64_option(
        options.options,
        {"glm_tts.reference_cache_slots", "reference_cache_slots"})
        .value_or(static_cast<int64_t>(kDefaultReferenceCacheSlots));
    if (slots < 0) {
        throw std::runtime_error(
            "glm_tts.reference_cache_slots must be non-negative");
    }
    if (static_cast<uint64_t>(slots) >
        static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error(
            "glm_tts.reference_cache_slots is too large");
    }
    return static_cast<size_t>(slots);
}

uint64_t mix_reference_key(uint64_t key, uint64_t value) {
    key ^= value;
    key *= 1099511628211ull;
    return key;
}

uint64_t hash_audio_samples(const runtime::AudioBuffer & audio) {
    uint64_t key = 1469598103934665603ull;
    for (const float sample : audio.samples) {
        uint32_t bits = 0;
        std::memcpy(&bits, &sample, sizeof(bits));
        key = mix_reference_key(key, static_cast<uint64_t>(bits));
    }
    return key;
}

modules::HiftVocoderConfig hift_config(
    assets::TensorStorageType storage_type) {
    modules::HiftVocoderConfig out;
    out.in_channels = 80;
    out.base_channels = 512;
    out.nb_harmonics = 8;
    out.sampling_rate = 24000;
    out.nsf_alpha = 0.1F;
    out.nsf_sigma = 0.003F;
    out.nsf_voiced_threshold = 10.0F;
    out.upsample_rates = {8, 5, 3};
    out.upsample_kernel_sizes = {16, 11, 7};
    out.istft_n_fft = 16;
    out.istft_hop = 4;
    out.resblock_kernel_sizes = {3, 7, 11};
    out.resblock_dilation_sizes = {
        {1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    out.source_resblock_kernel_sizes = {7, 7, 11};
    out.source_resblock_dilation_sizes = {
        {1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    out.lrelu_slope = 0.1F;
    out.audio_limit = 0.99F;
    out.f0_num_class = 1;
    out.f0_in_channels = 80;
    out.f0_cond_channels = 512;
    out.weight_storage_type = storage_type;
    out.weight_layout =
        modules::HiftVocoderWeightLayout::
            TorchParametrizedWeightNorm;
    return out;
}

std::vector<float> normal_noise(
    size_t count,
    uint32_t seed) {
    return sampling::generate_torch_cuda_randn(
        count,
        seed,
        sampling::TorchRandnPrecision::Float32);
}

std::vector<float> optional_f32_values(
    const runtime::TaskRequest & request,
    std::initializer_list<std::string_view> keys,
    const char * label) {
    const auto path = runtime::find_option(request.options, keys);
    if (!path.has_value() || path->empty()) {
        return {};
    }
    auto values = io::read_f32_file(*path);
    if (values.empty()) {
        throw std::runtime_error(
            std::string("GLM-TTS ") + label +
            " file is empty");
    }
    return values;
}

std::vector<float> frame_major_to_channel_major(
    const std::vector<float> & input,
    int64_t frames,
    int64_t channels) {
    if (static_cast<int64_t>(input.size()) != frames * channels) {
        throw std::runtime_error(
            "GLM-TTS mel transpose shape mismatch");
    }
    std::vector<float> out(input.size(), 0.0F);
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int64_t channel = 0; channel < channels; ++channel) {
            out[static_cast<size_t>(channel * frames + frame)] =
                input[static_cast<size_t>(frame * channels + channel)];
        }
    }
    return out;
}

}  // namespace

GlmTTSSession::GlmTTSSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const GlmTTSAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))),
      text_tokenizer_(
          assets_->resources.require_file("tokenizer_vocab"),
          assets_->resources.require_file("tokenizer_merges"),
          assets_->resources.require_file("tokenizer_config")),
      weight_storage_type_(requested_weight_type(this->options())),
      reference_cache_(
          requested_reference_cache_slots(this->options())) {
    if ((task_.task != runtime::VoiceTaskKind::Tts &&
         task_.task != runtime::VoiceTaskKind::VoiceCloning) ||
        task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error(
            "GLM-TTS supports offline TTS and voice cloning only");
    }
    validate_session_option_keys(options, *contract_);
    aggressive_mem_saver_ =
        requested_aggressive_mem_saver(this->options());
    mem_saver_ =
        aggressive_mem_saver_ ||
        requested_mem_saver(this->options());
}

GlmTTSSession::~GlmTTSSession() = default;

bool GlmTTSSession::ReferenceCacheKeyEqual::operator()(
    const ReferenceCacheKey & lhs,
    const ReferenceCacheKey & rhs) const {
    return lhs.sample_rate == rhs.sample_rate &&
        lhs.channels == rhs.channels &&
        lhs.sample_count == rhs.sample_count &&
        lhs.sample_hash == rhs.sample_hash;
}

std::string GlmTTSSession::family() const {
    return std::string(kFamily);
}

runtime::VoiceTaskKind GlmTTSSession::task_kind() const {
    return task_.task;
}

runtime::RunMode GlmTTSSession::run_mode() const {
    return task_.mode;
}

void GlmTTSSession::prepare(
    const runtime::SessionPreparationRequest &) {
    mark_prepared();
}

GlmTTSSpeechTokenizer & GlmTTSSession::speech_tokenizer() {
    if (speech_tokenizer_ == nullptr) {
        speech_tokenizer_ =
            std::make_unique<GlmTTSSpeechTokenizer>(
                assets_,
                options().backend,
                weight_storage_type_);
    }
    return *speech_tokenizer_;
}

GlmTTSLlamaRuntime & GlmTTSSession::llama() {
    if (llama_ == nullptr) {
        llama_ = std::make_unique<GlmTTSLlamaRuntime>(
            assets_,
            options().backend.type,
            options().backend.device,
            options().backend.threads,
            weight_storage_type_,
            runtime::parse_size_mb_option(
                options().options,
                {"glm_tts.llama_weight_context_mb"},
                8ull * 1024ull * 1024ull * 1024ull),
            runtime::parse_size_mb_option(
                options().options,
                {"glm_tts.constant_context_mb"},
                256ull * 1024ull * 1024ull));
    }
    return *llama_;
}

GlmTTSFlowRuntime & GlmTTSSession::flow() {
    if (flow_ == nullptr) {
        flow_ = std::make_unique<GlmTTSFlowRuntime>(
            assets_->flow_weights,
            options().backend,
            weight_storage_type_,
            assets_->config.flow);
    }
    return *flow_;
}

modules::CampplusEncoderComponent & GlmTTSSession::campplus() {
    if (campplus_ == nullptr) {
        modules::CampplusEncoderConfig config;
        config.feat_dim = 80;
        config.embedding_size = 192;
        config.weight_storage_type = weight_storage_type_;
        config.normalize_partial_segment_by_full_length = true;
        campplus_ =
            std::make_unique<modules::CampplusEncoderComponent>(
                modules::CampplusEncoderComponent::
                    load_from_tensor_source(
                        assets_->campplus_weights,
                        options().backend,
                        config));
    }
    return *campplus_;
}

modules::HiftVocoderComponent & GlmTTSSession::hift() {
    if (hift_ == nullptr) {
        hift_ = std::make_unique<modules::HiftVocoderComponent>(
            modules::HiftVocoderComponent::load_from_tensor_source(
                assets_->hift_weights,
                options().backend,
                hift_config(weight_storage_type_)));
    }
    return *hift_;
}

const GlmTTSSession::ReferenceCacheEntry &
GlmTTSSession::resolve_reference(
    const runtime::AudioBuffer & audio) {
    ReferenceCacheKey key;
    key.sample_rate = audio.sample_rate;
    key.channels = audio.channels;
    key.sample_count = static_cast<uint64_t>(audio.samples.size());
    key.sample_hash = hash_audio_samples(audio);
    if (const auto * cached = reference_cache_.find(key)) {
        debug::trace_log_scalar("glm_tts.reference_cache.hit", 1);
        debug::trace_log_scalar(
            "glm_tts.reference_cache.slots",
            static_cast<int64_t>(reference_cache_.capacity()));
        debug::trace_log_scalar(
            "glm_tts.reference_cache.entries",
            static_cast<int64_t>(reference_cache_.size()));
        debug::trace_log_scalar(
            "glm_tts.reference_cache.evicted", 0);
        return *cached;
    }

    // A reference cache miss needs the large speech tokenizer and CAMPPlus
    // weights. Balanced mem-saver keeps the generation path warm only while a
    // cached reference can be reused; release it before preparing a new voice
    // so the two groups do not overlap in VRAM.
    if (mem_saver_) {
        llama_.reset();
        flow_.reset();
        hift_.reset();
    }

    const bool will_evict =
        reference_cache_.capacity() > 0 &&
        reference_cache_.size() >= reference_cache_.capacity();
    const auto started = Clock::now();
    ReferenceCacheEntry entry;
    entry.speech_tokens = speech_tokenizer().encode(audio);
    entry.prompt_mel = compute_glm_tts_prompt_mel(audio);
    const auto fbank = compute_glm_tts_campplus_fbank(audio);
    auto speaker = campplus().embed_from_features(
        fbank.values, fbank.frames, fbank.dims);
    entry.speaker_embedding = std::move(speaker.embedding);
    debug::timing_log_scalar(
        "glm_tts.reference_prepare_ms",
        debug::elapsed_ms(started));
    if (mem_saver_) {
        speech_tokenizer_.reset();
        campplus_.reset();
    }

    if (reference_cache_.capacity() == 0) {
        uncached_reference_ = std::move(entry);
    } else {
        reference_cache_.put(key, std::move(entry));
    }
    debug::trace_log_scalar("glm_tts.reference_cache.hit", 0);
    debug::trace_log_scalar(
        "glm_tts.reference_cache.slots",
        static_cast<int64_t>(reference_cache_.capacity()));
    debug::trace_log_scalar(
        "glm_tts.reference_cache.entries",
        static_cast<int64_t>(reference_cache_.size()));
    debug::trace_log_scalar(
        "glm_tts.reference_cache.evicted", will_evict ? 1 : 0);
    if (reference_cache_.capacity() == 0) {
        return *uncached_reference_;
    }
    const auto * cached = reference_cache_.find(key);
    if (cached == nullptr) {
        throw std::runtime_error(
            "GLM-TTS reference cache insert failed");
    }
    return *cached;
}

runtime::TaskResult GlmTTSSession::run(
    const runtime::TaskRequest & request) {
    require_prepared("GLM-TTS run");
    const auto wall_start = Clock::now();
    if (!request.text_input.has_value() ||
        request.text_input->text.empty()) {
        throw std::runtime_error("GLM-TTS requires text input");
    }
    const auto * audio = reference_audio(request);
    if (audio == nullptr) {
        throw std::runtime_error(
            "GLM-TTS requires --voice-ref reference audio");
    }
    const auto reference_text = runtime::find_option(
        request.options, {"reference_text"});
    if (!reference_text.has_value() || reference_text->empty()) {
        throw std::runtime_error(
            "GLM-TTS requires --reference-text matching --voice-ref");
    }
    const uint32_t seed = runtime::parse_u32_option(
        request.options, {"seed"}).value_or(0);

    const auto frontend_start = Clock::now();
    const auto & reference = resolve_reference(*audio);
    debug::timing_log_scalar(
        "glm_tts.frontend_ms",
        debug::elapsed_ms(frontend_start));

    const auto prompt = build_glm_tts_prompt(
        assets_->config,
        text_tokenizer_,
        *reference_text,
        request.text_input->text,
        reference.speech_tokens);
    GlmTTSGenerateOptions generation;
    generation.seed = seed;
    generation.max_new_tokens = runtime::parse_i64_option(
        request.options, {"max_tokens"}).value_or(0);
    generation.top_k = runtime::parse_i64_option(
        request.options, {"top_k"}).value_or(25);
    generation.top_p = runtime::parse_finite_float_option(
        request.options, {"top_p"}).value_or(0.8F);
    generation.temperature = runtime::parse_finite_float_option(
        request.options, {"temperature"}).value_or(1.0F);
    const auto llama_start = Clock::now();
    auto generated = llama().generate(prompt, generation);
    debug::timing_log_scalar(
        "glm_tts.llama_ms", debug::elapsed_ms(llama_start));
    if (aggressive_mem_saver_) {
        llama_.reset();
    }

    std::vector<int32_t> full_tokens = reference.speech_tokens;
    full_tokens.insert(
        full_tokens.end(),
        generated.speech_tokens.begin(),
        generated.speech_tokens.end());
    const int64_t flow_frames = static_cast<int64_t>(
        static_cast<double>(full_tokens.size()) /
        static_cast<double>(assets_->config.flow.input_frame_rate) *
        static_cast<double>(assets_->config.flow.mel_framerate));
    GlmTTSFlowInput flow_input;
    flow_input.speech_tokens = std::move(full_tokens);
    flow_input.prompt_mel = reference.prompt_mel.values;
    flow_input.prompt_frames = reference.prompt_mel.frames;
    flow_input.speaker_embedding = reference.speaker_embedding;
    flow_input.inference_steps = static_cast<int>(
        runtime::parse_i64_option(
            request.options, {"num_inference_steps", "flow_steps"})
            .value_or(assets_->config.flow.inference_steps));
    flow_input.cfg_rate = runtime::parse_finite_float_option(
        request.options, {"flow_guidance_scale", "cfg_rate"})
        .value_or(assets_->config.flow.inference_cfg_rate);
    const size_t flow_noise_count = static_cast<size_t>(
        flow_frames * assets_->config.flow.mel_dim);
    flow_input.initial_noise = optional_f32_values(
        request, {"flow_noise_path", "flow_noise_file"}, "Flow noise");
    if (flow_input.initial_noise.empty()) {
        flow_input.initial_noise =
            normal_noise(flow_noise_count, seed + 1);
    } else if (flow_input.initial_noise.size() != flow_noise_count) {
        throw std::runtime_error(
            "GLM-TTS Flow noise file must contain exactly " +
            std::to_string(flow_noise_count) + " float32 values");
    }
    const auto flow_start = Clock::now();
    auto mel = flow().generate(flow_input);
    debug::timing_log_scalar(
        "glm_tts.flow_ms", debug::elapsed_ms(flow_start));
    if (aggressive_mem_saver_) {
        flow_.reset();
    }

    const auto channel_major = frame_major_to_channel_major(
        mel.mel, mel.frames, assets_->config.flow.mel_dim);
    const auto hift_source_random = optional_f32_values(
        request,
        {"hift_source_random_path", "hift_source_random_file"},
        "HiFT source-random");
    const auto * hift_source_random_ptr =
        hift_source_random.empty()
            ? nullptr
            : &hift_source_random;
    const uint64_t hift_prior_noise_values =
        runtime::parse_u64_option(
            request.options,
            {"hift_prior_noise_count", "hift_prior_noise_values"})
            .value_or(0);
    const auto vocoder_start = Clock::now();
    auto waveform = hift().synthesize(
        channel_major,
        mel.frames,
        seed + 2,
        hift_prior_noise_values,
        hift_source_random_ptr);
    debug::timing_log_scalar(
        "glm_tts.hift_ms", debug::elapsed_ms(vocoder_start));
    if (aggressive_mem_saver_) {
        hift_.reset();
    }

    runtime::TaskResult result;
    runtime::AudioBuffer output;
    output.sample_rate = waveform.sample_rate;
    output.channels = 1;
    output.samples = std::move(waveform.waveform);
    result.audio_output = std::move(output);
    debug::trace_log_scalar(
        "glm_tts.reference_tokens",
        static_cast<int64_t>(reference.speech_tokens.size()));
    debug::trace_log_scalar(
        "glm_tts.generated_tokens",
        static_cast<int64_t>(generated.speech_tokens.size()));
    debug::trace_log_scalar("glm_tts.mel_frames", mel.frames);
    debug::timing_log_scalar(
        "session.wall_ms", debug::elapsed_ms(wall_start));
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_glm_tts_loader() {
    runtime::SpecBackedVoiceModelConfig<GlmTTSAssets> config;
    config.family = std::string(kFamily);
    config.load_assets = load_glm_tts_assets;
    config.create_session = [](
                                const runtime::TaskSpec & task,
                                const runtime::SessionOptions & options,
                                std::shared_ptr<const GlmTTSAssets> assets,
                                std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        return std::make_unique<GlmTTSSession>(
            task,
            options,
            std::move(assets),
            std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::glm_tts
