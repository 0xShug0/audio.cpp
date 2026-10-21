#include "engine/models/ming3_audio/session.h"

#include "engine/models/ming3_audio/audio_encoder.h"
#include "engine/models/ming3_audio/backbone.h"
#include "engine/models/ming3_audio/frontend.h"
#include "engine/models/ming3_audio/talker.h"
#include "engine/models/ming3_audio/talker_frontend.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/reference_audio_frontend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/speaker_encoders/campplus_encoder.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/framework/text/chunking.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <optional>

namespace engine::models::ming3_audio {
namespace {

struct PreparedReference {
    runtime::AudioBuffer audio;
    std::vector<float> speaker;
    std::vector<float> latents;
    uint64_t seed = 0;
    uint64_t posterior_offset = 0;
};

class MingAudioSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    MingAudioSession(runtime::TaskSpec task, const runtime::SessionOptions & options,
                     std::shared_ptr<const MingAudioAssets> assets,
                     std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)), contract_(std::move(contract)) {
        runtime::validate_spec_backed_session_options(options, *contract_, "ming3_audio", "Ming3 Audio");
        if (task.mode != runtime::RunMode::Offline) throw std::runtime_error("Ming3 Audio supports offline sessions");
        thinker_options_.disk_experts = runtime::parse_bool_option(runtime::find_option(options.options,
            {"ming3_audio.disk_experts"}).value_or("false"), "ming3_audio.disk_experts");
        thinker_options_.pin_experts = runtime::parse_bool_option(runtime::find_option(options.options,
            {"ming3_audio.pin_experts"}).value_or("true"), "ming3_audio.pin_experts");
        thinker_options_.expert_cache_size = runtime::parse_positive_i64_option(options.options,
            {"ming3_audio.expert_cache_size"}, 8);
        if (task.task == runtime::VoiceTaskKind::Asr || task.task == runtime::VoiceTaskKind::SpeechToSpeech) {
            frontend_ = std::make_unique<Frontend>(assets_);
            if (!thinker_options_.disk_experts) encoder_ = std::make_unique<AudioEncoder>(execution_context(), assets_->thinker_weights,
                assets_->config.audio, assets_->config.thinker.hidden_size);
            thinker_ = std::make_unique<Thinker>(execution_context(), assets_->thinker_weights,
                assets_->config.thinker, 256, thinker_options_);
        }
        if (task.task != runtime::VoiceTaskKind::Asr) {
            if (task.task != runtime::VoiceTaskKind::SpeechToSpeech && task.task != runtime::VoiceTaskKind::Tts &&
                task.task != runtime::VoiceTaskKind::VoiceCloning) throw std::runtime_error("Ming3 Audio task is not supported");
            talker_frontend_ = std::make_unique<TalkerFrontend>(*assets_);
            if (!thinker_options_.disk_experts || task.task != runtime::VoiceTaskKind::SpeechToSpeech) {
                talker_ = std::make_unique<Talker>(execution_context(), assets_->talker_weights, assets_->talker);
                decoder_ = std::make_unique<modules::MingAudioDecoder>(assets_->vae_weights, assets_->talker.codec, execution_context());
            }
        }
        reference_cache_ = runtime::parse_bool_option(runtime::find_option(options.options,
            {"ming3_audio.reference_cache"}).value_or("true"), "ming3_audio.reference_cache");
    }

    std::string family() const override { return "ming3_audio"; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }
    void prepare(const runtime::SessionPreparationRequest &) override { mark_prepared(); }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("Ming3 Audio run");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "Ming3 Audio");
        const auto start = std::chrono::steady_clock::now();
        runtime::TaskResult result;
        if (thinker_) {
            result.text_output = runtime::Transcript{transcribe(request), ""};
        }
        if (talker_frontend_) {
            const std::string text = result.text_output ? result.text_output->text :
                request.text_input ? request.text_input->text : "";
            if (text.empty()) throw std::runtime_error("Ming3 Audio Talker requires nonempty text");
            result.audio_output = runtime::AudioBuffer{assets_->talker.codec.sample_rate, 1, synthesize(text, request)};
        }
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(start));
        return result;
    }

private:
    std::string transcribe(const runtime::TaskRequest & request) {
        if (!request.audio_input) throw std::runtime_error("Ming3 Audio requires input audio");
        if (thinker_options_.disk_experts && task_.task == runtime::VoiceTaskKind::SpeechToSpeech) {
            decoder_.reset();
            talker_.reset();
            reference_encoder_.reset();
        }
        const auto max_tokens = runtime::parse_positive_i64_option(request.options, {"max_tokens"}, 256);
        const auto prompt = runtime::find_option(request.options, {"prompt"}).value_or(
            request.text_input && !request.text_input->text.empty() ? request.text_input->text :
            task_.task == runtime::VoiceTaskKind::SpeechToSpeech ? "Please answer the question in the audio." :
                "Please recognize the language of this speech and transcribe it, placing a `\t` between the identified language and the transcribe.");
        auto stage = std::chrono::steady_clock::now();
        const auto features = frontend_->extract(*request.audio_input, options().backend.threads);
        const auto system = runtime::find_option(request.options, {"system_prompt"}).value_or("");
        const bool thinking = runtime::parse_bool_option(runtime::find_option(request.options,
            {"enable_thinking"}).value_or("false"), "enable_thinking");
        const auto ids = frontend_->prompt(prompt, features.frames, system, thinking);
        debug::timing_log_scalar("ming3_audio.frontend_ms", debug::elapsed_ms(stage));
        thinker_->reserve(thinker_options_.disk_experts
            ? runtime::bucketed_cache_steps(ids.size(), max_tokens)
            : ids.size() + max_tokens);
        if (!encoder_) encoder_ = std::make_unique<AudioEncoder>(execution_context(), assets_->thinker_weights,
            assets_->config.audio, assets_->config.thinker.hidden_size);
        stage = std::chrono::steady_clock::now();
        const auto audio = encoder_->encode(features.log_mel, features.frames);
        debug::timing_log_scalar("ming3_audio.encoder_ms", debug::elapsed_ms(stage));
        if (thinker_options_.disk_experts) encoder_.reset();
        stage = std::chrono::steady_clock::now();
        thinker_->reset();
        auto logits = thinker_->prefill(ids, audio, assets_->config.audio.token_id);
        std::vector<int32_t> tokens;
        for (int64_t i = 0; i < max_tokens; ++i) {
            const auto token = static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
            if (std::find(assets_->eos_token_ids.begin(), assets_->eos_token_ids.end(), token) != assets_->eos_token_ids.end()) break;
            tokens.push_back(token);
            if (i + 1 < max_tokens) logits = thinker_->step(token);
        }
        debug::trace_log_scalar("ming3_audio.thinker.generated_tokens", tokens.size());
        debug::trace_log_scalar("ming3_audio.thinker.hit_token_limit", tokens.size() == static_cast<size_t>(max_tokens));
        debug::timing_log_scalar("ming3_audio.decoder_ms", debug::elapsed_ms(stage));
        auto text = frontend_->decode(tokens);
        if (thinker_options_.disk_experts && task_.task == runtime::VoiceTaskKind::SpeechToSpeech) thinker_->release_workspace();
        return text;
    }

    const PreparedReference & prepare_reference(const runtime::AudioBuffer & audio, bool continuation,
                                                uint64_t seed, uint64_t & offset) {
        if (!reference_cache_ || !reference_ || reference_->audio.sample_rate != audio.sample_rate ||
            reference_->audio.channels != audio.channels || reference_->audio.samples != audio.samples) {
            if (!speaker_) {
                modules::CampplusEncoderConfig config;
                config.weight_layout = modules::CampplusEncoderWeightLayout::Fused;
                config.normalize_partial_segment_by_full_length = true;
                speaker_ = std::make_unique<modules::CampplusEncoderComponent>(modules::CampplusEncoderComponent::load_from_tensor_source(
                    assets_->speaker_weights, options().backend, config));
            }
            const auto features = audio::ReferenceAudioFrontend::campplus_fbank_16k(
                audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
                    audio.samples, audio.sample_rate, audio.channels, 16000));
            PreparedReference reference;
            reference.audio = audio;
            reference.speaker = speaker_->embed_from_features(features.values, features.frames, features.feature_dim).embedding;
            reference_ = std::move(reference);
        }
        if (continuation) {
            if (reference_->latents.empty() || reference_->seed != seed) {
                if (!reference_encoder_) reference_encoder_ = std::make_unique<modules::MingAudioEncoder>(
                    assets_->vae_weights, assets_->talker.codec, execution_context());
                auto waveform = audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
                    audio.samples, audio.sample_rate, audio.channels, assets_->talker.codec.sample_rate);
                const auto alignment = assets_->talker.codec.encoder.hop_size * assets_->talker.codec.patch_size;
                const auto padding = (alignment - waveform.size() % alignment) % alignment;
                waveform.insert(waveform.begin(), padding, 0.0f);
                const auto policy = sampling::resolve_torch_cuda_sampling_policy(execution_context().backend_type(),
                    options().backend.device, "ming3_audio", "Ming3 Audio reference");
                const auto count = reference_encoder_->latent_frames(waveform.size()) * assets_->talker.codec.encoder.latent_dim;
                const auto noise = sampling::generate_torch_cuda_tensor_iterator_randn(count, seed, offset, policy,
                    sampling::TorchRandnPrecision::BFloat16);
                reference_->posterior_offset = sampling::torch_cuda_tensor_iterator_offset_blocks(count, policy);
                reference_->latents = reference_encoder_->encode(waveform, noise);
                reference_->seed = seed;
            }
            offset += reference_->posterior_offset;
        }
        return *reference_;
    }

    std::vector<float> synthesize(std::string text, const runtime::TaskRequest & request) {
        if (!talker_) talker_ = std::make_unique<Talker>(execution_context(), assets_->talker_weights, assets_->talker);
        if (!decoder_) decoder_ = std::make_unique<modules::MingAudioDecoder>(assets_->vae_weights,
            assets_->talker.codec, execution_context());
        const auto & opts = request.options;
        TalkerGenerationOptions generation;
        generation.seed = runtime::parse_u64_option(opts, {"seed"}).value_or(42);
        generation.max_tokens = runtime::parse_positive_i64_option(opts, {"max_audio_tokens"}, 1000);
        generation.guidance_scale = runtime::parse_finite_float_option(opts, {"guidance_scale"}).value_or(2.0f);
        generation.sigma = runtime::parse_finite_float_option(opts, {"sigma"}).value_or(0.25f);
        generation.temperature = runtime::parse_finite_float_option(opts, {"temperature"}).value_or(0.0f);
        uint64_t offset = 0;
        const auto language = runtime::find_option(opts, {"language"}).value_or("");
        const auto voice = runtime::find_option(opts, {"voice_id"}).value_or(u8"\u7075\u5c0f\u67d4");
        const auto transcript = runtime::find_option(opts, {"reference_text"}).value_or("");
        const PreparedReference * reference = nullptr;
        if (request.voice && request.voice->speaker && request.voice->speaker->audio) {
            reference = &prepare_reference(*request.voice->speaker->audio, !transcript.empty(), generation.seed, offset);
        } else if (!transcript.empty()) throw std::runtime_error("Ming3 Audio reference_text requires voice_ref audio");
        const auto label = text.find('\t');
        if (label != std::string::npos && label > 0 && label <= 16 &&
            std::all_of(text.begin(), text.begin() + label, [](unsigned char c) { return std::isalpha(c); })) text.erase(0, label + 1);
        const bool raw = runtime::parse_bool_option(runtime::find_option(opts, {"raw_text"}).value_or("false"), "raw_text");
        const bool trim = runtime::parse_bool_option(runtime::find_option(opts, {"trim_silence"}).value_or("true"), "trim_silence");
        const auto chunks = raw ? std::vector<std::string>{text} : text::split_text_chunks(text,
            text::parse_text_chunk_size_override(opts).value_or(300),
            text::parse_text_chunk_mode_override(opts).value_or(text::TextChunkMode::TagAware));
        const auto latent_dim = assets_->talker.flow.latent_dim;
        const std::vector<float> speaker = reference ? reference->speaker : std::vector<float>{};
        const std::vector<float> latents = reference && !transcript.empty() ? reference->latents : std::vector<float>{};
        std::vector<float> history;
        if (!latents.empty()) history.assign(latents.end() - latent_dim, latents.end());
        std::vector<float> waveform;
        for (const auto & chunk : chunks) {
            const auto prompt = talker_frontend_->build(chunk, voice, language, reference != nullptr, transcript, latents.size() / latent_dim);
            const auto embeddings = talker_->prepare_inputs(prompt, speaker, latents);
            const auto generated = talker_->generate(embeddings, generation, offset, history);
            auto decoded = decoder_->decode(generated);
            if (trim) {
                const size_t frame = assets_->talker.codec.sample_rate / 10;
                const size_t tail = assets_->talker.codec.sample_rate * 3 / 10;
                const size_t frames = decoded.size() / frame;
                size_t last = frames;
                while (last > 0) {
                    double energy = 0;
                    for (size_t i = (last - 1) * frame; i < last * frame; ++i) energy += std::abs(decoded[i]);
                    if (energy / frame > 1e-3) break;
                    --last;
                }
                decoded.resize(std::min(frames ? frames * frame : decoded.size(), last * frame + tail));
            }
            waveform.insert(waveform.end(), decoded.begin(), decoded.end());
        }
        return waveform;
    }

    runtime::TaskSpec task_;
    std::shared_ptr<const MingAudioAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::unique_ptr<Frontend> frontend_;
    std::unique_ptr<AudioEncoder> encoder_;
    std::unique_ptr<Thinker> thinker_;
    std::unique_ptr<TalkerFrontend> talker_frontend_;
    std::unique_ptr<Talker> talker_;
    std::unique_ptr<modules::MingAudioDecoder> decoder_;
    std::unique_ptr<modules::MingAudioEncoder> reference_encoder_;
    std::unique_ptr<modules::CampplusEncoderComponent> speaker_;
    std::optional<PreparedReference> reference_;
    bool reference_cache_ = true;
    ThinkerOptions thinker_options_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_ming3_audio_loader() {
    runtime::SpecBackedVoiceModelConfig<MingAudioAssets> config;
    config.family = "ming3_audio";
    config.load_assets = load_ming3_audio_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const MingAudioAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<MingAudioSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::ming3_audio
