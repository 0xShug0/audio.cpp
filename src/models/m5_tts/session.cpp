#include "engine/models/m5_tts/session.h"

#include "engine/models/m5_tts/ar.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/runtime/streaming_audio.h"
#include "engine/framework/runtime/streaming_tts_session.h"
#include "engine/framework/sampling/hf_sampler.h"
#include "engine/framework/text/chinese_normalization.h"
#include "engine/framework/text/chunking.h"
#include "engine/framework/text/text_normalization.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <map>

namespace engine::models::m5_tts {
namespace {

std::pair<std::string, std::string> language_condition(const std::string & language) {
    static const std::map<std::string, std::pair<std::string, std::string>> names = {
        {"", {"Chinese", ""}}, {"auto", {"Chinese", ""}}, {"zh", {"Chinese", ""}},
        {"Chinese", {"Chinese", ""}}, {"yue", {"Chinese", "yue"}},
        {"zh-Sichuan", {"Chinese", "chuan"}},
        {"en", {"English", ""}}, {"English", {"English", ""}},
        {"de", {"German", ""}}, {"German", {"German", ""}},
        {"fr", {"French", ""}}, {"French", {"French", ""}},
        {"ja", {"Japanese", ""}}, {"Japanese", {"Japanese", ""}},
        {"ko", {"Korean", ""}}, {"Korean", {"Korean", ""}}};
    const auto found = names.find(language);
    if (found == names.end()) throw runtime::InvalidRequestError("M5-TTS does not support language: " + language);
    return found->second;
}

std::string conditioned_text(const std::string & value, const std::string & speaker,
    const std::string & language, const std::string & accent, const std::string & instruction) {
    for (const auto * text : {&value, &speaker, &language, &accent, &instruction}) {
        if (text->find("<|") != std::string::npos)
            throw runtime::InvalidRequestError("M5-TTS text contains a reserved conditioning delimiter");
    }
    return "<|caption_begin|>" + instruction + "<|caption_end|><|accent_begin|>" + accent +
        "<|accent_end|><|speaker_start|>" + speaker + "<|speaker_end|><|lang_begin|>" + language + "<|lang_end|>" + value;
}

runtime::AudioBuffer prepare_reference(const runtime::AudioBuffer & input) {
    auto mono = audio::mixdown_interleaved_to_mono_average(input.samples, input.channels);
    mono = audio::resample_mono_torchaudio_sinc_hann(mono, input.sample_rate, 24000,
        audio::torchaudio_sinc_hann_float32_options());
    if (mono.empty() || !std::all_of(mono.begin(), mono.end(), [](float x) { return std::isfinite(x); }))
        throw runtime::InvalidRequestError("M5-TTS reference audio must be nonempty and finite");
    float peak = 0;
    for (const auto x : mono) peak = std::max(peak, std::abs(x));
    if (peak == 0) throw runtime::InvalidRequestError("M5-TTS reference audio is silent");
    for (auto & x : mono) x /= peak + 1.0e-8F;
    const float threshold = mono.size() > 240000 ? 0.02F : 0.015F;
    std::vector<size_t> speech;
    for (size_t i = 0; i < mono.size(); ++i) if (std::abs(mono[i]) > threshold) speech.push_back(i);
    if (speech.empty()) throw runtime::InvalidRequestError("M5-TTS reference contains no speech above the silence threshold");
    std::vector<std::pair<size_t, size_t>> spans;
    size_t start = speech.front();
    for (size_t i = 1; i < speech.size(); ++i) {
        if (speech[i] - speech[i - 1] - 1 >= 3600) {
            spans.emplace_back(start, speech[i - 1] + 1);
            start = speech[i];
        }
    }
    spans.emplace_back(start, mono.size() - speech.back() - 1 >= 3600 ? speech.back() + 1 : mono.size());
    std::vector<float> result;
    for (const auto & span : spans) {
        const auto begin = span.first > 1200 ? span.first - 1200 : size_t{0};
        const auto end = std::min(mono.size(), span.second + size_t{1200});
        result.insert(result.end(), mono.begin() + begin, mono.begin() + end);
    }
    return {24000, 1, std::move(result)};
}

class Session final : public runtime::RuntimeSessionBase,
    public runtime::IOfflineVoiceTaskSession, public runtime::StreamingTtsSessionBase {
public:
    Session(runtime::TaskSpec task, const runtime::SessionOptions & options,
        std::shared_ptr<const ModelAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)), contract_(std::move(contract)),
          tokenizer_({{}, {}, assets_->resources.require_file("tokenizer_config"),
              assets_->resources.require_file("tokenizer_json"), tokenizers::LlamaBpePreTokenizer::Llama3, {}, {}, true}) {
        runtime::validate_spec_backed_session_options(options, *contract_, "m5_tts", "M5-TTS");
        const auto storage = assets::parse_tensor_storage_type(
            runtime::find_option(options.options, {"m5_tts.weight_type", "weight_type"}).value_or("native"));
        codec_storage_ = assets::parse_tensor_storage_type(
            runtime::find_option(options.options, {"m5_tts.codec_weight_type", "codec_weight_type"}).value_or("native"));
        const auto attention = core::parse_attention_preference(
            runtime::find_option(options.options, {"m5_tts.attention", "attention"}).value_or("auto"), "m5_tts.attention");
        flash_ = core::resolve_flash_attention(execution_context().backend(), 64, attention);
        ar_ = std::make_unique<M5LlamaRuntime>(*assets_, execution_context(), storage, attention);
        decoder_ = std::make_unique<codecs::Qwen3SpeechTokenizerDecoderRuntime>(assets_->codec,
            assets_->codec_config, execution_context(), 16 * 1024 * 1024, 4 * 1024 * 1024,
            codec_storage_, codec_storage_, flash_, true);
    }

    std::string family() const override { return "m5_tts"; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }
    void prepare(const runtime::SessionPreparationRequest & request) override {
        prepared_voice_ = request.voice;
        mark_prepared();
    }
    runtime::TaskResult run(const runtime::TaskRequest & request) override { return synthesize(request, false); }

private:
    runtime::TaskResult generate_stream(const runtime::TaskRequest & request) override { return synthesize(request, true); }
    void reset_stream_state() override {
        stream_.reset();
        decoder_->reset_stream();
    }

    const codecs::Qwen3SpeechCodes & reference_codes(const runtime::AudioBuffer & audio) {
        auto prepared = prepare_reference(audio);
        if (!reference_ || reference_->first != prepared.samples) {
            if (!encoder_) encoder_ = std::make_unique<codecs::Qwen3SpeechTokenizerEncoderRuntime>(
                assets_->codec, execution_context(), 16 * 1024 * 1024,
                codec_storage_, codec_storage_, flash_);
            auto codes = encoder_->encode(prepared);
            reference_ = std::make_pair(std::move(prepared.samples), std::move(codes));
        }
        return reference_->second;
    }

    runtime::TaskResult synthesize(const runtime::TaskRequest & request, bool streaming) {
        require_prepared("M5-TTS run");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "M5-TTS");
        if (!request.text_input || request.text_input->text.empty())
            throw runtime::InvalidRequestError("M5-TTS requires text input");
        const auto started = std::chrono::steady_clock::now();
        const auto & voice = request.voice ? request.voice : prepared_voice_;
        const auto * reference_audio = voice && voice->speaker && voice->speaker->audio ? &*voice->speaker->audio : nullptr;
        const bool cloning = reference_audio != nullptr;
        if (cloning != (assets_->variant == "voice_clone"))
            throw runtime::InvalidRequestError("M5-TTS VoiceClone requires reference audio; CustomVoice requires a preset speaker");
        if (task_.task == runtime::VoiceTaskKind::VoiceCloning && !cloning)
            throw runtime::InvalidRequestError("M5-TTS cloning requires reference audio");
        const auto reference_text = runtime::find_option(request.options, {"reference_text"}).value_or("");
        if (cloning && reference_text.empty()) throw runtime::InvalidRequestError("M5-TTS cloning requires reference_text");
        auto language_tag = request.text_input->language;
        if (voice && voice->style && voice->style->language) language_tag = *voice->style->language;
        const auto [language, accent] = language_condition(language_tag);
        const auto instruction = runtime::find_option(request.options, {"instruct"}).value_or("");
        auto speaker = cloning ? std::string("None") : runtime::find_option(request.options, {"voice_id"}).value_or("1");
        if (!cloning && voice && voice->speaker && voice->speaker->cached_voice_id &&
            *voice->speaker->cached_voice_id != "default") speaker = *voice->speaker->cached_voice_id;
        const auto frames = runtime::parse_i64_option(request.options, {"max_tokens"}).value_or(500);
        const auto seed = runtime::parse_i64_option(request.options, {"seed"}).value_or(2026);
        std::mt19937 rng(seed == -1 ? static_cast<uint32_t>(runtime::random_u64_seed()) : static_cast<uint32_t>(seed));
        sampling::HfSamplingOptions sampling;
        sampling.temperature = runtime::parse_finite_float_option(request.options, {"temperature"}).value_or(0.9F);
        sampling.top_k = runtime::parse_i64_option(request.options, {"top_k"}).value_or(10);
        sampling::HfSampler sampler;
        sampling::HfSamplerScratch scratch;
        const auto chunks = runtime::chunk_text_request(request,
            text::parse_text_chunk_size_override(request.options).value_or(200),
            text::parse_text_chunk_mode_override(request.options).value_or(text::TextChunkMode::Default));
        const codecs::Qwen3SpeechCodes * reference = cloning ? &reference_codes(*reference_audio) : nullptr;
        runtime::AudioBuffer output{24000, 1, {}};
        for (const auto & chunk : chunks) {
            auto target = chunk.text_input->text;
            if (language == "Chinese") target = text::normalize_chinese_text(target);
            else if (language == "English") target = text::normalize_english_text(target);
            const auto text = conditioned_text((cloning ? reference_text : "") + target, speaker, language, accent, instruction);
            auto ids = tokenizer_.encode(text, true);
            ids.insert(ids.begin(), assets_->bos_token_id);
            ids.push_back(assets_->eos_token_id);
            auto prompt = ar_->text_embeddings(ids);
            if (reference) {
                for (int64_t i = 0; i < reference->frames; ++i) {
                    AudioFrame frame;
                    for (size_t k = 0; k < frame.size(); ++k) frame[k] = reference->codes[static_cast<size_t>(i) * 16 + k] + 1;
                    const auto embedded = ar_->audio_embedding(frame);
                    prompt.insert(prompt.end(), embedded.begin(), embedded.end());
                }
            }
            ar_->begin(prompt, frames);
            codecs::Qwen3SpeechCodes generated;
            generated.code_groups = 16;
            std::vector<float> pending;
            if (streaming) {
                runtime::StreamingAudioConfig config;
                config.frame_width = 16;
                config.frames_per_chunk = 4;
                stream_.begin(config, [this](const runtime::StreamingAudioWindow<int32_t> & window) {
                    codecs::Qwen3SpeechCodes codes;
                    codes.code_groups = 16;
                    codes.frames = static_cast<int64_t>(window.frames());
                    codes.codes.assign(window.data, window.data + window.frames() * 16);
                    return decoder_->decode_stream_chunk(codes, 0, 4);
                }, [&](const runtime::StreamEvent & event) {
                    pending.insert(pending.end(), event.audio_output->samples.begin(), event.audio_output->samples.end());
                    // Upstream removes the last 100 ms. Hold it until EOS rather
                    // than publishing samples that would subsequently be removed.
                    if (pending.size() > 2400) {
                        runtime::StreamEvent emitted;
                        emitted.audio_output = runtime::AudioBuffer{24000, 1,
                            std::vector<float>(pending.begin(), pending.end() - 2400)};
                        if (stream_event_sink()) stream_event_sink()(emitted);
                        pending.erase(pending.begin(), pending.end() - 2400);
                    }
                }, {}, [this, reference] {
                    decoder_->reset_stream();
                    if (reference) {
                        for (int64_t offset = 0; offset < reference->frames; offset += 4) {
                            const auto count = std::min<int64_t>(4, reference->frames - offset);
                            codecs::Qwen3SpeechCodes codes;
                            codes.code_groups = 16;
                            codes.frames = count;
                            codes.codes.assign(reference->codes.begin() + offset * 16,
                                reference->codes.begin() + (offset + count) * 16);
                            (void)decoder_->decode_stream_chunk(codes, 0, 4);
                        }
                    }
                });
            }
            bool eos = false;
            for (int64_t index = 0; index < frames; ++index) {
                const auto frame = ar_->next_frame([&](int64_t, int64_t, const std::vector<float> & logits, bool allow_eos) {
                    auto scores = logits;
                    if (!allow_eos) scores[0] = -1.0e5F;
                    return sampler.sample(scores, {}, sampling, scratch, rng, nullptr, "M5-TTS");
                });
                if (std::all_of(frame.begin(), frame.end(), [](int32_t x) { return x == 0; })) { eos = true; break; }
                std::vector<int32_t> codes;
                for (const auto token : frame) codes.push_back(std::clamp(token - 1, int32_t{0}, int32_t{2047}));
                generated.codes.insert(generated.codes.end(), codes.begin(), codes.end());
                ++generated.frames;
                if (streaming) stream_.push(codes);
            }
            if (!generated.frames) throw std::runtime_error("M5-TTS generated no audio frames");
            if (!eos) debug::log_message(debug::LogLevel::Warning, "M5-TTS reached max_tokens before EOS");
            runtime::AudioBuffer audio;
            if (streaming) audio = stream_.finish();
            else {
                codecs::Qwen3SpeechCodes codes = generated;
                if (reference) {
                    codes.codes.insert(codes.codes.begin(), reference->codes.begin(), reference->codes.end());
                    codes.frames += reference->frames;
                }
                audio = decoder_->decode(codes);
                if (reference) audio.samples.erase(audio.samples.begin(),
                    audio.samples.begin() + reference->frames * codecs::Qwen3SpeechTokenizerDecoderRuntime::kSamplesPerFrame);
            }
            if (audio.samples.size() <= 2400) throw std::runtime_error("M5-TTS generated audio is shorter than its tail trim");
            audio.samples.resize(audio.samples.size() - 2400);
            output.samples.insert(output.samples.end(), audio.samples.begin(), audio.samples.end());
        }
        runtime::TaskResult result;
        result.audio_output = std::move(output);
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }

    runtime::TaskSpec task_;
    std::shared_ptr<const ModelAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    tokenizers::LlamaBpeTokenizer tokenizer_;
    std::unique_ptr<M5LlamaRuntime> ar_;
    std::unique_ptr<codecs::Qwen3SpeechTokenizerEncoderRuntime> encoder_;
    std::unique_ptr<codecs::Qwen3SpeechTokenizerDecoderRuntime> decoder_;
    runtime::StreamingAudioController<int32_t> stream_;
    std::optional<std::pair<std::vector<float>, codecs::Qwen3SpeechCodes>> reference_;
    std::optional<runtime::VoiceCondition> prepared_voice_;
    assets::TensorStorageType codec_storage_ = assets::TensorStorageType::Native;
    bool flash_ = false;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_m5_tts_loader() {
    runtime::SpecBackedVoiceModelConfig<ModelAssets> config;
    config.family = "m5_tts";
    config.load_assets = load_assets;
    config.create_session = [](const auto & task, const auto & options, auto assets, auto contract) {
        if ((task.task != runtime::VoiceTaskKind::Tts && task.task != runtime::VoiceTaskKind::VoiceCloning) ||
            (task.mode != runtime::RunMode::Offline && task.mode != runtime::RunMode::Streaming))
            throw runtime::InvalidRequestError("M5-TTS supports offline/streaming tts and clon");
        return std::make_unique<Session>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::m5_tts
