#include "engine/models/hviske_asr_v6/session.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/tokenizers/qwen_bpe_bundle.h"
#include "engine/models/hviske_asr_v6/decoder.h"
#include "engine/models/hviske_asr_v6/encoder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace engine::models::hviske_asr_v6 {
namespace {
constexpr const char * kFamily = "hviske_asr_v6";

std::string collapse_repeats(const std::string & text) {
    std::istringstream stream(text);
    std::vector<std::string> words;
    for (std::string word; stream >> word;) words.push_back(std::move(word));
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t size = 1; size <= 4; ++size) {
            std::vector<std::string> out;
            for (size_t i = 0; i < words.size();) {
                if (i + size > words.size()) {
                    out.insert(out.end(), words.begin() + i, words.end());
                    break;
                }
                size_t repeats = 1;
                while (i + (repeats + 1) * size <= words.size() &&
                    std::equal(words.begin() + i, words.begin() + i + size, words.begin() + i + repeats * size)) ++repeats;
                out.insert(out.end(), words.begin() + i, words.begin() + i + std::min<size_t>(3, repeats) * size);
                changed = changed || repeats > 3;
                i += repeats * size;
            }
            words = std::move(out);
        }
    }
    std::string out;
    for (const auto & word : words) {
        if (!out.empty()) out += ' ';
        out += word;
    }
    return out;
}

class HviskeV6Session final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    HviskeV6Session(const runtime::SessionOptions & options, std::shared_ptr<const HviskeV6Assets> assets,
        std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), assets_(std::move(assets)), contract_(std::move(contract)),
          tokenizer_(tokenizers::load_qwen_bpe_tokenizer(assets_->resources)), frontend_(assets_->config.encoder.mel_bins) {
        runtime::validate_spec_backed_session_options(options, *contract_, kFamily, "Hviske v6");
        const auto storage = assets::parse_tensor_storage_type(
            runtime::find_option(options.options, {"hviske_asr_v6.weight_type"}).value_or("native"));
        encoder_ = std::make_unique<HviskeV6WhisperRoPEEncoderRuntime>(assets_, execution_context(), storage);
        decoder_ = std::make_unique<HviskeV6Qwen3DecoderRuntime>(assets_, execution_context(), storage);
        assets_->weights->release_storage();
    }
    std::string family() const override { return kFamily; }
    runtime::VoiceTaskKind task_kind() const override { return runtime::VoiceTaskKind::Asr; }
    runtime::RunMode run_mode() const override { return runtime::RunMode::Offline; }
    void prepare(const runtime::SessionPreparationRequest & request) override {
        if (!request.audio) throw std::runtime_error("Hviske v6 requires audio preparation");
        mark_prepared();
    }
    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("Hviske v6 run");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "Hviske v6");
        if (!request.audio_input) throw std::runtime_error("Hviske v6 requires audio input");
        const auto started = std::chrono::steady_clock::now();
        const auto & audio = *request.audio_input;
        auto mono = audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
        if (audio.sample_rate != 16000) {
            audio::SoxrResampleOptions options;
            options.output_length_policy = audio::SoxrOutputLengthPolicy::ExactExpected;
            options.require_full_input = true;
            auto resampled = audio::try_resample_mono_soxr(mono, audio.sample_rate, 16000, options);
            if (!resampled) throw std::runtime_error("Hviske v6 needs SOXR for non-16-kHz input");
            mono = std::move(*resampled);
        }
        if (mono.empty()) mono.resize(8000, 0.0f);
        HviskeV6DecodingOptions options;
        options.max_tokens = runtime::parse_positive_i64_option(request.options, {"max_tokens"}, 440);
        options.num_beams = runtime::parse_positive_i64_option(request.options, {"num_beams"}, 2);
        options.length_penalty = runtime::parse_finite_float_option(request.options, {"length_penalty"}).value_or(1.0f);
        const auto cased = runtime::find_option(request.options, {"cased"});
        const auto punctuation = runtime::find_option(request.options, {"punctuated"});
        if (cased && punctuation) {
            const auto & c = assets_->config;
            options.control_tokens = {
                runtime::parse_bool_option(*cased, "cased") ? c.cased_token_id : c.nocase_token_id,
                runtime::parse_bool_option(*punctuation, "punctuated") ? c.punctuation_token_id : c.no_punctuation_token_id};
        }
        std::string text;
        const auto maximum = options.max_tokens;
        const auto chunk_mode = audio::parse_audio_chunk_mode(request.options);
        const double chunk_seconds = runtime::parse_finite_float_option(
            request.options, {"audio_chunk_duration_sec"}).value_or(28.0F);
        if (chunk_seconds < 0.001 || chunk_seconds > 30.0) {
            throw std::runtime_error("Hviske v6 audio_chunk_duration_sec must be between 0.001 and 30 seconds");
        }
        const auto chunk_samples = static_cast<int64_t>(std::llround(chunk_seconds * 16000));
        std::vector<runtime::TimeSpan> spans;
        if (chunk_mode == audio::AudioChunkMode::Auto || chunk_mode == audio::AudioChunkMode::Silence) {
            const auto trigger = chunk_mode == audio::AudioChunkMode::Auto ? 30 * 16000 : chunk_samples;
            spans = audio::plan_silence_audio_chunks(mono, {chunk_samples, trigger});
        } else if (chunk_mode == audio::AudioChunkMode::Fixed) {
            for (const auto & span : audio::plan_audio_chunks(mono.size(), {chunk_samples, chunk_samples})) {
                spans.push_back({span.output_start_sample, span.output_start_sample + span.valid_samples});
            }
        } else if (chunk_mode == audio::AudioChunkMode::None) {
            if (mono.size() > 30 * 16000) {
                throw std::runtime_error("Hviske v6 audio_chunk_mode=none requires audio of at most 30 seconds; use auto or fixed for longform");
            }
            spans.push_back({0, static_cast<int64_t>(mono.size())});
        } else {
            throw std::runtime_error("Hviske v6 audio_chunk_mode must be auto, silence, fixed, or none");
        }
        for (const auto & [begin, end] : spans) {
            std::vector<float> chunk(mono.begin() + begin, mono.begin() + end);
            options.max_tokens = std::min<int64_t>(maximum, (chunk.size() + 1999) / 2000 + 12);
            const auto features = frontend_.extract(chunk, execution_context().config().threads);
            const auto encoded = encoder_->encode(features);
            const auto ids = decoder_->generate(encoded, options);
            const auto part = collapse_repeats(tokenizer_->decode(ids, true));
            if (!part.empty()) {
                if (!text.empty()) text += ' ';
                text += part;
            }
        }
        runtime::TaskResult result;
        result.text_output = runtime::Transcript{text, "da"};
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }
private:
    std::shared_ptr<const HviskeV6Assets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::shared_ptr<tokenizers::LlamaBpeTokenizer> tokenizer_;
    HviskeV6WhisperFrontend frontend_;
    std::unique_ptr<HviskeV6WhisperRoPEEncoderRuntime> encoder_;
    std::unique_ptr<HviskeV6Qwen3DecoderRuntime> decoder_;
};
}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_hviske_asr_v6_loader() {
    runtime::SpecBackedVoiceModelConfig<HviskeV6Assets> config;
    config.family = kFamily;
    config.load_assets = load_hviske_v6_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const HviskeV6Assets> assets, std::shared_ptr<const model_spec::ModelContract> contract) {
        if (task.task != runtime::VoiceTaskKind::Asr || task.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("Hviske v6 supports offline ASR");
        }
        return std::make_unique<HviskeV6Session>(options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}
}  // namespace engine::models::hviske_asr_v6
