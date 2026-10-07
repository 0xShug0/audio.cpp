#include "engine/models/hviske_asr_v6/session.h"

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

std::vector<std::pair<int64_t, int64_t>> plan_chunks(const std::vector<float> & samples) {
    const auto n = static_cast<int64_t>(samples.size());
    if (n <= 30 * 16000) return {{0, n}};
    // librosa.effects.split defaults: centered 2048-sample RMS, 512-sample hop, top_db=30.
    std::vector<double> energy(static_cast<size_t>(1 + n / 512));
    double sum = 0;
    for (int64_t j = 0; j < std::min<int64_t>(n, 1024); ++j) sum += samples[j] * samples[j];
    for (size_t i = 0; i < energy.size(); ++i) {
        energy[i] = sum / 2048;
        const auto center = static_cast<int64_t>(i) * 512;
        for (int64_t j = center - 1024; j < center - 512; ++j) {
            if (j >= 0 && j < n) sum -= samples[j] * samples[j];
        }
        for (int64_t j = center + 1024; j < center + 1536; ++j) {
            if (j < n) sum += samples[j] * samples[j];
        }
    }
    const auto threshold = std::max(1.0e-10, *std::max_element(energy.begin(), energy.end())) * 0.001;
    std::vector<std::pair<int64_t, int64_t>> voiced;
    int64_t start = -1;
    for (size_t i = 0; i <= energy.size(); ++i) {
        const bool active = i < energy.size() && std::max(1.0e-10, energy[i]) > threshold;
        if (active && start < 0) start = i * 512;
        if (!active && start >= 0) {
            voiced.emplace_back(start, std::min<int64_t>(n, i * 512));
            start = -1;
        }
    }
    std::vector<int64_t> cuts{0};
    for (size_t i = 1; i < voiced.size(); ++i) cuts.push_back((voiced[i - 1].second + voiced[i].first) / 2);
    cuts.push_back(n);
    std::vector<std::pair<int64_t, int64_t>> spans;
    start = 0;
    for (size_t i = 1; i < cuts.size(); ++i) {
        while (cuts[i] - start > 28 * 16000) {
            const auto limit = start + 28 * 16000;
            const auto upper = std::upper_bound(cuts.begin(), cuts.end(), limit);
            const auto cut = upper != cuts.begin() && *std::prev(upper) > start ? *std::prev(upper) : limit;
            spans.emplace_back(start, cut);
            start = cut;
        }
    }
    if (start < n) spans.emplace_back(start, n);
    return spans;
}

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
        for (const auto & [begin, end] : plan_chunks(mono)) {
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
