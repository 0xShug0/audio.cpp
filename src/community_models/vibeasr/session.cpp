#include "engine/community_models/vibeasr/session.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/text.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::vibeasr {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kWeightContextBytes = 64ull * 1024ull * 1024ull;

// VibeASR resamples to 24 kHz and RMS-normalizes to -25 dBFS before the encoder;
// both numbers are fixed in the reference implementation, not in the checkpoint.
constexpr int kSampleRate = 24000;
constexpr float kTargetDbFs = -25.0F;
constexpr float kNormalizeEps = 1.0e-6F;

// Canonical HuggingFace ids for the VibeVoice special tokens. VibeASR inserts
// them numerically rather than through the tokenizer, because the GGUF vocab's
// text for these slots is Qwen2.5's original <|object_ref_start|> family while
// the embedding rows are the ones VibeVoice trained.
constexpr int32_t kEndOfText = 151643;
constexpr int32_t kImStart = 151644;
constexpr int32_t kImEnd = 151645;
constexpr int32_t kSpeechStart = 151646;
constexpr int32_t kSpeechEnd = 151647;
constexpr int32_t kSpeechPad = 151648;

constexpr const char * kSystemPrompt =
    "You are a helpful assistant that transcribes audio input into text output in JSON format.";

std::shared_ptr<const VibeASRAssets> require_assets(std::shared_ptr<const VibeASRAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("VibeASR session requires assets");
    }
    return assets;
}

const engine::model_spec::ModelContract & require_contract(
    const std::shared_ptr<const engine::model_spec::ModelContract> & contract) {
    if (contract == nullptr) {
        throw std::runtime_error("VibeASR session requires a model contract");
    }
    return *contract;
}

runtime::SessionOptions validate_session_setup(
    const runtime::TaskSpec & task,
    runtime::SessionOptions options,
    const engine::model_spec::ModelContract & contract) {
    if (task.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("VibeASR only supports VoiceTaskKind::Asr");
    }
    if (task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("VibeASR only supports offline sessions");
    }
    runtime::validate_spec_backed_session_options(options, contract, "vibeasr", "VibeASR");
    return options;
}

size_t encoder_graph_arena_bytes(const runtime::SessionOptions & options) {
    return runtime::parse_size_mb_option(
        options.options, {"vibeasr.encoder_graph_arena_mb"}, 64ull * 1024ull * 1024ull);
}

size_t prefill_graph_arena_bytes(const runtime::SessionOptions & options) {
    return runtime::parse_size_mb_option(
        options.options, {"vibeasr.prefill_graph_arena_mb"}, 256ull * 1024ull * 1024ull);
}

size_t decode_graph_arena_bytes(const runtime::SessionOptions & options) {
    return runtime::parse_size_mb_option(
        options.options, {"vibeasr.decode_graph_arena_mb"}, 256ull * 1024ull * 1024ull);
}

std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> load_tokenizer(const VibeASRAssets & assets) {
    // No merges.txt in the published package, so the tokenizer comes from
    // tokenizer.json alone.
    return engine::tokenizers::load_llama_bpe_tokenizer(engine::tokenizers::LlamaBpeTokenizerSpec{
        {},
        {},
        assets.resources.require_file("tokenizer_config"),
        assets.resources.require_file("tokenizer_json"),
        engine::tokenizers::LlamaBpePreTokenizer::Qwen2,
    });
}

std::string format_duration(float seconds) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.2f", static_cast<double>(seconds));
    return std::string(buffer);
}

}  // namespace

VibeASRSession::VibeASRSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const VibeASRAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(validate_session_setup(task, std::move(options), require_contract(contract))),
      task_(std::move(task)),
      assets_(require_assets(std::move(assets))),
      contract_(std::move(contract)),
      tokenizer_(load_tokenizer(*assets_)),
      encoder_(assets_->vae, execution_context(), encoder_graph_arena_bytes(RuntimeSessionBase::options())),
      lm_(assets_->lm_weights,
          assets_->lm,
          execution_context(),
          prefill_graph_arena_bytes(RuntimeSessionBase::options()),
          decode_graph_arena_bytes(RuntimeSessionBase::options()),
          kWeightContextBytes) {
    // Both weight stores have uploaded by now; drop the resident file blobs.
    assets_->vae->source->release_storage();
    assets_->lm_weights->release_storage();
}

VibeASRSession::~VibeASRSession() = default;

std::string VibeASRSession::family() const {
    return "vibeasr";
}

runtime::VoiceTaskKind VibeASRSession::task_kind() const {
    return task_.task;
}

runtime::RunMode VibeASRSession::run_mode() const {
    return task_.mode;
}

void VibeASRSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void)request;
    mark_prepared();
}

VibeASRSession::RequestOptions VibeASRSession::parse_request_options(const runtime::TaskRequest & request) const {
    runtime::validate_spec_backed_request_options(request.options, require_contract(contract_), "VibeASR");
    RequestOptions out;
    if (const auto value = runtime::find_option(request.options, {"output_format"}); value.has_value()) {
        if (*value != "text" && *value != "json") {
            throw std::runtime_error("VibeASR output_format must be text or json");
        }
        out.output_format = *value;
    }
    if (const auto value = runtime::find_option(request.options, {"context"}); value.has_value()) {
        out.context = *value;
    }
    out.max_new_tokens = runtime::parse_positive_i64_option(request.options, {"max_new_tokens"}, out.max_new_tokens);
    return out;
}

runtime::AudioBuffer VibeASRSession::normalize(const runtime::AudioBuffer & audio) const {
    if (audio.samples.empty()) {
        throw std::runtime_error("VibeASR requires non-empty audio");
    }
    auto mono = engine::audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
    if (audio.sample_rate != kSampleRate) {
        // VibeASR.cpp resamples with a naive linear kernel; audio.cpp's soxr path
        // is the better filter, so a non-24 kHz input will not match the
        // reference sample for sample.
        engine::audio::SoxrResampleOptions options;
        options.profile = engine::audio::SoxrResampleProfile::QualityOnly;
        options.output_length_policy = engine::audio::SoxrOutputLengthPolicy::ExactExpected;
        options.output_padding = 256;
        options.reject_empty_output = true;
        options.warning_context = "VibeASR audio";
        options.fallback_description = "linear resampling";
        mono = engine::audio::resample_mono_soxr_or_linear(mono, audio.sample_rate, kSampleRate, options);
    }
    double sum = 0.0;
    for (const float sample : mono) {
        sum += static_cast<double>(sample) * static_cast<double>(sample);
    }
    const float rms = std::sqrt(static_cast<float>(sum / std::max<size_t>(mono.size(), 1)));
    if (rms >= kNormalizeEps) {
        const float target = std::pow(10.0F, kTargetDbFs / 20.0F);
        const float gain = target / (rms + kNormalizeEps);
        float max_abs = 0.0F;
        for (float & sample : mono) {
            sample *= gain;
            max_abs = std::max(max_abs, std::abs(sample));
        }
        // Not in VibeASR.cpp, which can clip on a loud clip; audio.cpp's own
        // vibevoice_asr frontend clamps here and this port follows it.
        if (max_abs > 1.0F) {
            const float scale = max_abs + kNormalizeEps;
            for (float & sample : mono) {
                sample /= scale;
            }
        }
    }
    return runtime::AudioBuffer{kSampleRate, 1, std::move(mono)};
}

VibeASRSpeechEmbeddings VibeASRSession::encode_speech(const std::vector<float> & samples) {
    const auto encode_start = Clock::now();
    const auto acoustic = encoder_.encode_acoustic(samples);
    const auto semantic = encoder_.encode_semantic(samples);
    debug::timing_log_scalar("vibeasr.session.encoder_ms", engine::debug::elapsed_ms(encode_start));
    if (acoustic.frames != semantic.frames || acoustic.dim != semantic.dim) {
        throw std::runtime_error("VibeASR encoder branches disagree on the feature shape");
    }
    if (acoustic.dim != assets_->lm.hidden_size) {
        throw std::runtime_error("VibeASR connector width does not match the decoder hidden size");
    }

    // Both connectors are LM-width, so the reference sums them element-wise.
    VibeASRSpeechEmbeddings out;
    out.tokens = acoustic.frames;
    out.hidden_size = acoustic.dim;
    out.values.resize(acoustic.values.size());
    for (size_t i = 0; i < out.values.size(); ++i) {
        out.values[i] = acoustic.values[i] + semantic.values[i];
    }
    return out;
}

VibeASRLmPrompt VibeASRSession::build_prompt(
    int64_t speech_tokens,
    float duration_seconds,
    const RequestOptions & options) const {
    // Qwen2.5 ChatML, assembled exactly as VibeASR.cpp does it:
    //   <|im_start|>system\n{SYSTEM}<|im_end|>\n
    //   <|im_start|>user\n<|speech_start|><|speech_pad|>xN<|speech_end|>{suffix}<|im_end|>\n
    // There is deliberately no generation prompt -- the model emits the
    // <|im_start|>assistant\n header itself.
    const auto encode = [this](const std::string & text) {
        return tokenizer_->encode(text, false);
    };

    const std::string instruction = options.output_format == "json"
        ? "please transcribe it with these keys: Start, End, Speaker, Content"
        : "please transcribe it.";
    std::string suffix;
    if (options.context.empty()) {
        suffix = "\nThis is a " + format_duration(duration_seconds) + " seconds audio, " + instruction;
    } else {
        suffix = "\nThis is a " + format_duration(duration_seconds) + " seconds audio, with extra info: " +
            options.context + "\n\n" +
            (options.output_format == "json"
                 ? "Please transcribe it with these keys: Start, End, Speaker, Content"
                 : "Please transcribe it.");
    }

    const auto system_content = encode(std::string("system\n") + kSystemPrompt);
    const auto newline = encode("\n");
    const auto user_prefix = encode("user\n");
    const auto user_suffix = encode(suffix);

    VibeASRLmPrompt prompt;
    const auto append = [&prompt](const std::vector<int32_t> & ids) {
        prompt.input_ids.insert(prompt.input_ids.end(), ids.begin(), ids.end());
    };
    prompt.input_ids.push_back(kImStart);
    append(system_content);
    prompt.input_ids.push_back(kImEnd);
    append(newline);
    prompt.input_ids.push_back(kImStart);
    append(user_prefix);
    prompt.input_ids.push_back(kSpeechStart);
    // The reference builds ceil(samples / 3200) pads but only prefills
    // min(pads, frames) of them, so emitting exactly `frames` pads produces the
    // same sequence.
    for (int64_t i = 0; i < speech_tokens; ++i) {
        prompt.speech_positions.push_back(static_cast<int32_t>(prompt.input_ids.size()));
        prompt.input_ids.push_back(kSpeechPad);
    }
    prompt.input_ids.push_back(kSpeechEnd);
    append(user_suffix);
    prompt.input_ids.push_back(kImEnd);
    append(newline);
    return prompt;
}

std::string VibeASRSession::decode_tokens(const std::vector<int32_t> & token_ids) const {
    // The prompt carries no generation prompt, so the model emits its own
    // "<|im_start|>assistant\n" header; drop it exactly as the reference does.
    size_t begin = 0;
    const auto piece = [this](int32_t id) { return tokenizer_->decode({id}, true); };
    if (!token_ids.empty() && token_ids[0] == kImStart) {
        begin = 1;
        if (begin < token_ids.size() && piece(token_ids[begin]) == "assistant") {
            ++begin;
            if (begin < token_ids.size() && piece(token_ids[begin]) == "\n") {
                ++begin;
            }
        }
    }

    std::vector<int32_t> filtered;
    filtered.reserve(token_ids.size() - begin);
    for (size_t i = begin; i < token_ids.size(); ++i) {
        const int32_t id = token_ids[i];
        if (id == kSpeechPad || id == kSpeechStart || id == kSpeechEnd || id == kEndOfText ||
            tokenizer_->is_control_token_id(id)) {
            continue;
        }
        filtered.push_back(id);
    }
    if (filtered.empty()) {
        return "";
    }
    return engine::io::trim_ascii_whitespace(tokenizer_->decode(filtered, true));
}

runtime::TaskResult VibeASRSession::run(const runtime::TaskRequest & request) {
    require_prepared("VibeASR run()");
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("VibeASR run() requires audio_input");
    }
    const auto wall_start = Clock::now();
    const auto options = parse_request_options(request);
    const auto audio = normalize(*request.audio_input);
    const float duration_seconds =
        static_cast<float>(audio.samples.size()) / static_cast<float>(kSampleRate);

    auto speech = encode_speech(audio.samples);
    if (speech.tokens <= 0) {
        throw std::runtime_error("VibeASR audio is too short to produce a single encoder frame");
    }
    const auto prompt = build_prompt(speech.tokens, duration_seconds, options);

    VibeASRGenerationOptions generation;
    generation.max_new_tokens = options.max_new_tokens;
    generation.eos_token_ids = {kImEnd, kEndOfText};
    const auto generated = lm_.generate(prompt, speech, generation);

    runtime::TaskResult result;
    result.text_output = runtime::Transcript{decode_tokens(generated), ""};
    debug::trace_log_scalar("vibeasr.session.speech_tokens", speech.tokens);
    debug::trace_log_scalar("vibeasr.session.generated_tokens", static_cast<int64_t>(generated.size()));
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_vibeasr_loader() {
    runtime::SpecBackedVoiceModelConfig<VibeASRAssets> config;
    config.family = "vibeasr";
    config.load_assets = [](const std::filesystem::path & model_path) {
        return load_vibeasr_assets(model_path);
    };
    config.create_session = [](
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options,
        std::shared_ptr<const VibeASRAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        return std::make_unique<VibeASRSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::community_models::vibeasr
