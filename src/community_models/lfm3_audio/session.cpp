#include "engine/community_models/lfm3_audio/session.h"

#include "engine/community_models/lfm2_audio/tokenizer.h"
#include "engine/community_models/lfm3_audio/backbone.h"
#include "engine/community_models/lfm3_audio/chat.h"
#include "engine/community_models/lfm3_audio/codec.h"
#include "engine/community_models/lfm3_audio/debug.h"
#include "engine/community_models/lfm3_audio/depth.h"
#include "engine/community_models/lfm3_audio/encoder.h"
#include "engine/community_models/lfm3_audio/features.h"
#include "engine/community_models/lfm3_audio/generator.h"
#include "engine/community_models/lfm3_audio/prompt.h"
#include "engine/community_models/lfm3_audio/published_gguf.h"
#include "engine/community_models/lfm3_audio/tokenizer.h"
#include "engine/community_models/lfm3_audio/weights.h"
#include "engine/framework/audio/chunking.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/text.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/text/chunking.h"
#include "engine/framework/text/utf8.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::lfm3_audio {
namespace {

constexpr const char * kFamily = "lfm3_audio";
constexpr const char * kModelName = "LFM3-Audio";
constexpr int kSampleRate = 16000;
// ASR: one pass beat 30 s chunks on a long probe clip (chunks lost words at every cut), so auto keeps up to 300 s
// whole and splits longer input at VAD pauses; 300 s is 3750 audio tokens, leaving max_tokens room in 8192 steps.
constexpr float kDefaultChunkSeconds = 300.0f;
constexpr float kMinChunkSeconds = 1.0f;
constexpr float kDefaultMaxPassSeconds = 300.0f;
// TTS: 512 frames (41 s) per text chunk; 200 codepoints of text is about 13 s of English speech.
constexpr int64_t kDefaultMaxFrames = 512;
constexpr int64_t kDefaultTextChunkSize = 200;
constexpr int64_t kDefaultStreamFramesPerEvent = 1;
// S2S: text tokens and frames of the reply together; greedy audio tails can loop, so after the reply text ends
// at most 375 frames (30 s) are sampled before the turn is closed with an end-of-audio frame.
constexpr int64_t kDefaultReplySteps = 1024;
constexpr int64_t kDefaultAudioTailFrames = 375;

const engine::model_spec::ModelContract & require_contract(const std::shared_ptr<const engine::model_spec::ModelContract> & contract) {
    if (contract == nullptr) {
        throw std::runtime_error("LFM3-Audio session requires a model contract");
    }

    return *contract;
}

void warn(const std::string & message) {
    std::cerr << "[warning][" << kFamily << "] " << message << "\n";
}

std::string seconds_text(double seconds) {
    char out[32];
    std::snprintf(out, sizeof(out), "%.1f", seconds);
    return out;
}

std::string seconds_at(int64_t sample) {
    return seconds_text(static_cast<double>(sample) / kSampleRate);
}

int64_t frame_count(const runtime::AudioBuffer & audio) {
    return audio.sample_rate > 0 && audio.channels > 0 ? static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels)) : 0;
}

double parse_max_pass_seconds(const runtime::SessionOptions & options) {
    const auto value = runtime::parse_finite_float_option(options.options, {"lfm3_audio.max_pass_duration_sec"}).value_or(kDefaultMaxPassSeconds);
    if (value < kMinChunkSeconds) {
        throw std::runtime_error("lfm3_audio.max_pass_duration_sec must be at least 1");
    }

    return value;
}

Lfm3WeightOptions parse_weight_options(const runtime::SessionOptions & options) {
    Lfm3WeightOptions out;
    if (const auto value = runtime::find_option(options.options, {"lfm3_audio.weight_type"})) {
        if (*value == "native") {
            out.type = Lfm3WeightType::Native;
        } else if (*value == "f16") {
            out.type = Lfm3WeightType::F16;
        } else if (*value == "f32") {
            out.type = Lfm3WeightType::F32;
        } else {
            throw std::runtime_error("lfm3_audio.weight_type must be native, f16 or f32, not " + *value);
        }
    }

    const auto repack = runtime::find_option(options.options, {"lfm3_audio.cpu_repack"});
    out.cpu_repack = !repack.has_value() || runtime::parse_bool_option(*repack, "lfm3_audio.cpu_repack");
    return out;
}

// Checked in samples before the audio reaches the encoder, whose graph grows with it.
void require_one_pass(int64_t frames, int sample_rate, double limit, const std::string & what, const std::string & remedy) {
    if (frames > std::llround(limit * sample_rate)) {
        const auto tenths = (frames * 10 + sample_rate - 1) / sample_rate;
        throw runtime::CapacityError("LFM3-Audio encodes at most " + seconds_text(limit) +
                                     " s of audio in one pass (lfm3_audio.max_pass_duration_sec), and " + what + " " +
                                     seconds_text(static_cast<double>(tenths) / 10) + " s; " + remedy);
    }
}

runtime::SessionOptions validate_session_setup(const runtime::TaskSpec & task, runtime::SessionOptions options,
                                               const engine::model_spec::ModelContract & contract, runtime::VoiceTaskKind kind) {
    if (task.task != kind) {
        throw std::runtime_error("LFM3-Audio session created for the wrong task");
    }

    // ASR takes whole files: the transcript only starts once the audio is all in the prompt.
    if (task.mode != runtime::RunMode::Offline && !(kind != runtime::VoiceTaskKind::Asr && task.mode == runtime::RunMode::Streaming)) {
        throw std::runtime_error(kind == runtime::VoiceTaskKind::Tts   ? "LFM3-Audio TTS runs offline or streaming"
                                 : kind == runtime::VoiceTaskKind::Asr ? "LFM3-Audio ASR runs offline only"
                                                                       : "LFM3-Audio s2s runs offline or streaming");
    }

    runtime::validate_spec_backed_session_options(options, contract, kFamily, kModelName);
    return options;
}

// The spec lists one set of request options for all tasks; each session turns away the others' options.
void reject_options(const runtime::TaskRequest & request, std::initializer_list<const char *> names, const char * task) {
    for (const char * name : names) {
        if (request.options.count(name) != 0) {
            throw runtime::InvalidRequestError(std::string("LFM3-Audio ") + task + " does not take request option " + name);
        }
    }
}

void reject_history(const runtime::TaskRequest & request, const char * task) {
    for (const auto & artifact : request.input_artifacts) {
        if (artifact.id.rfind("lfm3_audio.", 0) == 0) {
            throw runtime::InvalidRequestError(std::string("LFM3-Audio ") + task + " takes no conversation history, so no " + artifact.id + " artifact");
        }
    }
}

bool parse_request_bool(const runtime::TaskRequest & request, const char * name) {
    const auto value = runtime::find_option(request.options, {name});
    if (!value.has_value()) {
        return false;
    }

    try {
        return runtime::parse_bool_option(*value, name);
    } catch (const std::runtime_error & error) {
        throw runtime::InvalidRequestError(error.what());
    }
}

Lfm3AudioSampling parse_audio_sampling(const runtime::TaskRequest & request, const Lfm3SpeechSampling & defaults) {
    Lfm3AudioSampling out;
    out.penalty_window = defaults.repetition_window;
    out.temperature = runtime::parse_finite_float_option(request.options, {"temperature"}).value_or(defaults.temperature);
    out.top_k = runtime::parse_i64_option(request.options, {"top_k"}).value_or(defaults.top_k);
    out.repetition_penalty = runtime::parse_finite_float_option(request.options, {"repetition_penalty"}).value_or(defaults.repetition_penalty);
    if (*out.temperature < 0.0f || out.top_k < 0 || out.repetition_penalty <= 0.0f) {
        throw runtime::InvalidRequestError("LFM3-Audio temperature and top_k must not be negative, and repetition_penalty must be positive");
    }

    return out;
}

Lfm3TextSampling parse_text_sampling(const runtime::TaskRequest & request) {
    Lfm3TextSampling out;
    out.temperature = runtime::parse_finite_float_option(request.options, {"text_temperature"}).value_or(*out.temperature);
    out.top_k = runtime::parse_i64_option(request.options, {"text_top_k"}).value_or(out.top_k);
    if (*out.temperature < 0.0f || out.top_k < 0) {
        throw runtime::InvalidRequestError("LFM3-Audio text_temperature and text_top_k must not be negative");
    }

    return out;
}

uint64_t parse_seed(const runtime::TaskRequest & request) {
    return runtime::parse_u64_option(request.options, {"seed"}).value_or(runtime::random_u64_seed());
}

bool is_ascii_alnum(char32_t value) {
    return (value >= '0' && value <= '9') || (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z');
}

// Chinese and Japanese put no spaces between words: CJK ideographs, kana, CJK punctuation and full-width forms.
bool is_unspaced(char32_t value) {
    return (value >= 0x2E80 && value <= 0x9FFF) || (value >= 0xF900 && value <= 0xFAFF) || (value >= 0xFF00 && value <= 0xFFEF) ||
           (value >= 0x20000 && value <= 0x3FFFF);
}

// The code point whose UTF-8 sequence starts at text[at]; a stray or cut-off byte reads as itself.
char32_t code_point_at(const std::string & text, size_t at) {
    const auto lead = static_cast<unsigned char>(text[at]);
    const size_t width = lead < 0xC0 ? 1 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4;
    if (width == 1 || at + width > text.size()) {
        return lead;
    }
    char32_t value = lead & (0x7Fu >> width);
    for (size_t i = 1; i < width; ++i) {
        value = (value << 6) | (static_cast<unsigned char>(text[at + i]) & 0x3Fu);
    }
    return value;
}

std::vector<runtime::TimeSpan> plan_fixed_chunks(int64_t samples, int64_t chunk_samples) {
    std::vector<runtime::TimeSpan> spans;
    for (const auto & chunk : audio::plan_audio_chunks(samples, {chunk_samples, chunk_samples})) {
        spans.push_back({chunk.output_start_sample, chunk.output_start_sample + chunk.valid_samples});
    }

    const auto min_samples = static_cast<int64_t>(kMinChunkSeconds * kSampleRate);
    if (spans.size() > 1 && spans.back().end_sample - spans.back().start_sample < min_samples) {
        spans.pop_back();
        spans.back().end_sample = samples;
    }

    return spans;
}

void dump_audio(const char * name, const std::vector<float> & samples) {
    if (lfm3_dump_enabled()) {
        lfm3_dump(name, samples.data(), {static_cast<int64_t>(samples.size())});
    }
}

bool is_end_of_audio_frame(const std::vector<int32_t> & codes, int32_t end_of_audio) {
    return std::find(codes.begin(), codes.end(), end_of_audio) != codes.end();
}

struct PulledSteps {
    std::vector<int32_t> tokens;
    std::vector<std::vector<int32_t>> frames;  // EOS frames left out: they carry no speech
    bool done = false;                         // the generator has no more steps
};

// The generator's steps until it has given `max_frames` speech frames, or until its end.
PulledSteps pull_steps(Lfm3Generator & generator, int32_t end_of_audio, size_t max_frames = std::numeric_limits<size_t>::max()) {
    PulledSteps out;
    while (out.frames.size() < max_frames) {
        auto step = generator.next();
        if (!step.has_value()) {
            out.done = true;
            break;
        }
        if (step->codes.empty()) {
            out.tokens.push_back(step->token);
        } else if (!is_end_of_audio_frame(step->codes, end_of_audio)) {
            out.frames.push_back(std::move(step->codes));
        }
    }
    return out;
}

struct Model {
    Model(std::shared_ptr<const Lfm3AudioAssets> assets_in, const runtime::SessionOptions & session, core::ExecutionContext & execution_in,
          bool audio_in, bool audio_out)
        : assets(std::move(assets_in)),
          execution(execution_in),
          weights(parse_weight_options(session)),
          max_pass_seconds(parse_max_pass_seconds(session)),
          tokenizer(assets->vocabulary.tokens.empty() ? Lfm3Tokenizer(assets->tokenizer_json) : Lfm3Tokenizer(assets->vocabulary)),
          text_vocab(tokenizer_size()),
          backbone(assets->model, assets->backbone, assets->depth.codebooks, assets->depth.audio_vocab_size, execution, weights) {
        if (audio_in) {
            encoder = std::make_unique<Lfm3AudioEncoderRuntime>(assets->model, assets->encoder, execution, weights);
        }
        if (audio_out) {
            depth = std::make_unique<Lfm3DepthDecoderRuntime>(assets->model, assets->depth, execution, weights);
            codec = std::make_unique<Lfm3CodecDecoder>(assets->codec_weights, assets->codec, execution, weights);
            assets->codec_weights->release_storage();
        }
        assets->model->release_storage();
    }

    // Ids past the tokenizer (untrained rows of the head) have no text, though a reply keeps them as steps.
    // Ids are contiguous from 0, so the first one decode rejects is the size.
    int32_t tokenizer_size() const {
        const auto known = [&](int64_t id) {
            try {
                (void)tokenizer.decode({static_cast<int32_t>(id)}, false);
                return true;
            } catch (const std::exception &) {
                return false;
            }
        };
        int64_t lo = 0, hi = assets->backbone.vocab_size;
        if (known(hi - 1)) {
            return static_cast<int32_t>(hi);
        }
        while (hi - lo > 1) {
            const int64_t mid = lo + (hi - lo) / 2;
            (known(mid) ? lo : hi) = mid;
        }
        return static_cast<int32_t>(hi);
    }

    // HF decode(skip_special_tokens=True) bytes of the text ids; not necessarily whole UTF-8 characters.
    std::string decode(const std::vector<int32_t> & ids) const {
        std::vector<int32_t> kept;
        std::copy_if(ids.begin(), ids.end(), std::back_inserter(kept), [&](int32_t id) { return id < text_vocab; });
        return tokenizer.decode(kept);
    }

    Lfm3AudioEmbeddings encode(const std::vector<float> & mono16k) {
        return encoder->encode(features.extract(mono16k, static_cast<size_t>(std::max(0, execution.config().threads))));
    }

    std::unique_ptr<Lfm3Generator> generate(Lfm3Prompt prompt, const Lfm3AudioEmbeddings & audio, Lfm3GenerationOptions options) {
        if (lfm3_dump_enabled()) {
            lfm3_dump("prompt_ids", prompt.ids.data(), {static_cast<int64_t>(prompt.ids.size())});
        }
        options.maskgit_orders = lfm3_injected_orders();
        return std::make_unique<Lfm3Generator>(backbone, depth.get(), tokenizer, assets->generation, assets->depth, std::move(prompt), audio,
                                               std::move(options));
    }

    std::vector<float> speak(const std::vector<std::vector<int32_t>> & frames) {
        auto samples = codec->decode(frames);
        if (lfm3_dump_enabled()) {
            std::vector<int32_t> codes;
            for (const auto & frame : frames) codes.insert(codes.end(), frame.begin(), frame.end());
            lfm3_dump("codes", codes.data(), {static_cast<int64_t>(frames.size()), assets->depth.codebooks});
            codec->start_stream();  // a second pass: spectrum() advances the stream without its ISTFT
            const auto rows = codec->spectrum(frames);
            const int64_t width = assets->codec.n_fft + 2;
            lfm3_dump("codec_spectrum", rows.data(), {static_cast<int64_t>(rows.size()) / width, width});
            dump_audio("audio24k", samples);
        }
        return samples;
    }

    [[nodiscard]] Lfm3ReplyCheckpoint reply_checkpoint() const {
        return {assets->depth.codebooks, assets->depth.audio_vocab_size, assets->backbone.vocab_size};
    }

    [[nodiscard]] int32_t end_of_audio() const { return assets->depth.end_of_audio(); }
    [[nodiscard]] int sample_rate() const { return assets->codec.sample_rate; }

    std::shared_ptr<const Lfm3AudioAssets> assets;
    core::ExecutionContext & execution;
    Lfm3WeightOptions weights;
    double max_pass_seconds;
    Lfm3Tokenizer tokenizer;
    int32_t text_vocab;
    Lfm3FeatureExtractor features;
    Lfm3BackboneRuntime backbone;
    std::unique_ptr<Lfm3AudioEncoderRuntime> encoder;
    std::unique_ptr<Lfm3DepthDecoderRuntime> depth;
    std::unique_ptr<Lfm3CodecDecoder> codec;
};

// A turn cut off at max_tokens is kept with a warning, as upstream keeps what it generated. Returns whether it was cut.
bool check_speech_end(const Lfm3Generator & generator, size_t frames, int64_t max_steps, const std::string & what) {
    if (frames == 0) {
        throw std::runtime_error(generator.ended() ? "LFM3-Audio produced no speech for " + what
                                                   : "LFM3-Audio reached max_tokens before any speech for " + what + "; raise max_tokens");
    }

    if (generator.ended()) {
        return false;
    }

    warn(what + " reached max_tokens=" + std::to_string(max_steps) + " and is cut off there; raise max_tokens for the rest");
    return true;
}

std::shared_ptr<const Lfm3AudioAssets> require_assets(std::shared_ptr<const Lfm3AudioAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("LFM3-Audio session requires assets");
    }
    return assets;
}

std::unique_ptr<runtime::IVoiceTaskSession> create_lfm3_audio_session(const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                                                                      std::shared_ptr<const Lfm3AudioAssets> assets,
                                                                      std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (task.task == runtime::VoiceTaskKind::Tts) {
        return std::make_unique<Lfm3TtsSession>(task, options, std::move(assets), std::move(contract));
    }

    if (task.task == runtime::VoiceTaskKind::SpeechToSpeech) {
        return std::make_unique<Lfm3ChatSession>(task, options, std::move(assets), std::move(contract));
    }

    if (task.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("LFM3-Audio supports the asr, tts and s2s tasks");
    }

    return std::make_unique<Lfm3AsrSession>(task, options, std::move(assets), std::move(contract));
}

}  // namespace

int64_t lfm3_asr_default_max_tokens(double seconds, bool chat, bool thinking) {
    if (chat) {
        return thinking ? 2048 : 512;
    }
    return std::max<int64_t>(512, static_cast<int64_t>(std::ceil(8.0 * std::max(0.0, seconds))));
}

std::optional<std::vector<int32_t>> lfm3_thinking_answer(const std::vector<int32_t> & tokens, int32_t think_end) {
    const auto close = std::find(tokens.begin(), tokens.end(), think_end);
    if (close == tokens.end()) {
        return std::nullopt;
    }
    return std::vector<int32_t>(close + 1, tokens.end());
}

void append_lfm3_chunk_text(std::string & text, std::string chunk) {
    chunk = io::trim_ascii_whitespace(std::move(chunk));
    if (chunk.empty()) {
        return;
    }

    if (!text.empty()) {
        size_t last = text.size() - 1;
        while (last > 0 && engine::text::is_utf8_continuation(static_cast<unsigned char>(text[last]))) {
            --last;
        }
        const char32_t before = code_point_at(text, last);
        const char32_t after = code_point_at(chunk, 0);
        if ((after >= 0x80 || is_ascii_alnum(after)) && !is_unspaced(before) && !is_unspaced(after)) {
            text.push_back(' ');
        }
    }

    text += chunk;
}

struct Lfm3AsrSession::Impl {
    struct RequestOptions {
        std::string system_prompt;
        std::string language;  // empty for auto
        bool chat = false;
        bool thinking = false;
        std::optional<int64_t> max_tokens;
        Lfm3TextSampling text;
        uint64_t seed = 0;
    };

    runtime::TaskSpec task;
    std::shared_ptr<const engine::model_spec::ModelContract> contract;
    Model model;
    std::filesystem::path vad_model_path;
    std::unique_ptr<runtime::ILoadedVoiceModel> vad_model;
    std::unique_ptr<runtime::IOfflineVoiceTaskSession> vad;
    bool reached_max_tokens = false;

    RequestOptions parse(const runtime::TaskRequest & request) const {
        runtime::validate_spec_backed_request_options(request.options, require_contract(contract), kModelName);
        reject_options(request,
                       {"temperature", "top_k", "repetition_penalty", "text_chunk_mode", "text_chunk_size", "stream_frames_per_event",
                        "return_codes", "max_audio_tail_frames"},
                       "ASR");
        reject_history(request, "ASR");

        RequestOptions out;
        out.chat = parse_request_bool(request, "chat");
        out.thinking = parse_request_bool(request, "enable_thinking");
        if (out.thinking && !out.chat) {
            throw runtime::InvalidRequestError("LFM3-Audio enable_thinking needs chat=true");
        }

        const auto language = runtime::find_option(request.options, {"language"}).value_or("auto");
        // The ASR system prompt is required: without one the model answers instead of transcribing.
        out.system_prompt = lfm3_asr_system_prompt(language);
        // The chat prompt has no system turn to name a language in, so a chat answer reports none.
        out.language = language == "auto" || out.chat ? "" : language;
        // An empty value (the CLI or a cleared form field) keeps the default, like the TTS and S2S paths.
        if (runtime::find_option(request.options, {"max_tokens"})) {
            out.max_tokens = runtime::parse_positive_i64_option(request.options, {"max_tokens"}, 1);
        }
        out.text = parse_text_sampling(request);
        out.seed = parse_seed(request);
        return out;
    }

    std::vector<runtime::TimeSpan> plan_chunks(const runtime::TaskRequest & request, const std::vector<float> & samples, bool chat) {
        const auto total = static_cast<int64_t>(samples.size());
        const auto mode = audio::parse_audio_chunk_mode(request.options);
        const auto requested = audio::parse_audio_chunk_seconds_override(request.options);
        const double seconds = requested ? *requested : std::min<double>(kDefaultChunkSeconds, model.max_pass_seconds);
        if (!std::isfinite(seconds) || seconds < kMinChunkSeconds) {
            throw runtime::InvalidRequestError("LFM3-Audio audio_chunk_duration_sec must be at least 1");
        }

        // A chat answers the whole input at once.
        if (mode == audio::AudioChunkMode::None || chat) {
            require_one_pass(total, kSampleRate, model.max_pass_seconds, "the input is",
                             chat ? "send a shorter question, or raise the limit" : "use audio_chunk_mode=auto, or raise the limit");
            return {{0, total}};
        }

        if (mode != audio::AudioChunkMode::Auto && mode != audio::AudioChunkMode::Fixed && mode != audio::AudioChunkMode::Vad) {
            throw runtime::InvalidRequestError("LFM3-Audio supports audio_chunk_mode=auto, fixed, vad, or none");
        }

        const auto chunk_samples = static_cast<int64_t>(std::llround(seconds * kSampleRate));
        require_one_pass(std::min(total, chunk_samples), kSampleRate, model.max_pass_seconds, "audio_chunk_duration_sec allows chunks of up to",
                         "lower audio_chunk_duration_sec, or raise the limit");
        if (mode == audio::AudioChunkMode::Auto && total <= chunk_samples) {
            return {{0, total}};
        }

        if (mode == audio::AudioChunkMode::Fixed || (mode == audio::AudioChunkMode::Auto && !io::is_existing_directory(vad_model_path))) {
            return plan_fixed_chunks(total, chunk_samples);
        }

        const audio::VadAudioChunkOptions options{chunk_samples, kSampleRate / 2, kSampleRate / 4};
        return audio::plan_vad_audio_chunks(runtime::AudioBuffer{kSampleRate, 1, samples}, vad_session(), options);
    }

    runtime::IOfflineVoiceTaskSession & vad_session() {
        if (vad == nullptr) {
            runtime::ModelLoadRequest load_request;
            load_request.model_path = vad_model_path;
            vad_model = runtime::load_vad_model(load_request);
            auto session = vad_model->create_task_session({runtime::VoiceTaskKind::Vad, runtime::RunMode::Offline},
                                                          runtime::SessionOptions{model.execution.config(), {}});
            auto * offline = dynamic_cast<runtime::IOfflineVoiceTaskSession *>(session.get());
            if (offline == nullptr) {
                throw std::runtime_error("LFM3-Audio VAD session does not support offline execution");
            }

            session.release();
            vad.reset(offline);
        }

        return *vad;
    }

    std::string transcribe(const std::vector<float> & samples, const runtime::TimeSpan & span, const RequestOptions & options) {
        const std::vector<float> chunk(samples.begin() + span.start_sample, samples.begin() + span.end_sample);
        const auto audio = model.encode(chunk);
        auto prompt = options.chat ? make_lfm3_chat_prompt(model.tokenizer, audio.tokens, options.thinking)
                                   : make_lfm3_asr_prompt(model.tokenizer, options.system_prompt, audio.tokens);

        const auto & ids = model.tokenizer.ids();
        const double seconds = static_cast<double>(chunk.size()) / kSampleRate;
        Lfm3GenerationOptions generation;
        // The default stops at the context's end: a 418 s pass (5244 prompt steps) would otherwise ask for 3345 more.
        const auto room = std::max<int64_t>(1, kLfm3MaxConversationSteps - static_cast<int64_t>(prompt.ids.size()));
        generation.max_steps = options.max_tokens.value_or(std::min(lfm3_asr_default_max_tokens(seconds, options.chat, options.thinking), room));
        generation.text = options.text;
        generation.seed = options.seed;
        // Text only: the mode markers would open audio this session cannot make.
        generation.suppress = {ids.mixed_start, ids.audio_start};
        const auto generator = model.generate(std::move(prompt), audio, generation);
        std::vector<int32_t> tokens;
        while (const auto step = generator->next()) {
            tokens.push_back(step->token);
        }

        const std::string where = seconds_at(span.start_sample) + "-" + seconds_at(span.end_sample) + " s";
        if (!generator->ended()) {
            reached_max_tokens = true;
            warn("the " + std::string(options.chat ? "answer to " : "transcript of ") + where + " reached max_tokens=" +
                 std::to_string(generation.max_steps) + " and is cut off there; raise max_tokens for the rest");
        }
        debug::trace_log_scalar("lfm3_audio.session.audio_tokens", audio.tokens);
        debug::trace_log_scalar("lfm3_audio.session.generated_tokens", static_cast<int64_t>(tokens.size()));

        // Thinking writes its reasoning first and the answer after </think>.
        if (options.thinking) {
            auto answer = lfm3_thinking_answer(tokens, ids.think_end);
            if (!answer) {
                warn(generator->ended()
                         ? "the answer to " + where + " ended inside <think> without closing </think>, so it is empty"
                         : "the answer to " + where + " thought until max_tokens=" + std::to_string(generation.max_steps) +
                               " without closing </think>, so it is empty; raise max_tokens");
                return {};
            }
            tokens = std::move(*answer);
        }

        // Whole characters only (byte-level tokens); the trim also drops a chat answer's leading newlines.
        auto bytes = model.decode(tokens);
        return io::trim_ascii_whitespace(lfm2_audio::lfm2_take_text(bytes));
    }
};

Lfm3AsrSession::Lfm3AsrSession(runtime::TaskSpec task, runtime::SessionOptions options, std::shared_ptr<const Lfm3AudioAssets> assets,
                               std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(validate_session_setup(task, std::move(options), require_contract(contract), runtime::VoiceTaskKind::Asr)) {
    const auto & session = RuntimeSessionBase::options();
    impl_.reset(new Impl{task, std::move(contract), Model(require_assets(std::move(assets)), session, execution_context(), true, false),
                         runtime::find_option(session.options, {"lfm3_audio.vad_model_path"})
                             .value_or((std::filesystem::path("assets") / "framework" / "models" / "silero_vad").string()),
                         nullptr, nullptr, false});
}

Lfm3AsrSession::~Lfm3AsrSession() = default;

std::string Lfm3AsrSession::family() const {
    return kFamily;
}

runtime::VoiceTaskKind Lfm3AsrSession::task_kind() const {
    return runtime::VoiceTaskKind::Asr;
}

runtime::RunMode Lfm3AsrSession::run_mode() const {
    return impl_->task.mode;
}

void Lfm3AsrSession::prepare(const runtime::SessionPreparationRequest &) {
    mark_prepared();
}

runtime::TaskResult Lfm3AsrSession::run(const runtime::TaskRequest & request) {
    require_prepared("LFM3-Audio run()");
    auto & m = *impl_;
    m.reached_max_tokens = false;
    if (!request.audio_input.has_value()) {
        throw runtime::InvalidRequestError("LFM3-Audio ASR requires audio_input");
    }

    const auto wall_start = std::chrono::steady_clock::now();
    const auto options = m.parse(request);
    const auto samples = lfm3_audio_mono_16k(*request.audio_input);
    dump_audio("wave16k", samples);

    std::string text;
    for (const auto & span : m.plan_chunks(request, samples, options.chat)) {
        append_lfm3_chunk_text(text, m.transcribe(samples, span, options));
    }

    runtime::TaskResult result;
    result.text_output = runtime::Transcript{text, options.language};
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

bool Lfm3AsrSession::reached_max_tokens() const {
    return impl_->reached_max_tokens;
}

struct Lfm3TtsSession::Impl {
    struct RequestOptions {
        std::string system_prompt;
        Lfm3GenerationOptions generation;
        int64_t stream_frames_per_event = 0;
        std::vector<std::string> texts;  // the text chunks, one turn each
    };

    struct Stream {
        RequestOptions options;
        size_t next_turn = 0;
        std::unique_ptr<Lfm3Generator> generator;
        size_t turn_frames = 0;
        runtime::AudioBuffer audio;  // everything emitted
    };

    runtime::TaskSpec task;
    std::shared_ptr<const engine::model_spec::ModelContract> contract;
    Model model;
    std::unique_ptr<Stream> stream;
    bool reached_max_tokens = false;

    RequestOptions parse(const runtime::TaskRequest & request) const {
        if (!request.text_input.has_value() || io::trim_ascii_whitespace(request.text_input->text).empty()) {
            throw runtime::InvalidRequestError("LFM3-Audio TTS requires text_input");
        }

        reject_options(request,
                       {"audio_chunk_mode", "audio_chunk_duration_sec", "audio_chunk_seconds", "text_temperature", "text_top_k", "return_codes",
                        "chat", "enable_thinking", "max_audio_tail_frames"},
                       "TTS");
        runtime::validate_spec_backed_request_options(request.options, require_contract(contract), kModelName);
        reject_history(request, "TTS");

        std::string language = request.text_input->language;
        if (const auto option = runtime::find_option(request.options, {"language"})) {
            language = *option;
        }

        std::string voice_id;
        if (request.voice.has_value() && request.voice->speaker.has_value()) {
            if (request.voice->speaker->audio.has_value()) {
                throw runtime::InvalidRequestError("LFM3-Audio speaks with its built-in voices and does not clone reference audio");
            }
            voice_id = request.voice->speaker->cached_voice_id.value_or("");
        }

        const auto & voice = lfm3_tts_voice(voice_id, language);
        RequestOptions out;
        out.system_prompt = lfm3_tts_system_prompt(voice);
        out.generation.max_steps = runtime::parse_positive_i64_option(request.options, {"max_tokens"}, kDefaultMaxFrames);
        out.generation.audio = parse_audio_sampling(request, model.assets->generation.speech);
        out.generation.seed = parse_seed(request);
        // The turn ends at its end-of-audio frame; the <|im_end|> after it is not worth a step.
        out.generation.stop_after_audio_eos = true;
        out.stream_frames_per_event = runtime::parse_positive_i64_option(request.options, {"stream_frames_per_event"}, kDefaultStreamFramesPerEvent);

        const auto chunk_size = text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
        const bool cjk = voice.language_name == "Japanese" || voice.language_name == "Chinese";
        const auto chunk_mode = text::parse_text_chunk_mode_override(request.options).value_or(cjk ? text::TextChunkMode::Japanese : text::TextChunkMode::Default);
        for (const auto & chunk : runtime::chunk_text_request(request, chunk_size, chunk_mode)) {
            out.texts.push_back(chunk.text_input->text);
            // Turn away a placeholder in any chunk now, not after the earlier chunks' audio.
            (void)make_lfm3_tts_prompt(model.tokenizer, out.system_prompt, out.texts.back());
        }

        return out;
    }

    // Turn i samples with seed + i, so each chunk is reproducible alone and streaming matches offline.
    std::unique_ptr<Lfm3Generator> start_turn(const RequestOptions & options, size_t turn) {
        auto generation = options.generation;
        generation.seed += turn;
        return model.generate(make_lfm3_tts_prompt(model.tokenizer, options.system_prompt, options.texts.at(turn)), {}, std::move(generation));
    }

    std::string turn_name(const RequestOptions & options, size_t turn) const {
        return options.texts.size() > 1 ? "the speech of text chunk " + std::to_string(turn + 1) + " of " + std::to_string(options.texts.size())
                                        : "the speech";
    }

    std::vector<float> speak(const RequestOptions & options, size_t turn) {
        const auto generator = start_turn(options, turn);
        const auto frames = pull_steps(*generator, model.end_of_audio()).frames;
        reached_max_tokens = check_speech_end(*generator, frames.size(), options.generation.max_steps, turn_name(options, turn)) || reached_max_tokens;
        debug::trace_log_scalar("lfm3_audio.session.audio_frames", static_cast<int64_t>(frames.size()));
        return model.speak(frames);
    }
};

Lfm3TtsSession::Lfm3TtsSession(runtime::TaskSpec task, runtime::SessionOptions options, std::shared_ptr<const Lfm3AudioAssets> assets,
                               std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(validate_session_setup(task, std::move(options), require_contract(contract), runtime::VoiceTaskKind::Tts)) {
    impl_.reset(new Impl{task, std::move(contract),
                         Model(require_assets(std::move(assets)), RuntimeSessionBase::options(), execution_context(), false, true), nullptr, false});
}

Lfm3TtsSession::~Lfm3TtsSession() = default;

std::string Lfm3TtsSession::family() const {
    return kFamily;
}

runtime::VoiceTaskKind Lfm3TtsSession::task_kind() const {
    return runtime::VoiceTaskKind::Tts;
}

runtime::RunMode Lfm3TtsSession::run_mode() const {
    return impl_->task.mode;
}

void Lfm3TtsSession::prepare(const runtime::SessionPreparationRequest &) {
    mark_prepared();
}

runtime::TaskResult Lfm3TtsSession::run(const runtime::TaskRequest & request) {
    require_prepared("LFM3-Audio run()");
    auto & m = *impl_;
    m.reached_max_tokens = false;
    const auto wall_start = std::chrono::steady_clock::now();
    const auto options = m.parse(request);

    runtime::AudioBuffer audio{m.model.sample_rate(), 1, {}};
    for (size_t turn = 0; turn < options.texts.size(); ++turn) {
        runtime::append_audio_buffer(audio, runtime::AudioBuffer{m.model.sample_rate(), 1, m.speak(options, turn)});
    }

    runtime::TaskResult result;
    result.audio_output = std::move(audio);
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

runtime::StreamingPolicy Lfm3TtsSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::None;
    policy.output = runtime::StreamingOutputKind::PullEvents;
    return policy;
}

void Lfm3TtsSession::start_stream(const runtime::TaskRequest & request) {
    require_prepared("LFM3-Audio start_stream()");
    auto & m = *impl_;
    m.reached_max_tokens = false;
    if (m.task.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("LFM3-Audio start_stream() needs a streaming session");
    }

    reset();
    auto stream = std::make_unique<Impl::Stream>();
    stream->options = m.parse(request);
    stream->audio = runtime::AudioBuffer{m.model.sample_rate(), 1, {}};
    m.stream = std::move(stream);
}

// The codec is causal and the ISTFT only waits for the windows that overlap a sample, so each event's samples
// are final: the events add up to what run() returns for the same seed.
std::optional<runtime::StreamEvent> Lfm3TtsSession::next_stream_event() {
    auto & m = *impl_;
    if (m.stream == nullptr) {
        throw std::runtime_error("LFM3-Audio streaming has not been started");
    }

    auto & st = *m.stream;
    auto & codec = *m.model.codec;
    std::vector<float> samples;
    while (samples.empty()) {
        if (st.generator == nullptr) {
            if (st.next_turn == st.options.texts.size()) {
                return std::nullopt;
            }

            st.generator = m.start_turn(st.options, st.next_turn++);
            st.turn_frames = 0;
            codec.start_stream();
        }

        const auto pulled = pull_steps(*st.generator, m.model.end_of_audio(), static_cast<size_t>(st.options.stream_frames_per_event));
        st.turn_frames += pulled.frames.size();
        samples = codec.push(pulled.frames);
        if (pulled.done) {
            const size_t turn = st.next_turn - 1;
            m.reached_max_tokens =
                check_speech_end(*st.generator, st.turn_frames, st.options.generation.max_steps, m.turn_name(st.options, turn)) || m.reached_max_tokens;
            const auto rest = codec.finish();
            samples.insert(samples.end(), rest.begin(), rest.end());
            st.generator.reset();
        }
    }

    runtime::AudioBuffer chunk{m.model.sample_rate(), 1, std::move(samples)};
    runtime::append_audio_buffer(st.audio, chunk);
    runtime::StreamEvent event;
    event.audio_output = std::move(chunk);
    return event;
}

void Lfm3TtsSession::set_stream_event_sink(runtime::StreamEventCallback) {
    // PullEvents drivers forward what next_stream_event returns; pushing here too would deliver every event twice.
}

runtime::TaskResult Lfm3TtsSession::finish_stream() {
    auto & m = *impl_;
    if (m.stream == nullptr) {
        throw std::runtime_error("LFM3-Audio streaming has not been started");
    }

    runtime::TaskResult result;
    result.audio_output = std::move(m.stream->audio);
    reset();
    return result;
}

void Lfm3TtsSession::reset() {
    impl_->stream.reset();
}

runtime::StreamEvent Lfm3TtsSession::process_audio_chunk(const runtime::AudioChunk &) {
    throw std::runtime_error("LFM3-Audio TTS does not take streamed audio input");
}

runtime::TaskResult Lfm3TtsSession::finalize() {
    return {};
}

bool Lfm3TtsSession::reached_max_tokens() const {
    return impl_->reached_max_tokens;
}

struct Lfm3ChatSession::Impl {
    struct RequestOptions {
        std::string system_prompt;
        Lfm3GenerationOptions generation;
        int64_t stream_frames_per_event = 0;
        bool return_codes = false;
        std::vector<Lfm3ConversationTurn> history;
    };

    struct Stream {
        RequestOptions options;
        runtime::AudioBuffer input;
        std::unique_ptr<Lfm3Generator> generator;
        bool done = false;
        size_t frames = 0;
        lfm2_audio::Lfm2StreamedText text;
        runtime::AudioBuffer audio;  // everything emitted
    };

    runtime::TaskSpec task;
    std::shared_ptr<const engine::model_spec::ModelContract> contract;
    Model model;
    std::unique_ptr<Stream> stream;
    bool reached_max_tokens = false;

    RequestOptions parse(const runtime::TaskRequest & request) const {
        reject_options(request,
                       {"audio_chunk_mode", "audio_chunk_duration_sec", "audio_chunk_seconds", "text_chunk_mode", "text_chunk_size", "chat",
                        "enable_thinking"},
                       "s2s");
        runtime::validate_spec_backed_request_options(request.options, require_contract(contract), kModelName);
        if (const auto language = runtime::find_option(request.options, {"language"}); language.has_value() && *language != "auto") {
            throw runtime::InvalidRequestError("LFM3-Audio s2s replies in the language of the conversation and takes no language");
        }
        if (request.voice.has_value() && request.voice->speaker.has_value()) {
            throw runtime::InvalidRequestError("LFM3-Audio s2s replies in the checkpoint's voice and takes no voice");
        }

        RequestOptions out;
        out.system_prompt = kLfm3S2sSystemPrompt;
        if (request.text_input.has_value() && !io::trim_ascii_whitespace(request.text_input->text).empty()) {
            out.system_prompt = request.text_input->text;
            // Here, so that a stream turns it away before its audio comes; the prompt builder checks again.
            require_lfm3_no_audio_placeholders(model.tokenizer.encode(out.system_prompt), model.tokenizer.ids());
        }

        auto & generation = out.generation;
        generation.max_steps = runtime::parse_positive_i64_option(request.options, {"max_tokens"}, kDefaultReplySteps);
        generation.audio = parse_audio_sampling(request, model.assets->generation.speech);
        generation.text = parse_text_sampling(request);
        generation.seed = parse_seed(request);
        generation.max_audio_tail_frames = runtime::parse_i64_option(request.options, {"max_audio_tail_frames"}).value_or(kDefaultAudioTailFrames);
        if (generation.max_audio_tail_frames < 0) {
            throw runtime::InvalidRequestError("LFM3-Audio max_audio_tail_frames must not be negative");
        }
        out.stream_frames_per_event = runtime::parse_positive_i64_option(request.options, {"stream_frames_per_event"}, kDefaultStreamFramesPerEvent);
        out.return_codes = parse_request_bool(request, "return_codes");

        // Read here, so that a stream turns away bad history before its audio comes.
        out.history = read_lfm3_conversation(request.input_artifacts, model.reply_checkpoint(), generation.max_steps);
        for (size_t turn = 0; turn < out.history.size(); ++turn) {
            const auto & question = out.history[turn].question;
            require_one_pass(frame_count(question), question.sample_rate, model.max_pass_seconds,
                             "the question of earlier turn " + std::to_string(turn + 1) + " is", "leave that turn out, or raise the limit");
        }

        return out;
    }

    // shortcut: no encoder or prefix cache across turns; add one when multi-turn latency matters.
    std::unique_ptr<Lfm3Generator> start_reply(const RequestOptions & options, const runtime::AudioBuffer & audio) {
        require_one_pass(frame_count(audio), audio.sample_rate, model.max_pass_seconds, "the question is", "send a shorter one, or raise the limit");
        // The prompt is built from the questions' lengths, so that one past the room is turned away before any encoding.
        std::vector<int64_t> question_tokens;
        std::vector<std::vector<Lfm3Step>> replies;
        for (const auto & turn : options.history) {
            question_tokens.push_back(lfm3_question_tokens(turn.question));
            replies.push_back(turn.reply);
        }

        auto prompt =
            make_lfm3_s2s_prompt(model.tokenizer, options.system_prompt, question_tokens, replies, lfm3_question_tokens(audio), model.end_of_audio());
        require_lfm3_conversation_room(static_cast<int64_t>(prompt.ids.size()), options.generation.max_steps);

        Lfm3AudioEmbeddings embeddings;
        const auto add = [&](const runtime::AudioBuffer & question) {
            const auto samples = lfm3_audio_mono_16k(question);
            const auto encoded = model.encode(samples);
            if (encoded.tokens != lfm3_question_tokens(question)) {
                throw std::runtime_error("LFM3-Audio encoded a question into " + std::to_string(encoded.tokens) + " tokens, and its prompt has " +
                                         std::to_string(lfm3_question_tokens(question)) + " slots for it");
            }

            embeddings.hidden = encoded.hidden;
            embeddings.tokens += encoded.tokens;
            embeddings.values.insert(embeddings.values.end(), encoded.values.begin(), encoded.values.end());
            return samples;
        };

        for (const auto & turn : options.history) {
            add(turn.question);
        }

        dump_audio("wave16k", add(audio));
        debug::trace_log_scalar("lfm3_audio.session.audio_tokens", embeddings.tokens);
        return model.generate(std::move(prompt), embeddings, options.generation);
    }

    runtime::Transcript transcript(std::string text) const {
        return runtime::Transcript{std::move(text), ""};
    }
};

Lfm3ChatSession::Lfm3ChatSession(runtime::TaskSpec task, runtime::SessionOptions options, std::shared_ptr<const Lfm3AudioAssets> assets,
                                 std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(validate_session_setup(task, std::move(options), require_contract(contract), runtime::VoiceTaskKind::SpeechToSpeech)) {
    impl_.reset(new Impl{task, std::move(contract),
                         Model(require_assets(std::move(assets)), RuntimeSessionBase::options(), execution_context(), true, true), nullptr, false});
}

Lfm3ChatSession::~Lfm3ChatSession() = default;

std::string Lfm3ChatSession::family() const {
    return kFamily;
}

runtime::VoiceTaskKind Lfm3ChatSession::task_kind() const {
    return runtime::VoiceTaskKind::SpeechToSpeech;
}

runtime::RunMode Lfm3ChatSession::run_mode() const {
    return impl_->task.mode;
}

void Lfm3ChatSession::prepare(const runtime::SessionPreparationRequest &) {
    mark_prepared();
}

runtime::TaskResult Lfm3ChatSession::run(const runtime::TaskRequest & request) {
    require_prepared("LFM3-Audio run()");
    auto & m = *impl_;
    m.reached_max_tokens = false;
    if (!request.audio_input.has_value()) {
        throw runtime::InvalidRequestError("LFM3-Audio s2s requires audio_input");
    }

    const auto wall_start = std::chrono::steady_clock::now();
    const auto options = m.parse(request);
    const auto generator = m.start_reply(options, *request.audio_input);
    const auto reply = pull_steps(*generator, m.model.end_of_audio());
    m.reached_max_tokens = check_speech_end(*generator, reply.frames.size(), options.generation.max_steps, "the reply");
    debug::trace_log_scalar("lfm3_audio.session.text_tokens", static_cast<int64_t>(reply.tokens.size()));
    debug::trace_log_scalar("lfm3_audio.session.audio_frames", static_cast<int64_t>(reply.frames.size()));

    auto bytes = m.model.decode(reply.tokens);
    runtime::TaskResult result;
    result.audio_output = runtime::AudioBuffer{m.model.sample_rate(), 1, m.model.speak(reply.frames)};
    result.text_output = m.transcript(lfm2_audio::lfm2_take_text(bytes));
    if (options.return_codes) {
        result.output_artifacts.push_back(make_lfm3_reply_artifact(generator->steps(), generator->ended(), m.model.reply_checkpoint()));
    }

    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

runtime::StreamingPolicy Lfm3ChatSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::AudioChunks;
    policy.output = runtime::StreamingOutputKind::PullEvents;
    // The user's turn is only collected, so the chunk size does not matter.
    policy.preferred_audio_chunk_seconds = 0.1;
    return policy;
}

void Lfm3ChatSession::start_stream(const runtime::TaskRequest & request) {
    require_prepared("LFM3-Audio start_stream()");
    auto & m = *impl_;
    m.reached_max_tokens = false;
    if (m.task.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("LFM3-Audio start_stream() needs a streaming session");
    }

    reset();
    auto stream = std::make_unique<Impl::Stream>();
    stream->options = m.parse(request);
    stream->audio = runtime::AudioBuffer{m.model.sample_rate(), 1, {}};
    m.stream = std::move(stream);
}

runtime::StreamEvent Lfm3ChatSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    auto & m = *impl_;
    if (m.stream == nullptr) {
        throw std::runtime_error("LFM3-Audio streaming has not been started");
    }

    auto & st = *m.stream;
    if (st.generator != nullptr || st.done) {
        throw std::runtime_error("LFM3-Audio s2s takes no more audio once the reply has started");
    }

    if (chunk.sample_rate <= 0 || chunk.channels <= 0 || chunk.samples.size() % static_cast<size_t>(chunk.channels) != 0) {
        throw std::runtime_error("LFM3-Audio audio chunks need a positive sample rate and channel count and whole frames");
    }

    if (st.input.samples.empty() && st.input.sample_rate == 0) {
        st.input.sample_rate = chunk.sample_rate;
        st.input.channels = chunk.channels;
    } else if (chunk.sample_rate != st.input.sample_rate || chunk.channels != st.input.channels) {
        throw std::runtime_error("LFM3-Audio audio chunk format changed during the stream");
    }

    st.input.samples.insert(st.input.samples.end(), chunk.samples.begin(), chunk.samples.end());
    require_one_pass(frame_count(st.input), st.input.sample_rate, m.model.max_pass_seconds, "the question so far is",
                     "send a shorter one, or raise the limit");
    return {};
}

// The reply starts once the driver pulls, after the user's audio has all come. Each event's samples are final
// (see the TTS stream) and its text is what the reply wrote since the last event, in whole characters.
std::optional<runtime::StreamEvent> Lfm3ChatSession::next_stream_event() {
    auto & m = *impl_;
    if (m.stream == nullptr) {
        throw std::runtime_error("LFM3-Audio streaming has not been started");
    }

    auto & st = *m.stream;
    if (st.done) {
        return std::nullopt;
    }

    auto & codec = *m.model.codec;
    if (st.generator == nullptr) {
        if (st.input.samples.empty()) {
            throw runtime::InvalidRequestError("LFM3-Audio s2s streaming received no audio");
        }

        st.generator = m.start_reply(st.options, st.input);
        codec.start_stream();
    }

    const auto pulled = pull_steps(*st.generator, m.model.end_of_audio(), static_cast<size_t>(st.options.stream_frames_per_event));
    st.frames += pulled.frames.size();
    auto samples = codec.push(pulled.frames);
    if (pulled.done) {
        m.reached_max_tokens = check_speech_end(*st.generator, st.frames, st.options.generation.max_steps, "the reply");
        const auto rest = codec.finish();
        samples.insert(samples.end(), rest.begin(), rest.end());
        st.done = true;
    }

    runtime::StreamEvent event;
    if (auto delta = st.text.add(m.model.decode(pulled.tokens)); !delta.empty()) {
        event.partial_text = m.transcript(std::move(delta));
    }

    if (!samples.empty()) {
        runtime::AudioBuffer chunk{m.model.sample_rate(), 1, std::move(samples)};
        runtime::append_audio_buffer(st.audio, chunk);
        event.audio_output = std::move(chunk);
    }

    return event;
}

void Lfm3ChatSession::set_stream_event_sink(runtime::StreamEventCallback) {
    // As for TTS: the driver forwards what next_stream_event returns.
}

runtime::TaskResult Lfm3ChatSession::finish_stream() {
    auto & m = *impl_;
    if (m.stream == nullptr) {
        throw std::runtime_error("LFM3-Audio streaming has not been started");
    }

    // A stream finished before its reply is over keeps the text and audio of its events, and its reply artifact
    // holds the steps so far, as not ended.
    auto & st = *m.stream;
    runtime::TaskResult result;
    result.audio_output = std::move(st.audio);
    result.text_output = m.transcript(st.text.text());
    if (st.options.return_codes) {
        const bool ended = st.generator != nullptr && st.generator->ended();
        const auto steps = st.generator != nullptr ? st.generator->steps() : std::vector<Lfm3Step>{};
        result.output_artifacts.push_back(make_lfm3_reply_artifact(steps, ended, m.model.reply_checkpoint()));
    }

    reset();
    return result;
}

void Lfm3ChatSession::reset() {
    impl_->stream.reset();
}

runtime::TaskResult Lfm3ChatSession::finalize() {
    return {};
}

bool Lfm3ChatSession::reached_max_tokens() const {
    return impl_->reached_max_tokens;
}

namespace {

// Liquid's published GGUFs have no model spec to resolve through, so they are
// claimed here, as lfm2_audio claims its own; every other path goes to the
// spec-backed loader, whose load_assets reads both layouts.
class Loader final : public runtime::IVoiceModelLoader {
public:
    explicit Loader(std::shared_ptr<runtime::IVoiceModelLoader> spec_backed) : spec_backed_(std::move(spec_backed)) {}

    std::string family() const override { return kFamily; }
    runtime::CapabilitySet advertised_capabilities() const override { return spec_backed_->advertised_capabilities(); }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint && *request.family_hint != kFamily) {
            return false;
        }

        return is_lfm3_audio_published_gguf(request.model_path) || spec_backed_->can_load(request);
    }

    bool identifies(const runtime::ModelLoadRequest & request) const override {
        return is_lfm3_audio_published_gguf(request.model_path) || spec_backed_->identifies(request);
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        if (!is_lfm3_audio_published_gguf(request.model_path)) {
            return spec_backed_->inspect(request);
        }

        const auto assets = load_lfm3_audio_assets(request.model_path);
        const auto contract = runtime::require_model_contract(kFamily);
        runtime::ModelInspection inspection;
        inspection.model_root = assets->published_backbone.parent_path();
        inspection.metadata = contract->metadata;
        inspection.capabilities = contract->capabilities;
        inspection.cli = contract->cli;
        inspection.discovered_weights = {{"model", assets->model->source_path()}, {"mmproj", assets->codec_weights->source_path()}};
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return spec_backed_->load(request);
    }

    runtime::ModelResourcePlan model_resources(const std::filesystem::path & model_path,
                                               const std::unordered_map<std::string, std::string> & session_options) const override {
        return spec_backed_->model_resources(model_path, session_options);
    }

private:
    std::shared_ptr<runtime::IVoiceModelLoader> spec_backed_;
};

}  // namespace

// The config.json model_type check keeps the spec-backed loader off other families' models, and puts it ahead of
// the families whose generic config.json + model.safetensors layout the upstream directory matches.
std::shared_ptr<runtime::IVoiceModelLoader> make_lfm3_audio_loader() {
    runtime::SpecBackedVoiceModelConfig<Lfm3AudioAssets> config;
    config.family = kFamily;
    config.load_assets = load_lfm3_audio_assets;
    config.create_session = create_lfm3_audio_session;
    config.accepts = [](const assets::ResourceBundle & resources) {
        // parse_json returns by value: keep it alive while model_type points into it.
        const auto config_json = resources.parse_json("config");
        const auto * model_type = config_json.find("model_type");
        return model_type != nullptr && model_type->is_string() && model_type->as_string() == kFamily;
    };
    return std::make_shared<Loader>(runtime::make_spec_backed_voice_loader(std::move(config)));
}

}  // namespace engine::community_models::lfm3_audio
