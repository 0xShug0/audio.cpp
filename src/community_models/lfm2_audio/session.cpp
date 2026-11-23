#include "engine/community_models/lfm2_audio/session.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/text.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/text/chunking.h"
#include "engine/framework/runtime/registry.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::lfm2_audio {
namespace {

constexpr const char * kFamily = "lfm2_audio";
constexpr const char * kModelName = "LFM2-Audio";
constexpr int kSampleRate = 16000;
constexpr float kDefaultChunkSeconds = 30.0f;
// A chunk under a second is at best a cut-off word, and under 10 ms too short
// for the features. The spec's min for audio_chunk_seconds matches; the
// framework does not enforce spec bounds, so plan_chunks does.
constexpr float kMinChunkSeconds = 1.0f;
// TTS defaults. 512 frames is generate_sequential's max_new_tokens in the
// README (41 s of audio); 200 codepoints of text is about 13 s of English
// speech, and at most about 30 s of Japanese.
constexpr int64_t kDefaultMaxFrames = 512;
constexpr int64_t kDefaultTextChunkSize = 200;
// Frames per streaming event: liquid-audio's demo decodes every frame.
constexpr int64_t kDefaultStreamFramesPerEvent = 1;
// S2S: liquid-audio's chat demo lets a reply run to 1024 steps (text tokens
// and audio frames together, about a minute of speech); the README's
// example stops at 512.
constexpr int64_t kDefaultReplySteps = 1024;
// The most audio the encoder takes in one pass by default: an ASR chunk (the
// whole input with audio_chunk_mode=none) or the S2S question. One pass is of
// little use by then (none is at 29% WER on the docs' 120 s English files,
// and Japanese loses sentences from about 41 to 45 s), and the encoder's
// buffer, linear in the length up to about 240 s, grows with its square past
// that.
constexpr float kDefaultMaxPassSeconds = 120.0f;

const engine::model_spec::ModelContract & require_contract(
    const std::shared_ptr<const engine::model_spec::ModelContract> & contract) {
    if (contract == nullptr) {
        throw std::runtime_error("LFM2-Audio session requires a model contract");
    }

    return *contract;
}

// Warnings go to stderr, as other families print theirs.
void warn(const std::string & message) {
    std::cerr << "[warning][" << kFamily << "] " << message << "\n";
}

// Seconds to a tenth.
std::string seconds_text(double seconds) {
    char out[32];
    std::snprintf(out, sizeof(out), "%.1f", seconds);
    return out;
}

// A position in the 16 kHz input.
std::string seconds_at(int64_t sample) {
    return seconds_text(static_cast<double>(sample) / kSampleRate);
}

// The audio's frames; none without a sample rate or channels, which the
// input checks reject.
int64_t frame_count(const runtime::AudioBuffer & audio) {
    return audio.sample_rate > 0 && audio.channels > 0
               ? static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels))
               : 0;
}

// A limit under kMinChunkSeconds would turn away even the shortest chunks
// audio_chunk_duration_sec allows; the spec's min matches.
double parse_max_pass_seconds(const runtime::SessionOptions & options) {
    const auto value =
        runtime::parse_finite_float_option(options.options, {"lfm2_audio.max_pass_duration_sec", "lfm2_audio.max_pass_seconds"}).value_or(kDefaultMaxPassSeconds);
    if (value < kMinChunkSeconds) {
        throw std::runtime_error("lfm2_audio.max_pass_duration_sec must be at least 1");
    }

    return value;
}

// lfm2_audio.cpu_repack: on the CPU, the kernels of ggml's CPU extra buffer
// types (repacked, or AMX) for the matmul weights they take.
bool parse_cpu_repack(const runtime::SessionOptions & options) {
    const auto value = runtime::find_option(options.options, {"lfm2_audio.cpu_repack"});
    return !value.has_value() || runtime::parse_bool_option(*value, "lfm2_audio.cpu_repack");
}

// lfm2_audio.conversation_cache_slots: how many conversations an S2S session
// keeps the questions and prefill blocks of, for their next turns.
size_t parse_conversation_cache_slots(const runtime::SessionOptions & options) {
    const int64_t slots = runtime::parse_i64_option(options.options, {"lfm2_audio.conversation_cache_slots"}).value_or(4);
    if (slots < 0) {
        throw std::runtime_error("lfm2_audio.conversation_cache_slots must be non-negative");
    }

    if (static_cast<std::uint64_t>(slots) > static_cast<std::uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("lfm2_audio.conversation_cache_slots is too large");
    }

    return static_cast<size_t>(slots);
}

// Whether two questions have the same samples, bit for bit.
bool same_samples(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

// Checked before the audio reaches the encoder, whose graph grows with the
// audio; nothing else stops a long pass before its allocation. The check is in
// samples, so audio exactly at a limit like 45.3 s passes, and the length is
// rounded up to a tenth, so audio just over the limit never reads as equal to
// it.
void require_one_pass(int64_t frames, int sample_rate, double limit, const std::string & what, const std::string & remedy) {
    if (frames > std::llround(limit * sample_rate)) {
        const auto tenths = (frames * 10 + sample_rate - 1) / sample_rate;
        throw runtime::CapacityError("LFM2-Audio encodes at most " + seconds_text(limit) +
                                     " s of audio in one pass (lfm2_audio.max_pass_duration_sec), and " + what + " " +
                                     seconds_text(static_cast<double>(tenths) / 10) + " s; " + remedy);
    }
}

runtime::SessionOptions validate_session_setup(
    const runtime::TaskSpec & task,
    runtime::SessionOptions options,
    const engine::model_spec::ModelContract & contract,
    runtime::VoiceTaskKind kind) {
    if (task.task != kind) {
        throw std::runtime_error("LFM2-Audio session created for the wrong task");
    }

    // Streaming is for the tasks that speak; ASR takes whole files.
    if (task.mode != runtime::RunMode::Offline && !(kind != runtime::VoiceTaskKind::Asr && task.mode == runtime::RunMode::Streaming)) {
        throw std::runtime_error(kind == runtime::VoiceTaskKind::Tts   ? "LFM2-Audio TTS runs offline or streaming"
                                 : kind == runtime::VoiceTaskKind::Asr ? "LFM2-Audio ASR runs offline only"
                                                                       : "LFM2-Audio s2s runs offline or streaming");
    }

    auto validation_options = options;
    validation_options.options.erase("lfm2_audio.max_pass_duration_sec");
    validation_options.options.erase("lfm2_audio.max_pass_seconds");
    runtime::validate_spec_backed_session_options(validation_options, contract, kFamily, kModelName);
    return options;
}

std::shared_ptr<const Lfm2AudioComponents> select_components(
    const std::shared_ptr<const Lfm2AudioAssets> & assets, const runtime::SessionOptions & options) {
    if (assets == nullptr) {
        throw std::runtime_error("LFM2-Audio session requires assets");
    }

    return load_lfm2_audio_components(
        *assets,
        runtime::find_option(options.options, {"lfm2_audio.model_gguf"}).value_or(""),
        runtime::find_option(options.options, {"lfm2_audio.mmproj_gguf"}).value_or(""));
}

// The spec lists one set of request options for both tasks, so the framework
// accepts either task's options; each session turns away the other's rather
// than ignoring them.
void reject_options(const runtime::TaskRequest & request, std::initializer_list<const char *> names, const char * task) {
    for (const char * name : names) {
        if (request.options.count(name) != 0) {
            throw runtime::InvalidRequestError(std::string("LFM2-Audio ") + task + " does not take request option " + name);
        }
    }
}

// Conversation history is for S2S. Other artifacts stay ignored, as before.
void reject_history(const runtime::TaskRequest & request, const char * task) {
    for (const auto & artifact : request.input_artifacts) {
        if (artifact.id.rfind("lfm2_audio.", 0) == 0) {
            throw runtime::InvalidRequestError(std::string("LFM2-Audio ") + task + " takes no conversation history, so no " + artifact.id +
                                               " artifact");
        }
    }
}

// A turn ends on end-of-audio, or at max_tokens, where its speech is cut off
// and kept, with a warning, as liquid-audio's generate_sequential keeps what
// it generated when max_new_tokens runs out. Either way it needs speech.
// Returns whether max_tokens cut it.
bool check_turn_end(const Lfm2SpeechGenerator & generator, size_t frames, size_t turn, size_t turns, int64_t max_frames) {
    if (frames == 0) {
        throw std::runtime_error(generator.ended() ? "LFM2-Audio produced no speech for the text"
                                                   : "LFM2-Audio reached max_tokens before any speech for the text; raise max_tokens");
    }

    if (generator.ended()) {
        return false;
    }

    const std::string which = turns > 1 ? " of text chunk " + std::to_string(turn + 1) + " of " + std::to_string(turns) : "";
    warn("the speech" + which + " reached max_tokens=" + std::to_string(max_frames) +
         " and is cut off there; raise max_tokens or lower text_chunk_size for the rest");
    return true;
}

// A reply ends on <|im_end|>, or at max_tokens, where its text and speech are
// cut off and kept, with a warning, as liquid-audio's generate_interleaved
// keeps what it generated when max_new_tokens runs out. Either way it needs
// speech. Returns whether max_tokens cut it.
std::shared_ptr<const Lfm2AudioOutputComponents> select_output_components(
    const std::shared_ptr<const Lfm2AudioAssets> & assets,
    const std::shared_ptr<const Lfm2AudioComponents> & components,
    const runtime::SessionOptions & options) {
    return load_lfm2_audio_output_components(
        *assets,
        *components,
        runtime::find_option(options.options, {"lfm2_audio.vocoder_gguf"}).value_or(""),
        runtime::find_option(options.options, {"lfm2_audio.detokenizer_gguf"}).value_or(""));
}

// The published checkpoints are single-language; the ASR and TTS system
// prompts are the ones each was trained with (liquid-audio README /
// README_JP). A GGUF without general.languages gets the English prompts, the
// base model's.
std::string model_language(const Lfm2AudioComponents & components) {
    const auto & languages = components.languages;
    const bool en = std::find(languages.begin(), languages.end(), "en") != languages.end();
    const bool ja = std::find(languages.begin(), languages.end(), "ja") != languages.end();
    if (en != ja) {
        return ja ? "ja" : "en";
    }

    if (languages.empty()) {
        return "en";
    }

    std::string listed;
    for (const auto & language : languages) {
        listed += (listed.empty() ? "" : ", ") + language;
    }

    throw std::runtime_error("LFM2-Audio has prompts for en or ja checkpoints, not general.languages = [" + listed + "]");
}

bool is_ascii_alnum(unsigned char value) {
    return (value >= '0' && value <= '9') || (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z');
}

// Chunk transcripts are joined with a space only after ASCII text and before
// an ASCII word, so Japanese text stays unbroken.
void append_chunk_text(std::string & merged, std::string chunk) {
    chunk = io::trim_ascii_whitespace(std::move(chunk));
    if (chunk.empty()) {
        return;
    }

    if (!merged.empty() && static_cast<unsigned char>(merged.back()) < 0x80 &&
        is_ascii_alnum(static_cast<unsigned char>(chunk.front()))) {
        merged.push_back(' ');
    }

    merged += chunk;
}

// Fixed chunks of audio_chunk_seconds. A short tail and the chunk before it
// split their audio evenly; joined, they would pass audio_chunk_seconds.
std::vector<runtime::TimeSpan> plan_fixed_chunks(int64_t samples, int64_t chunk_samples) {
    std::vector<runtime::TimeSpan> spans;
    for (const auto & chunk : audio::plan_audio_chunks(samples, {chunk_samples, chunk_samples})) {
        spans.push_back({chunk.output_start_sample, chunk.output_start_sample + chunk.valid_samples});
    }

    const auto min_samples = static_cast<int64_t>(kMinChunkSeconds * kSampleRate);
    if (spans.size() > 1 && spans.back().end_sample - spans.back().start_sample < min_samples) {
        auto & before = spans[spans.size() - 2];
        before.end_sample = (before.start_sample + samples) / 2;
        spans.back().start_sample = before.end_sample;
    }

    return spans;
}

std::filesystem::path default_vad_model_path() {
    return std::filesystem::path("assets") / "framework" / "models" / "silero_vad";
}

// Set LFM2_AUDIO_DUMP_DIR to write each stage (features, adapter output,
// prompt ids, first logits, tokens) as raw f32/i32 for stage-by-stage
// comparison against the reference implementation.
void debug_dump(const char * name, const void * data, size_t bytes) {
    const char * dir = std::getenv("LFM2_AUDIO_DUMP_DIR");
    if (dir == nullptr || *dir == '\0') {
        return;
    }

    std::ofstream(std::string(dir) + "/" + name, std::ios::binary).write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
}

}  // namespace

Lfm2AudioSession::Lfm2AudioSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const Lfm2AudioAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(validate_session_setup(task, std::move(options), require_contract(contract), runtime::VoiceTaskKind::Asr)),
      task_(std::move(task)),
      assets_(std::move(assets)),
      contract_(std::move(contract)),
      cpu_repack_(parse_cpu_repack(RuntimeSessionBase::options())),
      components_(select_components(assets_, RuntimeSessionBase::options())),
      tokenizer_(components_->vocabulary),
      features_(components_->encoder.n_mels, execution_context().config().threads),
      encoder_(components_->mmproj, components_->encoder, execution_context(), cpu_repack_),
      backbone_(components_->model, components_->backbone, execution_context(), nullptr, 0, 0, cpu_repack_),
      language_(model_language(*components_)),
      prompt_(make_lfm2_asr_prompt(tokenizer_, language_)),
      vad_model_path_(runtime::find_option(RuntimeSessionBase::options().options, {"lfm2_audio.vad_model_path"})
                          .value_or(default_vad_model_path().string())),
      max_pass_seconds_(parse_max_pass_seconds(RuntimeSessionBase::options())) {
    components_->model->release_storage();
    components_->mmproj->release_storage();
}

Lfm2AudioSession::~Lfm2AudioSession() = default;

std::string Lfm2AudioSession::family() const {
    return kFamily;
}

runtime::VoiceTaskKind Lfm2AudioSession::task_kind() const {
    return task_.task;
}

runtime::RunMode Lfm2AudioSession::run_mode() const {
    return task_.mode;
}

void Lfm2AudioSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void)request;
    mark_prepared();
}

Lfm2AudioSession::RequestOptions Lfm2AudioSession::parse_request_options(const runtime::TaskRequest & request) const {
    auto validation_options = request.options;
    validation_options.erase("audio_chunk_duration_sec");
    validation_options.erase("audio_chunk_seconds");
    runtime::validate_spec_backed_request_options(validation_options, require_contract(contract_), kModelName);
    reject_options(request,
        {"temperature", "top_k", "seed", "text_temperature", "text_top_k", "text_chunk_mode", "text_chunk_size", "stream_frames_per_event",
         "return_codes"},
        "ASR");
    reject_history(request, "ASR");

    RequestOptions out;
    out.max_tokens = runtime::parse_positive_i64_option(request.options, {"max_tokens"}, out.max_tokens);

    if (const auto language = runtime::find_option(request.options, {"language"});
        language.has_value() && *language != "auto" && *language != language_) {
        throw std::runtime_error("this LFM2-Audio checkpoint transcribes " + language_ + ", not " + *language);
    }

    return out;
}

// liquid-audio transcribes the whole input in one pass, which holds up to
// about a minute in English and about 40 s in Japanese: on joined LibriSpeech
// test-clean clips its WER is 2% at 60 s, 9% at 90 s (dropped words) and over
// 80% from 120 s (repetition); on joined Japanese clips it starts to skip
// whole sentences between 41 and 45 s.
// auto therefore keeps input that fits one chunk whole, like liquid-audio,
// and splits longer audio at pauses found by the bundled Silero VAD, as the
// other ASR families do (vad mode forces that). In English that stays at 1-2%
// WER up to 180 s; silence between the spans is not transcribed, so it cannot
// come back as words. fixed cuts at the chunk length regardless (1.8-2.9%:
// words get cut).
std::vector<runtime::TimeSpan> Lfm2AudioSession::plan_chunks(
    const runtime::TaskRequest & request, const std::vector<float> & samples) {
    const auto total = static_cast<int64_t>(samples.size());
    const auto mode = audio::parse_audio_chunk_mode(request.options);
    if (mode == audio::AudioChunkMode::None) {
        require_one_pass(total, kSampleRate, max_pass_seconds_, "the input is", "use audio_chunk_mode=auto, or raise the limit");
        return {{0, total}};
    }

    if (mode != audio::AudioChunkMode::Auto && mode != audio::AudioChunkMode::Fixed && mode != audio::AudioChunkMode::Vad) {
        throw std::runtime_error("LFM2-Audio supports audio_chunk_mode=auto, fixed, vad, or none");
    }

    // A limit under the default chunk length lowers the default, so that
    // requests which leave audio_chunk_seconds alone still run.
    const auto requested = audio::parse_audio_chunk_seconds_override(request.options);
    const double seconds = requested ? *requested : std::min<double>(kDefaultChunkSeconds, max_pass_seconds_);
    if (!std::isfinite(seconds) || seconds < kMinChunkSeconds) {
        throw std::runtime_error("LFM2-Audio audio_chunk_duration_sec must be at least 1");
    }

    const auto chunk_samples = static_cast<int64_t>(std::llround(seconds * kSampleRate));
    // Before any chunk is planned or encoded, so vad is held to the setting
    // rather than to its spans.
    require_one_pass(std::min(total, chunk_samples), kSampleRate, max_pass_seconds_, "audio_chunk_duration_sec allows chunks of up to",
                     "lower audio_chunk_duration_sec, or raise the limit");
    if (mode == audio::AudioChunkMode::Auto && total <= chunk_samples) {
        return {{0, total}};
    }

    // Without the bundled model, auto falls back to fixed chunks; vad requires it.
    if (mode == audio::AudioChunkMode::Fixed ||
        (mode == audio::AudioChunkMode::Auto && !io::is_existing_directory(vad_model_path_))) {
        return plan_fixed_chunks(total, chunk_samples);
    }

    const audio::VadAudioChunkOptions options{chunk_samples, kSampleRate / 2, kSampleRate / 4};
    return audio::plan_vad_audio_chunks(runtime::AudioBuffer{kSampleRate, 1, samples}, vad_session(), options);
}

runtime::IOfflineVoiceTaskSession & Lfm2AudioSession::vad_session() {
    if (vad_session_ == nullptr) {
        runtime::ModelLoadRequest load_request;
        load_request.model_path = vad_model_path_;
        vad_model_ = runtime::load_vad_model(load_request);
        auto session = vad_model_->create_task_session(
            {runtime::VoiceTaskKind::Vad, runtime::RunMode::Offline},
            runtime::SessionOptions{RuntimeSessionBase::options().backend, {}});
        auto * offline = dynamic_cast<runtime::IOfflineVoiceTaskSession *>(session.get());
        if (offline == nullptr) {
            throw std::runtime_error("LFM2-Audio VAD session does not support offline execution");
        }

        session.release();
        vad_session_.reset(offline);
    }

    return *vad_session_;
}

// A transcript that reaches max_tokens is cut off there, at the last whole
// character, and kept, with a warning, as liquid-audio's generate_sequential
// keeps what it generated when max_new_tokens runs out; the other chunks go on.
std::string Lfm2AudioSession::transcribe(
    const std::vector<float> & samples, const runtime::TimeSpan & span, const RequestOptions & options) {
    const std::vector<float> chunk(samples.begin() + span.start_sample, samples.begin() + span.end_sample);
    const auto features = features_.extract(chunk);
    debug_dump("mel.f32", features.values.data(), features.values.size() * sizeof(float));

    const auto audio = encoder_.encode(features);
    debug_dump("adapter.f32", audio.values.data(), audio.values.size() * sizeof(float));

    const auto prompt = prompt_.with_audio(audio.tokens);
    debug_dump("prompt_ids.i32", prompt.input_ids.data(), prompt.input_ids.size() * sizeof(int32_t));

    const auto result = backbone_.generate(prompt, audio, {options.max_tokens, prompt_.stop_token_ids});
    if (!result.stopped) {
        reached_max_tokens_ = true;
        warn("the transcript of " + seconds_at(span.start_sample) + "-" + seconds_at(span.end_sample) + " s reached max_tokens=" +
             std::to_string(options.max_tokens) + " and is cut off there; raise max_tokens for the rest");
    }

    debug_dump("prefill_logits.f32", result.prefill_logits.data(), result.prefill_logits.size() * sizeof(float));
    debug_dump("tokens.i32", result.tokens.data(), result.tokens.size() * sizeof(int32_t));
    debug::trace_log_scalar("lfm2_audio.session.audio_tokens", audio.tokens);
    debug::trace_log_scalar("lfm2_audio.session.generated_tokens", static_cast<int64_t>(result.tokens.size()));

    // Byte-level tokens need not end on a character boundary, and the next
    // chunk cannot complete one: end the text at its last whole character.
    // Bytes that can never make a character become U+FFFD (lfm2_take_text).
    auto bytes = tokenizer_.decode(result.tokens);
    return lfm2_take_text(bytes);
}

bool Lfm2AudioSession::reached_max_tokens() const {
    return reached_max_tokens_;
}

runtime::TaskResult Lfm2AudioSession::run(const runtime::TaskRequest & request) {
    require_prepared("LFM2-Audio run()");
    reached_max_tokens_ = false;
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("LFM2-Audio run() requires audio_input");
    }

    const auto wall_start = std::chrono::steady_clock::now();
    const auto options = parse_request_options(request);
    const auto samples = lfm2_audio_mono_16k(*request.audio_input);

    std::string text;
    for (const auto & span : plan_chunks(request, samples)) {
        append_chunk_text(text, transcribe(samples, span, options));
    }

    runtime::TaskResult result;
    result.text_output = runtime::Transcript{text, language_};
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

Lfm2AudioTtsSession::Lfm2AudioTtsSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const Lfm2AudioAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(validate_session_setup(task, std::move(options), require_contract(contract), runtime::VoiceTaskKind::Tts)),
      task_(std::move(task)),
      assets_(std::move(assets)),
      contract_(std::move(contract)),
      cpu_repack_(parse_cpu_repack(RuntimeSessionBase::options())),
      components_(select_components(assets_, RuntimeSessionBase::options())),
      output_(select_output_components(assets_, components_, RuntimeSessionBase::options())),
      tokenizer_(components_->vocabulary),
      backbone_(
          components_->model,
          components_->backbone,
          execution_context(),
          components_->mmproj,
          output_->depthformer.codebooks,
          output_->depthformer.audio_vocab_size,
          cpu_repack_),
      depthformer_(output_->vocoder, output_->depthformer, execution_context(), cpu_repack_),
      detokenizer_(
          output_->detokenizer,
          output_->vocoder,
          output_->detokenizer_config,
          execution_context(),
          Lfm2DetokenizerRuntime::kDefaultChunkFrames,
          cpu_repack_),
      language_(model_language(*components_)) {
    components_->model->release_storage();
    components_->mmproj->release_storage();
    output_->vocoder->release_storage();
    output_->detokenizer->release_storage();
}

Lfm2AudioTtsSession::~Lfm2AudioTtsSession() = default;

std::string Lfm2AudioTtsSession::family() const {
    return kFamily;
}

runtime::VoiceTaskKind Lfm2AudioTtsSession::task_kind() const {
    return task_.task;
}

runtime::RunMode Lfm2AudioTtsSession::run_mode() const {
    return task_.mode;
}

void Lfm2AudioTtsSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void)request;
    mark_prepared();
}

Lfm2AudioTtsSession::RequestOptions Lfm2AudioTtsSession::parse_request(const runtime::TaskRequest & request) const {
    if (!request.text_input.has_value() || io::trim_ascii_whitespace(request.text_input->text).empty()) {
        throw std::runtime_error("LFM2-Audio TTS requires text_input");
    }

    reject_options(request, {"audio_chunk_mode", "audio_chunk_duration_sec", "audio_chunk_seconds", "text_temperature", "text_top_k", "return_codes"}, "TTS");
    runtime::validate_spec_backed_request_options(request.options, require_contract(contract_), kModelName);
    reject_history(request, "TTS");

    std::string language = request.text_input->language;
    if (const auto option = runtime::find_option(request.options, {"language"})) {
        language = *option;
    }

    if (!language.empty() && language != "auto" && language != language_) {
        throw std::runtime_error("this LFM2-Audio checkpoint speaks " + language_ + ", not " + language);
    }

    std::string voice;
    if (request.voice.has_value() && request.voice->speaker.has_value()) {
        if (request.voice->speaker->audio.has_value()) {
            throw std::runtime_error("LFM2-Audio speaks with its built-in voices and does not clone reference audio");
        }

        voice = request.voice->speaker->cached_voice_id.value_or("");
    }

    RequestOptions out;
    out.system_prompt = lfm2_tts_system_prompt(language_, voice);
    out.speech.max_frames = runtime::parse_positive_i64_option(request.options, {"max_tokens"}, kDefaultMaxFrames);

    auto & sampling = out.speech.sampling;
    sampling.temperature = runtime::parse_finite_float_option(request.options, {"temperature"}).value_or(sampling.temperature);
    sampling.top_k = runtime::parse_int_option(request.options, {"top_k"}).value_or(static_cast<int>(sampling.top_k));
    if (sampling.temperature < 0.0f || sampling.top_k < 0) {
        throw std::runtime_error("LFM2-Audio temperature and top_k must not be negative");
    }

    out.text_chunk_size = text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    out.text_chunk_mode = text::parse_text_chunk_mode_override(request.options)
                              .value_or(language_ == "ja" ? text::TextChunkMode::Japanese : text::TextChunkMode::Default);
    out.stream_frames_per_event =
        runtime::parse_positive_i64_option(request.options, {"stream_frames_per_event"}, kDefaultStreamFramesPerEvent);
    out.seed = runtime::parse_u64_option(request.options, {"seed"}).value_or(runtime::random_u64_seed());

    // Each chunk is its own turn with the same voice prompt, as liquid-audio
    // would speak it.
    for (const auto & chunk : runtime::chunk_text_request(request, out.text_chunk_size, out.text_chunk_mode)) {
        out.texts.push_back(chunk.text_input->text);
    }

    return out;
}

// Turn i samples with seed + i, so each chunk is reproducible on its own and
// streaming gives the frames offline does.
std::unique_ptr<Lfm2SpeechGenerator> Lfm2AudioTtsSession::start_turn(const RequestOptions & options, size_t turn) {
    auto prompt = make_lfm2_tts_prompt(tokenizer_, options.system_prompt, options.texts.at(turn));
    debug_dump("prompt_ids.i32", prompt.input_ids.data(), prompt.input_ids.size() * sizeof(int32_t));

    auto speech_options = options.speech;
    speech_options.sampling.seed = options.seed + turn;
    return std::make_unique<Lfm2SpeechGenerator>(
        backbone_, depthformer_, tokenizer_, std::move(prompt), output_->depthformer.end_of_audio(), speech_options);
}

std::vector<float> Lfm2AudioTtsSession::speak(const RequestOptions & options, size_t turn) {
    const auto generator = start_turn(options, turn);
    std::vector<std::vector<int32_t>> frames;
    std::vector<int32_t> codes;
    while (auto frame = generator->next_frame()) {
        codes.insert(codes.end(), frame->begin(), frame->end());
        frames.push_back(std::move(*frame));
    }

    if (check_turn_end(*generator, frames.size(), turn, options.texts.size(), options.speech.max_frames)) {
        reached_max_tokens_ = true;
    }

    debug_dump("audio_codes.i32", codes.data(), codes.size() * sizeof(int32_t));
    debug::trace_log_scalar("lfm2_audio.session.audio_frames", static_cast<int64_t>(frames.size()));
    return detokenizer_.decode(frames);
}

bool Lfm2AudioTtsSession::reached_max_tokens() const {
    return reached_max_tokens_;
}

runtime::TaskResult Lfm2AudioTtsSession::run(const runtime::TaskRequest & request) {
    require_prepared("LFM2-Audio run()");
    reached_max_tokens_ = false;
    const auto wall_start = std::chrono::steady_clock::now();
    const auto options = parse_request(request);

    runtime::AudioBuffer audio;
    for (size_t turn = 0; turn < options.texts.size(); ++turn) {
        runtime::append_audio_buffer(audio, runtime::AudioBuffer{output_->detokenizer_config.sample_rate, 1, speak(options, turn)});
    }

    runtime::TaskResult result;
    result.audio_output = std::move(audio);
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

// One turn at a time: its frames, decoded as they come by the detokenizer's
// stream, and the ISTFT that turns their rows into final samples.
struct Lfm2AudioTtsSession::Stream {
    RequestOptions options;
    size_t next_turn = 0;
    std::unique_ptr<Lfm2SpeechGenerator> generator;
    size_t turn_frames = 0;
    std::unique_ptr<Lfm2StreamingIstft> istft;
    runtime::AudioBuffer audio;  // everything emitted
};

runtime::StreamingPolicy Lfm2AudioTtsSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::None;
    policy.output = runtime::StreamingOutputKind::PullEvents;
    return policy;
}

void Lfm2AudioTtsSession::start_stream(const runtime::TaskRequest & request) {
    require_prepared("LFM2-Audio start_stream()");
    reached_max_tokens_ = false;
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("LFM2-Audio start_stream() needs a streaming session");
    }

    reset();
    auto stream = std::make_unique<Stream>();
    stream->options = parse_request(request);
    stream_ = std::move(stream);
}

// The detokenizer is causal and the ISTFT only waits for the windows that
// overlap a sample, so each event's samples are final: the events add up to
// what run() returns for the same seed.
std::optional<runtime::StreamEvent> Lfm2AudioTtsSession::next_stream_event() {
    if (stream_ == nullptr) {
        throw std::runtime_error("LFM2-Audio streaming has not been started");
    }

    auto & st = *stream_;
    const auto & config = output_->detokenizer_config;
    std::vector<float> samples;
    while (samples.empty()) {
        if (st.generator == nullptr) {
            if (st.next_turn == st.options.texts.size()) {
                return std::nullopt;
            }

            st.generator = start_turn(st.options, st.next_turn++);
            st.turn_frames = 0;
            detokenizer_.start_stream();
            st.istft = std::make_unique<Lfm2StreamingIstft>(detokenizer_.window(), config.hop_length);
        }

        std::vector<std::vector<int32_t>> frames;
        bool turn_over = false;
        while (frames.size() < static_cast<size_t>(st.options.stream_frames_per_event)) {
            auto frame = st.generator->next_frame();
            if (!frame.has_value()) {
                turn_over = true;
                break;
            }

            frames.push_back(std::move(*frame));
        }

        if (!frames.empty()) {
            st.turn_frames += frames.size();
            const auto rows = static_cast<int64_t>(frames.size()) * config.upsample;
            samples = st.istft->push(detokenizer_.stream(frames), rows);
        }

        if (turn_over) {
            if (check_turn_end(*st.generator, st.turn_frames, st.next_turn - 1, st.options.texts.size(), st.options.speech.max_frames)) {
                reached_max_tokens_ = true;
            }

            const auto rest = st.istft->finish();
            samples.insert(samples.end(), rest.begin(), rest.end());
            st.generator.reset();
        }
    }

    runtime::AudioBuffer chunk{config.sample_rate, 1, std::move(samples)};
    runtime::append_audio_buffer(st.audio, chunk);
    runtime::StreamEvent event;
    event.audio_output = std::move(chunk);
    return event;
}

void Lfm2AudioTtsSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
    // PullEvents drivers forward what next_stream_event returns; pushing here
    // too would deliver every event twice.
    (void)sink;
}

runtime::TaskResult Lfm2AudioTtsSession::finish_stream() {
    if (stream_ == nullptr) {
        throw std::runtime_error("LFM2-Audio streaming has not been started");
    }

    runtime::TaskResult result;
    result.audio_output = std::move(stream_->audio);
    reset();
    return result;
}

void Lfm2AudioTtsSession::reset() {
    stream_.reset();
}

runtime::StreamEvent Lfm2AudioTtsSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    (void)chunk;
    throw std::runtime_error("LFM2-Audio TTS does not take streamed audio input");
}

runtime::TaskResult Lfm2AudioTtsSession::finalize() {
    return {};
}

Lfm2AudioChatSession::Lfm2AudioChatSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const Lfm2AudioAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(
          validate_session_setup(task, std::move(options), require_contract(contract), runtime::VoiceTaskKind::SpeechToSpeech)),
      task_(std::move(task)),
      assets_(std::move(assets)),
      contract_(std::move(contract)),
      cpu_repack_(parse_cpu_repack(RuntimeSessionBase::options())),
      conversation_slots_(parse_conversation_cache_slots(RuntimeSessionBase::options())),
      components_(select_components(assets_, RuntimeSessionBase::options())),
      output_(select_output_components(assets_, components_, RuntimeSessionBase::options())),
      tokenizer_(components_->vocabulary),
      features_(components_->encoder.n_mels, execution_context().config().threads),
      encoder_(components_->mmproj, components_->encoder, execution_context(), cpu_repack_),
      backbone_(
          components_->model,
          components_->backbone,
          execution_context(),
          components_->mmproj,
          output_->depthformer.codebooks,
          output_->depthformer.audio_vocab_size,
          cpu_repack_,
          conversation_slots_),
      depthformer_(output_->vocoder, output_->depthformer, execution_context(), cpu_repack_),
      detokenizer_(
          output_->detokenizer,
          output_->vocoder,
          output_->detokenizer_config,
          execution_context(),
          Lfm2DetokenizerRuntime::kDefaultChunkFrames,
          cpu_repack_),
      language_(model_language(*components_)),
      max_pass_seconds_(parse_max_pass_seconds(RuntimeSessionBase::options())) {
    components_->model->release_storage();
    components_->mmproj->release_storage();
    output_->vocoder->release_storage();
    output_->detokenizer->release_storage();
    bind_chat("LFM2-Audio", [this](const runtime::TaskRequest & request, chat::HistoryMode history_mode) {
        require_prepared("LFM2-Audio chat");
        auto options = std::make_shared<RequestOptions>(parse_request(request));
        options->retain_questions = options->return_codes || history_mode == chat::HistoryMode::Conversation;
        chat::UserTurn turn;
        turn.max_audio_seconds = max_pass_seconds_;
        turn.output_sample_rate = output_->detokenizer_config.sample_rate;
        turn.language = language_;
        turn.history_format = lfm2_history_format(reply_checkpoint());
        turn.max_reply_steps = options->reply.max_steps;
        turn.return_codes = options->return_codes;
        turn.history_prefix_steps = [this, options](const std::vector<chat::ConversationTurn> & history) {
            return options->text_steps = lfm2_chat_prompt_text_steps(tokenizer_, options->system_prompt, history);
        };
        turn.reply = [this, options](const runtime::AudioBuffer & input, const std::vector<chat::ConversationTurn> & history) {
            return bind_reply(start_reply(*options, input, history), *options);
        };
        return turn;
    });
}

Lfm2AudioChatSession::~Lfm2AudioChatSession() { chat::TaskSession::reset(); }

std::string Lfm2AudioChatSession::family() const {
    return kFamily;
}

runtime::VoiceTaskKind Lfm2AudioChatSession::task_kind() const {
    return task_.task;
}

runtime::RunMode Lfm2AudioChatSession::run_mode() const {
    return task_.mode;
}

void Lfm2AudioChatSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void)request;
    mark_prepared();
}

Lfm2AudioChatSession::RequestOptions Lfm2AudioChatSession::parse_request(const runtime::TaskRequest & request) const {
    reject_options(request, {"audio_chunk_mode", "audio_chunk_duration_sec", "audio_chunk_seconds", "text_chunk_mode", "text_chunk_size"}, "s2s");
    runtime::validate_spec_backed_request_options(request.options, require_contract(contract_), kModelName);

    if (const auto language = runtime::find_option(request.options, {"language"});
        language.has_value() && *language != "auto" && *language != language_) {
        throw std::runtime_error("this LFM2-Audio checkpoint speaks " + language_ + ", not " + *language);
    }

    if (request.voice.has_value() && request.voice->speaker.has_value()) {
        throw std::runtime_error("LFM2-Audio s2s replies in the checkpoint's voice and takes no voice");
    }

    RequestOptions out;
    out.system_prompt = kLfm2ChatSystemPrompt;
    if (request.text_input.has_value() && !io::trim_ascii_whitespace(request.text_input->text).empty()) {
        out.system_prompt = request.text_input->text;
    }

    out.reply.text_steps = output_->interleave.text_steps;
    out.reply.audio_steps = output_->interleave.audio_steps;
    out.reply.max_steps = runtime::parse_positive_i64_option(request.options, {"max_tokens"}, kDefaultReplySteps);

    auto & sampling = out.reply.sampling;
    sampling.temperature = runtime::parse_finite_float_option(request.options, {"temperature"}).value_or(sampling.temperature);
    sampling.top_k = runtime::parse_int_option(request.options, {"top_k"}).value_or(static_cast<int>(sampling.top_k));
    if (sampling.temperature < 0.0f || sampling.top_k < 0) {
        throw std::runtime_error("LFM2-Audio temperature and top_k must not be negative");
    }

    auto & text_sampling = out.reply.text_sampling;
    text_sampling.temperature = runtime::parse_finite_float_option(request.options, {"text_temperature"}).value_or(text_sampling.temperature);
    text_sampling.top_k = runtime::parse_int_option(request.options, {"text_top_k"}).value_or(static_cast<int>(text_sampling.top_k));
    if (text_sampling.temperature < 0.0f || text_sampling.top_k < 0) {
        throw std::runtime_error("LFM2-Audio text_temperature and text_top_k must not be negative");
    }

    sampling.seed = runtime::parse_u64_option(request.options, {"seed"}).value_or(runtime::random_u64_seed());
    out.stream_frames_per_event =
        runtime::parse_positive_i64_option(request.options, {"stream_frames_per_event"}, kDefaultStreamFramesPerEvent);
    if (const auto value = runtime::find_option(request.options, {"return_codes"})) {
        // The parser's only runtime_error is a value it does not read.
        try {
            out.return_codes = runtime::parse_bool_option(*value, "return_codes");
        } catch (const std::runtime_error & error) {
            throw runtime::InvalidRequestError(error.what());
        }
    }

    return out;
}

Lfm2ReplyCheckpoint Lfm2AudioChatSession::reply_checkpoint() const {
    return {language_, output_->depthformer.codebooks, output_->depthformer.audio_vocab_size, components_->backbone.vocab_size};
}

// The user's turn goes in the way ASR takes its audio, under the chat system
// prompt and after the earlier turns, as liquid-audio's ChatState holds them
// (chat.h). Each question goes through the encoder on its own: liquid-audio
// encodes them in one batch, whose padding the encoder masks.
//
// An earlier question with the very samples of one the session kept for a
// conversation takes that encoder output: on a session, the encoder gives the
// same bits for the same samples. The new question is always encoded.
std::unique_ptr<Lfm2InterleavedGenerator> Lfm2AudioChatSession::start_reply(
    const RequestOptions & options, const runtime::AudioBuffer & audio, const std::vector<chat::ConversationTurn> & conversation_history) {
    reused_questions_ = 0;
    // A request that may have a next turn keeps its questions for it, if the
    // session keeps any conversations.
    const bool conversation = conversation_slots_ > 0 && (!conversation_history.empty() || options.retain_questions);
    std::vector<EncodedQuestion> questions;
    // The prompt's steps so far: a conversation past the limit encodes no
    // more questions.
    int64_t steps = options.text_steps;
    Lfm2AudioEmbeddings embeddings;
    const auto encode = [&](const runtime::AudioBuffer & question, bool earlier) {
        auto samples = lfm2_audio_mono_16k(question);
        const EncodedQuestion * kept = nullptr;
        for (auto list = questions_.begin(); earlier && kept == nullptr && list != questions_.end(); ++list) {
            const auto found =
                std::find_if(list->begin(), list->end(), [&](const EncodedQuestion & q) { return same_samples(*q.samples, samples); });
            if (found != list->end()) {
                kept = &*found;
            }
        }

        EncodedQuestion entry;
        if (kept != nullptr) {
            entry = *kept;
            ++reused_questions_;
        } else {
            entry.encoded = std::make_shared<const Lfm2AudioEmbeddings>(encoder_.encode(features_.extract(samples)));
            entry.samples = std::make_shared<const std::vector<float>>(std::move(samples));
        }

        const auto & encoded = *entry.encoded;
        embeddings.hidden_size = encoded.hidden_size;
        embeddings.tokens += encoded.tokens;
        embeddings.values.insert(embeddings.values.end(), encoded.values.begin(), encoded.values.end());
        steps += encoded.tokens;
        const auto tokens = encoded.tokens;
        if (conversation) {
            questions.push_back(std::move(entry));
        }

        return tokens;
    };

    std::vector<Lfm2ChatTurn> history;
    for (const auto & turn : conversation_history) {
        history.push_back({encode(turn.question, true), turn.reply});
        require_lfm2_conversation_room(steps, true, options.reply.max_steps);
    }

    const auto question_tokens = encode(audio, false);
    auto prompt = make_lfm2_chat_prompt(tokenizer_, options.system_prompt, history, question_tokens);
    debug_dump("prompt_ids.i32", prompt.input_ids.data(), prompt.input_ids.size() * sizeof(int32_t));
    debug::trace_log_scalar("lfm2_audio.session.audio_tokens", embeddings.tokens);
    debug::trace_log_scalar("lfm2_audio.session.reused_questions", reused_questions_);

    // A first turn runs as S2S always has. A later one holds the whole
    // conversation, which the chunked prefill takes without the one-shot
    // prefill's prompt^2 attention scores.
    auto reply = options.reply;
    if (!history.empty()) {
        require_lfm2_conversation_room(static_cast<int64_t>(prompt.input_ids.size()), false, reply.max_steps);
        reply.prefill = Lfm2Prefill::Chunked;
    }

    if (conversation) {
        keep_questions(std::move(questions));
    }

    return std::make_unique<Lfm2InterleavedGenerator>(
        backbone_, depthformer_, tokenizer_, std::move(prompt), std::move(embeddings), output_->depthformer.end_of_audio(), reply);
}

// `questions`, the earlier ones then the new one, replace the kept questions
// of their conversation: of the kept lists whose last question they ask too
// (the turn before's new question, or this one when the turn runs again), the
// one that holds the most of the earlier ones, the most recent on a tie.
// Without one, they are a new conversation's, and the least recent beyond the
// slots is dropped.
void Lfm2AudioChatSession::keep_questions(std::vector<EncodedQuestion> questions) {
    const auto among = [](const EncodedQuestion & kept, auto first, auto last) {
        return std::any_of(first, last, [&](const EncodedQuestion & q) {
            return q.samples == kept.samples || same_samples(*q.samples, *kept.samples);
        });
    };
    const auto earlier = std::prev(questions.end());
    auto continued = questions_.end();
    std::ptrdiff_t most = -1;
    for (auto list = questions_.begin(); list != questions_.end(); ++list) {
        if (among(list->back(), questions.begin(), questions.end())) {
            const auto held = std::count_if(
                list->begin(), list->end(), [&](const EncodedQuestion & kept) { return among(kept, questions.begin(), earlier); });
            if (held > most) {
                most = held;
                continued = list;
            }
        }
    }

    if (continued != questions_.end()) {
        questions_.erase(continued);
    }

    questions_.insert(questions_.begin(), std::move(questions));
    if (questions_.size() > conversation_slots_) {
        questions_.pop_back();
    }
}

int64_t Lfm2AudioChatSession::resumed_prefill_steps() const {
    return backbone_.resumed_prefill_steps();
}

int64_t Lfm2AudioChatSession::reused_questions() const {
    return reused_questions_;
}

bool Lfm2AudioChatSession::reached_max_tokens() const {
    return reply_limited();
}

std::unique_ptr<chat::Reply> Lfm2AudioChatSession::bind_reply(std::shared_ptr<Lfm2InterleavedGenerator> generator, const RequestOptions & options) {
    auto source = std::move(generator);
    chat::ReplySource inference{
        [source] { return source->next(); },
        [source] { return source->ended(); },
        [this](const std::vector<int32_t> & tokens) { return tokenizer_.decode(tokens); }, {}};
    auto istft = std::make_shared<std::optional<Lfm2StreamingIstft>>();
    const auto & config = output_->detokenizer_config;
    chat::ReplyCodec audio{config.sample_rate, 1,
        [this](const chat::AudioCodes & frames) { return detokenizer_.decode(frames); },
        [this, istft] {
            detokenizer_.start_stream();
            istft->emplace(detokenizer_.window(), output_->detokenizer_config.hop_length);
        },
        [this, istft](const chat::AudioCodes & frames) {
            return istft->value().push(detokenizer_.stream(frames), static_cast<int64_t>(frames.size()) * output_->detokenizer_config.upsample);
        },
        [istft] { return istft->value().finish(); }};
    chat::ReplyConfig reply_config{output_->depthformer.end_of_audio(), static_cast<size_t>(options.stream_frames_per_event), language_, options.reply.max_steps};
    reply_config.publish_text_immediately = false;
    return std::make_unique<chat::Reply>(std::move(inference), std::move(audio), std::move(reply_config));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_lfm2_audio_loader() {
    class LoadedModel final : public runtime::ILoadedVoiceModel {
    public:
        explicit LoadedModel(std::shared_ptr<const Lfm2AudioAssets> assets)
            : assets_(std::move(assets)), contract_(runtime::require_model_contract(kFamily)) {}

        const runtime::ModelMetadata & metadata() const noexcept override { return contract_->metadata; }
        const runtime::CapabilitySet & capabilities() const noexcept override { return contract_->capabilities; }

        std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
            const runtime::TaskSpec & task, const runtime::SessionOptions & options) const override {
            if (task.task == runtime::VoiceTaskKind::Tts) {
                return std::make_unique<Lfm2AudioTtsSession>(task, options, assets_, contract_);
            }

            if (task.task == runtime::VoiceTaskKind::SpeechToSpeech) {
                return std::make_unique<Lfm2AudioChatSession>(task, options, assets_, contract_);
            }

            if (task.task != runtime::VoiceTaskKind::Asr) {
                throw std::runtime_error("LFM2-Audio supports the asr, tts and s2s tasks");
            }

            return std::make_unique<Lfm2AudioSession>(task, options, assets_, contract_);
        }

    private:
        std::shared_ptr<const Lfm2AudioAssets> assets_;
        std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    };

    // Not spec-backed: the package is a directory of several llama.cpp GGUFs
    // without an embedded model spec, so the loader is hand-written like
    // make_auk_loader (community_models/auk/session.cpp).
    class Loader final : public runtime::IVoiceModelLoader {
    public:
        std::string family() const override { return kFamily; }

        runtime::PackageLoadTarget resolve_package_load(const assets::ModelPackage & package) const override {
            runtime::PackageLoadTarget out{kFamily, package.root, {}, {}};
            if (package.format == "gguf") {
                for (const auto & file : package.files) {
                    if (file.extension() != ".gguf") continue;
                    const auto name = file.filename().string();
                    const char * option = "lfm2_audio.model_gguf";
                    if (name.rfind("mmproj-", 0) == 0) option = "lfm2_audio.mmproj_gguf";
                    else if (name.rfind("vocoder-", 0) == 0) option = "lfm2_audio.vocoder_gguf";
                    else if (name.rfind("tokenizer-", 0) == 0) option = "lfm2_audio.detokenizer_gguf";
                    out.session_options.emplace(option, file.generic_string());
                }
            }
            return out;
        }

        bool can_load(const runtime::ModelLoadRequest & request) const override {
            if (request.family_hint && *request.family_hint != kFamily) {
                return false;
            }

            // An incomplete package is still claimed, so load() reports what
            // is missing instead of the registry finding no loader at all.
            return has_lfm2_audio_component(request.model_path);
        }

        runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
            const auto assets = load_lfm2_audio_assets(request.model_path);
            const auto contract = runtime::require_model_contract(kFamily);

            runtime::ModelInspection inspection;
            inspection.model_root = assets->model_root;
            inspection.metadata = contract->metadata;
            inspection.capabilities = contract->capabilities;
            inspection.cli = contract->cli;
            inspection.discovered_configs =
                runtime::discover_named_assets(inspection.model_root, inspection.metadata.config_candidates);
            inspection.discovered_weights =
                runtime::discover_named_assets(inspection.model_root, inspection.metadata.weight_candidates);

            return inspection;
        }

        std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
            return std::make_unique<LoadedModel>(load_lfm2_audio_assets(request.model_path));
        }

        runtime::CapabilitySet advertised_capabilities() const override {
            return runtime::require_model_contract(kFamily)->capabilities;
        }
    };
    return std::make_shared<Loader>();
}

}  // namespace engine::community_models::lfm2_audio
