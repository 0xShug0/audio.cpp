#include "engine/models/fun_asr_nano/session.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/text.h"
#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace engine::models::fun_asr_nano {
namespace {

using Clock = std::chrono::steady_clock;
constexpr int64_t kDefaultMaxNewTokens = 512;
constexpr float kDefaultChunkSeconds = 30.0F;

std::shared_ptr<const FunAsrNanoAssets>
require_assets(std::shared_ptr<const FunAsrNanoAssets> assets) {
  if (assets == nullptr) {
    throw std::runtime_error("Fun-ASR-Nano session requires assets");
  }
  return assets;
}

void validate_weight_storage(assets::TensorStorageType storage,
                             const std::string &option) {
  if (storage == assets::TensorStorageType::Native ||
      storage == assets::TensorStorageType::F32 ||
      storage == assets::TensorStorageType::F16 ||
      storage == assets::TensorStorageType::BF16 ||
      storage == assets::TensorStorageType::Q8_0) {
    return;
  }
  throw std::runtime_error(option +
                           " supports only native, f32, f16, bf16, and q8_0");
}

assets::TensorStorageType
option_weight_type(const runtime::SessionOptions &options, const char *key,
                   assets::TensorStorageType fallback) {
  const auto value = options.options.find(key);
  return value == options.options.end()
             ? fallback
             : assets::parse_tensor_storage_type(value->second);
}

assets::TensorStorageType
decoder_weight_type(const runtime::SessionOptions &options,
                    core::BackendType backend_type) {
  const auto decoder = options.options.find("fun_asr_nano.decoder_weight_type");
  if (decoder != options.options.end()) {
    return assets::parse_tensor_storage_type(decoder->second);
  }
  auto shared = assets::TensorStorageType::Native;
  const auto shared_option = options.options.find("fun_asr_nano.weight_type");
  if (shared_option != options.options.end()) {
    shared = assets::parse_tensor_storage_type(shared_option->second);
  }
  // Native F16/Q8 decoder weights currently produce invalid logits with the
  // CUDA graph path. BF16 preserves the compact GPU representation and exact
  // transcript parity while encoder/adaptor weights keep the shared type.
  if (backend_type == core::BackendType::Cuda &&
      (shared == assets::TensorStorageType::Native ||
       shared == assets::TensorStorageType::F16 ||
       shared == assets::TensorStorageType::Q8_0)) {
    return assets::TensorStorageType::BF16;
  }
  return shared;
}

runtime::SessionOptions
validate_session_setup(const runtime::TaskSpec &task,
                       runtime::SessionOptions options) {
  if (task.task != runtime::VoiceTaskKind::Asr) {
    throw std::runtime_error("Fun-ASR-Nano only supports VoiceTaskKind::Asr");
  }
  if (task.mode != runtime::RunMode::Offline) {
    throw std::runtime_error(
        "Fun-ASR-Nano currently supports offline sessions");
  }
  const auto shared = option_weight_type(options, "fun_asr_nano.weight_type",
                                         assets::TensorStorageType::Native);
  validate_weight_storage(
      option_weight_type(options, "fun_asr_nano.encoder_weight_type", shared),
      "fun_asr_nano.encoder_weight_type");
  validate_weight_storage(
      option_weight_type(options, "fun_asr_nano.adaptor_weight_type", shared),
      "fun_asr_nano.adaptor_weight_type");
  validate_weight_storage(decoder_weight_type(options, options.backend.type),
                          "fun_asr_nano.decoder_weight_type");
  for (const auto &[key, value] : options.options) {
    (void)value;
    if (key.rfind("fun_asr_nano.", 0) == 0 &&
        key != "fun_asr_nano.weight_type" &&
        key != "fun_asr_nano.encoder_weight_type" &&
        key != "fun_asr_nano.adaptor_weight_type" &&
        key != "fun_asr_nano.decoder_weight_type" &&
        key != "fun_asr_nano.encoder_graph_arena_mb" &&
        key != "fun_asr_nano.adaptor_graph_arena_mb" &&
        key != "fun_asr_nano.decoder_prefill_graph_arena_mb" &&
        key != "fun_asr_nano.decoder_decode_graph_arena_mb" &&
        key != "fun_asr_nano.decoder_weight_context_mb") {
      throw std::runtime_error("unknown Fun-ASR-Nano session option: " + key);
    }
  }
  return options;
}

int64_t audio_frame_count(const runtime::AudioBuffer &audio) {
  if (audio.sample_rate <= 0) {
    throw std::runtime_error(
        "Fun-ASR-Nano audio requires a positive sample rate");
  }
  if (audio.channels <= 0) {
    throw std::runtime_error("Fun-ASR-Nano audio requires positive channels");
  }
  if (audio.samples.empty()) {
    throw std::runtime_error("Fun-ASR-Nano audio must not be empty");
  }
  if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
    throw std::runtime_error(
        "Fun-ASR-Nano audio samples must be divisible by channel count");
  }
  return static_cast<int64_t>(audio.samples.size() /
                              static_cast<size_t>(audio.channels));
}

bool ascii_word_boundary(const std::string &text, bool front) {
  if (text.empty()) {
    return false;
  }
  const unsigned char value =
      static_cast<unsigned char>(front ? text.front() : text.back());
  return (value >= '0' && value <= '9') || (value >= 'A' && value <= 'Z') ||
         (value >= 'a' && value <= 'z');
}

void append_chunk_text(std::string &merged, std::string chunk) {
  chunk = engine::io::trim_ascii_whitespace(std::move(chunk));
  if (chunk.empty()) {
    return;
  }
  if (!merged.empty() && ascii_word_boundary(merged, false) &&
      ascii_word_boundary(chunk, true)) {
    merged.push_back(' ');
  }
  merged += chunk;
}

void validate_request_options(
    const std::unordered_map<std::string, std::string> &options) {
  for (const auto &[key, value] : options) {
    (void)value;
    if (key != "language" && key != "enable_itn" && key != "max_tokens" &&
        key != "audio_chunk_mode" && key != "audio_chunk_seconds" &&
        key != "audio_chunk_duration_seconds" &&
        key != "audio_chunk_duration") {
      throw std::runtime_error("unknown Fun-ASR-Nano request option: " + key);
    }
  }
}

} // namespace

FunAsrNanoSession::FunAsrNanoSession(
    runtime::TaskSpec task, runtime::SessionOptions options,
    std::shared_ptr<const FunAsrNanoAssets> assets)
    : RuntimeSessionBase(validate_session_setup(task, options)), task_(task),
      assets_(require_assets(std::move(assets))),
      encoder_graph_arena_bytes_(runtime::parse_size_mb_option(
          options.options, {"fun_asr_nano.encoder_graph_arena_mb"},
          512ull * 1024ull * 1024ull)),
      adaptor_graph_arena_bytes_(runtime::parse_size_mb_option(
          options.options, {"fun_asr_nano.adaptor_graph_arena_mb"},
          128ull * 1024ull * 1024ull)),
      decoder_prefill_graph_arena_bytes_(runtime::parse_size_mb_option(
          options.options, {"fun_asr_nano.decoder_prefill_graph_arena_mb"},
          256ull * 1024ull * 1024ull)),
      decoder_decode_graph_arena_bytes_(runtime::parse_size_mb_option(
          options.options, {"fun_asr_nano.decoder_decode_graph_arena_mb"},
          128ull * 1024ull * 1024ull)),
      decoder_weight_context_bytes_(runtime::parse_size_mb_option(
          options.options, {"fun_asr_nano.decoder_weight_context_mb"},
          32ull * 1024ull * 1024ull)),
      encoder_weight_storage_type_(option_weight_type(
          options, "fun_asr_nano.encoder_weight_type",
          option_weight_type(options, "fun_asr_nano.weight_type",
                             assets::TensorStorageType::Native))),
      adaptor_weight_storage_type_(option_weight_type(
          options, "fun_asr_nano.adaptor_weight_type",
          option_weight_type(options, "fun_asr_nano.weight_type",
                             assets::TensorStorageType::Native))),
      decoder_weight_storage_type_(
          decoder_weight_type(options, execution_context().backend_type())),
      tokenizer_(assets_), frontend_(assets_->config.frontend),
      encoder_(assets_, execution_context(), encoder_graph_arena_bytes_,
               encoder_weight_storage_type_),
      adaptor_(assets_, execution_context(), adaptor_graph_arena_bytes_,
               adaptor_weight_storage_type_),
      prompt_builder_(assets_),
      decoder_(assets_, execution_context(), decoder_prefill_graph_arena_bytes_,
               decoder_decode_graph_arena_bytes_, decoder_weight_context_bytes_,
               decoder_weight_storage_type_) {}

FunAsrNanoSession::~FunAsrNanoSession() = default;

std::string FunAsrNanoSession::family() const { return "fun_asr_nano"; }

runtime::VoiceTaskKind FunAsrNanoSession::task_kind() const {
  return task_.task;
}

runtime::RunMode FunAsrNanoSession::run_mode() const { return task_.mode; }

void FunAsrNanoSession::prepare(
    const runtime::SessionPreparationRequest &request) {
  const auto started = Clock::now();
  if (!request.audio.has_value()) {
    throw std::runtime_error(
        "Fun-ASR-Nano prepare() requires an audio contract");
  }
  mark_prepared();
  debug::timing_log_scalar("fun_asr_nano.prepare_ms",
                           debug::elapsed_ms(started));
  debug::trace_log_scalar("fun_asr_nano.prepare.max_input_samples",
                          request.audio->max_input_samples);
}

runtime::TaskResult
FunAsrNanoSession::run(const runtime::TaskRequest &request) {
  require_prepared("Fun-ASR-Nano run()");
  validate_request_options(request.options);
  const auto chunks = audio_chunk_plan(request);
  if (chunks.empty()) {
    return run_single(make_request(request));
  }
  const auto &audio = *request.audio_input;
  if (chunks.size() == 1) {
    auto item = request;
    item.audio_input =
        engine::audio::slice_audio_buffer(audio, chunks.front().source_span);
    return run_single(make_request(item));
  }

  runtime::TaskResult merged;
  std::string text;
  for (const auto &chunk : chunks) {
    auto item = request;
    item.audio_input =
        engine::audio::slice_audio_buffer(audio, chunk.source_span);
    const auto result = run_single(make_request(item));
    if (result.text_output.has_value()) {
      append_chunk_text(text, result.text_output->text);
      if (!merged.text_output.has_value()) {
        merged.text_output =
            runtime::Transcript{"", result.text_output->language};
      }
    }
  }
  if (!merged.text_output.has_value()) {
    merged.text_output = runtime::Transcript{};
  }
  merged.text_output->text = std::move(text);
  return merged;
}

FunAsrNanoSession::AsrRequest
FunAsrNanoSession::make_request(const runtime::TaskRequest &request) const {
  if (!request.audio_input.has_value()) {
    throw std::runtime_error("Fun-ASR-Nano run() requires audio_input");
  }
  (void)audio_frame_count(*request.audio_input);

  AsrRequest out;
  out.audio = *request.audio_input;
  out.prompt.language = "auto";
  out.generation.max_new_tokens = kDefaultMaxNewTokens;
  if (request.text_input.has_value()) {
    out.prompt.prompt = request.text_input->text;
    if (!request.text_input->language.empty()) {
      out.prompt.language = request.text_input->language;
    }
  }
  if (const auto language =
          runtime::find_option(request.options, {"language"})) {
    out.prompt.language = *language;
  }
  if (const auto value =
          runtime::find_option(request.options, {"enable_itn"})) {
    out.prompt.enable_itn = runtime::parse_bool_option(*value, "enable_itn");
  }
  if (const auto value =
          runtime::parse_int_option(request.options, {"max_tokens"})) {
    if (*value <= 0) {
      throw std::runtime_error("Fun-ASR-Nano max_tokens must be positive");
    }
    out.generation.max_new_tokens = *value;
  }
  return out;
}

std::vector<FunAsrNanoSession::AudioChunkPlan>
FunAsrNanoSession::audio_chunk_plan(const runtime::TaskRequest &request) const {
  if (!request.audio_input.has_value()) {
    return {};
  }
  const auto mode = engine::audio::parse_audio_chunk_mode(request.options);
  if (mode == engine::audio::AudioChunkMode::None) {
    return {};
  }
  if (mode == engine::audio::AudioChunkMode::Vad ||
      mode == engine::audio::AudioChunkMode::QuietEnergy) {
    throw std::runtime_error(
        "Fun-ASR-Nano supports audio_chunk_mode=auto, fixed, or none");
  }

  const auto &audio = *request.audio_input;
  const int64_t frames = audio_frame_count(audio);
  const float seconds =
      engine::audio::parse_audio_chunk_seconds_override(request.options)
          .value_or(kDefaultChunkSeconds);
  if (!std::isfinite(seconds) || !(seconds > 0.0F)) {
    throw std::runtime_error(
        "Fun-ASR-Nano audio_chunk_seconds must be positive");
  }
  const double sample_count =
      static_cast<double>(seconds) * static_cast<double>(audio.sample_rate);
  if (sample_count >=
      static_cast<double>(std::numeric_limits<int64_t>::max())) {
    throw std::runtime_error("Fun-ASR-Nano audio_chunk_seconds is too large");
  }
  const int64_t samples = static_cast<int64_t>(std::llround(sample_count));
  if (samples <= 0) {
    throw std::runtime_error(
        "Fun-ASR-Nano audio_chunk_seconds produced an empty chunk");
  }
  const auto chunks = engine::audio::plan_audio_chunks(
      frames, {samples, samples, engine::audio::AudioChunkPadMode::Zero,
               engine::audio::AudioChunkTailAlignment::Start, 0});
  std::vector<AudioChunkPlan> plan;
  plan.reserve(chunks.size());
  for (const auto &chunk : chunks) {
    plan.push_back({{chunk.output_start_sample,
                     chunk.output_start_sample + chunk.valid_samples}});
  }
  return plan;
}

runtime::TaskResult FunAsrNanoSession::run_single(const AsrRequest &request) {
  const auto wall_start = Clock::now();

  const auto resample_start = Clock::now();
  const auto mono =
      engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
          request.audio.samples, request.audio.sample_rate,
          request.audio.channels, assets_->config.frontend.sample_rate);
  const auto resample_end = Clock::now();

  const auto frontend_start = Clock::now();
  const auto features =
      frontend_.extract(mono, assets_->config.frontend.sample_rate);
  const auto frontend_end = Clock::now();

  const auto encoder_start = Clock::now();
  const auto encoded = encoder_.encode(features);
  const auto encoder_end = Clock::now();

  std::vector<int32_t> mask(static_cast<size_t>(encoded.frames), 0);
  std::fill_n(mask.begin(), static_cast<size_t>(encoded.valid_frames), 1);
  const auto adaptor_start = Clock::now();
  const auto audio_embeddings = adaptor_.adapt(encoded, mask);
  const auto adaptor_end = Clock::now();

  const auto prompt_start = Clock::now();
  const auto prompt =
      prompt_builder_.build(request.prompt, audio_embeddings.tokens);
  const auto prompt_end = Clock::now();

  const auto decoder_start = Clock::now();
  const auto tokens =
      decoder_.generate(prompt, audio_embeddings, request.generation);
  const auto decoder_end = Clock::now();

  const auto decode_start = Clock::now();
  auto text =
      engine::io::trim_ascii_whitespace(tokenizer_.decode(tokens.token_ids));
  const auto decode_end = Clock::now();

  runtime::TaskResult result;
  result.text_output =
      runtime::Transcript{std::move(text), request.prompt.language == "auto"
                                               ? std::string{}
                                               : request.prompt.language};

  debug::timing_log_scalar("fun_asr_nano.resample_ms",
                           debug::elapsed_ms(resample_start, resample_end));
  debug::timing_log_scalar("fun_asr_nano.frontend_ms",
                           debug::elapsed_ms(frontend_start, frontend_end));
  debug::timing_log_scalar("fun_asr_nano.encoder_ms",
                           debug::elapsed_ms(encoder_start, encoder_end));
  debug::timing_log_scalar("fun_asr_nano.adaptor_ms",
                           debug::elapsed_ms(adaptor_start, adaptor_end));
  debug::timing_log_scalar("fun_asr_nano.prompt_ms",
                           debug::elapsed_ms(prompt_start, prompt_end));
  debug::timing_log_scalar("fun_asr_nano.decoder_ms",
                           debug::elapsed_ms(decoder_start, decoder_end));
  debug::timing_log_scalar("fun_asr_nano.decode_ms",
                           debug::elapsed_ms(decode_start, decode_end));
  debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(wall_start));
  debug::trace_log_scalar("fun_asr_nano.audio_input_frames",
                          audio_frame_count(request.audio));
  debug::trace_log_scalar("fun_asr_nano.frontend_frames", features.frames);
  debug::trace_log_scalar("fun_asr_nano.encoder_valid_frames",
                          encoded.valid_frames);
  debug::trace_log_scalar("fun_asr_nano.audio_tokens", audio_embeddings.tokens);
  debug::trace_log_scalar("fun_asr_nano.prompt_tokens",
                          prompt.input_ids.size());
  debug::trace_log_scalar("fun_asr_nano.generated_tokens",
                          tokens.token_ids.size());
  return result;
}

} // namespace engine::models::fun_asr_nano
