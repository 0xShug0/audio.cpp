#include "engine/models/irodori_tts/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/text/chunking.h"
#include "engine/models/irodori_tts/codec.h"
#include "engine/models/irodori_tts/condition_encoder.h"
#include "engine/models/irodori_tts/rf_dit.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::models::irodori_tts {
namespace {

using Clock = std::chrono::steady_clock;
constexpr const char *kFamily = "irodori_tts";

enum class IrodoriCodecBackend {
  Same,
  Cpu,
};

std::shared_ptr<const IrodoriTTSAssets>
require_assets(std::shared_ptr<const IrodoriTTSAssets> assets) {
  if (assets == nullptr) {
    throw std::runtime_error("Irodori-TTS session requires assets");
  }
  return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
  if (contract == nullptr) {
    throw std::runtime_error("Irodori-TTS session requires a model contract");
  }
  return contract;
}

IrodoriCodecBackend parse_codec_backend(const runtime::SessionOptions & options) {
  if (const auto value =
          runtime::find_option(options.options, {"irodori_tts.codec_backend"})) {
    if (*value == "same") {
      return IrodoriCodecBackend::Same;
    }
    if (*value == "cpu") {
      return IrodoriCodecBackend::Cpu;
    }
    throw std::runtime_error("Invalid irodori_tts.codec_backend: " + *value);
  }
  return IrodoriCodecBackend::Same;
}

runtime::SessionOptions normalize_session_options(runtime::SessionOptions options) {
  return runtime::apply_option_v1_compatibility(
      std::move(options),
      {
          {"mem_saver", "irodori_tts.mem_saver"},
          {"reference_cache_slots", "irodori_tts.reference_cache_slots"},
      },
      "Irodori-TTS");
}

runtime::SessionOptions require_supported_session_options(
    runtime::SessionOptions options,
    const std::shared_ptr<const engine::model_spec::ModelContract> &contract) {
  options = normalize_session_options(std::move(options));
  const auto checked_contract = require_contract(contract);
  auto validation_options = options;
  // Older standalone GGUF packages embed a v1 contract that predates this
  // workaround option; keep them usable while still validating the value below.
  if (checked_contract->session_option_keys.find("irodori_tts.codec_backend") ==
      checked_contract->session_option_keys.end()) {
    validation_options.options.erase("irodori_tts.codec_backend");
  }
  runtime::validate_spec_backed_session_options(
      validation_options, *checked_contract, kFamily, "Irodori-TTS");
  return options;
}

std::unordered_map<std::string, std::string> normalize_request_options(
    std::unordered_map<std::string, std::string> options) {
  return runtime::apply_option_v1_compatibility(
      std::move(options),
      {
          {"caption", "instruction"},
          {"duration_seconds", "duration_sec"},
          {"min_seconds", "min_duration_sec"},
          {"max_seconds", "max_duration_sec"},
      },
      "Irodori-TTS",
      "request");
}

std::unique_ptr<runtime::IVoiceTaskSession> create_irodori_tts_session(
    const runtime::TaskSpec &task,
    const runtime::SessionOptions &options,
    std::shared_ptr<const IrodoriTTSAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
  return std::make_unique<IrodoriTTSSession>(
      task, options, std::move(assets), std::move(contract));
}

IrodoriGenerationOptions
generation_options_from_request(const runtime::TaskRequest &request) {
  IrodoriGenerationOptions options;
  if (const auto value = runtime::find_option(request.options, {"language"})) {
    if (*value != "ja") {
      throw std::runtime_error("Irodori-TTS language must be ja");
    }
  }
  if (const auto value =
          runtime::parse_int_option(request.options, {"num_inference_steps"})) {
    if (*value <= 0) {
      throw std::runtime_error(
          "Irodori-TTS num_inference_steps must be positive");
    }
    options.num_inference_steps = *value;
  }
  if (const auto value = runtime::parse_float_option(request.options,
                                                     {"text_guidance_scale"})) {
    options.text_guidance_scale = *value;
  }
  if (const auto value = runtime::parse_float_option(
          request.options, {"caption_guidance_scale"})) {
    options.caption_guidance_scale = *value;
  }
  if (const auto value = runtime::parse_float_option(
          request.options, {"speaker_guidance_scale"})) {
    options.speaker_guidance_scale = *value;
  }
  if (const auto value =
          runtime::parse_float_option(request.options, {"guidance_scale"})) {
    options.text_guidance_scale = *value;
    options.caption_guidance_scale = *value;
    options.speaker_guidance_scale = *value;
  }
  if (const auto value =
          runtime::find_option(request.options, {"guidance_mode"})) {
    options.guidance_mode = *value;
  }
  if (const auto value =
          runtime::parse_float_option(request.options, {"guidance_min_t"})) {
    options.guidance_min_t = *value;
  }
  if (const auto value =
          runtime::parse_float_option(request.options, {"guidance_max_t"})) {
    options.guidance_max_t = *value;
  }
  if (const auto value =
          runtime::parse_float_option(request.options, {"duration_scale"})) {
    options.duration_scale = *value;
  }
  if (const auto value =
          runtime::parse_float_option(request.options, {"duration_sec"})) {
    if (*value > 0.0F) {
      options.duration_seconds = *value;
      options.duration_seconds_specified = true;
    }
  }
  if (const auto value =
          runtime::parse_float_option(request.options, {"min_duration_sec"})) {
    options.min_seconds = *value;
  }
  if (const auto value =
          runtime::parse_float_option(request.options, {"max_duration_sec"})) {
    options.max_seconds = *value;
  }
  if (const auto value = runtime::parse_u32_option(request.options, {"seed"})) {
    options.seed = *value;
    options.seed_specified = true;
  } else {
    options.seed = runtime::random_u32_seed();
  }
  if (const auto value = runtime::find_option(request.options, {"trim_tail"})) {
    options.trim_tail = runtime::parse_bool_option(*value, "trim_tail");
  }
  return options;
}

std::string normalize_text(std::string text) {
  auto replace_all = [&](const std::string &from, const std::string &to) {
    size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
      text.replace(pos, from.size(), to);
      pos += to.size();
    }
  };
  replace_all("\t", "");
  replace_all("[n]", "");
  replace_all("\\[n\\]", "");
  replace_all("\xE3\x80\x80", "");
  replace_all("\xEF\xBC\x9F", "?");
  replace_all("\xEF\xBC\x81", "!");
  replace_all("\xE2\x99\xA5", "\xE2\x99\xA1");
  replace_all("\xE2\x97\x8F", "\xE2\x97\x8B");
  replace_all("\xE2\x97\xAF", "\xE2\x97\x8B");
  replace_all("\xE3\x80\x87", "\xE2\x97\x8B");
  replace_all("...", "\xE2\x80\xA6");
  replace_all("..", "\xE2\x80\xA6");
  while (!text.empty() && (text.front() == ' ' || text.front() == '\n' ||
                           text.front() == '\r')) {
    text.erase(text.begin());
  }
  while (!text.empty() &&
         (text.back() == ' ' || text.back() == '\n' || text.back() == '\r')) {
    text.pop_back();
  }
  return text;
}

uint64_t mix_reference_audio_key(uint64_t key, uint64_t value) {
  key ^= value;
  key *= 1099511628211ull;
  return key;
}

uint64_t reference_audio_cache_key(const runtime::AudioBuffer &audio) {
  uint64_t key = 1469598103934665603ull;
  key = mix_reference_audio_key(key, static_cast<uint64_t>(audio.sample_rate));
  key = mix_reference_audio_key(key, static_cast<uint64_t>(audio.channels));
  key =
      mix_reference_audio_key(key, static_cast<uint64_t>(audio.samples.size()));
  for (float sample : audio.samples) {
    uint32_t bits = 0;
    std::memcpy(&bits, &sample, sizeof(bits));
    key = mix_reference_audio_key(key, static_cast<uint64_t>(bits));
  }
  return key;
}

std::size_t resolve_reference_cache_slots(const runtime::SessionOptions &options) {
  constexpr int64_t kDefaultCacheSlots = 1;
  const int64_t slots =
      runtime::parse_i64_option(
          options.options,
          {"irodori_tts.reference_cache_slots", "reference_cache_slots"})
          .value_or(kDefaultCacheSlots);
  if (slots < 0) {
    throw std::runtime_error(
        "irodori_tts.reference_cache_slots must be non-negative");
  }
  if (static_cast<std::uint64_t>(slots) >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    throw std::runtime_error("irodori_tts.reference_cache_slots is too large");
  }
  return static_cast<std::size_t>(slots);
}

IrodoriSpeakerCondition
no_reference_speaker_condition(const IrodoriModelConfig &config) {
  IrodoriSpeakerCondition out;
  out.tokens = 2;
  out.state.assign(static_cast<size_t>(out.tokens * config.speaker_dim), 0.0F);
  out.mask.assign(static_cast<size_t>(out.tokens), 0);
  out.has_speaker = false;
  return out;
}

std::string trim_ascii(std::string text) {
  while (!text.empty() && (text.front() == ' ' || text.front() == '\n' ||
                           text.front() == '\r' || text.front() == '\t')) {
    text.erase(text.begin());
  }
  while (!text.empty() && (text.back() == ' ' || text.back() == '\n' ||
                           text.back() == '\r' || text.back() == '\t')) {
    text.pop_back();
  }
  return text;
}

std::string lower_ascii(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  return text;
}

std::string escape_log_text(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (char ch : text) {
    if (ch == '\n') {
      out += "\\n";
    } else if (ch == '\r') {
      out += "\\r";
    } else if (ch == '\t') {
      out += "\\t";
    } else {
      out.push_back(ch);
    }
  }
  return out;
}

} // namespace

IrodoriTTSSession::IrodoriTTSSession(
    runtime::TaskSpec task, runtime::SessionOptions options,
    std::shared_ptr<const IrodoriTTSAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(require_supported_session_options(std::move(options), contract)),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))),
      tokenizer_(assets_),
      reference_speaker_cache_(resolve_reference_cache_slots(this->options())) {
  condition_graph_arena_bytes_ = runtime::parse_size_mb_option(
      this->options().options, {"irodori_tts.condition_graph_arena_mb"},
      condition_graph_arena_bytes_);
  rf_graph_arena_bytes_ = runtime::parse_size_mb_option(
      this->options().options, {"irodori_tts.rf_graph_arena_mb"},
      rf_graph_arena_bytes_);
  codec_graph_arena_bytes_ = runtime::parse_size_mb_option(
      this->options().options, {"irodori_tts.codec_graph_arena_mb"},
      codec_graph_arena_bytes_);
  condition_weight_context_bytes_ = runtime::parse_size_mb_option(
      this->options().options, {"irodori_tts.condition_weight_context_mb"},
      condition_weight_context_bytes_);
  rf_weight_context_bytes_ = runtime::parse_size_mb_option(
      this->options().options, {"irodori_tts.rf_weight_context_mb"},
      rf_weight_context_bytes_);
  codec_weight_context_bytes_ = runtime::parse_size_mb_option(
      this->options().options, {"irodori_tts.codec_weight_context_mb"},
      codec_weight_context_bytes_);
  weight_storage_type_ = runtime::parse_tensor_storage_option(
      this->options().options,
      "irodori_tts.weight_type",
      assets::TensorStorageType::Native,
      {
          assets::TensorStorageType::Native,
          assets::TensorStorageType::F32,
          assets::TensorStorageType::F16,
          assets::TensorStorageType::BF16,
          assets::TensorStorageType::Q8_0,
      });
  codec_weight_storage_type_ = runtime::parse_tensor_storage_option(
      this->options().options,
      "irodori_tts.codec_weight_type",
      assets::TensorStorageType::Native,
      {
          assets::TensorStorageType::Native,
          assets::TensorStorageType::F32,
          assets::TensorStorageType::F16,
          assets::TensorStorageType::Q8_0,
      });
  if (const auto value =
          runtime::find_option(this->options().options, {"irodori_tts.mem_saver"})) {
    mem_saver_ = runtime::parse_bool_option(*value, "irodori_tts.mem_saver");
  }
  if (task_.mode != runtime::RunMode::Offline) {
    throw std::runtime_error("Irodori-TTS only supports offline sessions");
  }
  if (task_.task != runtime::VoiceTaskKind::Tts &&
      task_.task != runtime::VoiceTaskKind::VoiceCloning &&
      task_.task != runtime::VoiceTaskKind::VoiceDesign) {
    throw std::runtime_error(
        "Irodori-TTS supports only TTS, voice-cloning, and voice-design offline tasks");
  }
  const auto codec_backend = parse_codec_backend(this->options());
  engine::core::ExecutionContext * codec_execution = &execution_context();
  if (codec_backend == IrodoriCodecBackend::Cpu) {
    auto codec_backend_config = this->options().backend;
    codec_backend_config.type = engine::core::BackendType::Cpu;
    codec_backend_config.device = 0;
    codec_execution_context_ =
        std::make_unique<engine::core::ExecutionContext>(codec_backend_config);
    codec_execution = codec_execution_context_.get();
  }
  condition_encoder_ = std::make_unique<IrodoriConditionEncoder>(
      assets_, execution_context(), condition_graph_arena_bytes_,
      condition_weight_context_bytes_, weight_storage_type_);
  rf_sampler_ = std::make_unique<IrodoriRfSampler>(
      assets_, execution_context(), rf_graph_arena_bytes_,
      rf_weight_context_bytes_, weight_storage_type_, mem_saver_);
  codec_ = std::make_unique<IrodoriCodec>(
      assets_, *codec_execution, codec_graph_arena_bytes_,
      codec_weight_context_bytes_, codec_weight_storage_type_);
  assets_->model_weights->release_storage();
  assets_->codec_weights->release_storage();
  debug::trace_log_scalar("irodori_tts.model_root",
                          assets_->resources.model_root().string());
  debug::trace_log_scalar("irodori_tts.config.use_speaker_condition",
                          assets_->config.use_speaker_condition);
  debug::trace_log_scalar("irodori_tts.config.use_caption_condition",
                          assets_->config.use_caption_condition);
  debug::trace_log_scalar("irodori_tts.config.max_text_len",
                          assets_->config.max_text_len);
  debug::trace_log_scalar("irodori_tts.config.max_caption_len",
                          assets_->config.max_caption_len);
  debug::trace_log_scalar(
      "irodori_tts.codec.backend",
      std::string_view(codec_backend == IrodoriCodecBackend::Cpu ? "cpu"
                                                                  : "same"));
}

IrodoriTTSSession::~IrodoriTTSSession() = default;

bool IrodoriTTSSession::ReferenceAudioCacheKeyEqual::operator()(
    const ReferenceAudioCacheKey &lhs,
    const ReferenceAudioCacheKey &rhs) const {
  return lhs.hash == rhs.hash && lhs.sample_rate == rhs.sample_rate &&
         lhs.channels == rhs.channels && lhs.sample_count == rhs.sample_count;
}

std::string IrodoriTTSSession::family() const { return kFamily; }

runtime::VoiceTaskKind IrodoriTTSSession::task_kind() const {
  return task_.task;
}

runtime::RunMode IrodoriTTSSession::run_mode() const { return task_.mode; }

void IrodoriTTSSession::prepare(
    const runtime::SessionPreparationRequest &request) {
  (void)request;
  mark_prepared();
}

runtime::TaskResult
IrodoriTTSSession::run(const runtime::TaskRequest &request) {
  require_prepared("Irodori-TTS run");
  auto normalized_request = request;
  normalized_request.options = normalize_request_options(request.options);
  auto validation_options = normalized_request.options;
  // Older standalone GGUF packages may embed a v1 contract that still advertised
  // caption. Generate with the normalized instruction key, but keep those
  // packages usable while unknown request options remain rejected.
  if (contract_->request_option_keys.find("instruction") ==
          contract_->request_option_keys.end() &&
      contract_->request_option_keys.find("caption") !=
          contract_->request_option_keys.end()) {
    validation_options.erase("instruction");
  }
  runtime::validate_spec_backed_request_options(
      validation_options, *contract_, "Irodori-TTS");
  const auto wall_start = Clock::now();
  const int64_t text_chunk_size =
      engine::text::parse_text_chunk_size_override(normalized_request.options)
          .value_or(assets_->config.max_text_len);
  const auto text_chunk_mode =
      engine::text::parse_text_chunk_mode_override(normalized_request.options)
          .value_or(engine::text::TextChunkMode::Endline);
  const auto chunk_requests =
      runtime::chunk_text_request(normalized_request, text_chunk_size, text_chunk_mode);
  if (chunk_requests.empty()) {
    throw std::runtime_error("Irodori-TTS text chunking produced no requests");
  }
  const IrodoriRequest first_request = make_request(chunk_requests.front());
  const auto reference_start = Clock::now();
  IrodoriSpeakerCondition speaker =
      no_reference_speaker_condition(assets_->config);
  bool reference_cache_hit = false;
  if (!first_request.no_ref) {
    if (!first_request.has_reference_audio) {
      throw std::runtime_error(
          "Irodori-TTS reference mode requires reference audio");
    }
    const ReferenceAudioCacheKey reference_key{
        reference_audio_cache_key(first_request.reference_audio),
        first_request.reference_audio.sample_rate,
        first_request.reference_audio.channels,
        first_request.reference_audio.samples.size(),
    };
    if (const auto *cached = reference_speaker_cache_.find(reference_key)) {
      reference_cache_hit = true;
      speaker.state = cached->state;
      speaker.mask = cached->mask;
      speaker.tokens = cached->tokens;
      speaker.has_speaker = cached->has_speaker;
      debug::trace_log_scalar("irodori_tts.reference_cache.hit", 1);
      debug::trace_log_scalar(
          "irodori_tts.reference_cache.slots",
          static_cast<int64_t>(reference_speaker_cache_.capacity()));
      debug::trace_log_scalar(
          "irodori_tts.reference_cache.entries",
          static_cast<int64_t>(reference_speaker_cache_.size()));
      debug::trace_log_scalar("irodori_tts.reference_cache.evicted", 0);
    } else {
      const bool will_evict = reference_speaker_cache_.capacity() > 0 &&
                              reference_speaker_cache_.size() >=
                                  reference_speaker_cache_.capacity();
      int64_t ref_latent_steps = 0;
      auto ref_latent = codec_->encode_reference(first_request.reference_audio,
                                                 ref_latent_steps);
      speaker = condition_encoder_->encode_speaker_reference(ref_latent,
                                                             ref_latent_steps);
      ReferenceSpeakerCacheEntry entry;
      entry.state = speaker.state;
      entry.mask = speaker.mask;
      entry.tokens = speaker.tokens;
      entry.has_speaker = speaker.has_speaker;
      reference_speaker_cache_.put(reference_key, std::move(entry));
      if (mem_saver_) {
        codec_->release_graphs();
        condition_encoder_->release_graphs();
      }
      debug::trace_log_scalar("irodori_tts.reference_cache.hit", 0);
      debug::trace_log_scalar(
          "irodori_tts.reference_cache.slots",
          static_cast<int64_t>(reference_speaker_cache_.capacity()));
      debug::trace_log_scalar(
          "irodori_tts.reference_cache.entries",
          static_cast<int64_t>(reference_speaker_cache_.size()));
      debug::trace_log_scalar("irodori_tts.reference_cache.evicted",
                              will_evict ? 1 : 0);
    }
  }
  const auto reference_end = Clock::now();
  IrodoriCaptionCondition caption;
  double tokenize_ms = 0.0;
  if (assets_->config.use_caption_condition) {
    const auto caption_start = Clock::now();
    const std::string caption_text = trim_ascii(first_request.caption);
    auto tokenized_caption =
        tokenizer_.encode_padded(caption_text, assets_->config.max_caption_len);
    caption.token_ids = std::move(tokenized_caption.token_ids);
    caption.mask = std::move(tokenized_caption.mask);
    caption.has_caption = !caption_text.empty();
    if (!caption.has_caption) {
      std::fill(caption.mask.begin(), caption.mask.end(), 0);
    }
    tokenize_ms += debug::elapsed_ms(caption_start);
  } else if (!trim_ascii(first_request.caption).empty()) {
    throw std::runtime_error("Irodori-TTS loaded checkpoint does not include "
                             "caption conditioning weights");
  }
  const int64_t rf_context_graph_rebuilds_before =
      rf_sampler_->context_graph_rebuilds();
  const int64_t rf_step_graph_rebuilds_before =
      rf_sampler_->step_graph_rebuilds();
  runtime::AudioBuffer merged_audio;
  double condition_ms = 0.0;
  double sample_rf_ms = 0.0;
  double rf_context_cond_ms = 0.0;
  double rf_context_cfg_ms = 0.0;
  double rf_step_cfg_ms = 0.0;
  double rf_step_cond_ms = 0.0;
  double decode_ms = 0.0;
  for (size_t chunk_index = 0; chunk_index < chunk_requests.size();
       ++chunk_index) {
    const auto &chunk_request = chunk_requests[chunk_index];
    const IrodoriRequest irodori_request = make_request(chunk_request);
    debug::trace_log_scalar(
        "irodori_tts.chunk." + std::to_string(chunk_index) + ".text",
        escape_log_text(irodori_request.text));
    debug::trace_log_scalar(
        "irodori_tts.chunk." + std::to_string(chunk_index) + ".seed",
        irodori_request.generation.seed);
    const auto text_start = Clock::now();
    const auto tokenized = tokenizer_.encode_padded(
        irodori_request.text, assets_->config.max_text_len);
    tokenize_ms += debug::elapsed_ms(text_start);

    const auto condition_start = Clock::now();
    const auto conditions = condition_encoder_->run(
        tokenized.token_ids, tokenized.mask, caption, speaker);
    condition_ms += debug::elapsed_ms(condition_start);
    if (mem_saver_) {
      condition_encoder_->release_graphs();
    }
    const auto sample_start = Clock::now();
    IrodoriRfSampleTiming rf_timing;
    IrodoriRfSampleRequest sample_request;
    sample_request.conditions = &conditions;
    sample_request.text_mask = &tokenized.mask;
    sample_request.caption = caption;
    sample_request.speaker = speaker;
    sample_request.generation = irodori_request.generation;
    auto sample = rf_sampler_->sample(sample_request, &rf_timing);
    rf_context_cond_ms += rf_timing.context_cond_ms;
    rf_context_cfg_ms += rf_timing.context_cfg_ms;
    rf_step_cond_ms += rf_timing.step_cond_ms;
    rf_step_cfg_ms += rf_timing.step_cfg_ms;
    sample_rf_ms += debug::elapsed_ms(sample_start);

    const auto decode_start = Clock::now();
    runtime::append_audio_buffer(merged_audio,
                                 codec_->decode(sample.latent,
                                                sample.latent_steps,
                                                sample.target_samples));
    decode_ms += debug::elapsed_ms(decode_start);
    if (mem_saver_) {
      codec_->release_graphs();
    }
  }
  runtime::TaskResult result;
  result.audio_output = std::move(merged_audio);
  const auto wall_end = Clock::now();
  debug::trace_log_scalar("irodori_tts.reference.used", !first_request.no_ref);
  debug::trace_log_scalar("irodori_tts.reference.cache_hit",
                          reference_cache_hit);
  debug::trace_log_scalar("irodori_tts.text_chunk_size", text_chunk_size);
  debug::trace_log_scalar("irodori_tts.text_chunk_mode",
                          engine::text::text_chunk_mode_name(text_chunk_mode));
  debug::trace_log_scalar("irodori_tts.text_chunk_count",
                          static_cast<int64_t>(chunk_requests.size()));
  debug::trace_log_scalar("irodori_tts.sample_rf.context_graph_rebuilds",
                           rf_sampler_->context_graph_rebuilds() -
                               rf_context_graph_rebuilds_before);
  debug::trace_log_scalar("irodori_tts.sample_rf.step_graph_rebuilds",
                           rf_sampler_->step_graph_rebuilds() -
                               rf_step_graph_rebuilds_before);
  debug::timing_log_scalar("irodori_tts.prepare_reference_ms",
                           debug::elapsed_ms(reference_start, reference_end));
  debug::timing_log_scalar("irodori_tts.tokenize_ms", tokenize_ms);
  debug::timing_log_scalar("irodori_tts.condition_ms", condition_ms);
  debug::timing_log_scalar("irodori_tts.sample_rf_ms", sample_rf_ms);
  debug::timing_log_scalar("irodori_tts.sample_rf.context_cond_ms",
                           rf_context_cond_ms);
  debug::timing_log_scalar("irodori_tts.sample_rf.context_cfg_ms",
                           rf_context_cfg_ms);
  debug::timing_log_scalar("irodori_tts.sample_rf.steps_cfg_ms",
                           rf_step_cfg_ms);
  debug::timing_log_scalar("irodori_tts.sample_rf.steps_cond_ms",
                           rf_step_cond_ms);
  debug::timing_log_scalar("irodori_tts.codec_decode_ms", decode_ms);
  debug::timing_log_scalar("session.wall_ms",
                           debug::elapsed_ms(wall_start, wall_end));
  return result;
}

IrodoriRequest
IrodoriTTSSession::make_request(const runtime::TaskRequest &request) const {
  if (!request.text_input.has_value()) {
    throw std::runtime_error("Irodori-TTS requires text input");
  }
  IrodoriRequest out;
  out.text = normalize_text(request.text_input->text);
  if (out.text.empty()) {
    throw std::runtime_error(
        "Irodori-TTS text became empty after normalization");
  }
  if (const auto caption = runtime::find_option(request.options, {"instruction"})) {
    out.caption = *caption;
  }
  out.no_ref = true;
  if (const auto value = runtime::find_option(request.options, {"no_ref"})) {
    out.no_ref = runtime::parse_bool_option(*value, "no_ref");
  }
  if (request.voice.has_value() && request.voice->speaker.has_value() &&
      request.voice->speaker->audio.has_value()) {
    out.reference_audio = *request.voice->speaker->audio;
    out.has_reference_audio = true;
    out.no_ref = false;
  } else if (request.audio_input.has_value()) {
    out.reference_audio = *request.audio_input;
    out.has_reference_audio = true;
    out.no_ref = false;
  }
  out.generation = generation_options_from_request(request);
  if (out.generation.duration_scale <= 0.0F) {
    throw std::runtime_error("Irodori-TTS duration_scale must be positive");
  }
  if (out.generation.min_seconds <= 0.0F ||
      out.generation.max_seconds < out.generation.min_seconds) {
    throw std::runtime_error("Irodori-TTS invalid duration bounds");
  }
  out.generation.guidance_mode =
      lower_ascii(trim_ascii(out.generation.guidance_mode));
  const std::string mode = out.generation.guidance_mode;
  if (mode != "independent" && mode != "joint" && mode != "alternating") {
    throw std::runtime_error(
        "Irodori-TTS guidance_mode must be independent, joint, or alternating");
  }
  if (out.generation.guidance_min_t > out.generation.guidance_max_t) {
    throw std::runtime_error(
        "Irodori-TTS guidance_min_t must be <= guidance_max_t");
  }
  return out;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_irodori_tts_loader() {
  runtime::SpecBackedVoiceModelConfig<IrodoriTTSAssets> config;
  config.family = kFamily;
  config.load_assets = load_irodori_tts_assets;
  config.create_session = create_irodori_tts_session;
  return runtime::make_spec_backed_voice_loader(std::move(config));
}

} // namespace engine::models::irodori_tts
