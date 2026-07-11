#include "engine/models/irodori_tts/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/framework/text/chunking.h"
#include "engine/models/irodori_tts/codec.h"
#include "engine/models/irodori_tts/condition_encoder.h"
#include "engine/models/irodori_tts/rf_dit.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::irodori_tts {
namespace {

using Clock = std::chrono::steady_clock;
std::shared_ptr<const IrodoriAssets>
require_assets(std::shared_ptr<const IrodoriAssets> assets) {
  if (assets == nullptr) {
    throw std::runtime_error("Irodori-TTS session requires assets");
  }
  return assets;
}

void validate_weight_storage(assets::TensorStorageType storage_type,
                             const char *option_name) {
  if (storage_type == assets::TensorStorageType::Native ||
      storage_type == assets::TensorStorageType::F32 ||
      storage_type == assets::TensorStorageType::F16 ||
      storage_type == assets::TensorStorageType::BF16 ||
      storage_type == assets::TensorStorageType::Q8_0) {
    return;
  }
  throw std::runtime_error(std::string(option_name) +
                           " supports only native, f32, f16, bf16, and q8_0");
}

void validate_codec_weight_storage(assets::TensorStorageType storage_type,
                                   const char *option_name) {
  if (storage_type == assets::TensorStorageType::Native ||
      storage_type == assets::TensorStorageType::F32 ||
      storage_type == assets::TensorStorageType::F16 ||
      storage_type == assets::TensorStorageType::Q8_0) {
    return;
  }
  throw std::runtime_error(std::string(option_name) +
                           " supports only native, f32, f16, and q8_0");
}

IrodoriGenerationOptions
generation_options_from_request(const runtime::TaskRequest &request) {
  IrodoriGenerationOptions options;
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
          runtime::parse_float_option(request.options, {"duration_seconds"})) {
    if (*value > 0.0F) {
      options.duration_seconds = *value;
      options.duration_seconds_specified = true;
    }
  }
  if (const auto value =
          runtime::parse_float_option(request.options, {"min_seconds"})) {
    options.min_seconds = *value;
  }
  if (const auto value =
          runtime::parse_float_option(request.options, {"max_seconds"})) {
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

int find_flattening_point(const std::vector<float> &latent, int64_t frames,
                          int64_t dim, int64_t window_size, float std_threshold,
                          float mean_threshold) {
  if (frames <= 0 || window_size <= 0) {
    return static_cast<int>(std::max<int64_t>(0, frames));
  }
  for (int64_t i = 0; i < frames; ++i) {
    double sum = 0.0;
    double sum_sq = 0.0;
    int64_t count = 0;
    for (int64_t w = 0; w < window_size; ++w) {
      const int64_t frame = i + w;
      for (int64_t d = 0; d < dim; ++d) {
        const float value = frame < frames
                                ? latent[static_cast<size_t>(frame * dim + d)]
                                : 0.0F;
        sum += value;
        sum_sq += static_cast<double>(value) * value;
        ++count;
      }
    }
    const double mean = sum / static_cast<double>(count);
    const double variance =
        std::max(0.0, sum_sq / static_cast<double>(count) - mean * mean);
    if (std::sqrt(variance) < std_threshold &&
        std::abs(mean) < mean_threshold) {
      return static_cast<int>(i);
    }
  }
  return static_cast<int>(frames);
}

} // namespace

IrodoriTTSSession::IrodoriTTSSession(
    runtime::TaskSpec task, runtime::SessionOptions options,
    std::shared_ptr<const IrodoriAssets> assets)
    : RuntimeSessionBase(options), task_(task),
      assets_(require_assets(std::move(assets))), tokenizer_(assets_),
      reference_speaker_cache_(resolve_reference_cache_slots(this->options())) {
  condition_graph_arena_bytes_ = runtime::parse_size_mb_option(
      options.options, {"irodori_tts.condition_graph_arena_mb"},
      condition_graph_arena_bytes_);
  rf_graph_arena_bytes_ = runtime::parse_size_mb_option(
      options.options, {"irodori_tts.rf_graph_arena_mb"},
      rf_graph_arena_bytes_);
  codec_graph_arena_bytes_ = runtime::parse_size_mb_option(
      options.options, {"irodori_tts.codec_graph_arena_mb"},
      codec_graph_arena_bytes_);
  condition_weight_context_bytes_ = runtime::parse_size_mb_option(
      options.options, {"irodori_tts.condition_weight_context_mb"},
      condition_weight_context_bytes_);
  rf_weight_context_bytes_ = runtime::parse_size_mb_option(
      options.options, {"irodori_tts.rf_weight_context_mb"},
      rf_weight_context_bytes_);
  codec_weight_context_bytes_ = runtime::parse_size_mb_option(
      options.options, {"irodori_tts.codec_weight_context_mb"},
      codec_weight_context_bytes_);
  if (const auto it = options.options.find("irodori_tts.weight_type");
      it != options.options.end()) {
    weight_storage_type_ = assets::parse_tensor_storage_type(it->second);
    validate_weight_storage(weight_storage_type_, "irodori_tts.weight_type");
  }
  if (const auto it = options.options.find("irodori_tts.codec_weight_type");
      it != options.options.end()) {
    codec_weight_storage_type_ = assets::parse_tensor_storage_type(it->second);
    validate_codec_weight_storage(codec_weight_storage_type_,
                                  "irodori_tts.codec_weight_type");
  }
  if (const auto value =
          runtime::find_option(options.options, {"irodori_tts.mem_saver", "mem_saver"})) {
    mem_saver_ = runtime::parse_bool_option(*value, "irodori_tts.mem_saver");
  }
  if (task_.mode != runtime::RunMode::Offline) {
    throw std::runtime_error("Irodori-TTS only supports offline sessions");
  }
  if (task_.task != runtime::VoiceTaskKind::Tts &&
      task_.task != runtime::VoiceTaskKind::VoiceDesign) {
    throw std::runtime_error(
        "Irodori-TTS supports only TTS and voice-design offline tasks");
  }
  condition_encoder_ = std::make_unique<IrodoriConditionEncoder>(
      assets_, execution_context(), condition_graph_arena_bytes_,
      condition_weight_context_bytes_, weight_storage_type_);
  rf_sampler_ = std::make_unique<IrodoriRfSampler>(
      assets_, execution_context(), rf_graph_arena_bytes_,
      rf_weight_context_bytes_, weight_storage_type_);
  codec_ = std::make_unique<IrodoriCodec>(
      assets_, execution_context(), codec_graph_arena_bytes_,
      codec_weight_context_bytes_, codec_weight_storage_type_);
  debug::trace_log_scalar("irodori_tts.model_root",
                          assets_->paths.model_root.string());
  debug::trace_log_scalar("irodori_tts.codec_root",
                          assets_->paths.codec_root.string());
  debug::trace_log_scalar("irodori_tts.config.use_speaker_condition",
                          assets_->config.use_speaker_condition);
  debug::trace_log_scalar("irodori_tts.config.use_caption_condition",
                          assets_->config.use_caption_condition);
  debug::trace_log_scalar("irodori_tts.config.max_text_len",
                          assets_->config.max_text_len);
  debug::trace_log_scalar("irodori_tts.config.max_caption_len",
                          assets_->config.max_caption_len);
}

IrodoriTTSSession::~IrodoriTTSSession() = default;

bool IrodoriTTSSession::ReferenceAudioCacheKeyEqual::operator()(
    const ReferenceAudioCacheKey &lhs,
    const ReferenceAudioCacheKey &rhs) const {
  return lhs.hash == rhs.hash && lhs.sample_rate == rhs.sample_rate &&
         lhs.channels == rhs.channels && lhs.sample_count == rhs.sample_count;
}

std::string IrodoriTTSSession::family() const { return "irodori_tts"; }

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
  const auto wall_start = Clock::now();
  const int64_t text_chunk_size =
      engine::text::parse_text_chunk_size_override(request.options)
          .value_or(assets_->config.max_text_len);
  const auto chunk_requests =
      runtime::chunk_text_request(request, text_chunk_size,
                                  engine::text::TextChunkMode::Japanese);
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
  const int hop_length = static_cast<int>(assets_->codec.hop_length);
  for (const auto &chunk_request : chunk_requests) {
    const IrodoriRequest irodori_request = make_request(chunk_request);
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
    IrodoriCaptionCondition rf_caption;
    std::vector<float> rf_caption_state;
    if (assets_->config.use_caption_condition) {
      const int64_t dim = assets_->config.caption_dim_resolved();
      rf_caption.mask = caption.mask;
      rf_caption.has_caption = caption.has_caption;
      rf_caption_state = conditions.caption_state;
      if (static_cast<int64_t>(rf_caption_state.size()) !=
          assets_->config.max_caption_len * dim) {
        throw std::runtime_error(
            "Irodori-TTS caption condition state shape mismatch");
      }
    }

    int64_t latent_steps = 0;
    int64_t target_samples = 0;
    if (irodori_request.generation.duration_seconds_specified) {
      const float seconds =
          std::min(irodori_request.generation.max_seconds,
                   std::max(irodori_request.generation.min_seconds,
                            irodori_request.generation.duration_seconds));
      target_samples = std::max<int64_t>(
          1, static_cast<int64_t>(seconds * assets_->codec.sample_rate));
      latent_steps = (target_samples + hop_length - 1) / hop_length;
    } else {
      const float pred_frames = std::expm1(conditions.predicted_log_frames);
      const float scaled_frames =
          pred_frames * irodori_request.generation.duration_scale;
      const int64_t min_frames =
          std::max<int64_t>(1, static_cast<int64_t>(std::ceil(
                                   irodori_request.generation.min_seconds *
                                   assets_->codec.sample_rate / hop_length)));
      const int64_t max_frames =
          std::max<int64_t>(1, static_cast<int64_t>(std::floor(
                                   irodori_request.generation.max_seconds *
                                   assets_->codec.sample_rate / hop_length)));
      latent_steps = static_cast<int64_t>(std::llround(scaled_frames));
      latent_steps = std::max(min_frames, std::min(max_frames, latent_steps));
      target_samples = latent_steps * hop_length;
    }
    const int64_t patched_steps =
        (latent_steps + assets_->config.latent_patch_size - 1) /
        assets_->config.latent_patch_size;
    const int64_t patched_dim = assets_->config.patched_latent_dim();
    const auto sample_start = Clock::now();
    std::vector<float> x_t = sampling::generate_torch_cuda_randn(
        static_cast<size_t>(patched_steps * patched_dim),
        irodori_request.generation.seed,
        sampling::TorchRandnPrecision::Float32);
    const bool text_cfg_enabled =
        irodori_request.generation.text_guidance_scale > 0.0F;
    const bool speaker_cfg_enabled =
        speaker.has_speaker &&
        irodori_request.generation.speaker_guidance_scale > 0.0F;
    const bool caption_cfg_enabled =
        rf_caption.has_caption &&
        irodori_request.generation.caption_guidance_scale > 0.0F;
    const bool any_cfg_enabled =
        text_cfg_enabled || speaker_cfg_enabled || caption_cfg_enabled;
    IrodoriRfSampler::ContextCache rf_context_cond;
    IrodoriRfSampler::ContextCache rf_context_cfg;
    if (any_cfg_enabled) {
      const auto rf_context_cfg_start = Clock::now();
      rf_context_cfg = rf_sampler_->build_context_cache(
          conditions.text_state, tokenized.mask, rf_caption_state, rf_caption,
          speaker, text_cfg_enabled, speaker_cfg_enabled, caption_cfg_enabled);
      rf_context_cfg_ms += debug::elapsed_ms(rf_context_cfg_start);
    } else {
      const auto rf_context_cond_start = Clock::now();
      rf_context_cond = rf_sampler_->build_context_cache(
          conditions.text_state, tokenized.mask, rf_caption_state, rf_caption,
          speaker, false, false, false);
      rf_context_cond_ms += debug::elapsed_ms(rf_context_cond_start);
      rf_context_cfg = rf_context_cond;
    }

    std::vector<float> timesteps(
        static_cast<size_t>(irodori_request.generation.num_inference_steps));
    for (int64_t step = 0;
         step < irodori_request.generation.num_inference_steps; ++step) {
      const float u =
          static_cast<float>(step) /
          static_cast<float>(irodori_request.generation.num_inference_steps);
      timesteps[static_cast<size_t>(step)] = (1.0F - u) * 0.999F;
    }
    const auto modulation_cache =
        rf_sampler_->build_modulation_cache(timesteps);
    std::vector<float> velocity(x_t.size());
    for (int64_t step = 0;
         step < irodori_request.generation.num_inference_steps; ++step) {
      const float u_next =
          static_cast<float>(step + 1) /
          static_cast<float>(irodori_request.generation.num_inference_steps);
      const float t = timesteps[static_cast<size_t>(step)];
      const float t_next = (1.0F - u_next) * 0.999F;
      const bool cfg_active = any_cfg_enabled &&
                              t >= irodori_request.generation.guidance_min_t &&
                              t <= irodori_request.generation.guidance_max_t;
      const auto &rf_context =
          cfg_active ? rf_context_cfg
                     : (any_cfg_enabled ? rf_context_cfg : rf_context_cond);
      const auto rf_step_start = Clock::now();
      rf_sampler_->run_step(x_t, step, modulation_cache, rf_context,
                            cfg_active && text_cfg_enabled,
                            cfg_active && speaker_cfg_enabled,
                            cfg_active && caption_cfg_enabled,
                            irodori_request.generation.text_guidance_scale,
                            irodori_request.generation.speaker_guidance_scale,
                            irodori_request.generation.caption_guidance_scale,
                            patched_steps, velocity);
      const double rf_step_ms = debug::elapsed_ms(rf_step_start);
      if (cfg_active) {
        rf_step_cfg_ms += rf_step_ms;
      } else {
        rf_step_cond_ms += rf_step_ms;
      }
      const int64_t sample_count = static_cast<int64_t>(x_t.size());
#ifdef _OPENMP
#pragma omp parallel for if(sample_count >= 4096)
#endif
      for (int64_t i = 0; i < sample_count; ++i) {
        const size_t index = static_cast<size_t>(i);
        x_t[index] += velocity[index] * (t_next - t);
      }
    }
    if (mem_saver_) {
      rf_context_cond = IrodoriRfSampler::ContextCache();
      rf_context_cfg = IrodoriRfSampler::ContextCache();
      rf_sampler_->release_graphs();
    }
    sample_rf_ms += debug::elapsed_ms(sample_start);

    std::vector<float> latent(
        static_cast<size_t>(latent_steps * assets_->config.latent_dim), 0.0F);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if(latent_steps * assets_->config.latent_dim >= 4096)
#endif
    for (int64_t frame = 0; frame < latent_steps; ++frame) {
      for (int64_t dim = 0; dim < assets_->config.latent_dim; ++dim) {
        const int64_t patched_frame = frame / assets_->config.latent_patch_size;
        const int64_t patch_offset = frame % assets_->config.latent_patch_size;
        latent[static_cast<size_t>(frame * assets_->config.latent_dim + dim)] =
            x_t[static_cast<size_t>(patched_frame * patched_dim +
                                    patch_offset * assets_->config.latent_dim +
                                    dim)];
      }
    }
    if (irodori_request.generation.trim_tail) {
      const int flat = find_flattening_point(
          latent, latent_steps, assets_->config.latent_dim,
          irodori_request.generation.tail_window_size,
          irodori_request.generation.tail_std_threshold,
          irodori_request.generation.tail_mean_threshold);
      const int64_t flattening_samples =
          static_cast<int64_t>(flat) * hop_length;
      if (flattening_samples > 0) {
        target_samples = std::min(target_samples, flattening_samples);
      }
    }
    const auto decode_start = Clock::now();
    runtime::append_audio_buffer(
        merged_audio, codec_->decode(latent, latent_steps, target_samples));
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
  if (const auto caption = runtime::find_option(request.options, {"caption"})) {
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
  const std::string mode = out.generation.guidance_mode;
  if (mode != "independent") {
    throw std::runtime_error(
        "Irodori-TTS native path currently supports independent CFG mode");
  }
  return out;
}

} // namespace engine::models::irodori_tts
