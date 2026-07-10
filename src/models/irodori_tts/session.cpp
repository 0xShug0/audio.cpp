#include "engine/models/irodori_tts/session.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"
#include "engine/framework/sampling/torch_random.h"

#include <ggml-backend.h>
#include <ggml.h>

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
constexpr float kNegInf = -1.0e9F;

struct GgmlContextDeleter {
  void operator()(ggml_context *ctx) const noexcept {
    if (ctx != nullptr) {
      ggml_free(ctx);
    }
  }
};

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
      throw std::runtime_error("Irodori-TTS num_inference_steps must be positive");
    }
    options.num_inference_steps = *value;
  }
  if (const auto value =
          runtime::parse_float_option(request.options, {"text_guidance_scale"})) {
    options.text_guidance_scale = *value;
  }
  if (const auto value =
          runtime::parse_float_option(request.options, {"caption_guidance_scale"})) {
    options.caption_guidance_scale = *value;
  }
  if (const auto value =
          runtime::parse_float_option(request.options, {"speaker_guidance_scale"})) {
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

std::vector<float> make_text_attention_mask(const std::vector<uint8_t> &mask,
                                            int64_t batch, int64_t heads,
                                            int64_t tokens) {
  std::vector<float> out(static_cast<size_t>(batch * heads * tokens * tokens),
                         kNegInf);
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t h = 0; h < heads; ++h) {
      for (int64_t q = 0; q < tokens; ++q) {
        for (int64_t k = 0; k < tokens; ++k) {
          const bool keep = mask[static_cast<size_t>(b * tokens + k)] != 0;
          out[static_cast<size_t>(((b * heads + h) * tokens + q) * tokens +
                                  k)] = keep ? 0.0F : kNegInf;
        }
      }
    }
  }
  return out;
}

std::vector<float>
make_rf_attention_mask(const std::vector<uint8_t> &text_mask,
                       const std::vector<uint8_t> &speaker_mask,
                       const std::vector<uint8_t> &caption_mask, int64_t batch,
                       int64_t latent_steps, int64_t text_tokens,
                       int64_t speaker_tokens, int64_t caption_tokens) {
  const int64_t keys =
      latent_steps + text_tokens + speaker_tokens + caption_tokens;
  std::vector<float> out(static_cast<size_t>(batch * latent_steps * keys),
                         -INFINITY);
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t q = 0; q < latent_steps; ++q) {
      for (int64_t k = 0; k < keys; ++k) {
        bool keep = k < latent_steps;
        if (k >= latent_steps && k < latent_steps + text_tokens) {
          keep = text_mask[static_cast<size_t>(b * text_tokens +
                                               (k - latent_steps))] != 0;
        } else if (k >= latent_steps + text_tokens &&
                   k < latent_steps + text_tokens + speaker_tokens) {
          keep =
              speaker_mask[static_cast<size_t>(
                  b * speaker_tokens + (k - latent_steps - text_tokens))] != 0;
        } else if (k >= latent_steps + text_tokens + speaker_tokens &&
                   caption_tokens > 0) {
          keep = caption_mask[static_cast<size_t>(
                     b * caption_tokens +
                     (k - latent_steps - text_tokens - speaker_tokens))] != 0;
        }
        out[static_cast<size_t>((b * latent_steps + q) * keys + k)] =
            keep ? 0.0F : -INFINITY;
      }
    }
  }
  return out;
}

std::vector<int32_t> positions(int64_t count) {
  std::vector<int32_t> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    out[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  }
  return out;
}

std::vector<float> reflect_pad_right_to_multiple(std::vector<float> samples,
                                                 int64_t multiple) {
  if (samples.empty() || multiple <= 0) {
    throw std::runtime_error("Irodori-TTS reference audio is invalid");
  }
  const int64_t length = static_cast<int64_t>(samples.size());
  const int64_t pad = (multiple - (length % multiple)) % multiple;
  if (pad == 0) {
    return samples;
  }
  if (pad >= length) {
    throw std::runtime_error(
        "Irodori-TTS reference audio is too short for DACVAE reflect padding");
  }
  samples.reserve(static_cast<size_t>(length + pad));
  for (int64_t i = 0; i < pad; ++i) {
    samples.push_back(samples[static_cast<size_t>(length - 2 - i)]);
  }
  return samples;
}

std::vector<float> lfilter_biquad(const std::vector<float> &input, double b0,
                                  double b1, double b2, double a1, double a2) {
  std::vector<float> output(input.size(), 0.0F);
  double x1 = 0.0;
  double x2 = 0.0;
  double y1 = 0.0;
  double y2 = 0.0;
  for (size_t i = 0; i < input.size(); ++i) {
    const double x0 = static_cast<double>(input[i]);
    const double y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    output[i] = static_cast<float>(y0);
    x2 = x1;
    x1 = x0;
    y2 = y1;
    y1 = y0;
  }
  return output;
}

double bs1770_loudness_48k(const std::vector<float> &mono) {
  constexpr int kSampleRate = 48000;
  constexpr int kMinSamples = kSampleRate / 2;
  constexpr int kBlockSamples = static_cast<int>(0.4 * kSampleRate);
  constexpr int kStrideSamples = kBlockSamples / 4;
  constexpr double kAbsoluteGate = -70.0;
  constexpr double kMinLoudness = -70.0;
  std::vector<float> working = mono;
  if (static_cast<int>(working.size()) < kMinSamples) {
    working.resize(kMinSamples, 0.0F);
  }
  working = lfilter_biquad(working, 1.5351828863637502, -2.691804030199196,
                           1.198426263333146, -1.6906995865986896,
                           0.7325047060963897);
  working = lfilter_biquad(working, 0.9950442970178917, -1.9900885940357833,
                           0.9950442970178917, -1.990076284018423,
                           0.9901009040531438);
  std::vector<double> block_energy;
  for (int64_t start = 0;
       start + kBlockSamples <= static_cast<int64_t>(working.size());
       start += kStrideSamples) {
    double sum_sq = 0.0;
    for (int i = 0; i < kBlockSamples; ++i) {
      const double sample = working[static_cast<size_t>(start + i)];
      sum_sq += sample * sample;
    }
    block_energy.push_back(sum_sq / static_cast<double>(kBlockSamples));
  }
  if (block_energy.empty()) {
    return kMinLoudness;
  }
  auto loudness_from_energy = [](double energy) {
    return -0.691 + 10.0 * std::log10(std::max(
                               energy, std::numeric_limits<double>::min()));
  };
  double gated_sum = 0.0;
  int gated_count = 0;
  std::vector<double> block_loudness(block_energy.size(), kMinLoudness);
  for (size_t i = 0; i < block_energy.size(); ++i) {
    block_loudness[i] = loudness_from_energy(block_energy[i]);
    if (block_loudness[i] > kAbsoluteGate) {
      gated_sum += block_energy[i];
      ++gated_count;
    }
  }
  if (gated_count == 0) {
    return kMinLoudness;
  }
  const double relative_gate =
      loudness_from_energy(gated_sum / static_cast<double>(gated_count)) - 10.0;
  double relative_sum = 0.0;
  int relative_count = 0;
  for (size_t i = 0; i < block_energy.size(); ++i) {
    if (block_loudness[i] > kAbsoluteGate &&
        block_loudness[i] > relative_gate) {
      relative_sum += block_energy[i];
      ++relative_count;
    }
  }
  if (relative_count == 0) {
    return kMinLoudness;
  }
  return std::max(
      loudness_from_energy(relative_sum / static_cast<double>(relative_count)),
      kMinLoudness);
}

void normalize_reference_audio_in_place(std::vector<float> &mono,
                                        double target_db) {
  constexpr double kGainFactor = 0.11512925464970229;
  const double loudness = bs1770_loudness_48k(mono);
  const double gain = std::exp((target_db - loudness) * kGainFactor);
  float peak = 0.0F;
  for (float &sample : mono) {
    sample = static_cast<float>(static_cast<double>(sample) * gain);
    peak = std::max(peak, std::abs(sample));
  }
  if (std::isfinite(peak) && peak > 1.0F) {
    const float peak_gain = 1.0F / peak;
    for (float &sample : mono) {
      sample *= peak_gain;
    }
  }
}

std::vector<float> prepend_masked_mean_token(const std::vector<float> &state,
                                             const std::vector<uint8_t> &mask,
                                             int64_t tokens, int64_t dim) {
  if (tokens <= 0 || dim <= 0 ||
      static_cast<int64_t>(state.size()) != tokens * dim ||
      static_cast<int64_t>(mask.size()) != tokens) {
    throw std::runtime_error("Irodori-TTS speaker state shape mismatch");
  }
  std::vector<float> out(static_cast<size_t>((tokens + 1) * dim), 0.0F);
  int64_t count = 0;
  for (int64_t token = 0; token < tokens; ++token) {
    if (mask[static_cast<size_t>(token)] == 0) {
      continue;
    }
    ++count;
    for (int64_t d = 0; d < dim; ++d) {
      out[static_cast<size_t>(d)] +=
          state[static_cast<size_t>(token * dim + d)];
    }
  }
  if (count > 0) {
    const float scale = 1.0F / static_cast<float>(count);
    for (int64_t d = 0; d < dim; ++d) {
      out[static_cast<size_t>(d)] *= scale;
    }
  }
  std::copy(state.begin(), state.end(),
            out.begin() + static_cast<std::ptrdiff_t>(dim));
  return out;
}

struct IrodoriSpeakerCondition {
  std::vector<float> state;
  std::vector<uint8_t> mask;
  bool has_speaker = false;
  int64_t tokens = 0;
};

struct IrodoriCaptionCondition {
  std::vector<int32_t> token_ids;
  std::vector<uint8_t> mask;
  bool has_caption = false;
};

uint64_t mix_reference_audio_key(uint64_t key, uint64_t value) {
  key ^= value;
  key *= 1099511628211ull;
  return key;
}

uint64_t reference_audio_cache_key(const runtime::AudioBuffer &audio) {
  uint64_t key = 1469598103934665603ull;
  key = mix_reference_audio_key(key, static_cast<uint64_t>(audio.sample_rate));
  key = mix_reference_audio_key(key, static_cast<uint64_t>(audio.channels));
  key = mix_reference_audio_key(key, static_cast<uint64_t>(audio.samples.size()));
  for (float sample : audio.samples) {
    uint32_t bits = 0;
    std::memcpy(&bits, &sample, sizeof(bits));
    key = mix_reference_audio_key(key, static_cast<uint64_t>(bits));
  }
  return key;
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

std::vector<float>
duration_speaker_state(const IrodoriSpeakerCondition &speaker,
                       int64_t speaker_dim) {
  std::vector<float> out(static_cast<size_t>(2 * speaker_dim), 0.0F);
  if (!speaker.state.empty()) {
    std::copy(speaker.state.begin(),
              speaker.state.begin() + static_cast<std::ptrdiff_t>(speaker_dim),
              out.begin());
  }
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

class IrodoriConditionRuntime {
public:
  struct Output {
    std::vector<float> text_state;
    std::vector<float> caption_state;
    float predicted_log_frames = 0.0F;
  };

  IrodoriConditionRuntime(std::shared_ptr<const IrodoriAssets> assets,
                          core::ExecutionContext &execution_context,
                          size_t graph_arena_bytes, size_t weight_context_bytes,
                          assets::TensorStorageType weight_storage_type)
      : assets_(std::move(assets)),
        weights_(load_irodori_condition_encoder_weights(
            *assets_, execution_context.backend(),
            execution_context.backend_type(), weight_context_bytes,
            weight_storage_type)),
        backend_(execution_context.backend()),
        backend_type_(execution_context.backend_type()),
        threads_(std::max(1, execution_context.config().threads)),
        graph_arena_bytes_(graph_arena_bytes) {
    if (assets_ == nullptr) {
      throw std::runtime_error("Irodori-TTS condition runtime requires assets");
    }
  }

  Output run(const std::vector<int32_t> &token_ids,
             const std::vector<uint8_t> &token_mask,
             const IrodoriCaptionCondition &caption,
             const IrodoriSpeakerCondition &speaker) {
    const auto &config = assets_->config;
    const int64_t tokens = config.max_text_len;
    if (static_cast<int64_t>(token_ids.size()) != tokens ||
        static_cast<int64_t>(token_mask.size()) != tokens) {
      throw std::runtime_error("Irodori-TTS condition input shape mismatch");
    }
    if (config.use_caption_condition &&
        (static_cast<int64_t>(caption.token_ids.size()) !=
             config.max_caption_len ||
         static_cast<int64_t>(caption.mask.size()) != config.max_caption_len)) {
      throw std::runtime_error(
          "Irodori-TTS caption condition input shape mismatch");
    }
    const bool graph_rebuild = graph_ == nullptr;
    if (graph_rebuild) {
      graph_ = std::make_unique<Graph>(*this, tokens, graph_arena_bytes_);
    }
    debug::timing_log_scalar("irodori_tts.condition.graph_rebuild",
                             graph_rebuild);
    return graph_->run(token_ids, token_mask, caption, speaker);
  }

  IrodoriSpeakerCondition
  encode_speaker_reference(const std::vector<float> &ref_latent,
                           int64_t ref_tokens) {
    const auto &config = assets_->config;
    if (ref_tokens <= 0 ||
        static_cast<int64_t>(ref_latent.size()) !=
            ref_tokens * config.speaker_patched_latent_dim()) {
      throw std::runtime_error("Irodori-TTS reference latent shape mismatch");
    }
    const bool graph_rebuild =
        speaker_graph_ == nullptr || speaker_graph_->tokens() != ref_tokens;
    if (graph_rebuild) {
      speaker_graph_ =
          std::make_unique<SpeakerGraph>(*this, ref_tokens, graph_arena_bytes_);
    }
    debug::timing_log_scalar("irodori_tts.speaker_encoder.graph_rebuild",
                             graph_rebuild);
    return speaker_graph_->run(ref_latent);
  }

private:
  class SpeakerGraph {
  public:
    SpeakerGraph(IrodoriConditionRuntime &runtime, int64_t tokens,
                 size_t graph_arena_bytes)
        : runtime_(&runtime), tokens_(tokens) {
      const auto &config = runtime.assets_->config;
      ggml_init_params params{graph_arena_bytes, nullptr, true};
      ctx_.reset(ggml_init(params));
      if (ctx_ == nullptr) {
        throw std::runtime_error(
            "failed to initialize Irodori-TTS speaker graph context");
      }
      core::ModuleBuildContext build_ctx{
          ctx_.get(), "irodori_tts.speaker_encoder", runtime.backend_type_};
      ref_latent_ = core::make_tensor(
          build_ctx, GGML_TYPE_F32,
          core::TensorShape::from_dims(
              {1, tokens_, config.speaker_patched_latent_dim()}));
      ref_mask_ = core::make_tensor(build_ctx, GGML_TYPE_I32,
                                    core::TensorShape::from_dims({1, tokens_}));
      attention_mask_ =
          core::make_tensor(build_ctx, GGML_TYPE_F32,
                            core::TensorShape::from_dims(
                                {1, config.speaker_heads, tokens_, tokens_}));
      positions_ = core::make_tensor(build_ctx, GGML_TYPE_I32,
                                     core::TensorShape::from_dims({tokens_}));
      ggml_set_input(ref_latent_.tensor);
      ggml_set_input(ref_mask_.tensor);
      ggml_set_input(attention_mask_.tensor);
      ggml_set_input(positions_.tensor);
      auto output = build_irodori_reference_latent_encoder(
          build_ctx, ref_latent_, ref_mask_, attention_mask_, positions_,
          runtime.weights_, config);
      output_ = core::ensure_backend_addressable_layout(build_ctx, output);
      ggml_set_output(output_.tensor);
      graph_ = ggml_new_graph_custom(ctx_.get(), 131072, false);
      ggml_build_forward_expand(graph_, output_.tensor);
      gallocr_ = ggml_gallocr_new(
          ggml_backend_get_default_buffer_type(runtime.backend_));
      if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) ||
          !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
        throw std::runtime_error(
            "failed to allocate Irodori-TTS speaker graph");
      }
      const auto mask = std::vector<uint8_t>(static_cast<size_t>(tokens_), 1);
      core::write_tensor_i32(
          ref_mask_, std::vector<int32_t>(static_cast<size_t>(tokens_), 1));
      core::write_tensor_f32(
          attention_mask_,
          make_text_attention_mask(mask, 1, config.speaker_heads, tokens_));
      core::write_tensor_i32(positions_, positions(tokens_));
    }

    ~SpeakerGraph() {
      engine::core::release_backend_graph_resources(runtime_->backend_, graph_);
      if (gallocr_ != nullptr) {
        ggml_gallocr_free(gallocr_);
      }
    }

    int64_t tokens() const noexcept { return tokens_; }

    IrodoriSpeakerCondition run(const std::vector<float> &ref_latent) {
      const auto &config = runtime_->assets_->config;
      core::write_tensor_f32(ref_latent_, ref_latent);
      core::set_backend_threads(runtime_->backend_, runtime_->threads_);
      const ggml_status status = core::compute_backend_graph(
          runtime_->backend_, graph_, nullptr, "irodori_tts.speaker_encoder");
      ggml_backend_synchronize(runtime_->backend_);
      if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Irodori-TTS speaker graph compute failed");
      }
      std::vector<uint8_t> mask(static_cast<size_t>(tokens_), 1);
      const auto encoded = core::read_tensor_f32(output_.tensor);
      IrodoriSpeakerCondition out;
      out.tokens = tokens_ + 1;
      out.mask.assign(static_cast<size_t>(out.tokens), 1);
      out.state =
          prepend_masked_mean_token(encoded, mask, tokens_, config.speaker_dim);
      out.has_speaker = true;
      return out;
    }

  private:
    IrodoriConditionRuntime *runtime_ = nullptr;
    int64_t tokens_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue ref_latent_;
    core::TensorValue ref_mask_;
    core::TensorValue attention_mask_;
    core::TensorValue positions_;
    core::TensorValue output_;
    ggml_cgraph *graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
  };

  class Graph {
  public:
    Graph(IrodoriConditionRuntime &runtime, int64_t tokens,
          size_t graph_arena_bytes)
        : runtime_(&runtime), tokens_(tokens) {
      const auto &config = runtime.assets_->config;
      ggml_init_params params{
          graph_arena_bytes,
          nullptr,
          true,
      };
      ctx_.reset(ggml_init(params));
      if (ctx_ == nullptr) {
        throw std::runtime_error(
            "failed to initialize Irodori-TTS condition graph context");
      }
      core::ModuleBuildContext build_ctx{ctx_.get(), "irodori_tts.condition",
                                         runtime.backend_type_};
      input_ids_ = core::make_tensor(
          build_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, tokens_}));
      text_mask_ = core::make_tensor(
          build_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, tokens_}));
      text_attention_mask_ =
          core::make_tensor(build_ctx, GGML_TYPE_F32,
                            core::TensorShape::from_dims(
                                {1, config.text_heads, tokens_, tokens_}));
      positions_ = core::make_tensor(build_ctx, GGML_TYPE_I32,
                                     core::TensorShape::from_dims({tokens_}));
      if (config.use_caption_condition) {
        caption_ids_ = core::make_tensor(
            build_ctx, GGML_TYPE_I32,
            core::TensorShape::from_dims({1, config.max_caption_len}));
        caption_mask_ = core::make_tensor(
            build_ctx, GGML_TYPE_I32,
            core::TensorShape::from_dims({1, config.max_caption_len}));
        caption_attention_mask_ = core::make_tensor(
            build_ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims({1, config.caption_heads_resolved(),
                                          config.max_caption_len,
                                          config.max_caption_len}));
        caption_positions_ = core::make_tensor(
            build_ctx, GGML_TYPE_I32,
            core::TensorShape::from_dims({config.max_caption_len}));
      }
      speaker_state_ = core::make_tensor(
          build_ctx, GGML_TYPE_F32,
          core::TensorShape::from_dims({1, 2, config.speaker_dim}));
      has_speaker_ = core::make_tensor(build_ctx, GGML_TYPE_I32,
                                       core::TensorShape::from_dims({1}));
      if (config.use_caption_condition) {
        has_caption_ = core::make_tensor(build_ctx, GGML_TYPE_I32,
                                         core::TensorShape::from_dims({1}));
      }
      ggml_set_input(input_ids_.tensor);
      ggml_set_input(text_mask_.tensor);
      ggml_set_input(text_attention_mask_.tensor);
      ggml_set_input(positions_.tensor);
      if (config.use_caption_condition) {
        ggml_set_input(caption_ids_.tensor);
        ggml_set_input(caption_mask_.tensor);
        ggml_set_input(caption_attention_mask_.tensor);
        ggml_set_input(caption_positions_.tensor);
      }
      ggml_set_input(speaker_state_.tensor);
      ggml_set_input(has_speaker_.tensor);
      if (config.use_caption_condition) {
        ggml_set_input(has_caption_.tensor);
      }

      auto text = build_irodori_text_encoder(build_ctx, input_ids_, text_mask_,
                                             text_attention_mask_, positions_,
                                             runtime.weights_, config);
      output_text_ = core::ensure_backend_addressable_layout(build_ctx, text);
      if (config.use_caption_condition) {
        auto caption = build_irodori_caption_encoder(
            build_ctx, caption_ids_, caption_mask_, caption_attention_mask_,
            caption_positions_, runtime.weights_, config);
        output_caption_ =
            core::ensure_backend_addressable_layout(build_ctx, caption);
      }
      auto duration = build_irodori_duration_predictor(
          build_ctx, output_text_, text_mask_, speaker_state_, has_speaker_,
          output_caption_, caption_mask_, has_caption_, runtime.weights_,
          config);
      output_duration_ =
          core::ensure_backend_addressable_layout(build_ctx, duration);
      ggml_set_output(output_text_.tensor);
      if (config.use_caption_condition) {
        ggml_set_output(output_caption_.tensor);
      }
      ggml_set_output(output_duration_.tensor);
      graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
      ggml_build_forward_expand(graph_, output_text_.tensor);
      if (config.use_caption_condition) {
        ggml_build_forward_expand(graph_, output_caption_.tensor);
      }
      ggml_build_forward_expand(graph_, output_duration_.tensor);
      gallocr_ = ggml_gallocr_new(
          ggml_backend_get_default_buffer_type(runtime.backend_));
      if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) ||
          !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
        throw std::runtime_error(
            "failed to allocate Irodori-TTS condition graph");
      }
      core::write_tensor_i32(positions_, positions(tokens_));
      if (config.use_caption_condition) {
        core::write_tensor_i32(caption_positions_,
                               positions(config.max_caption_len));
      }
      core::write_tensor_f32(
          speaker_state_,
          std::vector<float>(static_cast<size_t>(2 * config.speaker_dim),
                             0.0F));
      core::write_tensor_i32(has_speaker_, std::vector<int32_t>{0});
      if (config.use_caption_condition) {
        core::write_tensor_i32(has_caption_, std::vector<int32_t>{0});
      }
    }

    ~Graph() {
      engine::core::release_backend_graph_resources(runtime_->backend_, graph_);
      if (gallocr_ != nullptr) {
        ggml_gallocr_free(gallocr_);
      }
    }

    Output run(const std::vector<int32_t> &token_ids,
               const std::vector<uint8_t> &token_mask,
               const IrodoriCaptionCondition &caption,
               const IrodoriSpeakerCondition &speaker) {
      const auto &config = runtime_->assets_->config;
      std::vector<int32_t> mask_i32(token_mask.begin(), token_mask.end());
      core::write_tensor_i32(input_ids_, token_ids);
      core::write_tensor_i32(text_mask_, mask_i32);
      if (config.use_caption_condition) {
        std::vector<int32_t> caption_mask_i32(caption.mask.begin(),
                                              caption.mask.end());
        core::write_tensor_i32(caption_ids_, caption.token_ids);
        core::write_tensor_i32(caption_mask_, caption_mask_i32);
        core::write_tensor_i32(caption_positions_,
                               positions(config.max_caption_len));
        core::write_tensor_f32(
            caption_attention_mask_,
            make_text_attention_mask(caption.mask, 1,
                                     config.caption_heads_resolved(),
                                     config.max_caption_len));
      }
      core::write_tensor_f32(
          speaker_state_, duration_speaker_state(speaker, config.speaker_dim));
      core::write_tensor_i32(has_speaker_,
                             std::vector<int32_t>{speaker.has_speaker ? 1 : 0});
      if (config.use_caption_condition) {
        core::write_tensor_i32(
            has_caption_, std::vector<int32_t>{caption.has_caption ? 1 : 0});
      }
      core::write_tensor_f32(
          text_attention_mask_,
          make_text_attention_mask(token_mask, 1, config.text_heads, tokens_));
      core::write_tensor_i32(positions_, positions(tokens_));
      core::set_backend_threads(runtime_->backend_, runtime_->threads_);
      const ggml_status status = core::compute_backend_graph(
          runtime_->backend_, graph_, nullptr, "irodori_tts.condition");
      ggml_backend_synchronize(runtime_->backend_);
      if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Irodori-TTS condition graph compute failed");
      }
      Output output;
      output.text_state = core::read_tensor_f32(output_text_.tensor);
      if (config.use_caption_condition) {
        output.caption_state = core::read_tensor_f32(output_caption_.tensor);
      }
      const auto duration = core::read_tensor_f32(output_duration_.tensor);
      output.predicted_log_frames = duration.empty() ? 0.0F : duration.front();
      return output;
    }

  private:
    IrodoriConditionRuntime *runtime_ = nullptr;
    int64_t tokens_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue input_ids_;
    core::TensorValue text_mask_;
    core::TensorValue text_attention_mask_;
    core::TensorValue positions_;
    core::TensorValue caption_ids_;
    core::TensorValue caption_mask_;
    core::TensorValue caption_attention_mask_;
    core::TensorValue caption_positions_;
    core::TensorValue speaker_state_;
    core::TensorValue has_speaker_;
    core::TensorValue has_caption_;
    core::TensorValue output_text_;
    core::TensorValue output_caption_;
    core::TensorValue output_duration_;
    ggml_cgraph *graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
  };

  std::shared_ptr<const IrodoriAssets> assets_;
  IrodoriConditionEncoderWeights weights_;
  ggml_backend_t backend_ = nullptr;
  core::BackendType backend_type_ = core::BackendType::Cpu;
  int threads_ = 1;
  size_t graph_arena_bytes_ = 0;
  std::unique_ptr<Graph> graph_;
  std::unique_ptr<SpeakerGraph> speaker_graph_;
};

class IrodoriRfRuntime {
  class ContextGraph;
  class ModulationGraph;
  class Graph;

public:
  struct ContextCache {
    uint64_t id = 0;
    int64_t batch = 0;
    int64_t speaker_tokens = 0;
    int64_t caption_tokens = 0;
    std::vector<uint8_t> text_mask;
    std::vector<uint8_t> speaker_mask;
    std::vector<uint8_t> caption_mask;
    const std::vector<IrodoriLayerContextKV> *backend_kv = nullptr;
    const std::vector<IrodoriLayerContextKV> *first_branch_backend_kv = nullptr;
  };

  struct AdaLNModulationValues {
    std::vector<float> shift;
    std::vector<float> scale;
    std::vector<float> gate;
  };

  struct LayerAdaLNModulationValues {
    AdaLNModulationValues attention;
    AdaLNModulationValues mlp;
  };

  struct ModulationCache {
    int64_t steps = 0;
    std::vector<LayerAdaLNModulationValues> layers;
  };

  IrodoriRfRuntime(std::shared_ptr<const IrodoriAssets> assets,
                   core::ExecutionContext &execution_context,
                   size_t graph_arena_bytes, size_t weight_context_bytes,
                   assets::TensorStorageType weight_storage_type)
      : assets_(std::move(assets)),
        weights_(load_irodori_rf_dit_weights(
            *assets_, execution_context.backend(),
            execution_context.backend_type(), weight_context_bytes,
            weight_storage_type)),
        backend_(execution_context.backend()),
        backend_type_(execution_context.backend_type()),
        threads_(std::max(1, execution_context.config().threads)),
        graph_arena_bytes_(graph_arena_bytes) {
    if (assets_ == nullptr) {
      throw std::runtime_error("Irodori-TTS RF runtime requires assets");
    }
  }

  ContextCache build_context_cache(const std::vector<float> &text_state_cond,
                                   const std::vector<uint8_t> &text_mask_cond,
                                   const std::vector<float> &caption_state_cond,
                                   const IrodoriCaptionCondition &caption,
                                   const IrodoriSpeakerCondition &speaker,
                                   bool text_cfg_enabled,
                                   bool speaker_cfg_enabled,
                                   bool caption_cfg_enabled) {
    const auto &config = assets_->config;
    const int64_t batch = 1 + (text_cfg_enabled ? 1 : 0) +
                          (speaker_cfg_enabled ? 1 : 0) +
                          (caption_cfg_enabled ? 1 : 0);
    const int64_t caption_tokens =
        config.use_caption_condition ? static_cast<int64_t>(caption.mask.size())
                                     : 0;
    if (static_cast<int64_t>(text_state_cond.size()) !=
            config.max_text_len * config.text_dim ||
        static_cast<int64_t>(text_mask_cond.size()) != config.max_text_len ||
        static_cast<int64_t>(speaker.state.size()) !=
            speaker.tokens * config.speaker_dim ||
        static_cast<int64_t>(speaker.mask.size()) != speaker.tokens) {
      throw std::runtime_error("Irodori-TTS RF context input shape mismatch");
    }
    if (config.use_caption_condition && caption_tokens > 0 &&
        (static_cast<int64_t>(caption_state_cond.size()) !=
             caption_tokens * config.caption_dim_resolved() ||
         static_cast<int64_t>(caption.mask.size()) != caption_tokens)) {
      throw std::runtime_error(
          "Irodori-TTS RF caption context input shape mismatch");
    }

    ContextCache cache;
    cache.id = ++next_context_id_;
    cache.batch = batch;
    cache.speaker_tokens = speaker.tokens;
    cache.caption_tokens = caption_tokens;

    const int64_t text_elems = config.max_text_len * config.text_dim;
    std::vector<float> text_state(static_cast<size_t>(batch * text_elems),
                                  0.0F);
    cache.text_mask.assign(static_cast<size_t>(batch * config.max_text_len), 0);
    auto copy_text_branch = [&](int64_t branch, bool enabled) {
      if (enabled) {
        std::copy(text_state_cond.begin(), text_state_cond.end(),
                  text_state.begin() +
                      static_cast<std::ptrdiff_t>(branch * text_elems));
        std::copy(text_mask_cond.begin(), text_mask_cond.end(),
                  cache.text_mask.begin() + static_cast<std::ptrdiff_t>(
                                                branch * config.max_text_len));
      }
    };
    copy_text_branch(0, true);
    int64_t branch = 1;
    if (text_cfg_enabled) {
      copy_text_branch(branch, false);
      ++branch;
    }
    if (speaker_cfg_enabled) {
      copy_text_branch(branch, true);
      ++branch;
    }
    if (caption_cfg_enabled) {
      copy_text_branch(branch, true);
    }

    const int64_t speaker_elems = speaker.tokens * config.speaker_dim;
    std::vector<float> speaker_state(static_cast<size_t>(batch * speaker_elems),
                                     0.0F);
    cache.speaker_mask.assign(static_cast<size_t>(batch * speaker.tokens), 0);
    auto copy_speaker_branch = [&](int64_t target_branch, bool enabled) {
      if (enabled) {
        std::copy(speaker.state.begin(), speaker.state.end(),
                  speaker_state.begin() + static_cast<std::ptrdiff_t>(
                                              target_branch * speaker_elems));
        std::copy(
            speaker.mask.begin(), speaker.mask.end(),
            cache.speaker_mask.begin() +
                static_cast<std::ptrdiff_t>(target_branch * speaker.tokens));
      }
    };
    copy_speaker_branch(0, true);
    branch = 1;
    if (text_cfg_enabled) {
      copy_speaker_branch(branch, true);
      ++branch;
    }
    if (speaker_cfg_enabled) {
      copy_speaker_branch(branch, false);
      ++branch;
    }
    if (caption_cfg_enabled) {
      copy_speaker_branch(branch, true);
    }

    std::vector<float> caption_state;
    if (config.use_caption_condition && caption_tokens > 0) {
      const int64_t caption_elems =
          caption_tokens * config.caption_dim_resolved();
      caption_state.assign(static_cast<size_t>(batch * caption_elems), 0.0F);
      cache.caption_mask.assign(static_cast<size_t>(batch * caption_tokens), 0);
      auto copy_caption_branch = [&](int64_t target_branch, bool enabled) {
        if (enabled) {
          std::copy(caption_state_cond.begin(), caption_state_cond.end(),
                    caption_state.begin() + static_cast<std::ptrdiff_t>(
                                                target_branch * caption_elems));
          std::copy(
              caption.mask.begin(), caption.mask.end(),
              cache.caption_mask.begin() +
                  static_cast<std::ptrdiff_t>(target_branch * caption_tokens));
        }
      };
      copy_caption_branch(0, true);
      branch = 1;
      if (text_cfg_enabled) {
        copy_caption_branch(branch, true);
        ++branch;
      }
      if (speaker_cfg_enabled) {
        copy_caption_branch(branch, true);
        ++branch;
      }
      if (caption_cfg_enabled) {
        copy_caption_branch(branch, false);
      }
    }

    if (context_graph_ == nullptr ||
        context_graph_->speaker_tokens() != speaker.tokens ||
        context_graph_->caption_tokens() != caption_tokens ||
        context_graph_->batch() != batch) {
      ++context_graph_rebuilds_;
      context_graph_ = std::make_unique<ContextGraph>(
          *this, speaker.tokens, caption_tokens, batch, graph_arena_bytes_);
    }
    context_graph_->run(text_state, speaker_state, caption_state, cache);
    return cache;
  }

  ModulationCache build_modulation_cache(const std::vector<float> &timesteps) {
    const int64_t steps = static_cast<int64_t>(timesteps.size());
    if (steps <= 0) {
      throw std::runtime_error("Irodori-TTS RF modulation cache needs steps");
    }
    if (modulation_graph_ == nullptr || modulation_graph_->steps() != steps) {
      modulation_graph_ =
          std::make_unique<ModulationGraph>(*this, steps, graph_arena_bytes_);
    }
    return modulation_graph_->run(timesteps);
  }

  void run_step(const std::vector<float> &x_t, int64_t step,
                const ModulationCache &modulation_cache,
                const ContextCache &context_cache, bool cfg_active,
                bool text_cfg_enabled, bool speaker_cfg_enabled,
                bool caption_cfg_enabled, float text_guidance_scale,
                float speaker_guidance_scale, float caption_guidance_scale,
                int64_t latent_steps, std::vector<float> &velocity) {
    const int64_t batch = 1 + (text_cfg_enabled ? 1 : 0) +
                          (speaker_cfg_enabled ? 1 : 0) +
                          (caption_cfg_enabled ? 1 : 0);
    const int64_t caption_tokens = context_cache.caption_tokens;
    std::unique_ptr<Graph> &graph = batch == 1 ? cond_graph_ : cfg_graph_;
    if (graph == nullptr || graph->latent_steps() != latent_steps ||
        graph->speaker_tokens() != context_cache.speaker_tokens ||
        graph->caption_tokens() != caption_tokens || graph->batch() != batch) {
      ++step_graph_rebuilds_;
      graph = std::make_unique<Graph>(
          *this, latent_steps, context_cache.speaker_tokens, caption_tokens,
          batch, graph_arena_bytes_);
    }
    graph->set_context(context_cache);
    graph->run(x_t, step, modulation_cache, context_cache, cfg_active,
               text_cfg_enabled,
               speaker_cfg_enabled, caption_cfg_enabled, text_guidance_scale,
               speaker_guidance_scale, caption_guidance_scale, velocity);
  }

  int64_t context_graph_rebuilds() const noexcept {
    return context_graph_rebuilds_;
  }

  int64_t step_graph_rebuilds() const noexcept { return step_graph_rebuilds_; }

private:
  class ContextGraph {
  public:
    ContextGraph(IrodoriRfRuntime &runtime, int64_t speaker_tokens,
                 int64_t caption_tokens, int64_t batch,
                 size_t graph_arena_bytes)
        : runtime_(&runtime), speaker_tokens_(speaker_tokens),
          caption_tokens_(caption_tokens), batch_(batch) {
      const auto &config = runtime.assets_->config;
      ggml_init_params params{graph_arena_bytes, nullptr, true};
      ctx_.reset(ggml_init(params));
      if (ctx_ == nullptr) {
        throw std::runtime_error(
            "failed to initialize Irodori-TTS RF context graph context");
      }
      core::ModuleBuildContext build_ctx{ctx_.get(),
                                         "irodori_tts.rf_dit.context_cache",
                                         runtime.backend_type_};
      text_state_ = core::make_tensor(
          build_ctx, GGML_TYPE_F32,
          core::TensorShape::from_dims(
              {batch_, config.max_text_len, config.text_dim}));
      speaker_state_ =
          core::make_tensor(build_ctx, GGML_TYPE_F32,
                            core::TensorShape::from_dims(
                                {batch_, speaker_tokens_, config.speaker_dim}));
      if (config.use_caption_condition && caption_tokens_ > 0) {
        caption_state_ = core::make_tensor(
            build_ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims(
                {batch_, caption_tokens_, config.caption_dim_resolved()}));
      }
      ggml_set_input(text_state_.tensor);
      ggml_set_input(speaker_state_.tensor);
      if (config.use_caption_condition && caption_tokens_ > 0) {
        ggml_set_input(caption_state_.tensor);
      }
      outputs_ = build_irodori_context_kv_cache(build_ctx, text_state_,
                                                speaker_state_, caption_state_,
                                                runtime.weights_, config);
      if (batch_ > 1) {
        first_branch_outputs_.reserve(outputs_.size());
        for (auto &layer : outputs_) {
          IrodoriLayerContextKV first;
          first.k_context =
              modules::SliceModule({0, 0, 1}).build(build_ctx, layer.k_context);
          first.v_context =
              modules::SliceModule({0, 0, 1}).build(build_ctx, layer.v_context);
          first_branch_outputs_.push_back(first);
        }
      }
      for (auto &layer : outputs_) {
        ggml_set_output(layer.k_context.tensor);
        ggml_set_output(layer.v_context.tensor);
      }
      graph_ = ggml_new_graph_custom(ctx_.get(), 131072, false);
      for (auto &layer : outputs_) {
        ggml_build_forward_expand(graph_, layer.k_context.tensor);
        ggml_build_forward_expand(graph_, layer.v_context.tensor);
      }
      buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime.backend_);
      if (buffer_ == nullptr) {
        throw std::runtime_error(
            "failed to allocate Irodori-TTS RF context graph");
      }
    }

    ~ContextGraph() {
      engine::core::release_backend_graph_resources(runtime_->backend_, graph_);
      if (buffer_ != nullptr) {
        ggml_backend_buffer_free(buffer_);
      }
    }

    int64_t speaker_tokens() const noexcept { return speaker_tokens_; }

    int64_t caption_tokens() const noexcept { return caption_tokens_; }

    int64_t batch() const noexcept { return batch_; }

    void run(const std::vector<float> &text_state,
             const std::vector<float> &speaker_state,
             const std::vector<float> &caption_state,
             ContextCache &cache) const {
      const auto &config = runtime_->assets_->config;
      if (static_cast<int64_t>(text_state.size()) !=
              batch_ * config.max_text_len * config.text_dim ||
          static_cast<int64_t>(speaker_state.size()) !=
              batch_ * speaker_tokens_ * config.speaker_dim) {
        throw std::runtime_error(
            "Irodori-TTS RF context graph input shape mismatch");
      }
      if (config.use_caption_condition && caption_tokens_ > 0 &&
          static_cast<int64_t>(caption_state.size()) !=
              batch_ * caption_tokens_ * config.caption_dim_resolved()) {
        throw std::runtime_error(
            "Irodori-TTS RF caption context graph input shape mismatch");
      }
      core::write_tensor_f32(text_state_, text_state);
      core::write_tensor_f32(speaker_state_, speaker_state);
      if (config.use_caption_condition && caption_tokens_ > 0) {
        core::write_tensor_f32(caption_state_, caption_state);
      }
      core::set_backend_threads(runtime_->backend_, runtime_->threads_);
      const ggml_status status =
          core::compute_backend_graph(runtime_->backend_, graph_, nullptr,
                                      "irodori_tts.rf_dit.context_cache");
      ggml_backend_synchronize(runtime_->backend_);
      if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Irodori-TTS RF context graph compute failed");
      }
      cache.backend_kv = &outputs_;
      cache.first_branch_backend_kv =
          batch_ > 1 ? &first_branch_outputs_ : &outputs_;
    }

  private:
    IrodoriRfRuntime *runtime_ = nullptr;
    int64_t speaker_tokens_ = 0;
    int64_t caption_tokens_ = 0;
    int64_t batch_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue text_state_;
    core::TensorValue speaker_state_;
    core::TensorValue caption_state_;
    std::vector<IrodoriLayerContextKV> outputs_;
    std::vector<IrodoriLayerContextKV> first_branch_outputs_;
    ggml_cgraph *graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
  };

  class ModulationGraph {
  public:
    ModulationGraph(IrodoriRfRuntime &runtime, int64_t steps,
                    size_t graph_arena_bytes)
        : runtime_(&runtime), steps_(steps) {
      const auto &config = runtime.assets_->config;
      ggml_init_params params{graph_arena_bytes, nullptr, true};
      ctx_.reset(ggml_init(params));
      if (ctx_ == nullptr) {
        throw std::runtime_error(
            "failed to initialize Irodori-TTS RF modulation graph context");
      }
      core::ModuleBuildContext build_ctx{
          ctx_.get(), "irodori_tts.rf_dit.adaln_modulation",
          runtime.backend_type_};
      timesteps_ = core::make_tensor(build_ctx, GGML_TYPE_F32,
                                     core::TensorShape::from_dims({steps_}));
      ggml_set_input(timesteps_.tensor);
      outputs_ = build_irodori_adaln_modulation_cache(
          build_ctx, timesteps_, runtime.weights_, config);
      for (auto &layer : outputs_) {
        ggml_set_output(layer.attention.shift.tensor);
        ggml_set_output(layer.attention.scale.tensor);
        ggml_set_output(layer.attention.gate.tensor);
        ggml_set_output(layer.mlp.shift.tensor);
        ggml_set_output(layer.mlp.scale.tensor);
        ggml_set_output(layer.mlp.gate.tensor);
      }
      graph_ = ggml_new_graph_custom(ctx_.get(), 131072, false);
      for (auto &layer : outputs_) {
        ggml_build_forward_expand(graph_, layer.attention.shift.tensor);
        ggml_build_forward_expand(graph_, layer.attention.scale.tensor);
        ggml_build_forward_expand(graph_, layer.attention.gate.tensor);
        ggml_build_forward_expand(graph_, layer.mlp.shift.tensor);
        ggml_build_forward_expand(graph_, layer.mlp.scale.tensor);
        ggml_build_forward_expand(graph_, layer.mlp.gate.tensor);
      }
      buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime.backend_);
      if (buffer_ == nullptr) {
        throw std::runtime_error(
            "failed to allocate Irodori-TTS RF modulation graph");
      }
    }

    ~ModulationGraph() {
      engine::core::release_backend_graph_resources(runtime_->backend_, graph_);
      if (buffer_ != nullptr) {
        ggml_backend_buffer_free(buffer_);
      }
    }

    int64_t steps() const noexcept { return steps_; }

    ModulationCache run(const std::vector<float> &timesteps) const {
      if (static_cast<int64_t>(timesteps.size()) != steps_) {
        throw std::runtime_error(
            "Irodori-TTS RF modulation timestep count mismatch");
      }
      core::write_tensor_f32(timesteps_, timesteps);
      core::set_backend_threads(runtime_->backend_, runtime_->threads_);
      const ggml_status status = core::compute_backend_graph(
          runtime_->backend_, graph_, nullptr,
          "irodori_tts.rf_dit.adaln_modulation");
      ggml_backend_synchronize(runtime_->backend_);
      if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(
            "Irodori-TTS RF modulation graph compute failed");
      }
      ModulationCache cache;
      cache.steps = steps_;
      cache.layers.resize(outputs_.size());
      for (size_t layer = 0; layer < outputs_.size(); ++layer) {
        core::read_tensor_f32_into(outputs_[layer].attention.shift.tensor,
                                   cache.layers[layer].attention.shift);
        core::read_tensor_f32_into(outputs_[layer].attention.scale.tensor,
                                   cache.layers[layer].attention.scale);
        core::read_tensor_f32_into(outputs_[layer].attention.gate.tensor,
                                   cache.layers[layer].attention.gate);
        core::read_tensor_f32_into(outputs_[layer].mlp.shift.tensor,
                                   cache.layers[layer].mlp.shift);
        core::read_tensor_f32_into(outputs_[layer].mlp.scale.tensor,
                                   cache.layers[layer].mlp.scale);
        core::read_tensor_f32_into(outputs_[layer].mlp.gate.tensor,
                                   cache.layers[layer].mlp.gate);
      }
      return cache;
    }

  private:
    IrodoriRfRuntime *runtime_ = nullptr;
    int64_t steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue timesteps_;
    std::vector<IrodoriLayerAdaLNModulation> outputs_;
    ggml_cgraph *graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
  };

  class Graph {
  public:
    Graph(IrodoriRfRuntime &runtime, int64_t latent_steps,
          int64_t speaker_tokens, int64_t caption_tokens, int64_t batch,
          size_t graph_arena_bytes)
        : runtime_(&runtime), latent_steps_(latent_steps),
          speaker_tokens_(speaker_tokens), caption_tokens_(caption_tokens),
          batch_(batch) {
      const auto &config = runtime.assets_->config;
      ggml_init_params params{graph_arena_bytes, nullptr, true};
      ctx_.reset(ggml_init(params));
      if (ctx_ == nullptr) {
        throw std::runtime_error(
            "failed to initialize Irodori-TTS RF graph context");
      }
      core::ModuleBuildContext build_ctx{ctx_.get(), "irodori_tts.rf_dit",
                                         runtime.backend_type_};
      x_t_ = core::make_tensor(
          build_ctx, GGML_TYPE_F32,
          core::TensorShape::from_dims(
              {batch_, latent_steps_, config.patched_latent_dim()}));
      const int64_t dim = config.model_dim / config.num_heads;
      const int64_t context_tokens =
          config.max_text_len + speaker_tokens_ + caption_tokens_;
      context_kv_inputs_.reserve(runtime.weights_.blocks.size());
      for (size_t layer = 0; layer < runtime.weights_.blocks.size(); ++layer) {
        IrodoriLayerContextKV kv;
        kv.k_context = core::make_tensor(
            build_ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims(
                {batch_, context_tokens, config.num_heads, dim}));
        kv.v_context = core::make_tensor(
            build_ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims(
                {batch_, context_tokens, config.num_heads, dim}));
        ggml_set_input(kv.k_context.tensor);
        ggml_set_input(kv.v_context.tensor);
        context_kv_inputs_.push_back(kv);
      }
      modulation_inputs_.reserve(runtime.weights_.blocks.size());
      for (size_t layer = 0; layer < runtime.weights_.blocks.size(); ++layer) {
        IrodoriLayerAdaLNModulation modulation;
        auto make_mod_tensor = [&]() {
          auto tensor = core::make_tensor(
              build_ctx, GGML_TYPE_F32,
              core::TensorShape::from_dims({1, 1, config.model_dim}));
          ggml_set_input(tensor.tensor);
          return tensor;
        };
        modulation.attention.shift = make_mod_tensor();
        modulation.attention.scale = make_mod_tensor();
        modulation.attention.gate = make_mod_tensor();
        modulation.mlp.shift = make_mod_tensor();
        modulation.mlp.scale = make_mod_tensor();
        modulation.mlp.gate = make_mod_tensor();
        modulation_inputs_.push_back(modulation);
      }
      attention_mask_ =
          core::make_tensor(build_ctx, GGML_TYPE_F16,
                            core::TensorShape::from_dims(
                                {batch_, 1, latent_steps_,
                                 latent_steps_ + config.max_text_len +
                                     speaker_tokens_ + caption_tokens_}));
      positions_ =
          core::make_tensor(build_ctx, GGML_TYPE_I32,
                            core::TensorShape::from_dims({latent_steps_}));
      ggml_set_input(x_t_.tensor);
      ggml_set_input(attention_mask_.tensor);
      ggml_set_input(positions_.tensor);
      auto output = build_irodori_rf_dit(
          build_ctx, x_t_, {}, {}, {}, {}, attention_mask_, positions_,
          runtime.weights_, config, &context_kv_inputs_, &modulation_inputs_);
      output_ = core::ensure_backend_addressable_layout(build_ctx, output);
      ggml_set_output(output_.tensor);
      graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
      ggml_build_forward_expand(graph_, output_.tensor);
      buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime.backend_);
      if (buffer_ == nullptr) {
        throw std::runtime_error("failed to allocate Irodori-TTS RF graph");
      }
      core::write_tensor_i32(positions_, positions(latent_steps_));
    }

    ~Graph() {
      engine::core::release_backend_graph_resources(runtime_->backend_, graph_);
      if (buffer_ != nullptr) {
        ggml_backend_buffer_free(buffer_);
      }
    }

    int64_t latent_steps() const noexcept { return latent_steps_; }

    int64_t speaker_tokens() const noexcept { return speaker_tokens_; }

    int64_t caption_tokens() const noexcept { return caption_tokens_; }

    int64_t batch() const noexcept { return batch_; }

    void set_context(const ContextCache &cache) {
      if (loaded_context_id_ == cache.id) {
        return;
      }
      const auto &config = runtime_->assets_->config;
      const bool use_first_branch =
          cache.batch != batch_ && batch_ == 1 && cache.batch > 1;
      if ((!use_first_branch && cache.batch != batch_) ||
          cache.speaker_tokens != speaker_tokens_ ||
          cache.caption_tokens != caption_tokens_) {
        throw std::runtime_error("Irodori-TTS RF context cache shape mismatch");
      }
      if (cache.backend_kv == nullptr ||
          cache.backend_kv->size() != context_kv_inputs_.size()) {
        throw std::runtime_error(
            "Irodori-TTS RF context cache layer count mismatch");
      }
      const auto *source_kv = cache.backend_kv;
      if (use_first_branch) {
        source_kv = cache.first_branch_backend_kv;
        if (source_kv == nullptr ||
            source_kv->size() != context_kv_inputs_.size()) {
          throw std::runtime_error(
              "Irodori-TTS RF first-branch context cache layer count mismatch");
        }
      }
      const int64_t context_tokens =
          config.max_text_len + speaker_tokens_ + caption_tokens_;
      const size_t context_values =
          static_cast<size_t>(cache.batch * context_tokens * config.num_heads *
                              (config.model_dim / config.num_heads));
      const size_t write_context_values =
          static_cast<size_t>(batch_ * context_tokens * config.num_heads *
                              (config.model_dim / config.num_heads));
      auto upload_context_tensor = [&](const core::TensorValue &source,
                                       const core::TensorValue &target,
                                       size_t source_values,
                                       size_t target_values) {
        if (source.shape.num_elements() !=
                static_cast<int64_t>(source_values) ||
            target.shape.num_elements() !=
                static_cast<int64_t>(target_values)) {
          throw std::runtime_error(
              "Irodori-TTS RF context cache tensor shape mismatch");
        }
        ggml_backend_tensor_copy(source.tensor, target.tensor);
      };
      const size_t source_context_values =
          use_first_branch ? write_context_values : context_values;
      for (size_t layer = 0; layer < context_kv_inputs_.size(); ++layer) {
        const auto &source = (*source_kv)[layer];
        upload_context_tensor(source.k_context,
                              context_kv_inputs_[layer].k_context,
                              source_context_values, write_context_values);
        upload_context_tensor(source.v_context,
                              context_kv_inputs_[layer].v_context,
                              source_context_values, write_context_values);
      }
      const auto attention_mask_values = make_rf_attention_mask(
          cache.text_mask, cache.speaker_mask, cache.caption_mask, batch_,
          latent_steps_, config.max_text_len, speaker_tokens_, caption_tokens_);
      core::write_tensor_f16(attention_mask_, attention_mask_values);
      loaded_context_id_ = cache.id;
    }

    void run(const std::vector<float> &x_t, int64_t step,
             const ModulationCache &modulation_cache,
             const ContextCache &context_cache, bool cfg_active,
             bool text_cfg_enabled, bool speaker_cfg_enabled,
             bool caption_cfg_enabled, float text_guidance_scale,
             float speaker_guidance_scale, float caption_guidance_scale,
             std::vector<float> &velocity) {
      const auto &config = runtime_->assets_->config;
      const int64_t x_elems = latent_steps_ * config.patched_latent_dim();
      const bool use_first_branch =
          context_cache.batch != batch_ && batch_ == 1 &&
          context_cache.batch > 1;
      if (static_cast<int64_t>(x_t.size()) != x_elems ||
          (!use_first_branch && context_cache.batch != batch_) ||
          context_cache.speaker_tokens != speaker_tokens_ ||
          context_cache.caption_tokens != caption_tokens_ ||
          static_cast<int64_t>(context_cache.text_mask.size()) !=
              context_cache.batch * config.max_text_len ||
          static_cast<int64_t>(context_cache.speaker_mask.size()) !=
              context_cache.batch * speaker_tokens_) {
        throw std::runtime_error("Irodori-TTS RF step input shape mismatch");
      }
      if (config.use_caption_condition && caption_tokens_ > 0 &&
          static_cast<int64_t>(context_cache.caption_mask.size()) !=
              context_cache.batch * caption_tokens_) {
        throw std::runtime_error("Irodori-TTS RF caption input shape mismatch");
      }
      if (step < 0 || step >= modulation_cache.steps ||
          modulation_cache.layers.size() != modulation_inputs_.size()) {
        throw std::runtime_error("Irodori-TTS RF modulation cache mismatch");
      }
      const size_t modulation_offset =
          static_cast<size_t>(step * config.model_dim);
      auto write_modulation = [&](const AdaLNModulationValues &source,
                                  const IrodoriAdaLNModulation &target) {
        const size_t expected_values =
            static_cast<size_t>(modulation_cache.steps * config.model_dim);
        if (source.shift.size() != expected_values ||
            source.scale.size() != expected_values ||
            source.gate.size() != expected_values) {
          throw std::runtime_error(
              "Irodori-TTS RF modulation cache tensor shape mismatch");
        }
        core::write_tensor_f32(target.shift,
                               source.shift.data() + modulation_offset,
                               static_cast<size_t>(config.model_dim));
        core::write_tensor_f32(target.scale,
                               source.scale.data() + modulation_offset,
                               static_cast<size_t>(config.model_dim));
        core::write_tensor_f32(target.gate,
                               source.gate.data() + modulation_offset,
                               static_cast<size_t>(config.model_dim));
      };
      for (size_t layer = 0; layer < modulation_inputs_.size(); ++layer) {
        write_modulation(modulation_cache.layers[layer].attention,
                         modulation_inputs_[layer].attention);
        write_modulation(modulation_cache.layers[layer].mlp,
                         modulation_inputs_[layer].mlp);
      }
      if (batch_ == 1) {
        core::write_tensor_f32(x_t_, x_t.data(), x_t.size());
      } else {
        x_batch_scratch_.resize(static_cast<size_t>(batch_ * x_elems));
        for (int64_t b = 0; b < batch_; ++b) {
          std::copy(x_t.begin(), x_t.end(),
                    x_batch_scratch_.begin() +
                        static_cast<std::ptrdiff_t>(b * x_elems));
        }
        core::write_tensor_f32(x_t_, x_batch_scratch_.data(),
                               x_batch_scratch_.size());
      }

      core::set_backend_threads(runtime_->backend_, runtime_->threads_);
      const ggml_status status = core::compute_backend_graph(
          runtime_->backend_, graph_, nullptr, "irodori_tts.rf_dit");
      ggml_backend_synchronize(runtime_->backend_);
      if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Irodori-TTS RF graph compute failed");
      }
      core::read_tensor_f32_into(output_.tensor, output_scratch_);
      velocity.resize(static_cast<size_t>(x_elems));
      std::copy(output_scratch_.begin(),
                output_scratch_.begin() + static_cast<std::ptrdiff_t>(x_elems),
                velocity.begin());
      if (cfg_active) {
        int64_t branch = 1;
        if (text_cfg_enabled) {
          for (int64_t i = 0; i < x_elems; ++i) {
            const float cond = output_scratch_[static_cast<size_t>(i)];
            const float uncond =
                output_scratch_[static_cast<size_t>(branch * x_elems + i)];
            velocity[static_cast<size_t>(i)] +=
                text_guidance_scale * (cond - uncond);
          }
          ++branch;
        }
        if (speaker_cfg_enabled) {
          for (int64_t i = 0; i < x_elems; ++i) {
            const float cond = output_scratch_[static_cast<size_t>(i)];
            const float uncond =
                output_scratch_[static_cast<size_t>(branch * x_elems + i)];
            velocity[static_cast<size_t>(i)] +=
                speaker_guidance_scale * (cond - uncond);
          }
          ++branch;
        }
        if (caption_cfg_enabled) {
          for (int64_t i = 0; i < x_elems; ++i) {
            const float cond = output_scratch_[static_cast<size_t>(i)];
            const float uncond =
                output_scratch_[static_cast<size_t>(branch * x_elems + i)];
            velocity[static_cast<size_t>(i)] +=
                caption_guidance_scale * (cond - uncond);
          }
        }
      }
    }

  private:
    IrodoriRfRuntime *runtime_ = nullptr;
    int64_t latent_steps_ = 0;
    int64_t speaker_tokens_ = 0;
    int64_t caption_tokens_ = 0;
    int64_t batch_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue x_t_;
    std::vector<IrodoriLayerContextKV> context_kv_inputs_;
    std::vector<IrodoriLayerAdaLNModulation> modulation_inputs_;
    core::TensorValue attention_mask_;
    core::TensorValue positions_;
    core::TensorValue output_;
    std::vector<float> x_batch_scratch_;
    std::vector<float> output_scratch_;
    uint64_t loaded_context_id_ = 0;
    ggml_cgraph *graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
  };

  std::shared_ptr<const IrodoriAssets> assets_;
  IrodoriRfDitWeights weights_;
  ggml_backend_t backend_ = nullptr;
  core::BackendType backend_type_ = core::BackendType::Cpu;
  int threads_ = 1;
  size_t graph_arena_bytes_ = 0;
  uint64_t next_context_id_ = 0;
  int64_t context_graph_rebuilds_ = 0;
  int64_t step_graph_rebuilds_ = 0;
  std::unique_ptr<ContextGraph> context_graph_;
  std::unique_ptr<ModulationGraph> modulation_graph_;
  std::unique_ptr<Graph> cond_graph_;
  std::unique_ptr<Graph> cfg_graph_;
};

class IrodoriCodecRuntime {
public:
  IrodoriCodecRuntime(std::shared_ptr<const IrodoriAssets> assets,
                      core::ExecutionContext &execution_context,
                      size_t graph_arena_bytes, size_t weight_context_bytes,
                      assets::TensorStorageType weight_storage_type)
      : assets_(std::move(assets)),
        weights_(load_irodori_codec_weights(
            *assets_, execution_context.backend(),
            execution_context.backend_type(), weight_context_bytes,
            weight_storage_type)),
        backend_(execution_context.backend()),
        backend_type_(execution_context.backend_type()),
        threads_(std::max(1, execution_context.config().threads)),
        graph_arena_bytes_(graph_arena_bytes) {
    if (assets_ == nullptr) {
      throw std::runtime_error("Irodori-TTS codec runtime requires assets");
    }
  }

  runtime::AudioBuffer decode(const std::vector<float> &latent,
                              int64_t latent_steps, int64_t target_samples) {
    const bool graph_rebuild =
        graph_ == nullptr || graph_->latent_steps() != latent_steps;
    if (graph_rebuild) {
      graph_ = std::make_unique<Graph>(*this, latent_steps, graph_arena_bytes_);
    }
    debug::timing_log_scalar("irodori_tts.codec_decode.graph_rebuild",
                             graph_rebuild);
    return graph_->run(latent, target_samples);
  }

  std::vector<float> encode_reference(const runtime::AudioBuffer &audio,
                                      int64_t &latent_steps_out) {
    auto mono = engine::audio::mixdown_interleaved_to_mono_average(
        audio.samples, audio.channels);
    if (audio.sample_rate != assets_->codec.sample_rate) {
      engine::audio::TorchaudioSincHannResampleOptions options;
      options.kernel_mode = engine::audio::TorchaudioSincHannKernelMode::
          Float32ComputationStoredAsFloat32;
      options.accumulation =
          engine::audio::TorchaudioSincHannAccumulation::Float32;
      mono = engine::audio::resample_mono_torchaudio_sinc_hann(
          mono, audio.sample_rate, assets_->codec.sample_rate, options);
    }
    normalize_reference_audio_in_place(mono, -16.0);
    mono = reflect_pad_right_to_multiple(std::move(mono),
                                         assets_->codec.hop_length);
    const int64_t padded_samples = static_cast<int64_t>(mono.size());
    const bool graph_rebuild = encode_graph_ == nullptr ||
                               encode_graph_->padded_samples() != padded_samples;
    if (graph_rebuild) {
      encode_graph_ = std::make_unique<EncodeGraph>(*this, padded_samples,
                                                    graph_arena_bytes_);
    }
    debug::timing_log_scalar("irodori_tts.codec_encode.graph_rebuild",
                             graph_rebuild);
    return encode_graph_->run(mono, latent_steps_out);
  }

private:
  class EncodeGraph {
  public:
    EncodeGraph(IrodoriCodecRuntime &runtime, int64_t padded_samples,
                size_t graph_arena_bytes)
        : runtime_(&runtime), padded_samples_(padded_samples) {
      const auto &config = runtime.assets_->codec;
      ggml_init_params params{graph_arena_bytes, nullptr, true};
      ctx_.reset(ggml_init(params));
      if (ctx_ == nullptr) {
        throw std::runtime_error(
            "failed to initialize Irodori-TTS codec encode graph context");
      }
      core::ModuleBuildContext build_ctx{ctx_.get(), "irodori_tts.codec_encode",
                                         runtime.backend_type_};
      waveform_ = core::make_tensor(
          build_ctx, GGML_TYPE_F32,
          core::TensorShape::from_dims({1, 1, padded_samples_}));
      ggml_set_input(waveform_.tensor);
      auto output = build_irodori_codec_encode(build_ctx, waveform_,
                                               runtime.weights_, config);
      output_ = core::ensure_backend_addressable_layout(build_ctx, output);
      ggml_set_output(output_.tensor);
      graph_ = ggml_new_graph_custom(ctx_.get(), 131072, false);
      ggml_build_forward_expand(graph_, output_.tensor);
      gallocr_ = ggml_gallocr_new(
          ggml_backend_get_default_buffer_type(runtime.backend_));
      if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) ||
          !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
        throw std::runtime_error(
            "failed to allocate Irodori-TTS codec encode graph");
      }
    }

    ~EncodeGraph() {
      engine::core::release_backend_graph_resources(runtime_->backend_, graph_);
      if (gallocr_ != nullptr) {
        ggml_gallocr_free(gallocr_);
      }
    }

    int64_t padded_samples() const noexcept { return padded_samples_; }

    std::vector<float> run(const std::vector<float> &mono,
                           int64_t &latent_steps_out) {
      if (static_cast<int64_t>(mono.size()) != padded_samples_) {
        throw std::runtime_error(
            "Irodori-TTS codec encode audio size mismatch");
      }
      core::write_tensor_f32(waveform_, mono);
      core::set_backend_threads(runtime_->backend_, runtime_->threads_);
      const ggml_status status = core::compute_backend_graph(
          runtime_->backend_, graph_, nullptr, "irodori_tts.codec_encode");
      ggml_backend_synchronize(runtime_->backend_);
      if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(
            "Irodori-TTS codec encode graph compute failed");
      }
      latent_steps_out = output_.shape.dims[1];
      return core::read_tensor_f32(output_.tensor);
    }

  private:
    IrodoriCodecRuntime *runtime_ = nullptr;
    int64_t padded_samples_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue waveform_;
    core::TensorValue output_;
    ggml_cgraph *graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
  };

  class Graph {
  public:
    Graph(IrodoriCodecRuntime &runtime, int64_t latent_steps,
          size_t graph_arena_bytes)
        : runtime_(&runtime), latent_steps_(latent_steps) {
      const auto &config = runtime.assets_->config;
      ggml_init_params params{graph_arena_bytes, nullptr, true};
      ctx_.reset(ggml_init(params));
      if (ctx_ == nullptr) {
        throw std::runtime_error(
            "failed to initialize Irodori-TTS codec graph context");
      }
      core::ModuleBuildContext build_ctx{ctx_.get(), "irodori_tts.codec_decode",
                                         runtime.backend_type_};
      latent_ = core::make_tensor(
          build_ctx, GGML_TYPE_F32,
          core::TensorShape::from_dims({1, latent_steps_, config.latent_dim}));
      ggml_set_input(latent_.tensor);
      auto output = build_irodori_codec_decode(
          build_ctx, latent_, runtime.weights_, runtime.assets_->codec);
      output_ = core::ensure_backend_addressable_layout(build_ctx, output);
      ggml_set_output(output_.tensor);
      graph_ = ggml_new_graph_custom(ctx_.get(), 131072, false);
      ggml_build_forward_expand(graph_, output_.tensor);
      gallocr_ = ggml_gallocr_new(
          ggml_backend_get_default_buffer_type(runtime.backend_));
      if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) ||
          !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
        throw std::runtime_error("failed to allocate Irodori-TTS codec graph");
      }
    }

    ~Graph() {
      engine::core::release_backend_graph_resources(runtime_->backend_, graph_);
      if (gallocr_ != nullptr) {
        ggml_gallocr_free(gallocr_);
      }
    }

    int64_t latent_steps() const noexcept { return latent_steps_; }

    runtime::AudioBuffer run(const std::vector<float> &latent,
                             int64_t target_samples) {
      const auto &config = runtime_->assets_->config;
      if (static_cast<int64_t>(latent.size()) !=
          latent_steps_ * config.latent_dim) {
        throw std::runtime_error("Irodori-TTS codec latent size mismatch");
      }
      core::write_tensor_f32(latent_, latent);
      core::set_backend_threads(runtime_->backend_, runtime_->threads_);
      const ggml_status status = core::compute_backend_graph(
          runtime_->backend_, graph_, nullptr, "irodori_tts.codec_decode");
      ggml_backend_synchronize(runtime_->backend_);
      if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Irodori-TTS codec graph compute failed");
      }
      auto samples = core::read_tensor_f32(output_.tensor);
      if (target_samples > 0 &&
          static_cast<int64_t>(samples.size()) > target_samples) {
        samples.resize(static_cast<size_t>(target_samples));
      }
      return runtime::AudioBuffer{
          runtime_->assets_->codec.sample_rate,
          1,
          std::move(samples),
      };
    }

  private:
    IrodoriCodecRuntime *runtime_ = nullptr;
    int64_t latent_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue latent_;
    core::TensorValue output_;
    ggml_cgraph *graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
  };

  std::shared_ptr<const IrodoriAssets> assets_;
  IrodoriCodecWeights weights_;
  ggml_backend_t backend_ = nullptr;
  core::BackendType backend_type_ = core::BackendType::Cpu;
  int threads_ = 1;
  size_t graph_arena_bytes_ = 0;
  std::unique_ptr<Graph> graph_;
  std::unique_ptr<EncodeGraph> encode_graph_;
};

IrodoriTTSSession::IrodoriTTSSession(
    runtime::TaskSpec task, runtime::SessionOptions options,
    std::shared_ptr<const IrodoriAssets> assets)
    : RuntimeSessionBase(options), task_(task),
      assets_(require_assets(std::move(assets))), tokenizer_(assets_) {
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
  if (task_.mode != runtime::RunMode::Offline) {
    throw std::runtime_error("Irodori-TTS only supports offline sessions");
  }
  if (task_.task != runtime::VoiceTaskKind::Tts &&
      task_.task != runtime::VoiceTaskKind::VoiceDesign) {
    throw std::runtime_error(
        "Irodori-TTS supports only TTS and voice-design offline tasks");
  }
  condition_ = std::make_unique<IrodoriConditionRuntime>(
      assets_, execution_context(), condition_graph_arena_bytes_,
      condition_weight_context_bytes_, weight_storage_type_);
  rf_ = std::make_unique<IrodoriRfRuntime>(
      assets_, execution_context(), rf_graph_arena_bytes_,
      rf_weight_context_bytes_, weight_storage_type_);
  codec_ = std::make_unique<IrodoriCodecRuntime>(
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
  const auto chunk_requests = runtime::chunk_text_request(request, text_chunk_size);
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
    const uint64_t reference_key =
        reference_audio_cache_key(first_request.reference_audio);
    if (cached_reference_valid_ && cached_reference_key_ == reference_key &&
        cached_reference_sample_rate_ ==
            first_request.reference_audio.sample_rate &&
        cached_reference_channels_ ==
            first_request.reference_audio.channels &&
        cached_reference_samples_ ==
            first_request.reference_audio.samples.size()) {
      reference_cache_hit = true;
      speaker.state = cached_reference_speaker_state_;
      speaker.mask = cached_reference_speaker_mask_;
      speaker.tokens = cached_reference_speaker_tokens_;
      speaker.has_speaker = cached_reference_speaker_has_speaker_;
    } else {
      int64_t ref_latent_steps = 0;
      auto ref_latent = codec_->encode_reference(first_request.reference_audio,
                                                 ref_latent_steps);
      speaker =
          condition_->encode_speaker_reference(ref_latent, ref_latent_steps);
      cached_reference_key_ = reference_key;
      cached_reference_sample_rate_ =
          first_request.reference_audio.sample_rate;
      cached_reference_channels_ =
          first_request.reference_audio.channels;
      cached_reference_samples_ =
          first_request.reference_audio.samples.size();
      cached_reference_speaker_state_ = speaker.state;
      cached_reference_speaker_mask_ = speaker.mask;
      cached_reference_speaker_tokens_ = speaker.tokens;
      cached_reference_speaker_has_speaker_ = speaker.has_speaker;
      cached_reference_valid_ = true;
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
      rf_->context_graph_rebuilds();
  const int64_t rf_step_graph_rebuilds_before = rf_->step_graph_rebuilds();
  runtime::AudioBuffer merged_audio;
  double condition_ms = 0.0;
  double sample_rf_ms = 0.0;
  double rf_context_cond_ms = 0.0;
  double rf_context_cfg_ms = 0.0;
  double rf_step_cfg_ms = 0.0;
  double rf_step_cond_ms = 0.0;
  double decode_ms = 0.0;
  const int hop_length = static_cast<int>(assets_->codec.hop_length);
  for (const auto & chunk_request : chunk_requests) {
    const IrodoriRequest irodori_request = make_request(chunk_request);
    const auto text_start = Clock::now();
    const auto tokenized = tokenizer_.encode_padded(
        irodori_request.text, assets_->config.max_text_len);
    tokenize_ms += debug::elapsed_ms(text_start);

    const auto condition_start = Clock::now();
    const auto conditions =
        condition_->run(tokenized.token_ids, tokenized.mask, caption, speaker);
    condition_ms += debug::elapsed_ms(condition_start);
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
        irodori_request.generation.seed, sampling::TorchRandnPrecision::Float32);
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
    IrodoriRfRuntime::ContextCache rf_context_cond;
    IrodoriRfRuntime::ContextCache rf_context_cfg;
    if (any_cfg_enabled) {
      const auto rf_context_cfg_start = Clock::now();
      rf_context_cfg = rf_->build_context_cache(
          conditions.text_state, tokenized.mask, rf_caption_state, rf_caption,
          speaker, text_cfg_enabled, speaker_cfg_enabled, caption_cfg_enabled);
      rf_context_cfg_ms += debug::elapsed_ms(rf_context_cfg_start);
    } else {
      const auto rf_context_cond_start = Clock::now();
      rf_context_cond = rf_->build_context_cache(
          conditions.text_state, tokenized.mask, rf_caption_state, rf_caption,
          speaker, false, false, false);
      rf_context_cond_ms += debug::elapsed_ms(rf_context_cond_start);
      rf_context_cfg = rf_context_cond;
    }

    std::vector<float> timesteps(
        static_cast<size_t>(irodori_request.generation.num_inference_steps));
    for (int64_t step = 0;
         step < irodori_request.generation.num_inference_steps; ++step) {
      const float u = static_cast<float>(step) /
                      static_cast<float>(
                          irodori_request.generation.num_inference_steps);
      timesteps[static_cast<size_t>(step)] = (1.0F - u) * 0.999F;
    }
    const auto modulation_cache = rf_->build_modulation_cache(timesteps);
    std::vector<float> velocity(x_t.size());
    for (int64_t step = 0; step < irodori_request.generation.num_inference_steps; ++step) {
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
      rf_->run_step(
          x_t, step, modulation_cache, rf_context, cfg_active,
          cfg_active && text_cfg_enabled,
          cfg_active && speaker_cfg_enabled, cfg_active && caption_cfg_enabled,
          irodori_request.generation.text_guidance_scale,
          irodori_request.generation.speaker_guidance_scale,
          irodori_request.generation.caption_guidance_scale, patched_steps, velocity);
      const double rf_step_ms = debug::elapsed_ms(rf_step_start);
      if (cfg_active) {
        rf_step_cfg_ms += rf_step_ms;
      } else {
        rf_step_cond_ms += rf_step_ms;
      }
      for (size_t i = 0; i < x_t.size(); ++i) {
        x_t[i] += velocity[i] * (t_next - t);
      }
    }
    sample_rf_ms += debug::elapsed_ms(sample_start);

    std::vector<float> latent(
        static_cast<size_t>(latent_steps * assets_->config.latent_dim), 0.0F);
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
      const int flat =
          find_flattening_point(latent, latent_steps, assets_->config.latent_dim,
                                irodori_request.generation.tail_window_size,
                                irodori_request.generation.tail_std_threshold,
                                irodori_request.generation.tail_mean_threshold);
      const int64_t flattening_samples = static_cast<int64_t>(flat) * hop_length;
      if (flattening_samples > 0) {
        target_samples = std::min(target_samples, flattening_samples);
      }
    }
    const auto decode_start = Clock::now();
    runtime::append_audio_buffer(
        merged_audio, codec_->decode(latent, latent_steps, target_samples));
    decode_ms += debug::elapsed_ms(decode_start);
  }
  runtime::TaskResult result;
  result.audio_output = std::move(merged_audio);
  const auto wall_end = Clock::now();
  debug::timing_log_scalar("irodori_tts.reference.used",
                           !first_request.no_ref);
  debug::timing_log_scalar("irodori_tts.reference.cache_hit",
                           reference_cache_hit);
  debug::trace_log_scalar("irodori_tts.text_chunk_size", text_chunk_size);
  debug::trace_log_scalar("irodori_tts.text_chunk_count",
                          static_cast<int64_t>(chunk_requests.size()));
  debug::timing_log_scalar(
      "irodori_tts.sample_rf.context_graph_rebuilds",
      rf_->context_graph_rebuilds() - rf_context_graph_rebuilds_before);
  debug::timing_log_scalar(
      "irodori_tts.sample_rf.step_graph_rebuilds",
      rf_->step_graph_rebuilds() - rf_step_graph_rebuilds_before);
  debug::timing_log_scalar("irodori_tts.prepare_reference_ms",
                           debug::elapsed_ms(reference_start, reference_end));
  debug::timing_log_scalar("irodori_tts.tokenize_ms",
                           tokenize_ms);
  debug::timing_log_scalar("irodori_tts.condition_ms",
                           condition_ms);
  debug::timing_log_scalar("irodori_tts.sample_rf_ms",
                           sample_rf_ms);
  debug::timing_log_scalar(
      "irodori_tts.sample_rf.context_cond_ms",
      rf_context_cond_ms);
  debug::timing_log_scalar(
      "irodori_tts.sample_rf.context_cfg_ms",
      rf_context_cfg_ms);
  debug::timing_log_scalar("irodori_tts.sample_rf.steps_cfg_ms",
                           rf_step_cfg_ms);
  debug::timing_log_scalar("irodori_tts.sample_rf.steps_cond_ms",
                           rf_step_cond_ms);
  debug::timing_log_scalar("irodori_tts.codec_decode_ms",
                           decode_ms);
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
