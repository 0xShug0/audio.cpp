#include "engine/community_models/outetts/session.h"

#include "engine/framework/audio/fft.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/text/chunking.h"
#include "engine/framework/text/utf8.h"
#include "engine/models/qwen3_asr/assets.h"
#include "engine/models/qwen3_forced_aligner/session.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstring>
#include <deque>
#include <filesystem>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace engine::models::outetts {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int64_t kDefaultTextChunkSize = 256;
constexpr int64_t kAutomaticChunkTokenBudget = 4096;
constexpr size_t kDefaultReferenceCacheSlots = 1;

int64_t generation_budget_with_headroom(int64_t estimated_tokens) {
  // The upstream heuristic is a useful lower bound, but sampled generation can
  // occasionally need a little more room before emitting audio_end.
  return estimated_tokens + std::max<int64_t>(128, estimated_tokens / 4);
}

uint64_t mix_cache_key(uint64_t key, uint64_t value) {
  key ^= value;
  key *= 1099511628211ull;
  return key;
}

uint64_t reference_audio_hash(const runtime::AudioBuffer &audio) {
  uint64_t key = 1469598103934665603ull;
  key = mix_cache_key(key, static_cast<uint64_t>(audio.sample_rate));
  key = mix_cache_key(key, static_cast<uint64_t>(audio.channels));
  key = mix_cache_key(key, static_cast<uint64_t>(audio.samples.size()));
  for (const float sample : audio.samples) {
    uint32_t bits = 0;
    std::memcpy(&bits, &sample, sizeof(bits));
    key = mix_cache_key(key, static_cast<uint64_t>(bits));
  }
  return key;
}

size_t reference_cache_slots(const runtime::SessionOptions &options) {
  const int64_t slots = runtime::parse_i64_option(
                            options.options,
                            {"outetts.reference_cache_slots",
                             "reference_cache_slots"})
                            .value_or(
                                static_cast<int64_t>(
                                    kDefaultReferenceCacheSlots));
  if (slots < 0 ||
      static_cast<uint64_t>(slots) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw std::runtime_error(
        "outetts.reference_cache_slots must be a non-negative size");
  }
  return static_cast<size_t>(slots);
}

bool mem_saver_from_options(const runtime::SessionOptions &options) {
  if (const auto value = runtime::find_option(
          options.options, {"outetts.mem_saver", "mem_saver"})) {
    return runtime::parse_bool_option(*value, "outetts.mem_saver");
  }
  return false;
}

assets::TensorStorageType requested_weight_type(
    const runtime::SessionOptions &options) {
  const auto it = options.options.find("outetts.weight_type");
  return it == options.options.end()
             ? assets::TensorStorageType::Native
             : assets::parse_tensor_storage_type(it->second);
}

bool has_quantized_clone_weights(const runtime::SessionOptions &options,
                                 const OuteTTSAssets &model_assets) {
  constexpr std::string_view probe =
      "model.layers.0.self_attn.q_proj.weight";
  const auto source_type = assets::tensor_storage_type_for_dtype(
      model_assets.model_weights->require_metadata(probe).dtype);
  const auto requested_type = assets::resolve_tensor_storage_type(
      *model_assets.model_weights, probe, requested_weight_type(options));
  return ggml_is_quantized(
             assets::ggml_type_for_tensor_storage(source_type)) ||
         ggml_is_quantized(
             assets::ggml_type_for_tensor_storage(requested_type));
}

assets::TensorStorageType clone_weight_type(
    const runtime::SessionOptions &options,
    const OuteTTSAssets &model_assets) {
  if (options.backend.type == core::BackendType::Cuda &&
      has_quantized_clone_weights(options, model_assets)) {
    // CUDA execution with quantized OuteTTS weights diverges over the long
    // reference-codec prompt used for cloning. The GGUF stays quantized on
    // disk; only the in-memory language-model tensors are expanded to F32.
    // F16 still produces phonetic but unintelligible speech from Q8 source
    // tensors on this route, while F32 matches the coherent CPU decode.
    debug::trace_log_scalar("outetts.cuda_clone_quantized_f32_fallback", true);
    return assets::TensorStorageType::F32;
  }
  return requested_weight_type(options);
}

OuteTTSGenerateOptions
generation_options(const runtime::TaskRequest &request,
                   const OuteTTSGenerationConfig &defaults,
                   bool voice_cloning,
                   bool quantized_cloning,
                   int64_t automatic_max_new_tokens) {
  OuteTTSGenerateOptions out;
  out.max_new_tokens = automatic_max_new_tokens;
  out.temperature = voice_cloning ? 0.4F : defaults.temperature;
  out.repetition_penalty = defaults.repetition_penalty;
  out.repetition_window = defaults.repetition_window;
  out.top_k = voice_cloning ? 40 : defaults.top_k;
  out.top_p = voice_cloning ? 0.9F : defaults.top_p;
  out.min_p = voice_cloning ? 0.05F : defaults.min_p;
  if (const auto v = runtime::parse_i64_option(request.options, {"max_tokens"}))
    out.max_new_tokens = *v;
  if (const auto v =
          runtime::parse_finite_float_option(request.options, {"temperature"}))
    out.temperature = *v;
  if (const auto v = runtime::parse_finite_float_option(request.options,
                                                        {"repetition_penalty"}))
    out.repetition_penalty = *v;
  if (const auto v =
          runtime::parse_i64_option(request.options, {"repetition_window"}))
    out.repetition_window = *v;
  if (const auto v = runtime::parse_i64_option(request.options, {"top_k"}))
    out.top_k = *v;
  if (const auto v =
          runtime::parse_finite_float_option(request.options, {"top_p"}))
    out.top_p = *v;
  if (const auto v =
          runtime::parse_finite_float_option(request.options, {"min_p"}))
    out.min_p = *v;
  out.seed = runtime::parse_u32_option(request.options, {"seed"})
                 .value_or(voice_cloning
                               ? (quantized_cloning ? 42u : 4099u)
                               : runtime::random_u32_seed());
  if (out.max_new_tokens <= 0 || out.repetition_window < 0 || out.top_k < 0 ||
      out.temperature < 0.0F || out.repetition_penalty <= 0.0F ||
      out.top_p <= 0.0F || out.top_p > 1.0F || out.min_p < 0.0F ||
      out.min_p > 1.0F) {
    throw std::runtime_error("invalid OuteTTS generation options");
  }
  return out;
}

std::vector<runtime::TaskRequest> chunk_text_request_to_token_budget(
    const runtime::TaskRequest &request, int64_t requested_chunk_size,
    engine::text::TextChunkMode mode, int64_t token_budget) {
  auto initial = runtime::chunk_text_request(request, requested_chunk_size,
                                             mode);
  std::deque<runtime::TaskRequest> pending(initial.begin(), initial.end());
  std::vector<runtime::TaskRequest> chunks;
  while (!pending.empty()) {
    auto candidate = std::move(pending.front());
    pending.pop_front();
    if (!candidate.text_input.has_value()) {
      chunks.push_back(std::move(candidate));
      continue;
    }
    const auto budget =
        estimate_text_generation_budget(candidate.text_input->text);
    // The upstream estimate has an intentional 384-token floor. Preserve
    // explicitly smaller limits for short requests instead of inventing empty
    // or one-character chunks; a cap is still reported as an error after
    // generation if the model cannot stop within that caller-provided limit.
    if (budget.recommended_max_new_tokens <= token_budget ||
        token_budget < 384) {
      chunks.push_back(std::move(candidate));
      continue;
    }

    const int64_t codepoints = static_cast<int64_t>(
        engine::text::utf8_codepoint_count(candidate.text_input->text,
                                           "OuteTTS text chunk"));
    if (codepoints <= 1) {
      throw std::runtime_error(
          "OuteTTS max_tokens is too small for the requested text chunk");
    }
    const int64_t smaller_chunk_size = std::max<int64_t>(1, codepoints / 2);
    auto smaller = runtime::chunk_text_request(candidate, smaller_chunk_size,
                                               mode);
    if (smaller.size() <= 1) {
      throw std::runtime_error(
          "OuteTTS could not split text to fit the max_tokens budget");
    }
    for (auto it = smaller.rbegin(); it != smaller.rend(); ++it)
      pending.push_front(std::move(*it));
  }
  return chunks;
}

std::vector<std::string> split_words(const std::string &text) {
  std::istringstream input(text);
  std::vector<std::string> words;
  std::string word;
  while (input >> word)
    words.push_back(word);
  return words;
}

size_t utf8_length(const std::string &text) {
  return std::max<size_t>(
      1, static_cast<size_t>(
             std::count_if(text.begin(), text.end(), [](unsigned char value) {
               return (value & 0xc0) != 0x80;
             })));
}

OuteTTSVoiceFeatures audio_features(const std::vector<float> &samples,
                                    size_t begin, size_t end) {
  OuteTTSVoiceFeatures result;
  if (begin >= end || begin >= samples.size())
    return result;
  end = std::min(end, samples.size());
  double sum_sq = 0.0;
  for (size_t i = begin; i < end; ++i)
    sum_sq += static_cast<double>(samples[i]) * samples[i];
  const double rms = std::sqrt(sum_sq / static_cast<double>(end - begin));
  result.energy =
      static_cast<int>(std::clamp(std::lround(rms * 100.0), 0l, 100l));

  const size_t count = end - begin;
  std::vector<std::complex<float>> spectrum(count / 2u + 1u);
  const auto fft = engine::audio::get_real_fft_plan(count);
  fft->forward({count}, {static_cast<std::ptrdiff_t>(sizeof(float))},
               {static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))}, 0,
               samples.data() + static_cast<ptrdiff_t>(begin),
               spectrum.data());
  double magnitude_sum = 1.0e-10;
  double weighted_frequency = 0.0;
  for (size_t bin = 0; bin < spectrum.size(); ++bin) {
    const double magnitude = std::abs(spectrum[bin]);
    magnitude_sum += magnitude;
    weighted_frequency += magnitude * static_cast<double>(bin) *
                          24000.0 / static_cast<double>(count);
  }
  result.spectral_centroid = static_cast<int>(std::clamp(
      std::lround(weighted_frequency / magnitude_sum / 12000.0 * 100.0),
      0l, 100l));

  if (count >= 400 && sum_sq >= 1.0e-8) {
    constexpr int frame_length = 400;
    constexpr int hop_length = 160;
    constexpr int min_lag = 24000 / 600;
    constexpr int max_lag_exclusive = 24000 / 75;
    const size_t pad =
        (frame_length - (count % hop_length)) % hop_length;
    const size_t padded_count = count + pad;
    const size_t frames = 1u + (padded_count - frame_length) / hop_length;
    double pitch_sum = 0.0;
    std::vector<double> windowed(frame_length);
    std::vector<double> autocorrelation(frame_length);
    for (size_t frame = 0; frame < frames; ++frame) {
      const size_t offset = begin + frame * hop_length;
      for (int i = 0; i < frame_length; ++i) {
        const size_t source = offset + static_cast<size_t>(i);
        const double sample = source < end ? samples[source] : 0.0;
        const double window =
            0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * i /
                                 frame_length);
        windowed[static_cast<size_t>(i)] = sample * window;
      }
      for (int lag = 0; lag < frame_length; ++lag) {
        double value = 0.0;
        for (int i = 0; i + lag < frame_length; ++i)
          value += windowed[static_cast<size_t>(i)] *
                   windowed[static_cast<size_t>(i + lag)];
        autocorrelation[static_cast<size_t>(lag)] = value;
      }
      int best_lag = min_lag;
      for (int lag = min_lag + 1; lag < max_lag_exclusive; ++lag) {
        if (autocorrelation[static_cast<size_t>(lag)] >
            autocorrelation[static_cast<size_t>(best_lag)])
          best_lag = lag;
      }
      double frequency = 75.0;
      const double beta = autocorrelation[static_cast<size_t>(best_lag)];
      if (autocorrelation[0] > 1.0e-10 &&
          beta / autocorrelation[0] > 0.3) {
        const double alpha =
            autocorrelation[static_cast<size_t>(best_lag - 1)];
        const double gamma =
            autocorrelation[static_cast<size_t>(best_lag + 1)];
        const double delta = 0.5 * (alpha - gamma) /
                             (alpha - 2.0 * beta + gamma + 1.0e-8);
        frequency = std::clamp(24000.0 / (best_lag + delta), 75.0, 600.0);
      }
      pitch_sum += frequency;
    }
    const double average_pitch = pitch_sum / static_cast<double>(frames);
    result.pitch = static_cast<int>(std::clamp(
        std::lround((average_pitch - 75.0) / 525.0 * 100.0), 0l, 100l));
  }
  return result;
}

struct ReferenceAlignment {
  std::vector<runtime::WordTimestamp> words;
  int sample_rate = 16000;
};

OuteTTSVoiceProfile
make_voice_profile(OuteTTSDacDecoder::EncodedReference encoded,
                   std::string reference_text,
                   const ReferenceAlignment *alignment) {
  auto words = split_words(reference_text);
  if (words.empty())
    throw std::runtime_error("OuteTTS reference_text must not be empty");
  const size_t frame_count =
      std::min(encoded.codebook1.size(), encoded.codebook2.size());
  if (frame_count == 0)
    throw std::runtime_error(
        "OuteTTS DAC encoder produced no reference codec frames");
  if (alignment != nullptr) {
    words.clear();
    for (const auto &word : alignment->words)
      words.push_back(word.word);
  }
  if (words.size() > frame_count)
    words.resize(frame_count);
  std::vector<size_t> weights(words.size());
  size_t total_weight = 0;
  for (size_t i = 0; i < words.size(); ++i) {
    weights[i] = utf8_length(words[i]);
    total_weight += weights[i];
  }

  OuteTTSVoiceProfile profile;
  profile.text = reference_text;
  profile.global_features =
      audio_features(encoded.samples, 0, encoded.samples.size());
  debug::trace_log_scalar("outetts.reference.global.energy",
                          profile.global_features.energy);
  debug::trace_log_scalar("outetts.reference.global.spectral_centroid",
                          profile.global_features.spectral_centroid);
  debug::trace_log_scalar("outetts.reference.global.pitch",
                          profile.global_features.pitch);
  size_t start = 0;
  size_t cumulative_weight = 0;
  for (size_t word_index = 0; word_index < words.size(); ++word_index) {
    size_t feature_begin = 0;
    size_t feature_end = 0;
    size_t end = 0;
    if (alignment != nullptr && word_index < alignment->words.size()) {
      const auto &span = alignment->words[word_index].span;
      const double begin_seconds =
          static_cast<double>(span.start_sample) / alignment->sample_rate;
      const double end_seconds =
          static_cast<double>(span.end_sample) / alignment->sample_rate;
      if (word_index == 0) {
        const int64_t aligned_start =
            static_cast<int64_t>(begin_seconds * 75.0) - 20;
        start = static_cast<size_t>(std::clamp<int64_t>(
            aligned_start, 0, static_cast<int64_t>(frame_count - 1)));
      }
      int64_t aligned_end = static_cast<int64_t>(end_seconds * 75.0);
      if (word_index + 1 == words.size())
        aligned_end += 20;
      end = static_cast<size_t>(std::clamp<int64_t>(
          aligned_end, static_cast<int64_t>(start + 1),
          static_cast<int64_t>(frame_count)));
      feature_begin = static_cast<size_t>(std::clamp<int64_t>(
          static_cast<int64_t>(begin_seconds * 24000.0), 0,
          static_cast<int64_t>(encoded.samples.size())));
      feature_end = static_cast<size_t>(std::clamp<int64_t>(
          static_cast<int64_t>(end_seconds * 24000.0),
          static_cast<int64_t>(feature_begin),
          static_cast<int64_t>(encoded.samples.size())));
    } else {
      cumulative_weight += weights[word_index];
      end = word_index + 1 == words.size()
                ? frame_count
                : (frame_count * cumulative_weight + total_weight / 2) /
                      total_weight;
      feature_begin = start * 320u;
      feature_end = std::min(encoded.samples.size(), end * 320u);
    }
    end = std::max(end, std::min(frame_count, start + 1));
    OuteTTSVoiceWord word;
    word.text = words[word_index];
    word.duration =
        std::round(static_cast<double>(end - start) / 75.0 * 100.0) / 100.0;
    word.features = audio_features(encoded.samples, feature_begin, feature_end);
    word.codebook1.assign(
        encoded.codebook1.begin() + static_cast<ptrdiff_t>(start),
        encoded.codebook1.begin() + static_cast<ptrdiff_t>(end));
    word.codebook2.assign(
        encoded.codebook2.begin() + static_cast<ptrdiff_t>(start),
        encoded.codebook2.begin() + static_cast<ptrdiff_t>(end));
    const std::string trace_prefix =
        "outetts.reference.word." + std::to_string(word_index);
    debug::trace_log_scalar(trace_prefix + ".text", word.text);
    debug::trace_log_scalar(trace_prefix + ".duration", word.duration);
    debug::trace_log_scalar(trace_prefix + ".energy", word.features.energy);
    debug::trace_log_scalar(trace_prefix + ".spectral_centroid",
                            word.features.spectral_centroid);
    debug::trace_log_scalar(trace_prefix + ".pitch", word.features.pitch);
    profile.words.push_back(std::move(word));
    start = end;
  }
  return profile;
}

runtime::SessionOptions aligner_session_options(
    const runtime::SessionOptions &options) {
  runtime::SessionOptions out;
  out.backend = options.backend;
  for (const auto &[key, value] : options.options) {
    if (key.rfind("qwen3_forced_aligner.", 0) == 0)
      out.options.emplace(key, value);
  }
  return out;
}

std::shared_ptr<const engine::models::qwen3_asr::Qwen3ASRAssets>
resolve_aligner_assets(const runtime::SessionOptions &options,
                       const OuteTTSAssets &model_assets) {
  const auto model_path = runtime::find_option(
      options.options,
      {"outetts.aligner_model_path", "outetts.forced_aligner_model_path"});
  if (model_path.has_value()) {
    return engine::models::qwen3_asr::load_qwen3_asr_assets(
        std::filesystem::path(*model_path), "qwen3_forced_aligner");
  }
  return model_assets.embedded_aligner;
}

ReferenceAlignment align_reference(
    engine::models::qwen3_forced_aligner::Qwen3ForcedAlignerSession &session,
    const engine::models::qwen3_asr::Qwen3ASRAssets &aligner_assets,
    const runtime::AudioBuffer &audio,
    const std::string &text,
    const std::string &language) {
  runtime::TaskRequest request;
  request.audio_input = audio;
  request.text_input = runtime::Transcript{text, language};
  request.options["audio_chunk_mode"] = "none";
  session.prepare(runtime::build_preparation_request(request));
  auto result = session.run(request);
  if (result.word_timestamps.empty())
    throw std::runtime_error("OuteTTS reference aligner returned no words");
  return ReferenceAlignment{std::move(result.word_timestamps),
                            aligner_assets.config.sample_rate};
}

const runtime::AudioBuffer *
reference_audio(const runtime::TaskRequest &request) {
  if (request.voice.has_value() && request.voice->speaker.has_value() &&
      request.voice->speaker->audio.has_value()) {
    return &*request.voice->speaker->audio;
  }
  return request.audio_input.has_value() ? &*request.audio_input : nullptr;
}

} // namespace

OuteTTSSession::OuteTTSSession(runtime::TaskSpec task,
                               runtime::SessionOptions options,
                               std::shared_ptr<const OuteTTSAssets> assets)
    : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)),
      tokenizer_(assets_),
      dac_(assets_, execution_context(),
           runtime::parse_size_mb_option(options.options,
                                         {"outetts.dac_weight_context_mb"},
                                         1024ull * 1024ull * 1024ull),
           runtime::parse_size_mb_option(options.options,
                                         {"outetts.dac_graph_context_mb"},
                                         1536ull * 1024ull * 1024ull),
           assets::TensorStorageType::F32),
      mem_saver_(mem_saver_from_options(options)),
      reference_profile_cache_(reference_cache_slots(options)) {
  if (assets_ == nullptr)
    throw std::runtime_error("OuteTTS session requires assets");
  if ((task_.task != runtime::VoiceTaskKind::Tts &&
       task_.task != runtime::VoiceTaskKind::VoiceCloning) ||
      task_.mode != runtime::RunMode::Offline) {
    throw std::runtime_error(
        "OuteTTS supports offline TTS and voice cloning only");
  }
}

OuteTTSSession::~OuteTTSSession() = default;

OuteTTSLlamaRuntime &OuteTTSSession::llama(bool voice_cloning) {
  const auto ensure_start = Clock::now();
  const auto storage_type =
      voice_cloning ? clone_weight_type(options(), *assets_)
                    : requested_weight_type(options());
  const bool rebuilt =
      llama_ == nullptr || !llama_storage_type_.has_value() ||
      *llama_storage_type_ != storage_type;
  if (rebuilt) {
    // Keep only one language-model runtime resident. CUDA cloning may require
    // an F32 runtime for quantized source weights, so switching routes replaces
    // the previous runtime instead of retaining duplicate weights and graphs.
    llama_.reset();
    llama_ = std::make_unique<OuteTTSLlamaRuntime>(
        assets_, options().backend.type, options().backend.device,
        std::max(1, options().backend.threads),
        runtime::parse_size_mb_option(options().options,
                                      {"outetts.llama_weight_context_mb"},
                                      4096ull * 1024ull * 1024ull),
        runtime::parse_size_mb_option(options().options,
                                      {"outetts.constant_context_mb"},
                                      256ull * 1024ull * 1024ull),
        storage_type);
    llama_storage_type_ = storage_type;
  }
  debug::trace_log_scalar("outetts.llama.runtime_rebuilt", rebuilt);
  debug::trace_log_scalar("outetts.llama.runtime_reused", !rebuilt);
  debug::trace_log_scalar("outetts.llama.clone_route", voice_cloning);
  debug::timing_log_scalar(
      "outetts.llama.ensure_runtime_ms",
      rebuilt ? debug::elapsed_ms(ensure_start) : 0.0);
  return *llama_;
}

bool OuteTTSSession::ReferenceProfileCacheKeyEqual::operator()(
    const ReferenceProfileCacheKey &lhs,
    const ReferenceProfileCacheKey &rhs) const {
  return lhs.audio_hash == rhs.audio_hash &&
         lhs.sample_rate == rhs.sample_rate &&
         lhs.channels == rhs.channels &&
         lhs.sample_count == rhs.sample_count && lhs.text == rhs.text &&
         lhs.language == rhs.language;
}

OuteTTSVoiceProfile OuteTTSSession::prepare_voice_profile(
    const runtime::AudioBuffer &audio, const std::string &reference_text,
    const std::string &language, bool &cache_hit) {
  const auto total_start = Clock::now();
  ReferenceProfileCacheKey key{
      reference_audio_hash(audio), audio.sample_rate, audio.channels,
      audio.samples.size(), reference_text, language};
  if (const auto *cached = reference_profile_cache_.find(key)) {
    cache_hit = true;
    debug::trace_log_scalar("outetts.reference_cache.hit", true);
    debug::trace_log_scalar(
        "outetts.reference_cache.slots",
        static_cast<int64_t>(reference_profile_cache_.capacity()));
    debug::trace_log_scalar(
        "outetts.reference_cache.entries",
        static_cast<int64_t>(reference_profile_cache_.size()));
    debug::trace_log_scalar("outetts.reference_cache.evicted", false);
    debug::timing_log_scalar("outetts.reference.total_ms",
                             debug::elapsed_ms(total_start));
    return *cached;
  }

  cache_hit = false;
  const bool will_evict =
      reference_profile_cache_.capacity() > 0 &&
      reference_profile_cache_.size() >= reference_profile_cache_.capacity();
  const auto ensure_aligner_start = Clock::now();
  const bool aligner_built = aligner_session_ == nullptr;
  if (aligner_built) {
    if (aligner_assets_ == nullptr) {
      aligner_assets_ = resolve_aligner_assets(options(), *assets_);
    }
    if (aligner_assets_ == nullptr) {
      throw std::runtime_error(
          "OuteTTS voice cloning requires a GGUF with an embedded Qwen3 "
          "Forced Aligner or --session-option "
          "outetts.aligner_model_path=<path>");
    }
    aligner_session_ = std::make_unique<
        engine::models::qwen3_forced_aligner::Qwen3ForcedAlignerSession>(
        runtime::TaskSpec{runtime::VoiceTaskKind::Alignment,
                          runtime::RunMode::Offline},
        aligner_session_options(options()), aligner_assets_);
  }
  debug::trace_log_scalar("outetts.aligner.runtime_rebuilt", aligner_built);
  debug::trace_log_scalar("outetts.aligner.runtime_reused", !aligner_built);
  debug::timing_log_scalar(
      "outetts.aligner.ensure_runtime_ms",
      aligner_built ? debug::elapsed_ms(ensure_aligner_start) : 0.0);

  const auto align_start = Clock::now();
  const auto alignment = align_reference(*aligner_session_, *aligner_assets_,
                                         audio, reference_text, language);
  debug::timing_log_scalar("outetts.reference.align_ms",
                           debug::elapsed_ms(align_start));

  const auto encode_start = Clock::now();
  auto encoded = dac_.encode_reference(audio);
  debug::timing_log_scalar("outetts.reference.dac_encode_ms",
                           debug::elapsed_ms(encode_start));

  const auto profile_start = Clock::now();
  auto profile = make_voice_profile(std::move(encoded), reference_text,
                                    &alignment);
  debug::timing_log_scalar("outetts.reference.profile_ms",
                           debug::elapsed_ms(profile_start));
  reference_profile_cache_.put(std::move(key), profile);

  debug::trace_log_scalar("outetts.reference_cache.hit", false);
  debug::trace_log_scalar(
      "outetts.reference_cache.slots",
      static_cast<int64_t>(reference_profile_cache_.capacity()));
  debug::trace_log_scalar(
      "outetts.reference_cache.entries",
      static_cast<int64_t>(reference_profile_cache_.size()));
  debug::trace_log_scalar("outetts.reference_cache.evicted", will_evict);
  if (mem_saver_) {
    const auto release_start = Clock::now();
    aligner_session_.reset();
    debug::trace_log_scalar("outetts.aligner.runtime_released", true);
    debug::timing_log_scalar("outetts.aligner.release_ms",
                             debug::elapsed_ms(release_start));
  } else {
    debug::trace_log_scalar("outetts.aligner.runtime_released", false);
    debug::timing_log_scalar("outetts.aligner.release_ms", 0.0);
  }
  debug::timing_log_scalar("outetts.reference.total_ms",
                           debug::elapsed_ms(total_start));
  return profile;
}

std::string OuteTTSSession::family() const { return "outetts"; }
runtime::VoiceTaskKind OuteTTSSession::task_kind() const { return task_.task; }
runtime::RunMode OuteTTSSession::run_mode() const { return task_.mode; }
void OuteTTSSession::prepare(
    const runtime::SessionPreparationRequest &request) {
  voice_profile_.reset();
  if (request.voice.has_value() && request.voice->speaker.has_value() &&
      request.voice->speaker->audio.has_value()) {
    const auto reference_text =
        runtime::find_option(request.options, {"reference_text"}).value_or("");
    if (reference_text.empty()) {
      throw std::runtime_error(
          "OuteTTS voice cloning requires --reference-text");
    }
    const auto &audio = *request.voice->speaker->audio;
    const auto language = runtime::find_option(
                              request.options, {"reference_language"})
                              .value_or(request.text.has_value() &&
                                                !request.text->language.empty()
                                            ? request.text->language
                                            : "en");
    bool cache_hit = false;
    voice_profile_ =
        prepare_voice_profile(audio, reference_text, language, cache_hit);
    debug::trace_log_scalar("outetts.prepare.reference_cache_hit", cache_hit);
  }
  mark_prepared();
}

runtime::TaskResult OuteTTSSession::run(const runtime::TaskRequest &request) {
  const auto wall_start = Clock::now();
  require_prepared("OuteTTS run");
  if (!request.text_input.has_value() || request.text_input->text.empty()) {
    throw std::runtime_error("OuteTTS requires text input");
  }
  const auto *voice_audio = reference_audio(request);
  std::optional<OuteTTSVoiceProfile> request_profile;
  bool reference_cache_hit = false;
  if (voice_audio != nullptr) {
    const auto reference_text =
        runtime::find_option(request.options, {"reference_text"}).value_or("");
    if (reference_text.empty())
      throw std::runtime_error(
          "OuteTTS voice cloning requires --reference-text");
    const auto language = runtime::find_option(
                              request.options, {"reference_language"})
                              .value_or(request.text_input.has_value() &&
                                                !request.text_input->language.empty()
                                            ? request.text_input->language
                                            : "en");
    request_profile = prepare_voice_profile(
        *voice_audio, reference_text, language, reference_cache_hit);
  }
  const OuteTTSVoiceProfile *profile =
      request_profile.has_value()
          ? &*request_profile
          : (voice_profile_.has_value() ? &*voice_profile_ : nullptr);
  if (task_.task == runtime::VoiceTaskKind::VoiceCloning &&
      profile == nullptr) {
    throw std::runtime_error(
        "OuteTTS voice cloning requires --voice-ref and --reference-text");
  }

  const int64_t text_chunk_size =
      engine::text::parse_text_chunk_size_override(request.options)
          .value_or(kDefaultTextChunkSize);
  const auto text_chunk_mode =
      engine::text::parse_text_chunk_mode_override(request.options)
          .value_or(engine::text::TextChunkMode::Default);
  const auto explicit_max_tokens =
      runtime::parse_i64_option(request.options, {"max_tokens"});
  if (explicit_max_tokens.has_value() && *explicit_max_tokens <= 0) {
    throw std::runtime_error("OuteTTS max_tokens must be positive");
  }
  const int64_t chunk_token_budget = std::min<int64_t>(
      explicit_max_tokens.value_or(kAutomaticChunkTokenBudget),
      kAutomaticChunkTokenBudget);
  runtime::TaskRequest chunk_source = request;
  // Reference conditioning is already represented by profile. Avoid copying
  // the full reference waveform into every long-form chunk and retry.
  if (profile != nullptr) {
    chunk_source.voice.reset();
    chunk_source.audio_input.reset();
  }
  const auto initial_chunk_requests = chunk_text_request_to_token_budget(
      chunk_source, text_chunk_size, text_chunk_mode, chunk_token_budget);
  if (initial_chunk_requests.empty()) {
    throw std::runtime_error("OuteTTS text chunking produced no requests");
  }
  std::deque<runtime::TaskRequest> pending_chunks(
      initial_chunk_requests.begin(), initial_chunk_requests.end());
  debug::trace_log_scalar("outetts.text_chunk_size", text_chunk_size);
  debug::trace_log_scalar("outetts.text_chunk_mode",
                          engine::text::text_chunk_mode_name(text_chunk_mode));
  debug::trace_log_scalar("outetts.text_chunk_token_budget",
                          chunk_token_budget);
  debug::trace_log_scalar(
      "outetts.text_chunk_count",
      static_cast<int64_t>(initial_chunk_requests.size()));
  debug::trace_log_scalar("outetts.reference.cache_hit",
                          reference_cache_hit);

  const bool quantized_cloning =
      profile != nullptr && has_quantized_clone_weights(options(), *assets_);

  runtime::AudioBuffer merged_audio;
  double prompt_ms = 0.0;
  double generate_ms = 0.0;
  double decode_ms = 0.0;
  double release_ms = 0.0;
  int64_t generated_tokens = 0;
  int64_t retry_generated_tokens = 0;
  int64_t released_cache_capacity = 0;
  int64_t retry_count = 0;
  size_t chunk_index = 0;
  while (!pending_chunks.empty()) {
    auto chunk_request = std::move(pending_chunks.front());
    pending_chunks.pop_front();
    const auto prompt_start = Clock::now();
    const auto prompt =
        profile != nullptr
            ? tokenizer_.build_clone_prompt(chunk_request.text_input->text,
                                            *profile)
            : tokenizer_.build_prompt(chunk_request.text_input->text);
    prompt_ms += debug::elapsed_ms(prompt_start);

    const auto text_budget = estimate_text_generation_budget(
        chunk_request.text_input->text);
    const int64_t automatic_max_new_tokens = generation_budget_with_headroom(
        text_budget.recommended_max_new_tokens);
    const int64_t expected_generation_tokens = std::min(
        automatic_max_new_tokens,
        explicit_max_tokens.value_or(automatic_max_new_tokens));
    const int64_t available_context =
        assets_->generation.max_length - static_cast<int64_t>(prompt.size());
    if (available_context <= 0) {
      throw std::runtime_error(
          "OuteTTS chunk " + std::to_string(chunk_index + 1) +
          " prompt exhausts the generation context; use a shorter cloning "
          "reference");
    }
    if (expected_generation_tokens > available_context) {
      const int64_t codepoints = static_cast<int64_t>(
          engine::text::utf8_codepoint_count(chunk_request.text_input->text,
                                             "OuteTTS context retry"));
      auto smaller = runtime::chunk_text_request(
          chunk_request, std::max<int64_t>(1, codepoints / 2),
          text_chunk_mode);
      if (codepoints <= 1 || smaller.size() <= 1) {
        throw std::runtime_error(
            "OuteTTS chunk does not fit the generation context; reduce "
            "text_chunk_size or use a shorter cloning reference");
      }
      debug::trace_log_scalar(
          "outetts.retry." + std::to_string(retry_count) + ".reason",
          std::string_view("context_budget"));
      debug::trace_log_scalar(
          "outetts.retry." + std::to_string(retry_count) + ".split_count",
          static_cast<int64_t>(smaller.size()));
      ++retry_count;
      for (auto it = smaller.rbegin(); it != smaller.rend(); ++it)
        pending_chunks.push_front(std::move(*it));
      continue;
    }

    auto generate_options =
        generation_options(chunk_request, assets_->generation,
                           profile != nullptr, quantized_cloning,
                           automatic_max_new_tokens);
    generate_options.max_new_tokens =
        std::min(generate_options.max_new_tokens, available_context);
    const auto generate_start = Clock::now();
    const auto generation = llama(profile != nullptr).generate(
        prompt, generate_options, tokenizer_.eos_id(),
        tokenizer_.audio_end_id());
    generate_ms += debug::elapsed_ms(generate_start);
    const auto &generated = generation.tokens;

    if (generation.stop_reason == OuteTTSStopReason::MaxTokens ||
        generation.stop_reason == OuteTTSStopReason::ContextLimit) {
      retry_generated_tokens += static_cast<int64_t>(generated.size());
      const int64_t codepoints = static_cast<int64_t>(
          engine::text::utf8_codepoint_count(chunk_request.text_input->text,
                                             "OuteTTS generation retry"));
      auto smaller = runtime::chunk_text_request(
          chunk_request, std::max<int64_t>(1, codepoints / 2),
          text_chunk_mode);
      if (codepoints <= 1 || smaller.size() <= 1) {
        throw std::runtime_error(
            "OuteTTS reached " +
            std::string(outetts_stop_reason_name(generation.stop_reason)) +
            " before an audio end token and cannot split the remaining text "
            "further; increase max_tokens");
      }
      const std::string retry_prefix =
          "outetts.retry." + std::to_string(retry_count);
      debug::trace_log_scalar(retry_prefix + ".reason",
                              outetts_stop_reason_name(generation.stop_reason));
      debug::trace_log_scalar(retry_prefix + ".generated_tokens",
                              static_cast<int64_t>(generated.size()));
      debug::trace_log_scalar(retry_prefix + ".text_codepoints",
                              text_budget.non_whitespace_codepoints);
      debug::trace_log_scalar(retry_prefix + ".split_count",
                              static_cast<int64_t>(smaller.size()));
      ++retry_count;
      for (auto it = smaller.rbegin(); it != smaller.rend(); ++it)
        pending_chunks.push_front(std::move(*it));
      continue;
    }
    generated_tokens += static_cast<int64_t>(generated.size());

    debug::trace_log_scalar(
        "outetts.chunk." + std::to_string(chunk_index) + ".text_codepoints",
        text_budget.non_whitespace_codepoints);
    debug::trace_log_scalar(
        "outetts.chunk." + std::to_string(chunk_index) + ".text_words",
        text_budget.words);
    debug::trace_log_scalar(
        "outetts.chunk." + std::to_string(chunk_index) +
            ".recommended_max_new_tokens",
        text_budget.recommended_max_new_tokens);
    debug::trace_log_scalar(
        "outetts.chunk." + std::to_string(chunk_index) + ".max_new_tokens",
        generate_options.max_new_tokens);
    debug::trace_log_scalar(
        "outetts.chunk." + std::to_string(chunk_index) + ".stop_reason",
        outetts_stop_reason_name(generation.stop_reason));

    std::vector<int32_t> c1;
    std::vector<int32_t> c2;
    for (const int32_t token : generated)
      tokenizer_.append_audio_code(token, c1, c2);
    const size_t pairs = std::min(c1.size(), c2.size());
    c1.resize(pairs);
    c2.resize(pairs);
    if (pairs == 0) {
      std::string detail;
      for (size_t i = 0; i < std::min<size_t>(generated.size(), 12); ++i) {
        detail += (i == 0 ? "" : ",") + std::to_string(generated[i]);
      }
      throw std::runtime_error(
          "OuteTTS generated no complete DAC code pairs (tokens=" + detail +
          ")");
    }

    const auto decode_start = Clock::now();
    runtime::append_audio_buffer(merged_audio, dac_.decode(c1, c2));
    decode_ms += debug::elapsed_ms(decode_start);
    debug::trace_log_scalar(
        "outetts.chunk." + std::to_string(chunk_index) + ".prompt_tokens",
        static_cast<int64_t>(prompt.size()));
    debug::trace_log_scalar(
        "outetts.chunk." + std::to_string(chunk_index) + ".generated_tokens",
        static_cast<int64_t>(generated.size()));
    debug::trace_log_scalar(
        "outetts.chunk." + std::to_string(chunk_index) + ".codec_frames",
        static_cast<int64_t>(pairs));

    if (mem_saver_) {
      const auto release_start = Clock::now();
      released_cache_capacity += llama_->release_cached_step_graph();
      release_ms += debug::elapsed_ms(release_start);
    }
    ++chunk_index;
  }

  runtime::TaskResult result;
  result.audio_output = std::move(merged_audio);
  debug::trace_log_scalar("outetts.mem_saver", mem_saver_);
  debug::trace_log_scalar("outetts.generated_tokens", generated_tokens);
  debug::trace_log_scalar("outetts.retry_generated_tokens",
                          retry_generated_tokens);
  debug::trace_log_scalar("outetts.text_chunk_count_final",
                          static_cast<int64_t>(chunk_index));
  debug::trace_log_scalar("outetts.retry_count", retry_count);
  debug::trace_log_scalar("outetts.llama.step.released_cache_capacity",
                          released_cache_capacity);
  debug::timing_log_scalar("outetts.prompt_ms", prompt_ms);
  debug::timing_log_scalar("outetts.generate_ms", generate_ms);
  debug::timing_log_scalar("outetts.dac_decode_ms", decode_ms);
  debug::timing_log_scalar("outetts.llama.step.release_ms", release_ms);
  debug::timing_log_scalar("session.wall_ms",
                           debug::elapsed_ms(wall_start));
  return result;
}

} // namespace engine::models::outetts
