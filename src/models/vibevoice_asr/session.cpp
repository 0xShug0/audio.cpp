#include "engine/models/vibevoice_asr/session.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/models/silero_vad/session.h"

#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::vibevoice_asr {
namespace {

using Clock = std::chrono::steady_clock;
constexpr size_t kDefaultTokenizerWeightContextBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kDefaultConnectorWeightContextBytes = 128ull * 1024ull * 1024ull;
constexpr size_t kDefaultDecoderWeightContextBytes = 4096ull * 1024ull * 1024ull;
constexpr double kDefaultAudioChunkSeconds = 20.0 * 60.0;

std::shared_ptr<const VibeVoiceAssets> require_assets(std::shared_ptr<const VibeVoiceAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("VibeVoice-ASR session requires assets");
    }
    return assets;
}

assets::TensorStorageType option_weight_type(
    const runtime::SessionOptions & options,
    const char * key,
    assets::TensorStorageType fallback) {
    const auto it = options.options.find(key);
    if (it == options.options.end()) {
        return fallback;
    }
    return assets::parse_tensor_storage_type(it->second);
}

void validate_weight_storage(assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == assets::TensorStorageType::Native ||
        storage_type == assets::TensorStorageType::F32 ||
        storage_type == assets::TensorStorageType::F16 ||
        storage_type == assets::TensorStorageType::BF16 ||
        storage_type == assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " supports only native, f32, f16, bf16, and q8_0");
}

std::filesystem::path default_vad_model_path() {
    return std::filesystem::path("assets") / "framework" / "models" / "silero_vad";
}

int64_t audio_frame_count(const runtime::AudioBuffer & audio) {
    if (audio.channels <= 0) {
        throw std::runtime_error("VibeVoice-ASR audio chunking requires positive audio channels");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("VibeVoice-ASR audio samples must be divisible by channel count");
    }
    return static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels));
}

size_t common_prefix_size(const std::string & lhs, const std::string & rhs) {
    const size_t limit = std::min(lhs.size(), rhs.size());
    size_t size = 0;
    while (size < limit && lhs[size] == rhs[size]) {
        ++size;
    }
    return size;
}

void emit_transcript_delta(
    const runtime::StreamEventCallback & sink,
    const runtime::Transcript & transcript,
    std::string & emitted_text) {
    if (!sink || transcript.text.empty()) {
        return;
    }
    const size_t prefix_size = common_prefix_size(emitted_text, transcript.text);
    if (prefix_size == transcript.text.size()) {
        emitted_text = transcript.text;
        return;
    }
    runtime::StreamEvent event;
    event.partial_text = runtime::Transcript{transcript.text.substr(prefix_size), transcript.language};
    sink(event);
    emitted_text = transcript.text;
}

std::string append_streaming_transcript(
    runtime::TaskResult & total,
    const runtime::Transcript & chunk) {
    if (!total.text_output.has_value()) {
        total.text_output = runtime::Transcript{"", chunk.language};
    }
    if (!chunk.language.empty()) {
        total.text_output->language = chunk.language;
    }
    if (chunk.text.empty()) {
        return "";
    }
    std::string delta;
    if (!total.text_output->text.empty()) {
        total.text_output->text.push_back(' ');
        delta.push_back(' ');
    }
    total.text_output->text += chunk.text;
    delta += chunk.text;
    return delta;
}

void append_offset_speech_metadata(
    runtime::TaskResult & total,
    const runtime::TaskResult & chunk,
    const runtime::TimeSpan & source_span,
    const runtime::TimeSpan & keep_span) {
    for (const auto & segment : chunk.speech_segments) {
        auto shifted = segment;
        shifted.span.start_sample += source_span.start_sample;
        shifted.span.end_sample += source_span.start_sample;
        if (shifted.span.end_sample <= keep_span.start_sample ||
            shifted.span.start_sample >= keep_span.end_sample) {
            continue;
        }
        shifted.span.start_sample = std::max<int64_t>(shifted.span.start_sample, keep_span.start_sample);
        shifted.span.end_sample = std::min<int64_t>(shifted.span.end_sample, keep_span.end_sample);
        total.speech_segments.push_back(std::move(shifted));
    }
    for (const auto & turn : chunk.speaker_turns) {
        auto shifted = turn;
        shifted.span.start_sample += source_span.start_sample;
        shifted.span.end_sample += source_span.start_sample;
        if (shifted.span.end_sample <= keep_span.start_sample ||
            shifted.span.start_sample >= keep_span.end_sample) {
            continue;
        }
        shifted.span.start_sample = std::max<int64_t>(shifted.span.start_sample, keep_span.start_sample);
        shifted.span.end_sample = std::min<int64_t>(shifted.span.end_sample, keep_span.end_sample);
        total.speaker_turns.push_back(std::move(shifted));
    }
}

std::string decode_json_string_fragment(const std::string & value, size_t start, size_t * end) {
    std::string decoded;
    bool escaped = false;
    size_t i = start;
    for (; i < value.size(); ++i) {
        const char ch = value[i];
        if (escaped) {
            if (ch == 'n') {
                decoded.push_back('\n');
            } else if (ch == 'r') {
                decoded.push_back('\r');
            } else if (ch == 't') {
                decoded.push_back('\t');
            } else {
                decoded.push_back(ch);
            }
            escaped = false;
            continue;
        }
        if (ch == '\\') {
            escaped = true;
            continue;
        }
        if (ch == '"') {
            ++i;
            break;
        }
        decoded.push_back(ch);
    }
    *end = i;
    return decoded;
}

void append_streaming_json_content_for_key(
    const std::string & raw_text,
    const std::string & key,
    std::string & text) {
    const std::string quoted_key = "\"" + key + "\"";
    size_t search_pos = 0;
    while (true) {
        const size_t key_pos = raw_text.find(quoted_key, search_pos);
        if (key_pos == std::string::npos) {
            return;
        }
        size_t pos = key_pos + quoted_key.size();
        while (pos < raw_text.size() && std::isspace(static_cast<unsigned char>(raw_text[pos])) != 0) {
            ++pos;
        }
        if (pos >= raw_text.size() || raw_text[pos] != ':') {
            search_pos = key_pos + quoted_key.size();
            continue;
        }
        ++pos;
        while (pos < raw_text.size() && std::isspace(static_cast<unsigned char>(raw_text[pos])) != 0) {
            ++pos;
        }
        if (pos >= raw_text.size() || raw_text[pos] != '"') {
            search_pos = key_pos + quoted_key.size();
            continue;
        }
        size_t end = pos + 1;
        std::string fragment = decode_json_string_fragment(raw_text, pos + 1, &end);
        if (!fragment.empty()) {
            if (!text.empty()) {
                text.push_back(' ');
            }
            text += fragment;
        }
        search_pos = end;
    }
}

std::string streaming_transcript_from_vibevoice_raw(const std::string & raw_text) {
    std::string text;
    append_streaming_json_content_for_key(raw_text, "Content", text);
    append_streaming_json_content_for_key(raw_text, "text", text);
    return text;
}

float bf16_compare_value(float value) {
    return ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
}

int32_t argmax_token(const std::vector<float> & logits, bool compare_bf16) {
    if (logits.empty()) {
        throw std::runtime_error("VibeVoice-ASR decoder returned empty logits");
    }
    int32_t best = 0;
    float best_value = compare_bf16 ? bf16_compare_value(logits[0]) : logits[0];
    for (size_t i = 1; i < logits.size(); ++i) {
        const float value = compare_bf16 ? bf16_compare_value(logits[i]) : logits[i];
        if (value > best_value) {
            best = static_cast<int32_t>(i);
            best_value = value;
        }
    }
    return best;
}

void apply_repetition_penalty(
    std::vector<float> & logits,
    const std::vector<int32_t> & prompt_ids,
    const std::vector<int32_t> & generated,
    float penalty,
    std::vector<uint8_t> & seen) {
    if (penalty == 1.0F) {
        return;
    }
    if (!(penalty > 0.0F) || !std::isfinite(penalty)) {
        throw std::runtime_error("VibeVoice-ASR repetition_penalty must be finite and positive");
    }
    seen.assign(logits.size(), 0);
    auto visit = [&](int32_t token) {
        if (token < 0 || static_cast<size_t>(token) >= logits.size()) {
            return;
        }
        const size_t index = static_cast<size_t>(token);
        if (seen[index] != 0) {
            return;
        }
        seen[index] = 1;
        float & value = logits[index];
        value = value < 0.0F ? value * penalty : value / penalty;
    };
    for (const int32_t token : prompt_ids) {
        visit(token);
    }
    for (const int32_t token : generated) {
        visit(token);
    }
}

struct TokenLogProb {
    int32_t token = 0;
    double log_prob = -std::numeric_limits<double>::infinity();
};

std::vector<TokenLogProb> top_log_probs(
    const std::vector<float> & logits,
    int64_t count,
    bool compare_bf16) {
    if (logits.empty()) {
        throw std::runtime_error("VibeVoice-ASR decoder returned empty logits");
    }
    if (count <= 0) {
        throw std::runtime_error("VibeVoice-ASR top log-prob count must be positive");
    }
    const size_t keep = std::min<size_t>(logits.size(), static_cast<size_t>(count));
    double max_value = -std::numeric_limits<double>::infinity();
    for (float logit : logits) {
        const double value = compare_bf16 ? static_cast<double>(bf16_compare_value(logit)) : static_cast<double>(logit);
        if (value > max_value) {
            max_value = value;
        }
    }
    if (!std::isfinite(max_value)) {
        throw std::runtime_error("VibeVoice-ASR logits do not contain a finite value");
    }
    double sum = 0.0;
    std::vector<TokenLogProb> scores;
    scores.reserve(keep);
    auto better = [](const TokenLogProb & lhs, const TokenLogProb & rhs) {
        if (lhs.log_prob == rhs.log_prob) {
            return lhs.token < rhs.token;
        }
        return lhs.log_prob > rhs.log_prob;
    };
    for (size_t i = 0; i < logits.size(); ++i) {
        const float logit = logits[i];
        const double value = compare_bf16 ? static_cast<double>(bf16_compare_value(logit)) : static_cast<double>(logit);
        sum += std::exp(value - max_value);
        TokenLogProb candidate{static_cast<int32_t>(i), value};
        if (scores.size() < keep) {
            scores.push_back(candidate);
            if (scores.size() == keep) {
                std::sort(scores.begin(), scores.end(), better);
            }
        } else if (better(candidate, scores.back())) {
            scores.back() = candidate;
            for (size_t pos = scores.size() - 1; pos > 0 && better(scores[pos], scores[pos - 1]); --pos) {
                std::swap(scores[pos], scores[pos - 1]);
            }
        }
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        throw std::runtime_error("VibeVoice-ASR log-softmax normalization failed");
    }
    const double log_sum = max_value + std::log(sum);
    for (auto & score : scores) {
        score.log_prob -= log_sum;
    }
    std::sort(scores.begin(), scores.end(), better);
    return scores;
}

int32_t sample_token(
    const std::vector<float> & logits,
    float temperature,
    float top_p,
    int64_t top_k,
    uint64_t seed,
    uint64_t call_index,
    const sampling::TorchCudaSamplingPolicy & sampling_policy,
    std::mt19937 & rng,
    bool compare_bf16) {
    if (!(temperature > 0.0F) || !std::isfinite(temperature)) {
        throw std::runtime_error("VibeVoice-ASR sampling temperature must be finite and positive");
    }
    if (!(top_p > 0.0F && top_p <= 1.0F) || !std::isfinite(top_p)) {
        throw std::runtime_error("VibeVoice-ASR top_p must be finite and in (0, 1]");
    }
    if (top_k < 0) {
        throw std::runtime_error("VibeVoice-ASR top_k must be non-negative");
    }
    std::vector<float> scores(logits.size(), -std::numeric_limits<float>::infinity());
    for (size_t i = 0; i < logits.size(); ++i) {
        const float value = compare_bf16 ? bf16_compare_value(logits[i]) : logits[i];
        if (std::isfinite(value)) {
            scores[i] = value / temperature;
        }
    }
    float max_score = -std::numeric_limits<float>::infinity();
    for (const float score : scores) {
        if (std::isfinite(score)) {
            max_score = std::max(max_score, score);
        }
    }
    if (!std::isfinite(max_score)) {
        throw std::runtime_error("VibeVoice-ASR sampling found no finite logits");
    }
    if (top_k > 0 && static_cast<size_t>(top_k) < scores.size()) {
        std::vector<size_t> indices(scores.size());
        std::iota(indices.begin(), indices.end(), 0);
        auto better_index = [&](size_t lhs, size_t rhs) {
            if (scores[lhs] == scores[rhs]) {
                return lhs < rhs;
            }
            return scores[lhs] > scores[rhs];
        };
        const size_t keep = static_cast<size_t>(top_k);
        std::nth_element(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(keep), indices.end(), better_index);
        std::vector<uint8_t> keep_mask(scores.size(), 0);
        for (size_t i = 0; i < keep; ++i) {
            keep_mask[indices[i]] = 1;
        }
        for (size_t i = 0; i < scores.size(); ++i) {
            if (keep_mask[i] == 0) {
                scores[i] = -std::numeric_limits<float>::infinity();
            }
        }
        max_score = -std::numeric_limits<float>::infinity();
        for (const float score : scores) {
            if (std::isfinite(score)) {
                max_score = std::max(max_score, score);
            }
        }
        if (!std::isfinite(max_score)) {
            throw std::runtime_error("VibeVoice-ASR top-k sampling kept no finite logits");
        }
    }
    double total = 0.0;
    for (const float score : scores) {
        if (std::isfinite(score)) {
            total += std::exp(static_cast<double>(score - max_score));
        }
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        throw std::runtime_error("VibeVoice-ASR sampling probability mass is invalid");
    }
    if (top_p < 1.0F) {
        struct SortedScore {
            size_t index = 0;
            float score = -std::numeric_limits<float>::infinity();
            double weight = 0.0;
        };
        std::vector<SortedScore> sorted;
        sorted.reserve(scores.size());
        for (size_t i = 0; i < scores.size(); ++i) {
            if (!std::isfinite(scores[i])) {
                continue;
            }
            sorted.push_back({i, scores[i], std::exp(static_cast<double>(scores[i] - max_score))});
        }
        std::sort(sorted.begin(), sorted.end(), [](const SortedScore & lhs, const SortedScore & rhs) {
            if (lhs.score == rhs.score) {
                return lhs.index < rhs.index;
            }
            return lhs.score < rhs.score;
        });
        double cumulative = 0.0;
        const double remove_mass = 1.0 - static_cast<double>(top_p);
        const size_t keep_from = sorted.empty() ? 0 : sorted.size() - 1;
        for (size_t i = 0; i < sorted.size(); ++i) {
            cumulative += sorted[i].weight / total;
            if (i < keep_from && cumulative <= remove_mass) {
                scores[sorted[i].index] = -std::numeric_limits<float>::infinity();
            }
        }
        max_score = -std::numeric_limits<float>::infinity();
        for (const float score : scores) {
            if (std::isfinite(score)) {
                max_score = std::max(max_score, score);
            }
        }
        if (!std::isfinite(max_score)) {
            throw std::runtime_error("VibeVoice-ASR top-p sampling kept no finite logits");
        }
    }
    double kept_total = 0.0;
    for (const float score : scores) {
        if (std::isfinite(score)) {
            kept_total += std::exp(static_cast<double>(score - max_score));
        }
    }
    if (!(kept_total > 0.0) || !std::isfinite(kept_total)) {
        throw std::runtime_error("VibeVoice-ASR sampling kept probability mass is invalid");
    }
    if (sampling_policy.cuda_fast_path) {
        double best_rank = -std::numeric_limits<double>::infinity();
        int32_t best_token = -1;
        for (size_t index = 0; index < scores.size(); ++index) {
            if (!std::isfinite(scores[index])) {
                continue;
            }
            const double probability = std::exp(static_cast<double>(scores[index] - max_score)) / kept_total;
            const float exponential = sampling::torch_cuda_tensor_iterator_exponential_element(
                seed,
                static_cast<uint64_t>(scores.size()),
                static_cast<uint64_t>(index),
                call_index,
                sampling_policy.multiprocessor_count,
                sampling_policy.max_threads_per_multiprocessor);
            const double rank = probability / static_cast<double>(exponential);
            if (rank > best_rank) {
                best_rank = rank;
                best_token = static_cast<int32_t>(index);
            }
        }
        if (best_token < 0) {
            throw std::runtime_error("VibeVoice-ASR CUDA sampler failed to select a token");
        }
        return best_token;
    }
    std::vector<double> weights;
    weights.reserve(scores.size());
    std::vector<int32_t> tokens;
    tokens.reserve(scores.size());
    for (size_t index = 0; index < scores.size(); ++index) {
        if (!std::isfinite(scores[index])) {
            continue;
        }
        tokens.push_back(static_cast<int32_t>(index));
        weights.push_back(std::exp(static_cast<double>(scores[index] - max_score)));
    }
    std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
    return tokens[distribution(rng)];
}

}  // namespace

VibeVoiceASRSession::VibeVoiceASRSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const VibeVoiceAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      tokenizer_weight_context_bytes_(runtime::parse_size_mb_option(options.options, {"vibevoice_asr.tokenizer_weight_context_mb"}, kDefaultTokenizerWeightContextBytes)),
      connector_weight_context_bytes_(runtime::parse_size_mb_option(options.options, {"vibevoice_asr.connector_weight_context_mb"}, kDefaultConnectorWeightContextBytes)),
      decoder_weight_context_bytes_(runtime::parse_size_mb_option(options.options, {"vibevoice_asr.decoder_weight_context_mb"}, kDefaultDecoderWeightContextBytes)),
      tokenizer_weight_storage_type_(option_weight_type(
          options,
          "vibevoice_asr.tokenizer_weight_type",
          option_weight_type(options, "vibevoice_asr.weight_type", assets::TensorStorageType::Native))),
      connector_weight_storage_type_(option_weight_type(
          options,
          "vibevoice_asr.connector_weight_type",
          option_weight_type(options, "vibevoice_asr.weight_type", assets::TensorStorageType::Native))),
      decoder_weight_storage_type_(option_weight_type(
          options,
          "vibevoice_asr.decoder_weight_type",
          option_weight_type(options, "vibevoice_asr.weight_type", assets::TensorStorageType::Native))),
      greedy_compare_bf16_(
          decoder_weight_storage_type_ == assets::TensorStorageType::Native && options.backend.type == core::BackendType::Cuda),
      sampling_policy_(
          options.backend.type == core::BackendType::Cuda
              ? sampling::resolve_torch_cuda_sampling_policy(
                    options.backend.type,
                    options.backend.device,
                    "vibevoice_asr",
                    "VibeVoice-ASR",
                    sampling::TorchCudaSamplingPolicyFailureMode::StrictCuda)
              : sampling::TorchCudaSamplingPolicy{}),
      tokenizer_(assets_),
      frontend_(assets_),
      speech_encoder_(
          assets_,
          options.backend.type,
          options.backend.device,
          options.backend.threads,
          tokenizer_weight_context_bytes_,
          connector_weight_context_bytes_,
          tokenizer_weight_storage_type_,
          connector_weight_storage_type_),
      text_decoder_(
          assets_,
          options.backend.type,
          options.backend.device,
          options.backend.threads,
          decoder_weight_context_bytes_,
          128ull * 1024ull * 1024ull,
          decoder_weight_storage_type_),
      postprocessor_(tokenizer_),
      vad_model_path_(runtime::find_option(options.options, {"vibevoice_asr.vad_model_path"}).value_or(default_vad_model_path().string())) {
    if (task_.task != runtime::VoiceTaskKind::Asr || task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("VibeVoice-ASR streaming sessions are not supported");
    }
    validate_weight_storage(tokenizer_weight_storage_type_, "vibevoice_asr.tokenizer_weight_type");
    validate_weight_storage(connector_weight_storage_type_, "vibevoice_asr.connector_weight_type");
    validate_weight_storage(decoder_weight_storage_type_, "vibevoice_asr.decoder_weight_type");
    for (const auto & [key, value] : options.options) {
        (void)value;
        if (key.rfind("vibevoice_asr.", 0) == 0 &&
            key != "vibevoice_asr.weight_type" &&
            key != "vibevoice_asr.tokenizer_weight_type" &&
            key != "vibevoice_asr.connector_weight_type" &&
            key != "vibevoice_asr.decoder_weight_type" &&
            key != "vibevoice_asr.tokenizer_weight_context_mb" &&
            key != "vibevoice_asr.connector_weight_context_mb" &&
            key != "vibevoice_asr.decoder_weight_context_mb" &&
            key != "vibevoice_asr.vad_model_path") {
            throw std::runtime_error("unknown VibeVoice-ASR session option: " + key);
        }
    }
    assets_->model_weights->release_storage();
}

VibeVoiceASRSession::~VibeVoiceASRSession() = default;

std::string VibeVoiceASRSession::family() const {
    return "vibevoice_asr";
}

runtime::VoiceTaskKind VibeVoiceASRSession::task_kind() const {
    return task_.task;
}

runtime::RunMode VibeVoiceASRSession::run_mode() const {
    return task_.mode;
}

void VibeVoiceASRSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (!request.audio.has_value()) {
        throw std::runtime_error("VibeVoice-ASR prepare() requires an audio contract");
    }
    mark_prepared();
    debug::trace_log_scalar("vibevoice_asr.prepare.max_input_samples", request.audio->max_input_samples);
}

runtime::TaskResult VibeVoiceASRSession::run(const runtime::TaskRequest & request) {
    require_prepared("VibeVoice-ASR run()");
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("VibeVoice-ASR offline run called on non-offline session");
    }
    const auto chunks = audio_chunk_plan(request);
    if (chunks.empty()) {
        if (request.audio_input.has_value() &&
            engine::audio::parse_audio_chunk_mode(request.options) == engine::audio::AudioChunkMode::Vad) {
            runtime::TaskResult empty;
            empty.text_output = runtime::Transcript{"", request.text_input.has_value() ? request.text_input->language : ""};
            return empty;
        }
        return run_single(make_request(request));
    }
    const auto & audio = *request.audio_input;
    if (chunks.size() == 1) {
        auto item_request = request;
        item_request.audio_input = engine::audio::slice_audio_buffer(audio, chunks.front().source_span);
        auto item = run_single(make_request(item_request));
        runtime::TaskResult adjusted;
        if (item.text_output.has_value()) {
            adjusted.text_output = std::move(item.text_output);
        }
        adjusted.output_artifacts = std::move(item.output_artifacts);
        append_offset_speech_metadata(adjusted, item, chunks.front().source_span, chunks.front().keep_span);
        return adjusted;
    }
    runtime::TaskResult merged;
    std::ostringstream text;
    for (const auto & chunk : chunks) {
        auto item_request = request;
        item_request.audio_input = engine::audio::slice_audio_buffer(audio, chunk.source_span);
        auto item = run_single(make_request(item_request));
        if (item.text_output.has_value() && !item.text_output->text.empty()) {
            if (text.tellp() > 0) {
                text << ' ';
            }
            text << item.text_output->text;
            if (!merged.text_output.has_value()) {
                merged.text_output = runtime::Transcript{"", item.text_output->language};
            } else if (merged.text_output->language.empty()) {
                merged.text_output->language = item.text_output->language;
            }
        }
        append_offset_speech_metadata(merged, item, chunk.source_span, chunk.keep_span);
    }
    if (merged.text_output.has_value()) {
        merged.text_output->text = text.str();
    }
    return merged;
}

runtime::StreamingPolicy VibeVoiceASRSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::AudioChunks;
    policy.output = runtime::StreamingOutputKind::FinalResult;
    policy.preferred_audio_chunk_samples = assets_->processor.audio_processor.sample_rate;
    return policy;
}

void VibeVoiceASRSession::start_stream(const runtime::TaskRequest & request) {
    require_prepared("VibeVoice-ASR start_stream()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("VibeVoice-ASR start_stream called on non-streaming session");
    }
    reset();
    streaming_request_ = request;
    streaming_request_.audio_input = std::nullopt;
    streaming_result_ = runtime::TaskResult{};
    stream_started_ = true;
    streaming_chunks_processed_ = 0;
}

void VibeVoiceASRSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
    stream_event_sink_ = std::move(sink);
}

void VibeVoiceASRSession::reset() {
    require_prepared("VibeVoice-ASR reset()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("VibeVoice-ASR reset called on non-streaming session");
    }
    streaming_request_ = runtime::TaskRequest{};
    streaming_result_ = runtime::TaskResult{};
    stream_started_ = false;
    streaming_chunks_processed_ = 0;
}

runtime::StreamEvent VibeVoiceASRSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    require_prepared("VibeVoice-ASR process_audio_chunk()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("VibeVoice-ASR process_audio_chunk called on non-streaming session");
    }
    if (!stream_started_) {
        throw std::runtime_error("VibeVoice-ASR process_audio_chunk requires start_stream");
    }
    runtime::AudioBuffer audio;
    audio.sample_rate = chunk.sample_rate;
    audio.channels = chunk.channels;
    audio.samples = chunk.samples;

    auto request = streaming_request_;
    request.audio_input = std::move(audio);
    runtime::StreamEventCallback saved_sink;
    saved_sink.swap(stream_event_sink_);
    runtime::TaskResult result;
    try {
        result = run_single(make_request(request));
        saved_sink.swap(stream_event_sink_);
    } catch (...) {
        saved_sink.swap(stream_event_sink_);
        throw;
    }
    ++streaming_chunks_processed_;

    runtime::StreamEvent event;
    if (result.text_output.has_value()) {
        const std::string delta = append_streaming_transcript(streaming_result_, *result.text_output);
        if (!delta.empty()) {
            event.partial_text = runtime::Transcript{delta, streaming_result_.text_output->language};
        }
    }
    if (chunk.channels <= 0) {
        throw std::runtime_error("VibeVoice-ASR streamed audio chunk requires positive channels");
    }
    if (chunk.samples.size() % static_cast<size_t>(chunk.channels) != 0) {
        throw std::runtime_error("VibeVoice-ASR streamed audio samples must be divisible by channel count");
    }
    const runtime::TimeSpan streamed_span{
        chunk.start_sample,
        chunk.start_sample + static_cast<int64_t>(chunk.samples.size() / static_cast<size_t>(chunk.channels)),
    };
    append_offset_speech_metadata(streaming_result_, result, streamed_span, streamed_span);
    event.is_final = false;
    return event;
}

runtime::TaskResult VibeVoiceASRSession::finish_stream() {
    return finalize();
}

runtime::TaskResult VibeVoiceASRSession::finalize() {
    require_prepared("VibeVoice-ASR finalize()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("VibeVoice-ASR finalize called on non-streaming session");
    }
    if (!stream_started_) {
        throw std::runtime_error("VibeVoice-ASR finalize requires start_stream");
    }
    if (streaming_chunks_processed_ == 0) {
        throw std::runtime_error("VibeVoice-ASR finalize requires streamed audio");
    }
    if (!streaming_result_.text_output.has_value()) {
        streaming_result_.text_output = runtime::Transcript{};
    }
    return streaming_result_;
}

VibeVoiceASRRequest VibeVoiceASRSession::make_request(const runtime::TaskRequest & request) const {
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("VibeVoice-ASR run() requires audio_input");
    }
    VibeVoiceASRRequest out;
    out.audio = *request.audio_input;
    out.generation.max_new_tokens = 32768;
    if (request.text_input.has_value()) {
        out.context = request.text_input->text;
        out.language = request.text_input->language;
    }
    if (const auto language = runtime::find_option(request.options, {"language"})) {
        out.language = *language;
    }
    if (const auto value = runtime::parse_i64_option(request.options, {"max_tokens"})) {
        out.generation.max_new_tokens = *value;
    }
    if (const auto value = runtime::parse_u64_option(request.options, {"seed"})) {
        out.generation.seed = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"temperature"})) {
        out.generation.temperature = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"top_p"})) {
        out.generation.top_p = *value;
    }
    if (const auto value = runtime::parse_i64_option(request.options, {"top_k"})) {
        out.generation.top_k = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"repetition_penalty"})) {
        out.generation.repetition_penalty = *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"num_beams"})) {
        out.generation.num_beams = *value;
    }
    if (out.generation.max_new_tokens <= 0) {
        throw std::runtime_error("VibeVoice-ASR max_tokens must be positive");
    }
    if (out.generation.num_beams <= 0) {
        throw std::runtime_error("VibeVoice-ASR num_beams must be positive");
    }
    if (!(out.generation.top_p > 0.0F && out.generation.top_p <= 1.0F) || !std::isfinite(out.generation.top_p)) {
        throw std::runtime_error("VibeVoice-ASR top_p must be finite and in (0, 1]");
    }
    if (out.generation.top_k < 0) {
        throw std::runtime_error("VibeVoice-ASR top_k must be non-negative");
    }
    if (!(out.generation.repetition_penalty > 0.0F) || !std::isfinite(out.generation.repetition_penalty)) {
        throw std::runtime_error("VibeVoice-ASR repetition_penalty must be finite and positive");
    }
    if (out.generation.temperature < 0.0F || !std::isfinite(out.generation.temperature)) {
        throw std::runtime_error("VibeVoice-ASR temperature must be finite and non-negative");
    }
    return out;
}

std::vector<VibeVoiceASRSession::AudioChunkPlan> VibeVoiceASRSession::audio_chunk_plan(
    const runtime::TaskRequest & request) {
    if (!request.audio_input.has_value()) {
        return {};
    }
    const auto mode = engine::audio::parse_audio_chunk_mode(request.options);
    if (mode == engine::audio::AudioChunkMode::None) {
        return {};
    }
    if (mode == engine::audio::AudioChunkMode::QuietEnergy) {
        throw std::runtime_error("VibeVoice-ASR does not support audio_chunk_mode=quiet_energy");
    }

    const auto & audio = *request.audio_input;
    const int64_t frames = audio_frame_count(audio);
    if (mode == engine::audio::AudioChunkMode::Vad) {
        const auto seconds = engine::audio::parse_audio_chunk_seconds_override(request.options)
            .value_or(static_cast<float>(kDefaultAudioChunkSeconds));
        if (!(seconds > 0.0F)) {
            throw std::runtime_error("VibeVoice-ASR audio_chunk_seconds must be positive");
        }
        const auto options = engine::audio::VadAudioChunkOptions{
            static_cast<int64_t>(std::llround(static_cast<double>(seconds) * static_cast<double>(audio.sample_rate))),
            static_cast<int64_t>(std::llround(0.5 * static_cast<double>(audio.sample_rate))),
            static_cast<int64_t>(std::llround(0.25 * static_cast<double>(audio.sample_rate))),
        };
        if (options.max_chunk_samples <= 0) {
            throw std::runtime_error("VibeVoice-ASR audio_chunk_seconds produced an empty chunk");
        }
        const auto spans = engine::audio::plan_vad_audio_chunks(audio, vad_session(), options);
        std::vector<AudioChunkPlan> plan;
        plan.reserve(spans.size());
        for (const auto & span : spans) {
            plan.push_back(AudioChunkPlan{span, span});
        }
        return plan;
    }

    const auto seconds = engine::audio::parse_audio_chunk_seconds_override(request.options)
        .value_or(static_cast<float>(kDefaultAudioChunkSeconds));
    if (!(seconds > 0.0F)) {
        throw std::runtime_error("VibeVoice-ASR audio_chunk_seconds must be positive");
    }
    const int64_t samples = static_cast<int64_t>(
        std::llround(static_cast<double>(seconds) * static_cast<double>(audio.sample_rate)));
    if (samples <= 0) {
        throw std::runtime_error("VibeVoice-ASR audio_chunk_seconds produced an empty chunk");
    }
    const auto chunks = engine::audio::plan_audio_chunks(
        frames,
        engine::audio::AudioChunkSpec{
            samples,
            samples,
            engine::audio::AudioChunkPadMode::Zero,
            engine::audio::AudioChunkTailAlignment::Start,
            0,
        });
    std::vector<AudioChunkPlan> plan;
    plan.reserve(chunks.size());
    for (const auto & chunk : chunks) {
        const runtime::TimeSpan span{
            chunk.output_start_sample,
            chunk.output_start_sample + chunk.valid_samples,
        };
        plan.push_back(AudioChunkPlan{span, span});
    }
    return plan;
}

runtime::IOfflineVoiceTaskSession & VibeVoiceASRSession::vad_session() {
    if (vad_session_ == nullptr) {
        runtime::ModelLoadRequest load_request;
        load_request.model_path = vad_model_path_;
        vad_model_ = engine::models::silero_vad::load_silero_vad_model(load_request);
        auto session = vad_model_->create_task_session(
            runtime::TaskSpec{runtime::VoiceTaskKind::Vad, runtime::RunMode::Offline},
            runtime::SessionOptions{options().backend, {}});
        auto * offline = dynamic_cast<runtime::IOfflineVoiceTaskSession *>(session.get());
        if (offline == nullptr) {
            throw std::runtime_error("VibeVoice-ASR VAD helper did not create an offline session");
        }
        vad_session_.reset(offline);
        session.release();
    }
    return *vad_session_;
}

runtime::TaskResult VibeVoiceASRSession::run_single(const VibeVoiceASRRequest & request) {
    const auto wall_start = Clock::now();
    const auto frontend_start = Clock::now();
    const auto audio = frontend_.normalize(request.audio);
    const auto frontend_end = Clock::now();

    const auto speech_start = Clock::now();
    const auto speech = speech_encoder_.encode(audio, request.generation.seed);
    const auto speech_end = Clock::now();

    const auto prompt_start = Clock::now();
    auto prompt_request = request;
    prompt_request.audio = audio;
    const auto prompt = tokenizer_.build_prompt(prompt_request, speech.frames);
    const auto prompt_end = Clock::now();

    const auto decoder_start = Clock::now();
    auto prefill = text_decoder_.prefill_prompt(prompt.input_ids, speech.values, prompt.speech_positions);
    const uint64_t rng_call_offset = (speech.next_rng_index + 3ull) / 4ull;
    std::string emitted_text;
    std::function<void(const std::vector<int32_t> &)> token_callback;
    if (task_.mode == runtime::RunMode::Streaming && stream_event_sink_ != nullptr) {
        token_callback = [&](const std::vector<int32_t> & partial_tokens) {
            const std::string raw_text = tokenizer_.decode(partial_tokens, true);
            const std::string partial_text = streaming_transcript_from_vibevoice_raw(raw_text);
            emit_transcript_delta(
                stream_event_sink_,
                runtime::Transcript{partial_text, request.language},
                emitted_text);
        };
    }
    auto generated = generate_tokens(request, prompt, std::move(prefill), rng_call_offset, token_callback);
    const auto decoder_end = Clock::now();

    const auto post_start = Clock::now();
    const auto decoded = postprocessor_.decode(VibeVoiceASRGeneratedTokens{std::move(generated), {}});
    if (task_.mode == runtime::RunMode::Streaming && stream_event_sink_ != nullptr) {
        emit_transcript_delta(
            stream_event_sink_,
            runtime::Transcript{decoded.text, request.language},
            emitted_text);
    }
    const auto post_end = Clock::now();

    runtime::TaskResult result;
    result.text_output = runtime::Transcript{decoded.text, request.language};
    for (const auto & segment : decoded.segments) {
        runtime::SpeechSegment speech_segment;
        speech_segment.span.start_sample = static_cast<int64_t>(segment.start_time * audio.sample_rate);
        speech_segment.span.end_sample = static_cast<int64_t>(segment.end_time * audio.sample_rate);
        result.speech_segments.push_back(speech_segment);
        runtime::SpeakerTurn turn;
        turn.span = speech_segment.span;
        turn.speaker_id = segment.speaker_id;
        result.speaker_turns.push_back(std::move(turn));
    }

    debug::timing_log_scalar("vibevoice_asr.frontend_ms", engine::debug::elapsed_ms(frontend_start, frontend_end));
    debug::timing_log_scalar("vibevoice_asr.speech_encoder_ms", engine::debug::elapsed_ms(speech_start, speech_end));
    debug::timing_log_scalar("vibevoice_asr.prompt_ms", engine::debug::elapsed_ms(prompt_start, prompt_end));
    debug::timing_log_scalar("vibevoice_asr.text_decoder_ms", engine::debug::elapsed_ms(decoder_start, decoder_end));
    debug::timing_log_scalar("vibevoice_asr.postprocess_ms", engine::debug::elapsed_ms(post_start, post_end));
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    debug::trace_log_scalar("vibevoice_asr.prompt_tokens", prompt.input_ids.size());
    debug::trace_log_scalar("vibevoice_asr.speech_tokens", speech.frames);
    return result;
}

std::vector<int32_t> VibeVoiceASRSession::generate_tokens(
    const VibeVoiceASRRequest & request,
    const VibeVoiceASRPrompt & prompt,
    VibeVoiceDecoderPrefillOutput prefill,
    uint64_t rng_call_offset,
    const std::function<void(const std::vector<int32_t> &)> & token_callback) {
    if (request.generation.num_beams > 1) {
        return generate_beam(request, prompt, std::move(prefill), token_callback);
    }
    return generate_greedy_or_sample(request, prompt, std::move(prefill), rng_call_offset, token_callback);
}

std::vector<int32_t> VibeVoiceASRSession::generate_greedy_or_sample(
    const VibeVoiceASRRequest & request,
    const VibeVoiceASRPrompt & prompt,
    VibeVoiceDecoderPrefillOutput prefill,
    uint64_t rng_call_offset,
    const std::function<void(const std::vector<int32_t> &)> & token_callback) {
    VibeVoiceDecoderCachedState state;
    text_decoder_.reset_cached_state(state, std::move(prefill.state));
    std::vector<int32_t> generated;
    generated.reserve(static_cast<size_t>(std::min<int64_t>(request.generation.max_new_tokens, 4096)));
    std::vector<uint8_t> seen;
    std::mt19937 rng(static_cast<uint32_t>(request.generation.seed));
    auto logits = std::move(prefill.result.logits.values);
    const int64_t prompt_tokens = static_cast<int64_t>(prompt.input_ids.size());
    for (int64_t step = 0; step < request.generation.max_new_tokens; ++step) {
        apply_repetition_penalty(
            logits,
            prompt.input_ids,
            generated,
            request.generation.repetition_penalty,
            seen);
        const int32_t next = request.generation.temperature > 0.0F
            ? sample_token(
                  logits,
                  request.generation.temperature,
                  request.generation.top_p,
                  request.generation.top_k,
                  request.generation.seed,
                  rng_call_offset + static_cast<uint64_t>(step),
                  sampling_policy_,
                  rng,
                  greedy_compare_bf16_)
            : argmax_token(logits, false);
        generated.push_back(next);
        if (token_callback) {
            token_callback(generated);
        }
        if (next == tokenizer_.eos_id()) {
            break;
        }
        const auto next_embedding = text_decoder_.embed_tokens({next});
        const int64_t cache_capacity = prompt_tokens + static_cast<int64_t>(generated.size()) + 1;
        const auto result = text_decoder_.cached_step(next_embedding.values, state, cache_capacity);
        logits = result.logits.values;
    }
    return generated;
}

std::vector<int32_t> VibeVoiceASRSession::generate_beam(
    const VibeVoiceASRRequest & request,
    const VibeVoiceASRPrompt & prompt,
    VibeVoiceDecoderPrefillOutput prefill,
    const std::function<void(const std::vector<int32_t> &)> & token_callback) {
    struct Beam {
        std::vector<int32_t> generated;
        std::unique_ptr<VibeVoiceDecoderCachedState> state;
        double score = 0.0;
        bool done = false;
    };
    struct Candidate {
        size_t parent = 0;
        int32_t token = 0;
        double score = 0.0;
        double rank_score = 0.0;
    };
    const int64_t beam_count = request.generation.num_beams;
    const int64_t prompt_tokens = static_cast<int64_t>(prompt.input_ids.size());
    std::vector<uint8_t> seen;
    auto first_logits = std::move(prefill.result.logits.values);
    std::vector<int32_t> empty_generated;
    apply_repetition_penalty(
        first_logits,
        prompt.input_ids,
        empty_generated,
        request.generation.repetition_penalty,
        seen);
    const auto first = top_log_probs(first_logits, beam_count, greedy_compare_bf16_);
    std::vector<Beam> beams;
    beams.reserve(static_cast<size_t>(beam_count));
    for (const auto & item : first) {
        Beam beam;
        beam.generated.push_back(item.token);
        beam.state = std::make_unique<VibeVoiceDecoderCachedState>();
        text_decoder_.reset_cached_state(*beam.state, prefill.state);
        beam.score = item.log_prob;
        beam.done = item.token == tokenizer_.eos_id();
        beams.push_back(std::move(beam));
    }
    auto better_candidate = [](const Candidate & lhs, const Candidate & rhs) {
        if (lhs.rank_score == rhs.rank_score) {
            return lhs.token < rhs.token;
        }
        return lhs.rank_score > rhs.rank_score;
    };
    auto best_beam = [](std::vector<Beam> & values) {
        return std::max_element(values.begin(), values.end(), [](const Beam & lhs, const Beam & rhs) {
            const double lhs_len = std::max<size_t>(lhs.generated.size(), 1);
            const double rhs_len = std::max<size_t>(rhs.generated.size(), 1);
            const double lhs_score = lhs.score / lhs_len;
            const double rhs_score = rhs.score / rhs_len;
            if (lhs_score == rhs_score) {
                return lhs.generated > rhs.generated;
            }
            return lhs_score < rhs_score;
        });
    };
    if (token_callback && !beams.empty()) {
        token_callback(best_beam(beams)->generated);
    }
    const int64_t per_beam_candidates = std::max<int64_t>(beam_count * 2, 1);
    for (int64_t step = 1; step < request.generation.max_new_tokens; ++step) {
        std::vector<size_t> active_indices;
        active_indices.reserve(beams.size());
        for (size_t i = 0; i < beams.size(); ++i) {
            if (!beams[i].done) {
                active_indices.push_back(i);
            }
        }
        if (active_indices.empty()) {
            break;
        }

        std::vector<Candidate> candidates;
        candidates.reserve(static_cast<size_t>(beam_count * per_beam_candidates + beam_count));
        for (size_t beam_index = 0; beam_index < beams.size(); ++beam_index) {
            const auto & beam = beams[beam_index];
            if (beam.done) {
                const double length = std::max<size_t>(beam.generated.size(), 1);
                candidates.push_back({beam_index, tokenizer_.eos_id(), beam.score, beam.score / length});
            }
        }
        for (size_t row = 0; row < active_indices.size(); ++row) {
            const auto parent_index = active_indices[row];
            if (beams[parent_index].state == nullptr) {
                throw std::runtime_error("VibeVoice-ASR beam state is missing");
            }
            const auto embedding = text_decoder_.embed_tokens({beams[parent_index].generated.back()});
            const int64_t cache_capacity = prompt_tokens +
                static_cast<int64_t>(beams[parent_index].generated.size()) + 1;
            auto result = text_decoder_.cached_step(embedding.values, *beams[parent_index].state, cache_capacity);
            auto logits = std::move(result.logits.values);
            apply_repetition_penalty(
                logits,
                prompt.input_ids,
                beams[parent_index].generated,
                request.generation.repetition_penalty,
                seen);
            for (const auto & item : top_log_probs(logits, per_beam_candidates, greedy_compare_bf16_)) {
                const double score = beams[parent_index].score + item.log_prob;
                const double length = static_cast<double>(beams[parent_index].generated.size() + 1);
                candidates.push_back({parent_index, item.token, score, score / length});
            }
        }
        if (candidates.empty()) {
            break;
        }
        const size_t keep = std::min<size_t>(candidates.size(), static_cast<size_t>(beam_count));
        if (keep < candidates.size()) {
            std::nth_element(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(keep), candidates.end(), better_candidate);
            candidates.resize(keep);
        }
        std::sort(candidates.begin(), candidates.end(), better_candidate);
        std::vector<size_t> parent_use_count(beams.size(), 0);
        for (auto & candidate : candidates) {
            ++parent_use_count[candidate.parent];
        }
        std::vector<std::vector<std::unique_ptr<VibeVoiceDecoderCachedState>>> cloned_parent_states(beams.size());
        for (size_t parent = 0; parent < beams.size(); ++parent) {
            if (parent_use_count[parent] > 1 && beams[parent].state != nullptr) {
                auto & clones = cloned_parent_states[parent];
                clones.reserve(parent_use_count[parent] - 1);
                for (size_t clone = 1; clone < parent_use_count[parent]; ++clone) {
                    auto state = std::make_unique<VibeVoiceDecoderCachedState>();
                    const int64_t cache_capacity = prompt_tokens +
                        static_cast<int64_t>(beams[parent].generated.size()) + 1;
                    text_decoder_.clone_cached_state(*beams[parent].state, *state, cache_capacity);
                    clones.push_back(std::move(state));
                }
            }
        }

        std::vector<uint8_t> parent_moved(beams.size(), 0);
        std::vector<size_t> parent_clone_index(beams.size(), 0);
        std::vector<Beam> next_beams;
        next_beams.reserve(keep);
        for (const auto & candidate : candidates) {
            auto & parent = beams[candidate.parent];
            Beam child;
            child.generated = parent.generated;
            if (!parent.done) {
                child.generated.push_back(candidate.token);
            }
            child.score = candidate.score;
            child.done = parent.done || candidate.token == tokenizer_.eos_id();
            if (!child.done) {
                if (parent_moved[candidate.parent] == 0) {
                    child.state = std::move(parent.state);
                    parent_moved[candidate.parent] = 1;
                } else if (
                    parent_clone_index[candidate.parent] <
                    cloned_parent_states[candidate.parent].size()) {
                    child.state = std::move(
                        cloned_parent_states[candidate.parent][parent_clone_index[candidate.parent]++]);
                } else {
                    throw std::runtime_error("VibeVoice-ASR beam parent state was already moved");
                }
            }
            next_beams.push_back(std::move(child));
        }
        beams = std::move(next_beams);
        if (token_callback && !beams.empty()) {
            token_callback(best_beam(beams)->generated);
        }
    }
    if (beams.empty()) {
        return {};
    }
    return best_beam(beams)->generated;
}

}  // namespace engine::models::vibevoice_asr
