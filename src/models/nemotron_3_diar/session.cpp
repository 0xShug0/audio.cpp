#include "engine/models/nemotron_3_diar/session.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace engine::models::nemotron_3_diar {
namespace {

constexpr const char * kFamily = "nemotron_3_diar";
constexpr int64_t kSampleRate = 16000;
constexpr int64_t kOutputHopSamples = 160;
constexpr size_t kDefaultGraphArenaBytes = 1024ull * 1024ull * 1024ull;
constexpr size_t kDefaultWeightContextBytes = 1024ull * 1024ull * 1024ull;

template <typename T>
std::shared_ptr<const T> require_value(std::shared_ptr<const T> value, const char * label) {
    if (value == nullptr) throw std::runtime_error(std::string("Nemotron 3 diarization requires ") + label);
    return value;
}

std::string speaker_id(int64_t speaker) {
    return "speaker_" + std::to_string(speaker);
}

bool wants_frame_probabilities(const std::unordered_map<std::string, std::string> & options) {
    const auto value = runtime::find_option(options, {"return_frame_probabilities"});
    return value.has_value() && runtime::parse_bool_option(*value, "return_frame_probabilities");
}

// Raw little-endian F32 speaker activity, frame-major [frames, speakers], at
// the native 10 ms output cadence.
runtime::VoiceArtifact frame_probability_artifact(
    const std::vector<float> & probabilities, int64_t frames, int64_t speakers) {
    std::vector<std::byte> payload(static_cast<size_t>(frames * speakers) * sizeof(float));
    std::memcpy(payload.data(), probabilities.data(), payload.size());
    return runtime::make_voice_artifact(
        runtime::ArtifactKind::DiarizationState, "speaker_probabilities", std::move(payload),
        {{"extension", "f32"},
         {"mime", "application/octet-stream"},
         {"dtype", "f32"},
         {"layout", "frames,speakers"},
         {"frames", std::to_string(frames)},
         {"speakers", std::to_string(speakers)},
         {"frame_hop_samples", std::to_string(kOutputHopSamples)},
         {"sample_rate", std::to_string(kSampleRate)}});
}

}  // namespace

Session::Session(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const Assets> assets,
    std::shared_ptr<const model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_value(std::move(assets), "assets")),
      contract_(require_value(std::move(contract), "model contract")),
      streaming_config_(streaming_profile(assets_->model_config.streaming, options.options)),
      graph_arena_bytes_(runtime::parse_size_mb_option(
          options.options, {"nemotron_3_diar.graph_arena_mb"}, kDefaultGraphArenaBytes)),
      weight_context_bytes_(runtime::parse_size_mb_option(
          options.options, {"nemotron_3_diar.weight_context_mb"}, kDefaultWeightContextBytes)) {
    if (task_.task != runtime::VoiceTaskKind::Diarization) {
        throw std::runtime_error("Nemotron 3 Diarization only supports diarization");
    }
    runtime::validate_spec_backed_session_options(options, *contract_, kFamily, "Nemotron 3 diarization");
    const auto storage = runtime::parse_tensor_storage_option(
        options.options,
        "nemotron_3_diar.weight_type",
        "nemotron_3_diar.weight_type",
        assets::TensorStorageType::Native,
        {assets::TensorStorageType::Native, assets::TensorStorageType::F32,
         assets::TensorStorageType::F16, assets::TensorStorageType::BF16,
         assets::TensorStorageType::Q8_0});
    weights_ = load_weights(
        *assets_, execution_context().backend(), execution_context().backend_type(),
        storage, weight_context_bytes_);
    assets_->model_weights->release_storage();
    use_flash_attention_ = core::resolve_flash_attention(
        execution_context().backend(),
        assets_->model_config.encoder.hidden_size / assets_->model_config.encoder.heads,
        core::parse_attention_preference(
            runtime::find_option(options.options, {"nemotron_3_diar.attention"}).value_or("auto"),
            "nemotron_3_diar.attention"));
    stream_scheduler_ = std::make_unique<StreamScheduler>(
        assets_->feature_config, streaming_config_, assets_->model_config.encoder.subsampling_factor);
}

Session::~Session() = default;

std::string Session::family() const { return kFamily; }
runtime::VoiceTaskKind Session::task_kind() const { return task_.task; }
runtime::RunMode Session::run_mode() const { return task_.mode; }

void Session::prepare(const runtime::SessionPreparationRequest & request) {
    if (request.audio.has_value() && request.audio->sample_rate != 0 && request.audio->sample_rate != kSampleRate) {
        throw std::runtime_error("Nemotron 3 diarization requires 16 kHz audio");
    }
    mark_prepared();
    if (task_.mode == runtime::RunMode::Streaming) reset();
}

Session::DecodeConfig Session::decode_config(
    const std::unordered_map<std::string, std::string> & options) const {
    DecodeConfig config;
    if (const auto value = runtime::parse_finite_float_option(options, {"speaker_threshold"})) config.threshold = *value;
    if (const auto value = runtime::parse_i64_option(options, {"speaker_min_frames"})) config.min_frames = *value;
    if (const auto value = runtime::parse_i64_option(options, {"speaker_pad_frames"})) config.pad_frames = *value;
    if (config.threshold < 0.0F || config.threshold > 1.0F || config.min_frames < 0 || config.pad_frames < 0) {
        throw std::runtime_error("invalid Nemotron 3 diarization decode options");
    }
    return config;
}

std::vector<float> Session::pre_encode(const FeatureBatch & features) {
    ensure_pre_encode_graph(
        pre_encode_graph_, execution_context(), *assets_, *weights_, graph_arena_bytes_,
        features.batch, features.encoder_frames);
    core::write_tensor_f32(pre_encode_graph_->input, features.stacked);
    core::set_backend_threads(execution_context().backend(), pre_encode_graph_->threads);
    if (core::compute_backend_graph(
            execution_context().backend(), pre_encode_graph_->graph, pre_encode_graph_->plan) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Nemotron 3 diarization pre-encode graph failed");
    }
    std::vector<float> embeddings;
    core::read_tensor_f32_into(pre_encode_graph_->output.tensor, embeddings);
    return embeddings;
}

std::vector<float> Session::encode(
    const std::vector<float> & embeddings,
    int64_t batch,
    int64_t frames,
    const std::vector<int64_t> & valid_frames) {
    ensure_encoder_graph(
        encoder_graph_, execution_context(), *assets_, *weights_, graph_arena_bytes_,
        batch, frames, use_flash_attention_);
    core::write_tensor_f32(encoder_graph_->input, embeddings);
    core::write_tensor_f16(
        encoder_graph_->attention_mask,
        attention_mask(valid_frames, frames));
    core::write_tensor_f32(encoder_graph_->frame_mask, frame_mask(valid_frames, frames));
    core::set_backend_threads(execution_context().backend(), encoder_graph_->threads);
    if (core::compute_backend_graph(
            execution_context().backend(), encoder_graph_->graph, encoder_graph_->plan) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Nemotron 3 diarization encoder graph failed");
    }
    std::vector<float> probabilities;
    core::read_tensor_f32_into(encoder_graph_->probabilities.tensor, probabilities);
    return probabilities;
}

void Session::process_window_batch(
    const std::vector<StreamWindow> & windows,
    const std::vector<AoscState *> & states,
    const std::vector<std::vector<float> *> & timelines) {
    if (windows.empty()) return;
    if (windows.size() != states.size() || windows.size() != timelines.size()) {
        throw std::runtime_error("Nemotron 3 diarization streaming batch shape mismatch");
    }
    auto features = compute_stream_features(
        windows, *assets_, execution_context().config().threads);
    if (task_.mode == runtime::RunMode::Streaming) {
        if (windows.size() != 1) {
            throw std::runtime_error("Nemotron 3 diarization streaming requires a single graph row");
        }
        const int64_t capacity = streaming_config_.chunk_left_context +
            streaming_config_.chunk_len + streaming_config_.chunk_right_context;
        if (features.encoder_frames > capacity) {
            throw std::runtime_error("Nemotron 3 diarization streaming window exceeds graph capacity");
        }
        features.stacked.resize(static_cast<size_t>(
            capacity * assets_->model_config.encoder.subsampling_factor *
            assets_->feature_config.num_mel_bins), 0.0F);
        features.encoder_frames = capacity;
        features.feature_frames = capacity * assets_->model_config.encoder.subsampling_factor;
    }
    const auto chunk_embeddings = pre_encode(features);
    std::vector<int64_t> central_frames;
    std::vector<int64_t> left_context_frames;
    std::vector<int64_t> right_context_frames;
    central_frames.reserve(windows.size());
    left_context_frames.reserve(windows.size());
    right_context_frames.reserve(windows.size());
    for (size_t row = 0; row < windows.size(); ++row) {
        central_frames.push_back(windows[row].encoder_frames);
        left_context_frames.push_back(windows[row].left_context_frames);
        right_context_frames.push_back(windows[row].right_context_frames);
    }
    process_embedding_batch(
        chunk_embeddings, static_cast<int64_t>(windows.size()), features.encoder_frames,
        features.valid_encoder_frames, central_frames, left_context_frames,
        right_context_frames, states, timelines);
}

void Session::process_embedding_batch(
    const std::vector<float> & chunk_embeddings,
    int64_t batch,
    int64_t chunk_capacity,
    const std::vector<int64_t> & chunk_frames,
    const std::vector<int64_t> & central_frames,
    const std::vector<int64_t> & left_context_frames,
    const std::vector<int64_t> & right_context_frames,
    const std::vector<AoscState *> & states,
    const std::vector<std::vector<float> *> & timelines) {
    const size_t rows = static_cast<size_t>(batch);
    if (batch <= 0 || chunk_frames.size() != rows || central_frames.size() != rows ||
        left_context_frames.size() != rows || right_context_frames.size() != rows ||
        states.size() != rows || timelines.size() != rows) {
        throw std::runtime_error("Nemotron 3 diarization embedding batch shape mismatch");
    }
    const int64_t hidden = assets_->model_config.encoder.hidden_size;
    const int64_t speakers = assets_->model_config.num_speakers;
    const int64_t upsample = assets_->model_config.head.upsample_factor;
    if (chunk_embeddings.size() != rows * static_cast<size_t>(chunk_capacity * hidden)) {
        throw std::runtime_error("Nemotron 3 diarization embedding batch data mismatch");
    }

    std::vector<int64_t> packed_lengths;
    packed_lengths.reserve(rows);
    int64_t packed_capacity = 0;
    for (size_t row = 0; row < rows; ++row) {
        const int64_t state_frames = states[row]->speaker_cache_frames() + states[row]->fifo_frames();
        packed_lengths.push_back(state_frames + chunk_frames[row]);
        packed_capacity = std::max(packed_capacity, state_frames + chunk_frames[row]);
    }
    if (task_.mode == runtime::RunMode::Streaming) {
        const int64_t capacity = streaming_config_.spkcache_len + streaming_config_.fifo_len + chunk_capacity;
        if (packed_capacity > capacity) {
            throw std::runtime_error("Nemotron 3 diarization streaming state exceeds graph capacity");
        }
        packed_capacity = capacity;
    }
    std::vector<float> packed(
        rows * static_cast<size_t>(packed_capacity * hidden), 0.0F);
    for (size_t row = 0; row < rows; ++row) {
        auto * output = packed.data() + row * static_cast<size_t>(packed_capacity * hidden);
        const auto & cache = states[row]->speaker_cache();
        const auto & fifo = states[row]->fifo();
        std::copy(cache.begin(), cache.end(), output);
        std::copy(fifo.begin(), fifo.end(), output + cache.size());
        const auto * chunk = chunk_embeddings.data() + row * static_cast<size_t>(chunk_capacity * hidden);
        std::copy_n(
            chunk, static_cast<size_t>(chunk_frames[row] * hidden),
            output + cache.size() + fifo.size());
    }

    const auto high_resolution = encode(
        packed, batch, packed_capacity, packed_lengths);
    const int64_t high_capacity = packed_capacity * upsample;
    for (size_t row = 0; row < rows; ++row) {
        const int64_t old_cache = states[row]->speaker_cache_frames();
        const int64_t old_fifo = states[row]->fifo_frames();
        const int64_t valid_chunk_frames = chunk_frames[row];
        const int64_t left = std::min<int64_t>(left_context_frames[row], valid_chunk_frames);
        const int64_t right = std::min<int64_t>(
            right_context_frames[row], valid_chunk_frames - left);
        const int64_t central = std::min<int64_t>(
            central_frames[row], valid_chunk_frames - left - right);
        if (central <= 0) continue;
        const auto * row_high = high_resolution.data() + row * static_cast<size_t>(high_capacity * speakers);
        const int64_t high_start = (old_cache + old_fifo + left) * upsample;
        timelines[row]->insert(
            timelines[row]->end(),
            row_high + static_cast<size_t>(high_start * speakers),
            row_high + static_cast<size_t>((high_start + central * upsample) * speakers));

        std::vector<float> low_resolution(static_cast<size_t>(packed_lengths[row] * speakers), 0.0F);
        for (int64_t frame = 0; frame < packed_lengths[row]; ++frame) {
            for (int64_t sub = 0; sub < upsample; ++sub) {
                const float * source = row_high + static_cast<size_t>((frame * upsample + sub) * speakers);
                float * destination = low_resolution.data() + static_cast<size_t>(frame * speakers);
                for (int64_t speaker = 0; speaker < speakers; ++speaker) {
                    destination[speaker] += source[speaker] / static_cast<float>(upsample);
                }
            }
        }
        const auto * row_chunk = chunk_embeddings.data() + row * static_cast<size_t>(chunk_capacity * hidden);
        states[row]->update(
            row_chunk, static_cast<int>(valid_chunk_frames), low_resolution.data(),
            static_cast<int>(left), static_cast<int>(right));
    }
}

std::vector<runtime::SpeakerTurn> Session::decode_turns(
    const std::vector<float> & probabilities,
    int64_t frames,
    const DecodeConfig & config,
    bool include_open_turns) const {
    const int64_t speakers = assets_->model_config.num_speakers;
    std::vector<runtime::SpeakerTurn> turns;
    for (int64_t speaker = 0; speaker < speakers; ++speaker) {
        int64_t start = -1;
        for (int64_t frame = 0; frame <= frames; ++frame) {
            const bool active = frame < frames &&
                probabilities[static_cast<size_t>(frame * speakers + speaker)] >= config.threshold;
            if (active && start < 0) {
                start = frame;
            } else if (!active && start >= 0) {
                const int64_t end = frame;
                if (include_open_turns || end < frames) {
                    const int64_t padded_start = std::max<int64_t>(0, start - config.pad_frames);
                    const int64_t padded_end = std::min<int64_t>(frames, end + config.pad_frames);
                    if (padded_end - padded_start >= config.min_frames) {
                        float confidence = 0.0F;
                        for (int64_t index = start; index < end; ++index) {
                            confidence += probabilities[static_cast<size_t>(index * speakers + speaker)];
                        }
                        confidence /= static_cast<float>(std::max<int64_t>(1, end - start));
                        turns.push_back({
                            {padded_start * kOutputHopSamples, padded_end * kOutputHopSamples},
                            speaker_id(speaker), confidence, "",
                        });
                    }
                }
                start = -1;
            }
        }
    }
    std::sort(turns.begin(), turns.end(), [](const auto & left, const auto & right) {
        if (left.span.start_sample != right.span.start_sample) return left.span.start_sample < right.span.start_sample;
        return left.speaker_id < right.speaker_id;
    });
    return turns;
}

runtime::TaskResult Session::run(const runtime::TaskRequest & request) {
    auto results = run_batch({request});
    return std::move(results.front());
}

std::vector<runtime::TaskResult> Session::run_batch(
    const std::vector<runtime::TaskRequest> & requests) {
    std::vector<runtime::TaskResult> results(requests.size());
    run_batch(requests, [&](size_t index, runtime::TaskResult result) {
        results[index] = std::move(result);
    });
    return results;
}

void Session::run_batch(
    const std::vector<runtime::TaskRequest> & requests,
    const runtime::IBatchedOfflineVoiceTaskSession::ResultCallback & on_result) {
    require_prepared("Nemotron 3 diarization run_batch()");
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Nemotron 3 diarization run_batch() requires offline mode");
    }
    if (requests.empty()) throw std::runtime_error("Nemotron 3 diarization batch must not be empty");
    std::vector<DecodeConfig> decoding;
    decoding.reserve(requests.size());
    std::vector<std::unique_ptr<AoscState>> states;
    std::vector<std::vector<float>> probabilities(requests.size());
    states.reserve(requests.size());
    std::vector<runtime::AudioBuffer> audio;
    audio.reserve(requests.size());
    for (const auto & request : requests) {
        runtime::validate_spec_backed_request_options(request.options, *contract_, "Nemotron 3 diarization");
        if (!request.audio_input.has_value() || request.audio_input->sample_rate != kSampleRate ||
            request.audio_input->channels <= 0) {
            throw std::runtime_error("Nemotron 3 diarization requires 16 kHz audio input");
        }
        decoding.push_back(decode_config(request.options));
        audio.push_back(*request.audio_input);
        states.push_back(std::make_unique<AoscState>(
            streaming_config_, static_cast<int>(assets_->model_config.num_speakers),
            static_cast<int>(assets_->model_config.encoder.hidden_size), assets_->silence_embedding));
    }

    const auto features = compute_features(audio, *assets_, execution_context().config().threads);
    const auto full_embeddings = pre_encode(features);
    const int64_t hidden = assets_->model_config.encoder.hidden_size;
    const int64_t chunk_len = streaming_config_.chunk_len;
    const int64_t speakers = assets_->model_config.num_speakers;
    std::vector<bool> completed(requests.size(), false);
    auto finish_row = [&](size_t row) {
        const int64_t input_samples = static_cast<int64_t>(
            requests[row].audio_input->samples.size() /
            static_cast<size_t>(requests[row].audio_input->channels));
        const int64_t expected_frames = input_samples / kOutputHopSamples;
        const int64_t frames = std::min<int64_t>(
            expected_frames, static_cast<int64_t>(probabilities[row].size() / speakers));
        probabilities[row].resize(static_cast<size_t>(frames * speakers));
        runtime::TaskResult result;
        result.speaker_turns = decode_turns(probabilities[row], frames, decoding[row], true);
        if (wants_frame_probabilities(requests[row].options)) {
            result.output_artifacts.push_back(frame_probability_artifact(probabilities[row], frames, speakers));
        }
        completed[row] = true;
        on_result(row, std::move(result));
    };
    for (int64_t start = 0;; start += chunk_len) {
        bool any_active = false;
        int64_t chunk_capacity = 0;
        for (size_t row = 0; row < requests.size(); ++row) {
            if (start >= features.valid_encoder_frames[row]) continue;
            any_active = true;
            const int64_t left = std::min<int64_t>(streaming_config_.chunk_left_context, start);
            const int64_t end = std::min<int64_t>(start + chunk_len, features.valid_encoder_frames[row]);
            const int64_t right = std::min<int64_t>(
                streaming_config_.chunk_right_context, features.valid_encoder_frames[row] - end);
            chunk_capacity = std::max(chunk_capacity, left + end - start + right);
        }
        if (!any_active) break;

        std::vector<float> chunk_embeddings(
            requests.size() * static_cast<size_t>(chunk_capacity * hidden), 0.0F);
        std::vector<int64_t> chunk_frames;
        std::vector<int64_t> central_frames;
        std::vector<int64_t> left_context_frames;
        std::vector<int64_t> right_context_frames;
        std::vector<AoscState *> active_states;
        std::vector<std::vector<float> *> timelines;
        for (size_t row = 0; row < requests.size(); ++row) {
            active_states.push_back(states[row].get());
            timelines.push_back(&probabilities[row]);
            if (start >= features.valid_encoder_frames[row]) {
                chunk_frames.push_back(0);
                central_frames.push_back(0);
                left_context_frames.push_back(0);
                right_context_frames.push_back(0);
                continue;
            }
            const int64_t left = std::min<int64_t>(streaming_config_.chunk_left_context, start);
            const int64_t end = std::min<int64_t>(start + chunk_len, features.valid_encoder_frames[row]);
            const int64_t right = std::min<int64_t>(
                streaming_config_.chunk_right_context, features.valid_encoder_frames[row] - end);
            const int64_t frames = left + end - start + right;
            const auto * source = full_embeddings.data() +
                (row * static_cast<size_t>(features.encoder_frames) + static_cast<size_t>(start - left)) * hidden;
            auto * destination = chunk_embeddings.data() + row * static_cast<size_t>(chunk_capacity * hidden);
            std::copy_n(source, static_cast<size_t>(frames * hidden), destination);
            chunk_frames.push_back(frames);
            central_frames.push_back(end - start);
            left_context_frames.push_back(left);
            right_context_frames.push_back(right);
        }
        process_embedding_batch(
            chunk_embeddings, static_cast<int64_t>(requests.size()), chunk_capacity,
            chunk_frames, central_frames, left_context_frames, right_context_frames,
            active_states, timelines);
        for (size_t row = 0; row < requests.size(); ++row) {
            if (!completed[row] && start + chunk_len >= features.valid_encoder_frames[row]) {
                finish_row(row);
            }
        }
    }
    for (size_t row = 0; row < requests.size(); ++row) {
        if (!completed[row]) {
            finish_row(row);
        }
    }
}

runtime::StreamingPolicy Session::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::AudioChunks;
    policy.output = runtime::StreamingOutputKind::PullEvents;
    policy.preferred_audio_chunk_samples = streaming_config_.chunk_len * 1280;
    policy.preferred_audio_chunk_seconds =
        static_cast<double>(policy.preferred_audio_chunk_samples) / kSampleRate;
    return policy;
}

void Session::start_stream(const runtime::TaskRequest & request) {
    require_prepared("Nemotron 3 diarization start_stream()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Nemotron 3 diarization start_stream() requires streaming mode");
    }
    runtime::validate_spec_backed_request_options(request.options, *contract_, "Nemotron 3 diarization");
    reset();
    stream_request_ = request;
    stream_request_.audio_input.reset();
    stream_started_ = true;
}

void Session::set_stream_event_sink(runtime::StreamEventCallback sink) {
    stream_event_sink_ = std::move(sink);
}

void Session::reset() {
    stream_scheduler_->reset();
    stream_state_ = std::make_unique<AoscState>(
        streaming_config_, static_cast<int>(assets_->model_config.num_speakers),
        static_cast<int>(assets_->model_config.encoder.hidden_size), assets_->silence_embedding);
    stream_probabilities_.clear();
    stream_samples_ = 0;
    emitted_stream_turns_.clear();
    stream_request_ = {};
    stream_started_ = false;
}

runtime::StreamEvent Session::process_stream_windows(const std::vector<StreamWindow> & windows) {
    for (const auto & window : windows) {
        process_window_batch({window}, {stream_state_.get()}, {&stream_probabilities_});
    }
    const int64_t frames = static_cast<int64_t>(
        stream_probabilities_.size() / assets_->model_config.num_speakers);
    const auto turns = decode_turns(
        stream_probabilities_, frames, decode_config(stream_request_.options), false);
    runtime::StreamEvent event;
    for (const auto & turn : turns) {
        const auto key = std::make_tuple(
            turn.span.start_sample, turn.span.end_sample, turn.speaker_id);
        if (emitted_stream_turns_.insert(key).second) {
            event.speaker_turns.push_back(turn);
        }
    }
    if (!event.speaker_turns.empty() && stream_event_sink_) {
        stream_event_sink_(event);
        return {};
    }
    return event;
}

runtime::StreamEvent Session::process_audio_chunk(const runtime::AudioChunk & chunk) {
    require_prepared("Nemotron 3 diarization process_audio_chunk()");
    if (!stream_started_ || chunk.sample_rate != kSampleRate || chunk.channels != 1) {
        throw std::runtime_error("Nemotron 3 diarization received an invalid streaming audio chunk");
    }
    stream_samples_ += static_cast<int64_t>(chunk.samples.size());
    return process_stream_windows(stream_scheduler_->push_audio(chunk));
}

runtime::TaskResult Session::finalize() {
    require_prepared("Nemotron 3 diarization finalize()");
    if (!stream_started_) throw std::runtime_error("Nemotron 3 diarization has no active stream");
    process_stream_windows(stream_scheduler_->finalize());
    const int64_t speakers = assets_->model_config.num_speakers;
    const int64_t expected_frames = stream_samples_ / kOutputHopSamples;
    const int64_t frames = std::min<int64_t>(
        expected_frames, static_cast<int64_t>(stream_probabilities_.size() / speakers));
    stream_probabilities_.resize(static_cast<size_t>(frames * speakers));
    runtime::TaskResult result;
    result.speaker_turns = decode_turns(
        stream_probabilities_, frames, decode_config(stream_request_.options), true);
    if (wants_frame_probabilities(stream_request_.options)) {
        result.output_artifacts.push_back(frame_probability_artifact(stream_probabilities_, frames, speakers));
    }
    stream_started_ = false;
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_nemotron_3_diar_loader() {
    runtime::SpecBackedVoiceModelConfig<Assets> config;
    config.family = kFamily;
    config.load_assets = [](const std::filesystem::path & model_path) {
        return load_assets(model_path, model_spec::default_package_spec_path(kFamily));
    };
    config.create_session = [](
                                const runtime::TaskSpec & task,
                                const runtime::SessionOptions & options,
                                std::shared_ptr<const Assets> assets,
                                std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::unique_ptr<runtime::IVoiceTaskSession>(
            std::make_unique<Session>(task, options, std::move(assets), std::move(contract)));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::nemotron_3_diar
