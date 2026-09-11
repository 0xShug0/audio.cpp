#include "engine/community_models/sortformer_diar/session.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <utility>

namespace engine::community_models::sortformer_diar {
namespace {

constexpr const char * kFamily = "sortformer_diar_v2";
constexpr int64_t kSampleRate = 16000;
constexpr int64_t kFrameHopSamples = 1280;
constexpr size_t kDefaultGraphArenaBytes = 1024ull * 1024ull * 1024ull;
constexpr size_t kDefaultWeightContextBytes = 1024ull * 1024ull * 1024ull;

std::shared_ptr<const SortformerV2Assets> require_assets(
    std::shared_ptr<const SortformerV2Assets> assets) {
    if (assets == nullptr) throw std::runtime_error("Sortformer v2 session requires assets");
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) throw std::runtime_error("Sortformer v2 session requires a model contract");
    return contract;
}

int64_t conv_valid(int64_t value, int64_t kernel, int64_t stride, int64_t padding) {
    return sortformer_v2_conv_valid_length(value, kernel, stride, padding);
}

int64_t valid_encoder_frames(const SortformerV2FeatureBatch & features, const SortformerV2Assets & assets) {
    const auto & encoder = assets.model_config.fc_encoder;
    const int64_t padding = (encoder.subsampling_conv_kernel_size - 1) / 2;
    auto value = features.valid_frames;
    for (int i = 0; i < 3; ++i) {
        value = conv_valid(value, encoder.subsampling_conv_kernel_size, encoder.subsampling_conv_stride, padding);
    }
    return value;
}

std::string speaker_id(int64_t speaker) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "SPEAKER_%02lld", static_cast<long long>(speaker));
    return buffer;
}

}  // namespace

SortformerV2Session::SortformerV2Session(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const SortformerV2Assets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))),
      geometry_(),
      scoring_(SortformerV2DiarScoringConfig::from_model(assets_->model_config)),
      graph_arena_bytes_(runtime::parse_size_mb_option(
          options.options, {"sortformer_diar_v2.graph_arena_mb"}, kDefaultGraphArenaBytes)),
      weight_context_bytes_(runtime::parse_size_mb_option(
          options.options, {"sortformer_diar_v2.weight_context_mb"}, kDefaultWeightContextBytes)),
      matmul_weight_storage_type_(runtime::parse_tensor_storage_option(
          options.options,
          "sortformer_diar_v2.matmul_weight_type",
          "sortformer_diar_v2.weight_type",
          assets::TensorStorageType::F32,
          {assets::TensorStorageType::Native, assets::TensorStorageType::F32,
           assets::TensorStorageType::F16, assets::TensorStorageType::BF16,
           assets::TensorStorageType::Q8_0})),
      conv_weight_storage_type_(runtime::parse_tensor_storage_option(
          options.options,
          "sortformer_diar_v2.conv_weight_type",
          "sortformer_diar_v2.weight_type",
          assets::TensorStorageType::F32,
          {assets::TensorStorageType::Native, assets::TensorStorageType::F32,
           assets::TensorStorageType::F16})) {
    if (task_.task != runtime::VoiceTaskKind::Diarization) {
        throw std::runtime_error("Sortformer v2 only supports diarization");
    }
    if (task_.mode != runtime::RunMode::Offline && task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Sortformer v2 supports offline and streaming modes");
    }
    runtime::validate_spec_backed_session_options(options, *contract_, kFamily, "Sortformer v2");

    const auto geometry_name = runtime::find_option(
        options.options, {"sortformer_diar_v2.geometry"}).value_or("model");
    geometry_ = geometry_name == "model"
        ? SortformerV2DiarGeometry::from_model(assets_->model_config)
        : SortformerV2DiarGeometry::preset(geometry_name);
    geometry_.validate(
        static_cast<int>(assets_->model_config.num_speakers),
        scoring_.sil_frames_per_spk,
        static_cast<int>(assets_->model_config.fc_encoder.max_position_embeddings));
    default_decode_config_ = decode_config(options.options);

    weights_ = load_sortformer_v2_weights(
        *assets_,
        execution_context().backend(),
        execution_context().backend_type(),
        matmul_weight_storage_type_,
        conv_weight_storage_type_,
        weight_context_bytes_);
    assets_->model_weights->release_storage();
    scheduler_ = std::make_unique<SortformerV2StreamScheduler>(
        assets_->feature_config,
        geometry_,
        assets_->model_config.fc_encoder.subsampling_factor);
}

SortformerV2Session::~SortformerV2Session() = default;

std::string SortformerV2Session::family() const { return kFamily; }
runtime::VoiceTaskKind SortformerV2Session::task_kind() const { return task_.task; }
runtime::RunMode SortformerV2Session::run_mode() const { return task_.mode; }

SortformerV2Session::DecodeConfig SortformerV2Session::decode_config(
    const std::unordered_map<std::string, std::string> & options) const {
    DecodeConfig config;
    const auto threshold = runtime::parse_finite_float_option(options, {"speaker_threshold"});
    if (threshold.has_value()) config.threshold = *threshold;
    if (!(config.threshold >= 0.0f && config.threshold <= 1.0f)) {
        throw std::runtime_error("speaker_threshold must be between 0 and 1");
    }
    const auto min_frames = runtime::parse_i64_option(options, {"speaker_min_frames"});
    if (min_frames.has_value()) config.min_frames = *min_frames;
    const auto pad_frames = runtime::parse_i64_option(options, {"speaker_pad_frames"});
    if (pad_frames.has_value()) config.pad_frames = *pad_frames;
    if (config.min_frames < 0 || config.pad_frames < 0) {
        throw std::runtime_error("speaker_min_frames and speaker_pad_frames must be non-negative");
    }
    return config;
}

void SortformerV2Session::prepare(const runtime::SessionPreparationRequest & request) {
    if (request.audio.has_value()) {
        if (request.audio->sample_rate > 0 && request.audio->sample_rate != kSampleRate) {
            throw std::runtime_error("Sortformer v2 requires 16 kHz audio");
        }
        if (request.audio->channels <= 0) {
            throw std::runtime_error("Sortformer v2 requires a positive channel count");
        }
    }
    mark_prepared();
    if (task_.mode == runtime::RunMode::Streaming) reset();
}

void SortformerV2Session::prepare_graph(
    const SortformerV2FeatureBatch & features,
    int64_t state_frames,
    int64_t valid_frames) {
    const auto & encoder = assets_->model_config.fc_encoder;
    const int64_t padding = (encoder.subsampling_conv_kernel_size - 1) / 2;
    const int64_t stage1 = conv_valid(features.frames, encoder.subsampling_conv_kernel_size, encoder.subsampling_conv_stride, padding);
    const int64_t stage2 = conv_valid(stage1, encoder.subsampling_conv_kernel_size, encoder.subsampling_conv_stride, padding);
    const int64_t chunk_capacity = conv_valid(stage2, encoder.subsampling_conv_kernel_size, encoder.subsampling_conv_stride, padding);
    if (valid_frames <= 0 || valid_frames > chunk_capacity) {
        throw std::runtime_error("Sortformer v2 frontend produced invalid encoder frame count");
    }
    ensure_sortformer_v2_inference_graph(
        graph_, execution_context(), *assets_, *weights_, graph_arena_bytes_,
        features.frames, state_frames + chunk_capacity, state_frames);
}

std::vector<float> SortformerV2Session::run_graph(
    const SortformerV2FeatureBatch & features,
    int64_t state_frames,
    int64_t valid_frames,
    const std::vector<float> * state,
    std::vector<float> * chunk_embeddings) {
    prepare_graph(features, state_frames, valid_frames);
    const auto & encoder = assets_->model_config.fc_encoder;
    const int64_t padding = (encoder.subsampling_conv_kernel_size - 1) / 2;
    const int64_t stage1_valid = conv_valid(features.valid_frames, encoder.subsampling_conv_kernel_size, encoder.subsampling_conv_stride, padding);
    const int64_t stage2_valid = conv_valid(stage1_valid, encoder.subsampling_conv_kernel_size, encoder.subsampling_conv_stride, padding);
    const int64_t graph_valid = state_frames + valid_frames;

    core::write_tensor_f32(graph_->input, features.time_major);
    std::vector<int32_t> mask;
    fill_sortformer_v2_keep_mask(mask, graph_->mask1.shape.dims[1], stage1_valid);
    core::write_tensor_i32(graph_->mask1, mask);
    fill_sortformer_v2_keep_mask(mask, graph_->mask2.shape.dims[1], stage2_valid);
    core::write_tensor_i32(graph_->mask2, mask);
    fill_sortformer_v2_keep_mask(mask, graph_->encoder_keep_mask.shape.dims[1], graph_valid);
    core::write_tensor_i32(graph_->encoder_keep_mask, mask);
    std::vector<float> transformer_mask;
    fill_sortformer_v2_transformer_attention_mask(
        transformer_mask, graph_->encoder_frames, graph_valid);
    core::write_tensor_f32(graph_->transformer_mask, transformer_mask);

    if (state_frames > 0) {
        if (state == nullptr || state->size() != static_cast<size_t>(state_frames * encoder.hidden_size)) {
            throw std::runtime_error("Sortformer v2 state tensor has an unexpected size");
        }
        core::write_tensor_f32(graph_->state_input, *state);
    }

    core::set_backend_threads(execution_context().backend(), graph_->compute_threads);
    if (core::compute_backend_graph(execution_context().backend(), graph_->graph, graph_->plan) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Sortformer v2 graph compute failed");
    }

    std::vector<float> probabilities;
    core::read_tensor_f32_into(graph_->output_probabilities.tensor, probabilities);
    const size_t probability_size = static_cast<size_t>(graph_valid * assets_->model_config.num_speakers);
    if (probabilities.size() < probability_size) {
        throw std::runtime_error("Sortformer v2 graph returned too few probabilities");
    }
    probabilities.resize(probability_size);
    if (chunk_embeddings != nullptr) {
        std::vector<float> embeddings;
        core::read_tensor_f32_into(graph_->chunk_embeddings.tensor, embeddings);
        const size_t embedding_size = static_cast<size_t>(valid_frames * encoder.hidden_size);
        if (embeddings.size() < embedding_size) {
            throw std::runtime_error("Sortformer v2 graph returned too few chunk embeddings");
        }
        chunk_embeddings->assign(embeddings.begin(), embeddings.begin() + static_cast<std::ptrdiff_t>(embedding_size));
    }
    return probabilities;
}

std::vector<runtime::SpeakerTurn> SortformerV2Session::decode_turns(
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
                        double confidence = 0.0;
                        for (int64_t f = start; f < end; ++f) {
                            confidence += probabilities[static_cast<size_t>(f * speakers + speaker)];
                        }
                        confidence /= std::max<int64_t>(1, end - start);
                        turns.push_back({
                            {padded_start * kFrameHopSamples, padded_end * kFrameHopSamples},
                            speaker_id(speaker),
                            static_cast<float>(confidence),
                            "",
                        });
                    }
                }
                start = -1;
            }
        }
    }
    std::sort(turns.begin(), turns.end(), [](const runtime::SpeakerTurn & lhs, const runtime::SpeakerTurn & rhs) {
        if (lhs.span.start_sample != rhs.span.start_sample) return lhs.span.start_sample < rhs.span.start_sample;
        return lhs.speaker_id < rhs.speaker_id;
    });
    return turns;
}

runtime::TaskResult SortformerV2Session::run(const runtime::TaskRequest & request) {
    require_prepared("Sortformer v2 run()");
    if (task_.mode != runtime::RunMode::Offline) throw std::runtime_error("Sortformer v2 run() requires offline mode");
    if (!request.audio_input.has_value()) throw std::runtime_error("Sortformer v2 run() requires audio_input");
    runtime::validate_spec_backed_request_options(request.options, *contract_, "Sortformer v2");
    if (request.audio_input->sample_rate != kSampleRate || request.audio_input->channels <= 0 ||
        request.audio_input->samples.size() % static_cast<size_t>(request.audio_input->channels) != 0) {
        throw std::runtime_error("Sortformer v2 run() received an invalid audio layout");
    }
    const auto features = compute_sortformer_v2_features(
        *request.audio_input, *assets_, execution_context().config().threads);
    const int64_t valid_frames = valid_encoder_frames(features, *assets_);
    if (valid_frames <= 0) return {};
    const auto probabilities = run_graph(features, 0, valid_frames, nullptr, nullptr);
    runtime::TaskResult result;
    result.speaker_turns = decode_turns(
        probabilities, valid_frames, decode_config(request.options), true);
    return result;
}

runtime::StreamingPolicy SortformerV2Session::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::AudioChunks;
    policy.output = runtime::StreamingOutputKind::PullEvents;
    policy.preferred_audio_chunk_samples = geometry_.chunk_len * kFrameHopSamples;
    policy.preferred_audio_chunk_seconds =
        static_cast<double>(policy.preferred_audio_chunk_samples) / kSampleRate;
    return policy;
}

void SortformerV2Session::start_stream(const runtime::TaskRequest & request) {
    require_prepared("Sortformer v2 start_stream()");
    if (task_.mode != runtime::RunMode::Streaming) throw std::runtime_error("Sortformer v2 start_stream() requires streaming mode");
    runtime::validate_spec_backed_request_options(request.options, *contract_, "Sortformer v2");
    reset();
    streaming_decode_config_ = decode_config(request.options);
}

void SortformerV2Session::set_stream_event_sink(runtime::StreamEventCallback sink) {
    stream_event_sink_ = std::move(sink);
}

void SortformerV2Session::reset() {
    require_prepared("Sortformer v2 reset()");
    if (task_.mode != runtime::RunMode::Streaming) throw std::runtime_error("Sortformer v2 reset() requires streaming mode");
    scheduler_->reset();
    aosc_ = std::make_unique<SortformerV2AoscState>(
        geometry_, scoring_, static_cast<int>(assets_->model_config.num_speakers),
        static_cast<int>(assets_->model_config.fc_encoder.hidden_size));
    birth_gate_ = std::make_unique<SortformerV2ChannelBirthGate>(static_cast<int>(assets_->model_config.num_speakers));
    probabilities_.clear();
    probability_frames_ = 0;
    emitted_stream_turns_ = 0;
    graph_.reset();
    stream_started_ = true;
    finalized_ = false;
}

runtime::StreamEvent SortformerV2Session::process_window(const SortformerV2StreamWindow & window) {
    const auto features = compute_sortformer_v2_stream_features(
        window.mono_samples, *assets_, execution_context().config().threads,
        window.mel_start == 0, window.mel_frames);
    const int64_t expected_valid_frames =
        window.encoder_frames + window.left_context_frames + window.right_context_frames;
    if (features.valid_frames != window.mel_frames ||
        valid_encoder_frames(features, *assets_) != expected_valid_frames) {
        throw std::runtime_error("Sortformer v2 streaming frontend frame geometry mismatch");
    }
    const int64_t valid_frames = expected_valid_frames;
    const int64_t left = std::min<int64_t>(window.left_context_frames, valid_frames);
    const int64_t right = std::min<int64_t>(window.right_context_frames, valid_frames - left);
    if (valid_frames <= left + right) return {};

    const int64_t state_frames = aosc_->spkcache_frames() + aosc_->fifo_frames();
    std::vector<float> state;
    state.reserve(static_cast<size_t>(state_frames * assets_->model_config.fc_encoder.hidden_size));
    state.insert(state.end(), aosc_->spkcache().begin(), aosc_->spkcache().end());
    state.insert(state.end(), aosc_->fifo().begin(), aosc_->fifo().end());
    std::vector<float> embeddings;
    const auto probabilities = run_graph(features, state_frames, valid_frames, &state, &embeddings);
    const auto emitted = aosc_->update(
        embeddings.data(), static_cast<int>(valid_frames), probabilities.data(),
        static_cast<int>(left), static_cast<int>(right));
    birth_gate_->append(emitted, probabilities_);
    probability_frames_ += static_cast<int64_t>(emitted.size() / assets_->model_config.num_speakers);
    auto turns = decode_turns(probabilities_, probability_frames_, streaming_decode_config_, false);
    runtime::StreamEvent event;
    if (turns.size() > emitted_stream_turns_) {
        event.speaker_turns.assign(
            turns.begin() + static_cast<std::ptrdiff_t>(emitted_stream_turns_), turns.end());
        emitted_stream_turns_ = turns.size();
    }
    return event;
}

runtime::StreamEvent SortformerV2Session::process_windows(
    const std::vector<SortformerV2StreamWindow> & windows) {
    runtime::StreamEvent result;
    for (const auto & window : windows) {
        auto event = process_window(window);
        result.speaker_turns.insert(
            result.speaker_turns.end(), event.speaker_turns.begin(), event.speaker_turns.end());
    }
    if (!result.speaker_turns.empty() && stream_event_sink_) {
        stream_event_sink_(result);
        return {};
    }
    return result;
}

runtime::StreamEvent SortformerV2Session::process_audio_chunk(const runtime::AudioChunk & chunk) {
    require_prepared("Sortformer v2 process_audio_chunk()");
    if (task_.mode != runtime::RunMode::Streaming || !stream_started_ || finalized_) {
        throw std::runtime_error("Sortformer v2 audio chunk received outside an active stream");
    }
    return process_windows(scheduler_->push_audio(chunk));
}

runtime::TaskResult SortformerV2Session::finalize_stream_result() {
    const int64_t subsampling = assets_->model_config.fc_encoder.subsampling_factor;
    const int64_t expected_frames =
        (scheduler_->produced_mel_frames() + subsampling - 1) / subsampling;
    if (probability_frames_ > expected_frames) {
        const size_t keep = static_cast<size_t>(expected_frames) * assets_->model_config.num_speakers;
        probabilities_.resize(keep);
        probability_frames_ = expected_frames;
    }
    if (probability_frames_ != expected_frames) {
        throw std::runtime_error("Sortformer v2 streaming probability timeline has unexpected length");
    }
    const auto turns = decode_turns(
        probabilities_, probability_frames_, streaming_decode_config_, true);
    runtime::TaskResult result;
    result.speaker_turns = turns;
    return result;
}

runtime::TaskResult SortformerV2Session::finalize() {
    require_prepared("Sortformer v2 finalize()");
    if (task_.mode != runtime::RunMode::Streaming || !stream_started_ || finalized_) {
        throw std::runtime_error("Sortformer v2 finalize() requires an active streaming session");
    }
    process_windows(scheduler_->finalize());
    auto result = finalize_stream_result();
    finalized_ = true;
    stream_started_ = false;
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_sortformer_diar_v2_loader() {
    runtime::SpecBackedVoiceModelConfig<SortformerV2Assets> config;
    config.family = kFamily;
    config.load_assets = [](const std::filesystem::path & model_path) {
        return load_sortformer_v2_assets(
            model_path, engine::model_spec::default_package_spec_path(kFamily));
    };
    config.create_session = [](
                                const runtime::TaskSpec & task,
                                const runtime::SessionOptions & options,
                                std::shared_ptr<const SortformerV2Assets> assets,
                                std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        return std::unique_ptr<runtime::IVoiceTaskSession>(
            std::make_unique<SortformerV2Session>(
                task, options, std::move(assets), std::move(contract)));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::community_models::sortformer_diar
