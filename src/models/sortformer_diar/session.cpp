#include "engine/models/sortformer_diar/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/options.h"
#include "engine/models/sortformer_diar/frontend.h"
#include "engine/models/sortformer_diar/postprocess.h"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::sortformer_diar {

namespace {
using engine::debug::measure_ms;

int64_t context_sample_capacity(
    const SortformerFixedContextContract & contract,
    const SortformerAssets & assets) {
    return static_cast<int64_t>(
        std::llround(contract.session_len_sec * static_cast<double>(assets.feature_config.sample_rate)));
}

engine::assets::TensorStorageType option_weight_type(
    const runtime::SessionOptions & options,
    const char * key,
    engine::assets::TensorStorageType default_value) {
    const auto it = options.options.find(key);
    if (it == options.options.end()) {
        return default_value;
    }
    return engine::assets::parse_tensor_storage_type(it->second);
}

void validate_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " currently supports only native, f32, f16, bf16, and q8_0");
}

void reject_unknown_sortformer_options(const runtime::SessionOptions & options) {
    for (const auto & [key, value] : options.options) {
        (void) value;
        if (key.rfind("sortformer_diar.", 0) != 0) {
            continue;
        }
        if (key == "sortformer_diar.graph_context_mb" ||
            key == "sortformer_diar.weight_context_mb" ||
            key == "sortformer_diar.weight_type" ||
            key == "sortformer_diar.matmul_weight_type" ||
            key == "sortformer_diar.conv_weight_type") {
            continue;
        }
        throw std::runtime_error("unknown Sortformer diar session option: " + key);
    }
}

runtime::GraphCapacityMode default_graph_capacity_mode(const core::ExecutionContext & execution_context) {
    return execution_context.uses_host_graph_plan()
        ? runtime::GraphCapacityMode::Tiered
        : runtime::GraphCapacityMode::Fixed;
}

}  // namespace

SortformerDiarSession::SortformerDiarSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const SortformerAssets> assets)
    : RuntimeSessionBase(options),
      task_(std::move(task)),
      assets_(std::move(assets)),
      default_postprocess_(parse_sortformer_postprocess_config(RuntimeSessionBase::options())),
      graph_context_bytes_(
          runtime::parse_size_mb_option(RuntimeSessionBase::options().options, {"sortformer_diar.graph_context_mb"}, 512ull * 1024ull * 1024ull)),
      weight_context_bytes_(
          runtime::parse_size_mb_option(RuntimeSessionBase::options().options, {"sortformer_diar.weight_context_mb"}, 128ull * 1024ull * 1024ull)),
      matmul_weight_storage_type_(option_weight_type(
          RuntimeSessionBase::options(),
          "sortformer_diar.matmul_weight_type",
          option_weight_type(RuntimeSessionBase::options(), "sortformer_diar.weight_type", engine::assets::TensorStorageType::F32))),
      conv_weight_storage_type_(option_weight_type(
          RuntimeSessionBase::options(),
          "sortformer_diar.conv_weight_type",
          option_weight_type(RuntimeSessionBase::options(), "sortformer_diar.weight_type", engine::assets::TensorStorageType::F32))) {
    if (!assets_) {
        throw std::runtime_error("Sortformer diar session requires assets");
    }
    reject_unknown_sortformer_options(RuntimeSessionBase::options());
    validate_weight_storage(matmul_weight_storage_type_, "sortformer_diar.matmul_weight_type");
    validate_weight_storage(conv_weight_storage_type_, "sortformer_diar.conv_weight_type");
    weights_ = load_sortformer_diar_weights(
        *assets_,
        execution_context().backend(),
        execution_context().backend_type(),
        matmul_weight_storage_type_,
        conv_weight_storage_type_,
        weight_context_bytes_);
    assets_->model_weights->release_storage();
    const auto graph_capacity_mode = runtime::resolve_graph_capacity_mode(
        RuntimeSessionBase::options(),
        default_graph_capacity_mode(execution_context()),
        {"offline_graph_capacity_mode", "graph_capacity_mode"});
    if (graph_capacity_mode == runtime::GraphCapacityMode::Unsupported) {
        throw std::runtime_error("Sortformer diar graph_capacity_mode=unsupported is not implemented");
    }
    graph_capacity_controller_ = runtime::GraphCapacityController(graph_capacity_mode);
    base_context_ = parse_sortformer_fixed_context_contract(RuntimeSessionBase::options(), *assets_);
}

SortformerDiarSession::~SortformerDiarSession() = default;

std::string SortformerDiarSession::family() const {
    return "sortformer_diar";
}

runtime::VoiceTaskKind SortformerDiarSession::task_kind() const {
    return task_.task;
}

runtime::RunMode SortformerDiarSession::run_mode() const {
    return task_.mode;
}

int64_t SortformerDiarSession::base_graph_capacity_samples() const {
    return context_sample_capacity(base_context_, *assets_);
}

runtime::MappedGraphCapacityAdapter SortformerDiarSession::make_graph_capacity_adapter() {
    return runtime::MappedGraphCapacityAdapter(
        base_graph_capacity_samples(),
        base_graph_capacity_samples(),
        [](int64_t request_size) {
            if (request_size <= 0) {
                throw std::runtime_error("Sortformer graph capacity request size must be positive");
            }
            return request_size;
        },
        [this]() { return prepared_graph_capacities(); },
        [this](int64_t capacity) { prepare_graph_capacity(capacity); });
}

std::vector<int64_t> SortformerDiarSession::prepared_graph_capacities() const {
    std::vector<int64_t> capacities;
    capacities.reserve(inference_graphs_.size());
    for (const auto & [capacity, graph] : inference_graphs_) {
        if (graph) {
            capacities.push_back(capacity);
        }
    }
    return capacities;
}

void SortformerDiarSession::prepare_graph_capacity(int64_t capacity) {
    if (capacity <= 0) {
        throw std::runtime_error("Sortformer graph capacity must be positive");
    }
    if (inference_graphs_.find(capacity) != inference_graphs_.end()) {
        return;
    }
    const SortformerFixedContextContract context =
        capacity == base_graph_capacity_samples()
            ? base_context_
            : make_sortformer_fixed_context_contract_for_samples(capacity, *assets_);
    auto graph = std::make_unique<SortformerInferenceGraph>();
    ensure_sortformer_inference_graph(
        graph,
        execution_context(),
        *assets_,
        *weights_,
        graph_context_bytes_,
        context.feature_frames,
        context.encoder_frames);
    prepared_contexts_[capacity] = context;
    inference_graphs_[capacity] = std::move(graph);
}

void SortformerDiarSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (request.audio.has_value()) {
        if (request.audio->sample_rate > 0 && request.audio->sample_rate != assets_->feature_config.sample_rate) {
            throw std::runtime_error("Sortformer diar prepare sample_rate mismatch");
        }
        if (request.audio->channels > 0 && request.audio->channels != 1) {
            throw std::runtime_error("Sortformer diar prepare currently requires mono audio");
        }
    }
    auto adapter = make_graph_capacity_adapter();
    const int64_t request_size = request.audio.has_value() ? request.audio->max_input_samples : 0;
    graph_capacity_controller_.ensure_prepared(adapter, request_size);
    mark_prepared();
}

runtime::TaskResult SortformerDiarSession::run(const runtime::TaskRequest & request) {
    require_prepared("Sortformer run()");
    if (task_.task != runtime::VoiceTaskKind::Diarization) {
        throw std::runtime_error("Sortformer diar session only supports --task diar");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Sortformer diar session only supports offline mode");
    }
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("Sortformer diar offline run requires audio_input");
    }
    auto config = default_postprocess_;
    if (!request.options.empty()) {
        runtime::SessionOptions merged = options();
        for (const auto & [key, value] : request.options) {
            merged.options[key] = value;
        }
        config = parse_sortformer_postprocess_config(merged);
        return run_offline_diarization(*request.audio_input, config);
    }
    return run_offline_diarization(*request.audio_input, config);
}

runtime::TaskResult SortformerDiarSession::run_offline_diarization(
    const runtime::AudioBuffer & audio,
    const SortformerPostprocessConfig & config) {
    SortformerRunTimings timings;
    const auto wall_started = std::chrono::steady_clock::now();

    SortformerFeatureBatch features;
    timings.frontend_ms = measure_ms([&]() {
        features = compute_sortformer_features(
            audio,
            *assets_,
            execution_context().config().threads,
            &timings);
    });

    const int64_t kernel = assets_->model_config.fc_encoder.subsampling_conv_kernel_size;
    const int64_t stride = assets_->model_config.fc_encoder.subsampling_conv_stride;
    const int64_t padding = (kernel - 1) / 2;
    const int64_t valid1 = sortformer_conv_valid_length(features.valid_frames, kernel, stride, padding);
    const int64_t valid2 = sortformer_conv_valid_length(valid1, kernel, stride, padding);
    const int64_t valid3 = sortformer_conv_valid_length(valid2, kernel, stride, padding);
    if (valid3 <= 0) {
        runtime::TaskResult result;
        const auto wall_ended = std::chrono::steady_clock::now();
        timings.wall_ms = std::chrono::duration<double, std::milli>(wall_ended - wall_started).count();
        emit_sortformer_timings(timings);
        return result;
    }
    int64_t selected_capacity = 0;
    auto adapter = make_graph_capacity_adapter();
    timings.graph_ensure_ms = measure_ms([&]() {
        selected_capacity = graph_capacity_controller_.ensure_prepared(
            adapter,
            static_cast<int64_t>(audio.samples.size()));
    });
    const auto context_it = prepared_contexts_.find(selected_capacity);
    const auto graph_it = inference_graphs_.find(selected_capacity);
    if (context_it == prepared_contexts_.end() || graph_it == inference_graphs_.end() || !graph_it->second) {
        throw std::runtime_error("Sortformer diar selected graph capacity was not prepared");
    }
    const SortformerFixedContextContract & prepared_context = context_it->second;
    if (features.frames > prepared_context.feature_frames || valid3 > prepared_context.encoder_frames) {
        throw std::runtime_error(
            "Sortformer diar input exceeds prepared session context of " +
            std::to_string(prepared_context.session_len_sec) + " seconds");
    }

    auto & graph = *graph_it->second;

    timings.graph_prepare_ms = measure_ms([&]() {
        std::vector<float> padded_input(
            static_cast<size_t>(prepared_context.feature_frames * assets_->feature_config.num_mel_bins),
            0.0f);
        std::copy(features.time_major.begin(), features.time_major.end(), padded_input.begin());
        core::write_tensor_f32(graph.input, padded_input);

        std::vector<int32_t> keep_mask;
        fill_sortformer_keep_mask(keep_mask, graph.mask1.shape.dims[1], std::min<int64_t>(graph.mask1.shape.dims[1], valid1));
        core::write_tensor_i32(graph.mask1, keep_mask);
        fill_sortformer_keep_mask(keep_mask, graph.mask2.shape.dims[1], std::min<int64_t>(graph.mask2.shape.dims[1], valid2));
        core::write_tensor_i32(graph.mask2, keep_mask);
        fill_sortformer_keep_mask(keep_mask, graph.encoder_keep_mask.shape.dims[1], valid3);
        core::write_tensor_i32(graph.encoder_keep_mask, keep_mask);
        std::vector<float> tf_mask;
        fill_sortformer_transformer_attention_mask(tf_mask, graph.encoder_frames, valid3);
        core::write_tensor_f32(graph.transformer_mask, tf_mask);
    });

    timings.encoder_ms = measure_ms([&]() {
        timings.encoder_compute_ms = measure_ms([&]() {
            core::set_backend_threads(execution_context().backend(), graph.compute_threads);
            const ggml_status status =
                core::compute_backend_graph(execution_context().backend(), graph.graph, graph.plan);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Sortformer diar graph compute failed");
            }
        });
        timings.encoder_readback_ms = measure_ms([&]() {
            core::read_tensor_f32_into(graph.output_probabilities.tensor, probabilities_);
        });
    });

    runtime::TaskResult result;
    const int64_t frame_step_samples =
        assets_->feature_config.hop_length * assets_->model_config.fc_encoder.subsampling_factor;
    timings.postprocess_ms = measure_ms([&]() {
        result.speaker_turns = decode_sortformer_speaker_turns(
            probabilities_,
            graph.encoder_frames,
            valid3,
            assets_->model_config.modules.num_speakers,
            frame_step_samples,
            config);
    });
    const auto wall_ended = std::chrono::steady_clock::now();
    timings.wall_ms = std::chrono::duration<double, std::milli>(wall_ended - wall_started).count();
    emit_sortformer_timings(timings);
    return result;
}

}  // namespace engine::models::sortformer_diar
