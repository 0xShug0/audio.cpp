#include "engine/models/kokoro_tts/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/text/chunking.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include "engine/models/kokoro_tts/decoder.h"
#include "engine/models/kokoro_tts/frontend.h"
#include "engine/models/kokoro_tts/predictor.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <stdexcept>

namespace engine::models::kokoro_tts {

namespace {
using engine::debug::measure_ms;
constexpr int64_t kDefaultTextChunkSize = 240;
constexpr const char * kFamily = "kokoro_tts";
constexpr const char * kModelName = "Kokoro TTS";

}  // namespace

KokoroTTSSession::KokoroTTSSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const KokoroAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(std::move(task)),
      assets_(std::move(assets)),
      contract_(std::move(contract)) {
    if (!assets_ || !assets_->model_weights) {
        throw std::runtime_error("Kokoro TTS session requires loaded assets");
    }
    if (contract_ == nullptr) {
        throw std::runtime_error("Kokoro TTS session requires a model contract");
    }
    runtime::validate_spec_backed_session_options(RuntimeSessionBase::options(), *contract_, kFamily, kModelName);
    if (task_.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("Kokoro TTS only supports tts tasks");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Kokoro TTS only supports offline sessions");
    }
    const auto & session_options = RuntimeSessionBase::options().options;
    using T = engine::assets::TensorStorageType;
    matmul_weight_storage_type_ = runtime::parse_tensor_storage_option(
        session_options,
        "kokoro_tts.weight_type",
        matmul_weight_storage_type_,
        {T::Native, T::F32, T::F16, T::BF16, T::Q8_0});
    conv_weight_storage_type_ = runtime::parse_tensor_storage_option(
        session_options,
        "kokoro_tts.conv_weight_type",
        conv_weight_storage_type_,
        {T::Native, T::F32, T::F16});
    weight_context_bytes_ = runtime::parse_size_mb_option(
        session_options,
        {"kokoro_tts.weight_context_mb"},
        weight_context_bytes_);
    predictor_duration_graph_bytes_ = runtime::parse_size_mb_option(
        session_options,
        {"kokoro_tts.predictor_duration_graph_mb"},
        predictor_duration_graph_bytes_);
    predictor_text_graph_bytes_ = runtime::parse_size_mb_option(
        session_options,
        {"kokoro_tts.predictor_text_graph_mb"},
        predictor_text_graph_bytes_);
    predictor_tail_graph_bytes_ = runtime::parse_size_mb_option(
        session_options,
        {"kokoro_tts.predictor_tail_graph_mb"},
        predictor_tail_graph_bytes_);
    weights_ = load_kokoro_backend_weights(
        *assets_,
        execution_context().backend(),
        execution_context().backend_type(),
        matmul_weight_storage_type_,
        conv_weight_storage_type_,
        weight_context_bytes_);
    const auto graph_capacity_mode = runtime::resolve_graph_capacity_mode(
        RuntimeSessionBase::options(),
        runtime::GraphCapacityMode::Fixed,
        {"kokoro_tts.graph_capacity_mode"});
    if (graph_capacity_mode == runtime::GraphCapacityMode::Unsupported) {
        throw std::runtime_error("Kokoro TTS graph_capacity_mode=unsupported is not implemented");
    }
    graph_capacity_controller_ = runtime::GraphCapacityController(graph_capacity_mode);
    fixed_token_capacity_ = runtime::parse_positive_i64_option(
        session_options,
        {"kokoro_tts.max_input_tokens"},
        std::min<int64_t>(512, weights_->context_length));
    pre_tail_token_capacity_ = runtime::parse_i64_option(
        session_options,
        {"kokoro_tts.pre_tail_tokens"}).value_or(0);
    if (pre_tail_token_capacity_ < 0) {
        throw std::runtime_error("kokoro_tts.pre_tail_tokens must be non-negative");
    }
    rng_seed_ = runtime::random_u64_seed();
    if (fixed_token_capacity_ > weights_->context_length) {
        throw std::runtime_error("Kokoro fixed token capacity exceeds model context length");
    }
    if (pre_tail_token_capacity_ > weights_->context_length) {
        throw std::runtime_error("Kokoro pre-tail token capacity exceeds model context length");
    }
}

KokoroTTSSession::~KokoroTTSSession() = default;

std::string KokoroTTSSession::family() const {
    return kFamily;
}

runtime::VoiceTaskKind KokoroTTSSession::task_kind() const {
    return task_.task;
}

runtime::RunMode KokoroTTSSession::run_mode() const {
    return task_.mode;
}

runtime::MappedGraphCapacityAdapter KokoroTTSSession::make_graph_capacity_adapter() {
    return runtime::MappedGraphCapacityAdapter(
        fixed_token_capacity_,
        fixed_token_capacity_,
        [this](int64_t request_size) {
            if (request_size <= 0) {
                throw std::runtime_error("Kokoro graph capacity request size must be positive");
            }
            if (request_size > weights_->context_length) {
                throw std::runtime_error("Kokoro request exceeds model context length");
            }
            return request_size;
        },
        [this]() { return prepared_graph_capacities(); },
        [this](int64_t capacity) { prepare_graph_capacity(capacity); });
}

std::vector<int64_t> KokoroTTSSession::prepared_graph_capacities() const {
    std::vector<int64_t> capacities;
    if (prepared_predictor_ && prepared_session_capacity_ > 0) {
        capacities.push_back(prepared_session_capacity_);
    }
    return capacities;
}

KokoroTTSSession::DecoderCapacityContract KokoroTTSSession::make_decoder_capacity_contract(int64_t decoder_frame_capacity) const {
    if (decoder_frame_capacity <= 0) {
        throw std::runtime_error("Kokoro decoder frame capacity must be positive");
    }
    DecoderCapacityContract contract = {};
    contract.decoder_frame_capacity = decoder_frame_capacity;
    contract.conditioning_sample_capacity = contract.decoder_frame_capacity * 300;
    const int64_t pad = weights_->decoder.generator.gen_istft_n_fft / 2;
    contract.conditioning_frame_capacity =
        1 + (contract.conditioning_sample_capacity + 2 * pad - weights_->decoder.generator.gen_istft_n_fft) /
                weights_->decoder.generator.gen_istft_hop_size;
    if (contract.conditioning_sample_capacity <= 0 || contract.conditioning_frame_capacity <= 0) {
        throw std::runtime_error("Kokoro decoder capacity contract overflowed");
    }
    return contract;
}

void KokoroTTSSession::prepare_graph_capacity(int64_t capacity) {
    if (capacity <= 0) {
        throw std::runtime_error("Kokoro graph capacity must be positive");
    }
    if (capacity > weights_->context_length) {
        throw std::runtime_error("Kokoro graph capacity exceeds model context length");
    }
    if (prepared_predictor_ && prepared_session_capacity_ >= capacity) {
        return;
    }
    const int threads = std::max(1, execution_context().config().threads);
    const bool use_device_backend = !execution_context().uses_host_graph_plan();
    ggml_backend_t backend = execution_context().backend();
    const int64_t plbert_fixed_token_capacity = 0;
    const int64_t predictor_pre_tail_capacity = pre_tail_token_capacity_;
    prepared_predictor_.reset();
    prepared_session_capacity_ = 0;
    double build_ms = 0.0;
    kokoro_ggml::KokoroPredictorGraphConfig predictor_graph_config;
    predictor_graph_config.duration_graph_bytes = predictor_duration_graph_bytes_;
    predictor_graph_config.text_graph_bytes = predictor_text_graph_bytes_;
    predictor_graph_config.tail_graph_bytes = predictor_tail_graph_bytes_;
    build_ms = measure_ms([&]() {
        prepared_predictor_ = std::make_unique<kokoro_ggml::KokoroPredictorRuntime>(
            weights_,
            backend,
            threads,
            use_device_backend,
            plbert_fixed_token_capacity,
            predictor_pre_tail_capacity,
            predictor_graph_config);
    });
    prepared_session_capacity_ = capacity;
    engine::debug::timing_log_scalar("kokoro.prepare.predictor.graph.build_ms", build_ms);
}

void KokoroTTSSession::prepare_decoder_graph_capacity(int64_t capacity) {
    if (capacity <= 0) {
        throw std::runtime_error("Kokoro decoder graph capacity must be positive");
    }
    if (prepared_decoder_ && prepared_decoder_capacity_ == capacity) {
        return;
    }
    const int threads = std::max(1, execution_context().config().threads);
    const bool use_device_backend = !execution_context().uses_host_graph_plan();
    ggml_backend_t backend = execution_context().backend();
    const DecoderCapacityContract contract = make_decoder_capacity_contract(capacity);
    double build_ms = 0.0;
    kokoro_ggml::KokoroDecoderCapacityContract decoder_contract;
    decoder_contract.decoder_frames = contract.decoder_frame_capacity;
    decoder_contract.conditioning_frames = contract.conditioning_frame_capacity;
    if (prepared_decoder_) {
        build_ms = measure_ms([&]() {
            prepared_decoder_->prepare(decoder_contract);
        });
    } else {
        build_ms = measure_ms([&]() {
            prepared_decoder_ = std::make_unique<kokoro_ggml::KokoroDecoderRuntime>(
                weights_,
                backend,
                threads,
                use_device_backend,
                rng_seed_,
                decoder_contract);
        });
    }
    prepared_decoder_capacity_ = capacity;
    prepared_decoder_context_ = contract;
    engine::debug::timing_log_scalar("kokoro.prepare.decoder_runtime_build_ms", build_ms);
}

void KokoroTTSSession::prepare(const runtime::SessionPreparationRequest & request) {
    runtime::validate_spec_backed_request_options(request.options, *contract_, kModelName);
    if (const auto seed = runtime::parse_u64_option(request.options, {"seed"})) {
        if (rng_seed_ != *seed) {
            rng_seed_ = *seed;
            prepared_decoder_.reset();
            prepared_decoder_capacity_ = 0;
            prepared_decoder_context_ = {};
        }
    }
    auto adapter = make_graph_capacity_adapter();
    int64_t request_size = 0;
    if (request.text.has_value()) {
        const int64_t text_chunk_size =
            engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
        const auto text_chunks = engine::text::split_text_chunks(request.text->text, text_chunk_size);
        for (const auto & chunk : text_chunks) {
            runtime::SessionPreparationRequest chunk_request = request;
            chunk_request.text = runtime::Transcript{chunk, request.text->language};
            const auto frontend_state =
                resolve_kokoro_frontend_session_state(chunk_request.text, chunk_request.voice, *assets_);
            request_size = std::max(
                request_size,
                estimate_kokoro_request_tokens(chunk_request, frontend_state, *assets_));
        }
    }
    graph_capacity_controller_.ensure_prepared(adapter, request_size);
    mark_prepared();
}

runtime::TaskResult KokoroTTSSession::run(const runtime::TaskRequest & request) {
    require_prepared("Kokoro TTS run()");
    if (!request.text_input.has_value()) {
        throw std::runtime_error("Kokoro TTS run requires text_input");
    }
    runtime::validate_spec_backed_request_options(request.options, *contract_, kModelName);

    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    const auto chunk_requests = runtime::chunk_text_request(request, text_chunk_size);
    engine::debug::trace_log_scalar("kokoro.text_chunk_size", text_chunk_size);
    engine::debug::trace_log_scalar("kokoro.text_chunk_count", static_cast<int64_t>(chunk_requests.size()));
    double frontend_ms = 0.0;
    double inference_ms = 0.0;
    double predictor_ms = 0.0;
    double decoder_ms = 0.0;
    runtime::AudioBuffer merged_audio;
    for (const auto & chunk_request : chunk_requests) {
        const auto frontend_state =
            resolve_kokoro_frontend_session_state(chunk_request.text_input, chunk_request.voice, *assets_);
        const std::string cache_key =
            frontend_state.voice_id + ":" +
            frontend_state.language_code + ":" +
            std::to_string(frontend_state.speaking_rate) + ":" +
            std::to_string(chunk_request.text_input->text.size()) + ":" +
            chunk_request.text_input->text;
        KokoroSynthesisInput input;
        frontend_ms += measure_ms([&]() {
            if (!cache_key.empty() && cached_input_ && cache_key == cached_request_key_) {
                input = *cached_input_;
                return;
            }
            input = build_kokoro_synthesis_input(*chunk_request.text_input, frontend_state, *assets_);
            if (!cache_key.empty()) {
                cached_request_key_ = cache_key;
                cached_input_ = std::make_unique<KokoroSynthesisInput>(input);
            }
        });

        const auto inference_started = std::chrono::steady_clock::now();
        auto adapter = make_graph_capacity_adapter();
        const int64_t request_size = static_cast<int64_t>(input.input_ids.size());
        graph_capacity_controller_.ensure_prepared(adapter, request_size);
        const int64_t selected_capacity = graph_capacity_controller_.select_capacity_for_run(adapter, request_size);
        if (!prepared_predictor_ || prepared_session_capacity_ != selected_capacity) {
            throw std::runtime_error("Kokoro selected graph capacity was not prepared");
        }

        kokoro_ggml::PredictorOutputs predictor;
        predictor_ms += measure_ms([&]() {
            predictor = prepared_predictor_->predict(
                input.input_ids,
                input.style,
                input.speaking_rate);
        });
        const int64_t decoder_request_size = predictor.decoder_x_cols;
        if (predictor.decoder_x_on_backend && predictor.decoder_x_tensor == nullptr) {
            throw std::runtime_error("Kokoro predictor reported backend decoder features without a tensor");
        }
        if (decoder_request_size <= 0) {
            throw std::runtime_error("Kokoro predictor produced invalid decoder request size");
        }
        if (static_cast<int64_t>(predictor.f0_curve.size()) != decoder_request_size) {
            throw std::runtime_error("Kokoro predictor decoder and f0 frame counts diverged");
        }
        if (!prepared_decoder_ || prepared_decoder_capacity_ != decoder_request_size) {
            prepare_decoder_graph_capacity(decoder_request_size);
        }
        if (!prepared_decoder_ ||
            prepared_decoder_context_.decoder_frame_capacity <= 0 ||
            decoder_request_size != prepared_decoder_context_.decoder_frame_capacity) {
            throw std::runtime_error("Kokoro decoder runtime was not prepared");
        }
        std::vector<float> audio;
        decoder_ms += measure_ms([&]() {
            audio = prepared_decoder_->decode(
                predictor,
                input.style);
        });
        const auto inference_ended = std::chrono::steady_clock::now();
        inference_ms += std::chrono::duration<double, std::milli>(inference_ended - inference_started).count();
        runtime::append_audio_buffer(merged_audio, runtime::AudioBuffer{24000, 1, std::move(audio)});
    }

    runtime::TaskResult result;
    result.audio_output = std::move(merged_audio);
    const double wall_ms = frontend_ms + inference_ms;
    engine::debug::timing_log_scalar("kokoro.frontend_ms", frontend_ms);
    engine::debug::timing_log_scalar("kokoro.inference_ms", inference_ms);
    engine::debug::timing_log_scalar("kokoro.predictor_ms", predictor_ms);
    engine::debug::timing_log_scalar("kokoro.decoder_ms", decoder_ms);
    engine::debug::timing_log_scalar("session.wall_ms", wall_ms);
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_kokoro_tts_loader() {
    runtime::SpecBackedVoiceModelConfig<KokoroAssets> config;
    config.family = kFamily;
    config.load_assets = [](const std::filesystem::path & model_path) {
        return load_kokoro_assets(model_path);
    };
    config.create_session = [](
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options,
        std::shared_ptr<const KokoroAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        return std::make_unique<KokoroTTSSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::kokoro_tts
