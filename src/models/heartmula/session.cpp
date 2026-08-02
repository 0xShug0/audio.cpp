#include "engine/models/heartmula/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/text/chunking.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace engine::models::heartmula {
namespace {

using Clock = std::chrono::steady_clock;

constexpr const char * kFamily = "heartmula";

std::shared_ptr<const HeartMuLaAssets> require_assets(std::shared_ptr<const HeartMuLaAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("HeartMuLa session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("HeartMuLa session requires a model contract");
    }
    return contract;
}

runtime::SessionOptions normalize_session_options(runtime::SessionOptions options) {
    return runtime::apply_option_v1_compatibility(
        std::move(options),
        {
            {"heartmula.mula_weight_type", "heartmula.generator_weight_type"},
            {"heartmula.mula_weight_context_mb", "heartmula.generator_weight_context_mb"},
            {"heartmula.mula_constant_context_mb", "heartmula.generator_constant_context_mb"},
            {"heartmula.mula_backbone_prefill_graph_arena_mb", "heartmula.backbone_prefill_graph_arena_mb"},
            {"heartmula.mula_backbone_step_graph_arena_mb", "heartmula.backbone_step_graph_arena_mb"},
            {"heartmula.mula_decoder_prefill_graph_arena_mb", "heartmula.decoder_prefill_graph_arena_mb"},
            {"heartmula.mula_decoder_step_graph_arena_mb", "heartmula.decoder_step_graph_arena_mb"},
            {"heartmula.mula_frame_embedding_graph_arena_mb", "heartmula.frame_embedding_graph_arena_mb"},
            {"heartmula.codec_flow_estimator_graph_arena_mb", "heartmula.flow_estimator_graph_arena_mb"},
            {"heartmula.codec_conditioning_graph_arena_mb", "heartmula.conditioning_graph_arena_mb"},
            {"heartmula.codec_scalar_decoder_graph_arena_mb", "heartmula.scalar_decoder_graph_arena_mb"},
        },
        "HeartMuLa");
}

runtime::SessionOptions require_supported_session_options(
    runtime::SessionOptions options,
    const std::shared_ptr<const engine::model_spec::ModelContract> & contract) {
    options = normalize_session_options(std::move(options));
    const auto checked_contract = require_contract(contract);
    runtime::validate_spec_backed_session_options(options, *checked_contract, kFamily, "HeartMuLa");
    if (options.backend.threads <= 0) {
        throw std::runtime_error("HeartMuLa requires positive backend thread count");
    }
    if (const auto mem_saver = runtime::find_option(options.options, {"heartmula.mem_saver"})) {
        (void) runtime::parse_bool_option(*mem_saver, "heartmula.mem_saver");
    }
    return options;
}

HeartMuLaGenerationOptions generation_options_from_request(const runtime::TaskRequest & request) {
    HeartMuLaGenerationOptions options;
    if (const auto value = runtime::parse_positive_finite_float_option(request.options, {"duration_sec"})) {
        options.duration_seconds = *value;
    }
    if (const auto value = runtime::parse_positive_finite_float_option(request.options, {"temperature"})) {
        options.temperature = *value;
    }
    options.top_k = runtime::parse_positive_i64_option(request.options, {"top_k"}, options.top_k);
    if (const auto value = runtime::parse_positive_finite_float_option(request.options, {"guidance_scale"})) {
        options.guidance_scale = *value;
    }
    if (const auto value = runtime::parse_positive_finite_float_option(request.options, {"codec_duration_sec"})) {
        options.codec_duration = *value;
    }
    options.num_inference_steps = runtime::parse_positive_i64_option(
        request.options,
        {"num_inference_steps"},
        options.num_inference_steps);
    if (const auto value = runtime::parse_positive_finite_float_option(
            request.options,
            {"codec_guidance_scale"})) {
        options.codec_guidance_scale = *value;
    }
    if (const auto value = runtime::find_option(request.options, {"infinite_mode"})) {
        options.infinite_mode = runtime::parse_bool_option(*value, "infinite_mode");
    }
    if (const auto value = engine::text::parse_text_chunk_size_override(request.options)) {
        options.text_chunk_size = *value;
    }
    options.infinite_chunk_audio_length_ms = runtime::parse_positive_i64_option(
        request.options,
        {"infinite_chunk_audio_duration_ms"},
        options.infinite_chunk_audio_length_ms);
    return options;
}

runtime::TaskRequest normalize_request_options(runtime::TaskRequest request) {
    request.options = runtime::apply_option_v1_compatibility(
        std::move(request.options),
        {
            {"duration_seconds", "duration_sec"},
            {"codec_duration", "codec_duration_sec"},
            {"infinite_chunk_audio_length_ms", "infinite_chunk_audio_duration_ms"},
        },
        "HeartMuLa",
        "request");
    return request;
}

std::unique_ptr<runtime::IVoiceTaskSession> create_heartmula_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const HeartMuLaAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    return std::make_unique<HeartMuLaSession>(
        task,
        options,
        std::move(assets),
        std::move(contract));
}

std::string request_tags(const runtime::TaskRequest & request) {
    if (const auto tags = runtime::find_option(request.options, {"tags"})) {
        return *tags;
    }
    if (request.voice.has_value() && request.voice->style.has_value()) {
        const auto & style = *request.voice->style;
        if (const auto it = style.tags.find("tags"); it != style.tags.end()) {
            return it->second;
        }
        if (const auto it = style.tags.find("heartmula.tags"); it != style.tags.end()) {
            return it->second;
        }
    }
    return {};
}

std::string request_lyrics(const runtime::TaskRequest & request) {
    if (const auto lyrics = runtime::find_option(request.options, {"lyrics"})) {
        return *lyrics;
    }
    if (request.text_input.has_value()) {
        return request.text_input->text;
    }
    return {};
}

uint32_t request_seed(const runtime::TaskRequest & request) {
    return runtime::parse_u32_option(request.options, {"seed"}).value_or(1234U);
}

int64_t audio_duration_ms(const runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty()) {
        return 0;
    }
    const auto frames = static_cast<int64_t>(audio.samples.size()) / audio.channels;
    return (frames * 1000) / audio.sample_rate;
}

}  // namespace

HeartMuLaSession::HeartMuLaSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const HeartMuLaAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : runtime::RuntimeSessionBase(require_supported_session_options(std::move(options), contract)),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))),
      text_tokenizer_(assets_),
      mula_(
          assets_,
          RuntimeSessionBase::options().backend.type,
          RuntimeSessionBase::options().backend.device,
          RuntimeSessionBase::options().backend.threads,
          runtime::parse_size_mb_option(
              RuntimeSessionBase::options().options,
              {"heartmula.generator_weight_context_mb"},
              generator_weight_context_bytes_),
          runtime::parse_size_mb_option(
              RuntimeSessionBase::options().options,
              {"heartmula.generator_constant_context_mb"},
              generator_constant_context_bytes_),
          runtime::parse_size_mb_option(
              RuntimeSessionBase::options().options,
              {"heartmula.backbone_prefill_graph_arena_mb"},
              generator_backbone_prefill_graph_arena_bytes_),
          runtime::parse_size_mb_option(
              RuntimeSessionBase::options().options,
              {"heartmula.backbone_step_graph_arena_mb"},
              generator_backbone_step_graph_arena_bytes_),
          runtime::parse_size_mb_option(
              RuntimeSessionBase::options().options,
              {"heartmula.decoder_prefill_graph_arena_mb"},
              generator_decoder_prefill_graph_arena_bytes_),
          runtime::parse_size_mb_option(
              RuntimeSessionBase::options().options,
              {"heartmula.decoder_step_graph_arena_mb"},
              generator_decoder_step_graph_arena_bytes_),
          runtime::parse_size_mb_option(
              RuntimeSessionBase::options().options,
              {"heartmula.frame_embedding_graph_arena_mb"},
              generator_frame_embedding_graph_arena_bytes_),
          runtime::parse_tensor_storage_option(
              RuntimeSessionBase::options().options,
              "heartmula.generator_weight_type",
              "heartmula.weight_type",
              mula_weight_storage_type_,
              {
                  engine::assets::TensorStorageType::Native,
                  engine::assets::TensorStorageType::F32,
                  engine::assets::TensorStorageType::F16,
                  engine::assets::TensorStorageType::BF16,
                  engine::assets::TensorStorageType::Q8_0,
              })),
      codec_(
          assets_,
          execution_context(),
          runtime::parse_size_mb_option(
              RuntimeSessionBase::options().options,
              {"heartmula.codec_weight_context_mb"},
              codec_weight_context_bytes_),
          runtime::parse_size_mb_option(
              RuntimeSessionBase::options().options,
              {"heartmula.flow_estimator_graph_arena_mb"},
              codec_flow_estimator_graph_arena_bytes_),
          runtime::parse_size_mb_option(
              RuntimeSessionBase::options().options,
              {"heartmula.conditioning_graph_arena_mb"},
              codec_conditioning_graph_arena_bytes_),
          runtime::parse_size_mb_option(
              RuntimeSessionBase::options().options,
              {"heartmula.scalar_decoder_graph_arena_mb"},
              codec_scalar_decoder_graph_arena_bytes_),
          runtime::parse_tensor_storage_option(
              RuntimeSessionBase::options().options,
              "heartmula.codec_weight_type",
              "heartmula.weight_type",
              codec_weight_storage_type_,
              {
                  engine::assets::TensorStorageType::Native,
                  engine::assets::TensorStorageType::F32,
                  engine::assets::TensorStorageType::F16,
                  engine::assets::TensorStorageType::BF16,
                  engine::assets::TensorStorageType::Q8_0,
              })) {
    if (const auto mem_saver = runtime::find_option(RuntimeSessionBase::options().options, {"heartmula.mem_saver"})) {
        mem_saver_ = runtime::parse_bool_option(*mem_saver, "heartmula.mem_saver");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("HeartMuLa currently supports offline sessions");
    }
    if (task_.task != runtime::VoiceTaskKind::AudioGeneration) {
        throw std::runtime_error("HeartMuLa only supports the gen task");
    }
}

std::string HeartMuLaSession::family() const {
    return kFamily;
}

runtime::VoiceTaskKind HeartMuLaSession::task_kind() const {
    return task_.task;
}

runtime::RunMode HeartMuLaSession::run_mode() const {
    return task_.mode;
}

void HeartMuLaSession::prepare(const runtime::SessionPreparationRequest & request) {
    (void) request;
    mark_prepared();
}

runtime::TaskResult HeartMuLaSession::run(const runtime::TaskRequest & request) {
    require_prepared("HeartMuLa run");
    auto normalized_request = normalize_request_options(request);
    runtime::validate_spec_backed_request_options(normalized_request.options, *contract_, "HeartMuLa");
    const auto wall_start = Clock::now();
    const auto heartmula_request = make_request(normalized_request);
    const uint32_t seed = request_seed(normalized_request);
    if (heartmula_request.options.infinite_mode) {
        const auto chunks = engine::text::split_text_chunks(
            heartmula_request.lyrics,
            heartmula_request.options.text_chunk_size);
        if (chunks.empty()) {
            throw std::runtime_error("HeartMuLa infinite mode produced no text chunks");
        }
        engine::debug::trace_log_scalar("heartmula.infinite_mode", 1);
        engine::debug::trace_log_scalar(
            "heartmula.text_chunk_size",
            heartmula_request.options.text_chunk_size);
        engine::debug::trace_log_scalar("heartmula.infinite_text_chunk_count", static_cast<int64_t>(chunks.size()));
        engine::debug::trace_log_scalar(
            "heartmula.infinite_chunk_audio_length_ms",
            heartmula_request.options.infinite_chunk_audio_length_ms);

        runtime::AudioBuffer merged_audio;
        double ar_ms = 0.0;
        double codec_ms = 0.0;
        int64_t total_frames = 0;
        for (size_t index = 0; index < chunks.size(); ++index) {
            const int64_t remaining_ms =
                static_cast<int64_t>(heartmula_request.options.duration_seconds * 1000.0F) -
                audio_duration_ms(merged_audio);
            if (remaining_ms <= 0) {
                break;
            }
            const uint64_t chunk_seed = static_cast<uint64_t>(seed) + static_cast<uint64_t>(index);
            HeartMuLaPromptRequest chunk_request = heartmula_request;
            chunk_request.lyrics = chunks[index];
            chunk_request.options.infinite_mode = false;
            chunk_request.options.duration_seconds =
                static_cast<float>(std::min(heartmula_request.options.infinite_chunk_audio_length_ms, remaining_ms)) /
                1000.0F;
            engine::debug::trace_log_scalar("heartmula.infinite_chunk.index", static_cast<int64_t>(index));
            engine::debug::trace_log_scalar(
                "heartmula.infinite_chunk.duration_seconds",
                chunk_request.options.duration_seconds);

            const auto ar_start = Clock::now();
            const auto frames = generate_heartmula_frames(
                chunk_request,
                text_tokenizer_,
                mula_,
                chunk_seed);
            const auto ar_end = Clock::now();
            if (mem_saver_) {
                mula_.clear_graph_cache();
            } else {
                mula_.release_graph_workspaces();
            }
            const auto codec_start = Clock::now();
            auto decoded = codec_.detokenize_codes(
                frames.codes,
                frames.frames,
                frames.codebooks,
                chunk_request.options,
                chunk_seed,
                frames.codec_randn_philox_offset,
                frames.codec_randn_call_offset_blocks);
            const auto codec_end = Clock::now();
            ar_ms += engine::debug::elapsed_ms(ar_start, ar_end);
            codec_ms += engine::debug::elapsed_ms(codec_start, codec_end);
            total_frames += frames.frames;
            engine::debug::trace_log_scalar("heartmula.infinite_chunk.audio_frames", frames.frames);
            runtime::append_audio_buffer(
                merged_audio,
                runtime::AudioBuffer{
                    assets_->codec_config.sample_rate,
                    static_cast<int>(decoded.channels),
                    std::move(decoded.values),
                });
            if (mem_saver_) {
                codec_.clear_graph_cache();
            }
        }

        runtime::TaskResult result;
        result.audio_output = std::move(merged_audio);
        const auto wall_end = Clock::now();
        engine::debug::timing_log_scalar("heartmula.ar_ms", ar_ms);
        engine::debug::timing_log_scalar("heartmula.codec_ms", codec_ms);
        engine::debug::timing_log_scalar("heartmula.audio_frames", total_frames);
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, wall_end));
        return result;
    }

    const auto ar_start = Clock::now();
    const auto frames = generate_heartmula_frames(heartmula_request, text_tokenizer_, mula_, seed);
    const auto ar_end = Clock::now();
    if (mem_saver_) {
        const auto release_start = Clock::now();
        mula_.clear_graph_cache();
        engine::debug::timing_log_scalar(
            "heartmula.mula_release.runtime_graphs_ms",
            engine::debug::elapsed_ms(release_start, Clock::now()));
    } else {
        mula_.release_graph_workspaces();
    }
    const auto codec_start = Clock::now();
    auto decoded = codec_.detokenize_codes(
        frames.codes,
        frames.frames,
        frames.codebooks,
        heartmula_request.options,
        seed,
        frames.codec_randn_philox_offset,
        frames.codec_randn_call_offset_blocks);
    const auto codec_end = Clock::now();
    if (mem_saver_) {
        const auto release_start = Clock::now();
        codec_.clear_graph_cache();
        engine::debug::timing_log_scalar(
            "heartmula.codec_release.runtime_graphs_ms",
            engine::debug::elapsed_ms(release_start, Clock::now()));
    }

    runtime::TaskResult result;
    result.audio_output = runtime::AudioBuffer{
        assets_->codec_config.sample_rate,
        static_cast<int>(decoded.channels),
        std::move(decoded.values),
    };
    const auto wall_end = Clock::now();
    engine::debug::timing_log_scalar("heartmula.ar_ms", engine::debug::elapsed_ms(ar_start, ar_end));
    engine::debug::timing_log_scalar("heartmula.codec_ms", engine::debug::elapsed_ms(codec_start, codec_end));
    engine::debug::timing_log_scalar("heartmula.audio_frames", frames.frames);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, wall_end));
    return result;
}

HeartMuLaPromptRequest HeartMuLaSession::make_request(const runtime::TaskRequest & request) const {
    if (request.audio_input.has_value()) {
        throw std::runtime_error("HeartMuLa does not consume audio_input");
    }
    if (request.voice.has_value() && request.voice->speaker.has_value()) {
        throw std::runtime_error("HeartMuLa does not consume speaker references");
    }
    if (!request.input_artifacts.empty()) {
        throw std::runtime_error("HeartMuLa does not consume input artifacts");
    }
    HeartMuLaPromptRequest out;
    out.tags = request_tags(request);
    out.lyrics = request_lyrics(request);
    if (out.tags.empty()) {
        throw std::runtime_error("HeartMuLa requires non-empty tags");
    }
    if (out.lyrics.empty()) {
        throw std::runtime_error("HeartMuLa requires non-empty lyrics");
    }
    out.options = generation_options_from_request(request);
    return out;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_heartmula_loader() {
    runtime::SpecBackedVoiceModelConfig<HeartMuLaAssets> config;
    config.family = kFamily;
    config.load_assets = load_heartmula_assets;
    config.create_session = create_heartmula_session;
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::heartmula
