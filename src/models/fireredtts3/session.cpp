#include "engine/models/fireredtts3/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/text/chunking.h"
#include "engine/models/fireredtts3/pipeline.h"

#include <stdexcept>
#include <utility>

namespace engine::models::fireredtts3 {
namespace {

constexpr const char * kFamily = "fireredtts3";
constexpr const char * kBaseName = "FireRedTTS3 Base";
constexpr const char * kInstructName = "FireRedTTS3 Instruct";
constexpr int64_t kDefaultTextChunkSize = 600;
constexpr size_t kDefaultReferenceCacheSlots = 4;

const char * variant_name(FireRedTTS3Variant variant) {
    return variant == FireRedTTS3Variant::Instruct ? kInstructName : kBaseName;
}

bool is_instruct_variant(FireRedTTS3Variant variant) {
    return variant == FireRedTTS3Variant::Instruct;
}

std::shared_ptr<const FireRedTTS3Assets> require_assets(std::shared_ptr<const FireRedTTS3Assets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("FireRedTTS3 session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("FireRedTTS3 session requires a model contract");
    }
    return contract;
}

std::string request_language(const runtime::TaskRequest & request) {
    if (const auto language = runtime::find_option(request.options, {"language"})) {
        return *language;
    }
    if (request.text_input.has_value() && !request.text_input->language.empty()) {
        return request.text_input->language;
    }
    return "Chinese";
}

FireRedTTS3BaseRequest make_base_request(
    const FireRedTTS3TextTokenizer & tokenizer,
    const runtime::TaskRequest & request) {
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("FireRedTTS3 Base requires text input");
    }
    if (!request.voice.has_value() ||
        !request.voice->speaker.has_value() ||
        !request.voice->speaker->audio.has_value()) {
        throw std::runtime_error("FireRedTTS3 Base voice clone requires reference audio");
    }
    FireRedTTS3BaseRequest out;
    out.language = request_language(request);
    out.reference_text = runtime::find_option(request.options, {"reference_text"}).value_or("");
    out.prompt_audio = *request.voice->speaker->audio;
    out.token_ids = tokenizer.encode_base_prompt(out.language, out.reference_text, request.text_input->text);
    out.seed = runtime::parse_u32_option(request.options, {"seed"}).value_or(out.seed);
    out.num_inference_steps = runtime::parse_positive_i64_option(
        request.options,
        {"num_inference_steps"},
        out.num_inference_steps);
    if (const auto cfg = runtime::parse_positive_finite_float_option(request.options, {"guidance_scale"})) {
        out.guidance_scale = *cfg;
    }
    if (const auto stop = runtime::parse_finite_float_option(request.options, {"stop_threshold"})) {
        if (*stop < 0.0F || *stop > 1.0F) {
            throw std::runtime_error("FireRedTTS3 Base stop_threshold must be in [0, 1]");
        }
        out.stop_threshold = *stop;
    }
    return out;
}

void apply_common_generation_options(
    const runtime::TaskRequest & request,
    uint32_t default_seed,
    float default_cfg,
    FireRedTTS3InstructRequest & out) {
    out.seed = runtime::parse_u32_option(request.options, {"seed"}).value_or(default_seed);
    out.num_inference_steps = runtime::parse_positive_i64_option(
        request.options,
        {"num_inference_steps"},
        out.num_inference_steps);
    out.guidance_scale = default_cfg;
    if (const auto cfg = runtime::parse_positive_finite_float_option(request.options, {"guidance_scale"})) {
        out.guidance_scale = *cfg;
    }
    if (const auto stop = runtime::parse_finite_float_option(request.options, {"stop_threshold"})) {
        if (*stop < 0.0F || *stop > 1.0F) {
            throw std::runtime_error("FireRedTTS3 Instruct stop_threshold must be in [0, 1]");
        }
        out.stop_threshold = *stop;
    }
}

FireRedTTS3InstructRequest make_instruct_request(const runtime::TaskRequest & request) {
    const std::string mode = runtime::find_option(request.options, {"template_name"}).value_or(
        request.audio_input.has_value() ? "acoustic_edit" : "instruct_tts");
    FireRedTTS3InstructRequest out;
    out.text = request.text_input.has_value() ? request.text_input->text : "";
    out.instruction = runtime::find_option(request.options, {"instruction"}).value_or("");
    if (mode == "instruct_tts") {
        if (out.text.empty()) {
            throw std::runtime_error("FireRedTTS3 Instruct clone requires text input");
        }
        if (!request.voice.has_value() ||
            !request.voice->speaker.has_value() ||
            !request.voice->speaker->audio.has_value()) {
            throw std::runtime_error("FireRedTTS3 Instruct clone requires reference audio");
        }
        out.task = FireRedTTS3InstructTask::Clone;
        out.prompt_audio = *request.voice->speaker->audio;
        out.instruction = runtime::find_option(request.options, {"reference_text"}).value_or(out.instruction);
        apply_common_generation_options(request, 1234, 2.0F, out);
        return out;
    }
    if (mode == "voice_design") {
        if (out.text.empty() || out.instruction.empty()) {
            throw std::runtime_error("FireRedTTS3 voice design requires text and instruction");
        }
        out.task = FireRedTTS3InstructTask::VoiceDesign;
        out.infer_text = true;
        out.text_do_sample = true;
        apply_common_generation_options(request, 2, 1.2F, out);
        return out;
    }
    if (mode == "semantic_edit") {
        if (!request.audio_input.has_value() || out.instruction.empty()) {
            throw std::runtime_error("FireRedTTS3 semantic edit requires input audio and instruction");
        }
        out.task = FireRedTTS3InstructTask::SemanticEdit;
        out.input_audio = *request.audio_input;
        out.infer_text = true;
        out.text_do_sample = false;
        apply_common_generation_options(request, 1234, 1.2F, out);
        return out;
    }
    if (mode == "acoustic_edit") {
        if (!request.audio_input.has_value() || out.instruction.empty()) {
            throw std::runtime_error("FireRedTTS3 acoustic edit requires input audio and instruction");
        }
        out.task = FireRedTTS3InstructTask::AcousticEdit;
        out.input_audio = *request.audio_input;
        apply_common_generation_options(request, 1234, 1.2F, out);
        return out;
    }
    throw std::runtime_error("unknown FireRedTTS3 Instruct mode: " + mode);
}

std::vector<runtime::TaskRequest> split_request(const runtime::TaskRequest & request) {
    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    if (text_chunk_size <= 0) {
        throw std::runtime_error("FireRedTTS3 text_chunk_size must be positive");
    }
    const auto text_chunk_mode =
        engine::text::parse_text_chunk_mode_override(request.options).value_or(engine::text::TextChunkMode::Default);
    auto chunks = runtime::chunk_text_request(request, text_chunk_size, text_chunk_mode);
    engine::debug::trace_log_scalar("fireredtts3.text_chunk_size", text_chunk_size);
    engine::debug::trace_log_scalar("fireredtts3.text_chunk_mode", engine::text::text_chunk_mode_name(text_chunk_mode));
    engine::debug::trace_log_scalar("fireredtts3.text_chunk_count", static_cast<int64_t>(chunks.size()));
    return chunks;
}

std::unique_ptr<runtime::IVoiceTaskSession> create_fireredtts3_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const FireRedTTS3Assets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    return std::make_unique<FireRedTTS3Session>(
        task,
        options,
        std::move(assets),
        std::move(contract));
}

}  // namespace

FireRedTTS3Session::FireRedTTS3Session(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const FireRedTTS3Assets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : runtime::RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))),
      tokenizer_(std::make_unique<FireRedTTS3TextTokenizer>(assets_)) {
    const bool is_instruct = is_instruct_variant(assets_->variant);
    const char * name = variant_name(assets_->variant);
    runtime::validate_spec_backed_session_options(options, *contract_, kFamily, name);
    if (is_instruct) {
        if (task_.task != runtime::VoiceTaskKind::Tts &&
            task_.task != runtime::VoiceTaskKind::VoiceCloning &&
            task_.task != runtime::VoiceTaskKind::VoiceDesign) {
            throw std::runtime_error("FireRedTTS3 Instruct supports tts, clone, and voice design tasks");
        }
    } else if (task_.task != runtime::VoiceTaskKind::VoiceCloning) {
        throw std::runtime_error("FireRedTTS3 Base supports the voice clone task");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("FireRedTTS3 supports offline sessions");
    }
    using T = engine::assets::TensorStorageType;
    const auto storage_type = runtime::parse_tensor_storage_option(
        options.options,
        "fireredtts3.weight_type",
        T::Native,
        {T::Native, T::F32, T::F16, T::BF16, T::Q8_0});
    const auto graph_arena_bytes = runtime::parse_size_mb_option(
        options.options,
        {"fireredtts3.graph_arena_mb"},
        1024ull * 1024ull * 1024ull);
    const auto helper_graph_arena_bytes = runtime::parse_size_mb_option(
        options.options,
        {"fireredtts3.helper_graph_arena_mb"},
        256ull * 1024ull * 1024ull);
    const auto weight_context_bytes = runtime::parse_size_mb_option(
        options.options,
        {"fireredtts3.weight_context_mb"},
        512ull * 1024ull * 1024ull);
    if (const auto mem_saver = runtime::find_option(options.options, {"fireredtts3.mem_saver"})) {
        mem_saver_ = runtime::parse_bool_option(*mem_saver, "fireredtts3.mem_saver");
    }
    const int64_t reference_cache_slots = runtime::parse_i64_option(
        options.options,
        {"fireredtts3.reference_cache_slots"})
        .value_or(static_cast<int64_t>(kDefaultReferenceCacheSlots));
    if (reference_cache_slots < 0) {
        throw std::runtime_error("fireredtts3.reference_cache_slots must be non-negative");
    }
    if (is_instruct) {
        instruct_runtime_ = std::make_unique<FireRedTTS3InstructRuntime>(
            assets_,
            execution_context(),
            graph_arena_bytes,
            helper_graph_arena_bytes,
            weight_context_bytes,
            storage_type,
            mem_saver_);
    } else {
        runtime_ = std::make_unique<FireRedTTS3BaseRuntime>(
            assets_,
            execution_context(),
            graph_arena_bytes,
            helper_graph_arena_bytes,
            weight_context_bytes,
            storage_type,
            static_cast<size_t>(reference_cache_slots),
            mem_saver_);
    }
}

FireRedTTS3Session::~FireRedTTS3Session() = default;

std::string FireRedTTS3Session::family() const {
    return kFamily;
}

runtime::VoiceTaskKind FireRedTTS3Session::task_kind() const {
    return task_.task;
}

runtime::RunMode FireRedTTS3Session::run_mode() const {
    return task_.mode;
}

void FireRedTTS3Session::prepare(const runtime::SessionPreparationRequest & request) {
    runtime::validate_spec_backed_request_options(
        request.options,
        *contract_,
        variant_name(assets_->variant));
    mark_prepared();
}

runtime::TaskResult FireRedTTS3Session::run(const runtime::TaskRequest & request) {
    const auto wall_start = std::chrono::steady_clock::now();
    const bool is_instruct = is_instruct_variant(assets_->variant);
    runtime::validate_spec_backed_request_options(
        request.options,
        *contract_,
        variant_name(assets_->variant));
    require_prepared(is_instruct ? "FireRedTTS3 Instruct run" : "FireRedTTS3 Base run");
    runtime::TaskResult result;
    runtime::AudioBuffer merged_audio;
    std::string generated_text;
    try {
        auto chunk_requests = split_request(request);
        for (size_t i = 0; i < chunk_requests.size(); ++i) {
            if (is_instruct) {
                auto parsed = make_instruct_request(chunk_requests[i]);
                if (i > 0) {
                    parsed.seed += static_cast<uint32_t>(i);
                }
                auto generated = instruct_runtime_->generate(parsed);
                runtime::append_audio_buffer(merged_audio, generated.audio);
                if (!generated.generated_text.empty()) {
                    generated_text += generated.generated_text;
                }
            } else {
                auto parsed = make_base_request(*tokenizer_, chunk_requests[i]);
                if (i > 0) {
                    parsed.seed += static_cast<uint32_t>(i);
                }
                runtime::append_audio_buffer(merged_audio, runtime_->generate(parsed));
            }
        }
    } catch (...) {
        if (mem_saver_) {
            if (is_instruct) {
                instruct_runtime_->release_graphs();
            } else {
                runtime_->release_graphs();
            }
        }
        throw;
    }
    if (mem_saver_) {
        if (is_instruct) {
            instruct_runtime_->release_graphs();
        } else {
            runtime_->release_graphs();
        }
    }
    result.audio_output = std::move(merged_audio);
    if (!generated_text.empty()) {
        result.text_output = runtime::Transcript{std::move(generated_text), request_language(request)};
    }
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_fireredtts3_loader() {
    runtime::SpecBackedVoiceModelConfig<FireRedTTS3Assets> config;
    config.family = kFamily;
    config.load_assets = load_fireredtts3_assets;
    config.create_session = create_fireredtts3_session;
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::fireredtts3
