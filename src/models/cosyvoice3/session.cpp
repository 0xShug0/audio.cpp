#include "engine/models/cosyvoice3/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/rknn/engine.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/text/chunking.h"
#include "engine/models/cosyvoice3/ar.h"
#include "engine/models/cosyvoice3/flow.h"
#include "engine/models/cosyvoice3/frontend.h"
#include "engine/models/cosyvoice3/hift.h"
#include "engine/models/cosyvoice3/rknn_flow.h"

#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::cosyvoice3 {
namespace {

constexpr const char * kFamily = "cosyvoice3";
constexpr const char * kModelName = "CosyVoice3";
constexpr int64_t kDefaultTextChunkSize = 600;
constexpr size_t kDefaultReferenceCacheSlots = 4;

std::shared_ptr<const CosyVoice3Assets> require_assets(std::shared_ptr<const CosyVoice3Assets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("CosyVoice3 session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("CosyVoice3 session requires a model contract");
    }
    return contract;
}

std::string template_name(const runtime::TaskRequest & request) {
    return runtime::find_option(request.options, {"template_name"}).value_or("zero_shot");
}

std::string reference_text(const runtime::TaskRequest & request) {
    return runtime::find_option(request.options, {"reference_text"}).value_or("");
}

std::string instruction_text(const runtime::TaskRequest & request) {
    return runtime::find_option(request.options, {"instruct", "instruction"}).value_or("");
}

// The NPU sidecar takes a ladder of compiled engines in one option, comma-separated: which lengths
// get an engine is a deployment choice (each one is ~700 MB and covers one sequence length), and a
// list-typed session option would be a schema feature for one consumer.
std::vector<std::string> split_paths(const std::string & value) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= value.size()) {
        const auto end = value.find(',', start);
        const auto stop = end == std::string::npos ? value.size() : end;
        if (stop > start) {
            out.push_back(value.substr(start, stop - start));
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return out;
}

std::vector<runtime::TaskRequest> split_request(const runtime::TaskRequest & request) {    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    if (text_chunk_size <= 0) {
        throw std::runtime_error("CosyVoice3 text_chunk_size must be positive");
    }
    const auto text_chunk_mode =
        engine::text::parse_text_chunk_mode_override(request.options).value_or(engine::text::TextChunkMode::Default);
    return runtime::chunk_text_request(request, text_chunk_size, text_chunk_mode);
}

std::unique_ptr<runtime::IVoiceTaskSession> create_cosyvoice3_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const CosyVoice3Assets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    return std::make_unique<CosyVoice3Session>(
        task,
        options,
        std::move(assets),
        std::move(contract));
}

}  // namespace

CosyVoice3Session::CosyVoice3Session(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const CosyVoice3Assets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : runtime::RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))),
      tokenizer_(std::make_unique<CosyVoice3TextTokenizer>(assets_)) {
    auto validation_options = options;
    // This previously advertised option never selected convolution storage.
    validation_options.options.erase("cosyvoice3.conv_weight_type");
    // Older embedded contracts omit exec_plan; parse_exec_plan validates its value.
    if (contract_->session_option_keys.find("cosyvoice3.exec_plan") == contract_->session_option_keys.end()) {
        validation_options.options.erase("cosyvoice3.exec_plan");
    }
    runtime::validate_spec_backed_session_options(validation_options, *contract_, kFamily, kModelName);
    if (task_.task != runtime::VoiceTaskKind::Tts && task_.task != runtime::VoiceTaskKind::VoiceCloning) {
        throw std::runtime_error("CosyVoice3 supports tts and clone tasks");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("CosyVoice3 supports offline sessions");
    }
    using T = engine::assets::TensorStorageType;
    const auto storage_type = runtime::parse_tensor_storage_option(
        options.options,
        "cosyvoice3.weight_type",
        T::Native,
        {T::Native, T::F32, T::F16, T::BF16, T::Q8_0, T::Q4_0, T::Q4_K});
    const auto graph_arena_bytes = runtime::parse_size_mb_option(
        options.options,
        {"cosyvoice3.graph_arena_mb"},
        1024ull * 1024ull * 1024ull);
    const auto weight_context_bytes = runtime::parse_size_mb_option(
        options.options,
        {"cosyvoice3.weight_context_mb"},
        2048ull * 1024ull * 1024ull);
    if (const auto mem_saver = runtime::find_option(options.options, {"cosyvoice3.mem_saver"})) {
        mem_saver_ = runtime::parse_bool_option(*mem_saver, "cosyvoice3.mem_saver");
    }
    const int64_t reference_cache_slots = runtime::parse_i64_option(
        options.options,
        {"cosyvoice3.reference_cache_slots"})
        .value_or(static_cast<int64_t>(kDefaultReferenceCacheSlots));
    if (reference_cache_slots < 0) {
        throw std::runtime_error("cosyvoice3.reference_cache_slots must be non-negative");
    }

    frontend_ = std::make_unique<CosyVoice3Frontend>(
        assets_,
        execution_context(),
        graph_arena_bytes,
        weight_context_bytes,
        storage_type,
        static_cast<size_t>(reference_cache_slots));
    ar_ = std::make_unique<CosyVoice3Qwen2ARRuntime>(
        assets_,
        execution_context(),
        graph_arena_bytes,
        weight_context_bytes,
        storage_type);
    // The Rockchip NPU sidecar for the flow estimator: one compiled engine per sequence length,
    // loaded up front because an engine takes seconds to init and the ODE loop needs it per step.
    // The ladder travels as the component's main artifact, comma-separated, because the roles of
    // an exec_plan artifact block are a closed vocabulary while the ladder is not.
    const auto exec_plan = runtime::parse_exec_plan(
        options.options,
        "cosyvoice3.exec_plan",
        {{"flow", "ggml",
          {{"ggml", {options.backend.type}, {}, {}},
           {"rknn", {}, {"main"}, {}}}}});
    const auto & flow_plan = exec_plan.at("flow");
    std::vector<std::string> rknn_flow_engines = flow_plan.impl == "rknn"
        ? split_paths(flow_plan.artifacts.at("main"))
        : std::vector<std::string>{};
    for (auto & engine_path : rknn_flow_engines) {
        engine_path = engine::rknn::resolve_engine_path(engine_path, assets_->resources.model_directory());
    }
    flow_ = std::make_unique<CosyVoice3DiTFlowRuntime>(
        assets_,
        execution_context(),
        graph_arena_bytes,
        weight_context_bytes,
        storage_type,
        rknn_flow_engines);
    hift_ = std::make_unique<CosyVoice3HiftRuntime>(
        assets_,
        execution_context(),
        storage_type);
}

CosyVoice3Session::~CosyVoice3Session() = default;

std::string CosyVoice3Session::family() const {
    return kFamily;
}

runtime::VoiceTaskKind CosyVoice3Session::task_kind() const {
    return task_.task;
}

runtime::RunMode CosyVoice3Session::run_mode() const {
    return task_.mode;
}

void CosyVoice3Session::prepare(const runtime::SessionPreparationRequest & request) {
    auto validation_options = request.options;
    validation_options.erase("instruct");
    validation_options.erase("instruction");
    (void)runtime::find_option(request.options, {"instruct", "instruction"});
    runtime::validate_spec_backed_request_options(validation_options, *contract_, kModelName);
    mark_prepared();
}

runtime::TaskResult CosyVoice3Session::run(const runtime::TaskRequest & request) {
    const auto wall_start = std::chrono::steady_clock::now();
    auto validation_options = request.options;
    validation_options.erase("instruct");
    validation_options.erase("instruction");
    (void)runtime::find_option(request.options, {"instruct", "instruction"});
    runtime::validate_spec_backed_request_options(validation_options, *contract_, kModelName);
    require_prepared("CosyVoice3 run");
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("CosyVoice3 requires text input");
    }
    if (!request.voice.has_value() ||
        !request.voice->speaker.has_value() ||
        !request.voice->speaker->audio.has_value()) {
        throw std::runtime_error("CosyVoice3 requires reference audio");
    }

    runtime::AudioBuffer merged_audio;
    auto chunks = split_request(request);
    for (size_t index = 0; index < chunks.size(); ++index) {
        const auto & chunk = chunks[index];
        const auto mode = template_name(chunk);
        CosyVoice3TextTokens text_tokens;
        if (mode == "zero_shot") {
            text_tokens = tokenizer_->encode_zero_shot(chunk.text_input->text, reference_text(chunk));
        } else if (mode == "cross_lingual") {
            text_tokens = tokenizer_->encode_cross_lingual(chunk.text_input->text);
        } else if (mode == "instruct") {
            text_tokens = tokenizer_->encode_instruct(chunk.text_input->text, instruction_text(chunk));
        } else {
            throw std::runtime_error("unknown CosyVoice3 template_name: " + mode);
        }
        const auto & reference = frontend_->prepare_reference(*chunk.voice->speaker->audio);
        if (mem_saver_) {
            frontend_->release_graphs();
        }
        CosyVoice3ARRequest ar_request;
        ar_request.prompt_text_tokens = std::move(text_tokens.prompt);
        ar_request.target_text_tokens = std::move(text_tokens.target);
        if (mode == "zero_shot") {
            ar_request.prompt_speech_tokens = reference.speech_tokens;
        }
        ar_request.seed = runtime::parse_u32_option(chunk.options, {"seed"}).value_or(ar_request.seed);
        ar_request.top_k = runtime::parse_positive_i64_option(chunk.options, {"top_k"}, ar_request.top_k);
        ar_request.min_tokens = runtime::parse_i64_option(chunk.options, {"min_tokens"}).value_or(ar_request.min_tokens);
        ar_request.max_tokens = runtime::parse_i64_option(chunk.options, {"max_tokens"}).value_or(ar_request.max_tokens);
        if (index > 0) {
            ar_request.seed += static_cast<uint32_t>(index);
        }
        CosyVoice3AROutput ar_output = ar_->generate(ar_request);
        if (mem_saver_) {
            ar_->release_graphs();
        }

        CosyVoice3FlowRequest flow_request;
        flow_request.speech_tokens = std::move(ar_output.speech_tokens);
        flow_request.prompt_speech_tokens = reference.speech_tokens;
        flow_request.prompt_mel = reference.prompt_mel;
        flow_request.prompt_mel_frames = reference.prompt_mel_frames;
        flow_request.speaker_embedding = reference.speaker_embedding;
        flow_request.seed = ar_request.seed;
        flow_request.num_inference_steps = runtime::parse_positive_i64_option(
            chunk.options,
            {"num_inference_steps"},
            flow_request.num_inference_steps);
        CosyVoice3FlowOutput flow_output = flow_->generate(flow_request);
        if (mem_saver_) {
            flow_->release_graphs();
        }
        runtime::append_audio_buffer(
            merged_audio,
            hift_->synthesize(flow_output.mel, flow_output.frames, flow_request.seed));
        if (mem_saver_) {
            hift_->release_graphs();
        }
    }

    if (mem_saver_) {
        frontend_->release_graphs();
        ar_->release_graphs();
        flow_->release_graphs();
        hift_->release_graphs();
    }
    runtime::TaskResult result;
    result.audio_output = std::move(merged_audio);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_cosyvoice3_loader() {
    runtime::SpecBackedVoiceModelConfig<CosyVoice3Assets> config;
    config.family = kFamily;
    config.load_assets = load_cosyvoice3_assets;
    config.create_session = create_cosyvoice3_session;
    // The ladder's residency policy, stated by the family that owns it: the engines its plan
    // declares share one NPU aperture and evict to stay inside it, so a host consumer must not price
    // the set by its sum. The ceiling is the sidecar's (see rknn_flow.h).
    config.residency_cap_mib = rknn_aperture_budget_mib();
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::cosyvoice3
