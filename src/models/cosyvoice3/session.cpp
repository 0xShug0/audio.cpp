#include "engine/models/cosyvoice3/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/text/chunking.h"
#include "engine/models/cosyvoice3/ar.h"
#include "engine/models/cosyvoice3/flow.h"
#include "engine/models/cosyvoice3/frontend.h"
#include "engine/models/cosyvoice3/hift.h"

#include <chrono>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

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
    return runtime::find_option(request.options, {"instruction"}).value_or("");
}

std::unordered_map<std::string, std::string> validation_options(
    const std::unordered_map<std::string, std::string> & options) {
    auto out = options;
    out.erase("teacher_force_tokens");
    out.erase("teacher_force_source_random");
    return out;
}

std::vector<int32_t> read_teacher_force_tokens(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("CosyVoice3 teacher_force_tokens file cannot be opened: " + path);
    }
    std::vector<int32_t> out;
    std::string token;
    char ch = '\0';
    while (in.get(ch)) {
        if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '-') {
            token.push_back(ch);
        } else if (!token.empty()) {
            out.push_back(static_cast<int32_t>(std::stoi(token)));
            token.clear();
        }
    }
    if (!token.empty()) {
        out.push_back(static_cast<int32_t>(std::stoi(token)));
    }
    if (out.empty()) {
        throw std::runtime_error("CosyVoice3 teacher_force_tokens file contains no tokens: " + path);
    }
    return out;
}

std::vector<float> read_float_list(const std::string & path, const char * label) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error(std::string("CosyVoice3 ") + label + " file cannot be opened: " + path);
    }
    std::vector<float> out;
    std::string token;
    char ch = '\0';
    while (in.get(ch)) {
        if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '-' || ch == '+' || ch == '.' || ch == 'e' || ch == 'E') {
            token.push_back(ch);
        } else if (!token.empty()) {
            out.push_back(std::stof(token));
            token.clear();
        }
    }
    if (!token.empty()) {
        out.push_back(std::stof(token));
    }
    if (out.empty()) {
        throw std::runtime_error(std::string("CosyVoice3 ") + label + " file contains no values: " + path);
    }
    return out;
}

std::vector<runtime::TaskRequest> split_request(const runtime::TaskRequest & request) {
    const int64_t text_chunk_size =
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
    runtime::validate_spec_backed_session_options(options, *contract_, kFamily, kModelName);
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
    ar_ = std::make_unique<CosyVoice3ArRuntime>(
        assets_,
        execution_context(),
        graph_arena_bytes,
        weight_context_bytes,
        storage_type);
    flow_ = std::make_unique<CosyVoice3FlowRuntime>(
        assets_,
        execution_context(),
        graph_arena_bytes,
        weight_context_bytes,
        storage_type);
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
    runtime::validate_spec_backed_request_options(validation_options(request.options), *contract_, kModelName);
    mark_prepared();
}

runtime::TaskResult CosyVoice3Session::run(const runtime::TaskRequest & request) {
    const auto wall_start = std::chrono::steady_clock::now();
    runtime::validate_spec_backed_request_options(validation_options(request.options), *contract_, kModelName);
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
        CosyVoice3ArRequest ar_request;
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
        CosyVoice3ArOutput ar_output;
        if (const auto teacher_force_path = runtime::find_option(chunk.options, {"teacher_force_tokens"})) {
            ar_output.speech_tokens = read_teacher_force_tokens(*teacher_force_path);
        } else {
            ar_output = ar_->generate(ar_request);
        }
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
        std::vector<float> source_random_values;
        const std::vector<float> * source_random_ptr = nullptr;
        if (const auto random_path = runtime::find_option(chunk.options, {"teacher_force_source_random"})) {
            source_random_values = read_float_list(*random_path, "teacher_force_source_random");
            source_random_ptr = &source_random_values;
        }
        runtime::append_audio_buffer(
            merged_audio,
            hift_->synthesize(flow_output.mel, flow_output.frames, flow_request.seed, source_random_ptr));
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
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::cosyvoice3
