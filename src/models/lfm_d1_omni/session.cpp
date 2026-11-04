#include "engine/models/lfm_d1_omni/session.h"

#include "engine/models/lfm_d1_omni/assets.h"
#include "engine/models/lfm_d1_omni/audio.h"
#include "engine/models/lfm_d1_omni/questions.h"
#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/transformers/causal_decoder.h"
#include "engine/framework/modules/transformers/lfm_decoder_runtime.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <utility>

namespace engine::models::lfm_d1_omni {
namespace {

using io::json::Value;
constexpr const char * kFamily = "lfm_d1_omni";
constexpr const char * kName = "d1-Omni-2B";

class Session final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    Session(runtime::TaskSpec task, runtime::SessionOptions options, std::shared_ptr<const LfmD1OmniAssets> assets,
            std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)), contract_(std::move(contract)),
          weights_(execution_context().backend(), execution_context().backend_type(), "lfm_d1_omni.weights", 4 * 1024 * 1024) {
        if (task.task != runtime::VoiceTaskKind::AudioClassification || task.mode != runtime::RunMode::Offline) {
            throw runtime::InvalidRequestError("d1-Omni-2B supports offline audio decisions");
        }
        runtime::validate_spec_backed_session_options(options, *contract_, kFamily, kName);
        auto config = assets_->backbone.decoder;
        auto audio_config = assets_->audio;
        const auto preference = core::parse_attention_preference(
            runtime::find_option(options.options, {"lfm_d1_omni.attention"}).value_or("auto"), "lfm_d1_omni.attention");
        config.attention_lowering = core::resolve_flash_attention(execution_context().backend(), config.head_dim, preference)
            ? modules::GroupedQueryAttentionLowering::FlashGrouped : modules::GroupedQueryAttentionLowering::ManualRepeat;
        config.projection_precision = GGML_PREC_F32;
        config.attention_precision = GGML_PREC_F32;
        audio_config.decoder.attention_lowering = core::resolve_flash_attention(
            execution_context().backend(), audio_config.decoder.head_dim, preference)
            ? modules::GroupedQueryAttentionLowering::FlashGrouped : modules::GroupedQueryAttentionLowering::ManualRepeat;
        audio_config.decoder.projection_precision = GGML_PREC_F32;
        audio_config.decoder.attention_precision = GGML_PREC_F32;
        auto model = load_model_weights(*assets_, weights_);
        auto audio = load_audio_weights(*assets_, weights_);
        weights_.upload();
        frontend_ = std::make_unique<D1AudioFrontend>(audio_config);
        audio_ = std::make_unique<modules::Lfm3AudioEncoderRuntime>(std::move(audio_config), std::move(audio), execution_context());
        modules::LfmDecoderRuntimeWeights resolved{std::move(model.backbone), model.token_embedding, model.output_projection};
        decoder_ = std::make_unique<modules::LfmDecoderRuntime>(std::move(config), std::move(resolved), execution_context());
        questions_.codes = static_cast<int>(assets_->decision.codes);
        questions_.prompt = static_cast<int>(assets_->decision.prompt);
        questions_.numbers = assets_->decision.numbers;
        // These delimiter names belong to the model's protocol, not its decoder.
        for (const auto * name : {"<|startoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|audio|>"}) {
            if (!assets_->tokenizer->find_token_id(name)) { throw std::runtime_error(std::string("d1 tokenizer is missing ") + name); }

        }
        for (int32_t id = 0; id < assets_->backbone.vocab_size; ++id) {
            if (assets_->tokenizer->is_special_token_id(id)) { questions_.special_tokens.push_back(assets_->tokenizer->decode({id})); }
        }
        assets_->model->release_storage();
        assets_->mmproj->release_storage();
    }

    std::string family() const override { return kFamily; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }
    void prepare(const runtime::SessionPreparationRequest & request) override {
        if (!request.option_arrays.empty()) {
            throw runtime::InvalidRequestError("d1-Omni-2B does not accept list-valued request options");
        }
        try {
            runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, kName);
        } catch (const std::runtime_error & error) {
            throw runtime::InvalidRequestError(error.what());
        }
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("d1 audio decision");
        const auto started = std::chrono::steady_clock::now();
        if (!request.option_arrays.empty()) {
            throw runtime::InvalidRequestError("d1-Omni-2B does not accept list-valued request options");
        }
        try {
            runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, kName);
        } catch (const std::runtime_error & error) {
            throw runtime::InvalidRequestError(error.what());
        }
        if (!request.audio_input) { throw runtime::InvalidRequestError("d1-Omni-2B requires audio_input"); }
        if (request.voice || request.text_input || !request.input_artifacts.empty()) {
            throw runtime::InvalidRequestError("d1 audio decisions take audio and the questions option, not voice, text, or artifacts");
        }
        const auto raw = runtime::find_option(request.options, {"questions"});
        if (!raw) { throw runtime::InvalidRequestError("d1-Omni-2B requires questions as a JSON object"); }
        const auto named = parse_questions(*raw);
        const auto set_size = runtime::parse_positive_i64_option(request.options, {"set_size"}, assets_->decision.set_size);
        const auto score_readout = runtime::find_option(request.options, {"score_readout"}).value_or("median");
        if (score_readout != "median" && score_readout != "mean" && score_readout != "mode") {
            throw runtime::InvalidRequestError("score_readout must be median, mean, or mode");
        }
        const auto features = frontend_->extract(*request.audio_input, execution_context().config().threads);
        const auto audio = audio_->encode(features.values, features.frames);
        const auto & tokenizer = *assets_->tokenizer;
        std::string media = "<|vision_start|>";
        for (int64_t i = 0; i < audio.tokens; ++i) { media += "<|audio|>"; }
        media += "<|vision_end|>";
        const std::string prefix = "<|startoftext|><|im_start|>user\n" + media;
        Value::Object answers;
        int64_t total_tokens = 0;
        for (size_t begin = 0; begin < named.size(); begin += static_cast<size_t>(set_size)) {
            const size_t end = std::min(named.size(), begin + static_cast<size_t>(set_size));
            std::vector<QuestionSlots> rendered;
            for (size_t q = begin; q < end; ++q) { rendered.push_back(render_question(named[q].question, tokenizer, questions_)); }
            modules::LfmDecoderForward input;
            input.readout = modules::LfmDecoderReadout::Logits;
            input.token_ids = tokenizer.encode(prefix + rendered.front().suffix);
            const int64_t state = static_cast<int64_t>(input.token_ids.size() - rendered.front().suffix_ids.size());
            std::vector<int32_t> groups(static_cast<size_t>(state), -1);
            groups.resize(input.token_ids.size(), 1);
            std::vector<size_t> slot_counts;
            std::unordered_set<int32_t> seen;
            for (size_t q = 0; q < rendered.size(); ++q) {
                if (q) {
                    input.token_ids.insert(input.token_ids.end(), rendered[q].suffix_ids.begin(), rendered[q].suffix_ids.end());
                    groups.resize(input.token_ids.size(), static_cast<int32_t>(q + 1));
                }
                for (const auto & slot : rendered[q].slots) {
                    input.readout_positions.push_back(static_cast<int32_t>(static_cast<int64_t>(input.token_ids.size()) - 1 + slot.offset));
                    for (const auto & forms : slot.candidate_groups) {
                        for (const auto id : forms) { if (seen.insert(id).second) { input.output_tokens.push_back(id); } }
                    }
                }
                slot_counts.push_back(rendered[q].slots.size());
            }
            const auto steps = static_cast<int64_t>(input.token_ids.size());
            if (steps > assets_->backbone.context_length) { throw runtime::CapacityError("d1 questions and audio exceed model context"); }
            input.positions = modules::decoder_position_ids(steps);
            for (int64_t i = 0; i < steps; ++i) {
                if (input.token_ids[static_cast<size_t>(i)] == assets_->audio_token_id) { input.replacement_positions.push_back(static_cast<int32_t>(i)); }
            }
            if (input.replacement_positions.size() != static_cast<size_t>(audio.tokens)) {
                throw runtime::InvalidRequestError("d1 questions must not introduce audio placeholders");
            }
            input.replacement_tensor = &audio.projected;
            input.full_mask.resize(static_cast<size_t>(steps * steps));
            input.sliding_mask.resize(input.full_mask.size());
            for (int64_t q = 0; q < steps; ++q) {
                for (int64_t k = 0; k < steps; ++k) {
                    const size_t at = static_cast<size_t>(q * steps + k);
                    const bool visible = groups[static_cast<size_t>(q)] < 0 || groups[static_cast<size_t>(k)] < 0 ||
                        groups[static_cast<size_t>(q)] == groups[static_cast<size_t>(k)];
                    input.full_mask[at] = ggml_fp32_to_fp16(visible ? 0.0f : -INFINITY);
                    input.sliding_mask[at] = ggml_fp32_to_fp16(k <= q && q - k < assets_->backbone.sliding_window ? 0.0f : -INFINITY);
                }
            }
            const auto result = decoder_->forward(input);
            size_t row = 0;
            for (size_t q = 0; q < rendered.size(); ++q) {
                std::vector<SlotScores> scores;
                for (size_t s = 0; s < slot_counts[q]; ++s, ++row) {
                    const auto first = result.values.begin() + static_cast<std::ptrdiff_t>(row * input.output_tokens.size());
                    scores.push_back({input.output_tokens, std::vector<float>(first, first + input.output_tokens.size())});
                }
                answers.emplace(named[begin + q].name, decode_question(rendered[q], scores, assets_->calibration, score_readout));
            }
            total_tokens += steps;
        }
        runtime::TaskResult result;
        result.custom_schema_output = runtime::CustomSchemaOutput{"audio_decisions", Value::make_object({
            {"answers", Value::make_object(std::move(answers))},
            {"usage", Value::make_object({{"input_tokens", Value::make_number(total_tokens)}, {"output_tokens", Value::make_number(0)}})}})};
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const LfmD1OmniAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    core::BackendWeightStore weights_;
    std::unique_ptr<D1AudioFrontend> frontend_;
    std::unique_ptr<modules::Lfm3AudioEncoderRuntime> audio_;
    std::unique_ptr<modules::LfmDecoderRuntime> decoder_;
    QuestionConfig questions_;
};

} // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_lfm_d1_omni_loader() {
    runtime::SpecBackedVoiceModelConfig<LfmD1OmniAssets> config;
    config.family = kFamily;
    config.load_assets = [](const std::filesystem::path & path) { return load_lfm_d1_omni_assets(path); };
    config.package_load = [](const assets::ModelPackage & package) {
        std::vector<std::filesystem::path> main;
        for (const auto & file : package.files) {
            if (file.extension() == ".gguf" && file.filename().string().rfind("d1-Omni-2B-", 0) == 0) { main.push_back(file); }
        }
        if (main.size() != 1) { throw std::invalid_argument("d1 package must select one main GGUF"); }
        return runtime::PackageLoadTarget{kFamily, package.root / main.front(), {}, {}};
    };
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const LfmD1OmniAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<Session>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

} // namespace engine::models::lfm_d1_omni
