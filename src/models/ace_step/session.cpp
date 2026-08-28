#include "engine/models/ace_step/session.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/options.h"
#include "engine/models/ace_step/prompt_builder.h"
#include "engine/models/ace_step/repaint.h"
#include "engine/models/ace_step/request_parser.h"
#include "engine/models/ace_step/task_route.h"

#include <chrono>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace engine::models::ace_step {
namespace {

using Clock = std::chrono::steady_clock;

std::shared_ptr<const AceStepAssets> require_assets(std::shared_ptr<const AceStepAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("ACE-Step session requires assets");
    }
    return assets;
}

AceStepPlan make_direct_dit_plan(const AceStepRequest &request, const AceStepTaskRoute &route) {
    AceStepPlan plan;
    plan.caption = request.prompt;
    if (request.bpm.has_value() && *request.bpm > 0) {
        plan.metadata.bpm = request.bpm;
    }
    if (request.keyscale.has_value() && ace_step_metadata_text_is_provided(*request.keyscale)) {
        plan.metadata.keyscale = request.keyscale;
    }
    if (request.timesignature.has_value() && ace_step_metadata_text_is_provided(*request.timesignature)) {
        plan.metadata.timesignature = request.timesignature;
    }
    const bool locks_duration_to_source =
        (route.locks_duration_to_source || ace_step_request_uses_flow_edit_morph(request)) &&
        request.source_audio.has_value() &&
        request.audio_code_ids.empty();
    if (locks_duration_to_source) {
        const runtime::AudioBuffer &audio = *request.source_audio;
        plan.metadata.duration =
            static_cast<int64_t>(static_cast<float>(audio.samples.size() / static_cast<size_t>(audio.channels)) /
                                 static_cast<float>(audio.sample_rate));
    } else if (request.generation.duration_seconds > 0.0F) {
        plan.metadata.duration = static_cast<int64_t>(request.generation.duration_seconds);
    }
    plan.metadata.language = request.vocal_language;
    return plan;
}

void apply_request_audio_codes(AceStepPlan &plan, const AceStepRequest &request) {
    if (request.audio_code_ids.empty() || ace_step_request_uses_flow_edit_morph(request)) {
        return;
    }
    plan.audio_codes_text = request.audio_codes_text;
    plan.audio_code_ids = request.audio_code_ids;
    plan.frames_5hz = static_cast<int64_t>(plan.audio_code_ids.size());
}

bool ace_step_request_needs_lm_for_cot(const AceStepRequest &request) {
    return request.generation.use_cot_caption ||
           request.generation.use_cot_language ||
           request.generation.use_cot_metas;
}

AceStepGenerationOptions normalize_generation_options_for_model(
    const AceStepGenerationOptions & options,
    const AceStepAssets & assets) {
    AceStepGenerationOptions normalized = options;
    if (assets.config.diffusion.is_turbo && normalized.num_inference_steps > 8) {
        normalized.num_inference_steps = 8;
    }
    return normalized;
}

void validate_task_route_request(const AceStepRequest &request, const AceStepTaskRoute &route) {
    if (request.source_audio.has_value() &&
        (request.source_audio->channels <= 0 || request.source_audio->sample_rate <= 0)) {
        throw std::runtime_error("ACE-Step source audio metadata is invalid");
    }
    if (ace_step_route_has_missing_component(route)) {
        throw std::runtime_error("ACE-Step task '" + std::string(route.name) +
                                 "' requires missing component: " + std::string(route.missing_component));
    }
    if (ace_step_request_uses_flow_edit_morph(request)) {
        if (!request.source_audio.has_value()) {
            throw std::runtime_error("ACE-Step text2music flow_edit_morph requires source audio");
        }
        throw std::runtime_error("ACE-Step text2music flow_edit_morph requires missing component: flow-edit diffusion overlay");
    }
}

void validate_task_route_source_input(
    const AceStepRequest &request,
    const AceStepTaskRoute &route,
    const AceStepPlan &plan) {
    if (!ace_step_route_has_required_source_audio_input(route, request, plan)) {
        throw std::runtime_error("ACE-Step task '" + std::string(route.name) + "' requires source audio");
    }
}

assets::TensorStorageType parse_supported_weight_type(std::string_view option_name, std::string_view value) {
    const auto storage_type = assets::parse_tensor_storage_type(std::string(value));
    if (storage_type == assets::TensorStorageType::Native || storage_type == assets::TensorStorageType::F32 ||
        storage_type == assets::TensorStorageType::F16 || storage_type == assets::TensorStorageType::BF16 ||
        storage_type == assets::TensorStorageType::Q8_0) {
        return storage_type;
    }
    throw std::runtime_error(std::string(option_name) + " currently supports only native, f32, f16, bf16, and q8_0");
}

assets::TensorStorageType weight_type_from_options(const runtime::SessionOptions &options,
                                                   std::string_view specific_key) {
    const auto specific = options.options.find(std::string(specific_key));
    if (specific != options.options.end()) {
        return parse_supported_weight_type(specific_key, specific->second);
    }
    const auto shared = options.options.find("ace_step.weight_type");
    if (shared != options.options.end()) {
        return parse_supported_weight_type("ace_step.weight_type", shared->second);
    }
    return assets::TensorStorageType::Native;
}

assets::TensorStorageType dit_weight_type_from_options(const runtime::SessionOptions &options) {
    const auto storage_type = weight_type_from_options(options, "ace_step.dit_weight_type");
    if (storage_type == assets::TensorStorageType::F16) {
        return assets::TensorStorageType::BF16;
    }
    return storage_type;
}

bool mem_saver_from_options(const runtime::SessionOptions & options) {
    if (const auto value = runtime::find_option(options.options, {"ace_step.mem_saver", "mem_saver"})) {
        return runtime::parse_bool_option(*value, "ace_step.mem_saver");
    }
    return false;
}

bool rewrite_caption_requested(const std::unordered_map<std::string, std::string> & options) {
    const auto value = runtime::find_option(options, {"rewrite_caption"});
    return value.has_value() && runtime::parse_bool_option(*value, "rewrite_caption");
}

std::string plan_json(const AceStepPlan & plan, const AceStepRequest & request) {
    std::ostringstream out;
    out << "{"
        << "\"caption\":" << engine::io::json::stringify_string(plan.caption)
        << ",\"cot_caption\":" << engine::io::json::stringify_string(plan.cot_caption)
        << ",\"lyrics\":" << engine::io::json::stringify_string(request.lyrics);
    if (plan.metadata.bpm.has_value()) {
        out << ",\"bpm\":" << *plan.metadata.bpm;
    }
    if (plan.metadata.duration.has_value()) {
        out << ",\"duration_seconds\":" << *plan.metadata.duration;
    }
    if (plan.metadata.keyscale.has_value()) {
        out << ",\"keyscale\":" << engine::io::json::stringify_string(*plan.metadata.keyscale);
    }
    if (plan.metadata.timesignature.has_value()) {
        out << ",\"timesignature\":" << engine::io::json::stringify_string(*plan.metadata.timesignature);
    }
    if (plan.metadata.language.has_value()) {
        out << ",\"language\":" << engine::io::json::stringify_string(*plan.metadata.language);
    }
    if (plan.metadata.genres.has_value()) {
        out << ",\"genres\":" << engine::io::json::stringify_string(*plan.metadata.genres);
    }
    out << "}";
    return out.str();
}

}  // namespace

AceStepSession::AceStepSession(runtime::TaskSpec task, runtime::SessionOptions options,
                               std::shared_ptr<const AceStepAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      planner_weight_storage_type_(
          weight_type_from_options(RuntimeSessionBase::options(), "ace_step.planner_weight_type")),
      text_encoder_weight_storage_type_(
          weight_type_from_options(RuntimeSessionBase::options(), "ace_step.text_encoder_weight_type")),
      dit_weight_storage_type_(dit_weight_type_from_options(RuntimeSessionBase::options())),
      vae_weight_storage_type_(weight_type_from_options(RuntimeSessionBase::options(), "ace_step.vae_weight_type")),
      mem_saver_(mem_saver_from_options(RuntimeSessionBase::options())) {
    if (task_.task != runtime::VoiceTaskKind::AudioGeneration) {
        throw std::runtime_error("ACE-Step supports only the gen task");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("ACE-Step currently supports offline sessions only");
    }
}

std::string AceStepSession::family() const {
    return "ace_step";
}

runtime::VoiceTaskKind AceStepSession::task_kind() const {
    return task_.task;
}

runtime::RunMode AceStepSession::run_mode() const {
    return task_.mode;
}

void AceStepSession::prepare(const runtime::SessionPreparationRequest &request) {
    ensure_planner();
    if (rewrite_caption_requested(request.options)) {
        mark_prepared();
        return;
    }
    ensure_vae_decoder();
    ensure_pre_dit();
    pre_dit_->prepare_runtime();
    ensure_diffusion();
    mark_prepared();
}

runtime::TaskResult AceStepSession::run(const runtime::TaskRequest &request) {
    require_prepared("ACE-Step run()");
    const auto total_start = Clock::now();

    const auto parse_start = Clock::now();
    const AceStepRequest ace_request = ace_step_parse_request(request);
    const AceStepTaskRoute &route = ace_step_task_route(ace_request);
    validate_task_route_request(ace_request, route);
    engine::debug::timing_log_scalar("ace_step.session.parse_request_ms", engine::debug::elapsed_ms(parse_start, Clock::now()));
    const bool flow_edit_morph = ace_step_request_uses_flow_edit_morph(ace_request);
    if (ace_request.rewrite_caption) {
        if (flow_edit_morph || !route.uses_planner) {
            throw std::runtime_error("ACE-Step rewrite_caption requires a planner route");
        }
        const auto planner_ensure_start = Clock::now();
        ensure_planner();
        engine::debug::timing_log_scalar("ace_step.session.ensure_planner_ms",
                                         engine::debug::elapsed_ms(planner_ensure_start, Clock::now()));
        const auto planner_start = Clock::now();
        AceStepPlan plan = planner_->generate(ace_request, false);
        engine::debug::timing_log_scalar("ace_step.session.planner_generate_ms",
                                         engine::debug::elapsed_ms(planner_start, Clock::now()));
        const auto planner_release_start = Clock::now();
        planner_->release_graph_workspace();
        if (execution_context().backend_type() == core::BackendType::Metal) {
            planner_.reset();
        }
        engine::debug::timing_log_scalar("ace_step.session.planner_release.graph.workspace_ms",
                                         engine::debug::elapsed_ms(planner_release_start, Clock::now()));

        runtime::TaskResult result;
        result.text_output = runtime::Transcript{
            plan.caption,
            plan.metadata.language.value_or(ace_request.vocal_language),
        };
        result.output_artifacts.push_back(runtime::make_text_artifact(
            runtime::ArtifactKind::Custom,
            "ace_step_caption_plan",
            plan_json(plan, ace_request),
            {
                {"format", "json"},
                {"extension", "json"},
                {"mime", "application/json"},
            }));
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(total_start, Clock::now()));
        return result;
    }

    AceStepPlan plan;
    const bool has_request_audio_codes = !flow_edit_morph && !ace_request.audio_code_ids.empty();
    const bool use_planner =
        !flow_edit_morph &&
        route.uses_planner &&
        (ace_request.generation.thinking || ace_step_request_needs_lm_for_cot(ace_request));
    const bool generate_planner_audio_codes = ace_request.generation.thinking && !has_request_audio_codes;
    if (use_planner) {
        const auto planner_ensure_start = Clock::now();
        ensure_planner();
        engine::debug::timing_log_scalar("ace_step.session.ensure_planner_ms",
                                         engine::debug::elapsed_ms(planner_ensure_start, Clock::now()));
        const auto planner_start = Clock::now();
        plan = planner_->generate(ace_request, generate_planner_audio_codes);
        engine::debug::timing_log_scalar("ace_step.session.planner_generate_ms",
                                         engine::debug::elapsed_ms(planner_start, Clock::now()));

        const auto planner_release_start = Clock::now();
        planner_->release_graph_workspace();
        if (execution_context().backend_type() == core::BackendType::Metal) {
            planner_.reset();
        }
        engine::debug::timing_log_scalar("ace_step.session.planner_release.graph.workspace_ms",
                                         engine::debug::elapsed_ms(planner_release_start, Clock::now()));
    } else {
        engine::debug::timing_log_scalar("ace_step.session.ensure_planner_ms", 0.0);
        engine::debug::timing_log_scalar("ace_step.session.planner_generate_ms", 0.0);
        engine::debug::timing_log_scalar("ace_step.session.planner_release.graph.workspace_ms", 0.0);
        plan = make_direct_dit_plan(ace_request, route);
        if (execution_context().backend_type() == core::BackendType::Metal) {
            planner_.reset();
        }
    }
    apply_request_audio_codes(plan, ace_request);
    validate_task_route_source_input(ace_request, route, plan);

    const auto pre_dit_ensure_start = Clock::now();
    ensure_pre_dit();
    engine::debug::timing_log_scalar("ace_step.session.ensure_pre_dit_ms",
                                     engine::debug::elapsed_ms(pre_dit_ensure_start, Clock::now()));
    const auto pre_dit_start = Clock::now();
    AceStepPreDitInputs pre_dit = pre_dit_->prepare(ace_request, route, plan);
    engine::debug::timing_log_scalar("ace_step.session.pre_dit_prepare_ms", engine::debug::elapsed_ms(pre_dit_start, Clock::now()));

    const auto diffusion_ensure_start = Clock::now();
    ensure_diffusion();
    engine::debug::timing_log_scalar("ace_step.session.ensure_diffusion_ms",
                                     engine::debug::elapsed_ms(diffusion_ensure_start, Clock::now()));
    AceStepDiffusionConditioning conditioning = {};
    conditioning.request = ace_request;
    conditioning.pre_dit = std::move(pre_dit);
    if (mem_saver_) {
        const auto pre_dit_release_start = Clock::now();
        pre_dit_->release_runtime_graphs();
        engine::debug::timing_log_scalar(
            "ace_step.session.pre_dit_release.runtime_graphs_ms",
            engine::debug::elapsed_ms(pre_dit_release_start, Clock::now()));
    }
    const AceStepGenerationOptions generation_options =
        normalize_generation_options_for_model(ace_request.generation, *assets_);
    const auto diffusion_start = Clock::now();
    AceStepLatents latents = diffusion_->generate_latents(conditioning, generation_options);
    engine::debug::timing_log_scalar("ace_step.session.diffusion_generate_ms",
                                     engine::debug::elapsed_ms(diffusion_start, Clock::now()));

    const auto diffusion_release_start = Clock::now();
    diffusion_->release_graph_workspace();
    engine::debug::timing_log_scalar("ace_step.session.diffusion_release.graph.workspace_ms",
                                     engine::debug::elapsed_ms(diffusion_release_start, Clock::now()));

    const auto vae_ensure_start = Clock::now();
    ensure_vae_decoder();
    engine::debug::timing_log_scalar("ace_step.session.ensure_vae_decoder_ms",
                                     engine::debug::elapsed_ms(vae_ensure_start, Clock::now()));
    const auto vae_decode_start = Clock::now();
    runtime::AudioBuffer audio = vae_decoder_->decode(latents);
    engine::debug::timing_log_scalar("ace_step.session.vae_decode_ms", engine::debug::elapsed_ms(vae_decode_start, Clock::now()));
    if (mem_saver_) {
        const auto vae_release_start = Clock::now();
        vae_decoder_->release_runtime_graphs();
        engine::debug::timing_log_scalar(
            "ace_step.session.vae_release.runtime_graphs_ms",
            engine::debug::elapsed_ms(vae_release_start, Clock::now()));
    }

    const auto repaint_splice_start = Clock::now();
    if (route.task == AceStepTaskType::Repaint) {
        if (!conditioning.pre_dit.repaint_splice_audio.has_value()) {
            throw std::runtime_error("ACE-Step repaint waveform splice requires prepared source audio");
        }
        AceStepRepaintSpliceSource splice_source;
        splice_source.audio = std::move(*conditioning.pre_dit.repaint_splice_audio);
        splice_source.start_seconds = conditioning.pre_dit.repaint_splice_start_seconds;
        splice_source.end_seconds = conditioning.pre_dit.repaint_splice_end_seconds;
        ace_step_apply_repaint_waveform_splice(audio, splice_source, 0.05F * generation_options.repaint_injection_ratio);
    }
    engine::debug::timing_log_scalar(
        "ace_step.session.repaint_waveform_splice_ms",
        engine::debug::elapsed_ms(repaint_splice_start, Clock::now()));

    runtime::TaskResult result;
    result.audio_output = std::move(audio);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(total_start, Clock::now()));

    if (execution_context().backend_type() == core::BackendType::Metal) {
        vae_decoder_.reset();
        diffusion_.reset();
        pre_dit_.reset();
    }

    return result;
}

void AceStepSession::ensure_planner() {
    if (!planner_) {
        planner_ = std::make_unique<AceStepPlannerRuntime>(assets_, execution_context(), planner_weight_storage_type_,
                                                           256ull * 1024ull * 1024ull, 128ull * 1024ull * 1024ull,
                                                           AceStepPlannerRuntime::GenerationConfig{4096, 512, 4032});
    }
}

void AceStepSession::ensure_pre_dit() {
    ensure_dit_weights_runtime();
    if (!pre_dit_) {
        pre_dit_ = std::make_unique<AceStepPreDitRuntime>(execution_context(), assets_, dit_weights_runtime_,
                                                          dit_weight_storage_type_, vae_weight_storage_type_,
                                                          text_encoder_weight_storage_type_);
    }
}

void AceStepSession::ensure_diffusion() {
    ensure_dit_weights_runtime();
    if (!diffusion_) {
        diffusion_ = std::make_unique<AceStepDiffusionRuntime>(assets_, execution_context(), dit_weights_runtime_);
    }
}

void AceStepSession::ensure_dit_weights_runtime() {
    if (!dit_weights_runtime_) {
        dit_weights_runtime_ =
            std::make_shared<AceStepDitWeightsRuntime>(assets_, execution_context(), dit_weight_storage_type_);
    }
}

void AceStepSession::ensure_vae_decoder() {
    if (!vae_decoder_) {
        vae_decoder_ =
            std::make_shared<AceStepVAEDecoderRuntime>(assets_, execution_context(), vae_weight_storage_type_);
    }
}

}  // namespace engine::models::ace_step
