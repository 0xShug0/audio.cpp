#include "engine/models/ace_step/request_parser.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "engine/models/ace_step/task_route.h"
#include "engine/models/ace_step/prompt_builder.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/options.h"

namespace engine::models::ace_step {
namespace {

std::vector<std::string> option_string_list(const runtime::TaskRequest &request, std::string_view key) {
    const auto value = runtime::find_option(request.options, {key});
    if (!value.has_value() || value->empty()) {
        return {};
    }
    if (value->front() == '[') {
        const auto root = engine::io::json::parse(*value);
        std::vector<std::string> out;
        for (const auto & item : root.as_array()) {
            if (!item.is_null()) {
                out.push_back(item.as_string());
            }
        }
        return out;
    }
    return {*value};
}

std::vector<float> option_f32_list(const runtime::TaskRequest &request, std::string_view key) {
    const auto value = runtime::find_option(request.options, {key});
    if (!value.has_value()) {
        return {};
    }
    std::string raw = *value;
    raw.erase(std::remove_if(raw.begin(), raw.end(),
                             [](unsigned char ch) { return std::isspace(ch) != 0 || ch == '[' || ch == ']'; }),
              raw.end());
    if (raw.empty()) {
        return {};
    }
    std::vector<float> out;
    size_t start = 0;
    while (start < raw.size()) {
        const size_t end = raw.find(',', start);
        const std::string_view piece(raw.data() + static_cast<std::ptrdiff_t>(start),
                                     (end == std::string::npos ? raw.size() : end) - start);
        if (!piece.empty()) {
            out.push_back(std::stof(std::string(piece)));
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return out;
}

float clamp_repaint_strength(float strength) {
    return std::max(0.0F, std::min(1.0F, strength));
}

int64_t python_round_to_int(float value) {
    const double input = static_cast<double>(value);
    const double floor_value = std::floor(input);
    const double fraction = input - floor_value;
    if (fraction > 0.5) {
        return static_cast<int64_t>(floor_value + 1.0);
    }
    if (fraction < 0.5) {
        return static_cast<int64_t>(floor_value);
    }
    const int64_t floor_int = static_cast<int64_t>(floor_value);
    return (floor_int % 2 == 0) ? floor_int : floor_int + 1;
}

void apply_repaint_mode_strength(AceStepGenerationOptions & options, const runtime::TaskRequest & request) {
    const auto mode_value = runtime::find_option(request.options, {"repaint_mode"});
    const auto strength_value = runtime::find_option(request.options, {"repaint_strength"});
    if (!mode_value.has_value() && !strength_value.has_value()) {
        return;
    }

    const std::string mode = mode_value.value_or("balanced");
    const float strength = clamp_repaint_strength(
        strength_value.has_value() ? runtime::parse_float_option(request.options, {"repaint_strength"}).value() : 0.5F);
    if (mode == "aggressive") {
        options.repaint_injection_ratio = 0.0F;
        options.repaint_crossfade_frames = 0;
        return;
    }
    if (mode == "conservative") {
        options.repaint_injection_ratio = 1.0F;
        options.repaint_crossfade_frames = 25;
        return;
    }
    if (mode != "balanced") {
        throw std::runtime_error("ACE-Step repaint_mode must be balanced, conservative, or aggressive");
    }
    const float inv_strength = 1.0F - strength;
    options.repaint_injection_ratio = inv_strength;
    options.repaint_crossfade_frames = python_round_to_int(25.0F * inv_strength);
}

}  // namespace

AceStepRequest ace_step_parse_request(const runtime::TaskRequest &request) {
    if (!request.text_input.has_value()) {
        throw std::runtime_error("ACE-Step requires text_input");
    }
    AceStepRequest out;
    out.prompt = request.text_input->text;
    out.vocal_language = request.text_input->language.empty() ? "en" : request.text_input->language;
    out.source_audio = request.audio_input;
    if (const auto lyrics = runtime::find_option(request.options, {"lyrics"}); lyrics.has_value()) {
        out.lyrics = *lyrics;
    }
    if (const auto rewrite_caption = runtime::find_option(request.options, {"rewrite_caption"});
        rewrite_caption.has_value()) {
        out.rewrite_caption = runtime::parse_bool_option(*rewrite_caption, "rewrite_caption");
    }
    if (const auto negative = runtime::find_option(request.options, {"negative_prompt"}); negative.has_value()) {
        out.negative_prompt = *negative;
    }
    if (const auto instruction = runtime::find_option(request.options, {"instruction"}); instruction.has_value()) {
        out.instruction = *instruction;
    }
    const auto audio_codes = runtime::find_option(request.options, {"audio_codes"});
    const auto audio_code_string = runtime::find_option(request.options, {"audio_code_string"});
    const std::optional<std::string> audio_codes_value =
        audio_codes.has_value() ? audio_codes : audio_code_string;
    if (audio_codes_value.has_value() && !audio_codes_value->empty()) {
        AceStepPlan codes_plan = ace_step_parse_lm_output(*audio_codes_value);
        if (codes_plan.audio_code_ids.empty()) {
            throw std::runtime_error("ACE-Step audio_codes contains no audio code tokens");
        }
        out.audio_codes_text = std::move(codes_plan.audio_codes_text);
        out.audio_code_ids = std::move(codes_plan.audio_code_ids);
    }
    if (const auto track_name = runtime::find_option(request.options, {"track_name"}); track_name.has_value()) {
        out.track_name = *track_name;
    }
    out.complete_track_classes = option_string_list(request, "complete_track_classes");
    if (const auto route = runtime::find_option(request.options, {"route"}); route.has_value()) {
        out.task_type = *route;
    }
    out.task = ace_step_parse_task_type(out.task_type);
    out.task_type = std::string(ace_step_task_type_name(out.task));
    if (out.task == AceStepTaskType::Extract) {
        out.generation.thinking = false;
        out.generation.use_cot_metas = false;
        out.generation.use_cot_caption = false;
        out.generation.use_cot_language = false;
        out.generation.guidance_scale = 7.0F;
        out.generation.shift = 3.0F;
    }
    if (const auto bpm = runtime::find_option(request.options, {"bpm"}); bpm.has_value()) {
        out.bpm = runtime::parse_i64_option(request.options, {"bpm"}).value();
    }
    if (const auto keyscale = runtime::find_option(request.options, {"keyscale"}); keyscale.has_value()) {
        out.keyscale = *keyscale;
    }
    if (const auto timesignature = runtime::find_option(request.options, {"timesignature"}); timesignature.has_value()) {
        out.timesignature = *timesignature;
    }
    if (const auto chunk_mask_mode = runtime::find_option(request.options, {"chunk_mask_mode"});
        chunk_mask_mode.has_value()) {
        out.chunk_mask_mode = *chunk_mask_mode;
    }
    if (const auto repainting_start = runtime::parse_float_option(request.options, {"repainting_start"});
        repainting_start.has_value()) {
        out.repainting_start_seconds = *repainting_start >= 0.0F ?
            std::optional<float>(*repainting_start) : std::optional<float>{};
    }
    if (const auto repainting_end = runtime::parse_float_option(request.options, {"repainting_end"});
        repainting_end.has_value()) {
        out.repainting_end_seconds = *repainting_end >= 0.0F ?
            std::optional<float>(*repainting_end) : std::optional<float>{};
    }
    out.generation.duration_seconds = runtime::parse_float_option(request.options, {"duration_seconds"})
        .value_or(out.generation.duration_seconds);
    if (const auto thinking = runtime::find_option(request.options, {"thinking"}); thinking.has_value()) {
        out.generation.thinking = runtime::parse_bool_option(*thinking, "thinking");
    }
    if (const auto use_cot_metas = runtime::find_option(request.options, {"use_cot_metas"}); use_cot_metas.has_value()) {
        out.generation.use_cot_metas = runtime::parse_bool_option(*use_cot_metas, "use_cot_metas");
    }
    if (const auto use_cot_caption = runtime::find_option(request.options, {"use_cot_caption"});
        use_cot_caption.has_value()) {
        out.generation.use_cot_caption = runtime::parse_bool_option(*use_cot_caption, "use_cot_caption");
    }
    if (const auto use_cot_language = runtime::find_option(request.options, {"use_cot_language"});
        use_cot_language.has_value()) {
        out.generation.use_cot_language = runtime::parse_bool_option(*use_cot_language, "use_cot_language");
    }
    out.generation.num_inference_steps = runtime::parse_i64_option(request.options, {"num_inference_steps"})
        .value_or(out.generation.num_inference_steps);
    out.generation.guidance_scale = runtime::parse_float_option(request.options, {"guidance_scale"})
        .value_or(out.generation.guidance_scale);
    if (const auto use_adg = runtime::find_option(request.options, {"use_adg"}); use_adg.has_value()) {
        out.generation.use_adg = runtime::parse_bool_option(*use_adg, "use_adg");
    }
    out.generation.cfg_interval_start = runtime::parse_float_option(request.options, {"cfg_interval_start"})
        .value_or(out.generation.cfg_interval_start);
    out.generation.cfg_interval_end = runtime::parse_float_option(request.options, {"cfg_interval_end"})
        .value_or(out.generation.cfg_interval_end);
    out.generation.lm_temperature = runtime::parse_float_option(request.options, {"lm_temperature"})
        .value_or(out.generation.lm_temperature);
    out.generation.lm_cfg_scale = runtime::parse_float_option(request.options, {"lm_cfg_scale"})
        .value_or(out.generation.lm_cfg_scale);
    out.generation.lm_top_k = runtime::parse_i64_option(request.options, {"lm_top_k"})
        .value_or(out.generation.lm_top_k);
    out.generation.lm_top_p = runtime::parse_float_option(request.options, {"lm_top_p"})
        .value_or(out.generation.lm_top_p);
    out.generation.lm_repetition_penalty =
        runtime::parse_float_option(request.options, {"lm_repetition_penalty"})
            .value_or(out.generation.lm_repetition_penalty);
    out.generation.seed = runtime::parse_u32_option(request.options, {"seed"})
        .value_or(runtime::random_u32_seed());
    out.generation.shift = runtime::parse_float_option(request.options, {"shift"})
        .value_or(out.generation.shift);
    if (const auto infer_method = runtime::find_option(request.options, {"infer_method"}); infer_method.has_value()) {
        out.generation.infer_method = *infer_method;
    }
    out.generation.timesteps = option_f32_list(request, "timesteps");
    out.generation.audio_cover_strength =
        runtime::parse_float_option(request.options, {"audio_cover_strength"})
            .value_or(out.generation.audio_cover_strength);
    out.generation.cover_noise_strength =
        runtime::parse_float_option(request.options, {"cover_noise_strength"})
            .value_or(out.generation.cover_noise_strength);
    if (const auto retake_seed = runtime::find_option(request.options, {"retake_seed"}); retake_seed.has_value()) {
        const int64_t parsed = runtime::parse_i64_option(request.options, {"retake_seed"}).value();
        out.generation.retake_seed =
            parsed >= 0 ? std::optional<uint32_t>(static_cast<uint32_t>(parsed)) : std::nullopt;
    }
    out.generation.retake_variance = runtime::parse_float_option(request.options, {"retake_variance"})
        .value_or(out.generation.retake_variance);
    if (const auto sampler_mode = runtime::find_option(request.options, {"sampler_mode"}); sampler_mode.has_value()) {
        out.generation.sampler_mode = *sampler_mode;
    }
    out.generation.velocity_norm_threshold =
        runtime::parse_float_option(request.options, {"velocity_norm_threshold"})
            .value_or(out.generation.velocity_norm_threshold);
    out.generation.velocity_ema_factor = runtime::parse_float_option(request.options, {"velocity_ema_factor"})
        .value_or(out.generation.velocity_ema_factor);
    if (const auto dcw_enabled = runtime::find_option(request.options, {"dcw_enabled"}); dcw_enabled.has_value()) {
        out.generation.dcw_enabled = runtime::parse_bool_option(*dcw_enabled, "dcw_enabled");
    }
    if (const auto dcw_mode = runtime::find_option(request.options, {"dcw_mode"}); dcw_mode.has_value()) {
        out.generation.dcw_mode = *dcw_mode;
    }
    out.generation.dcw_scaler = runtime::parse_float_option(request.options, {"dcw_scaler"})
        .value_or(out.generation.dcw_scaler);
    out.generation.dcw_high_scaler = runtime::parse_float_option(request.options, {"dcw_high_scaler"})
        .value_or(out.generation.dcw_high_scaler);
    if (const auto dcw_wavelet = runtime::find_option(request.options, {"dcw_wavelet"}); dcw_wavelet.has_value()) {
        out.generation.dcw_wavelet = *dcw_wavelet;
    }
    apply_repaint_mode_strength(out.generation, request);
    out.generation.repaint_crossfade_frames =
        runtime::parse_i64_option(request.options, {"repaint_crossfade_frames"})
            .value_or(out.generation.repaint_crossfade_frames);
    out.generation.repaint_injection_ratio =
        runtime::parse_float_option(request.options, {"repaint_injection_ratio"})
            .value_or(out.generation.repaint_injection_ratio);
    if (const auto noise_file = runtime::find_option(request.options, {"noise_file"}); noise_file.has_value()) {
        out.generation.noise_file = *noise_file;
    }
    if (const auto flow_edit_morph = runtime::find_option(request.options, {"flow_edit_morph"});
        flow_edit_morph.has_value()) {
        out.generation.flow_edit_morph = runtime::parse_bool_option(*flow_edit_morph, "flow_edit_morph");
    }
    return out;
}

}  // namespace engine::models::ace_step
