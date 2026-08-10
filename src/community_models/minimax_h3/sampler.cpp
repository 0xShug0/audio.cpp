#include "engine/community_models/minimax_h3/sampler.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/structural_modules.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::minimax_h3 {

namespace core = engine::core;
namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

namespace {

struct SamplerCoefficients {
    float state = 1.0F;
    float velocity = 0.0F;
    float denoised = 0.0F;
    float previous_denoised = 0.0F;
    float previous_velocity = 0.0F;
};

const char * sampler_name(MiniMaxH3SamplerMode mode) {
    switch (mode) {
        case MiniMaxH3SamplerMode::Euler:
            return "euler";
        case MiniMaxH3SamplerMode::ResMultistep:
            return "res_multistep";
        case MiniMaxH3SamplerMode::Dpmpp2m:
            return "dpmpp_2m";
        case MiniMaxH3SamplerMode::UniPC:
            return "unipc";
    }
    throw std::runtime_error("MiniMax-H3 sampler mode is invalid");
}

SamplerCoefficients euler_coefficients(float sigma, float sigma_delta) {
    static_cast<void>(sigma);
    return SamplerCoefficients{
        1.0F,
        sigma_delta,
        0.0F,
        0.0F,
        0.0F,
    };
}

SamplerCoefficients x0_euler_coefficients(float sigma, float sigma_delta) {
    if (sigma <= 0.0F) {
        return SamplerCoefficients{};
    }
    const float sigma_next = std::max(0.0F, sigma - sigma_delta);
    const float ratio = sigma_next / sigma;
    return SamplerCoefficients{
        ratio,
        0.0F,
        1.0F - ratio,
        0.0F,
        0.0F,
    };
}

SamplerCoefficients dpmpp2m_coefficients(
    float sigma,
    float sigma_delta,
    float previous_sigma,
    bool has_history) {
    const float sigma_next = std::max(0.0F, sigma - sigma_delta);
    if (!has_history || sigma_next <= 0.0F || previous_sigma <= sigma || sigma <= 0.0F) {
        return x0_euler_coefficients(sigma, sigma_delta);
    }
    const float h = std::log(sigma / sigma_next);
    const float h_last = std::log(previous_sigma / sigma);
    if (h <= 0.0F || h_last <= 0.0F) {
        return x0_euler_coefficients(sigma, sigma_delta);
    }
    const float r = h_last / h;
    const float blend = 1.0F - sigma_next / sigma;
    return SamplerCoefficients{
        sigma_next / sigma,
        0.0F,
        blend * (1.0F + 1.0F / (2.0F * r)),
        -blend / (2.0F * r),
        0.0F,
    };
}

SamplerCoefficients res_multistep_coefficients(
    float sigma,
    float sigma_delta,
    float previous_sigma,
    bool has_history) {
    const float sigma_next = std::max(0.0F, sigma - sigma_delta);
    if (!has_history || sigma_next <= 0.0F || previous_sigma <= sigma || sigma <= 0.0F) {
        return x0_euler_coefficients(sigma, sigma_delta);
    }
    const float h = std::log(sigma / sigma_next);
    const float h_last = std::log(previous_sigma / sigma);
    if (h <= 0.0F || h_last <= 0.0F) {
        return x0_euler_coefficients(sigma, sigma_delta);
    }
    const float c2 = -h_last / h;
    if (std::abs(c2) <= 1.0e-6F) {
        return x0_euler_coefficients(sigma, sigma_delta);
    }
    const float phi1 = (1.0F - std::exp(-h)) / h;
    const float phi2 = (phi1 - 1.0F) / (-h);
    const float b2 = phi2 / c2;
    const float b1 = phi1 - b2;
    return SamplerCoefficients{
        sigma_next / sigma,
        0.0F,
        h * b1,
        h * b2,
        0.0F,
    };
}

SamplerCoefficients unipc_coefficients(
    float sigma,
    float sigma_delta,
    bool has_history) {
    if (!has_history) {
        return euler_coefficients(sigma, sigma_delta);
    }
    static_cast<void>(sigma);
    return SamplerCoefficients{
        1.0F,
        1.5F * sigma_delta,
        0.0F,
        0.0F,
        -0.5F * sigma_delta,
    };
}

SamplerCoefficients sampler_coefficients(
    MiniMaxH3SamplerMode mode,
    float sigma,
    float sigma_delta,
    float previous_sigma,
    bool has_history) {
    switch (mode) {
        case MiniMaxH3SamplerMode::Euler:
            return euler_coefficients(sigma, sigma_delta);
        case MiniMaxH3SamplerMode::ResMultistep:
            return res_multistep_coefficients(sigma, sigma_delta, previous_sigma, has_history);
        case MiniMaxH3SamplerMode::Dpmpp2m:
            return dpmpp2m_coefficients(sigma, sigma_delta, previous_sigma, has_history);
        case MiniMaxH3SamplerMode::UniPC:
            return unipc_coefficients(sigma, sigma_delta, has_history);
    }
    throw std::runtime_error("MiniMax-H3 sampler mode is invalid");
}

core::TensorValue scaled(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & scalar) {
    return core::wrap_tensor(
        ggml_mul(ctx.ggml, input.tensor, scalar.tensor),
        input.shape,
        GGML_TYPE_F32);
}

core::TensorValue add_terms(
    core::ModuleBuildContext & ctx,
    const std::array<core::TensorValue, 5> & terms) {
    auto out = core::wrap_tensor(
        ggml_add(ctx.ggml, terms[0].tensor, terms[1].tensor),
        terms[0].shape,
        GGML_TYPE_F32);
    for (size_t i = 2; i < terms.size(); ++i) {
        out = core::wrap_tensor(
            ggml_add(ctx.ggml, out.tensor, terms[i].tensor),
            out.shape,
            GGML_TYPE_F32);
    }
    return out;
}

std::vector<float> zeros_for_shape(const core::TensorShape & shape) {
    return std::vector<float>(static_cast<size_t>(shape.num_elements()), 0.0F);
}

}  // namespace

MiniMaxH3SamplerGraph::MiniMaxH3SamplerGraph(
    core::ExecutionContext & execution,
    const MiniMaxH3SamplerGraphSpec & spec,
    MiniMaxH3SamplerMode mode)
    : execution_(execution),
      spec_(spec),
      mode_(mode) {
    ctx_ = ggml_init({32 * 1024 * 1024, nullptr, true});
    if (ctx_ == nullptr) {
        throw std::runtime_error("failed to initialize MiniMax-H3 sampler graph context");
    }
    input_ctx_ = ggml_init({32 * 1024 * 1024, nullptr, true});
    if (input_ctx_ == nullptr) {
        throw std::runtime_error("failed to initialize MiniMax-H3 sampler input context");
    }

    const auto build_start = Clock::now();

    core::ModuleBuildContext input_ctx{input_ctx_, "minimax_h3.sampler.inputs", execution_.backend_type()};
    primary_state_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, spec_.primary.packed_state_shape);
    secondary_state_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, spec_.secondary.packed_state_shape);
    primary_velocity_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, spec_.primary.prediction_shape);
    secondary_velocity_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, spec_.secondary.prediction_shape);
    previous_primary_denoised_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, spec_.primary.prediction_shape);
    previous_secondary_denoised_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, spec_.secondary.prediction_shape);
    previous_primary_velocity_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, spec_.primary.prediction_shape);
    previous_secondary_velocity_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, spec_.secondary.prediction_shape);
    primary_sigma_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
    secondary_sigma_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
    primary_state_coeff_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
    primary_velocity_coeff_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
    primary_denoised_coeff_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
    primary_previous_denoised_coeff_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
    primary_previous_velocity_coeff_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
    secondary_state_coeff_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
    secondary_velocity_coeff_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
    secondary_denoised_coeff_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
    secondary_previous_denoised_coeff_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
    secondary_previous_velocity_coeff_t_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));

    for (auto * input : {
             primary_state_t_.tensor,
             secondary_state_t_.tensor,
             primary_velocity_t_.tensor,
             secondary_velocity_t_.tensor,
             previous_primary_denoised_t_.tensor,
             previous_secondary_denoised_t_.tensor,
             previous_primary_velocity_t_.tensor,
             previous_secondary_velocity_t_.tensor,
             primary_sigma_t_.tensor,
             secondary_sigma_t_.tensor,
             primary_state_coeff_t_.tensor,
             primary_velocity_coeff_t_.tensor,
             primary_denoised_coeff_t_.tensor,
             primary_previous_denoised_coeff_t_.tensor,
             primary_previous_velocity_coeff_t_.tensor,
             secondary_state_coeff_t_.tensor,
             secondary_velocity_coeff_t_.tensor,
             secondary_denoised_coeff_t_.tensor,
             secondary_previous_denoised_coeff_t_.tensor,
             secondary_previous_velocity_coeff_t_.tensor}) {
        ggml_set_input(input);
    }

    core::ModuleBuildContext build_ctx{ctx_, "minimax_h3.sampler", execution_.backend_type()};
    auto current_primary = modules::SliceModule({0, spec_.primary.active_row_start, spec_.primary.prediction_shape.at(0)})
                               .build(build_ctx, primary_state_t_);
    auto current_secondary = modules::SliceModule({0, spec_.secondary.active_row_start, spec_.secondary.prediction_shape.at(0)})
                                 .build(build_ctx, secondary_state_t_);
    denoised_primary_t_ = core::wrap_tensor(
        ggml_add(build_ctx.ggml, current_primary.tensor, scaled(build_ctx, primary_velocity_t_, primary_sigma_t_).tensor),
        spec_.primary.prediction_shape,
        GGML_TYPE_F32);
    denoised_secondary_t_ = core::wrap_tensor(
        ggml_add(build_ctx.ggml, current_secondary.tensor, scaled(build_ctx, secondary_velocity_t_, secondary_sigma_t_).tensor),
        spec_.secondary.prediction_shape,
        GGML_TYPE_F32);
    next_primary_t_ = add_terms(
        build_ctx,
        {
            scaled(build_ctx, current_primary, primary_state_coeff_t_),
            scaled(build_ctx, primary_velocity_t_, primary_velocity_coeff_t_),
            scaled(build_ctx, denoised_primary_t_, primary_denoised_coeff_t_),
            scaled(build_ctx, previous_primary_denoised_t_, primary_previous_denoised_coeff_t_),
            scaled(build_ctx, previous_primary_velocity_t_, primary_previous_velocity_coeff_t_),
        });
    next_secondary_t_ = add_terms(
        build_ctx,
        {
            scaled(build_ctx, current_secondary, secondary_state_coeff_t_),
            scaled(build_ctx, secondary_velocity_t_, secondary_velocity_coeff_t_),
            scaled(build_ctx, denoised_secondary_t_, secondary_denoised_coeff_t_),
            scaled(build_ctx, previous_secondary_denoised_t_, secondary_previous_denoised_coeff_t_),
            scaled(build_ctx, previous_secondary_velocity_t_, secondary_previous_velocity_coeff_t_),
        });

    graph_ = ggml_new_graph_custom(ctx_, 4096, false);
    for (auto * output : {next_primary_t_.tensor, next_secondary_t_.tensor, denoised_primary_t_.tensor, denoised_secondary_t_.tensor}) {
        ggml_set_output(output);
        ggml_build_forward_expand(graph_, output);
    }
    input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_, execution_.backend());
    if (input_buffer_ == nullptr) {
        throw std::runtime_error("failed to allocate MiniMax-H3 sampler inputs");
    }
    core::write_tensor_f32(previous_primary_denoised_t_, zeros_for_shape(spec_.primary.prediction_shape));
    core::write_tensor_f32(previous_secondary_denoised_t_, zeros_for_shape(spec_.secondary.prediction_shape));
    core::write_tensor_f32(previous_primary_velocity_t_, zeros_for_shape(spec_.primary.prediction_shape));
    core::write_tensor_f32(previous_secondary_velocity_t_, zeros_for_shape(spec_.secondary.prediction_shape));

    gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
    if (gallocr_ == nullptr ||
        !ggml_gallocr_reserve(gallocr_, graph_) ||
        !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
        throw std::runtime_error("failed to allocate MiniMax-H3 sampler graph");
    }
    core::prepare_host_graph_plan(execution_, graph_, plan_);
    engine::debug::timing_log_scalar(
        std::string("minimax_h3.sampler.") + sampler_name(mode_) + ".graph_build_ms",
        engine::debug::elapsed_ms(build_start, Clock::now()));
}

MiniMaxH3SamplerGraph::~MiniMaxH3SamplerGraph() {
    plan_.reset();
    if (graph_ != nullptr) {
        core::release_backend_graph_resources(execution_.backend(), graph_);
    }
    if (gallocr_ != nullptr) {
        ggml_gallocr_free(gallocr_);
    }
    if (input_buffer_ != nullptr) {
        ggml_backend_buffer_free(input_buffer_);
    }
    if (input_ctx_ != nullptr) {
        ggml_free(input_ctx_);
    }
    if (ctx_ != nullptr) {
        ggml_free(ctx_);
    }
}

void MiniMaxH3SamplerGraph::run(const MiniMaxH3SamplerInput & input, MiniMaxH3SamplerOutput & output) {
    const SamplerCoefficients primary_coeff =
        sampler_coefficients(mode_, input.primary_sigma, input.primary_sigma_delta, previous_primary_sigma_, has_history_);
    const SamplerCoefficients secondary_coeff =
        sampler_coefficients(mode_, input.secondary_sigma, input.secondary_sigma_delta, previous_secondary_sigma_, has_history_);
    const auto copy_start = Clock::now();
    ggml_backend_tensor_copy(input.primary_state, primary_state_t_.tensor);
    ggml_backend_tensor_copy(input.secondary_state, secondary_state_t_.tensor);
    ggml_backend_tensor_copy(input.primary_velocity, primary_velocity_t_.tensor);
    ggml_backend_tensor_copy(input.secondary_velocity, secondary_velocity_t_.tensor);
    core::write_tensor_f32(primary_sigma_t_, &input.primary_sigma, 1);
    core::write_tensor_f32(secondary_sigma_t_, &input.secondary_sigma, 1);
    core::write_tensor_f32(primary_state_coeff_t_, &primary_coeff.state, 1);
    core::write_tensor_f32(primary_velocity_coeff_t_, &primary_coeff.velocity, 1);
    core::write_tensor_f32(primary_denoised_coeff_t_, &primary_coeff.denoised, 1);
    core::write_tensor_f32(primary_previous_denoised_coeff_t_, &primary_coeff.previous_denoised, 1);
    core::write_tensor_f32(primary_previous_velocity_coeff_t_, &primary_coeff.previous_velocity, 1);
    core::write_tensor_f32(secondary_state_coeff_t_, &secondary_coeff.state, 1);
    core::write_tensor_f32(secondary_velocity_coeff_t_, &secondary_coeff.velocity, 1);
    core::write_tensor_f32(secondary_denoised_coeff_t_, &secondary_coeff.denoised, 1);
    core::write_tensor_f32(secondary_previous_denoised_coeff_t_, &secondary_coeff.previous_denoised, 1);
    core::write_tensor_f32(secondary_previous_velocity_coeff_t_, &secondary_coeff.previous_velocity, 1);
    device_copy_ms_ += engine::debug::elapsed_ms(copy_start, Clock::now());

    const auto compute_start = Clock::now();
    core::set_backend_threads(execution_.backend(), 8);
    const std::string graph_label = std::string("minimax_h3.sampler.") + sampler_name(mode_);
    const ggml_status status = core::compute_graph(execution_, graph_, plan_, graph_label.c_str());
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("MiniMax-H3 sampler graph compute failed for ") + sampler_name(mode_));
    }
    ggml_backend_synchronize(execution_.backend());
    const double compute_ms = engine::debug::elapsed_ms(compute_start, Clock::now());
    graph_compute_ms_ += compute_ms;
    engine::debug::timing_log_scalar(std::string("minimax_h3.sampler.") + sampler_name(mode_) + ".graph_compute_ms", compute_ms);

    const auto history_start = Clock::now();
    ggml_backend_tensor_copy(denoised_primary_t_.tensor, previous_primary_denoised_t_.tensor);
    ggml_backend_tensor_copy(denoised_secondary_t_.tensor, previous_secondary_denoised_t_.tensor);
    ggml_backend_tensor_copy(primary_velocity_t_.tensor, previous_primary_velocity_t_.tensor);
    ggml_backend_tensor_copy(secondary_velocity_t_.tensor, previous_secondary_velocity_t_.tensor);
    previous_primary_sigma_ = input.primary_sigma;
    previous_secondary_sigma_ = input.secondary_sigma;
    has_history_ = true;
    history_update_ms_ += engine::debug::elapsed_ms(history_start, Clock::now());

    const auto read_start = Clock::now();
    core::read_tensor_f32_into(next_primary_t_.tensor, output.next_primary);
    core::read_tensor_f32_into(next_secondary_t_.tensor, output.next_secondary);
    output_read_ms_ += engine::debug::elapsed_ms(read_start, Clock::now());
}

double MiniMaxH3SamplerGraph::device_copy_ms() const {
    return device_copy_ms_;
}

double MiniMaxH3SamplerGraph::graph_compute_ms() const {
    return graph_compute_ms_;
}

double MiniMaxH3SamplerGraph::output_read_ms() const {
    return output_read_ms_;
}

double MiniMaxH3SamplerGraph::history_update_ms() const {
    return history_update_ms_;
}

}  // namespace engine::models::minimax_h3
