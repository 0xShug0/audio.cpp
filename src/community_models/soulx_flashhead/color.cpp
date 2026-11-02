#include "engine/community_models/soulx_flashhead/color.h"
#include "graph.h"
#include <array>
#include <tuple>

namespace engine::community_models::soulx_flashhead {
namespace {
class ColorGraph final : public Graph {
  public:
    ColorGraph(core::ExecutionContext & execution, int64_t height, int64_t width,
               int64_t motion_frames = 0, bool correction = true)
        : Graph(execution, "soulx_flashhead.color") {
        const int64_t pixels = height * width;
        auto source = input("video", {3, 33, height, width});
        const int64_t frames = 33 - motion_frames;
        // Each frame is corrected independently; leading context is never delivered.
        if (motion_frames > 0)
            source = modules::SliceModule({1, motion_frames, frames}).build(ctx, source);
        if (!correction) {
            source = core::ensure_backend_addressable_layout(ctx, source);
            motion = modules::SliceModule({1, frames - motion_frames, motion_frames}).build(ctx, source);
            motion = core::ensure_backend_addressable_layout(ctx, motion);
            finish(source, {motion});
            return;
        }
        auto reference = input("portrait", {3, 1, height, width});
        auto strength = input("strength", {1});
        source = modules::TransposeModule({{1, 2, 3, 0}, 4}).build(ctx, source);
        reference = modules::TransposeModule({{1, 2, 3, 0}, 4}).build(ctx, reference);
        source = modules::ReshapeModule({TensorShape::from_dims({frames, pixels, 3})})
                     .build(ctx, core::ensure_backend_addressable_layout(ctx, source));
        reference = modules::ReshapeModule({TensorShape::from_dims({1, pixels, 3})})
                        .build(ctx, core::ensure_backend_addressable_layout(ctx, reference));
        source = scalar(source, .5F, .5F);
        reference = scalar(reference, .5F, .5F);
        auto lab = to_lab(source, "source");
        auto ref_lab = to_lab(reference, "reference");
        const auto stats = [&](TensorValue x) {
            x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
            x = core::ensure_backend_addressable_layout(ctx, x);
            auto shape = x.shape;
            shape.dims[2] = 1;
            auto mean = core::wrap_tensor(ggml_mean(ctx.ggml, x.tensor), shape);
            auto delta = core::wrap_tensor(ggml_sub(ctx.ggml, x.tensor, mean.tensor), x.shape);
            auto square = ggml_sqr(ctx.ggml, delta.tensor);
            auto variance = ggml_mean(ctx.ggml, square);
            auto std = core::wrap_tensor(ggml_sqrt(ctx.ggml, variance), shape);
            mean = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, mean);
            std = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, std);
            return std::array<TensorValue, 2>{mean, std};
        };
        auto a = stats(lab), b = stats(ref_lab);
        const float std_floor = ggml_bf16_to_fp32(ggml_fp32_to_bf16(1.e-8F));
        auto threshold = scalar(a[1], -1, std_floor);
        auto mask = core::wrap_tensor(ggml_step(ctx.ggml, threshold.tensor), threshold.shape);
        auto safe_std = select(mask, scalar(a[1], 0, 1), a[1]);
        auto ratio = core::wrap_tensor(
            ggml_div(ctx.ggml, ggml_repeat(ctx.ggml, b[1].tensor, safe_std.tensor), safe_std.tensor), safe_std.shape);
        auto corrected = core::wrap_tensor(ggml_sub(ctx.ggml, lab.tensor, a[0].tensor), lab.shape);
        corrected = core::wrap_tensor(ggml_mul(ctx.ggml, corrected.tensor, ratio.tensor), lab.shape);
        corrected = core::wrap_tensor(ggml_add(ctx.ggml, corrected.tensor, b[0].tensor), lab.shape);
        corrected = to_rgb(corrected);
        auto shape = source.shape;
        shape.dims = {1, 1, 1, 1};
        shape.rank = 3;
        strength = modules::ReshapeModule({shape}).build(ctx, strength);
        auto inverse = scalar(strength, -1, 1);
        auto old = core::wrap_tensor(ggml_mul(ctx.ggml, source.tensor, inverse.tensor), source.shape);
        corrected = core::wrap_tensor(ggml_mul(ctx.ggml, corrected.tensor, strength.tensor), corrected.shape);
        corrected = core::wrap_tensor(ggml_add(ctx.ggml, old.tensor, corrected.tensor), source.shape);
        corrected = scalar(corrected, 2, -1);
        corrected = modules::ReshapeModule({TensorShape::from_dims({frames, height, width, 3})})
                        .build(ctx, core::ensure_backend_addressable_layout(ctx, corrected));
        corrected = modules::TransposeModule({{3, 0, 1, 2}, 4}).build(ctx, corrected);
        corrected = core::ensure_backend_addressable_layout(ctx, corrected);
        if (motion_frames > 0) {
            motion = modules::SliceModule({1, frames - motion_frames, motion_frames}).build(ctx, corrected);
            motion = core::ensure_backend_addressable_layout(ctx, motion);
            finish(corrected, {motion});
        } else {
            finish(corrected);
        }
    }
    TensorValue motion;

  private:
    TensorValue scalar(const TensorValue & x, float scale, float bias) {
        auto value = core::ensure_backend_addressable_layout(ctx, x);
        value = modules::ReshapeModule({TensorShape::from_dims({x.shape.num_elements()})}).build(ctx, value);
        auto y = modules::ScaleBiasModule({scale, bias}).build(ctx, value);
        y = modules::ReshapeModule({x.shape}).build(ctx, y);
        return y;
    }
    TensorValue power(const TensorValue & x, float exponent) {
        // Unlike arithmetic scalars, torch.pow converts the exponent to the tensor dtype.
        exponent = ggml_bf16_to_fp32(ggml_fp32_to_bf16(exponent));
        if (exponent == 3)
            return modules::MulModule().build(ctx, modules::MulModule().build(ctx, x, x), x);
        auto safe = ggml_clamp(ctx.ggml, core::ensure_backend_addressable_layout(ctx, x).tensor, 1.e-30F, 1.e30F);
        return core::wrap_tensor(ggml_exp(ctx.ggml, ggml_scale(ctx.ggml, ggml_log(ctx.ggml, safe), exponent)), x.shape);
    }
    TensorValue select(const TensorValue & mask, const TensorValue & yes, const TensorValue & no) {
        auto inv = ggml_scale_bias(ctx.ggml, mask.tensor, -1, 1);
        return core::wrap_tensor(
            ggml_add(ctx.ggml, ggml_mul(ctx.ggml, mask.tensor, yes.tensor), ggml_mul(ctx.ggml, inv, no.tensor)),
            yes.shape);
    }
    TensorValue threshold(const TensorValue & x, float bound, const TensorValue & yes, const TensorValue & no) {
        bound = ggml_bf16_to_fp32(ggml_fp32_to_bf16(bound));
        auto diff = scalar(x, 1, -bound);
        auto mask = core::wrap_tensor(ggml_step(ctx.ggml, diff.tensor), x.shape);
        return select(mask, yes, no);
    }
    TensorValue channel(const TensorValue & x, int index) {
        return modules::SliceModule({2, index, 1}).build(ctx, x);
    }
    TensorValue combine(TensorValue x, TensorValue y, TensorValue z) {
        return modules::ConcatModule({2}).build(ctx, modules::ConcatModule({2}).build(ctx, x, y), z);
    }
    TensorValue matrix(TensorValue x, std::vector<float> values, const std::string & key) {
        core::round_f32_to_bf16_in_place(values);
        auto w = constant(key, {3, 3}, std::move(values));
        auto shape = TensorShape::from_dims({x.shape.dims[0], 3, 3});
        w = modules::ReshapeModule({TensorShape::from_dims({1, 3, 3})}).build(ctx, w);
        w = modules::RepeatModule({shape}).build(ctx, w);
        x = modules::MatMulModule().build(ctx, x, w);
        return x;
    }
    TensorValue to_lab(TensorValue x, const std::string & key) {
        auto gamma = power(scalar(x, 1 / 1.055F, .055F / 1.055F), 2.4F);
        x = threshold(x, .04045F, gamma, scalar(x, 1 / 12.92F, 0));
        x = matrix(x, {.4124564F, .2126729F, .0193339F, .3575761F, .7151522F, .119192F, .1804375F, .072175F, .9503041F},
                   key + ".rgb_to_xyz");
        auto white = constant(key + ".white", {1, 1, 3}, {.94921875F, 1, 1.0859375F});
        x = core::wrap_tensor(ggml_div(ctx.ggml, x.tensor, white.tensor), x.shape);
        x = core::wrap_tensor(ggml_clamp(ctx.ggml, x.tensor, ggml_bf16_to_fp32(ggml_fp32_to_bf16(1.e-8F)), 1), x.shape);
        auto f = threshold(x, .008856F, power(x, 1.F / 3), scalar(x, 903.3F / 116, 16.F / 116));
        auto l = scalar(channel(f, 1), 116, -16);
        auto a = core::wrap_tensor(ggml_sub(ctx.ggml, channel(f, 0).tensor, channel(f, 1).tensor), l.shape);
        a = scalar(a, 500, 0);
        auto b = core::wrap_tensor(ggml_sub(ctx.ggml, channel(f, 1).tensor, channel(f, 2).tensor), l.shape);
        b = scalar(b, 200, 0);
        return combine(l, a, b);
    }
    TensorValue to_rgb(TensorValue lab) {
        auto l = channel(lab, 0);
        auto fy = scalar(l, 1.F / 116, 16.F / 116);
        auto fx = core::wrap_tensor(ggml_add(ctx.ggml, scalar(channel(lab, 1), 1.F / 500, 0).tensor, fy.tensor), fy.shape);
        auto fz = core::wrap_tensor(ggml_sub(ctx.ggml, fy.tensor, scalar(channel(lab, 2), 1.F / 200, 0).tensor), fy.shape);
        auto cube_x = power(fx, 3), cube_z = power(fz, 3);
        auto x = threshold(cube_x, .008856F, cube_x, scalar(fx, 116 / 903.3F, -16 / 903.3F));
        auto y = threshold(l, 903.3F * .008856F, power(fy, 3), scalar(l, 1 / 903.3F, 0));
        auto z = threshold(cube_z, .008856F, cube_z, scalar(fz, 116 / 903.3F, -16 / 903.3F));
        auto xyz = combine(x, y, z);
        auto white = constant("output.white", {1, 1, 3}, {.94921875F, 1, 1.0859375F});
        xyz = core::wrap_tensor(ggml_mul(ctx.ggml, xyz.tensor, white.tensor), xyz.shape);
        auto rgb = matrix(
            xyz,
            {3.2404542F, -.969266F, .0556434F, -1.5371385F, 1.8760108F, -.2040259F, -.4985314F, .041556F, 1.0572252F},
            "output.xyz_to_rgb");
        rgb = threshold(rgb, .0031308F, scalar(power(rgb, 1 / 2.4F), 1.055F, -.055F),
                        scalar(rgb, 12.92F, 0));
        return core::wrap_tensor(ggml_clamp(ctx.ggml, rgb.tensor, 0, 1), rgb.shape);
    }
};
} // namespace
struct ColorMatcher::State {
    core::ExecutionContext & execution;
    std::map<std::tuple<int64_t, int64_t, int64_t, bool>, std::unique_ptr<ColorGraph>> graphs;
    explicit State(core::ExecutionContext & execution) : execution(execution) {}
};
ColorMatcher::ColorMatcher(core::ExecutionContext & execution) : state_(std::make_unique<State>(execution)) {}
ColorMatcher::~ColorMatcher() = default;
const std::vector<float> & ColorMatcher::match(const std::vector<float> & video, const std::vector<float> & portrait,
                                               int64_t height, int64_t width, float strength) {
    if (strength == 0)
        return video;
    auto & graph = state_->graphs[{height, width, 0, true}];
    if (!graph)
        graph = std::make_unique<ColorGraph>(state_->execution, height, width);
    const std::vector<float> blend{strength};
    return graph->run({{"video", &video}, {"portrait", &portrait}, {"strength", &blend}});
}
ColorMatchResult ColorMatcher::match_device(const TensorValue & video, const std::vector<float> & portrait,
    int64_t height, int64_t width, float strength, int64_t motion_frames) {
    if (motion_frames <= 0 || motion_frames > 33 - motion_frames)
        throw std::runtime_error("FlashHead motion tail must fit within the delivered frames");
    auto & graph = state_->graphs[{height, width, motion_frames, strength != 0}];
    if (!graph)
        graph = std::make_unique<ColorGraph>(state_->execution, height, width, motion_frames, strength != 0);
    std::vector<float> weight{strength};
    const auto output = strength == 0
        ? graph->run_device({}, {{"video", video}})
        : graph->run_device({{"portrait", &portrait}, {"strength", &weight}}, {{"video", video}});
    return {output, graph->motion};
}
} // namespace engine::community_models::soulx_flashhead
