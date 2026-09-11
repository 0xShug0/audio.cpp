#pragma once

#include "engine/framework/core/module.h"

#include <cstdint>
#include <optional>

namespace engine::modules {

struct Conv1dConfig {
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel_size = 0;
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    bool use_bias = true;
};

struct Conv1dWeights {
    core::TensorValue weight;
    std::optional<core::TensorValue> bias;
};

class Conv1dModule {
public:
    explicit Conv1dModule(Conv1dConfig config);

    const Conv1dConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const Conv1dWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    Conv1dConfig config_;
};

struct Conv2dConfig {
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel_height = 0;
    int64_t kernel_width = 0;
    int stride_height = 1;
    int stride_width = 1;
    int padding_height = 0;
    int padding_width = 0;
    int dilation_height = 1;
    int dilation_width = 1;
    bool use_bias = true;
};

struct Conv2dWeights {
    core::TensorValue weight;
    std::optional<core::TensorValue> bias;
};

class Conv2dModule {
public:
    explicit Conv2dModule(Conv2dConfig config);

    const Conv2dConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const Conv2dWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    Conv2dConfig config_;
};

struct Conv3dConfig {
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel_depth = 0;
    int64_t kernel_height = 0;
    int64_t kernel_width = 0;
    int stride_depth = 1;
    int stride_height = 1;
    int stride_width = 1;
    int padding_depth = 0;
    int padding_height = 0;
    int padding_width = 0;
    int dilation_depth = 1;
    int dilation_height = 1;
    int dilation_width = 1;
    bool use_bias = true;
};

struct Conv3dWeights {
    core::TensorValue weight;
    std::optional<core::TensorValue> bias;
};

class Conv3dModule {
public:
    explicit Conv3dModule(Conv3dConfig config);

    const Conv3dConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const Conv3dWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    Conv3dConfig config_;
};

struct CausalConv2dConfig {
    Conv2dConfig conv;
    int64_t pad_left = 0;
    int64_t pad_right = 0;
    int64_t pad_top = 0;
    int64_t pad_bottom = 0;
};

class CausalConv2dModule {
public:
    explicit CausalConv2dModule(CausalConv2dConfig config);

    const CausalConv2dConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const Conv2dWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    CausalConv2dConfig config_;
};

struct SameWidthCausalConv2dConfig {
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel_size = 0;
    bool use_bias = true;
};

class SameWidthCausalConv2dModule {
public:
    explicit SameWidthCausalConv2dModule(SameWidthCausalConv2dConfig config);

    const SameWidthCausalConv2dConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const Conv2dWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    SameWidthCausalConv2dConfig config_;
};

struct PixelNormCausalConv2dResBlockConfig {
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel_size = 3;
    int pixel_norm_axis = 1;
    float pixel_norm_eps = 1e-6f;
};

struct PixelNormCausalConv2dResBlockWeights {
    Conv2dWeights conv1;
    Conv2dWeights conv2;
    std::optional<Conv2dWeights> shortcut;
};

class PixelNormCausalConv2dResBlockModule {
public:
    explicit PixelNormCausalConv2dResBlockModule(PixelNormCausalConv2dResBlockConfig config);

    const PixelNormCausalConv2dResBlockConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const PixelNormCausalConv2dResBlockWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    PixelNormCausalConv2dResBlockConfig config_;
};

struct CausalConv2dUpsampleConfig {
    int64_t channels = 0;
    int64_t kernel_size = 3;
    int64_t scale_height = 2;
    int64_t scale_width = 2;
    int64_t crop_top = 1;
    int64_t crop_bottom = 0;
};

class CausalConv2dUpsampleModule {
public:
    explicit CausalConv2dUpsampleModule(CausalConv2dUpsampleConfig config);

    const CausalConv2dUpsampleConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const Conv2dWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    CausalConv2dUpsampleConfig config_;
};

struct DepthwiseConv2dConfig {
    int64_t channels = 0;
    int64_t kernel_height = 0;
    int64_t kernel_width = 0;
    int stride_height = 1;
    int stride_width = 1;
    int padding_height = 0;
    int padding_width = 0;
    int dilation_height = 1;
    int dilation_width = 1;
    bool use_bias = true;
};

class DepthwiseConv2dModule {
public:
    explicit DepthwiseConv2dModule(DepthwiseConv2dConfig config);

    const DepthwiseConv2dConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const Conv2dWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    DepthwiseConv2dConfig config_;
};

struct ConvTranspose1dConfig {
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel_size = 0;
    int stride = 1;
    int padding = 0;
    int dilation = 1;
    bool use_bias = true;
};

struct ConvTranspose1dWeights {
    core::TensorValue weight;
    std::optional<core::TensorValue> bias;
};

bool is_conv_transpose1d_col2im_fast_path_eligible(
    const core::ModuleBuildContext & ctx,
    const ConvTranspose1dConfig & config) noexcept;

class ConvTranspose1dModule {
public:
    explicit ConvTranspose1dModule(ConvTranspose1dConfig config);

    const ConvTranspose1dConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const ConvTranspose1dWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    ConvTranspose1dConfig config_;
};

}  // namespace engine::modules
