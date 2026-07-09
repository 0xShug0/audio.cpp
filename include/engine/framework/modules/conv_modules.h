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
