#pragma once

#include "engine/framework/core/module.h"

#include <optional>

namespace engine::modules {

struct NormConfig {
    int64_t hidden_size = 0;
    float eps = 1e-5f;
    bool use_weight = true;
    bool use_bias = true;
    bool preserve_input_layout = false;
};

struct NormWeights {
    std::optional<core::TensorValue> weight;
    std::optional<core::TensorValue> bias;
};

class LayerNormModule {
public:
    explicit LayerNormModule(NormConfig config);

    const core::ModuleSchema & schema() const noexcept;
    const NormConfig & config() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const NormWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    NormConfig config_;
};

class RMSNormModule {
public:
    explicit RMSNormModule(NormConfig config);

    const core::ModuleSchema & schema() const noexcept;
    const NormConfig & config() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const NormWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    NormConfig config_;
};

class GemmaRMSNormModule {
public:
    explicit GemmaRMSNormModule(NormConfig config);

    const core::ModuleSchema & schema() const noexcept;
    const NormConfig & config() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const NormWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    NormConfig config_;
};

struct GroupNormConfig {
    int64_t channels = 0;
    int64_t groups = 0;
    float eps = 1e-5F;
    bool use_weight = true;
    bool use_bias = true;
};

class GroupNormModule {
public:
    explicit GroupNormModule(GroupNormConfig config);

    const core::ModuleSchema & schema() const noexcept;
    const GroupNormConfig & config() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const NormWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    GroupNormConfig config_;
};

struct PixelNormConfig {
    int axis = 1;
    float eps = 1e-8f;
};

class PixelNormModule {
public:
    explicit PixelNormModule(PixelNormConfig config = {});

    const core::ModuleSchema & schema() const noexcept;
    const PixelNormConfig & config() const noexcept;

    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    PixelNormConfig config_;
};

struct BiasNormConfig {
    int64_t hidden_size = 0;
};

struct BiasNormWeights {
    core::TensorValue bias;
    float log_scale = 0.0f;
};

class BiasNormModule {
public:
    explicit BiasNormModule(BiasNormConfig config);

    const core::ModuleSchema & schema() const noexcept;
    const BiasNormConfig & config() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const BiasNormWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    BiasNormConfig config_;
};

struct AdaptiveInstanceNorm1dConfig {
    int64_t hidden_size = 0;
    float eps = 1e-5f;
};

struct AdaptiveInstanceNorm1dWeights {
    core::TensorValue gamma;
    core::TensorValue beta;
};

class AdaptiveInstanceNorm1dModule {
public:
    explicit AdaptiveInstanceNorm1dModule(AdaptiveInstanceNorm1dConfig config);

    const core::ModuleSchema & schema() const noexcept;
    const AdaptiveInstanceNorm1dConfig & config() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const AdaptiveInstanceNorm1dWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    AdaptiveInstanceNorm1dConfig config_;
};

struct BatchNorm1dEvalConfig {
    int64_t channels = 0;
};

struct BatchNorm1dEvalWeights {
    core::TensorValue scale;
    core::TensorValue bias;
};

class BatchNorm1dEvalModule {
public:
    explicit BatchNorm1dEvalModule(BatchNorm1dEvalConfig config);

    const core::ModuleSchema & schema() const noexcept;
    const BatchNorm1dEvalConfig & config() const noexcept;

    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const BatchNorm1dEvalWeights & weights) const;

    static const core::ModuleSchema & static_schema() noexcept;

private:
    BatchNorm1dEvalConfig config_;
};

}  // namespace engine::modules
