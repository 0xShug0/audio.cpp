#pragma once

#include "engine/framework/core/module.h"

namespace engine::modules {

enum class GeluApproximation {
    ExactErf,
    Tanh,
    Quick,
};

struct GeluConfig {
    GeluApproximation approximation = GeluApproximation::ExactErf;
};

class ReluModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;
};

struct LeakyReluConfig {
    float negative_slope = 0.01F;
};

class LeakyReluModule {
public:
    explicit LeakyReluModule(LeakyReluConfig config = {});

    const LeakyReluConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    LeakyReluConfig config_;
};

class SigmoidModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;
};

class TanhModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;
};

class SqrtModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;
};

class GeluModule {
public:
    explicit GeluModule(GeluConfig config = {});

    const GeluConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    GeluConfig config_;
};

class SiluModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;
};

class SwooshLModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;
};

class SwooshRModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;
};

class EluModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;
};

class SoftmaxModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;
};

class GLUModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;
};

struct Snake1dConfig {
    int64_t hidden_size = 0;
};

struct Snake1dWeights {
    core::TensorValue alpha;
};

class Snake1dModule {
public:
    explicit Snake1dModule(Snake1dConfig config);

    const Snake1dConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const Snake1dWeights & weights) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    Snake1dConfig config_;
};

enum class AliasFreeActivationKind {
    SnakeBeta,
};

struct AliasFreeActivationConfig {
    int64_t channels = 0;
    int64_t kernel_size = 0;
    int64_t upsample_ratio = 2;
    AliasFreeActivationKind kind = AliasFreeActivationKind::SnakeBeta;
};

struct AliasFreeActivationWeights {
    core::TensorValue alpha;
    core::TensorValue inv_beta;
    core::TensorValue up_filter_even;
    core::TensorValue up_filter_odd;
    core::TensorValue down_filter;
};

class AliasFreeActivationModule {
public:
    explicit AliasFreeActivationModule(AliasFreeActivationConfig config);

    const AliasFreeActivationConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const AliasFreeActivationWeights & weights) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    AliasFreeActivationConfig config_;
};

}  // namespace engine::modules
