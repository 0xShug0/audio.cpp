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

}  // namespace engine::modules
