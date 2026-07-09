#pragma once

#include "engine/framework/core/module.h"

namespace engine::modules {

class MatMulModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & lhs,
        const core::TensorValue & rhs) const;

    static const core::ModuleSchema & static_schema() noexcept;
};

class AddModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & lhs,
        const core::TensorValue & rhs) const;

    static const core::ModuleSchema & static_schema() noexcept;
};

class MulModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & lhs,
        const core::TensorValue & rhs) const;

    static const core::ModuleSchema & static_schema() noexcept;
};

class TimeMask4dModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & mask) const;

    static const core::ModuleSchema & static_schema() noexcept;
};

class ResidualAddModule {
public:
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & residual) const;

    static const core::ModuleSchema & static_schema() noexcept;
};

struct ReduceConfig {
    int axis = -1;
};

class ReduceMeanModule {
public:
    explicit ReduceMeanModule(ReduceConfig config);

    const ReduceConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    ReduceConfig config_;
};

class ReduceSumModule {
public:
    explicit ReduceSumModule(ReduceConfig config);

    const ReduceConfig & config() const noexcept;
    const core::ModuleSchema & schema() const noexcept;
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    static const core::ModuleSchema & static_schema() noexcept;

private:
    ReduceConfig config_;
};

}  // namespace engine::modules
