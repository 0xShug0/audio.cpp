#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::seed_vc {

struct SeedVcLengthRegulatorOutput {
    std::vector<float> values;
    int64_t batch = 0;
    int64_t tokens = 0;
    int64_t channels = 0;
};

class SeedVcDiscreteLengthRegulator {
public:
    SeedVcDiscreteLengthRegulator() = default;
    SeedVcDiscreteLengthRegulator(
        std::shared_ptr<const engine::assets::TensorSource> source,
        engine::core::BackendConfig backend,
        engine::assets::TensorStorageType storage_type,
        std::string prefix);
    ~SeedVcDiscreteLengthRegulator();

    SeedVcDiscreteLengthRegulator(SeedVcDiscreteLengthRegulator &&) noexcept;
    SeedVcDiscreteLengthRegulator & operator=(SeedVcDiscreteLengthRegulator &&) noexcept;
    SeedVcDiscreteLengthRegulator(const SeedVcDiscreteLengthRegulator &) = delete;
    SeedVcDiscreteLengthRegulator & operator=(const SeedVcDiscreteLengthRegulator &) = delete;

    int64_t codebook_size() const noexcept;
    int64_t channels() const noexcept;

    SeedVcLengthRegulatorOutput run(
        const std::vector<int32_t> & token_ids,
        int64_t batch,
        int64_t tokens) const;

private:
    struct State;

    std::string prefix_;
    int64_t codebook_size_ = 0;
    int64_t channels_ = 0;
    std::shared_ptr<State> state_;
};

class SeedVcCfmLengthRegulator {
public:
    SeedVcCfmLengthRegulator() = default;
    SeedVcCfmLengthRegulator(
        std::shared_ptr<const engine::assets::TensorSource> source,
        engine::core::BackendConfig backend,
        engine::assets::TensorStorageType storage_type,
        std::string prefix);
    ~SeedVcCfmLengthRegulator();

    SeedVcCfmLengthRegulator(SeedVcCfmLengthRegulator &&) noexcept;
    SeedVcCfmLengthRegulator & operator=(SeedVcCfmLengthRegulator &&) noexcept;
    SeedVcCfmLengthRegulator(const SeedVcCfmLengthRegulator &) = delete;
    SeedVcCfmLengthRegulator & operator=(const SeedVcCfmLengthRegulator &) = delete;

    int64_t codebook_size() const noexcept;
    int64_t channels() const noexcept;

    SeedVcLengthRegulatorOutput run(
        const std::vector<int32_t> & token_ids,
        const std::vector<int64_t> & output_lengths,
        int64_t batch,
        int64_t tokens) const;

private:
    struct State;

    std::string prefix_;
    int64_t codebook_size_ = 0;
    int64_t channels_ = 0;
    std::shared_ptr<State> state_;
};

struct SeedVcV1LengthRegulatorInput {
    std::vector<float> content;
    std::vector<float> f0;
    std::vector<int64_t> output_lengths;
    int64_t batch = 0;
    int64_t tokens = 0;
    int64_t f0_tokens = 0;
    bool has_f0 = false;
};

class SeedVcV1LengthRegulator {
public:
    SeedVcV1LengthRegulator() = default;
    SeedVcV1LengthRegulator(
        std::shared_ptr<const engine::assets::TensorSource> source,
        engine::core::BackendConfig backend,
        engine::assets::TensorStorageType storage_type,
        std::string prefix);
    ~SeedVcV1LengthRegulator();

    SeedVcV1LengthRegulator(SeedVcV1LengthRegulator &&) noexcept;
    SeedVcV1LengthRegulator & operator=(SeedVcV1LengthRegulator &&) noexcept;
    SeedVcV1LengthRegulator(const SeedVcV1LengthRegulator &) = delete;
    SeedVcV1LengthRegulator & operator=(const SeedVcV1LengthRegulator &) = delete;

    int64_t channels() const noexcept;
    int64_t input_channels() const noexcept;
    int64_t f0_bins() const noexcept;

    SeedVcLengthRegulatorOutput run(const SeedVcV1LengthRegulatorInput & input) const;

private:
    struct State;

    std::string prefix_;
    int64_t channels_ = 0;
    int64_t input_channels_ = 0;
    int64_t f0_bins_ = 0;
    std::shared_ptr<State> state_;
};

}  // namespace engine::models::seed_vc
