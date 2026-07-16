#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::seed_vc {

struct SeedVcAstralQuantizerOutput {
    std::vector<int32_t> indices;
    int64_t batch = 0;
    int64_t tokens = 0;
    int64_t code_dim = 0;
    int64_t codebook_size = 0;
};

class SeedVcAstralQuantizer {
public:
    SeedVcAstralQuantizer() = default;
    SeedVcAstralQuantizer(
        std::shared_ptr<const engine::assets::TensorSource> source,
        engine::core::BackendConfig backend,
        engine::assets::TensorStorageType storage_type,
        std::string prefix,
        int64_t input_channels,
        int64_t channels,
        int64_t intermediate_channels,
        int64_t blocks,
        int64_t codebook_size);
    ~SeedVcAstralQuantizer();

    SeedVcAstralQuantizer(SeedVcAstralQuantizer &&) noexcept;
    SeedVcAstralQuantizer & operator=(SeedVcAstralQuantizer &&) noexcept;
    SeedVcAstralQuantizer(const SeedVcAstralQuantizer &) = delete;
    SeedVcAstralQuantizer & operator=(const SeedVcAstralQuantizer &) = delete;

    int64_t input_channels() const noexcept;
    int64_t channels() const noexcept;
    int64_t code_dim() const noexcept;
    int64_t codebook_size() const noexcept;

    SeedVcAstralQuantizerOutput run(
        const std::vector<float> & hidden_states,
        int64_t batch,
        int64_t tokens) const;

private:
    struct State;

    std::string prefix_;
    int64_t input_channels_ = 0;
    int64_t channels_ = 0;
    int64_t intermediate_channels_ = 0;
    int64_t blocks_ = 0;
    int64_t code_dim_ = 0;
    int64_t codebook_size_ = 0;
    std::shared_ptr<State> state_;
};

}  // namespace engine::models::seed_vc
