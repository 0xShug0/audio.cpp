#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <functional>
#include <vector>

namespace engine::models::reuse {

struct ReuseAssets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> tensors;
};

std::shared_ptr<const ReuseAssets> load_reuse_assets(const std::filesystem::path & path);

class ReuseRuntime {
public:
    ReuseRuntime(std::shared_ptr<const ReuseAssets> assets, core::ExecutionContext & execution,
                 assets::TensorStorageType storage);
    ~ReuseRuntime();
    std::vector<float> restore(const std::vector<float> & waveform, int sample_rate);
    void process_chunks(size_t count, int sample_rate,
        const std::function<void(size_t, std::vector<float> &)> & source,
        const std::function<void(size_t, const std::vector<float> &)> & sink);
    std::vector<std::vector<float>> restore_batch(
        const std::vector<std::vector<float>> & waveforms,
        int sample_rate);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::reuse
