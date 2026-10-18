#pragma once

#include "engine/models/index_echo/assets.h"
#include "engine/models/qwen3_asr/types.h"

#include "engine/framework/core/execution_context.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::index_echo {

struct IndexEchoConnectedAudio {
    std::vector<float> values;
    int64_t tokens = 0;
    int64_t hidden_size = 0;
};

class IndexEchoAudioConnectorRuntime {
public:
    IndexEchoAudioConnectorRuntime(
        std::shared_ptr<const IndexEchoAssets> assets,
        core::ExecutionContext & execution,
        assets::TensorStorageType weight_storage_type);
    ~IndexEchoAudioConnectorRuntime();

    IndexEchoAudioConnectorRuntime(const IndexEchoAudioConnectorRuntime &) = delete;
    IndexEchoAudioConnectorRuntime & operator=(const IndexEchoAudioConnectorRuntime &) = delete;

    IndexEchoConnectedAudio connect(const qwen3_asr::Qwen3ASRAudioEmbeddings & input);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::index_echo
