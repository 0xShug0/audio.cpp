#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/community_models/parakeet_tdt/assets.h"
#include "engine/community_models/parakeet_tdt/decoder.h"
#include "engine/community_models/parakeet_tdt/encoder.h"
#include "engine/community_models/parakeet_tdt/frontend.h"
#include "engine/community_models/parakeet_tdt/weights.h"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

namespace engine::community_models::parakeet_tdt {

class ParakeetTDTSessionBase : public runtime::RuntimeSessionBase {
public:
    ParakeetTDTSessionBase(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const ParakeetTDTAssets> assets);
    ~ParakeetTDTSessionBase() override;

protected:
    std::string family_impl() const;
    runtime::VoiceTaskKind task_kind_impl() const;
    runtime::RunMode run_mode_impl() const;
    ParakeetDecodeOptions decode_options_for_request(const runtime::TaskRequest & request) const;

    runtime::TaskSpec task_;
    std::shared_ptr<const ParakeetTDTAssets> assets_;
    std::shared_ptr<const ParakeetWeights> weights_;
    size_t weight_context_bytes_ = 3072ull * 1024ull * 1024ull;
    size_t encoder_graph_arena_bytes_ = 1024ull * 1024ull * 1024ull;
    size_t decoder_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
    engine::assets::TensorStorageType matmul_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType conv_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    ParakeetFrontend frontend_;
    std::unique_ptr<ParakeetEncoderRuntime> encoder_;
    std::unique_ptr<ParakeetDecoderRuntime> decoder_;
};

class ParakeetTDTOfflineSession final
    : public ParakeetTDTSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    ParakeetTDTOfflineSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const ParakeetTDTAssets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;
};

}  // namespace engine::community_models::parakeet_tdt
