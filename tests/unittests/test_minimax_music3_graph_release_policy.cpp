#include "engine/community_models/minimax_music3/ar_runtime.h"
#include "engine/community_models/minimax_music3/condition_encoder.h"
#include "engine/community_models/minimax_music3/depth_decoder.h"
#include "engine/community_models/minimax_music3/flow_sampler.h"
#include "engine/community_models/minimax_music3/flow_transformer.h"
#include "engine/community_models/minimax_music3/vocoder.h"
#include "engine/framework/modules/transformers/qwen_causal_decode_runtime.h"

#include <iostream>
#include <memory>
#include <type_traits>

namespace {

using engine::assets::TensorStorageType;
using engine::core::ExecutionContext;
using engine::core::TensorValue;
using engine::models::minimax_music3::MiniMaxMusic3ArRuntime;
using engine::models::minimax_music3::MiniMaxMusic3Assets;
using engine::models::minimax_music3::MiniMaxMusic3ConditionEncoderRuntime;
using engine::models::minimax_music3::MiniMaxMusic3DepthDecoderRuntime;
using engine::models::minimax_music3::MiniMaxMusic3FlowSamplerRuntime;
using engine::models::minimax_music3::MiniMaxMusic3FlowTransformerRuntime;
using engine::models::minimax_music3::MiniMaxMusic3VocoderRuntime;
using engine::modules::QwenCausalDecodeRuntimeConfig;

using Assets = std::shared_ptr<const MiniMaxMusic3Assets>;

static_assert(std::is_same_v<decltype(QwenCausalDecodeRuntimeConfig{}.evict_cuda_graph_cache_on_release), bool>);
static_assert(std::is_constructible_v<
    MiniMaxMusic3ArRuntime,
    Assets,
    ExecutionContext &,
    size_t,
    size_t,
    TensorStorageType,
    bool>);
static_assert(std::is_constructible_v<
    MiniMaxMusic3DepthDecoderRuntime,
    Assets,
    TensorValue,
    ExecutionContext &,
    size_t,
    size_t,
    TensorStorageType,
    bool>);
static_assert(std::is_constructible_v<
    MiniMaxMusic3ConditionEncoderRuntime,
    Assets,
    ExecutionContext &,
    size_t,
    size_t,
    TensorStorageType,
    bool>);
static_assert(std::is_constructible_v<
    MiniMaxMusic3FlowSamplerRuntime,
    Assets,
    ExecutionContext &,
    size_t,
    size_t,
    TensorStorageType,
    bool>);
static_assert(std::is_constructible_v<
    MiniMaxMusic3FlowTransformerRuntime,
    Assets,
    ExecutionContext &,
    size_t,
    size_t,
    TensorStorageType,
    bool>);
static_assert(std::is_constructible_v<
    MiniMaxMusic3VocoderRuntime,
    Assets,
    ExecutionContext &,
    size_t,
    size_t,
    TensorStorageType,
    bool>);

}  // namespace

int main() {
    std::cout << "minimax_music3_graph_release_policy_test: ok\\n";
    return 0;
}
