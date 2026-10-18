#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/session.h"

#include <memory>

namespace engine::models::tone_color_vc {

class ToneColorRuntime {
public:
    ToneColorRuntime(std::shared_ptr<const assets::TensorSource> source,
                     core::ExecutionContext & execution, core::BackendConfig backend,
                     const io::json::Value & config);
    ~ToneColorRuntime();
    runtime::AudioBuffer convert(const runtime::AudioBuffer & source,
                                 const runtime::AudioBuffer & reference,
                                 float temperature, uint32_t seed);

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace engine::models::tone_color_vc
