#pragma once

#include <cstdint>
#include <string>
#include <vector>

#if defined(ENGINE_HAVE_RKNN)
#include <rknn_api.h>
#endif

namespace engine::rknn {

// Shared plumbing for the model-local Rockchip NPU sidecars (docs/build/rockchip-npu.md).
// A sidecar is not a ggml backend: RKNN has no runtime graph builder, so each
// graph-isolated stage is compiled offline for one fixed shape and voiced through
// `librknnrt` by a module in the model's own directory. What every one of those modules
// needs from the driver is the same, and it lives here: read the engine image, init the
// context, pick the NPU cores, read the tensor attributes, and run one step. What stays
// model-local is the tensor geometry — which named input is the latent, how the window
// walk slices the source, how the seams are stitched.

// Reads a compiled `.rknn` image. `label` prefixes every message so a failure names the
// stage that failed ("Qwen3 speech decoder RKNN: ...").
std::vector<char> read_engine(const std::string & path, const std::string & label);

// Formats one `RKNN_*` return code the same way every sidecar does.
std::string status(const std::string & label, const char * what, int code);

#if defined(ENGINE_HAVE_RKNN)

// One graph input for `Engine::run`. `size` is the size of the buffer in bytes: with
// `pass_through = 0` the runtime wants the width of the type it is handed, not
// `rknn_tensor_attr.size`, which reports the engine's native (often fp16) width.
struct Input {
    uint32_t index = 0;
    const void * buf = nullptr;
    uint32_t size = 0;
    rknn_tensor_type type = RKNN_TENSOR_FLOAT32;
    uint32_t pass_through = 0;
};

// One compiled engine: a `rknn_context` plus its input/output counts.
class Engine {
public:
    Engine() = default;
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;
    ~Engine();

    // Loads `path` and asks for all three NPU cores by name: RKNN_NPU_CORE_AUTO lets the
    // driver pick, and for these graphs it picks one core. `AUDIOCPP_RKNN_CORE_MASK`
    // (1/2/4 = core 0/1/2, 7 = all) overrides it, for measuring whether an fp16 engine is
    // core-invariant.
    void load(const std::string & path, const std::string & label);
    void release();
    bool loaded() const { return ctx_ != 0; }

    const std::string & label() const { return label_; }
    uint32_t input_count() const { return n_input_; }
    uint32_t output_count() const { return n_output_; }
    // Attributes are read once at load and cached, so the model can validate names and
    // shapes without a driver round trip per call.
    const rknn_tensor_attr & input_attr(uint32_t index) const;
    const rknn_tensor_attr & output_attr(uint32_t index) const;

    // One step: sets every input, runs, and copies out every output as fp32.
    void run(const std::vector<Input> & inputs, std::vector<std::vector<float>> & outputs);

private:
    std::string label_;
    rknn_context ctx_ = 0;
    std::vector<rknn_tensor_attr> inputs_;
    std::vector<rknn_tensor_attr> outputs_;
    uint32_t n_input_ = 0;
    uint32_t n_output_ = 0;
};

#endif  // ENGINE_HAVE_RKNN

}  // namespace engine::rknn
