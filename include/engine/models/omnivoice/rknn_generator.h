#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::omnivoice {

// One diffusion step in the engine's own row layout, out of the packed CFG inputs the ggml
// graph uses (generator.cpp). The audio rows carry the codebook offsets the packed layout adds
// (codebook * vocab_size); the engine adds them again from its own constant, so the sidecar
// subtracts them at copy time.
struct OmniVoiceRknnStepInputs {
    int64_t total_tokens = 0;    // conditional row length: style + text + reference + target
    int64_t audio_start = 0;     // first audio position of the conditional row (style + text)
    int64_t target_frames = 0;   // unconditional row length; the target block is the cond row tail
    std::array<std::vector<int32_t>, 8> conditional_ids;    // 8 rows, each total_tokens long
    std::array<std::vector<int32_t>, 8> unconditional_ids;  // 8 rows, first target_frames real
    std::vector<int32_t> conditional_text_ids;              // total_tokens long
    std::vector<int32_t> unconditional_text_ids;            // total_tokens long
    float guidance = 0.0F;
};

// Rockchip NPU (RKNN) backend for the diffusion-LM generator step: the 28-layer Qwen3 forward
// that the ggml graph in generator.cpp computes 32 times per utterance, and the stage that
// board profiling puts at 98.8% of the omnivoice runtime. RKNN has no dynamic shapes, so each
// engine is compiled offline for one fixed total-token length
// (tools/rknn/prepare_omnivoice_rknn_generator.py) and a ladder of them is registered here. A
// request of `total_tokens` runs on the smallest engine that fits it: rows right-padded,
// attention mask off the pad (pad rows attend to themselves only), output cropped to the
// conditional target block. Measured on the PC simulator, a padded run matches the exact fp32
// reference at 60-62 dB with pad 0/16/32 (argmax agreement 0.96-0.98 at fp16 weight precision),
// so the ladder is a time trade, not a quality one: the step graph costs ~S^2 in attention and
// every frame of pad is paid for once per diffusion step.
//
// The engine takes the packed 2-row classifier-free-guidance batch (conditional row first),
// the additive attention mask, the audio mask, the RoPE cos/sin tables (the host computes the
// angles in float64, mod 2*pi: angles up to ~1e3 rad quantized in an fp16 graph destroy
// attention, so the trig lives here and the tables enter as inputs), and the guidance scale as
// an input; it returns the guided conditional logits, the same tensor the ggml graph produces
// after its guidance combine. All scoring (log_softmax, penalties, gumbel, top-k) stays in the
// host loop, exactly where the ggml path's device_scoring=false branch reads it.
//
// A ladder compiled with the producer's `--front-k 28` (file names carry `front`) is the
// FRONT-SHARD contract: five inputs (no guidance) and the raw post-layer hidden state
// ``hidden [2,S,hidden_size]`` as output. The CFG tail - final RMS norm, audio heads, guided
// combine - then runs on the host in fp32 (the front-shard forward graph in generator.cpp),
// which is the measured fix for the fp16 heads' 26.9 dB uncond logits: the same 28-layer
// backbone leaves the hidden state at 67 dB, so with an fp32 tail the guided logits return to
// ~84 dB and guidance > 0 is usable on the NPU (PC simulator, real step-0 feed).
class OmniVoiceRknnGenerator {
public:
    // Registers the ladder without loading any of it: one engine holds the 0.6B-parameter
    // fp16 weight set (~1.2 GiB of the driver's own residency), so a ladder of them does not
    // fit together. The length an engine covers is read from its file name
    // (`omnivoice_generator_seq512_fp16.rknn`, the producer's convention) and validated against
    // the engine's own attributes when it loads.
    OmniVoiceRknnGenerator(
        const std::vector<std::string> & engine_paths,
        int64_t codebooks,
        int64_t vocab_size,
        int64_t head_dim,
        float rope_theta,
        int64_t hidden_size);
    ~OmniVoiceRknnGenerator();

    OmniVoiceRknnGenerator(const OmniVoiceRknnGenerator &) = delete;
    OmniVoiceRknnGenerator & operator=(const OmniVoiceRknnGenerator &) = delete;

    // Longest conditional row the ladder covers, in tokens. A longer request has no engine and
    // has to run on the ggml graph, which is why the caller keeps it.
    int64_t max_tokens() const;
    int64_t engine_count() const;

    // True when the registered ladder is the front-shard contract (name-detected at
    // registration, attribute-validated at load); run() then writes hidden rows, not logits.
    bool front_shard() const;

    // The smallest engine that covers `total_tokens`, without loading it: 0 when none does.
    int64_t bucket_for(int64_t total_tokens) const;

    // One full diffusion step at a bucket from bucket_for(). Full ladder: writes the guided
    // conditional logits of the target block as [target_frames][codebooks][vocab_size] floats
    // into `logits_out`, the layout the host scorer indexes. Front-shard ladder: writes the
    // raw hidden rows into `logits_out` in the packed host layout - the conditional row at
    // ``[0, total_tokens)`` and the unconditional row right behind it at
    // ``[total_tokens, total_tokens + target_frames)`` rows of `hidden_size` floats (pass
    // `row_capacity` = this request's total_tokens, the row stride; the buffer past the
    // written rows is untouched - the caller zero-pads it). The engine's own output attribute
    // layout (declared [1,codebooks,tokens,vocab] or the converter's NHWC relabel) is read at
    // load, so the crop follows the engine, not the producer's file.
    void run(
        int64_t bucket,
        const OmniVoiceRknnStepInputs & inputs,
        float * logits_out,
        int64_t row_capacity = 0);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The resident-set ceiling this sidecar enforces on its ladder, in MiB: the value
// `AUDIOCPP_RKNN_APERTURE_MIB` sets, else this board's measured ceiling, and 0 when the
// accounting is off and every declared rung is pinned.
int64_t aperture_budget_mib();

}  // namespace engine::models::omnivoice
