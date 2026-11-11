#include "engine/models/omnivoice/rknn_generator.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/rknn/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::omnivoice {

// The board's per-process NPU aperture, measured once on this board (see the cosyvoice3 flow
// sidecar, which pins its ladder against the same ceiling): engines ask the driver for their
// whole residency at `rknn_init`, and the resident set must stay inside the aperture.
constexpr int64_t kAperturePinBudgetMiB = 3900;

namespace {

#if defined(ENGINE_HAVE_RKNN)
using Clock = std::chrono::steady_clock;

// Prefix of every message this sidecar raises, shared with the plumbing in engine::rknn so a
// failure names the stage that failed.
const char * const kLabel = "OmniVoice generator RKNN";

// The graph order of tools/rknn/prepare_omnivoice_rknn_generator.py.
const char * const kInputNames[] = {
    "input_ids", "audio_mask", "attn_mask", "rope_cos", "rope_sin", "guidance"};
constexpr uint32_t kInputCount = 6;
// The front-shard export (the producer's --front-k): the same graph without the guidance input,
// ending at the post-layer hidden state; its CFG tail (final norm, audio heads, guided combine)
// runs on the host in fp32.
constexpr uint32_t kFrontInputCount = 5;
constexpr uint32_t kOutputCount = 1;
constexpr uint32_t kInputIds = 0;
constexpr uint32_t kInputAudioMask = 1;
constexpr uint32_t kInputAttnMask = 2;
constexpr uint32_t kInputRopeCos = 3;
constexpr uint32_t kInputRopeSin = 4;
constexpr uint32_t kInputGuidance = 5;

// The CFG pair: row 0 is the conditional sequence, row 1 its unconditional counterpart, and the
// engine mixes them internally (it is the whole step graph, guidance combine included).
constexpr int64_t kCfgBatch = 2;

// The additive attention mask value for masked positions, the exporter's MASK_VALUE.
constexpr float kMaskValue = -10000.0F;

// The token length an engine covers, read out of the producer's file name
// (`omnivoice_generator_seq512_fp16.rknn`). Reading it without loading is what keeps the ladder
// from being resident: one context costs ~1.2 GB, so a ladder of them never fits at once.
int64_t name_seq_len(const std::string & path) {
    const std::string stem = std::filesystem::path(path).stem().string();
    const auto at = stem.find("seq");
    if (at == std::string::npos) {
        return -1;
    }
    int64_t value = 0;
    size_t index = at + 3;
    for (; index < stem.size() && stem[index] >= '0' && stem[index] <= '9'; ++index) {
        value = value * 10 + (stem[index] - '0');
    }
    return index == at + 3 ? -1 : value;
}
#endif

}  // namespace

struct OmniVoiceRknnGenerator::Impl {
    // One registered engine of the ladder, plus the scratch the padding needs. The scratch is
    // sized on load and reused across calls because one utterance runs the same shape ~32 times.
    struct Slot {
        std::string path;
        int64_t tokens = 0;      // the conditional-row length this engine covers
        int64_t codebooks = 0;
        int64_t vocab_size = 0;
        // The front-shard contract (the producer's --front-k export, name-detected at
        // registration from the `front` in the file name): 5 inputs, output hidden rows, the
        // CFG tail on the host.
        bool front_shard = false;
#if defined(ENGINE_HAVE_RKNN)
        // What this engine will occupy once resident, read from the driver at registration; 0
        // means the driver could not say, and the file size is used for it.
        int64_t aperture_mib = 0;
        engine::rknn::Engine engine;
        bool loaded = false;
        // Feed type and buffer width of each input (host widths, see load()). The feed layout
        // is the engine's own: the 4-D attention mask is NHWC-relabelled by the converter on
        // some rungs, and a 4-D normalized input fed as UNDEFINED fails `rknn_inputs_set`.
        uint32_t input_sizes[kInputCount] = {};
        rknn_tensor_type input_types[kInputCount] = {};
        rknn_tensor_format input_fmts[kInputCount] = {};
        uint32_t input_pass_through[kInputCount] = {};
        std::vector<int64_t> ids;   // [2][codebooks][tokens]
        std::vector<float> audio_mask;   // [2][tokens]
        std::vector<float> attn_mask;    // [2][tokens][tokens], NHWC relabel is a flat copy
        std::vector<float> rope_cos;     // [2][tokens][head_dim/2], arange positions
        std::vector<float> rope_sin;
        float guidance = 0.0F;
        int64_t hidden_size = 0;   // front-shard output width (the transformer hidden size)

        void load(int64_t expected_codebooks, int64_t expected_vocab, int64_t head_dim,
                  float rope_theta, int64_t expected_hidden) {
            const auto start = Clock::now();
            engine.load(path, kLabel);
            const uint32_t expected_inputs = front_shard ? kFrontInputCount : kInputCount;
            if (engine.input_count() != expected_inputs || engine.output_count() != kOutputCount) {
                throw std::runtime_error(
                    std::string(kLabel) + ": expected the " + std::to_string(expected_inputs) +
                    "-input/1-output " + (front_shard ? "front-shard" : "step-graph") +
                    " export, got " + std::to_string(engine.input_count()) + "/" +
                    std::to_string(engine.output_count()));
            }
            codebooks = expected_codebooks;
            vocab_size = expected_vocab;
            const int64_t n_cfg = kCfgBatch * tokens;
            for (uint32_t index = 0; index < engine.input_count(); ++index) {
                const rknn_tensor_attr & attr = engine.input_attr(index);
                const std::string name(attr.name);
                debug::timing_log_scalar(
                    "omnivoice.generator.rknn.input_attr" + std::to_string(index),
                    name + " type=" + std::to_string(attr.type) + " size=" + std::to_string(attr.size) +
                        " elems=" + std::to_string(attr.n_elems) + " fmt=" + std::to_string(attr.fmt) +
                        " dims=" + std::to_string(attr.n_dims));
                if (name != kInputNames[index]) {
                    throw std::runtime_error(
                        std::string(kLabel) + ": " + path + " input " + std::to_string(index) + " is \"" +
                        name + "\", the export names it \"" + kInputNames[index] + "\"");
                }
                const int64_t expected_elems = index == kInputIds ? kCfgBatch * codebooks * tokens
                    : index == kInputAttnMask ? kCfgBatch * tokens * tokens
                    : index == kInputGuidance ? 1
                    : index == kInputRopeCos || index == kInputRopeSin
                          ? n_cfg * (head_dim / 2)
                    : n_cfg;  // audio_mask: [2, tokens]
                if (attr.n_elems != expected_elems) {
                    throw std::runtime_error(
                        std::string(kLabel) + ": " + path + " input " + name + " holds " +
                        std::to_string(attr.n_elems) + " elements, its name promises " +
                        std::to_string(expected_elems) + " at seq=" + std::to_string(tokens));
                }
                // Inputs travel at host width, not at the engine's native width: `attr.type`
                // reports FLOAT16 for the fp16 graph, and feeding that type over a float32
                // scratch reads it two bytes at a time (the qwen3 speech-decoder precedent:
                // with pass_through = 0 the declared type is the type of the buffer handed
                // over). The int64 ids pass through untouched; every float (mask, rope table,
                // guidance) enters as float32 and the runtime converts.
                if (attr.type != RKNN_TENSOR_INT64 && attr.type != RKNN_TENSOR_INT32 &&
                    attr.type != RKNN_TENSOR_FLOAT16 && attr.type != RKNN_TENSOR_FLOAT32) {
                    throw std::runtime_error(
                        std::string(kLabel) + ": " + path + " input " + name + " has type " +
                        std::to_string(attr.type) + ", which this sidecar cannot feed");
                }
                input_types[index] = attr.type == RKNN_TENSOR_INT64
                    ? RKNN_TENSOR_INT64
                    : attr.type == RKNN_TENSOR_INT32 ? RKNN_TENSOR_INT32 : RKNN_TENSOR_FLOAT32;
                input_sizes[index] = static_cast<uint32_t>(attr.n_elems * sizeof(float) *
                                                          (attr.type == RKNN_TENSOR_INT64 ? 2 : 1));
                input_pass_through[index] = attr.type == RKNN_TENSOR_INT64 ? 1U : 0U;
                input_fmts[index] = static_cast<rknn_tensor_format>(attr.fmt);
            }
            const rknn_tensor_attr & out = engine.output_attr(0);
            if (front_shard) {
                // Front-shard output: the raw hidden state [2, tokens, hidden_size]. It is a
                // 3-D (or NHWC-relabelled 3-D, which for a 3-D tensor is the same [2, tokens,
                // hidden] order) fp32 tensor, one row per CFG side. The CFG tail runs on the
                // host, so this engine has no vocab dimension.
                if (out.n_dims != 3 || out.dims[0] != kCfgBatch || out.dims[1] != tokens ||
                    out.dims[2] != expected_hidden ||
                    out.n_elems != kCfgBatch * tokens * expected_hidden) {
                    throw std::runtime_error(
                        std::string(kLabel) + ": " + path + " is a front-shard engine whose output " +
                        std::to_string(out.n_dims) + "D[" + std::to_string(out.dims[0]) + "," +
                        std::to_string(out.dims[1]) + "," + std::to_string(out.dims[2]) +
                        "] is not the expected [2," + std::to_string(tokens) + "," +
                        std::to_string(expected_hidden) + "] hidden state");
                }
                hidden_size = expected_hidden;
                ids.resize(static_cast<size_t>(kCfgBatch * codebooks * tokens));
                audio_mask.resize(static_cast<size_t>(n_cfg));
                attn_mask.resize(static_cast<size_t>(kCfgBatch * tokens * tokens));
                guidance = 0.0F;
                const int64_t half_h = head_dim / 2;
                rope_cos.resize(static_cast<size_t>(n_cfg * half_h));
                rope_sin.resize(static_cast<size_t>(n_cfg * half_h));
                for (int64_t token = 0; token < tokens; ++token) {
                    for (int64_t dim = 0; dim < half_h; ++dim) {
                        const double angle = std::fmod(
                            static_cast<double>(token) *
                                std::pow(
                                    static_cast<double>(rope_theta),
                                    -2.0 * static_cast<double>(dim) / static_cast<double>(head_dim)),
                            2.0 * 3.14159265358979323846);
                        rope_cos[static_cast<size_t>(token * half_h + dim)] =
                            static_cast<float>(std::cos(angle));
                        rope_sin[static_cast<size_t>(token * half_h + dim)] =
                            static_cast<float>(std::sin(angle));
                    }
                }
                std::memcpy(rope_cos.data() + static_cast<size_t>(tokens * half_h), rope_cos.data(),
                            static_cast<size_t>(tokens * half_h) * sizeof(float));
                std::memcpy(rope_sin.data() + static_cast<size_t>(tokens * half_h), rope_sin.data(),
                            static_cast<size_t>(tokens * half_h) * sizeof(float));
                loaded = true;
                return;
            }
            if (out.n_elems != codebooks * tokens * vocab_size) {
                throw std::runtime_error(
                    std::string(kLabel) + ": " + path + " returns " + std::to_string(out.n_elems) +
                    " values, expected " + std::to_string(codebooks * tokens * vocab_size));
            }
            // The crop follows the engine's own output layout: the producer declares
            // [1, codebooks, tokens, vocab], and the converter may relabel the 4-D output to
            // NHWC ([1, tokens, vocab, codebooks]). Anything else is a graph we do not know.
            output_row_major_codebook =
                out.n_dims == 4 && out.dims[1] == codebooks && out.dims[3] == vocab_size;
            const bool output_nhwc =
                out.n_dims == 4 && out.dims[1] == tokens && out.dims[2] == vocab_size &&
                out.dims[3] == codebooks;
            if (!output_row_major_codebook && !output_nhwc) {
                throw std::runtime_error(
                    std::string(kLabel) + ": " + path + " output layout " +
                    std::to_string(out.n_dims) + "D[" + std::to_string(out.dims[0]) + "," +
                    std::to_string(out.dims[1]) + "," + std::to_string(out.dims[2]) + "," +
                    std::to_string(out.dims[3]) + "] is neither of the two known ones");
            }
            ids.resize(static_cast<size_t>(kCfgBatch * codebooks * tokens));
            audio_mask.resize(static_cast<size_t>(n_cfg));
            attn_mask.resize(static_cast<size_t>(kCfgBatch * tokens * tokens));
            // The RoPE tables for the per-row arange positions (both CFG rows share the one
            // table; pads just continue the arange). Angles are reduced mod 2*pi in float64
            // before the trig: the graph's fp16 Cos/Sin of a raw ~1e3-rad argument quantizes
            // the argument (ulp ~0.5 rad) and turns the audio to noise, so the table is the
            // engine input (the qwen3 vocoder's freqs_in precedent).
            const int64_t half = head_dim / 2;
            rope_cos.resize(static_cast<size_t>(n_cfg * half));
            rope_sin.resize(static_cast<size_t>(n_cfg * half));
            for (int64_t token = 0; token < tokens; ++token) {
                for (int64_t dim = 0; dim < half; ++dim) {
                    const double angle = std::fmod(
                        static_cast<double>(token) *
                            std::pow(
                                static_cast<double>(rope_theta),
                                -2.0 * static_cast<double>(dim) / static_cast<double>(head_dim)),
                        2.0 * 3.14159265358979323846);
                    rope_cos[static_cast<size_t>(token * half + dim)] =
                        static_cast<float>(std::cos(angle));
                    rope_sin[static_cast<size_t>(token * half + dim)] =
                        static_cast<float>(std::sin(angle));
                }
            }
            std::memcpy(rope_cos.data() + static_cast<size_t>(tokens * half), rope_cos.data(),
                        static_cast<size_t>(tokens * half) * sizeof(float));
            std::memcpy(rope_sin.data() + static_cast<size_t>(tokens * half), rope_sin.data(),
                        static_cast<size_t>(tokens * half) * sizeof(float));
            debug::timing_log_scalar("omnivoice.generator.rknn.load_ms", debug::elapsed_ms(start));
            debug::trace_log_scalar("omnivoice.generator.rknn.loaded_tokens", tokens);
            // Only once the engine is validated and its scratch exists: a slot that threw halfway
            // through would otherwise be marked resident, and every later step would be handed an
            // engine with no input buffers instead of the same clear failure.
            loaded = true;
        }

        // One diffusion step at the engine's own length. The rows arrive at their real lengths
        // and are widened to `tokens` here; the result is cut back by the caller.
        void run(const OmniVoiceRknnStepInputs & in) {
            const int64_t total = in.total_tokens;
            const int64_t target = in.target_frames;
            const auto id_index = [this](int64_t row, int64_t codebook, int64_t token) {
                return static_cast<size_t>((row * codebooks + codebook) * tokens + token);
            };
            for (int64_t codebook = 0; codebook < codebooks; ++codebook) {
                const int32_t offset = static_cast<int32_t>(codebook * vocab_size);
                for (int64_t token = 0; token < tokens; ++token) {
                    const size_t cond = id_index(0, codebook, token);
                    const size_t uncond = id_index(1, codebook, token);
                    if (token >= total) {
                        // Pad: audio_mask is 0 there, so any in-vocab id works and 0 is one.
                        ids[cond] = 0;
                        ids[uncond] = 0;
                        continue;
                    }
                    if (codebook == 0) {
                        // Row 0 doubles as the text row (embed_tokens) and the codebook-0 audio
                        // row: at audio positions the packed layout stores the raw codebook-0 id
                        // in both places, at text positions the text id.
                        ids[cond] = token >= in.audio_start
                            ? in.conditional_ids[0][static_cast<size_t>(token)]
                            : in.conditional_text_ids[static_cast<size_t>(token)];
                    } else {
                        ids[cond] =
                            in.conditional_ids[static_cast<size_t>(codebook)][static_cast<size_t>(token)] -
                            offset;
                    }
                    ids[uncond] =
                        in.unconditional_ids[static_cast<size_t>(codebook)][static_cast<size_t>(token)] -
                        offset;
                }
            }
            for (int64_t token = 0; token < tokens; ++token) {
                audio_mask[static_cast<size_t>(token)] =
                    token >= in.audio_start && token < total ? 1.0F : 0.0F;
                audio_mask[static_cast<size_t>(tokens + token)] = token < target ? 1.0F : 0.0F;
            }
            // Additive mask: 0 keeps a key, kMaskValue masks it. The conditional row attends to
            // its own sequence, the unconditional row to its own, and every pad token attends to
            // itself only, so a padded run equals an exact one (measured on the simulator).
            std::fill(attn_mask.begin(), attn_mask.end(), kMaskValue);
            for (int64_t row = 0; row < kCfgBatch; ++row) {
                const int64_t real = row == 0 ? total : target;
                float * row_base = attn_mask.data() + row * tokens * tokens;
                for (int64_t r = 0; r < real; ++r) {
                    std::fill(row_base + r * tokens, row_base + r * tokens + real, 0.0F);
                }
                for (int64_t r = real; r < tokens; ++r) {
                    row_base[r * tokens + r] = 0.0F;
                }
            }
            guidance = in.guidance;

            std::vector<engine::rknn::Input> inputs(front_shard ? kFrontInputCount : kInputCount);
            inputs[kInputIds].buf = ids.data();
            inputs[kInputAudioMask].buf = audio_mask.data();
            inputs[kInputAttnMask].buf = attn_mask.data();
            inputs[kInputRopeCos].buf = rope_cos.data();
            inputs[kInputRopeSin].buf = rope_sin.data();
            if (!front_shard) {
                inputs[kInputGuidance].buf = &guidance;
            }
            for (uint32_t index = 0; index < inputs.size(); ++index) {
                inputs[index].index = index;
                inputs[index].size = input_sizes[index];
                inputs[index].type = input_types[index];
                inputs[index].fmt = input_fmts[index];
                inputs[index].pass_through = input_pass_through[index];
            }
            std::vector<std::vector<float>> outputs;
            engine.run(inputs, outputs);
            last_output.swap(outputs[0]);
        }

        // Cut the pad back out: the rows arrive packed on [0, total) and right-padded after,
        // so the conditional target block is the tail of the REAL row,
        // [total_tokens - target, total_tokens) - not of the padded row. The engine output is
        // [codebooks][tokens][vocab], the caller wants [target][codebooks][vocab].
        void crop(const OmniVoiceRknnStepInputs & in, float * out, int64_t row_capacity) const {
            const int64_t target = in.target_frames;
            const int64_t begin = in.total_tokens - target;
            if (front_shard) {
                // The hidden rows arrive packed on [0, total) and [0, target) and right-padded,
                // in [row][tokens][hidden] order. The host tail graph runs packed too: the
                // conditional row at [0, total) and the unconditional row right behind it at
                // [row_capacity, row_capacity + target) rows (the caller passes this request's
                // total_tokens as row_capacity and zero-pads the rest).
                const int64_t h = hidden_size;
                std::memcpy(out, last_output.data(),
                            static_cast<size_t>(in.total_tokens) * h * sizeof(float));
                std::memcpy(
                    out + static_cast<size_t>(row_capacity) * h,
                    last_output.data() + static_cast<size_t>(tokens) * h,
                    static_cast<size_t>(target) * h * sizeof(float));
                return;
            }
            if (output_row_major_codebook) {
                for (int64_t codebook = 0; codebook < codebooks; ++codebook) {
                    const float * from =
                        last_output.data() + static_cast<size_t>(codebook * tokens + begin) * vocab_size;
                    for (int64_t frame = 0; frame < target; ++frame) {
                        std::memcpy(
                            out + static_cast<size_t>(frame * codebooks + codebook) * vocab_size,
                            from + static_cast<size_t>(frame) * vocab_size,
                            static_cast<size_t>(vocab_size) * sizeof(float));
                    }
                }
                return;
            }
            // NHWC relabel [1, tokens, vocab, codebooks]: the vocab block is strided, so the
            // copy is element-wise. One step is codebooks * target * vocab floats, tens of
            // thousands, and it runs once per diffusion step.
            for (int64_t frame = 0; frame < target; ++frame) {
                const float * row =
                    last_output.data() + static_cast<size_t>(begin + frame) * vocab_size * codebooks;
                for (int64_t codebook = 0; codebook < codebooks; ++codebook) {
                    float * to = out + static_cast<size_t>(frame * codebooks + codebook) * vocab_size;
                    for (int64_t vocab = 0; vocab < vocab_size; ++vocab) {
                        to[vocab] = row[vocab * codebooks + codebook];
                    }
                }
            }
        }

        std::vector<float> last_output;
        // Engine output layout, read at load: true for the declared [1,codebooks,tokens,vocab],
        // false for the converter's NHWC relabel [1,tokens,vocab,codebooks].
        bool output_row_major_codebook = true;
#endif
    };

    // Held by pointer: Engine owns a rknn_context and is neither copyable nor movable, so the
    // ladder has to be a vector of slots, not of slots by value.
    std::vector<std::unique_ptr<Slot>> slots;
    int64_t codebooks = 0;
    int64_t vocab_size = 0;
    int64_t head_dim = 0;
    float rope_theta = 0.0F;
    int64_t hidden_size = 0;
    bool front_shard = false;
#if defined(ENGINE_HAVE_RKNN)
    // Resident-set accounting for the ladder, in the driver's own units (see kAperturePinBudgetMiB).
    engine::rknn::ApertureBudget budget_{kAperturePinBudgetMiB};

    void register_ladder(const std::vector<std::string> & engine_paths, int64_t n_codebooks,
                         int64_t vocab, int64_t n_head_dim, float n_rope_theta,
                         int64_t n_hidden_size) {
        codebooks = n_codebooks;
        vocab_size = vocab;
        head_dim = n_head_dim;
        rope_theta = n_rope_theta;
        hidden_size = n_hidden_size;
        for (const auto & path : engine_paths) {
            if (path.empty()) {
                continue;
            }
            const int64_t tokens = name_seq_len(path);
            if (tokens <= 0) {
                throw std::runtime_error(
                    std::string(kLabel) + ": " + path +
                    " does not name the length it covers; the producer names engines "
                    "omnivoice_generator_seq<tokens>_fp16.rknn and the length is read from the name so a "
                    "ladder can be chosen without loading all of it");
            }
            // Built in place: Slot owns an Engine, so it is neither copyable nor movable.
            auto slot = std::make_unique<Slot>();
            slot->path = path;
            slot->tokens = tokens;
            slot->codebooks = n_codebooks;
            slot->vocab_size = vocab;
            // Front-shard ladders carry `front<k>` between `generator` and `seq` in the producer's
            // name (omnivoice_generator_front28_seq512_fp16.rknn); the whole ladder is one
            // contract, and the first engine's name decides it (load() re-validates per engine).
            if (slots.empty()) {
                front_shard = std::filesystem::path(path).stem().string().find("front") !=
                    std::string::npos;
            }
            slot->front_shard = front_shard;
            // Ask the driver what this engine will cost before deciding to keep it: a model-info-only
            // init parses the image without allocating the network, which is what makes the
            // per-engine table a measurement instead of a guess. If the driver cannot say, the fp16
            // weight file is a tight proxy for this family (the residency is the weights plus a
            // modest workspace).
            slot->aperture_mib = engine::rknn::engine_memory_mib(path, kLabel);
            if (slot->aperture_mib <= 0) {
                std::error_code ec;
                const auto bytes = std::filesystem::file_size(path, ec);
                slot->aperture_mib = ec ? 0 : static_cast<int64_t>(bytes / (1024ull * 1024ull));
            }
            debug::trace_log_scalar("omnivoice.generator.rknn.aperture_mib", slot->aperture_mib);
            slots.push_back(std::move(slot));
        }
        std::sort(slots.begin(), slots.end(), [](const auto & a, const auto & b) {
            return a->tokens < b->tokens;
        });
        if (slots.empty()) {
            throw std::runtime_error(std::string(kLabel) + ": no engine paths given");
        }
        // The board states its own ceiling; 0 means pin every rung the deployment declares.
        const int64_t budget_mib = aperture_budget_mib();
        budget_ = engine::rknn::ApertureBudget(
            budget_mib > 0 ? budget_mib : std::numeric_limits<int64_t>::max());
        debug::trace_log_scalar("omnivoice.generator.rknn.aperture_budget_mib", budget_mib);
    }

    // The smallest engine that fits `tokens`. The long-sequence engines are loaded on demand and
    // evicted on the next length change, because the driver allocates the whole engine (weights
    // and workspace) per context and a 0.6B-parameter fp16 graph costs more than a gigabyte each.
    Slot & pick(int64_t tokens) {
        for (auto & holder : slots) {
            Slot & slot = *holder;
            if (slot.tokens < tokens) {
                continue;
            }
            if (!slot.loaded) {
                // The rung is kept (and the new one pinned) only while the whole resident set
                // stays inside the aperture; a rung above the budget can never share it with
                // anything, so it evicts everything and is itself evicted by the next length change.
                const int64_t cost_mib = slot.aperture_mib;
                const bool fits = cost_mib <= 0 || budget_.fits(cost_mib);
                if (!fits) {
                    for (auto & other : slots) {
                        if (other->loaded) {
                            other->engine.release();
                            other->loaded = false;
                        }
                    }
                    budget_.clear();
                }
                slot.load(codebooks, vocab_size, head_dim, rope_theta, hidden_size);
                if (cost_mib > 0) {
                    budget_.reserve(cost_mib);
                }
                debug::trace_log_scalar("omnivoice.generator.rknn.resident_mib", budget_.resident());
            }
            return slot;
        }
        throw std::runtime_error(
            std::string(kLabel) + ": no engine covers " + std::to_string(tokens) +
            " tokens, the largest is " + std::to_string(slots.back()->tokens));
    }
#endif
};

OmniVoiceRknnGenerator::OmniVoiceRknnGenerator(
    const std::vector<std::string> & engine_paths,
    int64_t codebooks,
    int64_t vocab_size,
    int64_t head_dim,
    float rope_theta,
    int64_t hidden_size)
    : impl_(std::make_unique<Impl>()) {
#if defined(ENGINE_HAVE_RKNN)
    impl_->register_ladder(engine_paths, codebooks, vocab_size, head_dim, rope_theta, hidden_size);
#else
    (void)engine_paths;
    (void)codebooks;
    (void)vocab_size;
    (void)head_dim;
    (void)rope_theta;
    (void)hidden_size;
    throw std::runtime_error("OmniVoice generator RKNN: built without ENGINE_ENABLE_RKNN");
#endif
}

OmniVoiceRknnGenerator::~OmniVoiceRknnGenerator() = default;

int64_t OmniVoiceRknnGenerator::max_tokens() const {
    return impl_->slots.empty() ? 0 : impl_->slots.back()->tokens;
}

int64_t OmniVoiceRknnGenerator::engine_count() const {
    return static_cast<int64_t>(impl_->slots.size());
}

bool OmniVoiceRknnGenerator::front_shard() const {
    return impl_->front_shard;
}

int64_t OmniVoiceRknnGenerator::bucket_for(int64_t total_tokens) const {
#if defined(ENGINE_HAVE_RKNN)
    for (const auto & slot : impl_->slots) {
        if (slot->tokens >= total_tokens) {
            return slot->tokens;
        }
    }
#else
    (void)total_tokens;
#endif
    return 0;
}

void OmniVoiceRknnGenerator::run(
    int64_t bucket,
    const OmniVoiceRknnStepInputs & inputs,
    float * logits_out,
    int64_t row_capacity) {
#if defined(ENGINE_HAVE_RKNN)
    const auto start = Clock::now();
    // The batch shares one bucket (the way the shared graph shares one capacity), so the slot is
    // picked by the bucket, not by this request's own length: a shorter request runs padded.
    auto & slot = impl_->pick(bucket);
    if (slot.tokens != bucket) {
        throw std::runtime_error(
            std::string(kLabel) + ": asked to run at bucket " + std::to_string(bucket) +
            ", the ladder picked " + std::to_string(slot.tokens));
    }
    slot.run(inputs);
    slot.crop(inputs, logits_out, row_capacity);
    debug::timing_log_scalar("omnivoice.generator.rknn.step_ms", debug::elapsed_ms(start));
    debug::trace_log_scalar("omnivoice.generator.rknn.engine_tokens", slot.tokens);
#else
    (void)bucket;
    (void)inputs;
    (void)logits_out;
    (void)row_capacity;
    throw std::runtime_error("OmniVoice generator RKNN: built without ENGINE_HAVE_RKNN");
#endif
}

// The ceiling the ladder evicts to, read once per call: a board states its own aperture with
// `AUDIOCPP_RKNN_APERTURE_MIB` in the spirit of `AUDIOCPP_RKNN_CORE_MASK`, and 0 (also the
// reading of a malformed value, so a typo cannot silently mean one megabyte) turns the accounting
// off and pins every rung the deployment declares.
int64_t aperture_budget_mib() {
    const char * const env = std::getenv("AUDIOCPP_RKNN_APERTURE_MIB");
    if (env == nullptr) {
        return kAperturePinBudgetMiB;
    }
    char * end = nullptr;
    const long long value = std::strtoll(env, &end, 10);
    return value > 0 && *end == '\0' ? static_cast<int64_t>(value) : 0;
}

}  // namespace engine::models::omnivoice
