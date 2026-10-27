// The depth decoder (one MaskGIT pass) and lfm3_maskgit_frame. On the
// synthetic package: pass logits against a host fp64 reference and greedy
// MaskGIT frames (EOS rule and early stop) against the reference's three full
// passes.
//
//   lfm3_audio_depth_test [--backend cpu|cuda|metal|best] [--threads N]
//
// A requested backend that is not available exits 77.

#include "lfm3_audio_reference.h"
#include "lfm3_audio_test_package.h"
#include "test_assert.h"

#include "engine/community_models/lfm3_audio/assets.h"
#include "engine/community_models/lfm3_audio/depth.h"
#include "engine/community_models/lfm3_audio/sampling.h"
#include "engine/framework/core/execution_context.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {

using namespace engine::community_models::lfm3_audio;
namespace core = engine::core;
namespace ref = lfm3_audio_test;
using engine::test::require;
using ref::report;
using Masked = std::array<bool, 12>;

constexpr int64_t kCodebooks = 12;
// Native weights: the CPU rounds activations to the weight type; max(2e-2, 2x the bf16 floor of 0.49%).
constexpr double kNativeTolerance = 2e-2;

int32_t first_argmax(const float * row, size_t count) {
    return static_cast<int32_t>(std::max_element(row, row + count) - row);
}

template <typename T>
int32_t first_argmax(const std::vector<T> & values, size_t offset, size_t count) {
    size_t best = offset;
    for (size_t i = offset + 1; i < offset + count; ++i) {
        if (values[i] > values[best]) best = i;
    }
    return static_cast<int32_t>(best - offset);
}

struct RefDepth {
    Lfm3DepthConfig config;
    ref::Values input_weight, input_bias, mask, embeddings, norms;  // embeddings [12 * vocab][d], norms [12][d]
    ref::RefStack stack;
};

RefDepth read_ref_depth(const engine::assets::TensorSource & source, const Lfm3DepthConfig & config) {
    RefDepth out{config, ref::to_values(source.require_f32("depth_linear.weight")), ref::to_values(source.require_f32("depth_linear.bias")),
                 ref::to_values(source.require_f32("mask_embedding")), {}, {}, ref::read_ref_stack(source, "depthformer.model.", config.stack)};
    for (int64_t i = 0; i < kCodebooks; ++i) {
        const std::string p = "depth_embeddings." + std::to_string(i) + ".";
        const auto e = source.require_f32(p + "embedding.weight");
        const auto n = source.require_f32(p + "embedding_norm.weight");
        out.embeddings.insert(out.embeddings.end(), e.begin(), e.end());
        out.norms.insert(out.norms.end(), n.begin(), n.end());
    }
    return out;
}

ref::Values ref_pass(const RefDepth & w, const std::vector<float> & hidden, const Lfm3Codes & codes, const Masked & masked) {
    const int64_t d = w.config.stack.hidden;
    const int64_t vocab = w.config.audio_vocab_size;
    auto x = ref::linear(ref::to_values(hidden), w.input_weight, w.config.input_size, kCodebooks * d);
    for (int64_t i = 0; i < kCodebooks; ++i) {
        for (int64_t c = 0; c < d; ++c) {
            const auto k = static_cast<size_t>(i);
            x[static_cast<size_t>(i * d + c)] +=
                w.input_bias[static_cast<size_t>(i * d + c)] +
                (masked[k] ? w.mask[static_cast<size_t>(c)] : w.embeddings[static_cast<size_t>((i * vocab + codes[k]) * d + c)]);
        }
    }
    const auto o = ref::stack(w.config.stack, w.stack, x, ref::positions_from(0, kCodebooks));
    ref::Values logits(static_cast<size_t>(kCodebooks * vocab));
    for (int64_t i = 0; i < kCodebooks; ++i) {
        const ref::Values oi(o.begin() + i * d, o.begin() + (i + 1) * d);
        const ref::Values ni(w.norms.begin() + i * d, w.norms.begin() + (i + 1) * d);
        const auto n = ref::rms_norm(oi, ni, d, w.config.stack.eps);
        for (int64_t v = 0; v < vocab; ++v) {
            double sum = 0.0;
            for (int64_t c = 0; c < d; ++c) sum += w.embeddings[static_cast<size_t>((i * vocab + v) * d + c)] * n[static_cast<size_t>(c)];
            logits[static_cast<size_t>(i * vocab + v)] = sum;
        }
    }
    return logits;
}

// The upstream frame: all passes always run; any end-of-audio code makes the frame all end-of-audio.
// eos_pass is the first pass that picked it (-1: none).
Lfm3Codes ref_maskgit(const RefDepth & w, const std::vector<float> & hidden, const Lfm3Codes & order, int & eos_pass) {
    const int64_t vocab = w.config.audio_vocab_size;
    const int32_t eos = w.config.end_of_audio();
    Lfm3Codes codes{};
    Masked masked;
    masked.fill(true);
    eos_pass = -1;
    size_t next = 0;
    for (size_t p = 0; p < w.config.maskgit_schedule.size(); ++p) {
        const auto logits = ref_pass(w, hidden, codes, masked);
        for (int64_t i = 0; i < w.config.maskgit_schedule[p]; ++i, ++next) {
            const auto k = static_cast<size_t>(order[next]);
            codes[k] = first_argmax(logits, k * static_cast<size_t>(vocab), static_cast<size_t>(vocab));
            masked[k] = false;
            if (codes[k] == eos && eos_pass < 0) eos_pass = static_cast<int>(p);
        }
    }
    if (eos_pass >= 0) codes.fill(eos);
    return codes;
}

Lfm3AudioSampling greedy() {
    Lfm3AudioSampling out;
    out.temperature = 0.0f;
    out.repetition_penalty = 1.0f;
    return out;
}

void test_synthetic(core::ExecutionContext & execution) {
    const auto package = ref::write_package("lfm3_audio_depth_test");
    const auto assets = load_lfm3_audio_assets(package.root);
    const auto & config = assets->depth;
    const auto reference = read_ref_depth(*assets->model, config);
    const int64_t vocab = config.audio_vocab_size;
    lfm2_audio_test::Random random(7);

    Lfm3DepthDecoderRuntime depth(assets->model, config, execution, {Lfm3WeightType::F32, true});
    Lfm3DepthDecoderRuntime native(assets->model, config, execution, {Lfm3WeightType::Native, true});
    const double tolerance = ref::reference_tolerance(execution);
    const bool gpu = execution.backend_type() != core::BackendType::Cpu;

    // Patterns: all masked, none masked, mixed.
    for (int pattern = 0; pattern < 3; ++pattern) {
        const auto hidden = random.uniform(static_cast<size_t>(config.input_size), 1.0f);
        Lfm3Codes codes;
        Masked masked;
        for (size_t k = 0; k < codes.size(); ++k) {
            codes[k] = static_cast<int32_t>(std::abs(random.uniform(1.0f)) * static_cast<float>(vocab - 1));
            masked[k] = pattern == 0 || (pattern == 2 && k % 3 == 1);
        }
        const auto expected = ref_pass(reference, hidden, codes, masked);
        report("synthetic pass logits f32, pattern " + std::to_string(pattern), ref::rel_error(depth.logits(hidden, codes, masked), expected),
               tolerance);
        report("synthetic pass logits native (bf16), pattern " + std::to_string(pattern),
               ref::rel_error(native.logits(hidden, codes, masked), expected), kNativeTolerance);
    }
    // Advances to the hidden states whose 120 MaskGIT frames below cover both outcomes and an early stop.
    (void)random.uniform(static_cast<size_t>(config.input_size), 1.0f);

    {
        const auto schedule = config.maskgit_schedule;
        Lfm3AudioSampler orders({}, 11);
        int eos_frames = 0;
        int early = 0;
        int mismatches = 0;
        constexpr int kFrames = 120;
        for (int f = 0; f < kFrames; ++f) {
            const auto hidden = random.uniform(static_cast<size_t>(config.input_size), 1.0f);
            const auto order = orders.next_order();
            Lfm3AudioSampler sampler(greedy(), 0);
            Lfm3FrameTaps taps;
            const auto codes = lfm3_maskgit_frame(depth, hidden, order, schedule, sampler, config.end_of_audio(), &taps);
            const bool eos = codes[0] == config.end_of_audio();
            size_t next = 0;
            for (size_t p = 0; p < taps.pass_logits.size(); ++p) {
                for (int64_t i = 0; i < schedule[p] && next < order.size(); ++i, ++next) {
                    const auto k = static_cast<size_t>(order[next]);
                    const auto pick = first_argmax(taps.pass_logits[p].data() + k * static_cast<size_t>(vocab), static_cast<size_t>(vocab));
                    if (pick == config.end_of_audio()) {
                        require(eos && p + 1 == taps.pass_logits.size(), "the frame must stop at its end-of-audio pass");
                        next = order.size();
                        break;
                    }
                    require(eos || pick == codes[k], "a pick must be its pass's argmax");
                }
            }
            require(eos || taps.pass_logits.size() == schedule.size(), "a frame without end-of-audio runs every pass");
            eos_frames += eos;
            early += eos && taps.pass_logits.size() < schedule.size();
            if (!gpu) {
                int eos_pass = -1;
                const auto expected = ref_maskgit(reference, hidden, order, eos_pass);
                mismatches += codes != expected;
                require(!eos || eos_pass + 1 == static_cast<int>(taps.pass_logits.size()), "end-of-audio pass differs from the reference");
            }
        }
        std::printf("PASS synthetic greedy MaskGIT: %d frames, %d end-of-audio (%d stopped early), %d reference mismatches%s\n", kFrames,
                    eos_frames, early, mismatches, gpu ? " (reference compared on CPU only)" : "");
        require(eos_frames > 0 && eos_frames < kFrames && early > 0, "the frames must cover both outcomes and an early stop");
        require(mismatches == 0, "greedy MaskGIT must reproduce the reference codes");
    }

    {
        const std::vector<float> hidden(static_cast<size_t>(config.input_size), 0.1f);
        Lfm3AudioSampler sampler(greedy(), 0);
        Lfm3Codes order;
        std::iota(order.begin(), order.end(), 0);
        auto bad_order = order;
        bad_order[3] = 2;
        ref::require_throws([&] { lfm3_maskgit_frame(depth, hidden, bad_order, config.maskgit_schedule, sampler, config.end_of_audio()); },
                            "repeated codebook in the order");
        ref::require_throws([&] { lfm3_maskgit_frame(depth, hidden, order, {4, 4, 3}, sampler, config.end_of_audio()); },
                            "schedule not summing to 12");
        Lfm3Codes codes{};
        codes[2] = static_cast<int32_t>(vocab);
        ref::require_throws([&] { (void)depth.logits(hidden, codes, Masked{}); }, "unmasked code out of range");
        ref::require_throws([&] { (void)depth.logits(std::vector<float>(3, 0.0f), Lfm3Codes{}, Masked{}); }, "wrong hidden size");
        std::printf("PASS error checks\n");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    int status = 0;
    const auto execution = ref::open_backend(argc, argv, status);
    if (!execution) {
        return status;
    }

    try {
        test_synthetic(*execution);
        std::printf("PASS: lfm3_audio depth (synthetic)\n");
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
