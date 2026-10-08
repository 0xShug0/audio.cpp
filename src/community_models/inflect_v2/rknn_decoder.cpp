#include "engine/community_models/inflect_v2/rknn_decoder.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/rknn/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ENGINE_HAVE_RKNN)
#include <rknn_api.h>
#endif

namespace engine::models::inflect_v2 {

namespace {

using Clock = std::chrono::steady_clock;

// The window contract of tools/rknn/prepare_inflect_v2_rknn_decoder.py. The engine is
// compiled at `window` slices; every step emits `chunk` of them and keeps `padding`
// real slices of context on each side, which is what makes a window decode
// identically to a full-length decode. 32 is measured, not guessed: the inverse
// flow's WaveNet couplings reach back ~23 slices, and at 480 slices a window scores
// 5.1 dB against a full-length decode with no context, 40.0 dB at 20, 82.1 dB at 24
// and 128.8 dB at 32.
constexpr int64_t kChunk = 192;
constexpr int64_t kPadding = 32;
// Seam stitching: compare this many samples of the previous window's tail against a
// search window in the next one, cut at the closest match, then blend the compare
// window. The search runs in steps of 2 samples.
constexpr int64_t kDepopCompare = 24;
constexpr int64_t kDepopSearch = 44;

#if defined(ENGINE_HAVE_RKNN)
// Prefix of every message this sidecar raises, and the label the shared engine
// plumbing stamps on the ones it raises on its behalf.
constexpr const char * kLabel = "Inflect v2 decoder RKNN";
#endif

}  // namespace

struct InflectV2RknnDecoder::Impl {
    int64_t window = 0;
#if defined(ENGINE_HAVE_RKNN)
    engine::rknn::Engine engine;
    int64_t step = 0;  // audio samples per latent slice
    std::vector<float> latent;  // [channels][window]
    std::vector<float> mask;    // [window]
    std::vector<float> audio;

    void load(const std::string & path, int64_t channels, int64_t samples_per_slice) {
        // The shared engine reads the image, inits the context, picks the cores
        // (all three by name, AUTO fallback, AUDIOCPP_RKNN_CORE_MASK override) and
        // caches every input/output attribute.
        engine.load(path, kLabel);

        if (engine.input_count() != 2 || engine.output_count() != 1) {
            throw std::runtime_error(
                "Inflect v2 decoder RKNN: expected the decoder slice (z + y_mask), got " +
                std::to_string(engine.input_count()) + " inputs / " +
                std::to_string(engine.output_count()) + " outputs");
        }
        const rknn_tensor_attr & input = engine.input_attr(0);
        const rknn_tensor_attr & mask_attr = engine.input_attr(1);
        const rknn_tensor_attr & output = engine.output_attr(0);
        debug::timing_log_scalar(
            "inflect_v2.decoder.rknn.io",
            std::string(input.name) + " type=" + std::to_string(input.type) +
                " elems=" + std::to_string(input.n_elems) +
                " + " + std::string(mask_attr.name) +
                " elems=" + std::to_string(mask_attr.n_elems) +
                " -> " + std::string(output.name) + " elems=" + std::to_string(output.n_elems));
        if (std::string(input.name) != "z" || std::string(mask_attr.name) != "y_mask") {
            throw std::runtime_error(
                "Inflect v2 decoder RKNN: model " + path + " inputs are '" +
                std::string(input.name) + "'/'" + std::string(mask_attr.name) +
                "', expected 'z'/'y_mask'; compile it with "
                "tools/rknn/prepare_inflect_v2_rknn_decoder.py");
        }
        if (input.n_dims != 3 || static_cast<int64_t>(input.dims[1]) != channels) {
            throw std::runtime_error(
                "Inflect v2 decoder RKNN: model " + path + " takes " +
                (input.n_dims == 3 ? std::to_string(input.dims[1]) : std::string{"?"}) +
                " latent channels, this model has " + std::to_string(channels) +
                "; compile the engine from this model "
                "(tools/rknn/prepare_inflect_v2_rknn_decoder.py)");
        }
        window = static_cast<int64_t>(input.dims[2]);
        // Exact, not a lower bound. A narrower engine would let `run_window` copy a
        // 256-slice window into a narrower row: overrun into the next channel and past
        // the end of the latent buffer on the last one.
        if (window != kChunk + 2 * kPadding) {
            throw std::runtime_error(
                "Inflect v2 decoder RKNN: model " + path + " has a " + std::to_string(window) +
                "-slice window, expected " + std::to_string(kChunk) + " emitted + 2*" +
                std::to_string(kPadding) + " context = " + std::to_string(kChunk + 2 * kPadding) +
                "; compile it with tools/rknn/prepare_inflect_v2_rknn_decoder.py --window " +
                std::to_string(kChunk + 2 * kPadding));
        }
        if (mask_attr.n_elems != static_cast<uint32_t>(window)) {
            throw std::runtime_error(
                "Inflect v2 decoder RKNN: model " + path + " masks " +
                std::to_string(mask_attr.n_elems) + " frames per " + std::to_string(window) +
                "-slice window");
        }
        step = static_cast<int64_t>(output.n_elems) / window;
        if (step * window != static_cast<int64_t>(output.n_elems) || step != samples_per_slice) {
            throw std::runtime_error(
                "Inflect v2 decoder RKNN: model " + path + " voices " +
                std::to_string(output.n_elems) + " samples per " + std::to_string(window) +
                "-slice window, this model expects " + std::to_string(samples_per_slice) +
                " samples per latent slice");
        }
        latent.assign(static_cast<size_t>(channels * window), 0.0F);
        mask.assign(static_cast<size_t>(window), 0.0F);
        audio.assign(static_cast<size_t>(output.n_elems), 0.0F);
        debug::timing_log_scalar("inflect_v2.decoder.rknn.window", window);
    }

    // One window: `slice_count` slices of `source` (row-major [channels][frames]) starting at
    // slice `lo`, zero-padded to the compiled width and masked off over the padding, voiced into
    // `audio`.
    void run_window(
        const float * source,
        int64_t channels,
        int64_t frames,
        int64_t lo,
        int64_t slice_count) {
        std::fill(latent.begin(), latent.end(), 0.0F);
        for (int64_t channel = 0; channel < channels; ++channel) {
            std::copy_n(
                source + channel * frames + lo,
                slice_count,
                latent.begin() + channel * window);
        }
        // 1 over the real slices, 0 over the padding: the flow multiplies its activations by the
        // mask after every convolution, and the sliced graph carries the same muls through the
        // decoder half, so a padded frame marked 1 propagates its conv biases into the real
        // frames.
        std::fill(mask.begin(), mask.end(), 0.0F);
        std::fill_n(mask.begin(), static_cast<size_t>(slice_count), 1.0F);

        // The buffers are fp32 and pass_through = 0, so each size is the element count
        // times sizeof(float); the engine's native (fp16) width is not the expected
        // size here (same as the piper_tts sidecar).
        std::vector<engine::rknn::Input> inputs;
        inputs.push_back(
            {0, latent.data(), static_cast<uint32_t>(latent.size() * sizeof(float)),
             RKNN_TENSOR_FLOAT32, 0});
        inputs.push_back(
            {1, mask.data(), static_cast<uint32_t>(mask.size() * sizeof(float)),
             RKNN_TENSOR_FLOAT32, 0});
        std::vector<std::vector<float>> outputs;
        engine.run(inputs, outputs);
        const std::vector<float> & out = outputs[0];
        const size_t count = std::min(audio.size(), out.size());
        std::copy_n(out.begin(), count, audio.begin());
        if (count != audio.size()) {
            throw std::runtime_error(
                "Inflect v2 decoder RKNN: window returned " + std::to_string(count) +
                " samples, expected " + std::to_string(audio.size()));
        }
    }
#endif
};

InflectV2RknnDecoder::InflectV2RknnDecoder(
    const std::string & model_path,
    int64_t channels,
    int64_t samples_per_slice)
    : impl_(std::make_unique<Impl>()) {
#if defined(ENGINE_HAVE_RKNN)
    impl_->load(model_path, channels, samples_per_slice);
#else
    (void)model_path;
    (void)channels;
    (void)samples_per_slice;
    throw std::runtime_error("Inflect v2 decoder RKNN: built without ENGINE_ENABLE_RKNN");
#endif
}

InflectV2RknnDecoder::~InflectV2RknnDecoder() = default;

int64_t InflectV2RknnDecoder::window() const { return impl_->window; }

int64_t InflectV2RknnDecoder::chunk() const { return kChunk; }

std::vector<float> InflectV2RknnDecoder::decode(
    const float * latent,
    int64_t channels,
    int64_t frames) {
#if defined(ENGINE_HAVE_RKNN)
    auto & impl = *impl_;
    const auto start = Clock::now();
    std::vector<float> out;
    out.reserve(static_cast<size_t>(frames * impl.step));
    int64_t windows = 0;
    for (int64_t emitted = 0; emitted < frames; emitted += kChunk) {
        // The window that reaches the end of the utterance emits to the end, and extends its
        // left context to fill the compiled shape: the flow's receptive field reaches back
        // further than kPadding.
        const bool tail = emitted + kChunk + kPadding >= frames;
        const int64_t lo = std::max(
            int64_t{0}, std::min(emitted - kPadding, frames - impl.window));
        const int64_t hi = tail ? frames : std::min(frames, emitted + kChunk + kPadding);
        const int64_t slice_count = hi - lo;
        const int64_t end_pad = tail ? 0 : kPadding;
        impl.run_window(latent, channels, frames, lo, slice_count);
        int64_t real_start = (emitted - lo) * impl.step;
        const int64_t real_end = (slice_count - end_pad) * impl.step;
        if (real_end <= real_start) {
            throw std::runtime_error("Inflect v2 decoder RKNN: empty window emission");
        }
        if (out.size() >= static_cast<size_t>(kDepopCompare) &&
            real_end - real_start >= kDepopSearch * 2) {
            // Stitch: cut where the previous tail best matches this window, then blend
            // the compare window so the seam carries no step.
            const float * prev = out.data() + out.size() - kDepopCompare;
            const int64_t base = real_start - std::min(real_start, kDepopCompare);
            int64_t best_start = real_start;
            double best_diff = -1.0;
            for (int64_t j = 0; j < kDepopSearch * 2; j += 2) {
                if (base + j + kDepopCompare > static_cast<int64_t>(impl.audio.size())) {
                    break;
                }
                double diff = 0.0;
                for (int64_t k = 0; k < kDepopCompare; ++k) {
                    diff += std::abs(prev[k] - impl.audio[static_cast<size_t>(base + j + k)]);
                }
                if (best_diff < 0.0 || diff < best_diff) {
                    best_diff = diff;
                    best_start = base + j + kDepopCompare;
                }
            }
            real_start = best_start;
            for (int64_t k = 0; k < kDepopCompare; ++k) {
                const float weight = static_cast<float>(k) / static_cast<float>(kDepopCompare);
                out[out.size() - kDepopCompare + static_cast<size_t>(k)] =
                    out[out.size() - kDepopCompare + static_cast<size_t>(k)] * (1.0F - weight) +
                    impl.audio[static_cast<size_t>(real_start - kDepopCompare + k)] * weight;
            }
        }
        out.insert(
            out.end(),
            impl.audio.begin() + real_start,
            impl.audio.begin() + real_end);
        ++windows;
        if (tail) {
            break;
        }
    }
    debug::timing_log_scalar("inflect_v2.decoder.rknn.windows", windows);
    debug::timing_log_scalar(
        "inflect_v2.decoder.rknn.step_ms",
        debug::elapsed_ms(start));
    return out;
#else
    (void)latent;
    (void)channels;
    (void)frames;
    throw std::runtime_error("Inflect v2 decoder RKNN: built without ENGINE_ENABLE_RKNN");
#endif
}

}  // namespace engine::models::inflect_v2
