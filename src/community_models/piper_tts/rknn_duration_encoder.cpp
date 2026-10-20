#include "engine/community_models/piper_tts/rknn_duration_encoder.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/rknn/engine.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(ENGINE_HAVE_RKNN)
#include <rknn_api.h>
#endif

namespace engine::models::piper_tts {

namespace {

using Clock = std::chrono::steady_clock;

#if defined(ENGINE_HAVE_RKNN)
// Prefix of every message this sidecar raises, and the label the shared engine
// plumbing stamps on the ones it raises on its behalf.
constexpr const char * kLabel = "Piper TTS duration encoder RKNN";
#endif

}  // namespace

struct PiperTtsRknnDurationEncoder::Impl {
#if defined(ENGINE_HAVE_RKNN)
    struct Bin {
        std::unique_ptr<engine::rknn::Engine> engine;  // Engine is not movable
        int64_t tokens = 0;                           // compiled token width
    };

    int64_t channels = 0;  // inter_channels; stats carry 2*channels rows
    std::vector<Bin> bins;  // ascending by tokens
    std::vector<int64_t> fed;      // [tokens] padded int64 tokens
    std::vector<int64_t> length;   // [1]
    std::vector<float> stats_out;
    std::vector<float> conditioning_out;

    void load(const std::string & model_paths, int64_t inter_channels) {
        channels = inter_channels;
        std::istringstream stream(model_paths);
        std::string path;
        while (std::getline(stream, path, ',')) {
            const auto at = path.find_first_not_of(" \t");
            if (at == std::string::npos) {
                continue;
            }
            path = path.substr(at, path.find_last_not_of(" \t") - at + 1);
            auto bin = Bin{std::make_unique<engine::rknn::Engine>(), 0};
            bin.engine->load(path, kLabel);
            const engine::rknn::Engine & engine = *bin.engine;
            if (engine.input_count() != 2 || engine.output_count() != 2) {
                throw std::runtime_error(
                    "Piper TTS duration encoder RKNN: model " + path +
                    " has " + std::to_string(engine.input_count()) + " inputs / " +
                    std::to_string(engine.output_count()) +
                    " outputs, expected input + input_lengths and stats + conditioning; "
                    "compile it with tools/prepare_piper_tts_rknn_vocoder.py --encoder");
            }
            const rknn_tensor_attr & input = engine.input_attr(0);
            const rknn_tensor_attr & length_attr = engine.input_attr(1);
            if (std::string(input.name) != "input" ||
                std::string(length_attr.name) != "input_lengths") {
                throw std::runtime_error(
                    "Piper TTS duration encoder RKNN: model " + path + " inputs are '" +
                    std::string(input.name) + "'/'" + std::string(length_attr.name) +
                    "', expected 'input'/'input_lengths'; compile it with "
                    "tools/prepare_piper_tts_rknn_vocoder.py --encoder");
            }
            bin.tokens = static_cast<int64_t>(input.n_elems);
            const rknn_tensor_attr & stats = engine.output_attr(0);
            const rknn_tensor_attr & conditioning = engine.output_attr(1);
            if (stats.n_elems !=
                    static_cast<uint32_t>(2 * channels * bin.tokens) ||
                conditioning.n_elems != static_cast<uint32_t>(channels * bin.tokens)) {
                throw std::runtime_error(
                    "Piper TTS duration encoder RKNN: model " + path + " emits " +
                    std::to_string(stats.n_elems) + " + " +
                    std::to_string(conditioning.n_elems) + " values for a " +
                    std::to_string(bin.tokens) + "-token width, this voice has " +
                    std::to_string(2 * channels) + " + " + std::to_string(channels) +
                    " rows; compile the engine from this voice "
                    "(tools/prepare_piper_tts_rknn_vocoder.py --encoder)");
            }
            bins.push_back(std::move(bin));
        }
        if (bins.empty()) {
            throw std::runtime_error(
                "Piper TTS duration encoder RKNN: no engine paths given");
        }
        std::sort(bins.begin(), bins.end(), [](const Bin & a, const Bin & b) {
            return a.tokens < b.tokens;
        });
        for (size_t index = 1; index < bins.size(); ++index) {
            if (bins[index].tokens == bins[index - 1].tokens) {
                throw std::runtime_error(
                    "Piper TTS duration encoder RKNN: two engines cover the same " +
                    std::to_string(bins[index].tokens) + "-token width");
            }
        }
        length.assign(1, 0);
    }
#endif
};

PiperTtsRknnDurationEncoder::PiperTtsRknnDurationEncoder(
    const std::string & model_paths,
    int64_t inter_channels)
    : impl_(std::make_unique<Impl>()) {
#if defined(ENGINE_HAVE_RKNN)
    impl_->load(model_paths, inter_channels);
#else
    (void)model_paths;
    (void)inter_channels;
    throw std::runtime_error("Piper TTS duration encoder RKNN: built without ENGINE_ENABLE_RKNN");
#endif
}

PiperTtsRknnDurationEncoder::~PiperTtsRknnDurationEncoder() = default;

int64_t PiperTtsRknnDurationEncoder::max_tokens() const {
#if defined(ENGINE_HAVE_RKNN)
    return impl_->bins.empty() ? 0 : impl_->bins.back().tokens;
#else
    return 0;
#endif
}

void PiperTtsRknnDurationEncoder::encode(
    const std::vector<int32_t> & token_ids,
    std::vector<float> & stats,
    std::vector<float> & conditioning) {
#if defined(ENGINE_HAVE_RKNN)
    auto & impl = *impl_;
    const int64_t tokens = static_cast<int64_t>(token_ids.size());
    auto bin = std::lower_bound(
        impl.bins.begin(),
        impl.bins.end(),
        tokens,
        [](const Impl::Bin & b, int64_t n) { return b.tokens < n; });
    if (bin == impl.bins.end()) {
        throw std::runtime_error(
            "Piper TTS duration encoder RKNN: " + std::to_string(tokens) +
            " tokens exceed the widest compiled engine (" +
            std::to_string(impl.bins.back().tokens) + ")");
    }
    const auto start = Clock::now();
    impl.fed.assign(static_cast<size_t>(bin->tokens), 0);
    std::copy(token_ids.begin(), token_ids.end(), impl.fed.begin());
    impl.length[0] = tokens;

    // int64 inputs ride pass_through (the qwen3_tts codes precedent): the engine's
    // native width for an index tensor is not a float conversion target.
    std::vector<engine::rknn::Input> inputs = {
        {0,
         impl.fed.data(),
         static_cast<uint32_t>(impl.fed.size() * sizeof(int64_t)),
         RKNN_TENSOR_INT64,
         1},
        {1,
         impl.length.data(),
         static_cast<uint32_t>(impl.length.size() * sizeof(int64_t)),
         RKNN_TENSOR_INT64,
         1},
    };
    std::vector<std::vector<float>> outputs;
    bin->engine->run(inputs, outputs);

    // De-pad the compiled width to the real token count: the outputs are
    // channel-major ([channels][width]), the runtime's vectors are [channels][tokens].
    const int64_t width = bin->tokens;
    stats.assign(static_cast<size_t>(2 * impl.channels * tokens), 0.0F);
    conditioning.assign(static_cast<size_t>(impl.channels * tokens), 0.0F);
    for (int64_t channel = 0; channel < 2 * impl.channels; ++channel) {
        std::copy_n(
            outputs[0].begin() + channel * width,
            tokens,
            stats.begin() + channel * tokens);
    }
    for (int64_t channel = 0; channel < impl.channels; ++channel) {
        std::copy_n(
            outputs[1].begin() + channel * width,
            tokens,
            conditioning.begin() + channel * tokens);
    }
    debug::timing_log_scalar("piper_tts.duration.rknn.bin", width);
    debug::timing_log_scalar("piper_tts.duration.rknn.step_ms", debug::elapsed_ms(start));
#else
    (void)token_ids;
    (void)stats;
    (void)conditioning;
    throw std::runtime_error("Piper TTS duration encoder RKNN: built without ENGINE_ENABLE_RKNN");
#endif
}

}  // namespace engine::models::piper_tts
