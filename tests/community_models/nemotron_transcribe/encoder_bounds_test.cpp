#include "engine/community_models/nemotron_transcribe/encoder.h"
#include <ggml-backend-impl.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>

using namespace engine::models::nemotron_transcribe;

namespace {
void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}
void rejects(const std::function<void()> & action, const char * message) {
    bool rejected = false;
    try { action(); } catch (const std::runtime_error &) { rejected = true; }
    require(rejected, message);
}
void bad_config(const std::function<void(EncoderConfig &)> & change) {
    EncoderConfig config;
    change(config);
    rejects([&] { validate_encoder_config(config); }, "invalid encoder configuration was accepted");
}

// Self-generated weights exercise feature stacking, the speaker cache, and
// optionally one attention layer per expert without a downloaded model fixture.
class CacheWeights final : public engine::assets::TensorSource {
public:
    explicit CacheWeights(bool attention = false) {
        for (const auto * prefix : {"asr", "diar"}) {
            add(std::string(prefix) + ".pre_encode.proj.weight", {16, 8});
            for (const auto * suffix : {"embed_norm", "final_norm"}) norm(std::string(prefix) + "." + suffix, 16);
            if (attention) {
                const std::string layer = std::string(prefix) + ".layers.0";
                norm(layer + ".norm1", 16);
                norm(layer + ".norm2", 16);
                norm(layer + ".attn.q_norm", 8);
                norm(layer + ".attn.k_norm", 8);
                add(layer + ".attn.w_qkv.weight", {48, 16});
                linear(layer + ".attn.out_proj", 16, 16);
                linear(layer + ".ffn.net.0", 32, 16);
                linear(layer + ".ffn.net.3", 16, 32);
            }
        }
        linear("diar.head.encoder_proj", 8, 16);
        add("diar.head.subpixel_upsample.weight", {16, 8, 3});
        add("diar.head.subpixel_upsample.bias", {16});
        linear("diar.head.first_hidden_to_hidden", 8, 8);
        linear("diar.head.single_hidden_to_spks", 2, 8);
        entries_.at("diar.head.single_hidden_to_spks.bias").values = {-0.5f, 0.5f};
        add("diar.head.learnable_sil_emb", {16});
        linear("fusion.proj", 12, 16);
        norm("fusion.asr_norm", 16);
        norm("fusion.diar_norm", 2);
        add("fusion.diar_kernel", {2, 16});
        if (attention) {
            for (auto & entry : entries_) {
                if (entry.second.metadata.shape.size() < 2) continue;
                for (size_t i = 0; i < entry.second.values.size(); ++i)
                    entry.second.values[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.03f;
            }
        }
    }
    const std::filesystem::path & source_path() const noexcept override { return path_; }
    bool has_tensor(std::string_view name) const noexcept override { return entries_.count(std::string(name)) != 0; }
    engine::assets::TensorMetadata require_metadata(std::string_view name) const override { return entries_.at(std::string(name)).metadata; }
    std::vector<engine::assets::TensorMetadata> tensors() const override {
        std::vector<engine::assets::TensorMetadata> result;
        for (const auto & entry : entries_) result.push_back(entry.second.metadata);
        return result;
    }
    engine::assets::RawTensorData require_tensor_data(std::string_view name) const override {
        const auto & entry = entries_.at(std::string(name));
        std::vector<std::byte> bytes(entry.values.size() * sizeof(float));
        std::memcpy(bytes.data(), entry.values.data(), bytes.size());
        return {entry.metadata, std::move(bytes)};
    }
    std::vector<float> require_f32(std::string_view name,
        const std::optional<std::vector<int64_t>> & shape = std::nullopt) const override {
        const auto & entry = entries_.at(std::string(name));
        require(!shape || *shape == entry.metadata.shape, "cache fixture shape mismatch");
        return entry.values;
    }
    std::optional<std::vector<float>> optional_f32(std::string_view name,
        const std::optional<std::vector<int64_t>> & shape = std::nullopt) const override {
        return has_tensor(name) ? std::optional<std::vector<float>>(require_f32(name, shape)) : std::nullopt;
    }
    int64_t require_i64_scalar(std::string_view) const override { throw std::runtime_error("no scalar fixture"); }
private:
    struct Entry { engine::assets::TensorMetadata metadata; std::vector<float> values; };
    void add(const std::string & name, std::vector<int64_t> shape, float value = 0.0f) {
        size_t count = 1;
        for (auto dim : shape) count *= static_cast<size_t>(dim);
        entries_.emplace(name, Entry{{name, "F32", std::move(shape)}, std::vector<float>(count, value)});
    }
    void linear(const std::string & prefix, int out, int in) { add(prefix + ".weight", {out, in}); add(prefix + ".bias", {out}); }
    void norm(const std::string & prefix, int size) { add(prefix + ".weight", {size}, 1.0f); add(prefix + ".bias", {size}); }
    std::filesystem::path path_ = "self-generated-cache-weights";
    std::map<std::string, Entry> entries_;
};

void extreme_cache_rates() {
    EncoderConfig config;
    config.asr.feature_size = config.diar.feature_size = 4;
    config.asr.hidden_size = config.diar.hidden_size = 16;
    config.asr.heads = config.diar.heads = 1;
    config.asr.layers = config.diar.layers = 0;
    config.asr.intermediate_size = config.diar.intermediate_size = 32;
    config.asr.subsampling_factor = config.diar.subsampling_factor = config.upsample_factor = 2;
    config.asr.rotary_dimensions = config.diar.rotary_dimensions = 8;
    config.projection_dim = 12;
    config.speaker_count = 2;
    config.speaker_hidden_size = 8;
    config.online_inference_length = 4;
    config.chunk_left_context = config.chunk_right_context = 1;
    config.speaker_cache.cache_frames = 6;
    config.speaker_cache.fifo_frames = 2;
    config.speaker_cache.update_period = 3;
    config.speaker_cache.strong_boost_rate = config.speaker_cache.weak_boost_rate =
        config.speaker_cache.min_positive_score_rate = std::numeric_limits<float>::max();
    CacheWeights source;
    engine::core::ExecutionContext execution({engine::core::BackendType::Cpu, 0, 1});
    ParallelExpertEncoder encoder(execution, source, config);
    MelFeatures mel{37, 4, 0.37, std::vector<float>(148, -1.0f)};
    const auto output = encoder.encode(mel);
    require(output.frames == 19 && output.native_speaker_frames == 37, "extreme cache fixture geometry");
    for (float value : output.speaker_probs10) require(std::isfinite(value), "extreme cache produced non-finite activity");
    const auto repeated = encoder.encode(mel);
    require(output.audio_embeddings == repeated.audio_embeddings && output.speaker_probs10 == repeated.speaker_probs10,
        "extreme cache rates changed later request output");
}

// Exercise the real encoder with Vulkan's contiguous-input NORM restriction,
// including on CPU-only CI. Restore the device callback before leaving the test.
struct ContiguousNormBackend {
    using Supports = bool (*)(ggml_backend_dev_t, const ggml_tensor *);
    static inline Supports original = nullptr;
    static inline int checked = 0;
    ggml_backend_dev_t device;
    explicit ContiguousNormBackend(ggml_backend_t backend) : device(ggml_backend_get_device(backend)) {
        original = device->iface.supports_op;
        checked = 0;
        device->iface.supports_op = [](ggml_backend_dev_t dev, const ggml_tensor * op) {
            if (op->op == GGML_OP_NORM) {
                ++checked;
                if (!ggml_is_contiguous(op->src[0])) return false;
            }
            return original(dev, op);
        };
    }
    ~ContiguousNormBackend() { device->iface.supports_op = original; }
};

void attention_norm_layout() {
    EncoderConfig config;
    config.asr.feature_size = config.diar.feature_size = 4;
    config.asr.hidden_size = config.diar.hidden_size = 16;
    config.asr.heads = config.diar.heads = 2;
    config.asr.layers = config.diar.layers = 1;
    config.asr.intermediate_size = config.diar.intermediate_size = 32;
    config.asr.subsampling_factor = config.diar.subsampling_factor = config.upsample_factor = 2;
    config.asr.rotary_dimensions = config.diar.rotary_dimensions = 8;
    config.asr.qk_norm = config.diar.qk_norm = true;
    config.projection_dim = 12;
    config.speaker_count = 2;
    config.speaker_hidden_size = 8;
    config.online_inference_length = 4;
    config.chunk_left_context = config.chunk_right_context = 1;
    config.speaker_cache.cache_frames = 6;
    config.speaker_cache.fifo_frames = 2;
    config.speaker_cache.update_period = 3;
    CacheWeights source(true);
    engine::core::ExecutionContext execution({engine::core::BackendType::Cpu, 0, 1});
    MelFeatures mel{37, 4, 0.37, std::vector<float>(148)};
    for (size_t i = 0; i < mel.values.size(); ++i) mel.values[i] = std::sin(static_cast<float>(i) * 0.17f);
    for (bool flash : {false, true}) {
        config.use_flash_attention = flash;
        ParallelExpertEncoder reference(execution, source, config);
        const auto expected = reference.encode(mel);
        ContiguousNormBackend restricted(execution.backend());
        ParallelExpertEncoder encoder(execution, source, config);
        const auto output = encoder.encode(mel);
        require(ContiguousNormBackend::checked > 0, "encoder layout test did not check normalization");
        require(output.asr_states == expected.asr_states && output.audio_embeddings == expected.audio_embeddings &&
            output.speaker_probs80 == expected.speaker_probs80 && output.speaker_probs10 == expected.speaker_probs10,
            "contiguous normalization changed encoder output");
        const auto repeated = encoder.encode(mel);
        require(output.audio_embeddings == repeated.audio_embeddings, "attention layout repeated request changed output");
    }
}
}

int main() {
    try {
        validate_encoder_config({});
        bad_config([](auto & c) { c.speaker_cache.latest_score_boost = 100.0f; });
        bad_config([](auto & c) { c.speaker_cache.latest_score_boost = -1000.0f; });
        bad_config([](auto & c) { c.asr.hidden_size = 1000000000; c.asr.heads = 1; });
        bad_config([](auto & c) { c.asr.feature_size = c.diar.feature_size = std::numeric_limits<int>::max(); });
        bad_config([](auto & c) { c.speaker_hidden_size = std::numeric_limits<int>::max(); });
        bad_config([](auto & c) { c.asr.layers = std::numeric_limits<int>::max(); });
        bad_config([](auto & c) { c.maximum_audio_seconds = 3600.01; });
        auto plan = plan_encoder_windows(360000, {});
        require(plan.size() == 90 && plan.back().core_frames == 500, "one-hour boundary changed");
        rejects([] { plan_encoder_windows(360001, {}); }, "over-hour mel grid was accepted");
        EncoderConfig fractional;
        fractional.maximum_audio_seconds = 0.29;
        require(plan_encoder_windows(29, fractional).front().core_frames == 4, "decimal duration lost a valid frame");
        rejects([&] { plan_encoder_windows(30, fractional); }, "decimal duration accepted an extra frame");
        const auto partial = plan_encoder_windows(4905, {});
        require(partial.size() == 2 && partial[0].mel_end == 4400 && partial[0].core_frames == 500 &&
            partial[1].mel_begin == 3600 && partial[1].left_drop == 50 && partial[1].core_frames == 114,
            "partial final window geometry changed");
        MelFeatures malformed{std::numeric_limits<int64_t>::max(), 128, 1.0, {}};
        rejects([&] { normalize_asr_features(malformed); }, "overflowing mel shape was accepted");
        malformed = {1, 1, 0.01, {std::numeric_limits<float>::quiet_NaN()}};
        rejects([&] { normalize_asr_features(malformed); }, "non-finite mel was accepted");
        const auto singleton = normalize_asr_features({1, 2, 0.01, {2.0f, -5.0f}});
        require(singleton == std::vector<float>({0.0f, 0.0f}), "singleton normalization changed");
        std::vector<float> fb(128 * 257, 0.0f);
        rejects([&] { Frontend({}, fb, std::vector<float>(399)); }, "wrong trained-window length accepted");
        std::vector<float> window(400, 1.0f);
        window[2] = std::numeric_limits<float>::infinity();
        rejects([&] { Frontend({}, fb, window); }, "non-finite trained window accepted");
        Frontend frontend({}, fb);
        engine::runtime::AudioBuffer audio{16000, 1, std::vector<float>(400, std::numeric_limits<float>::max())};
        rejects([&] { frontend.extract(audio, 1); }, "non-finite frontend output accepted");
        extreme_cache_rates();
        attention_norm_layout();
        std::cout << "PASS encoder bounds, window geometry, frontend checks and contiguous attention normalization\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
