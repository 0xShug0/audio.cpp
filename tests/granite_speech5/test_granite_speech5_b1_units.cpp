#include "engine/framework/debug/profiler.h"
#include "engine/community_models/granite_speech5/assets.h"
#include "engine/community_models/granite_speech5/frontend.h"
#include "test_assert.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace test = engine::test;
namespace g2b = engine::community_models::granite_speech5;
namespace assets_ns = engine::assets;

using g2b::granite_speech5_clip_too_short;
using g2b::granite_speech5_lengths_for_samples;

void require_lengths(int64_t samples, int64_t mel, int64_t steps, int64_t windows, int64_t placeholders) {
    const auto lengths = granite_speech5_lengths_for_samples(samples);
    test::require_eq(lengths.mel_frames, mel, "M for N=" + std::to_string(samples));
    test::require_eq(lengths.encoder_steps, steps, "T for N=" + std::to_string(samples));
    test::require_eq(lengths.projector_windows, windows, "W for N=" + std::to_string(samples));
    test::require_eq(lengths.audio_placeholders, placeholders, "A for N=" + std::to_string(samples));
}

void test_length_contract() {
    // Trace vectors from the pinned reference (ja/librispeech/multilingual).
    require_lengths(50304, 315, 157, 11, 33);
    require_lengths(56080, 351, 175, 12, 36);
    require_lengths(399040, 2495, 1247, 84, 252);
    // Brief boundaries: N=257 and 320 samples.
    require_lengths(257, 2, 1, 1, 3);
    require_lengths(320, 3, 1, 1, 3);
    // Even mel (N=320T-160) and odd mel (N=320T) must yield the same T.
    for (int64_t target : {14, 15, 16, 199, 200, 201}) {
        const auto even = granite_speech5_lengths_for_samples(320 * target - 160);
        const auto odd = granite_speech5_lengths_for_samples(320 * target);
        test::require_eq(even.encoder_steps, target, "even T");
        test::require_eq(odd.encoder_steps, target, "odd T");
        test::require_eq(even.mel_frames, 2 * target, "even M");
        test::require_eq(odd.mel_frames, 2 * target + 1, "odd M");
        const int64_t windows = (target + 14) / 15;
        test::require_eq(even.projector_windows, windows, "even W");
        test::require_eq(odd.projector_windows, windows, "odd W");
        test::require_eq(even.audio_placeholders, 3 * windows, "even A");
        test::require_eq(odd.audio_placeholders, 3 * windows, "odd A");
    }
    // Predicate + error boundaries.
    test::require(!granite_speech5_clip_too_short(0), "empty is not short");
    test::require(granite_speech5_clip_too_short(1), "1 sample");
    test::require(granite_speech5_clip_too_short(256), "256 samples");
    test::require(!granite_speech5_clip_too_short(257), "257 samples");
    for (int64_t bad : {0, 1, 256, -7}) {
        bool threw = false;
        try {
            (void) granite_speech5_lengths_for_samples(bad);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        test::require(threw, "lengths reject N=" + std::to_string(bad));
    }
}

std::shared_ptr<const g2b::GraniteSpeech5Assets> default_assets() {
    auto assets = std::make_shared<g2b::GraniteSpeech5Assets>();
    assets->config = g2b::GraniteSpeech5Config{};
    return assets;
}

engine::runtime::AudioBuffer mono_buffer(int sample_rate, std::vector<float> samples) {
    engine::runtime::AudioBuffer audio;
    audio.sample_rate = sample_rate;
    audio.channels = 1;
    audio.samples = std::move(samples);
    return audio;
}

void test_frontend_empty_and_rejects() {
    const g2b::GraniteSpeech5Frontend frontend(default_assets());
    const auto empty = frontend.extract(mono_buffer(16000, {}));
    test::require_eq(empty.mel_frames, int64_t{0}, "empty mel frames");
    test::require_eq(empty.encoder_steps, int64_t{0}, "empty steps");
    test::require(empty.mel.empty(), "empty mel values");
    test::require(empty.stacked.empty(), "empty stacked values");

    bool threw = false;
    try {
        (void) frontend.extract(mono_buffer(16000, std::vector<float>(100, 0.0F)));
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "100-sample clip rejected");

    engine::runtime::AudioBuffer bad_rate = mono_buffer(0, std::vector<float>(1600, 0.0F));
    threw = false;
    try {
        (void) frontend.extract(bad_rate);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "nonpositive rate rejected");

    engine::runtime::AudioBuffer bad_shape = mono_buffer(16000, {1.0F, 2.0F, 3.0F});
    bad_shape.channels = 2;
    threw = false;
    try {
        (void) frontend.extract(bad_shape);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "ragged interleave rejected");

    auto nonfinite = std::vector<float>(1600, 0.0F);
    nonfinite[7] = std::numeric_limits<float>::infinity();
    threw = false;
    try {
        (void) frontend.extract(mono_buffer(16000, nonfinite));
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "nonfinite amplitudes rejected");
}

void test_frontend_stacking_and_parity() {
    const g2b::GraniteSpeech5Frontend frontend(default_assets());
    // Impulse: N=1600 -> M=11, T=5, W=1, A=3.
    auto impulse = std::vector<float>(1600, 0.0F);
    impulse[0] = 1.0F;
    const auto out = frontend.extract(mono_buffer(16000, impulse));
    test::require_eq(out.post_resample_samples, int64_t{1600}, "post-resample N");
    test::require_eq(out.mel_frames, int64_t{11}, "impulse M");
    test::require_eq(out.encoder_steps, int64_t{5}, "impulse T");
    test::require_eq(out.lengths.projector_windows, int64_t{1}, "impulse W");
    test::require_eq(out.lengths.audio_placeholders, int64_t{3}, "impulse A");
    test::require_eq(static_cast<int64_t>(out.mel.size()), int64_t{11 * 80}, "mel values");
    test::require_eq(static_cast<int64_t>(out.stacked.size()), int64_t{5 * 160}, "stacked values");
    for (float value : out.mel) {
        test::require(std::isfinite(value), "mel finite");
    }
    // Pairwise stacking: row t is mel rows 2t/2t+1; the odd trailing mel
    // frame (index 10) is dropped, never padded.
    for (int64_t step = 0; step < 5; ++step) {
        for (int64_t m = 0; m < 80; ++m) {
            test::require_eq(
                out.stacked[static_cast<size_t>(step * 160 + m)],
                out.mel[static_cast<size_t>((2 * step) * 80 + m)], "stack left");
            test::require_eq(
                out.stacked[static_cast<size_t>(step * 160 + 80 + m)],
                out.mel[static_cast<size_t>((2 * step + 1) * 80 + m)], "stack right");
        }
    }
    // Silence is uniform across frames after the shared max floor.
    const auto silence = frontend.extract(mono_buffer(16000, std::vector<float>(1600, 0.0F)));
    for (size_t i = 1; i < silence.mel.size(); ++i) {
        test::require_eq(silence.mel[i], silence.mel[0], "silence uniform");
    }
    // Stereo parity: identical channels resample to identical PCM.
    std::vector<float> mono(1600, 0.0F);
    for (size_t i = 0; i < mono.size(); ++i) {
        mono[i] = 0.25F * std::sin(static_cast<float>(i) * 0.11F);
    }
    const auto mono_out = frontend.extract(mono_buffer(16000, mono));
    std::vector<float> stereo;
    stereo.reserve(3200);
    for (float sample : mono) {
        stereo.push_back(sample);
        stereo.push_back(sample);
    }
    engine::runtime::AudioBuffer stereo_audio;
    stereo_audio.sample_rate = 16000;
    stereo_audio.channels = 2;
    stereo_audio.samples = stereo;
    const auto stereo_out = frontend.extract(stereo_audio);
    test::require_eq(stereo_out.mel.size(), mono_out.mel.size(), "stereo mel size");
    for (size_t i = 0; i < mono_out.mel.size(); ++i) {
        test::require_eq(stereo_out.mel[i], mono_out.mel[i], "stereo parity");
    }
    // Resample path: 8 kHz stereo and mono share the same resampled PCM.
    std::vector<float> mono8(800, 0.0F);
    for (size_t i = 0; i < mono8.size(); ++i) {
        mono8[i] = 0.25F * std::sin(static_cast<float>(i) * 0.22F);
    }
    const auto out8 = frontend.extract(mono_buffer(8000, mono8));
    test::require_eq(out8.post_resample_samples, int64_t{1600}, "8k resample N");
    const auto ref_lengths = granite_speech5_lengths_for_samples(1600);
    test::require_eq(out8.lengths.mel_frames, ref_lengths.mel_frames, "8k M from resampled N");
    std::vector<float> stereo8;
    for (float sample : mono8) {
        stereo8.push_back(sample);
        stereo8.push_back(sample);
    }
    engine::runtime::AudioBuffer stereo8_audio;
    stereo8_audio.sample_rate = 8000;
    stereo8_audio.channels = 2;
    stereo8_audio.samples = stereo8;
    const auto stereo8_out = frontend.extract(stereo8_audio);
    test::require_eq(stereo8_out.mel.size(), out8.mel.size(), "8k stereo mel size");
    for (size_t i = 0; i < out8.mel.size(); ++i) {
        test::require_eq(stereo8_out.mel[i], out8.mel[i], "8k stereo parity");
    }
}

void write_text(const std::filesystem::path & path, const std::string & text) {
    std::ofstream out(path, std::ios::binary);
    out << text;
}

void test_can_load() {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "g2b_b1_can_load";
    fs::remove_all(root);
    // Missing directory and missing config both refuse.
    test::require(!g2b::granite_speech5_can_load(root / "absent"), "absent dir");
    const std::string triple = R"json({
        "model_type": "granite_speech",
        "has_lora_adapter": false,
        "encoder_config": {"model_type": "granite_speech_encoder"},
        "projector_config": {"model_type": "blip_2_qformer"},
        "text_config": {"model_type": "granite"}
    })json";
    const fs::path good = root / "good";
    fs::create_directories(good);
    write_text(good / "config.json", triple);
    test::require(g2b::granite_speech5_can_load(good), "conditional generation triple");
    // TurboCTC stamped with the generic name has no Q-Former/text decoder.
    const fs::path ctc = root / "ctc";
    fs::create_directories(ctc);
    write_text(ctc / "config.json", R"json({
        "model_type": "granite_speech",
        "has_lora_adapter": false,
        "encoder_config": {"model_type": "granite_speech5_ctc"}
    })json");
    test::require(!g2b::granite_speech5_can_load(ctc), "turboctc refused");
    const fs::path nar = root / "nar";
    fs::create_directories(nar);
    write_text(nar / "config.json", triple);
    // NAR advertises its own model_type.
    std::string nar_text = triple;
    nar_text.replace(nar_text.find("granite_speech5\""), std::string("granite_speech5\"").size(),
                     "granite_speech5_nar\"");
    write_text(nar / "config.json", nar_text);
    test::require(!g2b::granite_speech5_can_load(nar), "nar refused");
    const fs::path lora = root / "lora";
    fs::create_directories(lora);
    std::string lora_text = triple;
    lora_text.replace(lora_text.find("false"), std::string("false").size(), "true");
    write_text(lora / "config.json", lora_text);
    test::require(!g2b::granite_speech5_can_load(lora), "lora refused");
    fs::remove_all(root);
}

class FakeTensorSource : public assets_ns::TensorSource {
public:
    explicit FakeTensorSource(std::map<std::string, std::vector<int64_t>> shapes)
        : shapes_(std::move(shapes)) {}

    const std::filesystem::path & source_path() const noexcept override { return path_; }
    bool has_tensor(std::string_view name) const noexcept override {
        return shapes_.find(std::string(name)) != shapes_.end();
    }
    assets_ns::TensorMetadata require_metadata(std::string_view name) const override {
        const auto it = shapes_.find(std::string(name));
        if (it == shapes_.end()) {
            throw std::runtime_error("fake source missing tensor");
        }
        return {std::string(name), "BF16", it->second};
    }
    std::vector<assets_ns::TensorMetadata> tensors() const override {
        std::vector<assets_ns::TensorMetadata> out;
        for (const auto & [name, shape] : shapes_) {
            out.push_back({name, "BF16", shape});
        }
        return out;
    }
    assets_ns::RawTensorData require_tensor_data(std::string_view name) const override {
        (void) require_metadata(name);
        return {{std::string(name), "BF16", {}}, {}};
    }
    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        (void) require_metadata(name);
        (void) expected_shape;
        return {};
    }
    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        (void) expected_shape;
        if (!has_tensor(name)) {
            return std::nullopt;
        }
        return std::vector<float>{};
    }
    int64_t require_i64_scalar(std::string_view name) const override {
        (void) name;
        return 0;
    }

private:
    std::map<std::string, std::vector<int64_t>> shapes_;
    std::filesystem::path path_ = "/fake/model.safetensors";
};

std::map<std::string, std::vector<int64_t>> full_inventory_shapes() {
    std::map<std::string, std::vector<int64_t>> shapes;
    shapes["encoder.input_linear.weight"] = {1024, 160};
    shapes["encoder.input_linear.bias"] = {1024};
    shapes["encoder.out.weight"] = {348, 1024};
    shapes["encoder.out.bias"] = {348};
    shapes["encoder.out_mid.weight"] = {1024, 348};
    shapes["encoder.out_mid.bias"] = {1024};
    for (int layer = 0; layer < 16; ++layer) {
        const std::string prefix = "encoder.layers." + std::to_string(layer) + ".";
        shapes[prefix + "attn.pre_norm.weight"] = {1024};
        shapes[prefix + "attn.pre_norm.bias"] = {1024};
        shapes[prefix + "attn.rel_pos_emb.weight"] = {1025, 128};
        shapes[prefix + "attn.to_kv.weight"] = {2048, 1024};
        shapes[prefix + "attn.to_out.bias"] = {1024};
        shapes[prefix + "attn.to_out.weight"] = {1024, 1024};
        shapes[prefix + "attn.to_q.weight"] = {1024, 1024};
        shapes[prefix + "conv.batch_norm.bias"] = {2048};
        shapes[prefix + "conv.batch_norm.running_mean"] = {2048};
        shapes[prefix + "conv.batch_norm.running_var"] = {2048};
        shapes[prefix + "conv.batch_norm.weight"] = {2048};
        shapes[prefix + "conv.batch_norm.num_batches_tracked"] = {};
        shapes[prefix + "conv.depth_conv.conv.weight"] = {2048, 1, 15};
        shapes[prefix + "conv.down_conv.bias"] = {1024};
        shapes[prefix + "conv.down_conv.weight"] = {1024, 2048, 1};
        shapes[prefix + "conv.norm.bias"] = {1024};
        shapes[prefix + "conv.norm.weight"] = {1024};
        shapes[prefix + "conv.up_conv.bias"] = {4096};
        shapes[prefix + "conv.up_conv.weight"] = {4096, 1024, 1};
        for (const char * block : {"ff1", "ff2"}) {
            const std::string bp = prefix + block + ".";
            shapes[bp + "down_proj.bias"] = {1024};
            shapes[bp + "down_proj.weight"] = {1024, 4096};
            shapes[bp + "pre_norm.bias"] = {1024};
            shapes[bp + "pre_norm.weight"] = {1024};
            shapes[bp + "up_proj.bias"] = {4096};
            shapes[bp + "up_proj.weight"] = {4096, 1024};
        }
        shapes[prefix + "post_norm.bias"] = {1024};
        shapes[prefix + "post_norm.weight"] = {1024};
    }
    shapes["projector.query"] = {1, 3, 1024};
    shapes["projector.qformer.layernorm.weight"] = {1024};
    shapes["projector.qformer.layernorm.bias"] = {1024};
    shapes["projector.linear.weight"] = {2048, 1024};
    shapes["projector.linear.bias"] = {2048};
    for (int layer = 0; layer < 2; ++layer) {
        const std::string prefix = "projector.qformer.encoder.layer." + std::to_string(layer) + ".";
        for (const char * block : {"attention", "crossattention"}) {
            const std::string bp = prefix + block + ".";
            for (const char * proj : {"query", "key", "value"}) {
                shapes[bp + "attention." + proj + ".bias"] = {1024};
                shapes[bp + "attention." + proj + ".weight"] = {1024, 1024};
            }
            shapes[bp + "output.LayerNorm.bias"] = {1024};
            shapes[bp + "output.LayerNorm.weight"] = {1024};
            shapes[bp + "output.dense.bias"] = {1024};
            shapes[bp + "output.dense.weight"] = {1024, 1024};
        }
        shapes[prefix + "intermediate_query.dense.bias"] = {4096};
        shapes[prefix + "intermediate_query.dense.weight"] = {4096, 1024};
        shapes[prefix + "output_query.LayerNorm.bias"] = {1024};
        shapes[prefix + "output_query.LayerNorm.weight"] = {1024};
        shapes[prefix + "output_query.dense.bias"] = {1024};
        shapes[prefix + "output_query.dense.weight"] = {1024, 4096};
    }
    shapes["language_model.model.embed_tokens.weight"] = {100353, 2048};
    shapes["language_model.lm_head.weight"] = {100353, 2048};
    shapes["language_model.model.norm.weight"] = {2048};
    for (int layer = 0; layer < 40; ++layer) {
        const std::string prefix = "language_model.model.layers." + std::to_string(layer) + ".";
        shapes[prefix + "input_layernorm.weight"] = {2048};
        shapes[prefix + "post_attention_layernorm.weight"] = {2048};
        shapes[prefix + "mlp.down_proj.weight"] = {2048, 4096};
        shapes[prefix + "mlp.gate_proj.weight"] = {4096, 2048};
        shapes[prefix + "mlp.up_proj.weight"] = {4096, 2048};
        shapes[prefix + "self_attn.k_proj.weight"] = {512, 2048};
        shapes[prefix + "self_attn.o_proj.weight"] = {2048, 2048};
        shapes[prefix + "self_attn.q_proj.weight"] = {2048, 2048};
        shapes[prefix + "self_attn.v_proj.weight"] = {512, 2048};
    }
    return shapes;
}

void require_inventory_throws(
    const std::map<std::string, std::vector<int64_t>> & shapes, const std::string & label) {
    FakeTensorSource source(shapes);
    bool threw = false;
    try {
        g2b::check_granite_speech5_tensor_inventory(source);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, label);
}

void test_tensor_inventory() {
    const FakeTensorSource good(full_inventory_shapes());
    g2b::check_granite_speech5_tensor_inventory(good);

    auto missing_feedback = full_inventory_shapes();
    missing_feedback.erase("encoder.out.weight");
    require_inventory_throws(missing_feedback, "missing encoder.out");

    auto bad_shape = full_inventory_shapes();
    bad_shape["encoder.input_linear.weight"] = {160, 1024};
    require_inventory_throws(bad_shape, "transposed input_linear");

    auto short_text = full_inventory_shapes();
    for (auto it = short_text.begin(); it != short_text.end();) {
        if (it->first.rfind("language_model.model.layers.39.", 0) == 0) {
            it = short_text.erase(it);
        } else {
            ++it;
        }
    }
    require_inventory_throws(short_text, "39 text layers");

    auto extra_layer = full_inventory_shapes();
    extra_layer["language_model.model.layers.40.input_layernorm.weight"] = {2048};
    require_inventory_throws(extra_layer, "text layer 40");

    auto extra_encoder = full_inventory_shapes();
    extra_encoder["encoder.layers.16.attn.to_q.weight"] = {1024, 1024};
    require_inventory_throws(extra_encoder, "encoder layer 16");
}

}  // namespace

int main(int argc, char ** argv) {
    engine::debug::configure_logging({argc > 1 && std::string(argv[1]) == "--log"});
    const auto started = std::chrono::steady_clock::now();
    try {
        test_length_contract();
        test_frontend_empty_and_rejects();
        test_frontend_stacking_and_parity();
        test_can_load();
        test_tensor_inventory();
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    std::cout << "PASS: granite_speech5 B1 unit checks\n";
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        return 0;
}
