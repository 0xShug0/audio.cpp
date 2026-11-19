#include "engine/framework/debug/profiler.h"
#include "engine/community_models/granite_speech5/encoder.h"
#include "engine/community_models/granite_speech5/projector.h"
#include "engine/framework/core/execution_context.h"
#include "test_assert.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace test = engine::test;
namespace g2b = engine::community_models::granite_speech5;
namespace assets_ns = engine::assets;

// Fake source returning deterministic synthetic data: shapes for every
// encoder/projector tensor plus patterned F32 values, so the projector can
// actually run (window-isolation fixture) without the checkpoint.
class FakeDataSource : public assets_ns::TensorSource {
public:
    const std::filesystem::path & source_path() const noexcept override { return path_; }
    bool has_tensor(std::string_view name) const noexcept override {
        return shapes_.find(std::string(name)) != shapes_.end();
    }
    assets_ns::TensorMetadata require_metadata(std::string_view name) const override {
        const auto it = shapes_.find(std::string(name));
        if (it == shapes_.end()) {
            throw std::runtime_error("fake source missing tensor: " + std::string(name));
        }
        return {std::string(name), "F32", it->second};
    }
    std::vector<assets_ns::TensorMetadata> tensors() const override {
        std::vector<assets_ns::TensorMetadata> out;
        for (const auto & [name, shape] : shapes_) {
            out.push_back({name, "F32", shape});
        }
        return out;
    }
    assets_ns::RawTensorData require_tensor_data(std::string_view name) const override {
        const auto shape = require_metadata(name).shape;
        const auto values = require_f32(name, std::optional<std::vector<int64_t>>(shape));
        std::vector<std::byte> bytes(values.size() * sizeof(float));
        std::memcpy(bytes.data(), values.data(), bytes.size());
        return {{std::string(name), "F32", shape}, std::move(bytes)};
    }
    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        const auto shape = require_metadata(name).shape;
        if (expected_shape.has_value() && *expected_shape != shape) {
            throw std::runtime_error("fake source shape mismatch for " + std::string(name));
        }
        size_t count = 1;
        for (int64_t dim : shape) {
            count *= static_cast<size_t>(dim);
        }
        std::vector<float> values(count);
        uint64_t seed = 0x9e3779b9u;
        for (char ch : std::string(name)) {
            seed = seed * 1315423911u + static_cast<uint64_t>(ch);
        }
        for (size_t i = 0; i < count; ++i) {
            seed = seed * 6364136223846793005ull + 1442695040888963407ull;
            const double unit = static_cast<double>((seed >> 33) % 20001) / 20000.0;
            values[i] = static_cast<float>((unit - 0.5) * 0.4);
        }
        return values;
    }
    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        if (!has_tensor(name)) {
            return std::nullopt;
        }
        return require_f32(name, expected_shape);
    }
    int64_t require_i64_scalar(std::string_view name) const override {
        (void) name;
        return 0;
    }
    std::map<std::string, std::vector<int64_t>> shapes_;
    std::filesystem::path path_ = "/fake/model.safetensors";
};

void add_encoder_shapes(std::map<std::string, std::vector<int64_t>> & shapes) {
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
}

void add_projector_shapes(std::map<std::string, std::vector<int64_t>> & shapes) {
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
}

FakeDataSource full_shapes() {
    FakeDataSource source;
    add_encoder_shapes(source.shapes_);
    add_projector_shapes(source.shapes_);
    return source;
}

void require_throws_with(const std::string & text, const std::string & label) {
    test::require(!text.empty(), label);
    test::require(text.find("granite_speech5") != std::string::npos, label + ": family");
}

// Length-derived graph sizing: block/window boundaries incl. brief edges.
void test_sizing_rules() {
    // Encoder blocks: ceil(T/200), last block keeps its real length.
    for (int64_t steps : {1, 14, 15, 16, 199, 200}) {
        const auto plan = g2b::granite_speech5_encoder_blocks_for_steps(steps);
        test::require_eq(plan.num_blocks, int64_t{1}, "single block");
        test::require_eq(plan.last_block_len, steps, "short length");
    }
    {
        const auto plan = g2b::granite_speech5_encoder_blocks_for_steps(201);
        test::require_eq(plan.num_blocks, int64_t{2}, "two blocks");
        test::require_eq(plan.last_block_len, int64_t{1}, "tail of 1");
    }
    {
        const auto plan = g2b::granite_speech5_encoder_blocks_for_steps(400);
        test::require_eq(plan.num_blocks, int64_t{2}, "exact two blocks");
        test::require_eq(plan.last_block_len, int64_t{200}, "full tail");
    }
    {
        const auto plan = g2b::granite_speech5_encoder_blocks_for_steps(1247);
        test::require_eq(plan.num_blocks, int64_t{7}, "multilingual blocks");
        test::require_eq(plan.last_block_len, int64_t{47}, "multilingual tail");
    }
    for (int64_t bad : {0, -1, -200}) {
        bool threw = false;
        try {
            (void) g2b::granite_speech5_encoder_blocks_for_steps(bad);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        test::require(threw, "encoder blocks reject T=" + std::to_string(bad));
    }
    // Projector windows: ceil(T/15), right pad, A=3W audio placeholders.
    struct Case {
        int64_t steps, windows, padded, pad, placeholders;
    };
    for (const auto c : {Case{1, 1, 15, 14, 3},
                         Case{14, 1, 15, 1, 3},
                         Case{15, 1, 15, 0, 3},
                         Case{16, 2, 30, 14, 6},
                         Case{199, 14, 210, 11, 42},
                         Case{200, 14, 210, 10, 42},
                         Case{201, 14, 210, 9, 42},
                         Case{1247, 84, 1260, 13, 252},
                         Case{175, 12, 180, 5, 36},
                         Case{157, 11, 165, 8, 33}}) {
        const auto plan = g2b::granite_speech5_projector_plan_for_steps(c.steps);
        test::require_eq(plan.windows, c.windows, "W for T=" + std::to_string(c.steps));
        test::require_eq(plan.padded_steps, c.padded, "padded T");
        test::require_eq(plan.pad_steps, c.pad, "pad rows");
        test::require_eq(plan.audio_placeholders, c.placeholders, "A=3W");
    }
    for (int64_t bad : {0, -3}) {
        bool threw = false;
        try {
            (void) g2b::granite_speech5_projector_plan_for_steps(bad);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        test::require(threw, "projector plan reject T=" + std::to_string(bad));
    }
}

// No-weight negatives: every shape/length rule rejects with family error text.
void test_shape_negatives() {
    const FakeDataSource good = full_shapes();
    g2b::check_granite_speech5_encoder_weight_shapes(good);
    g2b::check_granite_speech5_projector_weight_shapes(good);

    auto expect_throw = [](FakeDataSource source, bool encoder, const std::string & label) {
        bool threw = false;
        try {
            if (encoder) {
                g2b::check_granite_speech5_encoder_weight_shapes(source);
            } else {
                g2b::check_granite_speech5_projector_weight_shapes(source);
            }
        } catch (const std::runtime_error &) {
            threw = true;
        }
        test::require(threw, label);
    };

    // First-block validation: a bad layer-0 tensor names layer 0.
    {
        FakeDataSource bad = full_shapes();
        bad.shapes_["encoder.layers.0.attn.to_q.weight"] = {1024, 512};
        std::string message;
        try {
            g2b::check_granite_speech5_encoder_weight_shapes(bad);
        } catch (const std::runtime_error & error) {
            message = error.what();
        }
        require_throws_with(message, "layer-0 to_q");
        test::require(message.find("encoder.layers.0.attn.to_q.weight") != std::string::npos, "names layer 0");
        test::require(message.find("1024x1024") != std::string::npos, "expected shape");
        test::require(message.find("1024x512") != std::string::npos, "actual shape");
    }
    {
        FakeDataSource bad = full_shapes();
        bad.shapes_.erase("encoder.layers.0.conv.batch_norm.running_var");
        expect_throw(bad, true, "missing layer-0 running_var");
    }
    {
        FakeDataSource bad = full_shapes();
        bad.shapes_["encoder.layers.7.conv.depth_conv.conv.weight"] = {2048, 1, 7};
        expect_throw(bad, true, "wrong kernel");
    }
    {
        FakeDataSource bad = full_shapes();
        bad.shapes_["encoder.out_mid.weight"] = {348, 1024};
        expect_throw(bad, true, "transposed out_mid");
    }
    {
        FakeDataSource bad = full_shapes();
        bad.shapes_["encoder.layers.16.attn.to_q.weight"] = {1024, 1024};
        expect_throw(bad, true, "extra encoder layer");
    }
    {
        FakeDataSource bad = full_shapes();
        bad.shapes_["projector.query"] = {1, 5, 1024};
        expect_throw(bad, false, "wrong query count");
    }
    {
        FakeDataSource bad = full_shapes();
        bad.shapes_["projector.qformer.encoder.layer.1.crossattention.attention.key.weight"] = {512, 1024};
        expect_throw(bad, false, "wrong cross key");
    }
    {
        FakeDataSource bad = full_shapes();
        bad.shapes_["projector.linear.weight"] = {1024, 2048};
        expect_throw(bad, false, "transposed final projection");
    }
    {
        FakeDataSource bad = full_shapes();
        bad.shapes_["projector.qformer.encoder.layer.2.attention.attention.query.weight"] = {1024, 1024};
        expect_throw(bad, false, "extra qformer layer");
    }
}

// Window isolation on a projector-only fixture: perturbing one window changes
// only that window's 3 outputs (bitwise, same process/threads).
void test_window_isolation() {
    FakeDataSource source = full_shapes();
    auto assets = std::make_shared<g2b::GraniteSpeech5Assets>();
    assets->config = g2b::GraniteSpeech5Config{};
    assets->source = std::make_shared<FakeDataSource>(std::move(source));
    engine::core::ExecutionContext execution(
        engine::core::BackendConfig{engine::core::BackendType::Cpu, 0, 1});
    g2b::GraniteSpeech5ProjectorRuntime projector(
        assets, execution, engine::assets::TensorStorageType::F32);

    // T=16: two windows, 14 zero-padded rows participating in window 1.
    const int64_t steps = 16;
    std::vector<float> enc(static_cast<size_t>(steps * 1024));
    for (size_t i = 0; i < enc.size(); ++i) {
        enc[i] = (static_cast<int64_t>(i % 97) - 48) * 0.01F;
    }
    const auto base = projector.project(enc, steps);
    test::require_eq(base.audio_placeholders, int64_t{6}, "A=3W");
    test::require_eq(static_cast<int64_t>(base.projected.size()), int64_t{6 * 2048}, "projected values");
    for (float value : base.projected) {
        test::require(std::isfinite(value), "projected finite");
    }

    std::vector<float> perturbed = enc;
    for (int64_t row = 15; row < 16; ++row) {
        for (int64_t d = 0; d < 1024; ++d) {
            perturbed[static_cast<size_t>(row * 1024 + d)] += 1.0F;
        }
    }
    const auto changed = projector.project(perturbed, steps);
    // Window 0 (rows 0..2) is bitwise identical; window 1 moved.
    for (int64_t row = 0; row < 3; ++row) {
        for (int64_t d = 0; d < 2048; ++d) {
            test::require_eq(
                changed.projected[static_cast<size_t>(row * 2048 + d)],
                base.projected[static_cast<size_t>(row * 2048 + d)],
                "window-0 isolation");
        }
    }
    double max_diff = 0.0;
    for (int64_t row = 3; row < 6; ++row) {
        for (int64_t d = 0; d < 2048; ++d) {
            max_diff = std::max(
                max_diff,
                std::abs(
                    static_cast<double>(changed.projected[static_cast<size_t>(row * 2048 + d)]) -
                    static_cast<double>(base.projected[static_cast<size_t>(row * 2048 + d)])));
        }
    }
    test::require(max_diff > 1e-6, "window-1 responds");

    // Single-window and fully-padded-tail shapes run clean.
    for (int64_t short_steps : {14, 15}) {
        std::vector<float> short_enc(static_cast<size_t>(short_steps * 1024), 0.01F);
        const auto out = projector.project(short_enc, short_steps);
        test::require_eq(out.audio_placeholders, int64_t{3}, "single window A");
        for (float value : out.projected) {
            test::require(std::isfinite(value), "short projected finite");
        }
    }

    // Input size mismatches throw before any graph work.
    bool threw = false;
    try {
        (void) projector.project(std::vector<float>(100, 0.0F), steps);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "projector rejects bad input size");
    threw = false;
    try {
        (void) projector.project(enc, 0);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "projector rejects T=0");
}

// Long-span capacity: a 180 s span (T=9000 steps, W=600 windows) must run
// without exhausting the projector graph node budget. Regression test for
// the fixed 65536-node budget that aborted on spans above ~60 s.
void test_long_span_projector_capacity() {
    FakeDataSource source = full_shapes();
    auto assets = std::make_shared<g2b::GraniteSpeech5Assets>();
    assets->config = g2b::GraniteSpeech5Config{};
    assets->source = std::make_shared<FakeDataSource>(std::move(source));
    engine::core::ExecutionContext execution(
        engine::core::BackendConfig{engine::core::BackendType::Cpu, 0, 1});
    g2b::GraniteSpeech5ProjectorRuntime projector(
        assets, execution, engine::assets::TensorStorageType::F32);

    const int64_t steps = 9000;
    std::vector<float> enc(static_cast<size_t>(steps * 1024));
    for (size_t i = 0; i < enc.size(); ++i) {
        enc[i] = (static_cast<int64_t>(i % 97) - 48) * 0.01F;
    }
    const auto out = projector.project(enc, steps);
    test::require_eq(out.audio_placeholders, int64_t{1800}, "A=3W for 600 windows");
    test::require_eq(
        static_cast<int64_t>(out.projected.size()), int64_t{1800 * 2048}, "projected values");
    for (float value : out.projected) {
        test::require(std::isfinite(value), "long-span projected finite");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    engine::debug::configure_logging({argc > 1 && std::string(argv[1]) == "--log"});
    const auto started = std::chrono::steady_clock::now();
    try {
        test_sizing_rules();
        test_shape_negatives();
        test_window_isolation();
        test_long_span_projector_capacity();
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    std::cout << "PASS: granite_speech5 B2 unit checks\n";
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        return 0;
}
