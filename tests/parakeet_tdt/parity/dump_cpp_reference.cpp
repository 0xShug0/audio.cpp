// Dumps C++ Parakeet-TDT activations for numerical parity comparison against
// the NeMo reference dump produced by dump_nemo_reference.py. Run both, then
// compare_parity.py --nemo-dir ... --cpp-dir ....
//
// Two things are checked here, deliberately NOT via ad hoc debug taps spliced
// into the shared production graph (see the comment on build_encoder_layer
// in encoder.h for why: an extra "dead end" output tensor added to that
// specific multi-thousand-node cached graph read back incorrect values
// during the original 2026-07 encoder debugging session, for reasons never
// fully root-caused — the safe, verified-reliable pattern is either a fresh
// small isolated graph, or the real production entry points as-is):
//
//   1. mel_features + enc_out: driven straight through the real, unmodified
//      production entry points (ParakeetFrontend::extract,
//      ParakeetEncoderRuntime::encode) exactly as inference does. This is
//      the most end-to-end signal available and needs no isolation.
//
//   2. layer_0: built via the exported build_encoder_layer(...) function in
//      a small, single-purpose graph — the SAME code the production encoder
//      graph calls per layer, just invoked directly on NeMo's own captured
//      pre_encode + pos_emb + real layer-0 weights, so the comparison is
//      exactly apples-to-apples with NeMo's layer_0.npy.
#include "engine/community_models/parakeet_tdt/assets.h"
#include "engine/community_models/parakeet_tdt/encoder.h"
#include "engine/community_models/parakeet_tdt/frontend.h"
#include "engine/community_models/parakeet_tdt/weights.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"

#include "npy_io.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>

using namespace engine::community_models::parakeet_tdt;
using engine::community_models::parakeet_tdt::parity::read_npy_f32;
using engine::community_models::parakeet_tdt::parity::write_npy_f32;

namespace {

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

// Builds and computes a single encoder layer (via the exported, real
// production build_encoder_layer) fed with externally-supplied input,
// positional encoding, and weights — no dependency on the cached production
// graph at all.
std::vector<float> run_isolated_layer(
    engine::core::ExecutionContext & exec,
    const ParakeetEncoderLayerWeights & layer_weights,
    const ParakeetEncoderConfig & enc_cfg,
    const std::vector<float> & pre_encode,  // [T, D], row-major, D fastest
    const std::vector<float> & pos_emb_raw,  // [P, D], row-major, D fastest (P = 2T-1)
    int64_t frames,
    bool use_flash_attention) {
    const int64_t d = enc_cfg.hidden_size;
    const int64_t pos_len = 2 * frames - 1;

    ggml_init_params params{256ull * 1024ull * 1024ull, nullptr, true};
    ggml_context * gctx = ggml_init(params);
    engine::core::ModuleBuildContext ctx{gctx, "parity.layer0", exec.backend_type()};

    auto input = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, frames, d}));
    ggml_set_input(input.tensor);
    auto pos_emb = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, pos_len, d}));
    ggml_set_input(pos_emb.tensor);
    auto attention_mask = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({frames, frames}));
    ggml_set_input(attention_mask.tensor);
    auto keep_mask = engine::core::make_tensor(ctx, GGML_TYPE_I32, engine::core::TensorShape::from_dims({1, frames}));
    ggml_set_input(keep_mask.tensor);

    auto projected_pos_emb = engine::modules::LinearModule({d, d, false})
                                  .build(ctx, pos_emb, {layer_weights.pos_weight, std::nullopt});

    auto layer_out = build_encoder_layer(
        ctx, input, attention_mask, keep_mask, projected_pos_emb, layer_weights,
        enc_cfg.hidden_size, enc_cfg.intermediate_size, enc_cfg.heads, enc_cfg.conv_kernel,
        use_flash_attention);
    ggml_set_output(layer_out.tensor);

    ggml_cgraph * graph = ggml_new_graph_custom(gctx, 65536, false);
    ggml_build_forward_expand(graph, layer_out.tensor);
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(exec.backend()));
    if (!ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        ggml_free(gctx);
        throw std::runtime_error("run_isolated_layer: graph allocation failed");
    }

    engine::core::write_tensor_f32(input, pre_encode);
    engine::core::write_tensor_f32(pos_emb, pos_emb_raw);
    std::vector<float> mask_zeros(static_cast<size_t>(frames * frames), 0.0f);
    engine::core::write_tensor_f32(attention_mask, mask_zeros);
    std::vector<int32_t> keep(static_cast<size_t>(frames), 1);
    engine::core::write_tensor_i32(keep_mask, keep);

    engine::core::set_backend_threads(exec.backend(), 1);
    if (ggml_backend_graph_compute(exec.backend(), graph) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(gallocr);
        ggml_free(gctx);
        throw std::runtime_error("run_isolated_layer: graph compute failed");
    }

    std::vector<float> out(static_cast<size_t>(frames * d));
    engine::core::read_tensor_f32_into(layer_out.tensor, out);

    ggml_gallocr_free(gallocr);
    ggml_free(gctx);
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::filesystem::path model_path = arg_value(argc, argv, "--model", "models/parakeet-tdt-0.6b-v3");
    const std::filesystem::path audio_path = arg_value(argc, argv, "--audio", "");
    const std::filesystem::path nemo_dir = arg_value(argc, argv, "--nemo-dir", "");
    const std::filesystem::path output_dir = arg_value(argc, argv, "--output-dir", "");
    const auto matmul_weight_type = engine::assets::parse_tensor_storage_type(
        arg_value(argc, argv, "--matmul-weight-type", "native"));
    const bool use_flash_attention = arg_value(argc, argv, "--flash-attention", "0") == "1";
    // Dumping on a non-CPU backend lets the same activations be compared across
    // backends, not just against NeMo — which is how you catch a change that is
    // correct on CPU but silently wrong on an accelerator (a strided view a CPU
    // op materializes and a GPU kernel reads differently, say).
    const std::string backend_name = arg_value(argc, argv, "--backend", "cpu");

    if (audio_path.empty() || nemo_dir.empty() || output_dir.empty()) {
        std::fprintf(
            stderr,
            "usage: %s --model <path> --audio <wav> --nemo-dir <dir with dump_nemo_reference.py output> "
            "--output-dir <dir to write .npy dumps into> [--matmul-weight-type native|f16|bf16|q8_0] "
            "[--flash-attention 1] [--backend cpu|cuda|vulkan]\n",
            argv[0]);
        return 2;
    }

    try {
        std::filesystem::create_directories(output_dir);

        auto assets = load_parakeet_assets(model_path);

        engine::core::BackendConfig backend_config;
        if (backend_name == "cpu") {
            backend_config.type = engine::core::BackendType::Cpu;
        } else if (backend_name == "cuda") {
            backend_config.type = engine::core::BackendType::Cuda;
        } else if (backend_name == "vulkan") {
            backend_config.type = engine::core::BackendType::Vulkan;
        } else {
            throw std::runtime_error("unsupported --backend: " + backend_name);
        }
        // Single-threaded on CPU so the dump is reproducible run to run; on an
        // accelerator this is the host-side thread count and does not affect
        // the result.
        backend_config.threads = 1;
        engine::core::ExecutionContext exec(backend_config);

        auto weights = load_parakeet_weights(
            *assets, exec.backend(), exec.backend_type(),
            matmul_weight_type,
            engine::assets::TensorStorageType::Native,
            3072ull * 1024ull * 1024ull);

        // --- 1a. frontend: mel features, straight through the real entry point ---
        ParakeetFrontend frontend(assets);
        const auto wav = engine::audio::read_wav_f32(audio_path);
        if (wav.channels != 1) {
            throw std::runtime_error("dump_cpp_reference requires mono audio");
        }
        engine::runtime::AudioBuffer buf{wav.sample_rate, wav.channels, wav.samples};
        const auto feats = frontend.extract(buf, true);
        // feats.values is time-major [T, feature_dim] (feature_dim fastest).
        write_npy_f32(
            (output_dir / "mel_features.npy").string(),
            feats.values,
            {feats.frames, feats.feature_dim});
        std::printf("mel_features: frames=%lld feature_dim=%lld\n", (long long)feats.frames, (long long)feats.feature_dim);

        // --- 1b. full encoder: straight through the real entry point ---
        ParakeetEncoderRuntime encoder(assets, weights, exec, 1024ull * 1024ull * 1024ull, use_flash_attention);
        encoder.prepare_capacity(feats.frames, feats.feature_dim);
        const auto encoded = encoder.encode(feats);
        // encoded.values is time-major [frames, hidden_size].
        write_npy_f32(
            (output_dir / "enc_out.npy").string(),
            encoded.values,
            {encoded.frames, encoded.hidden_size});
        std::printf(
            "enc_out: frames=%lld valid_frames=%lld hidden=%lld\n",
            (long long)encoded.frames, (long long)encoded.valid_frames, (long long)encoded.hidden_size);

        // --- 2. isolated layer 0, fed with NeMo's own pre_encode + pos_emb ---
        std::vector<int64_t> pre_encode_shape;
        auto pre_encode = read_npy_f32((nemo_dir / "pre_encode.npy").string(), &pre_encode_shape);
        std::vector<int64_t> pos_emb_shape;
        auto pos_emb = read_npy_f32((nemo_dir / "pos_emb.npy").string(), &pos_emb_shape);
        if (pre_encode_shape.size() != 3 || pos_emb_shape.size() != 3) {
            throw std::runtime_error("expected NeMo pre_encode/pos_emb dumps to be rank-3 [1, T, D]");
        }
        const int64_t frames = pre_encode_shape[1];
        const int64_t hidden = pre_encode_shape[2];
        if (hidden != assets->config.encoder.hidden_size) {
            throw std::runtime_error("NeMo pre_encode hidden_size does not match this model's config");
        }

        const auto & layer0_weights = weights->encoder.layers.at(0);
        auto layer0_out = run_isolated_layer(exec, layer0_weights, assets->config.encoder, pre_encode, pos_emb, frames, use_flash_attention);
        write_npy_f32((output_dir / "layer_0.npy").string(), layer0_out, {frames, hidden});
        std::printf("layer_0: frames=%lld hidden=%lld\n", (long long)frames, (long long)hidden);

        std::printf("\nWrote mel_features.npy, enc_out.npy, layer_0.npy to %s\n", output_dir.string().c_str());
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "dump_cpp_reference failed: %s\n", e.what());
        return 1;
    }
}
