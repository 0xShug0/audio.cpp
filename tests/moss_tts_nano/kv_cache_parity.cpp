/*
 * Incremental decode parity for MOSS-TTS-Nano.
 *
 * The runtime generates audio one frame at a time and keeps a per-layer K/V cache instead of
 * re-forwarding the whole transcript on every step (which used to cost O(n^2) work and also
 * reserved max_new_frames rows up front). This test pins down the equivalences that make the
 * incremental path safe, using only the public runtime API:
 *
 *   prefill_chunk  : prefill(prompt chunk) + prefill(rest, reuse_rows)  == one-shot prefill
 *   decode_step    : prefill(rows - 1) + step_into(last row)            == one-shot prefill
 *   prefix_reuse   : prefill(previous prompt) + prefill(new prompt, reuse_rows)
 *                                                                       == fresh full prefill
 *   local_frame    : two frames with identical inputs (and no history) produce identical codes,
 *                    i.e. the 18-slot frame cache is fully rewritten every frame
 *
 * A prefix-reuse bug shows up immediately here: the first implementation forwarded only the
 * uncached rows and never read the cached prefix, and prefill_chunk/prefix_reuse reported
 * grossly different hidden states.
 *
 *   moss_tts_nano_kv_cache_parity --model <model.gguf> [--threads 8] [--weight-type native]
 *                                  [--tolerance auto] [--rows 24] [--frames 8]
 *                                  [--backend cpu|cuda|vulkan]
 *
 * Tolerance defaults to "auto": 0.001 on CPU with unquantized weights (the incremental path is
 * bitwise-exact there), 0.05 on CPU with quantized weights (per-row dots stay exact for the decode
 * step, but re-batching the prefill shifts quantized GEMM tilings by a few percent), and 0.02 on
 * GPU backends (kernel selection depends on the batch shape; quantized storage is upgraded to f16
 * by the runtimes there, leaving ~1% cublas/GEMV-vs-GEMM batching residual).
 */

#include "engine/models/moss/moss_tts_nano/assets.h"
#include "engine/models/moss/moss_tts_nano/generator.h"
#include "engine/models/moss/moss_tts_nano/global_transformer.h"
#include "engine/models/moss/moss_tts_nano/local_frame_decoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace nano = engine::models::moss_tts_nano;

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

engine::assets::TensorStorageType parse_weight_type(const std::string & value) {
    if (value == "native") {
        return engine::assets::TensorStorageType::Native;
    }
    if (value == "f32") {
        return engine::assets::TensorStorageType::F32;
    }
    if (value == "f16") {
        return engine::assets::TensorStorageType::F16;
    }
    if (value == "bf16") {
        return engine::assets::TensorStorageType::BF16;
    }
    if (value == "q8_0") {
        return engine::assets::TensorStorageType::Q8_0;
    }
    throw std::runtime_error("unsupported --weight-type: " + value);
}

struct Deviation {
    float max_abs = 0.0F;
    float mean_abs = 0.0F;
    float max_relative = 0.0F;
    int64_t argmax = -1;
    int64_t non_finite = 0;
    int64_t count = 0;
};

// NaN compares false against everything, so a run that produces NaN would otherwise leave
// max_abs at zero and report a pass.
Deviation compare(const std::vector<float> & actual, const std::vector<float> & expected) {
    // An empty or size-mismatched comparison would report a perfect match, so refuse to run it.
    if (actual.empty() || actual.size() != expected.size()) {
        throw std::runtime_error(
            "compare: size mismatch (" + std::to_string(actual.size()) + " vs " +
            std::to_string(expected.size()) + ")");
    }
    Deviation out;
    double total = 0.0;
    const auto count = std::min(actual.size(), expected.size());
    out.count = static_cast<int64_t>(count);
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(actual[i])) {
            ++out.non_finite;
            continue;
        }
        const float delta = std::fabs(actual[i] - expected[i]);
        total += delta;
        if (delta > out.max_abs) {
            out.max_abs = delta;
            out.argmax = static_cast<int64_t>(i);
        }
    }
    out.mean_abs = count == 0 ? 0.0F : static_cast<float>(total / static_cast<double>(count));
    float scale = 0.0F;
    for (size_t i = 0; i < count; ++i) {
        scale = std::max(scale, std::fabs(expected[i]));
    }
    out.max_relative = scale > 0.0F ? out.max_abs / scale : out.max_abs;
    return out;
}

// Deterministic synthetic prompt: most audio channels stay at the pad code so the masked-embedding
// path is exercised, a few carry real codes.
std::vector<int32_t> build_rows(
    const nano::MossTTSNanoConfig & config,
    int64_t rows,
    int64_t row_width,
    uint32_t seed) {
    const int64_t n_vq = config.n_vq;
    std::vector<int32_t> out(static_cast<size_t>(rows * row_width), static_cast<int32_t>(config.audio_pad_token_id));
    std::mt19937 rng(seed);
    for (int64_t row = 0; row < rows; ++row) {
        const size_t offset = static_cast<size_t>(row * row_width);
        out[offset] = row == 0 ? static_cast<int32_t>(config.audio_start_token_id)
                               : static_cast<int32_t>(config.pad_token_id);
        for (int64_t q = 0; q < n_vq; ++q) {
            if ((row + q) % 3 != 0) {
                continue;
            }
            const int64_t codebook_size = config.audio_codebook_sizes.at(static_cast<size_t>(q));
            out[offset + static_cast<size_t>(1 + q)] =
                static_cast<int32_t>(rng() % static_cast<uint32_t>(codebook_size));
        }
    }
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::string model_path = arg_value(argc, argv, "--model", "");
        if (model_path.empty()) {
            std::cerr << "usage: moss_tts_nano_kv_cache_parity --model <model.gguf> "
                         "[--threads 8] [--weight-type native] [--tolerance auto] "
                         "[--rows 24] [--frames 8] [--backend cpu|cuda|vulkan]\n";
            return 2;
        }
        const auto weight_type = parse_weight_type(arg_value(argc, argv, "--weight-type", "native"));
        const std::string tolerance_arg = arg_value(argc, argv, "--tolerance", "auto");
        const int64_t rows_wanted = std::stoll(arg_value(argc, argv, "--rows", "24"));
        const int64_t frames = std::stoll(arg_value(argc, argv, "--frames", "8"));
        const std::string backend_name = arg_value(argc, argv, "--backend", "cpu");

        const auto assets = nano::load_moss_tts_nano_assets(model_path);
        if (assets == nullptr) {
            std::cerr << "FAIL: failed to load assets from " << model_path << "\n";
            return 1;
        }
        const auto & config = assets->config;
        const int64_t row_width = config.n_vq + 1;
        if (rows_wanted < 4) {
            std::cerr << "FAIL: --rows must be at least 4\n";
            return 2;
        }
        const int64_t rows = rows_wanted;
        const int64_t split = rows / 2;
        const int64_t budget = rows + frames + 8;
        float tolerance = 0.0F;
        if (tolerance_arg != "auto") {
            tolerance = std::stof(tolerance_arg);
        } else {
            const auto resolved_type = engine::assets::resolve_tensor_storage_type(
                *assets->model_weights, "transformer.h.0.attn.c_attn.weight", weight_type);
            const bool quantized = ggml_is_quantized(engine::assets::ggml_type_for_tensor_storage(resolved_type));
            tolerance = backend_name == "cpu" ? (quantized ? 0.05F : 0.001F) : 0.02F;
        }
        std::cout << "rows=" << rows << " split=" << split << " codebooks=" << config.n_vq
                  << " budget=" << budget << " weight_type=" << arg_value(argc, argv, "--weight-type", "native")
                  << " backend=" << backend_name << " tolerance=" << tolerance << "\n";

        engine::core::BackendType backend_type = engine::core::BackendType::Cpu;
        if (backend_name == "cpu") {
            backend_type = engine::core::BackendType::Cpu;
        } else if (backend_name == "cuda") {
            backend_type = engine::core::BackendType::Cuda;
        } else if (backend_name == "vulkan") {
            backend_type = engine::core::BackendType::Vulkan;
        } else {
            std::cerr << "FAIL: unsupported --backend " << backend_name << "\n";
            return 2;
        }
        engine::core::BackendConfig backend_config;
        backend_config.type = backend_type;
        backend_config.device = 0;
        backend_config.threads = std::stoi(arg_value(argc, argv, "--threads", "8"));
        engine::core::ExecutionContext execution_context(backend_config);

        const auto rows_a = build_rows(config, rows, row_width, 0x5eed1u);
        auto rows_b = rows_a;
        for (int64_t row = split; row < rows; ++row) {
            const size_t offset = static_cast<size_t>(row * row_width);
            rows_b[offset] = static_cast<int32_t>(config.pad_token_id); // divergent tail
            for (int64_t q = 0; q < config.n_vq; ++q) {
                rows_b[offset + static_cast<size_t>(1 + q)] = static_cast<int32_t>(config.audio_pad_token_id);
            }
        }

        nano::MossTTSNanoGlobalTransformerRuntime global(
            assets,
            execution_context,
            256ull * 1024ull * 1024ull,
            128ull * 1024ull * 1024ull,
            512ull * 1024ull * 1024ull,
            weight_type);

        // --- 1. one-shot prefill vs chunked prefill over the cache -------------------------------
        global.begin_generation(budget);
        const auto hidden_full = global.prefill(rows_a, rows, row_width, 0);
        const auto cached_after_full = global.cached_positions();

        const std::vector<int32_t> head_rows(
            rows_a.begin(), rows_a.begin() + static_cast<std::ptrdiff_t>(split * row_width));
        global.reset_cache();
        global.prefill(head_rows, split, row_width, 0);
        const auto hidden_chunked = global.prefill(rows_a, rows, row_width, split);
        const auto chunked_deviation = compare(hidden_chunked, hidden_full);
        std::cout << "prefill vs chunked prefill: max_abs=" << chunked_deviation.max_abs
                  << " mean_abs=" << chunked_deviation.mean_abs
                  << " max_rel=" << chunked_deviation.max_relative
                  << " non_finite=" << chunked_deviation.non_finite
                  << " (cached rows after full prefill=" << cached_after_full
                  << ", after chunked=" << global.cached_positions() << ")\n";

        // --- 2. decode step vs prefill for the same row -------------------------------------------
        global.reset_cache();
        const std::vector<int32_t> prefix_rows(
            rows_a.begin(), rows_a.begin() + static_cast<std::ptrdiff_t>((rows - 1) * row_width));
        const auto hidden_prefix = global.prefill(prefix_rows, rows - 1, row_width, 0);
        (void) hidden_prefix;
        std::vector<int32_t> last_codes(static_cast<size_t>(config.n_vq));
        for (int64_t q = 0; q < config.n_vq; ++q) {
            last_codes[static_cast<size_t>(q)] =
                rows_a[static_cast<size_t>((rows - 1) * row_width + 1 + q)];
        }
        std::vector<float> hidden_stepped;
        global.step_into(
            rows_a[static_cast<size_t>((rows - 1) * row_width)],
            last_codes,
            hidden_stepped);
        const auto step_deviation = compare(hidden_stepped, hidden_full);
        std::cout << "decode step vs prefill:     max_abs=" << step_deviation.max_abs
                  << " mean_abs=" << step_deviation.mean_abs
                  << " max_rel=" << step_deviation.max_relative
                  << " non_finite=" << step_deviation.non_finite
                  << " (cached rows=" << global.cached_positions() << ")\n";

        // --- 3. cross-utterance prefix reuse ------------------------------------------------------
        // rows_b shares its first `split` rows with rows_a, so those cached rows may be reused.
        global.reset_cache();
        global.prefill(rows_a, rows, row_width, 0);
        const auto hidden_reuse = global.prefill(rows_b, rows, row_width, split);

        global.reset_cache();
        const auto hidden_reference = global.prefill(rows_b, rows, row_width, 0);
        const auto reuse_deviation = compare(hidden_reuse, hidden_reference);
        std::cout << "prefix reuse vs full:       max_abs=" << reuse_deviation.max_abs
                  << " mean_abs=" << reuse_deviation.mean_abs
                  << " max_rel=" << reuse_deviation.max_relative
                  << " non_finite=" << reuse_deviation.non_finite
                  << " (reused rows=" << split << ")\n";

        // --- 3b. sensitivity guards -------------------------------------------------------------
        // Two ways the checks above could be vacuous, both asserted to be observable:
        //   * the compared hidden state must depend on the prompt at all (rows_a vs rows_b);
        //   * a cached prefix that does not match the prompt must change the result, which is what
        //     happened when the first implementation forwarded only the uncached rows and never
        //     read the cache.
        const auto prompt_sensitivity = compare(hidden_reference, hidden_full);
        std::cout << "prompt change (must differ):     max_abs=" << prompt_sensitivity.max_abs
                  << " max_rel=" << prompt_sensitivity.max_relative << "\n";

        auto rows_wrong_prefix = rows_a;
        rows_wrong_prefix[0] = static_cast<int32_t>(config.audio_end_token_id);
        global.reset_cache();
        global.prefill(rows_wrong_prefix, rows, row_width, 0);
        const auto hidden_wrong_prefix = global.prefill(rows_b, rows, row_width, split);
        const auto prefix_sensitivity = compare(hidden_wrong_prefix, hidden_reference);
        std::cout << "mismatched cached prefix:        max_abs=" << prefix_sensitivity.max_abs
                  << " max_rel=" << prefix_sensitivity.max_relative << " (must differ)\n";

        // --- 4. local frame decoder: identical frames must be identical ---------------------------
        nano::MossTTSNanoLocalFrameDecoderRuntime local(
            assets,
            execution_context,
            64ull * 1024ull * 1024ull,
            128ull * 1024ull * 1024ull,
            weight_type);
        const int64_t active = config.n_vq;
        local.prepare(active);
        nano::MossTTSNanoSamplingOptions sampling;
        sampling.do_sample = false; // greedy: no rng dependence
        std::mt19937 rng(1234);
        uint64_t sample_index_a = 0;
        uint64_t sample_index_b = 0;
        const auto frame_a = local.generate_frame(hidden_full, active, {}, sampling, rng, 7, sample_index_a);
        const auto frame_b = local.generate_frame(hidden_full, active, {}, sampling, rng, 7, sample_index_b);
        const bool frames_match = frame_a == frame_b && !frame_a.empty();
        bool tokens_in_range = !frame_a.empty();
        for (int64_t q = 0; q < static_cast<int64_t>(frame_a.size()); ++q) {
            const int64_t size = config.audio_codebook_sizes.at(static_cast<size_t>(q));
            if (frame_a[static_cast<size_t>(q)] < 0 || frame_a[static_cast<size_t>(q)] >= size) {
                tokens_in_range = false;
            }
        }
        std::cout << "local frame repeat:         codes_equal=" << (frames_match ? "yes" : "no")
                  << " codes=" << frame_a.size() << " in_range=" << (tokens_in_range ? "yes" : "no") << "\n";

        const bool passed =
            chunked_deviation.non_finite == 0 && step_deviation.non_finite == 0 &&
            reuse_deviation.non_finite == 0 &&
            chunked_deviation.max_relative <= tolerance &&
            step_deviation.max_relative <= tolerance &&
            reuse_deviation.max_relative <= tolerance &&
            cached_after_full == rows &&
            prompt_sensitivity.max_abs > 0.0F && prompt_sensitivity.non_finite == 0 &&
            prefix_sensitivity.max_abs > 0.0F && prefix_sensitivity.non_finite == 0 &&
            frames_match && tokens_in_range;
        std::cout << (passed ? "PASS" : "FAIL") << ": relative tolerance " << tolerance << "\n";
        return passed ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
