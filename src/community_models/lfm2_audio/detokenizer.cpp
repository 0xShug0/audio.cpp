#include "engine/community_models/lfm2_audio/detokenizer.h"

#include "lfm2_blocks.h"

#include "engine/framework/audio/fft.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/transformers/causal_decoder.h"
#include "engine/framework/runtime/errors.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace engine::community_models::lfm2_audio {
namespace {

namespace modules = engine::modules;
using core::TensorShape;
using core::TensorValue;
using lfm2_blocks::GgmlContextDeleter;
using lfm2_blocks::GgmlGallocrDeleter;

constexpr size_t kWeightContextBytes = 4ull * 1024ull * 1024ull;
constexpr size_t kGraphArenaBytes = 32ull * 1024ull * 1024ull;
constexpr size_t kStreamArenaBytes = 8ull * 1024ull * 1024ull;
constexpr size_t kGraphNodes = 8192;
// Graph sizes are rounded up to this many frames, so utterances of similar
// length reuse one graph; the padding frames only follow the real ones.
constexpr int64_t kGraphFrameStep = 16;
constexpr size_t kKeptGraphs = 3;

struct DetokenizerWeights {
    std::unique_ptr<core::BackendWeightStore> store;
    TensorValue code_embedding;  // [codebooks * codebook_size, hidden]
    std::vector<lfm2_blocks::LayerWeights> layers;
    modules::NormWeights final_norm;
    TensorValue head_weight;  // dense_2, [output_size, hidden]
    TensorValue head_bias;
    std::vector<float> window;
};

DetokenizerWeights load_weights(
    const assets::TensorSource & detokenizer,
    const assets::TensorSource & vocoder,
    const Lfm2DetokenizerConfig & config,
    core::ExecutionContext & execution,
    bool cpu_repack) {
    DetokenizerWeights out;
    core::BackendWeightStoreOptions options;
    options.cpu_extra_buffers = cpu_repack;
    out.store = std::make_unique<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(), "lfm2_audio.detokenizer.weights", kWeightContextBytes, options);
    auto & store = *out.store;
    const auto native = assets::TensorStorageType::Native;
    const int64_t d = config.lfm.hidden_size;
    const std::vector<int64_t> table_shape = {config.codebooks * config.codebook_size, d};

    // The detokenizer GGUF's token_embd is a text embedding the model never
    // uses; the codes are embedded with the vocoder's emb.emb.
    out.code_embedding = store.load_tensor(vocoder, "emb.emb.weight", native, table_shape);
    if (!lfm2_blocks::backend_gathers(execution.backend(), out.code_embedding.tensor->type)) {
        out.code_embedding = store.load_tensor(vocoder, "emb.emb.weight", assets::TensorStorageType::F16, table_shape);
    }

    out.layers = lfm2_blocks::load_layers(store, detokenizer, config.lfm);
    out.final_norm = {store.load_f32_tensor(detokenizer, "token_embd_norm.weight", {d}), std::nullopt};
    out.head_weight = store.load_tensor(detokenizer, "dense_2.weight", native, {config.output_size, d}, core::WeightUse::MatMulOnly);
    out.head_bias = store.load_f32_tensor(detokenizer, "dense_2.bias", {config.output_size});
    out.window = vocoder.require_f32_tensor("istft.window", {config.n_fft}).values;
    store.upload();
    return out;
}

// Steps whose output depends on a given input step: each attention layer
// looks back sliding_window - 1 steps and each short-conv layer
// conv_kernel_size - 1.
int64_t receptive_field(const Lfm2DetokenizerConfig & config) {
    int64_t steps = 0;
    for (int64_t layer = 0; layer < config.lfm.num_layers(); ++layer) {
        steps += config.lfm.is_attention_layer(layer) ? config.sliding_window - 1 : config.lfm.conv_kernel_size - 1;
    }

    return steps;
}

// Embedding rows of the frames' codes, each checked against its codebook.
std::vector<int32_t> embedding_rows(const std::vector<std::vector<int32_t>> & frames, const Lfm2DetokenizerConfig & config) {
    std::vector<int32_t> rows;
    rows.reserve(frames.size() * static_cast<size_t>(config.codebooks));
    for (const auto & frame : frames) {
        if (static_cast<int64_t>(frame.size()) != config.codebooks) {
            throw std::runtime_error("LFM2-Audio detokenizer frame needs one code per codebook");
        }

        for (int64_t codebook = 0; codebook < config.codebooks; ++codebook) {
            const int32_t code = frame[static_cast<size_t>(codebook)];
            if (code < 0 || code >= config.codebook_size) {
                throw std::runtime_error("LFM2-Audio detokenizer code " + std::to_string(code) + " is outside the codebook");
            }

            rows.push_back(static_cast<int32_t>(codebook * config.codebook_size + code));
        }
    }

    return rows;
}

// FusedEmbedding: the mean over each frame's codebooks, then nearest-exact
// upsampling, which at an integer factor repeats each frame. [1, steps, hidden].
TensorValue frame_steps(
    ggml_context * g, const DetokenizerWeights & weights, ggml_tensor * rows, const Lfm2DetokenizerConfig & config, int64_t frames) {
    const int64_t d = config.lfm.hidden_size;
    const int64_t steps = frames * config.upsample;
    auto * x = ggml_get_rows(g, weights.code_embedding.tensor, rows);
    x = ggml_reshape_3d(g, x, d, config.codebooks, frames);
    x = ggml_mean(g, ggml_cont(g, ggml_permute(g, x, 1, 0, 2, 3)));
    x = ggml_reshape_3d(g, x, d, 1, frames);
    x = ggml_repeat(g, x, ggml_new_tensor_3d(g, GGML_TYPE_F32, d, config.upsample, frames));
    return core::wrap_tensor(ggml_reshape_3d(g, ggml_cont(g, x), d, steps, 1), TensorShape::from_dims({1, steps, d}), GGML_TYPE_F32);
}

// Final norm and the head: log-magnitude and phase for each step.
ggml_tensor * head_rows(
    core::ModuleBuildContext & ctx, const TensorValue & hidden, const DetokenizerWeights & weights, const Lfm2DetokenizerConfig & config) {
    const auto normed = lfm2_blocks::rms_norm(ctx, hidden, weights.final_norm, config.lfm);
    return modules::LinearModule({config.lfm.hidden_size, config.output_size, true})
        .build(ctx, normed, {weights.head_weight, weights.head_bias})
        .tensor;
}

// The detokenizer's attention mask: step q sees steps q - window + 1 .. q.
std::vector<float> sliding_window_mask(int64_t steps, int64_t window) {
    std::vector<float> mask(static_cast<size_t>(steps * steps), -INFINITY);
    for (int64_t q = 0; q < steps; ++q) {
        for (int64_t k = std::max<int64_t>(0, q - window + 1); k <= q; ++k) {
            mask[static_cast<size_t>(q * steps + k)] = 0.0f;
        }
    }

    return mask;
}

// Head output for `frames` code frames starting at an absolute frame.
class ChunkGraph {
public:
    ChunkGraph(const DetokenizerWeights & weights, const Lfm2DetokenizerConfig & config, core::ExecutionContext & execution, int64_t frames)
        : config_(config), execution_(execution), frames_(frames), steps_(frames * config.upsample) {
        ctx_.reset(ggml_init({kGraphArenaBytes, nullptr, true}));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize the LFM2-Audio detokenizer graph context");
        }

        auto * g = ctx_.get();
        core::ModuleBuildContext ctx{g, "lfm2_audio.detokenizer", execution.backend_type()};

        rows_ = ggml_new_tensor_1d(g, GGML_TYPE_I32, frames * config.codebooks);
        positions_ = ggml_new_tensor_1d(g, GGML_TYPE_I32, steps_);
        mask_ = ggml_new_tensor_2d(g, GGML_TYPE_F32, steps_, steps_);
        ggml_set_input(rows_);
        ggml_set_input(positions_);
        ggml_set_input(mask_);

        auto hidden = frame_steps(g, weights, rows_, config, frames);
        const auto positions = core::wrap_tensor(positions_, TensorShape::from_dims({steps_}), GGML_TYPE_I32);
        const auto mask = core::wrap_tensor(mask_, TensorShape::from_dims({steps_, steps_}), GGML_TYPE_F32);
        hidden = lfm2_blocks::build_sequence(ctx, hidden, positions, weights.layers, config.lfm, mask);
        output_ = head_rows(ctx, hidden, weights, config);
        ggml_set_output(output_);

        graph_ = ggml_new_graph_custom(g, kGraphNodes, false);
        ggml_build_forward_expand(graph_, output_);
        core::validate_backend_graph_supported(execution.backend(), graph_, "LFM2-Audio detokenizer graph");

        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (allocator_ == nullptr || !ggml_gallocr_alloc_graph(allocator_.get(), graph_)) {
            throw runtime::CapacityError(
                "LFM2-Audio detokenizer graph does not fit in device memory at " + std::to_string(steps_) + " steps");
        }

        mask_values_ = sliding_window_mask(steps_, config.sliding_window);
    }

    ~ChunkGraph() { core::release_backend_graph_resources(execution_.backend(), graph_, true); }

    [[nodiscard]] int64_t frames() const noexcept { return frames_; }

    // `rows` holds frames() frames of embedding rows; positions count from
    // first_frame so RoPE sees the same angles as one pass over the audio.
    std::vector<float> run(const std::vector<int32_t> & rows, int64_t first_frame) {
        const auto positions = modules::decoder_position_ids(steps_, first_frame * config_.upsample);
        ggml_backend_tensor_set(rows_, rows.data(), 0, rows.size() * sizeof(int32_t));
        ggml_backend_tensor_set(positions_, positions.data(), 0, positions.size() * sizeof(int32_t));
        // Set on every run: the allocator may reuse an input's memory once
        // the graph is done reading it.
        ggml_backend_tensor_set(mask_, mask_values_.data(), 0, mask_values_.size() * sizeof(float));

        core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LFM2-Audio detokenizer graph compute failed");
        }

        std::vector<float> out(static_cast<size_t>(steps_ * config_.output_size));
        ggml_backend_tensor_get(output_, out.data(), 0, out.size() * sizeof(float));
        return out;
    }

private:
    const Lfm2DetokenizerConfig & config_;
    core::ExecutionContext & execution_;
    int64_t frames_ = 0;
    int64_t steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * rows_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * output_ = nullptr;
    std::vector<float> mask_values_;
    ggml_cgraph * graph_ = nullptr;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> allocator_;
};

// One frame of a stream through every layer, continuing from the state the
// frames before it left: each attention layer's last sliding_window - 1 keys
// and values (the most it looks back) and each short-conv layer's last
// kernel - 1 inputs. The graph moves that state on as it runs.
class StreamGraph {
public:
    StreamGraph(const DetokenizerWeights & weights, const Lfm2DetokenizerConfig & config, core::ExecutionContext & execution)
        : config_(config), execution_(execution), steps_(config.upsample), past_(config.sliding_window - 1) {
        ctx_.reset(ggml_init({kStreamArenaBytes, nullptr, true}));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize the LFM2-Audio detokenizer stream context");
        }

        auto * g = ctx_.get();
        core::ModuleBuildContext ctx{g, "lfm2_audio.detokenizer.stream", execution.backend_type()};
        const int64_t d = config.lfm.hidden_size;
        const int64_t k = config.lfm.conv_kernel_size;
        const int64_t keys = past_ + steps_;

        rows_ = ggml_new_tensor_1d(g, GGML_TYPE_I32, config.codebooks);
        positions_ = ggml_new_tensor_1d(g, GGML_TYPE_I32, steps_);
        mask_ = ggml_new_tensor_2d(g, GGML_TYPE_F32, keys, steps_);

        auto x = frame_steps(g, weights, rows_, config, 1);
        const auto positions = core::wrap_tensor(positions_, TensorShape::from_dims({steps_}), GGML_TYPE_I32);
        const auto mask = core::wrap_tensor(mask_, TensorShape::from_dims({steps_, keys}), GGML_TYPE_F32);

        std::vector<std::pair<ggml_tensor *, ggml_tensor *>> updates;  // next state, state
        for (int64_t layer = 0; layer < config.lfm.num_layers(); ++layer) {
            const auto & w = weights.layers[static_cast<size_t>(layer)];
            if (w.attention) {
                const TensorShape state_shape =
                    TensorShape::from_dims({1, past_, config.lfm.kv_heads[static_cast<size_t>(layer)], config.lfm.head_dim});
                const auto past_keys = core::make_tensor(ctx, GGML_TYPE_F32, state_shape);
                const auto past_values = core::make_tensor(ctx, GGML_TYPE_F32, state_shape);
                const auto out = modules::DecoderLayerModule(lfm2_blocks::attention_layer_config(config.lfm, layer))
                                     .build(ctx, x, positions, w.decoder, past_keys, past_values, mask);
                x = out.output;

                // The window moves on by this frame's steps.
                for (const auto & [state, fresh] : {std::pair{past_keys, out.key}, std::pair{past_values, out.value}}) {
                    const auto all = modules::ConcatModule({1}).build(ctx, state, fresh);
                    updates.emplace_back(lfm2_blocks::contiguous(ctx, modules::SliceModule({1, steps_, past_}).build(ctx, all)).tensor, state.tensor);
                }

                continue;
            }

            const auto tail = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, d, k - 1}));
            const auto in = lfm2_blocks::short_conv_input(ctx, lfm2_blocks::rms_norm(ctx, x, w.decoder.input_norm, config.lfm), w.conv, d);
            const auto window = modules::ConcatModule({2}).build(ctx, tail, in.conv_in);
            const auto conv = core::wrap_tensor(
                ggml_ssm_conv(g, window.tensor, lfm2_blocks::conv_kernel(ctx, w.conv, config.lfm).tensor),
                TensorShape::from_dims({1, steps_, d}),
                GGML_TYPE_F32);
            x = lfm2_blocks::short_conv_output(ctx, x, conv, in.gate, w.conv, d);
            x = lfm2_blocks::feed_forward(ctx, x, w, config.lfm);
            updates.emplace_back(lfm2_blocks::contiguous(ctx, modules::SliceModule({2, steps_, k - 1}).build(ctx, window)).tensor, tail.tensor);
        }

        output_ = head_rows(ctx, x, weights, config);
        ggml_set_output(output_);

        // The state is overwritten once everything that reads it has run.
        graph_ = ggml_new_graph_custom(g, kGraphNodes, false);
        ggml_build_forward_expand(graph_, output_);
        for (const auto & [next, state] : updates) {
            ggml_build_forward_expand(graph_, ggml_cpy(g, next, state));
        }

        core::validate_backend_graph_supported(execution.backend(), graph_, "LFM2-Audio detokenizer stream graph");

        // Every tensor has its own memory, so the state lasts from run to run.
        buffer_.reset(ggml_backend_alloc_ctx_tensors(g, execution.backend()));
        if (buffer_ == nullptr) {
            throw runtime::CapacityError("LFM2-Audio detokenizer stream graph does not fit in device memory");
        }

        reset();
    }

    ~StreamGraph() { core::release_backend_graph_resources(execution_.backend(), graph_, true); }

    // A stream starts from zeros, as a sequence does.
    void reset() {
        ggml_backend_buffer_clear(buffer_.get(), 0);
        frames_ = 0;
    }

    // The next frame's embedding rows in, its steps' head output back.
    std::vector<float> run(const std::vector<int32_t> & rows) {
        // A step sees the steps up to sliding_window - 1 before it; the
        // state's slots before the stream began hold nothing.
        const int64_t first_step = frames_ * steps_;
        const int64_t keys = past_ + steps_;
        std::vector<float> mask(static_cast<size_t>(steps_ * keys), -INFINITY);
        for (int64_t q = 0; q < steps_; ++q) {
            for (int64_t key = 0; key < keys; ++key) {
                const int64_t at = first_step - past_ + key;
                if (at >= 0 && at <= first_step + q && first_step + q - at < config_.sliding_window) {
                    mask[static_cast<size_t>(q * keys + key)] = 0.0f;
                }
            }
        }

        const auto positions = modules::decoder_position_ids(steps_, first_step);
        ggml_backend_tensor_set(rows_, rows.data(), 0, rows.size() * sizeof(int32_t));
        ggml_backend_tensor_set(positions_, positions.data(), 0, positions.size() * sizeof(int32_t));
        ggml_backend_tensor_set(mask_, mask.data(), 0, mask.size() * sizeof(float));

        core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LFM2-Audio detokenizer stream graph compute failed");
        }

        ++frames_;
        std::vector<float> out(static_cast<size_t>(steps_ * config_.output_size));
        ggml_backend_tensor_get(output_, out.data(), 0, out.size() * sizeof(float));
        return out;
    }

private:
    const Lfm2DetokenizerConfig & config_;
    core::ExecutionContext & execution_;
    int64_t steps_ = 0;
    int64_t past_ = 0;
    int64_t frames_ = 0;  // frames the stream has decoded
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * rows_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    std::unique_ptr<ggml_backend_buffer, lfm2_blocks::GgmlBufferDeleter> buffer_;
};

}  // namespace

struct Lfm2DetokenizerRuntime::Impl {
    Impl(std::shared_ptr<const assets::TensorSource> detokenizer_in,
         std::shared_ptr<const assets::TensorSource> vocoder_in,
         const Lfm2DetokenizerConfig & config_in,
         core::ExecutionContext & execution_in,
         int64_t chunk_frames_in,
         bool cpu_repack)
        : detokenizer(std::move(detokenizer_in)),
          vocoder(std::move(vocoder_in)),
          config(config_in),
          execution(execution_in),
          weights(load_weights(*detokenizer, *vocoder, config, execution_in, cpu_repack)),
          context_frames((receptive_field(config) + config.upsample - 1) / config.upsample),
          chunk_frames(chunk_frames_in) {
        if (chunk_frames <= 0) {
            throw std::runtime_error("LFM2-Audio detokenizer chunks must add at least one frame");
        }
    }

    // Utterances of different lengths use graphs of different sizes; the
    // most recently used few are kept.
    ChunkGraph & graph(int64_t frames) {
        auto it = graphs.find(frames);
        if (it == graphs.end()) {
            if (graphs.size() >= kKeptGraphs) {
                graphs.erase(std::min_element(graphs.begin(), graphs.end(), [](const auto & a, const auto & b) {
                    return a.second.last_use < b.second.last_use;
                }));
            }

            it = graphs.emplace(frames, KeptGraph{std::make_unique<ChunkGraph>(weights, config, execution, frames), 0}).first;
        }

        it->second.last_use = ++uses;
        return *it->second.graph;
    }

    struct KeptGraph {
        std::unique_ptr<ChunkGraph> graph;
        uint64_t last_use = 0;
    };

    std::shared_ptr<const assets::TensorSource> detokenizer;
    std::shared_ptr<const assets::TensorSource> vocoder;
    Lfm2DetokenizerConfig config;
    core::ExecutionContext & execution;
    DetokenizerWeights weights;
    int64_t context_frames = 0;
    int64_t chunk_frames = 0;
    std::map<int64_t, KeptGraph> graphs;
    uint64_t uses = 0;
    std::unique_ptr<StreamGraph> stream;
};

Lfm2DetokenizerRuntime::Lfm2DetokenizerRuntime(
    std::shared_ptr<const assets::TensorSource> detokenizer,
    std::shared_ptr<const assets::TensorSource> vocoder,
    const Lfm2DetokenizerConfig & config,
    core::ExecutionContext & execution,
    int64_t chunk_frames,
    bool cpu_repack)
    : impl_(std::make_unique<Impl>(std::move(detokenizer), std::move(vocoder), config, execution, chunk_frames, cpu_repack)) {}

Lfm2DetokenizerRuntime::~Lfm2DetokenizerRuntime() = default;

std::vector<float> Lfm2DetokenizerRuntime::spectrum(const std::vector<std::vector<int32_t>> & frames, int64_t first_frame) {
    const auto & config = impl_->config;
    const auto total = static_cast<int64_t>(frames.size());
    if (first_frame < 0 || first_frame >= total) {
        throw std::runtime_error("LFM2-Audio detokenizer needs at least one new audio frame");
    }

    const auto rows = embedding_rows(frames, config);

    // Each chunk starts context_frames before the first frame it keeps, or at
    // frame 0, where the reference starts too; the frames before that only
    // warm the state up. A chunk that runs past the end is padded with row 0;
    // causal layers do not let the padding reach earlier steps.
    const auto start_time = std::chrono::steady_clock::now();
    const auto chunk_start = [&](int64_t kept) { return std::max<int64_t>(0, kept - impl_->context_frames); };
    // 128 frames is 768 steps, and the attention scores stay under 40 MB a layer.
    const int64_t max_frames = impl_->context_frames + impl_->chunk_frames;
    const int64_t needed = total - chunk_start(first_frame);
    const int64_t chunk_frames = std::min(max_frames, (needed + kGraphFrameStep - 1) / kGraphFrameStep * kGraphFrameStep);
    auto & graph = impl_->graph(chunk_frames);
    const int64_t row_width = config.upsample * config.output_size;

    std::vector<float> out;
    out.reserve(static_cast<size_t>((total - first_frame) * row_width));
    for (int64_t kept = first_frame; kept < total;) {
        const int64_t first = chunk_start(kept);
        std::vector<int32_t> chunk_rows(static_cast<size_t>(chunk_frames * config.codebooks), 0);
        const int64_t available = std::min(chunk_frames, total - first);
        std::copy_n(rows.begin() + first * config.codebooks, available * config.codebooks, chunk_rows.begin());

        const auto values = graph.run(chunk_rows, first);
        const int64_t keep_until = std::min(total, first + chunk_frames);
        out.insert(out.end(), values.begin() + (kept - first) * row_width, values.begin() + (keep_until - first) * row_width);
        kept = keep_until;
    }

    debug::timing_log_scalar("lfm2_audio.detokenizer.ms", engine::debug::elapsed_ms(start_time));
    return out;
}

std::vector<float> Lfm2DetokenizerRuntime::decode(const std::vector<std::vector<int32_t>> & frames) {
    const auto & config = impl_->config;
    const auto values = spectrum(frames);
    return lfm2_audio_istft(values, static_cast<int64_t>(frames.size()) * config.upsample, impl_->weights.window, config.hop_length);
}

void Lfm2DetokenizerRuntime::start_stream() {
    if (impl_->stream == nullptr) {
        impl_->stream = std::make_unique<StreamGraph>(impl_->weights, impl_->config, impl_->execution);
    } else {
        impl_->stream->reset();
    }
}

std::vector<float> Lfm2DetokenizerRuntime::stream(const std::vector<std::vector<int32_t>> & frames) {
    if (impl_->stream == nullptr) {
        throw std::runtime_error("LFM2-Audio detokenizer stream has not been started");
    }

    const auto & config = impl_->config;
    const auto rows = embedding_rows(frames, config);
    const auto start_time = std::chrono::steady_clock::now();
    std::vector<float> out;
    out.reserve(frames.size() * static_cast<size_t>(config.upsample * config.output_size));
    for (size_t frame = 0; frame < frames.size(); ++frame) {
        const std::vector<int32_t> frame_rows(
            rows.begin() + static_cast<std::ptrdiff_t>(frame) * config.codebooks, rows.begin() + static_cast<std::ptrdiff_t>(frame + 1) * config.codebooks);
        const auto values = impl_->stream->run(frame_rows);
        out.insert(out.end(), values.begin(), values.end());
    }

    debug::timing_log_scalar("lfm2_audio.detokenizer.stream_ms", engine::debug::elapsed_ms(start_time));
    return out;
}

const std::vector<float> & Lfm2DetokenizerRuntime::window() const {
    return impl_->weights.window;
}

const Lfm2DetokenizerConfig & Lfm2DetokenizerRuntime::config() const {
    return impl_->config;
}

Lfm2StreamingIstft::Lfm2StreamingIstft(std::vector<float> window, int64_t hop_length)
    : window_(std::move(window)),
      n_fft_(static_cast<int64_t>(window_.size())),
      hop_(hop_length),
      pad_((n_fft_ - hop_length) / 2) {
    if (n_fft_ < 2 || hop_ <= 0 || hop_ > n_fft_) {
        throw std::runtime_error("LFM2-Audio ISTFT hop must be positive and at most the window length");
    }
}

std::vector<float> Lfm2StreamingIstft::push(const std::vector<float> & spectrum, int64_t rows) {
    // finish() has emitted every sample, past where new rows would add.
    if (finished_) {
        throw std::runtime_error("LFM2-Audio ISTFT takes no rows after finish()");
    }

    const int64_t bins = n_fft_ / 2 + 1;
    if (rows <= 0 || spectrum.size() != static_cast<size_t>(rows * 2 * bins)) {
        throw std::runtime_error("LFM2-Audio ISTFT input does not match the window");
    }

    std::vector<std::complex<float>> complex_spectrum(static_cast<size_t>(rows * bins));
    for (int64_t row = 0; row < rows; ++row) {
        const float * values = spectrum.data() + row * 2 * bins;
        for (int64_t bin = 0; bin < bins; ++bin) {
            complex_spectrum[static_cast<size_t>(row * bins + bin)] = std::polar(std::exp(values[bin]), values[bins + bin]);
        }
    }

    // irfft with norm="backward" scales by 1 / n_fft.
    std::vector<float> framed(static_cast<size_t>(rows * n_fft_));
    audio::real_fft_inverse(
        {static_cast<size_t>(rows), static_cast<size_t>(n_fft_)},
        {static_cast<std::ptrdiff_t>(bins * sizeof(std::complex<float>)), static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))},
        {static_cast<std::ptrdiff_t>(n_fft_ * sizeof(float)), static_cast<std::ptrdiff_t>(sizeof(float))},
        1,
        complex_spectrum.data(),
        framed.data(),
        1.0f / static_cast<float>(n_fft_));

    // Windowed overlap-add, row by row in order, as one call over all rows
    // would do it.
    const int64_t end = (rows_ + rows - 1) * hop_ + n_fft_;
    folded_.resize(static_cast<size_t>(end - base_), 0.0f);
    envelope_.resize(static_cast<size_t>(end - base_), 0.0f);
    for (int64_t row = 0; row < rows; ++row) {
        const int64_t start = (rows_ + row) * hop_ - base_;
        for (int64_t i = 0; i < n_fft_; ++i) {
            const float w = window_[static_cast<size_t>(i)];
            folded_[static_cast<size_t>(start + i)] += framed[static_cast<size_t>(row * n_fft_ + i)] * w;
            envelope_[static_cast<size_t>(start + i)] += w * w;
        }
    }

    rows_ += rows;
    // Every later row starts at rows_ * hop or after, so what lies before is
    // final; the output starts pad samples into the overlap-add.
    return emit(rows_ * hop_ - pad_);
}

std::vector<float> Lfm2StreamingIstft::finish() {
    finished_ = true;
    return emit(rows_ * hop_);
}

std::vector<float> Lfm2StreamingIstft::emit(int64_t until) {
    std::vector<float> out;
    if (until <= emitted_) {
        return out;
    }

    out.reserve(static_cast<size_t>(until - emitted_));
    for (int64_t i = emitted_; i < until; ++i) {
        const auto index = static_cast<size_t>(i + pad_ - base_);
        const float denominator = envelope_[index];
        if (!(denominator > 1e-11f)) {
            throw std::runtime_error("LFM2-Audio ISTFT window envelope is zero");
        }

        out.push_back(folded_[index] / denominator);
    }

    // Drop what has been emitted; later rows only add past it.
    const int64_t drop = until + pad_ - base_;
    folded_.erase(folded_.begin(), folded_.begin() + drop);
    envelope_.erase(envelope_.begin(), envelope_.begin() + drop);
    base_ += drop;
    emitted_ = until;
    return out;
}

std::vector<float> lfm2_audio_istft(
    const std::vector<float> & spectrum, int64_t rows, const std::vector<float> & window, int64_t hop_length) {
    Lfm2StreamingIstft istft(window, hop_length);
    auto out = istft.push(spectrum, rows);
    const auto rest = istft.finish();
    out.insert(out.end(), rest.begin(), rest.end());
    return out;
}

}  // namespace engine::community_models::lfm2_audio
