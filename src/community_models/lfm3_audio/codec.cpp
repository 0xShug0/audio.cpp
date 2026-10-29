#include "engine/community_models/lfm3_audio/codec.h"

#include "../lfm2_audio/weight_stores.h"

#include "engine/community_models/lfm2_audio/detokenizer.h"
#include "engine/community_models/lfm3_audio/blocks.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/runtime/errors.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::community_models::lfm3_audio {

namespace {

using core::TensorShape;
using core::TensorValue;

constexpr size_t kGraphNodes = 8192;
constexpr size_t kKeptGraphs = 3;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("LFM3-Audio codec: " + message);
}

struct CodecWeights {
    std::unique_ptr<WeightStores> stores;
    // project_out stays F32: its 48 inputs are not a multiple of the 32-wide blocks of BF16/F16 GPU kernels.
    TensorValue project_out, project_out_bias, from_quantizer, from_quantizer_bias, lin, lin_bias;
    blocks::StackWeights stack;
    std::vector<float> window;
};

CodecWeights load_weights(const assets::TensorSource & source, const Lfm3CodecConfig & config, core::ExecutionContext & execution,
                          const Lfm3WeightOptions & options) {
    const int64_t d = config.stack.hidden;
    const int64_t latent = config.codebooks * static_cast<int64_t>(config.fsq_levels.size());
    const int64_t q = config.quantizer_dim;
    auto * backend = execution.backend();

    CodecWeights out;
    out.stores = std::make_unique<WeightStores>(execution, "lfm3_audio.codec.weights", 1024 * ggml_tensor_overhead(), options.cpu_repack);
    auto & plain = out.stores->plain();
    out.project_out = plain.load_f32_tensor(source, "quantizer.project_out.weight", {q, latent});
    out.project_out_bias = plain.load_f32_tensor(source, "quantizer.project_out.bias", {q});
    out.from_quantizer = blocks::load_matmul(*out.stores, backend, source, "from_quantizer.weight", {d, q}, options);
    out.from_quantizer_bias = plain.load_f32_tensor(source, "from_quantizer.bias", {d});
    out.stack = blocks::load_stack(*out.stores, backend, source, "decoder.lfm.", config.stack, options);
    out.lin = blocks::load_matmul(*out.stores, backend, source, "decoder.lin.weight", {config.n_fft + 2, d}, options);
    out.lin_bias = plain.load_f32_tensor(source, "decoder.lin.bias", {config.n_fft + 2});
    out.window = source.require_f32_tensor("decoder.istft.window", {config.n_fft}).values;
    out.stores->upload();
    return out;
}

// Ring row r holds the latest position p with p % capacity == r. Query q sees positions q - window + 1 .. q,
// all still in the ring because capacity >= steps + window - 1.
std::vector<ggml_fp16_t> ring_mask(int64_t start, int64_t steps, int64_t capacity, int64_t window) {
    const ggml_fp16_t visible = ggml_fp32_to_fp16(0.0f);
    std::vector<ggml_fp16_t> out(static_cast<size_t>(steps * capacity), ggml_fp32_to_fp16(-INFINITY));
    for (int64_t i = 0; i < steps; ++i) {
        const int64_t q = start + i;
        for (int64_t p = std::max<int64_t>(0, q - window + 1); p <= q; ++p) {
            out[static_cast<size_t>(i * capacity + p % capacity)] = visible;
        }
    }
    return out;
}

struct ChunkOutput {
    std::vector<float> spectrum, project_out, from_quantizer, decoder;
};

// FSQ latents of `frames` frames to spectrum rows, continuing the stream state at an absolute step.
class ChunkGraph {
public:
    ChunkGraph(const CodecWeights & weights, const Lfm3CodecConfig & config, std::vector<blocks::LayerState> & state, int64_t capacity,
               core::ExecutionContext & execution, const Lfm3WeightOptions & options, int64_t frames)
        : execution_(execution), capacity_(capacity), window_(config.stack.sliding_window), steps_(frames * config.upsample) {
        ctx_ = blocks::graph_context(kGraphNodes, "codec");
        auto * g = ctx_.get();
        core::ModuleBuildContext ctx{g, "lfm3_audio.codec", execution.backend_type()};
        graph_ = ggml_new_graph_custom(g, kGraphNodes, false);

        const int64_t d = config.stack.hidden;
        const int64_t latent = config.codebooks * static_cast<int64_t>(config.fsq_levels.size());
        const int64_t q = config.quantizer_dim;
        auto input = [&](ggml_type type, std::initializer_list<int64_t> dims) {
            auto value = core::make_tensor(ctx, type, TensorShape::from_dims(dims));
            ggml_set_input(value.tensor);
            return value;
        };

        latents_ = input(GGML_TYPE_F32, {1, frames, latent});
        blocks::StepInputs in;
        in.positions = input(GGML_TYPE_I32, {steps_});
        in.slots = input(GGML_TYPE_I64, {steps_});
        in.sliding_mask = input(GGML_TYPE_F16, {steps_, capacity});
        in.kv_rows = capacity;
        positions_ = in.positions;
        slots_ = *in.slots;
        mask_ = *in.sliding_mask;

        project_out_ = modules::LinearModule({latent, q, true, blocks::kMatmulPrecision})
                           .build(ctx, latents_, {weights.project_out, weights.project_out_bias});
        from_quantizer_ = modules::LinearModule({q, d, true, blocks::kMatmulPrecision})
                              .build(ctx, project_out_, {weights.from_quantizer, weights.from_quantizer_bias});
        // nearest-exact upsampling at an integer factor: step p takes frame p / upsample.
        auto * x = ggml_repeat_4d(g, ggml_reshape_3d(g, from_quantizer_.tensor, d, 1, frames), d, config.upsample, frames, 1);
        in.x = core::wrap_tensor(ggml_reshape_3d(g, x, d, steps_, 1), TensorShape::from_dims({1, steps_, d}));

        decoder_ = blocks::build_stack(ctx, graph_, config.stack, weights.stack, in, &state, options);
        spectrum_ = modules::LinearModule({d, config.n_fft + 2, true, blocks::kMatmulPrecision})
                        .build(ctx, decoder_, {weights.lin, weights.lin_bias});
        for (const auto * out : {&spectrum_, &project_out_, &from_quantizer_, &decoder_}) {
            // A reshaped output reads its source's memory, which the allocator would otherwise reuse.
            ggml_set_output(out->tensor->view_src != nullptr ? out->tensor->view_src : out->tensor);
            ggml_set_output(out->tensor);
            ggml_build_forward_expand(graph_, out->tensor);
        }
        core::validate_backend_graph_with_cpu_extra_buffers(execution.backend(), graph_, "LFM3-Audio codec graph");

        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (allocator_ == nullptr || !ggml_gallocr_alloc_graph(allocator_.get(), graph_)) {
            throw runtime::CapacityError("LFM3-Audio codec graph does not fit in device memory at " + std::to_string(frames) + " frames");
        }
    }

    ~ChunkGraph() { core::release_backend_graph_resources(execution_.backend(), graph_, true); }

    ChunkGraph(const ChunkGraph &) = delete;
    ChunkGraph & operator=(const ChunkGraph &) = delete;

    ChunkOutput run(const std::vector<float> & latents, int64_t first_step, bool taps) {
        std::vector<int32_t> positions(static_cast<size_t>(steps_));
        std::vector<int64_t> slots(positions.size());
        for (int64_t i = 0; i < steps_; ++i) {
            positions[static_cast<size_t>(i)] = static_cast<int32_t>(first_step + i);
            slots[static_cast<size_t>(i)] = (first_step + i) % capacity_;
        }
        const auto mask = ring_mask(first_step, steps_, capacity_, window_);
        // Every input is set on every run: the allocator may reuse an input's memory once the graph has read it.
        core::write_tensor_f32(latents_, latents);
        core::write_tensor_i32(positions_, positions);
        ggml_backend_tensor_set(slots_.tensor, slots.data(), 0, slots.size() * sizeof(int64_t));
        ggml_backend_tensor_set(mask_.tensor, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        blocks::compute(execution_, graph_, "codec");

        ChunkOutput out;
        out.spectrum = core::read_tensor_f32(spectrum_.tensor);
        if (taps) {
            out.project_out = core::read_tensor_f32(project_out_.tensor);
            out.from_quantizer = core::read_tensor_f32(from_quantizer_.tensor);
            out.decoder = core::read_tensor_f32(decoder_.tensor);
        }
        return out;
    }

private:
    core::ExecutionContext & execution_;
    int64_t capacity_ = 0;
    int64_t window_ = 0;
    int64_t steps_ = 0;
    ggml_context_ptr ctx_;
    ggml_cgraph * graph_ = nullptr;
    TensorValue latents_, positions_, slots_, mask_;
    TensorValue project_out_, from_quantizer_, decoder_, spectrum_;
    ggml_gallocr_ptr allocator_;
};

void append(std::vector<float> & out, const std::vector<float> & values) {
    out.insert(out.end(), values.begin(), values.end());
}

}  // namespace

std::vector<float> lfm3_fsq_latents(const std::vector<std::vector<int32_t>> & frames, const std::vector<int64_t> & levels) {
    int64_t codes = 1;
    for (const auto level : levels) {
        if (level < 2) {
            fail("FSQ levels must be at least 2");
        }
        codes *= level;
    }
    if (levels.empty()) {
        fail("FSQ needs at least one level");
    }

    const size_t codebooks = frames.empty() ? 0 : frames.front().size();
    std::vector<float> out;
    out.reserve(frames.size() * codebooks * levels.size());
    for (const auto & frame : frames) {
        if (frame.size() != codebooks || codebooks == 0) {
            throw runtime::InvalidRequestError("LFM3-Audio codec frames need one code per codebook");
        }
        for (const int32_t code : frame) {
            if (code < 0 || code >= codes) {
                throw runtime::InvalidRequestError("LFM3-Audio codec code " + std::to_string(code) + " is outside [0, " +
                                                   std::to_string(codes) + "); EOS frames must be dropped before decoding");
            }
            int64_t basis = 1;
            for (const auto level : levels) {
                const int64_t half = level / 2;
                out.push_back(static_cast<float>((code / basis) % level - half) / static_cast<float>(half));
                basis *= level;
            }
        }
    }
    return out;
}

struct Lfm3CodecDecoder::Impl {
    Impl(std::shared_ptr<const assets::TensorSource> source_in, const Lfm3CodecConfig & config_in, core::ExecutionContext & execution_in,
         const Lfm3WeightOptions & options_in)
        : source(std::move(source_in)),
          config(config_in),
          execution(execution_in),
          options(options_in),
          weights(load_weights(*source, config, execution, options)),
          capacity(GGML_PAD(kChunkFrames * config.upsample + config.stack.sliding_window, blocks::kKvRowAlignment)) {
        if (static_cast<int64_t>(config.fsq_levels.size()) * config.codebooks <= 0 || config.upsample <= 0 || config.hop <= 0) {
            fail("config needs FSQ levels, codebooks, an upsample factor and a hop");
        }
        state.emplace(execution, config.stack, capacity, lfm3_reference_precision(options) ? GGML_TYPE_F32 : GGML_TYPE_F16, "codec");
    }

    ChunkGraph & graph(int64_t frames) {
        auto it = graphs.find(frames);
        if (it == graphs.end()) {
            if (graphs.size() >= kKeptGraphs) {
                graphs.erase(std::min_element(graphs.begin(), graphs.end(),
                                              [](const auto & a, const auto & b) { return a.second.last_use < b.second.last_use; }));
            }
            auto built = std::make_unique<ChunkGraph>(weights, config, state->layers, capacity, execution, options, frames);
            it = graphs.emplace(frames, KeptGraph{std::move(built), 0}).first;
        }
        it->second.last_use = ++uses;
        return *it->second.graph;
    }

    // Spectrum rows of `frames`, in pieces of at most kChunkFrames.
    std::vector<float> run(const std::vector<std::vector<int32_t>> & frames, Lfm3CodecTaps * taps) {
        if (!istft) {
            fail("the stream has not been started");
        }
        const auto latents = lfm3_fsq_latents(frames, config.fsq_levels);
        const auto total = static_cast<int64_t>(frames.size());
        const int64_t latent = config.codebooks * static_cast<int64_t>(config.fsq_levels.size());
        if (total > 0 && static_cast<int64_t>(latents.size()) != total * latent) {
            throw runtime::InvalidRequestError("LFM3-Audio codec frames need " + std::to_string(config.codebooks) + " codes");
        }

        std::vector<float> out;
        out.reserve(static_cast<size_t>(total * config.upsample * (config.n_fft + 2)));
        for (int64_t first = 0; first < total; first += kChunkFrames) {
            const int64_t count = std::min(kChunkFrames, total - first);
            const std::vector<float> part(latents.begin() + first * latent, latents.begin() + (first + count) * latent);
            auto chunk = graph(count).run(part, step, taps != nullptr);
            step += count * config.upsample;
            append(out, chunk.spectrum);
            if (taps != nullptr) {
                append(taps->project_out, chunk.project_out);
                append(taps->from_quantizer, chunk.from_quantizer);
                append(taps->decoder, chunk.decoder);
            }
        }
        return out;
    }

    struct KeptGraph {
        std::unique_ptr<ChunkGraph> graph;
        uint64_t last_use = 0;
    };

    std::shared_ptr<const assets::TensorSource> source;
    Lfm3CodecConfig config;
    core::ExecutionContext & execution;
    Lfm3WeightOptions options;
    CodecWeights weights;
    int64_t capacity = 0;
    std::optional<blocks::StackState> state;
    std::map<int64_t, KeptGraph> graphs;  // after the state they write into
    uint64_t uses = 0;
    int64_t step = 0;  // the stream's next absolute step
    std::optional<lfm2_audio::Lfm2StreamingIstft> istft;
};

Lfm3CodecDecoder::Lfm3CodecDecoder(std::shared_ptr<const assets::TensorSource> codec, const Lfm3CodecConfig & config,
                                   core::ExecutionContext & execution, const Lfm3WeightOptions & options)
    : impl_(std::make_unique<Impl>(std::move(codec), config, execution, options)) {}

Lfm3CodecDecoder::~Lfm3CodecDecoder() = default;

void Lfm3CodecDecoder::start_stream() {
    // A stream starts from zero history; stale ring rows are masked, but zeros keep NaN bytes out of q.k.
    impl_->state->clear();
    impl_->step = 0;
    impl_->istft.emplace(impl_->weights.window, impl_->config.hop);
}

std::vector<float> Lfm3CodecDecoder::push(const std::vector<std::vector<int32_t>> & frames) {
    if (frames.empty()) {
        return {};
    }
    const auto rows = impl_->run(frames, nullptr);
    return impl_->istft->push(rows, static_cast<int64_t>(frames.size()) * impl_->config.upsample);
}

std::vector<float> Lfm3CodecDecoder::finish() {
    if (!impl_->istft) {
        fail("the stream has not been started");
    }
    auto out = impl_->istft->finish();
    impl_->istft.reset();
    return out;
}

std::vector<float> Lfm3CodecDecoder::decode(const std::vector<std::vector<int32_t>> & frames) {
    start_stream();
    auto out = push(frames);
    append(out, finish());
    return out;
}

std::vector<float> Lfm3CodecDecoder::spectrum(const std::vector<std::vector<int32_t>> & frames, Lfm3CodecTaps * taps) {
    return impl_->run(frames, taps);
}

}  // namespace engine::community_models::lfm3_audio
