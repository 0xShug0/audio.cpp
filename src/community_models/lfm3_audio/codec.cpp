#include "engine/community_models/lfm3_audio/codec.h"

#include "engine/community_models/lfm2_audio/detokenizer.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/cache_slots.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
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

// FSQ latents of `frames` frames to spectrum rows, continuing the stream state at an absolute step.
class ChunkGraph {
public:
    ChunkGraph(const Lfm3CodecWeights & weights, const Lfm3CodecConfig & config, const std::vector<modules::LfmDecoderLayerState> & state, int64_t capacity,
               core::ExecutionContext & execution, const Lfm3WeightOptions & options, int64_t frames)
        : execution_(execution), capacity_(capacity), window_(config.sliding_window), steps_(frames * config.upsample) {
        ctx_.reset(ggml_init({kGraphNodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(kGraphNodes, false), nullptr, true}));
        if (!ctx_) fail("graph context allocation failed");
        auto * g = ctx_.get();
        core::ModuleBuildContext ctx{g, "lfm3_audio.codec", execution.backend_type()};
        graph_ = ggml_new_graph_custom(g, kGraphNodes, false);

        const int64_t d = config.stack.hidden_size;
        const int64_t latent = config.codebooks * static_cast<int64_t>(config.fsq_levels.size());
        const int64_t q = config.quantizer_dim;
        auto input = [&](ggml_type type, std::initializer_list<int64_t> dims) {
            auto value = core::make_tensor(ctx, type, TensorShape::from_dims(dims));
            ggml_set_input(value.tensor);
            return value;
        };

        latents_ = input(GGML_TYPE_F32, {1, frames, latent});
        modules::LfmDecoderInputs in;
        in.positions = input(GGML_TYPE_I32, {steps_});
        slots_ = input(GGML_TYPE_I64, {steps_});
        in.sliding_mask = input(GGML_TYPE_F16, {steps_, capacity});
        positions_ = in.positions;
        mask_ = *in.sliding_mask;

        const auto project_out = modules::LinearModule({latent, q, true, kLfm3MatmulPrecision})
                           .build(ctx, latents_, {weights.project_out, weights.project_out_bias});
        const auto from_quantizer = modules::LinearModule({q, d, true, kLfm3MatmulPrecision})
                              .build(ctx, project_out, {weights.from_quantizer, weights.from_quantizer_bias});
        // nearest-exact upsampling at an integer factor: step p takes frame p / upsample.
        auto * x = ggml_repeat_4d(g, ggml_reshape_3d(g, from_quantizer.tensor, d, 1, frames), d, config.upsample, frames, 1);
        in.hidden = core::wrap_tensor(ggml_reshape_3d(g, x, d, steps_, 1), TensorShape::from_dims({1, steps_, d}));

        const auto decoder = modules::build_lfm_decoder(ctx, graph_, in, weights.stack, lfm3_decoder_config(config.stack, options), {state, slots_, capacity}).hidden;
        spectrum_ = modules::LinearModule({d, config.n_fft + 2, true, kLfm3MatmulPrecision})
                        .build(ctx, decoder, {weights.lin, weights.lin_bias});
        ggml_set_output(spectrum_.tensor);
        ggml_build_forward_expand(graph_, spectrum_.tensor);
        core::validate_backend_graph_with_cpu_extra_buffers(execution.backend(), graph_, "LFM3-Audio codec graph");

        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (allocator_ == nullptr || !ggml_gallocr_alloc_graph(allocator_.get(), graph_)) {
            throw runtime::CapacityError("LFM3-Audio codec graph does not fit in device memory at " + std::to_string(frames) + " frames");
        }
    }

    ~ChunkGraph() { core::release_backend_graph_resources(execution_.backend(), graph_, true); }

    ChunkGraph(const ChunkGraph &) = delete;
    ChunkGraph & operator=(const ChunkGraph &) = delete;

    std::vector<float> run(const std::vector<float> & latents, int64_t first_step) {
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
        core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        if (core::compute_backend_graph(execution_.backend(), graph_) != GGML_STATUS_SUCCESS) fail("graph compute failed");
        ggml_backend_synchronize(execution_.backend());

        return core::read_tensor_f32(spectrum_.tensor);
    }

private:
    core::ExecutionContext & execution_;
    int64_t capacity_ = 0;
    int64_t window_ = 0;
    int64_t steps_ = 0;
    ggml_context_ptr ctx_;
    ggml_cgraph * graph_ = nullptr;
    TensorValue latents_, positions_, slots_, mask_;
    TensorValue spectrum_;
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
    Impl(Lfm3CodecWeights weights_in, const Lfm3CodecConfig & config_in, core::ExecutionContext & execution_in,
         const Lfm3WeightOptions & options_in)
        : config(config_in),
          execution(execution_in),
          options(options_in),
          weights(std::move(weights_in)),
          capacity(GGML_PAD(kChunkFrames * config.upsample + config.sliding_window, kLfm3KvAlignment)) {
        if (static_cast<int64_t>(config.fsq_levels.size()) * config.codebooks <= 0 || config.upsample <= 0 || config.hop <= 0) {
            fail("config needs FSQ levels, codebooks, an upsample factor and a hop");
        }
        state.emplace(execution, lfm3_decoder_config(config.stack, options), capacity, lfm3_reference_precision(options) ? GGML_TYPE_F32 : GGML_TYPE_F16);
    }

    ChunkGraph & graph(int64_t frames) {
        auto * cached = graphs.find(frames);
        if (!cached) {
            graphs.put(frames, std::make_unique<ChunkGraph>(weights, config, state->layers(), capacity, execution, options, frames));
            cached = graphs.find(frames);
        }
        return **cached;
    }

    // Spectrum rows of `frames`, in pieces of at most kChunkFrames.
    std::vector<float> run(const std::vector<std::vector<int32_t>> & frames) {
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
            auto chunk = graph(count).run(part, step);
            step += count * config.upsample;
            append(out, chunk);
        }
        return out;
    }

    Lfm3CodecConfig config;
    core::ExecutionContext & execution;
    Lfm3WeightOptions options;
    Lfm3CodecWeights weights;
    int64_t capacity;
    std::optional<modules::LfmDecoderState> state;
    runtime::CacheSlots<int64_t, std::unique_ptr<ChunkGraph>> graphs{kKeptGraphs};
    int64_t step = 0;  // the stream's next absolute step
    std::optional<lfm2_audio::Lfm2StreamingIstft> istft;
};

Lfm3CodecDecoder::Lfm3CodecDecoder(Lfm3CodecWeights weights, const Lfm3CodecConfig & config,
                                   core::ExecutionContext & execution, const Lfm3WeightOptions & options)
    : impl_(std::make_unique<Impl>(std::move(weights), config, execution, options)) {}

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
    const auto rows = impl_->run(frames);
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

std::vector<float> Lfm3CodecDecoder::spectrum(const std::vector<std::vector<int32_t>> & frames) {
    return impl_->run(frames);
}

}  // namespace engine::community_models::lfm3_audio
