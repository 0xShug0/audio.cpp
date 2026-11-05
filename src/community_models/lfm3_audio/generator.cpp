#include "engine/community_models/lfm3_audio/generator.h"
#include "engine/community_models/lfm3_audio/prompt.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/errors.h"
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <limits>
#include <numeric>
#include <string>

namespace engine::community_models::lfm3_audio {

namespace {
using core::TensorShape;
using core::TensorValue;
constexpr size_t kInputGraphNodes = 128;

// Model-specific codebook embedding: one frame is the ordered sum of its codebook rows.
class FrameGraph {
public:
    FrameGraph(const TensorValue & table, int64_t books, int64_t frames, core::ExecutionContext & execution)
        : execution_(execution), context_(ggml_init({kInputGraphNodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(kInputGraphNodes, false), nullptr, true})) {
        if (!context_) throw std::runtime_error("LFM3-Audio frame embedding context allocation failed");
        auto * g = context_.get();
        core::ModuleBuildContext ctx{g, "lfm3_audio.frame_embedding", execution.backend_type()};
        const auto ids = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({frames, books}));
        ids_ = ids.tensor;
        ggml_set_input(ids_);
        const auto d = table.shape.dims[1];
        const auto gathered = modules::EmbeddingModule({table.shape.dims[0], d}).build(ctx, ids, table);
        const auto sums = modules::ReduceSumModule({1}).build(ctx, gathered);
        output_ = core::reshape_tensor(ctx, sums, TensorShape::from_dims({frames, d}));
        ggml_set_output(output_.tensor);
        graph_ = ggml_new_graph_custom(g, kInputGraphNodes, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        core::validate_backend_graph_with_cpu_extra_buffers(execution.backend(), graph_, "LFM3 frame embedding");
        buffer_.reset(ggml_backend_alloc_ctx_tensors(g, execution.backend()));
        if (!buffer_) throw runtime::CapacityError("LFM3-Audio frame embedding allocation failed");
    }
    ~FrameGraph() { core::release_backend_graph_resources(execution_.backend(), graph_, true); }
    TensorValue run(const std::vector<int32_t> & ids) {
        ggml_backend_tensor_set(ids_, ids.data(), 0, ids.size() * sizeof(int32_t));
        core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        if (core::compute_backend_graph(execution_.backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LFM3-Audio frame embedding failed");
        }
        ggml_backend_synchronize(execution_.backend());
        return output_;
    }
private:
    core::ExecutionContext & execution_;
    ggml_context_ptr context_;
    ggml_tensor * ids_ = nullptr;
    TensorValue output_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_ptr buffer_;
};

// Only used when the caller explicitly projects an already-computed hidden row.
class HeadGraph {
public:
    HeadGraph(const TensorValue & weight, int64_t hidden, int64_t vocab, core::ExecutionContext & execution)
        : execution_(execution), context_(ggml_init({kInputGraphNodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(kInputGraphNodes, false), nullptr, true})) {
        if (!context_) throw std::runtime_error("LFM3-Audio text head context allocation failed");
        core::ModuleBuildContext ctx{context_.get(), "lfm3_audio.text_head", execution.backend_type()};
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 1, hidden}));
        ggml_set_input(input_.tensor);
        output_ = modules::LinearModule({hidden, vocab, false, kLfm3MatmulPrecision}).build(ctx, input_, {weight, std::nullopt});
        ggml_set_output(output_.tensor);
        graph_ = ggml_new_graph_custom(context_.get(), kInputGraphNodes, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        core::validate_backend_graph_with_cpu_extra_buffers(execution.backend(), graph_, "LFM3 text head");
        buffer_.reset(ggml_backend_alloc_ctx_tensors(context_.get(), execution.backend()));
        if (!buffer_) throw runtime::CapacityError("LFM3-Audio text head allocation failed");
    }
    ~HeadGraph() { core::release_backend_graph_resources(execution_.backend(), graph_, true); }
    std::vector<float> run(const std::vector<float> & hidden) {
        core::write_tensor_f32(input_, hidden);
        core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        if (core::compute_backend_graph(execution_.backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LFM3-Audio text head failed");
        }
        ggml_backend_synchronize(execution_.backend());
        return core::read_tensor_f32(output_.tensor);
    }
private:
    core::ExecutionContext & execution_;
    ggml_context_ptr context_;
    TensorValue input_, output_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_ptr buffer_;
};
} // namespace

struct Lfm3BackboneRuntime::Impl {
    Impl(Lfm3BackboneWeights weights_in, const Lfm3BackboneConfig & config_in, int64_t books, int64_t vocab,
         core::ExecutionContext & execution_in, const Lfm3WeightOptions & options_in)
        : weights(std::move(weights_in)), config(config_in), codebooks(books), audio_vocab(vocab), execution(execution_in), options(options_in),
          decoder(lfm3_decoder_config(config.stack, options), {weights.stack, weights.token_lookup, modules::LinearWeights{weights.text_head, std::nullopt}},
                  execution, kLfm3MaxConversationSteps / kPrefillBlock + 3) {}

    TensorValue frame_embedding(const std::vector<std::vector<int32_t>> & frames) {
        std::vector<int32_t> rows;
        rows.reserve(frames.size() * static_cast<size_t>(codebooks));
        for (const auto & frame : frames) {
            if (static_cast<int64_t>(frame.size()) != codebooks) throw runtime::InvalidRequestError("LFM3-Audio frame needs one code per codebook");
            for (size_t k = 0; k < frame.size(); ++k) {
                if (frame[k] < 0 || frame[k] >= audio_vocab) throw runtime::InvalidRequestError("LFM3-Audio code is outside its codebook");
                rows.push_back(static_cast<int32_t>(k * audio_vocab + frame[k]));
            }
        }
        const auto count = static_cast<int64_t>(frames.size());
        auto * cached = embeddings.find(count);
        if (!cached) {
            embeddings.put(count, std::make_unique<FrameGraph>(weights.frame_lookup, codebooks, count, execution));
            cached = embeddings.find(count);
        }
        return (*cached)->run(rows);
    }

    void start(int64_t capacity) {
        if (!stream || stream->cache_steps() != capacity) {
            stream = decoder.create_decode_session(capacity, config.sliding_window,
                lfm3_reference_precision(options) ? GGML_TYPE_F32 : GGML_TYPE_F16);
        }
    }

    Lfm3BackboneWeights weights;
    Lfm3BackboneConfig config;
    int64_t codebooks, audio_vocab;
    core::ExecutionContext & execution;
    Lfm3WeightOptions options;
    modules::LfmDecoderRuntime decoder;
    std::unique_ptr<modules::LfmDecoderRuntime::DecodeSession> stream;
    runtime::CacheSlots<int64_t, std::unique_ptr<FrameGraph>> embeddings{2};
    std::unique_ptr<HeadGraph> head;
};

Lfm3BackboneRuntime::Lfm3BackboneRuntime(Lfm3BackboneWeights weights, const Lfm3BackboneConfig & config,
    int64_t codebooks, int64_t audio_vocab_size, core::ExecutionContext & execution, const Lfm3WeightOptions & options)
    : impl_(std::make_unique<Impl>(std::move(weights), config, codebooks, audio_vocab_size, execution, options)) {}
Lfm3BackboneRuntime::~Lfm3BackboneRuntime() = default;

std::vector<float> Lfm3BackboneRuntime::prefill(const Lfm3Prompt & prompt, int32_t audio_id, int32_t audio_out_id,
    const Lfm3AudioEmbeddings & audio, int64_t max_new_steps, Lfm3Output output) {
    auto & m = *impl_;
    if (m.stream) m.stream->reset();
    const int64_t d = m.config.stack.hidden_size;
    const auto steps = static_cast<int64_t>(prompt.ids.size());
    if (steps == 0 || max_new_steps < 0) {
        throw runtime::InvalidRequestError("prefill needs a prompt and a nonnegative step budget");
    }
    const int64_t limit = GGML_PAD(kLfm3MaxConversationSteps, kPrefillBlock);
    if (steps > limit || max_new_steps > limit - steps) {
        throw runtime::CapacityError("LFM3-Audio request needs " + std::to_string(steps) + " prompt steps plus " +
                                     std::to_string(max_new_steps) + " more, more than " + std::to_string(limit));
    }

    int64_t audio_slots = 0;
    int64_t frame_slots = 0;
    for (const int32_t id : prompt.ids) {
        if (id < 0 || id >= m.config.vocab_size) {
            throw runtime::InvalidRequestError("prompt token id " + std::to_string(id) + " is outside the vocabulary");
        }
        audio_slots += id == audio_id ? 1 : 0;
        frame_slots += id == audio_out_id ? 1 : 0;
    }
    if (audio_slots != audio.tokens || audio.values.size() != static_cast<size_t>(audio.tokens * d) || (audio.tokens > 0 && audio.hidden != d)) {
        throw runtime::InvalidRequestError("the prompt has " + std::to_string(audio_slots) + " <|audio|> slots but " + std::to_string(audio.tokens) +
             " audio embeddings of the backbone width");
    }
    if (!std::all_of(audio.values.begin(), audio.values.end(), [](float v) { return std::isfinite(v); })) {
        throw runtime::InvalidRequestError("audio embeddings are not finite");
    }
    if (frame_slots != static_cast<int64_t>(prompt.frames.size())) {
        throw runtime::InvalidRequestError("the prompt has " + std::to_string(frame_slots) + " <|audio_out|> slots but " + std::to_string(prompt.frames.size()) + " frames");
    }

    const auto started = std::chrono::steady_clock::now();
    m.start(GGML_PAD(steps + max_new_steps, kPrefillBlock));
    const auto built = m.stream->graphs_built();
    const auto frames = embed_frames(prompt.frames);
    size_t audio_row = 0, frame_row = 0;
    std::vector<float> result;
    for (int64_t begin = 0; begin < steps; begin += kPrefillBlock) {
        const int64_t end = std::min(begin + kPrefillBlock, steps);
        modules::LfmDecoderForward request;
        request.token_ids.assign(prompt.ids.begin() + begin, prompt.ids.begin() + end);
        request.attention_rows = GGML_PAD(end, kPrefillBlock);
        for (int64_t i = begin; i < end; ++i) {
            const auto id = prompt.ids[static_cast<size_t>(i)];
            const float * row = id == audio_id ? audio.values.data() + (audio_row++) * d :
                                id == audio_out_id ? frames.data() + (frame_row++) * d : nullptr;
            if (row) {
                request.replacement_positions.push_back(static_cast<int32_t>(i - begin));
                request.replacement_values.insert(request.replacement_values.end(), row, row + d);
            }
        }
        if (end == steps) {
            request.readout_positions = {static_cast<int32_t>(end - begin - 1)};
            request.readout = output == Lfm3Output::Logits ? modules::LfmDecoderReadout::Logits : modules::LfmDecoderReadout::Hidden;
        }
        const auto encoded = m.stream->prefill_device(request);
        if (end == steps) result = core::read_tensor_f32(encoded.output.tensor);
    }
    debug::timing_log_scalar("lfm3_audio.backbone.prefill.ms", debug::elapsed_ms(started));
    debug::timing_log_scalar("lfm3_audio.backbone.prefill.built_blocks", m.stream->graphs_built() - built);
    return result;
}

std::vector<float> Lfm3BackboneRuntime::step_text(int32_t token, Lfm3Output output) {
    auto & m = *impl_;
    if (!m.stream || !m.stream->valid_steps()) throw std::logic_error("LFM3-Audio step before prefill");
    modules::LfmDecoderForward request;
    request.token_ids = {token};
    request.readout = output == Lfm3Output::Logits ? modules::LfmDecoderReadout::Logits : modules::LfmDecoderReadout::Hidden;
    return m.stream->decode_step(request).values;
}

std::vector<float> Lfm3BackboneRuntime::step_frame(const std::vector<int32_t> & codes, Lfm3Output output) {
    auto & m = *impl_;
    if (!m.stream || !m.stream->valid_steps()) throw std::logic_error("LFM3-Audio step before prefill");
    const auto frame = m.frame_embedding({codes});
    modules::LfmDecoderForward request;
    request.hidden_tensor = &frame;
    request.readout = output == Lfm3Output::Logits ? modules::LfmDecoderReadout::Logits : modules::LfmDecoderReadout::Hidden;
    return m.stream->decode_step(request).values;
}

std::vector<float> Lfm3BackboneRuntime::logits(const std::vector<float> & hidden) {
    auto & m = *impl_;
    if (hidden.size() != static_cast<size_t>(m.config.stack.hidden_size)) throw runtime::InvalidRequestError("LFM3-Audio text head requires one hidden row");
    if (!m.head) m.head = std::make_unique<HeadGraph>(m.weights.text_head, m.config.stack.hidden_size, m.config.vocab_size, m.execution);
    return m.head->run(hidden);
}

std::vector<float> Lfm3BackboneRuntime::embed_frames(const std::vector<std::vector<int32_t>> & frames) {
    if (frames.empty()) return {};
    return core::read_tensor_f32(impl_->frame_embedding(frames).tensor);
}

int64_t Lfm3BackboneRuntime::position() const noexcept { return impl_->stream ? impl_->stream->valid_steps() : 0; }
int64_t Lfm3BackboneRuntime::capacity() const noexcept { return impl_->stream ? impl_->stream->cache_steps() : 0; }

namespace {

constexpr int64_t kCodebooks = 12;

// The text stream's seed: splitmix64 of the request seed with "text" in its
// top bytes, so text and audio draws of one seed come from different streams.
uint64_t text_seed(uint64_t seed) {
    uint64_t z = (seed ^ 0x7465787400000000ull) + 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

void require_finite(const std::vector<float> & logits, const char * what) {
    if (logits.empty() || !std::all_of(logits.begin(), logits.end(), [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error(std::string("LFM3-Audio ") + what + " logits are empty or non-finite");
    }
}

// The upstream rule: greedy when the temperature is 0, or unset with no top-k; an unset temperature with top-k
// samples at 1.
sampling::HfSamplingOptions hf_options(std::optional<float> temperature, int64_t top_k, float repetition_penalty, const char * what) {
    if ((temperature && (!std::isfinite(*temperature) || *temperature < 0.0f)) || top_k < 0 || !std::isfinite(repetition_penalty) ||
        repetition_penalty <= 0.0f) {
        throw std::invalid_argument(std::string("LFM3-Audio ") + what + " sampling needs temperature >= 0, top_k >= 0 and repetition_penalty > 0");
    }
    sampling::HfSamplingOptions out;
    out.do_sample = !(temperature ? *temperature == 0.0f : top_k == 0);
    out.temperature = out.do_sample ? temperature.value_or(1.0f) : 1.0f;
    out.top_k = top_k;
    out.repetition_penalty = repetition_penalty;
    return out;
}

}  // namespace

Lfm3AudioSampler::Lfm3AudioSampler(const Lfm3AudioSampling & sampling, uint64_t seed)
    : sampling_(sampling),
      options_(hf_options(sampling.temperature, sampling.top_k, sampling.repetition_penalty, "audio")),
      rng_(static_cast<uint32_t>(seed)) {
    if (sampling.penalty_window < 0) {
        throw std::invalid_argument("LFM3-Audio audio repetition penalty window must be >= 0");
    }
}

Lfm3Codes Lfm3AudioSampler::next_order() {
    Lfm3Codes order;
    std::iota(order.begin(), order.end(), 0);
    for (size_t i = order.size() - 1; i > 0; --i) {
        std::swap(order[i], order[rng_() % (i + 1)]);
    }
    return order;
}

int32_t Lfm3AudioSampler::pick(int64_t codebook, std::vector<float> & logits) {
    if (codebook < 0 || codebook >= kCodebooks) {
        throw std::invalid_argument("LFM3-Audio audio codebook " + std::to_string(codebook) + " is outside [0, 12)");
    }
    require_finite(logits, "audio code");
    scratch_.reserve_vocab(logits.size());
    return sampler_.sample(logits, history_[static_cast<size_t>(codebook)], options_, scratch_, rng_, nullptr, "lfm3_audio audio code");
}

void Lfm3AudioSampler::commit(const Lfm3Codes & frame) {
    const auto window = static_cast<size_t>(sampling_.penalty_window);
    for (size_t k = 0; k < frame.size(); ++k) {
        if (frame[k] < 0) {
            throw std::invalid_argument("LFM3-Audio audio history got a negative code");
        }
        auto & history = history_[k];
        history.push_back(frame[k]);
        if (history.size() > window) {
            history.erase(history.begin(), history.end() - static_cast<std::ptrdiff_t>(window));
        }
    }
}

Lfm3TextSampler::Lfm3TextSampler(const Lfm3TextSampling & sampling, uint64_t seed)
    : options_(hf_options(sampling.temperature, sampling.top_k, 1.0f, "text")),
      rng_(static_cast<uint32_t>(text_seed(seed))) {}

int32_t Lfm3TextSampler::pick(std::vector<float> & logits, const std::vector<int32_t> & suppress) {
    require_finite(logits, "text");
    for (const auto id : suppress) {
        if (id < 0 || static_cast<size_t>(id) >= logits.size()) {
            throw std::invalid_argument("LFM3-Audio suppressed token " + std::to_string(id) + " is outside the vocabulary");
        }
        logits[static_cast<size_t>(id)] = -std::numeric_limits<float>::infinity();
    }
    scratch_.reserve_vocab(logits.size());
    return sampler_.sample(logits, {}, options_, scratch_, rng_, nullptr, "lfm3_audio text token");
}

Lfm3ModeMachine::Lfm3ModeMachine(Lfm3Mode initial, const Lfm3TokenIds & ids, std::vector<int32_t> eos_ids, int64_t n_text,
                                 int64_t n_audio, int32_t end_of_audio)
    : mode_(initial),
      text_end_(ids.text_end),
      audio_start_(ids.audio_start),
      mixed_start_(ids.mixed_start),
      eos_ids_(std::move(eos_ids)),
      n_text_(n_text),
      n_audio_(n_audio),
      end_of_audio_(end_of_audio),
      slots_left_(n_text) {
    if (n_text <= 0 || n_audio <= 0) {
        throw std::runtime_error("LFM3-Audio generation needs positive interleaved text and audio block sizes");
    }
}

Lfm3SlotKind Lfm3ModeMachine::next_slot() const {
    return mode_ == Lfm3Mode::Audio || (mode_ == Lfm3Mode::Interleaved && speaking_) ? Lfm3SlotKind::Audio : Lfm3SlotKind::Text;
}

void Lfm3ModeMachine::advance(const Lfm3Step & step) {
    if (finished_) {
        throw std::runtime_error("LFM3-Audio generation advanced past the end of the turn");
    }

    const bool frame = !step.codes.empty();
    if (frame != (next_slot() == Lfm3SlotKind::Audio)) {
        throw std::runtime_error(frame ? "LFM3-Audio generation got a frame for a text slot"
                                       : "LFM3-Audio generation got a text token for an audio slot");
    }

    // The block counter below follows the mode this slot started in.
    const bool interleaving = mode_ == Lfm3Mode::Interleaved;
    if (frame) {
        if (std::find(step.codes.begin(), step.codes.end(), end_of_audio_) != step.codes.end()) {
            mode_ = Lfm3Mode::Text;
        } else if (interleaving && text_done_) {
            ++tail_frames_;
        }
    } else if (step.token == text_end_) {
        // Checked before the eos ids, which list <|text_end|> too: inside a block it only ends the text.
        if (interleaving) {
            text_done_ = speaking_ = true;
        } else {
            finished_ = true;
        }
    } else if (std::find(eos_ids_.begin(), eos_ids_.end(), step.token) != eos_ids_.end()) {
        finished_ = true;
    } else if (interleaving) {
        // The mode markers only open a block; inside one they are ordinary text.
    } else if (step.token == audio_start_) {
        mode_ = Lfm3Mode::Audio;
    } else if (step.token == mixed_start_) {
        mode_ = Lfm3Mode::Interleaved;
        slots_left_ = n_text_;
        speaking_ = text_done_ = false;
        tail_frames_ = 0;
    }

    if (interleaving && !text_done_ && --slots_left_ == 0) {
        speaking_ = !speaking_;
        slots_left_ = speaking_ ? n_audio_ : n_text_;
    }
}

bool Lfm3ModeMachine::finished() const noexcept {
    return finished_;
}

Lfm3Mode Lfm3ModeMachine::mode() const noexcept {
    return mode_;
}

bool Lfm3ModeMachine::force_audio_eos(int64_t max_audio_tail_frames) const noexcept {
    return max_audio_tail_frames > 0 && mode_ == Lfm3Mode::Interleaved && text_done_ && tail_frames_ >= max_audio_tail_frames;
}

std::vector<int32_t> lfm3_text_suppress(const Lfm3TokenIds & ids) {
    return {ids.audio, ids.audio_out};
}

struct Lfm3Generator::Impl {
    Impl(Lfm3BackboneRuntime & backbone_in, Lfm3DepthDecoderRuntime * depth_in, const Lfm3Tokenizer & tokenizer,
         const Lfm3GenerationConfig & config, const Lfm3DepthConfig & depth_config, Lfm3Prompt prompt_in,
         const Lfm3AudioEmbeddings & audio_in, Lfm3GenerationOptions options_in)
        : backbone(backbone_in),
          depth(depth_in),
          ids(tokenizer.ids()),
          end_of_audio(depth_config.end_of_audio()),
          codebooks(depth_config.codebooks),
          schedule(depth_config.maskgit_schedule),
          prompt(std::move(prompt_in)),
          audio(audio_in),
          options(std::move(options_in)),
          machine(lfm3_initial_mode(prompt, ids, end_of_audio), ids, config.specials.eos_ids, config.interleaved_text,
                  config.interleaved_audio, end_of_audio),
          audio_sampler(options.audio, options.seed),
          text_sampler(options.text, options.seed),
          suppress(lfm3_text_suppress(ids)) {
        if (options.max_steps <= 0) {
            throw std::runtime_error("LFM3-Audio generation needs max_steps > 0");
        }
        if (codebooks != static_cast<int64_t>(Lfm3Codes{}.size())) {
            throw std::runtime_error("LFM3-Audio generation expects 12 codebooks");
        }
        suppress.insert(suppress.end(), options.suppress.begin(), options.suppress.end());
    }

    [[nodiscard]] Lfm3Output output_for_next() const {
        return machine.next_slot() == Lfm3SlotKind::Text ? Lfm3Output::Logits : Lfm3Output::Hidden;
    }

    Lfm3Codes next_frame() {
        if (depth == nullptr) {
            throw std::runtime_error("LFM3-Audio generation reached an audio slot without a depth decoder");
        }
        Lfm3Codes codes;
        if (machine.force_audio_eos(options.max_audio_tail_frames)) {
            codes.fill(end_of_audio);
            return codes;
        }
        codes = lfm3_maskgit_frame(*depth, out, audio_sampler.next_order(), schedule, audio_sampler, end_of_audio);
        if (codes[0] != end_of_audio) {  // lfm3_maskgit_frame makes an EOS frame all end_of_audio
            audio_sampler.commit(codes);
        }
        return codes;
    }

    Lfm3BackboneRuntime & backbone;
    Lfm3DepthDecoderRuntime * depth;
    Lfm3TokenIds ids;
    int32_t end_of_audio;
    int64_t codebooks;
    std::vector<int64_t> schedule;
    Lfm3Prompt prompt;
    Lfm3AudioEmbeddings audio;  // until the prefill
    Lfm3GenerationOptions options;
    Lfm3ModeMachine machine;
    Lfm3AudioSampler audio_sampler;
    Lfm3TextSampler text_sampler;
    std::vector<int32_t> suppress;
    std::vector<float> out;  // the backbone's output for the next slot: logits (text) or hidden (audio)
    std::vector<Lfm3Step> steps;
    bool started = false;
    bool advance_pending = false;
    bool ended = false;
    bool done = false;
};

Lfm3Generator::Lfm3Generator(Lfm3BackboneRuntime & backbone, Lfm3DepthDecoderRuntime * depth, const Lfm3Tokenizer & tokenizer,
                             const Lfm3GenerationConfig & config, const Lfm3DepthConfig & depth_config, Lfm3Prompt prompt,
                             const Lfm3AudioEmbeddings & audio, Lfm3GenerationOptions options)
    : impl_(std::make_unique<Impl>(backbone, depth, tokenizer, config, depth_config, std::move(prompt), audio, std::move(options))) {}

Lfm3Generator::~Lfm3Generator() = default;

// One slot per call, as upstream's generate() loop.
std::optional<Lfm3Step> Lfm3Generator::next() {
    auto & g = *impl_;
    if (g.done) {
        return std::nullopt;
    }
    if (!g.started) {
        g.out = g.backbone.prefill(g.prompt, g.ids.audio, g.ids.audio_out, g.audio, g.options.max_steps, g.output_for_next());
        g.audio = {};
        g.started = true;
    }
    if (g.advance_pending) {
        const auto & previous = g.steps.back();
        const auto output = g.output_for_next();
        g.out = previous.codes.empty() ? g.backbone.step_text(previous.token, output) : g.backbone.step_frame(previous.codes, output);
        g.advance_pending = false;
    }

    Lfm3Step step;
    if (g.machine.next_slot() == Lfm3SlotKind::Audio) {
        const auto codes = g.next_frame();
        step.codes.assign(codes.begin(), codes.end());
    } else {
        step.token = g.text_sampler.pick(g.out, g.suppress);
    }
    g.machine.advance(step);
    g.steps.push_back(step);

    const bool eos_frame = !step.codes.empty() && step.codes[0] == g.end_of_audio;
    g.ended = g.machine.finished() || (g.options.stop_after_audio_eos && eos_frame);
    g.done = g.ended || static_cast<int64_t>(g.steps.size()) >= g.options.max_steps;
    // Yield the requested output before doing work for the following pull.
    g.advance_pending = !g.done;
    return step;
}

bool Lfm3Generator::ended() const noexcept {
    return impl_->ended;
}

const std::vector<Lfm3Step> & Lfm3Generator::steps() const noexcept {
    return impl_->steps;
}

}  // namespace engine::community_models::lfm3_audio
