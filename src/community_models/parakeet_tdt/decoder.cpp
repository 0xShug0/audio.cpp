#include "engine/community_models/parakeet_tdt/decoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/recurrent_modules.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::community_models::parakeet_tdt {
namespace {

using Clock = std::chrono::steady_clock;
struct GgmlDeleter { void operator()(ggml_context* c) const noexcept { if (c) ggml_free(c); } };

int32_t argmax_vocab(const std::vector<float>& v, int64_t vocab_sz) {
    return static_cast<int32_t>(std::distance(v.begin(),
        std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(vocab_sz))));
}

int32_t argmax_dur(const std::vector<float>& v, int64_t vocab_sz, int64_t n_dur) {
    auto s = v.begin() + static_cast<std::ptrdiff_t>(vocab_sz);
    return static_cast<int32_t>(std::distance(s, std::max_element(s, s + static_cast<std::ptrdiff_t>(n_dur))));
}

}  // namespace

struct ParakeetDecoderRuntime::StepGraph {
    std::unique_ptr<ggml_context, GgmlDeleter> ggml;
    ggml_cgraph* graph = nullptr;
    ggml_gallocr_t alloc = nullptr;
    engine::core::HostGraphPlan plan;
    engine::core::TensorValue token_id, enc_frame;
    std::vector<engine::core::TensorValue> h_in, c_in, h_out, c_out;
    engine::core::TensorValue pred_cache, logits;
    ~StepGraph() { if (alloc) ggml_gallocr_free(alloc); }
};

struct ParakeetDecoderRuntime::JointGraph {
    std::unique_ptr<ggml_context, GgmlDeleter> ggml;
    ggml_cgraph* graph = nullptr;
    ggml_gallocr_t alloc = nullptr;
    engine::core::HostGraphPlan plan;
    engine::core::TensorValue enc_frame, pred_cache, logits;
    ~JointGraph() { if (alloc) ggml_gallocr_free(alloc); }
};

ParakeetDecoderRuntime::ParakeetDecoderRuntime(
    std::shared_ptr<const ParakeetTDTAssets> a, std::shared_ptr<const ParakeetWeights> w,
    engine::core::ExecutionContext& ec, size_t arena)
    : assets_(std::move(a)), weights_(std::move(w)), execution_context_(&ec), graph_arena_bytes_(arena) {
    if (!assets_ || !weights_) throw std::runtime_error("decoder requires assets/weights");
}
ParakeetDecoderRuntime::~ParakeetDecoderRuntime() = default;
void ParakeetDecoderRuntime::prepare() { ensure_step_graph(); ensure_joint_graph(); }

void ParakeetDecoderRuntime::ensure_step_graph() {
    if (step_graph_) return;
    const auto t0 = Clock::now();
    const auto& cfg = assets_->config; const auto& dw = weights_->decoder;
    auto g = std::make_unique<StepGraph>();
    ggml_init_params p{graph_arena_bytes_, nullptr, true};
    g->ggml.reset(ggml_init(p));
    engine::core::ModuleBuildContext ctx{g->ggml.get(), "parakeet.decoder", execution_context_->backend_type()};

    g->token_id = engine::core::make_tensor(ctx, GGML_TYPE_I32, engine::core::TensorShape::from_dims({1}));
    g->enc_frame = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, cfg.encoder.hidden_size}));
    for (int64_t l = 0; l < cfg.decoder_layers; ++l) {
        g->h_in.push_back(engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, cfg.decoder_hidden_size})));
        g->c_in.push_back(engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, cfg.decoder_hidden_size})));
    }
    ggml_set_input(g->token_id.tensor); ggml_set_input(g->enc_frame.tensor);
    for (int64_t l = 0; l < cfg.decoder_layers; ++l) { ggml_set_input(g->h_in[static_cast<size_t>(l)].tensor); ggml_set_input(g->c_in[static_cast<size_t>(l)].tensor); }

    auto h = engine::modules::EmbeddingModule({cfg.vocab_size, cfg.decoder_hidden_size}).build(ctx, g->token_id, dw.embedding);
    h = engine::core::reshape_tensor(ctx, h, engine::core::TensorShape::from_dims({1, cfg.decoder_hidden_size}));
    for (int64_t l = 0; l < cfg.decoder_layers; ++l) {
        auto out = engine::modules::LSTMCellModule({cfg.decoder_hidden_size, cfg.decoder_hidden_size})
                       .build(ctx, h, g->h_in[static_cast<size_t>(l)], g->c_in[static_cast<size_t>(l)], dw.lstm_layers[static_cast<size_t>(l)]);
        g->h_out.push_back(out.hidden); g->c_out.push_back(out.cell); h = g->h_out.back();
    }
    auto pred = engine::modules::LinearModule({cfg.decoder_hidden_size, cfg.decoder_hidden_size, true}).build(ctx, h, dw.decoder_projector);
    g->pred_cache = pred;
    auto proj_enc = engine::modules::LinearModule({cfg.encoder.hidden_size, cfg.decoder_hidden_size, true}).build(ctx, g->enc_frame, dw.joint_enc);
    auto joint = engine::modules::AddModule().build(ctx, proj_enc, g->pred_cache);
    joint = engine::modules::ReluModule().build(ctx, joint);
    g->logits = engine::modules::LinearModule({cfg.decoder_hidden_size, static_cast<int64_t>(cfg.vocab_size + cfg.durations.size()), true}).build(ctx, joint, dw.joint_head);

    ggml_set_output(g->logits.tensor); ggml_set_output(g->pred_cache.tensor);
    for (size_t l = 0; l < g->h_out.size(); ++l) { ggml_set_output(g->h_out[l].tensor); ggml_set_output(g->c_out[l].tensor); }

    g->graph = ggml_new_graph(g->ggml.get());
    ggml_build_forward_expand(g->graph, g->logits.tensor); ggml_build_forward_expand(g->graph, g->pred_cache.tensor);
    for (size_t l = 0; l < g->h_out.size(); ++l) { ggml_build_forward_expand(g->graph, g->h_out[l].tensor); ggml_build_forward_expand(g->graph, g->c_out[l].tensor); }

    g->alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_context_->backend()));
    if (!g->alloc || !ggml_gallocr_alloc_graph(g->alloc, g->graph)) throw std::runtime_error("step graph alloc failed");
    engine::core::validate_backend_graph_supported(execution_context_->backend(), g->graph, "Parakeet decoder");
    engine::core::prepare_host_graph_plan(*execution_context_, g->graph, g->plan);

    size_t vt = static_cast<size_t>(cfg.vocab_size + cfg.durations.size());
    logits_scratch_.assign(vt, 0.f);
    hidden_scratch_.assign(static_cast<size_t>(cfg.decoder_layers * cfg.decoder_hidden_size), 0.f);
    cell_scratch_.assign(static_cast<size_t>(cfg.decoder_layers * cfg.decoder_hidden_size), 0.f);
    decoder_cache_scratch_.assign(static_cast<size_t>(cfg.decoder_hidden_size), 0.f);
    hidden_read_scratch_.assign(static_cast<size_t>(cfg.decoder_hidden_size), 0.f);
    cell_read_scratch_.assign(static_cast<size_t>(cfg.decoder_hidden_size), 0.f);
    step_graph_ = std::move(g);
    debug::timing_log_scalar("parakeet.decoder.graph_build_ms", engine::debug::elapsed_ms(t0, Clock::now()));
}

void ParakeetDecoderRuntime::ensure_joint_graph() {
    if (joint_graph_) return;
    const auto t0 = Clock::now(); const auto& cfg = assets_->config;
    auto g = std::make_unique<JointGraph>();
    ggml_init_params p{graph_arena_bytes_, nullptr, true};
    g->ggml.reset(ggml_init(p));
    engine::core::ModuleBuildContext ctx{g->ggml.get(), "parakeet.decoder_joint", execution_context_->backend_type()};
    g->enc_frame = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, cfg.encoder.hidden_size}));
    g->pred_cache = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, cfg.decoder_hidden_size}));
    ggml_set_input(g->enc_frame.tensor); ggml_set_input(g->pred_cache.tensor);
    auto proj_enc = engine::modules::LinearModule({cfg.encoder.hidden_size, cfg.decoder_hidden_size, true}).build(ctx, g->enc_frame, weights_->decoder.joint_enc);
    auto joint = engine::modules::AddModule().build(ctx, proj_enc, g->pred_cache);
    joint = engine::modules::ReluModule().build(ctx, joint);
    g->logits = engine::modules::LinearModule({cfg.decoder_hidden_size, static_cast<int64_t>(cfg.vocab_size + cfg.durations.size()), true}).build(ctx, joint, weights_->decoder.joint_head);
    ggml_set_output(g->logits.tensor);
    g->graph = ggml_new_graph(g->ggml.get()); ggml_build_forward_expand(g->graph, g->logits.tensor);
    g->alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_context_->backend()));
    if (!g->alloc || !ggml_gallocr_alloc_graph(g->alloc, g->graph)) throw std::runtime_error("joint graph alloc failed");
    engine::core::validate_backend_graph_supported(execution_context_->backend(), g->graph, "Parakeet decoder joint");
    engine::core::prepare_host_graph_plan(*execution_context_, g->graph, g->plan);
    joint_graph_ = std::move(g);
    debug::timing_log_scalar("parakeet.decoder.joint_graph_build_ms", engine::debug::elapsed_ms(t0, Clock::now()));
}

int32_t ParakeetDecoderRuntime::run_joint_step(const float* enc, int32_t* out_dur_id) {
    auto& g = *joint_graph_;
    const auto& cfg = assets_->config;
    engine::core::write_tensor_f32(g.enc_frame, enc, static_cast<size_t>(cfg.encoder.hidden_size));
    engine::core::write_tensor_f32(g.pred_cache, decoder_cache_scratch_);
    if (engine::core::compute_graph(*execution_context_, g.graph, g.plan, "Parakeet joint") != GGML_STATUS_SUCCESS)
        throw std::runtime_error("joint compute failed");
    engine::core::read_tensor_f32_into(g.logits.tensor, logits_scratch_);
    if (out_dur_id) *out_dur_id = argmax_dur(logits_scratch_, assets_->config.vocab_size, static_cast<int64_t>(assets_->config.durations.size()));
    return argmax_vocab(logits_scratch_, assets_->config.vocab_size);
}

int32_t ParakeetDecoderRuntime::run_step(int32_t tok, const float* enc, bool pred_valid, int32_t* out_dur_id) {
    auto& g = *step_graph_;
    const auto& cfg = assets_->config;
    const int32_t blank = static_cast<int32_t>(cfg.blank_token_id);
    if (!pred_valid || tok != blank) {
        engine::core::write_tensor_i32(g.token_id, &tok, 1);
        engine::core::write_tensor_f32(g.enc_frame, enc, static_cast<size_t>(cfg.encoder.hidden_size));
        for (int64_t l = 0; l < cfg.decoder_layers; ++l) {
            size_t off = static_cast<size_t>(l * cfg.decoder_hidden_size);
            engine::core::write_tensor_f32(g.h_in[static_cast<size_t>(l)], hidden_scratch_.data() + off, static_cast<size_t>(cfg.decoder_hidden_size));
            engine::core::write_tensor_f32(g.c_in[static_cast<size_t>(l)], cell_scratch_.data() + off, static_cast<size_t>(cfg.decoder_hidden_size));
        }
        if (engine::core::compute_graph(*execution_context_, g.graph, g.plan, "Parakeet step") != GGML_STATUS_SUCCESS)
            throw std::runtime_error("step compute failed");
        engine::core::read_tensor_f32_into(g.logits.tensor, logits_scratch_);
        engine::core::read_tensor_f32_into(g.pred_cache.tensor, decoder_cache_scratch_);
        for (int64_t l = 0; l < cfg.decoder_layers; ++l) {
            size_t off = static_cast<size_t>(l * cfg.decoder_hidden_size);
            engine::core::read_tensor_f32_into(g.h_out[static_cast<size_t>(l)].tensor, hidden_read_scratch_);
            std::copy(hidden_read_scratch_.begin(), hidden_read_scratch_.end(), hidden_scratch_.begin() + static_cast<std::ptrdiff_t>(off));
            engine::core::read_tensor_f32_into(g.c_out[static_cast<size_t>(l)].tensor, cell_read_scratch_);
            std::copy(cell_read_scratch_.begin(), cell_read_scratch_.end(), cell_scratch_.begin() + static_cast<std::ptrdiff_t>(off));
        }
    } else {
        return run_joint_step(enc, out_dur_id);
    }
    if (out_dur_id) *out_dur_id = argmax_dur(logits_scratch_, assets_->config.vocab_size, static_cast<int64_t>(assets_->config.durations.size()));
    return argmax_vocab(logits_scratch_, assets_->config.vocab_size);
}

std::string ParakeetDecoderRuntime::decode_text(const std::vector<int32_t>& ids, bool keep_tags) const {
    std::vector<int32_t> f; f.reserve(ids.size());
    for (auto id : ids) {
        if (id == static_cast<int32_t>(assets_->config.blank_token_id) || id == static_cast<int32_t>(assets_->config.pad_token_id)) continue;
        if (!keep_tags && id >= 0 && id < static_cast<int32_t>(assets_->special_token_ids.size()) && assets_->special_token_ids[static_cast<size_t>(id)]) continue;
        f.push_back(id);
    }
    return assets_->tokenizer->decode_ids(f);
}

std::vector<runtime::WordTimestamp> ParakeetDecoderRuntime::build_token_timestamps(
    const std::vector<int32_t>& ids, const std::vector<int32_t>& durs) const {
    std::vector<runtime::WordTimestamp> out;
    const int64_t spf = assets_->config.frontend.hop_length * assets_->config.encoder.subsampling_factor;
    int64_t fi = 0;
    for (size_t i = 0; i < std::min(ids.size(), durs.size()); ++i) {
        int32_t tid = ids[i]; int64_t tf = fi; fi += std::max<int32_t>(durs[i], 0);
        if (tid == static_cast<int32_t>(assets_->config.blank_token_id) || tid == static_cast<int32_t>(assets_->config.pad_token_id)) continue;
        if (tid >= 0 && tid < static_cast<int32_t>(assets_->special_token_ids.size()) && assets_->special_token_ids[static_cast<size_t>(tid)]) continue;
        runtime::WordTimestamp ts;
        ts.span.start_sample = tf * spf; ts.span.end_sample = (tf + 1) * spf;
        ts.word = assets_->tokenizer->id_to_token()[static_cast<size_t>(tid)]; ts.confidence = 0.f;
        out.push_back(std::move(ts));
    }
    return out;
}

ParakeetDecodedText ParakeetDecoderRuntime::decode(const ParakeetEncodedAudio& enc, const ParakeetDecodeOptions& opts) {
    if (enc.valid_frames <= 0 || enc.hidden_size != assets_->config.encoder.hidden_size)
        throw std::runtime_error("decoder requires encoded frames");
    const auto t0 = Clock::now(); ensure_step_graph();
    engine::core::set_backend_threads(execution_context_->backend(), execution_context_->config().threads);
    const auto& cfg = assets_->config;
    int64_t max_tok = opts.max_tokens > 0 ? opts.max_tokens : (enc.valid_frames * cfg.max_symbols_per_step + 1);
    hidden_scratch_.assign(static_cast<size_t>(cfg.decoder_layers * cfg.decoder_hidden_size), 0.f);
    cell_scratch_.assign(static_cast<size_t>(cfg.decoder_layers * cfg.decoder_hidden_size), 0.f);
    decoder_cache_scratch_.assign(static_cast<size_t>(cfg.decoder_hidden_size), 0.f);

    ParakeetDecodedText out;
    out.token_ids.reserve(static_cast<size_t>(std::min(max_tok + 1, int64_t{4096})));
    out.durations.reserve(out.token_ids.capacity());
    out.token_ids.push_back(static_cast<int32_t>(cfg.blank_token_id));
    out.durations.push_back(0);

    int32_t blank = static_cast<int32_t>(cfg.blank_token_id);
    int32_t in_tok = blank;
    bool pred_valid = false;

    // TDT decode loop with duration-based frame skipping
    int64_t fi = 0;
    while (fi < enc.valid_frames && static_cast<int64_t>(out.token_ids.size()) - 1 < max_tok) {
        const float* f = enc.values.data() + static_cast<std::ptrdiff_t>(fi * enc.hidden_size);
        int inner = 0;
        while (inner < cfg.max_symbols_per_step) {
            int32_t dur_id = 0;
            int32_t tok = run_step(in_tok, f, pred_valid, &dur_id);
            pred_valid = true;

            out.token_ids.push_back(tok);
            int dur_skip = (dur_id >= 0 && dur_id < static_cast<int32_t>(cfg.durations.size())) ? cfg.durations[static_cast<size_t>(dur_id)] : 0;
            out.durations.push_back(dur_skip);
            in_tok = tok;

            if (tok == blank) {
                if (dur_skip > 0) { fi += dur_skip; break; }
                inner++; continue;
            }
            if (dur_skip > 0) { fi += dur_skip; break; }
            inner++;
        }
        if (inner >= cfg.max_symbols_per_step) ++fi;
    }
    out.text = decode_text(out.token_ids, opts.keep_language_tags);
    out.token_timestamps = build_token_timestamps(out.token_ids, out.durations);
    debug::timing_log_scalar("parakeet.decoder_ms", engine::debug::elapsed_ms(t0, Clock::now()));
    return out;
}

}  // namespace engine::community_models::parakeet_tdt
