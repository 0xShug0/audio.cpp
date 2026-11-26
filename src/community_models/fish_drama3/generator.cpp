#include "engine/framework/runtime/moe_options.h"
#include "engine/community_models/fish_drama3/generator.h"
#include "cache_key.h"
#include "prefill_cache.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/sampling/hf_sampler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <cstdlib>
#include <iostream>
#include <map>
#include <numeric>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>

namespace engine::models::fish_drama3 {
namespace {

constexpr size_t kDiskPrefillEntries = 4;
constexpr uint64_t kDiskPrefillBytes = 1024ULL * 1024 * 1024;

struct PerfPreset {
    std::string_view name;
    bool fused_moe;
    ggml_type kv_cache_type;
    int64_t kv_cache_alignment;
    bool device_sampling;
};

constexpr std::array<PerfPreset, 2> kPerfPresets{{
    {"off", false, GGML_TYPE_F16, 1, false},
    {"cuda_perf", true, GGML_TYPE_F16, 256, true},
}};

struct SamplerScratch {
    std::vector<int32_t> order;
    std::vector<double> weights;
    std::vector<double> probabilities;
};

int32_t sample(const std::vector<float> & logits, float temperature, int top_k,
               float top_p, std::mt19937 & random, SamplerScratch & scratch) {
    if (logits.empty()) throw std::runtime_error("Drama 3 sampler has no logits");
    if (temperature == 0 || top_k == 1)
        return sampling::HfLogitsProcessor::argmax(logits.data(), logits.size(), "Drama 3");
    auto & order = scratch.order;
    order.resize(logits.size());
    std::iota(order.begin(), order.end(), 0);
    const auto count = std::min(logits.size(), static_cast<size_t>(top_k));
    const auto compare = [&](int32_t left, int32_t right) {
        return logits[left] == logits[right] ? left < right : logits[left] > logits[right];
    };
    const double maximum = *std::max_element(logits.begin(), logits.end()) / temperature;
    double total = 0;
    // Top-p is defined against the full distribution, before top-k filtering.
    auto & weights = scratch.weights;
    weights.resize(logits.size());
    for (size_t index = 0; index < logits.size(); ++index) {
        weights[index] = std::exp(logits[index] / temperature - maximum);
        total += weights[index];
    }
    if (!(total > 0) || !std::isfinite(total)) throw std::runtime_error("Drama 3 sampler encountered nonfinite logits");
    size_t sorted = std::min(count, size_t{64});
    for (;;) {
        std::partial_sort(order.begin(), order.begin() + sorted, order.end(), compare);
        double mass = 0;
        for (size_t rank = 0; rank < sorted; ++rank) mass += weights[order[rank]] / total;
        if (sorted == count || mass > top_p) break;
        sorted = std::min(count, sorted * 2);
    }
    auto & probabilities = scratch.probabilities;
    probabilities.assign(count, 0);
    double cumulative = 0;
    for (size_t rank = 0; rank < sorted; ++rank) {
        cumulative += weights[order[rank]] / total;
        if (rank > 0 && cumulative > top_p) break;
        probabilities[rank] = weights[order[rank]];
    }
    std::discrete_distribution<size_t> choice(probabilities.begin(), probabilities.end());
    return order[choice(random)];
}

std::vector<float> last_hidden(const std::vector<float> & values, int64_t width) {
    if (values.size() < static_cast<size_t>(width)) throw std::runtime_error("Drama 3 backbone returned too few hidden values");
    return {values.end() - width, values.end()};
}

void cache_backend_identity(Sha256 & hash, const core::ExecutionContext & execution) {
    const auto & config = execution.config();
    cache_key_field(hash, std::to_string(static_cast<int>(execution.backend_type())));
    cache_key_field(hash, std::to_string(config.device));
    cache_key_field(hash, std::to_string(config.threads));
    cache_key_field(hash, config.impl);
    ggml_backend_dev_props props{};
    ggml_backend_dev_get_props(ggml_backend_get_device(execution.backend()), &props);
    cache_key_field(hash, props.name ? props.name : "");
    cache_key_field(hash, props.description ? props.description : "");
    cache_key_field(hash, props.device_id ? props.device_id : "");
    for (const auto * name : {"GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F", "GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F",
             "GGML_CUDA_DISABLE_FUSION", "GGML_CUDA_DISABLE_GRAPHS", "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG"}) {
        const auto * value = std::getenv(name);
        cache_key_field(hash, value ? ("=" + std::string(value)) : std::string{});
    }
}

}  // namespace

Drama3Generator::Drama3Generator(std::shared_ptr<const Drama3Assets> assets,
    core::ExecutionContext & execution, const runtime::SessionOptions & options, bool streaming)
    : assets_(std::move(assets)), tokenizer_(assets_->resources),
      cache_steps_(runtime::parse_positive_i64_option(options.options, {"fish_drama3.context_length"}, 2048)),
      reference_cache_(parse_reference_cache_slots(options.options)) {
    if (cache_steps_ > 262144) throw std::runtime_error("Drama 3 context_length exceeds the checkpoint's maximum position count");
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.profile_first_audio"})) {
        profile_first_audio_ = runtime::parse_bool_option(*value, "fish_drama3.profile_first_audio");
    }
    codec_batch_frames_ = static_cast<size_t>(runtime::parse_positive_i64_option(
        options.options, {"fish_drama3.codec_batch_frames"}, 4));
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.codec_async_decode"})) {
        codec_async_decode_ = runtime::parse_bool_option(*value, "fish_drama3.codec_async_decode");
    }
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.reuse_prefill"})) {
        reuse_prefill_ = runtime::parse_bool_option(*value, "fish_drama3.reuse_prefill");
    }
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.reuse_reference_prefill"})) {
        reuse_reference_prefill_ = runtime::parse_bool_option(*value, "fish_drama3.reuse_reference_prefill");
    }
    auto backbone_config = assets_->backbone;
    backbone_config.decode_cache_steps = cache_steps_;
    const auto perf_mode = runtime::find_option(options.options, {"fish_drama3.perf_mode"}).value_or("off");
    const auto preset = std::find_if(kPerfPresets.begin(), kPerfPresets.end(),
        [&](const auto & entry) { return entry.name == perf_mode; });
    if (preset == kPerfPresets.end()) throw std::runtime_error("Invalid fish_drama3.perf_mode: " + perf_mode);
    device_sampling_ = preset->device_sampling;
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.device_sampling"})) {
        device_sampling_ = runtime::parse_bool_option(*value, "fish_drama3.device_sampling");
    }
    backbone_config.fused_moe = preset->fused_moe;
    backbone_config.kv_cache_type = preset->kv_cache_type;
    backbone_config.kv_cache_alignment = preset->kv_cache_alignment;
    backbone_config.kv_cache_alignment = runtime::parse_positive_i64_option(
        options.options, {"fish_drama3.kv_cache_alignment"}, backbone_config.kv_cache_alignment);
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.fused_moe"}))
        backbone_config.fused_moe = runtime::parse_bool_option(*value, "fish_drama3.fused_moe");
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.kv_cache_type"})) {
        if (*value == "f16") backbone_config.kv_cache_type = GGML_TYPE_F16;
        else if (*value == "f32") backbone_config.kv_cache_type = GGML_TYPE_F32;
        else throw std::runtime_error("fish_drama3.kv_cache_type requires f32 or f16");
    }
    debug::trace_log_scalar("fish_drama3.perf_mode", perf_mode);
    debug::trace_log_scalar("fish_drama3.device_sampling", device_sampling_);
    debug::trace_log_scalar("fish_drama3.fused_moe", backbone_config.fused_moe);
    debug::trace_log_scalar("fish_drama3.kv_cache_type", std::string_view(backbone_config.kv_cache_type == GGML_TYPE_F16 ? "f16" : "f32"));
    debug::trace_log_scalar("fish_drama3.kv_cache_alignment", backbone_config.kv_cache_alignment);
    auto moe_defaults = backbone_config.moe;
    moe_defaults.cpu_layers = {0, 1, 2, 3};
    backbone_config.moe = runtime::resolve_moe_options(options.options, "fish_drama3",
        std::move(moe_defaults), static_cast<size_t>(backbone_config.layers));
    if (backbone_config.moe.pin_host_weights &&
        backbone_config.moe.residency != moe::Residency::Staged)
        throw std::runtime_error("fish_drama3.pin_experts requires fish_drama3.disk_experts=true");
    const auto backbone_prefill_cache_mb = runtime::parse_i64_option(options.options, {"fish_drama3.backbone_prefill_cache_mb"}).value_or(256);
    if (backbone_prefill_cache_mb < 0 || backbone_prefill_cache_mb > 4096) throw std::runtime_error("Drama 3 backbone_prefill_cache_mb must be between 0 and 4096");
    backbone_config.prefill_cache_bytes = static_cast<size_t>(backbone_prefill_cache_mb) * 1024u * 1024u;
    backbone_config.profile_prefill_cache = profile_first_audio_;
    backbone_ = std::make_unique<Drama3BackboneDecoderRuntime>(assets_->model_weights, backbone_config, execution,
        64 * 1024 * 1024, 8 * 1024 * 1024, assets::TensorStorageType::Native);
    disk_experts_ = backbone_config.moe.residency == moe::Residency::Staged;
    reuse_prefill_ = reuse_prefill_ && !streaming && !disk_experts_;
    reuse_reference_prefill_ = reuse_reference_prefill_ && !streaming && !disk_experts_;
    decoder_ = backbone_->create_decode_session(cache_steps_);
    const auto tower_prefill_cache_mb = runtime::parse_i64_option(options.options, {"fish_drama3.tower_prefill_cache_mb"}).value_or(256);
    if (tower_prefill_cache_mb < 0 || tower_prefill_cache_mb > 4096) throw std::runtime_error("Drama 3 tower_prefill_cache_mb must be between 0 and 4096");
    tower_ = std::make_unique<Drama3Tower>(assets_->model_weights, assets_->tower, execution, cache_steps_,
        static_cast<size_t>(tower_prefill_cache_mb) * 1024u * 1024u);
    auto codec_backend = options.backend;
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.codec_backend"})) {
        if (*value == "cpu") codec_backend.type = core::BackendType::Cpu;
        else if (*value != "same") throw std::runtime_error("fish_drama3.codec_backend requires cpu or same");
    }
    codec_execution_ = std::make_unique<core::ExecutionContext>(codec_backend);
    codecs::FishDacCodecRuntimeOptions codec_options;
    codec_options.weight_context_bytes = 16 * 1024 * 1024;
    codec_options.decoder_prepack_weights = streaming && codec_batch_frames_ > 1;
    codec_options.encoder_bf16_compute = true;
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.codec_encoder_bf16_compute"})) {
        codec_options.encoder_bf16_compute = runtime::parse_bool_option(*value, "fish_drama3.codec_encoder_bf16_compute");
    }
    const auto component = codecs::FishDacCodecComponent::load_from_tensor_source(assets_->codec_weights,
        assets_->codec, {}, codec_execution_->backend(), codec_backend.type, codec_options);
    bool codec_encoder_f32_accumulation = false;
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.codec_encoder_f32_accumulation"})) {
        codec_encoder_f32_accumulation = runtime::parse_bool_option(*value, "fish_drama3.codec_encoder_f32_accumulation");
    }
    const int64_t codec_encoder_chunk_samples = runtime::parse_i64_option(
        options.options, {"fish_drama3.codec_encoder_chunk_samples"}).value_or(0);
    codec_ = std::make_unique<codecs::FishDacCodecRuntime>(component, codec_backend,
        codec_backend.threads, codec_options.graph_arena_bytes, codec_encoder_f32_accumulation, codec_encoder_chunk_samples);
    const auto reference_dir = runtime::find_option(options.options, {"fish_drama3.reference_cache_dir"}).value_or("");
    const auto prefill_dir = runtime::find_option(options.options, {"fish_drama3.prefill_cache_dir"}).value_or("");
    const bool persist_prefill = reuse_reference_prefill_ && !prefill_dir.empty();
    if (!reference_dir.empty() || persist_prefill) {
        try {
            const auto build_id = drama3_cache_build_id();
            if (!build_id.empty()) {
                Sha256 identity;
                cache_key_field(identity, build_id);
                cache_file_identity(identity, assets_->codec_weights->source_path());
                cache_key_field(identity, assets_->resources.read_text("config"));
                if (!reference_dir.empty()) {
                    auto encoder = identity;
                    cache_backend_identity(encoder, *codec_execution_);
                    cache_key_field(encoder, codec_options.encoder_bf16_compute ? "bf16" : "native");
                    cache_key_field(encoder, codec_encoder_f32_accumulation ? "f32" : "native");
                    cache_key_field(encoder, std::to_string(codec_encoder_chunk_samples));
                    reference_cache_.set_disk_cache(std::filesystem::u8path(reference_dir), encoder.hex(),
                        assets_->tower.codebook_size);
                }
                if (persist_prefill) {
                    if (assets_->model_weights->source_path() != assets_->codec_weights->source_path()) {
                        cache_file_identity(identity, assets_->model_weights->source_path());
                    }
                    cache_backend_identity(identity, execution);
                    const std::map<std::string, std::string> sorted_options(options.options.begin(), options.options.end());
                    for (const auto & [key, value] : sorted_options) {
                        if (key == "fish_drama3.reference_cache_dir" || key == "fish_drama3.prefill_cache_dir" ||
                            key == "fish_drama3.reuse_prefill") continue;
                        cache_key_field(identity, key);
                        cache_key_field(identity, value);
                    }
                    prefill_identity_ = identity.hex();
                    prefill_disk_cache_.emplace(io::DiskCacheConfig{
                        std::filesystem::u8path(prefill_dir) / "fish_drama3-prefill", kDiskPrefillEntries, kDiskPrefillBytes});
                }
            } else {
                debug::log_message(debug::LogLevel::Warning, "fish_drama3",
                    "Disk cache unavailable: could not identify the build");
            }
        } catch (const std::exception & error) {
            debug::log_message(debug::LogLevel::Warning, "fish_drama3",
                "Disk cache setup failed: " + std::string(error.what()));
        }
    }
    assets_->model_weights->release_storage();
    assets_->codec_weights->release_storage();
}

Drama3Generator::~Drama3Generator() = default;

std::vector<float> Drama3Generator::embed_prompt(const std::vector<Drama3Frame> & prompt) {
    std::vector<int32_t> ids;
    ids.reserve(prompt.size());
    for (const auto & frame : prompt) ids.push_back(frame[0]);
    auto embeddings = backbone_->token_embedding(ids);
    tower_->mix_backbone_embeddings(embeddings, prompt);
    return embeddings;
}

Drama3Generator::PromptPrefill Drama3Generator::capture_prefill(const std::vector<Drama3Frame> & prompt,
    const std::vector<float> & hidden) const {
    return {prompt, decoder_->capture_prefill(), tower_->capture_prefill(), hidden};
}

void Drama3Generator::restore_prefill(const PromptPrefill & snapshot) {
    decoder_->restore_prefill(snapshot.backbone, static_cast<int64_t>(snapshot.prompt.size()));
    tower_->restore_prefill(snapshot.tower);
}

std::vector<float> Drama3Generator::prefill_reference_prompt(const std::vector<Drama3Frame> & prompt,
    size_t reference_end, double & embedding_ms, double & backbone_ms, double & tower_ms) {
    const bool memory_hit = reference_prefill_ && reference_prefill_->prompt.size() == reference_end &&
        std::equal(reference_prefill_->prompt.begin(), reference_prefill_->prompt.end(), prompt.begin());
    std::string disk_key;
    if (!memory_hit) {
        reference_prefill_.reset();
        if (prefill_disk_cache_) {
            const auto started = std::chrono::steady_clock::now();
            Sha256 key;
            cache_key_field(key, prefill_identity_);
            cache_key_field(key, cache_bytes(prompt.data(), reference_end));
            disk_key = "fish_drama3/prefill/v1/" + key.hex();
            try {
                if (const auto payload = prefill_disk_cache_->get(disk_key)) {
                    const auto unpack_started = std::chrono::steady_clock::now();
                    auto state = unpack_prefill(*payload, decoder_->prefill_snapshot_bytes(),
                        tower_->prefill_snapshot_bytes(), static_cast<size_t>(assets_->backbone.hidden_size));
                    if (state) {
                        reference_prefill_ = PromptPrefill{{prompt.begin(), prompt.begin() + reference_end},
                            std::move(state->backbone), std::move(state->tower), std::move(state->hidden)};
                        debug::timing_log_scalar("drama3.reference_prefill_disk_unpack_ms", debug::elapsed_ms(unpack_started));
                    }
                }
            } catch (const std::exception & error) {
                if (!std::exchange(prefill_disk_read_warned_, true))
                    debug::log_message(debug::LogLevel::Warning, "fish_drama3.prefill_cache",
                        "Cache read failed, recomputing the reference prefix: " + std::string(error.what()));
            }
            debug::trace_log_scalar("drama3.reference_prefill_disk", std::string_view(reference_prefill_ ? "hit" : "miss"));
            debug::timing_log_scalar("drama3.reference_prefill_disk_read_ms", debug::elapsed_ms(started));
        }
    }
    const auto run_backbone = [&](const std::vector<Drama3Frame> & rows, size_t offset) {
        auto started = std::chrono::steady_clock::now();
        auto embeddings = embed_prompt(rows);
        embedding_ms += debug::elapsed_ms(started);
        started = std::chrono::steady_clock::now();
        auto result = decoder_->prefill_embeddings(embeddings, static_cast<int64_t>(rows.size()), static_cast<int64_t>(offset));
        backbone_ms += debug::elapsed_ms(started);
        return std::move(result.hidden);
    };
    const bool hit = reference_prefill_.has_value();
    auto started = std::chrono::steady_clock::now();
    if (hit) {
        decoder_->restore_prefill(reference_prefill_->backbone, static_cast<int64_t>(reference_end));
        backbone_ms += debug::elapsed_ms(started);
        started = std::chrono::steady_clock::now();
        tower_->restore_prefill(reference_prefill_->tower);
        tower_ms += debug::elapsed_ms(started);
    } else {
        decoder_->reset();
        tower_->reset();
        const std::vector<Drama3Frame> prefix(prompt.begin(), prompt.begin() + reference_end);
        const auto hidden = run_backbone(prefix, 0);
        started = std::chrono::steady_clock::now();
        tower_->prefill(hidden, prefix);
        tower_ms += debug::elapsed_ms(started);
        reference_prefill_ = capture_prefill(prefix, last_hidden(hidden, assets_->backbone.hidden_size));
        if (!disk_key.empty()) {
            const auto & snapshot = *reference_prefill_;
            started = std::chrono::steady_clock::now();
            try {
                if (snapshot.backbone.size() + snapshot.tower.size() + snapshot.hidden.size() * sizeof(float) >
                    kDiskPrefillBytes) throw std::invalid_argument("disk cache entry exceeds byte limit");
                const auto payload = pack_prefill(snapshot.backbone, snapshot.tower, snapshot.hidden);
                debug::timing_log_scalar("drama3.reference_prefill_disk_pack_ms", debug::elapsed_ms(started));
                prefill_disk_cache_->put(disk_key, payload);
            } catch (const std::exception & error) {
                if (!std::exchange(prefill_disk_write_warned_, true))
                    debug::log_message(debug::LogLevel::Warning, "fish_drama3.prefill_cache",
                        "Cache write failed, using the computed reference prefix: " + std::string(error.what()));
            }
            debug::timing_log_scalar("drama3.reference_prefill_disk_write_ms", debug::elapsed_ms(started));
        }
    }
    debug::trace_log_scalar("drama3.prefill_reused_rows", hit ? reference_end : size_t{0});
    const auto hidden = run_backbone({prompt.begin() + reference_end, prompt.end()}, reference_end);
    started = std::chrono::steady_clock::now();
    // Tower row N is conditioned on row N+1. Resume one row before the backbone.
    auto tower_hidden = reference_prefill_->hidden;
    tower_hidden.insert(tower_hidden.end(), hidden.begin(), hidden.end());
    tower_->prefill(tower_hidden, {prompt.begin() + reference_end - 1, prompt.end()}, static_cast<int64_t>(reference_end - 1));
    tower_ms += debug::elapsed_ms(started);
    return last_hidden(hidden, assets_->backbone.hidden_size);
}

codecs::FishDacCodes Drama3Generator::encode_reference(const runtime::AudioBuffer & audio) {
    // Keep several short references in host memory. Exact waveform equality avoids
    // stale path-based hits when an uploaded file is replaced or re-decoded.
    Drama3ReferenceCache::Outcome outcome = Drama3ReferenceCache::Outcome::Miss;
    auto result = reference_cache_.lookup_or_encode(audio, [this](const runtime::AudioBuffer & clip) {
        auto codes = codec_->encode_codes(clip);
        codec_->release_encode_graph();
        validate_drama3_codes(codes, assets_->tower.codebook_size);
        return codes;
    }, &outcome);
    const char * label = outcome == Drama3ReferenceCache::Outcome::Hit ? "hit"
                       : outcome == Drama3ReferenceCache::Outcome::DiskHit ? "disk_hit"
                       : outcome == Drama3ReferenceCache::Outcome::Skip ? "skip" : "miss";
    debug::trace_log_scalar("drama3.reference_cache", std::string(label) + " " + std::to_string(reference_cache_.size()) +
                            "/" + std::to_string(reference_cache_.capacity()));
    return result;
}

void Drama3Generator::warm_streaming_codec(const codecs::FishDacCodes & reference) {
    validate_drama3_codes(reference, assets_->tower.codebook_size);
    codecs::FishDacCodes frame;
    frame.codebooks = reference.codebooks;
    frame.frames = 1;
    for (int64_t book = 0; book < reference.codebooks; ++book) {
        frame.codes.push_back(reference.codes[static_cast<size_t>(book * reference.frames)]);
    }
    // Initialize and capture the first-frame codec graph during preparation.
    // Discard decoded reference audio and reset all causal stream state.
    for (int pass = 0; pass < 4; ++pass) {
        codec_->reset_decode_stream();
        (void)codec_->decode_stream(frame);
    }
    codec_->reset_decode_stream();
}

Drama3Generation Drama3Generator::generate(const std::string & text, const std::vector<Drama3Take> & references,
    const std::vector<Drama3Take> & history, const Drama3Sampling & sampling, bool streaming,
    const std::function<void(const runtime::AudioBuffer &)> & on_audio, bool continuation,
    const std::vector<Drama3Take> & generated_history, bool allow_empty) {
    if (!std::isfinite(sampling.temperature) || sampling.temperature < 0 ||
        !std::isfinite(sampling.top_p) || sampling.top_p <= 0 || sampling.top_p > 1 ||
        sampling.top_k <= 0 || sampling.max_frames <= 0) throw runtime::InvalidRequestError("invalid Drama 3 sampling parameters");
    using ProfileClock = std::chrono::steady_clock;
    std::array<double, 9> component_ms{};
    auto component_started = ProfileClock::now();
    const auto profile_start = profile_first_audio_ ? ProfileClock::now() : ProfileClock::time_point{};
    auto profile_marked = profile_start;
    std::vector<std::pair<const char *, double>> first_audio_phases;
    const auto mark_phase = [&](const char * name) {
        if (!profile_first_audio_) return;
        const auto now = ProfileClock::now();
        first_audio_phases.emplace_back(name, std::chrono::duration<double, std::milli>(now - profile_marked).count());
        profile_marked = now;
    };
    size_t reference_end = 0;
    const auto prompt = build_drama3_prompt(*assets_, tokenizer_, text, references, history,
        generated_history, static_cast<size_t>(cache_steps_ - 1), reuse_reference_prefill_ ? &reference_end : nullptr);
    const auto max_frames = std::min(sampling.max_frames, cache_steps_ - static_cast<int64_t>(prompt.size()));
    debug::trace_log_scalar("drama3.prompt_rows", prompt.size());
    debug::trace_log_scalar("drama3.generation_budget", max_frames);
    if (max_frames < sampling.max_frames) {
        std::cerr << "[fish_drama3] max_tokens reduced from " << sampling.max_frames << " to " << max_frames
                  << " to fit fish_drama3.context_length after " << prompt.size() << " prompt rows\n";
    }
    auto & decoder = decoder_;
    // A continuation may extend the exact frames already evaluated. Never
    // carry recurrent state across independent requests or changed histories.
    const size_t prefix_rows = disk_experts_ && continuation && !cached_prefix_.empty() &&
        cached_prefix_.size() <= prompt.size() && std::equal(cached_prefix_.begin(), cached_prefix_.end(), prompt.begin())
        ? cached_prefix_.size() : 0;
    auto prefix_hidden = std::move(cached_hidden_);
    cached_prefix_.clear();
    cached_hidden_.clear();
    const bool cache_prompt = reuse_prefill_ && !streaming;
    const bool prompt_hit = cache_prompt && prompt_prefill_ && prompt_prefill_->prompt == prompt;
    Drama3BackboneDecoderForwardResult prefill;
    std::vector<float> hidden;
    if (prompt_hit) {
        restore_prefill(*prompt_prefill_);
        hidden = prompt_prefill_->hidden;
        debug::trace_log_scalar("drama3.prefill_reused_rows", prompt.size());
        mark_phase("restore_prefill_ms");
        component_ms[0] += debug::elapsed_ms(component_started);
    } else if (reuse_reference_prefill_ && !streaming && !references.empty()) {
        component_ms[0] += debug::elapsed_ms(component_started);
        hidden = prefill_reference_prompt(prompt, reference_end, component_ms[1], component_ms[2], component_ms[3]);
        mark_phase("reference_prefill_ms");
    } else {
        // Start independent requests from the same expert placement, but preserve
        // LRU reuse between continuation chunks within a request.
        if (prefix_rows == 0) decoder->reset(disk_experts_ && !continuation);
        tower_->reset();
        mark_phase("setup_ms");
        component_ms[0] += debug::elapsed_ms(component_started);
        component_started = ProfileClock::now();
        auto embeddings = embed_prompt(prompt);
        mark_phase("embedding_ms");
        component_ms[1] += debug::elapsed_ms(component_started);
        component_started = ProfileClock::now();
        if (prefix_rows != 0) {
            prefill.hidden = std::move(prefix_hidden);
            prefill.steps = static_cast<int64_t>(prompt.size());
            for (size_t row = prefix_rows; row < prompt.size(); ++row) {
                const auto begin = embeddings.begin() + row * assets_->backbone.hidden_size;
                const auto row_hidden = decoder->run_embedding_step({begin, begin + assets_->backbone.hidden_size});
                prefill.hidden.insert(prefill.hidden.end(), row_hidden.begin(), row_hidden.end());
            }
        } else {
            prefill = decoder->prefill_embeddings(embeddings, static_cast<int64_t>(prompt.size()));
        }
        debug::trace_log_scalar("drama3.prefill_reused_rows", prefix_rows);
        mark_phase("backbone_prefill_ms");
        component_ms[2] += debug::elapsed_ms(component_started);
        component_started = ProfileClock::now();
        tower_->prefill(prefill.hidden, prompt);
        mark_phase("tower_prefill_ms");
        component_ms[3] += debug::elapsed_ms(component_started);
        hidden = last_hidden(prefill.hidden, assets_->backbone.hidden_size);
    }
    if (cache_prompt && !prompt_hit) {
        component_started = ProfileClock::now();
        prompt_prefill_.reset();
        prompt_prefill_ = capture_prefill(prompt, hidden);
        component_ms[0] += debug::elapsed_ms(component_started);
    }
    Drama3Frame current = prompt.back();
    std::mt19937 random(sampling.seed);
    SamplerScratch sampler_scratch;
    std::deque<int32_t> recent;
    std::vector<Drama3Frame> output_frames;
    Drama3Generation result;
    std::mutex codec_mutex;
    std::condition_variable codec_ready;
    std::deque<Drama3Frame> codec_queue;
    bool codec_done = false;
    bool codec_cancelled = false;
    std::exception_ptr codec_error;
    std::thread codec_worker;
    if (streaming) {
        runtime::StreamingAudioConfig config;
        config.frames_per_chunk = codec_batch_frames_;
        config.frame_width = 10;
        streaming_audio_.begin(config, [&](const auto & window) {
            codecs::FishDacCodes codes;
            codes.codebooks = 10;
            codes.frames = static_cast<int64_t>(window.new_frames);
            codes.codes.resize(window.new_frames * 10);
            for (size_t book = 0; book < 10; ++book) {
                for (size_t frame = 0; frame < window.new_frames; ++frame) {
                    codes.codes[book * window.new_frames + frame] = window.data[frame * 10 + book];
                }
            }
            auto audio = codec_->decode_stream(codes);
            if (window.first_frame == 0) {
                mark_phase("codec_first_frame_ms");
                if (profile_first_audio_) {
                    std::cerr << "[drama3.first_audio] prompt_rows=" << prompt.size();
                    for (const auto & phase : first_audio_phases) std::cerr << " " << phase.first << "=" << phase.second;
                    std::cerr << " total_ms=" << std::chrono::duration<double, std::milli>(ProfileClock::now() - profile_start).count() << '\n';
                }
            }
            return audio;
        }, [&](const runtime::StreamEvent & event) {
            if (on_audio && event.audio_output) on_audio(*event.audio_output);
        }, {}, [&] { codec_->reset_decode_stream(); });
    }
    auto stop_codec = [&](bool cancel) {
        if (!codec_worker.joinable()) return;
        {
            std::lock_guard<std::mutex> lock(codec_mutex);
            codec_done = true;
            codec_cancelled = cancel;
            if (cancel) codec_queue.clear();
        }
        codec_ready.notify_all();
        codec_worker.join();
    };
    struct CodecJoin {
        decltype(stop_codec) & stop;
        runtime::StreamingAudioController<int32_t> & stream;
        ~CodecJoin() { stop(true); stream.reset(); }
    } codec_join{stop_codec, streaming_audio_};
    if (streaming && codec_async_decode_) {
        codec_worker = std::thread([&] {
            try {
                for (;;) {
                    Drama3Frame frame;
                    {
                        std::unique_lock<std::mutex> lock(codec_mutex);
                        codec_ready.wait(lock, [&] { return codec_done || !codec_queue.empty(); });
                        if (codec_queue.empty()) break;
                        frame = codec_queue.front();
                        codec_queue.pop_front();
                    }
                    codec_ready.notify_all();
                    streaming_audio_.push({frame.begin() + 1, frame.end()});
                }
                if (!codec_cancelled) result.audio = streaming_audio_.finish();
            } catch (...) {
                std::lock_guard<std::mutex> lock(codec_mutex);
                codec_error = std::current_exception();
                codec_queue.clear();
                codec_ready.notify_all();
            }
        });
    }
    for (int64_t step = 0; step < max_frames; ++step) {
        component_started = ProfileClock::now();
        auto logits = backbone_->lm_head(hidden);
        component_ms[4] += debug::elapsed_ms(component_started);
        component_started = ProfileClock::now();
        // Both drama3 (lm_head_token_ids) and drama3-small (lm_head_rows) return
        // compact [EOS, semantic 0..N-1] logits from the backbone.
        auto candidates = std::move(logits);
        const int32_t first = sample(candidates, sampling.temperature, sampling.top_k, sampling.top_p, random, sampler_scratch);
        const int32_t second = sample(candidates, 1.0F, 30, 0.95F, random, sampler_scratch);
        const int32_t first_token = first == 0 ? assets_->eos : assets_->tower.semantic_begin + first - 1;
        const bool repeats = first != 0 && std::find(recent.begin(), recent.end(), first_token) != recent.end();
        const int32_t selected = repeats ? second : first;
        component_ms[5] += debug::elapsed_ms(component_started);
        if (selected == 0) break;
        if (step == 0) mark_phase("semantic_ms");
        Drama3Frame next{};
        next[0] = assets_->tower.semantic_begin + selected - 1;
        next[1] = selected - 1;
        const auto tower_started = ProfileClock::now();
        const auto sampling_before_tower = component_ms[5];
        const int64_t position = decoder->valid_steps() - 1;
        const auto & config = assets_->tower;
        if (device_sampling_) {
            std::array<float, 9> uniforms{};
            if (sampling.temperature != 0 && sampling.top_k != 1) {
                for (auto & uniform : uniforms) uniform = static_cast<float>(std::generate_canonical<double, 53>(random));
            }
            next = tower_->run_frame(hidden, current, selected - 1, position,
                sampling.temperature, sampling.top_k, sampling.top_p, uniforms);
        } else {
            const auto shared_input = tower_->shared_input({current});
            auto tower_hidden = hidden;
            for (int64_t layer = 0; layer < config.shared_layers + config.residual_layers; ++layer) {
                const int64_t book = layer < config.shared_layers ? 0 : layer - config.shared_layers;
                const auto tower_output = tower_->run_layer(layer, tower_hidden, {next[1 + book]}, {1.0F},
                                                            layer == 0 ? shared_input : std::vector<float>{}, position);
                tower_hidden = tower_output.hidden;
                if (layer >= config.shared_layers) {
                    component_started = ProfileClock::now();
                    next[2 + book] = sample(tower_output.logits, sampling.temperature, sampling.top_k, sampling.top_p, random, sampler_scratch);
                    component_ms[5] += debug::elapsed_ms(component_started);
                }
            }
        }
        component_ms[6] += debug::elapsed_ms(tower_started) - (component_ms[5] - sampling_before_tower);
        output_frames.push_back(next);
        if (step == 0) mark_phase("tower_first_frame_ms");
        recent.push_back(next[0]);
        if (recent.size() > 16) recent.pop_front();
        if (streaming) {
            if (codec_async_decode_) {
                std::unique_lock<std::mutex> lock(codec_mutex);
                codec_ready.wait(lock, [&] { return codec_error || codec_queue.size() < 2 * codec_batch_frames_; });
                if (codec_error) std::rethrow_exception(codec_error);
                codec_queue.push_back(next);
                lock.unlock();
                codec_ready.notify_one();
            } else {
                streaming_audio_.push({next.begin() + 1, next.end()});
            }
        }
        if (step + 1 == max_frames) { result.reached_limit = true; break; }
        current = next;
        component_started = ProfileClock::now();
        auto embedding = backbone_->token_embedding({current[0]});
        tower_->mix_backbone_embeddings(embedding, {current});
        component_ms[1] += debug::elapsed_ms(component_started);
        component_started = ProfileClock::now();
        hidden = decoder->run_embedding_step(embedding);
        if (disk_experts_) prefill.hidden.insert(prefill.hidden.end(), hidden.begin(), hidden.end());
        component_ms[7] += debug::elapsed_ms(component_started);
    }
    if (streaming) {
        stop_codec(false);
        if (codec_error) std::rethrow_exception(codec_error);
        if (!codec_async_decode_) result.audio = streaming_audio_.finish();
        component_ms[8] += streaming_audio_.decode_ms() + streaming_audio_.publish_ms();
    }
    if (output_frames.empty()) {
        if (allow_empty) {
            std::cerr << "[fish_drama3] The model emitted EOS before any audio frame; skipping chunk\n";
            return result;
        }
        throw std::runtime_error("Drama 3 ended before producing an audio frame");
    }
    if (disk_experts_) {
        cached_prefix_ = prompt;
        cached_prefix_.insert(cached_prefix_.end(), output_frames.begin(), output_frames.end());
        // At max_tokens the final sampled frame has not entered the backbone.
        cached_prefix_.resize(static_cast<size_t>(decoder->valid_steps()));
        cached_hidden_ = std::move(prefill.hidden);
    }
    result.codes.codebooks = 10;
    result.codes.frames = static_cast<int64_t>(output_frames.size());
    result.codes.codes.resize(output_frames.size() * 10);
    for (int64_t book = 0; book < 10; ++book) {
        for (size_t row = 0; row < output_frames.size(); ++row) result.codes.codes[book * output_frames.size() + row] = output_frames[row][book + 1];
    }
    if (!streaming) {
        component_started = ProfileClock::now();
        if (disk_experts_) {
            // Preserve KV/recurrent state for continuation, but do not overlap
            // completed prompt workspaces with the waveform decoder workspace.
            backbone_->release_prefill_graphs();
            tower_->release_prefill_graphs();
        }
        result.audio = codec_->decode_codes(result.codes);
        codec_->release_runtime_graphs();
        component_ms[8] += debug::elapsed_ms(component_started);
    }
    const char * component_keys[] = {"drama3.setup_ms", "drama3.embedding_ms", "drama3.backbone_prefill_ms",
        "drama3.tower_prefill_ms", "drama3.lm_head_ms", "drama3.sampling_ms", "drama3.tower_ms",
        "drama3.backbone_step_ms", "drama3.codec_ms"};
    for (size_t i = 0; i < component_ms.size(); ++i) debug::timing_log_scalar(component_keys[i], component_ms[i]);
    debug::timing_log_scalar("drama3.generated_frames", output_frames.size());
    if (result.reached_limit) std::cerr << "[fish_drama3] generation budget reached; audio may be incomplete\n";
    return result;
}

}  // namespace engine::models::fish_drama3
