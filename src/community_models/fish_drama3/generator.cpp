#include "engine/community_models/fish_drama3/generator.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/sampling/hf_sampler.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <limits>
#include <sstream>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace engine::models::fish_drama3 {
namespace {

int32_t sample(const std::vector<float> & logits, float temperature, int top_k,
               float top_p, std::mt19937 & random) {
    if (logits.empty()) throw std::runtime_error("Drama 3 sampler has no logits");
    if (temperature == 0 || top_k == 1)
        return sampling::HfLogitsProcessor::argmax(logits.data(), logits.size(), "Drama 3");
    std::vector<int32_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    const auto count = std::min(logits.size(), static_cast<size_t>(top_k));
    std::partial_sort(order.begin(), order.begin() + count, order.end(), [&](int32_t left, int32_t right) {
        return logits[left] == logits[right] ? left < right : logits[left] > logits[right];
    });
    order.resize(count);
    const double maximum = logits[order.front()] / temperature;
    double total = 0;
    // Top-p is defined against the full distribution, before top-k filtering.
    for (const auto logit : logits) total += std::exp(logit / temperature - maximum);
    std::vector<double> probabilities(order.size());
    for (size_t rank = 0; rank < order.size(); ++rank) {
        probabilities[rank] = std::exp(logits[order[rank]] / temperature - maximum);
    }
    if (!(total > 0) || !std::isfinite(total)) throw std::runtime_error("Drama 3 sampler encountered nonfinite logits");
    double cumulative = 0;
    for (size_t rank = 0; rank < order.size(); ++rank) {
        cumulative += probabilities[rank] / total;
        if (rank > 0 && (cumulative > top_p || rank >= static_cast<size_t>(top_k))) probabilities[rank] = 0;
    }
    std::discrete_distribution<size_t> choice(probabilities.begin(), probabilities.end());
    return order[choice(random)];
}

// drama3-small pause clamp (request options max_pause_frames / silence_codes / pause_end_ratio /
// pause_cap_ratio, else AUDIOCPP_DRAMA3_* env). After max_pause_frames consecutive silence-code
// frames, silence codes are masked from the next main-token draw, so the model must resume speech
// or emit EOS (21.5 frames = 1 s). 0 = off.
struct PauseClamp {
    std::unordered_set<int32_t> codes;
    int64_t max_pause = 0;
    // Length guard (with the clamp): a pause after end_ratio x the text's expected frames ends the
    // take instead of forcing speech; cap_ratio x expected ends it regardless (babble backstop).
    double end_ratio = 2.0;
    double cap_ratio = 3.0;
};

// Expected frames for the spoken text, fitted on clean takes (21.5 frames/s):
// 3.88 per CJK/kana character + 2.00 per ASCII letter or digit + 23.9 per control span ([...], <...>,
// each opening bracket) + 8.5. A "Direction: ...\nLine:" prefix is not spoken.
double expected_frames(const std::string & text) {
    std::string spoken = text;
    if (spoken.rfind("Direction:", 0) == 0) {
        const auto line = spoken.find("\nLine:");
        if (line != std::string::npos) spoken = spoken.substr(line + 6);
    }
    double cjk = 0, latin = 0, spans = 0;
    int depth_square = 0, depth_angle = 0;
    for (size_t i = 0; i < spoken.size();) {
        const unsigned char c = static_cast<unsigned char>(spoken[i]);
        uint32_t cp = c;
        size_t len = 1;
        if (c >= 0xF0 && i + 3 < spoken.size()) { cp = ((c & 0x07) << 18) | ((spoken[i + 1] & 0x3F) << 12) | ((spoken[i + 2] & 0x3F) << 6) | (spoken[i + 3] & 0x3F); len = 4; }
        else if (c >= 0xE0 && i + 2 < spoken.size()) { cp = ((c & 0x0F) << 12) | ((spoken[i + 1] & 0x3F) << 6) | (spoken[i + 2] & 0x3F); len = 3; }
        else if (c >= 0xC0 && i + 1 < spoken.size()) { cp = ((c & 0x1F) << 6) | (spoken[i + 1] & 0x3F); len = 2; }
        i += len;
        if (cp == '[') { ++depth_square; ++spans; continue; }
        if (cp == ']') { if (depth_square > 0) --depth_square; continue; }
        if (cp == '<') { ++depth_angle; ++spans; continue; }
        if (cp == '>') { if (depth_angle > 0) --depth_angle; continue; }
        if (depth_square > 0 || depth_angle > 0) continue;
        const bool cjk_punct = cp == 0x3002 || cp == 0x3001 || cp == 0xFF01 || cp == 0xFF1F || cp == 0x300C ||
                               cp == 0x300D || cp == 0x2026 || cp == 0x30FB || cp == 0xFF08 || cp == 0xFF09;
        if (cp >= 0x3000 && !cjk_punct) cjk += 1;
        else if (cp < 0x80 && std::isalnum(static_cast<int>(cp))) latin += 1;
        else if (cp >= 0x80 && !cjk_punct) latin += 1;
    }
    return 3.88 * cjk + 2.00 * latin + 23.9 * spans + 8.5;
}

std::unordered_set<int32_t> parse_silence_codes(const std::string & spec) {
    static std::mutex mutex;
    static std::unordered_map<std::string, std::unordered_set<int32_t>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto it = cache.find(spec); it != cache.end()) return it->second;
    std::string text = spec;
    if (std::ifstream file(spec); file) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        text = buffer.str();
    }
    for (char & c : text) {
        if (!(c >= '0' && c <= '9')) c = ' ';
    }
    std::unordered_set<int32_t> codes;
    std::istringstream values(text);
    for (int32_t code; values >> code;) codes.insert(code);
    if (codes.empty()) throw std::runtime_error("Drama 3 silence_codes is empty");
    return cache.emplace(spec, std::move(codes)).first->second;
}

// Request options override the AUDIOCPP_DRAMA3_* environment defaults.
PauseClamp resolve_pause_clamp(const Drama3Sampling & sampling) {
    PauseClamp clamp;
    const char * env_frames = std::getenv("AUDIOCPP_DRAMA3_MAX_PAUSE_FRAMES");
    clamp.max_pause = sampling.max_pause_frames >= 0 ? sampling.max_pause_frames
                                                     : (env_frames != nullptr ? std::atoll(env_frames) : 0);
    if (clamp.max_pause <= 0) {
        clamp.max_pause = 0;
        return clamp;
    }
    std::string spec = sampling.silence_codes;
    if (spec.empty()) {
        const char * env_codes = std::getenv("AUDIOCPP_DRAMA3_SILENCE_CODES");
        if (env_codes == nullptr) throw std::runtime_error("Drama 3 pause clamp needs silence_codes or AUDIOCPP_DRAMA3_SILENCE_CODES");
        spec = env_codes;
    }
    clamp.codes = parse_silence_codes(spec);
    if (const char * v = std::getenv("AUDIOCPP_DRAMA3_END_RATIO")) clamp.end_ratio = std::atof(v);
    if (const char * v = std::getenv("AUDIOCPP_DRAMA3_CAP_RATIO")) clamp.cap_ratio = std::atof(v);
    if (sampling.pause_end_ratio > 0) clamp.end_ratio = sampling.pause_end_ratio;
    if (sampling.pause_cap_ratio > 0) clamp.cap_ratio = sampling.pause_cap_ratio;
    return clamp;
}

std::vector<float> last_hidden(const std::vector<float> & values, int64_t width) {
    if (values.size() < static_cast<size_t>(width)) throw std::runtime_error("Drama 3 backbone returned too few hidden values");
    return {values.end() - width, values.end()};
}

}  // namespace

Drama3Generator::Drama3Generator(std::shared_ptr<const Drama3Assets> assets,
    core::ExecutionContext & execution, const runtime::SessionOptions & options)
    : assets_(std::move(assets)), tokenizer_(assets_->resources),
      cache_steps_(runtime::parse_positive_i64_option(options.options, {"fish_drama3.context_length"}, 2048)) {
    if (cache_steps_ > 262144) throw std::runtime_error("Drama 3 context_length exceeds the checkpoint's maximum position count");
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.profile_first_audio"})) {
        profile_first_audio_ = runtime::parse_bool_option(*value, "fish_drama3.profile_first_audio");
    }
    auto backbone_config = assets_->backbone;
    backbone_config.decode_cache_steps = cache_steps_;
    backbone_config.cpu_expert_layers = runtime::parse_i64_option(options.options, {"fish_drama3.cpu_expert_layers"}).value_or(4);
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.disk_experts"}))
        backbone_config.disk_experts = runtime::parse_bool_option(*value, "fish_drama3.disk_experts");
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.pin_experts"}))
        backbone_config.pin_experts = runtime::parse_bool_option(*value, "fish_drama3.pin_experts");
    if (backbone_config.pin_experts && !backbone_config.disk_experts)
        throw std::runtime_error("fish_drama3.pin_experts requires fish_drama3.disk_experts=true");
    backbone_config.expert_cache_size = runtime::parse_positive_i64_option(
        options.options, {"fish_drama3.expert_cache_size"}, 32);
    const auto backbone_prefill_cache_mb = runtime::parse_i64_option(options.options, {"fish_drama3.backbone_prefill_cache_mb"}).value_or(256);
    if (backbone_prefill_cache_mb < 0 || backbone_prefill_cache_mb > 4096) throw std::runtime_error("Drama 3 backbone_prefill_cache_mb must be between 0 and 4096");
    backbone_config.prefill_cache_bytes = static_cast<size_t>(backbone_prefill_cache_mb) * 1024u * 1024u;
    backbone_config.profile_prefill_cache = profile_first_audio_;
    backbone_ = std::make_unique<Drama3BackboneDecoderRuntime>(assets_->model_weights, backbone_config, execution,
        64 * 1024 * 1024, 8 * 1024 * 1024, assets::TensorStorageType::Native);
    disk_experts_ = backbone_config.disk_experts;
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
    codec_ = std::make_unique<codecs::FishDacCodecRuntime>(component, codec_backend,
        codec_backend.threads, codec_options.graph_arena_bytes, codec_encoder_f32_accumulation);
    assets_->model_weights->release_storage();
    assets_->codec_weights->release_storage();
}

Drama3Generator::~Drama3Generator() = default;

codecs::FishDacCodes Drama3Generator::encode_reference(const runtime::AudioBuffer & audio) {
    if (cached_reference_audio_ && cached_reference_audio_->sample_rate == audio.sample_rate &&
        cached_reference_audio_->channels == audio.channels && cached_reference_audio_->samples == audio.samples) {
        return cached_reference_codes_;
    }
    auto result = codec_->encode_codes(audio);
    codec_->release_encode_graph();
    validate_drama3_codes(result, assets_->tower.codebook_size);
    // Keep one short reference in host memory. Exact waveform equality avoids
    // stale path-based hits when an uploaded file is replaced or re-decoded.
    constexpr size_t max_cached_samples = 44100 * 30;
    if (audio.samples.size() <= max_cached_samples) {
        cached_reference_audio_ = audio;
        cached_reference_codes_ = result;
    } else {
        cached_reference_audio_.reset();
        cached_reference_codes_ = {};
    }
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
    const std::function<void(const runtime::AudioBuffer &)> & on_audio, bool continuation) {
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
    const auto prompt = build_drama3_prompt(*assets_, tokenizer_, text, references, history);
    if (sampling.max_frames > cache_steps_ ||
        prompt.size() > static_cast<size_t>(cache_steps_ - sampling.max_frames)) {
        throw runtime::CapacityError("Drama 3 prompt plus max_tokens exceeds fish_drama3.context_length; increase it or shorten history");
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
    // Start independent requests from the same expert placement, but preserve
    // LRU reuse between continuation chunks within a request.
    if (prefix_rows == 0) decoder->reset(disk_experts_ && !continuation);
    tower_->reset();
    mark_phase("setup_ms");
    component_ms[0] += debug::elapsed_ms(component_started);
    component_started = ProfileClock::now();
    std::vector<int32_t> ids;
    for (const auto & frame : prompt) ids.push_back(frame[0]);
    auto embeddings = backbone_->token_embedding(ids);
    tower_->mix_backbone_embeddings(embeddings, prompt);
    mark_phase("embedding_ms");
    component_ms[1] += debug::elapsed_ms(component_started);
    component_started = ProfileClock::now();
    Drama3BackboneDecoderForwardResult prefill;
    if (prefix_rows != 0) {
        prefill.hidden = std::move(prefix_hidden);
        prefill.steps = static_cast<int64_t>(prompt.size());
        for (size_t row = prefix_rows; row < prompt.size(); ++row) {
            const auto begin = embeddings.begin() + row * assets_->backbone.hidden_size;
            const auto hidden = decoder->run_embedding_step({begin, begin + assets_->backbone.hidden_size});
            prefill.hidden.insert(prefill.hidden.end(), hidden.begin(), hidden.end());
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
    auto hidden = last_hidden(prefill.hidden, assets_->backbone.hidden_size);
    Drama3Frame current = prompt.back();
    std::mt19937 random(sampling.seed);
    std::deque<int32_t> recent;
    const PauseClamp clamp = resolve_pause_clamp(sampling);
    int64_t silent_run = 0;
    const double expected = expected_frames(text);
    const auto end_frames = static_cast<int64_t>(clamp.end_ratio * expected);
    const auto cap_frames = static_cast<int64_t>(clamp.cap_ratio * expected);
    std::vector<Drama3Frame> output_frames;
    Drama3Generation result;
    if (streaming) codec_->reset_decode_stream();
    for (int64_t step = 0; step < sampling.max_frames; ++step) {
        component_started = ProfileClock::now();
        auto logits = backbone_->lm_head(hidden);
        component_ms[4] += debug::elapsed_ms(component_started);
        component_started = ProfileClock::now();
        // Both drama3 (lm_head_token_ids) and drama3-small (lm_head_rows) return
        // compact [EOS, semantic 0..N-1] logits from the backbone.
        auto candidates = std::move(logits);
        if (clamp.max_pause > 0 && step >= cap_frames) {
            std::cerr << "[fish_drama3] length guard: stopped at " << step << " frames (expected " << expected << ")\n";
            break;
        }
        if (clamp.max_pause > 0 && silent_run >= clamp.max_pause && step >= end_frames) break;  // pause after the text is done
        if (clamp.max_pause > 0 && silent_run >= clamp.max_pause) {
            // Long enough silence: the next frame must be speech or EOS (candidate 0).
            for (const int32_t code : clamp.codes) {
                if (code + 1 < static_cast<int32_t>(candidates.size())) candidates[static_cast<size_t>(code) + 1] = -std::numeric_limits<float>::infinity();
            }
        }
        const int32_t first = sample(candidates, sampling.temperature, sampling.top_k, sampling.top_p, random);
        const int32_t second = sample(candidates, 1.0F, 30, 0.95F, random);
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
        const auto shared_input = tower_->shared_input({current});
        auto tower_hidden = hidden;
        const int64_t position = decoder->valid_steps() - 1;
        const auto & config = assets_->tower;
        for (int64_t layer = 0; layer < config.shared_layers + config.residual_layers; ++layer) {
            const int64_t book = layer < config.shared_layers ? 0 : layer - config.shared_layers;
            const auto tower_output = tower_->run_layer(layer, tower_hidden, {next[1 + book]}, {1.0F},
                                                        layer == 0 ? shared_input : std::vector<float>{}, position);
            tower_hidden = tower_output.hidden;
            if (layer >= config.shared_layers) {
                component_started = ProfileClock::now();
                next[2 + book] = sample(tower_output.logits, sampling.temperature, sampling.top_k, sampling.top_p, random);
                component_ms[5] += debug::elapsed_ms(component_started);
            }
        }
        component_ms[6] += debug::elapsed_ms(tower_started) - (component_ms[5] - sampling_before_tower);
        output_frames.push_back(next);
        if (step == 0) mark_phase("tower_first_frame_ms");
        silent_run = clamp.codes.count(next[1]) ? silent_run + 1 : 0;
        recent.push_back(next[0]);
        if (recent.size() > 16) recent.pop_front();
        if (streaming) {
            component_started = ProfileClock::now();
            codecs::FishDacCodes codes;
            codes.codebooks = 10;
            codes.frames = 1;
            codes.codes.assign(next.begin() + 1, next.end());
            auto chunk = codec_->decode_stream(codes);
            if (step == 0) mark_phase("codec_first_frame_ms");
            runtime::append_audio_buffer(result.audio, chunk);
            if (on_audio) on_audio(chunk);
            component_ms[8] += debug::elapsed_ms(component_started);
            if (step == 0 && profile_first_audio_) {
                std::cerr << "[drama3.first_audio] prompt_rows=" << prompt.size();
                for (const auto & phase : first_audio_phases) std::cerr << " " << phase.first << "=" << phase.second;
                std::cerr << " total_ms=" << std::chrono::duration<double, std::milli>(ProfileClock::now() - profile_start).count() << '\n';
            }
        }
        if (step + 1 == sampling.max_frames) { result.reached_limit = true; break; }
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
    if (output_frames.empty()) throw std::runtime_error("Drama 3 ended before producing an audio frame");
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
    if (result.reached_limit) std::cerr << "[fish_drama3] max_tokens reached; audio may be incomplete\n";
    return result;
}

}  // namespace engine::models::fish_drama3
