#include "engine/community_models/voice_isolator/config.h"

#include "engine/framework/io/json.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>

namespace engine::models::voice_isolator {
namespace {

namespace json = engine::io::json;

// Training-side fields that do not change inference.
const std::set<std::string> kTrainingOnly = {
    "activation_checkpointing", "checkpoint_every", "attention_backend", "offline_mode_prob",
    "dual_distill", "hint_sha256", "hint_bf16", "grid_width", "grid_hidden", "grid_kernel",
    "grid_stride", "grid_heads", "grid_qk_channels", "grid_output", "grid_normalize_input",
    "decision_head"};

// Fields this port implements, beyond the required values below.
const std::set<std::string> kKnown = {
    "width", "block_types", "num_heads", "n_fft", "hop_length", "sample_rate", "band_plan",
    "correction_bound", "chunk_seconds", "overlap_seconds", "exclusive_self_attention",
    "exclusive_axes", "mamba_state", "mamba_conv", "mamba_expand", "causal", "lookahead_frames",
    "lookahead_frames_choices", "dual_mode", "mode_film", "attention_window_frames",
    "offline_passes", "representation", "decoder_type", "hint_checkpoint"};

[[noreturn]] void refuse(const std::string & message) {
    throw std::runtime_error("Voice Isolator checkpoint config: " + message);
}

bool is_falsy(const json::Value & value) {
    switch (value.kind()) {
    case json::Value::Kind::Null:
        return true;
    case json::Value::Kind::Bool:
        return !value.as_bool();
    case json::Value::Kind::Number:
        return value.as_number() == 0.0;
    case json::Value::Kind::String:
        return value.as_string().empty();
    case json::Value::Kind::Array:
        return value.as_array().empty();
    case json::Value::Kind::Object:
        return value.as_object().empty();
    }
    return false;
}

// A null value counts as absent, as with every other key.
void read_f64(const json::Value & root, const char * key, double & out) {
    if (const auto * value = root.find(key); value != nullptr && !value->is_null()) {
        out = value->as_number();
    }
}

void require_string(const json::Value & root, const char * key, const char * expected) {
    const auto * value = root.find(key);
    if (value != nullptr && !value->is_null() && value->as_string() != expected) {
        refuse(std::string(key) + "=" + value->as_string() + " is not implemented (needs " + expected + ")");
    }
}

void validate(const Config & config) {
    if (config.width < 1 || config.num_heads < 1 || config.width % (2 * config.num_heads) != 0) {
        refuse("width must be a positive multiple of twice num_heads");
    }
    if (config.blocks.empty()) {
        refuse("block_types is empty");
    }
    if (config.n_fft < 4 || config.n_fft % 2 != 0 || config.hop_length <= 0 || config.hop_length > config.n_fft / 2) {
        refuse("n_fft must be even and hop_length within (0, n_fft / 2]");
    }
    if (config.sample_rate < 1 || config.band_plan.empty()) {
        refuse("band_plan needs positive widths and limits");
    }
    for (const auto & [width, upto] : config.band_plan) {
        if (width < 1 || upto < 1) {
            refuse("band_plan needs positive widths and limits");
        }
    }
    if (!(config.correction_bound > 0.0F && config.correction_bound < 1.0F)) {
        refuse("correction_bound must lie in (0, 1)");
    }
    if (config.chunk_seconds * static_cast<double>(config.sample_rate) < static_cast<double>(config.n_fft) ||
        !(config.overlap_seconds > 0.0 && config.overlap_seconds <= config.chunk_seconds / 2.0)) {
        refuse("chunk_seconds must cover one window and overlap_seconds lie in (0, chunk_seconds / 2]");
    }
    if (std::min({config.mamba_state, config.mamba_conv, config.mamba_expand}) < 1) {
        refuse("Mamba dimensions must be positive");
    }
    const auto & choices = config.lookahead_choices;
    if (!std::is_sorted(choices.begin(), choices.end()) ||
        std::adjacent_find(choices.begin(), choices.end()) != choices.end() ||
        (!choices.empty() && choices.front() < 0) || config.lookahead_frames < 0) {
        refuse("look-ahead choices must be distinct non-negative frame counts in ascending order");
    }
    if (!choices.empty() && std::find(choices.begin(), choices.end(), config.lookahead_frames) == choices.end()) {
        refuse("lookahead_frames must be one of lookahead_frames_choices");
    }
    if ((config.lookahead_frames > 0 || !choices.empty()) && !config.causal) {
        refuse("a look-ahead needs a causal model");
    }
    if (config.attention_window_frames < 0 || (config.attention_window_frames > 0 && !config.causal)) {
        refuse("attention_window_frames needs a causal model");
    }
}

}  // namespace

std::vector<int64_t> Config::streaming_lookaheads() const {
    if (!lookahead_choices.empty()) {
        return lookahead_choices;
    }
    return causal ? std::vector<int64_t>{lookahead_frames} : std::vector<int64_t>{};
}

Config parse_config(std::string_view text) {
    const auto root = json::parse(text);
    if (!root.is_object()) {
        refuse("not a JSON object");
    }
    require_string(root, "representation", "bands");
    require_string(root, "decoder_type", "complex_mask_residual");
    if (const auto * hint = root.find("hint_checkpoint"); hint != nullptr && !hint->is_null()) {
        refuse("hint_checkpoint is not implemented");
    }
    for (const auto & [key, value] : root.as_object()) {
        if (kKnown.count(key) == 0 && kTrainingOnly.count(key) == 0 && !is_falsy(value)) {
            refuse("uses the unimplemented field " + key);
        }
    }
    if (const auto * dual = root.find("dual_mode"); dual != nullptr && dual->as_bool()) {
        refuse("dual-mode checkpoints are not implemented");
    }
    if (const auto * film = root.find("mode_film"); film != nullptr && film->as_bool()) {
        refuse("mode_film is not implemented");
    }
    if (const auto * passes = root.find("offline_passes"); passes != nullptr && passes->as_i64() != 1) {
        refuse("offline_passes other than 1 is not implemented");
    }

    Config config;
    config.width = json::optional_nullable_i64(root, "width", config.width);
    config.num_heads = json::optional_nullable_i64(root, "num_heads", config.num_heads);
    config.n_fft = json::optional_nullable_i64(root, "n_fft", config.n_fft);
    config.hop_length = json::optional_nullable_i64(root, "hop_length", config.hop_length);
    config.sample_rate = json::optional_nullable_i64(root, "sample_rate", config.sample_rate);
    if (const auto * plan = root.find("band_plan")) {
        config.band_plan.clear();
        for (const auto & pair : plan->as_array()) {
            const auto & values = pair.as_array();
            if (values.size() != 2) {
                refuse("band_plan entries must be [width_hz, upto_hz]");
            }
            config.band_plan.emplace_back(values[0].as_i64(), values[1].as_i64());
        }
    }
    if (const auto * kinds = root.find("block_types")) {
        config.blocks.clear();
        for (const auto & kind : kinds->as_array()) {
            if (kind.as_string() == "transformer") {
                config.blocks.push_back(BlockKind::Attention);
            } else if (kind.as_string() == "mamba") {
                config.blocks.push_back(BlockKind::Mamba);
            } else {
                refuse("unknown block type " + kind.as_string());
            }
        }
    }
    config.correction_bound = json::optional_nullable_f32(root, "correction_bound", config.correction_bound);
    read_f64(root, "chunk_seconds", config.chunk_seconds);
    read_f64(root, "overlap_seconds", config.overlap_seconds);
    const bool exclusive = json::optional_nullable_bool(root, "exclusive_self_attention", true);
    const std::string axes = json::optional_nullable_string(root, "exclusive_axes", "time");
    if (axes != "both" && axes != "time" && axes != "band") {
        refuse("exclusive_axes must be both, time or band");
    }
    config.exclusive_time = exclusive && (axes == "both" || axes == "time");
    config.exclusive_band = exclusive && (axes == "both" || axes == "band");
    config.mamba_state = json::optional_nullable_i64(root, "mamba_state", config.mamba_state);
    config.mamba_conv = json::optional_nullable_i64(root, "mamba_conv", config.mamba_conv);
    config.mamba_expand = json::optional_nullable_i64(root, "mamba_expand", config.mamba_expand);
    config.causal = json::optional_nullable_bool(root, "causal", config.causal);
    config.lookahead_frames = json::optional_nullable_i64(root, "lookahead_frames", config.lookahead_frames);
    if (const auto * choices = root.find("lookahead_frames_choices")) {
        for (const auto & choice : choices->as_array()) {
            config.lookahead_choices.push_back(choice.as_i64());
        }
    }
    config.attention_window_frames = json::optional_nullable_i64(root, "attention_window_frames", config.attention_window_frames);
    validate(config);
    return config;
}

BandLayout make_band_layout(const Config & config) {
    BandLayout layout;
    layout.bins = config.bins();
    const double hz_per_bin = static_cast<double>(config.sample_rate) / static_cast<double>(config.n_fft);
    // Python's round(): half to even, as std::nearbyint in the default rounding mode.
    const auto round_bins = [&](int64_t hz) {
        return static_cast<int64_t>(std::nearbyint(static_cast<double>(hz) / hz_per_bin));
    };
    int64_t start = 0;
    for (const auto & [width_hz, upto_hz] : config.band_plan) {
        const int64_t width = std::max<int64_t>(1, round_bins(width_hz));
        const int64_t upto = std::min(layout.bins, round_bins(upto_hz));
        while (start < upto) {
            const int64_t end = std::min(start + width, upto);
            layout.edges.emplace_back(start, end);
            start = end;
        }
    }
    if (start < layout.bins) {
        layout.edges.emplace_back(start, layout.bins);
    }
    for (const auto & [first, last] : layout.edges) {
        layout.max_band_bins = std::max(layout.max_band_bins, last - first);
    }
    return layout;
}

}  // namespace engine::models::voice_isolator
