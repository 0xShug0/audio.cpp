#include "engine/models/parakeet_tdt/session.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/tokenizers/hf_tokenizer_json.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

namespace engine::models::parakeet_tdt {

namespace {

constexpr double kDefaultBufferedChunkSecs = 2.0;
constexpr double kDefaultStreamingChunkSecs = 2.0;
constexpr double kDefaultStreamingLeftContextSecs = 10.0;
constexpr double kDefaultStreamingRightContextSecs = 2.0;

[[maybe_unused]] void fill_first_batch_features_time_major(
    const ParakeetFrontendBatch & frontend,
    std::vector<float> & values) {
    values.assign(static_cast<size_t>(frontend.feature_dim * frontend.frames), 0.0f);
    for (int64_t t = 0; t < frontend.frames; ++t) {
        for (int64_t f = 0; f < frontend.feature_dim; ++f) {
            values[static_cast<size_t>(f * frontend.frames + t)] =
                frontend.features[static_cast<size_t>(t * frontend.feature_dim + f)];
        }
    }
}

std::shared_ptr<const ParakeetAssets> require_assets(std::shared_ptr<const ParakeetAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Parakeet TDT session requires assets");
    }
    return assets;
}

bool requests_long_context_buffered_encoder(const runtime::SessionOptions & options) {
    const auto it = options.options.find("encoder_variant");
    if (it == options.options.end() || it->second.empty() || it->second == "full_context") {
        return false;
    }
    if (it->second == "long_context" || it->second == "longform") {
        return true;
    }
    throw std::runtime_error("Parakeet TDT encoder_variant must be 'full_context' or 'long_context'");
}

double parse_double_option(
    const runtime::SessionOptions & options,
    const char * key,
    double default_value) {
    const auto it = options.options.find(key);
    if (it == options.options.end() || it->second.empty()) {
        return default_value;
    }
    return std::stod(it->second);
}

assets::TensorStorageType option_weight_type(
    const runtime::SessionOptions & options,
    const char * key,
    assets::TensorStorageType default_value) {
    const auto it = options.options.find(key);
    if (it == options.options.end() || it->second.empty()) {
        return default_value;
    }
    return assets::parse_tensor_storage_type(it->second);
}

void validate_weight_storage(assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == assets::TensorStorageType::Native ||
        storage_type == assets::TensorStorageType::F32 ||
        storage_type == assets::TensorStorageType::F16 ||
        storage_type == assets::TensorStorageType::BF16 ||
        storage_type == assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " currently supports only native, f32, f16, bf16, and q8_0");
}

void validate_session_options(const runtime::SessionOptions & options) {
    for (const auto & [key, value] : options.options) {
        (void) value;
        if (key.rfind("parakeet_tdt.", 0) == 0 &&
            key != "parakeet_tdt.weight_context_mb" &&
            key != "parakeet_tdt.weight_type" &&
            key != "parakeet_tdt.matmul_weight_type" &&
            key != "parakeet_tdt.conv_weight_type") {
            throw std::runtime_error("unknown Parakeet TDT session option: " + key);
        }
    }
}

int64_t make_divisible_by(int64_t value, int64_t factor) {
    if (factor <= 0) {
        throw std::runtime_error("Streaming factor must be positive");
    }
    return (value / factor) * factor;
}

int64_t conv_output_dim(int64_t input, int64_t kernel, int64_t stride, int64_t padding) {
    return (input + 2 * padding - kernel) / stride + 1;
}

int64_t frontend_frames_for_audio_samples(const ParakeetFeatureExtractorConfig & config, int64_t samples) {
    const int64_t pad = config.n_fft / 2;
    return 1 + (samples + 2 * pad - config.n_fft) / config.hop_length;
}

int64_t pre_encode_frames_for_frontend_frames(const ParakeetAssets & assets, int64_t frontend_frames) {
    const int64_t kernel = assets.model_config.encoder.subsampling_conv_kernel_size;
    const int64_t stride = assets.model_config.encoder.subsampling_conv_stride;
    const int64_t padding = (kernel - 1) / 2;
    const int64_t stage1_frames = conv_output_dim(frontend_frames, kernel, stride, padding);
    const int64_t stage2_frames = conv_output_dim(stage1_frames, kernel, stride, padding);
    return conv_output_dim(stage2_frames, kernel, stride, padding);
}

int64_t buffered_encoder_frame_to_samples(const ParakeetAssets & assets) {
    return static_cast<int64_t>(assets.feature_config.hop_length) *
        assets.model_config.encoder.subsampling_factor;
}

int64_t pre_encode_frames_for_audio_samples(const ParakeetAssets & assets, int64_t samples) {
    return pre_encode_frames_for_frontend_frames(
        assets,
        frontend_frames_for_audio_samples(assets.feature_config, samples));
}

int64_t max_audio_samples_for_pre_encode_capacity(const ParakeetAssets & assets, int64_t target_frames) {
    if (target_frames <= 0) {
        throw std::runtime_error("Parakeet capacity target frames must be positive");
    }
    const int64_t sample_quantum = buffered_encoder_frame_to_samples(assets);
    int64_t low = sample_quantum;
    int64_t high = sample_quantum;
    while (pre_encode_frames_for_audio_samples(assets, high) <= target_frames) {
        if (high > std::numeric_limits<int64_t>::max() / 2) {
            break;
        }
        high *= 2;
    }
    int64_t best = sample_quantum;
    while (low <= high) {
        const int64_t mid_steps = ((low + high) / 2) / sample_quantum;
        const int64_t mid = std::max<int64_t>(sample_quantum, mid_steps * sample_quantum);
        const int64_t actual_frames = pre_encode_frames_for_audio_samples(assets, mid);
        if (actual_frames <= target_frames) {
            best = mid;
            low = mid + sample_quantum;
        } else {
            high = mid - sample_quantum;
        }
    }
    return best;
}

}  // namespace

decoders::TdtDecoderAlgorithm ParakeetTDTSession::parse_decoder_algorithm_option(const runtime::SessionOptions & options) {
    const auto it = options.options.find("decoder_algorithm");
    if (it == options.options.end() || it->second.empty()) {
        return decoders::TdtDecoderAlgorithm::GreedyDurationLoop;
    }
    if (it->second == "greedy_customized") {
        return decoders::TdtDecoderAlgorithm::GreedyCustomized;
    }
    if (it->second == "greedy_duration_loop") {
        return decoders::TdtDecoderAlgorithm::GreedyDurationLoop;
    }
    if (it->second == "greedy_separate_heads") {
        return decoders::TdtDecoderAlgorithm::GreedySeparateHeads;
    }
    throw std::runtime_error(
        "Parakeet TDT decoder_algorithm must be 'greedy_customized', 'greedy_duration_loop', or 'greedy_separate_heads'");
}

ParakeetTDTSession::BufferedCapacityContract ParakeetTDTSession::make_capacity_contract_for_target_frames(
    const ParakeetAssets & assets,
    int64_t target_frames) {
    if (target_frames <= 0) {
        throw std::runtime_error("Parakeet capacity target frames must be positive");
    }
    if (target_frames > assets.model_config.encoder.max_position_embeddings) {
        throw std::runtime_error("Parakeet capacity target exceeds max_position_embeddings");
    }

    BufferedCapacityContract contract;
    contract.audio_samples = max_audio_samples_for_pre_encode_capacity(assets, target_frames);
    contract.pre_encode_frames = pre_encode_frames_for_audio_samples(assets, contract.audio_samples);
    if (contract.audio_samples <= 0 || contract.pre_encode_frames <= 0) {
        throw std::runtime_error("Failed to resolve Parakeet buffered capacity contract");
    }
    if (contract.pre_encode_frames > target_frames) {
        throw std::runtime_error("Resolved Parakeet buffered capacity exceeds requested encoder-frame target");
    }
    return contract;
}

ParakeetTDTSession::ParakeetTDTSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const ParakeetAssets> assets)
    : runtime::RuntimeSessionBase(options),
      task_(task),
      options_(std::move(options)),
      assets_(require_assets(std::move(assets))),
      decoder_algorithm_(parse_decoder_algorithm_option(options_)) {
    frontend_audio_batch_.resize(1);
    if (task_.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Parakeet TDT only supports VoiceTaskKind::Asr");
    }
    validate_session_options(options_);
    weight_context_bytes_ = runtime::parse_size_mb_option(
        options_.options,
        {"parakeet_tdt.weight_context_mb"},
        weight_context_bytes_);
    const auto default_weight_type = option_weight_type(
        options_,
        "parakeet_tdt.weight_type",
        assets::TensorStorageType::Native);
    matmul_weight_storage_type_ = option_weight_type(
        options_,
        "parakeet_tdt.matmul_weight_type",
        default_weight_type);
    conv_weight_storage_type_ = option_weight_type(
        options_,
        "parakeet_tdt.conv_weight_type",
        default_weight_type);
    validate_weight_storage(matmul_weight_storage_type_, "parakeet_tdt.matmul_weight_type");
    validate_weight_storage(conv_weight_storage_type_, "parakeet_tdt.conv_weight_type");
    weights_ = load_parakeet_tdt_weights(
        *assets_,
        execution_context().backend(),
        execution_context().backend_type(),
        matmul_weight_storage_type_,
        conv_weight_storage_type_,
        weight_context_bytes_);
    assets_->model_weights->release_storage();
    shared_decoder_ = std::make_unique<ParakeetSharedDecoder>(*assets_, *weights_);
    const auto full_context_mode = runtime::resolve_graph_capacity_mode(
        options_,
        runtime::GraphCapacityMode::Tiered,
        {"full_context_graph_capacity_mode", "offline_graph_capacity_mode", "graph_capacity_mode"});
    const auto long_context_mode = runtime::resolve_graph_capacity_mode(
        options_,
        runtime::GraphCapacityMode::Tiered,
        {"long_context_graph_capacity_mode", "offline_graph_capacity_mode", "graph_capacity_mode"});
    const auto streaming_mode = runtime::resolve_graph_capacity_mode(
        options_,
        runtime::GraphCapacityMode::Fixed,
        {"streaming_graph_capacity_mode", "graph_capacity_mode"});
    if (full_context_mode == runtime::GraphCapacityMode::Unsupported ||
        long_context_mode == runtime::GraphCapacityMode::Unsupported ||
        streaming_mode == runtime::GraphCapacityMode::Unsupported) {
        throw std::runtime_error("Parakeet TDT graph_capacity_mode=unsupported is not implemented");
    }
    full_context_capacity_controller_ = runtime::GraphCapacityController(full_context_mode);
    long_context_capacity_controller_ = runtime::GraphCapacityController(long_context_mode);
    initialize_buffered_capacity_contracts();
    if (task_.mode == runtime::RunMode::Streaming) {
        if (streaming_mode != runtime::GraphCapacityMode::Fixed) {
            throw std::runtime_error("Parakeet streaming currently requires graph_capacity_mode=fixed");
        }
        ensure_streaming_configured();
    } else {
        const bool use_long_context = requests_long_context_buffered_encoder(options_);
        if (use_long_context &&
            long_context_mode == runtime::GraphCapacityMode::Fixed &&
            long_context_tiers_.size() != 1) {
            throw std::runtime_error(
                "Parakeet long_context graph_capacity_mode=fixed requires explicit long_context_capacity_frames");
        }
        if (!use_long_context &&
            full_context_mode == runtime::GraphCapacityMode::Fixed &&
            full_context_tiers_.size() != 1) {
            throw std::runtime_error(
                "Parakeet full_context graph_capacity_mode=fixed requires explicit full_context_capacity_frames");
        }
    }
}

namespace {

std::string trace_mode_label(ParakeetTraceContractMode trace_mode) {
    switch (trace_mode) {
        case ParakeetTraceContractMode::Offline:
            return "offline";
        case ParakeetTraceContractMode::Longform:
            return "longform";
        case ParakeetTraceContractMode::Streaming:
            return "streaming";
    }
    throw std::runtime_error("Unsupported Parakeet trace mode");
}

std::string decoder_algorithm_label(decoders::TdtDecoderAlgorithm algorithm) {
    switch (algorithm) {
        case decoders::TdtDecoderAlgorithm::GreedyCustomized:
            return "greedy_customized";
        case decoders::TdtDecoderAlgorithm::GreedyDurationLoop:
            return "greedy_duration_loop";
        case decoders::TdtDecoderAlgorithm::GreedySeparateHeads:
            return "greedy_separate_heads";
    }
    throw std::runtime_error("Unsupported TDT decoder algorithm");
}

std::vector<int32_t> make_pad_mask_i32(int64_t frames, int64_t valid_frames) {
    std::vector<int32_t> mask(static_cast<size_t>(frames), 0);
    for (int64_t i = 0; i < frames; ++i) {
        mask[static_cast<size_t>(i)] = i < valid_frames ? 0 : 1;
    }
    return mask;
}

std::vector<int32_t> make_keep_mask_i32(int64_t frames, int64_t valid_frames) {
    std::vector<int32_t> mask(static_cast<size_t>(frames), 0);
    for (int64_t i = 0; i < frames; ++i) {
        mask[static_cast<size_t>(i)] = i < valid_frames ? 1 : 0;
    }
    return mask;
}

std::vector<float> make_global_attention_bias(int64_t frames, int64_t valid_frames) {
    std::vector<float> mask(static_cast<size_t>(frames * frames), -std::numeric_limits<float>::infinity());
    for (int64_t q = 0; q < valid_frames; ++q) {
        for (int64_t k = 0; k < valid_frames; ++k) {
            mask[static_cast<size_t>(q * frames + k)] = 0.0f;
        }
    }
    for (int64_t q = valid_frames; q < frames; ++q) {
        mask[static_cast<size_t>(q * frames + q)] = 0.0f;
    }
    return mask;
}

struct TimestampCharOffset {
    std::string text;
    std::string token;
    int32_t start_offset = 0;
    int32_t end_offset = 0;
    double start = 0.0;
    double end = 0.0;
};

struct TimestampWordOffset {
    std::string word;
    int32_t start_offset = 0;
    int32_t end_offset = 0;
    double start = 0.0;
    double end = 0.0;
};

struct TimestampSegmentOffset {
    std::string segment;
    int32_t start_offset = 0;
    int32_t end_offset = 0;
    double start = 0.0;
    double end = 0.0;
};

std::string lstrip_spaces(std::string text) {
    while (!text.empty() && text.front() == ' ') {
        text.erase(text.begin());
    }
    return text;
}

std::string decode_parakeet_tokens_to_str(const std::vector<std::string> & tokens) {
    std::string text;
    for (const auto & token : tokens) {
        text += token;
    }
    constexpr const char * kSentencePieceSpace = "\xE2\x96\x81";
    size_t pos = 0;
    while ((pos = text.find(kSentencePieceSpace, pos)) != std::string::npos) {
        text.replace(pos, 3, " ");
        pos += 1;
    }
    return lstrip_spaces(std::move(text));
}

std::vector<std::string> extract_supported_punctuation(const ParakeetAssets & assets) {
    std::vector<std::string> out;
    for (const auto & token : assets.tokenizer->id_to_token()) {
        if (token.empty()) {
            continue;
        }
        if (token.rfind("##", 0) == 0 || token.rfind("\xE2\x96\x81", 0) == 0 || token.front() == '<' || token.front() == '[') {
            continue;
        }
        for (unsigned char ch : token) {
            if (std::ispunct(ch) != 0) {
                std::string s(1, static_cast<char>(ch));
                if (std::find(out.begin(), out.end(), s) == out.end()) {
                    out.push_back(std::move(s));
                }
            }
        }
    }
    for (const std::string & token : {std::string("'"), std::string(","), std::string("."), std::string("!"), std::string("?")}) {
        if (std::find(out.begin(), out.end(), token) == out.end()) {
            out.push_back(token);
        }
    }
    return out;
}

std::vector<TimestampCharOffset> build_char_offsets(
    const ParakeetAssets & assets,
    const ParakeetDecodeResult & decoded,
    double window_stride,
    int64_t subsampling_factor) {
    std::vector<TimestampCharOffset> out;
    const auto & vocab = assets.tokenizer->id_to_token();
    const size_t count = std::min({decoded.token_ids.size(), decoded.token_timestamps.size(), decoded.token_durations.size()});
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const int32_t token_id = decoded.token_ids[i];
        if (token_id < 0 || static_cast<size_t>(token_id) >= vocab.size() || token_id == assets.model_config.blank_token_id) {
            continue;
        }
        TimestampCharOffset item;
        item.token = vocab[static_cast<size_t>(token_id)];
        item.text = assets.tokenizer->decode_ids({token_id}, assets.model_config.blank_token_id);
        item.start_offset = decoded.token_timestamps[i];
        item.end_offset = decoded.token_timestamps[i] + decoded.token_durations[i];
        item.start = static_cast<double>(item.start_offset) * window_stride * static_cast<double>(subsampling_factor);
        item.end = static_cast<double>(item.end_offset) * window_stride * static_cast<double>(subsampling_factor);
        out.push_back(std::move(item));
    }
    const auto supported_punctuation = extract_supported_punctuation(assets);
    for (size_t i = 1; i < out.size(); ++i) {
        if (std::find(supported_punctuation.begin(), supported_punctuation.end(), out[i].text) != supported_punctuation.end()) {
            out[i].start_offset = out[i - 1].end_offset;
            out[i].end_offset = out[i].start_offset;
            out[i].start = static_cast<double>(out[i].start_offset) * window_stride * static_cast<double>(subsampling_factor);
            out[i].end = static_cast<double>(out[i].end_offset) * window_stride * static_cast<double>(subsampling_factor);
        }
    }
    return out;
}

std::vector<TimestampWordOffset> build_word_offsets(
    const std::vector<TimestampCharOffset> & char_offsets) {
    if (char_offsets.empty()) {
        return {};
    }

    const std::vector<std::string> supported_punctuation = {",", ".", "!", "?"};
    std::vector<TimestampWordOffset> word_offsets;
    std::vector<std::string> built_tokens;
    int previous_token_index = 0;

    auto is_supported_punctuation = [&](const std::string & token) {
        return std::find(supported_punctuation.begin(), supported_punctuation.end(), token) != supported_punctuation.end();
    };

    auto next_non_delimiter_token = [&](size_t i) -> std::string {
        for (size_t j = i + 1; j < char_offsets.size(); ++j) {
            if (char_offsets[j].text != " ") {
                return char_offsets[j].text;
            }
        }
        return {};
    };

    auto append_word = [&](int start_index, int end_index) {
        TimestampWordOffset word;
        word.word = decode_parakeet_tokens_to_str(built_tokens);
        word.start_offset = char_offsets[static_cast<size_t>(start_index)].start_offset;
        word.end_offset = char_offsets[static_cast<size_t>(end_index)].end_offset;
        word.start = char_offsets[static_cast<size_t>(start_index)].start;
        word.end = char_offsets[static_cast<size_t>(end_index)].end;
        word_offsets.push_back(std::move(word));
    };

    for (size_t i = 0; i < char_offsets.size(); ++i) {
        const auto & char_offset = char_offsets[i];
        const std::string & char_text = char_offset.text;
        const std::string & char_token = char_offset.token;
        const bool curr_punctuation = is_supported_punctuation(char_text) && char_text != " ";
        const std::string next_token = next_non_delimiter_token(i);
        const bool word_start =
            (char_token != char_text) ||
            (char_text == " " && !is_supported_punctuation(next_token));

        if (word_start && !curr_punctuation) {
            if (!built_tokens.empty()) {
                append_word(previous_token_index, static_cast<int>(i - 1));
            }
            built_tokens.clear();
            if (char_text != " ") {
                built_tokens.push_back(char_token);
                previous_token_index = static_cast<int>(i);
            }
        } else if (curr_punctuation && built_tokens.empty() && !word_offsets.empty()) {
            auto & last_word = word_offsets.back();
            last_word.end_offset = char_offset.end_offset;
            last_word.end = char_offset.end;
            if (!last_word.word.empty() && last_word.word.back() == ' ') {
                last_word.word.pop_back();
            }
            last_word.word += char_text;
        } else if (curr_punctuation && !built_tokens.empty()) {
            if (!built_tokens.empty() &&
                (built_tokens.back() == " " || built_tokens.back() == "_" || built_tokens.back() == "\xE2\x96\x81")) {
                built_tokens.pop_back();
            }
            built_tokens.push_back(char_token);
        } else {
            if (built_tokens.empty()) {
                previous_token_index = static_cast<int>(i);
            }
            built_tokens.push_back(char_token);
        }
    }

    if (word_offsets.empty()) {
        if (!built_tokens.empty()) {
            TimestampWordOffset word;
            word.word = decode_parakeet_tokens_to_str(built_tokens);
            word.start_offset = char_offsets.front().start_offset;
            word.end_offset = char_offsets.back().end_offset;
            word.start = char_offsets.front().start;
            word.end = char_offsets.back().end;
            word_offsets.push_back(std::move(word));
        }
    } else {
        word_offsets.front().start_offset = char_offsets.front().start_offset;
        word_offsets.front().start = char_offsets.front().start;
        if (!built_tokens.empty()) {
            TimestampWordOffset word;
            word.word = decode_parakeet_tokens_to_str(built_tokens);
            word.start_offset = char_offsets[static_cast<size_t>(previous_token_index)].start_offset;
            word.end_offset = char_offsets.back().end_offset;
            word.start = char_offsets[static_cast<size_t>(previous_token_index)].start;
            word.end = char_offsets.back().end;
            word_offsets.push_back(std::move(word));
        }
    }

    return word_offsets;
}

[[maybe_unused]] std::vector<TimestampSegmentOffset> build_segment_offsets(
    const std::vector<TimestampWordOffset> & word_offsets) {
    if (word_offsets.empty()) {
        return {};
    }

    const std::vector<std::string> segment_delimiters = {".", "?", "!"};
    std::vector<TimestampSegmentOffset> segment_offsets;
    std::vector<std::string> segment_words;
    int previous_word_index = 0;

    auto is_segment_delimiter = [&](const std::string & word) {
        if (word.empty()) {
            return false;
        }
        if (std::find(segment_delimiters.begin(), segment_delimiters.end(), word) != segment_delimiters.end()) {
            return true;
        }
        return std::find(segment_delimiters.begin(), segment_delimiters.end(), std::string(1, word.back())) != segment_delimiters.end();
    };

    for (size_t i = 0; i < word_offsets.size(); ++i) {
        const auto & offset = word_offsets[i];
        if (is_segment_delimiter(offset.word)) {
            segment_words.push_back(offset.word);
            TimestampSegmentOffset segment;
            segment.segment = "";
            for (size_t j = 0; j < segment_words.size(); ++j) {
                if (j > 0) {
                    segment.segment += ' ';
                }
                segment.segment += segment_words[j];
            }
            segment.start_offset = word_offsets[static_cast<size_t>(previous_word_index)].start_offset;
            segment.end_offset = offset.end_offset;
            segment.start = word_offsets[static_cast<size_t>(previous_word_index)].start;
            segment.end = offset.end;
            segment_offsets.push_back(std::move(segment));
            segment_words.clear();
            previous_word_index = static_cast<int>(i + 1);
            continue;
        }
        segment_words.push_back(offset.word);
    }

    if (!segment_words.empty()) {
        TimestampSegmentOffset segment;
        for (size_t j = 0; j < segment_words.size(); ++j) {
            if (j > 0) {
                segment.segment += ' ';
            }
            segment.segment += segment_words[j];
        }
        segment.start_offset = word_offsets[static_cast<size_t>(previous_word_index)].start_offset;
        segment.end_offset = word_offsets.back().end_offset;
        segment.start = word_offsets[static_cast<size_t>(previous_word_index)].start;
        segment.end = word_offsets.back().end;
        segment_offsets.push_back(std::move(segment));
    }

    return segment_offsets;
}

}  // namespace

std::string ParakeetTDTSession::family() const {
    return "parakeet_tdt";
}

runtime::VoiceTaskKind ParakeetTDTSession::task_kind() const {
    return task_.task;
}

runtime::RunMode ParakeetTDTSession::run_mode() const {
    return task_.mode;
}

void ParakeetTDTSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (task_.mode == runtime::RunMode::Streaming) {
        if (request.audio.has_value()) {
            if (request.audio->sample_rate > 0 && request.audio->sample_rate != assets_->feature_config.sample_rate) {
                throw std::runtime_error("Parakeet streaming prepare sample_rate mismatch");
            }
            if (request.audio->channels > 0 && request.audio->channels != 1) {
                throw std::runtime_error("Parakeet streaming prepare currently requires mono audio");
            }
        }
        prepare_streaming_resources();
        mark_prepared();
        return;
    }

    const BufferedEncoderVariant encoder_variant =
        requests_long_context_buffered_encoder(options_) ? BufferedEncoderVariant::LongContext : BufferedEncoderVariant::FullContext;
    if (request.audio.has_value()) {
        if (request.audio->sample_rate > 0 && request.audio->sample_rate != assets_->feature_config.sample_rate) {
            throw std::runtime_error("Parakeet offline prepare sample_rate mismatch");
        }
        if (request.audio->channels > 0 && request.audio->channels != 1) {
            throw std::runtime_error("Parakeet offline prepare currently requires mono audio");
        }
    }
    const int64_t request_size =
        request.audio.has_value() ? request.audio->max_input_samples : 0;
    auto adapter = make_buffered_capacity_adapter(encoder_variant);
    buffered_capacity_controller(encoder_variant).ensure_prepared(adapter, request_size);
    mark_prepared();
}

std::pair<const ParakeetTDTSession::BufferedCapacityContract *, size_t>
ParakeetTDTSession::find_buffered_capacity_contract(
    int64_t capacity_audio_samples,
    BufferedEncoderVariant encoder_variant) {
    auto find_in_tiers = [&](const auto & tiers) -> std::pair<const BufferedCapacityContract *, size_t> {
        for (size_t i = 0; i < tiers.size(); ++i) {
            if (tiers[i].contract.audio_samples == capacity_audio_samples) {
                return {&tiers[i].contract, i};
            }
        }
        throw std::runtime_error("selected Parakeet graph capacity is not configured");
    };

    if (encoder_variant == BufferedEncoderVariant::LongContext) {
        return find_in_tiers(long_context_tiers_);
    }

    if (encoder_variant == BufferedEncoderVariant::FullContext) {
        return find_in_tiers(full_context_tiers_);
    }

    throw std::runtime_error("Parakeet buffered encoder variant is not supported");
}

runtime::GraphCapacityController & ParakeetTDTSession::buffered_capacity_controller(
    BufferedEncoderVariant encoder_variant) {
    if (encoder_variant == BufferedEncoderVariant::LongContext) {
        return long_context_capacity_controller_;
    }
    if (encoder_variant == BufferedEncoderVariant::FullContext) {
        return full_context_capacity_controller_;
    }
    throw std::runtime_error("Parakeet buffered encoder variant is not supported");
}

const runtime::GraphCapacityController & ParakeetTDTSession::buffered_capacity_controller(
    BufferedEncoderVariant encoder_variant) const {
    if (encoder_variant == BufferedEncoderVariant::LongContext) {
        return long_context_capacity_controller_;
    }
    if (encoder_variant == BufferedEncoderVariant::FullContext) {
        return full_context_capacity_controller_;
    }
    throw std::runtime_error("Parakeet buffered encoder variant is not supported");
}

std::vector<int64_t> ParakeetTDTSession::buffered_capacity_catalog(BufferedEncoderVariant encoder_variant) const {
    std::vector<int64_t> capacities;
    const auto append_tiers = [&](const auto & tiers) {
        capacities.reserve(tiers.size());
        for (const auto & tier : tiers) {
            capacities.push_back(tier.contract.audio_samples);
        }
    };
    if (encoder_variant == BufferedEncoderVariant::LongContext) {
        append_tiers(long_context_tiers_);
        return capacities;
    }
    if (encoder_variant == BufferedEncoderVariant::FullContext) {
        append_tiers(full_context_tiers_);
        return capacities;
    }
    throw std::runtime_error("Parakeet buffered encoder variant is not supported");
}

runtime::DiscreteGraphCapacityAdapter ParakeetTDTSession::make_buffered_capacity_adapter(
    BufferedEncoderVariant encoder_variant) {
    return runtime::DiscreteGraphCapacityAdapter(
        buffered_capacity_catalog(encoder_variant),
        [this, encoder_variant]() { return prepared_buffered_capacities(encoder_variant); },
        [this, encoder_variant](int64_t capacity) { prepare_buffered_capacity(encoder_variant, capacity); });
}

std::vector<int64_t> ParakeetTDTSession::prepared_buffered_capacities(BufferedEncoderVariant encoder_variant) const {
    std::vector<int64_t> capacities;
    const auto append_tiers = [&](const auto & tiers) {
        capacities.reserve(tiers.size());
        for (const auto & tier : tiers) {
            if (tier.pre_encode_scratch.graph != nullptr && tier.encoder.graph != nullptr) {
                capacities.push_back(tier.contract.audio_samples);
            }
        }
    };
    if (encoder_variant == BufferedEncoderVariant::LongContext) {
        append_tiers(long_context_tiers_);
        return capacities;
    }
    if (encoder_variant == BufferedEncoderVariant::FullContext) {
        append_tiers(full_context_tiers_);
        return capacities;
    }
    throw std::runtime_error("Parakeet buffered encoder variant is not supported");
}

void ParakeetTDTSession::prepare_buffered_capacity(
    BufferedEncoderVariant encoder_variant,
    int64_t capacity_audio_samples) {
    const auto [contract, contract_index] = find_buffered_capacity_contract(capacity_audio_samples, encoder_variant);
    (void)contract;
    prepare_buffered_tier(contract_index, encoder_variant);
}

void ParakeetTDTSession::initialize_shared_encoder_cache(
    const ParakeetPreEncodeBatch & pre_encode,
    BufferedEncoderVariant encoder_variant,
    FullContextEncoderState * full_context_encoder) {
    const int64_t frames = pre_encode.frames;
    const int64_t hidden = assets_->model_config.encoder.hidden_size;
    const bool use_long_context_encoder = encoder_variant == BufferedEncoderVariant::LongContext;
    const int64_t pos_frames = use_long_context_encoder ? (kParakeetLongformContext + 1) : frames;

    const bool position_frames_dirty = !use_long_context_encoder &&
        shared_encoder_.cached_position_frames != frames;
    const bool position_cache_dirty =
        position_frames_dirty ||
        shared_encoder_.cached_position_long_context != use_long_context_encoder ||
        shared_encoder_.cached_pos_frames != pos_frames;
    const bool mask_cache_dirty =
        shared_encoder_.cached_mask_frames != frames ||
        shared_encoder_.cached_mask_valid_frames != pre_encode.valid_frames;

    if (position_cache_dirty) {
        shared_encoder_.pos_emb_cache = compute_parakeet_relative_positional_encoding(
            1,
            hidden,
            pos_frames,
            assets_->model_config.encoder.max_position_embeddings);
        shared_encoder_.cached_position_frames = frames;
        shared_encoder_.cached_position_long_context = use_long_context_encoder;
        shared_encoder_.cached_pos_frames = pos_frames;
        ++shared_encoder_.position_cache_generation;
    }

    if (mask_cache_dirty) {
        shared_encoder_.pad_mask_cache = make_pad_mask_i32(frames, pre_encode.valid_frames);
        shared_encoder_.keep_mask_cache = make_keep_mask_i32(frames, pre_encode.valid_frames);
        shared_encoder_.cached_mask_frames = frames;
        shared_encoder_.cached_mask_valid_frames = pre_encode.valid_frames;
        ++shared_encoder_.mask_cache_generation;
    }

    if (!use_long_context_encoder && full_context_encoder != nullptr &&
        (full_context_encoder->cached_attention_frames != frames ||
         full_context_encoder->cached_attention_valid_frames != pre_encode.valid_frames)) {
        full_context_encoder->attention_bias_cache = make_global_attention_bias(frames, pre_encode.valid_frames);
        full_context_encoder->cached_attention_frames = frames;
        full_context_encoder->cached_attention_valid_frames = pre_encode.valid_frames;
        ++full_context_encoder->attention_cache_generation;
    }

}

ParakeetDecodeResult ParakeetTDTSession::run_shared_decoder(
    const std::vector<float> & encoder_projected,
    int64_t frames,
    decoders::TdtDecoderAlgorithm decoder_algorithm,
    ParakeetTraceContractMode) {
    return shared_decoder_->run(
        execution_context(),
        encoder_projected,
        frames,
        decoder_algorithm);
}

void ParakeetTDTSession::initialize_buffered_capacity_contracts() {
    const double default_chunk_secs = parse_double_option(options_, "buffered_chunk_secs", kDefaultBufferedChunkSecs);
    const int64_t sample_rate = assets_->feature_config.sample_rate;
    const int64_t default_tier_seconds = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(default_chunk_secs)));
    const int64_t max_audio_samples =
        make_capacity_contract_for_target_frames(*assets_, assets_->model_config.encoder.max_position_embeddings).audio_samples;
    const int64_t max_tier_seconds = std::max<int64_t>(default_tier_seconds, max_audio_samples / sample_rate);

    auto make_contract_for_audio_samples = [&](int64_t audio_samples) {
        BufferedCapacityContract contract;
        contract.audio_samples = audio_samples;
        contract.pre_encode_frames = pre_encode_frames_for_audio_samples(*assets_, audio_samples);
        if (contract.audio_samples <= 0 || contract.pre_encode_frames <= 0) {
            throw std::runtime_error("Failed to resolve Parakeet buffered capacity contract");
        }
        if (contract.pre_encode_frames > assets_->model_config.encoder.max_position_embeddings) {
            throw std::runtime_error("Parakeet buffered capacity tier exceeds max_position_embeddings");
        }
        return contract;
    };

    auto populate_second_buckets = [&](auto & tiers) {
        int64_t last_frames = -1;
        for (int64_t seconds = default_tier_seconds; seconds <= max_tier_seconds; ++seconds) {
            const int64_t audio_samples = seconds * sample_rate;
            auto contract = make_contract_for_audio_samples(audio_samples);
            if (contract.pre_encode_frames == last_frames) {
                continue;
            }
            typename std::remove_reference_t<decltype(tiers)>::value_type tier;
            tier.contract = contract;
            tiers.push_back(std::move(tier));
            last_frames = contract.pre_encode_frames;
        }
    };

    const auto full_it = options_.options.find("full_context_capacity_frames");
    const bool explicit_full_capacity = full_it != options_.options.end() && !full_it->second.empty();
    if (explicit_full_capacity) {
        FullContextCapacityTier tier;
        tier.contract = make_capacity_contract_for_target_frames(*assets_, std::stoll(full_it->second));
        full_context_tiers_.push_back(std::move(tier));
    } else {
        populate_second_buckets(full_context_tiers_);
    }

    const auto long_it = options_.options.find("long_context_capacity_frames");
    const bool explicit_long_capacity = long_it != options_.options.end() && !long_it->second.empty();
    if (explicit_long_capacity) {
        LongContextCapacityTier tier;
        tier.contract = make_capacity_contract_for_target_frames(*assets_, std::stoll(long_it->second));
        long_context_tiers_.push_back(std::move(tier));
    } else {
        populate_second_buckets(long_context_tiers_);
    }

    if (!explicit_full_capacity &&
        (full_context_tiers_.empty() ||
         full_context_tiers_.back().contract.audio_samples < max_audio_samples)) {
        FullContextCapacityTier final_tier;
        final_tier.contract = make_capacity_contract_for_target_frames(
            *assets_,
            assets_->model_config.encoder.max_position_embeddings);
        full_context_tiers_.push_back(std::move(final_tier));
    }
    if (!explicit_long_capacity &&
        (long_context_tiers_.empty() ||
         long_context_tiers_.back().contract.audio_samples < max_audio_samples)) {
        LongContextCapacityTier final_tier;
        final_tier.contract = make_capacity_contract_for_target_frames(
            *assets_,
            assets_->model_config.encoder.max_position_embeddings);
        long_context_tiers_.push_back(std::move(final_tier));
    }
}

void ParakeetTDTSession::prepare_buffered_tier(size_t contract_index, BufferedEncoderVariant encoder_variant) {
    const auto & contract = encoder_variant == BufferedEncoderVariant::LongContext
        ? long_context_tiers_.at(contract_index).contract
        : full_context_tiers_.at(contract_index).contract;
    const auto total_start = std::chrono::steady_clock::now();

    runtime::AudioBuffer audio;
    audio.sample_rate = assets_->feature_config.sample_rate;
    audio.channels = 1;
    audio.samples.assign(static_cast<size_t>(contract.audio_samples), 0.0f);

    std::vector<runtime::AudioBuffer> warmup_audio(1, audio);
    std::vector<int64_t> audio_lengths_override(1, contract.audio_samples);
    ParakeetFrontendBatch frontend;
    const auto frontend_start = std::chrono::steady_clock::now();
    compute_parakeet_frontend(
        warmup_audio,
        &audio_lengths_override,
        assets_->feature_config,
        execution_context(),
        frontend,
        frontend_scratch_);
    const auto frontend_end = std::chrono::steady_clock::now();

    ParakeetPreEncodeBatch pre_encode;
    if (encoder_variant == BufferedEncoderVariant::LongContext) {
        auto & tier = long_context_tiers_.at(contract_index);
        const bool had_pre_encode_graph = tier.pre_encode_scratch.graph != nullptr;
        const auto pre_encode_start = std::chrono::steady_clock::now();
        compute_parakeet_pre_encode(
            frontend,
            *assets_,
            *weights_,
            execution_context(),
            pre_encode,
            tier.pre_encode_scratch);
        const auto pre_encode_end = std::chrono::steady_clock::now();
        if (!had_pre_encode_graph && tier.pre_encode_scratch.graph != nullptr) {
            ++long_context_pre_encode_graphs_built_;
        }
        const bool had_encoder_graph = tier.encoder.graph != nullptr;
        const auto encoder_cache_start = std::chrono::steady_clock::now();
        initialize_shared_encoder_cache(pre_encode, encoder_variant, nullptr);
        const auto encoder_cache_end = std::chrono::steady_clock::now();
        const auto encoder_graph_start = std::chrono::steady_clock::now();
        ensure_long_context_encoder_graph(tier.encoder, pre_encode.frames);
        const auto encoder_graph_end = std::chrono::steady_clock::now();
        if (!had_encoder_graph && tier.encoder.graph != nullptr) {
            ++long_context_encoder_graphs_built_;
        }
        latest_long_context_tier_index_ = contract_index;
        const auto decoder_prepare_start = std::chrono::steady_clock::now();
        shared_decoder_->prepare(execution_context(), pre_encode.frames);
        const auto decoder_prepare_end = std::chrono::steady_clock::now();
        debug::trace_log_scalar("parakeet.buffered_prepare.long_context", true);
        debug::trace_log_scalar("parakeet.buffered_prepare.capacity_audio_samples", contract.audio_samples);
        debug::trace_log_scalar("parakeet.buffered_prepare.capacity_frames", pre_encode.frames);
        if (debug::timing_log_enabled()) {
            debug::timing_log_scalar("parakeet.buffered_prepare.frontend_ms", engine::debug::elapsed_ms(frontend_start, frontend_end));
            debug::timing_log_scalar("parakeet.buffered_prepare.pre_encode_ms", engine::debug::elapsed_ms(pre_encode_start, pre_encode_end));
            debug::timing_log_scalar(
                "parakeet.buffered_prepare.encoder_cache_ms",
                engine::debug::elapsed_ms(encoder_cache_start, encoder_cache_end));
            debug::timing_log_scalar(
                "parakeet.buffered_prepare.encoder.graph.total_ms",
                engine::debug::elapsed_ms(encoder_graph_start, encoder_graph_end));
            debug::timing_log_scalar(
                "parakeet.buffered_prepare.decoder_prepare_ms",
                engine::debug::elapsed_ms(decoder_prepare_start, decoder_prepare_end));
            debug::timing_log_scalar(
                "parakeet.buffered_prepare.total_ms",
                engine::debug::elapsed_ms(total_start, decoder_prepare_end));
        }
        return;
    }

    auto & tier = full_context_tiers_.at(contract_index);
    const bool had_pre_encode_graph = tier.pre_encode_scratch.graph != nullptr;
    const auto pre_encode_start = std::chrono::steady_clock::now();
    compute_parakeet_pre_encode(
        frontend,
        *assets_,
        *weights_,
        execution_context(),
        pre_encode,
        tier.pre_encode_scratch);
    const auto pre_encode_end = std::chrono::steady_clock::now();
    if (!had_pre_encode_graph && tier.pre_encode_scratch.graph != nullptr) {
        ++full_context_pre_encode_graphs_built_;
    }
    const bool had_encoder_graph = tier.encoder.graph != nullptr;
    const auto encoder_cache_start = std::chrono::steady_clock::now();
    initialize_shared_encoder_cache(pre_encode, encoder_variant, &tier.encoder);
    const auto encoder_cache_end = std::chrono::steady_clock::now();
    const auto encoder_graph_start = std::chrono::steady_clock::now();
    ensure_full_context_encoder_graph(tier.encoder, pre_encode.frames);
    const auto encoder_graph_end = std::chrono::steady_clock::now();
    if (!had_encoder_graph && tier.encoder.graph != nullptr) {
        ++full_context_encoder_graphs_built_;
    }
    latest_full_context_tier_index_ = contract_index;
    const auto decoder_prepare_start = std::chrono::steady_clock::now();
    shared_decoder_->prepare(execution_context(), pre_encode.frames);
    const auto decoder_prepare_end = std::chrono::steady_clock::now();
    debug::trace_log_scalar("parakeet.buffered_prepare.long_context", false);
    debug::trace_log_scalar("parakeet.buffered_prepare.capacity_audio_samples", contract.audio_samples);
    debug::trace_log_scalar("parakeet.buffered_prepare.capacity_frames", pre_encode.frames);
    if (debug::timing_log_enabled()) {
        debug::timing_log_scalar("parakeet.buffered_prepare.frontend_ms", engine::debug::elapsed_ms(frontend_start, frontend_end));
        debug::timing_log_scalar("parakeet.buffered_prepare.pre_encode_ms", engine::debug::elapsed_ms(pre_encode_start, pre_encode_end));
        debug::timing_log_scalar(
            "parakeet.buffered_prepare.encoder_cache_ms",
            engine::debug::elapsed_ms(encoder_cache_start, encoder_cache_end));
        debug::timing_log_scalar(
            "parakeet.buffered_prepare.encoder.graph.total_ms",
            engine::debug::elapsed_ms(encoder_graph_start, encoder_graph_end));
        debug::timing_log_scalar(
            "parakeet.buffered_prepare.decoder_prepare_ms",
            engine::debug::elapsed_ms(decoder_prepare_start, decoder_prepare_end));
        debug::timing_log_scalar(
            "parakeet.buffered_prepare.total_ms",
            engine::debug::elapsed_ms(total_start, decoder_prepare_end));
    }
}

void ParakeetTDTSession::prepare_streaming_resources() {
    ensure_streaming_configured();
    runtime::AudioBuffer audio;
    audio.sample_rate = assets_->feature_config.sample_rate;
    audio.channels = 1;
    audio.samples.assign(static_cast<size_t>(streaming_state_.context.window_samples), 0.0f);

    std::vector<runtime::AudioBuffer> warmup_audio(1, audio);
    std::vector<int64_t> audio_lengths_override(1, streaming_state_.context.window_samples);
    ParakeetFrontendBatch frontend;
    compute_parakeet_frontend(
        warmup_audio,
        &audio_lengths_override,
        assets_->feature_config,
        execution_context(),
        frontend,
        frontend_scratch_);

    ParakeetPreEncodeBatch pre_encode;
    compute_parakeet_pre_encode(
        frontend,
        *assets_,
        *weights_,
        execution_context(),
        pre_encode,
        streaming_pre_encode_scratch_);

    ensure_streaming_chunk_encoder_graph(
        pre_encode.frames,
        streaming_state_.context.left_frames,
        streaming_state_.context.right_frames);
    shared_decoder_->prepare(execution_context(), pre_encode.frames);
}

void ParakeetTDTSession::ensure_streaming_configured() {
    if (streaming_state_.configured) {
        return;
    }
    const double chunk_secs = parse_double_option(options_, "chunk_secs", kDefaultStreamingChunkSecs);
    const double left_context_secs = parse_double_option(options_, "left_context_secs", kDefaultStreamingLeftContextSecs);
    const double right_context_secs = parse_double_option(options_, "right_context_secs", kDefaultStreamingRightContextSecs);
    const double feature_stride_sec =
        static_cast<double>(assets_->feature_config.hop_length) / static_cast<double>(assets_->feature_config.sample_rate);
    const double features_per_sec = 1.0 / feature_stride_sec;
    const int64_t features_frame_to_samples = make_divisible_by(
        assets_->feature_config.sample_rate * assets_->feature_config.hop_length / assets_->feature_config.sample_rate,
        assets_->model_config.encoder.subsampling_factor);
    streaming_state_.context.encoder_frame_to_samples =
        features_frame_to_samples * assets_->model_config.encoder.subsampling_factor;
    streaming_state_.context.left_frames = static_cast<int64_t>(
        left_context_secs * features_per_sec / static_cast<double>(assets_->model_config.encoder.subsampling_factor));
    streaming_state_.context.chunk_frames = static_cast<int64_t>(
        chunk_secs * features_per_sec / static_cast<double>(assets_->model_config.encoder.subsampling_factor));
    streaming_state_.context.right_frames = static_cast<int64_t>(
        right_context_secs * features_per_sec / static_cast<double>(assets_->model_config.encoder.subsampling_factor));
    streaming_state_.context.left_samples =
        streaming_state_.context.left_frames * streaming_state_.context.encoder_frame_to_samples;
    streaming_state_.context.chunk_samples =
        streaming_state_.context.chunk_frames * streaming_state_.context.encoder_frame_to_samples;
    streaming_state_.context.right_samples =
        streaming_state_.context.right_frames * streaming_state_.context.encoder_frame_to_samples;
    streaming_state_.context.window_samples =
        streaming_state_.context.left_samples +
        streaming_state_.context.chunk_samples +
        streaming_state_.context.right_samples;
    streaming_state_.context.window_frames = pre_encode_frames_for_audio_samples(
        *assets_,
        streaming_state_.context.window_samples);
    if (streaming_state_.context.chunk_samples <= 0 || streaming_state_.context.encoder_frame_to_samples <= 0) {
        throw std::runtime_error("Parakeet TDT streaming chunk configuration must be positive");
    }
    if (streaming_state_.context.window_samples <= 0) {
        throw std::runtime_error("Parakeet TDT streaming window configuration must be positive");
    }
    streaming_state_.configured = true;
}

runtime::StreamEvent ParakeetTDTSession::make_stream_event_from_merged_decode() const {
    runtime::StreamEvent event;
    if (streaming_state_.has_latest_result) {
        event.partial_text = streaming_state_.latest_text;
        event.word_timestamps = streaming_state_.latest_word_timestamps;
        return event;
    }
    event.partial_text = runtime::Transcript{
        assets_->tokenizer->decode_ids(
            streaming_state_.merged_decoded.token_ids,
            assets_->model_config.blank_token_id),
        "en",
    };
    const int64_t samples_per_step =
        static_cast<int64_t>(assets_->feature_config.hop_length * assets_->model_config.encoder.subsampling_factor);
    const double window_stride =
        std::round(
            (static_cast<double>(assets_->feature_config.hop_length) / static_cast<double>(assets_->feature_config.sample_rate)) *
            1000000.0) /
        1000000.0;
    const auto char_offsets = build_char_offsets(
        *assets_,
        streaming_state_.merged_decoded,
        window_stride,
        assets_->model_config.encoder.subsampling_factor);
    const auto word_offsets = build_word_offsets(char_offsets);
    event.word_timestamps.reserve(word_offsets.size());
    for (const auto & word : word_offsets) {
        event.word_timestamps.push_back(runtime::WordTimestamp{
            runtime::TimeSpan{
                static_cast<int64_t>(word.start_offset) * samples_per_step,
                static_cast<int64_t>(word.end_offset) * samples_per_step,
            },
            word.word,
            0.0f,
        });
    }
    return event;
}

runtime::TaskResult ParakeetTDTSession::make_task_result_from_merged_decode() const {
    runtime::TaskResult result;
    if (streaming_state_.has_latest_result) {
        result.text_output = streaming_state_.latest_text;
        result.word_timestamps = streaming_state_.latest_word_timestamps;
        return result;
    }
    result.text_output = runtime::Transcript{
        assets_->tokenizer->decode_ids(
            streaming_state_.merged_decoded.token_ids,
            assets_->model_config.blank_token_id),
        "en",
    };
    const int64_t samples_per_step =
        static_cast<int64_t>(assets_->feature_config.hop_length * assets_->model_config.encoder.subsampling_factor);
    const double window_stride =
        std::round(
            (static_cast<double>(assets_->feature_config.hop_length) / static_cast<double>(assets_->feature_config.sample_rate)) *
            1000000.0) /
        1000000.0;
    const auto char_offsets = build_char_offsets(
        *assets_,
        streaming_state_.merged_decoded,
        window_stride,
        assets_->model_config.encoder.subsampling_factor);
    const auto word_offsets = build_word_offsets(char_offsets);
    result.word_timestamps.reserve(word_offsets.size());
    for (const auto & word : word_offsets) {
        result.word_timestamps.push_back(runtime::WordTimestamp{
            runtime::TimeSpan{
                static_cast<int64_t>(word.start_offset) * samples_per_step,
                static_cast<int64_t>(word.end_offset) * samples_per_step,
            },
            word.word,
            0.0f,
        });
    }
    return result;
}

std::optional<runtime::StreamEvent> ParakeetTDTSession::maybe_process_streaming_window(bool final_chunk) {
    ensure_streaming_configured();
    if (!final_chunk) {
        const int64_t needed = streaming_state_.next_chunk_start_sample +
            streaming_state_.context.chunk_samples +
            streaming_state_.context.right_samples;
        if (static_cast<int64_t>(stream_samples_.size()) < needed) {
            return std::nullopt;
        }
    } else if (streaming_state_.next_chunk_start_sample >= static_cast<int64_t>(stream_samples_.size())) {
        return std::nullopt;
    }

    const int64_t audio_size = static_cast<int64_t>(stream_samples_.size());
    const int64_t chunk_start = streaming_state_.next_chunk_start_sample;
    const int64_t chunk_end = std::min(chunk_start + streaming_state_.context.chunk_samples, audio_size);
    if (chunk_end <= chunk_start) {
        return std::nullopt;
    }
    streaming_state_.has_latest_result = false;
    const int64_t effective_left_samples = streaming_state_.context.left_samples;
    const int64_t effective_right_samples = streaming_state_.context.right_samples;
    const int64_t desired_window_start = chunk_start - effective_left_samples;
    const int64_t desired_window_end = chunk_start + streaming_state_.context.chunk_samples + effective_right_samples;
    const int64_t actual_window_start = std::max<int64_t>(0, desired_window_start);
    const int64_t actual_window_end = std::min(desired_window_end, audio_size);
    const int64_t buffered_valid_samples = std::max<int64_t>(0, actual_window_end - actual_window_start);

    auto & buffered = frontend_audio_batch_[0];
    buffered.sample_rate = stream_sample_rate_;
    buffered.channels = stream_channels_;
    buffered.samples.assign(static_cast<size_t>(streaming_state_.context.window_samples), 0.0f);
    if (buffered_valid_samples > 0) {
        std::copy(
            stream_samples_.begin() + static_cast<std::ptrdiff_t>(actual_window_start),
            stream_samples_.begin() + static_cast<std::ptrdiff_t>(actual_window_end),
            buffered.samples.begin());
    }
    frontend_audio_lengths_override_.assign(1, buffered_valid_samples);

    const auto frontend_start = std::chrono::steady_clock::now();
    compute_parakeet_frontend(
        frontend_audio_batch_,
        &frontend_audio_lengths_override_,
        assets_->feature_config,
        execution_context(),
        frontend_batch_,
        frontend_scratch_);
    const auto frontend_end = std::chrono::steady_clock::now();
    streaming_state_.timings.frontend_ms +=
        engine::debug::elapsed_ms(frontend_start, frontend_end);
    const auto pre_encode_start = std::chrono::steady_clock::now();
    compute_parakeet_pre_encode(
        frontend_batch_,
        *assets_,
        *weights_,
        execution_context(),
        pre_encode_batch_,
        streaming_pre_encode_scratch_);
    const auto pre_encode_end = std::chrono::steady_clock::now();
    streaming_state_.timings.pre_encode_ms +=
        engine::debug::elapsed_ms(pre_encode_start, pre_encode_end);
    if (pre_encode_batch_.frames != streaming_state_.context.window_frames) {
        throw std::runtime_error("Streaming pre-encode frames drifted from fixed streaming window frames");
    }
    const auto encoder_start = std::chrono::steady_clock::now();
    const auto & encoder_projected_full = run_streaming_chunk_graph_encoder(
        pre_encode_batch_,
        streaming_state_.context.left_frames,
        streaming_state_.context.chunk_frames,
        streaming_state_.context.right_frames);
    const auto encoder_end = std::chrono::steady_clock::now();
    streaming_state_.timings.encoder_ms +=
        engine::debug::elapsed_ms(encoder_start, encoder_end);

    const int64_t actual_left_samples = chunk_start - actual_window_start;
    const int64_t left_frames = actual_left_samples / streaming_state_.context.encoder_frame_to_samples;
    const int64_t requested_chunk_frames = streaming_state_.context.chunk_frames;
    const int64_t actual_valid_frames = pre_encode_batch_.valid_frames;
    const bool last_stream_chunk = final_chunk && actual_window_end >= audio_size;
    const int64_t chunk_frames = last_stream_chunk
        ? std::max<int64_t>(0, actual_valid_frames - left_frames)
        : std::min<int64_t>(requested_chunk_frames, std::max<int64_t>(0, actual_valid_frames - left_frames));
    if (chunk_frames <= 0) {
        streaming_state_.next_chunk_start_sample += streaming_state_.context.chunk_samples;
        return std::nullopt;
    }
    const int64_t hidden = assets_->model_config.decoder_hidden_size;
    std::vector<float> encoder_projected_chunk(
        static_cast<size_t>(chunk_frames * hidden),
        0.0f);
    for (int64_t t = 0; t < chunk_frames; ++t) {
        const float * src = encoder_projected_full.data() + static_cast<size_t>((left_frames + t) * hidden);
        float * dst = encoder_projected_chunk.data() + static_cast<size_t>(t * hidden);
        std::copy_n(src, static_cast<size_t>(hidden), dst);
    }

    const auto decoder_start = std::chrono::steady_clock::now();
    auto decoded = shared_decoder_->run_streaming_chunk(
        execution_context(),
        encoder_projected_chunk,
        chunk_frames,
        decoder_algorithm_,
        streaming_state_.decoder_state);
    const auto decoder_end = std::chrono::steady_clock::now();
    streaming_state_.timings.decoder_ms +=
        engine::debug::elapsed_ms(decoder_start, decoder_end);
    for (auto & timestamp : decoded.token_timestamps) {
        timestamp += static_cast<int32_t>(streaming_state_.decoded_frame_offset);
    }
    streaming_state_.decoded_frame_offset += chunk_frames;
    streaming_state_.merged_decoded.token_ids.insert(
        streaming_state_.merged_decoded.token_ids.end(),
        decoded.token_ids.begin(),
        decoded.token_ids.end());
    streaming_state_.merged_decoded.token_timestamps.insert(
        streaming_state_.merged_decoded.token_timestamps.end(),
        decoded.token_timestamps.begin(),
        decoded.token_timestamps.end());
    streaming_state_.merged_decoded.token_durations.insert(
        streaming_state_.merged_decoded.token_durations.end(),
        decoded.token_durations.begin(),
        decoded.token_durations.end());
    streaming_state_.merged_decoded.text = assets_->tokenizer->decode_ids(
        streaming_state_.merged_decoded.token_ids,
        assets_->model_config.blank_token_id);
    if (last_stream_chunk) {
        streaming_state_.next_chunk_start_sample = audio_size;
    } else {
        streaming_state_.next_chunk_start_sample += streaming_state_.context.chunk_samples;
    }
    streaming_state_.timings.chunk_windows += 1;
    return make_stream_event_from_merged_decode();
}

runtime::TaskResult ParakeetTDTSession::run(const runtime::TaskRequest & request) {
    require_prepared("Parakeet run()");
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Parakeet TDT offline run requires RunMode::Offline");
    }
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("Parakeet TDT offline inference requires audio_input");
    }

    return run_one_shot_request(*request.audio_input);
}

void ParakeetTDTSession::reset() {
    require_prepared("Parakeet reset()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Parakeet TDT streaming reset requires RunMode::Streaming");
    }
    stream_sample_rate_ = 0;
    stream_channels_ = 0;
    next_stream_sample_ = 0;
    stream_samples_.clear();
    frontend_audio_lengths_override_.clear();
    frontend_audio_batch_[0] = runtime::AudioBuffer{};
    streaming_state_ = StreamingState{};
}

runtime::StreamEvent ParakeetTDTSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    require_prepared("Parakeet process_audio_chunk()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Parakeet TDT streaming chunk processing requires RunMode::Streaming");
    }
    if (chunk.sample_rate <= 0) {
        throw std::runtime_error("Parakeet TDT streaming requires a positive sample rate");
    }
    if (chunk.channels != 1) {
        throw std::runtime_error("Parakeet TDT streaming currently requires mono audio");
    }
    if (stream_sample_rate_ == 0) {
        stream_sample_rate_ = chunk.sample_rate;
        stream_channels_ = chunk.channels;
        next_stream_sample_ = chunk.start_sample;
        stream_samples_.reserve(stream_samples_.size() + std::max<size_t>(chunk.samples.size(), 16384));
        if (debug::timing_log_enabled() && !streaming_state_.timings.wall_started) {
            streaming_state_.timings.wall_start = std::chrono::steady_clock::now();
            streaming_state_.timings.wall_started = true;
        }
    }
    if (chunk.sample_rate != stream_sample_rate_ || chunk.channels != stream_channels_) {
        throw std::runtime_error("Parakeet TDT streaming chunks must keep the same sample_rate/channels");
    }
    if (chunk.start_sample != next_stream_sample_) {
        throw std::runtime_error("Parakeet TDT streaming chunks must be contiguous");
    }
    const size_t required_samples = stream_samples_.size() + chunk.samples.size();
    if (required_samples > stream_samples_.capacity()) {
        stream_samples_.reserve(std::max(required_samples, stream_samples_.capacity() * 2));
    }
    stream_samples_.insert(stream_samples_.end(), chunk.samples.begin(), chunk.samples.end());
    next_stream_sample_ += static_cast<int64_t>(chunk.samples.size());
    auto event = maybe_process_streaming_window(false);
    if (event.has_value()) {
        return *event;
    }
    return runtime::StreamEvent{};
}

runtime::TaskResult ParakeetTDTSession::finalize() {
    require_prepared("Parakeet finalize()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Parakeet TDT streaming finalize requires RunMode::Streaming");
    }
    return flush_streaming_request();
}

runtime::TaskResult ParakeetTDTSession::run_one_shot_request(const runtime::AudioBuffer & audio) {
    const BufferedEncoderVariant encoder_variant =
        requests_long_context_buffered_encoder(options_) ? BufferedEncoderVariant::LongContext : BufferedEncoderVariant::FullContext;
    const ParakeetTraceContractMode trace_mode =
        encoder_variant == BufferedEncoderVariant::LongContext
        ? ParakeetTraceContractMode::Longform
        : ParakeetTraceContractMode::Offline;
    return transcribe_buffered_audio(audio, encoder_variant, trace_mode, false);
}

runtime::TaskResult ParakeetTDTSession::flush_streaming_request() {
    while (true) {
        const auto event = maybe_process_streaming_window(true);
        if (!event.has_value()) {
            break;
        }
    }
    const auto timestamps_start = std::chrono::steady_clock::now();
    auto result = make_task_result_from_merged_decode();
    const auto timestamps_end = std::chrono::steady_clock::now();
    streaming_state_.timings.timestamps_ms +=
        engine::debug::elapsed_ms(timestamps_start, timestamps_end);

    if (debug::timing_log_enabled()) {
        const auto wall_end = std::chrono::steady_clock::now();
        const double total_wall_ms = streaming_state_.timings.wall_started
            ? std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(wall_end - streaming_state_.timings.wall_start)
                  .count()
            : 0.0;
        debug::timing_log_scalar("parakeet.frontend_ms", streaming_state_.timings.frontend_ms);
        debug::timing_log_scalar("parakeet.pre_encode_ms", streaming_state_.timings.pre_encode_ms);
        debug::timing_log_scalar("parakeet.encoder_ms", streaming_state_.timings.encoder_ms);
        debug::timing_log_scalar(
            "parakeet.streaming_encoder.graph.ensure_ms",
            streaming_state_.timings.encoder_ensure_graph_ms);
        debug::timing_log_scalar(
            "parakeet.streaming_encoder.graph.build_ms",
            streaming_state_.timings.encoder_graph_build_ms);
        debug::timing_log_scalar(
            "parakeet.streaming_encoder.input_write_ms",
            streaming_state_.timings.encoder_input_write_ms);
        debug::timing_log_scalar(
            "parakeet.streaming_encoder.pos_emb_ms",
            streaming_state_.timings.encoder_pos_emb_ms);
        debug::timing_log_scalar(
            "parakeet.streaming_encoder.attention_mask_ms",
            streaming_state_.timings.encoder_attention_mask_ms);
        debug::timing_log_scalar(
            "parakeet.streaming_encoder.keep_mask_ms",
            streaming_state_.timings.encoder_keep_mask_ms);
        debug::timing_log_scalar(
            "parakeet.streaming_encoder.graph.compute_ms",
            streaming_state_.timings.encoder_graph_compute_ms);
        debug::timing_log_scalar(
            "parakeet.streaming_encoder.hidden_read_ms",
            streaming_state_.timings.encoder_hidden_read_ms);
        debug::timing_log_scalar(
            "parakeet.streaming_encoder.projected_read_ms",
            streaming_state_.timings.encoder_projected_read_ms);
        debug::timing_log_scalar(
            "parakeet.streaming_encoder.output_zero_ms",
            streaming_state_.timings.encoder_output_zero_ms);
        debug::timing_log_scalar(
            "parakeet.streaming_encoder.graph.rebuilds",
            streaming_state_.timings.encoder_graph_rebuilds);
        debug::timing_log_scalar("parakeet.decoder_ms", streaming_state_.timings.decoder_ms);
        debug::timing_log_scalar("parakeet.timestamps_ms", streaming_state_.timings.timestamps_ms);
        debug::timing_log_scalar(
            "session.wall_ms",
            total_wall_ms);
        debug::trace_log_scalar("parakeet.streaming_chunk_windows",
            streaming_state_.timings.chunk_windows);
    }
    return result;
}

ParakeetTDTSession::BufferedChunkDecode ParakeetTDTSession::decode_buffered_audio_chunk(
    const runtime::AudioBuffer & audio,
    const BufferedCapacityContract & contract,
    size_t contract_index,
    BufferedEncoderVariant encoder_variant,
    ParakeetTraceContractMode trace_mode,
    bool) {
    const int64_t audio_samples = static_cast<int64_t>(audio.samples.size());
    frontend_audio_batch_[0] = audio;
    frontend_audio_batch_[0].samples.resize(static_cast<size_t>(contract.audio_samples), 0.0f);
    frontend_audio_lengths_override_.assign(1, audio_samples);

    const auto frontend_start = std::chrono::steady_clock::now();
    compute_parakeet_frontend(
        frontend_audio_batch_,
        &frontend_audio_lengths_override_,
        assets_->feature_config,
        execution_context(),
        frontend_batch_,
        frontend_scratch_);
    const auto frontend_end = std::chrono::steady_clock::now();
    const auto & frontend = frontend_batch_;
    const int64_t expected_frontend_frames =
        frontend_frames_for_audio_samples(assets_->feature_config, contract.audio_samples);
    if (frontend.frames != expected_frontend_frames) {
        throw std::runtime_error("Parakeet frontend frames drifted from frozen buffered chunk shape");
    }

    const auto pre_encode_start = std::chrono::steady_clock::now();
    if (encoder_variant == BufferedEncoderVariant::LongContext) {
        const bool pre_encode_graph_was_missing = long_context_tiers_.at(contract_index).pre_encode_scratch.graph == nullptr;
        compute_parakeet_pre_encode(
            frontend,
            *assets_,
            *weights_,
            execution_context(),
            pre_encode_batch_,
            long_context_tiers_.at(contract_index).pre_encode_scratch);
        if (pre_encode_graph_was_missing &&
            long_context_tiers_.at(contract_index).pre_encode_scratch.graph != nullptr) {
            ++long_context_pre_encode_graphs_built_;
        }
        latest_long_context_tier_index_ = contract_index;
    } else {
        const bool pre_encode_graph_was_missing = full_context_tiers_.at(contract_index).pre_encode_scratch.graph == nullptr;
        compute_parakeet_pre_encode(
            frontend,
            *assets_,
            *weights_,
            execution_context(),
            pre_encode_batch_,
            full_context_tiers_.at(contract_index).pre_encode_scratch);
        if (pre_encode_graph_was_missing &&
            full_context_tiers_.at(contract_index).pre_encode_scratch.graph != nullptr) {
            ++full_context_pre_encode_graphs_built_;
        }
        latest_full_context_tier_index_ = contract_index;
    }
    const auto pre_encode_end = std::chrono::steady_clock::now();
    const auto & pre_encode = pre_encode_batch_;
    if (pre_encode.frames != contract.pre_encode_frames) {
        throw std::runtime_error("Parakeet pre-encode frames drifted from frozen buffered chunk shape");
    }

    const auto encoder_start = std::chrono::steady_clock::now();
    initialize_shared_encoder_cache(
        pre_encode,
        encoder_variant,
        encoder_variant == BufferedEncoderVariant::LongContext
            ? nullptr
            : &full_context_tiers_.at(contract_index).encoder);
    const bool encoder_graph_was_missing = encoder_variant == BufferedEncoderVariant::LongContext
        ? long_context_tiers_.at(contract_index).encoder.graph == nullptr
        : full_context_tiers_.at(contract_index).encoder.graph == nullptr;
    const auto & encoder_projected = encoder_variant == BufferedEncoderVariant::LongContext
        ? run_long_context_graph_encoder(pre_encode, long_context_tiers_.at(contract_index).encoder)
        : run_offline_graph_encoder(pre_encode, full_context_tiers_.at(contract_index).encoder);
    if (encoder_graph_was_missing) {
        if (encoder_variant == BufferedEncoderVariant::LongContext &&
            long_context_tiers_.at(contract_index).encoder.graph != nullptr) {
            ++long_context_encoder_graphs_built_;
        } else if (encoder_variant == BufferedEncoderVariant::FullContext &&
                   full_context_tiers_.at(contract_index).encoder.graph != nullptr) {
            ++full_context_encoder_graphs_built_;
        }
    }
    const auto encoder_end = std::chrono::steady_clock::now();
    const auto decoder_start = std::chrono::steady_clock::now();
    BufferedChunkDecode out;
    out.decoded = run_shared_decoder(
        encoder_projected,
        pre_encode.valid_frames,
        decoder_algorithm_,
        trace_mode);
    const auto decoder_end = std::chrono::steady_clock::now();

    out.timings.frontend_ms =
        engine::debug::elapsed_ms(frontend_start, frontend_end);
    out.timings.pre_encode_ms =
        engine::debug::elapsed_ms(pre_encode_start, pre_encode_end);
    out.timings.encoder_ms =
        engine::debug::elapsed_ms(encoder_start, encoder_end);
    out.timings.decoder_ms =
        engine::debug::elapsed_ms(decoder_start, decoder_end);
    out.timings.buffered_capacity_frames = pre_encode.frames;
    out.timings.buffered_valid_frames = pre_encode.valid_frames;
    return out;
}

runtime::TaskResult ParakeetTDTSession::transcribe_buffered_audio(
    const runtime::AudioBuffer & audio,
    BufferedEncoderVariant encoder_variant,
    ParakeetTraceContractMode trace_mode,
    bool emit_streaming_trace) {
    const bool trace_enabled = debug::trace_log_enabled();
    const auto wall_start = std::chrono::steady_clock::now();

    BufferedDecodeTimings decode_timings;
    ParakeetDecodeResult decoded;
    const int64_t request_size = static_cast<int64_t>(audio.samples.size());
    auto adapter = make_buffered_capacity_adapter(encoder_variant);
    const int64_t selected_capacity =
        buffered_capacity_controller(encoder_variant).ensure_prepared(adapter, request_size);
    const auto [contract, contract_index] = find_buffered_capacity_contract(
        selected_capacity,
        encoder_variant);
    auto chunk = decode_buffered_audio_chunk(
        audio,
        *contract,
        contract_index,
        encoder_variant,
        trace_mode,
        emit_streaming_trace);
    decoded = std::move(chunk.decoded);
    decode_timings = chunk.timings;

    runtime::TaskResult result;
    result.text_output = runtime::Transcript{decoded.text, "en"};
    const auto timestamps_start = std::chrono::steady_clock::now();
    const int64_t samples_per_step =
        static_cast<int64_t>(assets_->feature_config.hop_length * assets_->model_config.encoder.subsampling_factor);
    const double window_stride =
        std::round(
            (static_cast<double>(assets_->feature_config.hop_length) / static_cast<double>(assets_->feature_config.sample_rate)) *
            1000000.0) /
        1000000.0;
    const auto char_offsets = build_char_offsets(
        *assets_,
        decoded,
        window_stride,
        assets_->model_config.encoder.subsampling_factor);
    const auto word_offsets = build_word_offsets(char_offsets);
    const auto timestamps_end = std::chrono::steady_clock::now();
    result.word_timestamps.reserve(word_offsets.size());
    for (const auto & word : word_offsets) {
        result.word_timestamps.push_back(runtime::WordTimestamp{
            runtime::TimeSpan{
                static_cast<int64_t>(word.start_offset) * samples_per_step,
                static_cast<int64_t>(word.end_offset) * samples_per_step,
            },
            word.word,
            0.0f,
        });
    }
    if (trace_enabled) {
        debug::trace_log_scalar("parakeet.trace_mode", trace_mode_label(trace_mode));
        debug::trace_log_scalar(
            "parakeet.decoder_algorithm",
            decoder_algorithm_label(decoder_algorithm_));
    }
    if (debug::timing_log_enabled()) {
        const auto wall_end = std::chrono::steady_clock::now();
        const double timestamps_ms =
            engine::debug::elapsed_ms(timestamps_start, timestamps_end);
        const double total_wall_ms =
            engine::debug::elapsed_ms(wall_start, wall_end);
        debug::timing_log_scalar("parakeet.frontend_ms", decode_timings.frontend_ms);
        debug::timing_log_scalar("parakeet.pre_encode_ms", decode_timings.pre_encode_ms);
        debug::timing_log_scalar("parakeet.encoder_ms", decode_timings.encoder_ms);
        debug::timing_log_scalar("parakeet.decoder_ms", decode_timings.decoder_ms);
        debug::trace_log_scalar("parakeet.buffered_capacity_frames",
            decode_timings.buffered_capacity_frames);
        debug::trace_log_scalar("parakeet.buffered_valid_frames",
            decode_timings.buffered_valid_frames);
        debug::timing_log_scalar("parakeet.timestamps_ms", timestamps_ms);
        debug::timing_log_scalar("session.wall_ms", total_wall_ms);
    }
    return result;
}

}  // namespace engine::models::parakeet_tdt
