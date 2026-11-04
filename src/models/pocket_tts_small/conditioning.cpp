#include "engine/models/pocket_tts_small/conditioning.h"
#include "engine/models/pocket_tts_small/assets.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/waveform_ops.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/core/backend.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace engine::models::pocket_tts_small {

namespace {

template <typename BuildFn, typename InitFn, typename ReadFn>
auto run_graph_with_backend(
    const char * label,
    ggml_backend_t backend,
    core::BackendType backend_type,
    int threads,
    size_t graph_context_bytes,
    BuildFn build_fn,
    InitFn init_fn,
    ReadFn read_fn)
    -> decltype(read_fn(std::declval<ggml_context *>())) {
    ggml_context * ggml_ctx = ggml_init({graph_context_bytes, nullptr, true});
    if (ggml_ctx == nullptr) {
        throw std::runtime_error("Failed to initialize ggml context");
    }
    core::ModuleBuildContext ctx{ggml_ctx, "pocket_tts_small", backend_type};
    auto output = build_fn(ctx);
    ggml_cgraph * graph = ggml_new_graph_custom(ggml_ctx, 32768, false);
    ggml_build_forward_expand(graph, output.tensor);
    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (gallocr == nullptr ||
        !ggml_gallocr_reserve(gallocr, graph) ||
        !ggml_gallocr_alloc_graph(gallocr, graph)) {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
        }
        ggml_free(ggml_ctx);
        throw std::runtime_error(std::string(label) + " graph allocation failed");
    }
    init_fn();
    core::set_backend_threads(backend, threads);
    const ggml_status status = engine::core::compute_backend_graph(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        core::release_backend_graph_resources(backend, graph, true);
        ggml_gallocr_free(gallocr);
        ggml_free(ggml_ctx);
        throw std::runtime_error(std::string(label) + " compute failed: " + ggml_status_to_string(status));
    }
    auto result = read_fn(ggml_ctx);
    core::release_backend_graph_resources(backend, graph, true);
    ggml_gallocr_free(gallocr);
    ggml_free(ggml_ctx);
    return result;
}

struct CodePoint {
    uint32_t value = 0;
    size_t length = 0;
};

// Decodes the UTF-8 sequence starting at `start`. A malformed sequence is returned as a
// single opaque byte, so callers never read past it and never treat it as a letter.
CodePoint decode_utf8_at(const std::string & text, size_t start) {
    const auto lead = static_cast<unsigned char>(text[start]);
    if (lead < 0x80) {
        return {lead, 1};
    }
    size_t length = 0;
    uint32_t value = 0;
    if ((lead & 0xE0) == 0xC0) {
        length = 2;
        value = lead & 0x1F;
    } else if ((lead & 0xF0) == 0xE0) {
        length = 3;
        value = lead & 0x0F;
    } else if ((lead & 0xF8) == 0xF0) {
        length = 4;
        value = lead & 0x07;
    } else {
        return {0xFFFFFFFFU, 1};
    }
    if (start + length > text.size()) {
        return {0xFFFFFFFFU, 1};
    }
    for (size_t i = 1; i < length; ++i) {
        const auto next = static_cast<unsigned char>(text[start + i]);
        if ((next & 0xC0) != 0x80) {
            return {0xFFFFFFFFU, 1};
        }
        value = (value << 6) | (next & 0x3F);
    }
    return {value, length};
}

std::string encode_utf8(uint32_t value) {
    std::string out;
    if (value < 0x80) {
        out.push_back(static_cast<char>(value));
    } else if (value < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (value >> 6)));
        out.push_back(static_cast<char>(0x80 | (value & 0x3F)));
    } else if (value < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (value >> 12)));
        out.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (value & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (value >> 18)));
        out.push_back(static_cast<char>(0x80 | ((value >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (value & 0x3F)));
    }
    return out;
}

// Uppercase of a lowercase Latin letter, or 0 when it has no single-code-point uppercase:
// the reference capitalizes the first letter with str.upper(), which turns "état" into
// "État" where a byte-wise toupper() left it lowercase.
uint32_t latin_uppercase(uint32_t value) {
    if (value >= 'a' && value <= 'z') {
        return value - 0x20;
    }
    if (value >= 0x00E0 && value <= 0x00FE && value != 0x00F7) {
        return value - 0x20;
    }
    if (value == 0x00FF) {
        return 0x0178;
    }
    if (value == 0x0131) {
        return 'I';
    }
    const bool even_upper = (value >= 0x0100 && value <= 0x012F) || (value >= 0x0132 && value <= 0x0137) ||
        (value >= 0x014A && value <= 0x0177);
    if (even_upper && (value & 1U) == 1U) {
        return value - 1;
    }
    const bool odd_upper = (value >= 0x0139 && value <= 0x0148) || (value >= 0x0179 && value <= 0x017E);
    if (odd_upper && (value & 1U) == 0U) {
        return value - 1;
    }
    return 0;
}

std::string strip_ascii_whitespace(std::string text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
        text.erase(text.begin());
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
        text.pop_back();
    }
    return text;
}

std::string prepare_pocket_tts_small_text(
    std::string text,
    bool pad_with_spaces_for_short_inputs,
    bool remove_semicolons,
    const std::vector<std::pair<std::string, std::string>> & replacements) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
        text.erase(text.begin());
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
        text.pop_back();
    }
    for (const auto & [from, to] : replacements) {
        size_t offset = 0;
        while ((offset = text.find(from, offset)) != std::string::npos) {
            text.replace(offset, from.size(), to);
            offset += to.size();
        }
    }
    if (!replacements.empty()) {
        std::string normalized;
        bool space = false;
        for (const unsigned char ch : text) {
            if (std::isspace(ch)) {
                space = !normalized.empty();
            } else {
                if (space) normalized.push_back(' ');
                normalized.push_back(static_cast<char>(ch));
                space = false;
            }
        }
        text = std::move(normalized);
        for (size_t i = 0; i < text.size(); ++i) {
            if (text[i] != '.' && text[i] != '!' && text[i] != '?') continue;
            size_t next = text.find_first_not_of(' ', i + 1);
            if (next != std::string::npos && (text[next] == ',' || text[next] == ';' || text[next] == ':'))
                text.erase(i + 1, next - i);
        }
    }
    if (text.empty()) {
        throw std::runtime_error("PocketTTS text prompt cannot be empty");
    }

    for (char & ch : text) {
        if (ch == '\n' || ch == '\r') {
            ch = ' ';
        }
    }
    for (size_t i = 1; i < text.size();) {
        if (text[i] == ' ' && text[i - 1] == ' ') {
            text.erase(text.begin() + static_cast<ptrdiff_t>(i));
        } else {
            ++i;
        }
    }
    if (remove_semicolons) {
        std::replace(text.begin(), text.end(), ';', ',');
    }
    const CodePoint first = decode_utf8_at(text, 0);
    if (const uint32_t upper = latin_uppercase(first.value); upper != 0) {
        text.replace(0, first.length, encode_utf8(upper));
    }
    // Match upstream terminal punctuation, including trailing quotes/brackets.
    size_t core_end = text.size();
    uint32_t terminal = 0;
    while (core_end > 0) {
        size_t start = core_end - 1;
        while (start > 0 && (static_cast<unsigned char>(text[start]) & 0xC0) == 0x80) --start;
        terminal = decode_utf8_at(text, start).value;
        if (terminal != '"' && terminal != '\'' && terminal != ')' && terminal != ']' &&
            terminal != 0x201D && terminal != 0x2019 && terminal != 0x00BB && terminal != ' ') break;
        core_end = start;
    }
    if (core_end > 0 && terminal != '.' && terminal != '!' && terminal != '?' && terminal != 0x2026) {
        size_t weak_end = core_end;
        while (weak_end > 0) {
            size_t start = weak_end - 1;
            while (start > 0 && (static_cast<unsigned char>(text[start]) & 0xC0) == 0x80) --start;
            const auto value = decode_utf8_at(text, start).value;
            if (value != ',' && value != ';' && value != ':' && value != '-' &&
                value != 0x2013 && value != 0x2014 && value != ' ') break;
            weak_end = start;
        }
        if (weak_end < core_end) {
            text = text.substr(0, weak_end) + "." + strip_ascii_whitespace(text.substr(core_end));
        } else {
            text.push_back('.');
        }
    }
    if (pad_with_spaces_for_short_inputs) {
        int words = 0;
        bool in_word = false;
        for (const char ch : text) {
            if (std::isspace(static_cast<unsigned char>(ch))) {
                if (in_word) {
                    ++words;
                    in_word = false;
                }
            } else {
                in_word = true;
            }
        }
        if (in_word) {
            ++words;
        }
        if (words < 5) {
            text = "        " + text;
        }
    }
    return text;
}

std::vector<float> gather_token_embeddings(
    const assets::TensorDataF32 & embedding_table,
    const std::vector<int32_t> & tokens,
    int64_t hidden_size) {
    if (embedding_table.shape.rank != 2 || embedding_table.shape.dims[1] != hidden_size) {
        throw std::runtime_error("PocketTTS embedding table must have shape [vocab, hidden_size]");
    }
    const int64_t vocab = embedding_table.shape.dims[0];
    const int64_t dim = embedding_table.shape.dims[1];

    std::vector<float> output(static_cast<size_t>(tokens.size() * dim));
    for (size_t token_index = 0; token_index < tokens.size(); ++token_index) {
        const int32_t token = tokens[token_index];
        if (token < 0 || token >= vocab) {
            throw std::runtime_error("PocketTTS token id is out of range for embedding table");
        }
        const size_t src_offset = static_cast<size_t>(token) * static_cast<size_t>(dim);
        const size_t dst_offset = token_index * static_cast<size_t>(dim);
        std::copy_n(
            embedding_table.values.begin() + static_cast<ptrdiff_t>(src_offset),
            static_cast<size_t>(dim),
            output.begin() + static_cast<ptrdiff_t>(dst_offset));
    }
    return output;
}

// Token ids of `symbols` minus the first one, exactly as the reference builds its boundary
// sets: SentencePiece prefixes the string with its word marker, and only the ids that
// follow it stand for the punctuation.
std::unordered_set<int32_t> boundary_token_ids(
    const std::vector<tokenizers::SentencePiecePiece> & pieces,
    const std::string & symbols) {
    const std::vector<int32_t> ids = tokenizers::tokenize_sentencepiece(pieces, symbols);
    std::unordered_set<int32_t> out;
    for (size_t i = 1; i < ids.size(); ++i) {
        out.insert(ids[i]);
    }
    return out;
}

std::string decode_range(
    const std::vector<tokenizers::SentencePiecePiece> & pieces,
    const std::vector<int32_t> & tokens,
    size_t begin,
    size_t end) {
    return tokenizers::decode_sentencepiece(
        pieces,
        std::vector<int32_t>(
            tokens.begin() + static_cast<ptrdiff_t>(begin),
            tokens.begin() + static_cast<ptrdiff_t>(end)));
}

// "3.5": a period between two digits does not end a sentence.
bool is_decimal_period_boundary(
    const std::vector<tokenizers::SentencePiecePiece> & pieces,
    const std::vector<int32_t> & tokens,
    size_t segment_start) {
    const std::string prefix = decode_range(pieces, tokens, 0, segment_start);
    const std::string suffix = decode_range(pieces, tokens, segment_start, tokens.size());
    return prefix.size() >= 2 && prefix.back() == '.' &&
        std::isdigit(static_cast<unsigned char>(prefix[prefix.size() - 2])) != 0 && !suffix.empty() &&
        std::isdigit(static_cast<unsigned char>(suffix.front())) != 0;
}

// A segment starts at the first token that is not a boundary token after a run of boundary
// tokens. Returns [0, starts..., size], so consecutive pairs are the segments.
std::vector<size_t> find_boundary_indices(
    const std::vector<int32_t> & tokens,
    const std::unordered_set<int32_t> & boundary_ids,
    const std::vector<tokenizers::SentencePiecePiece> * pieces_for_decimal_check) {
    std::vector<size_t> indices{0};
    bool previous_was_boundary = false;
    for (size_t index = 0; index < tokens.size(); ++index) {
        if (boundary_ids.count(tokens[index]) != 0) {
            previous_was_boundary = true;
            continue;
        }
        if (previous_was_boundary &&
            !(pieces_for_decimal_check != nullptr &&
              is_decimal_period_boundary(*pieces_for_decimal_check, tokens, index))) {
            indices.push_back(index);
        }
        previous_was_boundary = false;
    }
    indices.push_back(tokens.size());
    return indices;
}

struct TextSegment {
    int64_t token_count = 0;
    std::string text;
};

std::vector<TextSegment> segments_from_boundaries(
    const std::vector<tokenizers::SentencePiecePiece> & pieces,
    const std::vector<int32_t> & tokens,
    const std::vector<size_t> & boundaries) {
    std::vector<TextSegment> segments;
    for (size_t i = 0; i + 1 < boundaries.size(); ++i) {
        segments.push_back(TextSegment{
            static_cast<int64_t>(boundaries[i + 1] - boundaries[i]),
            decode_range(pieces, tokens, boundaries[i], boundaries[i + 1]),
        });
    }
    return segments;
}

}  // namespace

TextConditioner::TextConditioner(TextConditionerConfig config) : config_(config) {}

TextConditioningResult TextConditioner::prepare(
    const models::pocket_tts_small::Assets & manifest,
    const models::pocket_tts_small::HostWeights & weights,
    const std::string & text,
    ggml_backend_t backend,
    int threads) const {
    TextConditioningResult result;
    const double prepare_ms = engine::debug::measure_ms([&]() {
        result.prepared_text = prepare_pocket_tts_small_text(
            text,
            manifest.model_config.pad_with_spaces_for_short_inputs,
            manifest.model_config.remove_semicolons,
            manifest.model_config.replace_characters);
        if (manifest.tokenizer_pieces.empty()) {
            throw std::runtime_error("PocketTTS tokenizer pieces are not loaded");
        }
        result.tokens = tokenizers::tokenize_sentencepiece(manifest.tokenizer_pieces, result.prepared_text);
        if (weights.conditioner_projection && !projected_embeddings_) {
            if (!backend) throw std::runtime_error("Factorized PocketTTS embeddings require a session backend");
            const auto & table = weights.conditioner_embedding_table;
            const auto & projection = *weights.conditioner_projection;
            core::TensorValue input, matrix, output;
            const auto values = run_graph_with_backend(
                "PocketTTS text projection", backend, core::backend_type(backend), threads,
                ggml_graph_overhead_custom(32768, false) + 16 * ggml_tensor_overhead(),
                [&](core::ModuleBuildContext & ctx) {
                    input = core::make_tensor(ctx, GGML_TYPE_F32, table.shape);
                    matrix = core::make_tensor(ctx, GGML_TYPE_F32, projection.shape);
                    output = modules::LinearModule({table.shape.dims[1], config_.hidden_size, false}).build(
                        ctx, input, {matrix, std::nullopt});
                    return output;
                },
                [&]() { core::write_tensor_f32(input, table.values); core::write_tensor_f32(matrix, projection.values); },
                [&](ggml_context *) { return core::read_tensor_f32(output.tensor); });
            projected_embeddings_ = assets::TensorDataF32{
                core::TensorShape::from_dims({table.shape.dims[0], config_.hidden_size}), values};
        }
        const auto & embedding_table = projected_embeddings_ ? *projected_embeddings_ : weights.conditioner_embedding_table;
        result.text_embeddings = gather_token_embeddings(embedding_table, result.tokens, config_.hidden_size);
    });
    engine::debug::timing_log_scalar("pocket_tts_small.text.prepare_ms", prepare_ms);
    return result;
}

std::vector<std::string> TextConditioner::split_into_sentence_chunks(
    const models::pocket_tts_small::Assets & manifest,
    const std::string & text,
    int64_t max_tokens) const {
    if (manifest.tokenizer_pieces.empty()) {
        throw std::runtime_error("PocketTTS tokenizer pieces are not loaded");
    }
    if (max_tokens <= 0) {
        throw std::runtime_error("PocketTTS max_tokens must be positive");
    }
    const auto & pieces = manifest.tokenizer_pieces;
    // Same normalization as the prompt the model will see, then stripped: the reference drops
    // the short-input padding here and adds it back chunk by chunk in prepare().
    const std::string prepared = strip_ascii_whitespace(prepare_pocket_tts_small_text(
        text,
        manifest.model_config.pad_with_spaces_for_short_inputs,
        manifest.model_config.remove_semicolons,
        manifest.model_config.replace_characters));
    const std::vector<int32_t> tokens = tokenizers::tokenize_sentencepiece(pieces, prepared);
    const std::vector<TextSegment> sentences = segments_from_boundaries(
        pieces,
        tokens,
        find_boundary_indices(tokens, boundary_token_ids(pieces, ".!...?"), &pieces));

    // A sentence longer than the budget is split again on commas, semicolons and colons,
    // which keeps the model from skipping words at the end of an overlong chunk.
    const std::unordered_set<int32_t> fallback_ids = boundary_token_ids(pieces, ",;:");
    std::vector<TextSegment> refined;
    for (const TextSegment & sentence : sentences) {
        if (sentence.token_count <= max_tokens) {
            refined.push_back(sentence);
            continue;
        }
        const std::vector<int32_t> sub_tokens =
            tokenizers::tokenize_sentencepiece(pieces, strip_ascii_whitespace(sentence.text));
        std::vector<TextSegment> parts = segments_from_boundaries(
            pieces,
            sub_tokens,
            find_boundary_indices(sub_tokens, fallback_ids, nullptr));
        if (parts.size() > 1) {
            refined.insert(refined.end(), parts.begin(), parts.end());
        } else {
            refined.push_back(sentence);
        }
    }

    std::vector<std::string> chunks;
    std::string current;
    int64_t current_tokens = 0;
    for (const TextSegment & segment : refined) {
        if (current.empty()) {
            current = segment.text;
            current_tokens = segment.token_count;
            continue;
        }
        if (current_tokens + segment.token_count > max_tokens) {
            chunks.push_back(strip_ascii_whitespace(current));
            current = segment.text;
            current_tokens = segment.token_count;
        } else {
            current += " " + segment.text;
            current_tokens += segment.token_count;
        }
    }
    if (!current.empty()) {
        chunks.push_back(strip_ascii_whitespace(current));
    }
    return chunks;
}

namespace {

std::vector<float> append_bos_voice_embedding(
    const models::pocket_tts_small::Assets & manifest,
    const models::pocket_tts_small::HostWeights & weights,
    const std::vector<float> & conditioning,
    int64_t frames) {
    const int64_t flow_hidden = manifest.model_config.flow_dim;
    if (!manifest.model_config.insert_bos_before_voice) {
        return conditioning;
    }
    if (!weights.bos_before_voice.has_value()) {
        throw std::runtime_error("flow_lm.bos_before_voice is required when insert_bos_before_voice is enabled");
    }
    const auto & bos = *weights.bos_before_voice;
    if (bos.shape.rank != 3 || bos.shape.dims[0] != 1 || bos.shape.dims[1] != 1 || bos.shape.dims[2] != flow_hidden) {
        throw std::runtime_error("flow_lm.bos_before_voice must have shape [1, 1, flow_dim]");
    }
    std::vector<float> output;
    output.reserve(static_cast<size_t>((frames + 1) * flow_hidden));
    output.insert(output.end(), bos.values.begin(), bos.values.end());
    output.insert(output.end(), conditioning.begin(), conditioning.end());
    return output;
}

MimiPromptEmbedding append_bos_voice_embedding(
    const models::pocket_tts_small::Assets & manifest,
    const models::pocket_tts_small::HostWeights & weights,
    MimiPromptEmbedding embedding) {
    const int64_t flow_hidden = manifest.model_config.flow_dim;
    embedding.values = append_bos_voice_embedding(manifest, weights, embedding.values, embedding.frames);
    embedding.frames = static_cast<int64_t>(embedding.values.size() / static_cast<size_t>(flow_hidden));
    return embedding;
}

std::vector<float> load_conditioning_audio(
    const std::filesystem::path & path,
    int target_sample_rate,
    int64_t frame_size,
    bool truncate) {
    auto resampled = audio::read_wav_f32_as_mono_linear_resampled(path, target_sample_rate);
    if (truncate) {
        const size_t max_samples = static_cast<size_t>(30 * target_sample_rate);
        if (resampled.size() > max_samples) {
            resampled.resize(max_samples);
        }
    }
    return audio::zero_pad_samples_to_multiple(resampled, frame_size);
}

std::vector<float> prepare_conditioning_audio(
    const runtime::AudioBuffer & audio_buffer,
    int target_sample_rate,
    int64_t frame_size,
    bool truncate) {
    auto resampled = audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio_buffer.samples,
        audio_buffer.sample_rate,
        audio_buffer.channels,
        target_sample_rate);
    if (truncate) {
        const size_t max_samples = static_cast<size_t>(30 * target_sample_rate);
        if (resampled.size() > max_samples) {
            resampled.resize(max_samples);
        }
    }
    return audio::zero_pad_samples_to_multiple(resampled, frame_size);
}

MimiPromptEmbedding project_speaker_latents(
    const models::pocket_tts_small::Assets & manifest,
    const SpeakerLatents & latents) {
    if (!manifest.host_weights.speaker_proj.has_value()) {
        throw std::runtime_error("This PocketTTS checkpoint cannot project speaker_wavs voice latents");
    }
    const auto & weight = *manifest.host_weights.speaker_proj;
    const int64_t out_dim = manifest.model_config.flow_dim;
    if (weight.shape.rank != 2 || weight.shape.dims[0] != out_dim || weight.shape.dims[1] != latents.channels) {
        throw std::runtime_error("speaker_wavs channels do not match the checkpoint's speaker projection");
    }
    MimiPromptEmbedding embedding;
    embedding.frames = latents.frames;
    embedding.values.resize(static_cast<size_t>(latents.frames * out_dim));
    for (int64_t frame = 0; frame < latents.frames; ++frame) {
        const float * x = latents.values.data() + frame * latents.channels;
        for (int64_t row = 0; row < out_dim; ++row) {
            const float * w = weight.values.data() + row * latents.channels;
            float sum = 0.0F;
            for (int64_t col = 0; col < latents.channels; ++col) {
                sum += w[col] * x[col];
            }
            embedding.values[static_cast<size_t>(frame * out_dim + row)] = sum;
        }
    }
    return embedding;
}

}  // namespace

VoiceConditioner::VoiceConditioner(FlowLMConfig flow_config) : flow_lm_(std::move(flow_config)) {}

VoiceConditioningResult VoiceConditioner::prepare(
    const models::pocket_tts_small::VoiceConditioningPlan & plan,
    const models::pocket_tts_small::Assets & manifest,
    const models::pocket_tts_small::BackendWeights & weights,
    ggml_backend_t backend,
    int threads,
    size_t mimi_encoder_graph_context_bytes,
    size_t flow_weights_view_context_bytes,
    size_t flow_step_graph_context_bytes) const {
    VoiceConditioningResult result;
    int64_t prompt_frames = 0;
    const double prepare_ms = engine::debug::measure_ms([&]() {
        if (plan.source != models::pocket_tts_small::VoiceSourceKind::CloneAudio) {
            if (const auto latents = models::pocket_tts_small::load_speaker_latents(plan.asset_path)) {
                auto prompt_embedding = append_bos_voice_embedding(
                    manifest,
                    weights.host,
                    project_speaker_latents(manifest, *latents));
                prompt_frames = prompt_embedding.frames;
                result = VoiceConditioningResult{
                    plan,
                    flow_lm_.prompt_state(
                        backend,
                        threads,
                        manifest,
                        weights,
                        prompt_embedding.values,
                        prompt_embedding.frames,
                        flow_lm_.make_empty_state(),
                        flow_weights_view_context_bytes,
                        flow_step_graph_context_bytes),
                };
                return;
            }
            const auto voice_assets = models::pocket_tts_small::load_voice_assets_for_plan(plan, manifest);
            result = VoiceConditioningResult{
                plan,
                flow_lm_.make_state(voice_assets),
            };
            return;
        }

        if (!weights.has_mimi_encoder) {
            throw std::runtime_error("This PocketTTS checkpoint has no Mimi encoder, so it cannot clone a voice from audio");
        }
        const int sample_rate = manifest.model_config.sample_rate;
        const auto frame_rate = manifest.model_config.mimi_frame_rate;
        const int64_t frame_size = static_cast<int64_t>(std::llround(static_cast<double>(sample_rate) / frame_rate));
        const auto audio = plan.clone_audio.has_value()
            ? prepare_conditioning_audio(*plan.clone_audio, sample_rate, frame_size, plan.truncate_clone_audio)
            : load_conditioning_audio(plan.clone_audio_path, sample_rate, frame_size, plan.truncate_clone_audio);
        auto prompt_embedding =
            mimi_encoder_.encode_prompt_embedding(backend, threads, manifest, weights, audio, mimi_encoder_graph_context_bytes);
        prompt_embedding = append_bos_voice_embedding(manifest, weights.host, std::move(prompt_embedding));
        prompt_frames = prompt_embedding.frames;
        result = VoiceConditioningResult{
            plan,
            flow_lm_.prompt_state(
                backend,
                threads,
                manifest,
                weights,
                prompt_embedding.values,
                prompt_embedding.frames,
                flow_lm_.make_empty_state(),
                flow_weights_view_context_bytes,
                flow_step_graph_context_bytes),
        };
    });
    auto source_name = std::string("unknown");
    switch (result.plan.source) {
        case models::pocket_tts_small::VoiceSourceKind::NamedPreset:
            source_name = "named_preset";
            break;
        case models::pocket_tts_small::VoiceSourceKind::PreparedEmbedding:
            source_name = "prepared_embedding";
            break;
        case models::pocket_tts_small::VoiceSourceKind::CloneAudio:
            source_name = "clone_audio";
            break;
    }
    engine::debug::timing_log_scalar("pocket_tts_small.voice.prepare_ms", prepare_ms);
    engine::debug::trace_log_scalar("pocket_tts_small.voice.source", source_name);
    if (prompt_frames > 0) {
        engine::debug::trace_log_scalar("pocket_tts_small.voice.prompt_frames", prompt_frames);
    }
    return result;
}

}  // namespace engine::models::pocket_tts_small
