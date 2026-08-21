#include "engine/community_models/moss_voicegen/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace engine::models::moss_voicegen {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int64_t kDefaultTextChunkSize = 200;

// Duration bounds, in codec frames at 12.5 frames a second. The model has no reference
// recording to anchor against, so these are derived from the text length.
//
// The rate is measured, not guessed: takes that read their text in full land at 0.92 to
// 1.02 frames per character, so ~0.95 is a natural reading. The floor sits at about half
// of that — brisk delivery is fine, stopping a third of the way through the sentence is
// not — and the ceiling leaves room for a slow reading with pauses.
constexpr double kFramesPerCharacter = 0.95;
constexpr double kFloorFraction = 0.45;
constexpr double kCeilingFraction = 1.6;
constexpr int64_t kMinFramesFloor = 12;
constexpr int64_t kCeilingSlackFrames = 25;

std::shared_ptr<const MossVoiceGenAssets> require_assets(std::shared_ptr<const MossVoiceGenAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("MOSS-VoiceGenerator session requires assets");
    }
    return assets;
}

engine::assets::TensorStorageType parse_weight_type(const std::string & value) {
    if (value == "native") {
        return engine::assets::TensorStorageType::Native;
    }
    if (value == "f32") {
        return engine::assets::TensorStorageType::F32;
    }
    if (value == "bf16") {
        return engine::assets::TensorStorageType::BF16;
    }
    if (value == "q8_0") {
        return engine::assets::TensorStorageType::Q8_0;
    }
    // f16 is deliberately not offered: this backbone's attention-sink activations run to
    // ~169k, far past f16's range, and produce NaN from the first position.
    throw std::runtime_error(
        "moss_voicegen.weight_type supports native, f32, bf16 and q8_0 (f16 produces NaN on this model)");
}

std::string option_string(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys,
    const std::string & fallback) {
    return runtime::find_option(options, keys).value_or(fallback);
}

float option_float(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys,
    float fallback) {
    const auto value = runtime::find_option(options, keys);
    return value.has_value() ? std::stof(*value) : fallback;
}

int option_int(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys,
    int fallback) {
    const auto value = runtime::find_option(options, keys);
    return value.has_value() ? std::stoi(*value) : fallback;
}

}  // namespace

MossVoiceGenSession::MossVoiceGenSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const MossVoiceGenAssets> assets)
    : runtime::RuntimeSessionBase(options),
      task_(std::move(task)),
      assets_(require_assets(std::move(assets))) {}

MossVoiceGenSession::~MossVoiceGenSession() = default;

std::string MossVoiceGenSession::family() const {
    return "moss_voicegen";
}

runtime::VoiceTaskKind MossVoiceGenSession::task_kind() const {
    return runtime::VoiceTaskKind::VoiceDesign;
}

runtime::RunMode MossVoiceGenSession::run_mode() const {
    return runtime::RunMode::Offline;
}

void MossVoiceGenSession::prepare(const runtime::SessionPreparationRequest &) {
    // The server calls prepare() on every request against one long-lived session, so this
    // has to be idempotent: building the runtimes here unconditionally re-uploaded the
    // whole 5.7 GB model per request, which cost about four seconds on top of roughly one
    // second of actual work. Nothing in this model depends on the request — there is no
    // reference audio to encode — so it is all built once.
    if (backbone_ != nullptr) {
        mark_prepared();
        return;
    }

    const auto & session_options = options().options;
    const auto weight_type = runtime::find_option(session_options, {"moss_voicegen.weight_type", "weight_type"});
    if (weight_type.has_value()) {
        weight_storage_type_ = parse_weight_type(*weight_type);
    }

    const auto & config = assets_->config;
    text_processor_ = std::make_unique<MossVoiceGenTextProcessor>(assets_);

    moss::AudioCodebookSpec codebook_spec;
    codebook_spec.hidden_size = config.backbone.hidden_size;
    codebook_spec.num_codebooks = config.num_codebooks;
    codebook_spec.audio_vocab_size = config.audio_vocab_size + 1;
    codebook_spec.audio_pad_token_id = config.audio_pad_code;
    // The delay family stores its per-codebook input embeddings as emb_ext.<i>, not under
    // the shared default prefix.
    codebook_spec.tensor_prefix = "emb_ext";
    codebooks_ = std::make_unique<moss::AudioCodebookEmbeddings>(*assets_->model_weights, codebook_spec);

    backbone_ = std::make_unique<MossVoiceGenBackboneRuntime>(
        assets_,
        execution_context(),
        backbone_graph_arena_bytes_,
        backbone_weight_context_bytes_,
        weight_storage_type_);
    heads_ = std::make_unique<MossVoiceGenHeadsRuntime>(
        assets_,
        execution_context(),
        heads_graph_arena_bytes_,
        heads_weight_context_bytes_,
        weight_storage_type_);
    codec_ = std::make_unique<moss::MossAudioTokenizerDecoder>(
        *assets_->audio_tokenizer_weights,
        execution_context(),
        config.num_codebooks,
        codec_weight_context_bytes_,
        codec_graph_arena_bytes_,
        moss::moss_audio_tokenizer_v1_config());

    mark_prepared();
}

MossVoiceGenSession::GeneratedChunk MossVoiceGenSession::generate_chunk(
    const std::string & text,
    const std::string & instruction,
    const std::optional<std::string> & language,
    const MossVoiceGenSamplingOptions & sampling,
    uint32_t seed,
    MossVoiceGenLengthBounds bounds_override) {
    const auto & config = assets_->config;
    const int64_t n_vq = config.num_codebooks;
    const int64_t hidden_size = config.backbone.hidden_size;

    const auto prompt = text_processor_->build_generation_prefix(
        text,
        instruction.empty() ? std::optional<std::string>() : std::optional<std::string>(instruction),
        language);
    const auto prompt_rows = static_cast<int64_t>(prompt.text_tokens.size());

    const auto characters = static_cast<int64_t>(text.size());
    const double expected_frames = static_cast<double>(characters) * kFramesPerCharacter;
    MossVoiceGenLengthBounds bounds = bounds_override;
    if (bounds.min_frames <= 0) {
        bounds.min_frames = std::max<int64_t>(kMinFramesFloor, static_cast<int64_t>(expected_frames * kFloorFraction));
    }
    if (bounds.max_frames <= 0) {
        bounds.max_frames =
            static_cast<int64_t>(expected_frames * kCeilingFraction) + kCeilingSlackFrames;
    }
    bounds.max_frames = std::max<int64_t>(bounds.max_frames, bounds.min_frames + n_vq);
    const int64_t max_steps = bounds.max_frames + n_vq + 4;

    std::vector<float> prompt_bias(static_cast<size_t>(prompt_rows * hidden_size), 0.0F);
    for (int64_t row = 0; row < prompt_rows; ++row) {
        codebooks_->add_bias(
            prompt.audio_codes.data() + static_cast<size_t>(row * n_vq),
            prompt_bias.data() + static_cast<size_t>(row * hidden_size));
    }

    MossVoiceGenDelayDecoder decoder(config, sampling, seed, bounds);
    backbone_->begin_generation(prompt_rows + max_steps + 8);
    auto hidden = backbone_->prefill(prompt.text_tokens, prompt_bias);

    MossVoiceGenStepLogits logits;
    std::vector<float> row_bias(static_cast<size_t>(hidden_size), 0.0F);
    GeneratedChunk chunk;
    for (int64_t step = 0; step < max_steps; ++step) {
        heads_->evaluate(hidden, logits);
        const auto row = decoder.step(logits);
        if (row.text_token == static_cast<int32_t>(config.audio_start_token_id)) {
            chunk.started_audio = true;
        }
        if (decoder.stopped()) {
            break;
        }
        std::fill(row_bias.begin(), row_bias.end(), 0.0F);
        codebooks_->add_bias(row.codes.data(), row_bias.data());
        hidden = backbone_->step(row.text_token, row_bias);
    }

    chunk.codes = decoder.extract_audio_codes(chunk.codebooks, chunk.frames);
    chunk.hit_frame_ceiling = chunk.frames >= bounds.max_frames;
    debug::trace_log_scalar("moss_voicegen.chunk.text_chars", characters);
    debug::trace_log_scalar("moss_voicegen.chunk.min_frames", bounds.min_frames);
    debug::trace_log_scalar("moss_voicegen.chunk.max_frames", bounds.max_frames);
    debug::trace_log_scalar("moss_voicegen.chunk.frames", chunk.frames);
    debug::trace_log_scalar("moss_voicegen.chunk.started_audio", chunk.started_audio);
    return chunk;
}

std::vector<float> MossVoiceGenSession::decode_codes(const GeneratedChunk & chunk) {
    if (chunk.frames <= 0) {
        return {};
    }
    std::vector<std::vector<int32_t>> codes(static_cast<size_t>(chunk.codebooks));
    for (int64_t codebook = 0; codebook < chunk.codebooks; ++codebook) {
        codes[static_cast<size_t>(codebook)].assign(
            chunk.codes.begin() + static_cast<int64_t>(codebook * chunk.frames),
            chunk.codes.begin() + static_cast<int64_t>((codebook + 1) * chunk.frames));
    }
    auto channels = codec_->decode(codes);
    if (channels.empty()) {
        throw std::runtime_error("MOSS-VoiceGenerator codec returned no audio");
    }
    return std::move(channels.front());
}

runtime::TaskResult MossVoiceGenSession::run(const runtime::TaskRequest & request) {
    require_prepared("MOSS-VoiceGenerator run()");
    const auto wall_start = Clock::now();
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("MOSS-VoiceGenerator requires text to speak");
    }

    // The voice description arrives either as a request option or as a style tag on the
    // voice condition, matching how qwen3_tts takes its voice-design instruction.
    std::string instruction = option_string(request.options, {"instruct"}, "");
    if (instruction.empty() && request.voice.has_value() && request.voice->style.has_value()) {
        const auto tag = request.voice->style->tags.find("instruct");
        if (tag != request.voice->style->tags.end()) {
            instruction = tag->second;
        }
    }

    // The model was trained on full language names; "en" means nothing to it.
    std::optional<std::string> language;
    const auto language_option = runtime::find_option(request.options, {"language"});
    if (language_option.has_value() && !language_option->empty()) {
        language = *language_option;
    }

    MossVoiceGenLengthBounds bounds_override;
    bounds_override.min_frames = option_int(request.options, {"min_frames"}, 0);
    bounds_override.max_frames = option_int(request.options, {"max_frames"}, 0);

    MossVoiceGenSamplingOptions sampling;
    sampling.text_temperature = option_float(request.options, {"moss_voicegen.text_temperature"}, sampling.text_temperature);
    sampling.audio_temperature = option_float(request.options, {"temperature"}, sampling.audio_temperature);
    sampling.audio_top_p = option_float(request.options, {"top_p"}, sampling.audio_top_p);
    sampling.audio_top_k = option_int(request.options, {"top_k"}, sampling.audio_top_k);
    sampling.audio_repetition_penalty =
        option_float(request.options, {"repetition_penalty"}, sampling.audio_repetition_penalty);
    const auto seed = static_cast<uint32_t>(option_int(request.options, {"seed"}, 0));

    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    const auto text_chunk_mode =
        engine::text::parse_text_chunk_mode_override(request.options).value_or(engine::text::TextChunkMode::Default);
    const auto chunk_requests = runtime::chunk_text_request(request, text_chunk_size, text_chunk_mode);

    runtime::AudioBuffer merged;
    merged.sample_rate = static_cast<int>(codec_->sampling_rate());
    merged.channels = 1;
    int64_t chunk_index = 0;
    int64_t silent_chunks = 0;
    for (const auto & chunk_request : chunk_requests) {
        // Each chunk gets its own seed offset so a long text does not repeat one take, and
        // stays reproducible for a given request seed.
        const auto chunk = generate_chunk(
            chunk_request.text_input->text,
            instruction,
            language,
            sampling,
            seed + static_cast<uint32_t>(chunk_index),
            bounds_override);
        if (chunk.frames <= 0) {
            // The model can answer in text rather than audio; upstream behaves the same way.
            ++silent_chunks;
            ++chunk_index;
            continue;
        }
        auto samples = decode_codes(chunk);
        merged.samples.insert(merged.samples.end(), samples.begin(), samples.end());
        ++chunk_index;
    }

    debug::trace_log_scalar("moss_voicegen.chunk_count", static_cast<int64_t>(chunk_requests.size()));
    debug::trace_log_scalar("moss_voicegen.silent_chunks", silent_chunks);
    debug::timing_log_scalar("moss_voicegen.run_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));

    if (merged.samples.empty()) {
        throw std::runtime_error(
            "MOSS-VoiceGenerator produced no audio: the model answered in text. Retry with another seed.");
    }

    runtime::TaskResult result;
    result.audio_output = std::move(merged);
    return result;
}

}  // namespace engine::models::moss_voicegen
