#include "engine/community_models/moss_tts_v15/session.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace engine::models::moss_tts_v15 {
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

std::shared_ptr<const Assets> require_assets(std::shared_ptr<const Assets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("MOSS-TTS-v1.5 session requires assets");
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
        "moss_tts_v15.weight_type supports native, f32, bf16 and q8_0 (f16 produces NaN on this model)");
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

MossTtsV15Session::MossTtsV15Session(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const Assets> assets)
    : runtime::RuntimeSessionBase(options),
      task_(std::move(task)),
      assets_(require_assets(std::move(assets))) {}

MossTtsV15Session::~MossTtsV15Session() = default;

std::string MossTtsV15Session::family() const {
    return "moss_tts_v15";
}

runtime::VoiceTaskKind MossTtsV15Session::task_kind() const {
    // tts and clone are the same path here, distinguished only by whether the request
    // carries a reference recording, so report whichever the caller asked for.
    return task_.task;
}

runtime::RunMode MossTtsV15Session::run_mode() const {
    return runtime::RunMode::Offline;
}

void MossTtsV15Session::prepare(const runtime::SessionPreparationRequest &) {
    // The server calls prepare() on every request against one long-lived session,
    // so this has to be idempotent: building the runtimes unconditionally would
    // re-upload the whole model per request. Nothing built here depends on the
    // request — the reference recording is encoded in run(), not here — so it is
    // all built once.
    //
    // The guard is on the last thing constructed, not the first. Guarding on
    // backbone_ would let a prepare() that threw while building the heads, the
    // codec or the embeddings be retried, take the early return, and report
    // itself prepared with those members still null; run() then dereferences a
    // null codec_. On a model this size an allocation failure partway through is
    // not hypothetical.
    if (codec_ != nullptr) {
        mark_prepared();
        return;
    }

    const auto & session_options = options().options;
    const auto weight_type = runtime::find_option(session_options, {"moss_tts_v15.weight_type", "weight_type"});
    if (weight_type.has_value()) {
        weight_storage_type_ = parse_weight_type(*weight_type);
    }

    const auto & config = assets_->config;

    // Same spec moss_voicegen uses: the checkpoint ships tokenizer.json and merges.txt
    // but no vocab.json, so the vocabulary and merge ranks come from tokenizer.json
    // rather than from a vocab/merges pair.
    engine::tokenizers::LlamaBpeTokenizerSpec tokenizer_spec;
    tokenizer_spec.tokenizer_config_path = assets_->resources.require_file("tokenizer_config");
    tokenizer_spec.tokenizer_json_path = assets_->resources.require_file("tokenizer_json");
    tokenizer_spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;
    tokenizer_ = engine::tokenizers::load_llama_bpe_tokenizer(tokenizer_spec);

    engine::modules::MultiCodebookEmbeddingSpec codebook_spec;
    codebook_spec.hidden_size = config.backbone.hidden_size;
    codebook_spec.num_codebooks = config.num_codebooks;
    codebook_spec.vocab_size = config.audio_vocab_size + 1;
    codebook_spec.pad_token_id = config.audio_pad_code;
    // The delay family stores its per-codebook input embeddings as emb_ext.<i>, not under
    // the shared default prefix.
    codebook_spec.tensor_prefix = "emb_ext";
    codebooks_ = std::make_unique<engine::modules::MultiCodebookEmbedding>(*assets_->model_weights, codebook_spec);

    const auto backend_type = execution_context().backend_type();
    const bool use_f16 = backend_type == core::BackendType::Cuda ||
                         backend_type == core::BackendType::Vulkan ||
                         backend_type == core::BackendType::Metal;
    backbone_ = std::make_unique<decoders::MossTtsDelayBackboneRuntime>(
        assets_->config,
        assets_->model_weights,
        execution_context(),
        backbone_graph_arena_bytes_,
        backbone_weight_context_bytes_,
        weight_storage_type_,
        use_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32);
    heads_ = std::make_unique<decoders::MossTtsDelayHeadsRuntime>(
        assets_->config,
        assets_->model_weights,
        execution_context(),
        heads_graph_arena_bytes_,
        heads_weight_context_bytes_,
        weight_storage_type_);
    codec_ = std::make_unique<engine::codecs::MossAudioTokenizerCodecRuntime>(
        assets_->audio_tokenizer_weights,
        execution_context(),
        config.num_codebooks,
        engine::codecs::MossAudioTokenizerCodecRuntimeOptions{
            codec_weight_context_bytes_,
            codec_graph_arena_bytes_,
            codec_graph_arena_bytes_,
            false,
            use_f16 ? assets::TensorStorageType::F16 : assets::TensorStorageType::F32,
        },
        engine::codecs::moss_audio_tokenizer_v1_config());
    codec_->prepare_decoder();
    // v1.5 clones from a reference recording, so unlike voice design it needs the
    // encoder as well. It is prepared lazily on the first request that supplies one.

    mark_prepared();
}

MossTtsV15Session::GeneratedChunk MossTtsV15Session::generate_chunk(
    const PromptFields & fields,
    const decoders::MossTtsDelaySamplingOptions & sampling,
    uint32_t seed,
    decoders::MossTtsDelayLengthBounds bounds_override,
    int64_t requested_frames) {
    const auto & config = assets_->config;
    const int64_t n_vq = config.num_codebooks;
    const int64_t hidden_size = config.backbone.hidden_size;

    const auto prompt = build_generation_prefix(fields, config, *tokenizer_);
    const auto prompt_rows = static_cast<int64_t>(prompt.text_tokens.size());

    const auto characters = static_cast<int64_t>(fields.text.size());
    // The family's "- Tokens:" field is an explicit duration budget in codec frames and
    // the model honours it (40 tokens produced 41 frames on the reference). When the
    // caller gives one, it is the estimate; otherwise fall back to the character rate,
    // which is all voice design ever had.
    const double expected_frames = requested_frames > 0
        ? static_cast<double>(requested_frames)
        : static_cast<double>(characters) * kFramesPerCharacter;
    decoders::MossTtsDelayLengthBounds bounds = bounds_override;
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

    decoders::MossTtsDelayDecoder decoder(config, sampling, seed, bounds);
    // A cloning prompt carries the reference recording's codes, and the reference
    // implementation penalises repetition against them like any earlier row.
    decoder.seed_prompt_codes(prompt.audio_codes.data(), prompt_rows);
    backbone_->begin_generation(prompt_rows + max_steps + 8);
    auto hidden = backbone_->prefill(prompt.text_tokens, prompt_bias);

    decoders::MossTtsDelayStepLogits logits;
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
    debug::trace_log_scalar("moss_tts_v15.chunk.text_chars", characters);
    debug::trace_log_scalar("moss_tts_v15.chunk.min_frames", bounds.min_frames);
    debug::trace_log_scalar("moss_tts_v15.chunk.max_frames", bounds.max_frames);
    debug::trace_log_scalar("moss_tts_v15.chunk.frames", chunk.frames);
    debug::trace_log_scalar("moss_tts_v15.chunk.started_audio", chunk.started_audio);
    return chunk;
}

ReferenceAudio MossTtsV15Session::encode_reference(const runtime::AudioBuffer & audio) {
    const auto rate = static_cast<int>(codec_->sampling_rate());
    if (audio.samples.empty()) {
        throw std::runtime_error("MOSS-TTS-v1.5 reference recording is empty");
    }

    // Downmix and resample to the codec's rate: it is mono at 24 kHz, and a reference
    // at any other shape would otherwise be encoded as though it were not.
    std::vector<float> mono = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples, audio.sample_rate, audio.channels, rate);

    // Whole frames only: a partial frame has no codes and would make the last column
    // of the reference span ambiguous.
    const auto samples_per_frame = static_cast<size_t>(rate) / 125 * 10;
    mono.resize(mono.size() / samples_per_frame * samples_per_frame);
    if (mono.empty()) {
        throw std::runtime_error("MOSS-TTS-v1.5 reference recording is shorter than one codec frame");
    }

    codec_->prepare_encoder();
    auto encoded = codec_->encode(engine::codecs::MossAudioTokenizerAudio{rate, {std::move(mono)}});
    ReferenceAudio reference;
    reference.frames = encoded.frames;
    reference.codes = std::move(encoded.codebooks);
    debug::trace_log_scalar("moss_tts_v15.reference_frames", reference.frames);
    return reference;
}

std::vector<float> MossTtsV15Session::decode_codes(const GeneratedChunk & chunk) {
    if (chunk.frames <= 0) {
        return {};
    }
    std::vector<std::vector<int32_t>> codes(static_cast<size_t>(chunk.codebooks));
    for (int64_t codebook = 0; codebook < chunk.codebooks; ++codebook) {
        codes[static_cast<size_t>(codebook)].assign(
            chunk.codes.begin() + static_cast<int64_t>(codebook * chunk.frames),
            chunk.codes.begin() + static_cast<int64_t>((codebook + 1) * chunk.frames));
    }
    auto audio = codec_->decode(engine::codecs::MossAudioTokenizerCodes{chunk.frames, std::move(codes)});
    if (audio.channels.empty()) {
        throw std::runtime_error("MOSS-TTS-v1.5 codec returned no audio");
    }
    return std::move(audio.channels.front());
}

runtime::TaskResult MossTtsV15Session::run(const runtime::TaskRequest & request) {
    require_prepared("MOSS-TTS-v1.5 run()");
    const auto wall_start = Clock::now();
    // Counters accumulate across calls, so without this a second request in the same
    // session reports the first one's time as well.
    backbone_->reset_timing();
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("MOSS-TTS-v1.5 requires text to speak");
    }

    // The voice description arrives either as a request option or as a style tag on the
    // voice condition, matching how qwen3_tts takes its voice-design instruction.
    // Accept both spellings: the OpenAI-compatible speech route maps its
    // `instructions` body field to `instruction`, which is the only name a
    // WebUI voice-design request can arrive under.
    std::string instruction = option_string(request.options, {"instruct", "instruction"}, "");
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

    decoders::MossTtsDelayLengthBounds bounds_override;
    bounds_override.min_frames = option_int(request.options, {"min_frames"}, 0);
    bounds_override.max_frames = option_int(request.options, {"max_frames"}, 0);

    // The family's own duration field, in codec frames at 12.5 a second.
    const int64_t requested_frames = option_int(request.options, {"tokens", "moss_tts_v15.tokens"}, 0);

    // A reference recording is what this checkpoint has that voice design does not.
    // It is encoded once per request and reused for every text chunk, so a long text
    // does not pay for the reference again on each take.
    std::vector<ReferenceAudio> references;
    if (request.voice.has_value() && request.voice->speaker.has_value()
        && request.voice->speaker->audio.has_value()) {
        references.push_back(encode_reference(*request.voice->speaker->audio));
    }

    decoders::MossTtsDelaySamplingOptions sampling;
    sampling.text_temperature = option_float(request.options, {"moss_tts_v15.text_temperature"}, sampling.text_temperature);
    sampling.audio_temperature = option_float(request.options, {"temperature"}, sampling.audio_temperature);
    sampling.audio_top_p = option_float(request.options, {"top_p"}, sampling.audio_top_p);
    sampling.audio_top_k = option_int(request.options, {"top_k"}, sampling.audio_top_k);
    sampling.audio_repetition_penalty =
        option_float(request.options, {"repetition_penalty"}, sampling.audio_repetition_penalty);
    const auto seed = static_cast<uint32_t>(option_int(request.options, {"seed"}, 0));

    // A per-request duration budget has to be shared out, not applied to every
    // chunk: "--tokens 40" on a text that splits three ways meant three takes of
    // 40 frames each, roughly three times what was asked for, with every chunk
    // also floored at 0.45 x 40 however short its own text was.
    const int64_t text_chunk_size =
        engine::text::parse_text_chunk_size_override(request.options).value_or(kDefaultTextChunkSize);
    const auto text_chunk_mode =
        engine::text::parse_text_chunk_mode_override(request.options).value_or(engine::text::TextChunkMode::Default);
    const auto chunk_requests = runtime::chunk_text_request(request, text_chunk_size, text_chunk_mode);

    int64_t total_chunk_characters = 0;
    for (const auto & chunk_request : chunk_requests) {
        if (chunk_request.text_input.has_value()) {
            total_chunk_characters += static_cast<int64_t>(chunk_request.text_input->text.size());
        }
    }

    runtime::AudioBuffer merged;
    merged.sample_rate = static_cast<int>(codec_->sampling_rate());
    merged.channels = 1;
    int64_t chunk_index = 0;
    int64_t silent_chunks = 0;
    for (const auto & chunk_request : chunk_requests) {
        // Each chunk gets its own seed offset so a long text does not repeat one take, and
        // stays reproducible for a given request seed.
        PromptFields fields;
        fields.text = chunk_request.text_input->text;
        // Share the request's budget across chunks in proportion to their length.
        int64_t chunk_frames = 0;
        if (requested_frames > 0 && total_chunk_characters > 0) {
            const auto chars = static_cast<int64_t>(chunk_request.text_input->text.size());
            chunk_frames = std::max<int64_t>(
                1, (requested_frames * chars + total_chunk_characters / 2) / total_chunk_characters);
        }
        if (!instruction.empty()) {
            fields.instruction = instruction;
        }
        fields.language = language;
        if (chunk_frames > 0) {
            fields.tokens = std::to_string(chunk_frames);
        }
        fields.references = references;
        const auto chunk = generate_chunk(
            fields,
            sampling,
            seed + static_cast<uint32_t>(chunk_index),
            bounds_override,
            chunk_frames);
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

    debug::trace_log_scalar("moss_tts_v15.chunk_count", static_cast<int64_t>(chunk_requests.size()));
    debug::trace_log_scalar("moss_tts_v15.silent_chunks", silent_chunks);
    // The backbone already counts its own prefill/step stages; nothing in the delay family
    // ever asked it to report them, so the instrumentation was dead. Same pair of calls
    // moss_tts_local makes (generator.cpp:265 resets, :401 reports).
    backbone_->log_timing();
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));

    if (merged.samples.empty()) {
        throw std::runtime_error(
            "MOSS-TTS-v1.5 produced no audio: the model answered in text. Retry with another seed.");
    }

    runtime::TaskResult result;
    result.audio_output = std::move(merged);
    return result;
}

}  // namespace engine::models::moss_tts_v15
