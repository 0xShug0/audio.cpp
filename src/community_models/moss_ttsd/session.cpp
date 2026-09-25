#include "engine/community_models/moss_ttsd/session.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace engine::models::moss_ttsd {
namespace {

using Clock = std::chrono::steady_clock;

// Codec frames per character of dialogue, at 12.5 frames a second. Only used to
// derive a ceiling: this checkpoint has no "- Tokens:" budget to honour, so the
// number is a guard against a run that never stops rather than a target.
constexpr double kFramesPerCharacter = 0.95;
constexpr double kCeilingFraction = 1.8;
constexpr int64_t kCeilingSlackFrames = 50;

std::shared_ptr<const Assets> require_assets(std::shared_ptr<const Assets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("MOSS-TTSD session requires assets");
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
        "moss_ttsd.weight_type supports native, f32, bf16 and q8_0 (f16 produces NaN on this model)");
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

std::vector<std::optional<std::string>> parse_speaker_paths(const std::string & value) {
    std::vector<std::optional<std::string>> out;
    if (value.empty()) {
        return out;
    }
    std::string current;
    const auto push = [&out](std::string entry) {
        const auto first = entry.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            out.emplace_back();          // named, not cloned
            return;
        }
        const auto last = entry.find_last_not_of(" \t\r\n");
        out.emplace_back(entry.substr(first, last - first + 1));
    };
    for (const char character : value) {
        if (character == ',') {
            push(current);
            current.clear();
            continue;
        }
        current += character;
    }
    push(current);
    return out;
}

MossTtsdSession::MossTtsdSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const Assets> assets)
    : runtime::RuntimeSessionBase(options),
      task_(std::move(task)),
      assets_(require_assets(std::move(assets))) {}

MossTtsdSession::~MossTtsdSession() = default;

std::string MossTtsdSession::family() const {
    return "moss_ttsd";
}

runtime::VoiceTaskKind MossTtsdSession::task_kind() const {
    return task_.task;
}

runtime::RunMode MossTtsdSession::run_mode() const {
    return runtime::RunMode::Offline;
}

void MossTtsdSession::prepare(const runtime::SessionPreparationRequest &) {
    // Idempotent, and guarded on the LAST thing constructed rather than the
    // first: a prepare() that threw partway through would otherwise be retried,
    // take the early return, and report itself prepared with null members.
    if (codec_ != nullptr) {
        mark_prepared();
        return;
    }

    const auto & session_options = options().options;
    const auto weight_type = runtime::find_option(session_options, {"moss_ttsd.weight_type", "weight_type"});
    if (weight_type.has_value()) {
        weight_storage_type_ = parse_weight_type(*weight_type);
    }

    const auto & config = assets_->config;

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
    codebook_spec.tensor_prefix = "emb_ext";
    codebooks_ = std::make_unique<engine::modules::MultiCodebookEmbedding>(*assets_->model_weights, codebook_spec);

    // Same GPU opt-in as moss_tts_v15: the packages store the codec at f16, so loading the decoder
    // at f32 only doubles its VRAM. The f16 KV cache is bit-identical on output.
    const auto backend_type = execution_context().backend_type();
    const bool use_f16 = backend_type == core::BackendType::Cuda ||
                         backend_type == core::BackendType::Vulkan ||
                         backend_type == core::BackendType::Metal;
    backbone_ = std::make_unique<decoders::MossTtsDelayBackboneRuntime>(
        assets_->config, assets_->model_weights, execution_context(),
        backbone_graph_arena_bytes_, backbone_weight_context_bytes_, weight_storage_type_,
        use_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32);
    heads_ = std::make_unique<decoders::MossTtsDelayHeadsRuntime>(
        assets_->config, assets_->model_weights, execution_context(),
        heads_graph_arena_bytes_, heads_weight_context_bytes_, weight_storage_type_);
    // ⚠ CONSTRUCTED WITH THE MODEL'S n_vq, NOT THE CODEC'S. MOSS-Audio-Tokenizer
    // v1 carries 32 RVQ layers and this checkpoint reads the first 16; the
    // reference truncates the tokenizer's output the same way ("Always follow
    // model RVQ channels"). Passing num_codebooks here loads only those 16
    // quantizers, so both encode and decode follow the model rather than the
    // codec, and no separate truncation step is needed.
    codec_ = std::make_unique<engine::codecs::MossAudioTokenizerCodecRuntime>(
        assets_->audio_tokenizer_weights, execution_context(), config.num_codebooks,
        engine::codecs::MossAudioTokenizerCodecRuntimeOptions{
            codec_weight_context_bytes_, codec_graph_arena_bytes_, codec_graph_arena_bytes_, false,
            use_f16 ? assets::TensorStorageType::F16 : assets::TensorStorageType::F32,
            // ⚠ THE ENCODER COMPUTES AT F32, UNLIKE moss_tts_v15. This model continues directly
            // from the reference's codes, and an f16 encoder changes 5-14% of the finer codebooks.
            // Over 26 seeds that produced silence or a re-spoken prompt 3 times, against none at
            // f32. Holding the weights at their stored type and widening them in the graph gives
            // the same f32 values -- the widening is exact -- without keeping an f32 copy resident.
            use_f16 ? assets::TensorStorageType::Native : assets::TensorStorageType::F32,
            use_f16,
        },
        engine::codecs::moss_audio_tokenizer_v1_config());
    codec_->prepare_decoder();

    mark_prepared();
}

MossTtsdSession::GeneratedChunk MossTtsdSession::generate_chunk(
    const PromptFields & fields,
    const std::optional<ReferenceAudio> & assistant_audio,
    const decoders::MossTtsDelaySamplingOptions & sampling,
    uint32_t seed,
    decoders::MossTtsDelayLengthBounds bounds_override,
    int64_t spoken_characters) {
    const auto & config = assets_->config;
    const int64_t n_vq = config.num_codebooks;
    const int64_t hidden_size = config.backbone.hidden_size;

    const auto prompt = assistant_audio.has_value()
        ? build_continuation_prefix(fields, *assistant_audio, config, *tokenizer_)
        : build_generation_prefix(fields, config, *tokenizer_);
    const auto prompt_rows = static_cast<int64_t>(prompt.text_tokens.size());

    // ⚠ NO FLOOR. v1.5 floors the length because voice design has nothing to
    // anchor duration against and will otherwise retire the codebooks on the
    // first frame. A dialogue has speaker tags and punctuation telling the model
    // how long it is, and a floor here would force it to keep talking past the
    // end of the script. Only the ceiling is kept, as a guard against a run that
    // never stops.
    decoders::MossTtsDelayLengthBounds bounds = bounds_override;
    if (bounds.max_frames <= 0) {
        const double expected = static_cast<double>(spoken_characters) * kFramesPerCharacter;
        bounds.max_frames = static_cast<int64_t>(expected * kCeilingFraction) + kCeilingSlackFrames;
    }
    // The ceiling counts from the start of the audio, and in continuation the
    // prompt's frames are already on that clock. Without this the budget for the
    // words actually being asked for shrinks by the length of the reference.
    if (assistant_audio.has_value()) {
        bounds.max_frames += assistant_audio->frames;
    }
    const int64_t max_steps = bounds.max_frames + n_vq + 4;

    std::vector<float> prompt_bias(static_cast<size_t>(prompt_rows * hidden_size), 0.0F);
    for (int64_t row = 0; row < prompt_rows; ++row) {
        codebooks_->add_bias(
            prompt.audio_codes.data() + static_cast<size_t>(row * n_vq),
            prompt_bias.data() + static_cast<size_t>(row * hidden_size));
    }

    decoders::MossTtsDelayDecoder decoder(config, sampling, seed, bounds);
    decoder.seed_prompt_codes(prompt.audio_codes.data(), prompt_rows);
    if (assistant_audio.has_value()) {
        decoder.begin_continuation(assistant_audio->frames);
    }
    backbone_->begin_generation(prompt_rows + max_steps + 8);
    auto hidden = backbone_->prefill(prompt.text_tokens, prompt_bias);

    decoders::MossTtsDelayStepLogits logits;
    std::vector<float> row_bias(static_cast<size_t>(hidden_size), 0.0F);
    GeneratedChunk chunk;
    // In continuation the audio is already running: the prefix ended mid-span, so
    // the model is not going to emit an audio_start and nothing should wait for one.
    chunk.started_audio = assistant_audio.has_value();
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
    debug::trace_log_scalar("moss_ttsd.chunk.text_chars", spoken_characters);
    debug::trace_log_scalar("moss_ttsd.chunk.max_frames", bounds.max_frames);
    debug::trace_log_scalar("moss_ttsd.chunk.frames", chunk.frames);
    debug::trace_log_scalar("moss_ttsd.chunk.continuation", assistant_audio.has_value() ? 1 : 0);
    return chunk;
}

ReferenceAudio MossTtsdSession::encode_reference(const runtime::AudioBuffer & audio) {
    const auto rate = static_cast<int>(codec_->sampling_rate());
    if (audio.samples.empty()) {
        throw std::runtime_error("MOSS-TTSD reference recording is empty");
    }
    std::vector<float> mono = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio.samples, audio.sample_rate, audio.channels, rate);

    // Whole frames only: a partial frame has no codes and would make the last
    // column of the reference span ambiguous.
    const auto samples_per_frame = static_cast<size_t>(rate) / 125 * 10;
    mono.resize(mono.size() / samples_per_frame * samples_per_frame);
    if (mono.empty()) {
        throw std::runtime_error("MOSS-TTSD reference recording is shorter than one codec frame");
    }

    codec_->prepare_encoder();
    auto encoded = codec_->encode(engine::codecs::MossAudioTokenizerAudio{rate, {std::move(mono)}});
    ReferenceAudio reference;
    reference.frames = encoded.frames;
    reference.codes = std::move(encoded.codebooks);
    return reference;
}

ReferenceAudio MossTtsdSession::concatenate_references(
    const std::vector<std::optional<ReferenceAudio>> & speakers) const {
    ReferenceAudio out;
    out.codes.assign(static_cast<size_t>(assets_->config.num_codebooks), {});
    for (const auto & speaker : speakers) {
        if (!speaker.has_value()) {
            continue;
        }
        if (static_cast<int64_t>(speaker->codes.size()) != assets_->config.num_codebooks) {
            throw std::runtime_error("MOSS-TTSD reference has the wrong codebook count");
        }
        for (size_t codebook = 0; codebook < speaker->codes.size(); ++codebook) {
            auto & row = out.codes[codebook];
            row.insert(row.end(), speaker->codes[codebook].begin(), speaker->codes[codebook].end());
        }
        out.frames += speaker->frames;
    }
    return out;
}

std::vector<float> MossTtsdSession::decode_codes(const GeneratedChunk & chunk) {
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
        throw std::runtime_error("MOSS-TTSD codec returned no audio");
    }
    return std::move(audio.channels.front());
}

runtime::TaskResult MossTtsdSession::run(const runtime::TaskRequest & request) {
    require_prepared("MOSS-TTSD run()");
    const auto wall_start = Clock::now();
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("MOSS-TTSD requires dialogue text to speak");
    }
    const std::string dialogue = request.text_input->text;

    // Positional speakers. An empty entry is a speaker that is named in the text
    // but not cloned, which renders "[S<n>]: None".
    std::vector<std::optional<ReferenceAudio>> speakers;
    const auto samples_option =
        runtime::find_option(request.options, {"moss_ttsd.voice_samples", "voice_samples"});
    const bool has_voice_ref = request.voice.has_value() && request.voice->speaker.has_value()
        && request.voice->speaker->audio.has_value();
    if (samples_option.has_value() && has_voice_ref) {
        throw std::runtime_error("MOSS-TTSD request cannot combine voice_samples with voice_ref");
    }
    if (samples_option.has_value()) {
        for (const auto & path : parse_speaker_paths(*samples_option)) {
            if (!path.has_value()) {
                speakers.emplace_back();
                continue;
            }
            const auto wav = engine::audio::read_wav_f32(std::filesystem::path(*path));
            speakers.emplace_back(encode_reference(
                runtime::AudioBuffer{wav.sample_rate, wav.channels, wav.samples}));
        }
    } else if (has_voice_ref) {
        speakers.emplace_back(encode_reference(*request.voice->speaker->audio));
    }

    // What the references say. The reference implementation prepends the
    // transcripts to the dialogue, so the continuation carries on from words it
    // has already spoken rather than restarting mid-sentence.
    const std::string reference_text =
        option_string(request.options, {"moss_ttsd.reference_text", "reference_text"}, "");

    PromptFields fields;
    fields.text = reference_text.empty() ? dialogue : (reference_text + " " + dialogue);
    fields.references = speakers;
    const auto instruction = option_string(request.options, {"instruct", "instruction"}, "");
    if (!instruction.empty()) {
        fields.instruction = instruction;
    }
    const auto language = option_string(request.options, {"language"}, "");
    if (!language.empty()) {
        fields.language = language;
    }

    // Only the dialogue is bounded, not the prepended transcripts: those are
    // already spoken by the audio the model is continuing, so counting them would
    // roughly double the ceiling.
    const auto spoken_characters = static_cast<int64_t>(dialogue.size());

    decoders::MossTtsDelaySamplingOptions sampling;
    sampling.text_temperature =
        option_float(request.options, {"moss_ttsd.text_temperature"}, sampling.text_temperature);
    sampling.audio_temperature = option_float(request.options, {"temperature"}, sampling.audio_temperature);
    sampling.audio_top_p = option_float(request.options, {"top_p"}, sampling.audio_top_p);
    sampling.audio_top_k = option_int(request.options, {"top_k"}, sampling.audio_top_k);
    sampling.audio_repetition_penalty =
        option_float(request.options, {"repetition_penalty"}, sampling.audio_repetition_penalty);
    const auto seed = static_cast<uint32_t>(option_int(request.options, {"seed"}, 0));

    decoders::MossTtsDelayLengthBounds bounds_override;
    bounds_override.max_frames = option_int(request.options, {"max_frames"}, 0);

    // ⚠ ONE TAKE, NOT CHUNKED. v1.5 splits long text and renders each piece
    // separately, which works because every piece is the same single voice. A
    // dialogue cannot be split that way: the speaker a chunk starts in depends on
    // the tags before it, and each chunk would restart the continuation from the
    // reference rather than from where the conversation had got to.
    std::optional<ReferenceAudio> assistant_audio;
    const bool cloning = std::any_of(
        speakers.begin(), speakers.end(),
        [](const std::optional<ReferenceAudio> & speaker) { return speaker.has_value(); });
    if (cloning) {
        assistant_audio = concatenate_references(speakers);
    }

    const auto chunk = generate_chunk(
        fields, assistant_audio, sampling, seed, bounds_override, spoken_characters);

    // ⚠ DECODE THE PROMPT AUDIO WITH IT, THEN CUT THE PROMPT BACK OFF. The codec
    // is causal, so decoding the generated frames alone starts it cold and the
    // first moments of the continuation are reconstructed without the context
    // that produced them -- audible as a seam. The reference does the same thing
    // and says so: "Keep codec causal context by decoding the whole first
    // segment first, then trim at waveform level according to start_length
    // ratio." The trim is by ratio rather than by a computed sample count for
    // the same reason it is there: the codec's frame-to-sample ratio is its own
    // business, and deriving the cut from the audio it actually returned cannot
    // drift from it.
    GeneratedChunk decoded = chunk;
    int64_t prompt_frames = 0;
    if (assistant_audio.has_value() && chunk.frames > 0) {
        prompt_frames = assistant_audio->frames;
        // ⚠ CHECKED, NOT ASSUMED. The copies below write each codebook's prompt
        // rows and then the generated rows at a fixed offset into one buffer
        // sized from these counts. A prompt row longer than `frames`, or fewer
        // codebooks than the chunk reports, would run past its slice and into
        // the next one -- or past the end of the buffer on the last codebook.
        // It holds by construction today; it is not worth finding out the hard
        // way if that changes.
        if (static_cast<int64_t>(assistant_audio->codes.size()) < chunk.codebooks) {
            throw std::runtime_error("MOSS-TTSD prompt audio has fewer codebooks than the generated chunk");
        }
        for (int64_t cb = 0; cb < chunk.codebooks; ++cb) {
            if (static_cast<int64_t>(assistant_audio->codes[static_cast<size_t>(cb)].size()) != prompt_frames) {
                throw std::runtime_error("MOSS-TTSD prompt audio codebook length does not match its frame count");
            }
        }
        decoded.frames = prompt_frames + chunk.frames;
        decoded.codes.assign(static_cast<size_t>(chunk.codebooks * decoded.frames), 0);
        for (int64_t cb = 0; cb < chunk.codebooks; ++cb) {
            auto * row = decoded.codes.data() + static_cast<size_t>(cb * decoded.frames);
            const auto & prompt_row = assistant_audio->codes[static_cast<size_t>(cb)];
            std::copy(prompt_row.begin(), prompt_row.end(), row);
            std::copy(chunk.codes.begin() + static_cast<int64_t>(cb * chunk.frames),
                      chunk.codes.begin() + static_cast<int64_t>((cb + 1) * chunk.frames),
                      row + prompt_frames);
        }
    }
    auto samples = decode_codes(decoded);
    if (prompt_frames > 0 && !samples.empty()) {
        const double ratio = static_cast<double>(prompt_frames) / static_cast<double>(decoded.frames);
        const auto cut = static_cast<size_t>(static_cast<double>(samples.size()) * ratio);
        samples.erase(samples.begin(), samples.begin() + static_cast<std::ptrdiff_t>(std::min(cut, samples.size())));
    }

    runtime::TaskResult result;
    result.audio_output = runtime::AudioBuffer{
        static_cast<int>(assets_->config.sampling_rate), 1, std::move(samples)};
    // The same name v1.5 reports under, so an RTF measured across the family is
    // measuring the same thing: the session's own work, not the process including
    // the seconds spent reading a multi-gigabyte package off disk.
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return result;
}

}  // namespace engine::models::moss_ttsd
