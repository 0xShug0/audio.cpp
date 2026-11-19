#include "engine/community_models/granite_speech5/session.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"
#include "engine/framework/io/text.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/registry.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::granite_speech5 {
namespace json = engine::io::json;

namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kDefaultGraphArenaBytes = 1024ull * 1024ull * 1024ull;
constexpr size_t kDecoderWeightContextBytes = 4ull * 1024ull * 1024ull;
constexpr int64_t kCanonicalSampleRate = 16000;
// VAD chunking follows the granite5asr family convention: half-second merge
// gaps, quarter-second padding around speech.
constexpr double kVadMergeGapSeconds = 0.5;
constexpr double kVadPaddingSeconds = 0.25;

std::filesystem::path default_spec_path() {
    return engine::model_spec::default_spec_path("granite_speech5");
}

std::shared_ptr<const GraniteSpeech5Assets> require_assets(
    std::shared_ptr<const GraniteSpeech5Assets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Granite Speech 2B session requires assets");
    }
    return assets;
}

assets::TensorStorageType parse_load_weight_storage(
    const std::unordered_map<std::string, std::string> & options) {
    // Load-scoped precision (spec options.load): native storage keeps the
    // checkpoint format as stored (GGUF mixed types included), F32 upcasts
    // for BF16-snapshot parity.
    return runtime::parse_tensor_storage_option(
        options,
        "granite_speech5.weight_type",
        assets::TensorStorageType::Native,
        {assets::TensorStorageType::Native, assets::TensorStorageType::F32});
}

size_t parse_arena_bytes(
    const runtime::SessionOptions & options, const std::string & name) {
    return runtime::parse_size_mb_option(
        options.options, {"granite_speech5." + name}, kDefaultGraphArenaBytes);
}

std::string parse_vad_model_path(const runtime::SessionOptions & options) {
    const auto it = options.options.find("granite_speech5.vad_model_path");
    if (it != options.options.end() && !it->second.empty()) {
        return it->second;
    }
    return "assets/framework/models/silero_vad";
}

bool is_gguf_path(const std::filesystem::path & path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return extension == ".gguf";
}

// UTF-8 helpers for the JP-aware join: only the boundary code points matter.
uint32_t decode_first_code_point(const std::string & text) {
    const auto * data = reinterpret_cast<const unsigned char *>(text.data());
    const size_t size = text.size();
    if (size == 0) {
        return 0;
    }
    const unsigned char lead = data[0];
    if (lead < 0x80) {
        return lead;
    }
    size_t length = 0;
    uint32_t code = 0;
    if ((lead & 0xE0) == 0xC0) {
        length = 2;
        code = lead & 0x1F;
    } else if ((lead & 0xF0) == 0xE0) {
        length = 3;
        code = lead & 0x0F;
    } else if ((lead & 0xF8) == 0xF0) {
        length = 4;
        code = lead & 0x07;
    } else {
        return 0xFFFD;
    }
    if (size < length) {
        return 0xFFFD;
    }
    for (size_t i = 1; i < length; ++i) {
        if ((data[i] & 0xC0) != 0x80) {
            return 0xFFFD;
        }
        code = (code << 6) | (data[i] & 0x3F);
    }
    return code;
}

uint32_t decode_last_code_point(const std::string & text) {
    const auto * data = reinterpret_cast<const unsigned char *>(text.data());
    size_t size = text.size();
    if (size == 0) {
        return 0;
    }
    size_t start = size - 1;
    while (start > 0 && (data[start] & 0xC0) == 0x80) {
        --start;
    }
    return decode_first_code_point(text.substr(start));
}

}  // namespace

GraniteSpeech5RequestOptions granite_speech5_resolve_request_options(
    const runtime::TaskRequest & request) {
    GraniteSpeech5RequestOptions resolved;
    // Language: explicit request option wins over text_input.language.
    std::string raw_language;
    const auto option_hint = request.options.find("language");
    if (option_hint != request.options.end()) {
        raw_language = option_hint->second;
    } else if (request.text_input.has_value()) {
        raw_language = request.text_input->language;
    }
    resolved.language = granite_speech5_normalize_language_code(raw_language);
    // Translation target: options-only key (never from text_input — the
    // transcript text/languages there describe the SOURCE side). Blank or
    // absent means ASR.
    std::string raw_target;
    const auto target_hint = request.options.find("target_language");
    if (target_hint != request.options.end()) {
        raw_target = target_hint->second;
    }
    resolved.target_language = granite_speech5_normalize_target_language_code(raw_target);
    // Keyword biasing terms: the whole text_input.text channel,
    // comma-separated (CLI --text, JSON text, multipart prompt|text).
    if (request.text_input.has_value()) {
        resolved.keywords =
            granite_speech5_parse_keyword_terms(request.text_input->text);
    }

    if (const auto max_tokens = runtime::parse_int_option(request.options, {"max_tokens"})) {
        if (*max_tokens < kGraniteSpeech5MinMaxTokens || *max_tokens > kGraniteSpeech5MaxMaxTokens) {
            throw std::runtime_error(
                "granite_speech5 max_tokens must be in [1, 16383], got " +
                std::to_string(*max_tokens));
        }
        resolved.max_tokens = *max_tokens;
    }

    resolved.chunk_mode = engine::audio::parse_audio_chunk_mode(request.options);
    if (resolved.chunk_mode != engine::audio::AudioChunkMode::Auto &&
        resolved.chunk_mode != engine::audio::AudioChunkMode::Fixed &&
        resolved.chunk_mode != engine::audio::AudioChunkMode::Vad &&
        resolved.chunk_mode != engine::audio::AudioChunkMode::None) {
        throw std::runtime_error(
            "granite_speech5 audio_chunk_mode must be auto, fixed, vad, or none");
    }

    if (resolved.chunk_mode != engine::audio::AudioChunkMode::None) {
        const auto seconds = engine::audio::parse_audio_chunk_seconds_override(request.options);
        if (seconds.has_value()) {
            if (!std::isfinite(*seconds) || *seconds <= 0.0F ||
                *seconds > kGraniteSpeech5MaxChunkSeconds) {
                throw std::runtime_error(
                    "granite_speech5 audio_chunk_duration_sec must be positive finite <= 120");
            }
            resolved.chunk_duration_sec = *seconds;
        }
    }
    return resolved;
}

std::vector<runtime::TimeSpan> granite_speech5_plan_fixed_spans(
    int64_t total_samples, int64_t chunk_samples) {
    if (total_samples <= 0) {
        throw std::runtime_error("granite_speech5 fixed planner requires positive input samples");
    }
    if (chunk_samples <= 0) {
        throw std::runtime_error("granite_speech5 fixed planner requires a positive chunk size");
    }
    std::vector<runtime::TimeSpan> spans;
    int64_t start = 0;
    while (start < total_samples) {
        const int64_t end = std::min(total_samples, start + chunk_samples);
        spans.push_back(runtime::TimeSpan{start, end});
        start = end;
    }
    return spans;
}

std::vector<runtime::TimeSpan> granite_speech5_merge_tiny_tails(
    std::vector<runtime::TimeSpan> spans,
    int64_t chunk_samples,
    bool vad_single_tiny_ok) {
    if (chunk_samples <= 0) {
        throw std::runtime_error("granite_speech5 tail merge requires a positive chunk size");
    }
    for (const auto & span : spans) {
        if (span.end_sample <= span.start_sample) {
            throw std::runtime_error("granite_speech5 tail merge found an empty span");
        }
    }
    // Join tiny tails into a neighbor: next when one exists, else previous.
    for (size_t i = 0; i < spans.size();) {
        const int64_t length = spans[i].end_sample - spans[i].start_sample;
        if (length <= 0 || length > kGraniteSpeech5MinSamples) {
            ++i;
            continue;
        }
        if (spans.size() == 1) {
            if (vad_single_tiny_ok) {
                return {};
            }
            throw std::runtime_error(
                "granite_speech5 clip too short after resample; need more than 256 samples");
        }
        if (i + 1 < spans.size()) {
            spans[i + 1].start_sample = spans[i].start_sample;
        } else {
            spans[i - 1].end_sample = spans[i].end_sample;
        }
        spans.erase(spans.begin() + static_cast<ptrdiff_t>(i));
    }
    // Rebalance a last span left over the cap by its merged tail: split the
    // last two spans' combined region into near-equal parts. The combined
    // region exceeds the cap by at most one tiny tail (<=256 samples), so
    // each half overshoots the cap by at most ~128 samples; both halves stay
    // far above the tiny threshold.
    if (spans.size() >= 2) {
        const size_t last = spans.size() - 1;
        const int64_t last_length = spans[last].end_sample - spans[last].start_sample;
        if (last_length > chunk_samples) {
            const int64_t start = spans[last - 1].start_sample;
            const int64_t end = spans[last].end_sample;
            const int64_t first_half = (end - start) / 2;
            spans[last - 1].end_sample = start + first_half;
            spans[last].start_sample = start + first_half;
        }
    } else if (spans.size() == 1) {
        // Unreachable via the fixed/VAD planners (they cap every span), kept
        // as a backstop so no over-cap span escapes.
        const int64_t length = spans[0].end_sample - spans[0].start_sample;
        if (length > chunk_samples) {
            const int64_t parts = (length + chunk_samples - 1) / chunk_samples;
            std::vector<runtime::TimeSpan> split;
            int64_t start = spans[0].start_sample;
            for (int64_t part = 0; part < parts; ++part) {
                const int64_t remaining = spans[0].end_sample - start;
                const int64_t take = (remaining + (parts - part) - 1) / (parts - part);
                split.push_back(runtime::TimeSpan{start, start + take});
                start += take;
            }
            return split;
        }
    }
    return spans;
}

void granite_speech5_append_chunk_text(std::string & merged, std::string chunk_text) {
    chunk_text = engine::io::trim_ascii_whitespace(std::move(chunk_text));
    if (chunk_text.empty()) {
        return;
    }
    if (!merged.empty()) {
        const uint32_t last = decode_last_code_point(merged);
        const uint32_t first = decode_first_code_point(chunk_text);
        if (last < 0x80 && first < 0x80) {
            merged.push_back(' ');
        }
    }
    merged += chunk_text;
}

GraniteSpeech5Session::GraniteSpeech5Session(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const GraniteSpeech5Assets> assets,
    assets::TensorStorageType weight_storage)
    : RuntimeSessionBase(std::move(options))
    , task_(std::move(task))
    , assets_(require_assets(std::move(assets)))
    , frontend_(assets_)
    , encoder_(
          assets_,
          execution_context(),
          weight_storage,
          parse_arena_bytes(RuntimeSessionBase::options(), "encoder_graph_arena_mb"))
    , projector_(
          assets_,
          execution_context(),
          weight_storage,
          parse_arena_bytes(RuntimeSessionBase::options(), "projector_graph_arena_mb"))
    , prompt_builder_(load_granite_speech5_tokenizer(assets_->resources))
    , decoder_(
          assets_->source,
          make_granite_speech5_greedy_decoder_spec(execution_context().backend_type()),
          execution_context(),
          parse_arena_bytes(RuntimeSessionBase::options(), "prefill_graph_arena_mb"),
          parse_arena_bytes(RuntimeSessionBase::options(), "decode_graph_arena_mb"),
          kDecoderWeightContextBytes,
          weight_storage)
    , vad_model_path_(parse_vad_model_path(RuntimeSessionBase::options())) {
    if (task_.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Granite Speech 2B only supports VoiceTaskKind::Asr");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Granite Speech 2B only supports offline sessions");
    }
    reserved_markers_ = granite_speech5_reserved_markers_from_tokenizer_json(
        assets_->resources.require_file("tokenizer_json"));
    // All graph uploads are done; the reopenable source may drop storage.
    // Later sessions re-read it lazily from the checkpoint file.
    assets_->source->release_storage();
}

GraniteSpeech5Session::~GraniteSpeech5Session() = default;

std::string GraniteSpeech5Session::family() const {
    return "granite_speech5";
}

runtime::VoiceTaskKind GraniteSpeech5Session::task_kind() const {
    return task_.task;
}

runtime::RunMode GraniteSpeech5Session::run_mode() const {
    return task_.mode;
}

void GraniteSpeech5Session::prepare(const runtime::SessionPreparationRequest &) {
    mark_prepared();
}

runtime::IOfflineVoiceTaskSession & GraniteSpeech5Session::vad_session() {
    if (vad_session_ == nullptr) {
        runtime::ModelLoadRequest load_request;
        load_request.model_path = vad_model_path_;
        vad_model_ = runtime::load_vad_model(load_request);
        auto session = vad_model_->create_task_session(
            runtime::TaskSpec{runtime::VoiceTaskKind::Vad, runtime::RunMode::Offline},
            runtime::SessionOptions{options().backend, {}});
        auto * offline = dynamic_cast<runtime::IOfflineVoiceTaskSession *>(session.get());
        if (offline == nullptr) {
            throw std::runtime_error("Granite Speech 2B internal VAD session does not support offline execution");
        }
        session.release();
        vad_session_.reset(offline);
    }
    return *vad_session_;
}

std::string GraniteSpeech5Session::transcribe_span(
    const runtime::AudioBuffer & clip,
    const GraniteSpeech5RequestOptions & options) {
    auto started = std::chrono::steady_clock::now();
    const auto features = frontend_.extract(clip);
    debug::timing_log_scalar("granite_speech5.frontend_ms", debug::elapsed_ms(started));
    if (features.stacked.empty() || features.encoder_steps <= 0) {
        return "";
    }
    started = std::chrono::steady_clock::now();
    const auto encoder_output = encoder_.encode(features.stacked, features.encoder_steps);
    debug::timing_log_scalar("granite_speech5.encoder_ms", debug::elapsed_ms(started));
    started = std::chrono::steady_clock::now();
    auto projected =
        projector_.project(encoder_output.last, encoder_output.steps).projected;
    debug::timing_log_scalar("granite_speech5.projector_ms", debug::elapsed_ms(started));

    // Single builder entry for all modes (plain ASR = empty target +
    // empty terms): no caller can silently receive the wrong composer.
    auto decoder_prompt = prompt_builder_.build_decoder_prompt_with_context(
        options.language, options.target_language, options.keywords,
        features.lengths.audio_placeholders, std::move(projected), reserved_markers_);
    // Budget is enforced on the same IDs the decoder consumes (rebuilt from
    // the prompt so the check and the generate input cannot diverge).
    GraniteSpeech5PromptIds budgeted;
    budgeted.input_ids = decoder_prompt.input_ids;
    budgeted.audio_positions = decoder_prompt.injection.positions;
    budgeted.audio_tokens = decoder_prompt.injection.tokens;
    const GraniteSpeech5PromptBudget budget = granite_speech5_prompt_budget(budgeted);
    granite_speech5_require_prompt_fits(budget);
    granite_speech5_require_generation_budget(
        budget.prompt_len, options.max_tokens, kGraniteSpeech5MaxPositionEmbeddings);

    started = std::chrono::steady_clock::now();
    const std::vector<int32_t> generated = decoder_.generate(decoder_prompt, options.max_tokens, true);
    debug::timing_log_scalar("granite_speech5.decoder_ms", debug::elapsed_ms(started));
    const bool hit_cap = static_cast<int64_t>(generated.size()) >= options.max_tokens;
    engine::debug::trace_log_scalar(
        "granite_speech5.prompt_tokens", static_cast<int64_t>(budget.prompt_len));
    engine::debug::trace_log_scalar(
        "granite_speech5.generated_tokens", static_cast<int64_t>(generated.size()));
    engine::debug::trace_log_scalar("granite_speech5.stop_cap", hit_cap);
    return prompt_builder_.decode_generated(generated);
}

runtime::TaskResult GraniteSpeech5Session::run(const runtime::TaskRequest & request) {
    std::lock_guard<std::mutex> guard(run_mutex_);
    require_prepared("Granite Speech 2B run()");
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("Granite Speech 2B run() requires audio_input");
    }
    const auto wall_start = Clock::now();
    const GraniteSpeech5RequestOptions options = granite_speech5_resolve_request_options(request);

    const bool vad_available = engine::io::is_existing_file(vad_model_path_);
    if (options.chunk_mode == engine::audio::AudioChunkMode::Vad && !vad_available) {
        throw std::runtime_error(
            "Granite Speech 2B audio_chunk_mode 'vad' requires granite_speech5.vad_model_path "
            "to point at an existing VAD weights file");
    }

    const runtime::AudioBuffer & input = *request.audio_input;
    if (input.channels <= 0) {
        throw std::runtime_error("Granite Speech 2B run() requires a positive channel count");
    }
    if (input.sample_rate <= 0) {
        throw std::runtime_error("Granite Speech 2B run() requires a positive sample rate");
    }
    if (input.samples.size() % static_cast<size_t>(input.channels) != 0) {
        throw std::runtime_error("Granite Speech 2B interleaved samples do not match the channel count");
    }
    runtime::TaskResult empty;
    // Translation labels the TARGET language; ASR labels the source hint
    // ("" when auto). Keywords alone never change the label.
    const std::string output_language = options.target_language.empty()
        ? options.language
        : options.target_language;
    empty.text_output = runtime::Transcript{"", output_language};
    if (input.samples.empty()) {
        return empty;
    }
    for (float sample : input.samples) {
        if (!std::isfinite(sample)) {
            throw std::runtime_error("Granite Speech 2B run() requires finite amplitudes");
        }
    }

    // Single sample clock: downmix/resample once to 16 kHz mono; every span
    // below is addressed in these samples.
    runtime::AudioBuffer canonical;
    canonical.sample_rate = static_cast<int>(kCanonicalSampleRate);
    canonical.channels = 1;
    canonical.samples = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        input.samples, input.sample_rate, input.channels, static_cast<int>(kCanonicalSampleRate));
    if (canonical.samples.empty()) {
        return empty;
    }
    const int64_t total_samples = static_cast<int64_t>(canonical.samples.size());

    const int64_t chunk_samples = static_cast<int64_t>(std::llround(
        static_cast<double>(options.chunk_duration_sec) * static_cast<double>(kCanonicalSampleRate)));

    std::vector<runtime::TimeSpan> spans;
    if (options.chunk_mode == engine::audio::AudioChunkMode::None || total_samples <= chunk_samples) {
        // Short inputs (and explicit none) run whole; a lone tiny clip
        // cannot be transcribed and rejects loudly.
        spans = granite_speech5_merge_tiny_tails(
            {runtime::TimeSpan{0, total_samples}}, chunk_samples, false);
    } else if (
        options.chunk_mode == engine::audio::AudioChunkMode::Vad ||
        options.chunk_mode == engine::audio::AudioChunkMode::Auto) {
        const bool have_vad = vad_available;
        if (!have_vad && options.chunk_mode == engine::audio::AudioChunkMode::Vad) {
            throw std::runtime_error(
                "Granite Speech 2B audio_chunk_mode 'vad' requires granite_speech5.vad_model_path "
                "to point at an existing VAD weights file");
        }
        bool planned_with_vad = false;
        if (have_vad) {
            try {
                const auto vad_options = engine::audio::VadAudioChunkOptions{
                    chunk_samples,
                    static_cast<int64_t>(std::llround(kVadMergeGapSeconds * kCanonicalSampleRate)),
                    static_cast<int64_t>(std::llround(kVadPaddingSeconds * kCanonicalSampleRate)),
                };
                auto vad_spans =
                    engine::audio::plan_vad_audio_chunks(canonical, vad_session(), vad_options);
                spans = granite_speech5_merge_tiny_tails(std::move(vad_spans), chunk_samples, true);
                planned_with_vad = true;
            } catch (const std::exception &) {
                if (options.chunk_mode == engine::audio::AudioChunkMode::Vad) {
                    throw;
                }
                spans.clear();
            }
        }
        if (!planned_with_vad) {
            spans = granite_speech5_merge_tiny_tails(
                granite_speech5_plan_fixed_spans(total_samples, chunk_samples), chunk_samples, false);
        }
    } else {
        spans = granite_speech5_merge_tiny_tails(
            granite_speech5_plan_fixed_spans(total_samples, chunk_samples), chunk_samples, false);
    }

    if (spans.empty()) {
        // VAD found no speech (or only a lone blip).
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
        return empty;
    }

    for (const auto & term : options.keywords) {
        granite_speech5_require_no_reserved_markers(term, reserved_markers_);
    }

    std::string merged;
    for (const auto & span : spans) {
        const runtime::AudioBuffer clip = engine::audio::slice_audio_buffer(canonical, span);
        granite_speech5_append_chunk_text(merged, transcribe_span(clip, options));
    }
    runtime::TaskResult result;
    result.text_output =
        runtime::Transcript{engine::io::trim_ascii_whitespace(std::move(merged)), output_language};
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

GraniteSpeech5LoadedModel::GraniteSpeech5LoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const GraniteSpeech5Assets> assets,
    assets::TensorStorageType weight_storage)
    : metadata_(std::move(metadata))
    , capabilities_(std::move(capabilities))
    , assets_(std::move(assets))
    , weight_storage_(weight_storage) {}

const runtime::ModelMetadata & GraniteSpeech5LoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & GraniteSpeech5LoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> GraniteSpeech5LoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    if (task.task != runtime::VoiceTaskKind::Asr || task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Granite Speech 2B only supports offline ASR sessions");
    }
    return std::make_unique<GraniteSpeech5Session>(task, options, assets_, weight_storage_);
}

std::unique_ptr<GraniteSpeech5LoadedModel> load_granite_speech5_model(
    const runtime::ModelLoadRequest & request) {
    if (request.family_hint.has_value() && *request.family_hint != "granite_speech5") {
        throw std::runtime_error("Granite Speech 2B loader cannot load family '" + *request.family_hint + "'");
    }
    // Both native checkpoint directories and GGUF packages resolve through
    // load_granite_speech5_assets (spec-backed bundle for GGUF, manual
    // bundle for native). Mixed GGUF types are kept as stored under the
    // default native weight storage; explicit f32 upcasts.
    const auto weight_storage = parse_load_weight_storage(request.options);
    auto assets = load_granite_speech5_assets(request.model_path);

    runtime::ModelMetadata metadata;
    metadata.family = "granite_speech5";
    metadata.variant = "5.0-2b";
    metadata.description = "IBM Granite Speech 5.0 2B autoregressive multilingual ASR model.";

    runtime::CapabilitySet capabilities;
    capabilities.supported_tasks = {
        {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}},
    };
    capabilities.languages = {"en", "fr", "de", "es", "pt", "ja"};
    capabilities.supports_timestamps = false;

    return std::make_unique<GraniteSpeech5LoadedModel>(
        std::move(metadata), std::move(capabilities), std::move(assets), weight_storage);
}

namespace {

class GraniteSpeech5Loader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "granite_speech5";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}},
        };
        out.languages = {"en", "fr", "de", "es", "pt", "ja"};
        out.supports_timestamps = false;
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        // GGUF packages claim through the spec-backed bundle (a GGUF
        // embedding another family's spec, or with no usable GGUF source,
        // is rejected). Native directories keep the Task D config-triple
        // probe: the exact granite_speech5 conditional-generation triple.
        // TurboCTC/NAR snapshots stamped "granite_speech5" return false via
        // the encoder/projector/text triple check.
        if (is_gguf_path(request.model_path)) {
            try {
                (void) engine::model_spec::load_resource_bundle_for_family(request.model_path, family());
                return true;
            } catch (...) {
                return false;
            }
        }
        if (granite_speech5_can_load(request.model_path)) {
            return true;
        }
        // Installed package directory: a single GGUF, no native checkpoint.
        // Native weight files on disk disqualify the directory even when a
        // stray GGUF sits beside them (foreign native checkpoint case).
        if (granite_speech5_directory_has_native_weights(request.model_path)) {
            return false;
        }
        try {
            if (!engine::assets::find_directory_gguf(request.model_path).has_value()) {
                return false;
            }
            (void) engine::model_spec::load_resource_bundle_for_family(request.model_path, family());
            return true;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        const auto resources = engine::model_spec::load_resource_bundle(
            request.model_path, default_spec_path());
        runtime::ModelInspection inspection;
        inspection.cli.request_options = {
            {"language",
             "code",
             "Source language hint: en, fr, de, es, pt, or ja (case-insensitive). "
             "Blank means automatic recognition with no language label.",
             false,
             std::nullopt,
             std::nullopt,
             std::nullopt},
            {"max_tokens",
             "int",
             "Maximum generated tokens per audio chunk. Must fit prompt_len + max_tokens <= 16384. Default: 512.",
             false,
             "512",
             "1",
             "16383"},
            {"target_language",
             "code",
             "Translation target: en, fr, de, es, pt, ja, it, or zh (case-insensitive ASCII; "
             "Italian and Mandarin render as Italian and Mandarin in the prompt). "
             "Blank (or omitted) means ASR transcription; setting a target selects speech "
             "translation into that language. Source speech is described by language; "
             "keyword biasing terms travel in text_input.text (CLI --text, JSON text, "
             "multipart prompt|text) and combine with either mode.",
             false,
             std::nullopt,
             std::nullopt,
             std::nullopt},
            {"audio_chunk_mode",
             "auto|fixed|vad|none",
             "Offline audio chunking mode: auto (short inputs whole, long inputs VAD when a VAD model is "
             "configured, else fixed), fixed, vad (requires granite_speech5.vad_model_path), or none. Default: auto.",
             false,
             "auto",
             std::nullopt,
             std::nullopt},
            {"audio_chunk_duration_sec",
             "float",
             "Positive maximum audio chunk duration in seconds, at most 120. Ignored in none mode. Default: 120.",
             false,
             "120",
             std::nullopt,
             "120"},
        };
        inspection.cli.session_options = {
            {"granite_speech5.vad_model_path",
             "path",
             "Path to a VAD weight file of any built-in VAD family (e.g. silero_vad_16k.safetensors), "
             "not a directory; VAD runs only when this path is a regular file.",
             false,
             std::nullopt,
             std::nullopt,
             std::nullopt},
            {"granite_speech5.encoder_graph_arena_mb",
             "int",
             "Maximum encoder graph metadata budget in MiB; shape-sized allocation. Default: 1024.",
             false,
             "1024",
             "1",
             std::nullopt},
            {"granite_speech5.projector_graph_arena_mb",
             "int",
             "Maximum projector graph metadata budget in MiB; shape-sized allocation. Default: 1024.",
             false,
             "1024",
             "1",
             std::nullopt},
            {"granite_speech5.prefill_graph_arena_mb",
             "int",
             "Decoder prefill graph arena size in MB. Must be positive. Default: 1024.",
             false,
             "1024",
             "1",
             std::nullopt},
            {"granite_speech5.decode_graph_arena_mb",
             "int",
             "Decoder step graph arena size in MB. Must be positive. Default: 1024.",
             false,
             "1024",
             "1",
             std::nullopt},
        };
        inspection.cli.load_options = {
            {"granite_speech5.weight_type",
             "native|f32",
             "Shared model weight storage type. Native keeps the stored dtype (native checkpoint or GGUF "
             "mixed package types); f32 upcasts for parity. "
             "Default: native.",
             false,
             "native",
             std::nullopt,
             std::nullopt},
        };
        inspection.model_root = resources.model_root();
        inspection.metadata.family = family();
        inspection.metadata.variant = "5.0-2b";
        inspection.metadata.description = "IBM Granite Speech 5.0 2B autoregressive multilingual ASR model.";
        inspection.capabilities = advertised_capabilities();
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path, default_spec_path(), engine::model_spec::ResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path, default_spec_path(), engine::model_spec::ResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_granite_speech5_model(request);
    }
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_granite_speech5_loader() {
    return std::make_shared<GraniteSpeech5Loader>();
}

}  // namespace engine::community_models::granite_speech5
