#include "engine/models/soulx_singer/runtime.h"
#include "engine/models/soulx_singer/routing.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/mel_spectrogram_frontend.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/speech_encoders/whisper_frontend.h"
#include "engine/framework/modules/pitch_extractors/rmvpe_pitch_extractor.h"
#include "engine/framework/modules/vocoders/vocos_vocoder.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/framework/runtime/cache_slots.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <stdexcept>

namespace engine::models::soulx_singer {
using core::TensorShape;
using core::TensorValue;

std::shared_ptr<const ModelAssets> load_assets(const std::filesystem::path &path) {
    auto a = std::make_shared<ModelAssets>();
    a->resources = model_spec::load_resource_bundle_for_family(path, "soulx_singer");
    const auto config = a->resources.parse_json("config");
    a->impl = &implementation_for_task(
        runtime::parse_voice_task_kind(config.require("task").as_string()));
    a->layers = config.require("layers").as_i64();
    a->hidden = config.require("hidden_size").as_i64();
    a->intermediate = config.require("intermediate_size").as_i64();
    a->heads = config.require("heads").as_i64();
    if (a->layers <= 0 || a->hidden <= 0 || a->intermediate <= 0 || a->heads <= 0 || a->hidden % a->heads)
        throw std::runtime_error("Invalid SoulX-Singer architecture");
    a->model = a->resources.open_tensor_source("model_weights");
    a->vocos = a->resources.open_tensor_source("vocos_weights");
    a->impl->load_conditioning_assets(*a);
    return a;
}

namespace {
modules::DitRuntimeConfig dit_config(const ModelAssets &a, assets::TensorStorageType storage,
                                     core::AttentionPreference preference,
                                     core::ExecutionContext &execution) {
    modules::DitRuntimeConfig c;
    c.label = "soulx_singer.dit";
    c.weight_type = storage;
    c.model.input_channels = c.model.output_channels = 128;
    c.model.hidden_size = a.hidden;
    c.model.intermediate_size = a.intermediate;
    c.model.layers = a.layers;
    c.model.norm_eps = 1e-6f;
    c.model.modulation = modules::DitModulation::AdaptiveRms;
    c.model.normalization = modules::DitNormalization::Rms;
    c.model.feed_forward = modules::DitFeedForward::SwiGlu;
    c.model.additive_frame_conditioning = true;
    auto &att = c.model.self_attention;
    att.heads = att.kv_heads = a.heads;
    att.head_dim = a.hidden / a.heads;
    att.rotary.mode = GGML_ROPE_TYPE_NEOX;
    att.rotary.theta = 10000;
    att.rotary.dimensions = att.head_dim;
    att.lowering = core::resolve_flash_attention(execution.backend(), att.head_dim, preference)
                       ? modules::ScaledDotProductAttentionLowering::Flash
                       : modules::ScaledDotProductAttentionLowering::Explicit;
    c.model.time_features = a.hidden;
    c.time_features = modules::DitTimeFeatures::SinCos;
    c.time_frequency_denominator = a.hidden / 2 - 1;
    c.guidance = modules::DitGuidance::Rescaled;
    c.guidance_rescale = 0.75f;
    return c;
}

modules::DitWeightNames dit_weight_names() {
    modules::DitWeightNames n;
    const auto linear = [](const std::string &name) {
        return modules::DitLinearNames{name + ".weight", name + ".bias"};
    };
    for (const auto *name : {"diff_step_mlp.0", "diff_step_mlp.2"})
        n.time_embedding.push_back(linear(name));
    for (const auto *name : {"mel_mlp.0", "mel_mlp.2"})
        n.input_embedding.push_back(linear(name));
    for (const auto *name : {"cond_mlp.0", "cond_mlp.2"})
        n.frame_embedding.push_back(linear(name));
    for (const auto *name : {"mel_out_mlp.0", "mel_out_mlp.2"})
        n.output_embedding.push_back(linear(name));
    const std::string p = "layers.{layer}.";
    n.layer.self_attention.query.weight = p + "self_attn.q_proj.weight";
    n.layer.self_attention.key.weight = p + "self_attn.k_proj.weight";
    n.layer.self_attention.value.weight = p + "self_attn.v_proj.weight";
    n.layer.self_attention.output.weight = p + "self_attn.o_proj.weight";
    n.layer.feed_forward_in.weight = p + "mlp.in.weight";
    n.layer.feed_forward_out.weight = p + "mlp.down_proj.weight";
    n.layer.self_scale = linear(p + "input_layernorm.to_weight");
    n.layer.feed_forward_scale = linear(p + "post_attention_layernorm.to_weight");
    n.final_modulation = linear("norm.to_weight");
    return n;
}
} // namespace

class SoulXSingerRuntime::Impl {
  public:
    std::shared_ptr<const ModelAssets> assets;
    core::ExecutionContext &execution;
    ConditioningComponents components;
    modules::VocosVocoder vocos;
    modules::DitRuntime dit;
    core::BackendWeightStore store;
    TensorValue f0_embedding;
    modules::LinearWeights projection;
    struct ConditioningGraph {
        std::unique_ptr<core::PreparedGraph> graph;
        TensorValue content, f0, output;
    };
    runtime::CacheSlots<int64_t, ConditioningGraph> conditioning{2};
    std::optional<EncodedAudio> reference;

    Impl(std::shared_ptr<const ModelAssets> a, core::ExecutionContext &e, assets::TensorStorageType storage,
         core::AttentionPreference attention)
        : assets(std::move(a)), execution(e),
          vocos(assets->vocos, e.backend(), e.config().threads, {128, 1920, 480, false}),
          dit(e, *assets->model, dit_config(*assets, storage, attention, e), dit_weight_names()),
          store(e.backend(), e.backend_type(), "soulx_singer.conditioning", 1024 * 1024) {
        components = assets->impl->create_conditioning(*assets, e);
        f0_embedding = store.load_f32_tensor(*assets->model, "f0_encoder.weight", {361, 512});
        projection = modules::binding::linear_from_source(store, *assets->model, "cond_emb",
                                                          assets::TensorStorageType::F32, 1024, 512, true);
        store.upload();
    }

    std::vector<float> project(const std::vector<float> &content, const std::vector<float> &f0, int shift) {
        const auto frames = static_cast<int64_t>(f0.size());
        int64_t capacity = 32;
        while (capacity < frames)
            capacity *= 2;
        auto *slot = conditioning.find(capacity);
        if (!slot) {
            conditioning.put(capacity, ConditioningGraph{});
            slot = conditioning.find(capacity);
        }
        auto &g = *slot;
        if (!g.graph) {
            g.graph = std::make_unique<core::PreparedGraph>(
                execution, core::PreparedGraphConfig{"soulx_singer.conditioning", 1024 * 1024, 256});
            auto &ctx = g.graph->build_context();
            g.content = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({capacity, 512}));
            g.f0 = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({capacity}));
            ggml_set_input(g.content.tensor);
            ggml_set_input(g.f0.tensor);
            const auto embedded = modules::EmbeddingModule({361, 512}).build(ctx, g.f0, f0_embedding);
            g.output = modules::LinearModule({512, 1024, true})
                           .build(ctx, modules::AddModule().build(ctx, g.content, embedded), projection);
            ggml_set_output(g.output.tensor);
            ggml_build_forward_expand(g.graph->graph(), g.output.tensor);
            g.graph->allocate();
        }
        std::vector<float> padded(static_cast<size_t>(capacity * 512), 0);
        std::copy_n(content.begin(), std::min(content.size(), static_cast<size_t>(frames * 512)),
                    padded.begin());
        std::vector<int32_t> ids(static_cast<size_t>(capacity), 0);
        for (int64_t i = 0; i < frames; ++i) {
            const float x = f0[static_cast<size_t>(i)];
            if (x > 0) {
                const auto coarse = std::clamp<int64_t>(
                    static_cast<int64_t>(
                        std::nearbyint(60 * std::log2(std::max(x, 32.7031956625f) / 32.7031956625f) + 1)),
                    1, 360);
                ids[static_cast<size_t>(i)] =
                    static_cast<int32_t>(std::clamp<int64_t>(coarse + shift * 5, 1, 360));
            }
        }
        core::write_tensor_f32(g.content, padded);
        ggml_backend_tensor_set(g.f0.tensor, ids.data(), 0, ids.size() * sizeof(int32_t));
        g.graph->compute();
        auto result = core::read_tensor_f32(g.output.tensor);
        result.resize(static_cast<size_t>(frames * 1024));
        return result;
    }

    std::vector<float> pitch_contour(const runtime::AudioBuffer &input) {
        const auto mono = audio::mixdown_interleaved_to_mono_average(input.samples, input.channels);
        const auto waveform = audio::try_resample_mono_soxr(
            mono, input.sample_rate, 16000,
            {audio::SoxrResampleProfile::QualityOnly, audio::SoxrOutputLengthPolicy::ExactExpected});
        if (!waveform)
            throw std::runtime_error("SoulX-Singer-SVC F0 preprocessing requires libsoxr");
        const auto raw =
            components.pitch->infer_16k_mono(*waveform, 0.03f, static_cast<size_t>(execution.config().threads));
        const auto frames =
            static_cast<size_t>(std::ceil(static_cast<double>(mono.size()) / input.sample_rate * 50));
        std::vector<float> out(frames, 0);
        for (size_t i = 0; i < frames && i * 2 < raw.size(); ++i)
            out[i] = raw[i * 2];
        return out;
    }

    audio::MelSpectrogramFeatures reference_mel(const std::vector<float> &waveform) const {
        audio::MelSpectrogramFrontendConfig config;
        config.sample_rate = 24000;
        config.n_fft = config.win_length = 1920;
        config.hop_length = 480;
        config.n_mels = 128;
        config.mel_fmax = 12000;
        config.magnitude_epsilon = 1e-9f;
        config.layout = audio::MelOutputLayout::TimeMajor;
        auto mel = audio::get_cached_mel_spectrogram_frontend(config)->extract_mono(
            waveform, static_cast<size_t>(execution.config().threads));
        for (auto &value : mel.values)
            value = (value + 4.92f) / std::sqrt(8.14f);
        return mel;
    }

    EncodedAudio encode(const runtime::AudioBuffer &input, bool prompt,
                        const std::vector<float> *supplied_f0 = nullptr) {
        if (!components.whisper || !components.pitch)
            throw std::logic_error("Audio conditioning is only available for SVC");
        EncodedAudio out;
        out.waveform = audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
            input.samples, input.sample_rate, input.channels, 24000,
            audio::torchaudio_sinc_hann_float32_options());
        if (out.waveform.size() < 1920 ||
            !std::all_of(out.waveform.begin(), out.waveform.end(), [](float x) { return std::isfinite(x); }))
            throw std::invalid_argument("SoulX-Singer-SVC needs finite audio at least 80 ms long");
        if (prompt && reference && reference->waveform == out.waveform)
            return *reference;
        auto at16 = audio::resample_mono_torchaudio_sinc_hann(out.waveform, 24000, 16000,
                                                              audio::torchaudio_sinc_hann_float32_options());
        out.frames = static_cast<int64_t>((out.waveform.size() + 479) / 480);
        out.f0 = supplied_f0 ? *supplied_f0 : pitch_contour(input);
        out.f0.resize(static_cast<size_t>(out.frames), 0);
        audio::MelSpectrogramFrontendConfig c;
        c.sample_rate = 16000;
        c.n_fft = c.win_length = 400;
        c.hop_length = 160;
        c.n_mels = 80;
        c.mel_fmax = 8000;
        c.stft_center = true;
        c.waveform_padding = audio::MelWaveformPadding::None;
        c.spectrum_mode = audio::MelSpectrumMode::PowerBeforeProjection;
        c.value_transform = audio::MelValueTransform::Log10;
        c.log_floor = 1e-10;
        c.log_dynamic_range = 8;
        c.log_shift = 4;
        c.log_divisor = 4;
        c.drop_last_frames = 1;
        const auto features = audio::get_cached_mel_spectrogram_frontend(c)->extract_mono(
            at16, static_cast<size_t>(execution.config().threads));
        std::vector<float> padded(80 * 3000, 0);
        const auto frames = std::min<int64_t>(features.frames, 3000);
        for (int64_t m = 0; m < 80; ++m)
            std::copy_n(features.values.begin() + m * features.frames, frames, padded.begin() + m * 3000);
        out.content = components.whisper->encode_log_mel(padded);
        out.content.resize(static_cast<size_t>(out.frames * 512), 0);
        if (prompt) {
            auto mel = reference_mel(out.waveform);
            out.frames = mel.frames;
            out.f0.resize(static_cast<size_t>(out.frames));
            out.content.resize(static_cast<size_t>(out.frames * 512), 0);
            out.mel = std::move(mel.values);
            reference = out;
        }
        return out;
    }
};

SoulXSingerRuntime::SoulXSingerRuntime(std::shared_ptr<const ModelAssets> a, core::ExecutionContext &e,
                                       assets::TensorStorageType storage, core::AttentionPreference attention)
    : impl_(std::make_unique<Impl>(std::move(a), e, storage, attention)) {}
SoulXSingerRuntime::~SoulXSingerRuntime() = default;
EncodedAudio SoulXSingerRuntime::encode(const runtime::AudioBuffer &a, bool reference) {
    return impl_->encode(a, reference);
}
std::pair<EncodedAudio, EncodedAudio>
SoulXSingerRuntime::encode_scores(const SingingScore &reference, const SingingScore &target, int shift,
                                  bool melody, const runtime::AudioBuffer &audio_input) {
    if (!impl_->components.score)
        throw std::logic_error("Score conditioning is only available for SVS");
    if (melody && (reference.f0.empty() || target.f0.empty()))
        throw std::invalid_argument("Melody control requires reference and target F0 contours");
    SingingScore joined = reference;
    const auto prefix_notes = static_cast<int32_t>(reference.phonemes.size());
    joined.phonemes.insert(joined.phonemes.end(), target.phonemes.begin(), target.phonemes.end());
    joined.pitches.insert(joined.pitches.end(), target.pitches.begin(), target.pitches.end());
    joined.types.insert(joined.types.end(), target.types.begin(), target.types.end());
    for (auto note : target.frame_notes)
        joined.frame_notes.push_back(note + prefix_notes);
    const auto content = impl_->components.score->encode(joined, shift, melody);
    EncodedAudio prompt, source;
    prompt.frames = static_cast<int64_t>(reference.frame_notes.size());
    source.frames = static_cast<int64_t>(target.frame_notes.size());
    prompt.f0 = melody ? reference.f0 : std::vector<float>(static_cast<size_t>(prompt.frames), 0);
    source.f0 = melody ? target.f0 : std::vector<float>(static_cast<size_t>(source.frames), 0);
    prompt.content.assign(content.begin(), content.begin() + prompt.frames * 512);
    source.content.assign(content.begin() + prompt.frames * 512, content.end());
    prompt.waveform = audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
        audio_input.samples, audio_input.sample_rate, audio_input.channels, 24000,
        audio::torchaudio_sinc_hann_float32_options());
    if (prompt.waveform.size() < static_cast<size_t>(prompt.frames * 480))
        throw std::invalid_argument("Reference recording is shorter than its score");
    prompt.waveform.resize(static_cast<size_t>(prompt.frames * 480));
    prompt.mel = impl_->reference_mel(prompt.waveform).values;
    if (prompt.mel.size() != static_cast<size_t>(prompt.frames * 128))
        throw std::runtime_error("Reference score and mel frames differ");
    return {std::move(prompt), std::move(source)};
}
void SoulXSingerRuntime::prepare_conditioning(const EncodedAudio &reference, const EncodedAudio &source,
                                              int shift) {
    modules::DitConditioning c;
    c.frames = reference.frames + source.frames;
    modules::DitFixedPrefixConditioning prefix;
    prefix.frames = static_cast<int64_t>(reference.mel.size() / 128);
    c.branches = 2;
    prefix.latent = reference.mel;
    c.frame_channels = {1024};
    auto features = impl_->project(reference.content, reference.f0, 0);
    const auto target = impl_->project(source.content, source.f0, shift);
    features.insert(features.end(), target.begin(), target.end());
    features.resize(static_cast<size_t>(2 * c.frames * 1024), 0);
    c.frame_features.push_back(std::move(features));
    prefix.key_mask.assign(static_cast<size_t>(2 * c.frames), 1);
    std::fill_n(prefix.key_mask.begin() + c.frames, reference.frames, 0);
    impl_->dit.prepare_conditioning(std::move(c), std::move(prefix));
}
std::vector<float> SoulXSingerRuntime::denoise(const std::vector<float> &noise, const GenerationOptions &o) {
    modules::DitSamplingRequest r;
    r.noise = noise;
    r.steps = o.steps;
    r.guidance = o.guidance + 1;
    for (int64_t i = 0; i <= o.steps; ++i)
        r.schedule.push_back((static_cast<float>(i) + 0.5f) / static_cast<float>(o.steps));
    return impl_->dit.sample(r);
}
std::vector<float> SoulXSingerRuntime::decode(const std::vector<float> &mel) {
    std::vector<float> rows(mel.size());
    const auto frames = mel.size() / 128;
    for (size_t t = 0; t < frames; ++t)
        for (size_t c = 0; c < 128; ++c)
            rows[t * 128 + c] = mel[c * frames + t];
    return impl_->vocos.decode(rows);
}

runtime::AudioBuffer SoulXSingerRuntime::convert(const runtime::AudioBuffer &input,
                                                 const runtime::AudioBuffer &voice,
                                                 const GenerationOptions &o) {
    const auto reference = encode(voice, true);
    const auto samples = audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
        input.samples, input.sample_rate, input.channels, 24000,
        audio::torchaudio_sinc_hann_float32_options());
    const auto full_f0 = impl_->pitch_contour(input);
    int shift = o.pitch_shift;
    if (o.auto_pitch_shift && shift == 0) {
        const auto median = [](const std::vector<float> &x) {
            auto v = x;
            v.erase(std::remove_if(v.begin(), v.end(), [](float f) { return f <= 0; }), v.end());
            if (v.empty())
                throw std::invalid_argument("SoulX-Singer-SVC automatic pitch shift requires voiced audio");
            std::sort(v.begin(), v.end());
            return v[(v.size() - 1) / 2];
        };
        shift = static_cast<int>(std::nearbyint(12 * std::log2(median(reference.f0) / median(full_f0))));
    }
    struct Segment {
        size_t context_begin, end, begin;
    };
    std::vector<Segment> segments;
    if (samples.size() < 30 * 24000)
        segments.push_back({0, full_f0.size(), 0});
    else {
        std::vector<size_t> points{0};
        const auto split = [&](size_t point) {
            while (point - points.back() > 1500)
                points.push_back(points.back() + 1500);
            if (point > points.back())
                points.push_back(point);
        };
        for (size_t i = 0; i < full_f0.size();) {
            if (full_f0[i] != 0) {
                ++i;
                continue;
            }
            const auto start = i;
            while (i < full_f0.size() && full_f0[i] == 0)
                ++i;
            if (i - start >= 10)
                split(std::max(i >= 5 ? i - 5 : size_t{0}, (start + i) / 2));
        }
        split(full_f0.size());
        for (size_t begin = 0; begin + 1 < points.size();) {
            auto end = begin + 1;
            while (end < points.size() && points[end] - points[begin] < 750)
                ++end;
            if (end >= points.size())
                end = points.size() - 1;
            const auto context = begin > 0 && points[end] - points[begin - 1] <= 1500 ? begin - 1 : begin;
            const auto first = points[context], last = points[end];
            const auto voiced = static_cast<size_t>(std::count_if(
                full_f0.begin() + static_cast<std::ptrdiff_t>(first),
                full_f0.begin() + static_cast<std::ptrdiff_t>(last), [](float f) { return f > 0; }));
            if (voiced >= 10 && double(voiced) / double(last - first) > 0.05)
                segments.push_back({first, last, points[begin]});
            begin = end;
        }
        if (segments.empty())
            throw std::invalid_argument("SoulX-Singer-SVC source contains no voiced segments");
    }
    runtime::AudioBuffer result{24000, 1, std::vector<float>(samples.size(), 0)};
    uint64_t offset = 0;
    for (const auto &segment : segments) {
        const auto start = segment.context_begin * 480, end = std::min(samples.size(), segment.end * 480);
        const std::vector<float> f0(full_f0.begin() + static_cast<std::ptrdiff_t>(segment.context_begin),
                                    full_f0.begin() + static_cast<std::ptrdiff_t>(segment.end));
        const auto source = impl_->encode({24000,
                                           1,
                                           {samples.begin() + static_cast<std::ptrdiff_t>(start),
                                            samples.begin() + static_cast<std::ptrdiff_t>(end)}},
                                          false, &f0);
        prepare_conditioning(reference, source, shift);
        auto noise = sampling::generate_torch_cuda_randn(static_cast<size_t>(source.frames * 128), o.seed,
                                                         sampling::TorchRandnPrecision::Float32, offset);
        offset += noise.size();
        auto wave = decode(denoise(noise, o));
        if (wave.size() < end - start)
            throw std::runtime_error("Vocos returned fewer samples than the SVC segment");
        wave.resize(end - start);
        const auto destination = segment.begin * 480, skip = destination - start;
        std::copy(wave.begin() + static_cast<std::ptrdiff_t>(skip), wave.end(),
                  result.samples.begin() + static_cast<std::ptrdiff_t>(destination));
    }
    return result;
}
} // namespace engine::models::soulx_singer
