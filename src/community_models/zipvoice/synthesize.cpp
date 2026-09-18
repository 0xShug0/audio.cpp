#include "engine/community_models/zipvoice/synthesize.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/modules/vocoders/vocos_vocoder.h"

#include "engine/community_models/zipvoice/emilia_tokenizer.h"
#include "engine/community_models/zipvoice/weights.h"
#include "engine/community_models/zipvoice/zipformer.h"

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/audio/espeak_phonemizer.h"
#include "engine/framework/audio/conversion.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>
#include <cctype>

namespace engine::models::zipvoice {
namespace {

// ---------------------------------------------------------------------------
// log-mel (VocosFbank parity: torchaudio MelSpectrogram 24k/1024/256/100,
// power=1, htk scale, f_max=12000, log clamp 1e-7, lhotse frame alignment)

constexpr float kPi = 3.14159265358979323846F;
constexpr int kSampleRate = 24000;
constexpr int kNfft = 1024;
constexpr int kHop = 256;
constexpr int kNMel = 100;
constexpr float kFMin = 0.0F;
constexpr float kFMax = 12000.0F;

float hz_to_mel_htk(float hz) { return 2595.0F * std::log10(1.0F + hz / 700.0F); }
float mel_to_hz_htk(float mel) { return 700.0F * (std::pow(10.0F, mel / 2595.0F) - 1.0F); }

std::vector<float> mel_filterbank() {
    std::vector<float> fb;
    {
        const int n_freqs = kNfft / 2 + 1;
        const float m_min = hz_to_mel_htk(kFMin);
        const float m_max = hz_to_mel_htk(kFMax);
        std::vector<float> mels(kNMel + 2);
        for (int i = 0; i < kNMel + 2; ++i) {
            mels[i] = mel_to_hz_htk(m_min + (m_max - m_min) * i / (kNMel + 1));
        }
        fb.assign(static_cast<size_t>(n_freqs) * kNMel, 0.0F);
        for (int m = 0; m < kNMel; ++m) {
            const float lo = mels[m];
            const float mid = mels[m + 1];
            const float hi = mels[m + 2];
            for (int f = 0; f < n_freqs; ++f) {
                const float freq = static_cast<float>(f) * kSampleRate / kNfft;
                if (freq <= lo || freq >= hi) continue;
                const float w = freq <= mid
                    ? (freq - lo) / (mid - lo)
                    : (hi - freq) / (hi - mid);
                fb[static_cast<size_t>(f) * kNMel + m] = w;
            }
        }
    }
    return fb;
}

void fft_inplace(std::vector<float> & re, std::vector<float> & im, bool inverse) {
    const size_t n = re.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const float ang = static_cast<float>(2.0 * kPi / static_cast<double>(len)) * (inverse ? 1 : -1);
        for (size_t i = 0; i < n; i += len) {
            for (size_t k = 0; k < len / 2; ++k) {
                const float wr = std::cos(ang * static_cast<float>(k));
                const float wi = std::sin(ang * static_cast<float>(k));
                const size_t a = i + k;
                const size_t b = i + k + len / 2;
                const float vr = re[b] * wr - im[b] * wi;
                const float vi = re[b] * wi + im[b] * wr;
                const float ur = re[a];
                const float ui = im[a];
                re[a] = ur + vr;
                im[a] = ui + vi;
                re[b] = ur - vr;
                im[b] = ui - vi;
            }
        }
    }
    if (inverse) {
        for (size_t i = 0; i < n; ++i) {
            re[i] /= static_cast<float>(n);
            im[i] /= static_cast<float>(n);
        }
    }
}

}  // namespace

std::vector<float> zipvoice_logmel(const std::vector<float> & wav) {
    if (wav.size() <= kNfft / 2) {
        throw std::invalid_argument("zipvoice: reference audio must contain more than 512 samples at 24 kHz");
    }
    const int n_freqs = kNfft / 2 + 1;
    // lhotse compute_num_frames: (samples + hop/2) / hop
    const int frames = std::max<int>(1, (static_cast<int>(wav.size()) + kHop / 2) / kHop);
    std::vector<float> hann(kNfft);
    for (int i = 0; i < kNfft; ++i) {
        hann[i] = 0.5F * (1.0F - std::cos(2.0F * static_cast<float>(kPi) * i / kNfft));
    }
    const auto & fb = mel_filterbank();
    std::vector<float> mel(static_cast<size_t>(frames) * kNMel);  // [t, mel] row-major
    std::vector<float> re(kNfft), im(kNfft), mag(n_freqs);
    for (int t = 0; t < frames; ++t) {
        std::fill(re.begin(), re.end(), 0.0F);
        std::fill(im.begin(), im.end(), 0.0F);
        const int start = (t - 2) * kHop;  // torchaudio center=True alignment
        for (int i = 0; i < kNfft; ++i) {
            int r = start + i;
            if (r < 0) r = -r;
            if (r >= static_cast<int>(wav.size())) r = 2 * static_cast<int>(wav.size()) - 2 - r;
            r = std::clamp(r, 0, static_cast<int>(wav.size()) - 1);
            re[i] = wav[static_cast<size_t>(r)] * hann[i];
        }
        fft_inplace(re, im, false);
        for (int f = 0; f < n_freqs; ++f) {
            mag[f] = std::sqrt(re[f] * re[f] + im[f] * im[f]);
        }
        for (int m = 0; m < kNMel; ++m) {
            float acc = 0.0F;
            for (int f = 0; f < n_freqs; ++f) {
                acc += fb[static_cast<size_t>(f) * kNMel + m] * mag[f];
            }
            mel[static_cast<size_t>(t) * kNMel + m] = std::log(std::max(acc, 1e-7F));
        }
    }
    return mel;
}

std::vector<float> zipvoice_vocos_decode(const std::string & path, const std::vector<float> & mel) {
    const auto free_backend = [](ggml_backend_t b) { ggml_backend_free(b); };
    std::unique_ptr<ggml_backend, decltype(free_backend)> backend(
        core::init_backend({core::BackendType::Cpu, 0, 4}), free_backend);
    modules::VocosVocoder vocos(path, backend.get(), 4);
    return vocos.decode(mel);
}

namespace {

// ---------------------------------------------------------------------------
// Session/device-owned weights and bounded runtime graph slots

std::unordered_map<std::string, int32_t> load_tokens(const std::filesystem::path & path);

struct LoadedModel {
    std::string path;
    ZipVoiceConfig config;
    ZipVoiceWeights weights;
    ggml_backend_t backend = nullptr;
    bool owns_backend = true;  // false when the backend was injected by the caller
    core::BackendType backend_type = core::BackendType::Cpu;  // concrete resolved type
    std::mutex mutex;  // serializes graph use (graphs are not thread-safe)
    runtime::CacheSlots<int64_t, std::unique_ptr<TextEncoderGraph>> text_graphs{1};
    struct FmKey {
        int64_t t;
        int64_t b;
        bool guidance;
        bool operator==(const FmKey & o) const { return t == o.t && b == o.b && guidance == o.guidance; }
    };
    runtime::CacheSlots<FmKey, std::unique_ptr<FmDecoderGraph>> fm_graphs{1};
    runtime::CacheSlots<std::vector<std::string>, std::unique_ptr<EmiliaTokenizer>> tokenizers{1};
    std::unique_ptr<modules::VocosVocoder> vocos_graph;
    std::string vocos_path;
    ~LoadedModel() {
        vocos_graph.reset();
        text_graphs.clear();
        fm_graphs.clear();
        weights = {};
        if (owns_backend) {
            ggml_backend_free(backend);
        }
    }
};

std::vector<float> decode_vocos(LoadedModel & model, const std::string & path,
                                const std::vector<float> & mel, int threads) {
    if (!model.vocos_graph || model.vocos_path != path) {
        model.vocos_graph.reset();
        model.vocos_graph = std::make_unique<modules::VocosVocoder>(path, model.backend, threads);
        model.vocos_path = path;
    }
    return model.vocos_graph->decode(mel);
}

// Sidecar resolution: spec-registered resources (GGUF-embedded sidecars
// materialized by the framework, or files found in the development directory)
// take priority; loose files next to the checkpoint remain the fallback for
// direct API entry points (parity harnesses) that pass no bundle.
struct SidecarResolver {
    const engine::assets::ResourceBundle * resources = nullptr;

    std::filesystem::path path(std::string_view id, const std::filesystem::path & dir,
                               std::string_view filename) const {
        if (resources != nullptr) {
            if (const auto * p = resources->find_file(id)) return *p;
        }
        return dir / filename;
    }
};

// Resolve the requested backend: an injected backend wins; otherwise the
// device's backend type (BestAvailable by default = GPU first, CPU fallback).
core::BackendType resolve_backend_type(const ZipVoiceComputeDevice & device) {
    if (device.backend != nullptr) {
        return core::backend_type(device.backend);
    }
    return device.backend_type;
}

} // namespace

class ZipVoiceRuntimeState {
public:
    std::mutex mutex;
    std::shared_ptr<LoadedModel> model;
    std::string key;
};

std::shared_ptr<ZipVoiceRuntimeState> make_zipvoice_runtime_state() {
    return std::make_shared<ZipVoiceRuntimeState>();
}

namespace {
std::shared_ptr<LoadedModel> load_model(
    const std::string & path, const SidecarResolver & sidecars,
    const ZipVoiceComputeDevice & device) {
    if (!device.runtime) throw std::invalid_argument("zipvoice: runtime is null");
    auto & state = *device.runtime;
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto cache_config_dir = std::filesystem::is_directory(path)
        ? std::filesystem::path(path) : std::filesystem::path(path).parent_path();
    const std::string key = path + "::" + sidecars.path("model_config", cache_config_dir, "model.json").string() +
        "::" + sidecars.path("tokens", cache_config_dir, "tokens.txt").string() +
        "::" + std::to_string(static_cast<int>(resolve_backend_type(device))) +
        "::" + std::to_string(device.device_index) + "::" + std::to_string(reinterpret_cast<uintptr_t>(device.backend));
    if (state.model && state.key == key) return state.model;
    state.model.reset();
    auto model = std::make_unique<LoadedModel>();
    model->path = path;
    namespace fs = std::filesystem;
    const fs::path p(path);
    std::string prefix = "";
    std::shared_ptr<const engine::assets::TensorSource> source;
    if (p.extension() == ".gguf") {
        source = engine::assets::make_prefixed_tensor_source(
            engine::assets::open_tensor_source(p), "model");
    } else {
        // development layout: directory with <name>-orig.safetensors
        std::vector<fs::path> candidates;
        if (fs::is_regular_file(p) && p.extension() == ".safetensors") {
            candidates.push_back(p);
        } else if (fs::is_directory(p)) {
            for (const auto & entry : fs::directory_iterator(p)) {
                if (entry.path().extension() == ".safetensors" &&
                    entry.path().filename().string().find("vocos") == std::string::npos) {
                    candidates.push_back(entry.path());
                }
            }
        }
        if (candidates.empty()) {
            throw std::runtime_error("zipvoice: no safetensors checkpoint in " + path);
        }
        std::sort(candidates.begin(), candidates.end());
        source = engine::assets::open_tensor_source(candidates.back());
    }
    const fs::path config_dir = fs::is_directory(p) ? p : p.parent_path();
    const auto config_path = sidecars.path("model_config", config_dir, "model.json");
    model->config = load_zipvoice_config(config_dir, source.get(), &config_path);
    // vocab size from the embedding itself
    model->config.vocab_size = static_cast<int>(
        source->require_metadata("embed.weight").shape[0]);
    // pad id from tokens.txt (the "_" entry per the reference tokenizers)
    {
        const auto vocab = load_tokens(sidecars.path("tokens", config_dir, "tokens.txt"));
        const auto it = vocab.find("_");
        if (it != vocab.end()) {
            model->config.pad_id = it->second;
        }
    }
    // Device-selected backend: a caller-injected backend (borrowed, not
    // owned) or one created here — CPU compute with n_threads, or a GPU
    // backend via BestAvailable (Metal on Apple Silicon, CUDA on NVIDIA).
    if (device.backend != nullptr) {
        model->backend = device.backend;
        model->owns_backend = false;
    } else {
        core::BackendConfig cfg;
        cfg.type = device.backend_type;
        cfg.device = device.device_index;
        cfg.threads = device.threads > 0
            ? device.threads
            : static_cast<int>(std::thread::hardware_concurrency());
        model->backend = core::init_backend(cfg);
        model->owns_backend = true;
    }
    if (model->backend == nullptr) {
        throw std::runtime_error("zipvoice: backend init failed");
    }
    // BestAvailable resolves to a concrete backend (Metal on Apple
    // Silicon); downstream allocation and vocoder decisions use it.
    model->backend_type = core::backend_type(model->backend);
    model->weights = load_zipvoice_weights(
        *source, "", model->config, model->backend, model->backend_type);

    state.key = key;
    state.model = std::move(model);
    return state.model;
}

// timestep_embedding host copy (matches zipvoice.modules.timestep_embedding)
std::vector<float> timestep_embedding(float t, int64_t dim) {
    const int64_t half = dim / 2;
    std::vector<float> out(static_cast<size_t>(dim));
    for (int64_t i = 0; i < half; ++i) {
        const float freq = std::exp(-std::log(10000.0F) * static_cast<float>(i) /
                                    static_cast<float>(half));
        const float arg = t * freq;
        out[static_cast<size_t>(i)] = std::cos(arg);
        out[static_cast<size_t>(half + i)] = std::sin(arg);
    }
    return out;
}

void upload_leaf(ggml_backend_t backend, ggml_tensor * leaf, const void * data, size_t bytes) {
    (void)backend;
    ggml_backend_tensor_set(leaf, data, 0, bytes);  // void return; CPU sync path
}

int compute_threads(const ZipVoiceComputeDevice & device) {
    return device.threads > 0 ? device.threads
                              : static_cast<int>(std::thread::hardware_concurrency());
}

// Fill per-stack pad bias / conv gate leaves given the valid frame count in
// the FULL-resolution sequence (frames >= valid are padding).
void fill_masks(
    ggml_backend_t backend,
    const ZipVoiceConfig & config,
    FmDecoderGraph & g,
    int64_t valid_frames) {
    for (size_t s = 0; s < config.fm_downsampling_factor.size(); ++s) {
        const int64_t ds = config.fm_downsampling_factor[s];
        // stack s runs at ceil(T / ds) frames (factors are relative to the
        // full frame rate); the ::ds-sampled mask keeps ceil(valid / ds)
        // valid frames.
        const int64_t T_s = (g.T + ds - 1) / ds;
        const int64_t valid_s = std::min<int64_t>((valid_frames + ds - 1) / ds, T_s);
        std::vector<float> bias(static_cast<size_t>(T_s));
        std::vector<float> gate(static_cast<size_t>(T_s));
        for (int64_t i = 0; i < T_s; ++i) {
            const bool pad = i >= valid_s;
            bias[static_cast<size_t>(i)] = pad ? -1000.0F : 0.0F;
            gate[static_cast<size_t>(i)] = pad ? 0.0F : 1.0F;
        }
        upload_leaf(backend, g.pad_bias[s], bias.data(), bias.size() * sizeof(float));
        upload_leaf(backend, g.conv_gate[s], gate.data(), gate.size() * sizeof(float));
    }
}

struct VelocityInputs {
    // xt, text_condition, speech_condition: [T, F] row-major (t, f)
    std::vector<float> xt, text_condition, speech_condition;
};

// Run one fm_decoder evaluation. B = 2 duplicates every input (CFG halves
// are prepared by the caller through `halves`).
std::vector<float> run_velocity(
    LoadedModel & model,
    const ZipVoiceComputeDevice & device,
    int64_t T,
    int64_t B,
    const float * xt,          // [B, T, F]
    const float * text_cond,   // [B, T, F]
    const float * speech_cond, // [B, T, F]
    int64_t valid_frames,
    float t_value,
    const float * guidance_embedding,  // nullptr for base model
    float * scratch_time = nullptr) {
    const auto & config = model.config;
    const bool with_guidance = guidance_embedding != nullptr;
    std::lock_guard<std::mutex> lock(model.mutex);
    FmDecoderGraph * g = nullptr;
    const LoadedModel::FmKey key{T, B, with_guidance};
    if (const auto * slot = model.fm_graphs.find(key)) {
        g = slot->get();
    } else {
        model.fm_graphs.clear();
        auto built = build_fm_decoder_graph(
            model.weights, config, T, B, with_guidance, false, model.backend);
        model.fm_graphs.put(key, std::make_unique<FmDecoderGraph>(std::move(built)));
        g = model.fm_graphs.find(key)->get();
    }

    // x_cat [3F, T, B]: interleave channels (xt | text | speech)
    const int64_t F = config.feat_dim;
    std::vector<float> x_cat(static_cast<size_t>(3 * F * T * B));
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t i = 0; i < T; ++i) {
            float * dst = x_cat.data() + static_cast<size_t>(3 * F * (i + b * T));
            const float * x_src = xt + static_cast<size_t>((i + b * T) * F);
            const float * t_src = text_cond + static_cast<size_t>((i + b * T) * F);
            const float * s_src = speech_cond + static_cast<size_t>((i + b * T) * F);
            std::memcpy(dst, x_src, F * sizeof(float));
            std::memcpy(dst + F, t_src, F * sizeof(float));
            std::memcpy(dst + 2 * F, s_src, F * sizeof(float));
        }
    }
    upload_leaf(model.backend, g->x_cat, x_cat.data(), x_cat.size() * sizeof(float));
    const auto time_values = scratch_time != nullptr
        ? std::vector<float>(scratch_time, scratch_time + config.time_embed_dim)
        : timestep_embedding(t_value, config.time_embed_dim);
    upload_leaf(model.backend, g->time_emb, time_values.data(),
                static_cast<size_t>(config.time_embed_dim) * sizeof(float));
    if (with_guidance) {
        upload_leaf(model.backend, g->guidance_emb, guidance_embedding,
                    static_cast<size_t>(config.time_embed_dim) * sizeof(float));
    }
    fill_masks(model.backend, config, *g, valid_frames);

    core::set_backend_threads(model.backend, compute_threads(device));
    const auto status = core::compute_backend_graph(model.backend, g->graph, nullptr, "zipvoice_fm");
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("zipvoice: fm decoder graph compute failed");
    }
    ggml_backend_synchronize(model.backend);

    // output [F, T, B] -> [B, T, F]
    std::vector<float> out_vec(ggml_nelements(g->output));
    ggml_backend_tensor_get(g->output, out_vec.data(), 0, out_vec.size() * sizeof(float));
    const float * out = out_vec.data();
    std::vector<float> v(static_cast<size_t>(T * F * B));
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t i = 0; i < T; ++i) {
            for (int64_t f = 0; f < F; ++f) {
                v[static_cast<size_t>((i + b * T) * F + f)] =
                    out[static_cast<size_t>(f + i * F + b * T * F)];
            }
        }
    }
    return v;
}

// forward_text_inference_ratio_duration (single utterance)
struct TextConditionResult {
    std::vector<float> condition;  // [T, F]
    int64_t T = 0;
};

TextConditionResult compute_text_condition(
    LoadedModel & model,
    const ZipVoiceComputeDevice & device,
    const std::vector<int32_t> & tokens,        // target tokens
    const std::vector<int32_t> & prompt_tokens, // prompt tokens
    int64_t prompt_features_len,
    float speed) {
    // pad_labels appends one trailing pad token; the original token count
    // defines the attention and convolution masks.
    // NOTE: tokens_lens in the reference EXCLUDES the pad token, so the
    // duration split uses token_count, not S.
    std::vector<int32_t> cat(prompt_tokens);
    cat.insert(cat.end(), tokens.begin(), tokens.end());
    const int64_t token_count = static_cast<int64_t>(cat.size());
    if (!std::isfinite(speed) || speed <= 0 || prompt_features_len <= 0 || tokens.empty()) {
        throw std::invalid_argument("zipvoice: nonempty tokens, positive prompt length and speed are required");
    }
    for (const int32_t token : cat) {
        if (token < 0 || token >= model.config.vocab_size) {
            throw std::invalid_argument("zipvoice: token id out of vocabulary range");
        }
    }
    cat.push_back(model.config.pad_id);
    const int64_t S = static_cast<int64_t>(cat.size());
    if (token_count == 0 || prompt_tokens.empty()) {
        throw std::runtime_error("zipvoice: empty token stream");
    }
    const int64_t features_len = prompt_features_len +
        static_cast<int64_t>(std::ceil(
            static_cast<double>(prompt_features_len) /
            static_cast<double>(prompt_tokens.size()) *
            static_cast<double>(tokens.size()) / speed));
    const int64_t T = features_len;

    std::lock_guard<std::mutex> lock(model.mutex);
    TextEncoderGraph * g = nullptr;
    if (const auto * slot = model.text_graphs.find(S)) {
        g = slot->get();
    } else {
        model.text_graphs.clear();
        auto built = build_text_encoder_graph(model.weights, model.config, S, false, model.backend);
        model.text_graphs.put(S, std::make_unique<TextEncoderGraph>(std::move(built)));
        g = model.text_graphs.find(S)->get();
    }
    upload_leaf(model.backend, g->token_ids, cat.data(), cat.size() * sizeof(int32_t));
    std::vector<float> zeros(static_cast<size_t>(S), 0.0F);
    std::vector<float> ones(static_cast<size_t>(S), 1.0F);
    zeros.back() = -1000.0F;
    ones.back() = 0.0F;
    upload_leaf(model.backend, g->pad_bias[0], zeros.data(), zeros.size() * sizeof(float));
    upload_leaf(model.backend, g->conv_gate[0], ones.data(), ones.size() * sizeof(float));

    core::set_backend_threads(model.backend, compute_threads(device));
    const auto status = core::compute_backend_graph(model.backend, g->graph, nullptr, "zipvoice_text");
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("zipvoice: text encoder graph compute failed");
    }
    ggml_backend_synchronize(model.backend);

    // embed output [feat_dim, S] -> gather per-frame rows
    std::vector<float> embed_vec(ggml_nelements(g->output));
    ggml_backend_tensor_get(g->output, embed_vec.data(), 0, embed_vec.size() * sizeof(float));
    const float * embed = embed_vec.data();
    const int64_t F = model.config.feat_dim;
    // prepare_avg_tokens_durations + get_tokens_index
    const int64_t avg = T / token_count;  // num_frames // num_tokens (B = 1)
    std::vector<int64_t> durations(static_cast<size_t>(token_count), avg);
    durations.push_back(T - avg * token_count);
    std::vector<int32_t> tokens_index(static_cast<size_t>(T));
    {
        int64_t cur = 0;
        for (int64_t i = 0; i < S; ++i) {
            const int64_t d = durations[static_cast<size_t>(i)];
            for (int64_t k = 0; k < d; ++k) {
                if (cur + k < T) tokens_index[static_cast<size_t>(cur + k)] = static_cast<int32_t>(i);
            }
            cur += d;
        }
    }
    TextConditionResult result;
    result.T = T;
    result.condition.resize(static_cast<size_t>(T * F));
    for (int64_t i = 0; i < T; ++i) {
        const int32_t tok = tokens_index[static_cast<size_t>(i)];
        std::memcpy(result.condition.data() + i * F,
                    embed + static_cast<size_t>(tok) * F, F * sizeof(float));
    }
    return result;
}

// Euler solver with t_shift; mirrors DiffusionModel/EulerSolver exactly.
std::vector<float> run_sampler(
    LoadedModel & model,
    const ZipVoiceComputeDevice & device,
    const TextConditionResult & text,
    const std::vector<float> & speech_condition,  // [T, F] padded prompt (zeros elsewhere)
    int64_t valid_frames,
    const std::vector<float> & x0,
    int num_step,
    float guidance_scale,
    float t_shift) {
    const auto & config = model.config;
    const int64_t T = text.T;
    const int64_t F = config.feat_dim;
    const bool distill = config.guidance_scale_embed;
    if (num_step <= 0 || num_step > 64 || !std::isfinite(t_shift) || t_shift <= 0 ||
        !std::isfinite(guidance_scale) || guidance_scale < 0 ||
        x0.size() != static_cast<size_t>(T * F)) {
        throw std::invalid_argument("zipvoice: invalid sampler parameters or noise shape");
    }

    // timesteps
    std::vector<float> ts(num_step + 1);
    for (int i = 0; i <= num_step; ++i) {
        ts[static_cast<size_t>(i)] = static_cast<float>(i) / num_step;
    }
    for (auto & v : ts) {
        v = t_shift * v / (1.0F + (t_shift - 1.0F) * v);
    }

    std::vector<float> x = x0;
    if (distill) {
        const auto guidance_embedding = timestep_embedding(guidance_scale, config.time_embed_dim);
        for (int step = 0; step < num_step; ++step) {
            const float t = ts[static_cast<size_t>(step)];
            const float dt = ts[static_cast<size_t>(step + 1)] - t;
            auto v = run_velocity(model, device, T, 1, x.data(), text.condition.data(),
                                  speech_condition.data(), valid_frames, t,
                                  guidance_embedding.data());
            for (size_t i = 0; i < x.size(); ++i) {
                x[i] += v[i] * dt;
            }
        }
        return x;
    }

    // base model: batched CFG
    const bool use_cfg = guidance_scale != 0.0F;
    if (!use_cfg) {
        for (int step = 0; step < num_step; ++step) {
            const float t = ts[static_cast<size_t>(step)];
            const float dt = ts[static_cast<size_t>(step + 1)] - t;
            auto v = run_velocity(model, device, T, 1, x.data(), text.condition.data(),
                                  speech_condition.data(), valid_frames, t, nullptr);
            for (size_t i = 0; i < x.size(); ++i) x[i] += v[i] * dt;
        }
        return x;
    }

    float effective = guidance_scale;
    std::vector<float> xt2, text2, speech2;
    for (int step = 0; step < num_step; ++step) {
        const float t = ts[static_cast<size_t>(step)];
        const float dt = ts[static_cast<size_t>(step + 1)] - t;
        float scale = effective;
        xt2.assign(x.data(), x.data() + x.size());
        xt2.insert(xt2.end(), x.begin(), x.end());
        text2.assign(text.condition.size(), 0.0F);
        text2.insert(text2.end(), text.condition.begin(), text.condition.end());
        if (t > 0.5F) {
            speech2.assign(speech_condition.size(), 0.0F);
            speech2.insert(speech2.end(), speech_condition.begin(), speech_condition.end());
        } else {
            scale = effective * 2.0F;
            speech2.assign(speech_condition.begin(), speech_condition.end());
            speech2.insert(speech2.end(), speech_condition.begin(), speech_condition.end());
        }
        auto v = run_velocity(model, device, T, 2, xt2.data(), text2.data(),
                              speech2.data(), valid_frames, t, nullptr);
        std::vector<float> combined(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            const float cond = v[x.size() + i];
            const float uncond = v[i];
            combined[i] = (1.0F + scale) * cond - scale * uncond;
        }
        for (size_t i = 0; i < x.size(); ++i) x[i] += combined[i] * dt;
    }
    return x;
}

struct Rng {
    uint64_t state;
    explicit Rng(uint64_t seed) : state(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t next_u64() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }
    float next_f32() {
        return static_cast<float>(static_cast<double>(next_u64() >> 11) / 9007199254740992.0);
    }
    float normal() {
        const float u1 = std::max(next_f32(), 1e-7F);
        const float u2 = next_f32();
        return std::sqrt(-2.0F * std::log(u1)) * std::cos(2.0F * static_cast<float>(kPi) * u2);
    }
};

// tokens.txt vocab: token -> id (tab separated)
std::unordered_map<std::string, int32_t> load_tokens(const std::filesystem::path & tokens_path) {
    std::unordered_map<std::string, int32_t> map;
    const auto & path = tokens_path;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("zipvoice: cannot open " + path.string());
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        map.emplace(line.substr(0, tab), std::stoi(line.substr(tab + 1)));
    }
    if (map.empty()) throw std::runtime_error("zipvoice: empty tokens.txt");
    return map;
}

std::vector<std::string> utf8_chars(const std::string & s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        size_t len = 1;
        const auto c = static_cast<unsigned char>(s[i]);
        if (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        out.emplace_back(s, i, len);
        i += len;
    }
    return out;
}

}  // namespace

void zipvoice_clear_runtime(const ZipVoiceComputeDevice & device) {
    if (!device.runtime) return;
    std::lock_guard<std::mutex> lock(device.runtime->mutex);
    device.runtime->model.reset();
    device.runtime->key.clear();
}

std::vector<float> zipvoice_vocos_decode_on_device(
    const std::string & model_path,
    const std::string & vocos_path,
    const std::vector<float> & mel_rows,
    const ZipVoiceComputeDevice & device,
    const engine::assets::ResourceBundle * resources) {
    auto owner = load_model(model_path, SidecarResolver{resources}, device);
    auto & model = *owner;
    std::lock_guard<std::mutex> lock(model.mutex);
    return decode_vocos(model, vocos_path, mel_rows, compute_threads(device));
}

std::vector<float> zipvoice_text_condition(
    const std::string & model_path,
    const std::vector<int32_t> & tokens,
    const std::vector<int32_t> & prompt_tokens,
    int64_t prompt_features_len,
    float speed,
    const ZipVoiceComputeDevice & device,
    const engine::assets::ResourceBundle * resources) {
    auto owner = load_model(model_path, SidecarResolver{resources}, device);
    auto & model = *owner;
    auto result = compute_text_condition(
        model, device, tokens, prompt_tokens, prompt_features_len, speed);
    return result.condition;
}

std::vector<float> zipvoice_text_encoder_raw(
    const std::string & model_path,
    const std::vector<int32_t> & token_ids,
    ZipVoiceLayerTaps * layer_taps,
    const ZipVoiceComputeDevice & device,
    const engine::assets::ResourceBundle * resources) {
    auto owner = load_model(model_path, SidecarResolver{resources}, device);
    auto & model = *owner;
    std::vector<int32_t> cat(token_ids);
    cat.push_back(model.config.pad_id);
    const int64_t S = static_cast<int64_t>(cat.size());

    std::lock_guard<std::mutex> lock(model.mutex);
    TextEncoderGraph * g = nullptr;
    if (const auto * slot = model.text_graphs.find(S)) {
        g = slot->get();
    } else {
        model.text_graphs.clear();
        auto built = build_text_encoder_graph(model.weights, model.config, S, false, model.backend);
        model.text_graphs.put(S, std::make_unique<TextEncoderGraph>(std::move(built)));
        g = model.text_graphs.find(S)->get();
    }
    upload_leaf(model.backend, g->token_ids, cat.data(), cat.size() * sizeof(int32_t));
    std::vector<float> zeros(static_cast<size_t>(S), 0.0F);
    std::vector<float> ones(static_cast<size_t>(S), 1.0F);
    zeros.back() = -1000.0F;
    ones.back() = 0.0F;
    upload_leaf(model.backend, g->pad_bias[0], zeros.data(), zeros.size() * sizeof(float));
    upload_leaf(model.backend, g->conv_gate[0], ones.data(), ones.size() * sizeof(float));

    core::set_backend_threads(model.backend, compute_threads(device));
    const auto status = core::compute_backend_graph(model.backend, g->graph, nullptr, "zipvoice_text");
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("zipvoice: text encoder graph compute failed");
    }
    ggml_backend_synchronize(model.backend);

    // output [feat_dim, S] -> [S, feat_dim] rows
    std::vector<float> out_vec(ggml_nelements(g->output));
    ggml_backend_tensor_get(g->output, out_vec.data(), 0, out_vec.size() * sizeof(float));
    const int64_t F = model.config.feat_dim;
    std::vector<float> rows(static_cast<size_t>(S * F));
    for (int64_t i = 0; i < S; ++i) {
        for (int64_t f = 0; f < F; ++f) {
            rows[static_cast<size_t>(i * F + f)] =
                out_vec[static_cast<size_t>(f + i * F)];
        }
    }
    if (layer_taps != nullptr) {
        layer_taps->layers.clear();
        layer_taps->stages.clear();
        for (ggml_tensor * tap : g->stage_taps) {
            // raw flat dump in ggml ne order; python/test side knows each
            // tap's shape and reorders
            std::vector<float> tap_vec(ggml_nelements(tap));
            ggml_backend_tensor_get(tap, tap_vec.data(), 0, tap_vec.size() * sizeof(float));
            layer_taps->stages.push_back(std::move(tap_vec));
        }
        for (ggml_tensor * tap : g->layer_taps) {
            const int64_t C = tap->ne[0];
            std::vector<float> tap_vec(ggml_nelements(tap));
            ggml_backend_tensor_get(tap, tap_vec.data(), 0, tap_vec.size() * sizeof(float));
            std::vector<float> tap_rows(static_cast<size_t>(S * C));
            for (int64_t i = 0; i < S; ++i) {
                for (int64_t c = 0; c < C; ++c) {
                    tap_rows[static_cast<size_t>(i * C + c)] =
                        tap_vec[static_cast<size_t>(c + i * C)];
                }
            }
            layer_taps->layers.push_back(std::move(tap_rows));
        }
    }
    return rows;
}

std::vector<float> zipvoice_velocity(
    const std::string & model_path,
    const std::vector<float> & xt,
    const std::vector<float> & text_condition,
    const std::vector<float> & speech_condition,
    int64_t features_len,
    float t,
    float guidance_scale,
    const ZipVoiceComputeDevice & device,
    int batch_size,
    const engine::assets::ResourceBundle * resources) {
    auto owner = load_model(model_path, SidecarResolver{resources}, device);
    auto & model = *owner;
    const int64_t T = features_len;
    const float * guidance = nullptr;
    std::vector<float> guidance_embedding;
    if (model.config.guidance_scale_embed) {
        guidance_embedding = timestep_embedding(guidance_scale, model.config.time_embed_dim);
        guidance = guidance_embedding.data();
    }
    const size_t expected = static_cast<size_t>(T * model.config.feat_dim * batch_size);
    if (T <= 0 || batch_size <= 0 || xt.size() != expected ||
        text_condition.size() != expected || speech_condition.size() != expected) {
        throw std::invalid_argument("zipvoice: velocity input shape mismatch");
    }
    return run_velocity(model, device, T, batch_size, xt.data(), text_condition.data(),
                        speech_condition.data(), features_len, t, guidance);
}

std::vector<float> zipvoice_sample(
    const std::string & model_path,
    const std::vector<int32_t> & tokens,
    const std::vector<int32_t> & prompt_tokens,
    const std::vector<float> & prompt_features,
    int64_t prompt_features_len,
    const std::vector<float> & x0,
    int num_steps,
    float guidance_scale,
    float t_shift,
    float speed,
    const ZipVoiceComputeDevice & device,
    const engine::assets::ResourceBundle * resources) {
    auto owner = load_model(model_path, SidecarResolver{resources}, device);
    auto & model = *owner;
    auto text = compute_text_condition(
        model, device, tokens, prompt_tokens, prompt_features_len, speed);
    const int64_t T = text.T;
    const int64_t F = model.config.feat_dim;
    if (prompt_features_len <= 0 || prompt_features.size() < static_cast<size_t>(prompt_features_len * F)) {
        throw std::invalid_argument("zipvoice: prompt feature shape mismatch");
    }
    std::vector<float> speech(static_cast<size_t>(T * F), 0.0F);
    for (int64_t i = 0; i < prompt_features_len && i < T; ++i) {
        std::memcpy(speech.data() + i * F,
                    prompt_features.data() + static_cast<size_t>(i * F), F * sizeof(float));
    }
    return run_sampler(model, device, text, speech, T, x0, num_steps, guidance_scale, t_shift);
}

ZipVoiceSynthesisResult zipvoice_synthesize(
    const std::string & model_path,
    const std::string & vocos_path,
    const ZipVoiceSynthesisRequest & request,
    const ZipVoiceComputeDevice & device,
    const engine::assets::ResourceBundle * resources) {
    if (request.ref_audio.empty() || request.ref_sample_rate <= 0 ||
        !std::isfinite(request.feat_scale) || request.feat_scale <= 0) {
        throw std::invalid_argument("zipvoice: valid reference audio and positive feature scale are required");
    }
    const SidecarResolver sidecars{resources};
    auto owner = load_model(model_path, sidecars, device);
    auto & model = *owner;

    // tokens
    std::vector<int32_t> tokens = request.token_ids;
    std::vector<int32_t> prompt_tokens = request.prompt_token_ids;
    if (tokens.empty() || prompt_tokens.empty()) {
        // Keep the dictionary frontend with the session, just like the graphs.
        // Loading its tables and Jieba dictionary for every chunk is avoidable.
        // Serialize encoding too: Emilia lazily initializes its phonemizer.
        std::lock_guard<std::mutex> lock(model.mutex);
        const std::filesystem::path path(model_path);
        const std::filesystem::path dir =
            std::filesystem::is_directory(path) ? path : path.parent_path();
        const auto vocab = load_tokens(sidecars.path("tokens", dir, "tokens.txt"));
        std::unique_ptr<audio::EspeakPhonemizer> phonemizer;
        EmiliaTokenizer * emilia = nullptr;
        if (request.tokenizer == "espeak") {
            phonemizer = std::make_unique<audio::EspeakPhonemizer>(
                std::filesystem::path{request.espeak_library_path}, request.espeak_data_path,
                std::vector<std::string>{request.lang});
        } else if (request.tokenizer == "emilia") {
            // Chinese/mixed frontend: needs the baked pypinyin tables (from
            // the bundle or next to tokens.txt) and an espeak-ng installation
            // for English segments.
            EmiliaTokenizer::TablePaths tables;
            tables.chars = sidecars.path("zh_chars", dir, "zh_chars.tsv");
            tables.phrases = sidecars.path("zh_phrases", dir, "zh_phrases.tsv");
            tables.syllables = sidecars.path("zh_syllables", dir, "zh_syllables.tsv");
            tables.jieba_dict = sidecars.path("zh_jieba_dict", dir, "zh_jieba_dict.txt");
            tables.hmm_model = sidecars.path("zh_hmm_model", dir, "zh_hmm_model.txt");
            const std::vector<std::string> key{
                sidecars.path("tokens", dir, "tokens.txt").string(),
                tables.chars.string(), tables.phrases.string(), tables.syllables.string(),
                tables.jieba_dict.string(), tables.hmm_model.string(),
                request.espeak_library_path, request.espeak_data_path, request.lang};
            if (!model.tokenizers.find(key)) {
                model.tokenizers.clear();
                model.tokenizers.put(key, std::make_unique<EmiliaTokenizer>(
                    tables, vocab, EmiliaTokenizer::EspeakConfig{
                        request.espeak_library_path, request.espeak_data_path, request.lang}));
            }
            emilia = model.tokenizers.find(key)->get();
        } else if (request.tokenizer != "simple") {
            throw std::invalid_argument("zipvoice: tokenizer must be espeak, emilia, or simple");
        }
        const auto encode = [&](const std::string & text) {
            if (emilia) {
                return emilia->encode(text);
            }
            std::string phones = text;
            if (phonemizer) {
                phones.clear();
                size_t start = 0;
                for (size_t i = 0; i <= text.size(); ++i) {
                    if (i != text.size() && std::string(";:,.!?\"").find(text[i]) == std::string::npos) continue;
                    const auto segment = text.substr(start, i - start);
                    if (!segment.empty()) {
                        if (!phones.empty() && std::isspace(static_cast<unsigned char>(segment.front()))) phones += ' ';
                        phones += phonemizer->phonemize(segment, 2);
                    }
                    if (i != text.size()) phones += text[i];
                    start = i + 1;
                }
            }
            std::vector<int32_t> ids;
            for (const auto & ch : utf8_chars(phones)) {
                if (const auto it = vocab.find(ch); it != vocab.end()) {
                    ids.push_back(it->second);
                }
            }
            return ids;
        };
        if (tokens.empty()) tokens = encode(request.text);
        if (prompt_tokens.empty()) prompt_tokens = encode(request.ref_text);
    }

    // prompt audio -> 24k mono
    auto wav = audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
        request.ref_audio, request.ref_sample_rate, request.ref_channels, kSampleRate);
    float prompt_rms = 0.0F;
    for (const float s : wav) prompt_rms += s * s;
    prompt_rms = std::sqrt(prompt_rms / std::max<size_t>(1, wav.size()));
    if (request.target_rms > 0.0F && prompt_rms < request.target_rms) {
        const float gain = request.target_rms / std::max(prompt_rms, 1e-8F);
        for (auto & s : wav) s *= gain;
    }

    const auto start = std::chrono::steady_clock::now();
    auto mel = zipvoice_logmel(wav);  // [T_p, 100]
    const int64_t prompt_features_len = static_cast<int64_t>(mel.size()) / kNMel;
    const float feat_scale = request.feat_scale;
    for (auto & v : mel) v *= feat_scale;

    auto text = compute_text_condition(
        model, device, tokens, prompt_tokens, prompt_features_len, request.speed);
    const int64_t T = text.T;
    const int64_t F = model.config.feat_dim;

    std::vector<float> speech(static_cast<size_t>(T * F), 0.0F);
    std::memcpy(speech.data(), mel.data(),
                static_cast<size_t>(prompt_features_len) * static_cast<size_t>(F) * sizeof(float));

    Rng rng(request.seed);
    std::vector<float> x0(static_cast<size_t>(T * F));
    for (auto & v : x0) v = rng.normal();

    auto x1 = run_sampler(
        model, device, text, speech, T, x0, request.num_steps,
        request.guidance_scale, request.t_shift);

    // strip prompt frames, unscale features
    const int64_t out_frames = T - prompt_features_len;
    std::vector<float> features(static_cast<size_t>(out_frames * F));
    for (int64_t i = 0; i < out_frames; ++i) {
        for (int64_t f = 0; f < F; ++f) {
            features[static_cast<size_t>(i * F + f)] =
                x1[static_cast<size_t>((prompt_features_len + i) * F + f)] / feat_scale;
        }
    }

    std::vector<float> audio;
    {
        std::lock_guard<std::mutex> lock(model.mutex);
        audio = decode_vocos(model, vocos_path, features, compute_threads(device));
    }
    for (auto & s : audio) s = std::clamp(s, -1.0F, 1.0F);
    if (request.target_rms > 0.0F && prompt_rms < request.target_rms && prompt_rms > 0.0F) {
        const float gain = prompt_rms / request.target_rms;
        for (auto & s : audio) s *= gain;
    }

    ZipVoiceSynthesisResult result;
    result.sample_rate = kSampleRate;
    result.audio = std::move(audio);
    const auto end = std::chrono::steady_clock::now();
    result.model_seconds = std::chrono::duration<double>(end - start).count();
    result.audio_seconds = static_cast<double>(result.audio.size()) / kSampleRate;
    return result;
}

}  // namespace engine::models::zipvoice
