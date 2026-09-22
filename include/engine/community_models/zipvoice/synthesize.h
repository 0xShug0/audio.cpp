#pragma once

#include "engine/framework/core/backend.h"

#include <cstdint>
#include <string>
#include <memory>
#include <vector>

namespace engine::assets {
class ResourceBundle;
}

namespace engine::models::zipvoice {

class ZipVoiceRuntimeState;
std::shared_ptr<ZipVoiceRuntimeState> make_zipvoice_runtime_state();

// Compute device for the whole zipvoice stack (text encoder, flow-matching
// decoder, Vocos backbone). GPU first: the default BestAvailable resolves to
// Metal/CUDA when a usable accelerator exists and falls back to CPU
// otherwise; `threads` applies to CPU compute. A runtime that already owns a
// backend may inject it via `backend`. Destroy/reset this device runtime before
// freeing a borrowed backend. Copies share runtime ownership and serialize use.
struct ZipVoiceComputeDevice {
    core::BackendType backend_type = core::BackendType::BestAvailable;
    int device_index = 0;
    int threads = 0;  // 0 = hardware concurrency
    ggml_backend_t backend = nullptr;
    std::shared_ptr<ZipVoiceRuntimeState> runtime = make_zipvoice_runtime_state();
};

// Release this session/device's weights and graphs; no process-global cache.
void zipvoice_clear_runtime(const ZipVoiceComputeDevice & device);

struct ZipVoiceSynthesisRequest {
    std::string text;
    std::string ref_text;
    std::vector<float> ref_audio;
    int ref_sample_rate = 24000;
    int ref_channels = 1;  // ref_audio is interleaved
    std::vector<int32_t> token_ids;        // optional: pre-tokenized target text
    std::vector<int32_t> prompt_token_ids;  // optional: pre-tokenized ref text
    std::string tokenizer = "emilia";  // fixed frontend: zh/en/mixed (espeak for en runs)
    std::string lang = "en-us";
    std::string espeak_library_path;
    std::string espeak_data_path;
    int num_steps = 8;
    float guidance_scale = 3.0F;
    float t_shift = 0.5F;
    float speed = 1.0F;
    float feat_scale = 0.1F;
    float target_rms = 0.1F;
    uint32_t seed = 666;
    bool fixed_seed = true;
};

struct ZipVoiceSynthesisResult {
    std::vector<float> audio;  // 24 kHz mono
    int sample_rate = 24000;
    double model_seconds = 0.0;      // wall time of the flow + vocoder
    double audio_seconds = 0.0;      // generated audio duration
};

// Full pipeline: tokenize (if needed) -> fbank -> duration prediction ->
// Euler flow-matching sampling -> Vocos decode. `model_path` is the GGUF
// package or the safetensors development directory. `resources` (optional)
// is the spec-resolved bundle: registered sidecars (tokens, model_config,
// zh_* tables, materialized from an embedded-sidecar GGUF or found in the
// development directory) take priority over loose files next to the
// checkpoint; direct API callers (parity harnesses) may pass nullptr.
ZipVoiceSynthesisResult zipvoice_synthesize(
    const std::string & model_path,
    const std::string & vocos_path,
    const ZipVoiceSynthesisRequest & request,
    const ZipVoiceComputeDevice & device = {},
    const engine::assets::ResourceBundle * resources = nullptr);

// --- test hooks (parity harness) -------------------------------------------

// Backend Vocos path using the same cached model/backend as synthesis.
std::vector<float> zipvoice_vocos_decode_on_device(
    const std::string & model_path,
    const std::string & vocos_path,
    const std::vector<float> & mel_rows,
    const ZipVoiceComputeDevice & device,
    const engine::assets::ResourceBundle * resources = nullptr);

// Text-condition stage: token ids -> text_condition [T, feat_dim] given
// prompt feature length. Mirrors forward_text_inference_ratio_duration.
std::vector<float> zipvoice_text_condition(
    const std::string & model_path,
    const std::vector<int32_t> & tokens,
    const std::vector<int32_t> & prompt_tokens,
    int64_t prompt_features_len,
    float speed,
    const ZipVoiceComputeDevice & device = {},
    const engine::assets::ResourceBundle * resources = nullptr);

// Raw text-encoder output for pre-tokenized ids (pad token appended
// internally, mirroring pad_labels): returns [S+1, feat_dim] row-major.
// `layer_taps` (optional) receives per-encoder-layer outputs as [S+1, C]
// rows for parity bisecting.
struct ZipVoiceLayerTaps {
    std::vector<std::vector<float>> layers;   // per-layer outputs [S+1, C]
    std::vector<std::vector<float>> stages;   // first-layer submodule taps
};
std::vector<float> zipvoice_text_encoder_raw(
    const std::string & model_path,
    const std::vector<int32_t> & token_ids,
    ZipVoiceLayerTaps * layer_taps = nullptr,
    const ZipVoiceComputeDevice & device = {},
    const engine::assets::ResourceBundle * resources = nullptr);

// One flow-matching velocity evaluation at time t (single batch, no CFG).
std::vector<float> zipvoice_velocity(
    const std::string & model_path,
    const std::vector<float> & xt,          // [T, feat_dim]
    const std::vector<float> & text_condition,
    const std::vector<float> & speech_condition,
    int64_t features_len,
    float t,
    float guidance_scale,
    const ZipVoiceComputeDevice & device = {},
    int batch_size = 1,
    const engine::assets::ResourceBundle * resources = nullptr);

// Full sampler: x0 -> x1 (mirrors ZipVoice::sample for duration="predict").
std::vector<float> zipvoice_sample(
    const std::string & model_path,
    const std::vector<int32_t> & tokens,
    const std::vector<int32_t> & prompt_tokens,
    const std::vector<float> & prompt_features,  // [T_prompt, feat_dim], scaled
    int64_t prompt_features_len,
    const std::vector<float> & x0,
    int num_steps,
    float guidance_scale,
    float t_shift,
    float speed,
    const ZipVoiceComputeDevice & device = {},
    const engine::assets::ResourceBundle * resources = nullptr);

// Log-mel filterbank identical to VocosFbank (24 kHz, 100 mels, hop 256,
// power=1, log clamp 1e-7, htk scale), with lhotse frame-count alignment.
std::vector<float> zipvoice_logmel(const std::vector<float> & wav);

// Vocos mel-24kHz decode (same weights as f5_tts uses).
std::vector<float> zipvoice_vocos_decode(
    const std::string & vocos_path, const std::vector<float> & mel_frames);

}  // namespace engine::models::zipvoice
