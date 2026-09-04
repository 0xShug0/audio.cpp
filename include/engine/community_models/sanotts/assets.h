#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::sanotts {

/**
 * Shape constants for one sanoTTS lineage.
 *
 * These arrive in the GGUF key/value header as `sanotts.*` and are checked
 * against the constants the vendored runtime was compiled with. A lineage
 * mismatch -- loading the 2.27M `heart` weights into a binary built for the
 * 294k `heart-nano` -- would otherwise read the blobs at the wrong offsets and
 * produce confident noise rather than an error.
 */
struct SanoTtsConfig {
    int64_t vocab = 0;
    int64_t sample_rate = 24000;
    int64_t hop = 256;
    int64_t n_fft = 1024;
    int64_t mels = 0;
    int64_t dim = 0;
    int64_t blocks = 0;
    int64_t noise_channels = 0;
    int64_t dur_hidden = 0;
    int64_t dur_depth = 0;
    int64_t ac_hidden = 0;
    int64_t ac_depth = 0;
    int64_t max_tokens = 0;
    int64_t weight_format = 0;   // 0 = int8 rows with per-row scale, 1 = f32 rows
    std::string voice;
};

/**
 * The two flat weight blobs the runtime consumes, rebuilt from GGUF tensors.
 *
 * The runtime addresses weights by byte offset because it was written for
 * microcontrollers, where parsing a container at load time is not affordable.
 * Rather than give it a second addressing scheme, the loader reassembles the
 * exact byte layout it expects. That reassembly is verified upstream: the
 * packaging tool rebuilds both blobs from the GGUF and requires byte equality
 * with the originals.
 */
struct SanoTtsAssets {
    assets::ResourceBundle resources;
    SanoTtsConfig config;
    std::vector<uint8_t> front_blob;
    std::vector<uint8_t> decoder_blob;
};

std::shared_ptr<const SanoTtsAssets> load_sanotts_assets(
    const std::filesystem::path & model_path);

}  // namespace engine::models::sanotts
