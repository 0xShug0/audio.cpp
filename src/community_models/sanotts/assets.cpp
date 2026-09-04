#include "engine/community_models/sanotts/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/io/validation.h"
#include "engine/framework/model_spec/resource_bundle_loader.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

extern "C" {
#include "nano_q8_meta.h"
}

namespace engine::models::sanotts {
namespace {

namespace json = engine::io::json;

constexpr const char * kFamily = "sanotts";

/**
 * The lineage the vendored runtime was compiled against.
 *
 * snt_nano.c takes its widths, block counts and kernel sizes from
 * nano_q8_meta.h at compile time, which is what lets it run on a
 * microcontroller with no parsing and no allocation. The consequence here is
 * that one binary serves one lineage. Loading another lineage's weights would
 * read the blobs at the wrong offsets and synthesize noise rather than fail,
 * so every constant is compared against the package's config.json before any
 * of it is used.
 */
struct CompiledLineage {
    static constexpr int64_t vocab = NANO_VOCAB;
    static constexpr int64_t mels = NANO_MELS;
    static constexpr int64_t dim = NANO_DIM;
    static constexpr int64_t blocks = NANO_BLOCKS;
    static constexpr int64_t noise_ch = NANO_NOISE_CH;
    static constexpr int64_t dur_hidden = NANO_DUR_HIDDEN;
    static constexpr int64_t dur_depth = NANO_DUR_DEPTH;
    static constexpr int64_t ac_hidden = NANO_AC_HIDDEN;
    static constexpr int64_t ac_depth = NANO_AC_DEPTH;
    static constexpr int64_t hop = NANO_HOP;
    static constexpr int64_t n_fft = NANO_N_FFT;
    static constexpr int64_t front_bytes = NANO_FRONT_BYTES;
    static constexpr int64_t decoder_bytes = NANO_DEC_BYTES;
    static constexpr int64_t weight_format = NANO_WEIGHT_FORMAT;
};

int64_t require_shape(
    const json::Value & shapes,
    const std::string & key,
    int64_t compiled) {
    const int64_t value = shapes.require(key).as_i64();
    if (value != compiled) {
        throw std::runtime_error(
            "sanoTTS lineage mismatch: config.json " + key + "=" +
            std::to_string(value) + " but this build was compiled for " +
            std::to_string(compiled) +
            ". This package is for a different sanoTTS lineage.");
    }
    return value;
}

SanoTtsConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    const auto architecture = root.require("architecture").as_string();
    if (architecture != kFamily) {
        throw std::runtime_error(
            "sanoTTS config.json architecture is '" + architecture + "', expected 'sanotts'");
    }
    const json::Value & shapes = root.require("shapes");

    SanoTtsConfig out;
    out.voice = root.require("voice").as_string();
    out.vocab = require_shape(shapes, "vocab", CompiledLineage::vocab);
    out.mels = require_shape(shapes, "mels", CompiledLineage::mels);
    out.dim = require_shape(shapes, "dim", CompiledLineage::dim);
    out.blocks = require_shape(shapes, "blocks", CompiledLineage::blocks);
    out.noise_channels = require_shape(shapes, "noise_ch", CompiledLineage::noise_ch);
    out.dur_hidden = require_shape(shapes, "dur_hidden", CompiledLineage::dur_hidden);
    out.dur_depth = require_shape(shapes, "dur_depth", CompiledLineage::dur_depth);
    out.ac_hidden = require_shape(shapes, "ac_hidden", CompiledLineage::ac_hidden);
    out.ac_depth = require_shape(shapes, "ac_depth", CompiledLineage::ac_depth);
    out.hop = require_shape(shapes, "hop", CompiledLineage::hop);
    out.n_fft = require_shape(shapes, "n_fft", CompiledLineage::n_fft);
    out.max_tokens = shapes.require("dur_max_tokens").as_i64();
    out.weight_format = shapes.require("weight_format").as_i64();
    if (out.weight_format != CompiledLineage::weight_format) {
        throw std::runtime_error(
            "sanoTTS weight format mismatch: package is " +
            std::string(out.weight_format == 1 ? "f32" : "int8") +
            " but this build expects " +
            std::string(CompiledLineage::weight_format == 1 ? "f32" : "int8"));
    }
    out.sample_rate = root.require("sample_rate").as_i64();
    engine::io::require_positive(out.sample_rate, "sanoTTS sample_rate");
    return out;
}

/**
 * Rebuild one flat blob by writing each named tensor at the byte offset the
 * runtime expects.
 *
 * config.json carries the (tensor, offset) pairs, so nothing here has to
 * translate region names -- the packaging tool emits the mapping from the
 * same code that verifies the reassembly, and it proves the result equals the
 * original blob byte for byte before publishing.
 *
 * Regions are 16-byte aligned, so a few hundred bytes between them are never
 * written; the blob is zero-initialised and the originals carry zeros there,
 * which is exactly what makes the round-trip byte-identical.
 */
void rebuild_blob(
    std::vector<uint8_t> & blob,
    const assets::TensorSource & weights,
    const json::Value & root,
    const char * key) {
    const auto & regions = root.require(key).as_array();
    if (regions.empty()) {
        throw std::runtime_error(std::string("sanoTTS config.json ") + key + " is empty");
    }
    for (const auto & region : regions) {
        const auto & tensor = region.require("tensor").as_string();
        const int64_t offset = region.require("offset").as_i64();
        const auto data = weights.require_tensor_data(tensor);
        if (offset < 0 ||
            static_cast<size_t>(offset) + data.bytes.size() > blob.size()) {
            throw std::runtime_error(
                "sanoTTS tensor '" + tensor + "' does not fit its blob at offset " +
                std::to_string(offset));
        }
        std::memcpy(blob.data() + offset, data.bytes.data(), data.bytes.size());
    }
}

}  // namespace

std::shared_ptr<const SanoTtsAssets> load_sanotts_assets(
    const std::filesystem::path & model_path) {
    auto resources = engine::model_spec::load_resource_bundle_for_family(model_path, kFamily);

    SanoTtsAssets out;
    out.config = parse_config(resources);
    const auto weights = resources.open_tensor_source("weights");
    const auto root = resources.parse_json("config");

    out.front_blob.assign(static_cast<size_t>(CompiledLineage::front_bytes), 0U);
    out.decoder_blob.assign(static_cast<size_t>(CompiledLineage::decoder_bytes), 0U);

    rebuild_blob(out.front_blob, *weights, root, "front_regions");
    rebuild_blob(out.decoder_blob, *weights, root, "decoder_regions");

    out.resources = std::move(resources);
    return std::make_shared<SanoTtsAssets>(std::move(out));
}

}  // namespace engine::models::sanotts
