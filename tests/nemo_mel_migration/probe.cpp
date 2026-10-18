#include "engine/community_models/parakeet_tdt/assets.h"
#include "engine/community_models/parakeet_tdt/frontend.h"
#include "engine/community_models/sortformer_diar/assets.h"
#include "engine/community_models/sortformer_diar/frontend.h"
#include "engine/community_models/granite5asr/frontend.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/debug/trace.h"
#include "engine/models/hviske_asr/assets.h"
#include "engine/models/hviske_asr/frontend.h"
#include "engine/models/citrinet_asr/runtime.h"
#include "engine/models/canary_asr/model.h"
#include "engine/models/cohere_asr/model.h"
#include "engine/models/sortformer_diar/assets.h"
#include "engine/models/sortformer_diar/frontend.h"
#include "engine/models/nemotron_asr/frontend.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string argument(int argc, char ** argv, const std::string & name) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) return argv[i + 1];
    }
    throw std::runtime_error("missing required argument: " + name);
}

void write_features(
    const std::filesystem::path & prefix,
    const std::vector<float> & values,
    int64_t frames,
    int64_t valid_frames,
    int64_t features,
    const char * layout) {
    if (frames < 0 || valid_frames < 0 || valid_frames > frames || features <= 0 ||
        values.size() != static_cast<size_t>(frames * features)) {
        throw std::runtime_error("frontend returned inconsistent shape or valid length");
    }
    std::ofstream data(prefix.string() + ".f32", std::ios::binary | std::ios::trunc);
    if (!data) throw std::runtime_error("cannot open feature output");
    data.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!data) throw std::runtime_error("cannot write feature output");
    std::ofstream meta(prefix.string() + ".meta", std::ios::trunc);
    if (!meta) throw std::runtime_error("cannot open metadata output");
    meta << frames << ' ' << valid_frames << ' ' << features << ' ' << layout << '\n';
    if (!meta) throw std::runtime_error("cannot write metadata output");
    std::cout << prefix << ": frames=" << frames << " valid=" << valid_frames
              << " features=" << features << " layout=" << layout << '\n';
}

}  // namespace

int main(int argc, char ** argv) try {
    const auto family = argument(argc, argv, "--family");
    const std::filesystem::path model = argument(argc, argv, "--model");
    const std::filesystem::path audio_path = argument(argc, argv, "--audio");
    const std::filesystem::path prefix = argument(argc, argv, "--out");
    engine::debug::configure_logging({true, argument(argc, argv, "--log")});
    const auto wav = engine::audio::read_wav_f32(audio_path);
    const engine::runtime::AudioBuffer input{wav.sample_rate, wav.channels, wav.samples};

    if (family == "parakeet_tdt") {
        engine::community_models::parakeet_tdt::ParakeetFrontend frontend(
            engine::community_models::parakeet_tdt::load_parakeet_assets(model));
        for (const bool centered : {true, false}) {
            const auto features = frontend.extract(input, centered);
            write_features(prefix.string() + (centered ? ".center" : ".no_center"),
                           features.values, features.frames, features.valid_frames,
                           features.feature_dim, "time_major");
        }
    } else if (family == "sortformer_diar") {
        const auto assets = engine::models::sortformer_diar::load_sortformer_assets(model);
        const auto features = engine::models::sortformer_diar::compute_sortformer_features(input, *assets, 8);
        write_features(prefix, features.time_major, features.frames, features.valid_frames,
                       assets->feature_config.num_mel_bins, "time_major");
    } else if (family == "sortformer_diar_v2") {
        const auto assets = engine::community_models::sortformer_diar::load_sortformer_v2_assets(
            model, model.parent_path() / "model_spec.json");
        const auto offline = engine::community_models::sortformer_diar::compute_sortformer_v2_features(input, *assets, 8);
        write_features(prefix.string() + ".offline", offline.time_major, offline.frames,
                       offline.valid_frames, assets->feature_config.num_mel_bins, "time_major");
        const auto mono = engine::audio::mixdown_interleaved_to_mono_average(input.samples, input.channels);
        const auto streaming = engine::community_models::sortformer_diar::compute_sortformer_v2_stream_features(
            mono, *assets, 8, false);
        write_features(prefix.string() + ".stream", streaming.time_major, streaming.frames,
                       streaming.valid_frames, assets->feature_config.num_mel_bins, "time_major");
    } else if (family == "hviske_asr") {
        engine::models::hviske_asr::HviskeFrontend frontend(
            engine::models::hviske_asr::load_hviske_asr_assets(model));
        const auto features = frontend.extract(input);
        write_features(prefix, features.values, features.frames, features.valid_frames,
                       features.feature_dim, "feature_major");
    } else if (family == "citrinet_asr") {
        const auto weights = engine::models::citrinet_asr::load_citrinet_weights_cached(model);
        const auto features = engine::models::citrinet_asr::extract_citrinet_frontend(input, *weights);
        write_features(prefix, features.values, features.padded_frames, features.raw_frames,
                       weights->config.n_mels, "time_major");
    } else if (family == "canary_asr") {
        const auto assets = engine::models::canary_asr::load_canary_assets(model);
        const auto mono = engine::audio::mixdown_interleaved_to_mono_average(input.samples, input.channels);
        const auto features = engine::models::canary_asr::extract_canary_frontend(mono, *assets, 8);
        write_features(prefix, features.values, features.raw_frames, features.valid_frames,
                       128, "feature_major");
    } else if (family == "cohere_asr") {
        const auto assets = engine::models::cohere_asr::load_cohere_assets(model);
        const auto mono = engine::audio::mixdown_interleaved_to_mono_average(input.samples, input.channels);
        const auto features = engine::models::cohere_asr::extract_cohere_frontend(mono, *assets, 8);
        write_features(prefix, features.values, features.shape.at(2),
                       static_cast<int64_t>(mono.size()) / 160, 128, "feature_major");
    } else if (family == "nemotron_asr") {
        engine::models::nemotron_asr::NemotronFrontend frontend(
            engine::models::nemotron_asr::load_nemotron_asr_assets(model));
        for (const bool centered : {true, false}) {
            const auto features = frontend.extract(input, centered);
            write_features(prefix.string() + (centered ? ".center" : ".no_center"),
                           features.values, features.frames, features.valid_frames,
                           features.feature_dim, "time_major");
        }
    } else if (family == "granite5asr") {
        engine::community_models::granite5asr::Granite5Frontend frontend(
            engine::community_models::granite5asr::load_granite5asr_assets(model));
        const auto features = frontend.extract(input);
        write_features(prefix, features.values, features.frames, features.frames,
                       features.feature_dim, "time_major");
    } else {
        throw std::runtime_error("unsupported family: " + family);
    }
    return 0;
} catch (const std::exception & error) {
    std::cerr << "nemo_mel_migration_probe: " << error.what() << '\n';
    return 1;
}
