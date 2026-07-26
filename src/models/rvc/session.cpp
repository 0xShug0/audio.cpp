#include "engine/models/rvc/session.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::rvc {
namespace {

std::string request_option_or_default(
    const runtime::TaskRequest & request,
    const std::string & prefixed_key,
    const std::string & bare_key,
    std::string default_value) {
    if (const auto it = request.options.find(prefixed_key); it != request.options.end()) {
        return it->second;
    }
    if (const auto it = request.options.find(bare_key); it != request.options.end()) {
        return it->second;
    }
    return default_value;
}

int request_int_or_default(
    const runtime::TaskRequest & request,
    const std::string & prefixed_key,
    const std::string & bare_key,
    int default_value) {
    const auto value = request_option_or_default(request, prefixed_key, bare_key, std::to_string(default_value));
    size_t consumed = 0;
    const int parsed = std::stoi(value, &consumed);
    if (consumed != value.size()) {
        throw std::runtime_error("invalid integer RVC option " + prefixed_key + ": " + value);
    }
    return parsed;
}

float request_float_or_default(
    const runtime::TaskRequest & request,
    const std::string & prefixed_key,
    const std::string & bare_key,
    float default_value) {
    const auto value = request_option_or_default(request, prefixed_key, bare_key, std::to_string(default_value));
    size_t consumed = 0;
    const float parsed = std::stof(value, &consumed);
    if (consumed != value.size()) {
        throw std::runtime_error("invalid float RVC option " + prefixed_key + ": " + value);
    }
    return parsed;
}

engine::assets::TensorStorageType rvc_weight_type_from_options(const runtime::SessionOptions & options) {
    const auto it = options.options.find("rvc.weight_type");
    if (it == options.options.end()) {
        return engine::assets::TensorStorageType::F32;
    }
    const auto storage_type = engine::assets::parse_tensor_storage_type(it->second);
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return storage_type;
    }
    throw std::runtime_error("rvc.weight_type currently supports only native, f32, f16, bf16, and q8_0");
}

std::size_t user_voice_cache_slots_from_options(const runtime::SessionOptions & options) {
    constexpr int64_t kDefaultCacheSlots = 4;
    const int64_t slots = runtime::parse_i64_option(
        options.options,
        {"rvc.voice_model_cache_slots", "voice_model_cache_slots"})
        .value_or(kDefaultCacheSlots);
    if (slots < 0) {
        throw std::runtime_error("rvc.voice_model_cache_slots must be non-negative");
    }
    if (static_cast<std::uint64_t>(slots) > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("rvc.voice_model_cache_slots is too large");
    }
    return static_cast<std::size_t>(slots);
}

const RvcVoiceModel & select_packaged_voice(const RvcAssets & assets, const runtime::TaskRequest & request) {
    std::string voice_id = "default";
    if (const auto it = request.options.find("target_voice"); it != request.options.end()) {
        voice_id = it->second;
    }
    voice_id = request_option_or_default(request, "rvc.voice", "voice", voice_id);
    const auto voice = assets.voices.find(voice_id);
    if (voice == assets.voices.end()) {
        throw std::runtime_error("unknown RVC voice id: " + voice_id);
    }
    return voice->second;
}

RvcInferenceConfig request_config(const runtime::TaskRequest & request) {
    RvcInferenceConfig config;
    config.f0_method = request_option_or_default(request, "rvc.f0_method", "f0_method", "rmvpe");
    config.f0_file = request_option_or_default(request, "rvc.f0_file", "f0_file", "");
    config.file_index = request_option_or_default(request, "rvc.file_index", "file_index", "");
    config.f0_up_key = request_int_or_default(request, "rvc.f0_up_key", "f0_up_key", 0);
    config.index_rate = request_float_or_default(request, "rvc.index_rate", "index_rate", 0.0F);
    config.filter_radius = request_int_or_default(request, "rvc.filter_radius", "filter_radius", 3);
    config.resample_sr = request_int_or_default(request, "rvc.resample_sr", "resample_sr", 0);
    config.rms_mix_rate = request_float_or_default(request, "rvc.rms_mix_rate", "rms_mix_rate", 0.25F);
    config.protect = request_float_or_default(request, "rvc.protect", "protect", 0.33F);
    config.speaker_id = request_int_or_default(request, "rvc.speaker_id", "sid", 0);
    config.x_pad = request_int_or_default(request, "rvc.x_pad", "x_pad", 1);
    config.x_query = request_int_or_default(request, "rvc.x_query", "x_query", 5);
    config.x_center = request_int_or_default(request, "rvc.x_center", "x_center", 30);
    config.x_max = request_int_or_default(request, "rvc.x_max", "x_max", 32);
    return config;
}

}  // namespace

RvcSession::RvcSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const RvcAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(std::move(assets)),
      weight_storage_type_(rvc_weight_type_from_options(RuntimeSessionBase::options())),
      pipeline_(assets_, execution_context().config(), weight_storage_type_),
      user_voice_cache_(user_voice_cache_slots_from_options(RuntimeSessionBase::options())) {
    if (assets_ == nullptr) {
        throw std::runtime_error("RVC session requires assets");
    }
    if (task_.task != runtime::VoiceTaskKind::VoiceConversion) {
        throw std::runtime_error("RVC models only support --task vc");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("RVC models only support offline mode");
    }
}

std::string RvcSession::family() const {
    return "rvc";
}

runtime::VoiceTaskKind RvcSession::task_kind() const {
    return task_.task;
}

runtime::RunMode RvcSession::run_mode() const {
    return task_.mode;
}

void RvcSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (!request.audio.has_value()) {
        throw std::runtime_error("RVC prepare() requires an audio contract");
    }
    if (request.audio->sample_rate <= 0 || request.audio->channels <= 0 || request.audio->max_input_samples <= 0) {
        throw std::runtime_error("RVC prepare() received an invalid audio contract");
    }
    mark_prepared();
}

runtime::TaskResult RvcSession::run(const runtime::TaskRequest & request) {
    require_prepared("RVC run()");
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("RVC run() requires audio_input");
    }
    const auto & input_audio = *request.audio_input;
    if (input_audio.sample_rate <= 0 || input_audio.channels <= 0 || input_audio.samples.empty()) {
        throw std::runtime_error("RVC run() received invalid audio_input");
    }

    const RvcVoiceModel * voice = nullptr;
    std::optional<RvcVoiceModel> uncached_voice;
    const auto voice_model_path = request_option_or_default(request, "rvc.voice_model", "voice_model", "");
    if (voice_model_path.empty()) {
        voice = &select_packaged_voice(*assets_, request);
    } else {
        const auto key = std::filesystem::absolute(std::filesystem::path(voice_model_path)).lexically_normal().string();
        voice = user_voice_cache_.find(key);
        if (voice == nullptr) {
            auto loaded_voice = load_rvc_voice_model(key);
            const bool will_evict =
                user_voice_cache_.capacity() > 0 &&
                user_voice_cache_.size() >= user_voice_cache_.capacity();
            if (user_voice_cache_.capacity() == 0) {
                uncached_voice = std::move(loaded_voice);
                voice = &*uncached_voice;
            } else {
                user_voice_cache_.put(key, std::move(loaded_voice));
                voice = user_voice_cache_.find(key);
                if (voice == nullptr) {
                    throw std::runtime_error("RVC voice model cache failed to retain loaded voice");
                }
            }
            engine::debug::trace_log_scalar("rvc.voice_model.cache_hit", 0);
            engine::debug::trace_log_scalar(
                "rvc.voice_model.cache_slots",
                static_cast<int64_t>(user_voice_cache_.capacity()));
            engine::debug::trace_log_scalar(
                "rvc.voice_model.cache_entries",
                static_cast<int64_t>(user_voice_cache_.size()));
            engine::debug::trace_log_scalar("rvc.voice_model.cache_evicted", will_evict ? 1 : 0);
        } else {
            engine::debug::trace_log_scalar("rvc.voice_model.cache_hit", 1);
            engine::debug::trace_log_scalar(
                "rvc.voice_model.cache_slots",
                static_cast<int64_t>(user_voice_cache_.capacity()));
            engine::debug::trace_log_scalar(
                "rvc.voice_model.cache_entries",
                static_cast<int64_t>(user_voice_cache_.size()));
            engine::debug::trace_log_scalar("rvc.voice_model.cache_evicted", 0);
        }
    }
    auto output = pipeline_.infer(
        input_audio,
        *voice,
        request_config(request),
        static_cast<size_t>(std::max(1, options().backend.threads)));
    runtime::TaskResult result;
    result.audio_output = std::move(output);
    return result;
}

}  // namespace engine::models::rvc
