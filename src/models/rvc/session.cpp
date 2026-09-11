#include "engine/models/rvc/session.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

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

constexpr const char * kFamily = "rvc";

std::shared_ptr<const RvcAssets> require_assets(std::shared_ptr<const RvcAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("RVC session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("RVC session requires a model contract");
    }
    return contract;
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
        {"rvc.voice_cache_slots"})
        .value_or(kDefaultCacheSlots);
    if (slots < 0) {
        throw std::runtime_error("rvc.voice_cache_slots must be non-negative");
    }
    if (static_cast<std::uint64_t>(slots) > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("rvc.voice_cache_slots is too large");
    }
    return static_cast<std::size_t>(slots);
}

const RvcVoiceModel & select_packaged_voice(const RvcAssets & assets, const runtime::TaskRequest & request) {
    std::string voice_id = runtime::find_option(request.options, {"voice_id"}).value_or("default");
    const auto voice = assets.voices.find(voice_id);
    if (voice == assets.voices.end()) {
        throw std::runtime_error("unknown RVC voice id: " + voice_id);
    }
    return voice->second;
}

RvcInferenceConfig request_config(const runtime::TaskRequest & request) {
    RvcInferenceConfig config;
    config.pitch_extractor = runtime::find_option(request.options, {"pitch_extractor"}).value_or("rmvpe");
    config.pitch_path = runtime::find_option(request.options, {"pitch_path"}).value_or("");
    config.retrieval_index_path = runtime::find_option(request.options, {"retrieval_index_path"}).value_or("");
    config.semitone_shift = runtime::parse_int_option(request.options, {"semitone_shift"}).value_or(0);
    config.retrieval_blend = runtime::parse_float_option(request.options, {"retrieval_blend"}).value_or(0.0F);
    config.pitch_filter_radius = runtime::parse_int_option(request.options, {"pitch_filter_radius"}).value_or(3);
    config.output_sample_rate = runtime::parse_int_option(request.options, {"output_sample_rate"}).value_or(0);
    config.rms_mix_rate = runtime::parse_float_option(request.options, {"rms_mix_rate"}).value_or(0.25F);
    config.unvoiced_protection = runtime::parse_float_option(request.options, {"unvoiced_protection"}).value_or(0.33F);
    config.speaker_id = runtime::parse_int_option(request.options, {"speaker_id"}).value_or(0);
    config.audio_pad_duration_sec = runtime::parse_int_option(request.options, {"audio_pad_duration_sec"}).value_or(1);
    config.split_query_sec = runtime::parse_int_option(request.options, {"split_query_sec"}).value_or(5);
    config.split_center_sec = runtime::parse_int_option(request.options, {"split_center_sec"}).value_or(30);
    config.split_threshold_sec = runtime::parse_int_option(request.options, {"split_threshold_sec"}).value_or(32);
    return config;
}

}  // namespace

RvcSession::RvcSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const RvcAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))),
      weight_storage_type_(rvc_weight_type_from_options(RuntimeSessionBase::options())),
      pipeline_(assets_, execution_context().config(), weight_storage_type_),
      user_voice_cache_(user_voice_cache_slots_from_options(RuntimeSessionBase::options())) {
    runtime::validate_spec_backed_session_options(RuntimeSessionBase::options(), *contract_, kFamily, "RVC");
    if (task_.task != runtime::VoiceTaskKind::VoiceConversion) {
        throw std::runtime_error("RVC models only support --task vc");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("RVC models only support offline mode");
    }
}

std::string RvcSession::family() const {
    return kFamily;
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
    runtime::validate_spec_backed_request_options(request.options, *contract_, "RVC");
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("RVC run() requires audio_input");
    }
    const auto & input_audio = *request.audio_input;
    if (input_audio.sample_rate <= 0 || input_audio.channels <= 0 || input_audio.samples.empty()) {
        throw std::runtime_error("RVC run() received invalid audio_input");
    }

    const RvcVoiceModel * voice = nullptr;
    std::optional<RvcVoiceModel> uncached_voice;
    const auto voice_model_path = runtime::find_option(request.options, {"voice_model_path"}).value_or("");
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

// Loading adapter: RVC uses the schema-v1 spec-backed loader, so the loader
// wiring stays beside the session it constructs.
std::shared_ptr<runtime::IVoiceModelLoader> make_rvc_loader() {
    runtime::SpecBackedVoiceModelConfig<RvcAssets> config;
    config.family = kFamily;
    config.load_assets = load_rvc_assets;
    config.create_session = [](const runtime::TaskSpec & task,
                                const runtime::SessionOptions & options,
                                std::shared_ptr<const RvcAssets> assets,
                                std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        return std::make_unique<RvcSession>(
            task,
            options,
            std::move(assets),
            std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::rvc
