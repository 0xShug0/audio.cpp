#include "engine/community_models/mms_forced_aligner/session.h"
#include "test_assert.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::test::require;

using engine::community_models::mms_forced_aligner::MmsForcedAlignerAssets;
using engine::community_models::mms_forced_aligner::MmsForcedAlignerSession;
using engine::community_models::mms_forced_aligner::make_mms_forced_aligner_loader;

// Minimal tensor source; the session only checks it for null-ness before the
// emission stage, so no tensors are served.
class DummyTensorSource final : public engine::assets::TensorSource {
public:
    const std::filesystem::path & source_path() const noexcept override {
        static const std::filesystem::path kPath = std::filesystem::path("dummy");
        return kPath;
    }
    bool has_tensor(std::string_view) const noexcept override {
        return false;
    }
    engine::assets::TensorMetadata require_metadata(std::string_view name) const override {
        throw std::runtime_error("dummy source has no metadata for " + std::string(name));
    }
    std::vector<engine::assets::TensorMetadata> tensors() const override {
        return {};
    }
    void release_storage() const override {}
    engine::assets::RawTensorData require_tensor_data(std::string_view name) const override {
        throw std::runtime_error("dummy source has no tensor data for " + std::string(name));
    }
    std::vector<float> require_f32(
        std::string_view,
        const std::optional<std::vector<int64_t>> &) const override {
        throw std::runtime_error("dummy source serves no f32 tensors");
    }
    std::optional<std::vector<float>> optional_f32(
        std::string_view,
        const std::optional<std::vector<int64_t>> &) const override {
        return std::nullopt;
    }
    int64_t require_i64_scalar(std::string_view name) const override {
        throw std::runtime_error("dummy source has no scalar " + std::string(name));
    }
};

std::shared_ptr<const MmsForcedAlignerAssets> dummy_assets() {
    auto assets = std::make_shared<MmsForcedAlignerAssets>();
    assets->model_weights = std::make_shared<DummyTensorSource>();
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> contract() {
    const auto out = engine::model_spec::model_contract("mms_forced_aligner");
    require(out.has_value(), "mms_forced_aligner spec must resolve from the workspace");
    return std::make_shared<engine::model_spec::ModelContract>(std::move(*out));
}

engine::core::BackendConfig cpu_backend() {
    return engine::core::BackendConfig{engine::core::BackendType::Cpu, 0, 1};
}

engine::runtime::TaskSpec align_task() {
    engine::runtime::TaskSpec task;
    task.task = engine::runtime::VoiceTaskKind::Alignment;
    task.mode = engine::runtime::RunMode::Offline;
    return task;
}

std::unique_ptr<MmsForcedAlignerSession> make_session(engine::runtime::SessionOptions options = {}) {
    options.backend = cpu_backend();
    return std::make_unique<MmsForcedAlignerSession>(align_task(), std::move(options), dummy_assets(), contract());
}

engine::runtime::TaskRequest valid_request() {
    engine::runtime::TaskRequest request;
    engine::runtime::AudioBuffer audio;
    audio.sample_rate = 48000;
    audio.channels = 1;
    audio.samples.assign(48000, 0.0F);
    request.audio_input = std::move(audio);
    request.text_input = engine::runtime::Transcript{"hello world", "en"};
    return request;
}

bool throws(std::function<void()> call) {
    try {
        call();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

void test_wrong_task() {
    engine::runtime::TaskSpec task = align_task();
    task.task = engine::runtime::VoiceTaskKind::Asr;
    engine::runtime::SessionOptions options;
    options.backend = cpu_backend();
    require(throws([&] {
        (void) MmsForcedAlignerSession(task, options, dummy_assets(), contract());
    }), "non-alignment task must be rejected");
}

void test_wrong_mode() {
    engine::runtime::TaskSpec task = align_task();
    task.mode = engine::runtime::RunMode::Streaming;
    engine::runtime::SessionOptions options;
    options.backend = cpu_backend();
    require(throws([&] {
        (void) MmsForcedAlignerSession(task, options, dummy_assets(), contract());
    }), "streaming mode must be rejected");
}

void test_unknown_session_option() {
    engine::runtime::SessionOptions options;
    options.options.emplace("mms_forced_aligner.bogus", "1");
    require(throws([&] { (void) make_session(std::move(options)); }),
            "unknown session option must be rejected");
}

void test_context_not_less_than_window() {
    engine::runtime::SessionOptions options;
    options.options.emplace("mms_forced_aligner.emission_context_sec", "30");
    options.options.emplace("mms_forced_aligner.emission_window_sec", "30");
    require(throws([&] { (void) make_session(std::move(options)); }),
            "context >= window must be rejected");
}

void test_negative_context() {
    engine::runtime::SessionOptions options;
    options.options.emplace("mms_forced_aligner.emission_context_sec", "-1");
    require(throws([&] { (void) make_session(std::move(options)); }),
            "negative context must be rejected");
}

void test_nonpositive_resource_limit() {
    engine::runtime::SessionOptions options;
    options.options.emplace("mms_forced_aligner.max_alignment_cells", "0");
    require(throws([&] { (void) make_session(std::move(options)); }),
            "nonpositive max_alignment_cells must be rejected");
    engine::runtime::SessionOptions options2;
    options2.options.emplace("mms_forced_aligner.max_target_tokens", "0");
    require(throws([&] { (void) make_session(std::move(options2)); }),
            "nonpositive max_target_tokens must be rejected");
}

void test_missing_prepare_contracts() {
    auto session = make_session();
    require(throws([&] { session->prepare({}); }),
            "prepare without audio/text contracts must be rejected");
}

void test_run_before_prepare() {
    auto session = make_session();
    require(throws([&] { (void) session->run(valid_request()); }),
            "run before prepare must be rejected");
}

void test_unsupported_chunk_modes() {
    auto session = make_session();
    engine::runtime::SessionPreparationRequest preparation;
    preparation.audio = engine::runtime::AudioPreparationContract{48000, 1, 48000 * 60};
    preparation.text = engine::runtime::Transcript{"hello world", "en"};
    session->prepare(preparation);
    for (const char * mode : {"fixed", "quiet_energy", "vad"}) {
        auto request = valid_request();
        request.options.emplace("audio_chunk_mode", mode);
        require(throws([&] { (void) session->run(request); }),
                std::string("chunk mode ") + mode + " must be rejected");
    }
}

void test_missing_audio() {
    auto session = make_session();
    engine::runtime::SessionPreparationRequest preparation;
    preparation.audio = engine::runtime::AudioPreparationContract{48000, 1, 48000 * 60};
    preparation.text = engine::runtime::Transcript{"hello world", "en"};
    session->prepare(preparation);
    auto request = valid_request();
    request.audio_input.reset();
    require(throws([&] { (void) session->run(request); }), "missing audio must be rejected");
}

void test_missing_transcript() {
    auto session = make_session();
    engine::runtime::SessionPreparationRequest preparation;
    preparation.audio = engine::runtime::AudioPreparationContract{48000, 1, 48000 * 60};
    preparation.text = engine::runtime::Transcript{"hello world", "en"};
    session->prepare(preparation);
    auto request = valid_request();
    request.text_input.reset();
    require(throws([&] { (void) session->run(request); }), "missing transcript must be rejected");
    request = valid_request();
    request.text_input = engine::runtime::Transcript{"", "en"};
    require(throws([&] { (void) session->run(request); }), "empty transcript must be rejected");
    request = valid_request();
    request.text_input = engine::runtime::Transcript{"hello", ""};
    require(throws([&] { (void) session->run(request); }), "empty language must be rejected");
}

void test_language_conflict() {
    auto session = make_session();
    engine::runtime::SessionPreparationRequest preparation;
    preparation.audio = engine::runtime::AudioPreparationContract{48000, 1, 48000 * 60};
    preparation.text = engine::runtime::Transcript{"hello world", "en"};
    session->prepare(preparation);
    auto request = valid_request();
    request.options.emplace("language", "nl");
    require(throws([&] { (void) session->run(request); }), "conflicting language must be rejected");
}

void test_unknown_request_option() {
    auto session = make_session();
    engine::runtime::SessionPreparationRequest preparation;
    preparation.audio = engine::runtime::AudioPreparationContract{48000, 1, 48000 * 60};
    preparation.text = engine::runtime::Transcript{"hello world", "en"};
    session->prepare(preparation);
    auto request = valid_request();
    request.options.emplace("bogus_option", "1");
    require(throws([&] { (void) session->run(request); }), "unknown request option must be rejected");
}

void test_loader_factory_family() {
    const auto loader = make_mms_forced_aligner_loader();
    require(loader != nullptr, "loader factory");
    require(loader->family() == "mms_forced_aligner", "loader family name");
}

}  // namespace

int main() {
    try {
        test_wrong_task();
        test_wrong_mode();
        test_unknown_session_option();
        test_context_not_less_than_window();
        test_negative_context();
        test_nonpositive_resource_limit();
        test_missing_prepare_contracts();
        test_run_before_prepare();
        test_unsupported_chunk_modes();
        test_missing_audio();
        test_missing_transcript();
        test_language_conflict();
        test_unknown_request_option();
        test_loader_factory_family();
        std::cout << "mms_forced_aligner_session_test passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "mms_forced_aligner_session_test: %s\n", error.what());
        return 1;
    }
}
