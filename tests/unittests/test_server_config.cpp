#include "busy_guard.h"
#include "config.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::filesystem::path make_temp_root() {
    const auto root = std::filesystem::temp_directory_path() / "audiocpp_server_config_test";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root);
    return root;
}

std::filesystem::path write_config(
    const std::filesystem::path & root,
    const std::string & name,
    const std::string & text) {
    const auto path = root / name;
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to create test config: " + path.string());
    }
    out << text;
    if (!out) {
        throw std::runtime_error("failed to write test config: " + path.string());
    }
    return path;
}

void test_inline_default_and_named_presets() {
    const auto root = make_temp_root();
    const auto config_path = write_config(
        root,
        "server.json",
        R"JSON({
  "host": "127.0.0.1",
  "port": 8080,
  "backend": "cpu",
  "models": [
    {
      "id": "tts",
      "family": "omnivoice",
      "path": "models/OmniVoice",
      "task": "tts",
      "default_voice_preset": {
        "voice_ref": "voices/default.wav",
        "reference_text": "default transcript"
      },
      "voice_presets": {
        "assistant": {
          "voice_ref": "voices/assistant.wav",
          "reference_text": "assistant transcript"
        },
        "builtin": {
          "voice_id": "alba"
        }
      }
    }
  ]
})JSON");

    const auto config = minitts::server::load_server_config(config_path);
    require(config.models.size() == 1, "one model parsed");
    const auto & model = config.models.front();
    require(model.path == root / "models/OmniVoice", "model path is resolved relative to config");
    require(model.default_voice_preset.has_value(), "inline default preset parsed");
    require(model.default_voice_preset->voice_ref == root / "voices/default.wav", "default voice_ref path resolved");
    require(model.default_voice_preset->reference_text == "default transcript", "default reference_text parsed");
    require(model.voice_presets.size() == 2, "named presets parsed");
    require(model.voice_presets.at("assistant").voice_ref == root / "voices/assistant.wav", "named voice_ref path resolved");
    require(model.voice_presets.at("assistant").reference_text == "assistant transcript", "named reference_text parsed");
    require(model.voice_presets.at("builtin").voice_id == "alba", "named voice_id parsed");
}

void test_default_preset_name() {
    const auto root = make_temp_root();
    const auto config_path = write_config(
        root,
        "server_named_default.json",
        R"JSON({
  "models": [
    {
      "id": "tts",
      "family": "pocket_tts",
      "path": "models/pocket-tts",
      "voice_presets": {
        "narrator": {
          "voice_id": "cosette"
        }
      },
      "default_voice_preset": "narrator"
    }
  ]
})JSON");

    const auto config = minitts::server::load_server_config(config_path);
    const auto & model = config.models.front();
    require(model.default_voice_preset_id == "narrator", "default preset name parsed");
    require(!model.default_voice_preset.has_value(), "named default does not create inline preset");
}

void test_missing_default_preset_name_is_rejected() {
    const auto root = make_temp_root();
    const auto config_path = write_config(
        root,
        "server_bad_default.json",
        R"JSON({
  "models": [
    {
      "id": "tts",
      "family": "pocket_tts",
      "path": "models/pocket-tts",
      "voice_presets": {
        "narrator": {
          "voice_id": "cosette"
        }
      },
      "default_voice_preset": "missing"
    }
  ]
})JSON");

    bool rejected = false;
    try {
        (void) minitts::server::load_server_config(config_path);
    } catch (const std::runtime_error & error) {
        rejected = std::string(error.what()).find("does not match") != std::string::npos;
    }
    require(rejected, "unknown default preset name is rejected");
}

const char * const kMinimalModel = R"JSON(
  "models": [
    {
      "id": "tts",
      "family": "pocket_tts",
      "path": "models/pocket-tts"
    }
  ]
)JSON";

void test_busy_timeout_defaults_and_overrides() {
    const auto root = make_temp_root();

    const auto default_path = write_config(
        root, "busy_default.json", std::string("{") + kMinimalModel + "}");
    require(
        minitts::server::load_server_config(default_path).busy_timeout_ms == 300000,
        "busy_timeout_ms defaults to 5 minutes when omitted");

    const auto override_path = write_config(
        root, "busy_override.json", std::string(R"JSON({"busy_timeout_ms": 90000,)JSON") + kMinimalModel + "}");
    require(
        minitts::server::load_server_config(override_path).busy_timeout_ms == 90000,
        "busy_timeout_ms is read from the config");

    const auto disabled_path = write_config(
        root, "busy_disabled.json", std::string(R"JSON({"busy_timeout_ms": 0,)JSON") + kMinimalModel + "}");
    require(
        minitts::server::load_server_config(disabled_path).busy_timeout_ms == 0,
        "busy_timeout_ms accepts 0 to disable the guard");
}

void test_negative_busy_timeout_is_rejected() {
    const auto root = make_temp_root();
    const auto config_path = write_config(
        root, "busy_negative.json", std::string(R"JSON({"busy_timeout_ms": -1,)JSON") + kMinimalModel + "}");

    bool rejected = false;
    try {
        (void) minitts::server::load_server_config(config_path);
    } catch (const std::runtime_error & error) {
        rejected = std::string(error.what()).find("busy_timeout_ms") != std::string::npos;
    }
    require(rejected, "negative busy_timeout_ms is rejected");
}

void test_per_model_busy_timeout() {
    const auto root = make_temp_root();
    const auto config_path = write_config(
        root,
        "busy_per_model.json",
        R"JSON({
  "busy_timeout_ms": 300000,
  "models": [
    {"id": "tts",   "family": "pocket_tts", "path": "models/a", "busy_timeout_ms": 30000},
    {"id": "music", "family": "pocket_tts", "path": "models/b", "busy_timeout_ms": 900000},
    {"id": "asr",   "family": "pocket_tts", "path": "models/c"}
  ]
})JSON");

    const auto config = minitts::server::load_server_config(config_path);
    require(config.models.at(0).busy_timeout_ms == 30000, "per-model busy_timeout_ms parsed");
    require(config.models.at(1).busy_timeout_ms == 900000, "a model may exceed the top-level value");
    require(
        !config.models.at(2).busy_timeout_ms.has_value(),
        "a model without the key inherits the top-level value");
}

void test_negative_per_model_busy_timeout_is_rejected() {
    const auto root = make_temp_root();
    const auto config_path = write_config(
        root,
        "busy_per_model_negative.json",
        R"JSON({
  "models": [
    {"id": "tts", "family": "pocket_tts", "path": "models/a", "busy_timeout_ms": -1}
  ]
})JSON");

    bool rejected = false;
    try {
        (void) minitts::server::load_server_config(config_path);
    } catch (const std::runtime_error & error) {
        rejected = std::string(error.what()).find("busy_timeout_ms for model tts") != std::string::npos;
    }
    require(rejected, "negative per-model busy_timeout_ms is rejected, naming the model");
}

// A request may shorten its own wait but must never lengthen it past server policy,
// otherwise a client could reintroduce the unbounded hang the guard prevents.
void test_request_timeout_is_clamped_to_policy() {
    using minitts::server::resolve_busy_timeout_ms;

    require(resolve_busy_timeout_ms(900000, std::nullopt) == 900000, "no request override uses policy");
    require(resolve_busy_timeout_ms(900000, 60000) == 60000, "a shorter request bound is honored");
    require(resolve_busy_timeout_ms(900000, 999999) == 900000, "a longer request bound is clamped");
    require(resolve_busy_timeout_ms(900000, 900000) == 900000, "an equal request bound is unchanged");

    // 0 means unbounded, so it must compare as +infinity rather than as a tiny value.
    require(
        resolve_busy_timeout_ms(900000, 0) == 900000,
        "a request asking for unbounded is still capped by policy");
    require(
        resolve_busy_timeout_ms(0, 60000) == 60000,
        "under unbounded policy a request may still bound its own wait");
    require(
        resolve_busy_timeout_ms(0, std::nullopt) == 0,
        "unbounded policy with no override stays unbounded");
    require(
        resolve_busy_timeout_ms(0, 0) == 0,
        "unbounded on both sides stays unbounded");
}

void test_model_run_overrun_predicate() {
    using minitts::server::model_run_has_overrun;

    require(!model_run_has_overrun(0, 10'000'000, 1000), "an idle model never counts as overrun");
    require(!model_run_has_overrun(1000, 1500, 1000), "a run inside the timeout waits normally");
    require(!model_run_has_overrun(1000, 2000, 1000), "a run exactly at the timeout is not yet overrun");
    require(model_run_has_overrun(1000, 2001, 1000), "a run past the timeout fails fast");
    require(!model_run_has_overrun(1000, 10'000'000, 0), "timeout 0 disables the guard");
    require(!model_run_has_overrun(1000, 10'000'000, -1), "a non-positive timeout disables the guard");
}

}  // namespace

int main() {
    try {
        test_inline_default_and_named_presets();
        test_default_preset_name();
        test_missing_default_preset_name_is_rejected();
        test_busy_timeout_defaults_and_overrides();
        test_negative_busy_timeout_is_rejected();
        test_per_model_busy_timeout();
        test_negative_per_model_busy_timeout_is_rejected();
        test_request_timeout_is_clamped_to_policy();
        test_model_run_overrun_predicate();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "server_config_test passed\n";
    return 0;
}
