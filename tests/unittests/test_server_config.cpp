#include "busy_guard.h"
#include "config.h"
#include "model_memory.h"

#include "engine/framework/io/json.h"

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

void test_default_request_options() {
    const auto root = make_temp_root();
    const auto config_path = write_config(
        root,
        "server_default_request_options.json",
        R"JSON({
  "models": [
    {
      "id": "tts",
      "family": "omnivoice",
      "path": "models/OmniVoice",
      "default_request_options": {
        "num_inference_steps": 8,
        "temperature": 0.6,
        "do_sample": false
      }
    }
  ]
})JSON");

    const auto config = minitts::server::load_server_config(config_path);
    const auto & options = config.models.front().default_request_options;
    require(options.at("num_inference_steps") == "8", "integer default request option parsed");
    require(options.at("temperature") == "0.6", "float default request option parsed");
    require(options.at("do_sample") == "false", "bool default request option parsed");
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

void test_duplicate_json_keys_are_rejected() {
    bool rejected = false;
    try {
        (void) engine::io::json::parse(R"JSON({
  "voice_presets": {
    "cosette": {
      "voice_id": "cosette"
    }
  },
  "voice_presets": {
    "anna": {
      "voice_id": "anna"
    }
  }
})JSON");
    } catch (const std::runtime_error & error) {
        rejected = std::string(error.what()).find("duplicate json object key: voice_presets") != std::string::npos;
    }
    require(rejected, "duplicate json object keys are rejected");
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

void test_max_request_body_defaults_and_overrides() {
    const auto root = make_temp_root();

    const auto default_path = write_config(
        root, "request_body_default.json", std::string("{") + kMinimalModel + "}");
    require(
        minitts::server::load_server_config(default_path).max_request_body_bytes ==
            minitts::server::kDefaultMaxRequestBodyBytes,
        "max_request_body_bytes defaults to 2 GiB when omitted");

    const auto override_path = write_config(
        root,
        "request_body_override.json",
        std::string(R"JSON({"max_request_body_bytes": 1048576,)JSON") + kMinimalModel + "}");
    require(
        minitts::server::load_server_config(override_path).max_request_body_bytes == 1048576,
        "max_request_body_bytes is read from the config");

    const auto zero_path = write_config(
        root, "request_body_zero.json", std::string(R"JSON({"max_request_body_bytes": 0,)JSON") + kMinimalModel + "}");
    require(
        minitts::server::load_server_config(zero_path).max_request_body_bytes == 0,
        "max_request_body_bytes accepts 0 to reject non-empty request bodies");
}

void test_negative_max_request_body_is_rejected() {
    const auto root = make_temp_root();
    const auto config_path = write_config(
        root,
        "request_body_negative.json",
        std::string(R"JSON({"max_request_body_bytes": -1,)JSON") + kMinimalModel + "}");

    bool rejected = false;
    try {
        (void) minitts::server::load_server_config(config_path);
    } catch (const std::runtime_error & error) {
        rejected = std::string(error.what()).find("max_request_body_bytes") != std::string::npos;
    }
    require(rejected, "negative max_request_body_bytes is rejected");
}

void test_unsafe_numeric_max_request_body_is_rejected() {
    const auto root = make_temp_root();
    const auto config_path = write_config(
        root,
        "request_body_unsafe_number.json",
        std::string(R"JSON({"max_request_body_bytes": 9007199254740993,)JSON") + kMinimalModel + "}");

    bool rejected = false;
    try {
        (void) minitts::server::load_server_config(config_path);
    } catch (const std::runtime_error & error) {
        rejected = std::string(error.what()).find("max_request_body_bytes") != std::string::npos;
    }
    require(rejected, "unsafe numeric max_request_body_bytes is rejected");
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

void test_max_loaded_models_defaults_and_overrides() {
    const auto root = make_temp_root();

    const auto default_path = write_config(
        root, "max_loaded_default.json", std::string("{") + kMinimalModel + "}");
    require(
        minitts::server::load_server_config(default_path).max_loaded_models == 0,
        "max_loaded_models defaults to 0 (no limit) when omitted");

    const auto single_path = write_config(
        root, "max_loaded_single.json", std::string(R"JSON({"max_loaded_models": 1,)JSON") + kMinimalModel + "}");
    require(
        minitts::server::load_server_config(single_path).max_loaded_models == 1,
        "max_loaded_models accepts 1 to enforce a single resident model");

    const auto multi_path = write_config(
        root, "max_loaded_multi.json", std::string(R"JSON({"max_loaded_models": 3,)JSON") + kMinimalModel + "}");
    require(
        minitts::server::load_server_config(multi_path).max_loaded_models == 3,
        "max_loaded_models is read from the config");
}

void test_negative_max_loaded_models_is_rejected() {
    const auto root = make_temp_root();
    const auto config_path = write_config(
        root, "max_loaded_negative.json", std::string(R"JSON({"max_loaded_models": -1,)JSON") + kMinimalModel + "}");

    bool rejected = false;
    try {
        (void) minitts::server::load_server_config(config_path);
    } catch (const std::runtime_error & error) {
        rejected = std::string(error.what()).find("max_loaded_models") != std::string::npos;
    }
    require(rejected, "negative max_loaded_models is rejected");
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

void test_ui_configuration() {
    const auto root = make_temp_root();
    const auto config_path = write_config(
        root,
        "ui.json",
        R"JSON({
  "ui": false,
  "ui_management": true,
  "models": []
})JSON");

    const auto config = minitts::server::load_server_config(config_path);
    require(!config.ui_enabled, "ui=false disables the embedded WebUI");
    require(config.ui_management, "ui_management=true enables dynamic model management");
    require(config.models.empty(), "management hosts may start without configured models");
}

void test_empty_models_require_ui_management() {
    const auto root = make_temp_root();
    const auto config_path = write_config(
        root,
        "empty_models.json",
        R"JSON({
  "models": []
})JSON");

    bool rejected = false;
    try {
        (void) minitts::server::load_server_config(config_path);
    } catch (const std::runtime_error & error) {
        rejected = std::string(error.what()).find("ui_management") != std::string::npos;
    }
    require(rejected, "an empty static server config requires ui_management");
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

void test_idle_unload_ms_defaults_and_overrides() {
    const auto root = make_temp_root();

    const auto default_path = write_config(
        root, "idle_unload_default.json", std::string("{") + kMinimalModel + "}");
    require(
        minitts::server::load_server_config(default_path).idle_unload_ms == 0,
        "idle_unload_ms defaults to 0 (disabled) when omitted");

    const auto set_path = write_config(
        root, "idle_unload_set.json", std::string(R"JSON({"idle_unload_ms": 300000,)JSON") + kMinimalModel + "}");
    require(
        minitts::server::load_server_config(set_path).idle_unload_ms == 300000,
        "idle_unload_ms is read from the config");
}

void test_negative_idle_unload_ms_is_rejected() {
    const auto root = make_temp_root();
    const auto config_path = write_config(
        root, "idle_unload_negative.json", std::string(R"JSON({"idle_unload_ms": -1,)JSON") + kMinimalModel + "}");

    bool rejected = false;
    try {
        (void) minitts::server::load_server_config(config_path);
    } catch (const std::runtime_error & error) {
        rejected = std::string(error.what()).find("idle_unload_ms") != std::string::npos;
    }
    require(rejected, "negative idle_unload_ms is rejected");
}

void test_min_free_memory_mb_defaults_and_overrides() {
    const auto root = make_temp_root();

    const auto default_path = write_config(
        root, "min_free_default.json", std::string("{") + kMinimalModel + "}");
    require(
        minitts::server::load_server_config(default_path).min_free_memory_mb == 0,
        "min_free_memory_mb defaults to 0 (guard disabled) when omitted");

    const auto set_path = write_config(
        root, "min_free_set.json", std::string(R"JSON({"min_free_memory_mb": 256,)JSON") + kMinimalModel + "}");
    require(
        minitts::server::load_server_config(set_path).min_free_memory_mb == 256,
        "min_free_memory_mb is read from the config to opt into the guard");
}

void test_negative_min_free_memory_mb_is_rejected() {
    const auto root = make_temp_root();
    const auto config_path = write_config(
        root, "min_free_negative.json", std::string(R"JSON({"min_free_memory_mb": -1,)JSON") + kMinimalModel + "}");

    bool rejected = false;
    try {
        (void) minitts::server::load_server_config(config_path);
    } catch (const std::runtime_error & error) {
        rejected = std::string(error.what()).find("min_free_memory_mb") != std::string::npos;
    }
    require(rejected, "negative min_free_memory_mb is rejected");
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

// Mirror of the estimator's formula: weights * 1.5 plus the fixed floor, so the
// tests assert absolute values rather than "greater than".
size_t expected_estimate(size_t weights) {
    constexpr double kRuntimeOverheadFactor = 1.5;
    constexpr size_t kFixedOverhead = 128ull * 1024 * 1024;
    return static_cast<size_t>(static_cast<double>(weights) * kRuntimeOverheadFactor) + kFixedOverhead;
}

void write_file(const std::filesystem::path & path, size_t bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << std::string(bytes, 'x');
    if (!out) {
        throw std::runtime_error("failed to write test file: " + path.string());
    }
}

void test_model_memory_estimator() {
    using minitts::server::estimate_model_memory_bytes;
    using minitts::server::ServerModelConfig;
    const auto root = make_temp_root();

    // A single-file model contributes that file.
    {
        const auto path = root / "single.gguf";
        write_file(path, 1000);
        ServerModelConfig model;
        model.path = path;
        const auto estimate = estimate_model_memory_bytes(model);
        require(estimate.has_value(), "single-file model has a determinate footprint");
        require(*estimate == expected_estimate(1000), "single-file estimate sums the file");
    }

    // A directory with exactly one GGUF contributes that file.
    {
        const auto dir = root / "sole";
        std::filesystem::create_directories(dir);
        write_file(dir / "variant.gguf", 2000);
        ServerModelConfig model;
        model.path = dir;
        const auto estimate = estimate_model_memory_bytes(model);
        require(estimate.has_value(), "sole-GGUF directory has a determinate footprint");
        require(*estimate == expected_estimate(2000), "the sole GGUF is selected");
    }

    // model.gguf disambiguates a multi-GGUF directory, ignoring other variants.
    {
        const auto dir = root / "named";
        std::filesystem::create_directories(dir);
        write_file(dir / "a.gguf", 3000);
        write_file(dir / "model.gguf", 4000);
        ServerModelConfig model;
        model.path = dir;
        const auto estimate = estimate_model_memory_bytes(model);
        require(estimate.has_value(), "model.gguf makes the directory determinate");
        require(*estimate == expected_estimate(4000), "model.gguf wins over the other variants");
    }

    // Several GGUFs and no model.gguf: the loader rejects the spec-driven case
    // with its own error, so the footprint is indeterminate and the guard skips.
    {
        const auto dir = root / "ambiguous";
        std::filesystem::create_directories(dir);
        write_file(dir / "a.gguf", 100);
        write_file(dir / "b.gguf", 100);
        ServerModelConfig model;
        model.path = dir;
        require(
            !estimate_model_memory_bytes(model).has_value(),
            "an ambiguous multi-GGUF directory has an indeterminate footprint");
    }

    // A directory with no GGUF is a safetensors/HF checkpoint: the tree is summed.
    {
        const auto dir = root / "tree";
        std::filesystem::create_directories(dir / "sub");
        write_file(dir / "model.safetensors", 5000);
        write_file(dir / "sub" / "chunk.bin", 6000);
        ServerModelConfig model;
        model.path = dir;
        const auto estimate = estimate_model_memory_bytes(model);
        require(estimate.has_value(), "a no-GGUF directory has a determinate footprint");
        require(*estimate == expected_estimate(11000), "a checkpoint tree is summed recursively");
    }

    // Relative auxiliary session files resolve against the model directory.
    {
        // Not named "aux": that is a reserved DOS device name and cannot be
        // created on Windows.
        const auto dir = root / "sidecar";
        std::filesystem::create_directories(dir);
        write_file(dir / "model.gguf", 7000);
        write_file(dir / "head.bin", 8000);
        ServerModelConfig model;
        model.path = dir;
        model.session_options["aux_path"] = "head.bin";
        const auto estimate = estimate_model_memory_bytes(model);
        require(estimate.has_value(), "an aux-resolved directory has a determinate footprint");
        require(*estimate == expected_estimate(15000), "a relative aux path resolves against the model directory");
    }
}

}  // namespace

int main() {
    try {
        test_inline_default_and_named_presets();
        test_default_preset_name();
        test_default_request_options();
        test_missing_default_preset_name_is_rejected();
        test_duplicate_json_keys_are_rejected();
        test_busy_timeout_defaults_and_overrides();
        test_max_request_body_defaults_and_overrides();
        test_negative_max_request_body_is_rejected();
        test_unsafe_numeric_max_request_body_is_rejected();
        test_negative_busy_timeout_is_rejected();
        test_max_loaded_models_defaults_and_overrides();
        test_negative_max_loaded_models_is_rejected();
        test_idle_unload_ms_defaults_and_overrides();
        test_negative_idle_unload_ms_is_rejected();
        test_min_free_memory_mb_defaults_and_overrides();
        test_negative_min_free_memory_mb_is_rejected();
        test_per_model_busy_timeout();
        test_negative_per_model_busy_timeout_is_rejected();
        test_ui_configuration();
        test_empty_models_require_ui_management();
        test_request_timeout_is_clamped_to_policy();
        test_model_run_overrun_predicate();
        test_model_memory_estimator();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "server_config_test passed\n";
    return 0;
}
