#include "busy_guard.h"
#include "config.h"
#include "model_memory.h"

#include "engine/framework/io/json.h"
#include "engine/framework/runtime/registry.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv(name);
    } else {
        setenv(name, value, 1);
    }
#endif
}

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
        },
        "embedding": {
          "options": {
            "speaker_embedding_file": "voices/embedding.safetensors",
            "num_steps": 8
          }
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
    require(model.voice_presets.size() == 3, "named presets parsed");
    require(model.voice_presets.at("assistant").voice_ref == root / "voices/assistant.wav", "named voice_ref path resolved");
    require(model.voice_presets.at("assistant").reference_text == "assistant transcript", "named reference_text parsed");
    require(model.voice_presets.at("builtin").voice_id == "alba", "named voice_id parsed");
    const auto & options = model.voice_presets.at("embedding").options;
    require(options.size() == 2, "preset options parsed");
    require(options.at("speaker_embedding_file") == "voices/embedding.safetensors",
            "preset option values are kept as given, not resolved as paths");
    require(options.at("num_steps") == "8", "numeric preset option parsed");
    require(model.voice_presets.at("assistant").options.empty(), "presets without options have none");
}

void test_bad_preset_options_are_rejected() {
    const auto root = make_temp_root();
    const auto not_object = write_config(root, "server_preset_options_type.json", R"JSON({
  "models": [{"id": "tts", "family": "irodori_tts", "path": "models/irodori",
              "voice_presets": {"narrator": {"options": "speaker.safetensors"}}}]
})JSON");
    bool rejected = false;
    try {
        (void) minitts::server::load_server_config(not_object);
    } catch (const std::runtime_error & error) {
        rejected = std::string(error.what()).find("options must be an object") != std::string::npos;
    }
    require(rejected, "non-object preset options are rejected");

    const auto empty = write_config(root, "server_preset_options_empty.json", R"JSON({
  "models": [{"id": "tts", "family": "irodori_tts", "path": "models/irodori",
              "voice_presets": {"narrator": {"options": {}}}}]
})JSON");
    rejected = false;
    try {
        (void) minitts::server::load_server_config(empty);
    } catch (const std::runtime_error & error) {
        rejected = std::string(error.what()).find("must set voice_id, voice_ref, reference_text, or options") !=
            std::string::npos;
    }
    require(rejected, "a preset with only empty options is rejected");
    std::filesystem::remove_all(root);
}

void test_model_spec_override_selection() {
    const auto root = make_temp_root();
    const auto path = write_config(root, "spec-override.json", R"JSON({
      "model_spec_override": "embedded",
      "models": [
        {"id": "embedded", "family": "kokoro_tts", "path": "model.gguf", "model_spec_override": "embedded"},
        {"id": "file", "family": "kokoro_tts", "path": "model.gguf", "model_spec_override": "specs/kokoro_tts.json"},
        {"id": "literal", "family": "kokoro_tts", "path": "model.gguf", "model_spec_override": "./embedded"}
      ]
    })JSON");
    const auto config = minitts::server::load_server_config(path);
    require(config.model_spec_override == std::filesystem::path("embedded"), "global embedded selector stays literal");
    require(config.models[0].model_spec_override == std::filesystem::path("embedded"), "per-model embedded selector stays literal");
    require(config.models[1].model_spec_override == root / "specs/kokoro_tts.json", "spec file still resolves relative to config");
    require(config.models[2].model_spec_override == (root / "./embedded"), "literal embedded path stays a path");
    std::filesystem::remove_all(root);
}

void test_builtin_utility_model_paths() {
    const auto root = make_temp_root();
    const auto absolute = (root / "weights" / "rnnoise.safetensors").generic_string();
    const auto config_path = write_config(root, "server.json",
        "{\"models\":["
        "{\"id\":\"relative\",\"family\":\"builtin_audio_utils\","
        "\"path\":\"weights/rnnoise.safetensors\",\"task\":\"s2s\","
        "\"load_options\":{\"utility\":\"rnnoise\"}},"
        "{\"id\":\"absolute\",\"family\":\"builtin_audio_utils\","
        "\"path\":\"" + absolute + "\",\"task\":\"s2s\","
        "\"load_options\":{\"utility\":\"rnnoise\"}}]}");
    const auto config = minitts::server::load_server_config(config_path);
    require(config.models.at(0).path == root / "weights/rnnoise.safetensors",
        "builtin utility relative path resolves against config directory");
    require(config.models.at(1).path == std::filesystem::path(absolute),
        "builtin utility absolute path is preserved");
    require(config.models.at(0).load_options.at("utility") == "rnnoise",
        "builtin utility selector is forwarded");
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

void test_default_session_options() {
    const auto root = make_temp_root();
    const auto config_path = write_config(
        root,
        "server_default_session_options.json",
        R"JSON({
  "default_session_options": {
    "irodori_tts.mem_saver": true,
    "irodori_tts.codec_decode_chunk_steps": 100,
    "qwen3_asr.max_tokens": 256
  },
  "models": [
    {
      "id": "tts",
      "family": "irodori_tts",
      "path": "models/irodori.gguf",
      "session_options": {
        "irodori_tts.codec_decode_chunk_steps": "50"
      }
    },
    {
      "id": "other",
      "family": "omnivoice",
      "path": "models/OmniVoice"
    }
  ]
})JSON");

    const auto config = minitts::server::load_server_config(config_path);
    require(config.default_session_options.size() == 3, "default session options parsed");
    const auto & tts = config.models.at(0).session_options;
    require(tts.at("irodori_tts.mem_saver") == "true", "default fills an unset session option");
    require(tts.at("irodori_tts.codec_decode_chunk_steps") == "50", "the model's own session option wins");
    require(tts.count("qwen3_asr.max_tokens") == 0, "defaults of another family are not applied");
    require(config.models.at(1).session_options.empty(), "a family without defaults is unchanged");

    // Models loaded later (POST /v1/models/load) get the same treatment.
    minitts::server::ServerModelConfig loaded;
    loaded.family = "irodori_tts";
    loaded.session_options["irodori_tts.mem_saver"] = "false";
    minitts::server::apply_default_session_options(config, loaded);
    require(loaded.session_options.at("irodori_tts.mem_saver") == "false", "a loaded model's own value wins");
    require(loaded.session_options.at("irodori_tts.codec_decode_chunk_steps") == "100",
            "a loaded model gets the family default");

    for (const std::string bad : {R"({"default_session_options": {"mem_saver": true}, "models": []})",
                                  R"({"default_session_options": {".mem_saver": true}, "models": []})",
                                  R"({"default_session_options": {"irodori_tts.": true}, "models": []})",
                                  R"({"default_session_options": ["irodori_tts.mem_saver"], "models": []})"}) {
        const auto bad_path = write_config(root, "server_bad_default_session_options.json", bad);
        bool rejected = false;
        try {
            (void) minitts::server::load_server_config(bad_path);
        } catch (const std::runtime_error & error) {
            rejected = std::string(error.what()).find("default_session_options") != std::string::npos;
        }
        require(rejected, "malformed default_session_options is rejected: " + bad);
    }
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
    require(!config.ui_requested, "ui=false does not ask for the WebUI");
    require(config.ui_management, "ui_management=true enables dynamic model management");
    require(config.models.empty(), "management hosts may start without configured models");

    const auto default_path = write_config(root, "ui_default.json", R"JSON({
  "models": [{"id": "tts", "family": "pocket_tts", "path": "models/pocket-tts"}]
})JSON");
    const auto by_default = minitts::server::load_server_config(default_path);
    require(by_default.ui_enabled && !by_default.ui_requested, "the WebUI is on by default but not asked for");

    const auto asked_path = write_config(root, "ui_asked.json", R"JSON({
  "ui": true,
  "models": [{"id": "tts", "family": "pocket_tts", "path": "models/pocket-tts"}]
})JSON");
    const auto asked = minitts::server::load_server_config(asked_path);
    require(asked.ui_enabled && asked.ui_requested, "ui=true asks for the WebUI");
}

void test_configured_model_fields_are_inherited() {
    const auto root = make_temp_root();
    const auto config_path = write_config(root, "inherit.json", R"JSON({
  "models": [
    {
      "id": "tts",
      "family": "irodori_tts",
      "path": "models/irodori",
      "busy_timeout_ms": 90000,
      "default_request_options": {"num_steps": "16"},
      "voice_presets": {"narrator": {"voice_ref": "voices/narrator.wav"}},
      "default_voice_preset": "narrator"
    }
  ]
})JSON");
    const auto config = minitts::server::load_server_config(config_path);
    const auto & configured = config.models.front();

    // What a WebUI load request carries: no request options or presets.
    minitts::server::ServerModelConfig loaded;
    loaded.id = "tts";
    loaded.family = "irodori_tts";
    loaded.path = "models/other-irodori";
    loaded.session_options = {{"irodori_tts.mem_saver", "true"}};
    minitts::server::inherit_configured_model_fields(configured, loaded);
    require(loaded.default_request_options.at("num_steps") == "16", "request options come from the config");
    require(loaded.voice_presets.count("narrator") == 1, "voice presets come from the config");
    require(loaded.default_voice_preset_id.value_or("") == "narrator", "the default preset comes from the config");
    require(loaded.busy_timeout_ms.value_or(0) == 90000, "the busy timeout comes from the config");
    require(loaded.path == "models/other-irodori", "the load keeps its own path");
    require(loaded.session_options.at("irodori_tts.mem_saver") == "true", "the load keeps its own session options");

    minitts::server::ServerModelConfig other;
    other.id = "tts";
    other.family = "pocket_tts";
    other.path = "models/pocket-tts";
    minitts::server::inherit_configured_model_fields(configured, other);
    require(other.default_request_options.empty() && other.voice_presets.empty() &&
                !other.default_voice_preset_id.has_value() && !other.busy_timeout_ms.has_value(),
            "a load of another family takes nothing from the config");

    minitts::server::ServerModelConfig design;
    design.id = "tts";
    design.family = "irodori_tts";
    design.task = "vdes";
    design.path = "models/irodori";
    minitts::server::inherit_configured_model_fields(configured, design);
    require(design.voice_presets.empty(), "a load of another task takes nothing from the config");
    std::filesystem::remove_all(root);
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

// The engines a plan declares are priced by the loader that owns the family, so a build that did not
// compile that family has no loader to ask and the guard has nothing to price: CI's sanitizer builds
// link one family (`builtin_audio_utils`), which reports no engines by contract. The cases below
// therefore state the priced expectation when the family is in this registry, and the empty-plan
// contract when it is not -- which is the same contract, seen from the other side.
bool family_compiled(const std::string & family) {
    return engine::runtime::make_default_registry().supports_family(family);
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

    // An exec_plan engine is counted by name, through the family's loader: the option value is JSON,
    // so the path scan sees nothing, and a name relative to the model directory is how a deployment
    // spells an engine installed with the weights. The names here are not engine images and this
    // gate has no NPU runtime, so the runtime cannot price them and they fall back to their file
    // size -- at 1.0, not the runtime factor, because a compiled engine is resident at its own
    // allocation and builds no graph.
    {
        const auto dir = root / "plan";
        std::filesystem::create_directories(dir);
        write_file(dir / "model.gguf", 7000);
        write_file(dir / "flow_seq110.rknn", 900);
        write_file(dir / "flow_seq200.rknn", 1100);
        const auto elsewhere = root / "plan-abs.rknn";
        write_file(elsewhere, 500);
        ServerModelConfig model;
        model.family = "cosyvoice3";
        model.path = dir;
        // generic_string(), not string(): on Windows the native separators are
        // backslashes, which are invalid escapes inside the JSON plan.
        model.session_options["cosyvoice3.exec_plan"] =
            R"({"flow":{"impl":"rknn","artifacts":{"main":"flow_seq110.rknn,flow_seq200.rknn,)" +
            elsewhere.generic_string() + R"("}}})";
        const auto estimate = estimate_model_memory_bytes(model);
        const auto priced = family_compiled("cosyvoice3") ? size_t{900 + 1100 + 500} : size_t{0};
        require(estimate.has_value(), "an exec_plan model has a determinate footprint");
        require(
            *estimate == expected_estimate(7000) + priced,
            "exec_plan engines resolve relative to the model directory, and an absolute path is used as given");
    }

    // A plan on a family that states no residency policy is priced by the sum of its engines, and
    // the engines of a ladder are not multiplied by the runtime factor.
    {
        const auto dir = root / "plan-sum";
        std::filesystem::create_directories(dir);
        write_file(dir / "model.gguf", 7000);
        std::string declared;
        for (const std::string rung : {"flow_seq250.rknn", "flow_seq400.rknn"}) {
            write_file(dir / rung, 1);  // resize_file needs the file to exist; the rest stays sparse
            std::filesystem::resize_file(dir / rung, 3ull * 1024 * 1024 * 1024);
            if (!declared.empty()) {
                declared += ',';
            }
            declared += rung;
        }
        ServerModelConfig model;
        model.family = "piper_tts";
        model.path = dir;
        model.session_options["piper_tts.exec_plan"] =
            R"({"speech_decoder":{"impl":"rknn","artifacts":{"main":")" + declared + R"("}}})";
        const auto estimate = estimate_model_memory_bytes(model);
        const auto priced =
            family_compiled("piper_tts") ? 6ull * 1024 * 1024 * 1024 : size_t{0};
        require(
            estimate.has_value() && *estimate == expected_estimate(7000) + priced,
            "a family without a residency policy is priced by the sum of its engines, at 1.0");
    }

    // The ceiling a family states for its engines caps the priced set: a ladder that evicts to stay
    // inside an NPU aperture is not resident at the sum of its parts. The ceiling CosyVoice3 states is
    // the aperture budget, so the knob that sets it (AUDIOCPP_RKNN_APERTURE_MIB, see rknn_flow.h) is
    // how this gate states a ceiling at all -- the family reports no cap of its own in a build without
    // an RKNN runtime.
    {
        const auto dir = root / "plan-capped";
        std::filesystem::create_directories(dir);
        write_file(dir / "model.gguf", 7000);
        std::string declared;
        for (const std::string rung : {"flow_seq110.rknn", "flow_seq200.rknn", "flow_seq250.rknn"}) {
            write_file(dir / rung, 1);
            std::filesystem::resize_file(dir / rung, 1024ull * 1024 * 1024);
            if (!declared.empty()) {
                declared += ',';
            }
            declared += rung;
        }
        ServerModelConfig model;
        model.family = "cosyvoice3";
        model.path = dir;
        model.session_options["cosyvoice3.exec_plan"] =
            R"({"flow":{"impl":"rknn","artifacts":{"main":")" + declared + R"("}}})";
        const auto uncapped = estimate_model_memory_bytes(model);
        set_env("AUDIOCPP_RKNN_APERTURE_MIB", "1024");
        const auto capped = estimate_model_memory_bytes(model);
        set_env("AUDIOCPP_RKNN_APERTURE_MIB", nullptr);
        const auto sum = family_compiled("cosyvoice3") ? 3ull * 1024 * 1024 * 1024 : size_t{0};
        const auto ceiling = family_compiled("cosyvoice3") ? 1024ull * 1024 * 1024 : size_t{0};
        require(
            uncapped.has_value() && *uncapped == expected_estimate(7000) + sum,
            "a ladder under its ceiling is priced by the sum of its engines");
        require(
            capped.has_value() && *capped == expected_estimate(7000) + ceiling,
            "a family's residency ceiling caps the engines it prices");

        // The eviction rule the ceiling describes admits one engine wider than the ceiling by
        // evicting the rest, so the cap is a sharing budget, not a hard ceiling: the priced set
        // floors at the largest single engine. A 3 GiB rung next to a 1 GiB one: with the
        // ladder's own default budget the 4 GiB sum caps to that budget, and with a 1 GiB
        // budget it caps to 1 GiB -- in both cases the floor lifts the estimate back to the
        // 3 GiB rung that the eviction rule can hold alone.
        const auto wide_dir = root / "plan-capped-wide";
        std::filesystem::create_directories(wide_dir);
        write_file(wide_dir / "model.gguf", 7000);
        const auto small_rung = wide_dir / "flow_seq110.rknn";
        const auto big_rung = wide_dir / "flow_seq1000.rknn";
        write_file(small_rung, 1);
        std::filesystem::resize_file(small_rung, 1024ull * 1024 * 1024);
        write_file(big_rung, 1);
        std::filesystem::resize_file(big_rung, 3ull * 1024 * 1024 * 1024);
        ServerModelConfig wide;
        wide.family = "cosyvoice3";
        wide.path = wide_dir;
        wide.session_options["cosyvoice3.exec_plan"] =
            R"({"flow":{"impl":"rknn","artifacts":{"main":"flow_seq110.rknn,flow_seq1000.rknn"}}})";
        const auto wide_default = estimate_model_memory_bytes(wide);
        set_env("AUDIOCPP_RKNN_APERTURE_MIB", "1024");
        const auto wide_capped = estimate_model_memory_bytes(wide);
        set_env("AUDIOCPP_RKNN_APERTURE_MIB", nullptr);
        const auto default_budget = family_compiled("cosyvoice3")
            ? 3900ull * 1024 * 1024
            : size_t{0};
        const auto wide_floor = family_compiled("cosyvoice3") ? 3ull * 1024 * 1024 * 1024 : size_t{0};
        require(
            wide_default.has_value() && *wide_default == expected_estimate(7000) + default_budget,
            "the sum of the set meets the ladder's own default budget when it fits above the floor");
        require(
            wide_capped.has_value() && *wide_capped == expected_estimate(7000) + wide_floor,
            "a ceiling floors at the largest single engine the eviction rule admits alone");
    }

    // An engine inside a summed checkpoint tree is priced by the plan, once: the plan is applied
    // before the scans, so the engine is counted at the engine price (the driver's residency, here
    // the file size) and the tree scan skips the same path -- it is not repriced at the runtime
    // factor the weights carry.
    {
        const auto dir = root / "plan-tree";
        std::filesystem::create_directories(dir);
        write_file(dir / "model.safetensors", 5000);
        write_file(dir / "vocoder.rknn", 600);
        ServerModelConfig model;
        model.family = "piper_tts";
        model.path = dir;
        model.session_options["piper_tts.exec_plan"] =
            R"({"decoder":{"impl":"rknn","artifacts":{"main":"vocoder.rknn","first":"absent.rknn"}}})";
        const auto estimate = estimate_model_memory_bytes(model);
        const auto priced = family_compiled("piper_tts")
            ? expected_estimate(5000) + 600
            : expected_estimate(5600);
        require(
            estimate.has_value() && *estimate == priced,
            "an engine declared by the plan is priced once, as an engine, and the tree scan skips it");
    }

    // A custom (non-spec-backed) loader answers the same hook: Qwen3-TTS's loader reads its own
    // plan, so its declared speech decoder reaches the guard the same way.
    {
        const auto dir = root / "plan-qwen3";
        std::filesystem::create_directories(dir);
        write_file(dir / "model.gguf", 7000);
        const auto vocoder = dir / "vocoder.rknn";
        write_file(vocoder, 1);
        std::filesystem::resize_file(vocoder, 1024ull * 1024 * 1024);
        ServerModelConfig model;
        model.family = "qwen3_tts";
        model.path = dir;
        model.session_options["qwen3_tts.exec_plan"] =
            R"({"speech_decoder":{"impl":"rknn","artifacts":{"main":"vocoder.rknn"}}})";
        const auto estimate = estimate_model_memory_bytes(model);
        const auto priced = family_compiled("qwen3_tts") ? 1024ull * 1024 * 1024 : size_t{0};
        require(
            estimate.has_value() && *estimate == expected_estimate(7000) + priced,
            "a custom loader reports its plan's engines through the same hook");
    }

    // A malformed plan is the session's error to report, not the guard's.
    {
        const auto path = root / "malformed.gguf";
        write_file(path, 1000);
        ServerModelConfig model;
        model.family = "piper_tts";
        model.path = path;
        model.session_options["piper_tts.exec_plan"] = "{not json";
        const auto estimate = estimate_model_memory_bytes(model);
        require(
            estimate.has_value() && *estimate == expected_estimate(1000),
            "a malformed exec plan contributes nothing to the estimate");
    }
}

void test_model_slots() {
    const auto root = make_temp_root();
    for (const std::string value : {"1", "2", "16", "0", "-1", "17", "null", "true", "\"2\"", "1.5", "4294967297"}) {
        const auto text = std::string(R"({"models":[{"id":"higgs","family":"higgs_audio_tts","path":"model.gguf","slots":)")
            + value + "}]}";
        const auto path = write_config(root, "slots.json", text);
        bool legacy_rejected = false;
        try { (void)minitts::server::load_server_config(path); }
        catch (const std::runtime_error & error) {
            legacy_rejected = std::string(error.what()).find("slots requires --parallel-jobs") != std::string::npos;
        }
        require(legacy_rejected, "legacy configuration accepted an explicit slots field");
        bool rejected = false;
        try {
            const auto config = minitts::server::load_server_config(path, true);
            require(config.models.front().slots == std::stoi(value), "slots value was not preserved");
        } catch (const std::runtime_error &) { rejected = true; }
        require(rejected == (value != "1" && value != "2" && value != "16"), "parallel capacity validation differs");
    }
    const auto path = write_config(root, "default-slots.json",
        R"({"models":[{"id":"higgs","family":"higgs_audio_tts","path":"model.gguf"}]})");
    for (bool enabled : {false, true}) {
        const auto config = minitts::server::load_server_config(path, enabled);
        require(config.models.front().slots == 1, "default capacity must remain one");
    }
}

void test_per_model_devices() {
    const auto root = make_temp_root();
    const auto path = write_config(root, "devices.json", R"JSON({
      "device": 2,
      "models": [
        {"id":"inherited","family":"kokoro_tts","path":"a.gguf"},
        {"id":"overridden","family":"kokoro_tts","path":"b.gguf","device":1}
      ]
    })JSON");
    const auto config = minitts::server::load_server_config(path);
    require(config.device == 2, "server device is preserved");
    require(!config.models[0].device.has_value(), "model without device inherits at runtime");
    require(config.models[1].device == 1, "per-model device is parsed");

    const auto bad = write_config(root, "negative-device.json", R"JSON({
      "models":[{"id":"bad","family":"kokoro_tts","path":"m.gguf","device":-1}]
    })JSON");
    bool rejected = false;
    try { (void)minitts::server::load_server_config(bad); }
    catch (const std::runtime_error & error) {
        rejected = std::string(error.what()).find("device for model bad") != std::string::npos;
    }
    require(rejected, "negative per-model device is rejected with the model id");
}

}  // namespace

int main() {
    try {
        test_per_model_devices();
        test_model_slots();
        test_inline_default_and_named_presets();
        test_model_spec_override_selection();
        test_builtin_utility_model_paths();
        test_default_preset_name();
        test_bad_preset_options_are_rejected();
        test_default_request_options();
        test_default_session_options();
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
        test_configured_model_fields_are_inherited();
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
