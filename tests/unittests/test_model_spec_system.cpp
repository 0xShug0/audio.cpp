#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/model_spec/schema.h"
#include "engine/framework/io/json.h"
#include "test_assert.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

namespace json = engine::io::json;

class ScopedCurrentPath {
public:
    explicit ScopedCurrentPath(const std::filesystem::path & path)
        : previous_(std::filesystem::current_path()) {
        std::filesystem::current_path(path);
    }

    ~ScopedCurrentPath() {
        std::filesystem::current_path(previous_);
    }

    ScopedCurrentPath(const ScopedCurrentPath &) = delete;
    ScopedCurrentPath & operator=(const ScopedCurrentPath &) = delete;

private:
    std::filesystem::path previous_;
};

std::filesystem::path make_temp_root() {
    const auto root = std::filesystem::temp_directory_path() / "audiocpp_model_spec_system_test";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root);
    return root;
}

std::filesystem::path write_text(
    const std::filesystem::path & root,
    const std::filesystem::path & relative_path,
    const std::string & text) {
    const auto path = root / relative_path;
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to create test file: " + path.string());
    }
    out << text;
    return path;
}

std::string legacy_spec_text(const std::string & dependencies) {
    return R"JSON({
  "family": "toy_model",
  "sources": [
    {
      "format": "safetensors",
      "roots": {
        "model": "."
      },
      "files": {
        "config": "model:config.json"
      },
      "optional_files": {
        "tokenizer": "model:tokenizer.json"
      },
      "tensors": {
        "weights": "model:model.safetensors"
      }
    }
  ],
  "dependencies": )JSON" + dependencies + R"JSON(
})JSON";
}

std::string schema_v1_spec_text(const std::string & dependencies) {
    return R"JSON({
  "schema_version": 1,
  "family": "toy_model",
  "display_name": "Toy Model",
  "description": "Toy model schema v1 fixture.",
  "category": "tts",
  "status": "experimental",
  "tasks": ["tts"],
  "modes": ["offline"],
  "languages": ["en"],
  "runtime": {"tags": ["gguf"]},
  "capabilities": {"tts": ["long_form"]},
  "options": {
    "request": [
      {
        "name": "return_timestamps",
        "type": "bool",
        "required": false,
        "default": false,
        "description": "Return timestamps when supported."
      },
      {
        "name": "audio_chunk_mode",
        "type": "enum",
        "values": ["auto", "fixed", "vad", "none"],
        "required": false,
        "default": "auto",
        "description": "Audio chunking mode."
      },
      {
        "name": "vad_path",
        "type": "path",
        "required": false,
        "description": "Optional request-scoped VAD path."
      }
    ],
    "session": [
      {
        "name": "peer_model_path",
        "type": "path",
        "required": false,
        "description": "Peer model path."
      },
      {
        "name": "vad_path",
        "type": "path",
        "required": false,
        "description": "Session-scoped VAD path."
      }
    ],
    "load": []
  },
  "packages": [
    {
      "id": "toy_model_q8",
      "display_name": "Toy Model Q8",
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "Toy-Model-GGUF",
      "download": {"kind": "unsupported", "reason": "test fixture"},
      "files": ["toy-model-q8_0.gguf"],
      "default": true
    }
  ],
  "dependencies": )JSON" + dependencies + R"JSON(,
  "ui": {
    "recommended_package": "toy_model_q8",
    "tags": ["TTS", "GGUF"],
    "docs": ["docs/tts.md"]
  },
  "sources": [
    {
      "format": "gguf",
      "roots": {
        "model": ".",
        "weights": "$gguf"
      },
      "files": {
        "config": "model:config.json"
      },
      "tensors": {
        "weights": "weights:"
      }
    }
  ]
})JSON";
}

std::string schema_v1_spec_with_options(const std::string & options) {
    return R"JSON({
  "schema_version": 1,
  "family": "toy_model",
  "display_name": "Toy Model",
  "description": "Toy model schema v1 fixture.",
  "category": "tts",
  "status": "experimental",
  "tasks": ["tts"],
  "modes": ["offline"],
  "languages": ["en"],
  "runtime": {"tags": ["gguf"]},
  "capabilities": {"tts": ["long_form"]},
  "options": )JSON" + options + R"JSON(,
  "packages": [
    {
      "id": "toy_model_q8",
      "display_name": "Toy Model Q8",
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "Toy-Model-GGUF",
      "download": {"kind": "unsupported", "reason": "test fixture"},
      "files": ["toy-model-q8_0.gguf"],
      "default": true
    }
  ],
  "dependencies": [],
  "ui": {
    "recommended_package": "toy_model_q8",
    "tags": ["TTS", "GGUF"],
    "docs": ["docs/tts.md"]
  },
  "sources": [
    {
      "format": "gguf",
      "roots": {
        "model": ".",
        "weights": "$gguf"
      },
      "files": {
        "config": "model:config.json"
      },
      "tensors": {
        "weights": "weights:"
      }
    }
  ]
})JSON";
}

void expect_rejects(const std::string & label, const std::string & spec_text, const std::string & needle) {
    bool rejected = false;
    try {
        engine::model_spec::validate_spec(json::parse(spec_text), label);
    } catch (const std::runtime_error & error) {
        rejected = std::string(error.what()).find(needle) != std::string::npos;
    }
    engine::test::require(rejected, label + " should reject with: " + needle);
}

void test_legacy_dependencies_schema() {
    // Valid legacy specs accept required model dependencies and conditional bundled dependencies.
    const auto spec = json::parse(schema_v1_spec_text(R"JSON([
    {
      "kind": "model",
      "family": "peer_model",
      "scope": "session",
      "option": "peer_model_path",
      "required": true
    },
    {
      "kind": "bundled_model",
      "family": "silero_vad",
      "path": "assets/framework/models/silero_vad",
      "scope": "request",
      "option": "vad_path",
      "required": false,
      "required_when": [
        {
          "scope": "request",
          "option_key": "audio_chunk_mode",
          "equals": "vad"
        }
      ]
    }
  ])JSON"));
    engine::model_spec::validate_spec(spec, "valid_legacy_dependencies");

    // Dependency rows must use the current local "option" field, not the old option_key field.
    expect_rejects(
        "old_option_key",
        schema_v1_spec_text(R"JSON([
          {
            "kind": "model",
            "family": "peer_model",
            "scope": "session",
            "option_key": "toy_model.peer_model_path",
            "required": true
          }
        ])JSON"),
        "missing required field 'option'");
    // Dependency scopes are typed and reject unknown values.
    expect_rejects(
        "invalid_scope",
        schema_v1_spec_text(R"JSON([
          {
            "kind": "model",
            "family": "peer_model",
            "scope": "prepare",
            "option": "peer_model_path",
            "required": true
          }
        ])JSON"),
        "unknown dependency scope 'prepare'");
    // Dependency options are local names; the family namespace is derived by the framework.
    expect_rejects(
        "namespaced_local_option",
        schema_v1_spec_text(R"JSON([
          {
            "kind": "model",
            "family": "peer_model",
            "scope": "session",
            "option": "toy_model.peer_model_path",
            "required": true
          }
        ])JSON"),
        "dependency option must be local");
    // Free-form required_for was replaced by typed required_when conditions.
    expect_rejects(
        "old_required_for",
        schema_v1_spec_text(R"JSON([
          {
            "kind": "model",
            "family": "peer_model",
            "scope": "session",
            "option": "peer_model_path",
            "required": false,
            "required_for": ["decode"]
          }
        ])JSON"),
        "use typed required_when conditions");
    // Optional dependencies must say exactly when they become required.
    expect_rejects(
        "optional_missing_condition",
        schema_v1_spec_text(R"JSON([
          {
            "kind": "model",
            "family": "peer_model",
            "scope": "session",
            "option": "peer_model_path",
            "required": false
          }
        ])JSON"),
        "optional dependencies require typed conditions");
    // Required dependencies are unconditional and must not also have required_when rows.
    expect_rejects(
        "required_with_condition",
        schema_v1_spec_text(R"JSON([
          {
            "kind": "model",
            "family": "peer_model",
            "scope": "session",
            "option": "peer_model_path",
            "required": true,
            "required_when": [
              {
                "scope": "request",
                "option_key": "return_timestamps",
                "equals": true
              }
            ]
          }
        ])JSON"),
        "required dependencies must not have conditions");
    // Dependency package hints are intentionally outside the core runtime contract.
    expect_rejects(
        "dependency_package_hint",
        schema_v1_spec_text(R"JSON([
          {
            "kind": "model",
            "family": "peer_model",
            "scope": "session",
            "option": "peer_model_path",
            "required": false,
            "required_when": [
              {
                "scope": "request",
                "option_key": "return_timestamps",
                "equals": true
              }
            ],
            "package": "peer_model_q8_0"
          }
        ])JSON"),
        "dependency package hints are not part of the core contract");
    // Bundled dependencies must provide the bundled asset path.
    expect_rejects(
        "bundled_missing_path",
        schema_v1_spec_text(R"JSON([
          {
            "kind": "bundled_model",
            "family": "silero_vad",
            "scope": "session",
            "option": "vad_path",
            "required": false,
            "required_when": [
              {
                "scope": "request",
                "option_key": "audio_chunk_mode",
                "equals": "vad"
              }
            ]
          }
        ])JSON"),
        "missing required field 'path'");
    // Dependency options must be declared under options.<scope>.
    expect_rejects(
        "undeclared_dependency_option",
        schema_v1_spec_text(R"JSON([
          {
            "kind": "model",
            "family": "peer_model",
            "scope": "session",
            "option": "missing_peer_path",
            "required": true
          }
        ])JSON"),
        "is not declared in options.session");
    // required_when option keys must refer to declared options in that scope.
    expect_rejects(
        "undeclared_required_when_option",
        schema_v1_spec_text(R"JSON([
          {
            "kind": "model",
            "family": "peer_model",
            "scope": "session",
            "option": "peer_model_path",
            "required": false,
            "required_when": [
              {
                "scope": "request",
                "option_key": "unknown_trigger",
                "equals": true
              }
            ]
          }
        ])JSON"),
        "is not declared in options.request");
    // Session-scoped required_when keys use the public <family>.<name> form.
    expect_rejects(
        "required_when_session_local_name",
        schema_v1_spec_text(R"JSON([
          {
            "kind": "model",
            "family": "peer_model",
            "scope": "session",
            "option": "peer_model_path",
            "required": false,
            "required_when": [
              {
                "scope": "session",
                "option_key": "peer_model_path",
                "equals": "/tmp/peer"
              }
            ]
          }
        ])JSON"),
        "is not declared in options.session");
}

void test_typed_schema_renamed_dependencies() {
    // Valid typed specs accept top-level languages, task-keyed capabilities, and typed dependency conditions.
    const auto typed = json::parse(R"JSON({
  "schema_version": 1,
  "family": "typed_model",
  "display_name": "Typed Model",
  "category": "asr",
  "status": "experimental",
  "tasks": ["asr"],
  "modes": ["offline"],
  "languages": ["en"],
  "runtime": {
    "tags": ["gguf"]
  },
  "capabilities": {
    "asr": ["word_timestamps"]
  },
  "options": {
    "request": [
      {
        "name": "return_timestamps",
        "type": "bool",
        "required": false,
        "default": false,
        "description": "Return word timestamps."
      },
      {
        "name": "audio_chunk_seconds",
        "type": "float",
        "required": false,
        "default": 15,
        "description": "Audio chunk duration in seconds."
      }
    ],
    "session": [
      {
        "name": "aligner_path",
        "type": "path",
        "required": false,
        "description": "Forced aligner model path."
      }
    ],
    "load": []
  },
  "packages": [
    {
      "id": "typed_model_q8",
      "display_name": "Typed Model Q8",
      "format": "gguf",
      "precision": "q8_0",
      "target_directory": "Typed-Model",
      "download": {
        "kind": "huggingface_snapshot",
        "repo": "audio-cpp/typed-model"
      },
      "files": ["typed-model-q8_0.gguf"],
      "default": true
    }
  ],
  "dependencies": [
    {
      "kind": "model",
      "family": "typed_aligner",
      "scope": "session",
      "option": "aligner_path",
      "required": false,
      "required_when": [
        {
          "scope": "request",
          "option_key": "return_timestamps",
          "equals": true
        },
        {
          "scope": "request",
          "option_key": "audio_chunk_seconds",
          "equals": 15
        }
      ]
    }
  ],
  "ui": {
    "recommended_package": "typed_model_q8",
    "tags": ["ASR"],
    "docs": ["docs/asr.md"]
  },
  "sources": [
    {
      "format": "gguf",
      "roots": {
        "model": ".",
        "weights": "$gguf"
      },
      "files": {
        "config": "model:config.json"
      },
      "tensors": {
        "weights": "weights:"
      }
    }
  ]
})JSON");
    engine::model_spec::validate_spec(typed, "typed_dependencies");

    // The old "companions" block is rejected in favor of typed dependencies.
    expect_rejects(
        "typed_rejects_companions",
        R"JSON({
          "schema_version": 1,
          "family": "typed_model",
          "display_name": "Typed Model",
          "category": "asr",
          "status": "experimental",
          "tasks": ["asr"],
          "modes": ["offline"],
          "languages": ["en"],
          "runtime": {"tags": ["gguf"]},
          "capabilities": {
            "asr": ["word_timestamps"]
          },
          "options": {"request": [], "session": [], "load": []},
          "packages": [
            {
              "id": "typed_model_q8",
              "display_name": "Typed Model Q8",
              "format": "gguf",
              "precision": "q8_0",
              "target_directory": "Typed-Model",
              "download": {"kind": "huggingface_snapshot", "repo": "audio-cpp/typed-model"},
              "files": ["typed-model-q8_0.gguf"],
              "default": true
            }
          ],
          "companions": [],
          "ui": {
            "recommended_package": "typed_model_q8",
            "tags": ["ASR"],
            "docs": ["docs/asr.md"]
          },
          "sources": [
            {
              "format": "gguf",
              "roots": {"model": ".", "weights": "$gguf"},
              "files": {"config": "model:config.json"},
              "tensors": {"weights": "weights:"}
            }
          ]
        })JSON",
        "missing required field 'dependencies'");

    // Task names are typed and reject unknown operation ids.
    expect_rejects(
        "typed_rejects_unknown_task",
        R"JSON({
          "schema_version": 1,
          "family": "typed_model",
          "display_name": "Typed Model",
          "category": "tts",
          "status": "experimental",
          "tasks": ["bad_task"],
          "modes": ["offline"],
          "languages": ["en"],
          "runtime": {"tags": ["gguf"]},
          "capabilities": {},
          "options": {"request": [], "session": [], "load": []},
          "packages": [
            {
              "id": "typed_model_q8",
              "display_name": "Typed Model Q8",
              "format": "gguf",
              "precision": "q8_0",
              "target_directory": "Typed-Model",
              "download": {"kind": "huggingface_snapshot", "repo": "audio-cpp/typed-model"},
              "files": ["typed-model-q8_0.gguf"],
              "default": true
            }
          ],
          "dependencies": [],
          "ui": {
            "recommended_package": "typed_model_q8",
            "tags": ["TTS"],
            "docs": ["docs/tts.md"]
          },
          "sources": [
            {
              "format": "gguf",
              "roots": {"model": ".", "weights": "$gguf"},
              "files": {"config": "model:config.json"},
              "tensors": {"weights": "weights:"}
            }
          ]
        })JSON",
        "unknown task 'bad_task'");

    // Capability keys must be one of the model's declared tasks.
    expect_rejects(
        "typed_rejects_capability_for_undeclared_task",
        R"JSON({
          "schema_version": 1,
          "family": "typed_model",
          "display_name": "Typed Model",
          "category": "tts",
          "status": "experimental",
          "tasks": ["tts"],
          "modes": ["offline"],
          "languages": ["en"],
          "runtime": {"tags": ["gguf"]},
          "capabilities": {"clone": ["speaker_reference"]},
          "options": {"request": [], "session": [], "load": []},
          "packages": [
            {
              "id": "typed_model_q8",
              "display_name": "Typed Model Q8",
              "format": "gguf",
              "precision": "q8_0",
              "target_directory": "Typed-Model",
              "download": {"kind": "huggingface_snapshot", "repo": "audio-cpp/typed-model"},
              "files": ["typed-model-q8_0.gguf"],
              "default": true
            }
          ],
          "dependencies": [],
          "ui": {
            "recommended_package": "typed_model_q8",
            "tags": ["TTS"],
            "docs": ["docs/tts.md"]
          },
          "sources": [
            {
              "format": "gguf",
              "roots": {"model": ".", "weights": "$gguf"},
              "files": {"config": "model:config.json"},
              "tensors": {"weights": "weights:"}
            }
          ]
        })JSON",
        "capability key must be one of this model's tasks");

    // Capability values are typed per task, so ASR timestamp capabilities cannot be attached to TTS.
    expect_rejects(
        "typed_rejects_unknown_capability",
        R"JSON({
          "schema_version": 1,
          "family": "typed_model",
          "display_name": "Typed Model",
          "category": "tts",
          "status": "experimental",
          "tasks": ["tts"],
          "modes": ["offline"],
          "languages": ["en"],
          "runtime": {"tags": ["gguf"]},
          "capabilities": {"tts": ["word_timestamps"]},
          "options": {"request": [], "session": [], "load": []},
          "packages": [
            {
              "id": "typed_model_q8",
              "display_name": "Typed Model Q8",
              "format": "gguf",
              "precision": "q8_0",
              "target_directory": "Typed-Model",
              "download": {"kind": "huggingface_snapshot", "repo": "audio-cpp/typed-model"},
              "files": ["typed-model-q8_0.gguf"],
              "default": true
            }
          ],
          "dependencies": [],
          "ui": {
            "recommended_package": "typed_model_q8",
            "tags": ["TTS"],
            "docs": ["docs/tts.md"]
          },
          "sources": [
            {
              "format": "gguf",
              "roots": {"model": ".", "weights": "$gguf"},
              "files": {"config": "model:config.json"},
              "tensors": {"weights": "weights:"}
            }
          ]
        })JSON",
        "unknown capability 'word_timestamps'");
}

void test_dependency_option_mapping_from_production_spec() {
    // The production catalog currently points at the legacy source-only specs.
    const auto miotts_dependencies = engine::model_spec::dependencies("miotts");
    engine::test::require(miotts_dependencies.empty(), "legacy miotts spec should not expose dependencies");

    const auto qwen_dependencies = engine::model_spec::dependencies("qwen3_asr");
    engine::test::require(qwen_dependencies.empty(), "legacy qwen3 ASR spec should not expose dependencies");
}

void test_options_schema() {
    // Valid option metadata accepts typed scopes, enum presets, structured list types, and load paths.
    const auto spec = json::parse(schema_v1_spec_with_options(R"JSON({
    "request": [
      {
        "name": "text_chunk_mode",
        "type": "enum",
        "preset": "text_chunk_mode_full",
        "required": false,
        "default": "word_budget",
        "description": "Text chunking mode."
      },
      {
        "name": "duration_seconds",
        "type": "float_list",
        "required": false,
        "min": 0,
        "description": "Per-prompt duration list."
      }
    ],
    "session": [
      {
        "name": "weight_type",
        "type": "enum",
        "preset": "weight_type_full",
        "required": false,
        "default": "native",
        "description": "Weight storage type."
      }
    ],
    "load": [
      {
        "name": "sidecar_path",
        "type": "path",
        "required": false,
        "description": "Optional sidecar path."
      }
    ]
  })JSON"));
    engine::model_spec::validate_spec(spec, "valid_options");

    // Option scopes are required so consumers do not guess missing categories.
    expect_rejects(
        "options_missing_scope",
        schema_v1_spec_with_options(R"JSON({
          "request": [],
          "session": []
        })JSON"),
        "missing required field 'load'");

    // Option type names are typed and reject unknown values.
    expect_rejects(
        "options_reject_unknown_type",
        schema_v1_spec_with_options(R"JSON({
          "request": [
            {
              "name": "seed",
              "type": "number",
              "required": false,
              "description": "Seed."
            }
          ],
          "session": [],
          "load": []
        })JSON"),
        "unknown option type 'number'");

    // Requiredness is explicit for every option; absence would make consumers guess.
    expect_rejects(
        "options_reject_missing_required",
        schema_v1_spec_with_options(R"JSON({
          "request": [
            {
              "name": "seed",
              "type": "int",
              "description": "Seed."
            }
          ],
          "session": [],
          "load": []
        })JSON"),
        "missing required field 'required'");

    // Enum options must list their accepted values directly or through a preset.
    expect_rejects(
        "options_reject_enum_without_values",
        schema_v1_spec_with_options(R"JSON({
          "request": [
            {
              "name": "text_chunk_mode",
              "type": "enum",
              "required": false,
              "description": "Text chunking mode."
            }
          ],
          "session": [],
          "load": []
        })JSON"),
        "enum option requires values or preset");

    // Defaults are typed; an int option cannot use a string default.
    expect_rejects(
        "options_reject_wrong_default_type",
        schema_v1_spec_with_options(R"JSON({
          "request": [
            {
              "name": "seed",
              "type": "int",
              "required": false,
              "default": "0",
              "description": "Seed."
            }
          ],
          "session": [],
          "load": []
        })JSON"),
        "expected numeric default");

    // Enum defaults must be part of the declared enum domain.
    expect_rejects(
        "options_reject_enum_default_outside_values",
        schema_v1_spec_with_options(R"JSON({
          "request": [
            {
              "name": "route",
              "type": "enum",
              "values": ["a", "b"],
              "required": false,
              "default": "c",
              "description": "Route."
            }
          ],
          "session": [],
          "load": []
        })JSON"),
        "default is not in enum values");

    // Numeric bounds are typed and ordered.
    expect_rejects(
        "options_reject_bad_numeric_bounds",
        schema_v1_spec_with_options(R"JSON({
          "request": [
            {
              "name": "top_k",
              "type": "int",
              "required": false,
              "min": 10,
              "max": 1,
              "description": "Top-k."
            }
          ],
          "session": [],
          "load": []
        })JSON"),
        "min must not exceed max");

    // Numeric defaults must stay inside declared bounds.
    expect_rejects(
        "options_reject_default_outside_bounds",
        schema_v1_spec_with_options(R"JSON({
          "request": [
            {
              "name": "top_p",
              "type": "float",
              "required": false,
              "default": 2.0,
              "min": 0.0,
              "max": 1.0,
              "description": "Top-p."
            }
          ],
          "session": [],
          "load": []
        })JSON"),
        "default exceeds max");

    // min/max are only meaningful for numeric options.
    expect_rejects(
        "options_reject_bounds_on_string",
        schema_v1_spec_with_options(R"JSON({
          "request": [
            {
              "name": "reference_text",
              "type": "string",
              "required": false,
              "min": 1,
              "description": "Reference text."
            }
          ],
          "session": [],
          "load": []
        })JSON"),
        "min/max are allowed only for int or float options");

    // Non-enum options cannot carry enum values.
    expect_rejects(
        "options_reject_values_on_non_enum",
        schema_v1_spec_with_options(R"JSON({
          "request": [
            {
              "name": "seed",
              "type": "int",
              "values": ["1", "2"],
              "required": false,
              "description": "Seed."
            }
          ],
          "session": [],
          "load": []
        })JSON"),
        "values are allowed only for enum options");

    // Enum presets replace repeated value arrays for shared option domains.
    expect_rejects(
        "options_reject_unknown_preset",
        schema_v1_spec_with_options(R"JSON({
          "request": [
            {
              "name": "text_chunk_mode",
              "type": "enum",
              "preset": "unknown_chunk_modes",
              "required": false,
              "description": "Text chunking mode."
            }
          ],
          "session": [],
          "load": []
        })JSON"),
        "unknown option preset 'unknown_chunk_modes'");

    // Enum options must use either inline values or a preset, not both.
    expect_rejects(
        "options_reject_preset_with_values",
        schema_v1_spec_with_options(R"JSON({
          "request": [
            {
              "name": "text_chunk_mode",
              "type": "enum",
              "preset": "text_chunk_mode_full",
              "values": ["word_budget"],
              "required": false,
              "description": "Text chunking mode."
            }
          ],
          "session": [],
          "load": []
        })JSON"),
        "enum options must use either preset or values");

    // Shared option names carry shared type contracts; top_k is not free-form text.
    expect_rejects(
        "options_reject_wrong_common_type",
        schema_v1_spec_with_options(R"JSON({
          "request": [
            {
              "name": "top_k",
              "type": "string",
              "required": false,
              "description": "Top-k sampling limit."
            }
          ],
          "session": [],
          "load": []
        })JSON"),
        "shared option 'top_k' must use its registered type");

    // Route choices are finite and must be machine-readable, not buried in free-form descriptions.
    expect_rejects(
        "options_reject_route_as_string",
        schema_v1_spec_with_options(R"JSON({
          "request": [
            {
              "name": "route",
              "type": "string",
              "required": false,
              "description": "Route name."
            }
          ],
          "session": [],
          "load": []
        })JSON"),
        "shared option 'route' must use its registered type");

    // Model-specific request options are owned by the model spec and do not need central registration.
    const auto local_request_option = json::parse(schema_v1_spec_with_options(R"JSON({
          "request": [
            {
              "name": "private_knob",
              "type": "bool",
              "required": true,
              "description": "Model-local switch."
            }
          ],
          "session": [],
          "load": []
        })JSON"));
    engine::model_spec::validate_spec(local_request_option, "local_request_option");

    // Dotted request options are public names and must stay in the model family namespace.
    expect_rejects(
        "options_reject_wrong_request_namespace",
        schema_v1_spec_with_options(R"JSON({
          "request": [
            {
              "name": "other_model.private_knob",
              "type": "bool",
              "required": false,
              "description": "Model-local switch."
            }
          ],
          "session": [],
          "load": []
        })JSON"),
        "dotted request option 'other_model.private_knob' must use namespace toy_model.<name>");

    // Session options are local names in the spec; the family prefix is derived by metadata readers.
    expect_rejects(
        "options_reject_prefixed_session_option",
        schema_v1_spec_with_options(R"JSON({
          "request": [],
          "session": [
            {
              "name": "toy_model.weight_type",
              "type": "enum",
              "values": ["native", "q8_0"],
              "required": false,
              "description": "Weight storage type."
            }
          ],
          "load": []
        })JSON"),
        "session option must be local");

    // Load options follow the same local-name rule as session options.
    expect_rejects(
        "options_reject_prefixed_load_option",
        schema_v1_spec_with_options(R"JSON({
          "request": [],
          "session": [],
          "load": [
            {
              "name": "toy_model.sidecar_path",
              "type": "path",
              "required": false,
              "description": "Optional sidecar path."
            }
          ]
        })JSON"),
        "load option must be local");
}

void test_option_name_mapping_from_production_spec() {
    // The production catalog currently points at the legacy source-only specs.
    const auto omnivoice_cli = engine::model_spec::cli_interface("omnivoice");
    engine::test::require(!omnivoice_cli.has_value(), "legacy omnivoice spec should not expose CLI metadata");

    const auto vibevoice_cli = engine::model_spec::cli_interface("vibevoice");
    engine::test::require(!vibevoice_cli.has_value(), "legacy vibevoice spec should not expose CLI metadata");
}

void test_schema_v1_metadata_projection() {
    const auto root = make_temp_root();
    write_text(root, "toy_model.json", schema_v1_spec_with_options(R"JSON({
      "request": [
        {
          "name": "language",
          "type": "string",
          "required": false,
          "default": "en",
          "description": "Target language."
        }
      ],
      "session": [
        {
          "name": "weight_type",
          "type": "enum",
          "preset": "weight_type_full",
          "required": false,
          "default": "native",
          "description": "Updated weight storage type."
        },
        {
          "name": "future_session_option",
          "type": "bool",
          "required": false,
          "default": false,
          "description": "New code-supported session option."
        }
      ],
      "load": []
    })JSON"));

    const engine::model_spec::ScopedSpecOverride spec_override(root);
    const auto metadata = engine::model_spec::model_metadata("toy_model");
    engine::test::require(metadata.has_value(), "schema v1 metadata should be projected");
    engine::test::require_eq(metadata->family, std::string("toy_model"), "metadata family");
    engine::test::require_eq(metadata->variant, std::string("Toy Model"), "metadata display name");
    engine::test::require_eq(
        metadata->description,
        std::string("Toy model schema v1 fixture."),
        "metadata description");
    engine::test::require(!metadata->config_candidates.empty(), "metadata config candidates");
    engine::test::require(!metadata->weight_candidates.empty(), "metadata weight candidates");

    const auto capabilities = engine::model_spec::advertised_capabilities("toy_model");
    engine::test::require(capabilities.has_value(), "schema v1 capabilities should be projected");
    engine::test::require_eq(capabilities->languages.size(), size_t{1}, "capability language count");
    engine::test::require_eq(capabilities->languages.front(), std::string("en"), "capability language");
    engine::test::require_eq(capabilities->supported_tasks.size(), size_t{1}, "capability task count");
    engine::test::require(
        capabilities->supported_tasks.front().task == engine::runtime::VoiceTaskKind::Tts,
        "capability task should come from v1 tasks");
    engine::test::require(
        capabilities->supported_tasks.front().modes.front() == engine::runtime::RunMode::Offline,
        "capability mode should come from v1 modes");

    const auto cli = engine::model_spec::cli_interface("toy_model");
    engine::test::require(cli.has_value(), "schema v1 CLI should be projected");
    engine::test::require_eq(cli->request_options.front().name, std::string("language"), "request option name");
    engine::test::require_eq(
        cli->session_options.front().name,
        std::string("toy_model.weight_type"),
        "session option public name");
    engine::test::require_eq(
        cli->session_options.front().description,
        std::string("Updated weight storage type."),
        "modified session option description");
    engine::test::require_eq(
        cli->session_options[1].name,
        std::string("toy_model.future_session_option"),
        "new session option public name");

    const auto contract = engine::model_spec::model_contract("toy_model");
    engine::test::require(contract.has_value(), "schema v1 contract should be projected");
    engine::test::require(
        contract->session_option_keys.find("toy_model.future_session_option") !=
            contract->session_option_keys.end(),
        "contract should include new session option");

    std::filesystem::remove_all(root);
}

void test_contract_spec_prefers_workspace_over_package_local_spec() {
    const auto root = make_temp_root();
    const auto workspace = root / "workspace";
    const auto model_root = root / "model";
    std::filesystem::create_directories(workspace);
    std::filesystem::create_directories(model_root);
    const auto workspace_spec = write_text(
        workspace,
        "model_specs/toy_model.json",
        schema_v1_spec_with_options(R"JSON({"request": [], "session": [], "load": []})JSON"));
    const auto package_spec = write_text(
        model_root,
        "model_specs/toy_model.json",
        schema_v1_spec_with_options(R"JSON({"request": [], "session": [], "load": []})JSON"));

    {
        ScopedCurrentPath cwd(workspace);
        const engine::model_spec::ScopedSpecOverride model_scope(std::nullopt, model_root);
        engine::test::require_eq(
            engine::model_spec::default_contract_spec_path("toy_model").string(),
            std::filesystem::weakly_canonical(workspace_spec).string(),
            "contract spec path");
        engine::test::require_eq(
            engine::model_spec::default_package_spec_path("toy_model").string(),
            std::filesystem::weakly_canonical(package_spec).string(),
            "package spec path");
    }

    std::filesystem::remove_all(root);
}

void test_loading_and_resource_bundle() {
    // Resource bundles resolve required files, tensor source files, and missing optional files.
    const auto root = make_temp_root();
    const auto model_root = root / "model";
    std::filesystem::create_directories(model_root);
    write_text(model_root, "config.json", "{\"ok\":true}");
    write_text(model_root, "model.safetensors", "");
    const auto spec_path = write_text(root, "toy_model.json", legacy_spec_text("[]"));

    const auto spec = engine::model_spec::load_spec(spec_path);
    engine::test::require_eq(
        spec.require("family").as_string(),
        std::string("toy_model"),
        "loaded spec family");
    const auto bundle = engine::model_spec::load_resource_bundle(model_root, spec_path);
    engine::test::require_eq(
        bundle.read_text("config"),
        std::string("{\"ok\":true}"),
        "resource bundle config");
    engine::test::require(bundle.has_file("weights"), "tensor source file should be registered");
    engine::test::require(!bundle.has_file("tokenizer"), "missing optional file should not be registered");

    std::filesystem::remove_all(root);
}

}  // namespace

int main() {
    try {
        test_legacy_dependencies_schema();
        test_typed_schema_renamed_dependencies();
        test_dependency_option_mapping_from_production_spec();
        test_options_schema();
        test_option_name_mapping_from_production_spec();
        test_schema_v1_metadata_projection();
        test_contract_spec_prefers_workspace_over_package_local_spec();
        test_loading_and_resource_bundle();
    } catch (const std::exception & error) {
        std::cerr << "model_spec_system_test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "model_spec_system_test passed\n";
    return 0;
}
