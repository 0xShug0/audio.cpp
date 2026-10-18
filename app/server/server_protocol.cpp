#include "server_protocol.h"

#include "speech_option_exceptions.h"
#include "../cli/request.h"
#include "../common/base64.h"

#include "engine/framework/debug/trace.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/model_spec/options.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/runtime/errors.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace minitts::server {
namespace {

using engine::io::json::Value;
using minitts::app::base64_decode;
using minitts::app::base64_encode;
using Clock = ServerClock;

} // namespace

// Per-request override for the busy timeout. Absent means "use the model's
// configured ceiling"; a value is clamped to that ceiling by
// resolve_busy_timeout_ms so a client can shorten its own wait but never weaken
// the guard.
std::optional<int> parse_busy_timeout_override(const Value &body) {
  const auto *value = body.find("busy_timeout_ms");
  if (value == nullptr) {
    return std::nullopt;
  }
  const auto requested =
      engine::io::json::optional_i32(body, "busy_timeout_ms", 0);
  if (requested < 0) {
    throw std::runtime_error(
        "busy_timeout_ms must be >= 0 (0 means no client-side bound)");
  }
  return requested;
}

engine::runtime::TaskRequest apply_default_request_options(
    const std::unordered_map<std::string, std::string> &default_options,
    engine::runtime::TaskRequest request) {
  if (default_options.empty()) {
    return request;
  }
  auto options = default_options;
  for (auto &[key, value] : request.options) {
    options[key] = std::move(value);
  }
  request.options = std::move(options);
  return request;
}

engine::runtime::TaskRequest
build_speech_request(const Value &body, const SpeechRequestContext &context) {
  engine::runtime::TaskRequest request;
  request.text_input = engine::runtime::Transcript{
      engine::io::json::require_string(body, "input"),
      engine::io::json::optional_string(body, "language", ""),
  };

  request.options = options_from_object(body.find("options"));
  add_option_from_json(request.options, body, "seed", "seed");
  add_option_from_json(request.options, body, "temperature", "temperature");
  add_option_from_json(request.options, body, "top_k", "top_k");
  add_option_from_json(request.options, body, "top_p", "top_p");
  add_option_from_json(request.options, body, "max_tokens", "max_tokens");
  add_option_from_json(request.options, body, "max_steps", "max_steps");
  add_option_from_json(request.options, body, "repetition_penalty",
                       "repetition_penalty");
  add_option_from_json(request.options, body, "guidance_scale",
                       "guidance_scale");
  add_option_from_json(request.options, body, "num_inference_steps",
                       "num_inference_steps");
  const auto *instruct = body.find("instruct");
  const auto *instructions = body.find("instructions");
  if (instruct != nullptr && instructions != nullptr) {
    throw engine::runtime::InvalidRequestError(
        "instruct and instructions cannot both be set");
  }
  if (instruct != nullptr) {
    request.options["instruct"] = instruct->as_string();
  } else if (instructions != nullptr) {
    request.options["instruction"] = instructions->as_string();
  }

  bool voice_field_is_preset = false;
  const RuntimeVoicePreset *preset = nullptr;
  if (const auto *value = body.find("voice")) {
    const auto it = context.voice_presets.find(value->as_string());
    if (it != context.voice_presets.end()) {
      voice_field_is_preset = true;
      preset = &it->second;
    }
  } else if (body.find("voice_ref") == nullptr &&
             context.default_voice_preset.has_value()) {
    preset = &*context.default_voice_preset;
  }

  engine::runtime::VoiceCondition voice;
  bool has_voice = false;
  if (preset != nullptr) {
    if (preset->voice_id.has_value()) {
      voice.speaker = engine::runtime::VoiceReference{};
      voice.speaker->cached_voice_id = *preset->voice_id;
      has_voice = true;
    }
    if (preset->audio.has_value()) {
      if (!voice.speaker.has_value()) {
        voice.speaker = engine::runtime::VoiceReference{};
      }
      voice.speaker->audio = *preset->audio;
      has_voice = true;
    }
    if (context.accepts_reference_text &&
        preset->reference_text.has_value() &&
        request.options.find("reference_text") == request.options.end()) {
      request.options["reference_text"] = *preset->reference_text;
    }
    request = apply_default_request_options(preset->options, std::move(request));
  }

  const bool has_explicit_voice_ref = body.find("voice_ref") != nullptr;
  if (const auto *value = body.find("voice");
      value != nullptr && !voice_field_is_preset) {
    bool voice_library_resolved = false;
    if (context.voice_dir.has_value() && !has_explicit_voice_ref) {
      const std::string voice_name = value->as_string();
      if (const auto wav =
              resolve_voice_library_wav(*context.voice_dir, voice_name)) {
        voice.speaker = engine::runtime::VoiceReference{};
        voice.speaker->audio = minitts::cli::read_audio_buffer(*wav);
        has_voice = true;
        voice_library_resolved = true;
        if (context.accepts_reference_text &&
            request.options.find("reference_text") == request.options.end()) {
          auto text = load_voice_library_text(*context.voice_dir, voice_name);
          if (text.has_value()) {
            request.options["reference_text"] = *text;
          }
        }
      }
    }
    if (!voice_library_resolved) {
      if (!voice.speaker.has_value()) {
        voice.speaker = engine::runtime::VoiceReference{};
      }
      voice.speaker->cached_voice_id = value->as_string();
      has_voice = true;
    }
  }

  if (const auto *value = body.find("voice_ref")) {
    if (!voice.speaker.has_value()) {
      voice.speaker = engine::runtime::VoiceReference{};
    }
    if (value->is_string()) {
      voice.speaker->audio = minitts::cli::read_audio_buffer(
          resolve_path(context.request_base, value->as_string()));
    } else if (value->is_object()) {
      const auto &type = engine::io::json::require_string(*value, "type");
      if (type == "path") {
        voice.speaker->audio = minitts::cli::read_audio_buffer(resolve_path(
            context.request_base,
            engine::io::json::require_string(*value, "path")));
      } else if (type == "base64") {
        constexpr size_t kMaxVoiceRefBytes = size_t{5} * 1024 * 1024;
        constexpr size_t kMaxVoiceRefB64Length =
            ((kMaxVoiceRefBytes + 2) / 3) * 4 + 4096;
        const auto &data = engine::io::json::require_string(*value, "data");
        if (data.size() > kMaxVoiceRefB64Length) {
          throw std::runtime_error(
              "voice_ref base64 data exceeds the 5 MiB limit");
        }
        const auto bytes = base64_decode(data);
        if (bytes.empty()) {
          throw std::runtime_error(
              "voice_ref base64 data decoded to an empty payload");
        }
        if (bytes.size() > kMaxVoiceRefBytes) {
          throw std::runtime_error(
              "voice_ref base64 data exceeds the 5 MiB limit");
        }
        voice.speaker->audio = minitts::cli::read_audio_buffer(std::string_view(
            reinterpret_cast<const char *>(bytes.data()), bytes.size()));
      } else {
        throw std::runtime_error(
            "voice_ref type must be \"path\" or \"base64\"");
      }
    } else {
      throw std::runtime_error(
          "voice_ref must be a path string or an object with type \"path\" or "
          "\"base64\"");
    }
    has_voice = true;
  }

  if (const auto *value = body.find("reference_text")) {
    request.options["reference_text"] = value->as_string();
  }
  const auto *speed = body.find("speed");
  if (speed == nullptr) {
    speed = body.find("speaking_rate");
  }
  if (speed != nullptr) {
    const double speed_value = speed->as_number();
    const float rate = static_cast<float>(speed_value);
    if (!std::isfinite(rate) || rate <= 0.0F) {
      throw std::runtime_error("speed must be a positive finite number");
    }
    if (context.accepts_speed || context.accepts_speaking_rate ||
        !is_neutral_speech_option("speed", speed_value)) {
      if (!context.accepts_speed && !context.accepts_speaking_rate) {
        throw std::runtime_error("speed is not supported by this model");
      }
      if (context.accepts_speed) {
        request.options["speed"] = std::to_string(rate);
      } else {
        request.options["speaking_rate"] = std::to_string(rate);
      }
      voice.style = engine::runtime::StyleCondition{};
      voice.style->speaking_rate = rate;
      has_voice = true;
    }
  }
  if (has_voice) {
    request.voice = std::move(voice);
  }
  const bool request_has_instruct =
      request.options.find("instruct") != request.options.end();
  const bool request_has_instruction =
      request.options.find("instruction") != request.options.end();
  auto merged = apply_default_request_options(context.default_options,
                                              std::move(request));
  if (request_has_instruct && !request_has_instruction) {
    merged.options.erase("instruction");
  } else if (request_has_instruction && !request_has_instruct) {
    merged.options.erase("instruct");
  }
  return merged;
}

static bool is_missing_model_contract_error(std::string_view message) {
  return message.find("model contract spec not found for family '") !=
             std::string_view::npos ||
         message.find("does not embed an audio.cpp model spec") !=
             std::string_view::npos ||
         message.find("embeds a legacy model spec") != std::string_view::npos;
}

bool model_accepts_request_option(
    std::string_view family, std::string_view option,
    const std::optional<std::filesystem::path> &model_spec_override,
    const std::filesystem::path &model_path) {
  std::optional<engine::model_spec::ModelContract> contract;
  {
    engine::model_spec::ScopedSpecOverride scoped(model_spec_override,
                                                  model_path);
    try {
      contract = engine::model_spec::model_contract(family);
    } catch (const std::runtime_error &ex) {
      if (!is_missing_model_contract_error(ex.what())) {
        throw;
      }
    }
  }
  if (!contract.has_value()) {
    return true;
  }
  return contract->request_option_keys.find(std::string(option)) !=
         contract->request_option_keys.end();
}

// The `options.request` rows of the model's spec, as a JSON array, for
// `GET /v1/models?include_params=true`. Rows are passed through as the spec
// declares them; an enum row that names a known `preset` also gains the
// preset's `values`, so a client needs no copy of the preset table. A spec need
// not be on schema_version 1: older-format specs may document their options
// too. Those skip the typed validator, so a row that is not an object is left
// out. A spec with no request options gets "[]".
std::string model_request_params_json(
    std::string_view family,
    const std::optional<std::filesystem::path> &model_spec_override,
    const std::filesystem::path &model_path) {
  namespace json = engine::io::json;
  json::Value spec;
  {
    engine::model_spec::ScopedSpecOverride scoped(model_spec_override,
                                                  model_path);
    try {
      const auto spec_path =
          engine::model_spec::find_contract_spec_path(family);
      if (!spec_path.has_value()) {
        return "[]";
      }
      spec = engine::model_spec::load_contract_spec(*spec_path);
    } catch (const std::runtime_error &ex) {
      if (!is_missing_model_contract_error(ex.what())) {
        throw;
      }
      return "[]";
    }
  }
  const auto *options = spec.find("options");
  const auto *request = options != nullptr && options->is_object()
                            ? options->find("request")
                            : nullptr;
  if (request == nullptr || !request->is_array()) {
    return "[]";
  }
  json::Value::Array rows;
  rows.reserve(request->as_array().size());
  const auto &presets = engine::model_spec::option_presets();
  for (const auto &row : request->as_array()) {
    if (!row.is_object()) {
      continue;
    }
    const auto *preset = row.find("preset");
    const auto known = preset != nullptr && preset->is_string()
                           ? presets.find(preset->as_string())
                           : presets.end();
    if (known == presets.end() || row.find("values") != nullptr) {
      rows.push_back(row);
      continue;
    }
    json::Value::Array values;
    for (const auto &value : known->second) {
      values.push_back(json::Value::make_string(value));
    }
    auto expanded = row.as_object();
    expanded["values"] = json::Value::make_array(std::move(values));
    rows.push_back(json::Value::make_object(std::move(expanded)));
  }
  return json::stringify(json::Value::make_array(std::move(rows)));
}

std::string json_quote(std::string_view value) {
  return engine::io::json::stringify_string(value);
}

std::filesystem::path resolve_path(const std::filesystem::path &base,
                                   const std::filesystem::path &path) {
  return path.is_absolute() ? path : base / path;
}

// Looks up `<voice_name>` in `<voice_dir>/prompt_text`, a mapping file with one
// `<basename>|<transcript>` line per built-in voice (same format the webui
// uses).
std::optional<std::string>
load_voice_library_text(const std::filesystem::path &voice_dir,
                        const std::string &voice_name) {
  std::ifstream f(voice_dir / "prompt_text");
  std::string line;
  while (std::getline(f, line)) {
    const auto sep = line.find('|');
    if (sep == std::string::npos)
      continue;
    std::string name = line.substr(0, sep);
    // trim trailing whitespace from name
    while (!name.empty() &&
           std::isspace(static_cast<unsigned char>(name.back()))) {
      name.pop_back();
    }
    if (name == voice_name) {
      return line.substr(sep + 1);
    }
  }
  return std::nullopt;
}

std::optional<std::filesystem::path>
resolve_voice_library_wav(const std::filesystem::path &voice_dir,
                          const std::string &voice_name) {
  const std::filesystem::path name_path(voice_name);
  if (voice_name.empty() || name_path.has_root_name() ||
      name_path.has_root_directory() || name_path.has_parent_path() ||
      name_path.filename().string() != voice_name || voice_name == "." ||
      voice_name == "..") {
    return std::nullopt;
  }
  const auto wav = voice_dir / (voice_name + ".wav");
  std::error_code ec;
  if (!std::filesystem::is_regular_file(wav, ec)) {
    return std::nullopt;
  }
  return wav;
}

std::unordered_map<std::string, std::string>
options_from_object(const Value *value);

std::string safe_upload_name(std::string value) {
  value = std::filesystem::path(value).filename().string();
  for (char &ch : value) {
    const auto uch = static_cast<unsigned char>(ch);
    if (!(std::isalnum(uch) || ch == '.' || ch == '-' || ch == '_')) {
      ch = '_';
    }
  }
  if (value.empty() || value == "." || value == "..") {
    value = "audio.wav";
  }
  return value;
}

ServerModelConfig
model_config_from_json(const Value &body,
                       const std::filesystem::path &request_base, bool lazy) {
  ServerModelConfig model;
  model.id = engine::io::json::require_string(body, "id");
  model.path = resolve_path(request_base,
                            engine::io::json::require_string(body, "path"));
  model.family = engine::io::json::require_string(body, "family");
  model.task = engine::io::json::optional_string(body, "task", model.task);
  model.mode = engine::io::json::optional_string(body, "mode", model.mode);
  model.lazy = lazy;
  if (const auto *value = body.find("model_spec_override")) {
    const auto &selection = value->as_string();
    model.model_spec_override = selection == "embedded"
        ? std::filesystem::path(selection) : resolve_path(request_base, selection);
  }
  if (const auto *value = body.find("config")) {
    model.config_id = value->as_string();
  }
  if (const auto *value = body.find("weight")) {
    model.weight_id = value->as_string();
  }
  model.load_options = options_from_object(body.find("load_options"));
  model.session_options = options_from_object(body.find("session_options"));
  return model;
}

// Minimal application/x-www-form-urlencoded query string lookup, e.g.
// query_param("model=pocket-tts&foo=bar", "model") -> "pocket-tts".
std::string query_param(const std::string &query, const std::string &key) {
  size_t pos = 0;
  while (pos < query.size()) {
    const size_t amp = query.find('&', pos);
    const std::string pair = query.substr(
        pos, amp == std::string::npos ? std::string::npos : amp - pos);
    const auto eq = pair.find('=');
    const std::string name = pair.substr(0, eq);
    if (name == key) {
      return eq == std::string::npos ? "" : pair.substr(eq + 1);
    }
    if (amp == std::string::npos) {
      break;
    }
    pos = amp + 1;
  }
  return {};
}

static int hex_value(char ch) {
  if (ch >= '0' && ch <= '9') {
    return ch - '0';
  }
  if (ch >= 'a' && ch <= 'f') {
    return ch - 'a' + 10;
  }
  if (ch >= 'A' && ch <= 'F') {
    return ch - 'A' + 10;
  }
  return -1;
}

static std::string url_decode_query_value(std::string_view value) {
  std::string out;
  out.reserve(value.size());
  for (size_t i = 0; i < value.size(); ++i) {
    if (value[i] == '+') {
      out.push_back(' ');
    } else if (value[i] == '%' && i + 2 < value.size()) {
      const int hi = hex_value(value[i + 1]);
      const int lo = hex_value(value[i + 2]);
      if (hi >= 0 && lo >= 0) {
        out.push_back(static_cast<char>((hi << 4) | lo));
        i += 2;
      } else {
        out.push_back(value[i]);
      }
    } else {
      out.push_back(value[i]);
    }
  }
  return out;
}

std::string decoded_query_param(const std::string &query,
                                const std::string &key) {
  return url_decode_query_value(query_param(query, key));
}

const char *backend_name(engine::core::BackendType type) {
  switch (type) {
  case engine::core::BackendType::Cpu:
    return "cpu";
  case engine::core::BackendType::Cuda:
    return "cuda";
  case engine::core::BackendType::Hip:
    return "hip";
  case engine::core::BackendType::Vulkan:
    return "vulkan";
  case engine::core::BackendType::Metal:
    return "metal";
  case engine::core::BackendType::BestAvailable:
    return "best";
  }
  return "unknown";
}

HttpResponse ui_service_worker_retirement_response() {
  HttpResponse response;
  response.status = 200;
  response.content_type = "application/javascript; charset=utf-8";
  response.body = R"JS(
// Retire service workers left by applications that previously used this origin.
self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});
self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const cacheNames = await caches.keys();
    await Promise.all(cacheNames.map((name) => caches.delete(name)));
    await self.clients.claim();
    const windows = await self.clients.matchAll({ type: "window", includeUncontrolled: true });
    await self.registration.unregister();
    await Promise.all(windows.map((client) => client.navigate(client.url)));
  })());
});
)JS";
  response.headers["Cache-Control"] =
      "no-store, no-cache, must-revalidate, max-age=0";
  response.headers["Clear-Site-Data"] = "\"cache\"";
  response.headers["Expires"] = "0";
  response.headers["Pragma"] = "no-cache";
  response.headers["Service-Worker-Allowed"] = "/";
  response.headers["X-Content-Type-Options"] = "nosniff";
  return response;
}

std::unordered_map<std::string, std::string>
options_from_object(const Value *value) {
  return minitts::cli::json_options_map(value);
}

std::string
options_json(const std::unordered_map<std::string, std::string> &options) {
  std::ostringstream out;
  out << "{";
  bool first = true;
  for (const auto &[key, value] : options) {
    if (!first) {
      out << ",";
    }
    first = false;
    out << json_quote(key) << ":" << json_quote(value);
  }
  out << "}";
  return out.str();
}

void add_option_from_json(std::unordered_map<std::string, std::string> &options,
                          const Value &object, const std::string &field,
                          const std::string &option_key) {
  const auto *value = object.find(field);
  if (value != nullptr && !value->is_null()) {
    options[option_key] = minitts::cli::json_option_string(*value);
  }
}

std::vector<uint8_t>
encode_pcm16_wav(const engine::runtime::AudioBuffer &audio) {
  if (audio.sample_rate <= 0) {
    throw std::runtime_error("audio output sample rate must be positive");
  }
  if (audio.channels <= 0) {
    throw std::runtime_error("audio output channel count must be positive");
  }
  if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
    throw std::runtime_error(
        "audio output sample count must be divisible by channel count");
  }

  const uint16_t channels = static_cast<uint16_t>(audio.channels);
  const uint16_t bits_per_sample = 16;
  const uint32_t data_bytes =
      static_cast<uint32_t>(audio.samples.size() * sizeof(int16_t));
  const uint32_t riff_size = 36 + data_bytes;
  const uint32_t byte_rate =
      static_cast<uint32_t>(audio.sample_rate) * channels * bits_per_sample / 8;
  const uint16_t block_align = channels * bits_per_sample / 8;

  std::vector<uint8_t> out;
  out.reserve(44 + data_bytes);
  auto append_bytes = [&](const void *data, size_t size) {
    const auto *bytes = static_cast<const uint8_t *>(data);
    out.insert(out.end(), bytes, bytes + size);
  };
  auto append_u16 = [&](uint16_t value) {
    append_bytes(&value, sizeof(value));
  };
  auto append_u32 = [&](uint32_t value) {
    append_bytes(&value, sizeof(value));
  };

  out.insert(out.end(), {'R', 'I', 'F', 'F'});
  append_u32(riff_size);
  out.insert(out.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
  append_u32(16);
  append_u16(1);
  append_u16(channels);
  append_u32(static_cast<uint32_t>(audio.sample_rate));
  append_u32(byte_rate);
  append_u16(block_align);
  append_u16(bits_per_sample);
  out.insert(out.end(), {'d', 'a', 't', 'a'});
  append_u32(data_bytes);
  for (float sample : audio.samples) {
    sample = std::max(-1.0F, std::min(1.0F, sample));
    const auto pcm = static_cast<int16_t>(std::lrint(sample * 32767.0F));
    append_bytes(&pcm, sizeof(pcm));
  }
  return out;
}

std::vector<uint8_t>
encode_pcm16_samples(const engine::runtime::AudioBuffer &audio) {
  if (audio.sample_rate <= 0) {
    throw std::runtime_error("audio output sample rate must be positive");
  }
  if (audio.channels <= 0) {
    throw std::runtime_error("audio output channel count must be positive");
  }
  if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
    throw std::runtime_error(
        "audio output sample count must be divisible by channel count");
  }

  std::vector<uint8_t> out;
  out.reserve(audio.samples.size() * sizeof(int16_t));
  auto append_bytes = [&](const void *data, size_t size) {
    const auto *bytes = static_cast<const uint8_t *>(data);
    out.insert(out.end(), bytes, bytes + size);
  };
  for (float sample : audio.samples) {
    sample = std::max(-1.0F, std::min(1.0F, sample));
    const auto pcm = static_cast<int16_t>(std::lrint(sample * 32767.0F));
    append_bytes(&pcm, sizeof(pcm));
  }
  return out;
}

void write_sse(HttpStreamWriter &writer, const std::string &json) {
  writer.write("data: " + json + "\n\n");
}

void write_sse_done(HttpStreamWriter &writer) {
  writer.write("data: [DONE]\n\n");
}

bool bool_field(const Value &object, const std::string &key,
                bool default_value) {
  const auto *value = object.find(key);
  if (value == nullptr || value->is_null()) {
    return default_value;
  }
  if (value->is_bool()) {
    return value->as_bool();
  }
  if (value->is_string()) {
    const auto str = value->as_string();
    if (str == "true" || str == "1") {
      return true;
    }
    if (str == "false" || str == "0") {
      return false;
    }
  }
  throw std::runtime_error(key + " must be a boolean");
}

HttpResponse sse_response(std::function<void(HttpStreamWriter &)> stream) {
  HttpResponse response;
  response.status = 200;
  response.content_type = "text/event-stream; charset=utf-8";
  response.headers.emplace("X-Accel-Buffering", "no");
  response.stream_body = std::move(stream);
  return response;
}

HttpResponse
chunked_audio_response(std::function<void(HttpStreamWriter &)> stream) {
  HttpResponse response;
  response.status = 200;
  response.content_type = "application/octet-stream";
  response.stream_body = std::move(stream);
  return response;
}

bool is_wav_upload_filename(const std::string &filename) {
  std::string ext = std::filesystem::path(filename).extension().string();
  for (char &ch : ext) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return ext.empty() || ext == ".wav";
}

static std::string lower_ascii(std::string value) {
  for (char &ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}

static std::string_view trim_ascii(std::string_view value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front())) != 0) {
    value.remove_prefix(1);
  }
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back())) != 0) {
    value.remove_suffix(1);
  }
  return value;
}

static bool is_json_content_type(const HttpRequest &request) {
  const auto it = request.headers.find("content-type");
  if (it == request.headers.end()) {
    return false;
  }
  const std::string content_type = lower_ascii(it->second);
  const auto media_type = trim_ascii(
      std::string_view(content_type).substr(0, content_type.find(';')));
  return media_type == "application/json" ||
         (media_type.size() > 5 &&
          media_type.substr(media_type.size() - 5) == "+json");
}

static std::string request_content_type(const HttpRequest &request) {
  const auto it = request.headers.find("content-type");
  return it == request.headers.end() ? "" : it->second;
}

void log_request_body_if_enabled(const ServerConfig &config,
                                 const HttpRequest &request) {
  if (!config.log_request_body || !engine::debug::log_enabled()) {
    return;
  }
  if (request.method != "POST") {
    return;
  }
  if (!request.body.empty() && is_json_content_type(request)) {
    engine::debug::log_message("[REQUEST_BODY] " + request.method + " " +
                               request.path);
    engine::debug::log_message(request.body);
    return;
  }
  if (request.body.empty() && request.body_stream == nullptr) {
    return;
  }
  std::ostringstream out;
  out << "[REQUEST_BODY_SKIPPED] " << request.method << " " << request.path
      << " content_type=" << json_quote(request_content_type(request));
  if (request.body_stream != nullptr) {
    out << " body=stream";
  } else {
    out << " body_bytes=" << request.body.size();
  }
  if (!request.query.empty()) {
    out << " query=" << json_quote(request.query);
  }
  engine::debug::log_message(out.str());
}

void log_multipart_request_summary_if_enabled(
    const ServerConfig &config, const std::vector<MultipartPart> &parts) {
  if (!config.log_request_body || !engine::debug::log_enabled()) {
    return;
  }
  for (const auto &part : parts) {
    if (part.filename.empty()) {
      continue;
    }
    std::ostringstream out;
    out << "[REQUEST_BODY_SKIPPED] multipart_file"
        << " field=" << json_quote(part.name)
        << " filename=" << json_quote(part.filename)
        << " bytes=" << part.data.size();
    engine::debug::log_message(out.str());
  }
}

double elapsed_ms(Clock::time_point started) {
  return std::chrono::duration<double, std::milli>(Clock::now() - started)
      .count();
}

double audio_duration_ms(const engine::runtime::AudioBuffer &audio) {
  if (audio.sample_rate <= 0 || audio.channels <= 0) {
    return 0.0;
  }
  return 1000.0 * static_cast<double>(audio.samples.size()) /
         static_cast<double>(audio.sample_rate * audio.channels);
}

double audio_rtf(double wall_ms, double duration_ms) {
  return duration_ms > 0.0 ? wall_ms / duration_ms : 0.0;
}

std::string timing_json(double wall_ms) {
  std::ostringstream out;
  out << "{\"wall_ms\":" << wall_ms << "}";
  return out.str();
}

std::string timing_json(double wall_ms,
                        const engine::runtime::AudioBuffer &audio) {
  const double duration_ms = audio_duration_ms(audio);
  std::ostringstream out;
  out << "{\"wall_ms\":" << wall_ms << ",\"audio_duration_ms\":" << duration_ms
      << ",\"rtf\":" << audio_rtf(wall_ms, duration_ms) << "}";
  return out.str();
}

std::string ttft_timing_json(double ttft_ms) {
  std::ostringstream out;
  out << "{\"ttft_ms\":" << ttft_ms << "}";
  return out.str();
}

std::string live_speech_timing_json(double request_start_to_first_audio_ms,
                                    double input_end_ms) {
  const double input_end_to_first_audio_ms =
      request_start_to_first_audio_ms - input_end_ms;
  const bool first_audio_before_input_end = input_end_to_first_audio_ms < 0.0;
  const double overlap_ms =
      first_audio_before_input_end ? -input_end_to_first_audio_ms : 0.0;
  std::ostringstream out;
  out << "{\"ttft_ms\":";
  if (first_audio_before_input_end) {
    out << "null";
  } else {
    out << input_end_to_first_audio_ms;
  }
  out << ",\"first_audio_before_input_end\":"
      << (first_audio_before_input_end ? "true" : "false")
      << ",\"request_start_to_first_audio_ms\":"
      << request_start_to_first_audio_ms << ",\"input_end_ms\":" << input_end_ms
      << ",\"overlap_ms\":" << overlap_ms << "}";
  return out.str();
}

bool stream_event_has_output(const engine::runtime::StreamEvent &event) {
  return (event.partial_text.has_value() &&
          !event.partial_text->text.empty()) ||
         (event.partial_text_snapshot.has_value() &&
          !event.partial_text_snapshot->text.empty()) ||
         event.audio_output.has_value() || !event.named_audio_outputs.empty();
}

bool task_result_has_output(const engine::runtime::TaskResult &result) {
  return result.text_output.has_value() || result.audio_output.has_value() ||
         !result.named_audio_outputs.empty() ||
         result.artifact_output.has_value() || !result.output_artifacts.empty();
}

static const char *artifact_kind_name(engine::runtime::ArtifactKind kind) {
  using engine::runtime::ArtifactKind;
  switch (kind) {
  case ArtifactKind::SpeakerEmbedding:
    return "speaker_embedding";
  case ArtifactKind::StyleEmbedding:
    return "style_embedding";
  case ArtifactKind::PromptEmbedding:
    return "prompt_embedding";
  case ArtifactKind::AcousticTokens:
    return "acoustic_tokens";
  case ArtifactKind::Midi:
    return "midi";
  case ArtifactKind::TranscriptAlignment:
    return "transcript_alignment";
  case ArtifactKind::DiarizationState:
    return "diarization_state";
  case ArtifactKind::VadState:
    return "vad_state";
  case ArtifactKind::Custom:
    return "custom";
  }
  return "custom";
}

double require_ttft_ms(const std::optional<double> &ttft_ms) {
  if (!ttft_ms.has_value()) {
    throw std::runtime_error("streaming response produced no TTFT event");
  }
  return *ttft_ms;
}

double require_input_end_ms(const std::optional<double> &input_end_ms) {
  if (!input_end_ms.has_value()) {
    throw std::runtime_error(
        "live streaming response did not observe input end");
  }
  return *input_end_ms;
}

std::unordered_map<std::string, std::string>
timing_headers(double wall_ms, const engine::runtime::AudioBuffer &audio) {
  const double duration_ms = audio_duration_ms(audio);
  return {
      {"X-AudioCPP-Wall-Ms", std::to_string(wall_ms)},
      {"X-AudioCPP-Audio-Duration-Ms", std::to_string(duration_ms)},
      {"X-AudioCPP-RTF", std::to_string(audio_rtf(wall_ms, duration_ms))},
  };
}

static std::string
speaker_turns_json(const std::vector<engine::runtime::SpeakerTurn> &turns) {
  std::ostringstream out;
  out << "[";
  for (size_t i = 0; i < turns.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    const auto &turn = turns[i];
    out << "{\"start_sample\":" << turn.span.start_sample
        << ",\"end_sample\":" << turn.span.end_sample
        << ",\"speaker_id\":" << json_quote(turn.speaker_id)
        << ",\"confidence\":" << turn.confidence;
    if (!turn.text.empty()) {
      out << ",\"text\":" << json_quote(turn.text);
    }
    out << "}";
  }
  out << "]";
  return out.str();
}

std::string
diarization_event_json(const std::vector<engine::runtime::SpeakerTurn> &turns,
                       int sample_rate, bool final,
                       const std::optional<double> &ttft_ms) {
  std::ostringstream out;
  out << "{\"type\":\"diarization." << (final ? "done" : "delta")
      << "\",\"speaker_turns\":" << speaker_turns_json(turns)
      << ",\"sample_rate\":" << sample_rate;
  if (final) {
    out << ",\"timing\":"
        << (ttft_ms ? ttft_timing_json(*ttft_ms) : "{\"ttft_ms\":null}");
  }
  out << "}";
  return out.str();
}

// Transcript detail arrays shared by /v1/tasks/run and
// /v1/audio/transcriptions. ASR models that produce timestamps populate only
// these fields, so a route that omits them silently discards work the model
// already did.
template <typename FieldFn>
static void
write_transcript_detail_fields_with(std::ostringstream &out,
                                    const engine::runtime::TaskResult &result,
                                    FieldFn field) {
  if (!result.speech_segments.empty()) {
    field("segments");
    out << "[";
    for (size_t i = 0; i < result.speech_segments.size(); ++i) {
      if (i != 0) {
        out << ",";
      }
      const auto &segment = result.speech_segments[i];
      out << "{\"start_sample\":" << segment.span.start_sample
          << ",\"end_sample\":" << segment.span.end_sample
          << ",\"confidence\":" << segment.confidence;
      if (!segment.text.empty()) {
        out << ",\"text\":" << json_quote(segment.text);
      }
      out << "}";
    }
    out << "]";
  }
  if (!result.speaker_turns.empty()) {
    field("speaker_turns");
    out << speaker_turns_json(result.speaker_turns);
  }
  if (!result.word_timestamps.empty()) {
    field("words");
    out << "[";
    for (size_t i = 0; i < result.word_timestamps.size(); ++i) {
      if (i != 0) {
        out << ",";
      }
      const auto &word = result.word_timestamps[i];
      out << "{\"word\":" << json_quote(word.word)
          << ",\"start_sample\":" << word.span.start_sample
          << ",\"end_sample\":" << word.span.end_sample
          << ",\"confidence\":" << word.confidence << "}";
    }
    out << "]";
  }
}

void write_transcript_detail_fields(std::ostringstream &out,
                                    const engine::runtime::TaskResult &result) {
  write_transcript_detail_fields_with(
      out, result,
      [&](const std::string &name) { out << "," << json_quote(name) << ":"; });
}

std::string
task_result_json_with_timing(const engine::runtime::TaskResult &result,
                             const std::string &timing) {
  std::ostringstream out;
  out << "{";
  bool first = true;
  auto field = [&](const std::string &name) {
    if (!first) {
      out << ",";
    }
    first = false;
    out << json_quote(name) << ":";
  };

  if (result.text_output.has_value()) {
    field("text");
    out << json_quote(result.text_output->text);
    if (!result.text_output->language.empty()) {
      field("language");
      out << json_quote(result.text_output->language);
    }
  }
  if (result.custom_schema_output.has_value()) {
    field("custom_schema_output");
    out << "{\"schema\":" << json_quote(result.custom_schema_output->schema)
        << ",\"data\":"
        << engine::io::json::stringify(result.custom_schema_output->data)
        << "}";
  }
  if (result.audio_output.has_value()) {
    const auto wav = encode_pcm16_wav(*result.audio_output);
    field("audio");
    out << json_quote(base64_encode(wav));
    field("sample_rate");
    out << result.audio_output->sample_rate;
    field("channels");
    out << result.audio_output->channels;
  }
  if (!result.named_audio_outputs.empty()) {
    field("named_audio_outputs");
    out << "[";
    for (size_t i = 0; i < result.named_audio_outputs.size(); ++i) {
      if (i != 0) {
        out << ",";
      }
      const auto wav = encode_pcm16_wav(result.named_audio_outputs[i].audio);
      out << "{\"id\":" << json_quote(result.named_audio_outputs[i].id)
          << ",\"audio\":" << json_quote(base64_encode(wav))
          << ",\"sample_rate\":"
          << result.named_audio_outputs[i].audio.sample_rate
          << ",\"channels\":" << result.named_audio_outputs[i].audio.channels
          << "}";
    }
    out << "]";
  }
  if (result.artifact_output.has_value() || !result.output_artifacts.empty()) {
    field("artifacts");
    out << "[";
    bool first_artifact = true;
    auto write_artifact = [&](const engine::runtime::VoiceArtifact &artifact) {
      if (!first_artifact)
        out << ",";
      first_artifact = false;
      out << "{\"id\":" << json_quote(artifact.id)
          << ",\"kind\":" << json_quote(artifact_kind_name(artifact.kind))
          << ",\"payload\":" << json_quote(base64_encode(artifact.payload))
          << ",\"meta\":{";
      bool first_meta = true;
      for (const auto &[key, value] : artifact.meta) {
        if (!first_meta)
          out << ",";
        first_meta = false;
        out << json_quote(key) << ":" << json_quote(value);
      }
      out << "}}";
    };
    if (result.artifact_output.has_value())
      write_artifact(*result.artifact_output);
    for (const auto &artifact : result.output_artifacts)
      write_artifact(artifact);
    out << "]";
  }
  write_transcript_detail_fields_with(out, result, field);
  field("timing");
  out << timing;
  out << "}";
  return out.str();
}

std::string task_result_json(const engine::runtime::TaskResult &result,
                             double wall_ms) {
  if (result.audio_output.has_value()) {
    return task_result_json_with_timing(
        result, timing_json(wall_ms, *result.audio_output));
  }
  if (result.named_audio_outputs.size() == 1) {
    return task_result_json_with_timing(
        result, timing_json(wall_ms, result.named_audio_outputs.front().audio));
  }
  return task_result_json_with_timing(result, timing_json(wall_ms));
}

std::string alignment_result_json(const engine::runtime::TaskResult &result,
                                  const engine::runtime::AudioBuffer &audio,
                                  double wall_ms) {
  if (audio.sample_rate <= 0) {
    throw std::runtime_error(
        "alignment timing requires a positive input sample rate");
  }

  std::ostringstream out;
  out << "{";
  bool first = true;
  auto field = [&](const std::string &name) {
    if (!first) {
      out << ",";
    }
    first = false;
    out << json_quote(name) << ":";
  };
  if (result.text_output.has_value()) {
    field("text");
    out << json_quote(result.text_output->text);
    if (!result.text_output->language.empty()) {
      field("language");
      out << json_quote(result.text_output->language);
    }
  }
  field("words");
  out << "[";
  for (size_t i = 0; i < result.word_timestamps.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    const auto &word = result.word_timestamps[i];
    out << "{\"word\":" << json_quote(word.word) << ",\"start\":"
        << (static_cast<double>(word.span.start_sample) / audio.sample_rate)
        << ",\"end\":"
        << (static_cast<double>(word.span.end_sample) / audio.sample_rate)
        << ",\"start_sample\":" << word.span.start_sample
        << ",\"end_sample\":" << word.span.end_sample
        << ",\"confidence\":" << word.confidence << "}";
  }
  out << "]";
  field("timing");
  out << timing_json(wall_ms, audio);
  out << "}";
  return out.str();
}

std::string
streaming_task_result_json(const engine::runtime::TaskResult &result,
                           const std::optional<double> &ttft_ms) {
  return task_result_json_with_timing(
      result, ttft_timing_json(require_ttft_ms(ttft_ms)));
}

std::string stream_event_json(const engine::runtime::StreamEvent &event,
                              bool diarization) {
  std::ostringstream out;
  out << "{";
  bool first = true;
  auto field = [&](const char *name) {
    if (!first) {
      out << ",";
    }
    first = false;
    out << "\"" << name << "\":";
  };
  if (event.partial_text.has_value()) {
    field("partial_text");
    out << "{\"text\":" << json_quote(event.partial_text->text)
        << ",\"language\":" << json_quote(event.partial_text->language) << "}";
  }
  if (event.partial_text_snapshot.has_value()) {
    field("partial_text_snapshot");
    out << "{\"text\":" << json_quote(event.partial_text_snapshot->text)
        << ",\"language\":" << json_quote(event.partial_text_snapshot->language)
        << "}";
  }
  if (event.audio_output.has_value()) {
    const auto wav = encode_pcm16_wav(*event.audio_output);
    field("audio");
    out << json_quote(base64_encode(wav));
  }
  if (!event.named_audio_outputs.empty()) {
    field("named_audio_outputs");
    out << "[";
    for (size_t i = 0; i < event.named_audio_outputs.size(); ++i) {
      if (i != 0) {
        out << ",";
      }
      const auto wav = encode_pcm16_wav(event.named_audio_outputs[i].audio);
      out << "{\"id\":" << json_quote(event.named_audio_outputs[i].id)
          << ",\"audio\":" << json_quote(base64_encode(wav))
          << ",\"format\":\"wav\"}";
    }
    out << "]";
  }
  if (!event.word_timestamps.empty()) {
    field("word_timestamps");
    out << "[";
    for (size_t i = 0; i < event.word_timestamps.size(); ++i) {
      if (i != 0) {
        out << ",";
      }
      out << "{\"start_sample\":" << event.word_timestamps[i].span.start_sample
          << ",\"end_sample\":" << event.word_timestamps[i].span.end_sample
          << ",\"word\":" << json_quote(event.word_timestamps[i].word)
          << ",\"confidence\":" << event.word_timestamps[i].confidence << "}";
    }
    out << "]";
  }
  if (diarization && !event.speaker_turns.empty()) {
    field("speaker_turns");
    out << speaker_turns_json(event.speaker_turns);
  }
  field("is_final");
  out << (event.is_final ? "true" : "false");
  out << "}";
  return out.str();
}

const engine::runtime::AudioBuffer &
select_audio_output(const engine::runtime::TaskResult &result) {
  if (result.audio_output.has_value()) {
    return *result.audio_output;
  }
  if (result.named_audio_outputs.size() == 1) {
    return result.named_audio_outputs.front().audio;
  }
  throw std::runtime_error(
      "model result did not contain exactly one audio output");
}

engine::runtime::TaskRequest build_openai_transcription_request(
    const Value &body, const std::filesystem::path &base_dir,
    bool accepts_language_option, const std::string *uploaded_audio_bytes) {
  const auto *audio = body.find("audio");
  if (audio == nullptr) {
    audio = body.find("audio_path");
  }
  if (audio == nullptr) {
    audio = body.find("file");
  }
  if (uploaded_audio_bytes == nullptr &&
      (audio == nullptr || !audio->is_string())) {
    throw std::runtime_error(
        "transcription request requires audio, audio_path, or file path");
  }

  engine::runtime::TaskRequest request;
  if (uploaded_audio_bytes == nullptr) {
    request.audio_input = minitts::cli::read_audio_buffer(
        resolve_path(base_dir, audio->as_string()));
  } else {
    request.audio_input = minitts::cli::read_audio_buffer(
        std::string_view(*uploaded_audio_bytes));
  }
  request.options = options_from_object(body.find("options"));
  std::string language;
  if (const auto *value = body.find("language")) {
    language = value->as_string();
    // Only forward a language the caller actually chose, and only to a model
    // whose contract accepts it. Clients send the field unconditionally, so
    // without both guards a model that validates request options strictly
    // rejects the whole request over an option the user never set. The
    // language still reaches the model through text_input either way.
    if (!language.empty() && accepts_language_option) {
      request.options["language"] = language;
    }
  }
  std::string context;
  if (const auto *value = body.find("text")) {
    context = value->as_string();
  }
  if (!language.empty() || !context.empty()) {
    request.text_input =
        engine::runtime::Transcript{std::move(context), std::move(language)};
  }
  return request;
}

template <typename Predicate>
static std::optional<std::filesystem::path>
find_ancestor(std::filesystem::path start, Predicate predicate) {
  std::error_code ec;
  start = std::filesystem::absolute(std::move(start), ec).lexically_normal();
  if (ec) {
    return std::nullopt;
  }
  for (;;) {
    if (predicate(start)) {
      return start;
    }
    const auto parent = start.parent_path();
    if (parent.empty() || parent == start) {
      return std::nullopt;
    }
    start = parent;
  }
}

std::optional<std::filesystem::path>
find_repository_root(const std::filesystem::path &request_base,
                     const std::filesystem::path &resource_anchor) {
  const auto has_model_specs = [](const std::filesystem::path &root) {
    return std::filesystem::is_directory(root / "model_specs");
  };
  if (auto found = find_ancestor(request_base, has_model_specs)) {
    return found;
  }
  if (!resource_anchor.empty()) {
    return find_ancestor(resource_anchor, has_model_specs);
  }
  return std::nullopt;
}

} // namespace minitts::server
