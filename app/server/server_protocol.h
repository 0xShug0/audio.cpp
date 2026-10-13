#pragma once

#include "config.h"
#include "http.h"
#include "multipart.h"

#include "engine/framework/io/json.h"
#include "engine/framework/runtime/session.h"

#include <chrono>
#include <filesystem>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace minitts::server {

using ServerClock = std::chrono::steady_clock;

std::optional<int>
parse_busy_timeout_override(const engine::io::json::Value &body);
bool model_accepts_request_option(
    std::string_view family, std::string_view option,
    const std::optional<std::filesystem::path> &model_spec_override,
    const std::filesystem::path &model_path);
std::string model_request_params_json(
    std::string_view family,
    const std::optional<std::filesystem::path> &model_spec_override,
    const std::filesystem::path &model_path);
std::string json_quote(std::string_view value);
std::filesystem::path resolve_path(const std::filesystem::path &base,
                                   const std::filesystem::path &path);
std::optional<std::string>
load_voice_library_text(const std::filesystem::path &voice_dir,
                        const std::string &voice_id);
std::optional<std::filesystem::path>
resolve_voice_library_wav(const std::filesystem::path &voice_dir,
                          const std::string &voice_id);
std::unordered_map<std::string, std::string>
options_from_object(const engine::io::json::Value *value);

struct RuntimeVoicePreset {
  std::optional<std::string> voice_id;
  std::optional<engine::runtime::AudioBuffer> audio;
  std::optional<std::string> reference_text;
};

struct SpeechRequestContext {
  const std::filesystem::path &request_base;
  const std::optional<std::filesystem::path> &voice_dir;
  const std::unordered_map<std::string, RuntimeVoicePreset> &voice_presets;
  const std::optional<RuntimeVoicePreset> &default_voice_preset;
  bool accepts_reference_text = true;
  bool accepts_speed = true;
  bool accepts_speaking_rate = true;
  const std::unordered_map<std::string, std::string> &default_options;
};

engine::runtime::TaskRequest build_speech_request(
    const engine::io::json::Value &body,
    const SpeechRequestContext &context);
engine::runtime::TaskRequest apply_default_request_options(
    const std::unordered_map<std::string, std::string> &default_options,
    engine::runtime::TaskRequest request);
std::string safe_upload_name(std::string value);
ServerModelConfig
model_config_from_json(const engine::io::json::Value &body,
                       const std::filesystem::path &request_base, bool lazy);
std::string query_param(const std::string &query, const std::string &key);
std::string decoded_query_param(const std::string &query,
                                const std::string &key);
const char *backend_name(engine::core::BackendType type);
HttpResponse ui_service_worker_retirement_response();
std::string
options_json(const std::unordered_map<std::string, std::string> &options);
void add_option_from_json(std::unordered_map<std::string, std::string> &options,
                          const engine::io::json::Value &object,
                          const std::string &field,
                          const std::string &option_key);
std::vector<uint8_t>
encode_pcm16_wav(const engine::runtime::AudioBuffer &audio);
std::vector<uint8_t>
encode_pcm16_samples(const engine::runtime::AudioBuffer &audio);
void write_sse(HttpStreamWriter &writer, const std::string &json);
void write_sse_done(HttpStreamWriter &writer);
bool bool_field(const engine::io::json::Value &object, const std::string &key,
                bool default_value);
HttpResponse sse_response(std::function<void(HttpStreamWriter &)> stream);
HttpResponse
chunked_audio_response(std::function<void(HttpStreamWriter &)> stream);
bool is_wav_upload_filename(const std::string &filename);
void log_request_body_if_enabled(const ServerConfig &config,
                                 const HttpRequest &request);
void log_multipart_request_summary_if_enabled(
    const ServerConfig &config, const std::vector<MultipartPart> &parts);
double elapsed_ms(ServerClock::time_point started);
double audio_duration_ms(const engine::runtime::AudioBuffer &audio);
double audio_rtf(double wall_ms, double duration_ms);
std::string timing_json(double wall_ms);
std::string timing_json(double wall_ms,
                        const engine::runtime::AudioBuffer &audio);
std::string ttft_timing_json(double ttft_ms);
std::string live_speech_timing_json(double request_start_to_first_audio_ms,
                                    double input_end_ms);
bool stream_event_has_output(const engine::runtime::StreamEvent &event);
bool task_result_has_output(const engine::runtime::TaskResult &result);
double require_ttft_ms(const std::optional<double> &ttft_ms);
double require_input_end_ms(const std::optional<double> &input_end_ms);
std::unordered_map<std::string, std::string>
timing_headers(double wall_ms, const engine::runtime::AudioBuffer &audio);
std::string
diarization_event_json(const std::vector<engine::runtime::SpeakerTurn> &turns,
                       int sample_rate, bool final,
                       const std::optional<double> &ttft_ms = std::nullopt);
void write_transcript_detail_fields(std::ostringstream &out,
                                    const engine::runtime::TaskResult &result);
std::string
task_result_json_with_timing(const engine::runtime::TaskResult &result,
                             const std::string &timing);
std::string task_result_json(const engine::runtime::TaskResult &result,
                             double wall_ms);
std::string alignment_result_json(const engine::runtime::TaskResult &result,
                                  const engine::runtime::AudioBuffer &audio,
                                  double wall_ms);
std::string
streaming_task_result_json(const engine::runtime::TaskResult &result,
                           const std::optional<double> &ttft_ms);
std::string stream_event_json(const engine::runtime::StreamEvent &event,
                              bool diarization);
const engine::runtime::AudioBuffer &
select_audio_output(const engine::runtime::TaskResult &result);
engine::runtime::TaskRequest build_openai_transcription_request(
    const engine::io::json::Value &body, const std::filesystem::path &base_dir,
    bool accepts_language_option,
    const std::string *uploaded_audio_bytes = nullptr);
std::optional<std::filesystem::path>
find_repository_root(const std::filesystem::path &request_base,
                     const std::filesystem::path &resource_anchor);

} // namespace minitts::server
