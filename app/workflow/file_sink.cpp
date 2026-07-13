#include "file_sink.h"

#include "engine/framework/audio/output.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace minitts::app {
namespace {

std::string quote_json(const std::string & value) {
    std::string out = "\"";
    for (char ch : value) {
        if (ch == '\\' || ch == '"') {
            out.push_back('\\');
        }
        out.push_back(ch);
    }
    out.push_back('"');
    return out;
}

std::string speech_segments_to_json(const std::vector<engine::runtime::SpeechSegment> & segments) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < segments.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << "{\"start_sample\":" << segments[i].span.start_sample
            << ",\"end_sample\":" << segments[i].span.end_sample
            << ",\"confidence\":" << segments[i].confidence
            << "}";
    }
    out << "]";
    return out.str();
}

std::string speaker_turns_to_json(const std::vector<engine::runtime::SpeakerTurn> & turns) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < turns.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << "{\"start_sample\":" << turns[i].span.start_sample
            << ",\"end_sample\":" << turns[i].span.end_sample
            << ",\"speaker_id\":" << quote_json(turns[i].speaker_id)
            << ",\"confidence\":" << turns[i].confidence
            << "}";
    }
    out << "]";
    return out.str();
}

const char * artifact_kind_name(engine::runtime::ArtifactKind kind) {
    switch (kind) {
    case engine::runtime::ArtifactKind::SpeakerEmbedding:
        return "speaker_embedding";
    case engine::runtime::ArtifactKind::StyleEmbedding:
        return "style_embedding";
    case engine::runtime::ArtifactKind::PromptEmbedding:
        return "prompt_embedding";
    case engine::runtime::ArtifactKind::AcousticTokens:
        return "acoustic_tokens";
    case engine::runtime::ArtifactKind::TranscriptAlignment:
        return "transcript_alignment";
    case engine::runtime::ArtifactKind::DiarizationState:
        return "diarization_state";
    case engine::runtime::ArtifactKind::VadState:
        return "vad_state";
    case engine::runtime::ArtifactKind::Custom:
        return "custom";
    }
    return "unknown";
}

std::string bytes_to_hex(const std::vector<std::byte> & bytes) {
    constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (const std::byte byte : bytes) {
        const auto value = std::to_integer<unsigned int>(byte);
        out.push_back(kHex[(value >> 4) & 0x0f]);
        out.push_back(kHex[value & 0x0f]);
    }
    return out;
}

std::string artifact_to_json(const engine::runtime::VoiceArtifact & artifact) {
    std::vector<std::string> meta_keys;
    meta_keys.reserve(artifact.meta.size());
    for (const auto & [key, _] : artifact.meta) {
        meta_keys.push_back(key);
    }
    std::sort(meta_keys.begin(), meta_keys.end());

    std::ostringstream out;
    out << "{"
        << "\"id\":" << quote_json(artifact.id)
        << ",\"kind\":" << quote_json(artifact_kind_name(artifact.kind))
        << ",\"bytes\":" << artifact.payload.size()
        << ",\"meta\":{";
    for (size_t i = 0; i < meta_keys.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        const auto & key = meta_keys[i];
        out << quote_json(key) << ":" << quote_json(artifact.meta.at(key));
    }
    out << "},\"payload_hex\":" << quote_json(bytes_to_hex(artifact.payload)) << "}";
    return out.str();
}

std::optional<std::filesystem::path> suffixed_json_path(
    const std::optional<std::filesystem::path> & base,
    const std::string & request_id) {
    if (!base.has_value()) {
        return std::nullopt;
    }
    return base->parent_path() / (base->stem().string() + "_" + request_id + base->extension().string());
}

void write_wav_output(
    const std::filesystem::path & path,
    const engine::audio::AudioBuffer & audio) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    const auto tmp = path.parent_path() / (path.filename().string() + ".tmp");
    std::filesystem::remove(tmp);
    engine::audio::WavPcm16Sink().write(tmp, audio);
    if (std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }
    std::filesystem::rename(tmp, path);
}

std::string batch_manifest_to_json(const AppBatchResult & batch) {
    std::ostringstream out;
    out << "{\"requests\":[";
    for (size_t i = 0; i < batch.results.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        const auto & item = batch.results[i];
        out << "{\"id\":" << quote_json(item.id);
        if (item.result.audio_output.has_value()) {
            out << ",\"sample_rate\":" << item.result.audio_output->sample_rate
                << ",\"channels\":" << item.result.audio_output->channels
                << ",\"samples\":" << item.result.audio_output->samples.size();
        }
        out << "}";
    }
    out << "],\"chapters\":[";
    for (size_t i = 0; i < batch.chapters.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << "{\"id\":" << quote_json(batch.chapters[i].id)
            << ",\"start_sample\":" << batch.chapters[i].start_sample
            << ",\"end_sample\":" << batch.chapters[i].end_sample
            << "}";
    }
    out << "]}";
    return out.str();
}

}  // namespace

std::string word_timestamps_to_json(const std::vector<engine::runtime::WordTimestamp> & words) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < words.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << "{\"start_sample\":" << words[i].span.start_sample
            << ",\"end_sample\":" << words[i].span.end_sample
            << ",\"word\":" << quote_json(words[i].word)
            << ",\"confidence\":" << words[i].confidence
            << "}";
    }
    out << "]";
    return out.str();
}

std::string safe_output_name(const std::string & value) {
    std::string out;
    out.reserve(value.size());
    for (const char ch : value) {
        const unsigned char uch = static_cast<unsigned char>(ch);
        out.push_back(std::isalnum(uch) != 0 || ch == '-' || ch == '_' ? ch : '_');
    }
    return out.empty() ? "request" : out;
}

void emit_task_result(
    const engine::runtime::TaskResult & result,
    const std::optional<std::filesystem::path> & audio_out,
    const std::optional<std::filesystem::path> & audio_out_dir,
    const std::optional<std::filesystem::path> & artifact_out_dir,
    const std::optional<std::filesystem::path> & segments_out,
    const std::optional<std::filesystem::path> & turns_out,
    const std::optional<std::filesystem::path> & words_out) {
    if (result.audio_output.has_value() && audio_out.has_value()) {
        write_wav_output(*audio_out, engine::audio::AudioBuffer{
            result.audio_output->sample_rate,
            result.audio_output->channels,
            result.audio_output->samples,
        });
        std::cout << "audio_out=" << audio_out->string() << "\n";
    } else if (!result.named_audio_outputs.empty() && audio_out.has_value()) {
        if (result.named_audio_outputs.size() != 1) {
            throw std::runtime_error("--out requires exactly one audio output");
        }
        const auto & audio = result.named_audio_outputs.front().audio;
        write_wav_output(*audio_out, engine::audio::AudioBuffer{
            audio.sample_rate,
            audio.channels,
            audio.samples,
        });
        std::cout << "audio_out=" << audio_out->string() << "\n";
    }

    if (!result.named_audio_outputs.empty()) {
        if (audio_out_dir.has_value()) {
            std::filesystem::create_directories(*audio_out_dir);
            for (const auto & output : result.named_audio_outputs) {
                const auto path = *audio_out_dir / (output.id + ".wav");
                write_wav_output(path, engine::audio::AudioBuffer{
                    output.audio.sample_rate,
                    output.audio.channels,
                    output.audio.samples,
                });
                std::cout << "audio_out[" << output.id << "]=" << path.string() << "\n";
            }
        } else {
            std::cout << "audio_outputs=" << result.named_audio_outputs.size() << "\n";
            for (const auto & output : result.named_audio_outputs) {
                std::cout << "audio_output_id=" << output.id
                          << " sample_rate=" << output.audio.sample_rate
                          << " channels=" << output.audio.channels
                          << " samples=" << output.audio.samples.size()
                          << "\n";
            }
        }
    }

    if (!result.speech_segments.empty()) {
        const auto json = speech_segments_to_json(result.speech_segments);
        if (segments_out.has_value()) {
            if (!segments_out->parent_path().empty()) {
                std::filesystem::create_directories(segments_out->parent_path());
            }
            std::ofstream(*segments_out) << json;
            std::cout << "segments_out=" << segments_out->string() << "\n";
        } else {
            std::cout << "speech_segments=" << json << "\n";
        }
    }

    if (!result.speaker_turns.empty()) {
        const auto json = speaker_turns_to_json(result.speaker_turns);
        if (turns_out.has_value()) {
            if (!turns_out->parent_path().empty()) {
                std::filesystem::create_directories(turns_out->parent_path());
            }
            std::ofstream(*turns_out) << json;
            std::cout << "turns_out=" << turns_out->string() << "\n";
        } else {
            std::cout << "speaker_turns=" << json << "\n";
        }
    }

    if (!result.word_timestamps.empty()) {
        const auto json = word_timestamps_to_json(result.word_timestamps);
        if (words_out.has_value()) {
            if (!words_out->parent_path().empty()) {
                std::filesystem::create_directories(words_out->parent_path());
            }
            std::ofstream(*words_out) << json;
            std::cout << "words_out=" << words_out->string() << "\n";
        } else {
            std::cout << "word_timestamps=" << json << "\n";
        }
    }

    if (result.text_output.has_value()) {
        std::cout << "text_output=" << result.text_output->text << "\n";
    }
    if (!result.output_artifacts.empty()) {
        std::cout << "artifacts=" << result.output_artifacts.size() << "\n";
        for (const auto & artifact : result.output_artifacts) {
            std::cout << "artifact=" << artifact.id
                      << " kind=" << artifact_kind_name(artifact.kind)
                      << " bytes=" << artifact.payload.size() << "\n";
            if (artifact_out_dir.has_value()) {
                std::filesystem::create_directories(*artifact_out_dir);
                const auto path = *artifact_out_dir / (safe_output_name(artifact.id) + ".json");
                std::ofstream(path) << artifact_to_json(artifact) << "\n";
                std::cout << "artifact_out[" << artifact.id << "]=" << path.string() << "\n";
            }
        }
    }
}

void emit_batch_result(
    const AppBatchResult & batch,
    const FileOutputPolicy & policy) {
    emit_batch_summary(batch, policy);
    for (size_t i = 0; i < batch.results.size(); ++i) {
        emit_batch_item_result(i, batch.results[i], policy);
    }

    std::cout << "[TIMING] session.prepare_ms " << batch.prepare_ms << "\n";
    std::cout << "[TIMING] session.wall_ms " << batch.session_wall_ms << "\n";
}

void emit_batch_summary(
    const AppBatchResult & batch,
    const FileOutputPolicy & policy) {
    if (batch.merged_audio.has_value()) {
        if (!policy.audio_out.has_value()) {
            throw std::runtime_error("merged batch audio requires --out");
        }
        write_wav_output(*policy.audio_out, engine::audio::AudioBuffer{
            batch.merged_audio->sample_rate,
            batch.merged_audio->channels,
            batch.merged_audio->samples,
        });
        std::cout << "merged_audio_out=" << policy.audio_out->string() << "\n";
    }

    if (policy.batch_manifest_out.has_value()) {
        if (!policy.batch_manifest_out->parent_path().empty()) {
            std::filesystem::create_directories(policy.batch_manifest_out->parent_path());
        }
        std::ofstream(*policy.batch_manifest_out) << batch_manifest_to_json(batch) << "\n";
        std::cout << "batch_manifest_out=" << policy.batch_manifest_out->string() << "\n";
    }
}

void emit_batch_item_result(
    size_t index,
    const AppRequestResult & item,
    const FileOutputPolicy & policy) {
    const std::string request_id = safe_output_name(item.id);
    std::cout << "request_index=" << index << "\n";
    std::cout << "request_id=" << request_id << "\n";
    std::cout << "[TIMING] request." << request_id << ".wall_ms " << item.wall_ms << "\n";

    std::optional<std::filesystem::path> audio_out;
    std::optional<std::filesystem::path> named_out_dir;
    std::optional<std::filesystem::path> artifact_out_dir;
    if (policy.output_dir.has_value()) {
        if (item.result.audio_output.has_value()) {
            audio_out = *policy.output_dir / (request_id + ".wav");
        }
        if (!item.result.named_audio_outputs.empty()) {
            named_out_dir = *policy.output_dir / request_id;
        }
        if (!item.result.output_artifacts.empty()) {
            artifact_out_dir = *policy.output_dir / request_id;
        }
    }
    emit_task_result(
        item.result,
        audio_out,
        named_out_dir,
        artifact_out_dir,
        suffixed_json_path(policy.segments_base, request_id),
        suffixed_json_path(policy.turns_base, request_id),
        suffixed_json_path(policy.words_base, request_id));
}

}  // namespace minitts::app
