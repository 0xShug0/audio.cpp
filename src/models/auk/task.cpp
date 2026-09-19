// GENERATED from AuK's src/auk/infer/pe.config.yaml by tools/community_models/auk_task_table.py
//
// These are the instruction templates the model was TRAINED on. AuK's shipped inference
// path reaches them through pe.py, a 2052-line prompt enhancer that asks another LLM to
// classify a user's free-form request into one of these task types and fill its slots.
//
// This table skips that entirely: the caller names the task and the parameters, and the
// exact template is rendered. That trades the natural-language surface for precision --
// "Raise the pitch by 4 semitones." instead of hoping a classifier reads "make it
// higher" the way you meant -- and removes a second LLM from the inference path.
//
// ⚠ 5 of the 16 task types have NO English template, only Chinese: enhance_speech,
// separate_speech, improve_quality, whisper_edit, accent_edit. They are rendered in
// Chinese because that is the only phrasing the model has seen for them.

#include "engine/models/auk/task.h"

#include <sstream>
#include <stdexcept>

namespace engine::models::auk {
namespace {

struct TemplateRow {
    const char * task;
    const char * variant;
    const char * language;
    const char * text;
};

constexpr TemplateRow kTemplates[] = {
    {"content_edit", "insert_before", "en", "Add '{text}' before '{anchor}'."},
    {"content_edit", "insert_after", "en", "Add '{text}' after '{anchor}'."},
    {"content_edit", "delete", "en", "Remove '{target}'."},
    {"content_edit", "delete_before", "en", "Remove '{target}' before '{anchor}'."},
    {"content_edit", "delete_after", "en", "Remove '{target}' after '{anchor}'."},
    {"content_edit", "replace", "en", "Replace '{orig}' with '{new}'."},
    {"content_edit", "insert_before", "zh", "在‘{anchor}’前面加上‘{text}’"},
    {"content_edit", "insert_after", "zh", "在‘{anchor}’后面加上‘{text}’"},
    {"content_edit", "delete", "zh", "删掉‘{target}’"},
    {"content_edit", "delete_before", "zh", "删掉‘{anchor}’前的‘{target}’"},
    {"content_edit", "delete_after", "zh", "删掉‘{anchor}’后的‘{target}’"},
    {"content_edit", "replace", "zh", "把‘{orig}’改成‘{new}’"},
    {"speed_edit", "", "en", "Adjust the speech speed to {speed_multiplier}x."},
    {"speed_edit", "", "zh", "将语速调整为{speed_multiplier}倍。"},
    {"volume_edit", "increase", "en", "Increase the volume by {gain_db} dB."},
    {"volume_edit", "decrease", "en", "Decrease the volume by {gain_db} dB."},
    {"volume_edit", "increase", "zh", "将音量升高{gain_db}分贝。"},
    {"volume_edit", "decrease", "zh", "将音量降低{gain_db}分贝。"},
    {"pitch_edit", "increase", "en", "Raise the pitch by {semitones} semitones."},
    {"pitch_edit", "decrease", "en", "Lower the pitch by {semitones} semitones."},
    {"pitch_edit", "increase", "zh", "将音调升高{semitones}个半音。"},
    {"pitch_edit", "decrease", "zh", "将音调降低{semitones}个半音。"},
    {"emotion_edit", "", "en", "Change the emotion to {emotion}."},
    {"emotion_edit", "", "zh", "将情感转变为{emotion}。"},
    {"enhance_speech", "", "zh", "请清理这段输入语音，不做说话人删除，并去除房间混响，输出干净的人声结果。"},
    {"separate_speech", "by_content", "zh", "请只保留说'{text}'的人，去掉其他说话人，输出等长纯净人声。"},
    {"separate_speech", "by_order", "zh", "这段音频中只保留第{n_zh}个开始说话的人对应的语音，去掉其余说话人。"},
    {"extract_vocals", "singing_only", "en", "Keep only the singing voice, drop everything else."},
    {"extract_vocals", "all_human_voices", "en", "Keep all human voices, speech and singing alike, drop everything else."},
    {"extract_vocals", "singing_only", "zh", "请只保留歌声，其余声音都去掉。"},
    {"extract_vocals", "all_human_voices", "zh", "请保留所有人声，说话和歌唱都算，其余声音都去掉。"},
    {"improve_quality", "bandwidth_extension", "zh", "请对这段语音做超分辨率/带宽扩展处理，恢复被削掉的高频成分，输出宽带纯净人声。"},
    {"improve_quality", "remove_effect", "zh", "请消除这段音频的{effect}音色，这段音频带有混响，请恢复成无混响的干声，输出自然清晰的人声。"},
    {"instruct_tts", "", "en", "Based on the following description: \"{style_desc}\", generate speech content \"{text}\"."},
    {"instruct_tts", "", "zh", "请基于下面的描述: \"{style_desc}\",生成语音内容\"{text}\"."},
    {"zero_shot_tts", "", "en", "Say the following with the same voice: \"{text}\""},
    {"zero_shot_tts", "", "zh", "Say the following with the same voice: \"{text}\""},
    {"voice_edit", "", "en", "Keep the spoken content unchanged and change the timbre to: \"{timbre_desc}\"."},
    {"voice_edit", "", "zh", "请将这段音频的音色修改为符合以下描述的声音：“{timbre_desc}”。"},
    {"nonverbal_edit", "delete", "en", "Remove all the {sound} from the audio."},
    {"nonverbal_edit", "delete", "zh", "删除音频中所有的{sound}。"},
    {"nonverbal_edit", "add_after", "zh", "在“{anchor}”后增加{sound}。"},
    {"nonverbal_edit", "add_before", "zh", "在“{anchor}”前增加{sound}。"},
    {"nonverbal_edit", "add_head", "zh", "在语音开头增加{sound}。"},
    {"nonverbal_edit", "add_tail", "zh", "在语音结尾增加{sound}。"},
    {"whisper_edit", "to_normal", "zh", "把这段耳语转换成正常说话的声音。"},
    {"whisper_edit", "to_whisper", "zh", "用小声耳语的方式把这段话说出来。"},
    {"vocal_edit", "", "en", "Change \"{orig}\" to \"{new}\" in the vocal recording."},
    {"vocal_edit", "", "zh", "把这段歌词中的“{orig}”改成“{new}”。"},
    {"accent_edit", "", "zh", "请去掉这段语音里的方言口音，保持说话人音色一致。"},
};

constexpr const char * kEmotions[] = {"happy", "angry", "sad", "fearful", "surprised", "disgusted", "calm", "excited"};

// Tasks that generate rather than edit: they take no source audio. Everything else is
// an edit of something, and refusing to run it without audio is better than returning
// four seconds of invented speech.
bool task_needs_audio(const std::string & task) {
    return task != "instruct_tts";
}

std::string fill_slots(const std::string & text, const std::map<std::string, std::string> & params) {
    std::string out;
    out.reserve(text.size());
    for (size_t index = 0; index < text.size();) {
        if (text[index] != '{') { out += text[index++]; continue; }
        const size_t close = text.find('}', index);
        if (close == std::string::npos) { out += text[index++]; continue; }
        const std::string name = text.substr(index + 1, close - index - 1);
        const auto found = params.find(name);
        if (found == params.end()) {
            throw std::runtime_error("AuK task template needs a value for {" + name + "}");
        }
        out += found->second;
        index = close + 1;
    }
    return out;
}

}  // namespace

std::vector<std::string> task_variants(const std::string & task);

RenderedTask render_task(const TaskRequest & request) {
    // Three questions in order, each with its own answer: does the task exist, does the
    // variant, and is it available in the language asked for. Collapsing them produced
    // "needs a variant" for a task that has none, which sent the reader looking for a
    // variant list that does not exist.
    bool task_exists = false;
    std::vector<const TemplateRow *> candidates;
    for (const auto & row : kTemplates) {
        if (request.task != row.task) continue;
        task_exists = true;
        if (request.variant != row.variant) continue;
        candidates.push_back(&row);
    }
    if (!task_exists) {
        throw std::runtime_error("unknown AuK task: " + request.task);
    }
    if (candidates.empty()) {
        const auto variants = task_variants(request.task);
        std::string message = "AuK task " + request.task;
        if (variants.empty()) {
            message += " takes no variant, but '" + request.variant + "' was given";
        } else {
            message += " needs one of these variants:";
            for (const auto & variant : variants) message += " " + variant;
        }
        throw std::runtime_error(message);
    }

    const TemplateRow * chosen = candidates.front();
    for (const auto * row : candidates) {
        if (request.language == row->language) { chosen = row; break; }
    }

    RenderedTask out;
    out.instruction = fill_slots(chosen->text, request.params);
    out.language = chosen->language;
    out.needs_audio = task_needs_audio(request.task);
    return out;
}

std::vector<std::string> task_names() {
    std::vector<std::string> names;
    for (const auto & row : kTemplates) {
        if (names.empty() || names.back() != row.task) {
            bool seen = false;
            for (const auto & name : names) seen |= name == row.task;
            if (!seen) names.emplace_back(row.task);
        }
    }
    return names;
}

std::vector<std::string> task_variants(const std::string & task) {
    std::vector<std::string> variants;
    for (const auto & row : kTemplates) {
        if (task != row.task || std::string(row.variant).empty()) continue;
        bool seen = false;
        for (const auto & variant : variants) seen |= variant == row.variant;
        if (!seen) variants.emplace_back(row.variant);
    }
    return variants;
}

std::vector<std::string> task_languages(const std::string & task) {
    std::vector<std::string> languages;
    for (const auto & row : kTemplates) {
        if (task != row.task) continue;
        bool seen = false;
        for (const auto & language : languages) seen |= language == row.language;
        if (!seen) languages.emplace_back(row.language);
    }
    return languages;
}

std::vector<std::string> task_parameters(const std::string & task) {
    std::vector<std::string> names;
    for (const auto & row : kTemplates) {
        if (task != row.task) continue;
        const std::string text = row.text;
        for (size_t index = 0; index < text.size();) {
            if (text[index] != '{') { ++index; continue; }
            const size_t close = text.find('}', index);
            if (close == std::string::npos) break;
            const std::string name = text.substr(index + 1, close - index - 1);
            bool seen = false;
            for (const auto & existing : names) seen |= existing == name;
            if (!seen) names.push_back(name);
            index = close + 1;
        }
    }
    return names;
}

std::vector<std::string> emotion_labels() {
    return std::vector<std::string>(std::begin(kEmotions), std::end(kEmotions));
}

}  // namespace engine::models::auk
