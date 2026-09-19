#pragma once

// AuK's task types, addressed directly.
//
// The model's own inference path reaches these through pe.py: a 2052-line prompt
// enhancer that asks a second LLM to classify a free-form request into one of 16 task
// types and fill its parameters. That is a reasonable product surface and a poor API --
// the behaviour depends on a classifier's reading of your phrasing, so the same request
// worded twice is two different requests.
//
// This names the task and its parameters instead. "Raise the pitch by 4 semitones."
// is rendered from pitch_edit{semitones=4}, not hoped for from "make it higher".
//
// ⚠ 5 of the 16 types have no English template, only Chinese. Those render in Chinese,
// because an English paraphrase of a Chinese-only instruction is a prompt the model has
// never seen rather than a translation of one it has.

#include <map>
#include <string>
#include <vector>

namespace engine::models::auk {

struct TaskRequest {
    std::string task;                            // e.g. "pitch_edit"
    std::string variant;                         // e.g. "increase"; empty when the task has one form
    std::map<std::string, std::string> params;   // e.g. {{"semitones", "4"}}
    std::string language = "en";                 // falls back to the only language available
};

struct RenderedTask {
    std::string instruction;
    std::string language;    // what it was ACTUALLY rendered in, which may not be what was asked
    bool needs_audio = true;
};

// Renders the canonical template with its slots filled. Throws when the task or variant
// is unknown, or when a slot has no value -- an unfilled "{semitones}" reaching the
// model is a silent quality failure, so it is refused here instead.
RenderedTask render_task(const TaskRequest & request);

std::vector<std::string> task_names();
std::vector<std::string> task_variants(const std::string & task);
std::vector<std::string> task_parameters(const std::string & task);
std::vector<std::string> task_languages(const std::string & task);
std::vector<std::string> emotion_labels();

}  // namespace engine::models::auk
