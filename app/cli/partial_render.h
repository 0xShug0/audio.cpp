#pragma once

#include <string>

namespace minitts::cli {

// Partials are the text newly decoded since the last one, so they concatenate into the transcript.
// A terminal can therefore just append them and the transcript scrolls like ordinary output, while
// redirected output keeps the labelled line-per-update form so pipes and logs stay parseable.
//
// Returns the bytes to write rather than taking a stream, so the caller decides when to flush and
// the behaviour can be tested without a terminal.
class PartialTextRenderer {
public:
    explicit PartialTextRenderer(bool interactive) : interactive_(interactive) {}

    std::string render(const std::string & text) {
        if (!interactive_) {
            return "partial_text=" + text + "\n";
        }
        wrote_ = wrote_ || !text.empty();
        return text;
    }

    // Terminates the transcript line so following output starts on a fresh row. Empty when no
    // line was left open, so a run that produced no partials does not print a stray blank line.
    std::string finish() {
        if (!interactive_ || !wrote_) {
            return {};
        }
        wrote_ = false;
        return "\n";
    }

private:
    bool interactive_ = false;
    bool wrote_ = false;
};

}  // namespace minitts::cli
