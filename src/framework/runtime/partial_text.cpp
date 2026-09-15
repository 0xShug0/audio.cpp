#include "engine/framework/runtime/partial_text.h"

namespace engine::runtime {

size_t transcript_common_prefix(const std::string & lhs, const std::string & rhs) {
    size_t size = 0;
    while (size < lhs.size() && size < rhs.size() && lhs[size] == rhs[size]) {
        ++size;
    }
    // A divergence inside a character would otherwise start the next delta on a
    // continuation byte.
    while (size > 0 && (static_cast<unsigned char>(rhs[size]) & 0xC0) == 0x80) {
        --size;
    }
    return size;
}

size_t transcript_publishable_end(const std::string & text) {
    size_t lead = text.size();
    while (lead > 0 && (static_cast<unsigned char>(text[lead - 1]) & 0xC0) == 0x80) {
        --lead;
    }
    if (lead == 0) {
        // All continuation bytes, or empty: nothing to anchor a decision on.
        return text.size();
    }
    --lead;
    const auto first = static_cast<unsigned char>(text[lead]);
    size_t needed = 0;
    if ((first & 0x80) == 0x00) {
        needed = 1;
    } else if ((first & 0xE0) == 0xC0) {
        needed = 2;
    } else if ((first & 0xF0) == 0xE0) {
        needed = 3;
    } else if ((first & 0xF8) == 0xF0) {
        needed = 4;
    } else {
        // Not a lead byte at all, so this is not text this can reason about.
        // Holding bytes back would lose them; publish and let the consumer see.
        return text.size();
    }
    return (text.size() - lead) < needed ? lead : text.size();
}

std::string PartialTextPublisher::publish(const std::string & transcript) {
    const size_t publishable = transcript_publishable_end(transcript);
    const size_t already = transcript_common_prefix(published_, transcript);
    std::string delta;
    if (already < publishable) {
        delta = transcript.substr(already, publishable - already);
    }
    // What actually went out, so a held-back character is reconsidered against
    // the update that completes it rather than skipped.
    published_.assign(transcript, 0, publishable);
    return delta;
}

}  // namespace engine::runtime
