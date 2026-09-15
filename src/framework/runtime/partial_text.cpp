#include "engine/framework/runtime/partial_text.h"

namespace engine::runtime {

std::size_t transcript_common_prefix(const std::string & lhs, const std::string & rhs) {
    std::size_t size = 0;
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

std::size_t transcript_publishable_end(const std::string & text) {
    std::size_t lead = text.size();
    while (lead > 0 && (static_cast<unsigned char>(text[lead - 1]) & 0xC0) == 0x80) {
        --lead;
    }
    if (lead == 0) {
        // All continuation bytes, or empty: nothing to anchor a decision on.
        return text.size();
    }
    --lead;
    const auto first = static_cast<unsigned char>(text[lead]);
    std::size_t needed = 0;
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
    const std::size_t publishable = transcript_publishable_end(transcript);
    const std::size_t already = transcript_common_prefix(published_, transcript);
    if (already >= publishable) {
        // Nothing new that is whole. Leave `published_` alone rather than
        // shortening it to this decode: a decode that truncates mid-character
        // would otherwise un-publish the character it cut, and the decode that
        // restores it would send it a second time.
        return {};
    }
    std::string delta = transcript.substr(already, publishable - already);
    // What has actually gone out, so a held-back character is reconsidered
    // against the update that completes it rather than skipped.
    published_.assign(transcript, 0, publishable);
    return delta;
}

}  // namespace engine::runtime
