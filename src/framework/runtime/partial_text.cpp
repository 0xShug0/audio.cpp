#include "engine/framework/runtime/partial_text.h"

#include <algorithm>
#include <cstring>
#include <iterator>

namespace engine::runtime {

namespace {
// The well-formed UTF-8 byte sequences (Unicode Table 3-7) by the range of
// their first byte: the range of the second byte, and the length. Any further
// bytes are 80..BF.
struct Utf8Sequence {
    unsigned char first_low;
    unsigned char first_high;
    unsigned char second_low;
    unsigned char second_high;
    size_t length;
};

constexpr Utf8Sequence kWellFormedUtf8[] = {
    {0x00, 0x7F, 0x00, 0x00, 1}, {0xC2, 0xDF, 0x80, 0xBF, 2}, {0xE0, 0xE0, 0xA0, 0xBF, 3},
    {0xE1, 0xEC, 0x80, 0xBF, 3}, {0xED, 0xED, 0x80, 0x9F, 3}, {0xEE, 0xEF, 0x80, 0xBF, 3},
    {0xF0, 0xF0, 0x90, 0xBF, 4}, {0xF1, 0xF3, 0x80, 0xBF, 4}, {0xF4, 0xF4, 0x80, 0x8F, 4},
};

}  // namespace

std::string consume_utf8_prefix(std::string & bytes) {
    std::string text;
    size_t pos = 0;
    while (pos < bytes.size()) {
        const auto first = static_cast<unsigned char>(bytes[pos]);
        const auto * sequence = std::find_if(std::begin(kWellFormedUtf8), std::end(kWellFormedUtf8),
            [&](const Utf8Sequence & row) { return first >= row.first_low && first <= row.first_high; });
        const size_t needed = sequence == std::end(kWellFormedUtf8) ? 0 : sequence->length;

        // How much of a well-formed sequence starts here: at least this byte,
        // which is the maximal subpart when no sequence starts with it.
        size_t length = 1;
        while (length < needed && pos + length < bytes.size()) {
            const auto byte = static_cast<unsigned char>(bytes[pos + length]);
            const bool fits = length == 1 ? byte >= sequence->second_low && byte <= sequence->second_high : (byte & 0xC0) == 0x80;
            if (!fits) {
                break;
            }

            ++length;
        }

        if (length == needed) {
            text.append(bytes, pos, length);
        } else if (needed != 0 && pos + length == bytes.size()) {
            break;  // the next bytes may finish it
        } else {
            text += "\xEF\xBF\xBD";
        }

        pos += length;
    }

    bytes.erase(0, pos);
    return text;
}


std::size_t transcript_common_prefix(const std::string & lhs, const std::string & rhs) {
    const std::size_t limit = std::min(lhs.size(), rhs.size());
    // The overwhelmingly common case is that the update only extended what was
    // published, so test that wholesale before walking byte by byte: a single
    // compare over the shared span instead of a per-byte loop.
    std::size_t size = 0;
    if (limit > 0 && std::memcmp(lhs.data(), rhs.data(), limit) == 0) {
        size = limit;
    } else {
        while (size < limit && lhs[size] == rhs[size]) {
            ++size;
        }
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

// How much of `published_` the update still agrees with.
//
// Checked over a bounded window rather than the whole transcript. A decode
// revises what it has just heard, not text from minutes ago -- and a delta that
// has gone out cannot be retracted anyway, so a divergence behind the window is
// not something this could act on even if it found it. Bounding the check keeps
// publish() O(1) in the length of the transcript instead of O(n), which is what
// makes it as cheap as the byte offset it replaces on a long session.
std::size_t agreed_prefix(const std::string & published, const std::string & transcript) {
    constexpr std::size_t kWindow = 256;
    if (transcript.size() >= published.size()) {
        const std::size_t start = published.size() > kWindow ? published.size() - kWindow : 0;
        if (std::memcmp(published.data() + start, transcript.data() + start,
                        published.size() - start) == 0) {
            return published.size();
        }
    }
    // Disagreed inside the window, or the transcript shrank: fall back to the
    // exact answer, which is rare enough to afford.
    return transcript_common_prefix(published, transcript);
}

std::string PartialTextPublisher::publish(const std::string & transcript) {
    const std::size_t publishable = transcript_publishable_end(transcript);
    const std::size_t already = agreed_prefix(published_, transcript);
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
    //
    // Appended rather than reassigned in the common case. A streaming
    // transcript is rebuilt from scratch on every update, so assigning the
    // whole thing here copies the entire transcript once per update -- O(n^2)
    // over a session, which is a real cost on a long one and the reason a
    // byte-offset is cheaper. When the update only extended what was already
    // published, appending the delta is O(delta) instead.
    if (already == published_.size()) {
        published_.append(delta);
    } else {
        published_.assign(transcript, 0, publishable);
    }
    return delta;
}

}  // namespace engine::runtime
