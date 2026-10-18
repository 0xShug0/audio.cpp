#include "engine/framework/runtime/partial_text.h"

#include "test_assert.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

using engine::runtime::PartialTextPublisher;
using engine::runtime::transcript_publishable_end;
using engine::test::require_eq;

// What a consumer assembles by appending every delta, which is the contract:
// partials concatenate into the transcript.
std::string appended(const std::vector<std::string> & decodes) {
    PartialTextPublisher publisher;
    std::string client;
    for (const auto & decode : decodes) {
        client += publisher.publish(decode);
    }
    return client;
}

void test_growing_transcript_yields_increments() {
    PartialTextPublisher publisher;
    require_eq(publisher.publish("Some call me nat"), std::string("Some call me nat"), "first");
    require_eq(publisher.publish("Some call me nature. Others"), std::string("ure. Others"), "second");
    require_eq(publisher.publish("Some call me nature. Others"), std::string(""), "unchanged");
}

// The reason this is shared rather than reimplemented per family: a tokenizer
// falls back to bytes for text its vocabulary does not cover, so a decode can
// stop part way through a character. Publishing that puts half a code point on
// the wire, where it reaches a JSON encoder as invalid UTF-8.
void test_partial_character_is_held_until_complete() {
    const std::string cjk = "\xE4\xB8\x80";  // U+4E00, three bytes
    PartialTextPublisher publisher;
    require_eq(publisher.publish("ab" + cjk.substr(0, 1)), std::string("ab"), "lead byte held");
    require_eq(publisher.publish("ab" + cjk.substr(0, 2)), std::string(""), "still incomplete");
    require_eq(publisher.publish("ab" + cjk), cjk, "released whole");
    require_eq(appended({"ab" + cjk.substr(0, 1), "ab" + cjk.substr(0, 2), "ab" + cjk}),
               "ab" + cjk, "assembled");
}

void test_two_byte_characters_are_held_too() {
    const std::string ru = "\xD0\x9F\xD1\x80\xD0\xB8";  // При
    require_eq(appended({ru.substr(0, 1), ru.substr(0, 3), ru.substr(0, 5), ru}), ru, "cyrillic");
}

void test_four_byte_character_is_held_until_complete() {
    const std::string emoji = "\xF0\x9F\x8E\xB5";  // U+1F3B5
    PartialTextPublisher publisher;
    require_eq(publisher.publish(emoji.substr(0, 3)), std::string(""), "incomplete");
    require_eq(publisher.publish(emoji), emoji, "released whole");
}

// A revision cannot be retracted -- the delta has already gone out -- but the
// next one must still start on a character boundary rather than inside one.
void test_revision_resumes_on_a_character_boundary() {
    const std::string cjk = "\xE4\xB8\x80";
    PartialTextPublisher publisher;
    require_eq(publisher.publish("ab" + cjk), "ab" + cjk, "published");
    // Same first byte of the character, different continuation.
    const std::string revised = "ab\xE4\xB8\x81";
    const std::string delta = publisher.publish(revised);
    require_eq(delta, std::string("\xE4\xB8\x81"), "whole character re-sent");
}

// A decode that truncates mid-character must not un-publish the character it
// cut: the decode that restores it would then send it twice, and a consumer
// that appends every delta would show it twice.
void test_truncated_decode_does_not_resend() {
    const std::string cjk = "\xE4\xB8\x80";
    PartialTextPublisher publisher;
    require_eq(publisher.publish("ab" + cjk), "ab" + cjk, "published whole");
    require_eq(publisher.publish("ab" + cjk.substr(0, 2)), std::string(""), "truncated");
    require_eq(publisher.publish("ab" + cjk), std::string(""), "not resent");
    require_eq(appended({"ab" + cjk, "ab" + cjk.substr(0, 2), "ab" + cjk}),
               "ab" + cjk, "consumer sees it once");
}

// The agreement check is bounded, so state what that buys and what it gives up.
// A revision near the end -- where a streaming decode actually revises -- is
// seen, and the delta resumes from it.
void test_revision_within_the_window_is_seen() {
    PartialTextPublisher publisher;
    require_eq(publisher.publish("the quick brown fox"), std::string("the quick brown fox"), "first");
    const std::string delta = publisher.publish("the quick brown dog");
    require_eq(delta, std::string("dog"), "resumes at the divergence");
}

// A revision further back than the window is not seen, and cannot be: the text
// it would correct has already gone out and nothing can retract it. The
// publisher keeps going forward instead of re-sending a transcript the consumer
// cannot un-append.
void test_revision_behind_the_window_does_not_resend_history() {
    PartialTextPublisher publisher;
    std::string transcript(600, 'a');
    require_eq(publisher.publish(transcript).size(), transcript.size(), "first");
    std::string revised = transcript;
    revised[0] = 'b';           // far behind the 256-byte window
    revised += "tail";
    require_eq(publisher.publish(revised), std::string("tail"), "only the new tail");
}

void test_shrinking_transcript_publishes_nothing() {
    PartialTextPublisher publisher;
    require_eq(publisher.publish("abcdef"), std::string("abcdef"), "first");
    require_eq(publisher.publish("abc"), std::string(""), "shrunk");
    // And the text it dropped is not re-sent when it comes back, because the
    // consumer was never told to remove it.
    require_eq(publisher.publish("abcdef"), std::string(""), "not resent");
}

void test_reset_forgets_the_published_prefix() {
    PartialTextPublisher publisher;
    require_eq(publisher.publish("hello"), std::string("hello"), "first");
    publisher.reset();
    require_eq(publisher.published(), std::string(""), "cleared");
    require_eq(publisher.publish("hello"), std::string("hello"), "republished after reset");
}

void test_empty_and_ascii_edges() {
    PartialTextPublisher publisher;
    require_eq(publisher.publish(""), std::string(""), "empty");
    require_eq(publisher.publish("a"), std::string("a"), "single byte");
}

// voxtral_realtime keeps its own byte offset and calls this directly, so pin it
// down on its own: the emoji split below is how Tekken byte tokens emit one.
void test_publishable_end_stops_before_an_incomplete_tail() {
    const std::string emoji = "\xF0\x9F\x98\x80";  // U+1F600, four bytes
    for (std::size_t n = 1; n < emoji.size(); ++n) {
        require_eq(transcript_publishable_end("hi " + emoji.substr(0, n)), std::size_t{3},
                   "incomplete emoji held");
    }
    require_eq(transcript_publishable_end("hi " + emoji), std::size_t{7}, "complete emoji");
    require_eq(transcript_publishable_end("\xC3"), std::size_t{0}, "lone two-byte lead");
    require_eq(transcript_publishable_end("\xC3\xA9"), std::size_t{2}, "complete two-byte");
    require_eq(transcript_publishable_end("abc"), std::size_t{3}, "ascii");
    require_eq(transcript_publishable_end(""), std::size_t{0}, "empty");
}

}  // namespace

int main() {
    try {
        test_growing_transcript_yields_increments();
        test_partial_character_is_held_until_complete();
        test_two_byte_characters_are_held_too();
        test_four_byte_character_is_held_until_complete();
        test_revision_resumes_on_a_character_boundary();
        test_truncated_decode_does_not_resend();
        test_revision_within_the_window_is_seen();
        test_revision_behind_the_window_does_not_resend_history();
        test_shrinking_transcript_publishes_nothing();
        test_reset_forgets_the_published_prefix();
        test_empty_and_ascii_edges();
        test_publishable_end_stops_before_an_incomplete_tail();
        std::cout << "partial_text_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "partial_text_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
