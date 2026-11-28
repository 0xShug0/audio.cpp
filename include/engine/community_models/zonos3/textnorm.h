#pragma once

#include <string>
#include <string_view>

namespace engine::models::zonos3 {

// The ZONOS3 language code for a request language: en_us, en_gb, fr_fr, de, es, it, pt_br, ja, cmn, ko, or the
// aliases en, fr, pt, zh. Throws InvalidRequestError otherwise.
std::string zonos3_language(std::string_view language);

// Spoken-form normalization (zyphra-textnorm) when enabled and linked; line breaks are collapsed either way.
// enabled=false returns the text unchanged.
std::string zonos3_normalize_text(const std::string & text, const std::string & language, bool enabled);

// Python \s*[^\S ]\s* -> " ": every whitespace run holding a non-space whitespace character becomes one space.
std::string zonos3_collapse_line_breaks(const std::string & text);

// Whether this build links zyphra-textnorm (AUDIOCPP_ZONOS3_TEXTNORM).
bool zonos3_textnorm_linked();

}  // namespace engine::models::zonos3
