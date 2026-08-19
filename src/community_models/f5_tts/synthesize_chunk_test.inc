static bool is_break_byte_here(const std::string & text, size_t i, size_t * len) {
    const unsigned char c = static_cast<unsigned char>(text[i]);
    if (c == '.' || c == '!' || c == '?' || c == '\n') {
        *len = 1;
        return true;
    }
    // Arabic semicolon U+061B (D8 9B) and comma U+060C (D8 8C)
    if (c == 0xD8 && i + 1 < text.size()) {
        const unsigned char c2 = static_cast<unsigned char>(text[i + 1]);
        if (c2 == 0x9B || c2 == 0x8C) {
            *len = 2;
            return true;
        }
    }
    return false;
}

static size_t utf8_char_count(const std::string & s) {
    size_t n = 0;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        i += c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : 4;
        ++n;
    }
    return n;
}

// Split text so each chunk has at most max_chars UTF-8 characters,
// preferring sentence/clause boundaries.
std::vector<std::string> chunk_text(const std::string & text, size_t max_chars) {
    std::vector<std::string> chunks;
    if (utf8_char_count(text) <= max_chars) {
        chunks.push_back(text);
        return chunks;
    }
    // collect sentence pieces [start, end) breaking AFTER break chars
    std::vector<std::pair<size_t, size_t>> pieces;
    size_t piece_start = 0;
    for (size_t i = 0; i < text.size();) {
        size_t blen = 0;
        if (is_break_byte_here(text, i, &blen)) {
            pieces.emplace_back(piece_start, i + blen);
            piece_start = i + blen;
            i += blen;
            continue;
        }
        const unsigned char c = static_cast<unsigned char>(text[i]);
        i += c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : 4;
    }
    if (piece_start < text.size()) {
        pieces.emplace_back(piece_start, text.size());
    }
    // greedy pack pieces into chunks <= max_chars chars
    for (const auto & [ps, pe] : pieces) {
        const std::string piece = text.substr(ps, pe - ps);
        const size_t pc = utf8_char_count(piece);
        if (pc > max_chars) {
            // oversize sentence: hard-split at character boundaries
            size_t cs = ps;
            while (cs < pe) {
                size_t cnt = 0, ce = cs;
                while (ce < pe && cnt < max_chars) {
                    const unsigned char c = static_cast<unsigned char>(text[ce]);
                    ce += c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : 4;
                    ++cnt;
                }
                chunks.push_back(text.substr(cs, ce - cs));
                cs = ce;
            }
            continue;
        }
        if (!chunks.empty()) {
            const std::string & prev = chunks.back();
            if (utf8_char_count(prev) + pc <= max_chars) {
                chunks.back() = prev + piece;
                continue;
            }
        }
        chunks.push_back(piece);
    }
    return chunks;
}

