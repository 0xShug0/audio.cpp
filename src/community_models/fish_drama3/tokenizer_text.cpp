#include "engine/community_models/fish_drama3/tokenizer_text.h"

#include "engine/framework/runtime/errors.h"

#include <utf8proc.h>

#include <cstdlib>
#include <limits>
#include <memory>

namespace engine::models::fish_drama3 {

Drama3TextTokenizer::Drama3TextTokenizer(const assets::ResourceBundle & resources)
    : tokenizer_(tokenizers::LlamaBpeTokenizerSpec{
          {},
          {},
          resources.require_file("tokenizer_config"),
          resources.require_file("tokenizer_json"),
          tokenizers::LlamaBpePreTokenizer::Qwen35,
      }) {}

std::vector<int32_t> Drama3TextTokenizer::encode(const std::string & text) const {
    if (text.empty()) return tokenizer_.encode(text, true);
    if (text.size() > static_cast<size_t>(std::numeric_limits<utf8proc_ssize_t>::max())) {
        throw runtime::InvalidRequestError("Drama 3 text is too large to normalize");
    }
    // The official tokenizer applies NFC before Qwen3.5 token splitting.
    utf8proc_uint8_t * output = nullptr;
    const auto size = utf8proc_map(
        reinterpret_cast<const utf8proc_uint8_t *>(text.data()),
        static_cast<utf8proc_ssize_t>(text.size()), &output,
        static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
    const std::unique_ptr<utf8proc_uint8_t, decltype(&std::free)> normalized(output, &std::free);
    if (size < 0) {
        throw runtime::InvalidRequestError(
            "Drama 3 text normalization: " + std::string(utf8proc_errmsg(size)));
    }
    return tokenizer_.encode(
        std::string(reinterpret_cast<const char *>(normalized.get()), static_cast<size_t>(size)), true);
}

}  // namespace engine::models::fish_drama3
