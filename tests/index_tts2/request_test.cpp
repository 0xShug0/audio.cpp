#include "engine/models/index_tts2/request.h"
#include "../unittests/test_assert.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    namespace runtime = engine::runtime;
    using engine::models::index_tts2::parse_index_tts2_request;
    using engine::test::require;
    using engine::test::require_eq;
    try {
        runtime::TaskRequest request;
        request.text_input = runtime::Transcript{"There are 12 items.", "en"};
        request.voice = runtime::VoiceCondition{
            runtime::VoiceReference{runtime::AudioBuffer{24000, 1, {0.25F}}, std::nullopt}, std::nullopt};
        const auto defaults = parse_index_tts2_request(request);
        require_eq(defaults.text_chunk_tokens, 120, "unchanged default segment budget");
        require(defaults.text_normalization, "normalization stays enabled by default");
        request.options["text_chunk_tokens"] = "60";
        request.options["text_normalization"] = "false";
        const auto custom = parse_index_tts2_request(request);
        require_eq(custom.text_chunk_tokens, 60, "custom segment budget");
        require(!custom.text_normalization, "explicit normalization bypass");
        for (const auto & value : {"0", "-1", "601", "1.5", "invalid"}) {
            request.options["text_chunk_tokens"] = value;
            bool rejected = false;
            try { (void)parse_index_tts2_request(request); }
            catch (const std::exception &) { rejected = true; }
            require(rejected, std::string("invalid segment budget: ") + value);
        }
        request.options["text_chunk_tokens"] = "600";
        require_eq(parse_index_tts2_request(request).text_chunk_tokens, 600, "upper segment budget");
        request.options["text_normalization"] = "invalid";
        bool rejected = false;
        try { (void)parse_index_tts2_request(request); }
        catch (const std::exception &) { rejected = true; }
        require(rejected, "invalid normalization boolean");
        std::cout << "index_tts2 request test passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
