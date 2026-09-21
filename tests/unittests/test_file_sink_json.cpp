#include "../../app/workflow/file_sink.h"

#include "engine/framework/io/json.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_word_timestamp_json_escapes_control_characters() {
    engine::runtime::WordTimestamp word;
    word.span.start_sample = 1;
    word.span.end_sample = 2;
    word.word = "line\nfeed\rreturn\ttab\x01" "control\\slash\"quote";

    const std::string json = minitts::app::word_timestamps_to_json({word});
    (void) engine::io::json::parse(json);

    require(json.find("\\n") != std::string::npos, "newline must be escaped");
    require(json.find("\\r") != std::string::npos, "carriage return must be escaped");
    require(json.find("\\t") != std::string::npos, "tab must be escaped");
    require(json.find("\\u0001") != std::string::npos, "control byte must be escaped");
    require(json.find("line\n") == std::string::npos, "raw newline must not appear in JSON");
    require(json.find("line\\nfeed") != std::string::npos, "escaped word content must be present");
}

}  // namespace

int main() {
    try {
        test_word_timestamp_json_escapes_control_characters();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "file_sink_json_test passed\n";
    return 0;
}
