#include "../../app/cli/partial_render.h"

#include "test_assert.h"

#include <exception>
#include <iostream>
#include <string>

namespace {

using engine::test::require_eq;
using minitts::cli::PartialTextRenderer;

// Partials carry only the text decoded since the last one, so a terminal appends them unlabelled
// and the transcript reads as running text.
void test_terminal_appends_partials_verbatim() {
    PartialTextRenderer renderer(true);
    require_eq(renderer.render(" Some"), std::string(" Some"), "first partial");
    require_eq(renderer.render(" call"), std::string(" call"), "second partial");
    require_eq(renderer.render(" me"), std::string(" me"), "third partial");
}

// Redirected output keeps the labelled line-per-update format so pipes and logs stay parseable.
void test_redirected_output_keeps_labelled_lines() {
    PartialTextRenderer renderer(false);
    require_eq(renderer.render(" Some"), std::string("partial_text= Some\n"), "first piped partial");
    require_eq(renderer.render(" call"), std::string("partial_text= call\n"), "second piped partial");
}

void test_finish_closes_only_an_open_line() {
    PartialTextRenderer opened(true);
    opened.render(" Some");
    require_eq(opened.finish(), std::string("\n"), "open line is closed");
    require_eq(opened.finish(), std::string(""), "line is closed only once");

    PartialTextRenderer untouched(true);
    require_eq(untouched.finish(), std::string(""), "a run with no partials prints no blank line");

    PartialTextRenderer piped(false);
    piped.render(" Some");
    require_eq(piped.finish(), std::string(""), "piped output already ends its lines");
}

}  // namespace

int main() {
    try {
        test_terminal_appends_partials_verbatim();
        test_redirected_output_keeps_labelled_lines();
        test_finish_closes_only_an_open_line();
    } catch (const std::exception & error) {
        std::cerr << "partial_text_render_test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "partial_text_render_test passed\n";
    return 0;
}
