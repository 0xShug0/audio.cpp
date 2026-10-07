// YuE2 takes no input artifacts. A request that carries one is the caller's
// to fix, so it is turned away with InvalidRequestError, which the server
// answers with 400, before anything else in it is read.
#include "engine/models/yue2/request.h"

#include "engine/framework/runtime/errors.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    namespace runtime = engine::runtime;
    runtime::TaskRequest request;
    request.input_artifacts.push_back(runtime::make_text_artifact(runtime::ArtifactKind::Custom, "example.state", "x"));
    try {
        (void) engine::models::yue2::parse_yue2_request(request, {});
    } catch (const runtime::InvalidRequestError & error) {
        if (std::string(error.what()) != "Yue2 does not consume input artifacts") {
            std::cerr << "input artifacts turned away with \"" << error.what() << "\"\n";
            return 1;
        }
        std::cout << "yue2 request test passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "input artifacts were not an InvalidRequestError: " << error.what() << '\n';
        return 1;
    }
    std::cerr << "input artifacts were accepted\n";
    return 1;
}
