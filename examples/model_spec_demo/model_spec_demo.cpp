#include "engine/framework/model_spec/package.h"

#include <exception>
#include <iostream>

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::cerr << "usage: model_spec_demo <spec.json> <model-dir>\n";
        return 2;
    }
    try {
        const std::filesystem::path spec_path(argv[1]);
        const std::filesystem::path model_path(argv[2]);
        const auto spec = engine::model_spec::load_spec(spec_path);
        const auto resources = engine::model_spec::load_resource_bundle(model_path, spec_path);
        std::cout << "family=" << spec.require("family").as_string() << "\n";
        std::cout << "display_name=" << spec.require("display_name").as_string() << "\n";
        std::cout << "model_root=" << resources.model_root().string() << "\n";
        std::cout << "files=" << resources.files().size() << "\n";
        const auto configs = engine::model_spec::discover_resources(
            model_path,
            spec_path,
            engine::model_spec::ResourceKind::Files);
        const auto tensors = engine::model_spec::discover_resources(
            model_path,
            spec_path,
            engine::model_spec::ResourceKind::Tensors);
        std::cout << "discovered_configs=" << configs.size() << "\n";
        std::cout << "discovered_tensors=" << tensors.size() << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "model_spec_demo failed: " << error.what() << "\n";
        return 1;
    }
}
