#include "engine/community_models/parakeet_tdt/bark_packed.h"
#include "engine/framework/io/json.h"
#include <fstream>
#include <iostream>

// External-fixture probe: dumps compensated FP32 values for comparison with
// the independently decoded official Python checkpoint, tensor by tensor.
int main(int argc, char ** argv) {
    if (argc != 3) { std::cerr << "usage: bark_packed_probe model.bark dump-prefix\n"; return 2; }
    try {
        namespace json = engine::io::json;
        const auto source = engine::community_models::parakeet_tdt::open_bark_packed_source(argv[1]);
        std::ofstream output(std::string(argv[2]) + ".bin", std::ios::binary);
        if (!output) throw std::runtime_error("cannot create dump");
        json::Value::Array manifest;
        size_t offset = 0;
        for (const auto & tensor : source->tensors()) {
            if (tensor.dtype == "I64" || tensor.dtype == "I32") { (void)source->require_i64_scalar(tensor.name); continue; }
            const auto values = source->require_f32(tensor.name);
            const size_t bytes = values.size() * sizeof(float);
            output.write(reinterpret_cast<const char *>(values.data()), bytes);
            if (!output) throw std::runtime_error("dump write failed");
            json::Value::Object row;
            row["name"] = json::Value::make_string(tensor.name);
            row["offset"] = json::Value::make_number(double(offset));
            row["bytes"] = json::Value::make_number(double(bytes));
            row["dtype"] = json::Value::make_string(tensor.dtype);
            manifest.push_back(json::Value::make_object(std::move(row)));
            offset += bytes;
        }
        std::ofstream(std::string(argv[2]) + ".json") << json::stringify(json::Value::make_array(std::move(manifest)));
        source->release_storage();
        std::cout << "dumped " << offset << " bytes\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
