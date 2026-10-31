#include "engine/community_models/parakeet_tdt/bark_packed.h"
#include "engine/framework/io/json.h"
#include "test_assert.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>

int main() {
    namespace json = engine::io::json;
    namespace bark = engine::community_models::parakeet_tdt;
    auto check = [](bool ok) { engine::test::require(ok, "Bark packed decoder regression"); };
    const auto fixture = std::filesystem::path(ENGINE_REPO_ROOT) / "tests/parakeet_tdt/assets/bark-packed";
    const auto temp = std::filesystem::temp_directory_path() /
        ("bark-packed-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(temp);
    try {
        auto source = bark::open_bark_packed_source(fixture / "model.bark");
        const auto expected = json::parse_file(fixture / "expected.json");
        for (const auto & row : expected.as_object()) {
            const auto actual = source->require_f32(row.first);
            check(actual.size() == row.second.as_array().size());
            for (size_t i = 0; i < actual.size(); ++i) {
                const float value = row.second.as_array()[i].as_f32();
                check(std::memcmp(&actual[i], &value, sizeof(float)) == 0);
            }
            const auto raw = source->require_tensor_data(row.first);
            check(!raw.bytes.empty());
        }
        check(source->require_i64_scalar("counter") == -7);
        check(!source->optional_f32("missing").has_value());
        const auto before = source->require_f32("levels3.weight");
        source->release_storage();
        check(before == source->require_f32("levels3.weight"));
        std::ifstream input(fixture / "model.bark", std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
        uint64_t header_size = 0;
        for (int i = 0; i < 8; ++i) header_size |= uint64_t(uint8_t(bytes[i])) << (8 * i);
        const auto header = json::parse(std::string_view(bytes.data() + 8, size_t(header_size)));
        for (int mutation = 0; mutation < 7; ++mutation) {
            auto object = header.as_object();
            auto index = header.require("index").as_array();
            auto first = index[0].as_object();
            if (mutation == 0) object["format"] = json::Value::make_string("unknown");
            if (mutation == 1) first["shape"] = json::Value::make_array({json::Value::make_number(0)});
            if (mutation == 2) first["shape"] = json::Value::make_array({json::Value::make_number(67108865)});
            if (mutation == 3) first["b"] = json::Value::make_number(268435457);
            index[0] = json::Value::make_object(std::move(first));
            if (mutation == 4) {
                auto duplicate = index[1].as_object();
                duplicate["n"] = index[0].require("n");
                index[1] = json::Value::make_object(std::move(duplicate));
            }
            if (mutation == 5) {
                auto level = index[6].as_object();
                level["K"] = json::Value::make_number(127);
                index[6] = json::Value::make_object(std::move(level));
            }
            if (mutation == 6) {
                auto last = index.back().as_object();
                last["shape"] = json::Value::make_array({json::Value::make_number(1)});
                index.back() = json::Value::make_object(std::move(last));
            }
            object["index"] = json::Value::make_array(std::move(index));
            const auto text = json::stringify(json::Value::make_object(std::move(object)));
            const auto path = temp / ("header-" + std::to_string(mutation) + ".bark");
            {
                std::ofstream out(path, std::ios::binary);
                const auto size = uint64_t(text.size());
                for (int i = 0; i < 8; ++i) out.put(char(size >> (8 * i)));
                out.write(text.data(), text.size());
                out.write(bytes.data() + 8 + header_size, bytes.size() - 8 - header_size);
            }
            bool rejected = false;
            try { (void)bark::open_bark_packed_source(path)->require_f32("xz.weight"); }
            catch (const std::exception &) { rejected = true; }
            check(rejected);
        }
        for (int mutation = 0; mutation < 4; ++mutation) {
            auto corrupt = bytes;
            if (mutation == 0) corrupt.resize(7);
            if (mutation == 1) corrupt.pop_back();
            if (mutation == 2) corrupt.push_back(0);
            if (mutation == 3) corrupt.back() ^= 1;
            const auto path = temp / (std::to_string(mutation) + ".bark");
            { std::ofstream out(path, std::ios::binary); out.write(corrupt.data(), corrupt.size()); }
            bool rejected = false;
            try { (void)bark::open_bark_packed_source(path)->require_f32("xz.weight"); }
            catch (const std::exception &) { rejected = true; }
            check(rejected);
        }
        std::filesystem::remove_all(temp);
        std::cout << "Bark packed decoder tests passed\n";
        return 0;
    } catch (const std::exception & e) {
        std::filesystem::remove_all(temp);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
