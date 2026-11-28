#include "engine/framework/runtime/model_routing.h"
#include "engine/framework/debug/profiler.h"
#include "test_assert.h"

#include <chrono>
#include <iostream>
#include <string_view>
#include <utility>

namespace {
struct Impl {
    int (*run)(int);
};

int increment(int value) { return value + 1; }
int double_value(int value) { return value * 2; }

enum class Variant { Base, Alternate, Unknown };

template<class Function>
void rejects(Function function, const char * expected) {
    try {
        function();
    } catch (const std::invalid_argument & error) {
        engine::test::require(std::string(error.what()).find(expected) != std::string::npos,
            "routing error lost its registration label/reason");
        return;
    }
    throw std::runtime_error("invalid implementation registration/selection was accepted");
}
} // namespace

int main(int argc, char ** argv) try {
    using namespace engine;
    using test::require;
    const auto started = std::chrono::steady_clock::now();
    require(argc == 3 && std::string_view(argv[1]) == "--log", "expected --log <file>");
    debug::configure_logging({true, argv[2]});
    static const Impl base{increment}, alternate{double_value};

    // Family names compare by content, not by the address of their spelling.
    static const runtime::ModelRouteEntry<std::string_view, Impl> families[]{
        {"base", &base}, {"alternate", &alternate},
    };
    const runtime::ModelRouteTable<std::string_view, Impl> family_table("families", families);
    const std::string dynamic_name = "alternate";
    require(&family_table.require(dynamic_name) == &alternate, "family selection copied or chose the wrong descriptor");
    rejects([&] { (void)family_table.require("missing"); }, "families: unregistered");

    // Checkpoint enums use the identical API. Retain the descriptor across
    // repeated execution, and select afresh when the request route changes.
    static const runtime::ModelRouteEntry<Variant, Impl> variants[]{
        {Variant::Base, &base}, {Variant::Alternate, &alternate},
    };
    const runtime::ModelRouteTable<Variant, Impl> variant_table("variants", variants);
    for (const auto route : {Variant::Base, Variant::Alternate, Variant::Base}) {
        const auto & selected = variant_table.require(route);
        require(&selected == (route == Variant::Base ? &base : &alternate), "changed request retained a stale route");
        for (int value = 0; value < 10; ++value)
            require(selected.run(value) == (route == Variant::Base ? value + 1 : value * 2), "bound operation differs");
    }
    rejects([&] { (void)variant_table.require(Variant::Unknown); }, "unregistered");

    // Compound request keys and independently owned model tables are supported
    // without adding family/variant/task interpretation to the framework.
    using Key = std::pair<Variant, int>;
    static const runtime::ModelRouteEntry<Key, Impl> tasks[]{
        {{Variant::Base, 0}, &alternate}, {{Variant::Base, 1}, &base},
    };
    const runtime::ModelRouteTable<Key, Impl> task_table("tasks", tasks);
    require(&task_table.require({Variant::Base, 0}) == &alternate, "tables leaked selection state");
    require(&task_table.require({Variant::Base, 1}) == &base, "compound key was partially matched");
    rejects([&] { (void)task_table.require({Variant::Alternate, 0}); }, "unregistered");

    static const runtime::ModelRouteEntry<Variant, Impl> duplicates[]{
        {Variant::Base, &base}, {Variant::Base, &alternate},
    };
    rejects([&] { runtime::ModelRouteTable<Variant, Impl> table("duplicate", duplicates); }, "duplicate implementation");
    static const runtime::ModelRouteEntry<Variant, Impl> null_entry[]{
        {Variant::Base, nullptr},
    };
    rejects([&] { runtime::ModelRouteTable<Variant, Impl> table("null", null_entry); }, "null implementation");

    debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
    std::cout << "model_routing: PASS\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
}
