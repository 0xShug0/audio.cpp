#pragma once

#include "engine/framework/io/json.h"

namespace engine::models::kitten_tts {

enum class KittenVariant { Mini08, Mini2 };

struct KittenRoute {
    KittenVariant variant;
    const char * default_voice;
};

const KittenRoute & resolve_kitten_route(const engine::io::json::Value & config);

} // namespace engine::models::kitten_tts
