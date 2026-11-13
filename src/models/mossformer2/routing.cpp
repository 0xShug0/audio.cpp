#include "engine/models/mossformer2/routing.h"
#include "engine/models/mossformer2/session.h"
#include "engine/framework/runtime/model_routing.h"

namespace engine::models::mossformer2 {
namespace {

const MossFormer2Implementation separation = [] {
    MossFormer2Implementation impl;
    impl.task = runtime::VoiceTaskKind::SourceSeparation;
    impl.dilated_fsmn = true;
    impl.create_session = create_separation_session;
    return impl;
}();

const MossFormer2Implementation enhancement = [] {
    MossFormer2Implementation impl;
    impl.create_session = create_restoration_session;
    return impl;
}();

const MossFormer2Implementation super_resolution = [] {
    auto impl = enhancement;
    impl.use_vocoder = true;
    return impl;
}();

// Implementation registrations. Missing variant metadata in legacy packages
// selects "ss" at the asset-loading boundary; unknown variants are rejected.
const runtime::ModelRouteEntry<std::string_view, MossFormer2Implementation> entries[] = {
    {"ss", &separation},
    {"se", &enhancement},
    {"sr", &super_resolution},
};
const runtime::ModelRouteTable<std::string_view, MossFormer2Implementation> table(
    "MossFormer2 variant", entries);

}  // namespace

const MossFormer2Implementation & implementation_for_variant(std::string_view variant) {
    return table.require(variant);
}

}  // namespace engine::models::mossformer2
