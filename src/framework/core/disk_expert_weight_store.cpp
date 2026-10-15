#include "engine/framework/core/disk_expert_weight_store.h"

#include "engine/framework/debug/trace.h"

#include <ggml-alloc.h>
#include <gguf.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace engine::core {

class DiskExpertWeightStore::Impl {
public:
    struct ContextDeleter { void operator()(ggml_context * ctx) const { ggml_free(ctx); } };
    struct HeaderDeleter { void operator()(gguf_context * ctx) const { gguf_free(ctx); } };
    struct Projection {
        ggml_type type;
        int64_t input, output, experts;
        size_t offset, expert_bytes;
    };
    struct Group {
        std::unique_ptr<ggml_context, ContextDeleter> context;
        ggml_backend_buffer_t buffer = nullptr;
        std::vector<Projection> projections;
        std::vector<ggml_tensor *> tensors;
        std::vector<int32_t> expert_ids;
        std::vector<uint64_t> last_used;
        ~Group() { if (buffer) ggml_backend_buffer_free(buffer); }
    };

    explicit Impl(const std::filesystem::path & path) : file(path, std::ios::binary) {
        if (!file) throw std::runtime_error("cannot open disk expert GGUF: " + path.string());
        ggml_context * tensors = nullptr;
        std::unique_ptr<gguf_context, HeaderDeleter> header(
            gguf_init_from_file(path.string().c_str(), {true, &tensors}));
        std::unique_ptr<ggml_context, ContextDeleter> context(tensors);
        if (!header || !context) throw std::runtime_error("cannot read disk expert GGUF header: " + path.string());
        const auto names = gguf_find_key(header.get(), "audiocpp.tensor_names");
        const auto count = gguf_get_n_tensors(header.get());
        if (names >= 0 && (gguf_get_kv_type(header.get(), names) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(header.get(), names) != GGUF_TYPE_STRING ||
            gguf_get_arr_n(header.get(), names) != static_cast<size_t>(count))) {
            throw std::runtime_error("invalid disk expert GGUF logical names");
        }
        file.seekg(0, std::ios::end);
        const auto file_bytes = file.tellg();
        for (int64_t i = 0; i < count; ++i) {
            const auto * physical = gguf_get_tensor_name(header.get(), i);
            const std::string name = names >= 0 ? gguf_get_arr_str(header.get(), names, i) : physical;
            const auto * tensor = ggml_get_tensor(context.get(), physical);
            if (tensor->ne[3] != 1) continue;
            const auto offset = gguf_get_data_offset(header.get()) + gguf_get_tensor_offset(header.get(), i);
            const auto bytes = gguf_get_tensor_size(header.get(), i);
            if (file_bytes < 0 || offset > static_cast<uint64_t>(file_bytes) ||
                bytes > static_cast<uint64_t>(file_bytes) - offset) {
                throw std::runtime_error("truncated disk expert tensor: " + name);
            }
            const auto expert_bytes = ggml_row_size(tensor->type, tensor->ne[0]) * tensor->ne[1];
            if (bytes != expert_bytes * tensor->ne[2]) throw std::runtime_error("invalid expert bank size: " + name);
            if (!projections.emplace(name, Projection{tensor->type, tensor->ne[0], tensor->ne[1],
                tensor->ne[2], offset, expert_bytes}).second) throw std::runtime_error("duplicate expert bank: " + name);
        }
    }

    void stage(ggml_tensor * ids_tensor, size_t index) {
        auto & group = *groups.at(index);
        std::vector<int32_t> ids(static_cast<size_t>(ggml_nelements(ids_tensor)));
        ggml_backend_tensor_get(ids_tensor, ids.data(), 0, ids.size() * sizeof(int32_t));
        const std::unordered_set<int32_t> required(ids.begin(), ids.end());
        if (required.size() > group.expert_ids.size()) throw std::runtime_error("routed experts exceed disk expert cache capacity");
        for (const auto id : required) {
            if (id < 0 || id >= group.projections.front().experts) throw std::runtime_error("disk expert routing ID out of range");
        }
        ++clock;
        for (const auto id : required) {
            const auto found = std::find(group.expert_ids.begin(), group.expert_ids.end(), id);
            size_t slot;
            if (found != group.expert_ids.end()) {
                slot = static_cast<size_t>(found - group.expert_ids.begin());
                ++stats.cache_hits;
            } else {
                slot = group.expert_ids.size();
                for (size_t candidate = 0; candidate < group.expert_ids.size(); ++candidate) {
                    if (required.count(group.expert_ids[candidate])) continue;
                    if (slot == group.expert_ids.size() || group.last_used[candidate] < group.last_used[slot]) slot = candidate;
                }
                group.expert_ids[slot] = -1;
                for (size_t p = 0; p < group.projections.size(); ++p) {
                    const auto & projection = group.projections[p];
                    scratch.resize(projection.expert_bytes);
                    file.clear();
                    file.seekg(static_cast<std::streamoff>(projection.offset + static_cast<size_t>(id) * projection.expert_bytes));
                    file.read(reinterpret_cast<char *>(scratch.data()), static_cast<std::streamsize>(scratch.size()));
                    if (!file) throw std::runtime_error("failed to read selected expert " + std::to_string(id));
                    ggml_backend_tensor_set(group.tensors[p], scratch.data(), slot * projection.expert_bytes, scratch.size());
                    stats.bytes_read += scratch.size();
                }
                group.expert_ids[slot] = id;
                ++stats.expert_loads;
            }
            group.last_used[slot] = clock;
        }
        for (auto & id : ids) id = static_cast<int32_t>(
            std::find(group.expert_ids.begin(), group.expert_ids.end(), id) - group.expert_ids.begin());
        ggml_backend_tensor_set(ids_tensor, ids.data(), 0, ids.size() * sizeof(int32_t));
    }

    std::ifstream file;
    std::map<std::string, Projection> projections;
    std::vector<std::unique_ptr<Group>> groups;
    std::map<std::string, size_t> routes;
    std::vector<std::byte> scratch;
    uint64_t clock = 0;
    Statistics stats;
};

DiskExpertWeightStore::DiskExpertWeightStore(const std::filesystem::path & path) : impl_(std::make_unique<Impl>(path)) {}
DiskExpertWeightStore::~DiskExpertWeightStore() {
    debug::timing_log_scalar("disk_experts.bytes_read", impl_->stats.bytes_read);
    debug::timing_log_scalar("disk_experts.expert_loads", impl_->stats.expert_loads);
    debug::timing_log_scalar("disk_experts.cache_hits", impl_->stats.cache_hits);
    debug::timing_log_scalar("disk_experts.buffer_mb", double(impl_->stats.buffer_bytes) / (1024 * 1024));
}

size_t DiskExpertWeightStore::add_group(const std::vector<std::string> & names, ggml_backend_t backend, int64_t slots) {
    if (names.empty() || !backend || slots <= 0) throw std::runtime_error("disk expert group requires projections, backend and positive capacity");
    auto group = std::make_unique<Impl::Group>();
    group->context.reset(ggml_init({names.size() * ggml_tensor_overhead() + 1024, nullptr, true}));
    if (!group->context) throw std::runtime_error("cannot allocate disk expert tensor context");
    for (const auto & name : names) {
        const auto entry = impl_->projections.find(name);
        if (entry == impl_->projections.end()) throw std::runtime_error("GGUF expert bank not found: " + name);
        const auto & p = entry->second;
        if (slots > p.experts || (!group->projections.empty() && p.experts != group->projections.front().experts))
            throw std::runtime_error("disk expert group has inconsistent expert counts or capacity");
        group->projections.push_back(p);
        auto * tensor = ggml_new_tensor_3d(group->context.get(), p.type, p.input, p.output, slots);
        ggml_set_name(tensor, name.c_str());
        ggml_set_param(tensor);
        group->tensors.push_back(tensor);
    }
    group->buffer = ggml_backend_alloc_ctx_tensors(group->context.get(), backend);
    if (!group->buffer) throw std::runtime_error("cannot allocate disk expert cache");
    ggml_backend_buffer_set_usage(group->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    group->expert_ids.assign(static_cast<size_t>(slots), -1);
    group->last_used.assign(static_cast<size_t>(slots), 0);
    impl_->stats.buffer_bytes += ggml_backend_buffer_get_size(group->buffer);
    const auto index = impl_->groups.size();
    impl_->groups.push_back(std::move(group));
    impl_->routes.emplace("disk_experts.route." + std::to_string(index), index);
    return index;
}

TensorValue DiskExpertWeightStore::tensor(size_t group, size_t projection) const {
    auto * tensor = impl_->groups.at(group)->tensors.at(projection);
    return wrap_tensor(tensor, TensorShape::from_dims({tensor->ne[2], tensor->ne[1], tensor->ne[0]}), tensor->type);
}

ggml_tensor * DiskExpertWeightStore::route(ModuleBuildContext & ctx, ggml_tensor * ids, size_t group) const {
    if (ids->type != GGML_TYPE_I32) throw std::runtime_error("disk expert routing requires I32 IDs");
    if (group >= impl_->groups.size()) throw std::out_of_range("disk expert group not found");
    auto * routed = ggml_dup(ctx.ggml, ids);
    ggml_set_name(routed, ("disk_experts.route." + std::to_string(group)).c_str());
    ggml_set_output(routed);
    return routed;
}

bool DiskExpertWeightStore::evaluate(ggml_tensor * tensor, bool ask, void * store) {
    auto & self = *static_cast<DiskExpertWeightStore *>(store);
    const auto found = self.impl_->routes.find(ggml_get_name(tensor));
    if (found == self.impl_->routes.end()) return false;
    if (!ask) self.impl_->stage(tensor, found->second);
    return true;
}

DiskExpertWeightStore::Statistics DiskExpertWeightStore::statistics() const noexcept { return impl_->stats; }

}  // namespace engine::core
