#include "engine/framework/core/disk_expert_weight_store.h"

#include "engine/framework/debug/trace.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/binary.h"

#include <ggml-alloc.h>
#include <gguf.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#if defined(__APPLE__) && !defined(_OPENMP)
#include <dispatch/dispatch.h>
#endif

namespace engine::core {

class DiskExpertWeightStore::Impl {
public:
    struct ContextDeleter { void operator()(ggml_context * ctx) const { ggml_free(ctx); } };
    struct HeaderDeleter { void operator()(gguf_context * ctx) const { gguf_free(ctx); } };
    struct Projection {
        ggml_type type;
        int64_t input, output, experts;
        size_t offset, expert_bytes;
        const std::byte * host_data = nullptr;
    };
    struct Group {
        std::unique_ptr<ggml_context, ContextDeleter> context;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_backend_buffer_t host_buffer = nullptr;
        ggml_backend_t backend = nullptr;
        std::vector<Projection> projections;
        std::vector<ggml_tensor *> tensors;
        std::vector<int32_t> expert_ids;
        std::vector<uint64_t> last_used;
        ~Group() {
            if (buffer) ggml_backend_buffer_free(buffer);
            if (host_buffer) ggml_backend_buffer_free(host_buffer);
        }
    };
    struct Staging {
        ggml_backend_buffer_t buffer = nullptr;
        std::vector<std::byte> bytes;
        size_t capacity = 0;
        ~Staging() { if (buffer) ggml_backend_buffer_free(buffer); }
        std::byte * data() { return buffer ? static_cast<std::byte *>(ggml_backend_buffer_get_base(buffer)) : bytes.data(); }
    };

    Impl(const std::filesystem::path & path, bool pin, size_t threads)
        : file(path, std::ios::binary), pin_experts(pin), read_threads(threads) {
        if (read_threads < 1 || read_threads > 64)
            throw std::invalid_argument("disk expert read_threads must be between 1 and 64");
#if !defined(_OPENMP) && !defined(__APPLE__)
        if (!pin_experts && read_threads > 1)
            throw std::runtime_error("parallel disk expert reads require ENGINE_ENABLE_OPENMP=ON");
#endif
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
                tensor->ne[2], offset, static_cast<size_t>(expert_bytes)}).second) throw std::runtime_error("duplicate expert bank: " + name);
        }
        mapped_file = io::read_binary_blob(path, true);
    }

    std::vector<int32_t> stage(std::vector<int32_t> ids, size_t index) {
        const auto stage_started = std::chrono::steady_clock::now();
        auto & group = *groups.at(index);
        const std::unordered_set<int32_t> required(ids.begin(), ids.end());
        if (required.size() > group.expert_ids.size()) throw std::runtime_error("routed experts exceed disk expert cache capacity");
        for (const auto id : required) {
            if (id < 0 || id >= group.projections.front().experts) throw std::runtime_error("disk expert routing ID out of range");
        }
        ++clock;
        struct Miss { int32_t id; size_t slot; };
        std::vector<Miss> misses;
        auto planned_ids = group.expert_ids;
        for (const auto id : required) {
            const auto found = std::find(planned_ids.begin(), planned_ids.end(), id);
            size_t slot;
            if (found != planned_ids.end()) {
                slot = static_cast<size_t>(found - planned_ids.begin());
                ++stats.cache_hits;
            } else {
                slot = planned_ids.size();
                for (size_t candidate = 0; candidate < planned_ids.size(); ++candidate) {
                    if (required.count(planned_ids[candidate])) continue;
                    if (slot == planned_ids.size() || group.last_used[candidate] < group.last_used[slot]) slot = candidate;
                }
                planned_ids[slot] = id;
                misses.push_back({id, slot});
                // This slot must not remain a hit if a later read/upload fails.
                group.expert_ids[slot] = -1;
            }
            group.last_used[slot] = clock;
        }
        std::sort(misses.begin(), misses.end(), [](const Miss & a, const Miss & b) { return a.slot < b.slot; });
        auto * staging = pin_experts ? nullptr : host_staging.at(group.backend)->data();
        if (!pin_experts && !misses.empty()) {
            file.clear();
            file.seekg(0, std::ios::end);
            if (file.tellg() < static_cast<std::streamoff>(mapped_file.size()))
                throw std::runtime_error("failed to read selected expert: GGUF was truncated");
        }
        size_t offset = 0;
        // Projection ranges in staging do not overlap. Post each upload before
        // reading the next projection, then wait before reusing any range.
        for (size_t p = 0; p < group.projections.size(); ++p) {
            const auto & projection = group.projections[p];
            const auto bytes = projection.expert_bytes;
            auto started = std::chrono::steady_clock::now();
            if (!pin_experts && !misses.empty()) {
                if (read_threads == 1 || misses.size() == 1) {
                    for (size_t m = 0; m < misses.size(); ++m) {
                        std::memcpy(staging + offset + m * bytes,
                            mapped_file.data() + projection.offset + static_cast<size_t>(misses[m].id) * bytes, bytes);
                    }
                } else {
#if defined(_OPENMP)
                    // Varying team sizes recreates OpenMP workers in the routing hot path.
                    const int worker_count = static_cast<int>(std::min(read_threads, group.expert_ids.size()));
                    #pragma omp parallel for num_threads(worker_count) schedule(static)
                    for (int64_t m = 0; m < static_cast<int64_t>(misses.size()); ++m) {
                        std::memcpy(staging + offset + m * bytes,
                            mapped_file.data() + projection.offset + static_cast<size_t>(misses[m].id) * bytes, bytes);
                    }
#elif defined(__APPLE__)
                    // Apple release builds omit OpenMP; dispatch joins all reads before upload.
                    const size_t workers = std::min(read_threads, misses.size());
                    struct ReadBatch {
                        std::byte * destination;
                        const std::byte * source;
                        const std::vector<Miss> & misses;
                        size_t bytes, workers;
                    } batch{staging + offset, mapped_file.data() + projection.offset, misses, bytes, workers};
                    dispatch_apply_f(workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                        &batch, [](void * context, size_t worker) {
                            const auto & batch = *static_cast<ReadBatch *>(context);
                            for (size_t m = worker; m < batch.misses.size(); m += batch.workers) {
                                std::memcpy(batch.destination + m * batch.bytes,
                                    batch.source + static_cast<size_t>(batch.misses[m].id) * batch.bytes,
                                    batch.bytes);
                            }
                        });
#endif
                }
                stats.bytes_read += misses.size() * bytes;
            }
            stats.read_ms += debug::elapsed_ms(started);
            started = std::chrono::steady_clock::now();
            for (size_t begin = 0; begin < misses.size();) {
                size_t end = begin + 1;
                while (end < misses.size() && misses[end].slot == misses[end - 1].slot + 1 &&
                    (!pin_experts || misses[end].id == misses[end - 1].id + 1)) ++end;
                const auto * data = pin_experts ? projection.host_data + static_cast<size_t>(misses[begin].id) * bytes
                                               : staging + offset + begin * bytes;
                ggml_backend_tensor_set_async(group.backend, group.tensors[p], data,
                    misses[begin].slot * bytes, (end - begin) * bytes);
                begin = end;
            }
            stats.upload_ms += debug::elapsed_ms(started);
            offset += misses.size() * bytes;
        }
        const auto started = std::chrono::steady_clock::now();
        if (!misses.empty()) ggml_backend_synchronize(group.backend);
        stats.upload_ms += debug::elapsed_ms(started);
        for (const auto & miss : misses) group.expert_ids[miss.slot] = miss.id;
        stats.expert_loads += misses.size();
        for (auto & id : ids) id = static_cast<int32_t>(
            std::find(group.expert_ids.begin(), group.expert_ids.end(), id) - group.expert_ids.begin());
        stats.stage_ms += debug::elapsed_ms(stage_started);
        return ids;
    }

    std::ifstream file;
    io::BinaryBlob mapped_file;
    bool pin_experts;
    size_t read_threads;
    std::map<std::string, Projection> projections;
    std::vector<std::unique_ptr<Group>> groups;
    std::map<std::string, size_t> routes;
    std::map<ggml_backend_t, std::unique_ptr<Staging>> host_staging;
    uint64_t clock = 0;
    Statistics stats;
};

DiskExpertWeightStore::DiskExpertWeightStore(const std::filesystem::path & path, bool pin_experts, size_t read_threads)
    : impl_(std::make_unique<Impl>(path, pin_experts, read_threads)) {}
DiskExpertWeightStore::~DiskExpertWeightStore() {
    debug::timing_log_scalar("disk_experts.bytes_read", impl_->stats.bytes_read);
    debug::timing_log_scalar("disk_experts.expert_loads", impl_->stats.expert_loads);
    debug::timing_log_scalar("disk_experts.cache_hits", impl_->stats.cache_hits);
    debug::timing_log_scalar("disk_experts.buffer_mb", double(impl_->stats.buffer_bytes) / (1024 * 1024));
    debug::timing_log_scalar("disk_experts.host_buffer_mb", double(impl_->stats.host_buffer_bytes) / (1024 * 1024));
    debug::timing_log_scalar("disk_experts.read_ms", impl_->stats.read_ms);
    debug::timing_log_scalar("disk_experts.upload_ms", impl_->stats.upload_ms);
    debug::timing_log_scalar("disk_experts.stage_ms", impl_->stats.stage_ms);
}

size_t DiskExpertWeightStore::add_group(const std::vector<std::string> & names, ggml_backend_t backend, int64_t slots) {
    if (names.empty() || !backend || slots <= 0) throw std::runtime_error("disk expert group requires projections, backend and positive capacity");
    auto group = std::make_unique<Impl::Group>();
    group->backend = backend;
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
    // Unoccupied slots are valid zero weights for padded batched expert work.
    ggml_backend_buffer_clear(group->buffer, 0);
    group->expert_ids.assign(static_cast<size_t>(slots), -1);
    group->last_used.assign(static_cast<size_t>(slots), 0);
    if (impl_->pin_experts) {
        const auto host_type = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
        if (!host_type) throw std::runtime_error("pinned experts require a backend with pinned host buffers");
        size_t host_bytes = 0;
        for (const auto & projection : group->projections) host_bytes += projection.expert_bytes * projection.experts;
        group->host_buffer = ggml_backend_buft_alloc_buffer(host_type, host_bytes);
        if (!group->host_buffer || ggml_backend_buffer_get_type(group->host_buffer) != host_type)
            throw std::runtime_error("cannot allocate pinned host expert banks");
        if (impl_->stats.host_buffer_bytes == 0)
            debug::trace_log_scalar("disk_experts.host_buffer_type", ggml_backend_buft_name(host_type));
        auto * data = static_cast<std::byte *>(ggml_backend_buffer_get_base(group->host_buffer));
        const auto started = std::chrono::steady_clock::now();
        for (auto & projection : group->projections) {
            const auto bytes = projection.expert_bytes * projection.experts;
            std::memcpy(data, impl_->mapped_file.data() + projection.offset, bytes);
            projection.host_data = data;
            data += bytes;
            impl_->stats.bytes_read += bytes;
            impl_->mapped_file.discard_range(projection.offset, bytes);
        }
        impl_->stats.read_ms += debug::elapsed_ms(started);
        impl_->stats.host_buffer_bytes += ggml_backend_buffer_get_size(group->host_buffer);
    } else {
        size_t staging_bytes = 0;
        for (const auto & projection : group->projections) staging_bytes += projection.expert_bytes * slots;
        auto & staging = impl_->host_staging[backend];
        if (!staging || staging->capacity < staging_bytes) {
            auto buffer = std::make_unique<Impl::Staging>();
            const auto host_type = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
            if (host_type) {
                buffer->buffer = ggml_backend_buft_alloc_buffer(host_type, staging_bytes);
                if (!buffer->buffer || ggml_backend_buffer_get_type(buffer->buffer) != host_type)
                    throw std::runtime_error("cannot allocate pinned expert upload staging buffer");
            } else {
                // Backends without pinned host buffers use ordinary host staging.
                buffer->bytes.resize(staging_bytes);
            }
            buffer->capacity = staging_bytes;
            staging = std::move(buffer);
            debug::trace_log_scalar("disk_experts.staging_backend", ggml_backend_name(backend));
            debug::trace_log_scalar("disk_experts.staging_type", host_type ? ggml_backend_buft_name(host_type) : "host");
            debug::timing_log_scalar("disk_experts.staging_mb", double(staging_bytes) / (1024 * 1024));
        }
    }
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

std::vector<int32_t> DiskExpertWeightStore::stage(size_t group, const std::vector<int32_t> & expert_ids) {
    return impl_->stage(expert_ids, group);
}

void DiskExpertWeightStore::invalidate_cache() {
    for (auto & group : impl_->groups) {
        std::fill(group->expert_ids.begin(), group->expert_ids.end(), -1);
        std::fill(group->last_used.begin(), group->last_used.end(), 0);
    }
    impl_->clock = 0;
}

bool DiskExpertWeightStore::evaluate(ggml_tensor * tensor, bool ask, void * store) {
    auto & self = *static_cast<DiskExpertWeightStore *>(store);
    const auto found = self.impl_->routes.find(ggml_get_name(tensor));
    if (found == self.impl_->routes.end()) return false;
    if (!ask) {
        auto started = std::chrono::steady_clock::now();
        std::vector<int32_t> ids(static_cast<size_t>(ggml_nelements(tensor)));
        ggml_backend_tensor_get(tensor, ids.data(), 0, ids.size() * sizeof(int32_t));
        self.impl_->stats.stage_ms += debug::elapsed_ms(started);
        ids = self.impl_->stage(std::move(ids), found->second);
        started = std::chrono::steady_clock::now();
        ggml_backend_tensor_set(tensor, ids.data(), 0, ids.size() * sizeof(int32_t));
        self.impl_->stats.stage_ms += debug::elapsed_ms(started);
    }
    return true;
}

DiskExpertWeightStore::Statistics DiskExpertWeightStore::statistics() const noexcept { return impl_->stats; }

}  // namespace engine::core
