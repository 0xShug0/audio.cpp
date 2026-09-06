#include "frontend.h"

#include "engine/framework/debug/trace.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace minitts::server {

namespace {

bool contract_value_matches(std::string_view lhs, std::string_view rhs) {
    return lhs == frontend_contracts::any || rhs == frontend_contracts::any || lhs == rhs;
}

bool contract_route_overlaps(std::string_view lhs, std::string_view rhs) {
    return contract_value_matches(lhs, rhs);
}

bool validate_pre_contracts(
    std::string_view previous_module,
    const FrontendPreContract & previous,
    std::string_view next_module,
    const FrontendPreContract & next) {
    if (!contract_route_overlaps(previous.method, next.method) ||
        !contract_route_overlaps(previous.path, next.path)) {
        return false;
    }
    if (!contract_value_matches(previous.request_out, next.request_in)) {
        throw std::runtime_error(
            "incompatible frontend pre-processing order: " + std::string(previous_module) +
            " outputs request state '" + std::string(previous.request_out) +
            "', but " + std::string(next_module) +
            " expects '" + std::string(next.request_in) + "'");
    }
    return true;
}

bool validate_post_contracts(
    std::string_view previous_module,
    const FrontendPostContract & previous,
    std::string_view next_module,
    const FrontendPostContract & next) {
    if (!contract_route_overlaps(previous.method, next.method) ||
        !contract_route_overlaps(previous.path, next.path)) {
        return false;
    }
    if (!contract_value_matches(previous.response_out, next.response_in)) {
        throw std::runtime_error(
            "incompatible frontend post-processing order: " + std::string(previous_module) +
            " outputs response state '" + std::string(previous.response_out) +
            "', but " + std::string(next_module) +
            " expects '" + std::string(next.response_in) + "'");
    }
    return true;
}

} // namespace

#include "server_frontend_module_declarations.inc"

std::optional<FrontendPreContract> ServerFrontendModule::pre_contract() const {
    return std::nullopt;
}

std::optional<FrontendPostContract> ServerFrontendModule::post_contract() const {
    return std::nullopt;
}

void ServerFrontendModule::pre_process(ServerFrontendContext & context, ServerFrontendRequest & request) {
    (void) context;
    (void) request;
}

void ServerFrontendModule::post_process(ServerFrontendContext & context, ServerFrontendResponse & response) {
    (void) context;
    (void) response;
}

void ServerFrontendRegistry::add(ServerFrontendModuleFactory factory) {
    if (factory == nullptr) {
        throw std::runtime_error("server frontend registration requires a module factory");
    }
    auto module = factory();
    if (!module) {
        throw std::runtime_error("server frontend module factory returned null");
    }

    const auto pre = module->pre_contract();
    const auto post = module->post_contract();
    for (auto it = modules_.rbegin(); it != modules_.rend(); ++it) {
        if (pre.has_value()) {
            if (const auto previous = (*it)->pre_contract()) {
                if (validate_pre_contracts((*it)->name(), *previous, module->name(), *pre)) {
                    break;
                }
            }
        }
    }
    for (auto it = modules_.rbegin(); it != modules_.rend(); ++it) {
        if (post.has_value()) {
            if (const auto previous = (*it)->post_contract()) {
                if (validate_post_contracts((*it)->name(), *previous, module->name(), *post)) {
                    break;
                }
            }
        }
    }

    modules_.push_back(std::move(module));
}

bool ServerFrontendRegistry::empty() const { return modules_.empty(); }

HttpResponse ServerFrontendRegistry::handle(ServerFrontendContext & context, const HttpRequest & request) const {
    ServerFrontendRequest frontend_request{request, std::nullopt};
    if (engine::debug::log_enabled()) {
        const auto content_type = request.headers.find("content-type");
        engine::debug::log_message(
            "[MP3_FRONTEND_DEBUG] frontend.enter method=" + request.method +
            " path=" + request.path +
            " content_type=" +
            (content_type == request.headers.end() ? std::string("<none>") : content_type->second) +
            " body_bytes=" + std::to_string(request.body.size()));
    }
    for (const auto & module : modules_) {
        if (engine::debug::log_enabled()) {
            engine::debug::log_message(
                "[MP3_FRONTEND_DEBUG] frontend.pre.begin module=" + std::string(module->name()) +
                " path=" + frontend_request.request.path +
                " body_bytes=" + std::to_string(frontend_request.request.body.size()));
        }
        module->pre_process(context, frontend_request);
        if (engine::debug::log_enabled()) {
            const auto content_type = frontend_request.request.headers.find("content-type");
            engine::debug::log_message(
                "[MP3_FRONTEND_DEBUG] frontend.pre.end module=" + std::string(module->name()) +
                " path=" + frontend_request.request.path +
                " content_type=" +
                (content_type == frontend_request.request.headers.end() ? std::string("<none>") : content_type->second) +
                " body_bytes=" + std::to_string(frontend_request.request.body.size()) +
                " short_circuit=" + (frontend_request.response.has_value() ? "true" : "false"));
        }
        if (frontend_request.response.has_value()) {
            return std::move(*frontend_request.response);
        }
    }

    auto core_response = context.forward_to_core(frontend_request.request);
    if (engine::debug::log_enabled()) {
        engine::debug::log_message(
            "[MP3_FRONTEND_DEBUG] frontend.core_response status=" + std::to_string(core_response.status) +
            " content_type=" + core_response.content_type +
            " body_bytes=" + std::to_string(core_response.body.size()));
    }
    ServerFrontendResponse frontend_response{request, frontend_request.request, std::move(core_response)};
    for (const auto & module : modules_) {
        module->post_process(context, frontend_response);
    }
    return std::move(frontend_response.response);
}

void register_static_server_frontends(ServerFrontendRegistry & registry) {
    (void) registry;
#include "server_frontend_module_registrations.inc"
}

#if !defined(AUDIOCPP_SERVER_FRONTEND_HAS_HTTPS)
void serve_frontend_https(
    const std::string & host,
    int port,
    IHttpHandler & handler,
    ShutdownRequested shutdown_requested,
    uint64_t max_request_body_bytes,
    const ServerFrontendHttpsConfig & config) {
    (void) host;
    (void) port;
    (void) handler;
    (void) shutdown_requested;
    (void) max_request_body_bytes;
    (void) config;
    throw std::runtime_error(
        "HTTPS frontend support is not available in this build; configure with "
        "-DAUDIOCPP_BUILD_SERVER_FRONTENDS=ON -DAUDIOCPP_SERVER_FRONTEND_MODULES=https");
}
#endif

} // namespace minitts::server
