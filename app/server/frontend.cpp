#include "frontend.h"

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
    for (const auto & module : modules_) {
        module->pre_process(context, frontend_request);
        if (frontend_request.response.has_value()) {
            return std::move(*frontend_request.response);
        }
    }

    auto core_response = context.forward_to_core(frontend_request.request);
    ServerFrontendResponse frontend_response{request, frontend_request.request, std::move(core_response)};
    for (const auto & module : modules_) {
        module->post_process(context, frontend_response);
    }
    return std::move(frontend_response.response);
}

void register_static_server_frontends(ServerFrontendRegistry & registry) {
#include "server_frontend_module_registrations.inc"
}

} // namespace minitts::server
