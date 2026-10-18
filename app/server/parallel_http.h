#pragma once
#include "http.h"
namespace minitts::server {
// Selected only by --parallel-jobs. Returns after all HTTP workers/callbacks
// have finished; shutdown interrupts socket I/O but cannot preempt GPU work.
void serve_parallel_http(const std::string & host, int port, IHttpHandler & handler,
                         ShutdownRequested shutdown_requested, uint64_t max_request_body_bytes);
} // namespace minitts::server
