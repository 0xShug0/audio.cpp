#!/usr/bin/env python3
"""Local HTTP fixture for server_model_installer_test; never used at runtime."""

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FILES = {
    "/org/repo/resolve/main/Demo/model-q8.gguf": b"q8-model-payload",
    "/org/repo/resolve/main/Demo/model-f16.gguf": b"f16-model-payload",
    "/org/repo/resolve/main/Demo/shared.json": b'{"shared":true}\n',
    "/org/repo/resolve/main/Slow/model.gguf": b"s" * (16 * 1024 * 1024),
}


class Handler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.respond(False)

    def do_GET(self):
        self.respond(True)

    def respond(self, body):
        payload = FILES.get(self.path)
        if payload is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("ETag", '"fixture-etag-' + str(len(payload)) + '"')
        self.send_header("X-Repo-Commit", "fixture-commit")
        self.end_headers()
        if body:
            try:
                for offset in range(0, len(payload), 64 * 1024):
                    self.wfile.write(payload[offset : offset + 64 * 1024])
                    self.wfile.flush()
                    if len(payload) > 1024 * 1024:
                        time.sleep(0.003)
            except (BrokenPipeError, ConnectionResetError):
                pass
        with self.server.remaining_lock:
            self.server.remaining -= 1
            if self.server.remaining <= 0:
                threading.Thread(target=self.server.shutdown, daemon=True).start()

    def log_message(self, *_args):
        pass


parser = argparse.ArgumentParser()
parser.add_argument("--requests", type=int, default=14)
args = parser.parse_args()
server = ThreadingHTTPServer(("127.0.0.1", 18991), Handler)
server.remaining = args.requests
server.remaining_lock = threading.Lock()
timeout = threading.Timer(30, server.shutdown)
timeout.daemon = True
timeout.start()
server.serve_forever()
