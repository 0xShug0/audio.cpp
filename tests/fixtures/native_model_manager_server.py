#!/usr/bin/env python3
"""Local HTTP fixture for server_model_installer_test; never used at runtime."""

import argparse
import hashlib
import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FILES = {
    "/org/repo/resolve/main/Demo/model-q8.gguf": b"q8-model-payload",
    "/org/repo/resolve/main/Demo/model-f16.gguf": b"f16-model-payload",
    "/org/repo/resolve/main/Demo/shared.json": b'{"shared":true}\n',
    "/org/repo/resolve/main/Slow/model.gguf": b"s" * (16 * 1024 * 1024),
}

# ModelScope-style repo: default branch is master, and the file-list API is
# the source of truth for per-file size and sha256.
MS_REPO = "ms/repo"
MS_REVISION = "master"
MS_FILES = {
    "Demo/model-q8.gguf": b"q8-model-payload",
    "Demo/shared.json": b'{"shared":true}\n',
}


def ms_listing_payload():
    files = [
        {
            "Path": path,
            "Size": len(payload),
            "Sha256": hashlib.sha256(payload).hexdigest(),
            "Type": "blob",
        }
        for path, payload in sorted(MS_FILES.items())
    ]
    return json.dumps({"Code": 200, "Success": True, "Data": {"Files": files}}).encode()


class Handler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.respond(False)

    def do_GET(self):
        self.respond(True)

    def respond(self, body):
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path.startswith("/api/v1/models/"):
            if not self.check_ms_auth():
                return
            self.respond_ms_listing(parsed, body)
            return
        if parsed.path.startswith("/models/"):
            if not self.check_ms_auth():
                return
            self.respond_ms_resolve(parsed.path, body)
            return
        payload = FILES.get(self.path)
        if payload is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("ETag", '"fixture-etag-' + str(len(payload)) + '"')
        self.send_header("X-Repo-Commit", "fixture-commit")
        self.end_headers()
        self.write_payload(payload, body)

    def check_ms_auth(self):
        # ModelScope requests must never carry the Hugging Face credential;
        # when --ms-token is given they must carry exactly that token.
        auth = self.headers.get("Authorization", "")
        hf_token = self.server.hf_token
        ms_token = self.server.ms_token
        if hf_token and auth == "Bearer " + hf_token:
            self.send_error(403, "HF token leaked into a ModelScope request")
            self.count_request()
            return False
        if ms_token and auth != "Bearer " + ms_token:
            self.send_error(401, "ModelScope request missing its own token")
            self.count_request()
            return False
        return True

    def respond_ms_listing(self, parsed, body):
        prefix = "/api/v1/models/" + MS_REPO + "/repo/files"
        query = urllib.parse.parse_qs(parsed.query)
        revision = query.get("Revision", [""])[0]
        if parsed.path != prefix or revision != MS_REVISION:
            self.send_error(404)
            return
        payload = ms_listing_payload()
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.write_payload(payload, body)

    def respond_ms_resolve(self, path, body):
        prefix = "/models/" + MS_REPO + "/resolve/" + MS_REVISION + "/"
        if not path.startswith(prefix):
            self.send_error(404)
            return
        payload = MS_FILES.get(path[len(prefix):])
        if payload is None:
            self.send_error(404)
            return
        etag = hashlib.sha256(payload).hexdigest()
        self.send_response(200)
        # Like real ModelScope resolve responses: no Content-Length and no
        # ETag on HEAD, only X-Linked-Etag.
        self.send_header("X-Linked-Etag", etag)
        if body:
            self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.write_payload(payload, body)

    def write_payload(self, payload, body):
        if body:
            try:
                for offset in range(0, len(payload), 64 * 1024):
                    self.wfile.write(payload[offset : offset + 64 * 1024])
                    self.wfile.flush()
                    if len(payload) > 1024 * 1024:
                        time.sleep(0.003)
            except (BrokenPipeError, ConnectionResetError):
                pass
        self.count_request()

    def count_request(self):
        with self.server.remaining_lock:
            self.server.remaining -= 1
            if self.server.remaining <= 0:
                threading.Thread(target=self.server.shutdown, daemon=True).start()

    def log_message(self, *_args):
        pass


parser = argparse.ArgumentParser()
parser.add_argument("--requests", type=int, default=14)
parser.add_argument("--port", type=int, default=18991)
parser.add_argument("--hf-token", default="")
parser.add_argument("--ms-token", default="")
args = parser.parse_args()
server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
server.remaining = args.requests
server.remaining_lock = threading.Lock()
server.hf_token = args.hf_token
server.ms_token = args.ms_token
timeout = threading.Timer(30, server.shutdown)
timeout.daemon = True
timeout.start()
server.serve_forever()
