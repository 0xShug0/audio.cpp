#!/usr/bin/env bash

for i in $(seq 1 60); do
  if curl -fsS -o /dev/null http://localhost:8080/health 2>/dev/null; then
    echo "Server is ready"
    exit 0
  fi
  sleep 1
done

echo "Server did not become healthy after 60s" >&2
exit 1
