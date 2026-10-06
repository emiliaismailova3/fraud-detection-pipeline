#!/usr/bin/env bash
# Start the API container, send real HTTP requests, then remove it.
# Usage (from the repo root, after docker build):  bash scripts/smoke_test_docker.sh
set -euo pipefail

docker rm -f fraud-api-test >/dev/null 2>&1 || true
docker run -d --name fraud-api-test -p 8080:8080 fraud-api >/dev/null
trap 'docker rm -f fraud-api-test >/dev/null' EXIT

until curl -s localhost:8080/health >/dev/null; do sleep 1; done
echo "GET /health  -> $(curl -s localhost:8080/health)"

payload='{"amt": 117.0, "hour": 3, "product_cd": "C", "card4": "visa", "card6": "credit", "uid_tx_last_24h": 6}'
echo "POST /predict -> $(curl -s -X POST localhost:8080/predict -H 'Content-Type: application/json' -d "$payload")"
echo "container user: $(docker exec fraud-api-test whoami)"
