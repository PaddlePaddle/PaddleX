#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SDK_DIR}/../../.." && pwd)"

docker run \
    -it \
    -e HOME=/tmp \
    -e PIP_CACHE_DIR=/tmp/pip-cache \
    -v "${SDK_DIR}":/workspace \
    -v "${REPO_ROOT}/paddlex/inference/serving/name_mappings.py":/workspace/_name_mappings.py:ro \
    -w /workspace \
    --rm \
    --user "$(id -u):$(id -g)" \
    python:3.10 \
    /bin/bash scripts/_assemble.sh "$@"
