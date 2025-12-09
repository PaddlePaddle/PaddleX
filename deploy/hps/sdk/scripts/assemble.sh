#!/usr/bin/env bash

docker run \
    -it \
    -e OUID="$(id -u)" \
    -e OGID="$(id -g)" \
    -v "$(pwd)":/workspace \
    -w /workspace \
    --rm \
    python:3.10 \
    /bin/bash scripts/_assemble.sh "$@"
