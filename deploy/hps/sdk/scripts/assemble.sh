#!/usr/bin/env bash

docker run \
    -it \
    -e OUID="$(id -u)" \
    -e OGID="$(id -g)" \
    -e PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    -v "$(pwd)":/workspace \
    -w /workspace \
    --rm \
    python:3.10@sha256:6ff000548a4fa34c1be02624836e75e212d4ead8227b4d4381c3ae998933a922 \
    /bin/bash scripts/_assemble.sh "$@"
