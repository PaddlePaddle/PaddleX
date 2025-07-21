#!/usr/bin/env bash

docker run \
    -it \
    -e OUID="$(id -u)" \
    -e OGID="$(id -g)" \
    -e PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    -v "$(pwd)":/workspace \
    -w /workspace \
    --rm \
    python3.10:sha256:875c3591e586f66aa65621926230925144920c951902a6c2eef005d9783a7ca7 \
    /bin/bash scripts/_assemble.sh "$@"
