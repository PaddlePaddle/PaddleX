#!/usr/bin/env bash

docker run \
    -it \
    -e OUID="$(id -u)" \
    -e OGID="$(id -g)" \
    -e PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    -v "$(pwd)":/workspace \
    -w /workspace \
    --rm \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/linmanhui/python:3.10 \
    /bin/bash scripts/_assemble.sh "$@"
