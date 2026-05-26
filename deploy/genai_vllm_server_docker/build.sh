#!/usr/bin/env bash

paddlex_version='>=3.6'
tag_suffix='latest'

while [[ $# -gt 0 ]]; do
    case $1 in
        --pdx-version)
            paddlex_version="==$2"
            shift
            shift
            ;;
        --tag-suffix)
            tag_suffix="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

docker build \
    -t "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-genai-vllm-server:${tag_suffix}" \
    --build-arg PADDLEX_VERSION="${paddlex_version}" \
    --build-arg http_proxy="${http_proxy}" \
    --build-arg https_proxy="${https_proxy}" \
    --build-arg no_proxy="${no_proxy}" \
    .
