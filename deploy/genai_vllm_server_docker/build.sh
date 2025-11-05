#!/usr/bin/env bash

paddlex_version='>=3.3.6,<3.4'
build_for_sm120='false'
tag_suffix='latest'

while [[ $# -gt 0 ]]; do
    case $1 in
        --pdx-version)
            paddlex_version="==$2"
            shift
            shift
            ;;
        --sm120)
            build_for_sm120='true'
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
    --build-arg BUILD_FOR_SM120="${build_for_sm120}" \
    --build-arg http_proxy="${http_proxy}" \
    --build-arg https_proxy="${https_proxy}" \
    --build-arg no_proxy="${no_proxy}" \
    .
