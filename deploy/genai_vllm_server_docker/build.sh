#!/usr/bin/env bash

set -e

usage() {
    echo "Usage: $0 (--pdx-version VERSION | --pdx-wheel PATH) [--sm120] [--tag-suffix TAG]"
}

paddlex_version=''
paddlex_wheel=''
build_for_sm120='false'
tag_suffix=''

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pdx-version|--pdx-wheel|--tag-suffix)
            if [[ -z "${2:-}" || "$2" == --* ]]; then
                echo "Missing value for $1" >&2
                usage >&2
                exit 2
            fi
            case "$1" in
                --pdx-version) paddlex_version="==$2" ;;
                --pdx-wheel) paddlex_wheel="$2" ;;
                --tag-suffix) tag_suffix="$2" ;;
            esac
            shift 2
            ;;
        --sm120)
            build_for_sm120='true'
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -n "$paddlex_version" && -n "$paddlex_wheel" ]] ||
   [[ -z "$paddlex_version" && -z "$paddlex_wheel" ]]; then
    usage >&2
    exit 2
fi

if [[ -z "$tag_suffix" ]]; then
    if [[ "$build_for_sm120" == 'true' ]]; then
        tag_suffix='latest-sm120'
    else
        tag_suffix='latest'
    fi
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
context_dir="$script_dir"
wheel_name=''
if [[ -n "$paddlex_wheel" ]]; then
    if [[ ! -f "$paddlex_wheel" || "${paddlex_wheel##*/}" != paddlex-*.whl ]]; then
        echo "Expected a PaddleX wheel file: $paddlex_wheel" >&2
        exit 2
    fi
    wheel_name="${paddlex_wheel##*/}"
    context_dir="$(mktemp -d)"
    trap 'rm -rf -- "$context_dir"' EXIT
    mkdir "$context_dir/paddlex-wheel"
    cp "$script_dir/Dockerfile" "$context_dir/"
    cp "$paddlex_wheel" "$context_dir/paddlex-wheel/$wheel_name"
fi

docker build \
    -t "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-genai-vllm-server:${tag_suffix}" \
    --build-arg PADDLEX_VERSION="$paddlex_version" \
    --build-arg PADDLEX_WHEEL="$wheel_name" \
    --build-arg BUILD_FOR_SM120="$build_for_sm120" \
    --build-arg http_proxy="${http_proxy:-}" \
    --build-arg https_proxy="${https_proxy:-}" \
    --build-arg no_proxy="${no_proxy:-}" \
    "$context_dir"
