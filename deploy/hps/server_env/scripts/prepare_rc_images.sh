#!/usr/bin/env bash

pip_index_url=''

while getopts 'p:' opt; do
    case "$opt" in
        p) pip_index_url="$OPTARG" ;;
        *) echo "Usage: $0 [-p pip_index_url]" >&2; exit 2 ;;
    esac
done
shift "$((OPTIND-1))"

for device_type in 'gpu' 'cpu'; do
    args=(
    -f Dockerfile
    -t "paddlex-hps-rc:${device_type}" 
    --target 'rc' 
    --build-arg DEVICE_TYPE="${device_type}" 
    --build-arg http_proxy="${http_proxy}" 
    --build-arg https_proxy="${https_proxy}" 
    )
    [[ -n "$pip_index_url" ]] && args+=( --build-arg PIP_INDEX_URL="${pip_index_url}" )

    DOCKER_BUILDKIT=1 docker build "${args[@]}" .

done
