#!/usr/bin/env bash

# Can we just do this in the Dockerfile? Like, use cache mounts?

for device_type in 'gpu' 'cpu'; do
    docker run \
        -it \
        -e DEVICE_TYPE="${device_type}" \
        -e HOME=/tmp \
        -e PIP_CACHE_DIR=/tmp/pip-cache \
        -v "$(pwd)/../../..":/workspace \
        -w /workspace/deploy/hps/server_env \
        --rm \
        --user "$(id -u):$(id -g)" \
        "paddlex-hps-rc:${device_type}" \
        /bin/bash scripts/_freeze_requirements.sh
done
