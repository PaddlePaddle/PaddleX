#!/usr/bin/env bash

# Can we just do this in the Dockerfile? Like, use cache mounts?

for device_type in 'gpu' 'cpu'; do
    docker run \
        -it \
        -e DEVICE_TYPE="${device_type}" \
        -e OUID="$(id -u)" \
        -e OGID="$(id -g)" \
        -v "$(pwd)":/workspace \
        -w /workspace \
        --rm \
        "paddlex-hps-rc:${device_type}" \
        /bin/bash scripts/_freeze_requirements.sh
done
