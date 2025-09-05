#!/usr/bin/env bash

for device_type in 'gpu' 'cpu'; do
    bash scripts/build_deployment_image.sh -k "${device_type}" -t "latest-${device_type}" "$@"
done
