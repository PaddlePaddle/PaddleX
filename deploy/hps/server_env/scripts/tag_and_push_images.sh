#!/usr/bin/env bash

set -euo pipefail

paddlex_version="$(git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//')"
if [ -z "${paddlex_version}" ]; then
    echo "Error: no git tag found to derive version" >&2
    exit 1
fi

for device_type in 'gpu' 'cpu'; do
    docker push "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:latest-${device_type}"
    for tag in "$(git rev-parse --short HEAD)-${device_type}" "paddlex${paddlex_version%.*}-${device_type}"; do
        docker tag "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:latest-${device_type}" "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:${tag}"
        docker push "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:${tag}"
    done
done
